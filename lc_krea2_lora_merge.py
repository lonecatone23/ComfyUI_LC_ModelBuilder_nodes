"""
LC Krea2 LoRA Merge
-------------------
Merges two to four Krea 2 LoRAs into one, with a 28-block A / B curve: per block, 0 = only LoRA A, 0.5 = both at
full strength, 1 = only LoRA B. The optional LoRAs C and D join at their own strength in every block. Patches the
model like a LoRA loader and outputs the merged LoRA for LC LoRA Save.

  sum     : the LoRAs added. Exact: the merged LoRA is them stacked (its rank is the sum of theirs; LC LoRA Save's
            rank shrinks it).
  ties    : TIES merging (Yadav et al., 2023) on each layer's full weight change: keep each LoRA's strongest changes
            (ties_density), let them vote on the direction of every weight, and drop the changes that fight the
            vote. Where the LoRAs pull the same way, the change is their average; where only one changes a weight,
            it keeps its full value. Re-fitted to low rank afterwards.
  mix     : one slider over the whole curve, like a plain merge: 0 = only A, 0.5 = the curve as set, 1 = only B.
  Face presets: B's face blocks into A (A = the main LoRA, B = the face). Where a LoRA keeps its face varies per
            LoRA, so they are starting points: adjust the bars, or wire LC Krea2 LoRA Block Scan's curve.
"""

from __future__ import annotations

import os

import torch

import comfy.lora
import comfy.model_management
import folder_paths
from comfy.weight_adapter.lora import LoRAAdapter

from .lc_krea2_lora_editor import (MODULE_LABELS, MODULES, NUM_BLOCKS, _key_of, _load_lora_file, _lora_scale,
                                   _lowrank_norm, _parse_curve, _plain, _where)

def _face(lo: int, hi: int) -> list[float]:
    """B inside the window, A outside, with a block at 'both full' on each side so the hand-over is not a cliff."""
    return [1.0 if lo <= i <= hi else 0.5 if i in (lo - 1, hi + 1) else 0.0 for i in range(NUM_BLOCKS)]


PRESETS = {
    "Both full": [0.5] * NUM_BLOCKS,
    "A early, B late": [0.0 if i < 14 else 1.0 for i in range(NUM_BLOCKS)],
    "B early, A late": [1.0 if i < 14 else 0.0 for i in range(NUM_BLOCKS)],
    "A only": [0.0] * NUM_BLOCKS,
    "B only": [1.0] * NUM_BLOCKS,
    "Face from B (8-20)": _face(8, 20),
    "Face from B (4-13)": _face(4, 13),
    "Face from B (16-21)": _face(16, 21),
}
PRESET_NAMES = [*PRESETS, "Custom"]
MAX_RANK = 128


def _weights(c: float):
    """Curve value -> (weight of A, weight of B). 0 = A only, 0.5 = both full, 1 = B only."""
    return min(1.0, 2.0 * (1.0 - c)), min(1.0, 2.0 * c)


def _parse01(text):
    return [max(0.0, min(1.0, v)) for v in _parse_curve(text)]


def _lowrank(adapter, key, shapes, dev):
    """(up, down) in float32 with the adapter's alpha folded in, or None if it is not a plain low-rank layer
    (those are expanded to a full delta by the caller)."""
    if not _plain(adapter):
        return None
    up = adapter.weights[0].to(dev, torch.float32) * _lora_scale(adapter)
    return up, adapter.weights[1].to(dev, torch.float32)


def _full(adapter, key, shape, dev):
    lr = _lowrank(adapter, key, None, dev)
    if lr is not None:
        return lr[0] @ lr[1]
    z = torch.zeros(shape, dtype=torch.float32, device=dev)
    return adapter.calculate_weight(z, key, 1.0, 1.0, None, lambda a: a, torch.float32)


def _trim(delta: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the largest |values| (top `density` share), zero the rest. Cutoff estimated from a sample."""
    if density >= 0.999:
        return delta
    flat = delta.abs().flatten()
    if flat.numel() > 2_000_000:
        idx = torch.randint(flat.numel(), (2_000_000,), device=flat.device, generator=None)
        sample = flat[idx]
    else:
        sample = flat
    cut = torch.quantile(sample.float(), 1.0 - density)
    return delta * (delta.abs() >= cut)


def _ties(deltas: list[torch.Tensor], density: float) -> torch.Tensor:
    ts = [_trim(d, density) for d in deltas]
    sign = torch.sign(sum(ts))
    total, count = torch.zeros_like(ts[0]), torch.zeros_like(ts[0])
    for t in ts:
        agree = (torch.sign(t) == sign) & (t != 0)
        total += t * agree
        count += agree
    return total / count.clamp(min=1.0)


def _refit(delta: torch.Tensor, rank: int):
    """Full delta -> (up, down) of the given rank, keeping the strongest directions."""
    rank = max(1, min(rank, min(delta.shape) - 1))
    torch.manual_seed(0)  # svd_lowrank is randomised: same input, same result
    u, s, v = torch.svd_lowrank(delta, q=min(rank + 8, min(delta.shape)), niter=2)
    return u[:, :rank] * s[:rank], v[:, :rank].T


class LCKrea2LoRAMerge:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        extra = ["None"] + loras
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Krea 2 model. The merged LoRA is applied to it, like a LoRA loader."}),
                "lora_a": (loras, {"tooltip": "LoRA A, from models/loras."}),
                "lora_b": (loras, {"tooltip": "LoRA B, from models/loras."}),
                "strength_a": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "strength_b": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "mix": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                        "tooltip": "One slider for the whole merge, like a plain model merge: 0 = only LoRA A, 1 = only "
                                   "LoRA B. At 0.5 the bars apply as set; toward 0 B fades out everywhere, toward 1 A does."}),
                "method": (["sum", "ties"], {"default": "sum",
                           "tooltip": "sum = the LoRAs added, exactly (fast). ties = keeps each LoRA's strongest changes and "
                                      "drops the ones that fight each other: for LoRAs that clash, e.g. two styles or "
                                      "two characters. Slower (works on every full layer)."}),
                "preset": (PRESET_NAMES, {"default": "Both full",
                           "tooltip": "Per block: 0 = only A, 0.5 = both at full strength, 1 = only B. Dragging the "
                                      "graph switches to Custom."}),
                "blocks": ("STRING", {"default": ", ".join(["0.5"] * NUM_BLOCKS),
                                      "tooltip": "28 block values, 0 (A) to 1 (B). Drag the bars on the node."}),
                "ties_density": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
                                 "tooltip": "ties only: share of each LoRA's changes that is kept (its strongest). "
                                            "Lower = only the most important changes survive."}),
            },
            "optional": {
                "blocks_in": ("STRING", {"forceInput": True,
                                         "tooltip": "A 28-block curve from another node. When wired it replaces the preset and the bars."}),
                "lora_c": (extra, {"default": "None", "tooltip": "Optional third LoRA, at strength_c in every block (not on the curve)."}),
                "strength_c": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "lora_d": (extra, {"default": "None", "tooltip": "Optional fourth LoRA, at strength_d in every block (not on the curve)."}),
                "strength_d": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MODEL", "LC_LORA")
    RETURN_NAMES = ("model", "merged_lora")
    OUTPUT_TOOLTIPS = ("The model with the merged LoRA applied.", "The merged LoRA, for LC LoRA Save.")
    FUNCTION = "merge"
    CATEGORY = "LC ModelBuilder/Krea2"
    DESCRIPTION = ("Merges two to four Krea 2 LoRAs with a per-block A / B curve (sum or TIES), applies the result to the "
                   "model and outputs it for LC LoRA Save. The heatmaps show A's and B's share and the merged result.")

    def merge(self, model, lora_a, lora_b, strength_a, strength_b, method, preset, blocks, ties_density, mix=0.5,
              blocks_in=None, lora_c="None", strength_c=1.0, lora_d="None", strength_d=1.0, **_old):
        # _old: settings from earlier saves that no longer exist (balance)
        curve = _parse01(blocks_in) if blocks_in else (PRESETS.get(preset) or _parse01(blocks))
        dev = comfy.model_management.get_torch_device()
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})

        def load(name):
            p = comfy.lora.load_lora(_load_lora_file(folder_paths.get_full_path_or_raise("loras", name))[0], key_map,
                                     log_missing=False)
            return {_key_of(k): v for k, v in p.items()}

        # (patches by layer, weight from the block's (wa, wb)): A and B follow the curve, C and D their own strength
        used = [lora_a, lora_b]
        src = [(load(lora_a), lambda wa, wb: wa * float(strength_a)), (load(lora_b), lambda wa, wb: wb * float(strength_b))]
        for name, st in ((lora_c, strength_c), (lora_d, strength_d)):
            if name and name != "None" and abs(float(st)) > 1e-6:
                used.append(name)
                src.append((load(name), lambda wa, wb, st=float(st): st))
        model_sd = None
        layers, patches = {}, {}
        maps = {n: [[0.0] * NUM_BLOCKS for _ in MODULES] for n in ("a", "b", "m")}
        outside = 0
        mix_a, mix_b = _weights(max(0.0, min(1.0, float(mix))))
        for key in sorted(set().union(*(set(p) for p, _ in src))):
            comfy.model_management.throw_exception_if_processing_interrupted()
            where = _where(key)
            # layers outside the 28 blocks (text fusion) follow the curve's average, so "A only" really is A only
            wa, wb = _weights(curve[where[0]] if where else sum(curve) / len(curve))
            wa, wb = wa * mix_a, wb * mix_b
            parts = []  # [source index, adapter, weight, (up, down) or None, unweighted norm or None]
            for n, (p, wf) in enumerate(src):
                w = wf(wa, wb)
                if key in p and abs(w) > 1e-6:
                    lr = _lowrank(p[key], key, None, dev)
                    parts.append([n, p[key], w, lr, _lowrank_norm(*lr) if lr else None])
            if not parts:
                continue
            need_full = (method == "ties" and len(parts) > 1) or any(x[3] is None for x in parts)
            fulls = {}
            if need_full:
                if model_sd is None:
                    model_sd = model.model.state_dict()
                shape = tuple(model_sd[key].shape) if key in model_sd else None
                if shape is None:
                    continue
                for x in parts:
                    fulls[x[0]] = _full(x[1], key, shape, dev)
                    if x[4] is None:
                        x[4] = float(fulls[x[0]].norm())
            norms = {x[0]: x[4] * abs(x[2]) for x in parts}
            if not need_full:
                up = torch.cat([x[3][0] * x[2] for x in parts], 1)
                down = torch.cat([x[3][1] for x in parts], 0)
            else:
                deltas = [fulls[x[0]] * x[2] for x in parts]
                delta = _ties(deltas, float(ties_density)) if method == "ties" and len(deltas) > 1 else sum(deltas)
                rank = sum(int(x[1].weights[1].shape[0]) for x in parts if _plain(x[1])) or 32
                up, down = _refit(delta.reshape(delta.shape[0], -1), min(MAX_RANK, rank))
                del deltas, delta, fulls
            norm_m = _lowrank_norm(up, down)
            up, down = up.to(torch.bfloat16).cpu(), down.to(torch.bfloat16).cpu()
            layers[key] = {"kind": "lora", "up": up, "down": down}
            patches[key] = LoRAAdapter(set(), (up, down, None, None, None, None))
            if where:
                b_, r_ = where
                maps["a"][r_][b_] = round(norms.get(0, 0.0), 5)
                maps["b"][r_][b_] = round(norms.get(1, 0.0), 5)
                maps["m"][r_][b_] = round(norm_m, 5)
            else:
                outside += 1

        m = model.clone()
        applied = m.add_patches(patches, 1.0)
        name = lambda p: os.path.splitext(os.path.basename(p))[0]
        note = f"{len(applied)} layers merged ({method}, {len(used)} LoRAs)"
        if outside:
            note += f" · {outside} outside the blocks"
        if len(applied) < len(patches):
            note += " · some layers do not match this model: are they all Krea 2 LoRAs?"
        lora_obj = {
            "arch": "krea2", "source": "+".join(name(u) for u in used), "layers": layers, "shapes": {},
            "metadata": {},
            "edit": {"merge": method, "lora_a": lora_a, "lora_b": lora_b, "strength_a": strength_a,
                     "strength_b": strength_b, "lora_c": lora_c, "strength_c": strength_c, "lora_d": lora_d,
                     "strength_d": strength_d, "mix": mix, "blocks": [round(v, 4) for v in curve],
                     "ties_density": ties_density},
        }
        ui = {"lc_lora_map": [{"maps": [["LoRA A", maps["a"]], ["LoRA B", maps["b"]], ["merged", maps["m"]]],
                               "rows": MODULE_LABELS, "note": note, "curve": [round(v, 4) for v in curve],
                               "wired": bool(blocks_in)}]}
        return {"ui": ui, "result": (m, lora_obj)}


NODE_CLASS_MAPPINGS = {"LCKrea2LoRAMerge": LCKrea2LoRAMerge}
NODE_DISPLAY_NAME_MAPPINGS = {"LCKrea2LoRAMerge": "LC Krea2 LoRA Merge"}
