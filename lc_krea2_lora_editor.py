"""
LC Krea2 LoRA Editor + LC LoRA Save
-----------------------------------
The Editor loads a LoRA onto a model like LoraLoaderModelOnly, but lets you turn each of the 28 Krea 2
blocks up or down, scale attention and MLP separately, and soften spikes (layers or directions that are
far stronger than the rest). The edited LoRA patches the model and also comes out as an LC_LORA, which
LC LoRA Save writes to disk as a normal LoRA with every edit baked in.

Loading goes through ComfyUI's own comfy.lora.load_lora, so every LoRA format core can read works.
Plain low-rank layers (the usual Krea 2 lora_A / lora_B files) are edited directly; other formats
(LoKr, LoHa, full diffs) are scaled per layer and converted to low rank only when saved.
"""

from __future__ import annotations

import json
import math
import os
import re

import torch

import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.utils
import folder_paths
from comfy.weight_adapter.lora import LoRAAdapter

from .lc_checkpoint_save import EMBED_WORKFLOW_INPUT, ENABLED_INPUT, METADATA_INPUT, trim_metadata, workflow_metadata
from .lc_lora_arch import ANIMA, KREA2

NUM_BLOCKS = 28
MODULES = ["attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate", "mlp.gate", "mlp.up", "mlp.down"]
MODULE_LABELS = ["q", "k", "v", "o", "gate", "m.gate", "m.up", "m.down"]
_BLOCK_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.((?:attn|mlp)\.\w+)\.weight$")

# Block multipliers per preset (1 = unchanged). Where a LoRA keeps its face lock differs per LoRA
# (tested: Better Lip Bite in blocks 16-21, Krea Amateur V4 in 4-9), so the presets are block groups
# to try, not named jobs. "Tame" is the recipe that fixed Lip Bite's whole-image change.
# Each group preset turns its blocks down to 0.3. Names follow what the blocks do in Krea 2 (block merge tests):
# 0-3 layout, 4-9 and 16-21 are where tested LoRAs held their face lock, 22-27 colour, skin and texture.
_GROUPS = {"Early blocks: structure (0-3)": (0, 3), "Face blocks (4-9)": (4, 9),
           "Mid blocks: face and body (10-15)": (10, 15), "Late face blocks (16-21)": (16, 21),
           "Finish: color and skin (22-27)": (22, 27)}
_TAME = "Tame whole image (4-15 at 0.7)"
# names used before 0.4.1, so saved workflows keep their preset
LEGACY = {"Blocks 0-3 down": "Early blocks: structure (0-3)", "Blocks 4-9 down": "Face blocks (4-9)",
          "Blocks 10-15 down": "Mid blocks: face and body (10-15)", "Blocks 16-21 down": "Late face blocks (16-21)",
          "Blocks 22-27 down": "Finish: color and skin (22-27)", "Tame (4-15 at 0.7)": _TAME}


def _preset_curve(name: str) -> list[float] | None:
    c = [1.0] * NUM_BLOCKS
    name = LEGACY.get(name, name)
    if name == "Full":
        return c
    if name == _TAME:
        return [0.7 if 4 <= i <= 15 else 1.0 for i in range(NUM_BLOCKS)]
    if name in _GROUPS:
        lo, hi = _GROUPS[name]
        return [0.3 if lo <= i <= hi else 1.0 for i in range(NUM_BLOCKS)]
    return None


PRESETS = ["Full", *_GROUPS, _TAME, "Custom"]


def _parse_curve(text) -> list[float]:
    vals = []
    for part in str(text or "").replace("\n", ",").split(","):
        try:
            vals.append(float(part.strip()))
        except ValueError:
            pass
    if len(vals) != NUM_BLOCKS:
        vals = (vals + [1.0] * NUM_BLOCKS)[:NUM_BLOCKS]
    return [max(0.0, min(2.0, v)) for v in vals]


def _key_of(patch_key):
    return patch_key if isinstance(patch_key, str) else patch_key[0]


def _slice_entry(entry, offset, weight, prev):
    """A LoRA that targets one slice of a fused layer (key, (dim, start, size)): widen up / down to the whole layer
    (zeros outside the slice) and stack it onto the slices already there, so LC LoRA Save keeps every slice, not the last.
    Falls back to the slice as it was when the layer's shape is unknown."""
    if weight is None or weight.ndim != 2 or len(offset) != 3:
        return entry
    dim, start, size = offset
    rows, cols = weight.shape
    up, down = entry["up"], entry["down"]
    if dim == 0 and down.shape[1] == cols and up.shape[0] == size:
        full_up = torch.zeros(rows, up.shape[1], dtype=up.dtype)
        full_up[start:start + size] = up
        full_down = down
    elif dim == 1 and up.shape[0] == rows and down.shape[1] == size:
        full_up = up
        full_down = torch.zeros(down.shape[0], cols, dtype=down.dtype)
        full_down[:, start:start + size] = down
    else:
        return entry
    if prev and prev.get("kind") == "lora" and prev["up"].shape[0] == rows and prev["down"].shape[1] == cols:
        full_up = torch.cat([prev["up"].to(full_up.dtype), full_up], 1)
        full_down = torch.cat([prev["down"].to(full_down.dtype), full_down], 0)
    return {"kind": "lora", "up": full_up, "down": full_down}


def _where(model_key: str):
    """(block, module row) for a Krea 2 block layer, or None for anything outside the 28 blocks."""
    m = _BLOCK_RE.search(model_key)
    if not m:
        return None
    mod = m.group(2)
    if mod not in MODULES:
        return None
    return int(m.group(1)), MODULES.index(mod)


def _plain(adapter) -> bool:
    """A plain 2-D low-rank layer we can edit directly (no mid / DoRA / reshape)."""
    if not isinstance(adapter, LoRAAdapter):
        return False
    up, down, _alpha, mid, dora, reshape = adapter.weights
    return mid is None and dora is None and reshape is None and up.ndim == 2 and down.ndim == 2


def _lora_scale(adapter) -> float:
    down, alpha = adapter.weights[1], adapter.weights[2]
    return (alpha / down.shape[0]) if alpha is not None else 1.0


def _lowrank_norm(up: torch.Tensor, down: torch.Tensor) -> float:
    # ||up @ down||_F without building the full matrix
    return float(torch.sqrt(torch.clamp(torch.trace((up.T @ up) @ (down @ down.T)), min=0)))


def _core_svd(up: torch.Tensor, down: torch.Tensor):
    """up @ down = (Qu U) diag(S) (Vh Qd^T): exact SVD of a low-rank product through its r x r core."""
    qu, ru = torch.linalg.qr(up)
    qd, rd = torch.linalg.qr(down.T)
    u, s, vh = torch.linalg.svd(ru @ rd.T)
    return qu @ u, s, vh @ qd.T


_CACHE = {}  # (path, mtime) -> (state dict, metadata); a few kept, so the Merge node's LoRAs are not re-read


def _load_lora_file(path: str):
    key = (path, os.path.getmtime(path))
    if key not in _CACHE:
        sd, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        while len(_CACHE) >= 5:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = (comfy.lora_convert.convert_lora(sd), meta)
    return _CACHE[key]


class LCKrea2LoRAEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Krea 2 model. The edited LoRA is applied to it, like a LoRA loader."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA to edit, from models/loras."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05,
                                       "tooltip": "Overall LoRA strength, same as a LoRA loader. Baked into the saved LoRA."}),
                "preset": (PRESETS, {"default": "Full",
                                     "tooltip": "Full = the LoRA as it is. Each block group preset turns those blocks down to 0.3. "
                                                "Face lock sits in different blocks for each LoRA (Lip Bite: late face blocks 16-21, "
                                                "Amateur V4: face blocks 4-9), so try each on a fixed seed batch and keep the one that "
                                                "frees the face. Tame whole image = blocks 4-15 at 0.7, which with spike_soften directions "
                                                "stopped Lip Bite changing the whole image. Dragging the graph switches to Custom."}),
                "blocks": ("STRING", {"default": ", ".join(["1"] * NUM_BLOCKS),
                                      "tooltip": "28 block multipliers, 0 to 2 (1 = unchanged). Drag the bars on the node."}),
                "attention": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                        "tooltip": "Multiplier for the attention layers (q, k, v, o, gate) in every block."}),
                "mlp": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                  "tooltip": "Multiplier for the MLP layers (gate, up, down) in every block. "
                                             "Most of a Krea 2 LoRA's weight usually sits here."}),
                "outside_blocks": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                             "tooltip": "Multiplier for layers outside the 28 blocks (text fusion, first, last), "
                                                        "if the LoRA has any. The heatmap note says how many."}),
                "spike_soften": (["off", "layers", "directions", "both"], {"default": "off",
                                 "tooltip": "layers = pull any layer that is far stronger than the same layer in other blocks "
                                            "down to the limit. directions = inside each layer, flatten the one or two "
                                            "directions that dominate it (plain lora_A / lora_B files only). both = both."}),
                "spike_limit": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1,
                                          "tooltip": "How strong a spike may be before it is softened. layers: times the median "
                                                     "of that layer across the 28 blocks. directions: times the layer's second "
                                                     "strongest direction (well-behaved LoRAs sit around 2). Lower = more softening."}),
            },
            "optional": {
                "blocks_in": ("STRING", {"forceInput": True,
                                         "tooltip": "A 28-block curve from another node, such as LC Krea2 LoRA Block Scan's "
                                                    "blocks output. When wired it replaces the preset and the bars."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, preset=None, **kwargs):
        # old preset names from saved workflows are mapped in edit()
        if preset is not None and preset not in PRESETS and preset not in LEGACY:
            return f"unknown preset: {preset}"
        return True

    RETURN_TYPES = ("MODEL", "LC_LORA")
    RETURN_NAMES = ("model", "edited_lora")
    OUTPUT_TOOLTIPS = ("The model with the edited LoRA applied.", "The edited LoRA, for LC LoRA Save.")
    FUNCTION = "edit"
    CATEGORY = "LC ModelBuilder/Krea2"
    DESCRIPTION = (
        "Loads a Krea 2 LoRA with per-block, attention / MLP and spike edits, applies it to the model and "
        "outputs the edited LoRA for LC LoRA Save. The heatmap shows each layer's strength before (outline) "
        "and after (fill) the edits."
    )

    ARCH = KREA2

    @classmethod
    def preset_curve(cls, name):
        return _preset_curve(name)

    def edit(self, model, lora_name, strength, preset, blocks, attention, mlp, outside_blocks,
             spike_soften, spike_limit, blocks_in=None):
        arch = self.ARCH
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd, meta = _load_lora_file(path)
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        patches = comfy.lora.load_lora(sd, key_map, log_missing=False)
        preset = LEGACY.get(preset, preset)
        curve = _parse_curve(blocks_in) if blocks_in else (self.preset_curve(preset) or _parse_curve(blocks))
        soften_layers = spike_soften in ("layers", "both")
        soften_dirs = spike_soften in ("directions", "both")
        dev = comfy.model_management.get_torch_device()
        model_sd = None

        # 1. original strength of every layer
        layers = []
        for pk, ad in patches.items():
            k = _key_of(pk)
            where = arch.where(k)
            if _plain(ad):
                up = ad.weights[0].to(dev, torch.float32)
                down = ad.weights[1].to(dev, torch.float32)
                norm = abs(_lora_scale(ad)) * _lowrank_norm(up, down)
            else:
                if model_sd is None:
                    model_sd = model.model.state_dict()
                w = model_sd.get(k)
                norm = 0.0
                if isinstance(ad, tuple) and ad[0] == "diff":  # a plain weight difference (e.g. a bias)
                    norm = float(ad[1][0].float().norm())
                elif w is not None:
                    try:
                        z = torch.zeros(w.shape, dtype=torch.float32, device=dev)
                        norm = float(ad.calculate_weight(z, k, 1.0, 1.0, None, lambda a: a, torch.float32).norm())
                        del z
                    except Exception:
                        norm = 0.0
            layers.append({"pk": pk, "key": k, "ad": ad, "where": where, "norm": norm})

        # 2. per-layer multiplier: block curve x attention / mlp, and the layer spike cap
        medians = {}
        for row in range(len(arch.rows)):
            vals = sorted(L["norm"] for L in layers if L["where"] and L["where"][1] == row and L["norm"] > 0)
            if vals:
                medians[row] = vals[len(vals) // 2]
        capped = 0
        for L in layers:
            if L["where"] is None:
                s = outside_blocks
            else:
                b, row = L["where"]
                s = curve[b] * arch.mult(row, attention, mlp)
                if soften_layers and row in medians and L["norm"] > spike_limit * medians[row]:
                    s *= spike_limit * medians[row] / L["norm"]
                    capped += 1
            L["scale"] = s

        # 3. build the edited layers; plain ones get every edit baked into up / down
        out_layers, new_patches, groups, shapes = {}, {}, {}, {}
        flattened = 0
        for L in layers:
            ad, s = L["ad"], L["scale"] * strength
            if _plain(ad):
                up = ad.weights[0].to(dev, torch.float32) * _lora_scale(ad)
                down = ad.weights[1].to(dev, torch.float32)
                if soften_dirs and up.shape[1] > 1:
                    u, sv, vh = _core_svd(up, down)
                    cap = spike_limit * float(sv[1])  # the top direction may be at most limit x the next one
                    if float(sv[0]) > cap:
                        flattened += 1
                        sv = torch.clamp(sv, max=cap)
                    up, down = u * sv, vh
                L["after"] = abs(L["scale"]) * _lowrank_norm(up, down)
                up = (up * s).to(ad.weights[0].dtype).cpu()
                down = down.to(ad.weights[1].dtype).cpu()
                entry = {"kind": "lora", "up": up, "down": down}
                if isinstance(L["pk"], tuple) and len(L["pk"]) > 1 and L["pk"][1] is not None:
                    if model_sd is None:
                        model_sd = model.model.state_dict()
                    entry = _slice_entry(entry, L["pk"][1], model_sd.get(L["key"]), out_layers.get(L["key"]))
                out_layers[L["key"]] = entry
                new_patches[L["pk"]] = LoRAAdapter(set(), (up, down, None, None, None, None))
            else:
                L["after"] = L["norm"] * abs(L["scale"])
                out_layers[L["key"]] = {"kind": "adapter", "adapter": ad, "scale": s}
                if model_sd is not None and L["key"] in model_sd:
                    shapes[L["key"]] = tuple(model_sd[L["key"]].shape)
                groups.setdefault(s, {})[L["pk"]] = ad

        m = model.clone()
        applied = m.add_patches(new_patches, 1.0) if new_patches else []
        for s, pd in groups.items():
            applied += m.add_patches(pd, s)
        if meta:
            m.set_attachments("lora_metadata", meta)

        # 4. heatmap for the node: one row per layer type x 28 blocks, before and after
        before = [[0.0] * NUM_BLOCKS for _ in arch.rows]
        after = [[0.0] * NUM_BLOCKS for _ in arch.rows]
        outside = 0
        for L in layers:
            if L["where"] is None:
                outside += 1
                continue
            b, row = L["where"]
            before[row][b] = round(L["norm"], 5)
            after[row][b] = round(L["after"], 5)
        tot_b = math.sqrt(sum(L["norm"] ** 2 for L in layers)) or 1.0
        tot_a = math.sqrt(sum(L["after"] ** 2 for L in layers))
        ranks = sorted({int(L["ad"].weights[1].shape[0]) for L in layers if _plain(L["ad"])})
        rank_txt = "/".join(map(str, ranks)) if len(ranks) <= 4 else f"{ranks[0]}-{ranks[-1]}"
        note = [f"{len(applied)} of {len(patches)} layers applied" + (f", rank {rank_txt}" if ranks else "")]
        note.append(f"strength kept {100 * tot_a / tot_b:.0f} %")
        if soften_layers:
            note.append(f"{capped} layers capped")
        if soften_dirs:
            note.append(f"{flattened} layers flattened")
        if outside:
            note.append(f"{outside} outside the blocks")
        if len(applied) < len(patches):
            note.append(f"some layers do not match this model: is it a {arch.title} LoRA?")

        lora_obj = {
            "arch": arch.name,
            "source": lora_name,
            "layers": out_layers,
            "shapes": shapes,
            "metadata": dict(meta or {}),
            "edit": {"strength": strength, "preset": preset, "blocks": [round(v, 4) for v in curve],
                     "attention": attention, "mlp": mlp, "outside_blocks": outside_blocks,
                     "spike_soften": spike_soften, "spike_limit": spike_limit},
        }
        ui = {"lc_lora_map": [{"before": before, "after": after, "rows": arch.labels, "note": " · ".join(note),
                                "curve": [round(v, 4) for v in curve], "wired": bool(blocks_in)}]}
        return {"ui": ui, "result": (m, lora_obj)}


# Anima block groups, named after the Animosity Anima block scan (README): turning 7-13 or 14-20 off changes the
# picture most (the LoRA's look lives there), 0-6 and 21-27 much less. Each preset turns its blocks down to 0.3.
ANIMA_GROUPS = {"Early blocks (0-6)": (0, 6), "Mid blocks: look (7-13)": (7, 13),
                "Mid blocks: look and character (14-20)": (14, 20), "Late blocks: finish (21-27)": (21, 27)}
ANIMA_PRESETS = ["Full", *ANIMA_GROUPS, "Custom"]


class LCAnimaLoRAEditor(LCKrea2LoRAEditor):
    """The LoRA Editor for Anima: same edits, Anima's 28 blocks and layer rows."""
    ARCH = ANIMA

    @classmethod
    def INPUT_TYPES(cls):
        d = super().INPUT_TYPES()
        r = d["required"]
        r["model"] = ("MODEL", {"tooltip": "Anima model. The edited LoRA is applied to it, like a LoRA loader."})
        r["preset"] = (ANIMA_PRESETS, {"default": "Full",
                                       "tooltip": "Full = the LoRA as it is. Each block group preset turns those blocks "
                                                  "down to 0.3. Try each on a fixed seed batch and keep the one that frees "
                                                  "the face. Dragging the graph switches to Custom."})
        r["attention"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                    "tooltip": "Multiplier for the attention layers (self and cross attention) in every block."})
        r["mlp"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                              "tooltip": "Multiplier for the MLP layers in every block."})
        r["outside_blocks"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                         "tooltip": "Multiplier for layers outside the 28 blocks (LLM adapter, final layer), "
                                                    "if the LoRA has any. The heatmap note says how many."})
        d["optional"]["blocks_in"] = ("STRING", {"forceInput": True,
                                                 "tooltip": "A 28-block curve from another node, such as LC Anima LoRA Block "
                                                            "Scan's blocks output. When wired it replaces the preset and the bars."})
        return d

    @classmethod
    def VALIDATE_INPUTS(cls, preset=None, **kwargs):
        return True if preset is None or preset in ANIMA_PRESETS else f"unknown preset: {preset}"

    @classmethod
    def preset_curve(cls, name):
        if name == "Full":
            return [1.0] * NUM_BLOCKS
        if name in ANIMA_GROUPS:
            lo, hi = ANIMA_GROUPS[name]
            return [0.3 if lo <= i <= hi else 1.0 for i in range(NUM_BLOCKS)]
        return None

    CATEGORY = "LC ModelBuilder/Anima"
    DESCRIPTION = (
        "Loads an Anima LoRA with per-block, attention / MLP and spike edits, applies it to the model and outputs the "
        "edited LoRA for LC LoRA Save. The heatmap shows each layer's strength before and after the edits."
    )


class LCLoRASave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": ("LC_LORA", {"tooltip": "Edited or merged LoRA from an LC LoRA Editor or LoRA Merge (Krea 2 or Anima)."}),
                "filename": ("STRING", {"default": "edited/%source%_edit",
                                        "tooltip": "Saved under models/loras. %source% = the original LoRA's name. "
                                                   "A number is added if the file exists."}),
                "rank": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1,
                                 "tooltip": "0 = keep each layer's rank. Any other number re-fits every layer to that rank "
                                            "(smaller file, keeps the strongest directions)."}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "enabled": ENABLED_INPUT,
                "embed_workflow": EMBED_WORKFLOW_INPUT,
            },
            "optional": {"metadata": METADATA_INPUT},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "LC ModelBuilder/Krea2"
    DESCRIPTION = ("Saves an edited LoRA as a standard lora_A / lora_B safetensors file with every edit baked in, "
                   "so it loads at strength 1.0 in any LoRA loader. The edit settings are stored in the metadata.")

    def save(self, lora, filename, rank, dtype, enabled=True, embed_workflow=True, metadata="keep all", prompt=None,
             extra_pnginfo=None):
        if not enabled:
            return {"ui": {"text": ["Save is off (enabled = false): nothing written."]}}
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        dev = comfy.model_management.get_torch_device()
        source = os.path.splitext(os.path.basename(lora.get("source", "lora")))[0]
        rel = (filename or "edited/%source%_edit").replace("%source%", source).strip().replace("\\", "/")
        rel = rel[:-12] if rel.lower().endswith(".safetensors") else rel
        root = folder_paths.get_folder_paths("loras")[0]
        base = os.path.abspath(os.path.join(root, rel))
        if not base.startswith(os.path.join(os.path.abspath(root), "")):  # trailing separator: "loras2" is outside
            raise ValueError("filename must stay inside models/loras")
        path, n = base + ".safetensors", 1
        while os.path.exists(path):
            path, n = f"{base}_{n:02d}.safetensors", n + 1
        os.makedirs(os.path.dirname(path), exist_ok=True)

        out, converted = {}, 0
        for key, L in lora["layers"].items():
            name = key[:-7] if key.endswith(".weight") else key
            if L["kind"] == "diff":  # a plain difference (bias or norm) from LoRA Merge
                out[f"{key[:-5]}.diff_b" if key.endswith(".bias") else f"{name}.diff"] = L["delta"].to(dt).cpu().contiguous()
                continue
            if L["kind"] == "lora":
                up, down = L["up"].to(dev, torch.float32), L["down"].to(dev, torch.float32)
                if rank and rank < up.shape[1]:
                    u, s, vh = _core_svd(up, down)
                    up, down = u[:, :rank] * s[:rank], vh[:rank]
            else:
                ad = L["adapter"]
                delta = self._delta(ad, key, lora)
                if delta is None:
                    continue
                delta = delta * L["scale"]
                if delta.ndim < 2:  # biases and norms are saved as plain differences
                    out[f"{key[:-5]}.diff_b" if key.endswith(".bias") else f"{name}.diff"] = delta.to(dt).cpu().contiguous()
                    continue
                r = rank or 32
                u, s, v = torch.svd_lowrank(delta.flatten(1), q=min(r + 8, min(delta.shape[0], delta.flatten(1).shape[1])))
                up, down = u[:, :r] * s[:r], v[:, :r].T
                converted += 1
            out[f"{name}.lora_B.weight"] = up.to(dt).cpu().contiguous()
            out[f"{name}.lora_A.weight"] = down.to(dt).cpu().contiguous()

        meta = {k: str(v) for k, v in (lora.get("metadata") or {}).items()}
        meta["lc_edit"] = json.dumps({"source": lora.get("source"), **lora.get("edit", {}), "rank": rank})
        if embed_workflow:
            meta.update(workflow_metadata(prompt, extra_pnginfo))
        meta = trim_metadata(meta, metadata)
        comfy.utils.save_torch_file(out, path, metadata=meta)
        n_layers = sum(1 for k in out if k.endswith((".lora_B.weight", ".diff", ".diff_b")))
        msg = f"Saved {os.path.relpath(path, root)} ({n_layers} layers"
        msg += f", {converted} converted to rank {rank or 32})" if converted else ")"
        print(f"[LC Modelbuilder] {msg}")
        return {"ui": {"text": [msg]}}

    @staticmethod
    def _delta(adapter, key, lora):
        shape = lora.get("shapes", {}).get(key)
        if shape is None:
            return None
        if isinstance(adapter, tuple) and adapter[0] == "diff":
            return adapter[1][0].to(comfy.model_management.get_torch_device(), torch.float32).reshape(shape)
        z = torch.zeros(shape, dtype=torch.float32, device=comfy.model_management.get_torch_device())
        return adapter.calculate_weight(z, key, 1.0, 1.0, None, lambda a: a, torch.float32)


NODE_CLASS_MAPPINGS = {"LCKrea2LoRAEditor": LCKrea2LoRAEditor, "LCLoRASave": LCLoRASave,
                       "LCAnimaLoRAEditor": LCAnimaLoRAEditor}
NODE_DISPLAY_NAME_MAPPINGS = {"LCKrea2LoRAEditor": "LC Krea2 LoRA Editor", "LCLoRASave": "LC LoRA Save",
                              "LCAnimaLoRAEditor": "LC Anima LoRA Editor"}
