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
  balanced: per layer, every LoRA's change is first brought to the same size (the median of theirs), then they are
            added and the result is set back to that size. A loud LoRA can't drown a quiet one, and two LoRAs at full
            come out as strong as one, not twice as strong. Strengths and the curve still apply on top.
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

from .lc_krea2_lora_editor import (NUM_BLOCKS, _core_svd, _key_of, _load_lora_file, _lora_scale, _lowrank_norm, _parse_curve,
                                   _plain)
from .lc_lora_arch import ANIMA, KREA2, MOSS


def _basic(n: int) -> dict:
    h = n // 2
    return {"Both full": [0.5] * n, "A early, B late": [0.0 if i < h else 1.0 for i in range(n)],
            "B early, A late": [1.0 if i < h else 0.0 for i in range(n)], "A only": [0.0] * n, "B only": [1.0] * n}


def _face(lo: int, hi: int) -> list[float]:
    """B inside the window, A outside, with a block at 'both full' on each side so the hand-over is not a cliff."""
    return [1.0 if lo <= i <= hi else 0.5 if i in (lo - 1, hi + 1) else 0.0 for i in range(NUM_BLOCKS)]


PRESETS = {
    **_basic(NUM_BLOCKS),
    "Face from B (8-20)": _face(8, 20),
    "Face from B (4-13)": _face(4, 13),
    "Face from B (16-21)": _face(16, 21),
}
PRESET_NAMES = [*PRESETS, "Custom"]
MAX_RANK = 128


def _weights(c: float):
    """Curve value -> (weight of A, weight of B). 0 = A only, 0.5 = both full, 1 = B only."""
    return min(1.0, 2.0 * (1.0 - c)), min(1.0, 2.0 * c)


def _parse01(text, n=NUM_BLOCKS):
    return [max(0.0, min(1.0, v)) for v in _parse_curve(text, n)]


def _widen(entries, rows_cols):
    """Slice LoRAs (q / k / v trained as parts of one fused qkv layer) -> one plain LoRA over the whole layer: each
    slice's up (or down) padded with zeros outside its slice, the slices stacked. None when it can't be done."""
    ups, downs = [], []
    rows, cols = rows_cols
    for off, ad in entries:
        if not _plain(ad) or off is None or len(off) != 3:
            return None
        dim, start, size = off
        up = ad.weights[0].float() * _lora_scale(ad)
        down = ad.weights[1].float()
        if dim == 0 and down.shape[1] == cols and up.shape[0] == size:
            full = torch.zeros(rows, up.shape[1])
            full[start:start + size] = up
            ups.append(full)
            downs.append(down)
        elif dim == 1 and up.shape[0] == rows and down.shape[1] == size:
            full = torch.zeros(down.shape[0], cols)
            full[:, start:start + size] = down
            ups.append(up)
            downs.append(full)
        else:
            return None
    return LoRAAdapter(set(), (torch.cat(ups, 1), torch.cat(downs, 0), None, None, None, None))


def _inner(a, b, key, shape, dev):
    """<delta A, delta B> and both sizes, for the conflict map. Plain LoRAs never build the full matrix."""
    """How A and B meet in one layer: (overlap 0..1, direction -1..1, size of A, size of B).
    Two LoRAs trained apart are almost at right angles in weight space (cosine ~0 even for two realism LoRAs), so a
    plain cosine says nothing. What matters is the part of each change that lands in the other's directions (in and
    out): overlap = that share (two random rank-32 LoRAs share ~0.001, a LoRA with itself 1). direction = cosine
    inside the shared part: + = they push the same way there (stacking doubles it), - = against (they cancel)."""
    def basis(ad, shp):
        lr = _lowrank(ad, key, None, dev)
        if lr is not None:
            u, s, vh = _core_svd(*lr)
            return u, s, vh.T
        if shp is None or len(shp) < 2:
            return None
        d = _full(ad, key, shp, dev).reshape(shp[0], -1)
        q = min(40, min(d.shape))
        cuda = [d.device.index or 0] if d.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda):
            torch.manual_seed(0)
            u, s, v = torch.svd_lowrank(d, q=q, niter=2)
        k = min(32, q)
        return u[:, :k], s[:k], v[:, :k]

    ba, bb = basis(a, shape), basis(b, shape)
    if ba is None or bb is None:
        return None
    (ua, sa, va), (ub, sb, vb) = ba, bb
    ea, eb = float((sa ** 2).sum()), float((sb ** 2).sum())
    if ea < 1e-24 or eb < 1e-24:
        return None
    m_out, m_in = ua.T @ ub, va.T @ vb          # how A's directions line up with B's, out and in
    x = m_out * sb[None, :] @ m_in.T             # B's change seen inside A's directions
    y = m_out.T * sa[None, :] @ m_in             # A's change seen inside B's directions
    nx, ny = float(x.norm()), float(y.norm())
    overlap = 0.5 * (nx * nx / eb + ny * ny / ea)
    dot = float((sa * x.diagonal()).sum())       # <A, B> = trace(diag(sa) x)
    den = (nx * ea ** 0.5 * ny * eb ** 0.5) ** 0.5
    direction = dot / den if den > 1e-24 else 0.0
    return overlap, max(-1.0, min(1.0, direction)), ea ** 0.5, eb ** 0.5


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
    if isinstance(adapter, tuple) and adapter[0] == "diff":  # a plain weight difference (e.g. a bias from LC LoRA Extract)
        return adapter[1][0].to(dev, torch.float32).reshape(shape)
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
    # svd_lowrank is randomised: same input, same result. fork_rng puts the global RNG back afterwards, so the
    # fixed seed does not leak into other nodes
    cuda = [delta.device.index or 0] if delta.device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda):
        torch.manual_seed(0)
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
                "lora_a": (loras, {"tooltip": "LoRA A: the main LoRA (body, style). Bars at 0 = only this one."}),
                "lora_b": (loras, {"tooltip": "LoRA B: the one you want part of (a face, a look). Bars at 1 = only this one."}),
                "strength_a": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05,
                               "tooltip": "A's strength before the merge, like a LoRA loader. Applies in every block, "
                                          "then the bars split it."}),
                "strength_b": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05,
                               "tooltip": "B's strength before the merge, like a LoRA loader. Applies in every block, "
                                          "then the bars split it."}),
                "mix": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                        "tooltip": "One slider for the whole merge. 0.5 = the bars as set. Toward 0 fades B out "
                                   "everywhere (0.25 = A full, B at half), toward 1 fades A out."}),
                "method": (["sum", "ties", "balanced"], {"default": "sum",
                           "tooltip": "sum: both added, exactly. Same picture as two LoRA loaders. Start here.\n"
                                      "balanced: each LoRA brought to the same size first, so a loud one can't drown a "
                                      "quiet one. Use it when the two overcook together.\n"
                                      "ties: keeps each LoRA's strongest changes and drops the ones that fight. For two "
                                      "styles or two characters that clash. Slower."}),
                "preset": (PRESET_NAMES, {"default": "Both full",
                           "tooltip": "Starting curve. Face from B (x-y) = B inside those blocks, A outside. Where a "
                                      "LoRA keeps its face differs per LoRA: check the LoRA B heatmap. Dragging a bar "
                                      "switches to Custom."}),
                "blocks": ("STRING", {"default": ", ".join(["0.5"] * NUM_BLOCKS),
                                      "tooltip": "28 block values: 0 = only A, 0.5 = both full, 1 = only B. Drag the bars on the node."}),
                "ties_density": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
                                 "tooltip": "ties only: how much of each LoRA survives (its strongest changes). 0.5 is a "
                                            "good start. Lower = only the most important changes."}),
            },
            "optional": {
                "blocks_in": ("STRING", {"forceInput": True,
                                         "tooltip": "A 28-block curve from another node. When wired it replaces the preset "
                                                    "and the bars. Dragging a bar unplugs it."}),
                "lora_c": (extra, {"default": "None", "tooltip": "Optional third LoRA at strength_c in every block, not on "
                                                                 "the bars. Good for a detail or lighting LoRA."}),
                "strength_c": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05,
                               "tooltip": "lora_c's strength, the same in every block."}),
                "lora_d": (extra, {"default": "None", "tooltip": "Optional fourth LoRA at strength_d in every block, not on "
                                                                 "the bars."}),
                "strength_d": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05,
                               "tooltip": "lora_d's strength, the same in every block."}),
            },
        }

    RETURN_TYPES = ("MODEL", "LC_LORA")
    RETURN_NAMES = ("model", "merged_lora")
    OUTPUT_TOOLTIPS = ("The model with the merged LoRA applied.", "The merged LoRA, for LC LoRA Save.")
    FUNCTION = "merge"
    CATEGORY = "LC ModelBuilder/Krea2"
    DESCRIPTION = ("Merges two to four Krea 2 LoRAs with a per-block A / B curve (sum, TIES or balanced), applies the result to the "
                   "model and outputs it for LC LoRA Save. The heatmaps show A's and B's share, the merged result, and where A and B "
                   "agree or fight.")

    ARCH = KREA2
    PRESET_TABLE = PRESETS

    def merge(self, model, lora_a, lora_b, strength_a, strength_b, method, preset, blocks, ties_density, mix=0.5,
              blocks_in=None, lora_c="None", strength_c=1.0, lora_d="None", strength_d=1.0, **_old):
        # _old: settings from earlier saves that no longer exist (balance)
        arch = self.ARCH
        curve = _parse01(blocks_in, arch.n) if blocks_in else (self.PRESET_TABLE.get(preset) or _parse01(blocks, arch.n))
        dev = comfy.model_management.get_torch_device()
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        model_sd = None
        sliced = [0, 0]  # slice layers widened to the whole layer, and ones that could not be (kept as the last slice)

        def load(name):
            nonlocal model_sd
            p = comfy.lora.load_lora(_load_lora_file(folder_paths.get_full_path_or_raise("loras", name))[0], key_map,
                                     log_missing=False)
            out, parts = {}, {}
            for k, v in p.items():
                if isinstance(k, str) or len(k) < 2 or k[1] is None:
                    out[_key_of(k)] = v
                else:  # a slice of a fused layer (MOSS q / k / v in qkv): collected, then widened below
                    parts.setdefault(k[0], []).append((k[1] if len(k) == 2 else None, v))
            for key, entries in parts.items():
                if model_sd is None:
                    model_sd = model.model.state_dict()
                w = model_sd.get(key)
                ad = _widen(entries, tuple(w.shape)) if w is not None and w.ndim == 2 else None
                if ad is None:
                    sliced[1] += 1
                    out[key] = entries[-1][1]
                else:
                    sliced[0] += 1
                    out[key] = ad
            return out

        # (patches by layer, weight from the block's (wa, wb)): A and B follow the curve, C and D their own strength
        used = [lora_a, lora_b]
        src = [(load(lora_a), lambda wa, wb: wa * float(strength_a)), (load(lora_b), lambda wa, wb: wb * float(strength_b))]
        for name, st in ((lora_c, strength_c), (lora_d, strength_d)):
            if name and name != "None" and abs(float(st)) > 1e-6:
                used.append(name)
                src.append((load(name), lambda wa, wb, st=float(st): st))
        layers, patches = {}, {}
        maps = {n: [[0.0] * arch.n for _ in arch.rows] for n in ("a", "b", "m", "ov", "dir")}
        outside = 0
        mix_a, mix_b = _weights(max(0.0, min(1.0, float(mix))))
        # A vs B per layer, before the curve: do they change the same thing, and the same way or against each other?
        sign_ab = 1.0 if float(strength_a) * float(strength_b) >= 0 else -1.0
        shared = []  # (overlap, direction, block) for layers where they clearly change the same thing
        for key in sorted(set().union(*(set(p) for p, _ in src))):
            comfy.model_management.throw_exception_if_processing_interrupted()
            where = arch.where(key)
            pa, pb = src[0][0].get(key), src[1][0].get(key)
            if where and pa is not None and pb is not None:
                shape = None
                if not (_plain(pa) and _plain(pb)):
                    if model_sd is None:
                        model_sd = model.model.state_dict()
                    shape = tuple(model_sd[key].shape) if key in model_sd else None
                got = _inner(pa, pb, key, shape, dev)
                if got:
                    ov, direction = got[0], sign_ab * got[1]
                    b_, r_ = where
                    maps["ov"][r_][b_] = round(ov, 4)
                    maps["dir"][r_][b_] = round(direction, 3)
                    if ov > 0.1:  # about 100x what two unrelated rank-32 LoRAs share
                        shared.append((ov, direction, b_))
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
            # balanced: every part scaled to the median size first, the sum set back to that size afterwards
            bal = method == "balanced" and len(parts) > 1
            if bal:
                sizes = sorted(x[4] for x in parts)
                mid = (sizes[(len(sizes) - 1) // 2] + sizes[len(sizes) // 2]) / 2
                eq = {x[0]: mid / max(x[4], 1e-12) for x in parts}
                target = mid * max(abs(x[2]) for x in parts)
            else:
                eq = {x[0]: 1.0 for x in parts}
            if not need_full:
                up = torch.cat([x[3][0] * (x[2] * eq[x[0]]) for x in parts], 1)
                down = torch.cat([x[3][1] for x in parts], 0)
                if bal:
                    now = _lowrank_norm(up, down)
                    if now > 1e-12:
                        up = up * (target / now)
            else:
                deltas = [fulls[x[0]] * (x[2] * eq[x[0]]) for x in parts]
                delta = _ties(deltas, float(ties_density)) if method == "ties" and len(deltas) > 1 else sum(deltas)
                if bal:
                    now = float(delta.norm())
                    if now > 1e-12:
                        delta = delta * (target / now)
                if delta.ndim < 2:  # biases and norms stay plain differences, no low-rank fit
                    d = delta.to(torch.bfloat16).cpu()
                    layers[key] = {"kind": "diff", "delta": d}
                    patches[key] = ("diff", (d,))
                    if not where:
                        outside += 1
                    continue
                # LoKr / LoHa / full differences have no rank of their own: give them the cap, or the refit squeezes
                # them down to the plain LoRAs' rank (a rank-16 LoRA + a LoKr kept only a sliver of the LoKr)
                rank = MAX_RANK if any(not _plain(x[1]) for x in parts) else (
                    sum(int(x[1].weights[1].shape[0]) for x in parts) or 32)
                up, down = _refit(delta.reshape(delta.shape[0], -1), min(MAX_RANK, rank))
                if bal:  # the refit drops some of a LoKr's spread-out change: land on the target size anyway
                    now = _lowrank_norm(up, down)
                    if now > 1e-12:
                        up = up * (target / now)
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
            note += f" · some layers do not match this model: are they all {arch.title} LoRAs?"
        if sliced[1]:
            note += f" · {sliced[1]} split layers could not be joined"
        if shared:
            per_block = {}
            for ov, _d, b_ in shared:
                per_block[b_] = per_block.get(b_, 0.0) + ov
            top = sorted(per_block, key=per_block.get, reverse=True)[:3]
            same = sum(1 for _o, d, _b in shared if d > 0.2)
            against = sum(1 for _o, d, _b in shared if d < -0.2)
            note += (f" · A and B change the same thing in {len(shared)} layers (most in blocks "
                     f"{', '.join(map(str, sorted(top)))}): {same} the same way, {against} against")
        elif any(any(row) for row in maps["ov"]):
            note += " · A and B change different weights everywhere (two looks that push the same way can still overcook)"
        lora_obj = {
            "arch": arch.name, "source": "+".join(name(u) for u in used), "layers": layers, "shapes": {},
            "metadata": {},
            "edit": {"merge": method, "lora_a": lora_a, "lora_b": lora_b, "strength_a": strength_a,
                     "strength_b": strength_b, "lora_c": lora_c, "strength_c": strength_c, "lora_d": lora_d,
                     "strength_d": strength_d, "mix": mix, "blocks": [round(v, 4) for v in curve],
                     "ties_density": ties_density},
        }
        ui = {"lc_lora_map": [{"maps": [["LoRA A", maps["a"]], ["LoRA B", maps["b"]], ["merged", maps["m"]]],
                               "conflict": {"ov": maps["ov"], "dir": maps["dir"]},
                               "rows": arch.labels, "note": note, "curve": [round(v, 4) for v in curve],
                               "wired": bool(blocks_in)}]}
        return {"ui": ui, "result": (m, lora_obj)}


ANIMA_PRESETS = _basic(NUM_BLOCKS)


class LCAnimaLoRAMerge(LCKrea2LoRAMerge):
    """The LoRA Merge for Anima: same methods, Anima's 28 blocks and layer rows."""
    ARCH = ANIMA
    PRESET_TABLE = ANIMA_PRESETS

    @classmethod
    def INPUT_TYPES(cls):
        d = super().INPUT_TYPES()
        d["required"]["model"] = ("MODEL", {"tooltip": "Anima model. The merged LoRA is applied to it, like a LoRA loader."})
        d["required"]["preset"] = ([*ANIMA_PRESETS, "Custom"], {"default": "Both full",
                                   "tooltip": "Per block: 0 = only A, 0.5 = both at full strength, 1 = only B. Dragging "
                                              "the graph switches to Custom."})
        return d

    CATEGORY = "LC ModelBuilder/Anima"
    DESCRIPTION = ("Merges two to four Anima LoRAs with a per-block A / B curve (sum, TIES or balanced), applies the result to the "
                   "model and outputs it for LC LoRA Save. The heatmaps show A's and B's share, the merged result, and where A and B "
                   "agree or fight.")


MOSS_PRESETS = _basic(MOSS.n)


class LCMOSSLoRAMerge(LCKrea2LoRAMerge):
    """The LoRA Merge for MOSS Image: same methods over MOSS's 42 layers. q / k / v LoRAs are joined into qkv."""
    ARCH = MOSS
    PRESET_TABLE = MOSS_PRESETS

    @classmethod
    def INPUT_TYPES(cls):
        d = super().INPUT_TYPES()
        r = d["required"]
        r["model"] = ("MODEL", {"tooltip": "MOSS Image model. The merged LoRA is applied to it, like a LoRA loader."})
        r["preset"] = ([*MOSS_PRESETS, "Custom"], {"default": "Both full",
                       "tooltip": "Per layer: 0 = only A, 0.5 = both at full strength, 1 = only B. Dragging the graph "
                                  "switches to Custom."})
        r["blocks"] = ("STRING", {"default": ", ".join(["0.5"] * MOSS.n),
                                  "tooltip": "42 layer values, 0 (A) to 1 (B). Drag the bars on the node."})
        d["optional"]["blocks_in"] = ("STRING", {"forceInput": True,
                                                 "tooltip": "A 42-layer curve from another node. When wired it replaces the "
                                                            "preset and the bars."})
        return d

    CATEGORY = "LC ModelBuilder/MOSS"
    DESCRIPTION = ("Merges two to four MOSS Image LoRAs with a per-layer A / B curve (sum, TIES or balanced), applies the "
                   "result to the model and outputs it for LC LoRA Save. The heatmaps show A's and B's share, the merged "
                   "result, and where A and B agree or fight.")


NODE_CLASS_MAPPINGS = {"LCKrea2LoRAMerge": LCKrea2LoRAMerge, "LCAnimaLoRAMerge": LCAnimaLoRAMerge,
                       "LCMOSSLoRAMerge": LCMOSSLoRAMerge}
NODE_DISPLAY_NAME_MAPPINGS = {"LCKrea2LoRAMerge": "LC Krea2 LoRA Merge", "LCAnimaLoRAMerge": "LC Anima LoRA Merge",
                              "LCMOSSLoRAMerge": "LC MOSS LoRA Merge"}
