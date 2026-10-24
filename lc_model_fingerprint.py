"""
LC Model Fingerprint: is a certain model, or a certain LoRA, inside this checkpoint? Reads the weights themselves, so it
works when the metadata says nothing (or lies).

- model: the checkpoint you are checking.
- reference (optional): the model you think it came from. Every layer both share is compared: identical, the same
  model at another precision, a finetune of it, or not related. A per-block chart shows a merge (some blocks from the
  reference, others not).
- lora_name (optional, needs the reference as the base): the LoRA's change is compared with the checkpoint's change
  from the base. If the LoRA was merged in, they line up, and the size of the match is roughly the merge strength.

Weights are read one layer at a time (quantized ones decoded first), so two big models do not need extra memory.
"""
from __future__ import annotations

import re
import statistics

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import comfy.lora
import comfy.model_management
import comfy.utils
import folder_paths
from comfy.weight_adapter.base import WeightAdapterBase
from comfy.weight_adapter.lora import LoRAAdapter

from .lc_model_merge_any import _dequant, _is_quant, _param, _precision

_SKIP = re.compile(r"(scale_weight|scale_input|comfy_quant|weight_scale|input_scale|\.absmax|\.quant_map)$")
_BLOCK = re.compile(r"((?:double_|single_|transformer_|input_|output_|middle_|joint_)?blocks|layers|down_blocks|up_blocks)\.(\d+)\.")
# cosine thresholds, from tests on Krealism V3.1: the same weights stored at int8 come out 0.9999, at nvfp4 0.995
CLOSE = 0.97


def _store(model, keys):
    """(how it is stored, the lowest similarity the SAME weights can show at that precision)."""
    kind = _precision(model, keys)
    t = None
    for k in keys:
        if ".blocks." in k:
            t = _param(model, k)
            if t is not None and getattr(t, "ndim", 0) == 2:
                break
    gguf = str(getattr(t, "tensor_type", "")).upper() if kind == "GGUF" else ""
    if gguf:
        kind = f"GGUF {gguf.split('.')[-1]}"
    low = kind.lower()
    if any(x in low for x in ("fp4", "w4", "int4", "q4", "q5", "q3", "q2", "4bit")):
        return kind, 0.99
    if any(x in low for x in ("fp8", "int8", "q8", "q6", "mxfp8", "quantized")):
        return kind, 0.999
    return kind, 0.9999


def _weight_keys(model):
    """Every weight of the diffusion model (biases and norms are too small to tell anything)."""
    keys = []
    for k in model.model_state_dict().keys():
        if not k.startswith("diffusion_model.") or not k.endswith(".weight") or _SKIP.search(k):
            continue
        keys.append(k)
    return keys


def _full(model, key, device):
    """One weight as the model really uses it: decoded if quantized, with any LoRA from a LoRA loader added on top."""
    t = _param(model, key)
    if t is None:
        return None
    w = _dequant(t, device).to(device, torch.float32) if _is_quant(t) else t.to(device, torch.float32)
    patches = getattr(model, "patches", {}).get(key)
    if patches:
        w = comfy.lora.calculate_weight(patches, w, key, intermediate_dtype=torch.float32)
    return w


def _lora_delta(p, mkey, key, base, device):
    """One LoRA layer's change to `base` at strength 1, or None when it cannot be read.
    A plain LoRA uses up x down x alpha / rank. LoHa, LoKr, LoCon with a mid, DoRA and the rest go through the adapter's
    own calculate_weight on the base weight (patched minus base), so every format is read the way ComfyUI applies it."""
    ws = getattr(p, "weights", None)
    if isinstance(p, LoRAAdapter) and len(ws) > 4 and ws[3] is None and ws[4] is None:
        up, down = ws[0].to(device, torch.float32), ws[1].to(device, torch.float32)
        alpha = float(ws[2]) if ws[2] is not None else down.shape[0]
        try:
            return (up.flatten(1) @ down.flatten(1)).reshape(base.shape) * (alpha / down.shape[0])
        except RuntimeError:
            return None
    if isinstance(mkey, tuple):  # a slice of a fused layer: not compared
        return None
    if isinstance(p, tuple):  # an old-style ("diff", (tensor,)) patch
        if len(p) > 1 and p[0] == "diff" and p[1] and isinstance(p[1][0], torch.Tensor):
            try:
                return p[1][0].to(device, torch.float32).reshape(base.shape)
            except RuntimeError:
                return None
        return None
    if not isinstance(p, WeightAdapterBase):
        return None
    try:
        patched = p.calculate_weight(base.clone(), key, 1.0, 1.0, None, lambda t: t, torch.float32)
    except Exception:
        return None
    if patched is None or patched.shape != base.shape:
        return None
    return patched.to(device, torch.float32) - base


def _block_of(key: str) -> str:
    m = _BLOCK.search(key)
    return f"{m.group(1)} {int(m.group(2)):02d}" if m else "other"


def _family(model) -> str:
    name = type(model.model).__name__
    n = sum(1 for _ in _weight_keys(model))
    return f"{name} ({n} weight layers)"


def _chart(rows, title, zoom=True):
    """rows: [(label, value 0..1, color)] -> a bar chart image tensor. zoom: the bars start just under the lowest value,
    so small differences near 1.0 show (the scale is printed in the title)."""
    lo = 0.0
    if zoom and rows:
        m = min(v for _, v, _ in rows)
        lo = max(0.0, m - 0.25 * (1.0 - m)) if m < 1.0 else 0.0
        title = f"{title}   (bar scale {lo:.4f} to 1.0)"
    w, row_h, label_w = 760, 18, 150
    h = 40 + row_h * max(1, len(rows))
    im = Image.new("RGB", (w, h), (24, 24, 24))
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("arial.ttf", 12)
        fb = ImageFont.truetype("arialbd.ttf", 14)
    except OSError:
        f = fb = ImageFont.load_default()
    d.text((10, 10), title, fill=(255, 200, 80), font=fb)
    for i, (label, v, col) in enumerate(rows):
        y = 36 + i * row_h
        d.text((10, y + 2), label[:22], fill=(210, 210, 210), font=f)
        bw = int((w - label_w - 70) * max(0.0, min(1.0, (v - lo) / (1.0 - lo) if lo < 1.0 else v)))
        d.rectangle([label_w, y + 3, label_w + bw, y + row_h - 3], fill=col)
        d.text((label_w + bw + 6, y + 2), f"{v:.4f}", fill=(200, 200, 200), font=f)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None]


def _color(cos, same):
    return (90, 200, 120) if cos >= same else (230, 190, 70) if cos >= CLOSE else (220, 90, 80)


class LCModelFingerprint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The checkpoint you are checking. Load it with your usual loader (any precision, GGUF too)."}),
            },
            "optional": {
                "reference": ("MODEL", {"tooltip": "The model you think it came from. Needed to compare, and as the base for the LoRA check."}),
                "lora_name": (["none"] + folder_paths.get_filename_list("loras"), {"default": "none",
                              "tooltip": "A LoRA you think was merged into the checkpoint. Needs the reference: the base model the LoRA was made for."}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("report", "chart")
    OUTPUT_TOOLTIPS = ("What was found, in plain words, with the numbers behind it.",
                       "Similarity per block (green = the same weights, yellow = close, red = different). With a LoRA: how strongly it shows per block.")
    FUNCTION = "check"
    OUTPUT_NODE = True
    CATEGORY = "LC Modelbuilder"
    DESCRIPTION = ("Is a model, or a LoRA, inside this checkpoint? Compares the weights themselves, so it works without "
                   "metadata: the same model at another precision, a finetune of it, a block merge, or a LoRA merged in "
                   "(with its rough strength).")

    def check(self, model, reference=None, lora_name="none"):
        device = comfy.model_management.get_torch_device()
        lines = [f"Model: {_family(model)}"]
        if reference is None:
            lines.append("Connect a reference model to compare (and as the base for the LoRA check).")
            text = "\n".join(lines)
            return {"ui": {"text": [text]}, "result": (text, _chart([], "No reference connected"))}
        lines.append(f"Reference: {_family(reference)}")
        keys = _weight_keys(model)
        ref_keys = set(_weight_keys(reference))
        (kind_m, floor_m), (kind_r, floor_r) = _store(model, keys), _store(reference, list(ref_keys))
        SAME = min(floor_m, floor_r)
        lines.append(f"Stored as: {kind_m} vs {kind_r}. At that precision the same weights still read {SAME} or higher.")
        shared = [k for k in keys if k in ref_keys]
        if not shared:
            lines.append("No layers in common: a different model family, so it was not built on the reference.")
            text = "\n".join(lines)
            return {"ui": {"text": [text]}, "result": (text, _chart([], "No layers in common"))}
        per_block, coss, exact, shape_diff = {}, [], 0, 0
        pbar = comfy.utils.ProgressBar(len(shared))
        for k in shared:
            a, b = _full(model, k, device), _full(reference, k, device)
            pbar.update(1)
            if a is None or b is None:
                continue
            if a.shape != b.shape:
                shape_diff += 1
                continue
            if torch.equal(a, b):
                exact += 1
                c = 1.0
            else:
                c = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
            coss.append(c)
            per_block.setdefault(_block_of(k), []).append(c)
        if not coss:
            lines.append("The shared layers all have different shapes: not built on the reference.")
            text = "\n".join(lines)
            return {"ui": {"text": [text]}, "result": (text, _chart([], "Different shapes"))}
        med, low = statistics.median(coss), min(coss)
        same_share = sum(c >= SAME for c in coss) / len(coss)
        blocks = sorted(per_block.items(), key=lambda kv: kv[0])
        bmed = {b: statistics.median(v) for b, v in blocks}
        same_blocks = [b for b, v in bmed.items() if v >= SAME]
        diff_blocks = [b for b, v in bmed.items() if v < CLOSE]
        lines.append(f"Compared {len(coss)} shared layers ({exact} bit-identical{f', {shape_diff} with different shapes' if shape_diff else ''}).")
        lines.append(f"Similarity (cosine, 1.0 = the same weights; {SAME} and up = the same model at these precisions): "
                     f"median {med:.5f}, lowest {low:.5f}, {same_share:.0%} of layers the same.")
        if exact == len(coss):
            verdict = "The same weights, bit for bit."
        elif same_share >= 0.98:
            verdict = ("The same model" + (", stored at a different precision" if kind_m != kind_r else "")
                       + ". A very light change (a LoRA merged in at low strength) also reads like this: use the LoRA check.")
        elif same_blocks and diff_blocks:
            verdict = (f"A block merge: {len(same_blocks)} block(s) are the reference's own weights, {len(diff_blocks)} are not. "
                       "See the chart for which.")
        elif med >= CLOSE:
            verdict = "Built on the reference: a finetune or a merge that kept it close."
        else:
            verdict = "Not built on the reference (or changed so much it no longer shows)."
        lines.append(f"Verdict: {verdict}")
        rows = [(b, v, _color(v, SAME)) for b, v in bmed.items()]
        title, zoom = "Similarity to the reference per block", True
        if lora_name and lora_name != "none":
            lora_lines, rows_l = self._lora(model, reference, lora_name, device)
            lines += [""] + lora_lines
            if rows_l:
                rows, title, zoom = rows_l, f"LoRA match per block: {lora_name.split(chr(92))[-1].split('/')[-1]}", False
        text = "\n".join(lines)
        print("[LC Model Fingerprint]\n" + text)
        return {"ui": {"text": [text]}, "result": (text, _chart(rows, title, zoom))}

    def _lora(self, model, reference, lora_name, device):
        """Each LoRA layer's change (up x down x alpha / rank) against the checkpoint's change from the base."""
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        key_map = comfy.lora.model_lora_keys_unet(reference.model, {})
        patches = comfy.lora.load_lora(sd, key_map, log_missing=False)
        per_block, corrs, strengths, weights = {}, [], [], []
        for mkey, p in patches.items():
            key = mkey[0] if isinstance(mkey, tuple) else mkey
            a, b = _full(model, key, device), _full(reference, key, device)
            if a is None or b is None or a.shape != b.shape:
                continue
            lw = _lora_delta(p, mkey, key, b, device)
            if lw is None:
                continue
            delta = (a - b).flatten()
            lw = lw.flatten()
            ll = float(lw @ lw)
            if ll <= 0:
                continue
            dl = float(delta @ lw)
            dd = float(delta @ delta)
            corr = dl / ((dd * ll) ** 0.5) if dd > 0 else 0.0
            corrs.append(corr)
            strengths.append(dl / ll)
            weights.append(ll ** 0.5)
            per_block.setdefault(_block_of(key), []).append(corr)
        name = lora_name.split("\\")[-1].split("/")[-1]
        if not corrs:
            return [f"LoRA {name}: none of its layers match this model family, so it cannot be inside it."], []
        tot = sum(weights)
        corr = sum(c * w for c, w in zip(corrs, weights)) / tot
        strength = statistics.median(strengths)
        lines = [f"LoRA {name}: {len(corrs)} of its layers checked.",
                 f"Match (how well the checkpoint's change from the reference lines up with the LoRA's change; 0 = not "
                 f"at all, 1 = it is exactly the LoRA): {corr:.3f}. Estimated merge strength: {strength:.2f}."]
        if corr >= 0.5 and strength >= 0.1:
            lines.append(f"Verdict: the LoRA is in it, merged at about {strength:.2f}.")
        elif corr >= 0.2 and strength >= 0.05:
            lines.append(f"Verdict: probably in it, weakly (about {strength:.2f}) or mixed with other changes.")
        else:
            lines.append("Verdict: not found.")
        rows = [(b, max(0.0, statistics.median(v)), (120, 160, 230)) for b, v in sorted(per_block.items())]
        return lines, rows


NODE_CLASS_MAPPINGS = {"LCModelFingerprint": LCModelFingerprint}
NODE_DISPLAY_NAME_MAPPINGS = {"LCModelFingerprint": "LC Model Fingerprint 🔍"}
