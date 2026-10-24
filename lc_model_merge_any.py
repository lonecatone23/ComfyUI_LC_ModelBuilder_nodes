"""
LC Model Merge (any precision)
------------------------------
Merges two models of the same architecture when they are stored in different
precisions: bf16 / fp16 / fp32 with fp8 (scaled), int8, nvfp4 or GGUF, in either order.

Core ModelMergeSimple patches model2's raw tensors onto model1. With a quantized model2
that hands over packed data plus its scale keys, and a full-precision model1 has nowhere
to put those ('Linear' object has no attribute 'weight_scale'). With a quantized model1
the merge runs but the result is rounded back to model1's format.

Here the quantized side is decoded to full precision first, only real weights that exist
in both models are merged (scale / quant keys are left out), and the full-precision model
becomes the base, so the result stays full precision. Two quantized models merge onto
model1 and the result keeps model1's format (the node says so).
"""

from __future__ import annotations

import re
import sys

import torch

import comfy.model_management as mm
import comfy.utils

_QUANT_CLASSES = ("QuantizedTensor", "GGMLTensor")
_SKIP_SUFFIX = ("weight_scale", "weight_scale_2", "input_scale", "pre_quant_scale", "scale_weight",
                "scale_input", "comfy_quant")
_BLOCK = re.compile(r"\.blocks\.(\d+)\.")


def _is_quant(t) -> bool:
    if any(c.__name__ in _QUANT_CLASSES for c in type(t).__mro__):
        return True
    return isinstance(t, torch.Tensor) and t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8, torch.uint8)


def _gguf_dequant():
    """The GGUF pack's own decoder, if that pack is loaded (its folder name has a hyphen, so look it up)."""
    for name, mod in list(sys.modules.items()):
        if name.endswith("dequant") and hasattr(mod, "dequantize_tensor"):
            return mod.dequantize_tensor
    return None


def _dequant(t, device) -> torch.Tensor:
    """Full-precision copy of one weight, returned as bf16 on the CPU."""
    kind = next((c.__name__ for c in type(t).__mro__ if c.__name__ in _QUANT_CLASSES), None)
    if kind == "GGMLTensor":
        fn = _gguf_dequant()
        if fn is None:
            raise RuntimeError("model has GGUF weights but the ComfyUI-GGUF pack is not loaded")
        out = fn(t.to(device), dtype=torch.float32)
    elif kind == "QuantizedTensor":
        try:
            out = t.to(device).dequantize()
        except Exception:
            out = t.dequantize()
    else:
        out = t.to(device, torch.float32)
    return out.to(torch.bfloat16).cpu()


def _param(model, key):
    """The live weight of one layer. model_state_dict() flattens quantized weights into raw numbers plus separate
    scale keys; the layer itself still holds a QuantizedTensor / GGMLTensor that decodes correctly.
    A model a sampler has loaded holds its patches (LoRAs, merges) baked into the layer, with the original kept in
    backup: read that one, like core's get_key_patches, so patches are not counted twice."""
    bk = getattr(model, "backup", {}).get(key)
    if bk is not None:
        return bk.weight
    try:
        return comfy.utils.get_attr(model.model, key)
    except AttributeError:
        return None


def _precision(model, keys) -> str:
    for k in keys:
        if ".blocks." in k and k.endswith(".weight"):
            t = _param(model, k)
            if t is None or getattr(t, "ndim", 0) != 2:
                continue
            kind = next((c.__name__ for c in type(t).__mro__ if c.__name__ in _QUANT_CLASSES), None)
            if kind == "GGMLTensor":
                return "GGUF"
            if kind == "QuantizedTensor":
                return f"quantized ({getattr(t, 'layout_cls', type(t)).__name__.strip('_').replace('Layout', '')})"
            return str(t.dtype).replace("torch.", "")
    return "unknown"


def _quantized(model, keys) -> bool:
    for k in keys:
        if ".blocks." in k and k.endswith(".weight"):
            t = _param(model, k)
            if t is not None and getattr(t, "ndim", 0) == 2:
                return _is_quant(t)
    return False


def _parse_blocks(text: str):
    vals = []
    for part in str(text or "").replace("\n", ",").split(","):
        part = part.strip()
        if part:
            vals.append(max(0.0, min(1.0, float(part))))
    return vals


class LCModelMergeAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model1": ("MODEL", {"tooltip": "Any precision: bf16, fp16, fp32, fp8, int8, nvfp4 or GGUF."}),
                "model2": ("MODEL", {"tooltip": "Same architecture as model1, any precision."}),
                "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                          "tooltip": "0 = all model1, 1 = all model2, 0.5 = half and half."}),
            },
            "optional": {
                "block_ratios": ("STRING", {"default": "", "multiline": False,
                                 "tooltip": "Optional per-block ratios for models with numbered blocks (Krea 2 has 28): "
                                            "comma list, block 0 first, same scale as ratio. Layers outside the "
                                            "blocks use ratio. Empty = ratio everywhere."}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "merge"
    CATEGORY = "LC Modelbuilder"
    DESCRIPTION = ("Merges two models stored in different precisions (bf16 with fp8, int8, nvfp4 or GGUF, in either "
                   "order). The quantized one is decoded to full precision first and the full-precision model is the "
                   "base, so the result stays full precision. Save it with LC Diffusion Model Save.")

    def merge(self, model1, model2, ratio, block_ratios=""):
        k1 = [k for k in model1.model_state_dict(filter_prefix="diffusion_model.") if not k.endswith(_SKIP_SUFFIX)]
        k2 = [k for k in model2.model_state_dict(filter_prefix="diffusion_model.") if not k.endswith(_SKIP_SUFFIX)]
        p1, p2 = _precision(model1, k1), _precision(model2, k2)
        q1, q2 = _quantized(model1, k1), _quantized(model2, k2)

        # the full-precision model is the base, so the result is not rounded to a quantized format
        swapped = q1 and not q2
        base, other, keys_base = (model2, model1, k2) if swapped else (model1, model2, k1)
        blocks = _parse_blocks(block_ratios)

        def share(key):  # how much of model2 this key gets
            m = _BLOCK.search(key)
            r = blocks[int(m.group(1))] if m and blocks and int(m.group(1)) < len(blocks) else ratio
            return 1.0 - r if swapped else r  # the patch source is "other": model1 when swapped

        device = mm.get_torch_device()
        groups: dict[float, dict] = {}
        merged = skipped = 0
        for key in keys_base:
            t_base, t_other = _param(base, key), _param(other, key)
            if t_base is None or t_other is None:
                continue
            if tuple(t_other.shape) != tuple(t_base.shape):
                skipped += 1
                continue
            w = _dequant(t_other, device) if _is_quant(t_other) else t_other
            groups.setdefault(round(share(key), 6), {})[key] = (w,)
            merged += 1

        if merged < 0.5 * max(1, sum(1 for k in keys_base if k.endswith(".weight"))):
            raise ValueError(f"LC Model Merge (any precision): only {merged} matching weights. "
                             "Are model1 and model2 the same architecture?")

        out = base.clone()
        for s, patches in groups.items():
            if s > 0.0:
                out.add_patches(patches, s, 1.0 - s)

        result = (p2 if swapped else p1) if not (q1 and q2) else p1
        lines = [f"model1: {p1}   model2: {p2}",
                 f"base: {'model2 (model1 is quantized)' if swapped else 'model1'}   result: {result}",
                 f"merged {merged} weights" + (f", skipped {skipped} (shape mismatch)" if skipped else "")]
        if q1 and q2:
            lines.append("both models are quantized: the result is rounded back to model1's format")
        if blocks:
            lines.append(f"block ratios: {len(blocks)} values")
        report = "\n".join(lines)
        print("[LC Model Merge (any precision)] " + report.replace("\n", " | "))
        return (out, report)


NODE_CLASS_MAPPINGS = {"LCModelMergeAny": LCModelMergeAny}
NODE_DISPLAY_NAME_MAPPINGS = {"LCModelMergeAny": "LC Model Merge (any precision)"}
