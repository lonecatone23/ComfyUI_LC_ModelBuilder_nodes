"""
LC LoRA Extract
---------------
Turns the difference between a finetuned model and the model it was built on into a LoRA:
finetuned - base, per layer, compressed with SVD. Either model can be stored in any precision
(bf16, fp8, int8, nvfp4, GGUF): weights are decoded to full precision first.

parts picks what goes in: all layers, the numbered blocks only, or a custom block list.
Layers that did not change (or only by rounding) are left out of the file entirely, so it
only holds what the finetune actually changed. Same key layout as ComfyUI's own LoRA saver
(diffusion_model.<layer>.lora_up / lora_down, .diff_b for biases), so every loader reads it.
"""

from __future__ import annotations

import json
import os
import re

import torch
from safetensors.torch import save_file

import comfy.model_management as mm
import folder_paths

from .lc_checkpoint_save import EMBED_WORKFLOW_INPUT, ENABLED_INPUT, METADATA_INPUT, trim_metadata, workflow_metadata
from .lc_model_merge_any import _SKIP_SUFFIX, _dequant, _is_quant, _param, _quantized

PARTS = ["all layers", "blocks only", "custom blocks"]
_BLOCK = re.compile(r"(?:^|\.)(?:blocks|layers)\.(\d+)\.")


def _parse_blocks(text):
    """'14-27' or '4-9, 16, 20-22' -> set of block numbers."""
    out = set()
    for part in str(text or "").replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _full(model, key, device, live=False):
    t = _param(model, key)
    if t is None:
        return None
    w = (_dequant(t, device) if _is_quant(t) else t).to(device, torch.float32)
    if live:
        # the weight as it really runs: LoRAs, merges and other patches on the MODEL wire, then weight wrappers
        # (anything that scales a weight), applied with ComfyUI's own patch maths
        import comfy.lora

        patches = getattr(model, "patches", {}).get(key)
        if patches:
            w = comfy.lora.calculate_weight(patches, w.clone(), key, intermediate_dtype=torch.float32)
        for f in getattr(model, "weight_wrapper_patches", {}).get(key, []):
            w = f(w)
    return w


class LCLoRAExtract:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "finetuned": ("MODEL", {"tooltip": "The finetune or merge (e.g. Krealism). Any precision."}),
                "base": ("MODEL", {"tooltip": "The model it was built on (e.g. Krea 2 Turbo). Any precision."}),
                "parts": (PARTS, {"default": "all layers",
                          "tooltip": "all layers = the whole difference. blocks only = just the numbered blocks "
                                     "(on Krea 2 this leaves out text fusion and the embedding layers; use it for a "
                                     "LoRA meant for Krea 2 Raw). custom blocks = the list below."}),
                "blocks": ("STRING", {"default": "14-27",
                           "tooltip": "Used with custom blocks: ranges and single numbers, e.g. 14-27 or 4-9, 16, 20-22."}),
                "rank": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1,
                         "tooltip": "Highest rank any layer may use. Layers that need less stay smaller."}),
                "keep": ("FLOAT", {"default": 0.90, "min": 0.50, "max": 1.0, "step": 0.01,
                         "tooltip": "How much of each layer's change to keep (0.90 = 90%). Higher = closer to the "
                                    "finetune, bigger file. Krealism: 0.90 at rank 256 matched the full model."}),
                "min_change": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 0.1, "step": 0.0005,
                               "tooltip": "Layers that changed less than this (relative to the layer) count as unchanged "
                                          "and are left out. 0.001 skips bf16 rounding noise."}),
                "filename": ("STRING", {"default": "Krea 2/Your Loras/extracted_lora",
                             "tooltip": "Saved under models/loras. A number is added instead of overwriting."}),
                "save_dtype": (["bf16", "fp16"], {"default": "bf16"}),
                "enabled": ENABLED_INPUT,
                "embed_workflow": EMBED_WORKFLOW_INPUT,
            },
            "optional": {
                "metadata": METADATA_INPUT,
                "applied_loras": ("BOOLEAN", {"default": False, "label_on": "included", "label_off": "file weights only",
                                  "tooltip": "included = read each layer as it really runs, with the LoRAs, merges and patches "
                                             "on the MODEL wire applied (LC LoRA Loader, LoRA Editor...). "
                                             "Extract 'Krealism + 3 LoRAs' as one LoRA. file weights only = the model files' own "
                                             "weights (the old behaviour)."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True
    FUNCTION = "extract"
    CATEGORY = "LC Modelbuilder"
    DESCRIPTION = ("Makes a LoRA from the difference between a finetuned model and its base (e.g. Krealism minus Krea 2 "
                   "Turbo). Any precision on either side. Unchanged layers are left out of the file.")

    def extract(self, finetuned, base, parts, blocks, rank, keep, min_change, filename, save_dtype,
                enabled=True, embed_workflow=True, metadata="keep all", applied_loras=False, prompt=None, extra_pnginfo=None):
        if not enabled:
            msg = "Extract is off (enabled = false): nothing extracted or written."
            return {"ui": {"text": [msg]}, "result": (msg,)}
        device = mm.get_torch_device()
        dtype = torch.bfloat16 if save_dtype == "bf16" else torch.float16
        wanted = _parse_blocks(blocks) if parts == "custom blocks" else None
        keys = [k for k in base.model_state_dict(filter_prefix="diffusion_model.")
                if k.endswith((".weight", ".bias")) and not k.endswith(_SKIP_SUFFIX)]

        out, kept, skipped_same, retained = {}, 0, 0, []
        for key in keys:
            m = _BLOCK.search(key)
            if parts == "blocks only" and not m:
                continue
            if wanted is not None and (not m or int(m.group(1)) not in wanted):
                continue
            w_base = _full(base, key, device, live=applied_loras)
            w_ft = _full(finetuned, key, device, live=applied_loras)
            if w_base is None or w_ft is None or w_base.shape != w_ft.shape:
                continue
            d = w_ft - w_base
            rel = float(d.norm() / w_base.norm().clamp(min=1e-12))
            if rel <= min_change:
                skipped_same += 1
                continue
            name = key[: -len(".weight")] if key.endswith(".weight") else key[: -len(".bias")]
            if key.endswith(".bias"):
                out[f"{name}.diff_b"] = d.to(dtype).cpu().contiguous()
            elif d.ndim != 2:
                out[f"{name}.diff"] = d.to(dtype).cpu().contiguous()
            else:
                u, s, vh = torch.linalg.svd(d, full_matrices=False)
                energy = torch.cumsum(s * s, 0) / (s * s).sum().clamp(min=1e-20)
                r = int(torch.searchsorted(energy, torch.tensor(keep * keep, device=energy.device))) + 1
                r = max(1, min(r, rank, s.numel()))
                retained.append(float(energy[r - 1].sqrt()))
                out[f"{name}.lora_up.weight"] = (u[:, :r] * s[:r]).to(dtype).cpu().contiguous()
                out[f"{name}.lora_down.weight"] = vh[:r].to(dtype).cpu().contiguous()
            kept += 1

        if not out:
            raise ValueError("LC LoRA Extract: no layer differs between the two models (or none in the chosen parts).")

        root = folder_paths.get_folder_paths("loras")[0]
        stem = os.path.join(root, filename.replace("\\", "/").strip("/"))
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        path, n = stem + ".safetensors", 1
        while os.path.exists(path):
            path, n = f"{stem}_{n:05d}.safetensors", n + 1
        meta = {"lc_lora_extract": json.dumps({"parts": parts, "blocks": blocks if wanted is not None else "",
                                                "rank": rank, "keep": keep, "min_change": min_change,
                                                "applied_loras": bool(applied_loras)})}
        if embed_workflow:
            meta.update(workflow_metadata(prompt, extra_pnginfo))
        meta = trim_metadata(meta, metadata)
        save_file(out, path, metadata=meta)

        size = os.path.getsize(path) / 1e9
        avg = sum(retained) / len(retained) if retained else 1.0
        low = sum(1 for x in retained if x < keep - 0.005)
        lines = [f"saved {os.path.relpath(path, root)} ({size:.2f} GB)",
                 f"parts: {parts}" + (f" ({blocks})" if wanted is not None else "")
                 + ("; LoRAs and patches on the wire included" if applied_loras else ""),
                 f"{kept} layers changed and saved, {skipped_same} unchanged left out",
                 f"kept {avg:.0%} of the change on average" + (f"; {low} layers hit the rank {rank} cap" if low else "")]
        if _quantized(finetuned, keys) or _quantized(base, keys):
            lines.append("⚠️ a quantized model (fp8 / int8 / nvfp4 / GGUF) adds its rounding to every layer, so unchanged "
                         "layers look changed and the file grows. Use bf16 versions of both models when you can.")
        report = "\n".join(lines)
        print("[LC LoRA Extract] " + report.replace("\n", " | "))
        return {"ui": {"text": [report]}, "result": (report,)}


NODE_CLASS_MAPPINGS = {"LCLoRAExtract": LCLoRAExtract}
NODE_DISPLAY_NAME_MAPPINGS = {"LCLoRAExtract": "LC LoRA Extract"}
