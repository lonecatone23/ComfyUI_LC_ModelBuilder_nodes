"""
LC LoRA Bake
------------
Bakes up to 4 LoRAs into a diffusion model file and saves the result as a new model: W + strength x (LoRA change).
Reads every LoRA format ComfyUI reads (PEFT lora_A / lora_B, kohya lora_up / lora_down with alpha, split q / k / v onto a
fused qkv, etc.), because the LoRA names are matched with ComfyUI's own table for the model, built on the meta device
from the file's header. The weights are patched with ComfyUI's own calculate_weight, one layer at a time.

Per-block control: blocks_curve multiplies each numbered block's strength (blocks.N / layers.N / ...), outside_blocks
the layers outside them. A layer the LoRA does not touch is copied byte for byte.

The recipe (model, LoRAs, strengths, curve) goes into the file's metadata as lc_merge.
"""

from __future__ import annotations

import json
import os
import re
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import comfy.lora
import comfy.model_detection
import comfy.utils
import folder_paths

from .lc_fingerprint_library import _DT, _header

_BLOCK = re.compile(r"(?:^|\.)(?:blocks|layers|double_blocks|single_blocks|transformer_blocks|joint_blocks)\.(\d+)\.")
_FULL = {"BF16", "F16", "F32"}
SLOTS = 4


def _meta_model(path):
    """(model on the meta device, file prefix) from the file header: no weights loaded."""
    h, _ = _header(path)
    sd = {k: torch.empty(v["shape"], dtype=_DT.get(v["dtype"], torch.float32), device="meta") for k, v in h.items()}
    prefix = "model.diffusion_model." if any(k.startswith("model.diffusion_model.") for k in sd) else ""
    cfg = comfy.model_detection.model_config_from_unet(sd, prefix)
    if cfg is None:
        raise ValueError("ComfyUI does not recognise this model, so it cannot match the LoRA names to its layers.")
    model = cfg.get_model({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}, "", device="meta")
    # the LoRA name table reads model.state_dict(). Under --fast / dynamic VRAM ComfyUI's ops build Linear layers
    # without registered weights, so the table missed every attention and MLP layer: name the file's own tensors instead.
    names = {"diffusion_model." + k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    model.state_dict = lambda *a, **kw: names
    return model, prefix, h


def parse_curve(text):
    return [float(x) for x in re.split(r"[,\s]+", str(text or "").strip()) if x.strip()]


def bake(model_path, loras, out_path, blocks_curve="", outside_blocks=1.0, device="cpu", log=print):
    """loras: [(path, strength)]. Returns an info dict."""
    t0 = time.time()
    model, prefix, header = _meta_model(model_path)
    quant = sorted({v["dtype"] for k, v in header.items() if k.endswith(".weight") and len(v["shape"]) == 2} - _FULL)
    if quant:
        raise ValueError(f"The model is stored as {', '.join(quant)}. Bake into the bf16 / fp16 / fp32 file, then quantize "
                         "the result with LC Model Quantize.")
    key_map = comfy.lora.model_lora_keys_unet(model, {})
    curve = parse_curve(blocks_curve)
    patches, report = {}, []
    for path, strength in loras:
        if not strength:
            continue
        lsd = comfy.utils.load_torch_file(path, safe_load=True)
        loaded = comfy.lora.load_lora(lsd, key_map, log_missing=False)
        n_lora = len({re.sub(r"\.(lora_A|lora_B|lora_up|lora_down|lora_mid|alpha|diff|diff_b|dora_scale|hada_\w+|lokr_\w+|set_weight)(\.weight)?$", "", k)
                      for k in lsd if not k.endswith(".bias")})
        hit = 0
        for k, v in loaded.items():
            key, offset, function = (k, None, None) if isinstance(k, str) else (k[0], k[1], k[2] if len(k) > 2 else None)
            m = _BLOCK.search(key)
            mult = (curve[int(m.group(1))] if m and int(m.group(1)) < len(curve) else 1.0) if m else outside_blocks
            if not mult:
                continue
            patches.setdefault(key, []).append((strength * mult, v, 1.0, offset, function))
            hit += 1
        report.append({"name": os.path.basename(path), "strength": strength, "matched": len(loaded), "baked": hit, "lora_modules": n_lora})
        log(f"[LC LoRA Bake] {os.path.basename(path)}: {len(loaded)} of {n_lora} LoRA modules matched, {hit} baked")
        if not loaded:
            lk = sorted(lsd)[:3]
            mk = sorted(k for k in key_map if isinstance(k, str) and k.startswith("diffusion_model."))[:3]
            raise ValueError(f"{os.path.basename(path)} matched no layer of this model ({type(model).__name__}, {len(key_map)} names): "
                             f"it was made for another model family. LoRA names look like {lk}, the model's like {mk}.")
    out, changed = {}, 0
    dev = torch.device("cuda") if device == "gpu" and torch.cuda.is_available() else torch.device("cpu")
    with safe_open(model_path, "pt") as f:
        meta = dict(f.metadata() or {})
        for fk in f.keys():
            w = f.get_tensor(fk)
            mk = "diffusion_model." + fk[len(prefix):] if fk.startswith(prefix) else fk
            p = patches.get(mk)
            if p:
                nw = comfy.lora.calculate_weight(p, w.to(dev, torch.float32), mk, intermediate_dtype=torch.float32)
                w = nw.to(w.dtype).cpu()
                changed += 1
            out[fk] = w
    recipe = {"base": os.path.basename(model_path), "loras": report, "blocks_curve": curve or None, "outside_blocks": outside_blocks}
    meta["lc_merge"] = json.dumps(recipe)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_file(out, out_path, metadata=meta)
    return {"path": out_path, "layers_changed": changed, "seconds": round(time.time() - t0, 1), "recipe": recipe}


class LCLoRABake:
    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        req = {
            "model_name": (folder_paths.get_filename_list("diffusion_models"),
                           {"tooltip": "The bf16 / fp16 model to bake into. Quantize the result afterwards with LC Model Quantize."}),
        }
        for i in range(1, SLOTS + 1):
            req[f"lora_{i}"] = (loras, {"default": "None", "tooltip": "A LoRA made for this model family."})
            req[f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05})
        req["blocks_curve"] = ("STRING", {"default": "", "tooltip": "Optional: one number per block (comma list), multiplies the "
                                                                    "strengths in that block. 0 leaves the block alone. Empty = 1 everywhere. "
                                                                    "Same lists as the LoRA Editor and Block Scan."})
        req["outside_blocks"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                                           "tooltip": "Multiplier for the layers outside the numbered blocks (embedders, text side, final layer)."})
        req["filename"] = ("STRING", {"default": "baked/my_model_baked", "tooltip": "Saved next to the source model, .safetensors added."})
        req["device"] = (["cpu", "gpu"], {"default": "cpu", "tooltip": "cpu: no VRAM used, about a minute. gpu: faster, needs a little VRAM."})
        return {"required": req}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("path", "info")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "LC ModelBuilder"
    DESCRIPTION = ("Bakes up to 4 LoRAs into a model file and saves it as a new model, with an optional per-block curve. "
                   "Every LoRA format ComfyUI reads. The recipe is written into the file.")

    def run(self, model_name, blocks_curve, outside_blocks, filename, device, **kw):
        src = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        loras = []
        for i in range(1, SLOTS + 1):
            name = kw.get(f"lora_{i}", "None")
            if name and name != "None":
                loras.append((folder_paths.get_full_path_or_raise("loras", name), float(kw.get(f"strength_{i}", 1.0))))
        if not loras:
            raise ValueError("Pick at least one LoRA.")
        name = filename.strip().replace("\\", "/").strip("/") or "baked/model_baked"
        if not name.endswith(".safetensors"):
            name += ".safetensors"
        dst = os.path.join(os.path.dirname(src), name)
        if os.path.abspath(dst) == os.path.abspath(src):
            raise ValueError("The filename would overwrite the source model.")
        r = bake(src, loras, dst, blocks_curve, outside_blocks, device)
        parts = ", ".join(f"{x['name']} x{x['strength']} ({x['matched']} of {x['lora_modules']} modules matched)" for x in r["recipe"]["loras"])
        info = f"Saved {name} in {r['seconds']} s: {r['layers_changed']} layers changed. {parts}."
        return {"ui": {"text": [info]}, "result": (dst, info)}


NODE_CLASS_MAPPINGS = {"LCLoRABake": LCLoRABake}
NODE_DISPLAY_NAME_MAPPINGS = {"LCLoRABake": "LC LoRA Bake 🍞"}
