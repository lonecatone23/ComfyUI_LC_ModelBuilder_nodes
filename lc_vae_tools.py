"""
LC VAE Merge / LC VAE Save / LC VAE Compare
-------------------------------------------
A diffusion model is trained on one VAE's latents, so only the decoder half of a VAE is free to change. These nodes
merge decoders (the encoder always stays the base VAE's), save the result to models/vae, and compare VAEs by decoding
the very same latents with each one.

VAEs are taken from ComfyUI's loader, which already puts diffusers-style files (up_blocks / mid_block) into the same
layout as the original ones (up / mid), so a diffusers finetune merges with an original-layout base.

Decoder regions, coarse to fine:
  start   decoder input + middle (and the post-quant conv): overall tone and colour
  coarse  first upsampling stage: large shapes
  medium  second stage
  fine    third stage
  finest  last stage and the output layer: fine texture, sharpness, the last colour mapping
The output layer always goes with the finest stage: on Wan / Qwen-Image style decoders an output layer from one VAE on
top of the other's finest stage falls apart (tested: grid pattern and a green cast).
"""

from __future__ import annotations

import json
import os
import re

import torch

import comfy.sd
import comfy.utils
import folder_paths

REGIONS = ["start", "coarse", "medium", "fine", "finest"]
_TIPS = {
    "start": "Decoder input and middle: overall tone and colour. 0 = all base (vae_a), 1 = all vae_b.",
    "coarse": "First upsampling stage: large shapes.",
    "medium": "Second upsampling stage.",
    "fine": "Third upsampling stage.",
    "finest": "Last upsampling stage and the output layer: fine texture, sharpness and the last colour mapping.",
}


def _region(key, n_up):
    """Which decoder region a (ComfyUI-layout) key belongs to, or None for the encoder side (kept from the base)."""
    if key.startswith("post_quant_conv.") or key.startswith("conv2.") or key.startswith("decoder.post_quant_conv."):
        return "start"
    if not key.startswith("decoder."):
        return None
    k = key[len("decoder."):]
    if k.startswith(("conv_in.", "mid.", "conv1.", "middle.")):
        return "start"
    if k.startswith(("norm_out.", "conv_out.", "head.")):
        return "finest"
    m = re.match(r"up\.(\d+)\.", k)  # SD / SDXL / Flux: up.3 runs first (coarsest), up.0 last (finest)
    if m:
        i = int(m.group(1))
        return REGIONS[1 + (n_up - 1 - i) * 4 // max(1, n_up)] if n_up else "medium"
    m = re.match(r"upsamples\.(\d+)\.", k)  # Wan / Qwen-Image: upsamples.0 runs first
    if m:
        return REGIONS[1 + min(3, int(m.group(1)) * 4 // max(1, n_up))]
    return "?"


def _n_up(sd):
    ups = {int(m.group(1)) for k in sd for m in [re.match(r"decoder\.up\.(\d+)\.", k)] if m}
    if ups:
        return max(ups) + 1
    ups = {int(m.group(1)) for k in sd for m in [re.match(r"decoder\.upsamples\.(\d+)\.", k)] if m}
    return max(ups) + 1 if ups else 0


def _decode(vae, samples):
    with torch.inference_mode():
        img = vae.decode(samples)
    if img.dim() == 5:  # video-style VAEs return (B, T, H, W, C)
        img = img.reshape(-1, *img.shape[-3:])
    return img.detach().float().clamp(0, 1).cpu()


def _sharpness(img):
    """Mean gradient size of the luma (0-255 scale): higher = more fine detail / sharper."""
    y = (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114) * 255
    gx = (y[:, :, 1:] - y[:, :, :-1]).abs().mean()
    gy = (y[:, 1:, :] - y[:, :-1, :]).abs().mean()
    return float((gx + gy) / 2)


# ------------------------------------------------------------------ merge
class LCVAEMerge:
    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "vae_a": ("VAE", {"tooltip": "The base. Its encoder is always kept, so the result still matches the "
                                         "models trained on it."}),
            "vae_b": ("VAE", {"tooltip": "Another VAE of the same family (e.g. a decoder finetune of the same base)."}),
        }
        for r in REGIONS:
            req[r] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": _TIPS[r]})
        return {"required": req}

    RETURN_TYPES = ("VAE", "STRING")
    RETURN_NAMES = ("vae", "report")
    FUNCTION = "merge"
    CATEGORY = "LC Modelbuilder"
    DESCRIPTION = ("Merges the decoders of two VAEs of the same family, region by region from coarse shapes to fine "
                   "texture. The encoder stays vae_a's, so the merged VAE still fits the models made for vae_a.")

    def merge(self, vae_a, vae_b, start, coarse, medium, fine, finest):
        ratio = dict(zip(REGIONS, (start, coarse, medium, fine, finest)))
        sa, sb = vae_a.get_sd(), vae_b.get_sd()
        bad = [k for k in sa if k in sb and tuple(sa[k].shape) != tuple(sb[k].shape)]
        missing = [k for k in sa if k not in sb and _region(k, 1) is not None]
        if bad or len(missing) > len(sa) // 10:
            raise ValueError("LC VAE Merge: these two VAEs are not the same family (their layers do not line up"
                             f"{': ' + bad[0] if bad else ''}). Merge VAEs made from the same base.")
        n_up = _n_up(sa)
        out, used, unknown = {}, {r: 0 for r in REGIONS}, 0
        dist = {r: [0.0, 0.0] for r in REGIONS}  # how far apart the two decoders are, per region
        mean = sum(ratio.values()) / len(ratio)
        for k, a in sa.items():
            reg = _region(k, n_up)
            if reg is None or k not in sb or not a.is_floating_point():
                out[k] = a.clone()
                continue
            t = mean if reg == "?" else ratio[reg]
            unknown += reg == "?"
            if reg != "?":
                used[reg] += 1
            bk = sb[k].to(a.device).float()
            if reg != "?" and a.dim() >= 2:
                dist[reg][0] += float((a.float() - bk).abs().mean())
                dist[reg][1] += float(a.float().abs().mean())
            out[k] = torch.lerp(a.float(), bk, float(t)).to(a.dtype)

        enc = [k for k in sa if _region(k, n_up) is None and k in sb and sa[k].is_floating_point()]
        diff = sum(float((sa[k].float() - sb[k].to(sa[k].device).float()).abs().mean()) for k in enc) / max(1, len(enc))
        size = sum(float(sa[k].float().abs().mean()) for k in enc) / max(1, len(enc))
        rel = diff / max(size, 1e-12)

        merged = comfy.sd.VAE(sd=out)
        merged.throw_exception_if_invalid()
        lines = ["Merged decoders (encoder kept from vae_a).",
                 "Regions: " + ", ".join(f"{r} {ratio[r]:.2f} ({used[r]} layers)" for r in REGIONS)]
        far = {r: d[0] / d[1] for r, d in dist.items() if d[1] > 0}
        lines.append("Decoder distance (how different the weights are, 0 % = identical): "
                     + ", ".join(f"{r} {v * 100:.1f} %" for r, v in far.items()))
        blended = [r for r, v in far.items() if v > 0.15 and 0.0 < ratio[r] < 1.0]
        if blended:
            lines.append("WARNING: the decoders are far apart in " + ", ".join(blended) + " (over 15 %): one was "
                         "retrained, not finetuned, from the other. Blending them makes artifacts (oversaturated colour, "
                         "noise). Merge VAEs that sit close (under about 10 %), or use 0 / 1 for those regions.")
        jumps = [f"{REGIONS[i]} -> {REGIONS[i + 1]}" for i in range(len(REGIONS) - 1)
                 if abs(ratio[REGIONS[i]] - ratio[REGIONS[i + 1]]) > 0.5]
        if jumps:
            lines.append("Note: a big jump between neighbouring regions (" + ", ".join(jumps) + "). Flux VAEs take it, "
                         "Wan / Qwen-Image style ones (Krea 2, Anima) can break: ramp it (e.g. 0, 0.25, 0.5, 0.75, 1).")
        if unknown:
            lines.append(f"{unknown} decoder layers have no region in this VAE type: they used the average ({mean:.2f}).")
        if rel < 1e-6:
            lines.append("The encoders are identical: vae_b is a decoder finetune of the same base. Safe.")
        else:
            lines.append(f"The encoders differ by {rel * 100:.1f} % on average. vae_b was trained on its own encoder, "
                         "so its decoder may not read vae_a's latents perfectly: check colours in LC VAE Compare.")
        report = "\n".join(lines)
        print("[LC VAE Merge] " + report.replace("\n", " | "))
        return (merged, report)


# ------------------------------------------------------------------ save
class LCVAESave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "filename": ("STRING", {"default": "LC_merged_vae", "tooltip": "Saved in models/vae, ready for "
                                        "Load VAE. A subfolder works. An existing file is never overwritten."}),
                "precision": (["fp32", "bf16", "fp16"], {
                    "default": "fp32",
                    "tooltip": "fp32 keeps every bit (the safest for a VAE). bf16 / fp16 halve the size; bf16 can band "
                               "a little in smooth gradients."}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off skips the save."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "LC Modelbuilder"
    DESCRIPTION = "Saves a VAE (a merged one, for example) to models/vae, in fp32, bf16 or fp16."

    def save(self, vae, filename, precision="fp32", enabled=True):
        if not enabled:
            return {"ui": {"text": ["skipped"]}, "result": ("",)}
        dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
        sd = {k: (v.to(dt) if v.is_floating_point() else v).contiguous().cpu() for k, v in vae.get_sd().items()}
        root = folder_paths.get_folder_paths("vae")[0]
        base = (filename.strip() or "LC_merged_vae").replace("\\", "/")
        out = os.path.join(root, base + ".safetensors")
        n = 1
        while os.path.exists(out):
            out = os.path.join(root, f"{base}_{n}.safetensors")
            n += 1
        os.makedirs(os.path.dirname(out), exist_ok=True)
        comfy.utils.save_torch_file(sd, out, metadata={"lc_vae_save": json.dumps({"precision": precision})})
        msg = f"Saved {os.path.relpath(out, root)} ({precision}, {os.path.getsize(out) / 1e6:.0f} MB)"
        print("[LC VAE Save] " + msg)
        return {"ui": {"text": [msg]}, "result": (out,)}


# ------------------------------------------------------------------ compare
class LCVAECompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Latents from a real render (KSampler output). Every VAE decodes "
                                                  "these same latents, which is what you actually see."}),
                "vae_1": ("VAE", {"tooltip": "The reference: the others are measured against it."}),
                "labels": ("STRING", {"default": "", "tooltip": "Names for the columns, comma separated. Empty = vae 1, "
                                                               "vae 2, ..."}),
                "crop_size": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 32,
                                      "tooltip": "A 1:1 crop from the middle of each decode, shown big under it, to "
                                                 "judge texture."}),
            },
            "optional": {f"vae_{i}": ("VAE",) for i in range(2, 7)},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("sheet", "images", "report")
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "LC Modelbuilder"
    DESCRIPTION = ("Decodes the same latents with up to six VAEs and lays them side by side with a 1:1 detail crop, "
                   "a sharpness number and how far each one lands from vae_1.")

    def compare(self, samples, vae_1, labels="", crop_size=256, **vaes):
        from PIL import Image, ImageDraw

        chosen = [("vae 1", vae_1)] + [(f"vae {i}", vaes[f"vae_{i}"]) for i in range(2, 7) if vaes.get(f"vae_{i}") is not None]
        names = [s.strip() for s in labels.split(",")] if labels.strip() else []
        lat = samples["samples"]
        decs, rows = [], []
        for i, (default, vae) in enumerate(chosen):
            name = names[i] if i < len(names) and names[i] else default
            try:
                img = _decode(vae, lat)
            except Exception as e:
                rows.append(f"{name}: could not decode these latents ({type(e).__name__}): not the same family")
                continue
            decs.append((name, img))
        if not decs:
            raise ValueError("LC VAE Compare: none of the VAEs could decode these latents.")
        ref = decs[0][1]
        report = ["Sharpness = average edge strength (higher = crisper). Drift = average difference from the first VAE "
                  "on a 0-255 scale (0 = identical, under 2 = hard to see, over 6 = visible)."]
        tiles = []
        for name, img in decs:
            sharp = _sharpness(img)
            drift = float((img - ref).abs().mean() * 255) if img.shape == ref.shape else float("nan")
            report.append(f"{name}: sharpness {sharp:.2f}, drift {drift:.2f}")
            tiles.append((name, img[0], sharp, drift))
        report += rows

        # sheet: one column per VAE, the full picture on top and a 1:1 crop from its middle below
        h, w = tiles[0][1].shape[:2]
        col_w = 360
        full_h = int(col_w * h / w)
        cs = min(crop_size, h, w)
        crop_h = col_w
        pad = 38
        sheet = Image.new("RGB", (col_w * len(tiles), pad + full_h + crop_h), (20, 20, 20))
        d = ImageDraw.Draw(sheet)
        for c, (name, img, sharp, drift) in enumerate(tiles):
            arr = (img.numpy() * 255).round().astype("uint8")
            pil = Image.fromarray(arr)
            sheet.paste(pil.resize((col_w, full_h), Image.LANCZOS), (c * col_w, pad))
            y0, x0 = (arr.shape[0] - cs) // 2, (arr.shape[1] - cs) // 2
            crop = pil.crop((x0, y0, x0 + cs, y0 + cs)).resize((col_w, crop_h), Image.NEAREST)
            sheet.paste(crop, (c * col_w, pad + full_h))
            d.text((c * col_w + 6, 4), name, fill=(235, 235, 235))
            d.text((c * col_w + 6, 20), f"sharp {sharp:.2f}  drift {drift:.2f}", fill=(170, 200, 170))
        import numpy as np

        sheet_t = torch.from_numpy(np.asarray(sheet, dtype=np.float32) / 255.0)[None]
        same = [img for _, img in decs if img.shape == ref.shape]
        text = "\n".join(report)
        print("[LC VAE Compare]\n" + text)
        return {"ui": {"text": [text]}, "result": (sheet_t, torch.cat(same, 0), text)}


NODE_CLASS_MAPPINGS = {"LCVAEMerge": LCVAEMerge, "LCVAESave": LCVAESave, "LCVAECompare": LCVAECompare}
NODE_DISPLAY_NAME_MAPPINGS = {"LCVAEMerge": "LC VAE Merge", "LCVAESave": "LC VAE Save", "LCVAECompare": "LC VAE Compare"}
