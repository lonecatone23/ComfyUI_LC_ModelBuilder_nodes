"""
LC Krea2 LoRA Block Scan
------------------------
Finds where a LoRA does its damage. Renders the same seed batch with no LoRA, with the full LoRA, and with
the LoRA turned off in one block group at a time, then scores every row: same-face score (face lock, via
LC Face Variety Scorer's InsightFace models) and how far each render moved from the base model and from the
full LoRA. Outputs a labeled grid, a report, and a suggested 28-block curve for LC Krea2 LoRA Editor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import comfy.model_management
import comfy.samplers
import comfy.utils
import folder_paths
import nodes

from .lc_krea2_lora_editor import NUM_BLOCKS, LCKrea2LoRAEditor

FACE_DROP = 0.03  # a group counts as face lock when turning it off lowers the same-face score by this much (8-seed noise is about 0.02)


def _groups(text: str):
    out = []
    for part in str(text or "").split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p in ("fusion", "outside", "txtfusion"):
            out.append(("fusion", None))
            continue
        try:
            lo, hi = (int(x) for x in p.split("-")) if "-" in p else (int(p), int(p))
        except ValueError:
            continue
        lo, hi = max(0, min(lo, hi)), min(NUM_BLOCKS - 1, max(lo, hi))
        out.append((f"{lo}-{hi}", (lo, hi)))
    return out


_LP = {}


def _change(a: torch.Tensor, b: torch.Tensor):
    """Perceptual distance per image pair, a / b = [B,H,W,3] 0..1. LPIPS when the lpips package is there,
    otherwise 1 - SSIM on a 256 px grey copy."""
    dev = comfy.model_management.get_torch_device()
    x = a.permute(0, 3, 1, 2).to(dev, torch.float32)
    y = b.permute(0, 3, 1, 2).to(dev, torch.float32)
    if "net" not in _LP:
        try:
            import lpips
            _LP["net"] = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
        except Exception:
            _LP["net"] = None
    if _LP["net"] is not None:
        with torch.no_grad():
            x = F.interpolate(x, size=(512, 512), mode="bilinear", antialias=True) * 2 - 1
            y = F.interpolate(y, size=(512, 512), mode="bilinear", antialias=True) * 2 - 1
            return [float(v) for v in _LP["net"].to(dev)(x, y).flatten()], "LPIPS"
    g = lambda t: F.interpolate(t.mean(1, keepdim=True), size=(256, 256), mode="area")
    x, y = g(x), g(y)
    k = torch.ones(1, 1, 7, 7, device=dev) / 49
    mx, my = F.conv2d(x, k), F.conv2d(y, k)
    vx = F.conv2d(x * x, k) - mx * mx
    vy = F.conv2d(y * y, k) - my * my
    cxy = F.conv2d(x * y, k) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return [float(1 - v) for v in ssim.mean(dim=(1, 2, 3))], "1 - SSIM"


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _grid(rows, tile=192, label_w=210):
    """rows = [(label lines, images [B,H,W,3])] -> one IMAGE [1,H,W,3]."""
    n = max(r[1].shape[0] for r in rows)
    th = int(tile * rows[0][1].shape[1] / rows[0][1].shape[2])
    W, H = label_w + n * (tile + 4), len(rows) * (th + 4)
    canvas = Image.new("RGB", (W, H), (24, 22, 16))
    d = ImageDraw.Draw(canvas)
    f_big, f_small = _font(18), _font(14)
    for r, (lines, imgs) in enumerate(rows):
        y = r * (th + 4)
        d.text((10, y + 10), lines[0], fill=(240, 217, 168), font=f_big)
        for i, t in enumerate(lines[1:]):
            d.text((10, y + 38 + i * 20), t, fill=(215, 215, 215), font=f_small)
        small = F.interpolate(imgs.permute(0, 3, 1, 2).float(), size=(th, tile), mode="bilinear", antialias=True)
        arr = (small.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        for i, a in enumerate(arr):
            canvas.paste(Image.fromarray(a), (label_w + i * (tile + 4), y))
    return torch.from_numpy(np.asarray(canvas).astype(np.float32) / 255.0)[None]


class LCKrea2LoRABlockScan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Krea 2 model, without the LoRA."}),
                "vae": ("VAE",),
                "positive": ("CONDITIONING", {"tooltip": "A prompt that shows a face (for face lock) and a scene (for whole-image change)."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA to scan, from models/loras."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "groups": ("STRING", {"default": "0-3, 4-9, 10-15, 16-21, 22-27, fusion",
                                      "tooltip": "Block groups to turn off one at a time, comma separated. fusion = the text fusion "
                                                 "layers outside the 28 blocks. Each group is one more row of renders."}),
                "off_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "What 'off' means for the group being tested. 0 = fully off."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": "fixed"}),
                "images": ("INT", {"default": 8, "min": 2, "max": 16,
                                   "tooltip": "Seeds per row (one batch). 8 gives a steady same-face score; fewer is faster but noisier."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
            },
            "optional": {
                "negative": ("CONDITIONING", {"tooltip": "Optional. Unwired = an empty (zeroed) negative, right for cfg 1."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("grid", "report", "blocks")
    OUTPUT_TOOLTIPS = ("One row per render set: no LoRA, full LoRA, then each group off.",
                       "Scores per row and what they suggest.",
                       "Suggested 28-block curve. Wire into LC Krea2 LoRA Editor's blocks (convert it to an input) or copy it.")
    OUTPUT_NODE = True
    FUNCTION = "scan"
    CATEGORY = "LC ModelBuilder/Krea2"
    DESCRIPTION = ("Renders a fixed seed batch without the LoRA, with it, and with each block group turned off, then "
                   "scores face lock (same-face score) and change against the base model and the full LoRA. Finds the "
                   "blocks to turn down in LC Krea2 LoRA Editor.")

    def scan(self, model, vae, positive, lora_name, strength, groups, off_value, seed, images, width, height,
             steps, cfg, sampler_name, scheduler, negative=None):
        from .lc_face_variety import _mean_pair_sim, _sessions, _detect, _embed  # needs onnxruntime

        if negative is None:
            negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
        latent = nodes.EmptyLatentImage().generate(width, height, images)[0]
        groups_ = _groups(groups)
        editor = LCKrea2LoRAEditor()
        ones = ", ".join(["1"] * NUM_BLOCKS)
        plan = [("No LoRA", None, None), ("Full LoRA", ones, 1.0)]
        for name, rng in groups_:
            if rng is None:
                plan.append(("Fusion off", ones, off_value))
            else:
                c = ["1"] * NUM_BLOCKS
                for i in range(rng[0], rng[1] + 1):
                    c[i] = f"{off_value:g}"
                plan.append((f"Blocks {name} off", ", ".join(c), 1.0))

        pbar = comfy.utils.ProgressBar(len(plan))
        renders = []
        for label, curve, outside in plan:
            comfy.model_management.throw_exception_if_processing_interrupted()
            m = model if curve is None else editor.edit(model, lora_name, strength, "Custom", curve, 1.0, 1.0, outside,
                                                        "off", 2.0)["result"][0]
            out = nodes.common_ksampler(m, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent)[0]
            img = vae.decode(out["samples"])
            if img.ndim == 5:
                img = img.reshape(-1, *img.shape[-3:])
            renders.append((label, curve, img[:, :, :, :3].float().cpu()))
            del m
            pbar.update(1)

        # scores
        det, rec = _sessions("auto")

        def face_score(imgs):
            embs = []
            for rgb in (imgs.clamp(0, 1).numpy() * 255).round().astype(np.uint8):
                rgb = np.ascontiguousarray(rgb)
                boxes, kps = _detect(det, rgb)
                ok = [j for j in range(len(boxes)) if boxes[j][2] - boxes[j][0] >= 40]
                if ok:
                    j = max(ok, key=lambda j: (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]))
                    embs.append(_embed(rec, rgb, kps[j]))
            s, sims = _mean_pair_sim(embs)
            return (s if sims else float("nan")), len(embs)

        base, full = renders[0][2], renders[1][2]
        rows, stats = [], []
        metric = "LPIPS"
        for label, curve, imgs in renders:
            fs, nf = face_score(imgs)
            vb, metric = _change(imgs, base)
            vf, _ = _change(imgs, full)
            st = {"label": label, "curve": curve, "face": fs, "faces": nf,
                  "vs_base": float(np.mean(vb)), "vs_full": float(np.mean(vf))}
            stats.append(st)
            lines = [label, f"same face {fs:.3f}" if fs == fs else "same face n/a (no faces)"]
            if label != "No LoRA":
                lines.append(f"change vs base {st['vs_base']:.3f}")
            if label not in ("No LoRA", "Full LoRA"):
                lines.append(f"change vs LoRA {st['vs_full']:.3f}")
            rows.append((lines, imgs))

        # suggestion: the group whose removal frees the face most, if it clearly does
        full_face = stats[1]["face"]
        tests = stats[2:]
        suggest = [1.0] * NUM_BLOCKS
        rep = [f"LC Krea2 LoRA Block Scan: {lora_name} at {strength:g}, {images} seeds from {seed}, change = {metric}", ""]
        rep.append(f"{'row':<18}{'same face':>10}{'vs base':>10}{'vs LoRA':>10}")
        for st in stats:
            f_ = f"{st['face']:.3f}" if st["face"] == st["face"] else "n/a"
            vb = "" if st["label"] == "No LoRA" else f"{st['vs_base']:.3f}"
            vf = "" if st["label"] in ("No LoRA", "Full LoRA") else f"{st['vs_full']:.3f}"
            rep.append(f"{st['label']:<18}{f_:>10}{vb:>10}{vf:>10}")
        rep.append("")
        valid = [t for t in tests if t["face"] == t["face"]]
        if full_face == full_face and valid:
            best = min(valid, key=lambda t: t["face"])
            drop = full_face - best["face"]
            lock = full_face - stats[0]["face"] if stats[0]["face"] == stats[0]["face"] else float("nan")
            if lock == lock:
                rep.append(f"Face lock: the LoRA moves the same-face score {lock:+.3f} from the base model.")
            if lock == lock and lock < FACE_DROP and drop < FACE_DROP:
                rep.append("That is within the seed noise (0.03): this LoRA barely locks the face, so there is nothing "
                           "to turn down for it. Best group anyway: "
                           f"{best['label']} ({best['face']:.3f}).")
            elif drop >= FACE_DROP and best["curve"] is not None and best["label"] != "Fusion off":
                for i, v in enumerate(best["curve"].split(",")):
                    if float(v) < 1:
                        suggest[i] = 0.3
                rep.append(f"{best['label']} frees the face most ({best['face']:.3f}, {-drop:+.3f}). Suggested curve: "
                           f"that group at 0.3.")
            elif drop >= FACE_DROP:
                rep.append(f"{best['label']} frees the face most ({best['face']:.3f}): set outside_blocks down in the Editor.")
            else:
                rep.append("No single group frees the face by more than the seed noise (0.03). Try smaller groups, "
                           "or spike_soften directions in the Editor.")
        else:
            rep.append("Not enough faces found to score face lock. Use a prompt with one clear face.")
        big = max(tests, key=lambda t: stats[1]["vs_base"] - t["vs_base"], default=None)
        if big is not None:
            rep.append(f"Whole-image change: {big['label']} cuts the change from the base model the most "
                       f"({stats[1]['vs_base']:.3f} -> {big['vs_base']:.3f}); 'vs LoRA' shows how much of the LoRA's own "
                       f"look that also costs.")
        report = "\n".join(rep)
        blocks = ", ".join(f"{v:g}" for v in suggest)
        grid = _grid(rows)
        ui = nodes.PreviewImage().save_images(grid, filename_prefix="LC_LoRA_BlockScan")["ui"]
        ui["text"] = [report]
        return {"ui": ui, "result": (grid, report, blocks)}


NODE_CLASS_MAPPINGS = {"LCKrea2LoRABlockScan": LCKrea2LoRABlockScan}
NODE_DISPLAY_NAME_MAPPINGS = {"LCKrea2LoRABlockScan": "LC Krea2 LoRA Block Scan"}
