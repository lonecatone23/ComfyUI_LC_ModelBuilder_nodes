"""
LC Model Quantize
-----------------
Makes a smaller copy of a diffusion model, from a MODEL wire (a merge, a LoRA-baked model, a checkpoint) or straight
from a file on disk, in one of:

  native ComfyUI formats (.safetensors, load with the stock Load Diffusion Model node):
      fp8 (scaled), mxfp8, nvfp4, int8 (convrot), int4 (convrot w4a4)
      ComfyUI's own quantizers write them, with the same per-layer comfy_quant tags ComfyUI saves.
  GGUF (.gguf, load with ComfyUI-GGUF's Unet Loader):
      Q8_0, Q6_K, Q5_K_M / S, Q4_K_M / S, Q3_K_M / S, Q2_K (lc_gguf_quants.py, ported from llama.cpp)

The sensitive layers (embedders, norms, modulation, the final layer, anything not 2-D or too small) stay full
precision, the way released quants do. The model you wire in is never changed: each layer is read, has its
LoRA / merge patches baked in, is quantized on the GPU and written straight to disk, so memory use stays at about one
layer. From a file it reads one tensor at a time, so a model bigger than your RAM still works.

Idea from ComfyUI_DiffusionModel_fp8_converter (Shiba-2-shiba): convert inside ComfyUI from the file you already have.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import time

import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths

from . import lc_gguf_quants as gq
from .lc_model_merge_any import _dequant, _is_quant

WIRED = "(use the model input)"

NATIVE = {
    "fp8 (scaled)": "float8_e4m3fn",
    "mxfp8 (50 series)": "mxfp8",
    "nvfp4 (50 series)": "nvfp4",
    "int8 (convrot)": "int8_tensorwise",
    "int4 (convrot w4a4)": "convrot_w4a4",
}
# GGUF: base type, and the type the first and last two blocks get for the _M variants
GGUF = {
    "GGUF Q8_0": ("Q8_0", None),
    "GGUF Q6_K": ("Q6_K", None),
    "GGUF Q5_K_M": ("Q5_K", "Q6_K"),
    "GGUF Q5_K_S": ("Q5_K", None),
    "GGUF Q4_K_M": ("Q4_K", "Q6_K"),
    "GGUF Q4_K_S": ("Q4_K", None),
    "GGUF Q3_K_M": ("Q3_K", "Q5_K"),
    "GGUF Q3_K_S": ("Q3_K", None),
    "GGUF Q2_K": ("Q2_K", "Q4_K"),
}
FORMATS = list(NATIVE) + list(GGUF)

# top-level modules that stay full precision (embedders, conditioning, final layer)
_TOP_KEEP = {
    "img_in", "txt_in", "time_in", "vector_in", "guidance_in", "x_embedder", "t_embedder", "y_embedder",
    "pos_embed", "pos_embedder", "patch_embed", "cap_embedder", "context_embedder", "time_text_embed", "adaln_single",
    "final_layer", "norm_out", "proj_out", "llm_adapter", "t_embedding_norm", "time_embed", "label_emb",
    "modulation", "txt_norm", "img_norm", "time_proj", "caption_projection", "t_block", "time_guidance_embed",
    "txt_in_norm", "distilled_guidance_layer", "time_caption_embed", "norm_final",
    "first", "last", "tmlp", "tproj", "txtmlp",  # Krea 2
    "video_embedder", "text_embedder", "time_embedding", "shared_block_modulation", "video_head_modulation",
    "final_norm_video", "final_linear_video",  # MOSS Image (Comfy-Org's int8 keeps all of these bf16)
}
# 4-bit formats that also round the activations (nvfp4, int4 w4a4) break on layers fed by the timestep (adaLN /
# modulation) and on the text side: those stay full precision there. Tested on Anima: noise without, a picture with.
_ACT4 = {"nvfp4", "convrot_w4a4"}
_ACT4_KEEP = re.compile(r"adaln|modulation|(^|\.)(img_|txt_)?mod(\.|$)|txtfusion|context_refiner|noise_refiner", re.I)
_MIN_ELEMENTS = 65536
_BLOCK = re.compile(r"^(?P<grp>[A-Za-z_]*blocks|layers|middle_block)\.(?P<idx>\d+)\.")
_GROUP_ORDER = ["input_blocks", "double_blocks", "joint_blocks", "blocks", "layers", "transformer_blocks",
                "middle_block", "single_blocks", "single_transformer_blocks", "output_blocks"]
_ST_DTYPE = {torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
             torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8",
             torch.bool: "BOOL", torch.float8_e4m3fn: "F8_E4M3", torch.float8_e5m2: "F8_E5M2"}
_ST_TORCH = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
             "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
             "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}


def _sources():
    out = [WIRED]
    for folder in ("diffusion_models", "checkpoints"):
        try:
            names = folder_paths.get_filename_list(folder)
        except Exception:
            names = []
        out += [f"{folder}/{n}" for n in names if n.lower().endswith(".safetensors")]
    return out


# ------------------------------------------------------------------ reading the source, one tensor at a time
class _ModelSource:
    """A MODEL wire: its diffusion model's weights with the patcher's patches (LoRAs, merges) baked in."""

    def __init__(self, model):
        mm.unload_all_models()  # puts back any weights a sampler patched in place, so patches are applied once
        mm.soft_empty_cache()
        self.mp = model
        self.sd = model.model.diffusion_model.state_dict()
        self.patches = getattr(model, "patches", {}) or {}
        self.name = "model"
        self.notes = []
        self.policy_prefix = ""
        big = [t for t in self.sd.values() if t.dim() == 2 and t.numel() >= _MIN_ELEMENTS]
        if any(_is_quant(t) for t in big):
            self.notes.append("The wired model is already quantized (it was loaded as fp8 / int8 / GGUF). It is decoded "
                              "first, but quantizing again adds to the loss: load the bf16 file (weight_dtype default).")

    def keys(self):
        return list(self.sd.keys())

    def shape(self, k):
        return tuple(self.sd[k].shape)

    def dtype(self, k):
        t = self.sd[k]
        return t.dtype if not _is_quant(t) else torch.bfloat16

    def get(self, k, dev):
        t = self.sd[k]
        w = _dequant(t, dev) if _is_quant(t) and t.dim() >= 2 else t
        w = w.to(dev, torch.float32) if w.is_floating_point() else w.to(dev)
        p = self.patches.get(f"diffusion_model.{k}")
        if p and w.is_floating_point():
            import comfy.lora

            w = comfy.lora.calculate_weight(p, w, f"diffusion_model.{k}")
        return w


class _FileSource:
    """A .safetensors file read one tensor at a time with plain reads (no memory map, so the file never sits in RAM).
    An all-in-one checkpoint gives only its diffusion model."""

    PREFIX = "model.diffusion_model."

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
        self.base = 8 + n
        head.pop("__metadata__", None)
        self.head = head
        allk = list(head.keys())
        self.pre = self.PREFIX if any(k.startswith(self.PREFIX) for k in allk) else ""
        self.map = {k[len(self.pre):]: k for k in allk if k.startswith(self.pre)}
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.notes = []
        # Cosmos / Anima files keep a "net." prefix: the names are written as they are, the layer rules look past it
        net = sum(k.startswith("net.") for k in self.map)
        self.policy_prefix = "net." if net > len(self.map) // 2 else ""
        if any(k.endswith("comfy_quant") or k.endswith("scaled_fp8") for k in self.map):
            raise ValueError("LC Model Quantize: this file is already quantized (it has scale / comfy_quant keys). "
                             "Pick the bf16 or fp16 version of the model.")
        big = [k for k in self.map if len(self.shape(k)) == 2 and math.prod(self.shape(k)) >= _MIN_ELEMENTS]
        if any(self._st_dtype(k) in ("F8_E4M3", "F8_E5M2", "I8", "U8") for k in big):
            self.notes.append("The source file stores its weights in fp8: quantizing again adds to the loss. "
                              "Use the bf16 / fp16 file if you have it.")
        self.f = open(path, "rb", buffering=0)
        self.buf = None  # one read buffer, sized to the biggest tensor and reused

    def _st_dtype(self, k):
        return self.head[self.map[k]]["dtype"]

    def keys(self):
        return list(self.map.keys())

    def shape(self, k):
        return tuple(self.head[self.map[k]]["shape"])

    def dtype(self, k):
        return _ST_TORCH.get(self._st_dtype(k), torch.float32)

    def get(self, k, dev):
        e = self.head[self.map[k]]
        a, b = e["data_offsets"]
        dt = _ST_TORCH.get(e["dtype"])
        if dt is None:
            raise ValueError(f"LC Model Quantize: unsupported dtype {e['dtype']} in {k}")
        n = b - a
        if n == 0:
            return torch.empty(e["shape"], dtype=dt, device=dev)
        if self.buf is None or len(self.buf) < n:
            self.buf = bytearray(max(n, max(v["data_offsets"][1] - v["data_offsets"][0] for v in self.head.values())))
        self.f.seek(self.base + a)
        self.f.readinto(memoryview(self.buf)[:n])
        t = torch.frombuffer(self.buf, dtype=dt, count=n // torch.empty((), dtype=dt).element_size()).reshape(e["shape"])
        # always a copy: the buffer is reused for the next tensor
        return t.to(dev, torch.float32, copy=True) if t.is_floating_point() else t.to(dev, copy=True)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


# ------------------------------------------------------------------ which layers get quantized
def _edges(keys, n):
    """Keys of the first n and last n transformer blocks (in model order, over all block groups)."""
    seen = {}
    for k in keys:
        m = _BLOCK.match(k)
        if m:
            seen.setdefault((m["grp"], int(m["idx"])), None)
    order = sorted(seen, key=lambda g: (_GROUP_ORDER.index(g[0]) if g[0] in _GROUP_ORDER else 99, g[0], g[1]))
    if len(order) <= 2 * n:
        return set(order[:1] + order[-1:]) if n and order else set()
    return set(order[:n] + order[-n:])


def _block_of(k):
    m = _BLOCK.match(k)
    return (m["grp"], int(m["idx"])) if m else None


def _quantizable(k, shape):
    if not k.endswith(".weight") or len(shape) != 2 or math.prod(shape) < _MIN_ELEMENTS:
        return False
    if k.split(".")[0] in _TOP_KEEP:
        return False
    last = k[: -len(".weight")].split(".")[-1].lower()
    return "norm" not in last and "embed" not in last


# ------------------------------------------------------------------ writers
class _StreamSafetensors:
    """Writes a .safetensors file tensor by tensor. Room for the header is reserved up front and filled in at the
    end (padded with spaces, which the format allows), so nothing has to be held in memory."""

    def __init__(self, path, reserve, metadata):
        self.path = path
        self.reserve = int(math.ceil((reserve + 65536) / 8) * 8)
        self.meta = metadata
        self.f = open(path, "wb")
        self.f.write(b"\0" * (8 + self.reserve))
        self.entries = {}
        self.pos = 0
        self.small = []

    def add(self, name, t):
        t = t.detach().contiguous().cpu()
        if t.dtype == getattr(torch, "float8_e8m0fnu", None):
            t = t.view(torch.uint8)  # safetensors has no e8m0; ComfyUI views the bytes back
        nbytes = t.numel() * t.element_size()
        if nbytes % 8 or nbytes < 4096:  # keep the big tensors 8-byte aligned: small odd ones go last
            self.small.append((name, t))
            return
        self._write(name, t)

    def _write(self, name, t):
        n = t.numel() * t.element_size()
        if n:
            self.f.write(t.reshape(-1).view(torch.uint8).numpy().data)
        self.entries[name] = {"dtype": _ST_DTYPE[t.dtype], "shape": list(t.shape),
                              "data_offsets": [self.pos, self.pos + n]}
        self.pos += n

    def close(self):
        for name, t in sorted(self.small, key=lambda nt: -nt[1].element_size()):
            self._write(name, t)
        head = json.dumps({"__metadata__": self.meta, **self.entries}, separators=(",", ":")).encode("utf-8")
        if len(head) > self.reserve:
            self.f.close()
            raise RuntimeError("LC Model Quantize: header bigger than the space reserved for it")
        head += b" " * (self.reserve - len(head))
        self.f.seek(0)
        self.f.write(struct.pack("<Q", self.reserve))
        self.f.write(head)
        self.f.close()

    def abort(self):
        try:
            self.f.close()
        finally:
            if os.path.exists(self.path):
                os.remove(self.path)


def _gguf_arch(keys):
    """An architecture name ComfyUI-GGUF accepts. Its own detector when it knows the model, otherwise "flux": the
    name is only checked against a list, ComfyUI works out the real model from the weights themselves."""
    import sys

    for name, mod in list(sys.modules.items()):
        if name.endswith("tools.convert") and hasattr(mod, "detect_arch"):
            try:
                return mod.detect_arch(set(keys)).arch
            except Exception:
                break
    return "flux"


# ------------------------------------------------------------------ the node
class LCModelQuantize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (_sources(), {
                    "tooltip": "Where the weights come from. '(use the model input)' takes the wired MODEL, LoRAs and "
                               "merges included (fine up to about 12-14B on 64 GB RAM). A file is read one layer at a "
                               "time, so any size works. Pick the bf16 / fp16 file, not an fp8 one."}),
                "format": (FORMATS, {
                    "default": "fp8 (scaled)",
                    "tooltip": "fp8: half size, near lossless, fast on 40 / 50 series. mxfp8, nvfp4: 50 series only. "
                               "int8 (convrot): half size, runs on 30 series too. int4: a quarter. GGUF needs "
                               "ComfyUI-GGUF to load: Q8_0 near lossless, Q6_K / Q5_K very close, Q4_K_M the usual "
                               "pick for small cards, Q3 / Q2 for when nothing else fits. _M keeps the first and last "
                               "two blocks at a higher type."}),
                "protect_edges": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep the first and last transformer block at full precision too (a little bigger, "
                               "a little closer to the original)."}),
                "filename": ("STRING", {
                    "default": "",
                    "tooltip": "Saved in models/diffusion_models. Empty = the source name plus the format. A "
                               "subfolder works: 'Krea2/Krealism'. An existing file is never overwritten."}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Used when source is '(use the model input)'."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "LC Modelbuilder"
    DESCRIPTION = (
        "Saves a smaller copy of a diffusion model: fp8, mxfp8, nvfp4, int8 or int4 (native ComfyUI files) or GGUF "
        "Q8_0 to Q2_K. From the wired MODEL (LoRAs and merges baked in) or straight from a file, one layer at a time. "
        "Embedders, norms and the final layer stay full precision. The wired model is not changed."
    )

    def run(self, source, format, protect_edges=False, filename="", model=None):
        t0 = time.time()
        if source == WIRED:
            if model is None:
                raise ValueError("LC Model Quantize: wire a MODEL in, or pick a file as the source.")
            src = _ModelSource(model)
        else:
            folder, name = source.split("/", 1)
            src = _FileSource(folder_paths.get_full_path_or_raise(folder, name))
        try:  # the source file is open from here: close it on any early error (it locks the file on Windows)
            dev = mm.get_torch_device()
            keys = src.keys()
            is_gguf = format in GGUF
            pp = src.policy_prefix

            def pk(k):
                return k[len(pp):] if pp and k.startswith(pp) else k

            # plan: what every tensor becomes
            pkeys = [pk(k) for k in keys]
            edges1 = _edges(pkeys, 1) if protect_edges else set()
            bump = set()
            if is_gguf and GGUF[format][1]:
                bump = _edges(pkeys, 2)
            plan = {}
            for k in keys:
                shape = src.shape(k)
                blk = _block_of(pk(k))
                act4 = not is_gguf and NATIVE[format] in _ACT4 and _ACT4_KEEP.search(pk(k))
                if not _quantizable(pk(k), shape) or (blk is not None and blk in edges1) or act4:
                    plan[k] = None
                    continue
                if is_gguf:
                    q = GGUF[format][1] if (blk is not None and blk in bump) else GGUF[format][0]
                    cols = shape[1]
                    if cols % gq.TYPES[q][1]:
                        q = "Q8_0" if cols % 32 == 0 else None
                    plan[k] = q
                else:
                    plan[k] = NATIVE[format]

            # output path, disk space
            base = filename.strip() or f"{src.name}-{format.split(' (')[0].replace('GGUF ', '').replace(' ', '_')}"
            ext = ".gguf" if is_gguf else ".safetensors"
            roots = folder_paths.get_folder_paths("diffusion_models")
            root = next((r for r in roots if os.path.basename(os.path.normpath(r)).lower() == "diffusion_models"), roots[0])
            base = base.replace("\\", "/").strip("/")
            for e in (".safetensors", ".gguf"):  # no double extension
                if base.lower().endswith(e):
                    base = base[: -len(e)]
            base = os.path.abspath(os.path.join(root, base or src.name))
            if not base.startswith(os.path.join(os.path.abspath(root), "")):  # trailing separator: a sibling folder is outside
                raise ValueError("LC Model Quantize: filename must stay inside models/diffusion_models")
            out = os.path.join(root, base + ext)
            n = 1
            while os.path.exists(out):
                out = os.path.join(root, f"{base}_{n}{ext}")
                n += 1
            os.makedirs(os.path.dirname(out), exist_ok=True)
            src_bytes = sum(math.prod(src.shape(k)) * 2 for k in keys)
            est = self._estimate(src, keys, plan, is_gguf)
            free = shutil.disk_usage(os.path.dirname(out)).free
            if est * 1.05 > free:
                raise RuntimeError(f"LC Model Quantize: about {est / 1e9:.1f} GB needed, only {free / 1e9:.1f} GB free on "
                                   f"that drive.")
        except BaseException:
            if hasattr(src, "close"):
                src.close()
            raise

        pbar = comfy.utils.ProgressBar(len(keys))
        stats = {"err": 0.0, "n": 0, "worst": (0.0, ""), "kinds": {}}
        try:
            if is_gguf:
                self._write_gguf(src, keys, plan, out, dev, stats, pbar)
            else:
                self._write_native(src, keys, plan, out, dev, stats, pbar, format)
        except BaseException:
            if os.path.exists(out):
                try:
                    os.remove(out)
                except OSError:
                    pass
            raise
        finally:
            if hasattr(src, "close"):
                src.close()
            mm.soft_empty_cache()

        size = os.path.getsize(out)
        drift = stats["err"] / max(1, stats["n"])
        kinds = ", ".join(f"{v} {k}" for k, v in sorted(stats["kinds"].items(), key=lambda kv: -kv[1]))
        lines = [
            f"Saved {os.path.relpath(out, root)} ({format})",
            f"Size: {size / 1e9:.2f} GB (the bf16 original is about {src_bytes / 1e9:.2f} GB)",
            f"Layers: {kinds}",
            f"Drift: {drift * 100:.2f} % on average, worst {stats['worst'][0] * 100:.1f} % ({stats['worst'][1]})",
            "  (how far the quantized weights sit from the originals, relative to their size: fp8 about 2-3 %, "
            "Q6_K about 2 %, Q4_K about 7 %, Q2_K about 30 %. Lower = closer)",
            f"Time: {time.time() - t0:.0f} s",
        ]
        if format in ("mxfp8 (50 series)", "nvfp4 (50 series)"):
            lines.append("Note: this format only runs fast on RTX 50 series cards.")
        if is_gguf:
            lines.append("Load it with Unet Loader (GGUF) from ComfyUI-GGUF.")
        params = sum(math.prod(src.shape(k)) for k in keys)
        if params < 6e9 and format in ("GGUF Q3_K_M", "GGUF Q3_K_S", "GGUF Q2_K", "int4 (convrot w4a4)"):
            lines.append(f"Note: this model has about {params / 1e9:.1f}B parameters. Below about 6B, 3-bit and lower lose a "
                         "lot (on Anima 2B, Q2_K is noise): check the images, Q4_K_M or higher is safer.")
        lines += src.notes
        report = "\n".join(lines)
        print("[LC Model Quantize]\n" + report)
        return {"ui": {"text": [report]}, "result": (report,)}

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _estimate(src, keys, plan, is_gguf):
        total = 0
        for k in keys:
            shape, q = src.shape(k), plan[k]
            numel = math.prod(shape)
            if q is None:
                total += numel * (4 if is_gguf and (len(shape) <= 1 or numel <= (1 << 20)) else 2)
            elif is_gguf:
                _, bs, size = gq.TYPES[q]
                total += numel // bs * size
            else:
                total += numel // (2 if q in ("nvfp4", "convrot_w4a4") else 1) + numel // 16
        return total

    @staticmethod
    def _track(stats, kind, w, deq, k):
        stats["kinds"][kind] = stats["kinds"].get(kind, 0) + 1
        if deq is None:
            return
        rel = float((deq.float() - w).abs().mean() / w.abs().mean().clamp(min=1e-12))
        n = w.numel()
        stats["err"] += rel * n
        stats["n"] += n
        if rel > stats["worst"][0]:
            stats["worst"] = (rel, k)

    @staticmethod
    def _keep_dtype(src, k, w):
        """Unquantized tensors: 1-D stays as stored, the rest bf16 (fp16 stays fp16)."""
        d = src.dtype(k)
        if not w.is_floating_point():
            return w
        if w.dim() <= 1:
            return w.to(d if d in (torch.float32, torch.float16, torch.bfloat16) else torch.float32)
        return w.to(torch.float16 if d == torch.float16 else torch.bfloat16)

    def _write_native(self, src, keys, plan, out, dev, stats, pbar, fmt_name):
        from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class

        meta = {"format": "pt", "lc_quantize": json.dumps({"format": fmt_name})}
        w_out = _StreamSafetensors(out, sum(len(k) + 160 for k in keys) * 4, meta)
        try:
            for k in keys:
                mm.throw_exception_if_processing_interrupted()
                fmt = plan[k]
                w = src.get(k, dev)
                done = False
                if fmt is not None:
                    try:
                        conf = QUANT_ALGOS[fmt]
                        layout = conf["comfy_tensor_layout"]
                        L = get_layout_class(layout)
                        qconf = {"format": fmt}
                        if fmt == "int8_tensorwise":
                            rot = w.shape[1] % 256 == 0
                            qd, params = L.quantize(w.to(torch.bfloat16), per_channel=True, convrot=rot,
                                                    convrot_groupsize=256)
                            if rot:
                                qconf.update(convrot=True, convrot_groupsize=256)
                        elif fmt == "convrot_w4a4":
                            qd, params = L.quantize(w.to(torch.bfloat16), convrot_groupsize=256)
                            qconf["convrot_groupsize"] = 256
                        else:
                            qd, params = L.quantize(w.to(torch.bfloat16), scale="recalculate")
                        qt = QuantizedTensor(qd, layout, params)
                        deq = qt.dequantize()
                        for name, t in qt.state_dict(k).items():
                            w_out.add(name, t)
                        tag = json.dumps(qconf).encode("utf-8")
                        tag += b" " * (-len(tag) % 8)  # JSON allows trailing spaces; keeps the file aligned
                        w_out.add(k[: -len("weight")] + "comfy_quant", torch.tensor(list(tag), dtype=torch.uint8))
                        self._track(stats, fmt, w, deq, k)
                        done = True
                    except Exception as e:  # a shape this format can not take: keep it full precision
                        print(f"[LC Model Quantize] {k}: kept full precision ({type(e).__name__}: {e})")
                if not done:
                    w_out.add(k, self._keep_dtype(src, k, w))
                    self._track(stats, "full precision", w, None, k)
                del w
                pbar.update(1)
            w_out.close()
        except BaseException:
            w_out.abort()
            raise

    def _write_gguf(self, src, keys, plan, out, dev, stats, pbar):
        import gguf
        import numpy as np

        too_long = [k for k in keys if len(k) >= 64]
        if too_long:
            raise ValueError(f"LC Model Quantize: GGUF tensor names must be under 64 characters: {too_long[:3]}")
        writer = gguf.GGUFWriter(path=None, arch=_gguf_arch(keys))
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
        writer.add_string("lc.quantize", "LC Model Quantize")
        kinds = {}
        for k in keys:
            shape, q = src.shape(k), plan[k]
            numel = math.prod(shape)
            if q is not None:
                qt = getattr(gguf.GGMLQuantizationType, q)
                _, bs, size = gq.TYPES[q]
                byte_shape = (shape[0], shape[1] // bs * size)
                writer.add_tensor_info(k, byte_shape, np.dtype(np.uint8), numel // bs * size, raw_dtype=qt)
                kinds[k] = q
            elif len(shape) <= 1 or numel <= (1 << 20) or not k.endswith(".weight") or not src.dtype(k).is_floating_point:
                # ComfyUI-GGUF hands F32 / F16 tensors over as they are; anything else is unpacked only inside layers,
                # so a raw parameter (Krea 2's mod.lin) has to be F32 or F16 or the model gets packed bytes
                writer.add_tensor_info(k, shape if shape else (1,), np.dtype(np.float32), max(1, numel) * 4)
                kinds[k] = "F32"
            else:
                writer.add_tensor_info(k, shape, np.dtype(np.float16), numel * 2)
                kinds[k] = "F16"
        writer.write_header_to_file(path=out)
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        try:
            for k in keys:
                mm.throw_exception_if_processing_interrupted()
                w = src.get(k, dev)
                kind = kinds[k]
                if kind == "F32":
                    arr = w.float().reshape(tuple(w.shape) or (1,)).cpu().numpy()
                    self._track(stats, "full precision", w, None, k)
                elif kind == "F16":
                    arr = w.to(torch.float16).cpu().numpy()
                    self._track(stats, "full precision", w, None, k)
                else:
                    raw, deq = gq.quantize(w, kind)
                    arr = raw.cpu().numpy()
                    self._track(stats, kind, w, deq, k)
                writer.write_tensor_data(np.ascontiguousarray(arr))
                del w
                pbar.update(1)
        finally:
            writer.close()


NODE_CLASS_MAPPINGS = {"LCModelQuantize": LCModelQuantize}
NODE_DISPLAY_NAME_MAPPINGS = {"LCModelQuantize": "LC Model Quantize 🗜️"}
