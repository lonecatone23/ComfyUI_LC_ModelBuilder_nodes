"""
LC Strip Metadata 🧹 — cleans the header of a model you already saved, without loading or re-saving its weights.

A .safetensors file is an 8-byte header length, a JSON header (tensor table + __metadata__), then the raw tensor bytes.
Only the __metadata__ part changes: the tensor table and every byte of the weights are copied as they are, so the
result loads exactly like the original. Uses the same "necessary only" rules as the save nodes' metadata switch.
"""

import json
import os
import shutil
import struct

import folder_paths

from .lc_checkpoint_save import trim_metadata

NODE_NAME = "LCStripMetadata"
_FOLDERS = ("diffusion_models", "checkpoints", "loras", "vae", "text_encoders")
_MAX_HEADER = 100 * 1024 * 1024
_KEEP_ALWAYS = ("_quantization_metadata",)  # other tools' quant formats need it to load


def _files():
    out = []
    for folder in _FOLDERS:
        try:
            names = folder_paths.get_filename_list(folder)
        except Exception:
            continue
        out += [f"{folder}/{n}" for n in names if n.lower().endswith((".safetensors", ".sft"))]
    return sorted(set(out)) or ["(no .safetensors files found)"]


def _resolve(entry):
    folder, _, name = entry.partition("/")
    path = folder_paths.get_full_path(folder, name) if folder in _FOLDERS else None
    if not path or not os.path.isfile(path):
        raise ValueError(f"LC Strip Metadata: can't find {entry}.")
    return path


def strip_file(src, dst, mode):
    """Writes dst with a cleaned header. Returns (old_meta, new_meta)."""
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > _MAX_HEADER:
            raise ValueError("LC Strip Metadata: this doesn't look like a .safetensors file.")
        header = json.loads(f.read(n))
        old = header.pop("__metadata__", {}) or {}
        if mode == "necessary only":
            new = trim_metadata(old, "necessary only")
        else:
            new = {k: v for k, v in old.items() if k in _KEEP_ALWAYS}
        if new:
            header = {"__metadata__": new, **header}
        raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
        raw += b" " * (-len(raw) % 8)  # keep the tensor data 8-byte aligned, like every safetensors writer does
        with open(dst, "wb") as out:
            out.write(struct.pack("<Q", len(raw)))
            out.write(raw)
            shutil.copyfileobj(f, out, 16 * 1024 * 1024)  # the weights, byte for byte
    return old, new


class LCStripMetadata:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (_files(), {"tooltip": "A saved model, checkpoint, LoRA, VAE or text encoder (.safetensors)."}),
                "mode": (["necessary only", "everything"], {
                    "default": "necessary only",
                    "tooltip": "necessary only = keep what loading needs plus the architecture and trigger words "
                               "(same as the save nodes' switch). everything = an empty header (quant settings other "
                               "tools need to load are still kept)."}),
                "save_as": (["new file (_clean)", "replace the original"], {
                    "default": "new file (_clean)",
                    "tooltip": "new file = writes name_clean.safetensors next to the original and leaves it alone. "
                               "replace = swaps the original for the cleaned copy once it's fully written."}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off skips it (nothing is written)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = "LC ModelBuilder"
    OUTPUT_NODE = True
    DESCRIPTION = ("Cleans the metadata out of a model you already saved: the embedded workflow, training details, "
                   "titles. The weights are copied byte for byte (never loaded or re-saved), so it loads exactly as "
                   "before. Same rules as the save nodes' metadata switch.")

    def run(self, file, mode, save_as, enabled=True):
        if not enabled:
            return {"ui": {"text": ["Off: nothing written."]}, "result": ("Off: nothing written.",)}
        src = _resolve(file)
        stem, ext = os.path.splitext(src)
        replace = save_as == "replace the original"
        dst = stem + ".lc_strip_tmp" if replace else stem + "_clean" + ext
        if not replace and os.path.exists(dst):
            raise ValueError(f"LC Strip Metadata: {os.path.basename(dst)} already exists. Delete or rename it first.")
        try:
            old, new = strip_file(src, dst, mode)
            if replace:
                os.replace(dst, src)  # only after the new file is complete
        except PermissionError as e:
            if os.path.exists(dst):
                os.remove(dst)
            raise ValueError("LC Strip Metadata: the file is in use (loaded somewhere?). Free the model or restart "
                             f"ComfyUI, then try again. ({e})") from e
        except Exception:
            if os.path.exists(dst):
                os.remove(dst)
            raise
        removed = sorted(k for k in old if k not in new)
        size = sum(len(str(old[k])) for k in removed)
        target = src if replace else dst
        msg = (f"{os.path.basename(target)}: removed {len(removed)} field(s), {size:,} characters"
               + (f" ({', '.join(removed[:8])}{' ...' if len(removed) > 8 else ''})" if removed else "")
               + f". Kept: {', '.join(new) if new else 'nothing'}.")
        print(f"[LC ModelBuilder] {msg}")
        return {"ui": {"text": [msg]}, "result": (msg,)}


NODE_CLASS_MAPPINGS = {NODE_NAME: LCStripMetadata}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_NAME: "LC Strip Metadata 🧹"}
