"""
LC Fingerprint Library: a small library of model and LoRA fingerprints, so a new file can be identified later
("most likely Krealism V3.1, with LoRA Y at about 0.6 and Z at about 0.3"), even after the originals are deleted.

A fingerprint is a sketch of every layer: a fixed random projection of its weights down to SKETCH numbers (count
sketch). Sketches keep distances and they can be subtracted: sketch(model) - sketch(base) = sketch(model - base). So a
mystery model is matched to its closest base, and the difference is explained as a mix of library LoRAs.

Files: models/fingerprints/<family>/<kind>/<name>.safetensors (about 1 MB each) + models/fingerprints/index.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import time
import zlib

import torch

import comfy.model_management
import comfy.utils
import folder_paths
from safetensors import safe_open

SKETCH = 1024  # numbers per layer
CHUNK = 1 << 22  # elements per hashing chunk (fixed, so every run hashes the same)
LIB = os.path.join(folder_paths.models_dir, "fingerprints")
_PREFIX = re.compile(r"^(model_diffusion_model_|diffusion_model_|base_model_model_|transformer_|unet_|net_|model_)+")
_LORA_KEY = re.compile(r"(lora_up|lora_down|lora_a|lora_b|lora\.up|lora\.down|hada_w|lokr_w|\.alpha$|lora_mid|dora_scale)", re.I)
_QUANT = ("U8", "I8", "I4", "U4")


# ------------------------------------------------------------------ reading files
def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", {}) or {}
    return h, meta


def _norm(key: str) -> str:
    """One name for a layer in every format: lower case, dots as underscores, wrappers stripped, no .weight."""
    k = key.lower()
    for tail in (".weight", ".lora_up", ".lora_down", ".lora_a", ".lora_b"):
        if k.endswith(tail):
            k = k[: -len(tail)]
    k = k.replace(".", "_")
    if k.startswith("lora_unet_"):
        k = k[len("lora_unet_"):]
    elif k.startswith("lora_transformer_"):
        k = k[len("lora_transformer_"):]
    return _PREFIX.sub("", k)


def _kind(keys, path):
    """'lora', 'model', or what else it is (skipped)."""
    low = [k.lower() for k in keys]
    if any(_LORA_KEY.search(k) for k in low):
        return "lora"
    if any(k.startswith("model.diffusion_model.") for k in low):
        return "model"  # a full checkpoint: model + VAE + text encoder
    j = " ".join(low[:4000])
    if "encoder.down" in j or ("decoder.up" in j and "encoder." in j) or "post_quant_conv" in j:
        return "vae"
    if "text_model" in j or "embed_tokens" in j or "encoder.block" in j or "model.layers.0.self_attn" in j:
        return "text encoder"
    if ("conv_first" in j or "body." in j) and len(low) < 1500 and "blocks" not in j:
        return "upscaler"
    return "model"


def _family(keys, shapes=None):
    """The model family from its layer names (model keys or LoRA module names, normalized)."""
    n = " ".join(k.lower().replace(".", "_") for k in keys[:6000])
    has = lambda s: s in n
    if has("txtfusion") or (has("attn_wk") and has("attn_wq") and has("attn_gate")):
        return "Krea 2"
    if has("cap_embedder") or (has("noise_refiner") and has("context_refiner")):
        return "Z-Image"
    if has("llm_adapter"):
        return "Anima"
    if has("shared_block_modulation") or (has("video_embedder") and has("text_embedder")):
        return "MOSS Image"
    if has("token_refiner") or has("audio_patch_proj") or has("adaln_basis"):
        return "MiniMax H3"
    if has("adaln_single") and has("audio"):
        return "LTX 2"
    if has("adaln_single"):
        return "LTX"
    if has("content_mode_blocks"):
        return "Flux 3 Klein"
    if has("double_stream_blocks"):
        return "HiDream"
    if has("distilled_guidance_layer"):
        return "Chroma"
    if has("double_blocks") and has("single_blocks"):
        return "Flux.1" if has("vector_in") or has("guidance_in") else "Flux.2 / Klein"
    if has("transformer_blocks") and (has("img_mod") or has("txt_mod") or has("img_mlp")):
        width = 0
        for k, s in (shapes or {}).items():
            if k.lower().replace(".", "_").endswith("img_in_weight") and s:
                width = s[0]
        return "Qwen-Image 2.1" if width >= 4096 else "Qwen-Image"
    if has("cross_attn") and has("self_attn") and (has("ffn_0") or has("patch_embedding")):
        return "Wan"
    if has("input_blocks") or has("down_blocks"):
        return "SDXL" if has("label_emb") or has("add_embedding") else "SD 1.5"
    return "unknown"


# rounding noise of a stored weight (relative rms), measured on Krealism V3.1 against its own bf16 file
_NOISE = {"bf16": 0.0015, "fp8": 0.026, "mxfp8": 0.026, "int8": 0.010, "w4a8": 0.072, "nvfp4": 0.092, "4bit": 0.09}


def precision_of(path, h):
    """(what the file is stored in, its rounding noise) from the header and the quant tags."""
    dt = {v["dtype"] for k, v in h.items() if k.endswith(".weight") and len(v["shape"]) >= 2}
    tags = [k for k in h if k.endswith(".comfy_quant")]
    fmt = ""
    if tags:
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                fmt = json.loads(bytes(f.get_tensor(tags[0]).tolist()).decode()).get("format", "").lower()
        except Exception:
            fmt = ""
    if not fmt:  # some converters (StarNodes) write the format into the header's metadata instead
        try:
            fmt = str(json.loads(_header(path)[1].get("_quantization_metadata", "{}")).get("format", "")).lower()
        except Exception:
            fmt = ""
    if fmt:
        for key, name in (("nvfp4", "nvfp4"), ("w4a8", "w4a8"), ("w4a4", "4bit"), ("int4", "4bit"), ("fp4", "4bit"), ("mxfp8", "mxfp8"),
                          ("int8", "int8"), ("float8", "fp8"), ("fp8", "fp8")):
            if key in fmt:
                return name, _NOISE[name]
        return "quantized", 0.05
    if dt & {"F8_E4M3", "F8_E5M2"}:
        return "fp8", _NOISE["fp8"]
    if dt & {"U8", "I8", "U4", "I4"}:
        return "quantized", 0.05
    return "bf16", _NOISE["bf16"]


def _scale_for(h, key):
    """fp8 weights saved with a separate scale (Comfy: <name>.scale_weight, others: <name>.weight_scale)."""
    base = key[: -len(".weight")] if key.endswith(".weight") else key
    for s in (base + ".scale_weight", base + ".weight_scale"):
        if s in h:
            return s
    return None


_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn,
       "F8_E5M2": torch.float8_e5m2, "U8": torch.uint8, "I8": torch.int8, "I32": torch.int32, "I64": torch.int64}


def _keymap_path(family):
    return os.path.join(LIB, "keymaps", f"{_safe(family)}.json")


def build_keymap(path, family):
    """ComfyUI's own LoRA name table for this model family (diffusers / kohya / native names -> the model's layers),
    built from the file's header on the meta device (no weights loaded), saved once per family."""
    dest = _keymap_path(family)
    if os.path.exists(dest):
        return
    import comfy.lora
    import comfy.model_detection
    h, _ = _header(path)
    sd = {k: torch.empty(v["shape"], dtype=_DT.get(v["dtype"], torch.float32), device="meta") for k, v in h.items()}
    prefix = "model.diffusion_model." if any(k.startswith("model.diffusion_model.") for k in sd) else ""
    cfg = comfy.model_detection.model_config_from_unet(sd, prefix)
    if cfg is None:
        return
    model = cfg.get_model({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}, "", device="meta")
    km = comfy.lora.model_lora_keys_unet(model, {})
    table = {}
    for lk, target in km.items():
        if isinstance(target, tuple):
            table[lk] = [target[0], list(target[1]) if len(target) > 1 and target[1] is not None else None]
        else:
            table[lk] = [target, None]
    shapes = {_norm(k): list(v.shape) for k, v in model.state_dict().items() if k.endswith(".weight") and v.ndim >= 2}
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump({"family": family, "map": table, "shapes": shapes}, open(dest, "w", encoding="utf-8"))


def _keymaps():
    d = os.path.join(LIB, "keymaps")
    out = {}
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn.endswith(".json"):
                try:
                    j = json.load(open(os.path.join(d, fn), encoding="utf-8"))
                    out[j["family"]] = j
                except Exception:
                    pass
    return out


_PARTS = ((".lora_up.weight", "up"), (".lora_down.weight", "down"), (".lora_b.weight", "up"), (".lora_a.weight", "down"),
          (".lora.up.weight", "up"), (".lora.down.weight", "down"), (".alpha", "alpha"),
          (".lokr_w1", "k1"), (".lokr_w2", "k2"), (".lokr_w1_a", "k1a"), (".lokr_w1_b", "k1b"), (".lokr_w2_a", "k2a"),
          (".lokr_w2_b", "k2b"), (".hada_w1_a", "h1a"), (".hada_w1_b", "h1b"), (".hada_w2_a", "h2a"), (".hada_w2_b", "h2b"),
          (".hada_t1", "t1"), (".hada_t2", "t2"), (".lokr_t2", "t2"))


def _lora_modules(h):
    """{module key as written in the LoRA: {part: tensor name}}: plain LoRA (up / down), LoKr or LoHa."""
    groups = {}
    for k in h:
        kl = k.lower()
        for a, b in _PARTS:
            if kl.endswith(a):
                groups.setdefault(k[: -len(a)], {})[b] = k
                break
    ok = {}
    for m, p in groups.items():
        lora = "up" in p and "down" in p
        lokr = ("k1" in p or ("k1a" in p and "k1b" in p)) and ("k2" in p or ("k2a" in p and "k2b" in p))
        loha = all(x in p for x in ("h1a", "h1b", "h2a", "h2b")) and "t1" not in p  # Tucker LoHa (conv) is left out
        if (lora or lokr or loha) and "t2" not in p:
            ok[m] = p
    return ok


def _delta(f, parts):
    """The change one module makes to its layer, scaled the way ComfyUI applies it at strength 1."""
    g = lambda n: f.get_tensor(parts[n]).float()
    alpha = float(g("alpha")) if "alpha" in parts else None
    if "up" in parts:
        up, down = g("up"), g("down")
        rank = down.shape[0]
        return (up.flatten(1) @ down.flatten(1)) * ((alpha if alpha is not None else rank) / rank)
    if "h1a" in parts:
        rank = g("h1b").shape[0]
        d = (g("h1a") @ g("h1b")) * (g("h2a") @ g("h2b"))
        return d * ((alpha if alpha is not None else rank) / rank)
    dim = None
    w1 = g("k1") if "k1" in parts else None
    if w1 is None:
        dim = g("k1b").shape[0]
        w1 = g("k1a") @ g("k1b")
    w2 = g("k2") if "k2" in parts else None
    if w2 is None:
        dim = g("k2b").shape[0]
        w2 = g("k2a").flatten(1) @ g("k2b").flatten(1)
    scale = alpha / dim if alpha is not None and dim is not None else 1.0
    return torch.kron(w1.flatten(1) if w1.ndim > 2 else w1, w2.flatten(1) if w2.ndim > 2 else w2) * scale


def _lora_family(mods, raw_keys):
    """(family, its keymap or None): the library family whose LoRA name table knows most of this LoRA's modules."""
    best, best_n, best_km = None, 0, None
    for fam, km in _keymaps().items():
        n = sum(1 for m in mods if m in km["map"])
        if n > best_n:
            best, best_n, best_km = fam, n, km
    if best and best_n >= max(3, len(mods) // 4):
        return best, best_km
    return _family(raw_keys), None


# ------------------------------------------------------------------ sketching
def _seed(name: str, numel: int) -> int:
    return zlib.crc32(f"{name}|{numel}".encode()) & 0x7FFFFFFF


@torch.no_grad()
def sketch(name: str, w: torch.Tensor, device) -> torch.Tensor:
    """Count sketch of one weight: SKETCH numbers. Linear: sketch(a - b) = sketch(a) - sketch(b) for the same layer."""
    flat = w.reshape(-1).to(device, torch.float32)
    n = flat.numel()
    out = torch.zeros(SKETCH, device=device, dtype=torch.float32)
    g = torch.Generator(device=device)
    g.manual_seed(_seed(name, n))
    for s in range(0, n, CHUNK):
        part = flat[s:s + CHUNK]
        m = part.numel()
        idx = torch.randint(0, SKETCH, (m,), generator=g, device=device)
        sign = torch.randint(0, 2, (m,), generator=g, device=device, dtype=torch.int8).float().mul_(2).sub_(1)
        out.index_add_(0, idx, part * sign)
    return out.cpu()


def _rng_tag():
    """Where sketches are hashed (device type + torch version): the random hash differs between CPU and GPU, and maybe
    between torch versions, so sketches only line up with library entries made the same way."""
    return f"{comfy.model_management.get_torch_device().type}, torch {torch.__version__.split('+')[0]}"


def _rng_note(entries):
    """A warning line when library entries were hashed on another device / torch (entries from before the tag: unknown)."""
    here = _rng_tag()
    other = sorted({e["rng"] for e in entries if e.get("rng") and e["rng"] != here})
    if not other:
        return []
    n = sum(1 for e in entries if e.get("rng") and e["rng"] != here)
    return [f"Warning: {n} library entr{'y was' if n == 1 else 'ies were'} fingerprinted on {', '.join(other)}, this run "
            f"on {here}. Those fingerprints do not line up with this one: rebuild them here (LC Fingerprint Library, "
            "skip_known off)."]


def _model_tensors(path, h):
    """(normalized name, weight) for every 2D+ weight of a model file, read one at a time. Quantized formats that need
    their pack to decode come back as None (the caller loads those through ComfyUI)."""
    weights = [k for k, v in h.items() if k.endswith(".weight") and len(v["shape"]) >= 2]
    if any(k.startswith("model.diffusion_model.") for k in weights):  # full checkpoint: leave the VAE and text encoder out
        weights = [k for k in weights if k.startswith("model.diffusion_model.")]
    if any(h[k]["dtype"] in _QUANT for k in weights) or any("_qdata" in k or k.endswith(".comfy_quant") for k in h):
        return None
    def gen():
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in weights:
                t = f.get_tensor(k)
                if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    t = t.float()
                    sk = _scale_for(h, k)
                    if sk:
                        t = t * f.get_tensor(sk).float()
                yield _norm(k), t
    return gen()


def _loaded_tensors(path):
    """Quantized files: load through ComfyUI (it knows every format), then decode one weight at a time."""
    import comfy.sd
    from .lc_model_merge_any import _dequant, _is_quant, _param
    try:
        model = comfy.sd.load_diffusion_model(path)
    except Exception:
        out = comfy.sd.load_checkpoint_guess_config(path, output_vae=False, output_clip=False, output_clipvision=False)
        model = out[0]
    dev = comfy.model_management.get_torch_device()
    for k in model.model_state_dict().keys():
        if not k.startswith("diffusion_model.") or not k.endswith(".weight"):
            continue
        t = _param(model, k)
        if t is None or getattr(t, "ndim", 0) < 2:
            continue
        yield _norm(k), (_dequant(t, dev) if _is_quant(t) else t)
    del model
    comfy.model_management.soft_empty_cache()


def _lora_deltas(path, mods, km):
    """(layer name, the change the LoRA makes to it = up x down x alpha / rank) per target layer. With the family's name
    table the LoRA's names become the model's own (split q / k / v LoRAs add into their slice of the fused layer)."""
    full = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for mod, parts in mods.items():
            try:
                d = _delta(f, parts)
            except RuntimeError:
                continue
            if km is None or mod not in km["map"]:
                yield _norm(mod), d, tuple(d.shape)
                continue
            target, off = km["map"][mod]
            name = _norm(target)
            if off is None:
                if name in full:
                    try:
                        full[name] = full[name] + d.reshape(full[name].shape)
                    except RuntimeError:
                        continue
                else:
                    full[name] = d
                continue
            shape = km["shapes"].get(name)
            if not shape:
                continue
            if name not in full:
                full[name] = torch.zeros(shape[0], int(torch.tensor(shape[1:]).prod()))
            dim, start, length = off
            sl = [slice(None)] * 2
            sl[dim] = slice(start, start + length)
            try:
                full[name][tuple(sl)] += d.reshape(full[name][tuple(sl)].shape)
            except RuntimeError:
                continue
    for name, d in full.items():
        yield name, d, tuple(d.shape)


# ------------------------------------------------------------------ the library
def _index():
    p = os.path.join(LIB, "index.json")
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return {}


def _save_index(ix):
    os.makedirs(LIB, exist_ok=True)
    tmp = os.path.join(LIB, "index.json.tmp")
    json.dump(ix, open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(LIB, "index.json"))


def _stamp(path):
    st = os.stat(path)
    return f"{st.st_size}:{int(st.st_mtime)}"


def _safe(name):
    return re.sub(r"[^\w\-. ]+", "_", name).strip()[:120] or "model"


def fingerprint_file(path, device):
    """-> (kind, family, {layer: sketch}, {layer: shape}, note) or (kind, family, None, None, why skipped)."""
    h, meta = _header(path)
    keys = list(h)
    kind = _kind(keys, path)
    shapes = {k: v["shape"] for k, v in h.items()}
    if kind == "lora":
        mods = _lora_modules(h)
        fam, km = _lora_family(mods, keys)
        sk, shp = {}, {}
        for name, d, s in _lora_deltas(path, mods, km):
            sk[name], shp[name] = sketch(name, d, device), list(s)
        return kind, fam, (sk or None), shp, ("" if sk else "no LoRA layers it can read (Tucker / conv-only LoRA)")
    if kind != "model":
        return kind, "", None, None, f"a {kind}, not a model or LoRA"
    fam = _family([k for k in keys if k.startswith("model.diffusion_model.")] or keys, shapes)
    if fam == "unknown":
        return kind, fam, None, None, "not an image or video model the library knows"
    tensors = _model_tensors(path, h)
    note = ""
    if tensors is None:
        tensors, note = _loaded_tensors(path), "quantized: decoded through ComfyUI"
    sk, shp = {}, {}
    for name, t in tensors:
        sk[name], shp[name] = sketch(name, t, device), list(t.shape)
    if len(sk) < 40:  # a ControlNet patch or another partial file, not a whole model
        return kind, fam, None, None, f"only {len(sk)} layer{'s' if len(sk) != 1 else ''}: a partial file (ControlNet patch, adapter), not a whole model"
    try:
        build_keymap(path, fam)  # so this family's LoRAs, in any naming, line up with its layers
    except Exception as e:
        print(f"[LC Fingerprint] no LoRA name table for {fam}: {e}")
    return kind, fam, sk, shp, note


def add_to_library(path, sk, shp, kind, family, note=""):
    name = os.path.splitext(os.path.basename(path))[0]
    fid = hashlib.sha1(os.path.abspath(path).lower().encode()).hexdigest()[:10]
    rel = os.path.join(_safe(family or "unknown"), kind, f"{_safe(name)}_{fid}.safetensors")
    dest = os.path.join(LIB, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    meta = {"name": name, "kind": kind, "family": family, "source": os.path.abspath(path), "note": note,
            "shapes": json.dumps(shp), "sketch": str(SKETCH), "added": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rng": _rng_tag()}
    comfy.utils.save_torch_file({k: v.contiguous() for k, v in sk.items()}, dest, metadata=meta)
    ix = _index()
    ix[os.path.abspath(path)] = {"stamp": _stamp(path), "file": rel, "name": name, "kind": kind, "family": family,
                                 "rng": _rng_tag()}
    _save_index(ix)
    return rel


def load_entries(family=None, kind=None):
    out = []
    for src, e in _index().items():
        if family and e.get("family") != family or kind and e.get("kind") != kind:
            continue
        p = os.path.join(LIB, e["file"])
        if not os.path.exists(p):
            continue
        sd = comfy.utils.load_torch_file(p, safe_load=True)
        out.append({**e, "source": src, "sketch": sd})
    return out


# ------------------------------------------------------------------ identifying
def _rel(a, b):
    """Relative difference |a - b| / |b| between two layers, from their sketches."""
    nb = float(b.norm())
    return float((a - b).norm()) / nb if nb > 0 else 1.0


def compare_models(sk, entry):
    """(similarity like a cosine, shared layer count) of a model against a library model."""
    shared = [k for k in sk if k in entry["sketch"] and sk[k].shape == entry["sketch"][k].shape]
    if not shared:
        return 0.0, 0
    rels = sorted(_rel(sk[k], entry["sketch"][k]) for k in shared)
    med = rels[len(rels) // 2]
    return max(0.0, 1.0 - med * med / 2.0), len(shared)


def explain(sk, base, loras, top=6):
    """The model's difference from its base as a mix of library LoRAs (non-negative least squares).
    -> ([(name, strength)], share of the difference they explain)."""
    layers = [k for k in sk if k in base["sketch"]]
    cand = [e for e in loras if any(k in e["sketch"] for k in layers)]
    if not layers or not cand:
        return [], 0.0
    d = torch.cat([sk[k] - base["sketch"][k] for k in layers])
    cols = []
    for e in cand:
        cols.append(torch.cat([e["sketch"][k] if k in e["sketch"] else torch.zeros(SKETCH) for k in layers]))
    A = torch.stack(cols, 1)
    try:
        from scipy.optimize import nnls
        c, _ = nnls(A.double().numpy(), d.double().numpy(), maxiter=5000)
        c = torch.from_numpy(c).float()
    except Exception:  # no scipy: projected gradient
        c = torch.zeros(A.shape[1])
        lr = 1.0 / float((A.T @ A).diagonal().max().clamp(min=1e-12))
        for _ in range(3000):
            c = (c - lr * (A.T @ (A @ c - d))).clamp(min=0)
    resid = d - A @ c
    dd = float(d @ d)
    explained = 1.0 - float(resid @ resid) / dd if dd > 0 else 0.0
    found = sorted(((cand[i]["name"], float(c[i])) for i in range(len(cand)) if c[i] > 0.05), key=lambda x: -x[1])[:top]
    return found, explained


def identify_sketch(kind, family, sk, precision=("bf16", 0.0015)):
    """-> (report lines, heatmap image or None)."""
    lines = []
    if kind == "lora":
        same = []
        lib_loras = load_entries(family, "lora")
        lines += _rng_note(lib_loras)
        for e in lib_loras:
            shared = [k for k in sk if k in e["sketch"]]
            if not shared:
                continue
            a = torch.cat([sk[k] for k in shared])
            b = torch.cat([e["sketch"][k] for k in shared])
            cos = float(a @ b) / max(1e-12, float(a.norm() * b.norm()))
            same.append((cos, e["name"]))
        same.sort(reverse=True)
        hits = [f"{n} ({c:.3f})" for c, n in same[:3] if c > 0.5]
        lines.append("Matches in the library: " + (", ".join(hits) if hits else "none (a LoRA the library has not seen)."))
        if same and same[0][0] > 0.98:
            lines.append(f"It is {same[0][1]}, or a copy of it.")
        return lines, None
    from . import lc_fingerprint_recipe as rc
    models = load_entries(family, "model")
    if not models:
        return [f"No {family} models in the library yet to compare with."], None
    lines += _rng_note(models)
    store, noise = precision
    scored = sorted(((compare_models(sk, e)[0], e) for e in models), key=lambda x: -x[0])
    exclude = []

    def same(e):  # every layer (95th percentile) within the rounding of both files: a finetune fails on its changed layers
        allowed = 1.3 * (noise ** 2 + (_NOISE["fp8"] if rc._low_precision(e) else _NOISE["bf16"]) ** 2) ** 0.5 + 0.003
        return rc._rel95({"sketch": sk}, e) < allowed
    if same(scored[0][1]):  # the file itself (or a copy) is in the library: name it, then show how it was made
        lines.append(f"It is {scored[0][1]['name']} (in your library)" + (f", stored as {store}." if store != "bf16" else "."))
        # leave out only it and its own copies at other precisions (at fp8 and lower every close relative clears the floor)
        best = scored[0][1]
        exclude = [e["name"] for _, e in scored if e is best or rc._same_model(e, best, rc._low_precision(e))]
        lines.append("How it was made, from the rest of the library:")
    rest = [e for e in models if e["name"] not in exclude]
    if not rest:
        return lines, None
    base = rc.pick_base(rest, sk)
    r = rc.recipe(sk, models, load_entries(family, "lora"), base, noise=noise, exclude=exclude)
    lines += rc.words(r)
    if noise > 0.005 and not lines[-1].startswith("Can't read"):
        lines.append(f"({store} file: its rounding (about {noise:.1%} per weight) is allowed for. It blurs small changes, so "
                     f"the shares are rougher and the confidence is lower than for a bf16 file.)")
    return lines, rc.heatmap(r, f"{family}: what it is made of")


def _walk(paths, recursive):
    for p in paths:
        p = p.strip().strip('"').strip("'")
        if not p:
            continue
        if p in folder_paths.folder_names_and_paths:  # "loras", "diffusion_models": every folder ComfyUI reads them from
            yield from _walk([d for d in folder_paths.get_folder_paths(p) if os.path.isdir(d)], recursive)
            continue
        if not os.path.isabs(p) and not os.path.exists(p):  # "models/loras": relative to the ComfyUI folder
            for root in (folder_paths.base_path, folder_paths.models_dir):
                if os.path.exists(os.path.join(root, p)):
                    p = os.path.join(root, p)
                    break
        if os.path.isfile(p):
            if p.lower().endswith(".safetensors"):
                yield p
            continue
        if not os.path.isdir(p):
            continue
        for root, dirs, files in os.walk(p):
            for fn in sorted(files):
                if fn.lower().endswith(".safetensors"):
                    yield os.path.join(root, fn)
            if not recursive:
                break


# ------------------------------------------------------------------ nodes
class LCFingerprintLibrary:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folders": ("STRING", {"multiline": True, "default": "",
                            "tooltip": "Folders (or single files) to fingerprint, one per line: full paths, paths from the ComfyUI folder (models/loras), or a model folder name (loras, diffusion_models, checkpoints) for every place ComfyUI reads it from. Every .safetensors inside is checked: "
                                       "models and LoRAs go in the library, VAEs / text encoders / upscalers are skipped."}),
                "recursive": ("BOOLEAN", {"default": True, "tooltip": "Also the folders inside them."}),
                "include": (["models and LoRAs", "models only", "LoRAs only"], {"default": "models and LoRAs"}),
                "skip_known": ("BOOLEAN", {"default": True, "tooltip": "Skip files already in the library (same size and date), so a rerun only adds what is new."}),
                "precision": (["bf16 / fp16 / fp8 only", "every precision"], {"default": "bf16 / fp16 / fp8 only",
                              "tooltip": "Models only: leave out int8, mxfp8, nvfp4 and other lower quants. They are the same model as their bf16 / fp8 file, just rounded, so they add nothing."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "build"
    OUTPUT_NODE = True
    CATEGORY = "LC Modelbuilder"
    DESCRIPTION = ("Builds the fingerprint library: goes through your folders and stores a small fingerprint (about 1 MB) "
                   "of every model and LoRA, with its family. Once a file is in, the original can go. "
                   "LC Fingerprint Identify uses the library to work out what a new file is made of.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def build(self, folders, recursive, include, skip_known, precision="bf16 / fp16 / fp8 only"):
        device = comfy.model_management.get_torch_device()
        files = list(_walk(folders.splitlines(), recursive))

        def order(pth):  # models first: they bring their family's LoRA name table for the LoRAs after them
            try:
                return 0 if _kind(list(_header(pth)[0]), pth) == "model" else 1
            except Exception:
                return 2
        files.sort(key=order)
        ix = _index()
        lines, added, skipped = [], 0, 0
        pbar = comfy.utils.ProgressBar(max(1, len(files)))
        for n, path in enumerate(files, 1):
            comfy.model_management.throw_exception_if_processing_interrupted()  # Cancel works between files
            ap = os.path.abspath(path)
            name = os.path.basename(path)
            try:
                if skip_known and ap in ix and ix[ap].get("stamp") == _stamp(path):
                    skipped += 1
                    pbar.update(1)
                    continue
                h, _ = _header(path)
                kind = _kind(list(h), path)
                if kind not in ("model", "lora") or (include == "models only" and kind != "model") or (include == "LoRAs only" and kind != "lora"):
                    lines.append(f"skip  {name}: {kind}")
                    pbar.update(1)
                    continue
                if kind == "model" and precision != "every precision" and precision_of(path, h)[0] not in ("bf16", "fp8"):
                    lines.append(f"skip  {name}: a lower quant (its bf16 / fp8 file is enough)")
                    pbar.update(1)
                    continue
                t = time.time()
                kind, fam, sk, shp, note = fingerprint_file(path, device)
                if not sk:
                    lines.append(f"skip  {name}: {note or 'no weights to fingerprint'}")
                else:
                    add_to_library(path, sk, shp, kind, fam, note)
                    added += 1
                    lines.append(f"added {name}: {kind}, {fam}, {len(sk)} layers ({time.time() - t:.0f} s){', ' + note if note else ''}")
                print(f"[LC Fingerprint] {n}/{len(files)} {lines[-1]}", flush=True)
            except Exception as e:
                lines.append(f"error {name}: {e}")
                print(f"[LC Fingerprint] {n}/{len(files)} error {name}: {e}", flush=True)
            pbar.update(1)
        head = f"{added} added, {skipped} already in the library, {len(files) - added - skipped} skipped or failed, of {len(files)} files."
        text = head + "\n\n" + "\n".join(lines)
        return {"ui": {"text": [text]}, "result": (text,)}


class LCFingerprintIdentify:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "files": ("STRING", {"multiline": True, "default": "",
                          "tooltip": "Files or folders to identify, one per line: full paths, paths from the ComfyUI folder, or a model folder name such as diffusion_models (a folder of downloads works)."}),
                "recursive": ("BOOLEAN", {"default": False}),
                "add_to_library": ("BOOLEAN", {"default": False, "tooltip": "Also store each one in the library afterwards, so it is known next time."}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("report", "heatmap")
    OUTPUT_TOOLTIPS = ("What each file is made of, in words.",
                       "Per block: which library model it came from (brighter = a bigger share), LoRAs and what is not in the library. One image per model.")
    FUNCTION = "identify"
    OUTPUT_NODE = True
    CATEGORY = "LC Modelbuilder"
    DESCRIPTION = ("What is this file made of? For a model: the recipe from the fingerprint library, block by block (which "
                   "models were merged, at what ratio, which LoRAs went in), how sure it is, and a heatmap. A LoRA is "
                   "matched against the library's LoRAs (renamed copies show up). It can only name what is in the library.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def identify(self, files, recursive, add_to_library=False):
        device = comfy.model_management.get_torch_device()
        paths = list(_walk(files.splitlines(), recursive))
        out, images = [], []
        pbar = comfy.utils.ProgressBar(max(1, len(paths)))
        for path in paths:
            comfy.model_management.throw_exception_if_processing_interrupted()
            name = os.path.basename(path)
            try:
                kind, fam, sk, shp, note = fingerprint_file(path, device)
                block = [f"## {name}", f"{kind.capitalize()}, {fam or 'family unknown'}{' (' + note + ')' if note else ''}."]
                if sk:
                    found, img = identify_sketch(kind, fam, sk, precision=precision_of(path, _header(path)[0]))
                    block += found
                    if img is not None:
                        images.append(img)
                    if add_to_library:
                        add_to_library_fn(path, sk, shp, kind, fam, note)
                        block.append("Added to the library.")
                out.append("\n".join(block))
            except Exception as e:
                out.append(f"## {name}\nerror: {e}")
            print(f"[LC Fingerprint] {out[-1]}", flush=True)
            pbar.update(1)
        text = "\n\n".join(out) if out else "No .safetensors files found."
        return {"ui": {"text": [text]}, "result": (text, _batch(images))}


def _batch(images):
    """One IMAGE batch from heatmaps of different sizes (padded to the largest)."""
    if not images:
        return torch.zeros(1, 64, 64, 3)
    h = max(i.shape[1] for i in images)
    w = max(i.shape[2] for i in images)
    out = torch.full((len(images), h, w, 3), 24 / 255.0)
    for n, i in enumerate(images):
        out[n, : i.shape[1], : i.shape[2]] = i[0]
    return out


add_to_library_fn = add_to_library

NODE_CLASS_MAPPINGS = {"LCFingerprintLibrary": LCFingerprintLibrary, "LCFingerprintIdentify": LCFingerprintIdentify}
NODE_DISPLAY_NAME_MAPPINGS = {"LCFingerprintLibrary": "LC Fingerprint Library 📚", "LCFingerprintIdentify": "LC Fingerprint Identify 🔎"}
