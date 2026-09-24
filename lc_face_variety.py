"""
LC Face Variety Scorer
----------------------
Scores how different the faces in an image batch are, to catch "same face"
in a model. Lower score = more variety.

Face detection (SCRFD det_10g) and recognition (ArcFace w600k_r50) are the
InsightFace buffalo_l models, run straight through onnxruntime. Neither the insightface
python package nor opencv is required (both are common sources of broken
installs). Only onnxruntime, numpy and torch, which install.py checks for. Detection decode, NMS and 5-point alignment below follow InsightFace's
own reference code (MIT): https://github.com/deepinsight/insightface

Credits:
  InsightFace project, Jia Guo and Jiankang Deng et al.
  ArcFace: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019
  SCRFD: Guo et al., "Sample and Computation Redistribution for Efficient Face Detection", 2021
  The buffalo_l model files are released by InsightFace for non-commercial research use only.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths

BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DET_FILE = "det_10g.onnx"
REC_FILE = "w600k_r50.onnx"
# sha256 of the two files we use, from the official buffalo_l.zip (v0.7)
SHA256 = {
    DET_FILE: "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    REC_FILE: "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}
ORT_HINT = (
    "LC Face Variety Scorer needs onnxruntime. Run this with ComfyUI's own python, then restart ComfyUI:\n"
    "    python -m pip install onnxruntime"
)
_DEFAULT_DIR = os.path.join(folder_paths.models_dir, "insightface", "models", "buffalo_l")
_SEARCH_DIRS = [
    _DEFAULT_DIR,
    os.path.join(folder_paths.models_dir, "insightface", "buffalo_l"),
    os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l"),
]

GRADES = [
    (0.20, "Clearly different people"),
    (0.35, "Different people who share a few features"),
    (0.50, "Could be sisters, or the same person in a different photo"),
    (1.01, "Same person"),
]

ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)

_CACHE = {}


def grade(score: float) -> str:
    for hi, label in GRADES:
        if score < hi:
            return label
    return GRADES[-1][1]


# ---------------------------------------------------------------- model files

def _find_models():
    for d in _SEARCH_DIRS:
        if os.path.isfile(os.path.join(d, DET_FILE)) and os.path.isfile(os.path.join(d, REC_FILE)):
            return d
    return None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_models():
    from comfy.utils import ProgressBar

    os.makedirs(_DEFAULT_DIR, exist_ok=True)
    zpath = os.path.join(_DEFAULT_DIR, "buffalo_l.zip.part")
    print(f"[LC Modelbuilder] downloading InsightFace buffalo_l (~280 MB) to {_DEFAULT_DIR}")
    try:
        with urllib.request.urlopen(BUFFALO_URL, timeout=60) as r, open(zpath, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            pbar = ProgressBar(total) if total else None
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if pbar:
                    pbar.update_absolute(done, total)
        if total and os.path.getsize(zpath) != total:
            raise RuntimeError("download was cut off")
        with zipfile.ZipFile(zpath) as z:
            if z.testzip() is not None:
                raise RuntimeError("zip is corrupt")
            for name in z.namelist():
                base = os.path.basename(name)
                if base.endswith(".onnx"):
                    tmp = os.path.join(_DEFAULT_DIR, base + ".part")
                    with z.open(name) as src, open(tmp, "wb") as dst:
                        dst.write(src.read())
                    if base in SHA256 and _sha256(tmp) != SHA256[base]:
                        os.remove(tmp)
                        raise RuntimeError(f"{base} failed its checksum")
                    os.replace(tmp, os.path.join(_DEFAULT_DIR, base))
    except Exception as e:
        raise RuntimeError(
            f"LC Face Variety Scorer: buffalo_l download failed ({e}). Nothing half-finished was kept. "
            f"Run again, or download {BUFFALO_URL} yourself and unzip det_10g.onnx and w600k_r50.onnx into {_DEFAULT_DIR}"
        ) from e
    finally:
        if os.path.exists(zpath):
            os.remove(zpath)
    print("[LC Modelbuilder] buffalo_l ready (InsightFace, non-commercial research license)")
    return _DEFAULT_DIR


def _sessions(device: str):
    if device in _CACHE:
        return _CACHE[device]
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError(f"{ORT_HINT}\n(import error: {e})") from e

    d = _find_models() or _download_models()
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    if device == "auto" and "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            ort.preload_dlls()  # borrow torch's CUDA/cuDNN DLLs, fixes most "error 126" installs
        except Exception:
            pass
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        det = ort.InferenceSession(os.path.join(d, DET_FILE), opts, providers=providers)
        rec = ort.InferenceSession(os.path.join(d, REC_FILE), opts, providers=providers)
    except Exception as e:
        if providers == ["CPUExecutionProvider"]:
            raise RuntimeError(f"LC Face Variety Scorer: could not load the face models in {d}: {e}") from e
        print(f"[LC Modelbuilder] CUDA not usable for face models, using CPU: {e}")
        det = ort.InferenceSession(os.path.join(d, DET_FILE), opts, providers=["CPUExecutionProvider"])
        rec = ort.InferenceSession(os.path.join(d, REC_FILE), opts, providers=["CPUExecutionProvider"])
    print(f"[LC Modelbuilder] face models on {det.get_providers()[0]}")
    _CACHE[device] = (det, rec)
    return _CACHE[device]


# ---------------------------------------------------------------- image helpers (no opencv)

def _resize(rgb, w, h):
    """Bilinear, half-pixel centers, no antialias (matches cv2.INTER_LINEAR)."""
    t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()


def _blob(rgb, scale, mean):
    """RGB uint8 HWC -> normalized float32 NCHW."""
    return ((rgb.astype(np.float32) - mean) * scale).transpose(2, 0, 1)[None].copy()


def _warp_affine(rgb, M, size):
    """Same as an opencv warpAffine with a black border: bilinear, zeros outside."""
    h, w = rgb.shape[:2]
    Minv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    src = Minv @ np.stack([xs.ravel(), ys.ravel(), np.ones(size * size)])
    gx = 2.0 * src[0] / (w - 1) - 1.0
    gy = 2.0 * src[1] / (h - 1) - 1.0
    grid = torch.from_numpy(np.stack([gx, gy], -1).reshape(1, size, size, 2)).float()
    t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()
    out = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()


# ---------------------------------------------------------------- SCRFD detect

def _distance2bbox(points, d):
    return np.stack([points[:, 0] - d[:, 0], points[:, 1] - d[:, 1], points[:, 0] + d[:, 2], points[:, 1] + d[:, 3]], axis=-1)


def _distance2kps(points, d):
    out = []
    for i in range(0, d.shape[1], 2):
        out.append(points[:, i % 2] + d[:, i])
        out.append(points[:, i % 2 + 1] + d[:, i + 1])
    return np.stack(out, axis=-1)


def _nms(dets, thresh=0.4):
    x1, y1, x2, y2, s = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


def _detect(det, rgb, size=640, thresh=0.5):
    h, w = rgb.shape[:2]
    if h / w > 1.0:
        nh, nw = size, int(size / (h / w))
    else:
        nw, nh = size, int(size * (h / w))
    scale = nh / h
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = _resize(rgb, nw, nh)
    blob = _blob(canvas, 1.0 / 128.0, 127.5)
    outs = det.run(None, {det.get_inputs()[0].name: blob})
    if outs[0].ndim == 3:
        outs = [o[0] for o in outs]
    fmc = 3
    scores_l, boxes_l, kps_l = [], [], []
    for idx, stride in enumerate((8, 16, 32)):
        scores = outs[idx]
        boxes = outs[idx + fmc] * stride
        kps = outs[idx + fmc * 2] * stride
        fh, fw = size // stride, size // stride
        centers = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
        centers = np.stack([(centers * stride).reshape(-1, 2)] * 2, axis=1).reshape(-1, 2)
        pos = np.where(scores >= thresh)[0]
        scores_l.append(scores[pos])
        boxes_l.append(_distance2bbox(centers, boxes)[pos])
        kps_l.append(_distance2kps(centers, kps).reshape(-1, 5, 2)[pos])
    scores = np.vstack(scores_l).ravel()
    if scores.size == 0:
        return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
    order = scores.argsort()[::-1]
    boxes = np.vstack(boxes_l) / scale
    kps = np.vstack(kps_l) / scale
    pre = np.hstack((boxes, scores[:, None])).astype(np.float32)[order]
    keep = _nms(pre)
    return pre[keep], kps[order][keep]


# ---------------------------------------------------------------- ArcFace embed

def _umeyama(src, dst):
    """Similarity transform src -> dst (same math as skimage SimilarityTransform)."""
    n, dim = src.shape
    sm, dm = src.mean(0), dst.mean(0)
    sd, dd = src - sm, dst - dm
    A = dd.T @ sd / n
    d = np.ones(dim)
    if np.linalg.det(A) < 0:
        d[-1] = -1
    U, S, V = np.linalg.svd(A)
    T = np.eye(dim + 1)
    T[:dim, :dim] = U @ np.diag(d) @ V
    scale = 1.0 / sd.var(0).sum() * (S @ d)
    T[:dim, dim] = dm - scale * (T[:dim, :dim] @ sm)
    T[:dim, :dim] *= scale
    return T[:2]


def _embed(rec, rgb, kps):
    M = _umeyama(kps.astype(np.float64), ARCFACE_DST.astype(np.float64))
    aligned = _warp_affine(rgb, M, 112)
    blob = _blob(aligned, 1.0 / 127.5, 127.5)
    e = rec.run(None, {rec.get_inputs()[0].name: blob})[0].flatten()
    return e / (np.linalg.norm(e) + 1e-12)


def _mean_pair_sim(E):
    sims = [float(a @ b) for a, b in itertools.combinations(E, 2)]
    return (float(np.mean(sims)) if sims else float("nan")), sims


def _grid(crops, tile=128, cols=10):
    if not crops:
        return torch.zeros((1, tile, tile, 3))
    cols = min(cols, len(crops))
    rows = (len(crops) + cols - 1) // cols
    g = np.full((rows * tile, cols * tile, 3), 16, dtype=np.uint8)
    for i, c in enumerate(crops):
        r, k = divmod(i, cols)
        g[r * tile:(r + 1) * tile, k * tile:(k + 1) * tile] = np.asarray(Image.fromarray(c).resize((tile, tile), Image.LANCZOS))
    return torch.from_numpy(g).float().unsqueeze(0) / 255.0


class LCFaceVarietyScorer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "A batch of generations to test. Same prompt, different seeds works best. 8 or more images."}),
                "same_threshold": ("FLOAT", {
                    "default": 0.35, "min": 0.1, "max": 0.9, "step": 0.01,
                    "tooltip": "Two faces scoring above this count as 'could be the same person' for same_pct.",
                }),
                "min_face_px": ("INT", {
                    "default": 40, "min": 8, "max": 1024,
                    "tooltip": "Ignore detected faces narrower than this (background people, reflections).",
                }),
                "device": (["cpu", "auto"], {"tooltip": "cpu works on every install and takes seconds for a normal test. auto tries CUDA first and falls back to CPU."}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "STRING", "STRING", "IMAGE", "INT")
    RETURN_NAMES = ("score", "same_pct", "crowd_score", "grade", "report", "faces", "faces_found")
    FUNCTION = "score"
    CATEGORY = "LC ModelBuilder/testing"
    DESCRIPTION = (
        "Same-face test. Finds the main face in every image, compares every face against every other "
        "with ArcFace, and averages it. score 0.0 - 1.0, lower = more variety. crowd_score compares the "
        "faces inside each image (group shots), -1 when no image has 2+ faces. Uses InsightFace buffalo_l (auto-downloads once, "
        "non-commercial research license)."
    )

    def score(self, images, same_threshold=0.35, min_face_px=40, device="cpu"):
        det, rec = _sessions(device)
        arr = (images.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        main, crops, crowd, missed = [], [], [], []
        for i, rgb in enumerate(arr):
            rgb = np.ascontiguousarray(rgb[:, :, :3])
            boxes, kps = _detect(det, rgb)
            wide = [j for j in range(len(boxes)) if boxes[j][2] - boxes[j][0] >= min_face_px]
            if not wide:
                missed.append(i + 1)
                continue
            embs = [_embed(rec, rgb, kps[j]) for j in wide]
            big = max(range(len(wide)), key=lambda j: (boxes[wide[j]][2] - boxes[wide[j]][0]) * (boxes[wide[j]][3] - boxes[wide[j]][1]))
            main.append(embs[big])
            x0, y0, x1, y1 = boxes[wide[big]][:4].astype(int)
            p = int(0.35 * (x1 - x0))
            crops.append(rgb[max(0, y0 - p):y1 + p, max(0, x0 - p):x1 + p])
            if len(embs) >= 2:
                crowd.append(_mean_pair_sim(embs)[0])

        score, sims = _mean_pair_sim(main)
        same_pct = 100.0 * float(np.mean(np.array(sims) > same_threshold)) if sims else float("nan")
        crowd_score = float(np.mean(crowd)) if crowd else -1.0
        g = grade(score) if sims else "Need at least 2 faces"

        lines = [
            "LC Face Variety Scorer",
            f"Images: {len(arr)}   Faces used: {len(main)}" + (f"   No face found in image(s): {missed}" if missed else ""),
            "",
            f"Same face score: {score:.3f}   (0.0 - 1.0, lower = more variety)",
            f"Grade: {g}",
            f"Pairs above {same_threshold:.2f} (could be the same person): {same_pct:.1f}%",
            ("Crowd score: n/a   (no image had 2+ faces)" if not crowd else f"Crowd score: {crowd_score:.3f}   ({len(crowd)} group image(s))"),
            "",
            "Scale:",
            "  0.00 - 0.20  Clearly different people",
            "  0.20 - 0.35  Different people who share a few features",
            "  0.35 - 0.50  Could be sisters, or the same person in a different photo",
            "  0.50 - 1.00  Same person",
            "",
            "Face models: InsightFace buffalo_l (SCRFD + ArcFace), github.com/deepinsight/insightface",
        ]
        return (score, same_pct, crowd_score, g, "\n".join(lines), _grid(crops), len(main))


NODE_CLASS_MAPPINGS = {"LCFaceVarietyScorer": LCFaceVarietyScorer}
NODE_DISPLAY_NAME_MAPPINGS = {"LCFaceVarietyScorer": "LC Face Variety Scorer 🧑‍🤝‍🧑"}
