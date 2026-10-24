"""
Anime same-character score with CCIP (deepghs, OpenRAIL): a feature model that turns an image into a character
vector and a tiny metric model that gives the difference between two of them. Below the threshold (about 0.18)
CCIP calls two images the same character. Face models trained on photos (ArcFace) miss on anime; CCIP is made for it.

Faces are found first with deepghs' anime face detector (YOLOv8, MIT) and CCIP compares the face crops: on whole
images a small character in a busy scene reads as "the same" far too often.

Files (about 150 MB + 45 MB) download once to models/ccip/, checksum-verified. Copies already in the Hugging Face
cache (dghs-imgutils) are used as they are.
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import os
import urllib.request

import numpy as np

import folder_paths

MODEL = "ccip-caformer-24-randaug-pruned"
REPO = "https://huggingface.co/deepghs/ccip_onnx/resolve/main"
FILES = {"model_feat.onnx": "4ea118d16496274f4f6e08d3afc768cc592389e8f7f32f8732ce2215c228ac5f",
         "model_metrics.onnx": "7e4646fdfbe369ad485b227ec47ac4092c98b8deef3af44e19fdbd8ba7bc25c1"}
THRESHOLD = 0.17847511429108218  # from the model's metrics.json
DET_URL = "https://huggingface.co/deepghs/anime_face_detection/resolve/main/face_detect_v1.4_s/model.onnx"
DET_SHA = "403b5bc93b6ff789b7d183418df4a1364049bac00c24acd927604a7ff6891483"
_DET_PATH = os.path.join(folder_paths.models_dir, "ccip", "face_detect_v1.4_s.onnx")
_DIR = os.path.join(folder_paths.models_dir, "ccip", MODEL)
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)[:, None, None]
_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)[:, None, None]
_SESS = {}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find():
    hf = os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub", "models--deepghs--ccip_onnx", "snapshots", "*", MODEL))
    for d in [_DIR, *glob.glob(hf)]:
        if all(os.path.isfile(os.path.join(d, f)) for f in FILES):
            return d
    return None


def _fetch(url, tmp):
    """Download to a .part file; a network error removes the partial file instead of leaving it behind."""
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _download():
    os.makedirs(_DIR, exist_ok=True)
    for name, sha in FILES.items():
        dst = os.path.join(_DIR, name)
        if os.path.isfile(dst) and _sha256(dst) == sha:
            continue
        tmp = dst + ".part"
        print(f"[LC Modelbuilder] downloading CCIP {name} ...")
        _fetch(f"{REPO}/{MODEL}/{name}", tmp)
        if _sha256(tmp) != sha:
            os.remove(tmp)
            raise RuntimeError(f"CCIP {name}: checksum mismatch after download. Try again, or place the file in {_DIR}.")
        os.replace(tmp, dst)
    return _DIR


def _sessions():
    if "feat" not in _SESS:
        import onnxruntime as ort
        d = _find() or _download()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        try:  # CUDA when onnxruntime-gpu works, otherwise CPU (about 2 s per image)
            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
            feat = ort.InferenceSession(os.path.join(d, "model_feat.onnx"), opts, providers=prov)
            feat.run(["output"], {"input": np.zeros((1, 3, 384, 384), np.float32)})
        except Exception:
            feat = ort.InferenceSession(os.path.join(d, "model_feat.onnx"), opts, providers=["CPUExecutionProvider"])
        _SESS["feat"] = feat
        _SESS["metric"] = ort.InferenceSession(os.path.join(d, "model_metrics.onnx"), opts, providers=["CPUExecutionProvider"])
    return _SESS["feat"], _SESS["metric"]


def _detector():
    if "det" not in _SESS:
        import onnxruntime as ort
        hf = glob.glob(os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub", "models--deepghs--anime_face_detection",
                                                       "snapshots", "*", "face_detect_v1.4_s", "model.onnx")))
        path = _DET_PATH if os.path.isfile(_DET_PATH) else (hf[0] if hf else None)
        if path is None:
            os.makedirs(os.path.dirname(_DET_PATH), exist_ok=True)
            tmp = _DET_PATH + ".part"
            print("[LC Modelbuilder] downloading the anime face detector ...")
            _fetch(DET_URL, tmp)
            if _sha256(tmp) != DET_SHA:
                os.remove(tmp)
                raise RuntimeError("anime face detector: checksum mismatch after download. Try again.")
            os.replace(tmp, _DET_PATH)
            path = _DET_PATH
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        _SESS["det"] = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    return _SESS["det"]


def _nms(boxes, scores, iou=0.7):
    order, keep = scores.argsort()[::-1], []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        order = order[1:][inter / (area[i] + area[order[1:]] - inter + 1e-9) <= iou]
    return keep


def main_face(rgb: np.ndarray, conf=0.25):
    """Largest anime face in an RGB uint8 image as (x0, y0, x1, y1), or None."""
    from PIL import Image
    h, w = rgb.shape[:2]
    x = np.asarray(Image.fromarray(rgb).resize((640, 640), Image.BILINEAR), np.float32).transpose(2, 0, 1)[None] / 255.0
    out = _detector().run(["output0"], {"images": x})[0][0]  # [5, anchors]: cx, cy, w, h, score
    out = out[:, out[4] > conf].T
    if not len(out):
        return None
    b = np.stack([out[:, 0] - out[:, 2] / 2, out[:, 1] - out[:, 3] / 2,
                  out[:, 0] + out[:, 2] / 2, out[:, 1] + out[:, 3] / 2], 1)
    b = b[_nms(b, out[:, 4])] * np.array([w / 640, h / 640, w / 640, h / 640])
    x0, y0, x1, y1 = max(b, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    return int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))


def face_crops(images):
    """Padded crops of the main face per image (None where no face is found)."""
    arr = images.detach().cpu().numpy() if hasattr(images, "detach") else np.asarray(images)
    out = []
    for a in arr:
        rgb = (np.clip(a[..., :3], 0, 1) * 255).round().astype(np.uint8)
        box = main_face(rgb)
        if box is None:
            out.append(None)
            continue
        x0, y0, x1, y1 = box
        pw, ph = (x1 - x0) // 2, (y1 - y0) // 2
        out.append(np.ascontiguousarray(rgb[max(0, y0 - ph):y1 + ph, max(0, x0 - pw):x1 + pw]))
    return out


def features(images) -> np.ndarray:
    """images: [B,H,W,3] float 0..1 (torch or numpy) -> [B,768] CCIP features."""
    from PIL import Image
    feat, _ = _sessions()
    arr = images if isinstance(images, list) else (images.detach().cpu().numpy() if hasattr(images, "detach") else np.asarray(images))
    out = []
    for a in arr:
        a = a if a.dtype == np.uint8 else (np.clip(a[..., :3], 0, 1) * 255).round().astype(np.uint8)
        im = Image.fromarray(np.ascontiguousarray(a[..., :3])).resize((384, 384), Image.BILINEAR)
        x = (np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0 - _MEAN) / _STD
        out.append(feat.run(["output"], {"input": x[None].astype(np.float32)})[0][0])
    return np.stack(out)


def same_character(images, faces=True):
    """(share of image pairs CCIP calls the same character 0..1, mean difference, images used) over a batch. Higher
    share = more alike: different characters sit near 0, a locked face near 1. faces=True compares face crops (images
    without a face are left out); with fewer than 2 faces it falls back to whole images."""
    used = images
    if faces:
        crops = [c for c in face_crops(images) if c is not None]
        if len(crops) >= 2:
            used = crops
    f = features(used)
    if len(f) < 2:
        return float("nan"), float("nan"), len(f)
    _, metric = _sessions()
    d = metric.run(["output"], {"input": f.astype(np.float32)})[0]
    pairs = [float(d[i, j]) for i, j in itertools.combinations(range(len(f)), 2)]
    return float(np.mean([p < THRESHOLD for p in pairs])), float(np.mean(pairs)), len(f)
