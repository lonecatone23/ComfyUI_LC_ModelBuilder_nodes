"""
LC Modelbuilder Nodes -- install.py
-----------------------------------
Auto-run by ComfyUI-Manager after clone/update. Makes sure LC Face Variety
Scorer can run. Its only extra dependency is onnxruntime (no insightface, no
opencv, no onnx/protobuf, no compiler).

Rules, same spirit as the LC Vision installer:
- If a working onnxruntime (any flavor) is already there, change nothing.
- If none is installed, install the plain CPU `onnxruntime`. Never
  onnxruntime-gpu: its CUDA/cuDNN matching is the #1 cause of
  "LoadLibrary failed with error 126", and the CPU package runs on NVIDIA,
  AMD, Intel and Mac alike. Scoring a test batch on CPU takes seconds.
- Never uninstall or upgrade anything. If onnxruntime is installed but
  broken, print the exact repair commands and stop.
- Snapshot `pip freeze` to install_backups/ before any change.
- --no-deps, then add only dependencies that are missing outright.
"""

from __future__ import annotations

import base64
import importlib
import importlib.metadata as md
import re
import subprocess
import sys
import time
from pathlib import Path

# A 1-op ONNX model (Identity), so the self-test needs no model download and no `onnx` package.
_TEST_MODEL = base64.b64decode("CAg6NwoQCgF4EgF5IghJZGVudGl0eRIBdFoPCgF4EgoKCAgBEgQKAggBYg8KAXkSCgoICAESBAoCCAFCBAoAEA0=")
_FLAVORS = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino", "onnxruntime-silicon")


def _log(msg: str) -> None:
    print(f"[LC Modelbuilder install] {msg}")


def _installed_flavors() -> list[str]:
    found = []
    for name in _FLAVORS:
        try:
            found.append(f"{name}=={md.version(name)}")
        except md.PackageNotFoundError:
            pass
    return found


def _self_test() -> str | None:
    """None when onnxruntime imports and runs a model on CPU, else the error text."""
    for mod in [m for m in sys.modules if m == "onnxruntime" or m.startswith("onnxruntime.")]:
        del sys.modules[mod]
    importlib.invalidate_caches()
    try:
        import numpy as np
        import onnxruntime as ort

        s = ort.InferenceSession(_TEST_MODEL, providers=["CPUExecutionProvider"])
        out = s.run(None, {"x": np.array([1.5], dtype=np.float32)})[0]
        return None if float(out[0]) == 1.5 else "self-test returned a wrong value"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _pip(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "pip"] + args
    _log("running: " + " ".join(cmd))
    return subprocess.run(cmd, text=True, capture_output=True)


def _snapshot() -> None:
    r = _pip(["freeze"])
    if r.returncode == 0:
        d = Path(__file__).parent / "install_backups"
        d.mkdir(exist_ok=True)
        p = d / f"pip_freeze_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        p.write_text(r.stdout, encoding="utf-8")
        _log(f"Snapshotted the current environment to {p} before making any changes.")


def _missing_module(err: str) -> str | None:
    m = re.search(r"No module named '([\w\.]+)'", err)
    return m.group(1).split(".")[0] if m else None


def main() -> int:
    err = _self_test()
    if err is None:
        _log(f"onnxruntime OK ({', '.join(_installed_flavors()) or 'unknown build'}). Nothing to do.")
        return 0

    flavors = _installed_flavors()
    if flavors and not _missing_module(err):
        _log(f"onnxruntime is installed ({', '.join(flavors)}) but will not run: {err}")
        if len(flavors) > 1:
            _log("More than one onnxruntime package is installed. They share one folder and overwrite each other.")
        _log("Nothing was changed. To repair, run these with ComfyUI's own python and restart ComfyUI:")
        _log("    python -m pip uninstall -y " + " ".join(f.split("==")[0] for f in flavors))
        _log("    python -m pip install onnxruntime")
        _log("(Other node packs that want GPU onnxruntime can use onnxruntime-gpu instead of onnxruntime.)")
        return 1

    _snapshot()
    if not flavors:
        r = _pip(["install", "--no-deps", "onnxruntime"])
        if r.returncode != 0:
            _log("pip could not install onnxruntime:\n" + (r.stderr or r.stdout)[-1500:])
            _log("Check https://onnxruntime.ai for a build that matches your Python version.")
            return 1

    # Pull in only dependencies that are missing outright. Never upgrade what is already there.
    for _ in range(6):
        err = _self_test()
        if err is None:
            break
        mod = _missing_module(err)
        if not mod:
            break
        pkg = {"google": "protobuf", "flatbuffers": "flatbuffers", "sympy": "sympy", "coloredlogs": "coloredlogs",
               "packaging": "packaging", "mpmath": "mpmath", "humanfriendly": "humanfriendly"}.get(mod, mod)
        _log(f"onnxruntime needs '{mod}', installing {pkg}")
        _pip(["install", "--no-deps", pkg])

    err = _self_test()
    if err is None:
        _log(f"onnxruntime ready ({', '.join(_installed_flavors())}). LC Face Variety Scorer can run.")
        return 0
    _log(f"onnxruntime still will not run: {err}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
