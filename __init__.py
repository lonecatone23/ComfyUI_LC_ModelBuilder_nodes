"""
ComfyUI_LC_ModelBuilder_nodes — LC Modelbuilder Nodes by lonecatone23

Model/checkpoint merging tools. Separate repo, separate Registry name.
Shares no code with ComfyUI_LC123_nodes or ComfyUI_LC_AV_nodes on purpose —
this pack is for model-building (merge / blend / block-inject), not image
or audio/video pipelines.

https://github.com/lonecatone23
https://ko-fi.com/lonecatone
"""

import os as _os

_PACK_DIR = _os.path.dirname(_os.path.abspath(__file__))
_nested = _os.path.join(_PACK_DIR, "ComfyUI_LC_ModelBuilder_nodes", "__init__.py")
if _os.path.isfile(_nested):
    print(
        "[LC Modelbuilder] WARNING: nested pack folder detected. "
        f"{_nested} will be ignored. Unzip so __init__.py sits in {_PACK_DIR}"
    )

NODE_CLASS_MAPPINGS = {}
_FAILED = []  # modules that did not load (shown in full, and counted on the banner line)
NODE_DISPLAY_NAME_MAPPINGS = {}


def _load(module_name: str) -> None:
    """Import a submodule and merge its mappings. Log and skip on failure."""
    import importlib
    import traceback

    try:
        mod = importlib.import_module(f".{module_name}", __name__)
        maps = getattr(mod, "NODE_CLASS_MAPPINGS", None) or {}
        disp = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", None) or {}
        NODE_CLASS_MAPPINGS.update(maps)
        NODE_DISPLAY_NAME_MAPPINGS.update(disp)
    except Exception as e:
        _FAILED.append(module_name)
        print(f"[LC Modelbuilder] ! failed to load {module_name}: {e}")
        traceback.print_exc()


_load("lc_krea2_block_merge")
_load("lc_krea2_semantic_tuner")
_load("lc_checkpoint_save")
_load("lc_diffusion_model_save")
_load("lc_face_variety")
_load("lc_krea2_lora_editor")
_load("lc_krea2_lora_scan")
_load("lc_krea2_lora_merge")

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

try:  # the Lonecat banner (first LC pack to load) or this pack's one line under it
    from .lc_banner import pack_version as _pv, show as _banner

    _banner("LC ModelBuilder", _pv(_PACK_DIR), len(NODE_CLASS_MAPPINGS), len(_FAILED))
except Exception as _banner_e:
    print(f"[LC Modelbuilder] {len(NODE_CLASS_MAPPINGS)} nodes ({_banner_e})")
