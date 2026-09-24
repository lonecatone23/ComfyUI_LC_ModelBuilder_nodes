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
print(f"[LC Modelbuilder] loading from {_PACK_DIR}")
_nested = _os.path.join(_PACK_DIR, "ComfyUI_LC_ModelBuilder_nodes", "__init__.py")
if _os.path.isfile(_nested):
    print(
        "[LC Modelbuilder] WARNING: nested pack folder detected. "
        f"{_nested} will be ignored. Unzip so __init__.py sits in {_PACK_DIR}"
    )

NODE_CLASS_MAPPINGS = {}
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
        print(f"[LC Modelbuilder] + {module_name}: {list(maps.keys())}")
    except Exception as e:
        print(f"[LC Modelbuilder] ! failed to load {module_name}: {e}")
        traceback.print_exc()


_load("lc_krea2_block_merge")
_load("lc_krea2_semantic_tuner")
_load("lc_checkpoint_save")
_load("lc_diffusion_model_save")
_load("lc_face_variety")

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print(f"[LC Modelbuilder] total {len(NODE_CLASS_MAPPINGS)} nodes: {sorted(NODE_CLASS_MAPPINGS.keys())}")
