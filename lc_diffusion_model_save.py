"""
LC Diffusion Model Save
------------------------
Diffusion-model-only saver: only a MODEL input, no clip/vae sockets at
all -- the counterpart to LC Checkpoint Save's all-in-one file. Same
enabled/embed_workflow controls, same metadata mechanism (imported from
lc_checkpoint_save.py rather than duplicated), just clip=None/vae=None
at the comfy.sd.save_checkpoint() call.

The written file still carries the "model.diffusion_model." prefix core's
save path always applies (comfy.supported_models_base.py's
process_unet_state_dict_for_saving) -- this is not a bare-key
diffusion_models/ export. That prefix is not a problem for loading it
back: comfy.sd.load_diffusion_model_state_dict() auto-detects and strips
known unet prefixes (model_detection.unet_prefix_from_state_dict), so the
file loads fine through the stock "Load Diffusion Model" node. It just
means the file will be a little larger on disk than a from-source
diffusion_models/ release (no clip/vae, but the same key-naming
convention as a full checkpoint's UNet half).
"""

from __future__ import annotations

import os

import comfy.sd
import folder_paths

from .lc_checkpoint_save import build_save_metadata, ENABLED_INPUT, EMBED_WORKFLOW_INPUT


class LCDiffusionModelSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Diffusion model to save. No clip/vae -- model only."}),
                "filename_prefix": (
                    "STRING",
                    {"default": "diffusion_models/LC_ModelBuilder", "tooltip": "Subfolder/prefix under the output directory."},
                ),
                "enabled": ENABLED_INPUT,
                "embed_workflow": EMBED_WORKFLOW_INPUT,
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "LC ModelBuilder"
    DESCRIPTION = (
        "Save just the diffusion model -- no clip, no vae. Same mechanism and metadata as LC Checkpoint "
        "Save (comfy.sd.save_checkpoint(), core's modelspec detection), just without the text encoder / "
        "VAE half of an all-in-one file. Loads back fine through the stock Load Diffusion Model node -- "
        "it auto-strips the model.diffusion_model. prefix this save path applies. Same enabled and "
        "embed_workflow toggles as LC Checkpoint Save."
    )

    def save(self, model, filename_prefix, enabled=True, embed_workflow=True, prompt=None, extra_pnginfo=None):
        if not enabled:
            return {}

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir
        )
        metadata, extra_keys = build_save_metadata(model, filename, counter, embed_workflow, prompt, extra_pnginfo)

        output_path = f"{filename}_{counter:05}_.safetensors"
        output_path = os.path.join(full_output_folder, output_path)

        comfy.sd.save_checkpoint(output_path, model, None, None, metadata=metadata, extra_keys=extra_keys)
        return {}


NODE_CLASS_MAPPINGS = {"LCDiffusionModelSave": LCDiffusionModelSave}
NODE_DISPLAY_NAME_MAPPINGS = {"LCDiffusionModelSave": "LC Diffusion Model Save"}
