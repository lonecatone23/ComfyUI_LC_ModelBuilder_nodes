"""
LC Checkpoint Save
-------------------
All-in-one checkpoint saver: model + clip + vae, all required -- a
"checkpoint" means a real AIO file, not a model-only export. For a
diffusion-model-only save, see LC Diffusion Model Save
(lc_diffusion_model_save.py), which imports build_save_metadata() from
here rather than duplicating the metadata logic.

Reuses ComfyUI core's own comfy.sd.save_checkpoint() for the actual file
write, and copies core's CheckpointSave node's modelspec-metadata
detection verbatim (see comfy_extras/nodes_model_merging.py's own
save_checkpoint() helper) so this produces the same file format and
metadata core's stock "Save Checkpoint" node does.

Two per-node controls core's stock node doesn't offer:

- enabled: skip the save entirely (no file written) without physically
  bypassing the node. Convert to an input to gate it from a switch
  elsewhere in the graph -- useful since a save node has no outputs, so
  there's nothing for ComfyUI's native bypass to pass through.
- embed_workflow: per-save override for whether the prompt/workflow JSON
  goes into the file's metadata, independent of the global
  --disable-metadata launch flag core's own node is stuck reading.
"""

from __future__ import annotations

import json
import os

import torch

import comfy.sd
import comfy.model_base
import comfy.model_sampling
import folder_paths


def build_save_metadata(model, filename, counter, embed_workflow, prompt, extra_pnginfo):
    """Shared with LCDiffusionModelSave. Mirrors core's own save_checkpoint()
    helper in comfy_extras/nodes_model_merging.py -- modelspec detection plus
    the embed_workflow-gated prompt/workflow metadata. Returns (metadata, extra_keys)."""
    metadata = {}
    enable_modelspec = True
    if isinstance(model.model, comfy.model_base.SDXL):
        if isinstance(model.model, comfy.model_base.SDXL_instructpix2pix):
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-edit"
        else:
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-base"
    elif isinstance(model.model, comfy.model_base.SDXLRefiner):
        metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-refiner"
    elif isinstance(model.model, comfy.model_base.SVD_img2vid):
        metadata["modelspec.architecture"] = "stable-video-diffusion-img2vid-v1"
    elif isinstance(model.model, comfy.model_base.SD3):
        metadata["modelspec.architecture"] = "stable-diffusion-v3-medium"
    else:
        enable_modelspec = False

    if enable_modelspec:
        metadata["modelspec.sai_model_spec"] = "1.0.0"
        metadata["modelspec.implementation"] = "sgm"
        metadata["modelspec.title"] = "{} {}".format(filename, counter)

    extra_keys = {}
    model_sampling = model.get_model_object("model_sampling")
    if isinstance(model_sampling, comfy.model_sampling.ModelSamplingContinuousEDM):
        if isinstance(model_sampling, comfy.model_sampling.V_PREDICTION):
            extra_keys["edm_vpred.sigma_max"] = torch.tensor(model_sampling.sigma_max).float()
            extra_keys["edm_vpred.sigma_min"] = torch.tensor(model_sampling.sigma_min).float()

    if model.model.model_type == comfy.model_base.ModelType.EPS:
        metadata["modelspec.predict_key"] = "epsilon"
    elif model.model.model_type == comfy.model_base.ModelType.V_PREDICTION:
        metadata["modelspec.predict_key"] = "v"
        extra_keys["v_pred"] = torch.tensor([])
        if getattr(model_sampling, "zsnr", False):
            extra_keys["ztsnr"] = torch.tensor([])

    if embed_workflow:
        metadata["prompt"] = json.dumps(prompt) if prompt is not None else ""
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])

    return metadata, extra_keys


ENABLED_INPUT = (
    "BOOLEAN",
    {
        "default": True,
        "tooltip": "Off = skip the save entirely (no file written), without physically bypassing the "
        "node. Convert to an input to gate saving from a switch elsewhere in the graph.",
    },
)

EMBED_WORKFLOW_INPUT = (
    "BOOLEAN",
    {
        "default": True,
        "tooltip": "Include the prompt and workflow JSON in the saved file's metadata. Turn off before "
        "sharing/publishing a checkpoint if you don't want your exact node graph attached.",
    },
)


class LCCheckpointSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Diffusion model to save."}),
                "clip": ("CLIP", {"tooltip": "Required -- a checkpoint is an all-in-one file."}),
                "vae": ("VAE", {"tooltip": "Required -- a checkpoint is an all-in-one file."}),
                "filename_prefix": (
                    "STRING",
                    {"default": "checkpoints/LC_ModelBuilder", "tooltip": "Subfolder/prefix under the output directory."},
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
        "Save an all-in-one checkpoint (model + clip + vae, all required). Reuses ComfyUI core's own "
        "comfy.sd.save_checkpoint() and modelspec-metadata detection -- same file format core's Save "
        "Checkpoint node produces. Adds an enabled toggle (skip the save without bypassing the node, "
        "since a save node has no outputs to pass through) and a per-save embed_workflow toggle, "
        "independent of the global --disable-metadata launch flag. For a diffusion-model-only file, use "
        "LC Diffusion Model Save instead."
    )

    def save(self, model, clip, vae, filename_prefix, enabled=True, embed_workflow=True, prompt=None, extra_pnginfo=None):
        if not enabled:
            return {}

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir
        )
        metadata, extra_keys = build_save_metadata(model, filename, counter, embed_workflow, prompt, extra_pnginfo)

        output_path = f"{filename}_{counter:05}_.safetensors"
        output_path = os.path.join(full_output_folder, output_path)

        comfy.sd.save_checkpoint(output_path, model, clip, vae, metadata=metadata, extra_keys=extra_keys)
        return {}


NODE_CLASS_MAPPINGS = {"LCCheckpointSave": LCCheckpointSave}
NODE_DISPLAY_NAME_MAPPINGS = {"LCCheckpointSave": "LC Checkpoint Save"}
