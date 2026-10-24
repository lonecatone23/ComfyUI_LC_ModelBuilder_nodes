"""
LC MOSS Block Merge Advanced
----------------------------
The Krea2 Block Merge Advanced for MOSS Image: a draggable ratio curve over MOSS's 42 layers (one stream: text and
picture go through the same layers) plus sliders for the named parts outside them (picture in, prompt in, timestep,
shared modulation, final layer). Same mechanism: ComfyUI's own ModelMergeBlocks (longest-prefix-match + add_patches),
no custom tensor math. Core has no MOSS merge node.

Presets save under web/moss_block_presets/ as {name, blocks, named}.json.
"""

from __future__ import annotations

import os

from .lc_krea2_block_merge import (LCKrea2BlockMergeAdvanced, _PACK, _preset_choices, _save_preset,
                                   lerp_curve, parse_curve)

_NUM_LAYERS = 42
_PRESET_DIR = os.path.join(_PACK, "web", "moss_block_presets")

_NAMED_PREFIXES = [
    ("video_embedder.", "image_in"),
    ("text_embedder.", "text_in"),
    ("time_embedding.", "timestep"),
    ("shared_block_modulation.", "modulation"),
    ("video_head_modulation.", "head_modulation"),
    ("final_norm_video.", "final_norm"),
    ("final_linear_video.", "final_layer"),
]
_NAMED_KEYS = [key for _, key in _NAMED_PREFIXES]
_NAMED_TOOLTIPS = {
    "image_in": "video_embedder. Turns the picture latent into tokens, before layer 0.",
    "text_in": "text_embedder. Maps the Qwen3.5 text encoder's output into the model (how the prompt is read).",
    "timestep": "time_embedding. The timestep (how noisy the picture is).",
    "modulation": "shared_block_modulation. One modulation table every layer shares: how strongly each layer acts per step.",
    "head_modulation": "video_head_modulation. Modulation of the output head.",
    "final_norm": "final_norm_video. Norm before the output layer.",
    "final_layer": "final_linear_video. Output layer, after layer 41.",
}
_BUILTIN_PRESET_NAMES = ["50_50", "model2_early", "model2_look", "model2_late"]


class LCMOSSBlockMergeAdvanced(LCKrea2BlockMergeAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        d = super().INPUT_TYPES()
        old = d["required"]
        req = {}
        for k in ("model1", "model2"):
            req[k] = old[k]
        req["preset"] = (_preset_choices(_PRESET_DIR, _BUILTIN_PRESET_NAMES),
                         {"default": "50_50",
                          "tooltip": "Bundled: 50_50, model2_early (layers 0-13 from model2), model2_look (14-27), model2_late "
                                     "(28-41: SFT + Base test, these carried most of the pose and face), plus anything you save. "
                                     "Loading one sets the curve and the named sliders. Custom keeps what you dragged."})
        for k in ("edit_mode", "smooth_radius"):
            req[k] = old[k]
        req["blocks_curve"] = ("STRING", {"default": ", ".join(["0.5"] * _NUM_LAYERS), "multiline": False,
                                          "tooltip": f"Comma list of {_NUM_LAYERS} ratios (layers.0. - layers.{_NUM_LAYERS - 1}.). "
                                                     "0 = model1, 1 = model2. Drag the graph to set it."})
        for prefix, key in _NAMED_PREFIXES:
            req[key] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": _NAMED_TOOLTIPS[key]})
        req["save_name"] = ("STRING", {"default": "", "tooltip": "Filename under web/moss_block_presets/ (no extension)."})
        req["save_curve"] = ("BOOLEAN", {"default": False, "tooltip": "On run, write save_name.json into web/moss_block_presets/."})
        return {"required": req}

    CATEGORY = "LC ModelBuilder/MOSS Image"
    DESCRIPTION = (
        "MOSS Image merge with a draggable per-layer ratio curve across the 42 layers, plus sliders for the named parts "
        "outside them (picture in, prompt in, timestep, modulation, final layer). Reuses ComfyUI's own ModelMergeBlocks. "
        "Save writes web/moss_block_presets/<name>.json."
    )

    def merge_advanced(self, model1, model2, preset="Custom", edit_mode="smooth", smooth_radius=2, blocks_curve="",
                       save_name="", save_curve=False, **named):
        blocks = parse_curve(blocks_curve)
        if len(blocks) != _NUM_LAYERS:
            blocks = lerp_curve(blocks, _NUM_LAYERS) if len(blocks) >= 2 else [0.5] * _NUM_LAYERS
        named_vals = {key: float(named.get(key, 0.5)) for key in _NAMED_KEYS}
        if save_curve and str(save_name).strip():
            try:
                _save_preset(save_name, blocks, named_vals, _PRESET_DIR)
            except OSError as e:
                print(f"[LC Modelbuilder] block merge preset save failed: {e}")
        # core's ratio is model1's strength; the UI is 0 = model1, 1 = model2 (see the Krea2 node)
        kwargs = {prefix: 1.0 - named_vals[key] for prefix, key in _NAMED_PREFIXES}
        for i, v in enumerate(blocks):
            kwargs[f"layers.{i}."] = 1.0 - float(v)
        return super(LCKrea2BlockMergeAdvanced, self).merge(model1, model2, **kwargs)


NODE_CLASS_MAPPINGS = {"LCMOSSBlockMergeAdvanced": LCMOSSBlockMergeAdvanced}
NODE_DISPLAY_NAME_MAPPINGS = {"LCMOSSBlockMergeAdvanced": "LC MOSS Block Merge Advanced"}
