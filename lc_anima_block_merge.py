"""
LC Anima Block Merge Advanced
-----------------------------
The Krea2 Block Merge Advanced for Anima: a draggable ratio curve over Anima's 28 blocks plus sliders for the
named parts outside them (x embedder, timestep embedder and norm, LLM adapter, final layer). Same mechanism:
ComfyUI's own ModelMergeBlocks (longest-prefix-match + add_patches), no custom tensor math. Core's
ModelMergeCosmosPredict2_2B has no LLM adapter slider; this one does.

Presets save under web/anima_block_presets/ as {name, blocks, named}.json.
"""

from __future__ import annotations

import os

from .lc_krea2_block_merge import (LCKrea2BlockMergeAdvanced, _NUM_BLOCKS, _PACK, _preset_choices, _save_preset,
                                   lerp_curve, parse_curve)

_PRESET_DIR = os.path.join(_PACK, "web", "anima_block_presets")

_NAMED_PREFIXES = [
    ("x_embedder.", "x_embedder"),
    ("t_embedder.", "t_embedder"),
    ("t_embedding_norm.", "t_embedding_norm"),
    ("llm_adapter.", "llm_adapter"),
    ("final_layer.", "final_layer"),
]
_NAMED_KEYS = [key for _, key in _NAMED_PREFIXES]
_NAMED_TOOLTIPS = {
    "x_embedder": "x_embedder. — turns the image latent into tokens, before block 0.",
    "t_embedder": "t_embedder. — timestep embedding.",
    "t_embedding_norm": "t_embedding_norm. — norm on the timestep embedding.",
    "llm_adapter": "llm_adapter. — maps the Qwen text encoder's output into the model (how the prompt is read).",
    "final_layer": "final_layer. — output layer, after block 27.",
}
_BUILTIN_PRESET_NAMES = ["50_50", "model2_early", "model2_look", "model2_late"]


class LCAnimaBlockMergeAdvanced(LCKrea2BlockMergeAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        d = super().INPUT_TYPES()
        old = d["required"]
        req = {}
        for k in ("model1", "model2"):
            req[k] = old[k]
        req["preset"] = (_preset_choices(_PRESET_DIR, _BUILTIN_PRESET_NAMES),
                         {"default": "50_50",
                          "tooltip": "Bundled: 50_50, model2_early (blocks 0-6 from model2), model2_look (blocks 7-20, "
                                     "where an Anima finetune's look sits), model2_late (21-27), plus anything you save. "
                                     "Loading one sets the curve and the named sliders. Custom keeps what you dragged."})
        for k in ("edit_mode", "smooth_radius", "blocks_curve"):
            req[k] = old[k]
        for prefix, key in _NAMED_PREFIXES:
            req[key] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": _NAMED_TOOLTIPS[key]})
        req["save_name"] = ("STRING", {"default": "", "tooltip": "Filename under web/anima_block_presets/ (no extension)."})
        req["save_curve"] = ("BOOLEAN", {"default": False, "tooltip": "On run, write save_name.json into web/anima_block_presets/."})
        return {"required": req}

    CATEGORY = "LC ModelBuilder/Anima"
    DESCRIPTION = (
        "Anima merge with a draggable per-block ratio curve across the 28 blocks, plus sliders for the named parts "
        "outside them (x embedder, timestep embedder and norm, LLM adapter, final layer). Reuses ComfyUI's own "
        "ModelMergeBlocks. Save writes web/anima_block_presets/<name>.json."
    )

    def merge_advanced(self, model1, model2, preset="Custom", edit_mode="smooth", smooth_radius=2, blocks_curve="",
                       save_name="", save_curve=False, **named):
        blocks = parse_curve(blocks_curve)
        if len(blocks) != _NUM_BLOCKS:
            blocks = lerp_curve(blocks, _NUM_BLOCKS) if len(blocks) >= 2 else [0.5] * _NUM_BLOCKS
        named_vals = {key: float(named.get(key, 0.5)) for key in _NAMED_KEYS}
        if save_curve and str(save_name).strip():
            try:
                _save_preset(save_name, blocks, named_vals, _PRESET_DIR)
            except OSError as e:
                print(f"[LC Modelbuilder] block merge preset save failed: {e}")
        # core's ratio is model1's strength; the UI is 0 = model1, 1 = model2 (see the Krea2 node)
        kwargs = {prefix: 1.0 - named_vals[key] for prefix, key in _NAMED_PREFIXES}
        for i, v in enumerate(blocks):
            kwargs[f"blocks.{i}."] = 1.0 - float(v)
        return super(LCKrea2BlockMergeAdvanced, self).merge(model1, model2, **kwargs)


NODE_CLASS_MAPPINGS = {"LCAnimaBlockMergeAdvanced": LCAnimaBlockMergeAdvanced}
NODE_DISPLAY_NAME_MAPPINGS = {"LCAnimaBlockMergeAdvanced": "LC Anima Block Merge Advanced"}
