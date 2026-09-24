"""
LC Krea2 Semantic Vector Tuner
-------------------------------
Single-model weight amplifier/attenuator, not a merge — one MODEL in, the
same MODEL out with specific regions scaled up or down. Ported from the
Arthemy SDXL Suite's ArthemySDXLModelTuner mechanism (same repo, same
ComfyUI instance): for each real weight tensor, bake any existing patches
via comfy.lora.calculate_weight, scale the baked tensor by a per-region
target multiplier, and add the difference back as a patch via
ModelPatcher.add_patches. No hand-rolled patching semantics — this is
Arthemy's own proven approach, re-pointed at Krea2's real prefixes instead
of SDXL's input_blocks/middle_block/output_blocks layout.

Region grouping reuses the exact same 28-block low/mid/high zones and the
10 named prefixes as LC Krea2 Block Merge Advanced (see
lc_krea2_block_merge.py) — shared infrastructure across the pack rather
than a second, competing map of the architecture. Krea2's 28 main blocks
are architecturally uniform (unlike SDXL's UNet, which genuinely
specializes by depth), so this node does not claim fine-grained semantic
labels within them the way Arthemy's SDXL tuner does — it offers 3 zone
sliders plus an optional per-block override string for users who want
finer control than that.
"""

from __future__ import annotations

import re

import comfy.lora

from .lc_krea2_block_merge import (
    _NAMED_PREFIXES,
    _NAMED_KEYS,
    _NAMED_TOOLTIPS,
    _NUM_BLOCKS,
    _ZONE_BOUNDS,
    parse_curve,
)

_ZONE_LOW_END, _ZONE_MID_END = _ZONE_BOUNDS


def _soft_curve(w: float) -> float:
    """Arthemy's 'Soft Value' response curve: gentle below 1.0, heavily
    compressed above it, so a 0-2 slider never wildly overdrives a weight."""
    if w <= 1.0:
        return max(0.0, -1.02 * (w ** 2) + 2.02 * w)
    return 1.0 + (w - 1.0) * 0.133


class LCKrea2SemanticVectorTuner:
    @classmethod
    def INPUT_TYPES(cls):
        slider = {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}
        required = {
            "model": (
                "MODEL",
                {"tooltip": "Single checkpoint to amplify/attenuate by region. This is not a merge — no second model."},
            ),
            "mode": (
                ("Soft Value", "Real Value"),
                {
                    "default": "Soft Value",
                    "tooltip": "Soft Value dampens the sliders so 0-2 stays safe (values above 1.0 land around "
                    "1.0-1.13x). Real Value applies each slider as a literal multiplier.",
                },
            ),
            "base_scale": (
                "FLOAT",
                {**slider, "tooltip": "Fallback multiplier for any weight that isn't one of the 28 blocks or the 10 named regions."},
            ),
            "low_blocks": ("FLOAT", {**slider, "tooltip": f"Scales blocks.0. - blocks.{_ZONE_LOW_END - 1}. (low zone)."}),
            "mid_blocks": ("FLOAT", {**slider, "tooltip": f"Scales blocks.{_ZONE_LOW_END}. - blocks.{_ZONE_MID_END - 1}. (mid zone)."}),
            "high_blocks": ("FLOAT", {**slider, "tooltip": f"Scales blocks.{_ZONE_MID_END}. - blocks.{_NUM_BLOCKS - 1}. (high zone)."}),
            "blocks_override": (
                "STRING",
                {
                    "default": "",
                    "multiline": False,
                    "tooltip": f"Optional comma list of exactly {_NUM_BLOCKS} multipliers, one per block. "
                    "Overrides low_blocks/mid_blocks/high_blocks for the 28 blocks when present. "
                    "Named regions below are unaffected — they always come from their own sliders.",
                },
            ),
        }
        for prefix, key in _NAMED_PREFIXES:
            required[key] = ("FLOAT", {**slider, "tooltip": _NAMED_TOOLTIPS[key]})
        return {"required": required}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "tune_model"
    CATEGORY = "LC ModelBuilder/Krea2"
    DESCRIPTION = (
        "Amplify or attenuate specific regions of a single Krea2 checkpoint's weights. Not a merge. "
        "Reuses Arthemy SDXL Suite's own tuning mechanism (bake existing patches, scale, re-patch the "
        "delta via ModelPatcher.add_patches) pointed at Krea2's real block/region prefixes instead of "
        "SDXL's UNet layout. 3 zone sliders cover the 28 uniform blocks (with an optional per-block "
        "override string for finer control); the 10 named, architecturally distinct regions each get "
        "their own slider."
    )

    def tune_model(
        self,
        model,
        mode="Soft Value",
        base_scale=1.0,
        low_blocks=1.0,
        mid_blocks=1.0,
        high_blocks=1.0,
        blocks_override="",
        **named,
    ):
        def target(w):
            return float(w) if mode == "Real Value" else _soft_curve(float(w))

        block_scale = (
            [low_blocks] * _ZONE_LOW_END
            + [mid_blocks] * (_ZONE_MID_END - _ZONE_LOW_END)
            + [high_blocks] * (_NUM_BLOCKS - _ZONE_MID_END)
        )
        override_vals = parse_curve(blocks_override)
        if len(override_vals) == _NUM_BLOCKS:
            block_scale = override_vals

        block_targets = [target(v) for v in block_scale]
        named_targets = {key: target(named.get(key, 1.0)) for key in _NAMED_KEYS}
        base_target = target(base_scale)

        m = model.clone()
        base_sd = m.model.state_dict()
        patches_to_add = {}
        active_patches = 0
        valid_keys = m.model_keys if hasattr(m, "model_keys") else None

        for k, base_weight in base_sd.items():
            # m.model.state_dict() keys already carry the "diffusion_model." prefix
            # (BaseModel.diffusion_model is a real submodule -- PyTorch's own
            # state_dict() naming, not something we add). Strip it only for
            # matching against the bare block/named prefixes; the patch key
            # itself must stay exactly as m.model.state_dict() produced it,
            # since that's what add_patches() looks up against internally.
            k_unet = k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k

            target_weight = base_target
            block_match = re.match(r"blocks\.(\d+)\.", k_unet)
            if block_match:
                idx = int(block_match.group(1))
                if 0 <= idx < _NUM_BLOCKS:
                    target_weight = block_targets[idx]
            else:
                for prefix, key in _NAMED_PREFIXES:
                    if k_unet.startswith(prefix):
                        target_weight = named_targets[key]
                        break

            strength = target_weight - 1.0
            if strength == 0:
                continue
            if valid_keys is not None and k not in valid_keys:
                continue

            patch_key = k

            current_patches = m.patches.get(patch_key, [])
            if current_patches:
                baked_weight = comfy.lora.calculate_weight(current_patches, base_weight.clone(), patch_key)
            else:
                baked_weight = base_weight.clone()

            baked_float = baked_weight.float()
            v_modified = baked_float + (baked_float * strength)
            v_target = v_modified.to(base_weight.dtype)
            delta = v_target.float() - baked_float

            patches_to_add[patch_key] = (delta.cpu(),)
            active_patches += 1

        if patches_to_add:
            m.add_patches(patches_to_add, 1.0)

        info = (
            f"Krea2 Tuned | mode={mode} | base={base_scale:.2f} "
            f"low={low_blocks:.2f} mid={mid_blocks:.2f} high={high_blocks:.2f} | patches={active_patches}"
        )
        return (m, info)


NODE_CLASS_MAPPINGS = {"LCKrea2SemanticVectorTuner": LCKrea2SemanticVectorTuner}
NODE_DISPLAY_NAME_MAPPINGS = {"LCKrea2SemanticVectorTuner": "LC Krea2 Semantic Vector Tuner"}
