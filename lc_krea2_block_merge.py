"""
LC Krea2 Block Merge Advanced
------------------------------
Per-block ratio curve (the 28 uniform MMDiT blocks) plus individual sliders
for the 10 named, architecturally-distinct regions. No hand-rolled tensor
math here: merge() builds the same prefix->ratio kwargs ComfyUI's own
ModelMergeKrea2 uses and hands them straight to ModelMergeBlocks.merge(),
so this node rides the exact core longest-prefix-match / add_patches path.

Curve is a comma list, source of truth when preset is Custom. Presets save
under web/krea2_block_presets/ as {name, blocks, named}.json.
"""

from __future__ import annotations

import json
import os
import re

from comfy_extras.nodes_model_merging import ModelMergeBlocks

_PACK = os.path.dirname(os.path.abspath(__file__))
_PRESET_DIR = os.path.join(_PACK, "web", "krea2_block_presets")

_NUM_BLOCKS = 28

# (real Krea2 state-dict prefix, python-safe kwarg name for the node's own signature)
_NAMED_PREFIXES = [
    ("first.", "first"),
    ("tmlp.", "tmlp"),
    ("txtmlp.", "txtmlp"),
    ("tproj.", "tproj"),
    ("txtfusion.layerwise_blocks.0.", "txtfusion_layer0"),
    ("txtfusion.layerwise_blocks.1.", "txtfusion_layer1"),
    ("txtfusion.projector.", "txtfusion_projector"),
    ("txtfusion.refiner_blocks.0.", "txtfusion_refiner0"),
    ("txtfusion.refiner_blocks.1.", "txtfusion_refiner1"),
    ("last.", "last"),
]
_NAMED_KEYS = [key for _, key in _NAMED_PREFIXES]

# Named-region tooltips. These prefixes come straight from ComfyUI core's own
# ModelMergeKrea2 (comfy_extras/nodes_model_merging_model_specific.py) — the
# descriptions below go by what the prefix names imply, not verified internals.
_NAMED_TOOLTIPS = {
    "first": "first. — earliest input/embedding weights, before block 0.",
    "tmlp": "tmlp. — timestep-embedding MLP (timestep conditioning).",
    "txtmlp": "txtmlp. — text-embedding MLP (text conditioning).",
    "tproj": "tproj. — timestep projection layer.",
    "txtfusion_layer0": "txtfusion.layerwise_blocks.0. — first text/image fusion block.",
    "txtfusion_layer1": "txtfusion.layerwise_blocks.1. — second text/image fusion block.",
    "txtfusion_projector": "txtfusion.projector. — projects fused text features into the main stream.",
    "txtfusion_refiner0": "txtfusion.refiner_blocks.0. — first fusion refiner block.",
    "txtfusion_refiner1": "txtfusion.refiner_blocks.1. — second fusion refiner block.",
    "last": "last. — final output layer, after block 27.",
}

# Low/mid/high zone split for the 28 uniform blocks, shared with the JS graph.
_ZONE_BOUNDS = (_NUM_BLOCKS // 3, (_NUM_BLOCKS * 2) // 3)  # (9, 18) -> low 0-8, mid 9-17, high 18-27

# Bundled presets shipped under web/krea2_block_presets/. Kept first (in this
# order) in the preset combo, ahead of anything the user saves themselves.
_BUILTIN_PRESET_NAMES = [
    "50_50",
    "face_favor",
    "detail_focus",
    "style_favor",
    "sine_wave",
    "resonance_blocks",
    "even_blend",
    "core_focus",
    "heavy_transplant_anchored",
]


def _ensure_dir():
    os.makedirs(_PRESET_DIR, exist_ok=True)


def _sanitize(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^\w.\-]+", "_", name)
    return name[:80] or "custom"


def _list_saved() -> list[str]:
    _ensure_dir()
    names = set()
    try:
        for fn in os.listdir(_PRESET_DIR):
            if fn.endswith(".json") and fn != "index.json":
                names.add(fn[:-5])
    except OSError:
        pass
    return sorted(names)


def _preset_choices() -> list[str]:
    saved = _list_saved()
    builtins = [n for n in _BUILTIN_PRESET_NAMES if n in saved]
    custom = sorted(n for n in saved if n not in _BUILTIN_PRESET_NAMES)
    out = list(builtins)
    if custom:
        if out:
            out.append("────────")
        out += custom
    if out:
        out.append("────────")
    out.append("Custom")
    return out


def parse_curve(text) -> list[float]:
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        out = []
        for x in text:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                pass
        return out
    s = str(text).replace("[", " ").replace("]", " ").replace("\n", ",")
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


def format_curve(vals: list[float]) -> str:
    return ", ".join(f"{v:.6g}" for v in vals)


def lerp_curve(vals: list[float], new_len: int) -> list[float]:
    if new_len < 1:
        return []
    if len(vals) < 2:
        return [vals[0] if vals else 0.5] * new_len
    out = []
    for i in range(new_len):
        x = i / max(new_len - 1, 1)
        pos = (len(vals) - 1) * x
        idx = int(pos)
        frac = pos - idx
        if idx >= len(vals) - 1:
            out.append(vals[-1])
        else:
            out.append((1.0 - frac) * vals[idx] + frac * vals[idx + 1])
    return out


def _write_index():
    names = _list_saved()
    try:
        with open(os.path.join(_PRESET_DIR, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"presets": names}, f)
    except OSError:
        pass


def _load_saved(name: str) -> dict | None:
    want = (_sanitize(name) + ".json").lower()
    try:
        for fn in os.listdir(_PRESET_DIR):
            if fn.lower() == want:
                with open(os.path.join(_PRESET_DIR, fn), "r", encoding="utf-8") as f:
                    return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _save_preset(name: str, blocks: list[float], named: dict) -> None:
    _ensure_dir()
    safe = _sanitize(name)
    payload = {
        "name": safe,
        "blocks": format_curve(blocks),
        "named": {k: float(v) for k, v in named.items()},
    }
    with open(os.path.join(_PRESET_DIR, safe + ".json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _write_index()


class LCKrea2BlockMergeAdvanced(ModelMergeBlocks):
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model1": ("MODEL", {"tooltip": "Ratio 0 side. Blocks/regions at 0.0 come entirely from this checkpoint."}),
            "model2": ("MODEL", {"tooltip": "Ratio 1 side. Blocks/regions at 1.0 come entirely from this checkpoint."}),
            "preset": (
                _preset_choices(),
                {
                    "default": "50_50",
                    "tooltip": "9 bundled presets (50_50, face_favor, detail_focus, style_favor, sine_wave, "
                    "resonance_blocks, even_blend, core_focus, heavy_transplant_anchored) plus anything "
                    "you save yourself. Loading one sets both the 28-block curve and the 10 named sliders. "
                    "Custom keeps whatever is currently dragged/typed.",
                },
            ),
            "edit_mode": (
                ("smooth", "spike"),
                {
                    "default": "smooth",
                    "tooltip": "smooth = drag one block and blend neighbors. spike = move one block only.",
                },
            ),
            "smooth_radius": (
                "INT",
                {
                    "default": 2,
                    "min": 0,
                    "max": 8,
                    "tooltip": "Neighbor window (in blocks) for smooth mode (0 = spike-like).",
                },
            ),
            "blocks_curve": (
                "STRING",
                {
                    "default": ", ".join(["0.5"] * _NUM_BLOCKS),
                    "multiline": False,
                    "tooltip": f"Comma list of {_NUM_BLOCKS} ratios (blocks.0. - blocks.{_NUM_BLOCKS - 1}.). "
                    "0 = all model1, 1 = all model2. Source of truth when preset is Custom.",
                },
            ),
        }
        for prefix, key in _NAMED_PREFIXES:
            required[key] = (
                "FLOAT",
                {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": _NAMED_TOOLTIPS[key]},
            )
        required["save_name"] = (
            "STRING",
            {"default": "", "tooltip": "Filename under web/krea2_block_presets/ (no extension)."},
        )
        required["save_curve"] = (
            "BOOLEAN",
            {"default": False, "tooltip": "On run, write save_name.json into web/krea2_block_presets/."},
        )
        return {"required": required}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "merge_advanced"
    CATEGORY = "LC ModelBuilder/Krea2"
    DESCRIPTION = (
        "Krea2 checkpoint merge with a draggable per-block ratio curve across the 28 uniform "
        "MMDiT blocks, plus individual sliders for the 10 named regions (first/tmlp/txtmlp/tproj, "
        "txtfusion.* x5, last). Reuses ComfyUI's own ModelMergeBlocks patch mechanism "
        "(get_key_patches + add_patches, longest-prefix-match) — no custom tensor math. "
        "Save writes web/krea2_block_presets/<name>.json (blocks + named sliders together)."
    )

    def merge_advanced(
        self,
        model1,
        model2,
        preset="Custom",
        edit_mode="smooth",
        smooth_radius=2,
        blocks_curve="",
        save_name="",
        save_curve=False,
        **named,
    ):
        _ = preset, edit_mode, smooth_radius  # client-side drag/preset UX only

        blocks = parse_curve(blocks_curve)
        if len(blocks) != _NUM_BLOCKS:
            blocks = lerp_curve(blocks, _NUM_BLOCKS) if len(blocks) >= 2 else [0.5] * _NUM_BLOCKS

        named_vals = {key: float(named.get(key, 0.5)) for key in _NAMED_KEYS}

        if save_curve and str(save_name).strip():
            try:
                _save_preset(save_name, blocks, named_vals)
            except OSError as e:
                print(f"[LC Modelbuilder] block merge preset save failed: {e}")

        # ModelMergeBlocks.merge()'s own "ratio" is the strength of model1, not model2:
        # add_patches({k: kp[k]}, 1.0 - ratio, ratio) resolves to
        # ratio*model1 + (1-ratio)*model2 (verified empirically against real
        # comfy.lora.calculate_weight with distinguishable dummy tensors -- ratio=0
        # returns model2's value, ratio=1 returns model1's). Our UI documents the
        # opposite convention everywhere (0 = model1, 1 = model2, matching every
        # preset, tooltip, and the graph's own axis labels), so invert here at the
        # one point we hand values to core rather than flip the UI/docs/presets.
        kwargs = {prefix: 1.0 - named_vals[key] for prefix, key in _NAMED_PREFIXES}
        for i, v in enumerate(blocks):
            kwargs[f"blocks.{i}."] = 1.0 - float(v)

        return super().merge(model1, model2, **kwargs)


NODE_CLASS_MAPPINGS = {"LCKrea2BlockMergeAdvanced": LCKrea2BlockMergeAdvanced}
NODE_DISPLAY_NAME_MAPPINGS = {"LCKrea2BlockMergeAdvanced": "LC Krea2 Block Merge Advanced"}
