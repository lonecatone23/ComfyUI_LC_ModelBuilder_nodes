"""
Block maps for the LoRA tools (Editor, Merge, Block Scan): where each model layer sits (block, heatmap row),
and which rows the attention / mlp sliders scale. Krea 2 and Anima both have 28 numbered blocks; everything
else (Krea 2's text fusion, Anima's LLM adapter and final layer) counts as outside the blocks.
"""

from __future__ import annotations

import re

NUM_BLOCKS = 28


class Arch:
    def __init__(self, name, title, block_re, rows, kinds, outside_name):
        self.name = name                  # stored in the LoRA object ("krea2", "anima")
        self.title = title                # for notes: "is it a Krea 2 LoRA?"
        self.block_re = re.compile(block_re)
        self.rows = rows                  # [(label, module path)] in heatmap order
        self.kinds = kinds                # per row: "attn", "mlp" or "block" (block curve only)
        self.outside_name = outside_name  # what outside_blocks scales
        self._paths = {path: i for i, (_, path) in enumerate(rows)}

    @property
    def labels(self):
        return [label for label, _ in self.rows]

    def where(self, model_key: str):
        """(block, row) for a layer inside the numbered blocks, else None."""
        m = self.block_re.search(model_key)
        if not m:
            return None
        row = self._paths.get(m.group(2))
        if row is None:
            row = self._match_prefix(m.group(2))
        return None if row is None else (int(m.group(1)), row)

    def _match_prefix(self, path):
        # rows whose path ends in "*" take every module that starts with it (Anima's adaln_modulation_*)
        for i, (_, p) in enumerate(self.rows):
            if p.endswith("*") and path.startswith(p[:-1]):
                return i
        return None

    def mult(self, row: int, attention: float, mlp: float) -> float:
        k = self.kinds[row]
        return attention if k == "attn" else mlp if k == "mlp" else 1.0


KREA2 = Arch(
    "krea2", "Krea 2",
    r"(?:^|\.)blocks\.(\d+)\.((?:attn|mlp)\.\w+)\.weight$",
    [("q", "attn.wq"), ("k", "attn.wk"), ("v", "attn.wv"), ("o", "attn.wo"), ("gate", "attn.gate"),
     ("m.gate", "mlp.gate"), ("m.up", "mlp.up"), ("m.down", "mlp.down")],
    ["attn"] * 5 + ["mlp"] * 3,
    "text fusion, first and last layers",
)

# Anima (Cosmos Predict2 2B blocks): self-attention, cross-attention to the text, MLP, and the timestep modulation
# (adaln) per block. Anchored on diffusion_model.blocks so the LLM adapter's own blocks.N stay outside.
ANIMA = Arch(
    "anima", "Anima",
    r"^diffusion_model\.blocks\.(\d+)\.(.+)\.weight$",
    [("sa.q", "self_attn.q_proj"), ("sa.k", "self_attn.k_proj"), ("sa.v", "self_attn.v_proj"),
     ("sa.o", "self_attn.output_proj"), ("ca.q", "cross_attn.q_proj"), ("ca.k", "cross_attn.k_proj"),
     ("ca.v", "cross_attn.v_proj"), ("ca.o", "cross_attn.output_proj"), ("mlp.1", "mlp.layer1"),
     ("mlp.2", "mlp.layer2"), ("adaln", "adaln_modulation*")],
    ["attn"] * 8 + ["mlp"] * 2 + ["block"],
    "LLM adapter and final layer",
)
