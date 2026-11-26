"""
Random (stochastic) rounding to bf16 for saved models.

Rounding to the nearest bf16 value drops a small change: a LoRA or merge that moves a weight by less than half a bf16
step leaves it where it was, so a subtle LoRA baked into a bf16 file keeps only part of its effect (HaDeS-MOSS-Suite
measured 80-85%). Random rounding rounds up or down with odds set by the distance, so on average every change survives.
Seeded per tensor name: the same inputs save the same file.

ComfyUI's own save path (comfy.float.stochastic_rounding) rounds bf16 to nearest; only fp8 is random there.
"""

from __future__ import annotations

import contextlib
import zlib

import torch

_CHUNK = 1 << 24  # elements per slice, keeps the int32 work under ~200 MB


def stochastic_bf16(x: torch.Tensor, seed: int | str = 0) -> torch.Tensor:
    """x (any float dtype) -> bf16, rounded at random in proportion to the distance to the two neighbours."""
    if isinstance(seed, str):
        seed = zlib.crc32(seed.encode())
    g = torch.Generator(device=x.device)
    g.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    src = x.detach().to(torch.float32).contiguous().reshape(-1)
    out = torch.empty(src.shape, dtype=torch.bfloat16, device=x.device)
    for i in range(0, src.numel(), _CHUNK):
        s = src[i:i + _CHUNK]
        bits = s.view(torch.int32)
        noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=x.device, generator=g)
        r = ((bits + noise) & -65536).view(torch.float32)  # -65536 = 0xFFFF0000: keep the bf16 part
        r = torch.where(torch.isfinite(s) & torch.isfinite(r), r, s)  # nan/inf (and overflow at the top) stay nearest
        out[i:i + _CHUNK] = r.to(torch.bfloat16)
    return out.reshape(x.shape)


@contextlib.contextmanager
def random_rounding_for_save():
    """While saving through comfy.sd.save_checkpoint: patched bf16 weights are rounded at random, not to nearest.
    Unloads the loaded models first, so weights already patched for sampling (rounded to nearest then) are patched
    again from the originals."""
    import comfy.float
    import comfy.model_management

    comfy.model_management.unload_all_models()
    original = comfy.float.stochastic_rounding

    def rounding(value, dtype, seed=0):
        if dtype == torch.bfloat16 and value.dtype != torch.bfloat16:
            return stochastic_bf16(value, seed)
        return original(value, dtype, seed=seed)

    comfy.float.stochastic_rounding = rounding
    try:
        yield
    finally:
        comfy.float.stochastic_rounding = original
