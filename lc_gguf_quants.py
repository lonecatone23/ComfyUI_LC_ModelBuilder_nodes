"""
GGUF block quantizers in torch (GPU or CPU), ported from llama.cpp's reference code (ggml-quants.c, MIT):
quantize_row_q8_0_ref, quantize_row_q{2,3,4,5,6}_K_ref with make_qkx2_quants, make_qx_quants and make_q3_quants.
The python gguf package can read these types but only writes Q8_0 and the legacy ones, so the K-quants are here.

Each quantizer takes a 2-D float tensor (rows, cols), cols a multiple of the block size, and returns
(raw bytes as uint8 (rows, cols / block * bytes_per_block), the dequantized weights for the error report).
Byte layouts follow ggml's block structs exactly; tests check them against ComfyUI-GGUF's own decoder.
"""

from __future__ import annotations

import torch

QK_K = 256
EPS = 1e-15  # GROUP_MAX_EPS

# ggml type id, block size, bytes per block
TYPES = {
    "Q8_0": (8, 32, 34),
    "Q2_K": (10, 256, 84),
    "Q3_K": (11, 256, 110),
    "Q4_K": (12, 256, 144),
    "Q5_K": (13, 256, 176),
    "Q6_K": (14, 256, 210),
}


def _rnd(x):
    """nearest_int(): round half to even, like ggml's float trick."""
    return torch.round(x)


def _half_bytes(v):
    """(B,) float -> fp16 as (B, 2) little-endian bytes, and its float value back."""
    h = v.to(torch.float16)
    return h.view(torch.uint8).view(-1, 2), h.float()


def _qkx2(x, w, nmax, rmin, rdelta, nstep, use_mad):
    """make_qkx2_quants over rows of x (N, n): scale (N,), the_min (N,) (>= 0, subtracted), L (N, n)."""
    mn = x.min(1).values.clamp(max=0.0)
    mx = x.max(1).values
    sum_w = w.sum(1)
    sum_x = (w * x).sum(1)
    flat = mx == mn
    rng = torch.where(flat, torch.ones_like(mx), mx - mn)

    def err(d):
        return d.abs() if use_mad else d * d

    iscale = nmax / rng
    scale = 1.0 / iscale
    L = (_rnd(iscale[:, None] * (x - mn[:, None]))).clamp(0, nmax)
    best = (w * err(scale[:, None] * L + mn[:, None] - x)).sum(1)
    for i in range(nstep + 1):
        isc = (rmin + rdelta * i + nmax) / rng
        La = (_rnd(isc[:, None] * (x - mn[:, None]))).clamp(0, nmax)
        sum_l = (w * La).sum(1)
        sum_l2 = (w * La * La).sum(1)
        sum_xl = (w * La * x).sum(1)
        D = sum_w * sum_l2 - sum_l * sum_l
        Ds = torch.where(D > 0, D, torch.ones_like(D))
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / Ds
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / Ds
        pos = this_min > 0
        this_scale = torch.where(pos, sum_xl / sum_l2.clamp(min=1e-30), this_scale)
        this_min = torch.where(pos, torch.zeros_like(this_min), this_min)
        mad = (w * err(this_scale[:, None] * La + this_min[:, None] - x)).sum(1)
        better = (D > 0) & (mad < best)
        L = torch.where(better[:, None], La, L)
        best = torch.where(better, mad, best)
        scale = torch.where(better, this_scale, scale)
        mn = torch.where(better, this_min, mn)
    scale = torch.where(flat, torch.zeros_like(scale), scale)
    L = torch.where(flat[:, None], torch.zeros_like(L), L)
    return scale, -mn, L


def _qx(x, nmax):
    """make_qx_quants(rmse_type = 1) over rows of x (N, n): scale (N,), L (N, n) in 0..2*nmax-1."""
    amax, idx = x.abs().max(1)
    mx = x.gather(1, idx[:, None])[:, 0]
    small = amax < EPS
    mxs = torch.where(small, torch.ones_like(mx), mx)
    w = x * x

    def ev(isc):
        l = (_rnd(isc[:, None] * x)).clamp(-nmax, nmax - 1)
        return l, (w * x * l).sum(1), (w * l * l).sum(1)

    L, sumlx, suml2 = ev(-nmax / mxs)
    scale = torch.where(suml2 > 0, sumlx / suml2.clamp(min=1e-30), torch.zeros_like(sumlx))
    best = scale * sumlx
    for i in range(-9, 10):
        if i == 0:
            continue
        l, slx, sl2 = ev(-(nmax + 0.1 * i) / mxs)
        better = (sl2 > 0) & (slx * slx > best * sl2)
        L = torch.where(better[:, None], l, L)
        scale = torch.where(better, slx / sl2.clamp(min=1e-30), scale)
        best = torch.where(better, scale * slx, best)
    L = L + nmax
    L = torch.where(small[:, None], torch.zeros_like(L), L)
    scale = torch.where(small, torch.zeros_like(scale), scale)
    return scale, L


def _q3(x, nmax=4):
    """make_q3_quants(do_rmse = true) over rows of x (N, n): scale (N,), L (N, n) in 0..2*nmax-1."""
    amax, idx = x.abs().max(1)
    mx = x.gather(1, idx[:, None])[:, 0]
    small = amax < EPS
    mxs = torch.where(small, torch.ones_like(mx), mx)
    l = (_rnd((-nmax / mxs)[:, None] * x)).clamp(-nmax, nmax - 1)
    w = x * x
    sumlx = (w * x * l).sum(1)
    suml2 = (w * l * l).sum(1)
    for _ in range(5):
        for i in range(x.shape[1]):
            wi, xi, li = w[:, i], x[:, i], l[:, i]
            slx = sumlx - wi * xi * li
            sl2 = suml2 - wi * li * li
            ok = slx > 0
            new_l = (_rnd(xi * sl2 / torch.where(ok, slx, torch.ones_like(slx)))).clamp(-nmax, nmax - 1)
            slx2 = slx + wi * xi * new_l
            sl22 = sl2 + wi * new_l * new_l
            take = ok & (new_l != li) & (sl22 > 0) & (slx2 * slx2 * suml2 > sumlx * sumlx * sl22)
            l[:, i] = torch.where(take, new_l, li)
            sumlx = torch.where(take, slx2, sumlx)
            suml2 = torch.where(take, sl22, suml2)
    scale = torch.where(small, torch.zeros_like(sumlx), sumlx / suml2.clamp(min=1e-30))
    L = torch.where(small[:, None], torch.zeros_like(l), l + nmax)
    return scale, L


def _pack2(L):
    """(B, 256) 2-bit values -> (B, 64): q[j/4 + l] = L[j+l] | L[j+l+32] << 2 | L[j+l+64] << 4 | L[j+l+96] << 6."""
    L = L.to(torch.int32).view(-1, 2, 4, 32)
    return (L[:, :, 0] | (L[:, :, 1] << 2) | (L[:, :, 2] << 4) | (L[:, :, 3] << 6)).reshape(-1, 64).to(torch.uint8)


def _pack_scales_k4(ls, lm):
    """6-bit scales and mins (B, 8) -> the 12-byte K4 layout (Q4_K, Q5_K)."""
    ls, lm = ls.to(torch.int32), lm.to(torch.int32)
    s = torch.zeros(ls.shape[0], 12, dtype=torch.int32, device=ls.device)
    s[:, 0:4] = ls[:, 0:4]
    s[:, 4:8] = lm[:, 0:4]
    s[:, 8:12] = (ls[:, 4:8] & 0xF) | ((lm[:, 4:8] & 0xF) << 4)
    s[:, 0:4] |= (ls[:, 4:8] >> 4) << 6
    s[:, 4:8] |= (lm[:, 4:8] >> 4) << 6
    return s.to(torch.uint8)


def _k4(x, nmax, rmin, rdelta, nstep):
    """Shared body of Q4_K / Q5_K: (B, 256) -> d bytes, dmin bytes, scales (B, 12), L (B, 256), dequant."""
    B = x.shape[0]
    sub = x.view(B, 8, 32)
    av = (sub.pow(2).sum(-1) / 32).sqrt()
    w = av[..., None] + sub.abs()
    sc, mins, _ = _qkx2(sub.reshape(-1, 32), w.reshape(-1, 32), nmax, rmin, rdelta, nstep, False)
    sc, mins = sc.view(B, 8), mins.view(B, 8)
    max_scale, max_min = sc.max(1).values, mins.max(1).values
    inv_s = torch.where(max_scale > 0, 63.0 / max_scale.clamp(min=1e-30), torch.zeros_like(max_scale))
    inv_m = torch.where(max_min > 0, 63.0 / max_min.clamp(min=1e-30), torch.zeros_like(max_min))
    ls = _rnd(inv_s[:, None] * sc).clamp(0, 63)
    lm = _rnd(inv_m[:, None] * mins).clamp(0, 63)
    d_b, d = _half_bytes(max_scale / 63.0)
    m_b, dmin = _half_bytes(max_min / 63.0)
    dd = d[:, None] * ls
    dm = dmin[:, None] * lm
    q = _rnd((sub + dm[..., None]) / torch.where(dd != 0, dd, torch.ones_like(dd))[..., None]).clamp(0, nmax)
    q = torch.where((dd != 0)[..., None], q, torch.zeros_like(q))
    deq = dd[..., None] * q - dm[..., None]
    return d_b, m_b, _pack_scales_k4(ls, lm), q.view(B, 256), deq.view(B, 256)


def q4_k(x):
    d_b, m_b, scales, L, deq = _k4(x, 15, -1.0, 0.1, 20)
    L = L.to(torch.int32).view(-1, 4, 2, 32)
    qs = (L[:, :, 0] | (L[:, :, 1] << 4)).reshape(-1, 128).to(torch.uint8)
    return torch.cat([d_b, m_b, scales, qs], 1), deq


def q5_k(x):
    d_b, m_b, scales, L, deq = _k4(x, 31, -0.5, 0.1, 15)
    L = L.to(torch.int32).view(-1, 4, 2, 32)  # (B, chunk, half, 32)
    hi = (L >> 4) & 1
    lo = L & 0xF
    qh = torch.zeros(L.shape[0], 32, dtype=torch.int32, device=L.device)
    for c in range(4):
        qh |= hi[:, c, 0] << (2 * c)
        qh |= hi[:, c, 1] << (2 * c + 1)
    ql = (lo[:, :, 0] | (lo[:, :, 1] << 4)).reshape(-1, 128)
    return torch.cat([d_b, m_b, scales, qh.to(torch.uint8), ql.to(torch.uint8)], 1), deq


def q6_k(x):
    B = x.shape[0]
    sub = x.view(B, 16, 16)
    sc, _ = _qx(sub.reshape(-1, 16), 32)
    sc = sc.view(B, 16)
    amax, idx = sc.abs().max(1)
    max_scale = sc.gather(1, idx[:, None])[:, 0]
    small = amax < EPS
    iscale = -128.0 / torch.where(small, torch.ones_like(max_scale), max_scale)
    d_b, d = _half_bytes(torch.where(small, torch.zeros_like(iscale), 1.0 / iscale))
    ls = _rnd(iscale[:, None] * sc).clamp(max=127)
    ls = torch.where(small[:, None], torch.zeros_like(ls), ls)
    dd = d[:, None] * ls
    q = _rnd(sub / torch.where(dd != 0, dd, torch.ones_like(dd))[..., None]).clamp(-32, 31) + 32
    q = torch.where((dd != 0)[..., None], q, torch.full_like(q, 32))
    deq = (dd[..., None] * (q - 32)).view(B, 256)
    L = q.view(B, 256).to(torch.int32).view(B, 2, 4, 32)  # (B, half, quarter, 32)
    lo, hi = L & 0xF, L >> 4
    ql = torch.cat([lo[:, :, 0] | (lo[:, :, 2] << 4), lo[:, :, 1] | (lo[:, :, 3] << 4)], 2).reshape(B, 128)
    qh = (hi[:, :, 0] | (hi[:, :, 1] << 2) | (hi[:, :, 2] << 4) | (hi[:, :, 3] << 6)).reshape(B, 64)
    scales = ls.to(torch.int8).view(torch.uint8)
    out = torch.cat([ql.to(torch.uint8), qh.to(torch.uint8), scales, d_b], 1)
    out = torch.where(small[:, None], torch.zeros_like(out), out)
    deq = torch.where(small[:, None], torch.zeros_like(deq), deq)
    return out, deq


def q3_k(x):
    B = x.shape[0]
    sub = x.view(B, 16, 16)
    sc, _ = _q3(sub.reshape(-1, 16), 4)
    sc = sc.view(B, 16)
    amax, idx = sc.abs().max(1)
    max_scale = sc.gather(1, idx[:, None])[:, 0]
    has = max_scale != 0
    iscale = -32.0 / torch.where(has, max_scale, torch.ones_like(max_scale))
    l = (_rnd(iscale[:, None] * sc).clamp(-32, 31) + 32).to(torch.int32)
    l = torch.where(has[:, None], l, torch.zeros_like(l))
    s = torch.zeros(B, 12, dtype=torch.int32, device=x.device)
    s[:, 0:8] = l[:, 0:8] & 0xF
    s[:, 0:8] |= (l[:, 8:16] & 0xF) << 4
    hi = l >> 4
    for j in range(16):
        s[:, 8 + j % 4] |= hi[:, j] << (2 * (j // 4))
    d_b, d = _half_bytes(torch.where(has, 1.0 / iscale, torch.zeros_like(iscale)))
    scb = torch.where(has[:, None], (l - 32).float(), torch.zeros_like(sc))
    dd = d[:, None] * scb
    q = _rnd(sub / torch.where(dd != 0, dd, torch.ones_like(dd))[..., None]).clamp(-4, 3) + 4
    q = torch.where((dd != 0)[..., None], q, torch.full_like(q, 4))
    deq = (dd[..., None] * (q - 4)).view(B, 256)
    L = q.view(B, 256).to(torch.int32)
    hbit = (L > 3).to(torch.int32)
    L = L - 4 * hbit
    hmask = (hbit.view(B, 8, 32) << torch.arange(8, device=x.device, dtype=torch.int32)[None, :, None]).sum(1)
    out = torch.cat([hmask.to(torch.uint8), _pack2(L), s.to(torch.uint8), d_b], 1)
    return out, deq


def q2_k(x):
    B = x.shape[0]
    sub = x.view(B, 16, 16)
    sc, mins, _ = _qkx2(sub.reshape(-1, 16), sub.abs().reshape(-1, 16), 3, -0.5, 0.1, 15, True)
    sc, mins = sc.view(B, 16), mins.view(B, 16)
    max_scale, max_min = sc.max(1).values, mins.max(1).values
    s4 = torch.where((max_scale > 0)[:, None], _rnd(15.0 / max_scale.clamp(min=1e-30)[:, None] * sc), torch.zeros_like(sc))
    m4 = torch.where((max_min > 0)[:, None], _rnd(15.0 / max_min.clamp(min=1e-30)[:, None] * mins), torch.zeros_like(mins))
    d_b, d = _half_bytes(torch.where(max_scale > 0, max_scale / 15.0, torch.zeros_like(max_scale)))
    m_b, dmin = _half_bytes(torch.where(max_min > 0, max_min / 15.0, torch.zeros_like(max_min)))
    dd = d[:, None] * s4
    dm = dmin[:, None] * m4
    q = _rnd((sub + dm[..., None]) / torch.where(dd != 0, dd, torch.ones_like(dd))[..., None]).clamp(0, 3)
    q = torch.where((dd != 0)[..., None], q, torch.zeros_like(q))
    deq = (dd[..., None] * q - dm[..., None]).view(B, 256)
    scales = (s4.to(torch.int32) | (m4.to(torch.int32) << 4)).to(torch.uint8)
    return torch.cat([scales, _pack2(q.view(B, 256)), d_b, m_b], 1), deq


def q8_0(x):
    B = x.shape[0]
    blk = x.view(B, 32)
    amax = blk.abs().max(1).values
    d_b, _ = _half_bytes(amax / 127.0)
    d = amax / 127.0
    idv = torch.where(d > 0, 1.0 / d.clamp(min=1e-30), torch.zeros_like(d))
    v = blk * idv[:, None]
    q = torch.sign(v) * torch.floor(v.abs() + 0.5)  # roundf: half away from zero
    q = q.clamp(-127, 127)
    deq = q * d_b.view(torch.float16).float().view(-1, 1)
    return torch.cat([d_b, q.to(torch.int8).view(torch.uint8)], 1), deq


_FN = {"Q8_0": q8_0, "Q2_K": q2_k, "Q3_K": q3_k, "Q4_K": q4_k, "Q5_K": q5_k, "Q6_K": q6_k}


def quantize(w: torch.Tensor, qtype: str, chunk_blocks: int = 1 << 18):
    """2-D weight (rows, cols) -> (uint8 bytes (rows, cols / block * size), dequantized float32 (rows, cols)).
    Works in chunks so a big layer never needs more than a few hundred MB of scratch."""
    _, bs, size = TYPES[qtype]
    rows, cols = w.shape
    assert cols % bs == 0, f"{qtype} needs a column count divisible by {bs} (got {cols})"
    x = w.reshape(-1, bs).float()
    outs, deqs = [], []
    for i in range(0, x.shape[0], chunk_blocks):
        o, d = _FN[qtype](x[i:i + chunk_blocks].contiguous())
        outs.append(o)
        deqs.append(d)
    return torch.cat(outs).view(rows, cols // bs * size), torch.cat(deqs).view(rows, cols)
