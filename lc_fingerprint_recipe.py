"""
LC Fingerprint recipe: what a model is made of, block by block, from the fingerprint library.

Every block of the mystery model is fitted as a mix of library models (weights >= 0 that add up to 1, like a merge
ratio) plus library LoRAs on top (strengths >= 0, the same for every block). Sketches are linear, so a weighted merge,
a block merge and a merged-in LoRA all come back exactly; whatever is left over is the part the library can't explain
(training after the merge, or an ingredient that isn't in the library).

Confidence of each ingredient = how much worse the fit gets without it (a model another one can stand in for scores
low). The recipe's likelihood = the share of the change from the family's base it explains x how sure each named
ingredient is.
"""
from __future__ import annotations

import math
import re

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont



def group_of(name: str) -> str:
    """The block a layer belongs to: blocks_12, double_blocks_3, txtfusion_refiner_blocks_0, ... else its first part."""
    parts = name.split("_")
    for i in range(1, len(parts)):
        if parts[i].isdigit() and (parts[i - 1].endswith("blocks") or parts[i - 1] == "layers"):
            return "_".join(parts[: i + 1])
    return parts[0]


def _order(g: str):
    m = re.match(r"^(.*?)(\d+)$", g)
    return (0 if m and m.group(1) in ("blocks_", "double_blocks_", "input_blocks_", "layers_") else 1,
            m.group(1) if m else g, int(m.group(2)) if m else -1)


def label_of(g: str) -> str:
    return g.replace("txtfusion_", "txt ").replace("_blocks_", " ").replace("blocks_", "block ").replace("_", " ")


def _cos(a, b):
    return float(a @ b) / max(1e-12, float(a.norm() * b.norm()))


def _simplex(v):
    """Euclidean projection onto {w >= 0, sum w = 1}."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    k = np.arange(1, len(v) + 1)
    rho = int(np.nonzero(u - css / k > 0)[0].max())
    return np.maximum(v - css[rho] / (rho + 1), 0)


def _pos(v):
    return np.maximum(v, 0)


def _pg(G, b, x, proj, iters=200):
    """min 0.5 x'Gx - b'x over a convex set, accelerated projected gradient (G is small)."""
    if len(x) == 0:
        return x
    L = max(float(np.linalg.eigvalsh(G).max()), 1e-20)
    y, t = x.copy(), 1.0
    for _ in range(iters):
        nx = proj(y - (G @ y - b) / L)
        nt = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = nx + (t - 1) / nt * (nx - x)
        x, t = nx, nt
    return x


class Problem:
    """Per-block Gram matrices of the mystery model against the candidate models and LoRAs (all the solving after this
    is on small matrices, so trying recipes without one ingredient is quick)."""

    def __init__(self, sk, models, loras, base):
        self.groups = sorted({group_of(k) for k in sk if all(k in m["sketch"] for m in models)}, key=_order)
        self.models, self.loras = models, loras
        self.G = []
        anchor = base["sketch"]
        for g in self.groups:
            keys = [k for k in sk if group_of(k) == g and all(k in m["sketch"] for m in models)]
            y = torch.cat([sk[k] - anchor[k] for k in keys]).double().numpy()
            M = torch.stack([torch.cat([m["sketch"][k] - anchor[k] for k in keys]) for m in models], 1).double().numpy()
            if loras:
                Lc = torch.stack([torch.cat([e["sketch"][k] if k in e["sketch"] else torch.zeros_like(sk[k]) for k in keys])
                                  for e in loras], 1).double().numpy()
            else:
                Lc = np.zeros((len(y), 0))
            size = float(torch.cat([sk[k] for k in keys]).double().pow(2).sum())  # |model| in this block (noise scale)
            self.G.append({"yy": float(y @ y), "My": M.T @ y, "MM": M.T @ M, "Ly": Lc.T @ y, "LL": Lc.T @ Lc,
                           "ML": M.T @ Lc, "size": size, "n": len(keys), "D": len(y)})

    def solve(self, mi, li, iters=8):
        """mi / li: the model and LoRA indices allowed. -> (weights per block [B x M], LoRA strengths [L], leftover per
        block [B], LoRA part per block [B])."""
        nm, nl = len(self.models), len(self.loras)
        mi, li = np.asarray(mi, dtype=int), np.asarray(li, dtype=int)
        W = np.zeros((len(self.groups), nm))
        c = np.zeros(nl)
        for b, g in enumerate(self.G):  # start: the closest single candidate per block
            err = -2 * g["My"][mi] + np.diag(g["MM"])[mi]
            W[b, mi[int(err.argmin())]] = 1.0
        GL = sum(g["LL"][np.ix_(li, li)] for g in self.G) if len(li) else None
        for _ in range(iters if len(li) else 1):
            for b, g in enumerate(self.G):
                rhs = g["My"][mi] - (g["ML"][np.ix_(mi, li)] @ c[li] if len(li) else 0)
                W[b, mi] = _pg(g["MM"][np.ix_(mi, mi)], rhs, W[b, mi], _simplex)
            if len(li):
                bl = sum(g["Ly"][li] - g["ML"][np.ix_(mi, li)].T @ W[b, mi] for b, g in enumerate(self.G))
                c[li] = _pg(GL, bl, c[li], _pos, 300)
        left = np.zeros(len(self.groups))
        lp = np.zeros(len(self.groups))
        for b, g in enumerate(self.G):
            w = W[b]
            fit = g["yy"] - 2 * (w @ g["My"]) - 2 * (c @ g["Ly"]) + w @ g["MM"] @ w + c @ g["LL"] @ c + 2 * (w @ g["ML"] @ c)
            left[b] = max(0.0, float(fit))
            lp[b] = float(c @ g["LL"] @ c)
        return W, c, left, lp

    def lora_scores(self, W):
        """How well each LoRA lines up with what the models leave over (to pick the few worth trying)."""
        s = np.zeros(len(self.loras))
        for b, g in enumerate(self.G):
            s += g["Ly"] - g["ML"].T @ W[b]  # L'(y - Mw)
        n = np.sqrt(np.maximum(sum(np.diag(g["LL"]) for g in self.G), 1e-30))
        return s / n


def recipe(sk, models, loras, base, noise=0.0015, exclude=()):
    """The recipe: per block weights, LoRAs, leftovers and confidences. models/loras are library entries (sketch dict).
    base = the family's reference model (the "change" is measured from it)."""
    models = _dedupe([m for m in models if m["name"] not in exclude], keep=base["name"])
    loras = [e for e in loras if any(k in e["sketch"] for k in sk)]
    P = Problem(sk, models, loras, base)
    # rounding noise the file carries: leftovers below it are not "unexplained". Only 85% of it is allowed for, so a
    # noisy file never reads as a perfect fit.
    noise_rel = noise
    noise = np.array([0.85 * noise_rel ** 2 * g["size"] for g in P.G])
    change = np.array([g["yy"] for g in P.G])  # distance from the base, per block
    allm = list(range(len(models)))

    def unx(left):
        return np.maximum(left - noise, 0)
    # models alone, then the LoRAs that line up best with what is left, pruned until each one is needed
    W, c, left, lp = P.solve(allm, [])
    li = []
    if loras:
        sc = P.lora_scores(W)
        li = [int(j) for j in np.argsort(-sc)[:12] if sc[j] > 0]
    mi = list(allm)
    bi = next(i for i, m in enumerate(models) if m["name"] == base["name"])
    # drop the weakest ingredient until every one left is needed, one at a time (dropping one can make another needed).
    # The base is never dropped: where a model is an exact copy of the base (a block merge), the base gets the credit.
    for _ in range(30):
        W, c, left, lp = P.solve(mi, li)
        mi = [i for i in mi if i == bi or float(W[:, i].max()) >= 0.02]
        li = [j for j in li if c[j] >= 0.05]
        here = unx(left)
        weak = []
        for i in [k for k in mi if k != bi]:
            where = W[:, i] >= 0.02
            _, _, l2, _ = P.solve([k for k in mi if k != i], li, iters=4)
            n = _necessity(here[where].sum(), unx(l2)[where].sum())
            if n < 0.1:
                weak.append((n, "m", i))
        for j in li:
            _, _, l2, _ = P.solve(mi, [k for k in li if k != j], iters=4)
            n = _necessity(here.sum(), unx(l2).sum())
            if n < 0.15:
                weak.append((n, "l", j))
        if not weak:
            break
        _, kind, idx = min(weak)
        (mi if kind == "m" else li).remove(idx)
    W, c, left, lp = P.solve(mi, li)
    unexpl = unx(left)
    used_m = [i for i in mi if float(W[:, i].max()) >= 0.02]
    used_l = [j for j in li if c[j] >= 0.05]
    # what changed from the base beyond the rounding: the part a recipe can explain
    signal = np.maximum(change - noise / 0.85, 0)
    total_change = float(signal.sum())
    explained = 1.0 - float(unexpl.sum()) / total_change if total_change > 0 else 1.0
    clarity = total_change / max(total_change + float((noise / 0.85).sum()), 1e-30)  # 1 = rounding hides nothing
    conf = {}
    for i in used_m:  # refit without it, judged on the blocks it was used in
        where = W[:, i] >= 0.02
        rest = [k for k in used_m if k != i]
        if not rest:
            conf[("m", i)] = 1.0
            continue
        _, _, l2, _ = P.solve(rest, used_l)
        conf[("m", i)] = _necessity(float(unexpl[where].sum()), float(unx(l2)[where].sum()))
    D = float(sum(g["D"] for g in P.G))
    z = {}
    for j in used_l:  # how much better the fit is with it than a random direction would make it (on the raw leftover)
        _, _, l2, _ = P.solve(used_m, [k for k in used_l if k != j])
        z[j] = max(0.0, float(l2.sum() - left.sum())) / max(float(left.sum()) / D, 1e-30)
        conf[("l", j)] = _significance(z[j]) * _steady(P, W, c, j)

    def t(a):
        return torch.from_numpy(np.asarray(a, dtype=np.float64))
    return {"groups": P.groups, "labels": [label_of(g) for g in P.groups], "models": models, "loras": loras,
            "W": t(W), "c": t(c), "unexplained": t(unexpl), "lora_part": t(lp), "change": t(change),
            "z": z, "size": t([g["n"] for g in P.G]), "explained": max(0.0, min(1.0, explained)), "clarity": clarity, "used_m": used_m, "used_l": used_l, "conf": conf, "base": base}


def _steady(P, W, c, j):
    """0..1: does LoRA j sit at the same strength in every block (a baked-in LoRA does), or only line up in some (a
    lookalike)? Strength per block from what is left for it once everything else is taken out."""
    s, w = [], []
    for b, g in enumerate(P.G):
        ljj = float(g["LL"][j, j])
        if ljj <= 0:
            continue
        rest = g["Ly"][j] - float(W[b] @ g["ML"][:, j]) - float(g["LL"][j] @ c) + ljj * float(c[j])
        s.append(rest / ljj)
        w.append(ljj)
    if len(s) < 3:
        return 0.5
    s, w = np.array(s), np.array(w)
    mean = float((s * w).sum() / w.sum())
    spread = float(np.sqrt(((s - mean) ** 2 * w).sum() / w.sum()))
    ratio = max(mean, 0.0) / max(spread, 1e-9)
    return float(ratio / (ratio + 2.0))


def _significance(z):
    """0..1 from z (fit gain / what a random direction gives): a few times random is noise, hundreds is a real ingredient."""
    return float(z / (z + 400.0))


def _necessity(left, left_without):
    """0..1: how much of the fit is lost without this ingredient (1 = nothing else in the library can stand in for it)."""
    lost = max(0.0, float(left_without) - float(left))
    return lost / max(lost + float(left), 1e-12)


def _low_precision(e):
    """Stored at fp8 or lower (from the source file's header when it is still there, else from the name)."""
    src = e.get("source") or ""
    try:
        import json
        import struct
        with open(src, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        dt = [v["dtype"] for k, v in h.items() if k.endswith(".weight") and len(v["shape"]) >= 2]
        return any(d not in ("BF16", "F16", "F32") for d in dt) or any("_qdata" in k or k.endswith(".comfy_quant") for k in h)
    except Exception:
        n = e["name"].lower()
        return any(t in n for t in ("fp8", "int8", "nvfp4", "q8", "q4", "gguf", "mxfp", "w4a"))


def _rel95(a, b):
    shared = [k for k in a["sketch"] if k in b["sketch"] and a["sketch"][k].shape == b["sketch"][k].shape]
    if not shared:
        return 1.0
    rel = sorted(float((a["sketch"][k] - b["sketch"][k]).norm()) / max(1e-12, float(b["sketch"][k].norm())) for k in shared)
    return rel[int(len(rel) * 0.95)]


def _same_model(a, b, low_a):
    """The same weights: a near exact copy, or a low precision copy of a full precision entry (a finetune is a few
    percent off its base too, so only a low precision file gets that much room)."""
    r = _rel95(a, b)
    return r < 0.004 or (low_a and r < 0.08)


def _dedupe(models, keep=None):
    """Keep one entry per model (the same weights at bf16 and fp8 would split the weight between them)."""
    low = {id(m): _low_precision(m) for m in models}
    out = []
    for m in sorted(models, key=lambda e: (e["name"] != keep, low[id(e)])):
        if not any(_same_model(m, o, low[id(m)]) for o in out):
            out.append(m)
    return out


def pick_base(models, sk):
    """The family's reference: the library model the others sit closest to (for Krea 2 that is Turbo, since most
    finetunes start from it)."""
    if len(models) == 1:
        return models[0]
    keys = [k for k in sk if all(k in m["sketch"] for m in models)]
    vec = [torch.cat([m["sketch"][k] for k in keys]) for m in models]
    best, bi = None, 0
    for i, a in enumerate(vec):
        d = sum(float((a - b).norm()) for b in vec)
        if best is None or d < best:
            best, bi = d, i
    return models[bi]


# ------------------------------------------------------------------ the answer in words
def likelihood(r):
    """0..1 for the whole recipe: the share explained x how sure the named ingredients are."""
    named = [r["conf"][("m", i)] for i in r["used_m"] if _share(r, i) >= 0.1] + [r["conf"][("l", j)] for j in r["used_l"]]
    sure = min(named) if named else 1.0
    # a noisy (low precision) file can't be as sure: the rounding hides part of what changed
    return r["explained"] * (0.5 + 0.5 * sure) * (0.6 + 0.4 * r.get("clarity", 1.0))


def _share(r, i):
    """Share of the model's layers that come from model i."""
    n = r["size"]
    return float((r["W"][:, i] * n).sum() / n.sum())


def _ranges(idx, labels):
    """[3, 4, 5, 9] -> 'block 3-5, block 9'."""
    out, run = [], []
    for i in idx:
        if run and i == run[-1] + 1 and labels[i].split(" ")[0] == labels[run[-1]].split(" ")[0]:
            run.append(i)
            continue
        if run:
            out.append(run)
        run = [i]
    if run:
        out.append(run)
    txt = []
    for rr in out:
        a, b = labels[rr[0]], labels[rr[-1]]
        txt.append(a if len(rr) == 1 else f"{a}-{b.split(' ')[-1]}")
    return ", ".join(txt)


def words(r):
    """Plain sentences: the recipe, where each part sits, how sure."""
    W, labels, models = r["W"], r["labels"], r["models"]
    L = likelihood(r)
    lines = []
    parts = []
    for i in sorted(r["used_m"], key=lambda i: -_share(r, i)):
        sh = _share(r, i)
        if sh < 0.02:
            continue
        parts.append(f"{models[i]['name']} {sh:.0%}")
    for j in r["used_l"]:
        parts.append(f"LoRA {r['loras'][j]['name']} at ~{float(r['c'][j]):.2f}")
    if r.get("clarity", 1.0) < 0.3:  # the rounding is bigger than what changed: no recipe can be read from this file
        return [f"Can't read a recipe from this file: its rounding is bigger than the differences between the library "
                f"models (only {r['clarity']:.0%} of what changed rises above it). Best guess: " + (" + ".join(parts) or "none")
                + ". A bf16, fp16 or fp8 copy of the same model gives a real answer."]
    if r["explained"] < 0.25:
        top = max(r["used_m"], key=lambda i: _share(r, i)) if r["used_m"] else None
        return [f"Not a merge of anything in your library: {1 - r['explained']:.0%} of what changed is not in it. "
                + (f"Closest: {models[top]['name']}. " if top is not None else "")
                + "That is either the maker's own training or models and LoRAs your library doesn't have; the "
                  "fingerprint can't tell those two apart."]
    word = "Very likely" if L >= 0.85 else "Likely" if L >= 0.65 else "Possibly" if L >= 0.4 else "Unlikely"
    lines.append(f"{word} ({L:.0%}): " + (" + ".join(parts) if parts else "nothing in the library fits"))
    # block merge or a plain merge? Blocks with the same mix (to the nearest 0.05) are listed together.
    sig = []
    for b in range(len(labels)):
        parts_b = [(i, min(1.0, round(float(W[b, i]) * 20) / 20) if float(W[b, i]) < 0.95 else 1.0) for i in r["used_m"] if float(W[b, i]) >= 0.1]
        sig.append(tuple(sorted(parts_b, key=lambda x: -x[1])))
    kinds = list(dict.fromkeys(sig))
    if len(kinds) == 1 and len(kinds[0]) > 1:
        ratio = ", ".join(f"{models[i]['name']} {w:.2f}" for i, w in kinds[0])
        lines.append(f"A plain weighted merge (the same mix in every block): {ratio}.")
    elif len(kinds) > 1:
        lines.append("A block merge: different blocks come from different models.")
        shown = [k for k in sorted(kinds, key=lambda k: -sum(1 for x in sig if x == k)) if k]
        for n_, k in enumerate(shown):
            if n_ == 5:
                lines.append(f"  ... and {len(shown) - 5} more mixes (see the heatmap)")
                break
            who = " + ".join(models[i]["name"] + (f" {w:.2f}" if w < 0.95 else "") for i, w in k)
            lines.append(f"  {who}: {_ranges([b for b in range(len(labels)) if sig[b] == k], labels)}")
    for i in r["used_m"]:
        if _share(r, i) >= 0.02:
            lines.append(f"  sure of {models[i]['name']}: {r['conf'][('m', i)]:.0%}"
                         + ("  (another library model could stand in for it)" if r["conf"][("m", i)] < 0.5 else ""))
    for j in r["used_l"]:
        lines.append(f"  sure of LoRA {r['loras'][j]['name']}: {r['conf'][('l', j)]:.0%}")
    lines.append(f"Explains {r['explained']:.0%} of what changed from {r['base']['name']}.")
    if r["explained"] < 0.9:
        worst = sorted(range(len(labels)), key=lambda b: -float(r["unexplained"][b] / max(1e-12, r["change"][b])))[:4]
        lines.append(f"Not in your library: {1 - r['explained']:.0%}, most in {', '.join(labels[b] for b in sorted(worst))}. "
                     f"Either the maker's own training or a model or LoRA your library doesn't have (it can't tell which).")
    return lines


# ------------------------------------------------------------------ heatmap
_COLORS = [(86, 180, 233), (230, 159, 0), (0, 158, 115), (204, 121, 167), (240, 228, 66), (0, 114, 178),
           (213, 94, 0), (150, 150, 150)]


def heatmap(r, title):
    """Rows = blocks, columns = the models used, then LoRAs and what is not in the library. Brighter = a bigger share."""
    labels, W = r["labels"], r["W"]
    cols = [(r["models"][i]["name"], W[:, i].tolist(), _COLORS[n % len(_COLORS)]) for n, i in enumerate(r["used_m"])]
    ch = r["change"].clamp(min=1e-12)
    if r["used_l"]:  # where the LoRAs act, 1.00 = their strongest block
        lp = r["lora_part"].clamp(min=0).sqrt()
        cols.append(("LoRAs (where they act)", (lp / lp.max().clamp(min=1e-12)).tolist(), (200, 200, 200)))
    cols.append(("not in library", (r["unexplained"] / ch).clamp(0, 1).tolist(), (220, 80, 80)))
    try:
        f = ImageFont.truetype("arial.ttf", 12)
        fb = ImageFont.truetype("arialbd.ttf", 14)
    except OSError:
        f = fb = ImageFont.load_default()
    row_h, label_w, cell_w = 16, 110, 92
    head = 126
    w = label_w + cell_w * len(cols) + 20
    w = max(w, 560)
    h = head + row_h * len(labels) + 16
    im = Image.new("RGB", (w, h), (24, 24, 24))
    d = ImageDraw.Draw(im)
    d.text((10, 8), title[:90], fill=(255, 200, 80), font=fb)
    for k, line in enumerate(_wrap(words(r)[0], max(60, (w - 20) // 7))[:2]):
        d.text((10, 28 + k * 15), line, fill=(230, 230, 230), font=f)
    for n, (name, vals, col) in enumerate(cols):
        x = label_w + n * cell_w
        tx = Image.new("RGBA", (cell_w - 4, 54), (0, 0, 0, 0))
        dd = ImageDraw.Draw(tx)
        words_ = _wrap(name, 13)[:4]
        for k, line in enumerate(words_):
            dd.text((0, k * 13), line, fill=col, font=f)
        im.paste(tx, (x + 2, 66), tx)
        for b, v in enumerate(vals):
            y = head + b * row_h
            v = max(0.0, min(1.0, float(v)))
            fill = tuple(int(24 + (c - 24) * v) for c in col)
            d.rectangle([x + 2, y + 1, x + cell_w - 2, y + row_h - 1], fill=fill)
            if v >= 0.05:
                d.text((x + 6, y + 1), f"{v:.2f}", fill=(0, 0, 0) if v > 0.55 else (220, 220, 220), font=f)
    for b, lab in enumerate(labels):
        d.text((10, head + b * row_h + 1), lab[:16], fill=(200, 200, 200), font=f)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None]


def _wrap(s, n):
    out, cur = [], ""
    for ch in re.split(r"([ _\-.])", s):
        if len(cur) + len(ch) > n and cur:
            out.append(cur)
            cur = ch.lstrip(" _")
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out
