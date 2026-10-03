// LC Krea2 LoRA Editor and LC Krea2 LoRA Merge: 28 draggable block bars and heatmaps (8 layer rows x 28 blocks)
// filled in after each run. One DOM canvas, so it looks the same in Nodes classic and Nodes 2.0.
//   Editor: bars 0..2 (1 = unchanged) over the LoRA's own strength per block; heatmaps before / after.
//   Merge : bars 0..1 (0 = LoRA A only, 0.5 = both full, 1 = LoRA B only); heatmaps A, B and merged.
import { app } from "../../scripts/app.js";

const N = 28;
const ROWS = ["q", "k", "v", "o", "gate", "m.gate", "m.up", "m.down"];
const H_BARS = 120;
const CELL_H = 9;
const hTotal = (cfg) => H_BARS + 22 + cfg.maps * (ROWS.length * CELL_H + 14) + 18;

const CFG = {
  LCKrea2LoRAEditor: {
    max: 2, rest: 1, ticks: ["2", "1", "0"], dividers: [4, 10, 16, 22], maps: 2, backdrop: true,
    low: "rgba(120,170,255,0.75)", high: "rgba(255,150,80,0.8)",
    // Mirrors _preset_curve in lc_krea2_lora_editor.py
    preset(name) {
      if (name === "Full") return new Array(N).fill(1);
      if (name === "Tame (4-15 at 0.7)") return Array.from({ length: N }, (_, i) => (i >= 4 && i <= 15 ? 0.7 : 1));
      const m = /^Blocks (\d+)-(\d+) down$/.exec(name || "");
      if (!m) return null;
      const lo = +m[1], hi = +m[2];
      return Array.from({ length: N }, (_, i) => (i >= lo && i <= hi ? 0.3 : 1));
    },
  },
  LCKrea2LoRAMerge: {
    max: 1, rest: 0.5, ticks: ["B", "both", "A"], dividers: [14], maps: 3, backdrop: false,
    low: "rgba(120,170,255,0.75)", high: "rgba(255,150,80,0.8)",
    // Mirrors PRESETS in lc_krea2_lora_merge.py
    preset(name) {
      const f = (fn) => Array.from({ length: N }, (_, i) => fn(i));
      const face = (lo, hi) => f((i) => (i >= lo && i <= hi ? 1 : i === lo - 1 || i === hi + 1 ? 0.5 : 0));
      return {
        "Both full": f(() => 0.5), "A early, B late": f((i) => (i < 14 ? 0 : 1)), "B early, A late": f((i) => (i < 14 ? 1 : 0)),
        "A only": f(() => 0), "B only": f(() => 1),
        "Face from B (8-20)": face(8, 20), "Face from B (4-13)": face(4, 13), "Face from B (16-21)": face(16, 21),
      }[name] || null;
    },
  },
};

const W = (node, name) => (node.widgets || []).find((w) => w.name === name);

function wired(node) {
  const i = (node.inputs || []).find((x) => x.name === "blocks_in");
  return i?.link != null;
}

function curveOf(node) {
  if (wired(node) && node._lcMap?.curve?.length === N) return node._lcMap.curve;
  const v = String(W(node, "blocks")?.value ?? "")
    .split(",")
    .map((x) => parseFloat(x))
    .filter((x) => Number.isFinite(x));
  const cfg = node._lcCfg;
  while (v.length < N) v.push(cfg.rest);
  return v.slice(0, N).map((x) => Math.max(0, Math.min(cfg.max, x)));
}

function setCurve(node, vals) {
  const w = W(node, "blocks");
  if (w) w.value = vals.map((x) => +x.toFixed(3)).join(", ");
}

function heat(t) {
  // dark -> amber -> pale yellow
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(40 + 215 * Math.min(1, t * 1.6));
  const g = Math.round(30 + 190 * Math.max(0, t - 0.25) / 0.75);
  const b = Math.round(20 + 120 * Math.max(0, t - 0.7) / 0.3);
  return `rgb(${r},${g},${b})`;
}

function draw(node) {
  const ui = node._lcLE;
  if (!ui) return;
  const cfg = node._lcCfg;
  const H_TOTAL = hTotal(cfg);
  const cv = ui.canvas;
  const dpr = window.devicePixelRatio || 1;
  const cw = Math.max(200, cv.clientWidth || 300);
  if (cv.width !== Math.round(cw * dpr) || cv.height !== Math.round(H_TOTAL * dpr)) {
    cv.width = Math.round(cw * dpr);
    cv.height = Math.round(H_TOTAL * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, H_TOTAL);
  const L = 40, R = 6;
  const gw = cw - L - R;
  const bw = gw / N;
  const vals = curveOf(node);
  const map = node._lcMap;
  const maps = map ? map.maps || [["before", map.before], ["after (same scale)", map.after]] : null;

  // bars, with the LoRA's own strength per block behind them (Editor)
  const top = 6, h = H_BARS - 12;
  ctx.fillStyle = "rgba(0,0,0,0.28)";
  ctx.fillRect(L, top, gw, h);
  if (maps && cfg.backdrop) {
    const tot = (m) => Array.from({ length: N }, (_, b) => Math.sqrt(m.reduce((s, row) => s + row[b] * row[b], 0)));
    const before = tot(maps[0][1]);
    const mx = Math.max(...before) || 1;
    ctx.fillStyle = "rgba(255,255,255,0.10)";
    before.forEach((v, b) => {
      const hh = (v / mx) * h;
      ctx.fillRect(L + b * bw + 1, top + h - hh, bw - 2, hh);
    });
  }
  const y1 = top + h / 2;
  ctx.strokeStyle = "rgba(255,255,255,0.25)";
  ctx.beginPath();
  ctx.moveTo(L, y1);
  ctx.lineTo(L + gw, y1);
  ctx.stroke();
  // the preset block groups
  ctx.font = "9px sans-serif";
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.setLineDash([3, 3]);
  cfg.dividers.forEach((b) => {
    ctx.beginPath();
    ctx.moveTo(L + b * bw, top);
    ctx.lineTo(L + b * bw, top + h);
    ctx.stroke();
  });
  ctx.setLineDash([]);
  vals.forEach((v, b) => {
    const y = top + h - (v / cfg.max) * h;
    const hot = b === ui.hover || b === ui.drag;
    ctx.fillStyle = v < cfg.rest ? cfg.low : v > cfg.rest ? cfg.high : "rgba(212,164,79,0.7)";
    if (hot) ctx.fillStyle = "#f0d9a8";
    ctx.fillRect(L + b * bw + 2, Math.min(y, y1), bw - 4, Math.max(1.5, Math.abs(y - y1)));
    ctx.fillRect(L + b * bw + 1, y - 1, bw - 2, 2);
  });
  ctx.textAlign = "right";
  ctx.fillStyle = "rgba(255,255,255,0.55)";
  ctx.fillText(cfg.ticks[0], L - 4, top + 8);
  ctx.fillText(cfg.ticks[1], L - 4, y1 + 3);
  ctx.fillText(cfg.ticks[2], L - 4, top + h);
  ctx.textAlign = "center";
  for (let b = 0; b < N; b += 3) ctx.fillText(String(b), L + (b + 0.5) * bw, top + h + 10);
  if (ui.hover >= 0 || ui.drag >= 0) {
    const b = ui.drag >= 0 ? ui.drag : ui.hover;
    ctx.textAlign = "left";
    ctx.fillStyle = "#f0d9a8";
    ctx.fillText(`block ${b}: ${vals[b].toFixed(2)}`, L + 4, top + h - 4);
  }

  // heatmaps
  let y = H_BARS + 10;
  const drawMap = (label, m, mx) => {
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.font = "9px sans-serif";
    ctx.fillText(label, 2, y + 7);
    y += 10;
    ROWS.forEach((name, r) => {
      ctx.textAlign = "right";
      ctx.fillStyle = "rgba(255,255,255,0.4)";
      ctx.font = "8px sans-serif";
      ctx.fillText(name, L - 4, y + r * CELL_H + 7);
      for (let b = 0; b < N; b++) {
        const v = m ? m[r][b] : 0;
        ctx.fillStyle = m ? heat(Math.sqrt(v / mx)) : "rgba(255,255,255,0.05)";
        ctx.fillRect(L + b * bw + 0.5, y + r * CELL_H + 0.5, bw - 1, CELL_H - 1);
      }
    });
    y += ROWS.length * CELL_H + 4;
  };
  const mx = maps ? Math.max(1e-9, ...maps.flatMap(([, m]) => m.flat())) : 1;
  const labels = cfg.maps === 3 ? ["LoRA A", "LoRA B", "merged"] : ["before", "after (same scale)"];
  for (let i = 0; i < cfg.maps; i++) drawMap(maps ? maps[i][0] : labels[i], maps ? maps[i][1] : null, mx);
  ctx.textAlign = "left";
  ctx.fillStyle = "rgba(255,255,255,0.65)";
  ctx.font = "10px sans-serif";
  const wiredNote = wired(node) ? (map?.wired ? "bars from blocks_in · " : "blocks_in wired: run to see its curve · ") : "";
  ctx.fillText(wiredNote + (map ? map.note : "Run once to see the LoRA's layer strengths."), 2, H_TOTAL - 5, cw - 4);
}

function blockAt(node, ev) {
  const cv = node._lcLE.canvas;
  const r = cv.getBoundingClientRect();
  const sx = (cv.clientWidth || r.width) / r.width; // canvas zoom-independent
  const x = (ev.clientX - r.left) * sx;
  const y = (ev.clientY - r.top) * sx;
  const L = 40, gw = (cv.clientWidth || 300) - L - 6;
  if (y > H_BARS || x < L || x > L + gw) return null;
  const mxv = node._lcCfg.max;
  return { b: Math.max(0, Math.min(N - 1, Math.floor(((x - L) / gw) * N))), v: Math.max(0, Math.min(mxv, mxv * (1 - (y - 6) / (H_BARS - 12)))) };
}

function setup(node, cfg) {
  if (node._lcLE) return;
  node._lcCfg = cfg;
  const H_TOTAL = hTotal(cfg);
  const bw = W(node, "blocks");
  if (bw) {
    bw.hidden = true;
    bw.type = "hidden";
    bw.computeSize = () => [0, -4];
  }
  const canvas = document.createElement("canvas");
  canvas.style.cssText = `width:100%;height:${H_TOTAL}px;display:block;cursor:ns-resize;touch-action:none`;
  const wrap = document.createElement("div");
  wrap.style.cssText = `width:100%;height:${H_TOTAL}px;`;
  wrap.appendChild(canvas);
  node._lcLE = { canvas, hover: -1, drag: -1 };
  node.addDOMWidget("lc_lora_graph", "lc_lora_graph", wrap, {
    serialize: false,
    getMinHeight: () => H_TOTAL,
    getMaxHeight: () => H_TOTAL,
  });

  const put = (hit, ev) => {
    const vals = curveOf(node);
    vals[hit.b] = ev.shiftKey ? cfg.rest : Math.round(hit.v * 20) / 20; // 0.05 steps, shift = reset
    setCurve(node, vals);
    const p = W(node, "preset");
    if (p && p.value !== "Custom") p.value = "Custom";
    node.setDirtyCanvas?.(true, true);
    draw(node);
  };
  canvas.addEventListener("pointerdown", (ev) => {
    const hit = blockAt(node, ev);
    if (!hit || wired(node)) return; // a wired curve comes from blocks_in, not the bars
    ev.stopPropagation();
    ev.preventDefault();
    canvas.setPointerCapture(ev.pointerId);
    node._lcLE.drag = hit.b;
    put(hit, ev);
  });
  canvas.addEventListener("pointermove", (ev) => {
    const hit = blockAt(node, ev);
    if (node._lcLE.drag >= 0) {
      ev.stopPropagation();
      if (hit) {
        node._lcLE.drag = hit.b; // sweep across bars to paint them
        put(hit, ev);
      }
      return;
    }
    const h = hit ? hit.b : -1;
    if (h !== node._lcLE.hover) {
      node._lcLE.hover = h;
      draw(node);
    }
  });
  const end = () => {
    node._lcLE.drag = -1;
    draw(node);
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("pointerleave", () => {
    node._lcLE.hover = -1;
    if (node._lcLE.drag < 0) draw(node);
  });
  canvas.addEventListener("dblclick", (ev) => {
    const hit = blockAt(node, ev);
    if (!hit) return;
    ev.stopPropagation();
    put({ b: hit.b, v: cfg.rest }, ev);
  });

  const preset = W(node, "preset");
  if (preset) {
    const prev = preset.callback;
    preset.callback = function (v) {
      prev?.apply(this, arguments);
      const c = cfg.preset(v);
      if (c) setCurve(node, c);
      draw(node);
    };
  }
  new ResizeObserver(() => draw(node)).observe(canvas);
  setTimeout(() => draw(node), 50);
}

app.registerExtension({
  name: "LCModelBuilder.Krea2LoRAEditor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const cfg = CFG[nodeData.name];
    if (!cfg) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      setup(this, cfg);
      if (this.size[0] < 380) this.setSize?.([380, this.size[1]]);
      // the first size is taken before the DOM widget counts: grow to fit once it does
      requestAnimationFrame(() => {
        const need = this.computeSize?.()?.[1];
        if (need && this.size[1] < need) this.setSize([this.size[0], need]);
      });
      return r;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      setTimeout(() => draw(this), 50);
      return r;
    };
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      onExecuted?.apply(this, arguments);
      const m = msg?.lc_lora_map?.[0];
      if (m) {
        this._lcMap = m;
        draw(this);
      }
    };
  },
});
