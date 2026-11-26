// LC Krea2 / Anima / MOSS LoRA Editor and LoRA Merge: draggable block bars (28 blocks, MOSS 42 layers) and heatmaps
// filled in after each run. One DOM canvas, so it looks the same in Nodes classic and Nodes 2.0.
//   Editor: bars 0..2 (1 = unchanged) over the LoRA's own strength per block; heatmaps before / after.
//   Merge : bars 0..1 (0 = LoRA A only, 0.5 = both full, 1 = LoRA B only); heatmaps A, B, merged, and A vs B
//           (green = they pull the same way, red = they fight, brightness = how much both change that layer).
import { app } from "../../scripts/app.js";

const N = 28;
const ROWS = ["q", "k", "v", "o", "gate", "m.gate", "m.up", "m.down"];
const ANIMA_ROWS = ["sa.q", "sa.k", "sa.v", "sa.o", "ca.q", "ca.k", "ca.v", "ca.o", "mlp.1", "mlp.2", "adaln"];
const MOSS_ROWS = ["qkv", "o", "ffn.up", "ffn.down"];
const H3_ROWS = ["qkv", "o", "gate", "fc1", "fc2", "adaln"];
const nOf = (node) => node._lcCfg?.n || N;
const H_BARS = 240;
const MIN_W = 380;
const CELL_H = 9;
const hTotal = (cfg) => H_BARS + 22 + cfg.maps * ((cfg.rows || ROWS).length * CELL_H + 14) + 18;

const CFG = {
  LCKrea2LoRAEditor: {
    max: 2, rest: 1, ticks: ["2", "1", "0"], dividers: [4, 10, 16, 22], maps: 2, backdrop: true,
    low: "rgba(120,170,255,0.75)", high: "rgba(255,150,80,0.8)",
    // Mirrors _preset_curve in lc_krea2_lora_editor.py
    legacy: {
      "Blocks 0-3 down": "Early blocks: structure (0-3)", "Blocks 4-9 down": "Face blocks (4-9)",
      "Blocks 10-15 down": "Mid blocks: face and body (10-15)", "Blocks 16-21 down": "Late face blocks (16-21)",
      "Blocks 22-27 down": "Finish: color and skin (22-27)", "Tame (4-15 at 0.7)": "Tame whole image (4-15 at 0.7)",
    },
    preset(name) {
      name = this.legacy[name] || name;
      if (name === "Full") return new Array(N).fill(1);
      const m = /\((\d+)-(\d+)(?: at ([\d.]+))?\)$/.exec(name || "");
      if (!m) return null;
      const lo = +m[1], hi = +m[2], v = m[3] ? +m[3] : 0.3;
      return Array.from({ length: N }, (_, i) => (i >= lo && i <= hi ? v : 1));
    },
  },
  LCAnimaLoRAEditor: {
    max: 2, rest: 1, ticks: ["2", "1", "0"], dividers: [7, 14, 21], maps: 2, backdrop: true, rows: ANIMA_ROWS,
    low: "rgba(120,170,255,0.75)", high: "rgba(255,150,80,0.8)",
    // Mirrors ANIMA_GROUPS in lc_krea2_lora_editor.py
    preset(name) {
      if (name === "Full") return new Array(N).fill(1);
      const m = /\((\d+)-(\d+)\)$|^Blocks (\d+)-(\d+)$/.exec(name || "");
      if (!m) return null;
      const lo = +(m[1] ?? m[3]), hi = +(m[2] ?? m[4]);
      return Array.from({ length: N }, (_, i) => (i >= lo && i <= hi ? 0.3 : 1));
    },
  },
  LCKrea2LoRAMerge: {
    max: 1, rest: 0.5, ticks: ["B", "both", "A"], dividers: [14], maps: 4, backdrop: false,
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
// Anima Merge: Krea Merge's bars with Anima's rows and presets (mirrors ANIMA_PRESETS in lc_krea2_lora_merge.py)
CFG.LCAnimaLoRAMerge = { ...CFG.LCKrea2LoRAMerge, rows: ANIMA_ROWS };
// MOSS: 42 layers. Editor groups mirror MOSS_GROUPS in lc_krea2_lora_editor.py, Merge presets MOSS_PRESETS (_basic)
CFG.LCMOSSLoRAEditor = {
  ...CFG.LCAnimaLoRAEditor, n: 42, dividers: [14, 28], rows: MOSS_ROWS,
  preset(name) {
    const n = 42;
    if (name === "Full") return new Array(n).fill(1);
    const m = /\((\d+)-(\d+)\)$/.exec(name || "");
    if (!m) return null;
    const lo = +m[1], hi = +m[2];
    return Array.from({ length: n }, (_, i) => (i >= lo && i <= hi ? 0.3 : 1));
  },
};
// H3 / H3 Image: 50 blocks. Groups mirror H3_GROUPS in lc_krea2_lora_editor.py
CFG.LCH3LoRAEditor = { ...CFG.LCMOSSLoRAEditor, n: 50, dividers: [17, 34], rows: H3_ROWS,
  preset(name) {
    const n = 50;
    if (name === "Full") return new Array(n).fill(1);
    const m = /\((\d+)-(\d+)\)$/.exec(name || "");
    if (!m) return null;
    const lo = +m[1], hi = +m[2];
    return Array.from({ length: n }, (_, i) => (i >= lo && i <= hi ? 0.3 : 1));
  },
};
// LC H3 LoRA Blocks: the H3 editor's bars and presets, no heatmaps
CFG.LCH3LoRABlocks = { ...CFG.LCH3LoRAEditor, maps: 0, hint: "Run once to see the LoRA's own strength per block behind the bars." };
// LC Krea2 LoRA Blocks: the Krea 2 editor's bars and presets, no heatmaps
CFG.LCKrea2LoRABlocks = { ...CFG.LCKrea2LoRAEditor, maps: 0, hint: CFG.LCH3LoRABlocks.hint };
CFG.LCMOSSLoRAMerge = {
  ...CFG.LCKrea2LoRAMerge, n: 42, dividers: [21], rows: MOSS_ROWS,
  preset(name) {
    const n = 42, f = (fn) => Array.from({ length: n }, (_, i) => fn(i));
    return {
      "Both full": f(() => 0.5), "A early, B late": f((i) => (i < 21 ? 0 : 1)), "B early, A late": f((i) => (i < 21 ? 1 : 0)),
      "A only": f(() => 0), "B only": f(() => 1),
    }[name] || null;
  },
};

function conflictColor(dir, t) {
  // green = same way, red = against, grey = shared but neither; t (0..1) = how much they change the same thing
  const c = Math.max(-1, Math.min(1, dir));
  const k = 0.12 + 0.88 * Math.max(0, Math.min(1, t));
  const base = c >= 0 ? [90, 210, 110] : [240, 80, 65];
  const a = Math.min(1, Math.abs(c) * 1.5);
  const mixc = (i) => Math.round((150 + (base[i] - 150) * a) * k);
  return `rgb(${mixc(0)},${mixc(1)},${mixc(2)})`;
}

const W = (node, name) => (node.widgets || []).find((w) => w.name === name);

function wired(node) {
  const i = (node.inputs || []).find((x) => x.name === "blocks_in");
  return i?.link != null;
}

function curveOf(node) {
  const N = nOf(node);
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

function rowsOf(node) {
  return node._lcMap?.rows || node._lcCfg.rows || ROWS;
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
  const N = nOf(node);
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
  const drawMap = (label, m, mx, cell) => {
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.font = "9px sans-serif";
    ctx.fillText(label, 2, y + 7);
    y += 10;
    rowsOf(node).forEach((name, r) => {
      ctx.textAlign = "right";
      ctx.fillStyle = "rgba(255,255,255,0.4)";
      ctx.font = "8px sans-serif";
      ctx.fillText(name, L - 4, y + r * CELL_H + 7);
      for (let b = 0; b < N; b++) {
        const v = m ? m[r]?.[b] ?? 0 : 0;
        ctx.fillStyle = !m ? "rgba(255,255,255,0.05)" : cell ? cell(r, b) : heat(Math.sqrt(v / mx));
        ctx.fillRect(L + b * bw + 0.5, y + r * CELL_H + 0.5, bw - 1, CELL_H - 1);
      }
    });
    y += rowsOf(node).length * CELL_H + 4;
  };
  const mx = maps ? Math.max(1e-9, ...maps.flatMap(([, m]) => m.flat())) : 1;
  const labels = cfg.maps >= 3 ? ["LoRA A", "LoRA B", "merged"] : ["before", "after (same scale)"];
  for (let i = 0; i < Math.min(cfg.maps, 3); i++) drawMap(maps ? maps[i][0] : labels[i], maps ? maps[i][1] : null, mx);
  if (cfg.maps === 4) {
    // brightness = how much A and B change the same thing (0.3 shared = full), color = same way / against there
    const cf = map?.conflict?.ov ? map.conflict : null;
    const cell = cf
      ? (r, b) => {
          const ov = cf.ov[r]?.[b] ?? 0;
          return ov > 0 ? conflictColor(cf.dir[r][b], Math.sqrt(Math.min(1, ov / 0.3))) : "rgba(255,255,255,0.05)";
        }
      : null;
    drawMap("A vs B: bright = same thing changed, green = same way, red = against", cf ? cf.ov : null, 1, cell);
  }
  ctx.textAlign = "left";
  ctx.fillStyle = "rgba(255,255,255,0.65)";
  ctx.font = "10px sans-serif";
  const wiredNote = wired(node) ? (map?.wired ? "bars from blocks_in (drag to edit: unplugs it) · " : "blocks_in wired (drag to edit: unplugs it) · ") : "";
  ctx.fillText(wiredNote + (map ? map.note : cfg.hint || "Run once to see the LoRA's layer strengths."), 2, H_TOTAL - 5, cw - 4);
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
  const N = nOf(node);
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
  const gw = node.addDOMWidget("lc_lora_graph", "lc_lora_graph", wrap, {
    serialize: false,
    getMinHeight: () => H_TOTAL,
    getMaxHeight: () => H_TOTAL,
  });
  // the option alone still saved an empty value: with it, the saved count matched the frontend's old-save cleanup
  // (widgets + the forceInput blocks_in) and it dropped a value, shifting lora_c / strength_c / lora_d / strength_d
  if (gw) gw.serialize = false;

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
    if (!hit) return;
    if (wired(node)) {
      // editing by hand takes over from blocks_in: keep its curve as the starting point, then unplug it
      const start = curveOf(node).slice();
      node.disconnectInput(node.inputs.findIndex((x) => x.name === "blocks_in"));
      setCurve(node, start);
    }
    ev.stopPropagation();
    ev.preventDefault();
    try {
      canvas.setPointerCapture(ev.pointerId);
    } catch {
      // no capture: the drag still works while the pointer stays on the graph
    }
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
    if (!hit || wired(node)) return;
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
      if (this.size[0] < MIN_W) this.setSize?.([MIN_W, this.size[1]]);
      // the first size is taken before the DOM widget counts: grow to fit once it does
      requestAnimationFrame(() => {
        const need = this.computeSize?.()?.[1];
        if (need && this.size[1] < need) this.setSize([this.size[0], need]);
      });
      return r;
    };
    // workflows saved with the graph's empty value: drop it before the frontend's cleanup miscounts and shifts values
    const configure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      const vals = info?.widgets_values;
      const gi = (this.widgets || []).findIndex((w) => w.name === "lc_lora_graph");
      if (Array.isArray(vals) && gi >= 0 && vals.length === this.widgets.length && vals[gi] === "") {
        info.widgets_values = vals.filter((_, i) => i !== gi);
      }
      return configure.apply(this, arguments);
    };
    // never narrower than MIN_W: below it the pickers cut off and the bars and maps get too thin to use
    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      if (size && size[0] < MIN_W) size[0] = MIN_W;
      return onResize?.apply(this, arguments);
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      if (this.size[0] < MIN_W) this.size[0] = MIN_W; // saved narrower before the lock
      const p = W(this, "preset");
      if (p && cfg.legacy?.[p.value]) p.value = cfg.legacy[p.value];
      // saved before the A vs B map existed: grow once so the new map is not cut off
      requestAnimationFrame(() => {
        const need = this.computeSize?.()?.[1];
        if (need && this.size[1] < need) this.setSize([this.size[0], need]);
      });
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
