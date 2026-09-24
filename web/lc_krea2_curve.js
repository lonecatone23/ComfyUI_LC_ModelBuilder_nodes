import { app } from "../../scripts/app.js";

const NUM_BLOCKS = 28;
// Mirrors _ZONE_BOUNDS in lc_krea2_block_merge.py — low 0-8, mid 9-17, high 18-27.
const ZONE_LOW_END = Math.floor(NUM_BLOCKS / 3);
const ZONE_MID_END = Math.floor((NUM_BLOCKS * 2) / 3);
const NAMED_KEYS = [
    "first", "tmlp", "txtmlp", "tproj",
    "txtfusion_layer0", "txtfusion_layer1", "txtfusion_projector",
    "txtfusion_refiner0", "txtfusion_refiner1", "last",
];
const PRESET_ROOTS = [
    "/extensions/ComfyUI_LC_ModelBuilder_nodes/krea2_block_presets/",
    "/extensions/comfyui_lc_modelbuilder_nodes/krea2_block_presets/",
];
// Mirrors _BUILTIN_PRESET_NAMES in lc_krea2_block_merge.py — kept first, in this order.
const BUILTIN_PRESETS = [
    "50_50", "face_favor", "detail_focus", "style_favor", "sine_wave",
    "resonance_blocks", "even_blend", "core_focus", "heavy_transplant_anchored",
];

function zoneLabel(i) {
    if (i < ZONE_LOW_END) return "Low block";
    if (i < ZONE_MID_END) return "Mid block";
    return "High block";
}

function parseCurve(text) {
    if (!text) return [];
    return String(text)
        .replace(/[\[\]]/g, " ")
        .split(/[,\s]+/)
        .map((p) => p.trim())
        .filter(Boolean)
        .map(Number)
        .filter((n) => Number.isFinite(n));
}

function formatCurve(vals) {
    return vals.map((v) => {
        const n = Number(v);
        if (!Number.isFinite(n)) return "0";
        return n.toFixed(6).replace(/\.?0+$/, "");
    }).join(", ");
}

function lerpCurve(vals, newLen) {
    if (newLen < 1) return [];
    if (vals.length < 2) return new Array(newLen).fill(vals[0] ?? 0.5);
    const out = [];
    for (let i = 0; i < newLen; i++) {
        const x = i / Math.max(newLen - 1, 1);
        const pos = (vals.length - 1) * x;
        const idx = Math.floor(pos);
        const frac = pos - idx;
        if (idx >= vals.length - 1) out.push(vals[vals.length - 1]);
        else out.push((1 - frac) * vals[idx] + frac * vals[idx + 1]);
    }
    return out;
}

function widget(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

function widgetVal(node, name, fallback) {
    const w = widget(node, name);
    if (!w) return fallback;
    return w.value;
}

function setWidget(node, name, value) {
    const w = widget(node, name);
    if (!w) return;
    w.value = value;
}

function markCustom(node) {
    const p = widget(node, "preset");
    if (p && p.value !== "Custom") p.value = "Custom";
}

function blockVals(node) {
    let vals = parseCurve(widgetVal(node, "blocks_curve", ""));
    if (vals.length !== NUM_BLOCKS) {
        vals = vals.length >= 2 ? lerpCurve(vals, NUM_BLOCKS) : new Array(NUM_BLOCKS).fill(0.5);
    }
    return vals;
}

app.registerExtension({
    name: "LCModelBuilder.Krea2BlockMerge",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LCKrea2BlockMergeAdvanced") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);
            this.title = "LC Krea2 Block Merge Advanced";
            this.lcDrag = -1;
            this.lcGraphH = 160;
            this.lcHover = -1;
            this.lcHoverPos = null;

            const self = this;

            const sn = widget(this, "save_name");
            if (sn && (sn.value == null || String(sn.value) === "null")) sn.value = "";
            const sc = widget(this, "save_curve");
            if (sc && sc.value == null) sc.value = false;

            this.widgets = (this.widgets || []).filter((w) => w && w.type !== "button");

            if (!this.widgets.some((w) => w && w.type === "button")) {
                const btn = this.addWidget("button", "Save preset", "save", () => {
                    const nameW = widget(self, "save_name");
                    if (nameW && !String(nameW.value || "").trim()) nameW.value = "custom_preset";
                    setWidget(self, "save_curve", true);
                    const nm = nameW ? String(nameW.value).trim() : "";
                    if (nm && typeof setPresetList === "function") setPresetList([nm]);
                });
                if (btn) btn.serialize = false;
            }

            const presetCombo = widget(this, "preset");
            const nodeRef = this;

            function existingSaved() {
                const cur = (presetCombo && presetCombo.options && presetCombo.options.values) || [];
                return cur.filter((n) => n && n !== "Custom" && !String(n).startsWith("─"));
            }
            function setPresetList(names) {
                if (!presetCombo || !presetCombo.options) return;
                const extra = {};
                existingSaved().forEach((n) => { extra[n] = true; });
                (names || []).forEach((n) => {
                    if (n && n !== "index" && n !== "Custom") extra[n] = true;
                });
                const all = Object.keys(extra);
                const builtins = BUILTIN_PRESETS.filter((n) => all.includes(n));
                const custom = all.filter((n) => !BUILTIN_PRESETS.includes(n)).sort();
                const vals = [...builtins];
                if (custom.length) {
                    if (vals.length) vals.push("────────");
                    vals.push(...custom);
                }
                if (vals.length) vals.push("────────");
                vals.push("Custom");
                presetCombo.options.values = vals;
                try {
                    const spec = nodeRef.constructor.nodeData.input.required.preset;
                    if (Array.isArray(spec)) spec[0] = vals;
                } catch (e) { /* ignore */ }
            }

            if (presetCombo) {
                const prev = presetCombo.callback;
                presetCombo.callback = function (v) {
                    if (prev) prev.apply(this, arguments);
                    if (!v || String(v).startsWith("─") || v === "Custom") {
                        self.setDirtyCanvas(true, true);
                        return;
                    }
                    for (const root of PRESET_ROOTS) {
                        fetch(root + encodeURIComponent(v) + ".json" + "?t=" + Date.now())
                            .then((r) => (r.ok ? r.json() : null))
                            .then((body) => {
                                if (!body) return;
                                const pts = parseCurve(body.blocks);
                                if (pts.length >= 2) {
                                    setWidget(self, "blocks_curve", formatCurve(lerpCurve(pts, NUM_BLOCKS)));
                                }
                                const named = body.named || {};
                                NAMED_KEYS.forEach((k) => {
                                    if (typeof named[k] === "number") setWidget(self, k, named[k]);
                                });
                                self.setDirtyCanvas(true, true);
                            })
                            .catch(() => {});
                    }
                };
            }

            // Any manual edit of a merge parameter (named slider or the curve string
            // itself) flips preset back to Custom — loading a preset uses setWidget()
            // directly and never touches .callback, so this never fires on preset load.
            ["blocks_curve", ...NAMED_KEYS].forEach((name) => {
                const w = widget(self, name);
                if (!w) return;
                const prev = w.callback;
                w.callback = function (v) {
                    if (prev) prev.apply(this, arguments);
                    markCustom(self);
                    self.setDirtyCanvas(true, true);
                };
            });

            (async () => {
                for (const root of PRESET_ROOTS) {
                    try {
                        const res = await fetch(root + "index.json?t=" + Date.now());
                        if (!res.ok) continue;
                        const data = await res.json();
                        setPresetList((data && data.presets) || []);
                        nodeRef.setDirtyCanvas(true, true);
                        return;
                    } catch (e) { /* next root */ }
                }
            })();
        };

        const GRAPH_H = 160;
        const _computeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function () {
            const base = _computeSize ? _computeSize.apply(this, arguments) : [320, 260];
            return [Math.max(320, base[0] || 320), (base[1] || 260) + GRAPH_H + 22];
        };
        nodeType.prototype.lcGraphRect = function () {
            const ws = this.widgets || [];
            let top = LiteGraph.NODE_TITLE_HEIGHT + 8;
            for (let i = 0; i < ws.length; i++) {
                if (ws[i] && ws[i].last_y != null) top = Math.max(top, ws[i].last_y + 34);
            }
            const h = Math.max(GRAPH_H, this.size[1] - top - 10);
            return { x: 28, y: top, w: Math.max(40, this.size[0] - 40), h };
        };
        nodeType.prototype.lcEndDrag = function () {
            this.lcDrag = -1;
            this.lcDragOrig = null;
            if (this._lcUp) {
                window.removeEventListener("pointerup", this._lcUp, true);
                window.removeEventListener("mouseup", this._lcUp, true);
                this._lcUp = null;
            }
        };
        nodeType.prototype.lcIndexAt = function (r, pos, vals) {
            const n = Math.max(vals.length - 1, 1);
            return Math.max(0, Math.min(vals.length - 1, Math.round(((pos[0] - r.x) / r.w) * n)));
        };
        nodeType.prototype.lcClearHoverSoon = function () {
            const node = this;
            clearTimeout(this._lcHoverTimer);
            this._lcHoverTimer = setTimeout(() => {
                if (node.lcDrag < 0) {
                    node.lcHover = -1;
                    node.setDirtyCanvas(true, true);
                }
            }, 150);
        };
        nodeType.prototype.onMouseDown = function (e, pos) {
            const r = this.lcGraphRect();
            if (pos[0] < r.x || pos[0] > r.x + r.w || pos[1] < r.y || pos[1] > r.y + r.h) return false;
            const vals = blockVals(this);
            this.lcDrag = this.lcIndexAt(r, pos, vals);
            this.lcDragOrig = vals.slice();
            this.lcHover = this.lcDrag;
            this.lcHoverPos = pos.slice();
            const node = this;
            if (this._lcUp) {
                window.removeEventListener("pointerup", this._lcUp, true);
                window.removeEventListener("mouseup", this._lcUp, true);
            }
            this._lcUp = function () {
                node.lcEndDrag();
                node.setDirtyCanvas(true, true);
            };
            window.addEventListener("pointerup", this._lcUp, true);
            window.addEventListener("mouseup", this._lcUp, true);
            return true;
        };
        nodeType.prototype.onMouseMove = function (e, pos) {
            const r = this.lcGraphRect();
            const inside = pos[0] >= r.x && pos[0] <= r.x + r.w && pos[1] >= r.y && pos[1] <= r.y + r.h;

            if (this.lcDrag < 0) {
                if (inside) {
                    const vals = blockVals(this);
                    this.lcHover = this.lcIndexAt(r, pos, vals);
                    this.lcHoverPos = pos.slice();
                    this.lcClearHoverSoon();
                    this.setDirtyCanvas(true, true);
                } else if (this.lcHover >= 0) {
                    this.lcHover = -1;
                    this.setDirtyCanvas(true, true);
                }
                return false;
            }
            if (e && typeof e.buttons === "number" && e.buttons === 0) {
                this.lcEndDrag();
                return false;
            }
            const vals = this.lcDragOrig ? this.lcDragOrig.slice() : blockVals(this);
            let ny = Math.max(0, Math.min(1, 1 - (pos[1] - r.y) / r.h));
            const mode = String(widgetVal(this, "edit_mode", "smooth"));
            const radius = Math.max(0, Number(widgetVal(this, "smooth_radius", 2)) || 0);
            const k = this.lcDrag;
            if (mode === "spike" || radius <= 0) {
                vals[k] = ny;
            } else {
                for (let i = 0; i < vals.length; i++) {
                    const d = (i - k) / Math.max(radius, 0.001);
                    const wgt = Math.exp(-0.5 * d * d);
                    vals[i] = (1 - wgt) * vals[i] + wgt * ny;
                }
            }
            setWidget(this, "blocks_curve", formatCurve(vals.map((v) => Math.max(0, Math.min(1, v)))));
            markCustom(this);
            this.lcHover = k;
            this.lcHoverPos = pos.slice();
            this.setDirtyCanvas(true, true);
            return true;
        };
        nodeType.prototype.onMouseUp = function () {
            this.lcEndDrag();
            this.lcClearHoverSoon();
            return false;
        };
        const onDrawFg = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (onDrawFg) onDrawFg.apply(this, arguments);
            if (this.flags && this.flags.collapsed) return;
            const r = this.lcGraphRect();
            const vals = blockVals(this);
            if (vals.length < 2) return;
            const n = vals.length - 1;
            ctx.save();
            ctx.fillStyle = "rgba(0,0,0,0.28)";
            ctx.fillRect(r.x, r.y, r.w, r.h);
            ctx.strokeStyle = "rgba(255,255,255,0.25)";
            ctx.strokeRect(r.x, r.y, r.w, r.h);
            ctx.beginPath();
            ctx.rect(r.x, r.y, r.w, r.h);
            ctx.clip();
            // horizontal gridlines: faint quarter lines, brighter labeled midline
            ctx.strokeStyle = "rgba(255,255,255,0.08)";
            ctx.lineWidth = 1;
            [0.25, 0.75].forEach((frac) => {
                const y = r.y + r.h * (1 - frac);
                ctx.beginPath();
                ctx.moveTo(r.x, y);
                ctx.lineTo(r.x + r.w, y);
                ctx.stroke();
            });
            const midY = r.y + r.h * 0.5;
            ctx.strokeStyle = "rgba(255,255,255,0.22)";
            ctx.beginPath();
            ctx.moveTo(r.x, midY);
            ctx.lineTo(r.x + r.w, midY);
            ctx.stroke();

            // low/mid/high zone dividers
            ctx.strokeStyle = "rgba(255,255,255,0.18)";
            ctx.setLineDash([3, 3]);
            [ZONE_LOW_END - 0.5, ZONE_MID_END - 0.5].forEach((idx) => {
                const x = r.x + (idx / n) * r.w;
                ctx.beginPath();
                ctx.moveTo(x, r.y);
                ctx.lineTo(x, r.y + r.h);
                ctx.stroke();
            });
            ctx.setLineDash([]);

            ctx.strokeStyle = "#d4a44f";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            vals.forEach((v, i) => {
                const px = r.x + (i / n) * r.w;
                const py = r.y + (1 - v) * r.h;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            });
            ctx.stroke();
            const activeIdx = this.lcDrag >= 0 ? this.lcDrag : this.lcHover;
            ctx.fillStyle = "#f0d9a8";
            vals.forEach((v, i) => {
                const px = r.x + (i / n) * r.w;
                const py = r.y + (1 - v) * r.h;
                const active = i === activeIdx;
                if (active) {
                    ctx.save();
                    ctx.strokeStyle = "rgba(255,255,255,0.35)";
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(px, r.y);
                    ctx.lineTo(px, r.y + r.h);
                    ctx.stroke();
                    ctx.restore();
                }
                ctx.beginPath();
                ctx.arc(px, py, active ? 4.5 : 2.2, 0, Math.PI * 2);
                ctx.fill();
            });
            ctx.restore();

            // axis + zone labels
            ctx.fillStyle = "rgba(255,255,255,0.55)";
            ctx.font = "10px sans-serif";
            ctx.fillText("model2  1", 4, r.y + 10);
            ctx.fillText("0.5", 4, r.y + r.h * 0.5 + 3);
            ctx.fillText("model1  0", 4, r.y + r.h - 2);

            ctx.font = "9px sans-serif";
            ctx.fillStyle = "rgba(255,255,255,0.4)";
            ctx.textAlign = "center";
            const zoneY = r.y - 3;
            ctx.fillText("LOW", r.x + ((ZONE_LOW_END - 1) / 2 / n) * r.w, zoneY);
            ctx.fillText("MID", r.x + ((ZONE_LOW_END + ZONE_MID_END - 1) / 2 / n) * r.w, zoneY);
            ctx.fillText("HIGH", r.x + ((ZONE_MID_END + n) / 2 / n) * r.w, zoneY);
            ctx.textAlign = "left";

            // hover/drag tooltip
            if (activeIdx >= 0 && this.lcHoverPos) {
                const v = vals[activeIdx];
                const lines = [zoneLabel(activeIdx), `Block ${activeIdx}`, `Value: ${v.toFixed(3)}`];
                ctx.save();
                ctx.font = "10px sans-serif";
                const padding = 6;
                const lineH = 13;
                const boxW = Math.max(...lines.map((l) => ctx.measureText(l).width)) + padding * 2;
                const boxH = lines.length * lineH + padding * 2 - 3;
                let tx = this.lcHoverPos[0] + 14;
                let ty = this.lcHoverPos[1] - boxH - 10;
                // keep the tooltip inside the graph rect so it never sits under the widget rows above
                if (tx + boxW > r.x + r.w) tx = this.lcHoverPos[0] - boxW - 14;
                if (tx < r.x) tx = r.x + 2;
                if (ty < r.y) ty = this.lcHoverPos[1] + 14;
                if (ty + boxH > r.y + r.h) ty = r.y + r.h - boxH - 2;
                ctx.fillStyle = "rgba(20,18,10,0.92)";
                ctx.strokeStyle = "rgba(255,255,255,0.25)";
                ctx.lineWidth = 1;
                ctx.beginPath();
                if (ctx.roundRect) ctx.roundRect(tx, ty, boxW, boxH, 4);
                else ctx.rect(tx, ty, boxW, boxH);
                ctx.fill();
                ctx.stroke();
                ctx.font = "bold 10px sans-serif";
                ctx.fillStyle = "#f0d9a8";
                ctx.fillText(lines[0], tx + padding, ty + padding + 9);
                ctx.font = "10px sans-serif";
                ctx.fillStyle = "rgba(255,255,255,0.85)";
                ctx.fillText(lines[1], tx + padding, ty + padding + 9 + lineH);
                ctx.fillText(lines[2], tx + padding, ty + padding + 9 + lineH * 2);
                ctx.restore();
            }
        };
    },
});
