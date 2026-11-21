/**
 * LC Block Scan viewer: the scan's renders at full size. The node's grid tiles are 192 px; after a run the
 * "🔍 View full size" button opens a full-window gallery (one row per render set, labeled), click a picture to see it
 * at full resolution. Wheel = zoom at the cursor, drag = pan, double-click = 100% / fit, ←/→ = the other seeds,
 * ↑/↓ = the same seed in the other rows (no LoRA, full LoRA, each group off), Esc = back / close.
 * The pictures live in ComfyUI's temp folder, so they last until ComfyUI restarts.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TYPES = new Set(["LCKrea2LoRABlockScan", "LCAnimaLoRABlockScan", "LCMOSSLoRABlockScan"]);
const url = (im) =>
  api.apiURL(`/view?filename=${encodeURIComponent(im.filename)}&subfolder=${encodeURIComponent(im.subfolder || "")}` +
    `&type=${encodeURIComponent(im.type || "temp")}`);

const CSS = `
.lcbsv{position:fixed;inset:0;z-index:2147483000;background:rgba(12,11,9,.97);color:#ddd;font:13px/1.35 system-ui,sans-serif;
  display:flex;flex-direction:column;user-select:none}
.lcbsv-top{display:flex;align-items:center;gap:12px;padding:8px 14px;background:#1d1a14;border-bottom:1px solid #3a3326}
.lcbsv-top b{color:#f0d9a8}.lcbsv-top .sp{flex:1}.lcbsv-top .k{color:#999}
.lcbsv-x{cursor:pointer;font-size:20px;padding:0 6px;color:#ccc}.lcbsv-x:hover{color:#fff}
.lcbsv-gal{flex:1;overflow:auto;padding:12px 14px}
.lcbsv-row{display:flex;gap:6px;margin-bottom:8px;align-items:flex-start}
.lcbsv-lab{width:190px;flex:none;padding-top:4px}.lcbsv-lab b{color:#f0d9a8;display:block;font-size:14px;margin-bottom:3px}
.lcbsv-lab span{display:block;color:#bbb}
.lcbsv-row img{width:var(--t);height:auto;cursor:zoom-in;border:2px solid transparent;border-radius:3px;background:#222}
.lcbsv-row img:hover{border-color:#c9a45a}
.lcbsv-one{flex:1;position:relative;overflow:hidden;cursor:grab}.lcbsv-one.drag{cursor:grabbing}
.lcbsv-one img{position:absolute;left:0;top:0;transform-origin:0 0;image-rendering:auto;max-width:none}
.lcbsv-one.px img{image-rendering:pixelated}
`;

let styled = false;
function style() {
  if (styled) return;
  styled = true;
  const s = document.createElement("style");
  s.textContent = CSS;
  document.head.appendChild(s);
}

function open(scan) {
  style();
  const rows = scan.rows || [];
  if (!rows.length) return;
  const n = Math.max(...rows.map((r) => r.images.length));
  const root = document.createElement("div");
  root.className = "lcbsv";
  root.tabIndex = -1;
  document.body.appendChild(root);

  const top = document.createElement("div");
  top.className = "lcbsv-top";
  const body = document.createElement("div");
  body.style.cssText = "flex:1;display:flex;min-height:0";
  root.append(top, body);

  let cur = null; // {r, i} in the single view, null in the gallery
  let view = null; // {s, x, y} zoom and pan of the single view

  const close = () => {
    window.removeEventListener("keydown", onKey, true);
    root.remove();
  };

  function header(html) {
    top.innerHTML = `${html}<span class="sp"></span><span class="lcbsv-x" title="Close (Esc)">✕</span>`;
    top.querySelector(".lcbsv-x").onclick = close;
  }

  function gallery() {
    cur = null;
    header(`<b>LC Block Scan</b><span>${scan.title || ""}</span>` +
      `<span class="k">Click a picture for full size</span>` +
      (scan.mb != null ? `<span class="k" title="${scan.temp_dir || ""}">${scan.mb} MB in ComfyUI's temp folder, cleared on restart</span>` : ""));
    body.innerHTML = "";
    const gal = document.createElement("div");
    gal.className = "lcbsv-gal";
    const avail = Math.max(120, (window.innerWidth - 230 - 6 * n) / n);
    gal.style.setProperty("--t", `${Math.floor(Math.min(320, avail))}px`);
    rows.forEach((row, r) => {
      const el = document.createElement("div");
      el.className = "lcbsv-row";
      el.innerHTML = `<div class="lcbsv-lab"><b>${row.label}</b>${(row.info || []).map((t) => `<span>${t}</span>`).join("")}</div>`;
      row.images.forEach((im, i) => {
        const img = document.createElement("img");
        img.loading = "lazy";
        img.src = url(im);
        img.title = `${row.label}, image ${i + 1}`;
        img.onclick = () => single(r, i);
        el.appendChild(img);
      });
      gal.appendChild(el);
    });
    body.appendChild(gal);
  }

  function single(r, i) {
    const row = rows[r];
    const im = row?.images[i];
    if (!im) return;
    const keep = cur && view ? { ...view } : null; // ↑/↓ and ←/→ keep the zoom and the spot
    cur = { r, i };
    header(`<b>${row.label}</b><span>${(row.info || []).join(" · ")}</span>` +
      `<span class="k">image ${i + 1}/${row.images.length} · row ${r + 1}/${rows.length}</span>` +
      `<span class="k" data-size></span>` +
      `<span class="k">wheel zoom · drag pan · double-click 100% / fit · ←/→ seed · ↑/↓ row · Esc back</span>`);
    body.innerHTML = "";
    const box = document.createElement("div");
    box.className = "lcbsv-one";
    const img = document.createElement("img");
    box.appendChild(img);
    body.appendChild(box);

    const fit = () => {
      const s = Math.min(box.clientWidth / img.naturalWidth, box.clientHeight / img.naturalHeight, 1);
      return { s, x: (box.clientWidth - img.naturalWidth * s) / 2, y: (box.clientHeight - img.naturalHeight * s) / 2 };
    };
    const apply = () => {
      img.style.transform = `translate(${view.x}px,${view.y}px) scale(${view.s})`;
      box.classList.toggle("px", view.s > 1.5);
      const z = top.querySelector("[data-size]");
      if (z) z.textContent = `${img.naturalWidth}×${img.naturalHeight} · ${Math.round(view.s * 100)}%`;
    };
    img.onload = () => {
      view = keep && keep.w === img.naturalWidth ? keep : fit();
      view.w = img.naturalWidth;
      apply();
    };
    img.src = url(im);

    box.onwheel = (e) => {
      e.preventDefault();
      if (!view) return;
      const b = box.getBoundingClientRect();
      const mx = e.clientX - b.left, my = e.clientY - b.top;
      const s = Math.min(16, Math.max(0.05, view.s * (e.deltaY < 0 ? 1.2 : 1 / 1.2)));
      view.x = mx - ((mx - view.x) * s) / view.s;
      view.y = my - ((my - view.y) * s) / view.s;
      view.s = s;
      apply();
    };
    box.ondblclick = (e) => {
      if (!view) return;
      const f = fit();
      if (Math.abs(view.s - 1) < 0.01) {
        view = { ...f, w: view.w };
      } else {
        const b = box.getBoundingClientRect();
        const mx = e.clientX - b.left, my = e.clientY - b.top;
        view.x = mx - (mx - view.x) / view.s;
        view.y = my - (my - view.y) / view.s;
        view.s = 1;
      }
      apply();
    };
    box.onpointerdown = (e) => {
      if (!view || e.button !== 0) return;
      const sx = e.clientX - view.x, sy = e.clientY - view.y;
      box.classList.add("drag");
      box.setPointerCapture(e.pointerId);
      box.onpointermove = (m) => {
        view.x = m.clientX - sx;
        view.y = m.clientY - sy;
        apply();
      };
      box.onpointerup = () => {
        box.classList.remove("drag");
        box.onpointermove = box.onpointerup = null;
      };
    };
  }

  function onKey(e) {
    if (e.key === "Escape") {
      if (cur) gallery();
      else close();
    } else if (cur && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) {
      const { r, i } = cur;
      if (e.key === "ArrowLeft") single(r, (i - 1 + rows[r].images.length) % rows[r].images.length);
      if (e.key === "ArrowRight") single(r, (i + 1) % rows[r].images.length);
      if (e.key === "ArrowUp") single((r - 1 + rows.length) % rows.length, i);
      if (e.key === "ArrowDown") single((r + 1) % rows.length, i);
    } else {
      return;
    }
    e.preventDefault();
    e.stopPropagation();
  }
  window.addEventListener("keydown", onKey, true);
  gallery();
  root.focus();
}

app.registerExtension({
  name: "LC.ModelBuilder.BlockScanViewer",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TYPES.has(nodeData?.name)) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const w = this.addWidget("button", "🔍 View full size", null, () => {
        if (this.__lcScan) open(this.__lcScan);
        else app.extensionManager?.toast?.add?.({ severity: "info", summary: "LC Block Scan",
          detail: "Run the scan first: the full-size pictures appear after a run.", life: 4000 });
      }, { serialize: false });
      w.serialize = false;
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted?.apply(this, arguments);
      const scan = message?.lc_scan?.[0];
      if (scan) this.__lcScan = scan;
      return r;
    };
  },
});
