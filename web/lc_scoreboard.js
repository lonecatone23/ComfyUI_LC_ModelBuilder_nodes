/**
 * LC Scoreboard: three buttons, "#1 Win" / "Tie" / "#2 Win". A win for one side is a loss for the other, so one
 * press scores both. Reset all clears the board; the leader and each side's record are drawn under the buttons.
 * The counts are the node's own INT widgets (wins_1, ties, wins_2), so they save with the workflow.
 */
import { app } from "../../scripts/app.js";

const TYPE = "LCScoreboard";
const BOX_H = 84;
const C1 = "#6fcf7f", C2 = "#e07a6a", CT = "#c9b26a";

const wget = (node, name) => (node.widgets || []).find((w) => w.name === name);
const num = (node, name) => Math.max(0, Math.round(Number(wget(node, name)?.value) || 0));

function add(node, name, by) {
  const w = wget(node, name);
  if (!w) return;
  w.value = by === 0 ? 0 : num(node, name) + by;
  w.callback?.(w.value);
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "LC.ModelBuilder.Scoreboard",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const labels = { wins_1: "#1 wins", ties: "ties", wins_2: "#2 wins" };
      for (const [k, l] of Object.entries(labels)) { const w = wget(this, k); if (w) w.label = l; }
      this.addWidget("button", "#1 Win", null, () => add(this, "wins_1", 1), { serialize: false });
      this.addWidget("button", "Tie", null, () => add(this, "ties", 1), { serialize: false });
      this.addWidget("button", "#2 Win", null, () => add(this, "wins_2", 1), { serialize: false });
      this.addWidget("button", "Reset all", null, () => ["wins_1", "ties", "wins_2"].forEach((k) => add(this, k, 0)), { serialize: false });
      this.setSize(this.computeSize());
      return r;
    };

    // room under the widgets for the leader box
    const computeSize = nodeType.prototype.computeSize;
    nodeType.prototype.computeSize = function () {
      const s = computeSize ? computeSize.apply(this, arguments) : [300, 200];
      return [Math.max(300, s[0]), s[1] + BOX_H];
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      const r = onDrawForeground?.apply(this, arguments);
      if (this.flags?.collapsed) return r;
      const w1 = num(this, "wins_1"), w2 = num(this, "wins_2"), t = num(this, "ties");
      const total = w1 + w2 + t;
      const pct = (n) => (total ? Math.round((100 * n) / total) : 0);
      const x0 = 12, w = this.size[0] - 24, y0 = this.size[1] - BOX_H + 6;

      ctx.save();
      ctx.fillStyle = "rgba(0,0,0,0.28)";
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x0, y0, w, BOX_H - 14, 6);
      else ctx.rect(x0, y0, w, BOX_H - 14);
      ctx.fill();

      ctx.textAlign = "center";
      ctx.font = "bold 16px Arial";
      if (!total) { ctx.fillStyle = "#aaa"; ctx.fillText("No results yet", x0 + w / 2, y0 + 24); }
      else if (w1 === w2) { ctx.fillStyle = "#ddd"; ctx.fillText(`Even (${w1} - ${w2})`, x0 + w / 2, y0 + 24); }
      else if (w1 > w2) { ctx.fillStyle = C1; ctx.fillText(`#1 leads: ${pct(w1)}%`, x0 + w / 2, y0 + 24); }
      else { ctx.fillStyle = C2; ctx.fillText(`#2 leads: ${pct(w2)}%`, x0 + w / 2, y0 + 24); }

      // share bar: #1 wins | ties | #2 wins
      const by = y0 + 34, bh = 6, bw = w - 16;
      let bx = x0 + 8;
      ctx.fillStyle = "rgba(255,255,255,0.08)";
      ctx.fillRect(bx, by, bw, bh);
      if (total) {
        for (const [n, c] of [[w1, C1], [t, CT], [w2, C2]]) { const seg = (bw * n) / total; ctx.fillStyle = c; ctx.fillRect(bx, by, seg, bh); bx += seg; }
      }
      // records: wins - losses - ties
      ctx.font = "12px Arial";
      ctx.fillStyle = C1; ctx.fillText(`#1  ${w1}-${w2}-${t}  (${pct(w1)}%)`, x0 + w * 0.27, by + 22);
      ctx.fillStyle = CT; ctx.fillText(`ties ${pct(t)}%`, x0 + w * 0.5, by + 40);
      ctx.fillStyle = C2; ctx.fillText(`#2  ${w2}-${w1}-${t}  (${pct(w2)}%)`, x0 + w * 0.73, by + 22);
      ctx.restore();
      return r;
    };
  },
});
