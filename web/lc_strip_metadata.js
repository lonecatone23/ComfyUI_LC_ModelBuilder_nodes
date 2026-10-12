/** LC Strip Metadata 🧹: shows what the last run removed, in a read-only box on the node. */
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
  name: "LCModelBuilder.StripMetadata",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "LCStripMetadata") return;
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      onExecuted?.apply(this, arguments);
      const text = Array.isArray(msg?.text) ? msg.text.join("\n") : msg?.text;
      if (text == null) return;
      let w = this.widgets?.find((x) => x.name === "lc_report");
      if (!w) {
        w = ComfyWidgets["STRING"](this, "lc_report", ["STRING", { multiline: true }], app).widget;
        w.serialize = false;
        w.options = { ...(w.options || {}), serialize: false };
        if (w.inputEl) w.inputEl.readOnly = true;
        this.setSize([Math.max(this.size[0], 380), Math.max(this.size[1], 220)]);
      }
      w.value = String(text);
      if (w.inputEl) w.inputEl.value = String(text);
      this.graph?.setDirtyCanvas?.(true, true);
    };
  },
});
