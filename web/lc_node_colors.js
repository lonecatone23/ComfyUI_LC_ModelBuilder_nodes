/**
 * Default node color for LC ModelBuilder: every node in the pack wears the same deep blue (#191964), VAE nodes
 * included. Picked by pack, not by a list, so new nodes get it too. A color saved in a workflow still wins.
 */
import { app } from "../../scripts/app.js";
import "./lc_color.js"; // keeps a saved "No color" and palette colors on reload and paste

const PACK_COLOR = "#191964";
const isMine = (nodeData) => String(nodeData?.python_module || "").includes("ComfyUI_LC_ModelBuilder_nodes");

app.registerExtension({
    name: "LC.ModelBuilder.NodeColors",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isMine(nodeData)) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);
            this.color = PACK_COLOR;
            this.bgcolor = PACK_COLOR;
        };
    },
});
