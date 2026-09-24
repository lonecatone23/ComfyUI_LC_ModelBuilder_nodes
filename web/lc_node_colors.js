/**
 * Default node colors for LC Modelbuilder Nodes.
 *
 * Brass/gold family, distinct from LC123 (teal) and LC AV (blue-teal) —
 * signals "model building" at a glance across the three packs.
 */
import { app } from "../../scripts/app.js";

const COLORS = {
    LCKrea2BlockMergeAdvanced: { color: "#5a4a1a", bgcolor: "#5a4a1a" },
    LCKrea2SemanticVectorTuner: { color: "#6a5220", bgcolor: "#6a5220" },
    // Not Krea2-specific -- same neutral shade for both save nodes since
    // they're siblings (same category, same mechanism, model-only vs AIO).
    LCCheckpointSave: { color: "#3a3a2a", bgcolor: "#3a3a2a" },
    LCDiffusionModelSave: { color: "#3a3a2a", bgcolor: "#3a3a2a" },
    LCFaceVarietyScorer: { color: "#5a4a1a", bgcolor: "#5a4a1a" },
};

app.registerExtension({
    name: "LC.ModelBuilder.NodeColors",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const cfg = COLORS[nodeData.name];
        if (!cfg) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);
            this.color = cfg.color;
            this.bgcolor = cfg.bgcolor;
            if (cfg.size) this.size = cfg.size.slice();
        };
    },
});
