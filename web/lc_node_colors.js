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
    // the save nodes share the pack's brass, like the LoRA tools
    LCCheckpointSave: { color: "#6a5220", bgcolor: "#6a5220" },
    LCDiffusionModelSave: { color: "#6a5220", bgcolor: "#6a5220" },
    LCFaceVarietyScorer: { color: "#5a4a1a", bgcolor: "#5a4a1a" },
    LCKrea2LoRAEditor: { color: "#6a5220", bgcolor: "#6a5220" },
    LCLoRASave: { color: "#6a5220", bgcolor: "#6a5220" },
    LCKrea2LoRABlockScan: { color: "#6a5220", bgcolor: "#6a5220" },
    LCKrea2LoRAMerge: { color: "#6a5220", bgcolor: "#6a5220" },
    LCLoRAExtract: { color: "#6a5220", bgcolor: "#6a5220" },
    LCModelMergeAny: { color: "#5a4a1a", bgcolor: "#5a4a1a" },
    LCScoreboard: { color: "#5a4a1a", bgcolor: "#5a4a1a" },
    LCAnimaLoRAEditor: { color: "#6a5220", bgcolor: "#6a5220" },
    LCAnimaLoRABlockScan: { color: "#6a5220", bgcolor: "#6a5220" },
    LCAnimaLoRAMerge: { color: "#6a5220", bgcolor: "#6a5220" },
    LCAnimaBlockMergeAdvanced: { color: "#5a4a1a", bgcolor: "#5a4a1a" },
    LCMOSSBlockMergeAdvanced: { color: "#5a4a1a", bgcolor: "#5a4a1a" },
    LCModelQuantize: { color: "#6a5220", bgcolor: "#6a5220" },
    LCStripMetadata: { color: "#6a5220", bgcolor: "#6a5220" },
    // VAE nodes wear ComfyUI's standard red, like Load VAE
    LCVAEMerge: { color: "#322", bgcolor: "#533" },
    LCVAESave: { color: "#322", bgcolor: "#533" },
    LCVAECompare: { color: "#6a5220", bgcolor: "#6a5220" },
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
