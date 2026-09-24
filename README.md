# ComfyUI LC Modelbuilder Nodes

Model and checkpoint merging tools by lonecatone23. Companion pack to [ComfyUI_LC123_nodes](https://github.com/lonecatone23/ComfyUI_LC123_nodes) and [ComfyUI_LC_AV_nodes](https://github.com/lonecatone23/ComfyUI_LC_AV_nodes), kept separate so LC123 stays dependency-light. This pack is where merge, blend, and block-inject nodes live, plus the tools to test what a model actually does (like the same-face test in **LC Face Variety Scorer**).

## Nodes

| Node | Purpose |
|---|---|
| LC Krea2 Block Merge Advanced | Merge two Krea2 checkpoints with a draggable per-block ratio curve across the 28 uniform MMDiT blocks, plus individual sliders for the 10 named regions. Save/load named presets. |
| LC Krea2 Semantic Vector Tuner | Amplify or attenuate regions of a single Krea2 checkpoint's own weights. Not a merge — one model in, one model out. |
| LC Checkpoint Save | All-in-one checkpoint saver — model + clip + vae, all required. Adds an `enabled` toggle and a per-save `embed_workflow` toggle that core's stock Save Checkpoint node doesn't have. |
| LC Diffusion Model Save | Same as LC Checkpoint Save but model-only — no clip/vae sockets at all. |
| LC Face Variety Scorer 🧑‍🤝‍🧑 | Same-face test for a model. Scores how different the faces in an image batch are (0.0 - 1.0, lower = more variety), plus a group-shot score and a grid of the faces it found. |

### LC Krea2 Block Merge Advanced

Krea2 is a single-stream MMDiT: its 28 main blocks are architecturally uniform, so one continuous curve across them makes sense in a way it wouldn't for a double-stream model like Flux. The 10 named regions (`first`, `tmlp`, `txtmlp`, `tproj`, and the `txtfusion.*` group, plus `last`) are architecturally distinct, so they get their own sliders instead of living on the curve.

💡 The merge itself isn't custom tensor math. It builds the same prefix-to-ratio mapping ComfyUI's own `ModelMergeKrea2` node uses and hands it to ComfyUI's own `ModelMergeBlocks.merge()` (longest-prefix-match, `get_key_patches` / `add_patches`). This node is a UI on top of a mechanism ComfyUI core already ships and tests.

- **Curve drag**: click and drag anywhere in the graph. `edit_mode` is `smooth` (drag one block, neighbors blend in with a falloff set by `smooth_radius`) or `spike` (only the block under the cursor moves). Hover or drag to see a live tooltip with the zone, block number, and value. Zone dividers (low/mid/high) and a labeled 0.5 midline are drawn on the graph.
- **Presets**: pick a name from the `preset` combo to load it, or type a name into `save_name` and hit **Save preset** (writes on the next queue). A preset stores the curve and all 10 named sliders together. Touching any slider or the curve by hand snaps `preset` back to `Custom` immediately.
- **0 = all model1, 1 = all model2** at every block and every named region.

#### Bundled presets

`50_50` is the default on a fresh node.

| Preset | Idea |
|---|---|
| `50_50` | Flat 0.5 everywhere — every block and every named slider. A neutral, even blend to dial from. Default preset on launch. |
| `face_favor` | Smooth low→high ramp (0.15→0.75). Keeps early blocks and the input/timestep sliders close to model1 to anchor identity/structure, lets late blocks pick up more of model2. |
| `detail_focus` | Low and mid blocks stay flat and close to model1; only the high-zone blocks (and `last`) ramp hard toward model2 — a surgical late-block detail swap. |
| `style_favor` | The mirror of `face_favor`: high→low ramp (0.75→0.15). Early structure comes from model2, late refinement/output stays with model1. |
| `sine_wave` | Two full oscillations across the 28 blocks, centered on 0.5. A stress-test / exploratory curve rather than a targeted merge. |
| `resonance_blocks` | Spikes blocks 1, 3, 5, 7, 11, and 15 toward model2 (0.85), everything else stays low (0.3) — the "certain blocks dominate" pattern. 💡 I read "top model" as model2 (the one merged in); flip the spike/baseline values if you meant the opposite. |
| `even_blend` | Plain linear 0→1 ramp across all 28 blocks. The reference/baseline preset. |
| `core_focus` | Gaussian bump centered on the mid zone — pulls model2's "core" understanding into the middle of the network while leaving the edges (and the `txtfusion.*` sliders moderately) closer to model1. |
| `heavy_transplant_anchored` | Nearly all model2 (0.9) except block 0 and the `first`/`tproj` sliders, which stay anchored to model1 to avoid the merge drifting off-structure. |

### LC Krea2 Semantic Vector Tuner

Not a merge — this one takes a single checkpoint and turns specific regions up or down. Ported from the Arthemy SDXL Suite's `ArthemySDXLModelTuner` mechanism: it bakes any patches already on a weight, scales the baked tensor by a target multiplier, and re-patches the difference. Same proven patching approach, re-pointed at Krea2's real prefixes instead of SDXL's UNet layout.

- **`mode`**: `Soft Value` dampens the 0-2 sliders so nothing above 1.0 overdrives past roughly 1.13x — safe for exploring. `Real Value` applies the slider as a literal multiplier.
- **`low_blocks` / `mid_blocks` / `high_blocks`**: scale the same three zones (0-8 / 9-17 / 18-27) as Block Merge Advanced's graph. 💡 Krea2's 28 blocks are architecturally uniform — unlike SDXL's UNet, there's no verified evidence they specialize by depth, so this node doesn't invent named sub-groups the way Arthemy's SDXL tuner does. `blocks_override` (a 28-value comma list) is there if you want finer control than three zones.
- **10 named-region sliders**: same prefixes, same tooltips as Block Merge Advanced.
- **`base_scale`**: fallback multiplier for anything that isn't one of the 28 blocks or the 10 named regions.

### LC Checkpoint Save / LC Diffusion Model Save

A checkpoint means an all-in-one file, so these are two separate nodes rather than one node with optional sockets. Both reuse ComfyUI core's own `comfy.sd.save_checkpoint()` and the same modelspec-metadata detection core's stock "Save Checkpoint" node uses, so the files they write are the same format. Both share these two things core's node doesn't give you:

- **`enabled`**: off skips the save entirely, no file written. A save node has no outputs, so ComfyUI's native bypass has nothing to pass through — this is a real toggle you can convert to an input and wire from a switch elsewhere in the graph.
- **`embed_workflow`**: per-save control over whether the prompt/workflow JSON goes into the file's metadata, independent of the `--disable-metadata` launch flag core's node is stuck reading. 💡 Turn this off before sharing a model if you don't want your exact node graph attached to the file.

**LC Checkpoint Save**: `model`, `clip`, and `vae` are all required. `filename_prefix` defaults to `checkpoints/LC_ModelBuilder`.

**LC Diffusion Model Save**: only `model` — no clip/vae sockets at all. `filename_prefix` defaults to `diffusion_models/LC_ModelBuilder`. 💡 The file still carries the same `model.diffusion_model.` prefix a full checkpoint's UNet half uses, not a bare-key export — that's not a loading problem (`Load Diffusion Model` auto-strips known prefixes) but the file will be a little larger than a from-source `diffusion_models/` release.

### LC Face Variety Scorer 🧑‍🤝‍🧑

Catches "same face" in a model. Feed it a batch of generations (same prompt, different seeds works best, 8 or more images) and it finds the main face in each one, compares every face against every other face with a face recognition model, and averages it.

- **score**: 0.0 - 1.0, **lower = more variety**. It only looks at the face, not hair, clothes or lighting.

| Score | What it looks like |
|---|---|
| 0.0 - 0.2 | Clearly different people |
| 0.2 - 0.35 | Different people who share a few features |
| 0.35 - 0.5 | Could be sisters, or the same person in a different photo |
| 0.5 - 1.0 | Same person |

- **same_pct**: the percent of face pairs above `same_threshold` (0.35), i.e. "could be the same person."
- **crowd_score**: same scale, but compares the faces *inside* each image. For group shots, lower = the crowd is different people, not clones.
- **faces**: a grid of the faces it used, so you can see what it measured.
- **report**: all of the above as text, ready for a Show Text / Preview Any node.
- 💡 Pair it with **LC Image Batch From Folder** (LC123) to score a whole folder of test images. The example workflow is in `workflows/`.

**Face models (auto-download):** the first run downloads InsightFace's **buffalo_l** pack (~280 MB) into `ComfyUI/models/insightface/models/buffalo_l/`. If you already have it there (IPAdapter FaceID, ReActor, etc. use the same folder), nothing is downloaded. The download is checksum-verified, so a cut-off download never leaves broken files behind.

**No insightface, no opencv:** the models run straight through `onnxruntime`. The `insightface` Python package (the one that needs a C++ compiler and fights with numpy/protobuf) is **not** needed, and neither is opencv.

- **install.py** (run by ComfyUI-Manager) checks for a working onnxruntime. Already have one (any flavor)? It changes nothing. None at all? It installs the plain CPU `onnxruntime`. It never uninstalls or upgrades anything, and it saves a `pip freeze` backup to `install_backups/` before any change. If your onnxruntime is installed but broken, it prints the exact repair commands instead of guessing.
- **device**: `cpu` is the default and works on every install (NVIDIA, AMD, Intel, Mac). A normal test batch takes seconds. `auto` tries CUDA first and quietly falls back to CPU.

⚠️ The buffalo_l model files are released by InsightFace for **non-commercial research use only**. Check their license before using this commercially.

## Credits

- **InsightFace** by Jia Guo, Jiankang Deng and contributors: [github.com/deepinsight/insightface](https://github.com/deepinsight/insightface). The face detection decode and alignment in `lc_face_variety.py` follow InsightFace's own reference code (MIT).
- **ArcFace**: Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019.
- **SCRFD**: Guo et al., *Sample and Computation Redistribution for Efficient Face Detection*, 2021.

## Install

1. Clone into `ComfyUI/custom_nodes/`:
   ```
   git clone https://github.com/lonecatone23/ComfyUI_LC_ModelBuilder_nodes
   ```
2. Check the folder structure. `__init__.py` should sit directly in `ComfyUI/custom_nodes/ComfyUI_LC_ModelBuilder_nodes/`, not nested one level deeper.
3. Restart ComfyUI.
4. 💡 No `requirements.txt`. The merge and save nodes only use what ships with ComfyUI core. **LC Face Variety Scorer** needs `onnxruntime`, and `install.py` handles that for you (ComfyUI-Manager runs it on install and update). Installing by hand with `git clone`? Run `python install.py` once with ComfyUI's own python.
5. 💡 The **LC Face Variety Test** workflow in `workflows/` also uses **LC Image Batch From Folder** from [ComfyUI_LC123_nodes](https://github.com/lonecatone23/ComfyUI_LC123_nodes) (v1.40.0 or newer).
6. 💡 Hard-refresh your browser (`Ctrl+Shift+R`) after updating, so the new JS chrome actually loads.

---
"True Nothing is. Permitted Everything is"- Yoda Auditore, Assassin's Wars
