# ComfyUI LC Modelbuilder Nodes

Model and checkpoint merging tools by lonecatone23. Companion pack to [ComfyUI_LC123_nodes](https://github.com/lonecatone23/ComfyUI_LC123_nodes) and [ComfyUI_LC_AV_nodes](https://github.com/lonecatone23/ComfyUI_LC_AV_nodes), kept separate so LC123 stays dependency-light. This pack is where merge, blend, and block-inject nodes live, plus the tools to test what a model actually does (like the same-face test in **LC Face Variety Scorer**).

## Nodes

| Node | Purpose |
|---|---|
| LC Krea2 Block Merge Advanced | Merge two Krea2 checkpoints with a draggable per-block ratio curve across the 28 uniform MMDiT blocks, plus individual sliders for the 10 named regions. Save/load named presets. |
| LC Anima Block Merge Advanced | The same for Anima: a curve over its 28 blocks, plus sliders for the x embedder, timestep embedder and norm, LLM adapter and final layer. |
| LC Krea2 Semantic Vector Tuner | Amplify or attenuate regions of a single Krea2 checkpoint's own weights. Not a merge — one model in, one model out. |
| LC Checkpoint Save | All-in-one checkpoint saver — model + clip + vae, all required. Adds an `enabled` toggle and a per-save `embed_workflow` toggle that core's stock Save Checkpoint node doesn't have. |
| LC Diffusion Model Save | Same as LC Checkpoint Save but model-only — no clip/vae sockets at all. |
| LC Face Variety Scorer 🧑‍🤝‍🧑 | Same-face test for a model. Scores how different the faces in an image batch are (0.0 - 1.0, lower = more variety), plus a group-shot score and a grid of the faces it found. Photo (ArcFace) or anime (CCIP) mode. |
| LC Krea2 LoRA Editor | 🧪 **BETA, still in testing.** Load a Krea 2 LoRA with each of the 28 blocks turned up or down, attention and MLP scaled separately, and spikes softened. Patches the model like a LoRA loader and outputs the edited LoRA. For face lock and for LoRAs that change the whole image. |
| LC LoRA Save | 🧪 **BETA, still in testing.** Saves an edited LoRA as a standard file with every edit baked in, optionally at a new rank. |
| LC Krea2 LoRA Block Scan | 🧪 **BETA, still in testing.** Finds where a LoRA locks the face or changes the whole image: renders a seed batch with each block group turned off, scores every row, and suggests a block curve for the Editor. |
| LC Krea2 LoRA Merge | 🧪 **BETA, still in testing.** Merges two to four Krea 2 LoRAs with a per-block A / B curve (sum or TIES), applies the result like a LoRA loader and outputs it for LC LoRA Save. |
| LC Model Merge (any precision) | Merges two models stored in different precisions (bf16 with fp8, int8, nvfp4 or GGUF, in either order). The result stays full precision. |
| LC Model Quantize 🗜️ | Saves a smaller copy of a model: fp8, mxfp8, nvfp4, int8 or int4 (load with the stock loader) or GGUF Q8_0 to Q2_K. From the wired MODEL (LoRAs and merges baked in) or straight from a file of any size. |
| LC VAE Merge | Merges the decoders of two VAEs of the same family, coarse shapes to fine texture. The encoder stays the base's, so the result still fits the models made for it. |
| LC VAE Save | Saves a VAE to models/vae in fp32, bf16 or fp16. |
| LC VAE Compare | Decodes the same latents with up to six VAEs, side by side with a 1:1 detail crop, a sharpness number and how far each lands from the first. |
| LC LoRA Extract | Turns a finetune into a LoRA: finetuned minus its base, all layers, blocks only or a custom block list. Unchanged layers are left out of the file. |
| LC Anima LoRA Editor / Block Scan / Merge | 🧪 **BETA, still in testing.** The three Krea 2 LoRA tools for Anima: same controls, Anima's 28 blocks and layer types. The Block Scan scores anime faces with CCIP. |
| LC Scoreboard 🏆 | Head-to-head counter for A / B tests: #1 Win / Tie / #2 Win buttons, a reset, and the leader and percentages shown on the node. |

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

### LC Anima Block Merge Advanced

The Krea2 Block Merge Advanced for Anima: same curve, drag modes and preset saving, with Anima's own named sliders (**x_embedder**, **t_embedder**, **t_embedding_norm**, **llm_adapter**, **final_layer**). Core's Cosmos merge node has no LLM adapter slider; this one does.

- **0 = all model1, 1 = all model2**, per block and per named slider.
- Bundled presets: **50_50**, **model2_early** (blocks 0-6 from model2), **model2_look** (blocks 7-20, where an Anima finetune's look sits), **model2_late** (21-27). Your own save to `web/anima_block_presets/`.

### LC Krea2 Semantic Vector Tuner

Not a merge — this one takes a single checkpoint and turns specific regions up or down. Ported from the Arthemy SDXL Suite's `ArthemySDXLModelTuner` mechanism: it bakes any patches already on a weight, scales the baked tensor by a target multiplier, and re-patches the difference. Same proven patching approach, re-pointed at Krea2's real prefixes instead of SDXL's UNet layout.

- **`mode`**: `Soft Value` dampens the 0-2 sliders so nothing above 1.0 overdrives past roughly 1.13x — safe for exploring. `Real Value` applies the slider as a literal multiplier.
- **`low_blocks` / `mid_blocks` / `high_blocks`**: scale the same three zones (0-8 / 9-17 / 18-27) as Block Merge Advanced's graph. 💡 Krea2's 28 blocks are architecturally uniform — unlike SDXL's UNet, there's no verified evidence they specialize by depth, so this node doesn't invent named sub-groups the way Arthemy's SDXL tuner does. `blocks_override` (a 28-value comma list) is there if you want finer control than three zones.
- **10 named-region sliders**: same prefixes, same tooltips as Block Merge Advanced.
- **`base_scale`**: fallback multiplier for anything that isn't one of the 28 blocks or the 10 named regions.

### LC Krea2 LoRA Editor / LC LoRA Save

**⚠️🧪 BETA: still in testing**

- The four LoRA nodes (**Editor**, **LoRA Save**, **Block Scan** and **Merge**) work, but they are still being tested on more LoRAs.
- Presets, defaults and settings can change between versions. A saved workflow may need its bars set again after an update.
- Test every edit on a fixed seed before you save, and always keep your original LoRA files.
- Found something odd? Open an issue on this repo with the LoRA name and your settings.

Edit a LoRA instead of retraining it. The Editor sits where a LoRA loader would (model in, model out, into your sampler), so you can test every edit on a fixed seed before you save anything.

- **Block bars**: 28 bars, 0 to 2 (1 = unchanged), on a tall graph so small changes are easy. Drag or sweep across them; double-click or shift-click a bar to reset it to 1. The grey columns behind the bars show how strong the LoRA is in each block.
- **`preset`**: each block group preset turns its blocks down to 0.3: **Early blocks: structure (0-3)**, **Face blocks (4-9)**, **Mid blocks: face and body (10-15)**, **Late face blocks (16-21)**, **Finish: color and skin (22-27)**. Face lock sits in different blocks for each LoRA (Better Lip Bite: late face blocks, Krea Amateur V4: face blocks), so try each on the same seed batch and keep the one that frees the face. **Tame whole image** sets blocks 4-15 to 0.7. Workflows saved with the old `Blocks X-Y down` names load the matching new preset.
- **`attention` / `mlp` / `outside_blocks`**: scale the attention layers, the MLP layers, or anything outside the 28 blocks (text fusion) in one go.
- **`spike_soften`**: `layers` pulls a layer that is far stronger than the same layer in the other blocks down to `spike_limit` x their median. `directions` flattens the one direction that dominates a layer to `spike_limit` x the next one (well-behaved LoRAs sit around 2). `directions` works on plain lora_A / lora_B files.
- **`blocks_in`**: wire a 28-block curve here (for example the Block Scan's `blocks` output) and it replaces the preset and the bars. Dragging a bar takes over: it unplugs `blocks_in` and keeps its curve as your starting point.
- **Heatmap**: after a run, 8 layer rows x 28 blocks, before and after on the same scale, plus a note (layers applied, rank, how much strength is left).

Tested on Better Lip Bite, which both locks the face and changes the whole image (LPIPS against the base model, person / background, 4 seeds; same-face score over 8 seeds):

| Edit | Same face | Person change | Background change |
|---|---|---|---|
| Base model, no LoRA | 0.445 | | |
| LoRA as is | 0.510 | 0.571 | 0.564 |
| Late face blocks (16-21) | 0.423 | 0.518 | 0.501 |
| Blocks 16-21 at 0.3, 4-15 at 0.7, directions at 2 | 0.464 | 0.419 | 0.262 |

💡 For comparison, the well-behaved Braces LoRA changes the background by 0.260, so the last row brings Lip Bite down to that level.

**LC Krea2 LoRA Block Scan** does the block hunting for you. It renders the same seed batch with no LoRA, with the full LoRA, and with the LoRA turned off in one block group at a time (`groups`, default `0-3, 4-9, 10-15, 16-21, 22-27, fusion`). Then it scores every row: the same-face score (the LC Face Variety Scorer's face models, so it needs onnxruntime), and how far each row moved from the base model and from the full LoRA. You get a labeled grid, a report, and a suggested curve (`blocks`) that sets the group that frees the face most to 0.3. A group only counts when it lowers the same-face score by 0.03 or more, since 8 seeds wobble by about 0.02. Use a prompt with one clear face. Each row is one batch, so 6 groups at 8 images is 64 renders.

**LC Krea2 LoRA Merge** combines two to four LoRAs into one. The bars set, per block, how much of each you get: **0 = only A**, **0.5 = both at full strength**, **1 = only B**. Layers outside the blocks follow the bars' average.

- **method**: **sum** adds both, exactly (loading the saved merge gives the same image as the node). **ties** keeps each LoRA's strongest changes (**ties_density**) and drops the ones where the two pull against each other: for two styles or two characters that clash. Slower.
- **mix**: one slider over the whole merge, like a plain model merge: **0 = only A**, **1 = only B**. At 0.5 the bars apply as set.
- **preset**: Both full, A early / B late, B early / A late, A only, B only, and **Face from B** (8-20, 4-13, 16-21). Dragging a bar switches to Custom.
- **Face from B**: put the main LoRA (body, style) in A and the face LoRA in B. B gets the window, A the rest. Where a LoRA keeps its face changes from LoRA to LoRA, so these are starting points: move the bars, or wire **LC Krea2 LoRA Block Scan**'s curve into **blocks_in**.
- **lora_c / lora_d** (optional): a third and fourth LoRA at their own strength in every block (not on the bars).
- Heatmaps: LoRA A's share, LoRA B's share and the merged result, per layer and block.
- 💡 Two LoRAs stacked in two loaders and the same two merged with **sum** give the same picture, but not bit for bit: one combined patch rounds slightly differently from two separate ones.

**LC LoRA Save** writes the edited LoRA to `models/loras` (`%source%` in the filename = the original LoRA's name, a number is added if the file exists) with every edit and the strength baked in, so it loads at 1.0 in any LoRA loader. Saving and reloading gives the same image as the Editor. `rank` 0 keeps each layer's rank; any other number re-fits every layer to that rank. LoRAs in other formats (LoKr, LoHa) are edited per layer and converted to lora_A / lora_B when saved. The edit settings go into the file's metadata. Same **enabled** and **embed_workflow** switches as LC Checkpoint Save (see below): off skips the save, and embed_workflow controls whether your graph goes into the file.

### LC Anima LoRA Editor / LC Anima LoRA Block Scan / LC Anima LoRA Merge

🧪 **BETA, still in testing.** The same three tools for Anima LoRAs, and LC LoRA Save saves their output too. Everything above applies; what differs:

- **Layer rows**: Anima blocks have self attention (`sa`), cross attention to the text (`ca`), an MLP and the timestep modulation (`adaln`). **attention** scales `sa` and `ca`, **mlp** the MLP, and **outside_blocks** the LLM adapter and the final layer.
- **Presets**: **Early blocks (0-6)**, **Mid blocks: look (7-13)**, **Mid blocks: look and character (14-20)**, **Late blocks: finish (21-27)**, each at 0.3. On the Animosity Anima LoRA, turning off 7-13 or 14-20 changed the picture most, 0-6 and 21-27 far less.
- **Block Scan** scores face lock with **CCIP**, a model made for anime characters (photo face models miss on anime), on each image's main face. The score is the share of seed pairs CCIP calls the same character: about 0.1 to 0.2 for the base model, 1.0 for a LoRA that draws one person every time. Noise is about 0.15 at 8 seeds. The models download once to `models/ccip/` (about 200 MB, checksum-verified).
- Defaults follow Anima: 30 steps, cfg 4.5, er_sde, 896 x 1152. Wire a real negative.
- ⚠️ On a LoRA that bakes one face into every layer (like a style LoRA trained on one person), no single block group frees it. Turn several down a little, or remove the face LoRA itself with LoRA Merge at a negative strength.

### LC Model Merge (any precision)

Core's merge nodes break when the two models are stored differently. A bf16 **model1** with an fp8 **model2** fails (`'Linear' object has no attribute 'weight_scale'`), and an fp8 **model1** works but rounds the whole result back down to fp8.

- The quantized model (fp8, int8, nvfp4, GGUF) is decoded to full precision first, and the full-precision model becomes the base, so the result stays bf16 in either order.
- **ratio**: 0 = all model1, 1 = all model2.
- **block_ratios** (optional): one value per numbered block, comma list, block 0 first (Krea 2 has 28). Layers outside the blocks use **ratio**.
- **report**: what each model was stored as, which one became the base and what the result is.
- *note:* two quantized models still merge, but the result is rounded back to model1's format (the report says so).
- Tested on Krea 2: Krealism fp8 + Turbo bf16 lands within normal fp8 rounding of the same merge done in pure bf16.

### LC Model Quantize 🗜️

Makes the fp8 / GGUF / nvfp4 version of a model right inside ComfyUI, from the file you already have or from whatever is on the MODEL wire (a merge, a LoRA baked in).

- **source**: **(use the model input)** takes the wired MODEL, LoRAs and merges included. Or pick a file from `diffusion_models` or `checkpoints` (an all-in-one checkpoint gives just its diffusion model).
  - *note:* the wired model has to fit in RAM in bf16 (about 2 GB per billion parameters: fine up to 12-14B on 64 GB). A file is read one layer at a time, so even Flux 2 dev works on 32 GB.
- **format**:

Tested on Krealism V3.1 (12B, bf16 = 25.6 GB), straight from the file, 25 to 55 seconds each on an RTX 5090:

| Format | Size | Drift | Load with | Runs fast on |
|---|---|---|---|---|
| fp8 (scaled) | 13.1 GB | 2.2 % | Load Diffusion Model | 40 / 50 series |
| mxfp8 | about 13 GB | about 2 % | Load Diffusion Model | 50 series |
| int8 (convrot) | 13.2 GB | 1.0 % | Load Diffusion Model | any card with int8 (30 series too) |
| nvfp4 | 8.2 GB | 9.1 % | Load Diffusion Model | 50 series |
| int4 (convrot w4a4) | about 7 GB | about 15 % | Load Diffusion Model | any card with int8 |
| GGUF Q8_0 | 13.9 GB | 0.6 % | Unet Loader (GGUF) | any |
| GGUF Q5_K_M | 9.5 GB | 3.8 % | Unet Loader (GGUF) | any |
| GGUF Q4_K_M | 8.1 GB | 7.2 % | Unet Loader (GGUF) | any |
| GGUF Q3_K_M | 6.5 GB | 15.1 % | Unet Loader (GGUF) | any |
| GGUF Q2_K | 5.2 GB | 28.7 % | Unet Loader (GGUF) | any |

**Things to notice:**
- **Everything renders** on a 12B model, Q2_K included: the layout shifts more as you go down, but nothing breaks.
- **int8 and Q8_0** come out the closest to bf16. fp8 and Q5_K_M are next.
- **nvfp4** made here lands a little closer to bf16 than the older Krealism nvfp4 release.
- ⚠️ **Small models are fragile:** on Anima (2B), Q3 is rough and Q2_K is noise. The report warns below 6B.

- **Drift** = how far the quantized weights sit from the originals, on average, relative to their size (0 % = identical, lower = closer). It's a quick check before rendering, not a quality score: fp8 at 2 % and Q8_0 at 0.6 % both look the same as bf16.
- **_M / _S** (GGUF): _M keeps the first and last two blocks one type higher, _S doesn't. _M is the usual pick.
- **protect_edges**: keeps the first and last block at full precision too (a little bigger, a little closer).
- **filename**: saved in `models/diffusion_models`. Empty = the source name plus the format. Subfolders work (`Krea2/Krealism`). It never overwrites: a number is added.
- Embedders, norms, the global modulation and the final layer always stay full precision, like released quants.
- **nvfp4 / int4** also round the activations to 4-bit, and the layers fed by the timestep (adaLN / modulation) and the text side can't take that (Anima came out as pure noise). Those stay full precision in these two formats.
- **Memory:** from a file it used about 2 GB of system RAM on the 12B model (plus a few GB of GPU scratch). From the MODEL wire the model itself is already in RAM.
- The wired model is not changed: the node works on a copy, one layer at a time on the GPU, and writes straight to disk. It checks the free disk space first.
- ⚠️ Quantize from **bf16 / fp16**. An fp8 or GGUF source gets decoded first, but its rounding stacks with the new one. The report warns you.
- 💡 GGUF needs [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) to load.

### LC VAE Merge / LC VAE Save / LC VAE Compare

A diffusion model is trained on one VAE's latents, so only the **decoder** half of a VAE is free to change. These three merge decoders, save the result and compare VAEs on the same latents.

**LC VAE Merge**
- **vae_a** is the base: its encoder is always kept, so the merged VAE still fits every model trained on it. **vae_b** gives its decoder.
- Five sliders, coarse to fine (0 = all vae_a, 1 = all vae_b):
  - **start**: decoder input and middle (overall tone, colour)
  - **coarse** / **medium** / **fine**: the upsampling stages
  - **finest**: the last stage and the output layer (texture, sharpness, the last colour mapping)
- **report**: how far apart the two decoders are per region, and whether the encoders match.
- ⚠️ Only merge VAEs that sit close: **under about 10 %** in the report. A decoder that was retrained instead of finetuned can't be blended (over 15 % the node warns), you get oversaturated colour or noise.
- 💡 On Wan / Qwen-Image style VAEs (Krea 2, Anima) **ramp** the sliders instead of jumping (e.g. 0, 0.25, 0.5, 0.75, 1). Flux VAEs take hard jumps fine.
- Diffusers-style files (up_blocks / mid_block) merge with original-layout ones: ComfyUI's loader puts them in the same layout first.

**Which of your VAEs merge** (all of them share the same encoder per family):

| Family | Close pairs (merge) | Far apart (don't blend) |
|---|---|---|
| Flux 2 | flux2-vae + flux2UltrahdRAWVAE (1 %) | |
| Krea 2 / Anima (Wan lineage) | wan21-vae + krea2RealVae (3-5 %) | |
| Krea 2 / Anima (Qwen-Image lineage) | qwen_image_vae + hdrVAE (0.1 %) | either of these with the Wan pair (30-70 %) |
| Flux 1 | ae + ultrafluxVAEImproved (1-14 %) | their output layers (94 %): keep **finest** at 0 or 1 |

**LC VAE Save**
- Saves to `models/vae`, ready for Load VAE. **fp32** (the safest for a VAE), **bf16** or **fp16** (half the size). It never overwrites: a number is added.

**LC VAE Compare**
- Wire **samples** from a real render (the KSampler output) and up to six VAEs. Every VAE decodes those same latents, which is what you actually see.
- **sheet**: one column per VAE, the full picture on top and a 1:1 crop from its middle below (**crop_size**). **labels** names the columns.
- **Sharpness** = average edge strength (higher = crisper). **Drift** = average difference from the first VAE, 0-255 scale (0 = identical, under 2 = hard to see, over 6 = visible).

### LC LoRA Extract

Pulls a finetune out into a LoRA: **finetuned** minus **base**, compressed per layer. This is how the Krealism Turbo and Raw LoRAs were made.

- **parts**: **all layers** (the whole finetune), **blocks only** (on Krea 2 this leaves out text fusion and the embedding layers, the version that works on Krea 2 Raw) or **custom blocks** (e.g. `14-27` or `4-9, 16, 20-22`).
- **rank**: the highest rank any layer may use. **keep**: how much of each layer's change to keep (0.90 = 90%). Krealism at 256 / 0.90 matched the full model on Turbo.
- **min_change**: layers that changed less than this are left out of the file, so it only holds what the finetune really changed.
- Saved under `models/loras` (a number is added instead of overwriting), same key layout as ComfyUI's own LoRA saver, so any loader reads it.
- **enabled** / **embed_workflow**: same switches as the save nodes. Off skips the extraction entirely. 💡 Turn embed_workflow off before sharing the LoRA if you don't want your graph attached.
- ⚠️ Extract from **bf16** versions of both models. A quantized model adds its rounding to every layer: the file comes out bigger and holds noise. The report warns you.

### LC Scoreboard 🏆

Keeps score while you compare two models, LoRAs or settings side by side.

- **#1 Win**, **Tie**, **#2 Win**: one press scores both sides (a win for #1 is a loss for #2). **Reset all** clears the board.
- The node shows who leads and by how much (e.g. **#1 leads: 60%**), a share bar, and each side's win-loss-tie record. Equal wins show **Even**.
- The score saves with the workflow. No need to run the graph; the **wins_1 / ties / wins_2 / summary** outputs are there if you want to log it.

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
- **face_model = anime (CCIP)** for anime and illustration models (Anima, Illustrious, Pony), where photo face models miss. It finds each image's main face with an anime face detector, then CCIP scores the share of image pairs that show the same character: an anime base model sits around 0.1 to 0.25, a locked face near 1.0. The two models download once to `models/ccip/` (about 200 MB, checksum-verified). crowd_score is not used in this mode.

**Face models (auto-download):** the first run downloads InsightFace's **buffalo_l** pack (~280 MB) into `ComfyUI/models/insightface/models/buffalo_l/`. If you already have it there (IPAdapter FaceID, ReActor, etc. use the same folder), nothing is downloaded. The download is checksum-verified, so a cut-off download never leaves broken files behind.

**No insightface, no opencv:** the models run straight through `onnxruntime`. The `insightface` Python package (the one that needs a C++ compiler and fights with numpy/protobuf) is **not** needed, and neither is opencv.

- **install.py** (run by ComfyUI-Manager) checks for a working onnxruntime. Already have one (any flavor)? It changes nothing. None at all? It installs the plain CPU `onnxruntime`. It never uninstalls or upgrades anything, and it saves a `pip freeze` backup to `install_backups/` before any change. If your onnxruntime is installed but broken, it prints the exact repair commands instead of guessing.
- **device**: `cpu` is the default and works on every install (NVIDIA, AMD, Intel, Mac). A normal test batch takes seconds. `auto` tries CUDA first and quietly falls back to CPU.

⚠️ The buffalo_l model files are released by InsightFace for **non-commercial research use only**. Check their license before using this commercially.

## Credits

- **InsightFace** by Jia Guo, Jiankang Deng and contributors: [github.com/deepinsight/insightface](https://github.com/deepinsight/insightface). The face detection decode and alignment in `lc_face_variety.py` follow InsightFace's own reference code (MIT).
- **CCIP** by deepghs (OpenRAIL): [huggingface.co/deepghs/ccip_onnx](https://huggingface.co/deepghs/ccip_onnx), and the **anime face detector** by deepghs (MIT): [huggingface.co/deepghs/anime_face_detection](https://huggingface.co/deepghs/anime_face_detection). Used by the Anima Block Scan and the Face Variety Scorer's anime mode.
- **llama.cpp** (ggml, MIT): [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The GGUF K-quant quantizers in `lc_gguf_quants.py` are ported from its reference code (`ggml-quants.c`).
- **ComfyUI_DiffusionModel_fp8_converter** by Shiba-2-shiba: [github.com/Shiba-2-shiba/ComfyUI_DiffusionModel_fp8_converter](https://github.com/Shiba-2-shiba/ComfyUI_DiffusionModel_fp8_converter), the idea of converting inside ComfyUI.
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
   - **LC Model Merge Any Precision Test**, **LC LoRA Extract Test**, **LoRA Test** and **Model test** (A / B tests with **LC Scoreboard 🏆**) are in `workflows/` too. They use LC123 nodes, and **LoRA Test** also uses LC Vision.
6. 💡 Hard-refresh your browser (`Ctrl+Shift+R`) after updating, so the new JS chrome actually loads.

---
"True Nothing is. Permitted Everything is"- Yoda Auditore, Assassin's Wars
