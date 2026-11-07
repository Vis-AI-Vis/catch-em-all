# Topic: Getting started

- **Date:** 2026-11-04
- **Model(s) Used:** Online: Gemini Argon - Thinking Medium
- **User:** K1427462
- **Overall Objective:** Getting started and make some first analyses

---

## Interaction 1: Phase I Dataset Strategy & Batch Inference Pipeline Generation

### 0. Model
- **Model:** Online: Gemini Argon - Thinking Medium

### 1. Prompt / Input
- **Context:** Empty start but included all necessary information regarding the project; it used the Web; Prompt
- **Prompt:**
> Provided full assignment brief (Vis4AI Phase I & II requirements, AI4Vis pair comparison rules, Nov 11 midpoint submission deliverables), project description ("CATCH EM ALL"), baseline single-image PyTorch inference code for `imzynoxprince/pokemons-image-classifier-gen1-gen9`, and asked:
> 1. Whether to use a different dataset or stick with the AM-2k 1,800 foreground images (`./fg` folder) used by the non-AI pair.
> 2. How best to get started analyzing the model in Phase I and whether anything important was overlooked.
> 3. Requested a structured GitHub AI interaction log template filled out for every turn.

### 2. AI Output
- **Output / Summary:** 
  1. **Dataset & Alignment Recommendation:** Advised keeping the AM-2k dataset to preserve the required shared baseline with the human-only pair for the AI4Vis comparison, while extending it via synthetic background/spurious-feature interventions (white vs. black vs. high-saturation/edge-filtered versions of `./fg`).
  2. **Identified Blind Spots:** Pointed out that naive `.convert("RGB")` on transparent PNGs in `./fg` creates inconsistent black/white background artifacts, which acts as an uncontrolled spurious feature since Pokémon artwork is typically trained on solid white/transparent backgrounds.
  3. **Batch Inference & Visual Feature Code:** Provided a complete PyTorch batch-processing script over `./fg` that composites alpha channels cleanly, computes model uncertainty metrics (`entropy`, `normalized_entropy`, `top1_conf`, `conf_margin_1_2`), extracts black-box input image statistics (`mean_saturation`, `mean_brightness`, `contrast_std`, `edge_density`, `fg_coverage`), and saves the structured output to `phase1_fg_analysis.csv`.
  4. **Visualization Concepts:** Proposed three Phase I visualizations beyond aggregate tables: a Confidence–Entropy Phase Space scatter plot, an "Attractor Pokémon" bipartite flow diagram, and a Paired Intervention Shift plot comparing spurious modifications.

### 3. Reflection
- **Strengths:** Caught issues with background maybe making difference (black vs white); 
- **Limitations:** long overarching output: understanding and veryfiyng the output can take a long time; many assumptions
- **Impact:** Got a massive head-start because part of the output was also the code to test the AIs assumptions

---

## Interaction 2: Analyzing Initial Phase I CSV, Recovering AM-2k Categories & Designing Distance/Confusion Visualizations

### 0. Model
- **Model:** Online: Gemini Argon - Thinking Medium

### 1. Prompt / Input
- **Context:** We ran the batch inference script on 1,767 foreground images from the AM-2k dataset (`./fg` folder) and generated `phase1_fg_analysis.csv`. We observed that certain Pokémon (such as `Bouffalant`) trigger extreme confidence (~98.8%) with very low entropy (`0.1191`), while entropy rises steeply within the first ~30–40 ranked images. However, the AM-2k filenames are hashes (`m_00308c60.png`), so we lack the ground-truth real-world animal categories needed to build an animal-to-Pokémon confusion/distance matrix and bubble plots sized by entropy.
- **Prompt:**
> Uploaded `phase1_fg_analysis.csv` and asked:
> 1. Whether ground-truth animal category labels exist for the AM-2k dataset.
> 2. How to construct a distance/confusion matrix showing which real animals are confused with which Pokémon, encoding entropy/normalized entropy via point size.
> 3. What initial insights and next steps emerge from our `phase1_fg_analysis.csv` output for Phase I.


### 2. AI Output
- **Output / Summary:**
  1. **Exploratory Diagnosis of `phase1_fg_analysis.csv`:**
     - **The Entropy Cliff:** Confirmed that out of 1,767 OOD animal images, only 76 images (4.30%) achieve $\text{Top-1 Confidence} \ge 50\%$. Entropy starts at `0.1191` (`normalized_entropy = 0.0172`) and crosses `normalized_entropy = 0.30` by rank 39.
     - **Two Distinct OOD Attractor Regimes:** Identified a sharp split between **Hyper-Confident Traps** (`Bouffalant` with 45.1% mean confidence and 23 high-confidence catches; `Sawsbuck` with 39.0% mean confidence; `Mightyena` with 30.9% mean confidence) and **High-Entropy Default Sinks** (`Linoone` with 200 predictions at 15.7% mean confidence, and `Muddy Goomy` with 197 predictions at 9.9% mean confidence and zero high-confidence catches).
     - **Preprocessing Bug Identified:** Noticed `fg_coverage` was `1.0` across all rows and `mean_brightness` averaged `0.170` because AM-2k `./fg` images are saved as RGB with solid black `(0,0,0)` backgrounds rather than RGBA alpha channels.
  2. **Recovering Official AM-2k Categories:** Identified the exact 20 animal categories from the official AM-2k paper (Li et al., IJCV 2022: `alpaca`, `antelope`, `bear`, `camel`, `cat`, `cattle`, `deer`, `dog`, `elephant`, `giraffe`, `horse`, `kangaroo`, `leopard`, `lion`, `monkey`, `rabbit`, `rhinoceros`, `sheep`, `tiger`, `zebra`) and wrote a zero-shot CLIP script constrained to those 20 classes to auto-label all 1,767 images and extract pairwise cosine distance embeddings.
  3. **Visualizations & Code Provided:** Generated a 4-panel diagnostic figure (`phase1_initial_diagnosis.png`) directly from the CSV and provided a script to produce:
     - A **Bipartite Bubble Confusion Matrix** (`20 Animals × Top 15 Pokémon`) where bubble size encodes assignment percentage and color encodes mean normalized entropy.
     - A **2D Cosine-Distance t-SNE Map** where color encodes the true animal species, point size encodes certainty ($1 - H_{\text{norm}}$), and high-confidence Pokémon traps are annotated.


### 3. Reflection
- **Strengths:** extremely good visualization and anaylsis of csv; fast code generation (vibe coding), found categories of used dataset
- **Limitations:** code is not always correct; strange assumptions to correct; lots of idesa which need to be kept in line
- **Impact:** created first visualizations, much faster than alone, suggested new visulization idea not included in my prompt

---

## Interaction 3: Model collapse

### 0. Model
- **Model:** Online: Gemini Argon - Thinking Medium

### 1. Prompt / Input
- **Context:** Entie context from previous 2 tasks
- **Prompt:**
> some minor correction sin code and strategy --> model broke down

---

## Interaction 4: New Chat: Refining t-SNE Visualization & Adding OOD Network Graphs

### 0. Model
- **Model:** Gemini 4 Online  (Thinking)

### 1. Prompt / Input
- **Context:** The Phase 1 Python script generated a 2D t-SNE plot of 1,767 AM-2k animal images mapped to Pokémon classes, but the high-confidence text labels overlapped heavily. We needed to fix the t-SNE layout and evaluate whether adding network graphs (Pokémon-only vs. Animal-to-Pokémon) made analytical sense.
- **Prompt:**
> [Provided Phase 1 CLIP + t-SNE Python script]
> Can we improve the Phase 1 t-SNE plot since the labels are overlapping and cluttered? Also, would it make sense to add a network graph analysis—either connecting Pokémon that frequently share high-confidence images, or including the animal categories as well?

### 2. AI Output
- **Output / Summary:** Provided drop-in Python (`matplotlib`/`networkx`) code with three updates:
  1. **Cleaned 2D t-SNE Plot:** Placed animal category names at cluster medians (with white text halos) and de-duplicated Pokémon trap callouts (max 1 per cluster), spacing them radially around the outer margin with arrows to prevent overlap.
  2. **Bipartite Attractor Graph (Animals ↔ Pokémon):** Connected real animal nodes to Pokémon attractor nodes, scaling edge width by the percentage of an animal routed to that Pokémon and color by mean confidence.
  3. **Pokémon Co-Confusion Graph (Pokémon ↔ Pokémon):** Connected Pokémon that compete on the same images/visual neighborhoods, using edge width for shared image frequency and red edge labels to show the bridging animal.

### 3. Reflection
- **Strengths:** [strengths if any, compare if necessary]
- **Limitations:** [Issues /Limitation/ Problems]
- **Impact:** [What did we do with it?]