# Topic: Other patterns

- **Date:** 2026.11.07 & 08
- **Model(s) Used:** Gemini 4 Thinking Medium - Online, Gemini 3 Flash - Thinking Medium Online
- **User:** k1427462
- **Overall Objective:** What leads to high confidence in animals? Are there patterns in saturation eedges etc across categories?

---

## Interaction 1: Getting Started: Automated Discovery of High-Confidence Visual Triggers

### 0. Model
- **Model:** Gemini 4 Thinking Medium - Online

### 1. Prompt / Input
- **Context:** After identifying which animal categories map to which Pokémon attractors, we wanted to start a new mini-phase investigating *why* specific individual images trigger high-confidence (>85%) predictions while others of the same animal fall into low-confidence sinks. We needed an automated pipeline to test continuous image features (brightness, saturation, animal coverage, pose, silhouette complexity) and generate side-by-side image thumbnail drill-downs as shown in Lecture Slide 2.
- **Prompt:**
> Let's start a new mini-phase to automatically investigate why specific images trigger >85% confidence. Can we write a standalone Python script that correlates continuous image characteristics (like brightness, color saturation, foreground coverage, full-body vs. close-up pose, and pattern complexity) against model confidence, and automatically generates side-by-side image thumbnail comparisons of the top highest-confidence vs. lowest-confidence images within the same animal class?

### 2. AI Output
- **Output / Summary:** Created a standalone Python script (`Phase1_DeepDive_Confidence_Triggers.py`) that automates feature extraction, statistical correlation testing, and item-level visualization:
  1. **Automated Multi-Feature Extraction (Pixel + CLIP Zero-Shot Probing):** Extracted 8 continuous metrics per image: low-level geometry/color stats (`fg_brightness_only`, `fg_saturation_only`, `fg_coverage`, internal luminance `fg_contrast`, Sobel `edge_complexity`, and `bbox_aspect_ratio`) plus high-level CLIP zero-shot semantic scores (`clip_full_body_score` for full-body side profile vs. cropped face, and `clip_pattern_score` for horns/antlers/stripes vs. plain fur).
  2. **Statistical Ranking & Continuous Scatter Dashboard (`phase1_feature_correlations_dashboard.png`):** Computed Spearman rank correlations ($r$, $p$-values) and Cohen's $d$ effect sizes between >85% high-confidence traps and <40% low-confidence sinks, plotting a 6-panel scatter and decile-trendline dashboard.
  3. **Feature Shift Violin & Effect-Size Plot (`phase1_high_vs_low_conf_distributions.png`):** Visualized standardized Z-score distribution shifts showing which visual properties most strongly separate high-confidence traps from uncertain predictions.
  4. **Automated Item-Level Thumbnail Grid (`phase1_item_drilldown_thumbnails.png`):** Implemented the Slide 2 drill-down workflow by automatically selecting the top animal classes and rendering an 8-column thumbnail grid comparing the 4 highest-confidence images (>85% traps) against the 4 lowest-confidence images (<40% sinks) with their corresponding brightness, saturation, and pose scores.

### 3. Reflection
- **Strengths:** immediately created code with an AI model (openai/clip-vit-base-patch32) which is good at analyzing what I wanted to analyze
- **Limitations:** As always too much info difficult to work through code, A lot has to checked over (i found nothing) and then we have to doublecheck if teh outcome can be correct etc. 
- **Impact:** Used to generate complex code for otherwise very complex task. 

---

## Interaction 2: Deep-Dive Statistical & Visual Pattern Analysis of Enriched AM-2k Features

### 0. Model
- **Model:** Gemini 4 Thinking Medium - Online

### 1. Prompt / Input
- **Context:** After generating `phase1_deepdive_features.csv` and `phase1_fg_with_animals.csv` (containing 27 low-level visual metrics, CLIP semantic pose/pattern scores, and model predictions for all 1,767 AM-2k images), we wanted a comprehensive statistical analysis and publication-quality visualizations uncovering the hidden behavioral patterns in the data.
- **Prompt:**
> [Uploaded `phase1_deepdive_features.csv` and `phase1_fg_with_animals.csv`]
> Analyze these files in detail and give me patterns, knowledge, and visualizations back. Find all patterns that could interest us in the data and create informative, well-designed visualizations based on them.

### 2. AI Output
- **Output / Summary:** Conducted a full statistical analysis of the 1,767 images and generated 4 multi-panel visualizations revealing four core behavioral findings:
  1. **Bimodal OOD Regime & Logit Co-Activation (`vis1_bimodal_attractor_taxonomy.png`):** Showed that 93.2% of images fall into low-confidence (<40%) generic fallback sinks (*Linoone*, *Muddy Goomy*, *Mabosstiff* with <19% class purity and ~0.75–0.80 entropy), while 90% of extreme traps (≥85% confidence) concentrate in three high-purity taxonomic attractors (`cattle` → *Bouffalant*, `cat` → *Mightyena*, `antelope` → *Sawsbuck*). Verified that Top-1 and Top-2 predictions co-activate biologically homologous Pokémon pairs (e.g., *Bouffalant* → *Tauros*, *Sawsbuck* → *Stantler*, *Mightyena* → *Poochyena*).
  2. **Simpson's Paradox in Visual Feature Triggers (`vis2_simpsons_paradox_feature_reversals.png`):** Discovered that global correlations look weak ($|r| < 0.15$) because opposite visual properties trigger different Pokémon classes: *Bouffalant* is triggered by close-up cattle heads/horns ($r = -0.28$ with full-body pose, $r = +0.28$ with frame coverage), whereas *Sawsbuck/Stantler* require full-body standing profiles with lower frame coverage ($r = +0.27$ pose, $r = -0.33$ coverage); similarly, *Mightyena* requires desaturated black/grey fur ($r = -0.30$ saturation) while *Bouffalant* requires warm brown saturation.
  3. **Projection Hotspot Landscape vs. Big-Cat Dead Zone (`vis3_projection_hotspot_comparison.png`):** Mapped 2D KDE confidence contours over the t-SNE manifold, showing that high-confidence predictions cluster in 3 morphological islands while heavy mammals and big cats (`tiger`, `lion`, `leopard`, `elephant`) form a "dead zone" (mean confidence 9.8%–11.4%, 0% ≥50% for tigers/lions).
  4. **Standardized Archetype Fingerprints (`vis4_visual_trigger_archetypes.png`):** Built a parallel Z-score coordinate plot and Cohen's $d$ effect-size chart (`clip_pattern_score` $d = +0.78$, `fg_saturation_only` $d = +0.53$, `clip_full_body_score` $d = -0.48$) profiling the exact multi-feature signatures of the four OOD trap archetypes.

### 3. Reflection
- **Strengths:** extremely good visualizations, and analysis
- **Limitations:** Generation code is unreadable to me, cannot verify by this if anything went wrong
- **Impact:** very good pattern analysis and visualizations to keep

---


## Interaction 3: Designing the Input Perturbation Study (`Original` vs. `FG` vs. `Mask`)

### 0. Model
- **Model:** Gemini 4 Thinking Medium - Online

### 1. Prompt / Input
- **Context:** After completing our deep-dive statistical analysis and 4 visualizations on the `fg` (black background) dataset, we wanted to plan the final Phase 1 experiment: comparing model predictions and confidence shifts across all three AM-2k input modalities (`original` with natural background, `fg` on black background, and `mask` binary silhouettes).
- **Prompt:**
> Analyze how we should design and execute the Input Perturbation Comparison (`Original` vs. `FG` vs. `Mask`) to visualize confidence changes and prediction shifts across the three image versions, and give me concrete visualization ideas and code to finish Phase 1.

### 2. AI Output
- **Output / Summary:** Designed a controlled counterfactual perturbation pipeline comparing the exact same 1,767 animal images across `Original`, `FG`, and `Mask` to isolate three inductive biases:
  1. **Experimental Design (Isolating Background, Texture, and Shape):**
     - **`Original` → `FG` (Background Bias):** Tests whether removing natural backgrounds (green grass, sky, water) and replacing them with black pixels shifts predictions away from habitat-driven Pokémon toward dark/brown fallback sinks (*Linoone*, *Muddy Goomy*, *Mightyena*).
     - **`FG` → `Mask` (Texture/Color vs. Silhouette Shape Bias):** Tests which high-confidence traps rely on internal surface texture/color (`zebra` → *Zebstrika* stripes, `cattle` → *Bouffalant* brown coat) versus 2D boundary geometry (`antelope`/`deer` → *Sawsbuck*/*Stantler* branching antler silhouettes).
  2. **Visualization Roadmap & Code (`Phase1_Perturbation_Comparison.py`):**
     - **3-Stage Alluvial / Redistribution Flow:** Tracks how top Pokémon predictions shift across `Original` → `FG` → `Mask`.
     - **Archetype Survival Trajectory Plot:** Measures the percentage of each animal class assigned to its specialist Pokémon across the three stages to separate texture-dependent traps from shape-dependent traps.
     - **Confidence & Entropy Shift Violin Plot ($\Delta \text{Confidence}$):** Visualizes prediction stability and uncertainty changes when background or texture is stripped.
     - **Triplet Item Drill-Down Grid:** Displays side-by-side `[Original | FG | Mask]` image triplets with their shifting Top-1 Pokémon predictions and confidences.

### 3. Reflection
- **Strengths:** [strengths if any, compare if necessary]
- **Limitations:** [Issues /Limitation/ Problems]
- **Impact:** [What did we do with it?]


---


## Interaction 4: Code Corrections and creation of CSVs

### 0. Model
- **Model:** Gemini 3 Flash - Thinking Medium Online

### 1. Prompt / Input
- **Context:** Old Code for file generation
- **Prompt:**
> Create Code variations to analyze the mask and original datsets exactly the same as the fg datset. 

### 2. AI Output
- **Output / Summary:** Code (Multiple prompts)

### 3. Reflection
- **Strengths:** in the end cerated code which used the correct names for files, columns etc. 
- **Limitations:** Issues at understanding complex task with much code 
- **Impact:** used as input for perturbation code and analysis


---


## Interaction 5: Cross-Modal Counterfactual Perturbation Pipeline & Feature Attribution Analysis (`Original` vs. `FG` vs. `Mask`)

### 0. Model
- **Model:** Gemini 4 Argon Thinking Medium - Online

### 1. Prompt / Input
- **Context:** After establishing the Phase 1 analysis pipeline on the `fg` (black background) dataset, we needed to extend both the ViT inference script (`ViTForImageClassification`) and the CLIP + t-SNE visualization pipeline to the `original` (natural background) and `mask` (binary silhouette) image directories, and then synthesize a cross-modal statistical and visual analysis across all three image versions.
- **Prompt:**
> Adapt the Phase 1 ViT inference and CLIP/t-SNE visualization scripts to generate the enriched datasets and diagnostic plots for the `original` and `mask` folders. Then analyze in detail what patterns emerge across the three input modalities (`Original` vs. `FG` vs. `Mask`), provide a cross-modal visualization script to illustrate these findings, and summarize the interaction in our standard documentation format.

### 2. AI Output
- **Output / Summary:** Adapted the end-to-end ViT inference and CLIP/t-SNE visualization pipelines for both `original` and `mask` modalities and designed a cross-modal counterfactual analysis script (`phase1_cross_modal_analysis.py`) isolating three core inductive biases across the 1,767 matched image triplets:
  1. **Three-Tier Causal Feature Attribution (`vis_comp1_alluvial_redistribution.png`):** Joined all three modalities by `filename` to track Top-1 Pokémon prediction flows across `Original` $\rightarrow$ `FG` $\rightarrow$ `Mask`. Separated OOD misclassifications into **Scene-Driven Illusions** (predictions triggered by green grass, water, or sky in `Original` that vanish in `FG`), **Texture/Color Traps** (predictions retained in `FG` but lost in `Mask`, such as `zebra` $\rightarrow$ *Zebstrika* stripes or `cattle` $\rightarrow$ *Bouffalant* brown fur), and **Silhouette-Resilient Traps** (predictions surviving in `Mask` via 2D contour geometry alone, such as `antelope`/`deer` $\rightarrow$ *Sawsbuck* / *Stantler* branching antlers).
  2. **Background Suppression vs. Sprite Isolation Boost (`vis_comp2_archetype_survival_trajectories.png`):** Identified that stripping natural backgrounds (`Original` $\rightarrow$ `FG`) actually *increases* confidence for morphological specialist attractors because natural scenery acts as competing spatial noise in ViT self-attention, whereas isolated animals on a uniform background mimic clean Pokémon training sprites.
  3. **Counterfactual Confidence Regimes & Manifold Collapse (`vis_comp3_confidence_shift_regimes.png`):** Quantified per-class background sensitivity ($\Delta \text{Conf}_{\text{bg}} = \text{Conf}_{\text{fg}} - \text{Conf}_{\text{orig}}$) versus texture sensitivity ($\Delta \text{Conf}_{\text{tex}} = \text{Conf}_{\text{mask}} - \text{Conf}_{\text{fg}}$), showing that binary masks induce latent t-SNE manifold collapse where non-horned mammals merge into a generic shape-coverage sink while high-perimeter silhouettes remain distinct.
  4. **Counterfactual Triplet Drill-Down Grid (`vis_comp4_counterfactual_triplet_grid.png`):** Built an automated 4-row `[Original | FG | Mask]` visual inspection grid sampling real image triplets from each causal regime (Shape-Resilient, Texture-Dependent, Background-Suppressed, and Habitat Illusion) annotated with shifting Top-1 predictions, confidences, and normalized entropies.

  
### 3. Reflection
- **Strengths:** visualization ideas
- **Limitations:** issues with the code (manual repair), not always fitting analysis for our goal (tries to compare things we do not need)
- **Impact:** final visualizations for this phase