# Topic: Next steps

- **Date:** 2026-11-07
- **Model(s) Used:** Gemini 4 (Thinking:High, Online)
- **User:** k1427462
- **Overall Objective:** Assess status and next steps

---

## Interaction 3: Evaluating Initial Visualizations & Planning Deep-Dive Behavioral Analysis

### 0. Model
- **Model:** Gemini 4 (Thinking:High, Online)

### 1. Prompt / Input
- **Context:** After generating our initial Phase 1 visualizations (Animal-to-Pokémon Bubble Confusion Matrix, 2D t-SNE Projection, and two Network Graphs), we compared our outputs with lecture Info and project goals, also first ideas on how to process where inputted by user
- **Prompt:**
> [Attached 4 generated visualization PNGs and lecture info]
> Do we have a good start based on the lecture slides and our current plots? Since our bubble chart already acts as a confusion matrix, should we focus on scatter plots/projections next? Specifically, we are considering: (1) filtering out low-confidence noise to investigate why specific animal-Pokémon pairs have high confidence, (2) comparing model behavior across `original`, `fg` (black background), and `mask` datasets, (3) refining our projection techniques for Phase 1, and (4) potentially using XAI methods like LIME or Grad-CAM.

### 2. AI Output
- **Output / Summary:** Evaluated the current visualizations against the Phase 1 assignment criteria and proposed a concrete roadmap:
  1. **Assessment of Current Plots & Colorbar Fix:** Confirmed the Bubble Matrix effectively captures Slide 3's Confusion Matrix concept—separating semantic/morphological matches (`cattle` → *Bouffalant*, `antelope` → *Sawsbuck*, `zebra` → *Zebstrika*) from uncertain default OOD sinks (*Linoone*, *Muddy Goomy*). Identified an inverted colorbar interpretation where low normalized entropy (`0.1–0.3`, teal/yellow) actually represents high confidence and dark purple (`0.8`) represents high uncertainty.
  2. **High-Confidence Deep Dive (Idea 1):** Recommended filtering out high-entropy fallback connections in the network graphs and correlating model confidence with image-level metrics (foreground coverage, brightness, saturation) and specific visual poses.
  3. **Controlled Input Ablation Across `Original`, `FG`, and `Mask` (Idea 2):** Highlighted this as the strongest Phase 1 experiment to test *"what goes in and what comes out"*—isolating **Background Bias** (`original` vs. `fg`) and **Texture/Color vs. Shape Silhouette Bias** (`fg` vs. `mask`).
  4. **Upgrading Projections & Deferring XAI (Ideas 3 & 4):** Suggested aligning the t-SNE plot with Slide 2 (*Drill down → Understand → Summarize → Annotate*) by coloring points by model prediction/confidence (or projecting model logits) and embedding actual image thumbnails on outlier callouts. Advised saving white-box methods (LIME, Grad-CAM) for Phase 2 since Phase 1 focuses on black-box input-output behavior.

### 3. Reflection
- **Strengths:** found issue with colors in already checked output, good in analysing ideas, good ideas on how to implement rough user ideas (what to compare how)
- **Limitations:** no good prioritization; code needed to be regenerated, no new ideas (only used user ideas); Used frontier model with thinking high: immense token usage
- **Impact:** copied output to plan next stpes myself with this as idea on what I can do as next steps. 

---

## Interaction 2: Small fix of Visualization

### 0. Model
- **Model:** Gemini 4 (Thinking:High, Online)

### 1. Prompt / Input
- **Context:** Code
- **Prompt:**
> [Fix my issue with wrong colors]

### 2. AI Output
- **Output / Summary:** New fixed code

### 3. Reflection
- **Strengths:** Very good at coding
- **Limitations:** Code has to be rechecked
- **Impact:** Fixed a visualization issue
