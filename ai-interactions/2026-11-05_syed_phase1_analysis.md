# Phase I reproducible OOD analysis

- **Date:** 2026-11-05
- **Model Used:** OpenAI Codex (GPT-5)
- **User:** Muhid Syed Muhammad (`thenorthsolo`)
- **Overall Objective:** Recover the team task, prepare a reproducible local analysis, and avoid pushing until the work is reviewed.

---

## Interaction 1: Task recovery and scope

### 0. Model
- **Model:** OpenAI Codex (GPT-5)

### 1. Prompt / Input
- **Context:** The user asked Codex to read the AI&Vis Discord conversation, extract links and assigned work for `@SYEDMUHID`, find the GitHub repository and referenced Markdown/notebook material, then start locally with read-only behavior.
- **Prompt:**
> Read the AI&Vis discussion, identify my tasks and links, locate the repository and Markdown material, then work through the task one step at a time without pushing yet.

### 2. AI Output
- **Output / Summary:** Codex identified the `Vis-AI-Vis/catch-em-all` repository, recovered the preliminary FG OOD notebook, summarized the 1,800-image findings, cloned the repository, and created a local feature branch.

### 3. Reflection
- **Strengths:** Consolidated scattered team context into a concrete local scope. Preserved the existing repository and avoided remote writes.
- **Limitations:** Discord messages and attached notebook provided results but no executable code or category mapping.
- **Impact:** The recovered notebook became a reference result, not an unverified source of implementation details.

---

## Interaction 2: Reproducible inference implementation

### 0. Model
- **Model:** OpenAI Codex (GPT-5)

### 1. Prompt / Input
- **Context:** The user authorized local work while waiting for repository access.
- **Prompt:**
> Start working on everything with read-only access. We can push later.

### 2. AI Output
- **Output / Summary:** Codex implemented package setup, image validation, batched Hugging Face ViT inference, confidence, normalized entropy, top-2 margin, top-five output, CSV exports, plots, comparison montages, and automated tests.

### 3. Reflection
- **Strengths:** Pipeline is reproducible, device-aware, and separates model inference from category metadata. Six tests pass.
- **Limitations:** The public folder download contained 1,766 of 1,800 FG files. The official split manifest is shuffled and contains no category labels.
- **Impact:** Global predictions reproduce the team result closely: 38 high-confidence cases locally versus 39 in the complete team run.

---

## Interaction 3: AI-assisted category suggestions

### 0. Model
- **Model:** `openai/clip-vit-base-patch32`, orchestrated by OpenAI Codex

### 1. Prompt / Input
- **Context:** Category metadata was absent, but the project requires visual comparison within cattle, antelope, and cat categories.
- **Prompt:**
> Use AI-assisted analysis to identify useful visual groups, while keeping limitations visible.

### 2. AI Output
- **Output / Summary:** A separate script created CLIP zero-shot category suggestions with top-two scores and margins. High-confidence groups were visually checked against the team notebook. Suggestions were used only for exploratory grouping.

### 3. Reflection
- **Strengths:** Made AI involvement auditable and preserved uncertainty columns. Cattle and cat suggestions were strong enough for exploratory montages.
- **Limitations:** CLIP under-counted antelope and over-counted sheep. Pseudo-labels are not ground truth and must not replace verified category metadata in final statistics.
- **Impact:** Enabled focused high-versus-low visual comparisons without presenting generated labels as official annotations.
