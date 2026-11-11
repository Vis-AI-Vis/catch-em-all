# Midpoint AI reflection - Phase I OOD analysis

AI support was most useful for converting an unstructured collaboration history into a reproducible implementation plan. Codex recovered the relevant repository, dataset and model links, inspected the preliminary notebook, and created a tested inference pipeline. This reduced setup time and exposed several hidden assumptions in the prior analysis.

The most important correction came from validation rather than generation. The public AM-2k split manifest lists 1,800 shuffled filenames but does not contain category labels. An initial row-order hypothesis failed immediately when a visual contact sheet showed mixed animal categories in every block. The category-building script was removed instead of preserving a convenient but false mapping.

The local Pokémon ViT run on 1,766 available foreground images produced 38 cases at or above 70% confidence, close to the team's 39 of 1,800 result. Label counts also matched the team pattern: Bouffalant 17, Sawsbuck 7, Mightyena 6, Linoone 3, Glastrier 2, Mabosstiff 2, and Zebstrika 1. This agreement increases confidence in the model and metric implementation.

CLIP zero-shot category suggestions helped construct exploratory cattle, antelope, and cat comparison grids. They remain imperfect: antelope was under-detected and sheep was over-detected. For that reason, generated category labels are stored separately with scores, margins, and an explicit AI-source field. Final category rates should use official labels or the team's verified mapping, not the pseudo-label table.

Next work should test causal visual hypotheses with controlled perturbations. Strong candidates include horn shape and shaggy texture for cattle-to-Bouffalant, body silhouette and horn geometry for antelope-to-Sawsbuck, and black-white coat contrast plus pointed ears for cat-to-Mightyena. Occlusion or patch-attribution experiments can separate these cues from foreground-extraction artifacts.


---

## 2. Reflection Phase 1: Thomas – Gemini 4

We worked as a team on the AI track, with each person using a different AI system.

I used AI mainly as a collaborator and as a tool to write code. I also tried generating visualizations directly within the chat interface, which worked very well; however, having the actual code allowed me to rerun everything locally, understand exactly what happened under the hood, and ensure reproducibility, so I preferred that approach.

My primary model was Gemini Argon (Gemini 4), a frontier model that I used through the web chat interface so it could leverage web search. Typically, I only adjusted the thinking level between medium and high.

Overall, the AI functioned really well as a coworker whenever I gave it enough context and information. However, virtually every prompt resulted in pages of detailed text and long scripts. This quickly became overwhelming, so I frequently had to intervene and rein it in by stating the exact analytical direction I wanted to pursue for Phase 1.

At the same time, the AI helped immensely in identifying those directions in the first place. Whenever I didn't rely strictly on the lecture scripts or my own intuition about what might be interesting, it supplied most of the strongest ideas. I have read that people collaborating with AI tend to default toward agreeing with its suggestions; even though I made a conscious effort to challenge its ideas, that same tendency still occurred throughout this project.

The sheer volume of information was also challenging to manage, particularly regarding the code it produced. The AI writes code in a style very different from my own, which made verification difficult. Because reading through thousands of lines of code line-by-line was nearly impossible, I tried to verify its accuracy by understanding the overarching logic, double-checking specific implementations with the AI itself, and carefully auditing the generated data and visual outputs.

Ultimately, I noticed that AI had a massive influence on the directions and scope I chose, which became especially striking once we compared what we accomplished in the AI team against what our colleagues in the non-AI team did.

### Key Decisions & Collaboration Examples

- **Comparing Original vs. FG vs. Mask:** The decision to run a 3-stage controlled perturbation across all three image sets was proposed entirely by the AI.
- **Designing Visualizations:** The conceptual ideas of what to compare came from me, while the specific visual techniques and plotting code were suggested and built by the AI.
- **Continuous Feature Probing:** This was our most successful collaborative moment. I wanted to investigate what visual properties drive high confidence across individual images but had no idea how to extract and quantify them easily. The AI helped massively by implementing this.

### Patterns of Issues & Failures

- **Visual Blindness & Clutter:** Because the AI cannot see its own output, many generated charts had severe label overlapping and visual clutter, making them difficult to read.
- **Inverted Metrics & Colors:** The AI once generated a confusion plot where the colorbar mapping was inverted, it caught the issue itself in a later turn after reviewing the numbers.
- **Confirmation Bias & Stubborn Assumptions:** The AI frequently enters a chat with strong initial assumptions that it aggressively tries to prove. It has to be actively stopped from some assumptions then.