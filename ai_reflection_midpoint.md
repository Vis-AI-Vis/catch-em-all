# Midpoint AI reflection - Phase I OOD analysis

AI support was most useful for converting an unstructured collaboration history into a reproducible implementation plan. Codex recovered the relevant repository, dataset and model links, inspected the preliminary notebook, and created a tested inference pipeline. This reduced setup time and exposed several hidden assumptions in the prior analysis.

The most important correction came from validation rather than generation. The public AM-2k split manifest lists 1,800 shuffled filenames but does not contain category labels. An initial row-order hypothesis failed immediately when a visual contact sheet showed mixed animal categories in every block. The category-building script was removed instead of preserving a convenient but false mapping.

The local Pokémon ViT run on 1,766 available foreground images produced 38 cases at or above 70% confidence, close to the team's 39 of 1,800 result. Label counts also matched the team pattern: Bouffalant 17, Sawsbuck 7, Mightyena 6, Linoone 3, Glastrier 2, Mabosstiff 2, and Zebstrika 1. This agreement increases confidence in the model and metric implementation.

CLIP zero-shot category suggestions helped construct exploratory cattle, antelope, and cat comparison grids. They remain imperfect: antelope was under-detected and sheep was over-detected. For that reason, generated category labels are stored separately with scores, margins, and an explicit AI-source field. Final category rates should use official labels or the team's verified mapping, not the pseudo-label table.

Next work should test causal visual hypotheses with controlled perturbations. Strong candidates include horn shape and shaggy texture for cattle-to-Bouffalant, body silhouette and horn geometry for antelope-to-Sawsbuck, and black-white coat contrast plus pointed ears for cat-to-Mightyena. Occlusion or patch-attribution experiments can separate these cues from foreground-extraction artifacts.
