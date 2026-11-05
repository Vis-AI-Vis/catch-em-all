# Syed's Phase I Contribution

## What I worked on

My Phase I work focused on checking how a Pokémon image classifier responds to
real animal photographs. I reproduced the analysis locally and concentrated on
cattle, antelope, and cat images because these groups gave useful examples of
both high- and low-confidence predictions.

The strongest repeated matches were cattle predicted as Bouffalant, antelope
predicted as Sawsbuck, and cats predicted as Mightyena. I compared the visual
differences between confident and uncertain examples, including horn shape,
body outline, coat contrast, pose, cropping, and the quality of the isolated
foreground. These comparisons are working hypotheses for Phase II, not proof
that the classifier used one specific feature.

## What the code does

The code provides a reproducible pipeline for the AM-2k foreground animal
images. It:

- finds and validates the input images;
- loads the Pokémon Vision Transformer classifier;
- runs inference in batches on CPU, CUDA, or Apple MPS;
- records the top predictions, confidence, normalized entropy, top-two margin,
  and top-five probabilities for every image;
- exports summary CSV files and visual comparisons; and
- produces category-level tables when verified animal metadata is available.

There is also an optional CLIP script for exploratory category suggestions.
Those suggestions are stored separately and are not treated as verified animal
labels. This matters because the automatic suggestions were useful for cattle
and cats but less reliable for visually similar groups such as antelope, deer,
and sheep.

## Checks completed

The implementation includes tests for the confidence metrics, image discovery,
metadata validation, and output generation. Generated datasets, model files,
and analysis outputs remain outside Git so the repository only contains the
code and documentation needed to reproduce the work.
