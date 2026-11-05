# CATCH EM ALL

Task:
We investigate Out-of-Distribution (OOD) misidentifications in a Pokémon classifier using a mixed dataset of real Pokémon, maybe look-alike cartoons, and real-world animals (axolotls, caterpillars, pigeons, etc.) and maybe objects (keychains), systematically tagged with spurious features (background type, outlines, color saturation, and texture).
Our goal is to visually analyze model confidence and entropy in Phase I to identify which visual triggers cause high-confidence real-world misidentifications versus when they are successfully avoided. Building on these findings, we will open the black box in Phase II using different methods to trace how real-world misidentifications are created or avoided across layers, pinpointing the exact features and layers that pull a real photo into a Pokémon cluster or keep the model properly uncertain.
•    Dataset: https://www.kaggle.com/datasets/hlrhegemony/pokemon-image-dataset (plus a curated OOD test set of real-world animals, etc.)

Model:
 We use a Vision Transformer (ViT-base, PyTorch/Hugging Face), pretrained on ImageNet-21k and fine-tuned on 1,028 output labels in the current checkpoint (Gen 1–9).
Depending on our findings, we will also compare it with the original pretrained base model (google/vit-base-patch16-224) without Pokémon fine-tuning, particularly in Phase II to investigate how fine-tuning warps the representation space and at which layer a real-world animal shifts from a biological representation to a Pokémon misidentification.
•    Model checkpoints: https://huggingface.co/imzynoxprince/pokemons-image-classifier-gen1-gen9
•    Base model: https://huggingface.co/google/vit-base-patch16-224

## Reproducible Phase I analysis

The repository now includes a local, read-only inference pipeline for the AM-2k
foreground images. It records the top two Pokémon predictions, top-1 confidence,
normalized entropy, top-2 margin, and top five probabilities for every image.
It also produces the all-confidence and high-confidence prediction matrices needed
for the next comparison.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run

```bash
catch-em-all-phase1 \
  --images data/am2k/fg \
  --metadata data/am2k/metadata.csv \
  --output outputs/phase1_fg \
  --device auto \
  --batch-size 16
```

The input data, model cache, and generated outputs are intentionally excluded from
Git. The optional metadata CSV must contain `filename` and `category` columns. This
keeps category provenance visible instead of embedding an undocumented filename
rule in the analysis.

AM-2k's public `filenames_train.txt` is a shuffled split manifest, not a category
mapping. Do not infer categories from row order. If official category metadata is
unavailable, `scripts/pseudo_label_am2k.py` can create clearly marked CLIP
zero-shot suggestions for exploratory visual grouping. Those suggestions are not
ground truth and should not replace verified labels in final statistics.

```bash
python scripts/pseudo_label_am2k.py \
  --images data/am2k/fg \
  --output outputs/phase1_fg/ai_category_suggestions.csv
```

### Tests

```bash
pytest
```
