# CATCH EM ALL

Task:
We investigate Out-of-Distribution (OOD) misidentifications in a Pokémon classifier using a mixed dataset of real Pokémon, maybe look-alike cartoons, and real-world animals (axolotls, caterpillars, pigeons, etc.) and maybe objects (keychains), systematically tagged with spurious features (background type, outlines, color saturation, and texture).
Our goal is to visually analyze model confidence and entropy in Phase I to identify which visual triggers cause high-confidence real-world misidentifications versus when they are successfully avoided. Building on these findings, we will open the black box in Phase II using different methods to trace how real-world misidentifications are created or avoided across layers, pinpointing the exact features and layers that pull a real photo into a Pokémon cluster or keep the model properly uncertain.
•    Dataset: https://www.kaggle.com/datasets/hlrhegemony/pokemon-image-dataset (plus a curated OOD test set of real-world animals, etc.)

Model:
 We use a Vision Transformer (ViT-base, PyTorch/Hugging Face), pretrained on ImageNet-21k and fine-tuned on 1,025 Pokémon classes (Gen 1–9).
Depending on our findings, we will also compare it with the original pretrained base model (google/vit-base-patch16-224) without Pokémon fine-tuning, particularly in Phase II to investigate how fine-tuning warps the representation space and at which layer a real-world animal shifts from a biological representation to a Pokémon misidentification.
•    Model checkpoints: https://huggingface.co/imzynoxprince/pokemons-image-classifier-gen1-gen9
•    Base model: https://huggingface.co/google/vit-base-patch16-224