# CATCH EM ALL

Task:
We investigate Out-of-Distribution (OOD) misidentifications in a Pokémon classifier using a mixed dataset of real Pokémon, maybe look-alike cartoons, and real-world animals (axolotls, caterpillars, pigeons, etc.) and maybe objects (keychains), systematically tagged with spurious features (background type, outlines, color saturation, and texture).
Our goal is to visually analyze model confidence and entropy in Phase I to identify which visual triggers cause high-confidence real-world misidentifications versus when they are successfully avoided. Building on these findings, we will open the black box in Phase II using different methods to trace how real-world misidentifications are created or avoided across layers, pinpointing the exact features and layers that pull a real photo into a Pokémon cluster or keep the model properly uncertain.

*Pokemon Dataset:* https://www.kaggle.com/datasets/hlrhegemony/pokemon-image-dataset

OOD Datasets - a set of real world animals:
*Animal Image Dataset (Zooming in on Wildlife: 5400 Animal Images Across 90 Diverse Classes)*: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/data 

*AM2K (20 Classes, 100 images per class, including segmentation mask that can be used to remove background):*
https://github.com/bruinxiong/animal-matting respectively https://drive.google.com/drive/folders/1SReB9Zma0TDfDhow7P5kiZNMwY9j9xMA 

Model:
 We use a Vision Transformer (ViT-base, PyTorch/Hugging Face), pretrained on ImageNet-21k and fine-tuned on 1,025 Pokémon classes (Gen 1–9):
 https://huggingface.co/skshmjn/Pokemon-classifier-gen9-1025 

Depending on our findings, we will also compare it with the original pretrained base model (google/vit-base-patch16-224) without Pokémon fine-tuning, particularly in Phase II to investigate how fine-tuning warps the representation space and at which layer a real-world animal shifts from a biological representation to a Pokémon misidentification.
•    Model checkpoints: https://huggingface.co/imzynoxprince/pokemons-image-classifier-gen1-gen9
•    Base model: https://huggingface.co/google/vit-base-patch16-224