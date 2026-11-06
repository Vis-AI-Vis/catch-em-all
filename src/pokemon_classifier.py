"""
An introduction on how to use the base model to classify a single image can be found here:
https://huggingface.co/skshmjn/Pokemon-classifier-gen9-1025
This was used to generate a custom dataset loader for the Pokemon- and Animals-datasets.
"""

from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import os
import pandas as pd
import numpy as np

from src.data import PokemonDataset
from src.visualisation import visualise_confidence_vs_entropy
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
IMAGES_DIR = REPO_ROOT / "images"
PREDICTIONS_DIR = REPO_ROOT / "predictions"

# --- Input image directories
POKEMON_DIR = IMAGES_DIR / "pokemon"
POKEMON_CLASSES = os.listdir(POKEMON_DIR)
ANIMAL_DIR = IMAGES_DIR / "animals"
ANIMAL_CLASSES = os.listdir(ANIMAL_DIR)
# AM-2k: the same 2000 animal photos, once unedited and once on a white background
# (cut out with the dataset's masks), so they look more like the Pokémon images
AM2K_ORIGINAL_DIR = IMAGES_DIR / "AM-2k-original"
AM2K_BG_REMOVED_DIR = IMAGES_DIR / "AM-2k-bg-removed"

MODEL_ID = "skshmjn/Pokemon-classifier-gen9-1025"

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"


def predict_image(image_path, model, image_processor): ...


def predict_dataset(dataset, model, image_processor, batch_size=32):
    # Create a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    confidences = []
    predicted_ids = []
    predicted_labels = []
    true_ids = []
    true_labels = []
    image_paths = []

    with torch.no_grad():
        # Iterate over the dataset
        for batch_inputs, batch_true_ids, batch_indices in tqdm(
            dataloader, desc=f"Predicting {dataset.name}", unit="batch"
        ):
            batch_inputs = batch_inputs.to(device)

            # Predict (no labels passed: we don't need the loss, and animals have id -1)
            outputs = model(pixel_values=batch_inputs)
            # Save predictions
            predicted_ids.extend(outputs.logits.argmax(-1).tolist())
            confidences.extend(outputs.logits.softmax(-1).tolist())
            true_ids.extend(batch_true_ids.tolist())
            image_paths.extend(dataset.get_paths(batch_indices.tolist()))

    confidences = np.array(confidences, dtype=np.float32)
    predicted_labels = [model.config.id2label[i] for i in predicted_ids]
    # Folder name as true label, so it also works for non-Pokémon images (id -1)
    true_labels = [path.parent.name for path in image_paths]

    predictions_df = pd.DataFrame(
        {
            "image_path": [str(path) for path in image_paths],
            "true_label_id": true_ids,
            "true_label": true_labels,
            "predicted_label_id": predicted_ids,
            "predicted_label": predicted_labels,
            "confidence": confidences.max(axis=1),
            "entropy": entropy(confidences),
            "entropy_norm": normalized_entropy(confidences),
        }
    )

    return predictions_df, confidences


def entropy(probs, eps=1e-12):
    """Entropy (in nats) of each row of a probability matrix."""
    return -np.sum(probs * np.log(probs + eps), axis=1)


def normalized_entropy(probs):
    """Entropy divided by its maximum log(num_classes), so it lies in [0, 1]."""
    return entropy(probs) / np.log(probs.shape[1])


def load_model(model_id=MODEL_ID):
    """
    Load the model and image processor.
    This will automatically download the model weights etc. into the models directory.
    """
    model = ViTForImageClassification.from_pretrained(
        model_id, cache_dir=MODELS_DIR
    ).to(device)
    model.eval()
    image_processor = ViTImageProcessor.from_pretrained(model_id)
    return model, image_processor


def load_datasets(model, image_processor):
    """Pokémon (known classes of the model) and animal datasets (unknown to the model)."""
    return [
        PokemonDataset(
            POKEMON_DIR, image_processor, label2id=model.config.label2id, name="pokemon"
        ),
        PokemonDataset(ANIMAL_DIR, image_processor, label2id=None, name="animals"),
        PokemonDataset(
            AM2K_ORIGINAL_DIR, image_processor, label2id=None, name="am2k_original"
        ),
        PokemonDataset(
            AM2K_BG_REMOVED_DIR, image_processor, label2id=None, name="am2k_bg_removed"
        ),
    ]


def load_or_predict(dataset, model, image_processor):
    """Load saved predictions of a dataset, or predict and save them if there are none."""
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    predictions_csv = PREDICTIONS_DIR / f"{dataset.name}_predictions.csv"
    confidences_npy = PREDICTIONS_DIR / f"{dataset.name}_confidences.npy"

    if predictions_csv.exists() and confidences_npy.exists():
        print(f"Loading saved predictions for {dataset.name}")
        predictions_df = pd.read_csv(predictions_csv)
        confidences = np.load(confidences_npy)
        # Older saved predictions don't have this column yet
        if "entropy_norm" not in predictions_df:
            predictions_df["entropy_norm"] = normalized_entropy(confidences)
            predictions_df.to_csv(predictions_csv, index=False)
    else:
        print(f"Predicting {len(dataset)} images of {dataset.name}")
        predictions_df, confidences = predict_dataset(dataset, model, image_processor)
        predictions_df.to_csv(predictions_csv, index=False)
        np.save(confidences_npy, confidences)

    return predictions_df, confidences


if __name__ == "__main__":
    model, image_processor = load_model()
    datasets = load_datasets(model, image_processor)

    predictions = {}
    for dataset in datasets:
        predictions_df, _ = load_or_predict(dataset, model, image_processor)
        predictions[dataset.name] = predictions_df

        print(
            f"{dataset.name}: mean confidence {predictions_df['confidence'].mean():.3f}, "
            f"mean normalized entropy {predictions_df['entropy_norm'].mean():.3f}"
        )

    pokemon_df = predictions["pokemon"]
    accuracy = (pokemon_df["true_label_id"] == pokemon_df["predicted_label_id"]).mean()
    print(f"Pokémon accuracy: {accuracy:.3f}")

    visualise_confidence_vs_entropy(
        predictions,
        save_path=PREDICTIONS_DIR / "confidence_vs_entropy.png",
    )
