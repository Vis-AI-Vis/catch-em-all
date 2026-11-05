"""Batched Hugging Face ViT inference for foreground animal images."""

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .metrics import prediction_metrics


def choose_device(requested: str) -> torch.device:
    """Resolve ``auto`` to MPS, CUDA, or CPU in that order."""

    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _batches(items: list[Path], size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_inference(
    image_paths: list[Path],
    category_by_filename: dict[str, str],
    model_name: str,
    batch_size: int,
    device_name: str,
) -> pd.DataFrame:
    """Run model inference and return one row of metrics per image."""

    if not image_paths:
        raise ValueError("no images were supplied")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    device = choose_device(device_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    id2label = {int(key): value for key, value in model.config.id2label.items()}

    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for paths in tqdm(
            _batches(image_paths, batch_size),
            total=(len(image_paths) + batch_size - 1) // batch_size,
            desc=f"Inference on {device}",
        ):
            images = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {name: value.to(device) for name, value in inputs.items()}
            logits = model(**inputs).logits.detach().float().cpu().numpy()

            for path, sample_logits in zip(paths, logits):
                metrics = prediction_metrics(sample_logits, top_k=5)
                labels = [id2label[int(index)] for index in metrics.top_indices]
                probabilities = metrics.top_probabilities.tolist()
                rows.append(
                    {
                        "filename": path.name,
                        "path": str(path),
                        "category": category_by_filename.get(path.name),
                        "top1_label": labels[0],
                        "top1_confidence": probabilities[0],
                        "top2_label": labels[1],
                        "top2_confidence": probabilities[1],
                        "top2_margin": metrics.top2_margin,
                        "normalized_entropy": metrics.normalized_entropy,
                        "top5_json": json.dumps(
                            [
                                {"label": label, "probability": probability}
                                for label, probability in zip(labels, probabilities)
                            ]
                        ),
                        "num_model_labels": len(id2label),
                    }
                )
    return pd.DataFrame(rows)
