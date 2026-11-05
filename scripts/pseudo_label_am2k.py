#!/usr/bin/env python3
"""Create auditable AI-assisted AM-2k category suggestions with CLIP.

These are analysis aids, not ground-truth labels. The output includes the top-two
scores and margin so uncertain assignments remain visible.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from catch_em_all.data import discover_images
from catch_em_all.inference import choose_device

CATEGORIES = [
    "alpaca", "antelope", "bear", "camel", "cat", "cattle", "deer", "dog",
    "elephant", "giraffe", "horse", "kangaroo", "leopard", "lion", "monkey",
    "rabbit", "rhinoceros", "sheep", "tiger", "zebra",
]
PROMPTS = {
    "alpaca": "a photograph of an alpaca or llama",
    "antelope": "a photograph of an antelope or gazelle",
    "bear": "a photograph of a bear",
    "camel": "a photograph of a camel",
    "cat": "a photograph of a domestic cat",
    "cattle": "a photograph of cattle, a cow, bull, or ox",
    "deer": "a photograph of a deer",
    "dog": "a photograph of a domestic dog",
    "elephant": "a photograph of an elephant",
    "giraffe": "a photograph of a giraffe",
    "horse": "a photograph of a horse",
    "kangaroo": "a photograph of a kangaroo",
    "leopard": "a photograph of a leopard or cheetah",
    "lion": "a photograph of a lion",
    "monkey": "a photograph of a monkey or ape",
    "rabbit": "a photograph of a rabbit or hare",
    "rhinoceros": "a photograph of a rhinoceros",
    "sheep": "a photograph of a sheep or ram",
    "tiger": "a photograph of a tiger",
    "zebra": "a photograph of a zebra",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    paths = discover_images(args.images)
    device = choose_device(args.device)
    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    text_inputs = processor(
        text=[PROMPTS[category] for category in CATEGORIES],
        return_tensors="pt",
        padding=True,
    )
    text_inputs = {name: value.to(device) for name, value in text_inputs.items()}
    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    rows = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(paths), args.batch_size), desc=f"CLIP on {device}"):
            batch_paths = paths[start : start + args.batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (100 * image_features @ text_features.T).softmax(dim=-1).cpu()
            scores, indices = similarities.topk(2, dim=-1)
            for path, sample_scores, sample_indices in zip(batch_paths, scores, indices):
                rows.append(
                    {
                        "filename": path.name,
                        "pseudo_category": CATEGORIES[int(sample_indices[0])],
                        "pseudo_score": float(sample_scores[0]),
                        "second_category": CATEGORIES[int(sample_indices[1])],
                        "second_score": float(sample_scores[1]),
                        "pseudo_margin": float(sample_scores[0] - sample_scores[1]),
                        "label_source": f"AI-assisted zero-shot suggestion ({args.model})",
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} category suggestions to {args.output}")


if __name__ == "__main__":
    main()
