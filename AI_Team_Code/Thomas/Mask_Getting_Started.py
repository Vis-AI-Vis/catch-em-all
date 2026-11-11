import glob
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageStat
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

# 1. Configuration & Device Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "imzynoxprince/pokemons-image-classifier-gen1-gen9"
IMAGE_DIR = os.path.join(SCRIPT_DIR, "mask")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "phase1_mask_analysis.csv")
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"--> Using device: {device}")

# 2. Load Processor and Model
processor = ViTImageProcessor.from_pretrained(MODEL_ID)
model = ViTForImageClassification.from_pretrained(MODEL_ID).to(device)
model.eval()


# 3. Helper: Load Mask & Extract Shape/Silhouette Features
def load_and_extract_features(mask_path):
    raw_mask = Image.open(mask_path).convert("L")
    mask_arr = np.array(raw_mask)

    # Binary foreground coverage (white pixels)
    fg_bool = mask_arr > 128
    fg_coverage = float(np.mean(fg_bool))

    # Mask silhouette edge density (boundary complexity of the animal shape)
    edges = raw_mask.filter(ImageFilter.FIND_EDGES)
    edge_density = ImageStat.Stat(edges).mean[0] / 255.0

    # Photometric stats for binary silhouettes
    mean_brightness = float(mask_arr.mean() / 255.0)
    contrast_std = float(mask_arr.std() / 255.0)
    mean_saturation = 0.0  # Grayscale masks have no chromatic saturation

    features = {
        "mean_saturation": round(mean_saturation, 4),
        "mean_brightness": round(mean_brightness, 4),
        "contrast_std": round(contrast_std, 4),
        "edge_density": round(edge_density, 4),
        "fg_coverage": round(fg_coverage, 4),
    }

    # Convert to 3-channel RGB so ViT processor accepts it without shape mismatch
    rgb_img = raw_mask.convert("RGB")
    return rgb_img, features


# 4. Gather Images in ./mask
image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp")
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, "**", ext), recursive=True))
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
image_paths = sorted(list(set(image_paths)))

print(f"--> Found {len(image_paths)} images in '{IMAGE_DIR}'")

# 5. Batch Inference Loop
results = []
max_entropy = float(np.log(model.config.num_labels))

for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Running ViT Inference (Masks)"):
    batch_paths = image_paths[i : i + BATCH_SIZE]
    batch_images = []
    batch_meta = []

    for path in batch_paths:
        try:
            img, feats = load_and_extract_features(path)
            batch_images.append(img)
            batch_meta.append((path, feats))
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not batch_images:
        continue

    inputs = processor(images=batch_images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Normalized Shannon Entropy
        entropies = -(probs * log_probs).sum(dim=-1).cpu().numpy()

        # Top-5 Predictions
        top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)
        top5_probs = top5_probs.cpu().numpy()
        top5_indices = top5_indices.cpu().numpy()

    for idx, (path, feats) in enumerate(batch_meta):
        t_probs = top5_probs[idx]
        t_indices = top5_indices[idx]
        t_labels = [model.config.id2label[int(c_idx)] for c_idx in t_indices]

        row = {
            "filename": os.path.basename(path),
            "filepath": path,
            "entropy": round(float(entropies[idx]), 4),
            "normalized_entropy": round(float(entropies[idx] / max_entropy), 4),
            "top1_pokemon": t_labels[0],
            "top1_conf": round(float(t_probs[0]), 4),
            "top2_pokemon": t_labels[1],
            "top2_conf": round(float(t_probs[1]), 4),
            "conf_margin_1_2": round(float(t_probs[0] - t_probs[1]), 4),
            "top5_pokemons": "|".join(t_labels),
            **feats,
        }
        results.append(row)

# 6. Save Results to CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n--> Saved {len(df)} rows to {OUTPUT_CSV}")