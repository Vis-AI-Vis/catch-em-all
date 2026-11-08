"""
Preprocessing that makes photos look more like the Pokémon artwork, to test whether a
more "cartoon like" style increases the model's confidence on out-of-distribution images.

Needs OpenCV: pip install opencv-python
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
OUTLINE_COLOUR = (25, 25, 25)


def _foreground_mask(rgb, background_threshold):
    """
    True for the animal, False for the white background.

    Background = near-white pixels connected to the image border, so white fur inside
    the animal (e.g. a polar bear) is not mistaken for background.
    """
    near_white = np.all(rgb >= background_threshold, axis=2).astype(np.uint8)
    _, components = cv2.connectedComponents(near_white)
    border = np.concatenate(
        [components[0], components[-1], components[:, 0], components[:, -1]]
    )
    border_labels = np.unique(border[border > 0])  # label 0 = not near-white
    background = np.isin(components, border_labels)
    # Smooth the silhouette a little, so its outline is not jagged
    foreground = cv2.GaussianBlur((~background).astype(np.float32), (5, 5), 0) > 0.5
    return foreground


def _remove_short_lines(edges, min_length):
    """Drop edge fragments with fewer than min_length pixels (fur texture, noise)."""
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        edges, connectivity=8
    )
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_length
    return np.where(keep[components], 255, 0).astype(np.uint8)


def cartoonize_image(
    img,
    max_size=384,
    smoothing=6,
    colours=6,
    cleanup=9,
    saturation=1.8,
    brightness=1.25,
    outline_thresholds=(80, 180),
    min_line_length=25,
    outline_width=2,
    silhouette_width=3,
    background_threshold=235,
):
    """
    Make a photo look more like Pokémon artwork: flat, light and vivid colour areas,
    a dark outline around the animal and a few dark lines inside it.

    img: PIL image (returns a PIL image, so it works like the other images in the notebook)
    max_size: longest side in pixels (Pokémon images are ~350 px; smaller = bigger flat areas)
    smoothing: number of bilateral filter passes; removes fur/texture detail but keeps edges
    colours: number of colours of the animal after k-means quantization (fewer = flatter)
    cleanup: median filter size that removes small colour blotches (odd number, 0 = off)
    saturation: 1.0 = unchanged, >1 = more colourful
    brightness: 1.0 = unchanged, >1 = lighter colours
    outline_thresholds: (low, high) Canny thresholds for the lines inside the animal;
                        higher = fewer lines, None = no inner lines
    min_line_length: inner lines shorter than this (in pixels) are dropped
    outline_width: thickness of the inner lines in pixels
    silhouette_width: thickness of the outline around the animal (0 = none)
    background_threshold: pixels brighter than this (in all channels) that touch the image
                          border count as white background; it stays pure white.
                          None = no background handling (e.g. for photos with a background)
    """
    rgb = np.array(img.convert("RGB"))
    scale = max_size / max(rgb.shape[:2])
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if background_threshold is not None:
        foreground = _foreground_mask(rgb, background_threshold)
    else:
        foreground = np.ones(rgb.shape[:2], dtype=bool)
    if not foreground.any():  # nothing found, treat the whole image as the animal
        foreground[:] = True

    # 1. Smooth away texture while keeping edges sharp
    smooth = rgb
    for _ in range(smoothing):
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. Flat colour areas: k-means on the animal only, so the background
    #    doesn't use up one of the colours
    pixels = smooth[foreground].astype(np.float32)
    k = min(colours, len(pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centres = cv2.kmeans(
        pixels, k, None, criteria, attempts=1, flags=cv2.KMEANS_PP_CENTERS
    )
    cartoon = np.full_like(smooth, 255)
    cartoon[foreground] = centres.astype(np.uint8)[labels.flatten()]
    if cleanup:
        cartoon = cv2.medianBlur(cartoon, cleanup)

    # 3. Lighter, more vivid colours, like the Pokémon artwork
    hsv = cv2.cvtColor(cartoon, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness, 0, 255)
    cartoon = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    cartoon[~foreground] = 255

    # 4. A few dark lines inside the animal, along its strongest edges
    if outline_thresholds is not None:
        gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, *outline_thresholds)
        edges[~foreground] = 0
        edges = _remove_short_lines(edges, min_line_length)
        if outline_width > 1:
            kernel = np.ones((outline_width, outline_width), np.uint8)
            edges = cv2.dilate(edges, kernel)
        cartoon[edges > 0] = OUTLINE_COLOUR

    # 5. Dark outline around the animal (not along the image border, where the
    #    animal is cut off by the photo - that would look like a frame)
    if silhouette_width and not foreground.all():
        contours, _ = cv2.findContours(
            foreground.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        silhouette = np.zeros(foreground.shape, dtype=np.uint8)
        cv2.drawContours(silhouette, contours, -1, 255, silhouette_width)
        band = silhouette_width + 1
        silhouette[:band] = silhouette[-band:] = 0
        silhouette[:, :band] = silhouette[:, -band:] = 0
        cartoon[silhouette > 0] = OUTLINE_COLOUR

    return Image.fromarray(cartoon)


def create_cartoon_dataset(src_dir, dst_dir, overwrite=False, **cartoon_kwargs):
    """
    Cartoonize every image in src_dir and save it under dst_dir with the same
    <class>/<file> structure, so it can be loaded like the other datasets.

    Images that already exist in dst_dir are skipped unless overwrite=True.
    cartoon_kwargs are passed on to cartoonize_image (e.g. colours=8).
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    paths = sorted(
        path for path in src_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )  # [::20]
    created = 0
    for path in tqdm(paths, desc=f"Cartoonizing {src_dir.name}", unit="image"):
        out_path = (dst_dir / path.relative_to(src_dir)).with_suffix(".jpg")
        if out_path.exists() and not overwrite:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as img:
            cartoonize_image(img, **cartoon_kwargs).save(out_path)  # , quality=100)
        created += 1
    print(
        f"{created} images created, {len(paths) - created} already existed in {dst_dir}"
    )
    return dst_dir
