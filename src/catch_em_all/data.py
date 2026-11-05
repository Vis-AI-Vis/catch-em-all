"""Image and metadata loading with explicit validation."""

from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def discover_images(image_dir: Path) -> list[Path]:
    """Return all supported images below ``image_dir`` in stable path order."""

    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_metadata(metadata_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    """Load a filename/category mapping, if supplied."""

    if metadata_csv is None:
        return None
    metadata = pd.read_csv(metadata_csv)
    required = {"filename", "category"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    if metadata["filename"].duplicated().any():
        raise ValueError("metadata contains duplicate filenames")
    if metadata[["filename", "category"]].isna().any().any():
        raise ValueError("metadata contains missing filenames or categories")
    return metadata[["filename", "category"]].copy()


def validate_images(paths: list[Path]) -> list[tuple[Path, str]]:
    """Decode every image and return any failures without stopping at the first one."""

    failures: list[tuple[Path, str]] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:  # Pillow exposes format-specific exception classes.
            failures.append((path, str(exc)))
    return failures


def category_lookup(metadata: Optional[pd.DataFrame]) -> dict[str, str]:
    if metadata is None:
        return {}
    return dict(zip(metadata["filename"], metadata["category"]))
