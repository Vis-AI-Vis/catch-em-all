from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from catch_em_all.data import discover_images, load_metadata, validate_images


def test_discover_and_validate_images(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    Image.new("RGB", (4, 4), "red").save(nested / "valid.png")
    (nested / "ignore.txt").write_text("not an image")
    paths = discover_images(tmp_path)
    assert [path.name for path in paths] == ["valid.png"]
    assert validate_images(paths) == []


def test_metadata_requires_columns(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    pd.DataFrame({"filename": ["one.png"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="category"):
        load_metadata(path)


def test_metadata_rejects_missing_category(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    pd.DataFrame({"filename": ["one.png"], "category": [None]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_metadata(path)
