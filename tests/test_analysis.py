from pathlib import Path

import pandas as pd

from catch_em_all.analysis import create_analysis


def test_analysis_without_metadata_writes_core_outputs(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "filename": ["one.png", "two.png"],
            "path": ["one.png", "two.png"],
            "category": [None, None],
            "top1_label": ["Pikachu", "Eevee"],
            "top1_confidence": [0.8, 0.2],
            "normalized_entropy": [0.2, 0.8],
            "num_model_labels": [1028, 1028],
        }
    )
    create_analysis(predictions, tmp_path, 0.7)
    assert (tmp_path / "predictions.csv").is_file()
    assert (tmp_path / "high_confidence_predictions.csv").is_file()
    assert "1 (50.00%)" in (tmp_path / "summary.md").read_text()
