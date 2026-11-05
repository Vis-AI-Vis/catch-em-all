"""Command-line entry point for the Phase I analysis."""

import argparse
from pathlib import Path

from .analysis import create_analysis
from .data import category_lookup, discover_images, load_metadata, validate_images
from .inference import run_inference

DEFAULT_MODEL = "imzynoxprince/pokemons-image-classifier-gen1-gen9"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory of input images")
    parser.add_argument("--metadata", type=Path, help="CSV with filename and category columns")
    parser.add_argument("--output", type=Path, default=Path("outputs/phase1_fg"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--skip-image-validation", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 < args.threshold < 1:
        raise SystemExit("--threshold must be between 0 and 1")
    paths = discover_images(args.images)
    if not paths:
        raise SystemExit(f"No supported images found below {args.images}")
    if not args.skip_image_validation:
        failures = validate_images(paths)
        if failures:
            details = "\n".join(f"- {path}: {error}" for path, error in failures[:10])
            raise SystemExit(f"{len(failures)} images failed validation:\n{details}")
    metadata = load_metadata(args.metadata)
    if metadata is not None:
        image_names = {path.name for path in paths}
        missing = image_names.difference(metadata["filename"])
        if missing:
            raise SystemExit(f"metadata has no category for {len(missing)} images")

    predictions = run_inference(
        image_paths=paths,
        category_by_filename=category_lookup(metadata),
        model_name=args.model,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    create_analysis(predictions, args.output, args.threshold)
    print(f"Wrote analysis to {args.output.resolve()}")


if __name__ == "__main__":
    main()
