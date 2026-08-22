"""Command-line interface for PSG feature extraction.

This interface intentionally contains no workstation-specific paths.
Users must explicitly provide authorized local input and output
directories.

The scientific calculations are delegated to the reconstructed package
modules rather than duplicated here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .batch_processing import (
    DEFAULT_BATCH_SIZE,
    write_feature_batches,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="psg-adhd-extract",
        description=(
            "Extract historical-compatible participant-level "
            "sleep-stage graph features from MNE Epochs FIF files."
        ),
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing authorized local MNE Epochs "
            "FIF files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Directory where generated feature CSV files "
            "will be written."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of FIF files processed together "
            f"(default: {DEFAULT_BATCH_SIZE})."
        ),
    )

    parser.add_argument(
        "--pattern",
        default="*.fif",
        help=(
            "Input filename glob pattern "
            '(default: "*.fif").'
        ),
    )

    parser.add_argument(
        "--output-prefix",
        default="features_batch",
        help=(
            "Prefix for generated CSV files "
            '(default: "features_batch").'
        ),
    )

    parser.add_argument(
        "--id-column",
        default="ID",
        help=(
            'Participant metadata column '
            '(historical default: "ID").'
        ),
    )

    parser.add_argument(
        "--label-column",
        default="ADHD",
        help=(
            'Classification-label metadata column '
            '(historical default: "ADHD").'
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output CSV files to be replaced.",
    )

    return parser


def run(
    argv: Sequence[str] | None = None,
) -> list[Path]:
    """Run feature extraction from command-line-style arguments."""
    parser = build_parser()

    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        parser.error(
            "--batch-size must be a positive integer."
        )

    generated = write_feature_batches(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        pattern=args.pattern,
        output_prefix=args.output_prefix,
        overwrite=args.overwrite,
        id_column=args.id_column,
        label_column=args.label_column,
    )

    print("")
    print("Feature extraction complete.")
    print(f"Generated {len(generated)} file(s):")

    for path in generated:
        print(f"  {path}")

    return generated


def main() -> None:
    """CLI entry point."""
    run()


if __name__ == "__main__":
    main()