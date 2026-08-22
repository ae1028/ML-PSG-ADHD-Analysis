"""Command-line interface for the reconstructed ADHD modeling workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .model_workflow import (
    run_modeling_from_directory,
    save_modeling_results,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the modeling command-line parser."""
    parser = argparse.ArgumentParser(
        prog="psg-adhd-model",
        description=(
            "Run the historical-compatible Random Forest modeling "
            "workflow on participant-level feature CSV files."
        ),
    )

    parser.add_argument(
        "--feature-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing generated participant-level "
            "feature CSV files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Directory where modeling result tables will be written."
        ),
    )

    parser.add_argument(
        "--outer-splits",
        type=int,
        default=5,
        help=(
            "Number of outer KFold splits "
            "(historical default: 5)."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing result CSV files to be replaced.",
    )

    return parser


def run(
    argv: Sequence[str] | None = None,
):
    """Run the modeling workflow from command-line-style arguments."""
    parser = build_parser()

    args = parser.parse_args(
        argv
    )

    if args.outer_splits <= 1:
        parser.error(
            "--outer-splits must be greater than 1."
        )

    result = run_modeling_from_directory(
        feature_directory=args.feature_dir,
        n_splits=args.outer_splits,
    )

    generated = save_modeling_results(
        result,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    print("")
    print("Modeling workflow complete.")
    print("")
    print("Average metrics:")

    for row in result.summary_metrics.itertuples(
        index=False
    ):
        print(
            f"  {row.Metric}: "
            f"{row.Mean:.4f}"
        )

    print("")
    print(
        f"Generated {len(generated)} result file(s):"
    )

    for path in generated:
        print(
            f"  {path}"
        )

    print("")
    print(
        "Note: historical shuffle, outer KFold, and Random Forest "
        "operations were unseeded; repeated runs may differ."
    )

    return result, generated


def main() -> None:
    """CLI entry point."""
    run()


if __name__ == "__main__":
    main()