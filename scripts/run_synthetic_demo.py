"""Run the complete public synthetic ML-PSG-ADHD demonstration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SOURCE_ROOT),
    )


from psg_adhd.demo import run_synthetic_demo  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the synthetic-demo command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete ML-PSG-ADHD software pipeline using "
            "fully artificial demonstration data."
        )
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(
            "outputs/synthetic_demo"
        ),
        help=(
            "Directory for generated demonstration artifacts "
            "(default: outputs/synthetic_demo)."
        ),
    )

    parser.add_argument(
        "--participants",
        type=int,
        default=10,
        help=(
            "Number of artificial participants "
            "(default: 10; minimum: 10)."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help=(
            "Synthetic PSG generation seed "
            "(default: 2026)."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated demonstration artifacts.",
    )

    return parser


def main() -> None:
    """Run the synthetic demonstration."""
    parser = build_parser()

    args = parser.parse_args()

    result = run_synthetic_demo(
        work_dir=args.work_dir,
        n_participants=args.participants,
        random_state=args.seed,
        overwrite=args.overwrite,
    )

    print("")
    print("============================================================")
    print("SYNTHETIC DEMONSTRATION COMPLETE")
    print("============================================================")

    print("")
    print(
        f"Synthetic PSG files: {len(result.psg_files)}"
    )

    print(
        f"Feature CSV files: {len(result.feature_files)}"
    )

    print(
        f"Modeling result files: {len(result.result_files)}"
    )

    print("")
    print("Modeling smoke-test metrics:")

    for row in result.summary_metrics.itertuples(
        index=False
    ):
        print(
            f"  {row.Metric}: {row.Mean:.4f}"
        )

    print("")
    print(
        "IMPORTANT: These metrics come from artificial software-demo "
        "data and are not scientific or clinical results."
    )

    print("")
    print(
        "Historical-compatible modeling retains unseeded stochastic "
        "operations, so repeated demo metrics may differ."
    )


if __name__ == "__main__":
    main()