"""Generate fully artificial MNE Epochs FIF files.

These files are for software demonstration only and contain no clinical
or participant-derived data.
"""

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


from psg_adhd.synthetic_psg import (  # noqa: E402
    generate_synthetic_psg_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fully artificial MNE Epochs FIF files for "
            "testing the ML-PSG-ADHD software pipeline."
        )
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for generated synthetic FIF files.",
    )

    parser.add_argument(
        "--participants",
        type=int,
        default=10,
        help="Number of artificial participants (default: 10).",
    )

    parser.add_argument(
        "--epochs-per-stage",
        type=int,
        default=2,
        help=(
            "Artificial epochs generated for each of five stages "
            "(default: 2)."
        ),
    )

    parser.add_argument(
        "--sfreq",
        type=float,
        default=128.0,
        help=(
            "Synthetic sampling frequency in Hz "
            "(default: 128)."
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=256,
        help=(
            "Samples in each artificial epoch "
            "(default: 256)."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help=(
            "Example-only random seed "
            "(default: 2026)."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing synthetic FIF files.",
    )

    return parser


def main() -> None:
    parser = build_parser()

    args = parser.parse_args()

    generated = generate_synthetic_psg_files(
        output_dir=args.output_dir,
        n_participants=args.participants,
        random_state=args.seed,
        epochs_per_stage=args.epochs_per_stage,
        sfreq=args.sfreq,
        n_samples=args.samples,
        overwrite=args.overwrite,
    )

    print("")
    print("Synthetic PSG generation complete.")
    print(
        f"Generated {len(generated)} artificial FIF file(s):"
    )

    for path in generated:
        print(
            f"  {path}"
        )

    print("")
    print(
        "These files are synthetic software-demo artifacts only."
    )


if __name__ == "__main__":
    main()