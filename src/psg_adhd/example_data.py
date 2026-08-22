"""Synthetic public example data for repository demonstrations.

The functions in this module generate completely artificial
participant-level feature tables that match the schema expected by the
reconstructed ML-PSG-ADHD modeling workflow.

IMPORTANT
---------
The generated values:

- are not derived from any study participant;
- are not transformed versions of clinical PSG data;
- are not intended to approximate published biological measurements;
- have no diagnostic or clinical interpretation.

They exist only for software testing, documentation, and demonstration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SYNTHETIC_FEATURE_COLUMNS = [
    "Sleep_ID_1",
    "Sleep_ID_2",
    "Sleep_ID_3",
    "Sleep_ID_4",
    "Sleep_ID_5",
]


def generate_synthetic_feature_table(
    n_participants: int = 20,
    random_state: int = 2026,
) -> pd.DataFrame:
    """Generate a deterministic artificial participant feature table.

    Parameters
    ----------
    n_participants
        Number of artificial participants.

    random_state
        Seed used solely to make the public example reproducible.

    Returns
    -------
    pandas.DataFrame
        Synthetic table with the historical modeling schema:

        ``Patient_ID | Sleep_ID_1 ... Sleep_ID_5 | ADHD``

    Notes
    -----
    A small artificial class-dependent offset is deliberately introduced
    so that example modeling commands have nontrivial structure to
    process.

    This pattern is a software-demo construction only and must not be
    interpreted as a representation of ADHD physiology or of the
    original clinical dataset.
    """
    if n_participants < 2:
        raise ValueError(
            "n_participants must be at least 2."
        )

    rng = np.random.default_rng(
        random_state
    )

    labels = np.array(
        [
            "N" if index % 2 == 0 else "Y"
            for index in range(
                n_participants
            )
        ]
    )

    class_indicator = (
        labels == "Y"
    ).astype(float)

    base = rng.normal(
        loc=0.45,
        scale=0.07,
        size=(
            n_participants,
            len(SYNTHETIC_FEATURE_COLUMNS),
        ),
    )

    artificial_offsets = np.array(
        [
            0.03,
            -0.02,
            0.04,
            -0.01,
            0.02,
        ]
    )

    features = (
        base
        + class_indicator[:, None]
        * artificial_offsets[None, :]
    )

    # Keep example values positive and compact.
    features = np.clip(
        features,
        0.05,
        0.95,
    )

    table = pd.DataFrame(
        features,
        columns=SYNTHETIC_FEATURE_COLUMNS,
    )

    table.insert(
        0,
        "Patient_ID",
        [
            f"SYN{index:03d}"
            for index in range(
                1,
                n_participants + 1,
            )
        ],
    )

    table["ADHD"] = labels

    return table


def write_synthetic_feature_batches(
    output_dir: str | Path,
    n_participants: int = 20,
    batch_size: int = 10,
    random_state: int = 2026,
    overwrite: bool = False,
) -> list[Path]:
    """Generate and write artificial feature CSV batches.

    Output names follow the reconstructed feature-extraction interface:

    ``features_batch_1.csv``
    ``features_batch_2.csv``
    ...

    These files contain synthetic demonstration data only.
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be a positive integer."
        )

    table = generate_synthetic_feature_table(
        n_participants=n_participants,
        random_state=random_state,
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated = []

    for batch_number, start in enumerate(
        range(
            0,
            len(table),
            batch_size,
        ),
        start=1,
    ):
        batch = table.iloc[
            start : start + batch_size
        ].copy()

        output_path = (
            output_dir
            / f"features_batch_{batch_number}.csv"
        )

        if (
            output_path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"Output already exists: {output_path}"
            )

        batch.to_csv(
            output_path,
            index=False,
        )

        generated.append(
            output_path
        )

    return generated