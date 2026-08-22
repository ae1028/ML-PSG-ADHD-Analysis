"""Tests for public synthetic example data."""

import pandas as pd
import pytest

from psg_adhd.example_data import (
    SYNTHETIC_FEATURE_COLUMNS,
    generate_synthetic_feature_table,
    write_synthetic_feature_batches,
)


def test_synthetic_table_schema():
    table = (
        generate_synthetic_feature_table(
            n_participants=20,
            random_state=2026,
        )
    )

    assert list(
        table.columns
    ) == [
        "Patient_ID",
        *SYNTHETIC_FEATURE_COLUMNS,
        "ADHD",
    ]

    assert len(table) == 20


def test_synthetic_table_is_deterministic():
    first = (
        generate_synthetic_feature_table(
            n_participants=20,
            random_state=2026,
        )
    )

    second = (
        generate_synthetic_feature_table(
            n_participants=20,
            random_state=2026,
        )
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_synthetic_ids_and_labels_are_artificial():
    table = (
        generate_synthetic_feature_table(
            n_participants=6,
            random_state=2026,
        )
    )

    assert list(
        table["Patient_ID"]
    ) == [
        "SYN001",
        "SYN002",
        "SYN003",
        "SYN004",
        "SYN005",
        "SYN006",
    ]

    assert list(
        table["ADHD"]
    ) == [
        "N",
        "Y",
        "N",
        "Y",
        "N",
        "Y",
    ]


def test_synthetic_batch_writer(
    tmp_path,
):
    generated = (
        write_synthetic_feature_batches(
            output_dir=tmp_path,
            n_participants=20,
            batch_size=10,
            random_state=2026,
        )
    )

    assert [
        path.name
        for path in generated
    ] == [
        "features_batch_1.csv",
        "features_batch_2.csv",
    ]

    first = pd.read_csv(
        generated[0]
    )

    second = pd.read_csv(
        generated[1]
    )

    assert len(first) == 10
    assert len(second) == 10

    assert list(
        first.columns
    ) == [
        "Patient_ID",
        "Sleep_ID_1",
        "Sleep_ID_2",
        "Sleep_ID_3",
        "Sleep_ID_4",
        "Sleep_ID_5",
        "ADHD",
    ]


def test_synthetic_generator_rejects_too_few_participants():
    with pytest.raises(
        ValueError
    ):
        generate_synthetic_feature_table(
            n_participants=1
        )