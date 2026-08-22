"""Tests for reconstructed FIF batch-processing orchestration."""

from glob import glob
from pathlib import Path

import pandas as pd
import pytest

import psg_adhd.batch_processing as batch_processing


def test_iter_file_batches_preserves_order():
    paths = [
        Path(f"subject_{index}.fif")
        for index in range(7)
    ]

    batches = list(
        batch_processing.iter_file_batches(
            paths,
            batch_size=3,
        )
    )

    assert batches == [
        paths[0:3],
        paths[3:6],
        paths[6:7],
    ]


def test_iter_file_batches_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        list(
            batch_processing.iter_file_batches(
                ["a.fif"],
                batch_size=0,
            )
        )


def test_discovery_matches_direct_glob_behavior(tmp_path):
    for filename in [
        "a.fif",
        "b.fif",
        "ignore.txt",
        "c.fif",
    ]:
        (
            tmp_path / filename
        ).write_text(
            "test",
            encoding="utf-8",
        )

    expected = [
        Path(path)
        for path in glob(
            str(tmp_path / "*.fif")
        )
    ]

    actual = (
        batch_processing.discover_epoch_files(
            tmp_path
        )
    )

    assert actual == expected


def test_process_epoch_batch_orchestration(monkeypatch):
    loaded = []

    def fake_loader(path):
        loaded.append(Path(path))
        return f"epochs:{Path(path).name}"

    concatenated = object()

    def fake_concatenate(items):
        assert items == [
            "epochs:a.fif",
            "epochs:b.fif",
        ]
        return concatenated

    expected_table = pd.DataFrame(
        {
            "Patient_ID": ["P01"],
            "Sleep_ID_1": [0.5],
            "ADHD": ["N"],
        }
    )

    def fake_build_feature_table(
        epochs,
        id_column,
        label_column,
    ):
        assert epochs is concatenated
        assert id_column == "ID"
        assert label_column == "ADHD"
        return expected_table

    monkeypatch.setattr(
        batch_processing,
        "load_epochs_data",
        fake_loader,
    )

    monkeypatch.setattr(
        batch_processing.mne,
        "concatenate_epochs",
        fake_concatenate,
    )

    monkeypatch.setattr(
        batch_processing,
        "build_feature_table",
        fake_build_feature_table,
    )

    actual = (
        batch_processing.process_epoch_batch(
            [
                "a.fif",
                "b.fif",
            ]
        )
    )

    assert loaded == [
        Path("a.fif"),
        Path("b.fif"),
    ]

    pd.testing.assert_frame_equal(
        actual,
        expected_table,
    )


def test_empty_epoch_batch_is_rejected():
    with pytest.raises(ValueError):
        batch_processing.process_epoch_batch(
            []
        )


def test_write_feature_batches(
    tmp_path,
    monkeypatch,
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    input_dir.mkdir()

    fake_files = [
        input_dir / f"subject_{index}.fif"
        for index in range(5)
    ]

    for path in fake_files:
        path.write_text(
            "placeholder",
            encoding="utf-8",
        )

    calls = []

    def fake_process(
        file_paths,
        id_column,
        label_column,
    ):
        calls.append(
            list(file_paths)
        )

        return pd.DataFrame(
            {
                "Patient_ID": [
                    f"P{len(calls):02d}"
                ],
                "Sleep_ID_1": [
                    float(len(calls))
                ],
                "ADHD": ["N"],
            }
        )

    monkeypatch.setattr(
        batch_processing,
        "process_epoch_batch",
        fake_process,
    )

    generated = (
        batch_processing.write_feature_batches(
            input_dir=input_dir,
            output_dir=output_dir,
            batch_size=2,
        )
    )

    assert len(generated) == 3

    assert [
        path.name
        for path in generated
    ] == [
        "features_batch_1.csv",
        "features_batch_2.csv",
        "features_batch_3.csv",
    ]

    assert all(
        path.exists()
        for path in generated
    )

    assert [
        len(batch)
        for batch in calls
    ] == [
        2,
        2,
        1,
    ]


def test_write_feature_batches_refuses_overwrite(
    tmp_path,
    monkeypatch,
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    input_dir.mkdir()
    output_dir.mkdir()

    (
        input_dir / "subject.fif"
    ).write_text(
        "placeholder",
        encoding="utf-8",
    )

    existing = (
        output_dir / "features_batch_1.csv"
    )

    existing.write_text(
        "existing",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        batch_processing.write_feature_batches(
            input_dir=input_dir,
            output_dir=output_dir,
            overwrite=False,
        )


def test_write_feature_batches_requires_input_files(
    tmp_path,
):
    with pytest.raises(FileNotFoundError):
        batch_processing.write_feature_batches(
            input_dir=tmp_path,
            output_dir=tmp_path / "output",
        )