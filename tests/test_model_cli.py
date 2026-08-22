"""Tests for the reconstructed modeling CLI."""

from pathlib import Path

import pandas as pd
import pytest

import psg_adhd.model_cli as model_cli


def test_model_cli_defaults():
    parser = (
        model_cli.build_parser()
    )

    args = parser.parse_args(
        [
            "--feature-dir",
            "features",
            "--output-dir",
            "results",
        ]
    )

    assert (
        args.feature_dir
        == Path("features")
    )

    assert (
        args.output_dir
        == Path("results")
    )

    assert args.outer_splits == 5
    assert args.overwrite is False


def test_model_cli_delegates_to_workflow(
    monkeypatch,
    tmp_path,
):
    feature_dir = (
        tmp_path / "features"
    )

    output_dir = (
        tmp_path / "results"
    )

    fake_result = type(
        "FakeResult",
        (),
        {
            "summary_metrics": pd.DataFrame(
                {
                    "Metric": [
                        "Accuracy"
                    ],
                    "Mean": [
                        0.72
                    ],
                }
            )
        },
    )()

    workflow_call = {}
    save_call = {}

    def fake_run_modeling_from_directory(
        feature_directory,
        n_splits,
    ):
        workflow_call[
            "feature_directory"
        ] = feature_directory

        workflow_call[
            "n_splits"
        ] = n_splits

        return fake_result

    def fake_save_modeling_results(
        result,
        output_dir,
        overwrite,
    ):
        save_call[
            "result"
        ] = result

        save_call[
            "output_dir"
        ] = output_dir

        save_call[
            "overwrite"
        ] = overwrite

        return [
            Path(output_dir)
            / "summary_metrics.csv"
        ]

    monkeypatch.setattr(
        model_cli,
        "run_modeling_from_directory",
        fake_run_modeling_from_directory,
    )

    monkeypatch.setattr(
        model_cli,
        "save_modeling_results",
        fake_save_modeling_results,
    )

    result, generated = model_cli.run(
        [
            "--feature-dir",
            str(feature_dir),
            "--output-dir",
            str(output_dir),
            "--outer-splits",
            "4",
            "--overwrite",
        ]
    )

    assert result is fake_result

    assert workflow_call == {
        "feature_directory": feature_dir,
        "n_splits": 4,
    }

    assert save_call == {
        "result": fake_result,
        "output_dir": output_dir,
        "overwrite": True,
    }

    assert len(generated) == 1


def test_model_cli_rejects_invalid_outer_splits():
    with pytest.raises(
        SystemExit
    ):
        model_cli.run(
            [
                "--feature-dir",
                "features",
                "--output-dir",
                "results",
                "--outer-splits",
                "1",
            ]
        )