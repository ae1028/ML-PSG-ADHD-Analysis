"""Tests for public synthetic-demo orchestration."""

from pathlib import Path

import pandas as pd
import pytest

import psg_adhd.demo as demo


def test_demo_parameter_grid_is_lightweight():
    assert demo.DEMO_PARAMETER_GRID == {
        "n_estimators": [20],
        "max_depth": [5],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
    }


def test_demo_requires_at_least_ten_participants(
    tmp_path,
):
    with pytest.raises(
        ValueError
    ):
        demo.run_synthetic_demo(
            work_dir=tmp_path,
            n_participants=9,
        )


def test_demo_orchestration(
    tmp_path,
    monkeypatch,
):
    psg_paths = [
        tmp_path / "psg" / "SYN001-epo.fif"
    ]

    feature_paths = [
        tmp_path
        / "features"
        / "features_batch_1.csv"
    ]

    result_paths = [
        tmp_path
        / "results"
        / "summary_metrics.csv"
    ]

    features_df = pd.DataFrame(
        {
            "Patient_ID": [
                "SYN001"
            ],
            "Sleep_ID_1": [
                0.1
            ],
            "Sleep_ID_2": [
                0.2
            ],
            "Sleep_ID_3": [
                0.3
            ],
            "Sleep_ID_4": [
                0.4
            ],
            "Sleep_ID_5": [
                0.5
            ],
            "ADHD": [
                "N"
            ],
        }
    )

    summary = pd.DataFrame(
        {
            "Metric": [
                "Accuracy"
            ],
            "Mean": [
                0.5
            ],
        }
    )

    fake_model_result = type(
        "FakeModelResult",
        (),
        {
            "summary_metrics": summary
        },
    )()

    calls = {}

    def fake_generate(**kwargs):
        calls["generate"] = kwargs
        return psg_paths

    def fake_extract(**kwargs):
        calls["extract"] = kwargs
        return feature_paths

    def fake_load(path):
        calls["load"] = path
        return features_df

    def fake_model(**kwargs):
        calls["model"] = kwargs
        return fake_model_result

    def fake_save(**kwargs):
        calls["save"] = kwargs
        return result_paths

    monkeypatch.setattr(
        demo,
        "generate_synthetic_psg_files",
        fake_generate,
    )

    monkeypatch.setattr(
        demo,
        "write_feature_batches",
        fake_extract,
    )

    monkeypatch.setattr(
        demo,
        "load_features",
        fake_load,
    )

    monkeypatch.setattr(
        demo,
        "run_modeling_workflow",
        fake_model,
    )

    monkeypatch.setattr(
        demo,
        "save_modeling_results",
        fake_save,
    )

    result = demo.run_synthetic_demo(
        work_dir=tmp_path,
        n_participants=10,
        random_state=2026,
        overwrite=True,
    )

    assert result.psg_files == psg_paths
    assert result.feature_files == feature_paths
    assert result.result_files == result_paths

    pd.testing.assert_frame_equal(
        result.summary_metrics,
        summary,
    )

    assert calls["generate"][
        "n_participants"
    ] == 10

    assert calls["generate"][
        "random_state"
    ] == 2026

    assert calls["generate"][
        "epochs_per_stage"
    ] == 1

    assert calls["extract"][
        "batch_size"
    ] == 10

    assert calls["extract"][
        "pattern"
    ] == "*-epo.fif"

    assert calls["model"][
        "param_grid"
    ] == demo.DEMO_PARAMETER_GRID

    assert calls["model"][
        "n_splits"
    ] == 2