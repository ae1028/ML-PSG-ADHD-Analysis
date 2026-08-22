"""Tests for complete reconstructed modeling orchestration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import psg_adhd.model_workflow as workflow


def make_features():
    return pd.DataFrame(
        {
            "Patient_ID": [
                "P01",
                "P02",
                "P03",
            ],
            "Sleep_ID_1": [
                1.0,
                2.0,
                3.0,
            ],
            "Sleep_ID_2": [
                4.0,
                5.0,
                6.0,
            ],
            "ADHD": [
                "N",
                "Y",
                "N",
            ],
        }
    )


def test_workflow_builds_expected_result_tables(
    monkeypatch,
):
    features_df = make_features()

    X_combined = np.array(
        [
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
            [-1.0, -4.0],
            [-2.0, -5.0],
            [-3.0, -6.0],
        ]
    )

    y_combined = np.array(
        [
            0,
            1,
            0,
            0,
            1,
            0,
        ]
    )

    fake_model = object()

    monkeypatch.setattr(
        workflow,
        "prepare_data",
        lambda df: (
            X_combined,
            y_combined,
            X_combined[:3],
            y_combined[:3],
        ),
    )

    monkeypatch.setattr(
        workflow,
        "perform_nested_cross_validation",
        lambda X, y, param_grid, n_splits: (
            0.70,
            0.60,
            0.80,
            0.68,
            [0.60, 0.80],
            [0.50, 0.70],
            [0.70, 0.90],
            [0.58, 0.78],
            fake_model,
        ),
    )

    permutation_df = pd.DataFrame(
        {
            "Feature": [
                "Sleep Stage 1",
                "Wake",
            ],
            "Importance Mean": [
                0.4,
                0.2,
            ],
            "Importance Std": [
                0.04,
                0.02,
            ],
        }
    )

    monkeypatch.setattr(
        workflow,
        "refit_and_compute_importances",
        lambda model, X, y, df: (
            np.array(
                [0.25, 0.75]
            ),
            [
                "Wake",
                "Sleep Stage 1",
            ],
            permutation_df,
        ),
    )

    result = workflow.run_modeling_workflow(
        features_df,
        param_grid={
            "n_estimators": [100]
        },
        n_splits=2,
    )

    assert list(
        result.summary_metrics["Metric"]
    ) == [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]

    np.testing.assert_allclose(
        result.summary_metrics["Mean"],
        np.array(
            [
                0.70,
                0.60,
                0.80,
                0.68,
            ]
        ),
    )

    assert list(
        result.fold_metrics.columns
    ) == [
        "Fold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]

    assert list(
        result.fold_metrics["Fold"]
    ) == [
        1,
        2,
    ]

    assert list(
        result.impurity_importance[
            "Feature"
        ]
    ) == [
        "Wake",
        "Sleep Stage 1",
    ]

    np.testing.assert_array_equal(
        result.impurity_importance[
            "Importance"
        ],
        np.array(
            [0.25, 0.75]
        ),
    )

    pd.testing.assert_frame_equal(
        result.permutation_importance,
        permutation_df,
    )

    assert result.best_model is fake_model


def test_workflow_does_not_rename_input_dataframe(
    monkeypatch,
):
    features_df = make_features()

    original_columns = list(
        features_df.columns
    )

    monkeypatch.setattr(
        workflow,
        "prepare_data",
        lambda df: (
            np.ones((6, 2)),
            np.array(
                [0, 1, 0, 0, 1, 0]
            ),
            np.ones((3, 2)),
            np.array(
                [0, 1, 0]
            ),
        ),
    )

    monkeypatch.setattr(
        workflow,
        "perform_nested_cross_validation",
        lambda *args, **kwargs: (
            1.0,
            1.0,
            1.0,
            1.0,
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            object(),
        ),
    )

    def fake_importance(
        model,
        X,
        y,
        df,
    ):
        df.rename(
            columns={
                "Sleep_ID_1": "Wake",
            },
            inplace=True,
        )

        return (
            np.array(
                [0.5, 0.5]
            ),
            [
                "Wake",
                "Sleep_ID_2",
            ],
            pd.DataFrame(
                {
                    "Feature": [],
                    "Importance Mean": [],
                    "Importance Std": [],
                }
            ),
        )

    monkeypatch.setattr(
        workflow,
        "refit_and_compute_importances",
        fake_importance,
    )

    workflow.run_modeling_workflow(
        features_df,
        param_grid={
            "n_estimators": [100]
        },
    )

    assert list(
        features_df.columns
    ) == original_columns


def test_empty_feature_table_is_rejected():
    with pytest.raises(ValueError):
        workflow.run_modeling_workflow(
            pd.DataFrame()
        )


def test_save_modeling_results_creates_four_csvs(
    tmp_path,
):
    result = workflow.ModelingWorkflowResult(
        summary_metrics=pd.DataFrame(
            {
                "Metric": ["Accuracy"],
                "Mean": [0.72],
            }
        ),
        fold_metrics=pd.DataFrame(
            {
                "Fold": [1],
                "Accuracy": [0.72],
            }
        ),
        impurity_importance=pd.DataFrame(
            {
                "Feature": ["Wake"],
                "Importance": [0.5],
            }
        ),
        permutation_importance=pd.DataFrame(
            {
                "Feature": ["Wake"],
                "Importance Mean": [0.4],
                "Importance Std": [0.1],
            }
        ),
        best_model=object(),
    )

    generated = workflow.save_modeling_results(
        result,
        tmp_path,
    )

    assert [
        path.name
        for path in generated
    ] == [
        "summary_metrics.csv",
        "fold_metrics.csv",
        "impurity_feature_importance.csv",
        "permutation_feature_importance.csv",
    ]

    assert all(
        path.exists()
        for path in generated
    )


def test_save_modeling_results_refuses_overwrite(
    tmp_path,
):
    existing = (
        tmp_path /
        "summary_metrics.csv"
    )

    existing.write_text(
        "existing",
        encoding="utf-8",
    )

    result = workflow.ModelingWorkflowResult(
        summary_metrics=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        impurity_importance=pd.DataFrame(),
        permutation_importance=pd.DataFrame(),
        best_model=object(),
    )

    with pytest.raises(
        FileExistsError
    ):
        workflow.save_modeling_results(
            result,
            tmp_path,
            overwrite=False,
        )