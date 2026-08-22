"""Executable orchestration for the reconstructed ADHD modeling workflow.

This module connects the historical-compatible components implemented in:

- ``modeling.py``
- ``evaluation.py``

The scientific behavior remains delegated to those modules.

The orchestration layer provides a clean repository interface for:

1. loading feature CSV files;
2. historical sign-inversion augmentation;
3. historical nested Random Forest cross-validation;
4. aggregate and fold-level metric tables;
5. full-data refitting of the final-fold best estimator;
6. impurity-based feature importance;
7. permutation feature importance;
8. safe result-table export.

Important
---------
The historical modeling workflow contains unseeded stochastic operations:

- ``sklearn.utils.shuffle``;
- outer ``KFold(..., shuffle=True)``;
- ``RandomForestClassifier()``.

Therefore separate executions are not guaranteed to produce identical
numerical results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import (
    load_features,
    refit_and_compute_importances,
)
from .modeling import (
    HISTORICAL_PARAMETER_GRID,
    perform_nested_cross_validation,
    prepare_data,
)


@dataclass
class ModelingWorkflowResult:
    """Structured outputs from one historical-compatible modeling run."""

    summary_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    impurity_importance: pd.DataFrame
    permutation_importance: pd.DataFrame
    best_model: Any


def run_modeling_workflow(
    features_df: pd.DataFrame,
    param_grid: dict[str, list[Any]] | None = None,
    n_splits: int = 5,
) -> ModelingWorkflowResult:
    """Run the complete reconstructed historical modeling workflow.

    Parameters
    ----------
    features_df
        Participant-level feature table.

    param_grid
        Random Forest grid. If omitted, the historical grid is used.

    n_splits
        Number of outer KFold splits. Historical default: 5.

    Returns
    -------
    ModelingWorkflowResult
        Summary metrics, fold metrics, both feature-importance tables,
        and the refit best estimator.

    Notes
    -----
    ``features_df`` is copied before the historical presentation-name
    mutation performed by the feature-importance code. This keeps the
    public workflow interface from unexpectedly renaming the caller's
    original DataFrame while preserving the numerical historical
    calculations.
    """
    if features_df is None or features_df.empty:
        raise ValueError(
            "features_df must contain at least one participant."
        )

    if param_grid is None:
        param_grid = HISTORICAL_PARAMETER_GRID

    (
        X_combined,
        y_combined,
        _X_original,
        _y_original,
    ) = prepare_data(
        features_df
    )

    (
        avg_accuracy,
        avg_precision,
        avg_recall,
        avg_f1,
        accuracy_scores,
        precision_scores,
        recall_scores,
        f1_scores,
        best_rf_model,
    ) = perform_nested_cross_validation(
        X_combined,
        y_combined,
        param_grid=param_grid,
        n_splits=n_splits,
    )

    summary_metrics = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
            ],
            "Mean": [
                float(avg_accuracy),
                float(avg_precision),
                float(avg_recall),
                float(avg_f1),
            ],
        }
    )

    fold_metrics = pd.DataFrame(
        {
            "Fold": np.arange(
                1,
                len(accuracy_scores) + 1,
            ),
            "Accuracy": accuracy_scores,
            "Precision": precision_scores,
            "Recall": recall_scores,
            "F1": f1_scores,
        }
    )

    importance_features_df = (
        features_df.copy()
    )

    (
        impurity_values,
        feature_names,
        permutation_df,
    ) = refit_and_compute_importances(
        best_rf_model,
        X_combined,
        y_combined,
        importance_features_df,
    )

    impurity_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": impurity_values,
        }
    )

    return ModelingWorkflowResult(
        summary_metrics=summary_metrics,
        fold_metrics=fold_metrics,
        impurity_importance=impurity_df,
        permutation_importance=permutation_df,
        best_model=best_rf_model,
    )


def run_modeling_from_directory(
    feature_directory: str | Path,
    param_grid: dict[str, list[Any]] | None = None,
    n_splits: int = 5,
) -> ModelingWorkflowResult:
    """Load historical feature CSV files and run the modeling workflow."""
    features_df = load_features(
        feature_directory
    )

    if features_df is None or features_df.empty:
        raise FileNotFoundError(
            "No usable feature CSV files were found in "
            f"{str(feature_directory)!r}."
        )

    return run_modeling_workflow(
        features_df,
        param_grid=param_grid,
        n_splits=n_splits,
    )


def save_modeling_results(
    result: ModelingWorkflowResult,
    output_dir: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    """Write numerical modeling outputs as repository-safe CSV tables.

    Generated files
    ---------------
    - ``summary_metrics.csv``
    - ``fold_metrics.csv``
    - ``impurity_feature_importance.csv``
    - ``permutation_feature_importance.csv``

    The trained estimator itself is intentionally not serialized in this
    reconstruction step. The historical script did not establish a
    canonical persisted model artifact.
    """
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "summary_metrics.csv": (
            result.summary_metrics
        ),
        "fold_metrics.csv": (
            result.fold_metrics
        ),
        "impurity_feature_importance.csv": (
            result.impurity_importance
        ),
        "permutation_feature_importance.csv": (
            result.permutation_importance
        ),
    }

    output_paths = [
        output_dir / filename
        for filename in outputs
    ]

    if not overwrite:
        existing = [
            path
            for path in output_paths
            if path.exists()
        ]

        if existing:
            formatted = ", ".join(
                str(path)
                for path in existing
            )

            raise FileExistsError(
                "Refusing to overwrite existing result file(s): "
                f"{formatted}"
            )

    for filename, dataframe in outputs.items():
        dataframe.to_csv(
            output_dir / filename,
            index=False,
        )

    return output_paths