"""Historical-compatible ADHD machine-learning workflow.

This module reconstructs the machine-learning behavior encoded in the
historical:

    ADHD_Final_Code.py

preserved under:

    reference_implementations/published_pipeline/

The goal of these functions is behavioral fidelity to the historical
implementation.

Important historical behaviors preserved
----------------------------------------
1. Features are all columns between ``Patient_ID`` and ``ADHD``.
2. ``ADHD == "Y"`` is encoded as class 1; all other values become 0.
3. Feature-value sign inversion is applied before cross-validation:

       X_augmented = [X; -X]
       y_augmented = [y; y]

4. The augmented dataset is shuffled without ``random_state``.
5. Outer KFold uses ``shuffle=True`` without ``random_state``.
6. RandomForestClassifier is instantiated without ``random_state``.
7. Inner GridSearchCV uses ``cv=5``.
8. Grid search evaluates accuracy, precision, recall, and F1.
9. GridSearchCV refits according to accuracy.
10. The model returned by nested cross-validation is the best estimator
    from the final outer fold.

Because several historical stochastic operations were unseeded,
repeated executions are not guaranteed to produce identical folds,
hyperparameters, or performance values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
)
from sklearn.utils import shuffle


HISTORICAL_PARAMETER_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}


def prepare_data(
    features_df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Prepare and augment the participant feature table.

    This reproduces the historical implementation::

        features = features_df.iloc[:, 1:-1].values

        labels = (
            features_df["ADHD"] == "Y"
        ).astype(int).values

        X_augmented = np.vstack([
            features,
            -features,
        ])

        y_augmented = np.hstack([
            labels,
            labels,
        ])

        X_combined, y_combined = shuffle(
            X_augmented,
            y_augmented,
        )

    Parameters
    ----------
    features_df
        Participant-level feature table whose first column is the
        participant identifier and whose final column is ``ADHD``.

    Returns
    -------
    X_combined
        Sign-inverted augmented and shuffled feature matrix.

    y_combined
        Duplicated and correspondingly shuffled labels.

    X
        Original, non-augmented feature matrix.

    y
        Original binary labels.

    Notes
    -----
    The historical shuffle did not specify ``random_state``.
    This behavior is intentionally preserved.
    """
    features = (
        features_df.iloc[:, 1:-1]
        .values
    )

    labels = (
        (features_df["ADHD"] == "Y")
        .astype(int)
        .values
    )

    X_augmented = np.vstack(
        [
            features,
            -features,
        ]
    )

    y_augmented = np.hstack(
        [
            labels,
            labels,
        ]
    )

    X_combined, y_combined = shuffle(
        X_augmented,
        y_augmented,
    )

    return (
        X_combined,
        y_combined,
        features_df.iloc[:, 1:-1].values,
        labels,
    )


def historical_scoring() -> dict[str, Any]:
    """Return the historical GridSearchCV scoring dictionary."""
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score
        ),
        "recall": make_scorer(
            recall_score
        ),
        "f1": make_scorer(
            f1_score
        ),
    }


def perform_nested_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict[str, list[Any]] | None = None,
    n_splits: int = 5,
):
    """Run the historical nested Random Forest cross-validation.

    Parameters
    ----------
    X
        Augmented and shuffled feature matrix.

    y
        Corresponding binary labels.

    param_grid
        Random Forest parameter grid. If omitted, the exact historical
        parameter grid is used.

    n_splits
        Number of outer KFold splits. Historical default: 5.

    Returns
    -------
    avg_accuracy
    avg_precision
    avg_recall
    avg_f1
    outer_accuracy_scores
    outer_precision_scores
    outer_recall_scores
    outer_f1_scores
    best_rf_model

    Notes
    -----
    Historical stochastic behavior is preserved:

    - outer KFold has ``shuffle=True`` and no random seed;
    - RandomForestClassifier has no random seed;
    - the inner GridSearchCV receives ``cv=5`` as an integer;
    - the returned best model is the estimator from the final
      outer fold.
    """
    if param_grid is None:
        param_grid = HISTORICAL_PARAMETER_GRID

    outer_kf = KFold(
        n_splits=n_splits,
        shuffle=True,
    )

    outer_accuracy_scores = []
    outer_precision_scores = []
    outer_recall_scores = []
    outer_f1_scores = []

    best_rf_model = None

    for train_val_index, test_index in outer_kf.split(X):
        X_train_val = X[train_val_index]
        X_test = X[test_index]

        y_train_val = y[train_val_index]
        y_test = y[test_index]

        scoring = historical_scoring()

        rf_classifier = (
            RandomForestClassifier()
        )

        grid_search = GridSearchCV(
            estimator=rf_classifier,
            param_grid=param_grid,
            cv=5,
            scoring=scoring,
            refit="accuracy",
            n_jobs=-1,
        )

        grid_search.fit(
            X_train_val,
            y_train_val,
        )

        best_rf_model = (
            grid_search.best_estimator_
        )

        y_test_pred = (
            best_rf_model.predict(
                X_test
            )
        )

        outer_accuracy_scores.append(
            accuracy_score(
                y_test,
                y_test_pred,
            )
        )

        outer_precision_scores.append(
            precision_score(
                y_test,
                y_test_pred,
            )
        )

        outer_recall_scores.append(
            recall_score(
                y_test,
                y_test_pred,
            )
        )

        outer_f1_scores.append(
            f1_score(
                y_test,
                y_test_pred,
            )
        )

    avg_accuracy = np.mean(
        outer_accuracy_scores
    )

    avg_precision = np.mean(
        outer_precision_scores
    )

    avg_recall = np.mean(
        outer_recall_scores
    )

    avg_f1 = np.mean(
        outer_f1_scores
    )

    return (
        avg_accuracy,
        avg_precision,
        avg_recall,
        avg_f1,
        outer_accuracy_scores,
        outer_precision_scores,
        outer_recall_scores,
        outer_f1_scores,
        best_rf_model,
    )