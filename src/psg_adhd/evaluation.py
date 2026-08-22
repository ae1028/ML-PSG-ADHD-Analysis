"""Historical-compatible model evaluation and feature importance.

This module reconstructs the post-cross-validation behavior encoded in
the historical ``ADHD_Final_Code.py``.

Historical behaviors preserved
------------------------------
1. Feature CSV files are discovered using ``os.listdir``.
2. Every filename ending in ``.csv`` is loaded.
3. CSV tables are concatenated with ``ignore_index=True``.
4. If no CSV files exist, the historical loader returns ``None``.
5. The best Random Forest estimator returned from the FINAL outer fold
   is refit on the complete augmented dataset.
6. Random Forest impurity-based ``feature_importances_`` are then read.
7. Sleep-stage feature names are changed for presentation.
8. Permutation importance is calculated using:

       n_repeats=10
       random_state=42

9. Permutation results are sorted by descending mean importance.

These functions preserve historical analytical behavior while keeping
plotting separate from numerical calculations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


HISTORICAL_STAGE_RENAME = {
    "Sleep_ID_1": "Wake",
    "Sleep_ID_2": "Sleep Stage 1",
    "Sleep_ID_3": "Sleep Stage 2",
    "Sleep_ID_4": "Sleep Stage 3-4",
    "Sleep_ID_5": "Rapid Eye Movement",
}


def load_features(
    directory: str | Path,
) -> pd.DataFrame | None:
    """Load and concatenate historical feature CSV files.

    This reproduces the historical implementation::

        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".csv")
        ]

        df_list = [
            pd.read_csv(file)
            for file in files
        ]

        if not df_list:
            return None

        full_df = pd.concat(
            df_list,
            ignore_index=True,
        )

    Notes
    -----
    No additional sorting rule is introduced. The order follows the
    result returned by ``os.listdir`` as in the historical script.
    """
    directory = str(directory)

    files = [
        os.path.join(
            directory,
            filename,
        )
        for filename in os.listdir(
            directory
        )
        if filename.endswith(".csv")
    ]

    df_list = [
        pd.read_csv(file_path)
        for file_path in files
    ]

    if not df_list:
        return None

    return pd.concat(
        df_list,
        ignore_index=True,
    )


def apply_historical_stage_names(
    features_df: pd.DataFrame,
) -> list[str]:
    """Apply historical sleep-stage names to the feature table.

    The historical script used ``DataFrame.rename(..., inplace=True)``
    before plotting feature importances. That mutation is intentionally
    preserved here.

    Parameters
    ----------
    features_df
        Participant feature table.

    Returns
    -------
    list[str]
        Feature names after the historical rename operation.
    """
    features_df.rename(
        columns=HISTORICAL_STAGE_RENAME,
        inplace=True,
    )

    return list(
        features_df.columns[1:-1]
    )


def refit_and_compute_importances(
    best_rf_model,
    X_combined: np.ndarray,
    y_combined: np.ndarray,
    features_df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    list[str],
    pd.DataFrame,
]:
    """Reproduce the historical post-CV feature-importance workflow.

    Historical sequence
    -------------------
    1. Refit the best estimator from the final outer fold on the full
       augmented dataset.
    2. Read Random Forest ``feature_importances_``.
    3. Rename sleep-stage columns for presentation.
    4. Calculate permutation importance on that same full augmented
       dataset using ``n_repeats=10`` and ``random_state=42``.
    5. Sort permutation importance by decreasing mean value.

    Parameters
    ----------
    best_rf_model
        Best Random Forest estimator returned from the final outer fold.

    X_combined
        Complete historical sign-inverted augmented feature matrix.

    y_combined
        Corresponding duplicated labels.

    features_df
        Historical participant-level feature DataFrame. Its feature
        columns are renamed in place, matching the original script.

    Returns
    -------
    impurity_importances
        Random Forest impurity-based feature importance vector.

    feature_names
        Historical presentation names for the sleep-stage features.

    permutation_df
        DataFrame containing permutation-importance mean and standard
        deviation, sorted by descending mean importance.
    """
    best_rf_model.fit(
        X_combined,
        y_combined,
    )

    impurity_importances = np.asarray(
        best_rf_model.feature_importances_,
        dtype=float,
    )

    feature_names = (
        apply_historical_stage_names(
            features_df
        )
    )

    perm_importance = (
        permutation_importance(
            best_rf_model,
            X_combined,
            y_combined,
            n_repeats=10,
            random_state=42,
        )
    )

    permutation_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance Mean": (
                perm_importance.importances_mean
            ),
            "Importance Std": (
                perm_importance.importances_std
            ),
        }
    ).sort_values(
        by="Importance Mean",
        ascending=False,
    )

    return (
        impurity_importances,
        feature_names,
        permutation_df,
    )