"""End-to-end synthetic demonstration of the reconstructed repository.

This module provides a public software smoke test requiring no clinical
or participant-derived data.

The demonstration performs:

1. fully artificial PSG generation;
2. MNE Epochs FIF serialization;
3. historical-compatible graph-feature extraction;
4. participant feature-table generation;
5. a lightweight Random Forest modeling smoke test;
6. numerical result-table export.

Important
---------
The synthetic signals, labels, extracted values, performance metrics,
and feature importances produced by this demonstration have no clinical
or scientific interpretation.

The modeling smoke test intentionally uses a much smaller hyperparameter
grid than the historical experiment so the public example runs quickly.
It is NOT a published-result reproduction run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .batch_processing import write_feature_batches
from .evaluation import load_features
from .model_workflow import (
    ModelingWorkflowResult,
    run_modeling_workflow,
    save_modeling_results,
)
from .synthetic_psg import generate_synthetic_psg_files


DEMO_PARAMETER_GRID = {
    "n_estimators": [20],
    "max_depth": [5],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
}


@dataclass
class SyntheticDemoResult:
    """Artifacts produced by one synthetic demonstration run."""

    psg_files: list[Path]
    feature_files: list[Path]
    result_files: list[Path]
    summary_metrics: pd.DataFrame


def run_synthetic_demo(
    work_dir: str | Path,
    n_participants: int = 10,
    random_state: int = 2026,
    overwrite: bool = False,
) -> SyntheticDemoResult:
    """Run the complete synthetic software demonstration.

    Parameters
    ----------
    work_dir
        Root directory for generated demo artifacts.

    n_participants
        Number of artificial participants.

    random_state
        Seed used only for synthetic PSG generation.

    overwrite
        Whether existing generated artifacts may be replaced.

    Returns
    -------
    SyntheticDemoResult
        Generated artifact paths and modeling summary metrics.

    Notes
    -----
    The historical-compatible modeling implementation itself retains its
    original unseeded shuffle, outer KFold, and Random Forest behavior.

    Therefore modeling metrics may differ between repeated demo runs even
    though the synthetic PSG signals are generated deterministically.
    """
    if n_participants < 10:
        raise ValueError(
            "The public modeling demonstration requires at least "
            "10 synthetic participants."
        )

    work_dir = Path(work_dir)

    psg_dir = work_dir / "psg"
    feature_dir = work_dir / "features"
    result_dir = work_dir / "results"

    psg_files = generate_synthetic_psg_files(
        output_dir=psg_dir,
        n_participants=n_participants,
        random_state=random_state,
        epochs_per_stage=1,
        sfreq=64.0,
        n_samples=128,
        overwrite=overwrite,
    )

    feature_files = write_feature_batches(
        input_dir=psg_dir,
        output_dir=feature_dir,
        batch_size=10,
        pattern="*-epo.fif",
        output_prefix="features_batch",
        overwrite=overwrite,
        id_column="ID",
        label_column="ADHD",
    )

    features_df = load_features(
        feature_dir
    )

    if features_df is None or features_df.empty:
        raise RuntimeError(
            "Synthetic feature extraction produced no usable data."
        )

    modeling_result: ModelingWorkflowResult = (
        run_modeling_workflow(
            features_df=features_df,
            param_grid=DEMO_PARAMETER_GRID,
            n_splits=2,
        )
    )

    result_files = save_modeling_results(
        result=modeling_result,
        output_dir=result_dir,
        overwrite=overwrite,
    )

    return SyntheticDemoResult(
        psg_files=psg_files,
        feature_files=feature_files,
        result_files=result_files,
        summary_metrics=modeling_result.summary_metrics,
    )