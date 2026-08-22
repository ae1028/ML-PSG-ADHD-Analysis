"""Historical-compatible participant-level PSG feature extraction.

This module connects the reconstructed data-selection and graph-feature
utilities into the participant-by-sleep-stage feature table used by the
historical ML-PSG-ADHD workflow.

The implementation is intentionally traceable to:

    reference_implementations/published_pipeline/
    Feature extraction (Sleep stages).py

At this stage the goal is behavioral fidelity, not methodological
redesign.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .data import (
    get_patient_ids,
    get_patient_label,
    get_sleep_ids,
    impute_missing_stage_features,
    select_patient_epochs,
    select_sleep_stage_epochs,
)
from .graph_features import compute_stage_graph_feature


def _evoked_epoch_arrays(stage_epochs) -> list[np.ndarray]:
    """Convert historical ``iter_evoked()`` output to epoch arrays.

    The original implementation used::

        for epoch in sleep_epochs.iter_evoked():
            correlation_matrix = np.abs(
                np.corrcoef(epoch.data)
            )

    This helper preserves that route rather than replacing it with
    direct access to the Epochs data array.
    """
    return [
        np.asarray(evoked.data, dtype=float)
        for evoked in stage_epochs.iter_evoked()
    ]


def compute_participant_stage_features(
    epochs,
    patient_ids: Sequence | None = None,
    sleep_ids: Sequence | None = None,
    id_column: str = "ID",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute historical graph features for participant-stage pairs.

    Parameters
    ----------
    epochs
        Concatenated MNE Epochs object containing participant metadata
        and sleep-stage event codes.

    patient_ids
        Participant IDs to process. If omitted, values are obtained
        using the historical ``np.unique(metadata["ID"])`` behavior.

    sleep_ids
        Sleep-stage event IDs to process. If omitted, values are
        obtained using the historical
        ``np.unique(events[:, 2])`` behavior.

    id_column
        Participant-ID metadata column. Historical default is ``ID``.

    Returns
    -------
    feature_matrix
        Imputed participant-by-sleep-stage feature matrix.

    patient_ids
        Ordered participant IDs.

    sleep_ids
        Ordered sleep-stage IDs.

    Notes
    -----
    Missing participant-stage combinations are initially represented
    as NaN.

    After all participants are processed, missing values are replaced
    using the historical sleep-stage mean imputation performed across
    participants in the current batch.
    """
    if patient_ids is None:
        patient_ids = get_patient_ids(
            epochs,
            id_column=id_column,
        )

    if sleep_ids is None:
        sleep_ids = get_sleep_ids(epochs)

    patient_ids = np.asarray(patient_ids)
    sleep_ids = np.asarray(sleep_ids)

    raw_features = np.full(
        (
            len(patient_ids),
            len(sleep_ids),
        ),
        np.nan,
        dtype=float,
    )

    for patient_index, patient_id in enumerate(
        patient_ids
    ):
        patient_epochs = select_patient_epochs(
            epochs,
            patient_id,
            id_column=id_column,
        )

        for sleep_index, sleep_id in enumerate(
            sleep_ids
        ):
            stage_epochs = select_sleep_stage_epochs(
                patient_epochs,
                sleep_id,
            )

            epoch_arrays = _evoked_epoch_arrays(
                stage_epochs
            )

            if not epoch_arrays:
                # Historical implementation appended np.nan
                # when the participant did not contain this stage.
                continue

            feature, _ = compute_stage_graph_feature(
                epoch_arrays
            )

            raw_features[
                patient_index,
                sleep_index,
            ] = feature

    imputed_features, _ = (
        impute_missing_stage_features(
            raw_features,
            sleep_ids,
        )
    )

    return (
        imputed_features,
        patient_ids,
        sleep_ids,
    )


def build_feature_table(
    epochs,
    id_column: str = "ID",
    label_column: str = "ADHD",
) -> pd.DataFrame:
    """Build the historical participant-level feature table.

    The resulting schema mirrors the historical CSV construction::

        Patient_ID
        Sleep_ID_<event code>
        ...
        ADHD

    Sleep-stage columns reflect the sleep IDs present in the supplied
    Epochs object. They are not globally forced to five columns here,
    because the historical batch implementation inferred them
    independently for each concatenated batch.
    """
    patient_ids = get_patient_ids(
        epochs,
        id_column=id_column,
    )

    sleep_ids = get_sleep_ids(epochs)

    feature_matrix, patient_ids, sleep_ids = (
        compute_participant_stage_features(
            epochs,
            patient_ids=patient_ids,
            sleep_ids=sleep_ids,
            id_column=id_column,
        )
    )

    feature_columns = [
        f"Sleep_ID_{int(sleep_id)}"
        for sleep_id in sleep_ids
    ]

    feature_df = pd.DataFrame(
        feature_matrix,
        index=patient_ids,
        columns=feature_columns,
    )

    feature_df.reset_index(
        inplace=True
    )

    feature_df.rename(
        columns={
            "index": "Patient_ID"
        },
        inplace=True,
    )

    feature_df[label_column] = [
        get_patient_label(
            epochs,
            patient_id,
            id_column=id_column,
            label_column=label_column,
        )
        for patient_id in patient_ids
    ]

    return feature_df