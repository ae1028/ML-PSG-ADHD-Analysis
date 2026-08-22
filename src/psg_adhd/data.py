"""PSG data and sleep-stage handling utilities.

This module reconstructs data-selection behavior from the historical
ML-PSG-ADHD implementation preserved under:

    reference_implementations/published_pipeline/
    Feature extraction (Sleep stages).py

Historical behavior preserved here
----------------------------------
1. Participant IDs are taken from ``epochs.metadata["ID"]``.
2. Sleep-stage IDs are taken directly from ``epochs.events[:, 2]``.
3. Participant epochs are selected using the metadata ID.
4. Sleep-stage epochs are selected using the event code.
5. Missing participant-stage features are replaced using the mean
   feature for that sleep ID among participants with available values
   in the current processing batch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mne
import numpy as np


def load_epochs_data(file_path: str | Path):
    """Load one historical MNE Epochs FIF file.

    Parameters
    ----------
    file_path
        Path to an MNE ``.fif`` epochs file.

    Returns
    -------
    mne.Epochs
        Epochs loaded with ``preload=True``.

    Notes
    -----
    This mirrors the historical call::

        mne.read_epochs(file_path, preload=True)
    """
    return mne.read_epochs(
        str(file_path),
        preload=True,
    )


def get_patient_ids(
    epochs,
    id_column: str = "ID",
) -> np.ndarray:
    """Return unique participant IDs using historical ordering logic.

    The historical implementation used::

        np.unique(concatenated_epochs.metadata["ID"])

    ``numpy.unique`` sorts the returned values.
    """
    if epochs.metadata is None:
        raise ValueError(
            "Epochs metadata is required to identify participants."
        )

    if id_column not in epochs.metadata.columns:
        raise ValueError(
            f"Required metadata column {id_column!r} was not found."
        )

    return np.unique(
        epochs.metadata[id_column].to_numpy()
    )


def get_sleep_ids(epochs) -> np.ndarray:
    """Return unique sleep-stage event codes.

    This reproduces::

        np.unique(concatenated_epochs.events[:, 2])

    Sleep-stage IDs are therefore inferred from the current Epochs
    object rather than forced to a predefined global list.
    """
    events = np.asarray(epochs.events)

    if events.ndim != 2 or events.shape[1] < 3:
        raise ValueError(
            "epochs.events must have at least three columns."
        )

    return np.unique(events[:, 2])


def select_patient_epochs(
    epochs,
    patient_id,
    id_column: str = "ID",
):
    """Select all epochs belonging to one participant.

    This reproduces the historical selection::

        epochs_data[
            epochs_data.metadata["ID"] == patient_id
        ]
    """
    if epochs.metadata is None:
        raise ValueError(
            "Epochs metadata is required to select a participant."
        )

    if id_column not in epochs.metadata.columns:
        raise ValueError(
            f"Required metadata column {id_column!r} was not found."
        )

    mask = (
        epochs.metadata[id_column].to_numpy()
        == patient_id
    )

    return epochs[mask]


def select_sleep_stage_epochs(
    patient_epochs,
    sleep_id,
):
    """Select epochs belonging to one sleep-stage event code.

    This reproduces the historical selection::

        patient_epochs[
            patient_epochs.events[:, 2] == sleep_id
        ]
    """
    events = np.asarray(patient_epochs.events)

    if events.ndim != 2 or events.shape[1] < 3:
        raise ValueError(
            "patient_epochs.events must have at least three columns."
        )

    mask = events[:, 2] == sleep_id

    return patient_epochs[mask]


def get_patient_label(
    epochs,
    patient_id,
    id_column: str = "ID",
    label_column: str = "ADHD",
):
    """Return the historical participant label.

    The historical feature-table construction used the first label
    associated with the participant::

        metadata[
            metadata["ID"] == patient_id
        ]["ADHD"].iloc[0]

    This function intentionally preserves that behavior.
    """
    if epochs.metadata is None:
        raise ValueError(
            "Epochs metadata is required to retrieve labels."
        )

    for column in (id_column, label_column):
        if column not in epochs.metadata.columns:
            raise ValueError(
                f"Required metadata column {column!r} was not found."
            )

    patient_rows = epochs.metadata[
        epochs.metadata[id_column] == patient_id
    ]

    if patient_rows.empty:
        raise ValueError(
            f"No metadata rows found for participant {patient_id!r}."
        )

    return patient_rows[label_column].iloc[0]


def impute_missing_stage_features(
    feature_matrix: np.ndarray,
    sleep_ids: Sequence,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply historical sleep-stage mean imputation.

    Parameters
    ----------
    feature_matrix
        Array with shape ``(n_participants, n_sleep_ids)``.
        Missing participant-stage values must be represented by NaN.

    sleep_ids
        Sleep-stage IDs corresponding to matrix columns.

    Returns
    -------
    imputed_matrix
        Copy of the feature matrix with missing values replaced by the
        observed mean for their corresponding sleep-stage column.

    stage_means
        Mean feature value for each sleep-stage column.

    Notes
    -----
    The historical implementation collected available graph features
    separately for every sleep ID, calculated the mean for each stage,
    and then replaced missing participant-stage values with the
    corresponding stage mean.

    Therefore imputation occurs across participants *within the current
    processing batch*.

    If an entire sleep-stage column contains no observed values, its
    mean remains NaN and missing entries for that stage remain NaN.
    """
    matrix = np.asarray(
        feature_matrix,
        dtype=float,
    )

    sleep_ids_array = np.asarray(
        list(sleep_ids)
    )

    if matrix.ndim != 2:
        raise ValueError(
            "feature_matrix must be two-dimensional."
        )

    if matrix.shape[1] != len(sleep_ids_array):
        raise ValueError(
            "The number of feature columns must match "
            "the number of sleep IDs."
        )

    # Equivalent numerical behavior to the historical
    # per-sleep-ID np.nanmean calculation.
    with np.errstate(invalid="ignore"):
        stage_means = np.nanmean(
            matrix,
            axis=0,
        )

    imputed = matrix.copy()

    for column_index, stage_mean in enumerate(
        stage_means
    ):
        missing = np.isnan(
            imputed[:, column_index]
        )

        imputed[
            missing,
            column_index
        ] = stage_mean

    return imputed, stage_means