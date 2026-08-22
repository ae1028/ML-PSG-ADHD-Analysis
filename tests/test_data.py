"""Tests for historical PSG participant/stage handling."""

import warnings

import mne
import numpy as np
import pandas as pd

from psg_adhd.data import (
    get_patient_ids,
    get_patient_label,
    get_sleep_ids,
    impute_missing_stage_features,
    select_patient_epochs,
    select_sleep_stage_epochs,
)


def make_synthetic_epochs():
    """Create a small MNE EpochsArray with historical metadata fields."""
    rng = np.random.default_rng(123)

    n_epochs = 6
    n_channels = 3
    n_samples = 100

    data = rng.normal(
        size=(
            n_epochs,
            n_channels,
            n_samples,
        )
    )

    info = mne.create_info(
        ch_names=[
            "EEG1",
            "EEG2",
            "EEG3",
        ],
        sfreq=100.0,
        ch_types="eeg",
    )

    sleep_codes = np.array(
        [1, 2, 1, 3, 2, 3],
        dtype=int,
    )

    events = np.column_stack(
        [
            np.arange(n_epochs) * n_samples,
            np.zeros(n_epochs, dtype=int),
            sleep_codes,
        ]
    )

    metadata = pd.DataFrame(
        {
            "ID": [
                "P02",
                "P02",
                "P01",
                "P01",
                "P03",
                "P03",
            ],
            "ADHD": [
                "Y",
                "Y",
                "N",
                "N",
                "Y",
                "Y",
            ],
        }
    )

    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={
            "Wake": 1,
            "Stage1": 2,
            "Stage2": 3,
        },
        tmin=0.0,
        metadata=metadata,
        verbose=False,
    )

    return epochs


def test_patient_ids_match_numpy_unique_behavior():
    epochs = make_synthetic_epochs()

    actual = get_patient_ids(epochs)

    expected = np.unique(
        epochs.metadata["ID"]
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )

    np.testing.assert_array_equal(
        actual,
        np.array(
            ["P01", "P02", "P03"]
        ),
    )


def test_sleep_ids_match_event_codes():
    epochs = make_synthetic_epochs()

    actual = get_sleep_ids(epochs)

    expected = np.unique(
        epochs.events[:, 2]
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )

    np.testing.assert_array_equal(
        actual,
        np.array([1, 2, 3]),
    )


def test_patient_selection_matches_historical_mask():
    epochs = make_synthetic_epochs()

    patient_epochs = select_patient_epochs(
        epochs,
        "P02",
    )

    assert len(patient_epochs) == 2

    np.testing.assert_array_equal(
        patient_epochs.metadata[
            "ID"
        ].to_numpy(),
        np.array(["P02", "P02"]),
    )

    np.testing.assert_array_equal(
        patient_epochs.events[:, 2],
        np.array([1, 2]),
    )


def test_sleep_stage_selection_matches_event_mask():
    epochs = make_synthetic_epochs()

    patient_epochs = select_patient_epochs(
        epochs,
        "P01",
    )

    stage_epochs = select_sleep_stage_epochs(
        patient_epochs,
        3,
    )

    assert len(stage_epochs) == 1

    np.testing.assert_array_equal(
        stage_epochs.events[:, 2],
        np.array([3]),
    )


def test_patient_label_uses_first_matching_metadata_row():
    epochs = make_synthetic_epochs()

    assert (
        get_patient_label(
            epochs,
            "P01",
        )
        == "N"
    )

    assert (
        get_patient_label(
            epochs,
            "P02",
        )
        == "Y"
    )


def test_stage_mean_imputation_matches_historical_behavior():
    features = np.array(
        [
            [1.0, np.nan, 3.0],
            [2.0, 4.0, np.nan],
            [np.nan, 6.0, 9.0],
        ]
    )

    sleep_ids = np.array(
        [1, 2, 3]
    )

    imputed, stage_means = (
        impute_missing_stage_features(
            features,
            sleep_ids,
        )
    )

    expected_means = np.array(
        [1.5, 5.0, 6.0]
    )

    expected_imputed = np.array(
        [
            [1.0, 5.0, 3.0],
            [2.0, 4.0, 6.0],
            [1.5, 6.0, 9.0],
        ]
    )

    np.testing.assert_allclose(
        stage_means,
        expected_means,
    )

    np.testing.assert_allclose(
        imputed,
        expected_imputed,
    )


def test_entirely_missing_stage_remains_nan():
    features = np.array(
        [
            [1.0, np.nan],
            [2.0, np.nan],
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )

        imputed, stage_means = (
            impute_missing_stage_features(
                features,
                [1, 2],
            )
        )

    assert stage_means[0] == 1.5
    assert np.isnan(stage_means[1])

    assert np.isnan(
        imputed[:, 1]
    ).all()