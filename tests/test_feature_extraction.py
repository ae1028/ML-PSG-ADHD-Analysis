"""Tests for the complete historical feature-extraction workflow."""

import networkx as nx
import mne
import numpy as np
import pandas as pd

from psg_adhd.feature_extraction import (
    build_feature_table,
    compute_participant_stage_features,
)


def make_test_epochs():
    """Construct deterministic synthetic participant/stage PSG data."""
    rng = np.random.default_rng(2468)

    # Participant-stage layout:
    #
    # P01 -> stages 1, 2, 3
    # P02 -> stages 1, 2
    # P03 -> stages 1, 3
    #
    # This deliberately creates missing participant-stage combinations
    # so that historical stage-mean imputation is exercised.

    participant_ids = [
        "P01",
        "P01",
        "P01",
        "P02",
        "P02",
        "P03",
        "P03",
    ]

    labels = [
        "N",
        "N",
        "N",
        "Y",
        "Y",
        "Y",
        "Y",
    ]

    sleep_codes = np.array(
        [
            1,
            2,
            3,
            1,
            2,
            1,
            3,
        ],
        dtype=int,
    )

    n_epochs = len(participant_ids)
    n_channels = 4
    n_samples = 128

    data = rng.normal(
        size=(
            n_epochs,
            n_channels,
            n_samples,
        )
    )

    info = mne.create_info(
        ch_names=[
            "C1",
            "C2",
            "C3",
            "C4",
        ],
        sfreq=128.0,
        ch_types="eeg",
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
            "ID": participant_ids,
            "ADHD": labels,
        }
    )

    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={
            "Stage1": 1,
            "Stage2": 2,
            "Stage3": 3,
        },
        tmin=0.0,
        metadata=metadata,
        verbose=False,
    )


def literal_historical_feature(epochs):
    """Direct transcription of the historical feature logic."""
    patient_ids = np.unique(
        epochs.metadata["ID"]
    )

    sleep_ids = np.unique(
        epochs.events[:, 2]
    )

    by_patient = {}

    all_shortest_paths = {
        sleep_id: []
        for sleep_id in sleep_ids
    }

    for patient_id in patient_ids:
        patient_epochs = epochs[
            epochs.metadata["ID"]
            == patient_id
        ]

        shortest_paths_for_patient = []

        for sleep_id in sleep_ids:
            sleep_epochs = patient_epochs[
                patient_epochs.events[:, 2]
                == sleep_id
            ]

            adjacency_matrices = []

            for epoch in sleep_epochs.iter_evoked():
                correlation_matrix = np.abs(
                    np.corrcoef(epoch.data)
                )

                adjacency_matrices.append(
                    correlation_matrix
                )

            if adjacency_matrices:
                average_adjacency_matrix = np.mean(
                    adjacency_matrices,
                    axis=0,
                )

                graph = nx.from_numpy_array(
                    average_adjacency_matrix
                )

                shortest_path_lengths = dict(
                    nx.shortest_path_length(
                        graph,
                        weight="weight",
                    )
                )

                total_paths = sum(
                    len(v)
                    for v
                    in shortest_path_lengths.values()
                )

                total_length = sum(
                    sum(v.values())
                    for v
                    in shortest_path_lengths.values()
                )

                avg_shortest_path_length = (
                    total_length
                    / total_paths
                )

                shortest_paths_for_patient.append(
                    avg_shortest_path_length
                )

                all_shortest_paths[
                    sleep_id
                ].append(
                    avg_shortest_path_length
                )

            else:
                shortest_paths_for_patient.append(
                    np.nan
                )

        by_patient[
            patient_id
        ] = shortest_paths_for_patient

    averages_per_sleep_id = {
        sleep_id: np.nanmean(
            all_shortest_paths[sleep_id]
        )
        for sleep_id in sleep_ids
    }

    for patient_id, paths in by_patient.items():
        by_patient[patient_id] = [
            (
                averages_per_sleep_id[sleep_id]
                if np.isnan(path)
                else path
            )
            for sleep_id, path
            in zip(
                sleep_ids,
                paths,
            )
        ]

    return (
        np.array(
            list(
                by_patient.values()
            )
        ),
        patient_ids,
        sleep_ids,
    )


def test_complete_feature_matrix_matches_literal_history():
    epochs = make_test_epochs()

    expected, expected_patients, expected_sleep_ids = (
        literal_historical_feature(
            epochs
        )
    )

    actual, actual_patients, actual_sleep_ids = (
        compute_participant_stage_features(
            epochs
        )
    )

    np.testing.assert_array_equal(
        actual_patients,
        expected_patients,
    )

    np.testing.assert_array_equal(
        actual_sleep_ids,
        expected_sleep_ids,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_feature_table_schema_matches_historical_csv():
    epochs = make_test_epochs()

    feature_df = build_feature_table(
        epochs
    )

    assert list(
        feature_df.columns
    ) == [
        "Patient_ID",
        "Sleep_ID_1",
        "Sleep_ID_2",
        "Sleep_ID_3",
        "ADHD",
    ]


def test_feature_table_patient_order_matches_numpy_unique():
    epochs = make_test_epochs()

    feature_df = build_feature_table(
        epochs
    )

    assert list(
        feature_df["Patient_ID"]
    ) == [
        "P01",
        "P02",
        "P03",
    ]


def test_feature_table_labels_match_first_historical_row():
    epochs = make_test_epochs()

    feature_df = build_feature_table(
        epochs
    )

    labels = dict(
        zip(
            feature_df["Patient_ID"],
            feature_df["ADHD"],
        )
    )

    assert labels == {
        "P01": "N",
        "P02": "Y",
        "P03": "Y",
    }


def test_missing_participant_stage_is_imputed():
    epochs = make_test_epochs()

    feature_df = build_feature_table(
        epochs
    )

    feature_columns = [
        "Sleep_ID_1",
        "Sleep_ID_2",
        "Sleep_ID_3",
    ]

    assert not feature_df[
        feature_columns
    ].isna().any().any()