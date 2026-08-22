"""Tests for fully artificial MNE/PSG demonstration data."""

import mne
import numpy as np
import pytest

from psg_adhd.synthetic_psg import (
    SYNTHETIC_CHANNEL_NAMES,
    SYNTHETIC_EVENT_ID,
    generate_synthetic_participant_epochs,
    generate_synthetic_psg_files,
)


def test_synthetic_epochs_have_expected_structure():
    epochs = generate_synthetic_participant_epochs(
        patient_id="SYN001",
        label="N",
        random_state=2027,
        epochs_per_stage=2,
        sfreq=128.0,
        n_samples=256,
    )

    assert len(
        epochs.ch_names
    ) == 17

    assert epochs.ch_names == (
        SYNTHETIC_CHANNEL_NAMES
    )

    assert len(epochs) == 10

    np.testing.assert_array_equal(
        np.unique(
            epochs.events[:, 2]
        ),
        np.array(
            [1, 2, 3, 4, 5]
        ),
    )

    assert epochs.event_id == (
        SYNTHETIC_EVENT_ID
    )

    assert set(
        epochs.metadata["ID"]
    ) == {
        "SYN001"
    }

    assert set(
        epochs.metadata["ADHD"]
    ) == {
        "N"
    }


def test_synthetic_epochs_are_deterministic():
    first = generate_synthetic_participant_epochs(
        patient_id="SYN001",
        label="N",
        random_state=2027,
        epochs_per_stage=1,
        sfreq=128.0,
        n_samples=64,
    )

    second = generate_synthetic_participant_epochs(
        patient_id="SYN001",
        label="N",
        random_state=2027,
        epochs_per_stage=1,
        sfreq=128.0,
        n_samples=64,
    )

    np.testing.assert_array_equal(
        first.get_data(),
        second.get_data(),
    )

    np.testing.assert_array_equal(
        first.events,
        second.events,
    )


def test_synthetic_generator_rejects_invalid_label():
    with pytest.raises(
        ValueError
    ):
        generate_synthetic_participant_epochs(
            patient_id="SYN001",
            label="UNKNOWN",
            random_state=2027,
        )


def test_synthetic_fif_writer_and_reload(
    tmp_path,
):
    generated = generate_synthetic_psg_files(
        output_dir=tmp_path,
        n_participants=2,
        random_state=2026,
        epochs_per_stage=1,
        sfreq=64.0,
        n_samples=64,
    )

    assert [
        path.name
        for path in generated
    ] == [
        "SYN001-epo.fif",
        "SYN002-epo.fif",
    ]

    assert all(
        path.exists()
        for path in generated
    )

    first = mne.read_epochs(
        generated[0],
        preload=True,
        verbose=False,
    )

    second = mne.read_epochs(
        generated[1],
        preload=True,
        verbose=False,
    )

    assert set(
        first.metadata["ID"]
    ) == {
        "SYN001"
    }

    assert set(
        first.metadata["ADHD"]
    ) == {
        "N"
    }

    assert set(
        second.metadata["ID"]
    ) == {
        "SYN002"
    }

    assert set(
        second.metadata["ADHD"]
    ) == {
        "Y"
    }


def test_synthetic_fif_writer_refuses_overwrite(
    tmp_path,
):
    generate_synthetic_psg_files(
        output_dir=tmp_path,
        n_participants=1,
        random_state=2026,
        epochs_per_stage=1,
        sfreq=64.0,
        n_samples=64,
    )

    with pytest.raises(
        FileExistsError
    ):
        generate_synthetic_psg_files(
            output_dir=tmp_path,
            n_participants=1,
            random_state=2026,
            epochs_per_stage=1,
            sfreq=64.0,
            n_samples=64,
            overwrite=False,
        )