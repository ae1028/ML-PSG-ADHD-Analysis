"""Generate fully artificial MNE Epochs data for software demonstrations.

The synthetic PSG data created by this module are completely artificial.

They:

- do not contain study-participant data;
- are not anonymized or transformed clinical recordings;
- do not reproduce the original PSG signal distributions;
- are not intended to model ADHD physiology;
- have no diagnostic or clinical interpretation.

Their only purpose is to make the repository's software interfaces
executable without access to protected clinical data.

The synthetic files preserve several structural properties useful for
testing the reconstructed workflow:

- 17 channels;
- participant-level ``ID`` metadata;
- participant-level ``ADHD`` demonstration labels;
- five sleep-stage event codes;
- MNE Epochs FIF output.

The lightweight default sampling rate and epoch length are deliberately
smaller than the historical study acquisition settings so the example
remains inexpensive to generate and run.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd


SYNTHETIC_EVENT_ID = {
    "Wake": 1,
    "Sleep Stage 1": 2,
    "Sleep Stage 2": 3,
    "Sleep Stage 3-4": 4,
    "Rapid Eye Movement": 5,
}


SYNTHETIC_CHANNEL_NAMES = [
    "EEG_Fp1",
    "EEG_Fp2",
    "EEG_F3",
    "EEG_F4",
    "EEG_C3",
    "EEG_C4",
    "EEG_O1",
    "EEG_O2",
    "EOG_L",
    "EOG_R",
    "EMG_CHIN1",
    "EMG_CHIN2",
    "EMG_LEG_L",
    "EMG_LEG_R",
    "EMG_AUX",
    "ECG_1",
    "ECG_2",
]


SYNTHETIC_CHANNEL_TYPES = [
    "eeg",
    "eeg",
    "eeg",
    "eeg",
    "eeg",
    "eeg",
    "eeg",
    "eeg",
    "eog",
    "eog",
    "emg",
    "emg",
    "emg",
    "emg",
    "emg",
    "ecg",
    "ecg",
]


def generate_synthetic_participant_epochs(
    patient_id: str,
    label: str,
    random_state: int,
    epochs_per_stage: int = 2,
    sfreq: float = 128.0,
    n_samples: int = 256,
) -> mne.EpochsArray:
    """Generate one fully artificial participant's MNE Epochs object.

    Parameters
    ----------
    patient_id
        Artificial participant identifier, for example ``SYN001``.

    label
        Artificial demonstration label, expected to be ``Y`` or ``N``.

    random_state
        Seed used only for deterministic synthetic-data generation.

    epochs_per_stage
        Number of artificial epochs generated for every sleep-stage
        event code.

    sfreq
        Lightweight synthetic sampling frequency.

    n_samples
        Number of samples in each artificial epoch.

    Returns
    -------
    mne.EpochsArray
        Synthetic Epochs object with 17 channels, five sleep-stage event
        codes, and ``ID`` / ``ADHD`` metadata.
    """
    if label not in {"Y", "N"}:
        raise ValueError(
            "label must be either 'Y' or 'N'."
        )

    if epochs_per_stage <= 0:
        raise ValueError(
            "epochs_per_stage must be a positive integer."
        )

    if sfreq <= 0:
        raise ValueError(
            "sfreq must be positive."
        )

    if n_samples < 2:
        raise ValueError(
            "n_samples must be at least 2."
        )

    rng = np.random.default_rng(
        random_state
    )

    sleep_codes = np.repeat(
        np.array(
            list(
                SYNTHETIC_EVENT_ID.values()
            ),
            dtype=int,
        ),
        epochs_per_stage,
    )

    n_epochs = len(
        sleep_codes
    )

    n_channels = len(
        SYNTHETIC_CHANNEL_NAMES
    )

    data = np.empty(
        (
            n_epochs,
            n_channels,
            n_samples,
        ),
        dtype=float,
    )

    # Create artificial correlated signals so that correlation matrices
    # and graph features are numerically well behaved.
    #
    # No label-dependent signal is introduced. The signals have no
    # physiological or diagnostic meaning.
    for epoch_index in range(
        n_epochs
    ):
        common_component = rng.normal(
            loc=0.0,
            scale=1.0,
            size=n_samples,
        )

        independent_noise = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(
                n_channels,
                n_samples,
            ),
        )

        data[
            epoch_index
        ] = (
            0.20
            * common_component[
                None,
                :
            ]
            + 0.80
            * independent_noise
        )

    # Use small arbitrary signal amplitudes. Correlation calculations
    # are scale-invariant, and these values are not intended to
    # represent actual physiological amplitudes.
    data *= 1e-6

    info = mne.create_info(
        ch_names=SYNTHETIC_CHANNEL_NAMES,
        sfreq=sfreq,
        ch_types=SYNTHETIC_CHANNEL_TYPES,
    )

    events = np.column_stack(
        [
            np.arange(
                n_epochs,
                dtype=int,
            )
            * (
                n_samples
                + 1
            ),
            np.zeros(
                n_epochs,
                dtype=int,
            ),
            sleep_codes,
        ]
    )

    metadata = pd.DataFrame(
        {
            "ID": [
                patient_id
            ]
            * n_epochs,
            "ADHD": [
                label
            ]
            * n_epochs,
        }
    )

    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id=SYNTHETIC_EVENT_ID,
        tmin=0.0,
        metadata=metadata,
        verbose=False,
    )


def generate_synthetic_psg_files(
    output_dir: str | Path,
    n_participants: int = 10,
    random_state: int = 2026,
    epochs_per_stage: int = 2,
    sfreq: float = 128.0,
    n_samples: int = 256,
    overwrite: bool = False,
) -> list[Path]:
    """Generate artificial participant-level Epochs FIF files.

    Output filenames use the standard MNE Epochs suffix:

    ``SYN001-epo.fif``
    ``SYN002-epo.fif``
    ...

    Labels alternate between ``N`` and ``Y``.

    Generated FIF files are intended as local demonstration artifacts
    and should remain excluded from Git.
    """
    if n_participants < 1:
        raise ValueError(
            "n_participants must be at least 1."
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated = []

    for participant_index in range(
        1,
        n_participants + 1,
    ):
        patient_id = (
            f"SYN{participant_index:03d}"
        )

        label = (
            "N"
            if participant_index % 2 == 1
            else "Y"
        )

        output_path = (
            output_dir
            / f"{patient_id}-epo.fif"
        )

        if (
            output_path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"Output already exists: {output_path}"
            )

        epochs = (
            generate_synthetic_participant_epochs(
                patient_id=patient_id,
                label=label,
                random_state=(
                    random_state
                    + participant_index
                ),
                epochs_per_stage=epochs_per_stage,
                sfreq=sfreq,
                n_samples=n_samples,
            )
        )

        epochs.save(
            output_path,
            overwrite=overwrite,
            verbose=False,
        )

        generated.append(
            output_path
        )

    return generated