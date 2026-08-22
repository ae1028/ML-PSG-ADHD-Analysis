"""Batch processing for historical-compatible PSG feature extraction.

This module reconstructs the file-level orchestration surrounding the
validated participant/sleep-stage feature extractor.

Historical workflow
-------------------
The original project processed MNE Epochs FIF files in batches, loaded
the Epochs objects with preload enabled, concatenated the Epochs in the
current batch, and then calculated participant-level sleep-stage graph
features.

The scientific feature calculations are implemented separately under
``psg_adhd.feature_extraction``.

Raw clinical PSG data are not distributed with this repository.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Iterable, Sequence

import mne
import pandas as pd

from .data import load_epochs_data
from .feature_extraction import build_feature_table


DEFAULT_BATCH_SIZE = 10


def discover_epoch_files(
    input_dir: str | Path,
    pattern: str = "*.fif",
) -> list[Path]:
    """Discover candidate Epochs files in one directory.

    Parameters
    ----------
    input_dir
        Directory containing authorized local Epochs FIF files.

    pattern
        Glob pattern used for file discovery.

    Returns
    -------
    list[pathlib.Path]
        Files returned by Python's historical-style ``glob`` call.

    Notes
    -----
    The returned paths are intentionally not re-sorted here. This keeps
    the discovery behavior close to a direct historical ``glob(...)``
    workflow rather than silently imposing a new ordering rule.
    """
    search_pattern = str(
        Path(input_dir) / pattern
    )

    return [
        Path(file_path)
        for file_path in glob(search_pattern)
    ]


def iter_file_batches(
    file_paths: Sequence[str | Path],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterable[list[Path]]:
    """Yield consecutive file batches while preserving input order."""
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be a positive integer."
        )

    normalized = [
        Path(file_path)
        for file_path in file_paths
    ]

    for start in range(
        0,
        len(normalized),
        batch_size,
    ):
        yield normalized[
            start : start + batch_size
        ]


def process_epoch_batch(
    file_paths: Sequence[str | Path],
    id_column: str = "ID",
    label_column: str = "ADHD",
) -> pd.DataFrame:
    """Process one collection of Epochs FIF files.

    Each FIF file is loaded using the reconstructed historical loader,
    the Epochs objects are concatenated with MNE, and the complete
    participant-level feature table is generated.
    """
    paths = [
        Path(file_path)
        for file_path in file_paths
    ]

    if not paths:
        raise ValueError(
            "At least one FIF file is required to process a batch."
        )

    epochs_list = [
        load_epochs_data(path)
        for path in paths
    ]

    concatenated_epochs = mne.concatenate_epochs(
        epochs_list
    )

    return build_feature_table(
        concatenated_epochs,
        id_column=id_column,
        label_column=label_column,
    )


def write_feature_batches(
    input_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pattern: str = "*.fif",
    output_prefix: str = "features_batch",
    overwrite: bool = False,
    id_column: str = "ID",
    label_column: str = "ADHD",
) -> list[Path]:
    """Process a directory of FIF files and write batch feature CSVs.

    Parameters
    ----------
    input_dir
        Directory containing locally authorized FIF files.

    output_dir
        Directory where generated feature tables will be written.

    batch_size
        Number of FIF files processed together. Historical default: 10.

    pattern
        File-discovery pattern.

    output_prefix
        Prefix used for cleaned repository output filenames.

    overwrite
        Whether existing output CSVs may be replaced.

    id_column, label_column
        Metadata column names used by the historical workflow.

    Returns
    -------
    list[pathlib.Path]
        Paths of generated CSV files.

    Notes
    -----
    Generated filenames such as ``features_batch_1.csv`` are part of
    the cleaned repository interface and are not treated as scientific
    results or historical provenance records.
    """
    discovered = discover_epoch_files(
        input_dir,
        pattern=pattern,
    )

    if not discovered:
        raise FileNotFoundError(
            f"No files matching {pattern!r} were found in "
            f"{str(input_dir)!r}."
        )

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_files: list[Path] = []

    for batch_number, batch_paths in enumerate(
        iter_file_batches(
            discovered,
            batch_size=batch_size,
        ),
        start=1,
    ):
        csv_path = output_path / (
            f"{output_prefix}_{batch_number}.csv"
        )

        if csv_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {csv_path}. "
                "Use overwrite=True to replace it."
            )

        feature_table = process_epoch_batch(
            batch_paths,
            id_column=id_column,
            label_column=label_column,
        )

        feature_table.to_csv(
            csv_path,
            index=False,
        )

        generated_files.append(
            csv_path
        )

    return generated_files