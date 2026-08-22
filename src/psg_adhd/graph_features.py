"""Graph-based PSG feature extraction.

This module reconstructs the graph-feature calculations used in the
historical ML-PSG-ADHD analysis.

The primary goal at this stage is behavioral fidelity to the historical
implementation preserved in:

    reference_implementations/published_pipeline/
    Feature extraction (Sleep stages).py

Important
---------
The historical implementation treats the absolute Pearson-correlation
values themselves as NetworkX edge weights when calculating weighted
shortest paths.

That behavior is preserved intentionally for reproducibility. No
conversion such as ``1 - correlation`` is applied here.
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np


def compute_absolute_correlation(epoch_data: np.ndarray) -> np.ndarray:
    """Compute the historical channel-by-channel adjacency matrix.

    Parameters
    ----------
    epoch_data
        Two-dimensional array with shape ``(n_channels, n_samples)``.

    Returns
    -------
    numpy.ndarray
        Absolute Pearson-correlation matrix with shape
        ``(n_channels, n_channels)``.

    Notes
    -----
    This reproduces the historical expression::

        np.abs(np.corrcoef(epoch.data))
    """
    data = np.asarray(epoch_data, dtype=float)

    if data.ndim != 2:
        raise ValueError(
            "epoch_data must be a 2-D array with shape "
            "(n_channels, n_samples)."
        )

    return np.abs(np.corrcoef(data))


def average_stage_adjacency(
    epoch_arrays: Iterable[np.ndarray],
) -> np.ndarray:
    """Average epoch-level adjacency matrices for one sleep stage.

    The historical implementation first calculated one absolute
    correlation matrix per epoch and then averaged those matrices
    before constructing the NetworkX graph.

    Parameters
    ----------
    epoch_arrays
        Iterable of epoch arrays. Each array must have shape
        ``(n_channels, n_samples)``.

    Returns
    -------
    numpy.ndarray
        Mean adjacency matrix across all supplied epochs.

    Raises
    ------
    ValueError
        If no epochs are supplied or the adjacency matrices do not
        share the same shape.
    """
    adjacency_matrices = [
        compute_absolute_correlation(epoch)
        for epoch in epoch_arrays
    ]

    if not adjacency_matrices:
        raise ValueError(
            "At least one epoch is required to calculate a "
            "sleep-stage adjacency matrix."
        )

    first_shape = adjacency_matrices[0].shape

    if any(matrix.shape != first_shape for matrix in adjacency_matrices):
        raise ValueError(
            "All epochs must contain the same number of channels."
        )

    return np.mean(adjacency_matrices, axis=0)


def historical_average_shortest_path(
    adjacency_matrix: np.ndarray,
) -> float:
    """Calculate the graph feature exactly as in the historical code.

    Historical behavior
    -------------------
    The original implementation used::

        graph = nx.from_numpy_array(average_adjacency_matrix)
        shortest_path_lengths = dict(
            nx.shortest_path_length(graph, weight="weight")
        )
        total_paths = sum(
            len(v) for v in shortest_path_lengths.values()
        )
        total_length = sum(
            sum(v.values()) for v in shortest_path_lengths.values()
        )
        feature = total_length / total_paths

    This function intentionally preserves that calculation.

    Parameters
    ----------
    adjacency_matrix
        Square weighted adjacency matrix.

    Returns
    -------
    float
        Historical average shortest-path feature.
    """
    adjacency = np.asarray(adjacency_matrix, dtype=float)

    if adjacency.ndim != 2:
        raise ValueError("adjacency_matrix must be two-dimensional.")

    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency_matrix must be square.")

    graph = nx.from_numpy_array(adjacency)

    shortest_path_lengths = dict(
        nx.shortest_path_length(graph, weight="weight")
    )

    total_paths = sum(
        len(lengths)
        for lengths in shortest_path_lengths.values()
    )

    if total_paths == 0:
        raise ValueError(
            "No shortest paths were returned for the supplied graph."
        )

    total_length = sum(
        sum(lengths.values())
        for lengths in shortest_path_lengths.values()
    )

    return float(total_length / total_paths)


def compute_stage_graph_feature(
    epoch_arrays: Iterable[np.ndarray],
) -> tuple[float, np.ndarray]:
    """Compute one historical graph feature for one sleep stage.

    Parameters
    ----------
    epoch_arrays
        Epoch arrays belonging to a single participant and sleep stage.

    Returns
    -------
    feature
        Historical average-shortest-path feature.

    average_adjacency
        Mean absolute-correlation adjacency matrix used to create the
        graph.
    """
    average_adjacency = average_stage_adjacency(epoch_arrays)

    feature = historical_average_shortest_path(
        average_adjacency
    )

    return feature, average_adjacency