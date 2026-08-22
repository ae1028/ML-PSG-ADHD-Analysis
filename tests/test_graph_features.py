"""Tests for historical graph-feature reconstruction."""

import networkx as nx
import numpy as np

from psg_adhd.graph_features import (
    average_stage_adjacency,
    compute_absolute_correlation,
    compute_stage_graph_feature,
    historical_average_shortest_path,
)


def historical_reference_feature(epoch_arrays):
    """Minimal direct transcription of historical graph logic."""
    adjacency_matrices = []

    for epoch_data in epoch_arrays:
        correlation_matrix = np.abs(
            np.corrcoef(epoch_data)
        )
        adjacency_matrices.append(correlation_matrix)

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
        for v in shortest_path_lengths.values()
    )

    total_length = sum(
        sum(v.values())
        for v in shortest_path_lengths.values()
    )

    return (
        total_length / total_paths,
        average_adjacency_matrix,
    )


def test_absolute_correlation_matches_historical_expression():
    rng = np.random.default_rng(10)

    epoch = rng.normal(
        size=(17, 200)
    )

    expected = np.abs(
        np.corrcoef(epoch)
    )

    actual = compute_absolute_correlation(epoch)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_stage_adjacency_matches_historical_average():
    rng = np.random.default_rng(20)

    epochs = [
        rng.normal(size=(17, 150))
        for _ in range(4)
    ]

    expected = np.mean(
        [
            np.abs(np.corrcoef(epoch))
            for epoch in epochs
        ],
        axis=0,
    )

    actual = average_stage_adjacency(epochs)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_shortest_path_matches_historical_formula():
    adjacency = np.array(
        [
            [1.0, 0.20, 0.60],
            [0.20, 1.0, 0.30],
            [0.60, 0.30, 1.0],
        ],
        dtype=float,
    )

    graph = nx.from_numpy_array(adjacency)

    lengths = dict(
        nx.shortest_path_length(
            graph,
            weight="weight",
        )
    )

    expected = (
        sum(sum(v.values()) for v in lengths.values())
        / sum(len(v) for v in lengths.values())
    )

    actual = historical_average_shortest_path(
        adjacency
    )

    assert actual == expected


def test_complete_stage_feature_matches_historical_code():
    rng = np.random.default_rng(42)

    epochs = [
        rng.normal(size=(17, 256))
        for _ in range(5)
    ]

    expected_feature, expected_adjacency = (
        historical_reference_feature(epochs)
    )

    actual_feature, actual_adjacency = (
        compute_stage_graph_feature(epochs)
    )

    np.testing.assert_allclose(
        actual_adjacency,
        expected_adjacency,
        rtol=0.0,
        atol=0.0,
    )

    assert actual_feature == expected_feature


def test_empty_stage_is_rejected():
    try:
        average_stage_adjacency([])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for an empty sleep stage."
        )


def test_non_square_adjacency_is_rejected():
    adjacency = np.ones((3, 4))

    try:
        historical_average_shortest_path(
            adjacency
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for non-square adjacency."
        )