# Method

This repository contains a cleaned, testable reconstruction of the
sleep-stage-specific PSG feature-extraction workflow used in the
historical ML-PSG-ADHD analysis.

The exact historical scripts are preserved without scientific
refactoring under:

`reference_implementations/published_pipeline/`

The reusable implementation is located under:

`src/psg_adhd/`

## Feature-extraction workflow

For each processing batch, the reconstructed historical workflow is:

1. Discover local MNE Epochs FIF files.
2. Process files in consecutive batches of 10 by default.
3. Load each file with `mne.read_epochs(..., preload=True)`.
4. Concatenate the Epochs objects in the current batch.
5. Obtain participant IDs from `metadata["ID"]` using `numpy.unique`.
6. Obtain sleep-stage IDs from `events[:, 2]` using `numpy.unique`.
7. Select each participant's epochs.
8. Select each sleep stage for that participant.
9. Iterate through stage epochs using `iter_evoked()`.
10. Calculate the channel-by-channel Pearson correlation matrix for
    every epoch.
11. Take the absolute value of each correlation matrix.
12. Average the adjacency matrices across epochs belonging to the same
    participant and sleep stage.
13. Construct a weighted NetworkX graph from the mean adjacency matrix.
14. Calculate weighted shortest paths.
15. Average the returned shortest-path lengths to obtain one graph
    feature for that participant-stage combination.
16. Represent unavailable participant-stage combinations as missing.
17. Replace missing values using the mean feature for the corresponding
    sleep-stage ID among participants available in the current batch.
18. Construct the participant-level feature table.

The resulting historical-style feature table has the form:

`Patient_ID | Sleep_ID_* | ADHD`

## Important historical graph-weight behavior

The historical implementation uses the absolute Pearson-correlation
values directly as NetworkX edge weights:

```python
graph = nx.from_numpy_array(average_adjacency_matrix)

nx.shortest_path_length(
    graph,
    weight="weight",
)
```

This means a stronger absolute correlation receives a larger numerical
edge weight.

For a conventional graph-distance interpretation, one might instead
transform similarity into distance. The historical code did not do so.

The reconstructed historical-compatible implementation therefore does
not silently introduce transformations such as:

```python
1 - abs(correlation)
```

Preserving this behavior is necessary for provenance and reproducibility.

## Order of operations

The historical implementation averages adjacency matrices before graph
feature calculation:

`epochs -> correlation matrices -> mean adjacency -> graph feature`

It does not calculate one graph feature per epoch and then average the
graph features.

## Participant and sleep-stage discovery

Participant IDs are inferred from:

```python
np.unique(epochs.metadata["ID"])
```

Sleep-stage IDs are inferred from:

```python
np.unique(epochs.events[:, 2])
```

The historical extraction therefore did not enforce a global sleep-stage
schema during each batch.

## Missing-stage handling

If a participant does not contain epochs for a sleep-stage ID present in
the current batch, that participant-stage feature is initially missing.

The historical workflow then calculates the mean observed feature for
that sleep-stage ID across participants in the current batch and uses
that value for imputation.

This is preserved as historical behavior. It should not be interpreted
as a recommendation for leakage-safe model-evaluation preprocessing.

## Reconstruction architecture

The historical workflow has been separated into testable components:

- `data.py` — loading, participant selection, stage selection, labels,
  and historical stage-mean imputation.
- `graph_features.py` — correlation matrices, adjacency aggregation,
  graph construction, and shortest-path feature calculation.
- `feature_extraction.py` — participant-by-stage feature-table
  construction.
- `batch_processing.py` — FIF discovery, 10-file batching,
  concatenation, and CSV generation.
- `cli.py` — safe command-line interface.

The historical scripts remain unchanged so that the reconstructed
implementation can be audited against them.