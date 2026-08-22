# Installation

This repository distinguishes between the historical scientific
environment and the development/test environment.

## Historical scientific environment

The reconstructed historical project environment used:

```text
Python          3.9.19
mne             1.6.1
numpy           1.26.4
pandas          2.2.2
networkx        3.2.1
matplotlib      3.8.4
seaborn         0.13.2
scikit-learn    1.4.2
```

These versions are recorded in:

`requirements-historical.txt`

The same scientific dependency pins are used in:

`requirements.txt`

and:

`pyproject.toml`

## Recommended clean installation

Create a new Python 3.9 environment rather than modifying an existing
research environment.

Example:

```bash
python -m venv .venv
```

Activate the environment and install the runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Development and testing

For repository development and automated tests:

```bash
python -m pip install -r requirements-dev.txt
```

This includes:

```text
pytest==8.4.1
```

in addition to the historical scientific dependencies.

## Editable installation

The repository follows a `src/` package layout.

Install the package in editable mode with:

```bash
python -m pip install -e .
```

For editable development installation:

```bash
python -m pip install -e ".[dev]"
```

## Verify the installation

Run the automated test suite:

```bash
python -m pytest -q
```

The repository also provides executable interfaces:

```bash
python scripts/extract_features.py --help
python scripts/run_modeling.py --help
python scripts/generate_synthetic_psg.py --help
python scripts/run_synthetic_demo.py --help
```

## Public synthetic demonstration

The complete software path can be exercised without access to protected
clinical PSG data:

```bash
python scripts/run_synthetic_demo.py \
    --work-dir outputs/synthetic_demo \
    --participants 10 \
    --overwrite
```

This generates artificial PSG files, extracts sleep-stage graph
features, and executes a lightweight modeling smoke test.

The resulting synthetic signals, labels, performance metrics, and
feature importances have no clinical interpretation.

## Reproducibility scope

Exact dependency versions improve software reproducibility, but the
historical standalone modeling pipeline is not numerically deterministic.

The historical implementation leaves the following stochastic operations
unseeded:

- augmented-data shuffle;
- outer KFold splitting;
- Random Forest fitting.

Consequently, repeated historical-compatible modeling runs may produce
different folds, selected hyperparameters, metrics, and feature
importances.

Permutation importance is explicitly seeded with `random_state=42`.

See:

- `docs/REPRODUCIBILITY.md`
- `docs/MODELING.md`
- `docs/DATA_BOUNDARY.md`

for further details.