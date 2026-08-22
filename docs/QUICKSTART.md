# Quick Start

This repository can be exercised end-to-end without access to the
original clinical PSG recordings.

## One-command synthetic demonstration

From a source checkout:

```bash
python scripts/run_synthetic_demo.py \
    --work-dir outputs/synthetic_demo \
    --participants 10 \
    --overwrite
```

On Windows PowerShell:

```powershell
python scripts\run_synthetic_demo.py `
    --work-dir "outputs\synthetic_demo" `
    --participants 10 `
    --overwrite
```

The command runs:

```text
fully artificial PSG signals
        |
        v
synthetic MNE Epochs FIF files
        |
        v
historical-compatible graph feature extraction
        |
        v
participant-level feature CSV
        |
        v
lightweight Random Forest smoke test
        |
        v
metrics and feature-importance CSV files
```

## Generated artifacts

By default, demonstration outputs are placed under:

`outputs/synthetic_demo/`

The directory contains three groups of local artifacts.

### `psg/`

Fully artificial MNE Epochs FIF files such as:

`SYN001-epo.fif`

### `features/`

Participant-level graph-feature CSV files such as:

`features_batch_1.csv`

### `results/`

Modeling smoke-test outputs:

- `summary_metrics.csv`
- `fold_metrics.csv`
- `impurity_feature_importance.csv`
- `permutation_feature_importance.csv`

The `outputs/` directory and FIF files are excluded from version control.

## Important interpretation boundary

This demonstration validates software connectivity.

It does not validate:

- ADHD biomarkers;
- diagnostic performance;
- published numerical results;
- physiological realism;
- clinical generalization.

The synthetic PSG generator deliberately creates artificial signals with
no physiological or diagnostic meaning.

The modeling demonstration also uses a deliberately reduced Random
Forest parameter grid so it can run quickly.

## Historical reproduction

The historical-compatible implementation and the public smoke-test mode
serve different purposes.

The historical modeling parameter grid is documented in:

`docs/MODELING.md`

and:

`configs/historical_modeling_pipeline.yaml`

The synthetic demo does not replace or redefine that historical
configuration.