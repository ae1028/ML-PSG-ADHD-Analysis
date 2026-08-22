# Reproducibility

The repository distinguishes between historical implementation
preservation and cleaned software reconstruction.

## Historical environment

The recorded historical environment is:

- Python 3.9.19
- MNE 1.6.1
- NumPy 1.26.4
- pandas 2.2.2
- NetworkX 3.2.1
- Matplotlib 3.8.4
- seaborn 0.13.2
- scikit-learn 1.4.2

These versions are recorded in `requirements.txt`.

## Exact historical source preservation

The original repository implementations are preserved under:

`reference_implementations/published_pipeline/`

They are retained for provenance and are not the recommended execution
interface.

The reconstructed modules under `src/psg_adhd/` are designed so that
historical calculations can be tested independently.

## Running the test suite

From a Python environment containing the required dependencies:

```bash
python -m pytest -q
```

The tests use synthetic data and therefore do not require access to the
original clinical PSG recordings.

The test suite checks, among other behavior:

- absolute Pearson-correlation calculation;
- averaging adjacency matrices across epochs;
- historical NetworkX weighted shortest paths;
- participant selection;
- sleep-stage selection;
- historical stage-mean imputation;
- complete participant-stage feature extraction;
- batch orchestration;
- command-line behavior.

## Running feature extraction

From a source checkout:

```bash
python scripts/extract_features.py \
    --input-dir <authorized-fif-directory> \
    --output-dir <output-directory>
```

On Windows PowerShell, for example:

```powershell
python scripts\extract_features.py `
    --input-dir "C:\path\to\authorized\fif\data" `
    --output-dir "C:\path\to\output"
```

The default batch size is 10 files.

Additional options can be viewed with:

```bash
python scripts/extract_features.py --help
```

## Reproduction boundary

Passing the software equivalence tests demonstrates that the cleaned
implementation reproduces the explicitly encoded historical calculations
on controlled inputs.

It does not by itself prove reproduction of every numerical value in the
published study because the original clinical PSG recordings and final
experiment artifacts are not distributed publicly.

Published-result reconciliation will therefore be treated separately
from software-level behavioral equivalence.

## Environment policy

The historical environment is recorded for provenance.

Future supported environments may be added only after compatibility
testing. Historical dependency versions must not be silently replaced
and described as though they were the original environment.

## Historical modeling reproducibility

The standalone historical modeling script contains several unseeded
stochastic operations:

- `sklearn.utils.shuffle(...)` after sign-inversion augmentation;
- outer `KFold(..., shuffle=True)`;
- `RandomForestClassifier()`.

None of these operations specifies `random_state`.

Accordingly, two executions of the historical-compatible workflow can
produce different:

- augmented-row orderings;
- outer-fold memberships;
- fitted Random Forests;
- selected hyperparameters;
- fold-level metrics;
- mean metrics;
- impurity-based feature importance.

Permutation importance is different: the historical script explicitly
uses `random_state=42` and `n_repeats=10`.

Therefore the public reconstruction distinguishes:

1. **behavioral reproduction** — the same algorithmic operations are
   implemented and tested;
2. **exact numerical reproduction** — the same numerical outcome from a
   particular historical run.

The first is supported by the reconstruction and automated tests.

The second cannot currently be claimed solely from the historical code,
because the historical stochastic state for the unseeded operations was
not recorded.

## Modeling command

The reconstructed historical-compatible modeling workflow can be
started with:

```bash
python scripts/run_modeling.py \
    --feature-dir <feature-csv-directory> \
    --output-dir <result-directory>
```

The workflow exports numerical CSV result tables and explicitly warns
that repeated historical-compatible runs may differ.