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