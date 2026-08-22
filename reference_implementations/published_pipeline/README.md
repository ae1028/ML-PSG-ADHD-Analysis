# Published-Pipeline Reference Implementations

This directory preserves the exact historical Python scripts that were
present in the original `ML-PSG-ADHD-Analysis` repository before its
professional reconstruction.

## Files

### `Feature extraction (Sleep stages).py`

Historical implementation of the sleep-stage-specific PSG feature
extraction workflow.

The preserved file is intentionally retained without scientific
refactoring so that the development history and published workflow can
be audited against the reconstructed reusable implementation.

### `ADHD_Final_Code.py`

Historical implementation of the Random Forest modeling, nested
cross-validation, evaluation, and feature-importance workflow.

The preserved file is intentionally retained without scientific
refactoring.

## Important distinction

These files are **provenance references**, not the recommended reusable
API of the reconstructed repository.

The cleaned implementation developed during repository reconstruction
will live under:

`src/psg_adhd/`

Historical behavior will not be silently changed during refactoring.
Any intentional differences between the reusable implementation and
these preserved scripts must be documented explicitly.

## Historical environment

The original project environment was recorded as:

- Python 3.9.19
- MNE 1.6.1
- NumPy 1.26.4
- pandas 2.2.2
- NetworkX 3.2.1
- Matplotlib 3.8.4
- seaborn 0.13.2
- scikit-learn 1.4.2

## Data boundary

The original PSG recordings are not distributed through this
repository. Local/raw clinical data must remain outside version control.