# Data

## Public data boundary

The original clinical polysomnography recordings are not distributed
through this repository.

No participant-derived raw PSG files are required to be committed to
Git in order to inspect, test, or understand the reconstructed software.

The repository `.gitignore` excludes common PSG formats and local data
directories.

## Historical input representation

The reconstructed feature-extraction workflow expects MNE Epochs FIF
files compatible with:

```python
mne.read_epochs(file_path, preload=True)
```

The historical analysis relies on two metadata concepts.

### Participant ID

Participant membership is obtained from the Epochs metadata column:

```text
ID
```

### ADHD label

The historical participant-level label is obtained from:

```text
ADHD
```

The historical implementation uses the first matching metadata row for
the participant when constructing the final feature table.

## Sleep-stage representation

Sleep stages are represented by the integer event code in:

```python
epochs.events[:, 2]
```

The reconstructed historical-compatible implementation discovers the
sleep IDs present in each concatenated processing batch rather than
forcing a global list during extraction.

Historical study documentation describes five sleep-stage-derived
features:

- Wake
- Stage 1
- Stage 2
- Stage 3-4
- REM

The numerical event-code mapping must come from the authorized local
Epochs data and its event definitions. The public repository does not
invent an event-code mapping that cannot be verified from the original
data.

## Generated feature table

For sleep IDs present in a given batch, output columns follow the
historical naming convention:

```text
Patient_ID
Sleep_ID_1
Sleep_ID_2
...
ADHD
```

Generated feature CSV files are derived artifacts, not raw PSG data.

## Privacy and data governance

Users of this repository are responsible for obtaining appropriate
authorization for any clinical or participant-level data they process.

Raw recordings, identifying metadata, participant-derived screenshots,
and other sensitive clinical artifacts must remain outside the public
repository.

## Public repository data boundary

The original participant-level PSG recordings are not included in this
repository.

The checked-in CSV files under `examples/synthetic_features/` are
completely artificial and use synthetic `SYN###` participant
identifiers.

Users who want to exercise the PSG-to-feature software path can generate
fully artificial MNE Epochs FIF files locally with:

```bash
python scripts/generate_synthetic_psg.py \
    --output-dir examples/generated_synthetic_psg \
    --participants 10 \
    --overwrite
```

Generated FIF files remain local and are excluded from version control.

For the full public-data policy, see `docs/DATA_BOUNDARY.md`.