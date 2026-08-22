# Synthetic PSG Demonstration Data

The repository includes a generator for fully artificial MNE Epochs
files.

No synthetic FIF binaries are committed to Git. Users generate them
locally when they want to exercise the feature-extraction pipeline.

## Purpose

The generator allows the software path to be tested without access to
the protected clinical PSG recordings.

The generated files contain:

- 17 artificial signal channels;
- five sleep-stage event codes;
- artificial participant IDs using the `SYN###` format;
- artificial `Y` / `N` demonstration labels;
- MNE Epochs metadata compatible with the reconstructed feature
  extractor.

## Important limitation

The generated signals are not physiological simulations.

They are not intended to reproduce:

- EEG morphology;
- EOG morphology;
- EMG morphology;
- ECG morphology;
- ADHD biomarkers;
- published feature distributions;
- the historical acquisition protocol.

The default synthetic sampling frequency and epoch length are
deliberately lightweight so the public demonstration can run quickly.

## Generate local synthetic FIF files

From a source checkout:

```bash
python scripts/generate_synthetic_psg.py \
    --output-dir examples/generated_synthetic_psg \
    --participants 10 \
    --overwrite
```

The resulting files have names such as:

```text
SYN001-epo.fif
SYN002-epo.fif
...
SYN010-epo.fif
```

The repository `.gitignore` excludes FIF files, so these generated
artifacts remain local.

## Historical stage codes

The synthetic demonstration uses:

| Event code | Demonstration stage |
| ---: | --- |
| 1 | Wake |
| 2 | Sleep Stage 1 |
| 3 | Sleep Stage 2 |
| 4 | Sleep Stage 3-4 |
| 5 | Rapid Eye Movement |

This mapping is chosen to match the historical feature-column naming
used elsewhere in the repository.

## Next stage

The locally generated synthetic FIF files can be passed to the
reconstructed feature-extraction command.

That provides a complete software demonstration path without distributing
clinical data.