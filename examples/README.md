# Public Synthetic Examples

This directory contains artificial data for demonstrating the software
interfaces in this repository.

## Important data boundary

The files in `synthetic_features/` are completely synthetic.

They:

- do not contain information from study participants;
- are not anonymized versions of the original PSG data;
- are not derived from the clinical dataset;
- do not reproduce real patient feature distributions;
- have no diagnostic interpretation.

They are included solely so that users can inspect the expected schema
and exercise repository code without access to protected PSG data.

## Feature-table schema

The synthetic CSV files follow the historical modeling structure:

| Column | Purpose |
| --- | --- |
| `Patient_ID` | Artificial example identifier |
| `Sleep_ID_1` | Synthetic numerical feature |
| `Sleep_ID_2` | Synthetic numerical feature |
| `Sleep_ID_3` | Synthetic numerical feature |
| `Sleep_ID_4` | Synthetic numerical feature |
| `Sleep_ID_5` | Synthetic numerical feature |
| `ADHD` | Artificial `Y` / `N` demonstration label |

Historical presentation names are:

| Historical column | Presentation name |
| --- | --- |
| `Sleep_ID_1` | Wake |
| `Sleep_ID_2` | Sleep Stage 1 |
| `Sleep_ID_3` | Sleep Stage 2 |
| `Sleep_ID_4` | Sleep Stage 3-4 |
| `Sleep_ID_5` | Rapid Eye Movement |

## Synthetic construction

The public example is generated deterministically with a fixed
example-only random seed.

A small artificial label-dependent numerical offset is included only so
that the machine-learning demonstration has some structure to process.

That artificial relationship must not be interpreted as a statement
about ADHD, sleep physiology, PSG biomarkers, or the original study.

## Regenerating the example

The checked-in example CSV files can be recreated with:

```python
from psg_adhd.example_data import write_synthetic_feature_batches

write_synthetic_feature_batches(
    "examples/synthetic_features",
    n_participants=20,
    batch_size=10,
    random_state=2026,
    overwrite=True,
)
```

The resulting files are intended for software demonstrations, not
scientific reproduction.