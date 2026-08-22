# Configurations

This directory contains explicit configuration records for the
reconstructed repository.

## `historical_integrated_pipeline.yaml`

This file records behavior directly observed in the historical
`Feature extraction (Sleep stages).py` implementation.

It should not yet be interpreted as proof that every setting reproduced
the final published numerical results.

During later validation batches, the historical implementations,
archived outputs, and published results will be compared before a
publication-reproduction configuration is declared canonical.

## `historical_modeling_pipeline.yaml`

This file records the behavior directly observed in the standalone
historical `ADHD_Final_Code.py` modeling implementation.

It intentionally records unseeded values as `null` for:

- augmented-data shuffle;
- outer KFold;
- Random Forest classifier.

This configuration is distinct from
`historical_integrated_pipeline.yaml`.

The repository does not assume that the two historical scripts used
identical stochastic settings merely because they belong to the same
research project.