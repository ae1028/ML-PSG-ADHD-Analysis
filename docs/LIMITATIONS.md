# Limitations

This repository reconstructs a research workflow developed for a
specific polysomnography-based ADHD study.

## Study scope

The historical study used a relatively small participant cohort and
should not be interpreted as establishing a general-purpose clinical
diagnostic system.

## Clinical-data availability

The original PSG recordings are not publicly distributed through this
repository.

Consequently, the public repository can validate software behavior with
synthetic tests but cannot independently reproduce the complete clinical
experiment without authorized access to the original data.

## Historical graph interpretation

The historical implementation uses absolute correlation coefficients
directly as shortest-path edge weights.

Because stronger correlations therefore have larger numerical traversal
costs, this differs from a conventional similarity-to-distance graph
construction.

The reconstructed historical-compatible pipeline preserves this behavior
rather than silently correcting it.

## Batch-dependent preprocessing

Sleep-stage IDs are inferred independently from the stages present in
each processing batch.

Missing participant-stage values are imputed using the corresponding
stage mean among participants in that same batch.

This historical preprocessing behavior can make derived values dependent
on batch composition.

It is preserved for provenance and should not automatically be adopted
for a new prospective modeling study.

## Model-evaluation implications

Stage-mean imputation is performed before the historical downstream
classification workflow.

A modern evaluation intended to estimate prospective generalization
should consider fitting preprocessing operations strictly within
training folds.

That methodological alternative is separate from historical
reproduction and will not be substituted silently.

## Research use

The repository is intended for research, methodological inspection, and
software reproducibility.

It is not a medical device and should not be used as an independent
clinical diagnostic tool.