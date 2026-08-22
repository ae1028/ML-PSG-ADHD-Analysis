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

## Historical augmentation and cross-validation dependence

The historical modeling workflow performs sign-inversion augmentation
before nested cross-validation.

Each original feature vector and its sign-inverted counterpart therefore
exist in the combined dataset before the outer folds are generated.

Because ordinary row-level `KFold` is subsequently applied to that
augmented dataset, paired augmented representations are not explicitly
kept within the same fold.

As a result, related representations derived from the same participant
may be assigned to different training and test partitions.

This behavior is preserved for historical provenance, but a new
prospective evaluation should consider participant- or group-aware
splitting and should perform any learned or synthetic preprocessing only
within the appropriate training partition.

## Historical stochasticity

The historical augmented-data shuffle, outer KFold, and Random Forest
classifier were unseeded.

Consequently, the exact fold memberships, selected models, metrics, and
feature importances from a historical execution are not deterministically
recoverable from the source code alone.

The reconstruction preserves this fact rather than adding an undocumented
seed and presenting the resulting values as historical results.

## Post-CV feature-importance interpretation

The historical workflow takes the best estimator from the final outer
fold, refits it on the entire augmented dataset, and then calculates
feature importance on that same complete dataset.

These importance values are therefore descriptive outputs of the
historical fitted workflow and should not be interpreted as independently
validated measures of causal or clinical importance.