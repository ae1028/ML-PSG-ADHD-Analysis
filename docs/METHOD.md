# Method

This document will describe the reconstructed scientific workflow and
its relationship to the exact historical implementation.

The final documentation will cover:

1. PSG organization by sleep stage.
2. Channel-wise correlation estimation.
3. Graph construction.
4. Graph-derived feature extraction.
5. Participant-level feature representation.
6. Random Forest classification.
7. Nested cross-validation.
8. Feature-importance analysis.

Exact historical source code is preserved under:

`reference_implementations/published_pipeline/`

The reusable implementation under `src/psg_adhd/` must not silently
change historical scientific behavior.