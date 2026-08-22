# Repository Map

This document describes the role of the major repository components.

## `src/psg_adhd/`

Reusable implementation of the reconstructed software workflow.

### `data.py`

MNE Epochs loading and participant/stage data access.

### `graph_features.py`

Correlation adjacency construction and weighted graph-feature
calculation.

### `feature_extraction.py`

Participant-level sleep-stage graph-feature extraction.

### `batch_processing.py`

Historical-compatible multi-file batch processing and feature-table
writing.

### `modeling.py`

Historical-compatible Random Forest nested-cross-validation logic.

### `evaluation.py`

Feature loading, performance evaluation, and feature-importance
utilities.

### `model_workflow.py`

High-level modeling orchestration.

### `synthetic_psg.py`

Fully artificial PSG generation for public software testing.

### `example_data.py`

Artificial participant-level feature-table generation.

### `demo.py`

End-to-end public synthetic demonstration.

---

## `scripts/`

Executable repository interfaces:

- `extract_features.py`
- `run_modeling.py`
- `generate_synthetic_psg.py`
- `run_synthetic_demo.py`
- `validate_repository.py`

---

## `reference_implementations/published_pipeline/`

Exact historical source-code records preserved for provenance.

These files are intentionally not modernized in place.

The cleaned implementation under `src/` should be used for reusable
software development.

---

## `configs/`

Machine-readable historical workflow configuration.

The configuration files document both feature-extraction and modeling
behavior observed in the preserved source implementations.

---

## `examples/`

Public artificial examples.

No real participant-level PSG data should be stored here.

---

## `results/`

Canonical publication-level result records.

Synthetic demo outputs must not be placed in this directory.

---

## `docs/`

Detailed documentation covering:

- method;
- modeling;
- installation;
- data boundaries;
- reproducibility;
- limitations;
- results;
- citation;
- validation;
- publication metadata.

---

## `tests/`

Automated behavioral, provenance, synthetic-data, modeling, CLI, and
publication-metadata tests.

---

## `.github/workflows/`

GitHub Actions continuous-integration configuration.

---

## Root metadata

### `README.md`

Primary project overview.

### `CITATION.cff`

GitHub-compatible citation metadata.

### `CITATION.bib`

BibTeX citation for the associated IEEE ISBI publication.

### `Final_Manuscript.pdf`

Preserved author-side manuscript record.

### `requirements-historical.txt`

Recorded historical scientific dependency versions.

### `requirements.txt`

Runtime environment specification.

### `requirements-dev.txt`

Development/test environment.

### `pyproject.toml`

Python packaging metadata.