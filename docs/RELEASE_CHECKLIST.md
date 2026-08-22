# Public Release Checklist

Use this checklist before treating the repository as a finalized public
research-software release.

## Repository integrity

- [ ] `python scripts/validate_repository.py` passes.
- [ ] `python -m pytest -q` passes.
- [ ] Historical implementation hashes are exact.
- [ ] Manuscript hash is exact.
- [ ] No raw PSG files are tracked.
- [ ] No generated runtime outputs are tracked.
- [ ] No Python/test caches are tracked.
- [ ] No workstation-specific paths appear in public source or
      documentation.

## Scientific record

- [ ] Published metrics match the canonical publication record.
- [ ] Synthetic outputs are clearly labeled as non-scientific.
- [ ] Data availability statements do not imply public access to
      protected PSG data.
- [ ] Historical stochastic limitations are documented.
- [ ] DOI and citation metadata are correct.

## GitHub presentation

Suggested repository description:

`Graph-based PSG analysis and machine learning for sleep-stage ADHD biomarkers — reproducible reconstruction of an IEEE ISBI 2025 study.`

Suggested GitHub topics:

- `polysomnography`
- `adhd`
- `machine-learning`
- `graph-analysis`
- `mne-python`
- `random-forest`
- `biomedical-signal-processing`
- `reproducible-research`
- `research-software`

## After final push

- [ ] Confirm the GitHub Actions workflow is green.
- [ ] Confirm the README renders correctly.
- [ ] Confirm Mermaid diagrams render correctly.
- [ ] Confirm GitHub recognizes `CITATION.cff`.
- [ ] Confirm the DOI link opens the intended publication.
- [ ] Add the suggested repository description and topics.
- [ ] Pin the repository if it belongs in the primary academic
      portfolio.
- [ ] Consider creating a `v0.1.0` GitHub release after the final audit.