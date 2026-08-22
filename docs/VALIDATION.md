# Repository Validation

The repository provides automated validation for both local development
and GitHub-hosted continuous integration.

## Local validation

From the repository root:

```bash
python scripts/validate_repository.py
```

The validator checks:

- required repository structure;
- historical source-code SHA-256 hashes;
- manuscript-record SHA-256 hash;
- canonical publication metrics;
- result provenance;
- public synthetic feature schema;
- synthetic participant identifiers;
- absence of tracked raw PSG data;
- absence of tracked Python/test caches;
- absence of known workstation-specific paths;
- absence of generated modeling outputs in version control.

## Automated tests

Run:

```bash
python -m pytest -q
```

The test suite covers:

- data handling;
- graph-feature calculations;
- participant/stage feature extraction;
- batch processing;
- command-line interfaces;
- historical-compatible Random Forest modeling;
- feature importance;
- modeling orchestration;
- synthetic feature generation;
- synthetic PSG generation;
- end-to-end public demonstration;
- canonical study-result records.

## GitHub Actions

`.github/workflows/ci.yml`

runs automatically for:

- pushes to `main`;
- pull requests targeting `main`;
- manual workflow dispatch.

The workflow uses Python 3.9.19, installs the recorded development
dependencies, installs the project in editable mode, runs repository
validation, executes the full test suite, and checks all public CLI
interfaces.

## Validation scope

Passing CI demonstrates repository consistency and software integrity.

It does not establish exact numerical reproduction of the historical
published experiment because the original standalone modeling code
contains unseeded stochastic operations and the protected research PSG
data are not distributed publicly.