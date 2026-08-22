"""Repository-level validation for ML-PSG-ADHD-Analysis.

This script performs public-release safety and provenance checks that
should remain true across local development and GitHub CI.

It intentionally avoids executing the full historical modeling grid.

Checks include:

- required repository files;
- preserved historical source hashes;
- preserved manuscript hash;
- canonical published-result records;
- synthetic/public data boundary;
- absence of tracked raw PSG files;
- absence of tracked Python caches;
- absence of targeted workstation-specific paths;
- public synthetic feature-table schema;
- expected documentation and executable scripts.

This validator checks repository integrity and software packaging.
It does not claim exact numerical reproduction of the published study.
"""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_HASHES = {
    (
        ROOT
        / "reference_implementations"
        / "published_pipeline"
        / "ADHD_Final_Code.py"
    ): (
        "9E4EF5BAA345B836030DE04608D312B9C7B0C4FC607CC600014476B4BEE08ADC"
    ),
    (
        ROOT
        / "reference_implementations"
        / "published_pipeline"
        / "Feature extraction (Sleep stages).py"
    ): (
        "86E50456199D3EA9974108A56C6934574B686E4A317EC1F46BAB222A5F4131C4"
    ),
    ROOT / "Final_Manuscript.pdf": (
        "B8FB9FFFDD7BD71C7FFC62DECFB4F97D91BEBD8B3B00906E87605A77B04E431A"
    ),
}


REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "requirements-historical.txt",
    "requirements-dev.txt",

    "docs/DATA.md",
    "docs/DATA_BOUNDARY.md",
    "docs/FEATURE_SCHEMA.md",
    "docs/INSTALLATION.md",
    "docs/LIMITATIONS.md",
    "docs/METHOD.md",
    "docs/MODELING.md",
    "docs/PUBLICATION.md",
    "docs/QUICKSTART.md",
    "docs/REPRODUCIBILITY.md",
    "docs/RESULTS.md",
    "docs/SYNTHETIC_PSG.md",

    "configs/historical_integrated_pipeline.yaml",
    "configs/historical_modeling_pipeline.yaml",

    "scripts/extract_features.py",
    "scripts/generate_synthetic_psg.py",
    "scripts/run_modeling.py",
    "scripts/run_synthetic_demo.py",

    "src/psg_adhd/__init__.py",
    "src/psg_adhd/batch_processing.py",
    "src/psg_adhd/cli.py",
    "src/psg_adhd/data.py",
    "src/psg_adhd/demo.py",
    "src/psg_adhd/evaluation.py",
    "src/psg_adhd/example_data.py",
    "src/psg_adhd/feature_extraction.py",
    "src/psg_adhd/graph_features.py",
    "src/psg_adhd/model_cli.py",
    "src/psg_adhd/model_workflow.py",
    "src/psg_adhd/modeling.py",
    "src/psg_adhd/synthetic_psg.py",

    "reference_implementations/published_pipeline/ADHD_Final_Code.py",
    "reference_implementations/published_pipeline/Feature extraction (Sleep stages).py",
    "reference_implementations/published_pipeline/README.md",

    "results/README.md",
    "results/RESULTS_PROVENANCE.csv",
    "results/published_performance_metrics.csv",
    "results/published_study_characteristics.csv",
    "results/sleep_stage_mapping.csv",

    "examples/README.md",
    "examples/synthetic_features/features_batch_1.csv",
    "examples/synthetic_features/features_batch_2.csv",
]


RAW_PSG_SUFFIXES = {
    ".fif",
    ".edf",
    ".bdf",
    ".set",
    ".vhdr",
    ".vmrk",
    ".eeg",
}


TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".csv",
}


SCAN_ROOTS = [
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "docs",
    ROOT / "configs",
    ROOT / "examples",
    ROOT / "results",
]


# Construct workstation-specific audit targets dynamically so this
# validator does not flag its own pattern definitions while scanning
# repository source files.
FORBIDDEN_WORKSTATION_STRINGS = [
    "D:" + "\\Amir",
    "E:" + "\\MSU",
    "C:" + "\\Users" + "\\ae1028",
    "OneDrive - " + "Mississippi State University",
]


EXPECTED_FEATURE_COLUMNS = [
    "Patient_ID",
    "Sleep_ID_1",
    "Sleep_ID_2",
    "Sleep_ID_3",
    "Sleep_ID_4",
    "Sleep_ID_5",
    "ADHD",
]


EXPECTED_METRICS = {
    "Accuracy": 0.72,
    "Precision": 0.71,
    "Recall": 0.85,
    "F1": 0.76,
}


def print_header(title: str) -> None:
    print("")
    print("=" * 68)
    print(title)
    print("=" * 68)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest().upper()


def git_tracked_files() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def check_required_files(
    failures: list[str],
) -> None:
    print_header(
        "1. REQUIRED REPOSITORY FILES"
    )

    missing = []

    for relative in REQUIRED_FILES:
        path = ROOT / relative

        if path.exists():
            print(
                f"[OK] {relative}"
            )
        else:
            print(
                f"[FAIL] Missing: {relative}"
            )

            missing.append(
                relative
            )

    if missing:
        failures.append(
            "Missing required repository files: "
            + ", ".join(missing)
        )


def check_historical_hashes(
    failures: list[str],
) -> None:
    print_header(
        "2. HISTORICAL RECORD INTEGRITY"
    )

    for path, expected in EXPECTED_HASHES.items():
        relative = path.relative_to(
            ROOT
        )

        if not path.exists():
            print(
                f"[FAIL] Missing: {relative}"
            )

            failures.append(
                f"Missing historical record: {relative}"
            )

            continue

        actual = sha256(
            path
        )

        if actual == expected:
            print(
                f"[OK] {relative}"
            )
        else:
            print(
                f"[FAIL] Hash mismatch: {relative}"
            )

            print(
                f"       Expected: {expected}"
            )

            print(
                f"       Actual:   {actual}"
            )

            failures.append(
                f"Historical hash mismatch: {relative}"
            )


def check_tracked_raw_psg(
    tracked: list[str],
    failures: list[str],
) -> None:
    print_header(
        "3. TRACKED RAW PSG DATA"
    )

    raw = [
        item
        for item in tracked
        if Path(
            item
        ).suffix.lower()
        in RAW_PSG_SUFFIXES
    ]

    if not raw:
        print(
            "[OK] No raw PSG files are tracked."
        )

        return

    for item in raw:
        print(
            f"[FAIL] Raw PSG file tracked: {item}"
        )

    failures.append(
        "Raw PSG files are tracked."
    )


def check_tracked_caches(
    tracked: list[str],
    failures: list[str],
) -> None:
    print_header(
        "4. TRACKED GENERATED CACHES"
    )

    bad = []

    for item in tracked:
        normalized = item.replace(
            "\\",
            "/",
        )

        if (
            "__pycache__/" in normalized
            or normalized.endswith(
                ".pyc"
            )
            or ".pytest_cache/" in normalized
        ):
            bad.append(
                item
            )

    if not bad:
        print(
            "[OK] No generated Python/test caches are tracked."
        )

        return

    for item in bad:
        print(
            f"[FAIL] Cache tracked: {item}"
        )

    failures.append(
        "Generated caches are tracked."
    )


def check_workstation_paths(
    failures: list[str],
) -> None:
    print_header(
        "5. WORKSTATION-SPECIFIC PATHS"
    )

    matches = []

    for root in SCAN_ROOTS:

        if not root.exists():
            continue

        for path in root.rglob(
            "*"
        ):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                not in TEXT_SUFFIXES
            ):
                continue

            if "__pycache__" in path.parts:
                continue

            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except OSError:
                continue

            for forbidden in (
                FORBIDDEN_WORKSTATION_STRINGS
            ):

                if forbidden in text:
                    matches.append(
                        (
                            path.relative_to(
                                ROOT
                            ),
                            forbidden,
                        )
                    )

    if not matches:
        print(
            "[OK] No targeted workstation paths found."
        )

        return

    for path, text in matches:
        print(
            f"[FAIL] {path}: {text}"
        )

    failures.append(
        "Workstation-specific paths were found."
    )


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


def check_public_synthetic_features(
    failures: list[str],
) -> None:
    print_header(
        "6. PUBLIC SYNTHETIC FEATURE DATA"
    )

    paths = [
        ROOT
        / "examples"
        / "synthetic_features"
        / "features_batch_1.csv",
        ROOT
        / "examples"
        / "synthetic_features"
        / "features_batch_2.csv",
    ]

    rows = []

    for path in paths:

        if not path.exists():
            failures.append(
                f"Missing synthetic example: {path.name}"
            )

            print(
                f"[FAIL] Missing: {path}"
            )

            return

        file_rows = load_csv(
            path
        )

        if not file_rows:
            failures.append(
                f"Synthetic example is empty: {path.name}"
            )

            print(
                f"[FAIL] Empty: {path.name}"
            )

            return

        if (
            list(
                file_rows[0].keys()
            )
            != EXPECTED_FEATURE_COLUMNS
        ):
            failures.append(
                f"Synthetic schema mismatch: {path.name}"
            )

            print(
                f"[FAIL] Schema mismatch: {path.name}"
            )

            return

        rows.extend(
            file_rows
        )

    if len(rows) != 20:
        print(
            f"[FAIL] Expected 20 synthetic rows; "
            f"found {len(rows)}."
        )

        failures.append(
            "Public synthetic participant count is incorrect."
        )

        return

    identifiers = [
        row[
            "Patient_ID"
        ]
        for row in rows
    ]

    if not all(
        re.fullmatch(
            r"SYN\d{3}",
            identifier,
        )
        for identifier in identifiers
    ):
        failures.append(
            "Public example contains a non-SYN identifier."
        )

        print(
            "[FAIL] Unexpected public synthetic identifier."
        )

        return

    if len(
        set(
            identifiers
        )
    ) != len(
        identifiers
    ):
        failures.append(
            "Duplicate public synthetic identifiers."
        )

        print(
            "[FAIL] Duplicate public synthetic identifiers."
        )

        return

    labels = {
        row[
            "ADHD"
        ]
        for row in rows
    }

    if labels != {
        "Y",
        "N",
    }:
        failures.append(
            "Unexpected synthetic label set."
        )

        print(
            f"[FAIL] Unexpected labels: {sorted(labels)}"
        )

        return

    print(
        "[OK] 20 artificial participants."
    )

    print(
        "[OK] Exact historical feature-table schema."
    )

    print(
        "[OK] All identifiers use SYN###."
    )

    print(
        "[OK] All identifiers are unique."
    )

    print(
        "[OK] Synthetic labels are exactly N and Y."
    )


def check_canonical_results(
    failures: list[str],
) -> None:
    print_header(
        "7. CANONICAL RESULT RECORDS"
    )

    metrics_path = (
        ROOT
        / "results"
        / "published_performance_metrics.csv"
    )

    rows = load_csv(
        metrics_path
    )

    actual = {
        row[
            "Metric"
        ]: float(
            row[
                "Value"
            ]
        )
        for row in rows
    }

    if actual != EXPECTED_METRICS:
        print(
            f"[FAIL] Unexpected canonical metrics: {actual}"
        )

        failures.append(
            "Canonical published metrics changed."
        )

    else:
        print(
            "[OK] Published metrics are exact."
        )

    provenance = (
        ROOT
        / "results"
        / "RESULTS_PROVENANCE.csv"
    ).read_text(
        encoding="utf-8"
    )

    required_phrases = [
        "Synthetic demo outputs",
        "Not scientific results",
    ]

    for phrase in required_phrases:

        if phrase not in provenance:
            print(
                f"[FAIL] Missing provenance phrase: {phrase}"
            )

            failures.append(
                "Result provenance boundary is incomplete."
            )

        else:
            print(
                f"[OK] Provenance contains: {phrase}"
            )


def check_runtime_output_boundary(
    tracked: list[str],
    failures: list[str],
) -> None:
    print_header(
        "8. GENERATED OUTPUT BOUNDARY"
    )

    generated_result_names = {
        "summary_metrics.csv",
        "fold_metrics.csv",
        "impurity_feature_importance.csv",
        "permutation_feature_importance.csv",
    }

    bad = []

    for item in tracked:
        path = Path(
            item
        )

        normalized = item.replace(
            "\\",
            "/",
        )

        if (
            normalized.startswith(
                "outputs/"
            )
            or path.name
            in generated_result_names
        ):
            bad.append(
                item
            )

    if not bad:
        print(
            "[OK] No generated runtime/model outputs are tracked."
        )

        return

    for item in bad:
        print(
            f"[FAIL] Generated artifact tracked: {item}"
        )

    failures.append(
        "Generated runtime outputs are tracked."
    )


def check_python_version() -> None:
    print_header(
        "9. PYTHON VERSION"
    )

    print(
        "Python:",
        sys.version.split()[0],
    )

    if sys.version_info < (
        3,
        9,
    ):
        raise RuntimeError(
            "Python 3.9 or newer is required."
        )

    print(
        "[OK] Supported Python major/minor version."
    )


def main() -> int:
    failures: list[str] = []

    check_python_version()

    tracked = git_tracked_files()

    check_required_files(
        failures
    )

    check_historical_hashes(
        failures
    )

    check_tracked_raw_psg(
        tracked,
        failures,
    )

    check_tracked_caches(
        tracked,
        failures,
    )

    check_workstation_paths(
        failures
    )

    check_public_synthetic_features(
        failures
    )

    check_canonical_results(
        failures
    )

    check_runtime_output_boundary(
        tracked,
        failures,
    )

    print_header(
        "REPOSITORY VALIDATION RESULT"
    )

    if failures:

        print(
            f"[FAIL] {len(failures)} validation issue(s):"
        )

        for failure in failures:
            print(
                f" - {failure}"
            )

        return 1

    print(
        "[OK] Repository validation passed."
    )

    print(
        "[OK] Historical provenance is intact."
    )

    print(
        "[OK] Public data boundary is intact."
    )

    print(
        "[OK] Canonical result boundary is intact."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )