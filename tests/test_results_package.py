"""Tests for canonical publication-result records."""

from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPOSITORY_ROOT / "results"


def test_published_metrics_are_exact():
    metrics = pd.read_csv(
        RESULTS_DIR
        / "published_performance_metrics.csv"
    )

    actual = dict(
        zip(
            metrics["Metric"],
            metrics["Value"],
        )
    )

    assert actual == {
        "Accuracy": 0.72,
        "Precision": 0.71,
        "Recall": 0.85,
        "F1": 0.76,
    }


def test_published_study_characteristics():
    characteristics = pd.read_csv(
        RESULTS_DIR
        / "published_study_characteristics.csv"
    )

    values = dict(
        zip(
            characteristics["Characteristic"],
            characteristics["Value"],
        )
    )

    assert values["Participants"] == "48"
    assert values["ADHD participants"] == "25"
    assert values["Control participants"] == "23"
    assert values["ADHD epochs"] == "20294"
    assert values["Control epochs"] == "19968"
    assert values["PSG channels"] == "17"


def test_sleep_stage_mapping():
    mapping = pd.read_csv(
        RESULTS_DIR
        / "sleep_stage_mapping.csv"
    )

    assert list(
        mapping["Historical_Feature"]
    ) == [
        "Sleep_ID_1",
        "Sleep_ID_2",
        "Sleep_ID_3",
        "Sleep_ID_4",
        "Sleep_ID_5",
    ]

    assert list(
        mapping["Published_Presentation_Name"]
    ) == [
        "Wake",
        "Sleep Stage 1",
        "Sleep Stage 2",
        "Sleep Stage 3-4",
        "Rapid Eye Movement",
    ]


def test_results_provenance_distinguishes_synthetic_outputs():
    provenance = (
        RESULTS_DIR
        / "RESULTS_PROVENANCE.csv"
    ).read_text(
        encoding="utf-8"
    )

    assert "Synthetic" in provenance
    assert "Not scientific results" in provenance


def test_results_documentation_exists():
    required = [
        RESULTS_DIR / "README.md",
        REPOSITORY_ROOT
        / "docs"
        / "RESULTS.md",
        REPOSITORY_ROOT
        / "docs"
        / "PUBLICATION.md",
    ]

    assert all(
        path.exists()
        for path in required
    )