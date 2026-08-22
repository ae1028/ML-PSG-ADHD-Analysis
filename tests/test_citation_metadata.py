"""Tests for repository and publication citation metadata."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CFF = ROOT / "CITATION.cff"
BIB = ROOT / "CITATION.bib"
CITATION_DOC = ROOT / "docs" / "CITATION.md"


TITLE = (
    "Machine Learning-Based Polysomnography Data Analysis for ADHD "
    "Diagnosis: A Focus on Sleep Stage-Based Biomarkers"
)

DOI = "10.1109/ISBI60581.2025.10981031"


def test_citation_files_exist():
    assert CFF.exists()
    assert BIB.exists()
    assert CITATION_DOC.exists()


def test_cff_contains_publication_metadata():
    text = CFF.read_text(
        encoding="utf-8"
    )

    assert "cff-version: 1.2.0" in text
    assert "type: software" in text
    assert TITLE in text
    assert DOI in text
    assert "preferred-citation:" in text


def test_preferred_citation_author_order():
    text = CFF.read_text(
        encoding="utf-8"
    )

    preferred = text.split(
        "preferred-citation:",
        1,
    )[1]

    family_names = [
        "Eskorouchi",
        "Wang",
        "Lee",
        "Nayak",
        "Ojeda",
        "Fan",
    ]

    positions = [
        preferred.index(
            f"family-names: {name}"
        )
        for name in family_names
    ]

    assert positions == sorted(
        positions
    )


def test_bibtex_contains_authoritative_citation():
    text = BIB.read_text(
        encoding="utf-8"
    )

    assert "@inproceedings{Eskorouchi2025ADHDPSG" in text
    assert TITLE in text
    assert DOI in text
    assert "pages     = {1--4}" in text

    expected_author_sequence = (
        "Eskorouchi, Amir and Wang, H. and Lee, J. W. "
        "and Nayak, V. H. and Ojeda, N. B. and Fan, L.-W."
    )

    assert expected_author_sequence in text