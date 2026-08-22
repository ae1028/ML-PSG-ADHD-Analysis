"""Tests for the public feature-extraction CLI."""

from pathlib import Path

import pytest

import psg_adhd.cli as cli


def test_parser_historical_defaults():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "--input-dir",
            "input",
            "--output-dir",
            "output",
        ]
    )

    assert args.input_dir == Path("input")
    assert args.output_dir == Path("output")
    assert args.batch_size == 10
    assert args.pattern == "*.fif"
    assert args.output_prefix == "features_batch"
    assert args.id_column == "ID"
    assert args.label_column == "ADHD"
    assert args.overwrite is False


def test_cli_delegates_to_batch_processor(
    tmp_path,
    monkeypatch,
    capsys,
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    generated = [
        output_dir / "features_batch_1.csv",
        output_dir / "features_batch_2.csv",
    ]

    captured_call = {}

    def fake_write_feature_batches(**kwargs):
        captured_call.update(kwargs)
        return generated

    monkeypatch.setattr(
        cli,
        "write_feature_batches",
        fake_write_feature_batches,
    )

    actual = cli.run(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "7",
            "--pattern",
            "*-epo.fif",
            "--output-prefix",
            "study_features",
            "--id-column",
            "ID",
            "--label-column",
            "ADHD",
            "--overwrite",
        ]
    )

    assert actual == generated

    assert captured_call == {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "batch_size": 7,
        "pattern": "*-epo.fif",
        "output_prefix": "study_features",
        "overwrite": True,
        "id_column": "ID",
        "label_column": "ADHD",
    }

    output = capsys.readouterr().out

    assert "Feature extraction complete." in output
    assert "Generated 2 file(s)" in output


def test_cli_rejects_nonpositive_batch_size():
    with pytest.raises(SystemExit):
        cli.run(
            [
                "--input-dir",
                "input",
                "--output-dir",
                "output",
                "--batch-size",
                "0",
            ]
        )