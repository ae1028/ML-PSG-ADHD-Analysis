"""Tests for historical post-modeling evaluation behavior."""

import os

import numpy as np
import pandas as pd

import psg_adhd.evaluation as evaluation


def test_load_features_matches_historical_csv_concatenation(
    tmp_path,
    monkeypatch,
):
    first = pd.DataFrame(
        {
            "Patient_ID": ["P02"],
            "Sleep_ID_1": [2.0],
            "ADHD": ["Y"],
        }
    )

    second = pd.DataFrame(
        {
            "Patient_ID": ["P01"],
            "Sleep_ID_1": [1.0],
            "ADHD": ["N"],
        }
    )

    first.to_csv(
        tmp_path / "b.csv",
        index=False,
    )

    second.to_csv(
        tmp_path / "a.csv",
        index=False,
    )

    (
        tmp_path / "ignore.txt"
    ).write_text(
        "not a CSV",
        encoding="utf-8",
    )

    real_listdir = os.listdir

    def fake_listdir(directory):
        assert str(directory) == str(tmp_path)

        return [
            "b.csv",
            "ignore.txt",
            "a.csv",
        ]

    monkeypatch.setattr(
        evaluation.os,
        "listdir",
        fake_listdir,
    )

    actual = evaluation.load_features(
        tmp_path
    )

    expected = pd.concat(
        [
            first,
            second,
        ],
        ignore_index=True,
    )

    pd.testing.assert_frame_equal(
        actual,
        expected,
    )

    monkeypatch.setattr(
        evaluation.os,
        "listdir",
        real_listdir,
    )


def test_load_features_returns_none_when_no_csv_exists(
    tmp_path,
):
    (
        tmp_path / "notes.txt"
    ).write_text(
        "nothing",
        encoding="utf-8",
    )

    actual = evaluation.load_features(
        tmp_path
    )

    assert actual is None


def test_historical_stage_rename_is_exact():
    feature_df = pd.DataFrame(
        {
            "Patient_ID": ["P01"],
            "Sleep_ID_1": [1.0],
            "Sleep_ID_2": [2.0],
            "Sleep_ID_3": [3.0],
            "Sleep_ID_4": [4.0],
            "Sleep_ID_5": [5.0],
            "ADHD": ["N"],
        }
    )

    feature_names = (
        evaluation.apply_historical_stage_names(
            feature_df
        )
    )

    assert feature_names == [
        "Wake",
        "Sleep Stage 1",
        "Sleep Stage 2",
        "Sleep Stage 3-4",
        "Rapid Eye Movement",
    ]

    assert list(
        feature_df.columns
    ) == [
        "Patient_ID",
        "Wake",
        "Sleep Stage 1",
        "Sleep Stage 2",
        "Sleep Stage 3-4",
        "Rapid Eye Movement",
        "ADHD",
    ]


def test_refit_uses_complete_augmented_dataset(
    monkeypatch,
):
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [-1.0, -2.0],
            [-3.0, -4.0],
        ]
    )

    y = np.array(
        [
            0,
            1,
            0,
            1,
        ]
    )

    feature_df = pd.DataFrame(
        {
            "Patient_ID": [
                "P01",
                "P02",
            ],
            "Sleep_ID_1": [
                1.0,
                3.0,
            ],
            "Sleep_ID_2": [
                2.0,
                4.0,
            ],
            "ADHD": [
                "N",
                "Y",
            ],
        }
    )

    class FakeModel:
        def __init__(self):
            self.fit_X = None
            self.fit_y = None
            self.feature_importances_ = np.array(
                [0.25, 0.75]
            )

        def fit(self, X_input, y_input):
            self.fit_X = X_input.copy()
            self.fit_y = y_input.copy()
            return self

    class FakePermutationResult:
        importances_mean = np.array(
            [0.10, 0.40]
        )

        importances_std = np.array(
            [0.01, 0.02]
        )

    permutation_call = {}

    def fake_permutation_importance(
        estimator,
        X_input,
        y_input,
        n_repeats,
        random_state,
    ):
        permutation_call[
            "estimator"
        ] = estimator

        permutation_call[
            "X"
        ] = X_input.copy()

        permutation_call[
            "y"
        ] = y_input.copy()

        permutation_call[
            "n_repeats"
        ] = n_repeats

        permutation_call[
            "random_state"
        ] = random_state

        return FakePermutationResult()

    monkeypatch.setattr(
        evaluation,
        "permutation_importance",
        fake_permutation_importance,
    )

    model = FakeModel()

    (
        impurity,
        feature_names,
        permutation_df,
    ) = evaluation.refit_and_compute_importances(
        model,
        X,
        y,
        feature_df,
    )

    np.testing.assert_array_equal(
        model.fit_X,
        X,
    )

    np.testing.assert_array_equal(
        model.fit_y,
        y,
    )

    np.testing.assert_array_equal(
        impurity,
        np.array(
            [0.25, 0.75]
        ),
    )

    assert feature_names == [
        "Wake",
        "Sleep Stage 1",
    ]

    assert (
        permutation_call[
            "estimator"
        ]
        is model
    )

    np.testing.assert_array_equal(
        permutation_call["X"],
        X,
    )

    np.testing.assert_array_equal(
        permutation_call["y"],
        y,
    )

    assert (
        permutation_call[
            "n_repeats"
        ]
        == 10
    )

    assert (
        permutation_call[
            "random_state"
        ]
        == 42
    )

    assert list(
        permutation_df["Feature"]
    ) == [
        "Sleep Stage 1",
        "Wake",
    ]


def test_permutation_dataframe_columns_and_sorting(
    monkeypatch,
):
    X = np.ones(
        (4, 3)
    )

    y = np.array(
        [0, 1, 0, 1]
    )

    feature_df = pd.DataFrame(
        {
            "Patient_ID": [
                "P01",
                "P02",
            ],
            "Sleep_ID_1": [
                1.0,
                2.0,
            ],
            "Sleep_ID_2": [
                2.0,
                3.0,
            ],
            "Sleep_ID_3": [
                3.0,
                4.0,
            ],
            "ADHD": [
                "N",
                "Y",
            ],
        }
    )

    class FakeModel:
        feature_importances_ = np.array(
            [0.2, 0.3, 0.5]
        )

        def fit(self, X, y):
            return self

    class FakePermutationResult:
        importances_mean = np.array(
            [
                0.15,
                0.60,
                0.30,
            ]
        )

        importances_std = np.array(
            [
                0.01,
                0.04,
                0.02,
            ]
        )

    monkeypatch.setattr(
        evaluation,
        "permutation_importance",
        lambda *args, **kwargs: FakePermutationResult(),
    )

    _, _, result = (
        evaluation.refit_and_compute_importances(
            FakeModel(),
            X,
            y,
            feature_df,
        )
    )

    assert list(
        result.columns
    ) == [
        "Feature",
        "Importance Mean",
        "Importance Std",
    ]

    assert list(
        result["Importance Mean"]
    ) == [
        0.60,
        0.30,
        0.15,
    ]

    assert list(
        result["Feature"]
    ) == [
        "Sleep Stage 1",
        "Sleep Stage 2",
        "Wake",
    ]