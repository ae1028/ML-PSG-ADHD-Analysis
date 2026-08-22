"""Tests for the historical-compatible modeling workflow."""

import numpy as np
import pandas as pd

import psg_adhd.modeling as modeling


def make_feature_table():
    """Create a deterministic participant-level feature table."""
    return pd.DataFrame(
        {
            "Patient_ID": [
                "P01",
                "P02",
                "P03",
            ],
            "Sleep_ID_1": [
                1.0,
                2.0,
                3.0,
            ],
            "Sleep_ID_2": [
                4.0,
                5.0,
                6.0,
            ],
            "ADHD": [
                "N",
                "Y",
                "N",
            ],
        }
    )


def test_historical_parameter_grid_is_exact():
    assert modeling.HISTORICAL_PARAMETER_GRID == {
        "n_estimators": [
            100,
            200,
            300,
        ],
        "max_depth": [
            10,
            20,
            30,
        ],
        "min_samples_split": [
            2,
            5,
            10,
        ],
        "min_samples_leaf": [
            1,
            2,
            4,
        ],
    }


def test_prepare_data_matches_historical_augmentation(
    monkeypatch,
):
    feature_df = make_feature_table()

    captured = {}

    def fake_shuffle(X, y):
        captured["X"] = X.copy()
        captured["y"] = y.copy()

        # Return unchanged so the exact pre-shuffle historical
        # augmentation can be inspected deterministically.
        return X, y

    monkeypatch.setattr(
        modeling,
        "shuffle",
        fake_shuffle,
    )

    X_combined, y_combined, X, y = (
        modeling.prepare_data(
            feature_df
        )
    )

    expected_X = np.array(
        [
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
        ]
    )

    expected_y = np.array(
        [
            0,
            1,
            0,
        ]
    )

    expected_augmented_X = np.vstack(
        [
            expected_X,
            -expected_X,
        ]
    )

    expected_augmented_y = np.hstack(
        [
            expected_y,
            expected_y,
        ]
    )

    np.testing.assert_array_equal(
        X,
        expected_X,
    )

    np.testing.assert_array_equal(
        y,
        expected_y,
    )

    np.testing.assert_array_equal(
        captured["X"],
        expected_augmented_X,
    )

    np.testing.assert_array_equal(
        captured["y"],
        expected_augmented_y,
    )

    np.testing.assert_array_equal(
        X_combined,
        expected_augmented_X,
    )

    np.testing.assert_array_equal(
        y_combined,
        expected_augmented_y,
    )


def test_prepare_data_treats_only_Y_as_positive(
    monkeypatch,
):
    feature_df = pd.DataFrame(
        {
            "Patient_ID": [
                "P01",
                "P02",
                "P03",
                "P04",
            ],
            "Sleep_ID_1": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "ADHD": [
                "Y",
                "N",
                "Unknown",
                "",
            ],
        }
    )

    monkeypatch.setattr(
        modeling,
        "shuffle",
        lambda X, y: (X, y),
    )

    _, _, _, labels = (
        modeling.prepare_data(
            feature_df
        )
    )

    np.testing.assert_array_equal(
        labels,
        np.array(
            [1, 0, 0, 0]
        ),
    )


def test_historical_scoring_has_expected_metrics():
    scoring = (
        modeling.historical_scoring()
    )

    assert set(scoring) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
    }

    assert scoring["accuracy"] == "accuracy"


def test_nested_cv_preserves_historical_configuration(
    monkeypatch,
):
    X = np.arange(
        40,
        dtype=float,
    ).reshape(
        10,
        4,
    )

    y = np.array(
        [
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
        ]
    )

    calls = {
        "kfold": [],
        "rf": [],
        "grid": [],
        "fit": [],
    }

    class FakeKFold:
        def __init__(
            self,
            n_splits,
            shuffle,
        ):
            calls["kfold"].append(
                {
                    "n_splits": n_splits,
                    "shuffle": shuffle,
                }
            )

        def split(self, X):
            # Two deterministic outer folds are enough to verify the
            # orchestration without fitting real forests.
            yield (
                np.array(
                    [0, 1, 2, 3, 4]
                ),
                np.array(
                    [5, 6, 7, 8, 9]
                ),
            )

            yield (
                np.array(
                    [5, 6, 7, 8, 9]
                ),
                np.array(
                    [0, 1, 2, 3, 4]
                ),
            )

    class FakeRandomForest:
        def __init__(self):
            calls["rf"].append({})

    class FakeBestModel:
        def __init__(self, prediction):
            self.prediction = prediction

        def predict(self, X):
            return np.full(
                len(X),
                self.prediction,
                dtype=int,
            )

    class FakeGridSearch:
        counter = 0

        def __init__(
            self,
            estimator,
            param_grid,
            cv,
            scoring,
            refit,
            n_jobs,
        ):
            calls["grid"].append(
                {
                    "estimator": estimator,
                    "param_grid": param_grid,
                    "cv": cv,
                    "scoring_keys": set(
                        scoring
                    ),
                    "refit": refit,
                    "n_jobs": n_jobs,
                }
            )

            prediction = (
                FakeGridSearch.counter
                % 2
            )

            self.best_estimator_ = (
                FakeBestModel(
                    prediction
                )
            )

            FakeGridSearch.counter += 1

        def fit(self, X, y):
            calls["fit"].append(
                (
                    X.copy(),
                    y.copy(),
                )
            )

            return self

    monkeypatch.setattr(
        modeling,
        "KFold",
        FakeKFold,
    )

    monkeypatch.setattr(
        modeling,
        "RandomForestClassifier",
        FakeRandomForest,
    )

    monkeypatch.setattr(
        modeling,
        "GridSearchCV",
        FakeGridSearch,
    )

    test_grid = {
        "n_estimators": [100]
    }

    result = (
        modeling.perform_nested_cross_validation(
            X,
            y,
            param_grid=test_grid,
            n_splits=2,
        )
    )

    assert calls["kfold"] == [
        {
            "n_splits": 2,
            "shuffle": True,
        }
    ]

    assert len(
        calls["rf"]
    ) == 2

    assert len(
        calls["grid"]
    ) == 2

    for call in calls["grid"]:
        assert (
            call["param_grid"]
            == test_grid
        )

        assert call["cv"] == 5

        assert call[
            "scoring_keys"
        ] == {
            "accuracy",
            "precision",
            "recall",
            "f1",
        }

        assert (
            call["refit"]
            == "accuracy"
        )

        assert (
            call["n_jobs"]
            == -1
        )

    assert len(
        calls["fit"]
    ) == 2

    # Historical return structure:
    # four means + four fold-score lists + final-fold model.
    assert len(result) == 9

    assert len(result[4]) == 2
    assert len(result[5]) == 2
    assert len(result[6]) == 2
    assert len(result[7]) == 2

    # The historical function returns the estimator generated
    # in the FINAL outer fold.
    assert isinstance(
        result[8],
        FakeBestModel,
    )

    assert (
        result[8].prediction
        == 1
    )


def test_nested_cv_averages_fold_metrics(
    monkeypatch,
):
    X = np.arange(
        24,
        dtype=float,
    ).reshape(
        6,
        4,
    )

    y = np.array(
        [
            0,
            1,
            0,
            1,
            0,
            1,
        ]
    )

    class FakeKFold:
        def __init__(
            self,
            n_splits,
            shuffle,
        ):
            pass

        def split(self, X):
            yield (
                np.array(
                    [0, 1, 2]
                ),
                np.array(
                    [3, 4, 5]
                ),
            )

            yield (
                np.array(
                    [3, 4, 5]
                ),
                np.array(
                    [0, 1, 2]
                ),
            )

    class FakeRandomForest:
        pass

    class PerfectModel:
        def __init__(self):
            self.expected = None

        def predict(self, X):
            # The test deliberately constructs predictions from
            # row identity so both outer folds are perfect.
            row_ids = (
                X[:, 0] // 4
            ).astype(int)

            return np.array(
                [
                    y[row_id]
                    for row_id
                    in row_ids
                ]
            )

    class FakeGridSearch:
        def __init__(
            self,
            estimator,
            param_grid,
            cv,
            scoring,
            refit,
            n_jobs,
        ):
            self.best_estimator_ = (
                PerfectModel()
            )

        def fit(self, X, y):
            return self

    monkeypatch.setattr(
        modeling,
        "KFold",
        FakeKFold,
    )

    monkeypatch.setattr(
        modeling,
        "RandomForestClassifier",
        FakeRandomForest,
    )

    monkeypatch.setattr(
        modeling,
        "GridSearchCV",
        FakeGridSearch,
    )

    (
        avg_accuracy,
        avg_precision,
        avg_recall,
        avg_f1,
        accuracy_scores,
        precision_scores,
        recall_scores,
        f1_scores,
        _,
    ) = modeling.perform_nested_cross_validation(
        X,
        y,
        param_grid={
            "n_estimators": [100]
        },
        n_splits=2,
    )

    assert avg_accuracy == 1.0
    assert avg_precision == 1.0
    assert avg_recall == 1.0
    assert avg_f1 == 1.0

    assert accuracy_scores == [
        1.0,
        1.0,
    ]

    assert precision_scores == [
        1.0,
        1.0,
    ]

    assert recall_scores == [
        1.0,
        1.0,
    ]

    assert f1_scores == [
        1.0,
        1.0,
    ]