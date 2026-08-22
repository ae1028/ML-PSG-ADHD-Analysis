# Historical Modeling Workflow

This repository contains a cleaned and tested reconstruction of the
machine-learning workflow encoded in the historical:

`ADHD_Final_Code.py`

The exact historical source is preserved under:

`reference_implementations/published_pipeline/ADHD_Final_Code.py`

The reusable implementation is separated into:

- `src/psg_adhd/modeling.py`
- `src/psg_adhd/evaluation.py`
- `src/psg_adhd/model_workflow.py`
- `src/psg_adhd/model_cli.py`

## Input feature table

The historical modeling script assumes a participant-level table whose:

- first column is the participant identifier;
- final column is `ADHD`;
- intermediate columns are the numerical model features.

The historical code selects features positionally:

```python
features = features_df.iloc[:, 1:-1].values
```

Labels are encoded using:

```python
labels = (
    features_df["ADHD"] == "Y"
).astype(int).values
```

Therefore:

- `Y` becomes `1`;
- every other value becomes `0`.

## Historical sign-inversion augmentation

Before cross-validation, the historical script doubles the feature
matrix by appending the sign-inverted copy:

```python
X_augmented = np.vstack([
    features,
    -features,
])

y_augmented = np.hstack([
    labels,
    labels,
])
```

The resulting augmented dataset is then shuffled:

```python
X_combined, y_combined = shuffle(
    X_augmented,
    y_augmented,
)
```

No `random_state` was supplied to this shuffle.

This order is important:

`original participant features -> sign inversion -> duplicate labels -> shuffle -> nested CV`

The reconstruction preserves this historical behavior.

## Nested cross-validation

The historical outer loop is:

```python
KFold(
    n_splits=5,
    shuffle=True,
)
```

No `random_state` was supplied.

For every outer training fold, the historical script creates:

```python
RandomForestClassifier()
```

Again, no `random_state` was supplied.

The inner search is:

```python
GridSearchCV(
    estimator=rf_classifier,
    param_grid=param_grid,
    cv=5,
    scoring=scoring,
    refit="accuracy",
    n_jobs=-1,
)
```

The scoring dictionary contains:

- Accuracy
- Precision
- Recall
- F1

The parameter grid is:

| Hyperparameter | Historical values |
| --- | --- |
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 10, 20, 30 |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |

## Outer-fold evaluation

The selected estimator for each outer fold predicts the held-out outer
test set.

The historical script records:

```python
accuracy_score(...)
precision_score(...)
recall_score(...)
f1_score(...)
```

Average values are calculated with `numpy.mean` across the five outer
folds.

## Final-fold estimator

An important historical detail is that the function returns:

`best_rf_model`

after the outer loop finishes.

Consequently, the returned estimator is the best estimator selected
during the final outer fold, rather than an estimator chosen by
aggregating hyperparameter decisions across all outer folds.

This behavior is preserved explicitly in the reconstruction.

## Feature importance

After nested cross-validation, the historical script refits that
final-fold best estimator on the complete augmented dataset:

```python
best_rf_model.fit(
    X_combined,
    y_combined,
)
```

It then calculates two types of feature importance.

### Random Forest impurity importance

The historical script reads:

```python
best_rf_model.feature_importances_
```

### Permutation importance

The historical script also calculates:

```python
permutation_importance(
    best_rf_model,
    X_combined,
    y_combined,
    n_repeats=10,
    random_state=42,
)
```

Permutation importance is therefore the one explicitly seeded stochastic
operation in this part of the historical workflow.

## Historical feature display names

The historical presentation mapping is:

| Feature column | Display name |
| --- | --- |
| `Sleep_ID_1` | Wake |
| `Sleep_ID_2` | Sleep Stage 1 |
| `Sleep_ID_3` | Sleep Stage 2 |
| `Sleep_ID_4` | Sleep Stage 3-4 |
| `Sleep_ID_5` | Rapid Eye Movement |

The reconstruction preserves these historical presentation names.

## Running the reconstructed modeling workflow

From a source checkout:

```bash
python scripts/run_modeling.py \
    --feature-dir <feature-csv-directory> \
    --output-dir <result-directory>
```

The workflow writes:

- `summary_metrics.csv`
- `fold_metrics.csv`
- `impurity_feature_importance.csv`
- `permutation_feature_importance.csv`

The reconstructed workflow does not automatically serialize the fitted
Random Forest because the historical script did not establish a
canonical persisted model artifact.

## Historical versus recommended methodology

The functions described here reproduce historical behavior.

They are not intended to silently redefine or modernize that behavior.

Any future leakage-safe, deterministic, group-aware, or otherwise
methodologically revised workflow should be implemented as a clearly
separate analysis mode rather than replacing the historical
reconstruction.