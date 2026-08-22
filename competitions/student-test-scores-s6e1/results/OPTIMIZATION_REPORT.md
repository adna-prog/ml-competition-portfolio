# S6E1 optimization report

## Protected baseline

```text
Submission: 55693166
Private RMSE: 8.74614
Local OOF RMSE: 8.759351
```

## Optuna

Eight bounded LightGBM trials were screened on one fold and the best three were
promoted to all three frozen folds.

Best parameters:

```text
learning_rate     0.020685
num_leaves        63
min_data_in_leaf 550
feature_fraction  0.692812
bagging_fraction  0.926591
lambda_l1         0.012046
lambda_l2         1.109286
iterations        1799
```

```text
Optuna OOF RMSE : 8.755139
Gain            : -0.004212 RMSE
Private RMSE    : 8.74005
Submission      : 55695288
```

## Public clean feature reproduction

An audited public notebook suggested three unsupervised features:

- high attendance × study hours;
- ideal sleep;
- ideal study.

Reproduction with LightGBM, RMSE and frozen folds:

```text
Optuna baseline       : 8.755139 OOF
Public clean features : 8.753247 OOF
Gain                  : -0.001892 RMSE
Private RMSE         : 8.73988
Submission             : 55695651
```

This is the current protected S6E1 result.

## Capping test

Fold-fitted ±3 standard-deviation capping was neutral:

```text
Capping OOF RMSE : 8.753247
```

No submission was generated.

## Decision

Current best:

```text
Private RMSE : 8.73988
Submission    : 55695651
```

The public notebook advertises `8.68897`, but its validation is not directly
comparable to our frozen three-fold protocol and no external predictions were
used in our reproduction. Further gains require a genuinely new representation
or a carefully validated blend; broad tuning and capping are closed.
