# Gap localization — local CV vs Kaggle S6E7

## Observed gap

- Local LightGBM OOF: `0.967069417`
- Kaggle private: `0.874570`
- Difference: `0.092499417`

The submission format is not the cause: columns, row count, IDs, labels and
missing values were all verified.

## Shift tests

- Linear adversarial train/test AUC: `0.525074`
- LightGBM adversarial train/test AUC: `0.652806` in 3-fold CV.
- Full in-sample LightGBM adversarial AUC: `0.667500`.

Therefore the train and test sets have similar marginal/linear distributions,
but substantially different multivariate interactions.

## Main shift contributors

The adversarial LightGBM gain importance is concentrated in:

1. `water_intake`
2. `calorie_expenditure`
3. `bmi`
4. `physical_activity_level`
5. `smoking_alcohol`
6. `step_count`
7. `diet_type`

Dropping the four leading features reduces adversarial AUC from approximately
`0.6675` to `0.5594`; dropping only water/calories gives `0.6052`.

The largest numeric joint-distribution divergence is:

- `calorie_expenditure × water_intake`: Jensen–Shannon distance `0.0613`.
- `bmi × step_count`: `0.0250`.
- `bmi × exercise_duration`: `0.0124`.

Single-feature means and missingness rates are nearly identical, which explains
why the earlier simple adversarial check missed the problem.

## Target model context

The target LightGBM relies most on:

- `stress_level`
- `sleep_duration`
- `physical_activity_level`
- `bmi`

This overlaps with the shifted multivariate blocks, especially activity/body
features. The local model is therefore learning relationships that do not
transfer to Kaggle test as well as the random/block CV suggests.

## Current conclusion

The discrepancy appears primarily at the **multivariate covariate-shift level**,
not at file format, ID leakage, class-prior drift, or simple marginal shift.

Next technically justified experiment: adversarial reweighting or a model trained
on features less sensitive to the shifted interaction blocks, evaluated using
importance-weighted CV. Do not use the private score for fitting weights.
