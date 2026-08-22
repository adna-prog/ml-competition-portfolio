# Public writeup audit — S6E7

## Critical protocol correction

The official metric is **balanced accuracy**, not ordinary accuracy.

Verified from:

- public repository `Assem-ElQersh/Student-Health-Risk-Kaggle-s6e7`, README;
- notebook code using `balanced_accuracy_score`;
- leaderboard context and multiple independent public repositories.

Our earlier local and submission analysis optimized ordinary accuracy. This makes
our comparisons invalid for the competition objective and explains much of the
apparent local/private discrepancy.

## Public reference pipeline

The audited repository reports an organic public LB around `0.94986` to
`0.95034`, with the following important ingredients:

1. **Balanced accuracy evaluation** on every fold.
2. `class_weight='balanced'`, `auto_class_weights='Balanced'`, or balanced
   sample weights.
3. Categorical target encoding, computed fold-by-fold:
   - `stress_level`
   - `physical_activity_level`
   - `diet_type`
   - `gender`
   - `smoking_alcohol`
   - `sleep_quality`
   - engineered `stress_level × sleep_bin`.
4. Numeric median imputation.
5. CatBoost / LightGBM / XGBoost / HistGradientBoosting bake-off.
6. Probability blending and, later, carefully learned per-class thresholds.
7. High-confidence pseudo-labeling was tested, but must remain a separate
   artifact-assisted experiment, not part of the first autonomous benchmark.

The repository explicitly reports:

- categorical target-encoded CatBoost: approximately `0.9494` OOF;
- uniform tree blend: approximately `0.94955` OOF;
- public LB around `0.94986`;
- later clean organic variants around `0.95034`.

## What this rules out

- Our `0.967` accuracy OOF was the wrong metric.
- The blend regression from `0.87457` to `0.87180` is not informative for balanced
  accuracy optimization.
- The adversarial-shift investigation remains useful, but its conclusions must
  be revisited under balanced accuracy.
- We should not continue tuning ordinary-accuracy LightGBM.

## Reproduction policy

The next run will reproduce only the validation-safe core:

- balanced accuracy;
- fold-safe categorical target encoding;
- balanced class weights;
- `stress_level × sleep_bin` interaction;
- CatBoost/LightGBM comparison.

Pseudo-labeling, public anchor files, and any external predictions remain
excluded until the clean baseline is established.

Source commit inspected: `7566766399e099512d0bd4016b13956bc1bc2825`.
