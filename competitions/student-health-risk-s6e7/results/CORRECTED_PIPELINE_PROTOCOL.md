# S6E7 — Corrected end-to-end pipeline protocol

## Objective

Build an autonomous, reproducible pipeline for the Kaggle Playground Series S6E7
competition. The official metric is **balanced accuracy**, not ordinary accuracy.
The current protected Kaggle reference is submission `55674809` at private
balanced accuracy `0.874570`.

No public solution code, public predictions, pseudo-labels, anchor CSVs or
external data enter the first corrected benchmark.

## Phase 0 — Frozen truth and safety gates

1. Read `train.csv`, `test.csv`, `sample_submission.csv`.
2. Verify target classes and submission schema.
3. Exclude `id` from every model feature matrix.
4. Record SHA-256 hashes and row/column counts.
5. Freeze one `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` split.
6. Never select parameters from the Kaggle private score.
7. Preserve the current Kaggle champion as rollback.

## Phase 1 — Metric-correct audit

Report for every model:

- balanced accuracy;
- ordinary accuracy only as a secondary diagnostic;
- macro-F1;
- recall and balanced recall for `fit`, `unhealthy`, `at-risk`;
- confusion matrix;
- fold mean and standard deviation.

The majority baseline is expected to have balanced accuracy near `1/3`, despite
high ordinary accuracy.

## Phase 2 — Fold-safe feature construction

### Raw features

- Numeric variables: median imputation fitted inside each fold.
- Categorical variables: explicit `Missing` category.
- No ID-derived features.

### Engineered categorical interaction

Create `sleep_bin` from `sleep_duration` using quantile edges fitted on the
outer training fold only. Apply those edges unchanged to outer validation/test.
Create:

```text
stress_sleep_interact = stress_level + "_" + sleep_bin
```

### Categorical target encoding

For each outer fold:

1. Fit an inner cross-fitting scheme on the outer training portion.
2. Encode each categorical/interactions column as a three-class probability
   vector using only inner-train labels.
3. Use the full outer-training mapping only for outer validation/test.
4. Unseen categories receive the outer-training class-prior vector.
5. No target encoding is computed from outer validation labels.

Encoded columns:

```text
stress_level
physical_activity_level
diet_type
gender
smoking_alcohol
sleep_quality
stress_sleep_interact
```

Numeric columns are concatenated after fold-fitted median imputation.

## Phase 3 — Corrected model bake-off

Run the following fixed models before tuning:

1. CatBoost on raw mixed features with balanced class weights.
2. LightGBM on cross-fitted target-encoded features with balanced sample weights.
3. XGBoost or HistGradientBoosting only if the first two disagree materially.

Use conservative fixed parameters first. Save OOF probability matrices for all
models. Do not select using ordinary accuracy.

## Phase 4 — OOF selection and blend

1. Compute balanced accuracy from each OOF matrix.
2. Test only a small predefined blend grid, e.g. CatBoost weights:
   `0.0, 0.25, 0.50, 0.75, 1.0`.
3. Select only if the gain is stable across folds, not just on aggregate OOF.
4. Any per-class threshold correction must be learned from OOF probabilities
   only, with a small bounded search and a no-regression guardrail.
5. Preserve the best raw argmax model as rollback.

Promotion gate for a new local candidate:

```text
balanced-accuracy gain >= 0.001
and no fold regression larger than 0.002
and minority recalls do not collapse
```

## Phase 5 — Optional extensions, kept separate

Only after the clean core is established:

- high-confidence pseudo-labeling, marked artifact-assisted;
- adversarial weighting, evaluated under balanced accuracy;
- public-method reproduction, separate from autonomous score;
- Optuna, bounded around the corrected core.

None of these may overwrite the clean benchmark.

## Phase 6 — Final training and submission gate

1. Freeze the selected feature recipe and parameters.
2. Refit on all training rows.
3. Predict test labels.
4. Verify exact sample columns, row count, ID order, labels, uniqueness and no
   missing values.
5. Save metadata with fold scores, model parameters, hash and artifact class.
6. Submit at most one promoted candidate.
7. Record public/private Kaggle score and compare to the protected champion.
8. Roll back immediately if private balanced accuracy regresses.

## Expected deliverables

```text
output/corrected_audit.json
output/corrected_oof_metrics.json
output/corrected_oof_predictions.npz
output/submission_corrected_core.csv
output/CORRECTED_PIPELINE_REPORT.md
```

This protocol is frozen before execution.
