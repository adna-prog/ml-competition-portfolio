# Competition-grade Kaggle pipeline — S6E1 to S6E6

This is the standard operating procedure for future Playground Series work.
It is designed from the lessons of S6E7, S6E8 and the previous competition audits.

## Operating principles

1. Verify the official metric before reading scores or training models.
2. Keep a clean from-scratch track separate from public-artifact-assisted work.
3. Freeze validation before model search.
4. Protect every real leaderboard champion and maintain rollback files.
5. Never select on ordinary accuracy when the official metric is different.
6. Treat public leaderboard feedback as secondary; private/final score is decisive.
7. Stop experiments when the predeclared gain gate is not met.
8. Publish only after the competition is frozen and the result is reproducible.

## Phase 0 — Competition intake

For every competition:

- record URL, title, season/episode, start/deadline and status;
- read description, evaluation and rules pages;
- identify target, submission schema and official metric;
- determine whether the competition is active or closed;
- download train/test/sample submission;
- record SHA-256 hashes, sizes and row/column counts;
- create a project brief and a frozen experiment registry.

Required metric contract:

```text
official_metric
optimization_metric
secondary_diagnostics
public_score_semantics
private_score_semantics
```

## Phase 1 — Baseline audit

Before optimization:

- validate schema and ID order;
- inspect target balance and missingness;
- detect exact duplicates and train/test ID overlap;
- run adversarial validation, but exclude ID from model features unless a
  hypothesis is explicitly tested;
- create frozen stratified/group/time folds appropriate to the data;
- establish majority, linear and one tree baseline;
- submit exactly one baseline when the competition is active.

Record:

```text
local OOF score
holdout score if available
public score
private score when available
submission ID
runtime and cost
```

## Phase 2 — Correct validation

Choose validation based on the data:

- stratified K-fold for IID classification;
- group/location folds for grouped data;
- temporal walk-forward for time series;
- spatial lockbox for geographic data;
- historical-year walk-forward for historical competitions.

For large or expensive pipelines:

- freeze folds and seeds;
- use a development OOF partition for fitting blend/calibration;
- keep a decision-validation OOF partition untouched;
- preserve a real holdout and open it only under a predeclared gate.

Every expensive run must save:

```text
oof_predictions.npz
 test_predictions.npz
fold_metadata.json
experiment_metadata.json
submission.csv
```

## Phase 3 — Model ladder

Run one family at a time:

1. simple statistical/linear baseline;
2. CatBoost or equivalent native categorical model;
3. LightGBM/XGBoost;
4. one orthogonal neural or foundation model only if diversity is plausible;
5. imputation and feature blocks one at a time;
6. blend only after measuring prediction correlation.

Promotion gate:

```text
improvement >= predeclared useful gain
positive on all/most frozen folds
no protected subgroup collapse
no holdout regression
```

Tiny OOF gains below the noise floor are not promoted.

## Phase 4 — Optuna policy

Optuna is used only after:

- the metric is correct;
- the validation protocol is frozen;
- the baseline and one strong model are known;
- a structural feature/model hypothesis exists.

Recommended sequence:

1. cheap one-fold screening;
2. promote only the top few trials;
3. evaluate promoted trials on all frozen folds;
4. validate on untouched decision OOF/holdout;
5. retrain once on full train;
6. submit only if the useful-gain gate passes.

Rules:

- 10–30 trials by default;
- bounded search spaces;
- fixed seed and study artifact;
- no FLAML/AutoML when rules prohibit it;
- no private score in the objective;
- never overwrite the best rollback.

## Phase 5 — Score-gap diagnosis

After the first real submission, compare local and Kaggle results:

```text
local OOF → holdout → public → private
```

If the gap is large, test in this order:

1. submission schema and ID order;
2. official metric mismatch;
3. target encoding/imputation leakage;
4. target/class prior mismatch;
5. marginal train/test shift;
6. multivariate adversarial shift;
7. conditional shift;
8. public/private leaderboard split and noise.

Do not launch more tuning until the gap is explained.

## Phase 6 — Closed competition research

Only if the competition is closed or the user authorizes public research:

- search Kaggle writeups, notebooks and GitHub;
- inspect code and metadata in read-only mode;
- classify each idea as clean, code-assisted, public-artifact-assisted,
  leaderboard-probed or external-submission arbitration;
- do not copy external predictions, labels, anchors or ID overrides;
- reproduce only clean methodological ideas;
- keep public probing in a separate non-benchmark track.

## Phase 7 — Final freeze

Freeze when:

- a candidate passes the metric-correct validation gate;
- no new structural idea remains with a credible expected gain;
- private score is stable or the competition has ended;
- all rejected experiments are documented;
- rollback submission and hashes are preserved.

Freeze report must contain:

```text
final protected score
best submission ID
metric and validation protocol
best local metrics
public/private comparison
rejected experiments
cost/runtime
known limitations
```

## Phase 8 — GitHub publication

After freeze only:

- copy clean code and reports to the portfolio;
- exclude raw data, credentials, caches and private probability archives;
- add provenance notice;
- separate autonomous, assisted and probed results;
- run syntax/structure/secret checks;
- commit and push;
- verify the remote tree and commit hash.

## S6E1–S6E6 execution order

For each episode, run the complete phases in order. Never optimize all six in
parallel before each one has a verified baseline and metric contract.

Recommended status registry:

```text
episode | status | metric | baseline | best_private | phase | next_gate
```
