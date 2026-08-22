# S6E7 lessons transferred to S6E8

S6E8 keeps its official metric (**ROC AUC**) and existing frozen validation
protocol. The following guardrails are now mandatory for future S6E8 sprints.

## 1. Metric contract

Every sprint must declare:

```text
official_metric: ROC AUC
optimization_metric: ROC AUC
secondary_metrics: PR AUC, calibration, subgroup AUC
public_score: informational only
private_score: final criterion
```

Accuracy, F1, or decision thresholds must never replace ROC AUC for model
selection.

## 2. Separate calibration from decision validation

When tuning blend weights, calibration, or meta-models:

- use one OOF partition for fitting weights/calibration;
- use a separate untouched OOF partition for accepting/rejecting the decision;
- preserve the prior frozen blend as rollback.

A tiny full-OOF gain is not sufficient evidence.

## 3. Mandatory probability archives

Every expensive model family must save and verify:

```text
oof_predictions.npz
test_predictions.npz
fold_metadata.json
blend_selection.json
submission.csv
```

Metadata must include seeds, fold IDs, feature recipe, code version, shapes,
ID order and file hashes. Remote jobs must write these artifacts to persistent
storage before shutdown.

## 4. Generalization gates

A candidate should pass, in order:

```text
OOF gain
→ frozen holdout/leave-one-fold-out gain
→ public score (secondary evidence)
→ private score (final evidence)
```

Reject candidates that rely on one seed, one fold, a micro-gain below the noise
floor, or public leaderboard behavior alone.

## 5. Provenance taxonomy

Every artifact must be labelled:

```text
clean-from-scratch
code-assisted
public-artifact-assisted
leaderboard-probed
external-submission-arbitration
```

Only the first category counts toward the autonomous S6E8 benchmark. Public OOF,
external submissions, test-ID overrides and manual flips must never enter the
clean track.

## 6. Remote compute discipline

For RunPod/Kaggle GPU jobs:

- predeclare the experiment and stop condition;
- use persistent storage;
- checkpoint models and probabilities;
- verify output hashes and sizes;
- stop the pod immediately after artifact recovery;
- record cost and runtime.

## 7. AUC-specific caution

S6E7's balanced-accuracy calibration does not transfer directly to S6E8.
S6E8 should preserve probability ranking and evaluate ROC AUC. Class weights,
prior correction and class thresholds require separate AUC validation and must
not be imported automatically.
