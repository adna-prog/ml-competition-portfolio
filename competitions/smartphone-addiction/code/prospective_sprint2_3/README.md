# S6E8 Prospective Sprints 2–3 — blend TabM puis CatBoost

Reproducible blend specs for the Sprint 2 (TabM) and Sprint 3 (CatBoost) autonomous benchmarks.
The small configs below record the frozen weights and validation gates without publishing OOF or
test predictions.

## Third-party code attribution

- **TabM**: official implementation, [yandex-research/tabm](https://github.com/yandex-research/tabm),
  Apache License 2.0. All weights and predictions used here were trained by this project.
- **CatBoost**: official package, [catboost/catboost](https://github.com/catboost/catboost),
  Apache License 2.0. All fitted trees and predictions used here were produced by this project.
- RealMLP provenance for the Sprint 1 member is documented separately in
  `../prospective_sprint1/THIRD_PARTY_NOTICE.md`.

## Validation protocol (shared with Sprint 1)

- Development frozen: 553 095 rows, 5 folds of 110 619 (80 %); **historical holdout opened once**
  (138 274 rows, code `fold=-2`) — never used for selection of representation, model or blend weight.
- All decisions on development OOF only; holdout never read or scored for Sprint 2/3 selection.
- Minimum useful gain predeclared: `+0.00015` AUC OOF.
- A candidate joins the blend only if global gain ≥ minimum, positive on 5/5 folds at fixed weights,
  and selected by leave-one-fold-out on 5/5 partitions.
- Qualification: **from-scratch-code-assisted** — no public predictions, OOF or trained weights imported.
  TabM / CatBoost architectures or packages used with attribution; all weights and predictions retrained.

## Sprint 2 — TabM enters the blend (public 0.96923)

- TabM exact-value TE (inner cross-fit) + fit-only frequencies + fold-local quantile transform:
  development OOF **0.9671391** (folds 0.967425 / 0.967518 / 0.967086 / 0.967222 / 0.967440).
- Baseline Sprint 1 blend (30 % XGB / 70 % RealMLP): OOF 0.9681391.
- Rank-average blend XGB 21.75 % / RealMLP 50.75 % / TabM 27.50 %: OOF **0.9682957**,
  gain **+0.0001566**, positive on 5/5 folds, leave-one-fold-out positive on 5/5.
- Public: **0.96923**. Config frozen in `frozen_tabm_te_blend_spec.json`.

## Sprint 3 — CatBoost composition enters (public 0.96952, submission 55504165)

- CatBoost exact-category + 12 fit-only compositions (no target leakage): development OOF **0.9678329**
  (folds 0.967882 / 0.968236 / 0.967610 / 0.967588 / 0.967857), 6 000 trees max, early stopping,
  runtime ≈ 30.6 min on free Kaggle GPU, cost `0 $`.
- Rank-average blend XGB **13.59375 %** / RealMLP **31.71875 %** / TabM **17.1875 %** / CatBoost **37.5 %**:
  OOF **0.9685551**, gain **+0.0002594** vs Sprint 2, positive on 5/5 folds, leave-one-fold-out
  selects 37.5 % on 5/5 (mean held gain +0.0002495).
- Public: **0.96952** (gain +0.00029), very close to the predicted OOF gain.
- Config frozen in `frozen_catboost_blend_spec.json`.

## Explicit NO-GOs documented this round

- **LightGBM exact-TE** (OOF 0.9664086): rank correlation with XGB ≈ 0.9934, marginal blend gain
  ≈ +0.000026 — below the useful gate. Rejected despite a positive micro-gain.
- **Decimal lattice** (both the wide 45-var variant and the faithful published reproduction): fold-0
  deltas −0.000070 and −0.000267. Rejected: a feature's external reputation does not replace validation
  on our folds.
- **xRFM**: full five-fold run NO-GO on the free P100 (`sm_60` lacks tensor cores; quadratic cost too
  risky). Micro-screening only, explicitly not comparable to a full-fold AUC.
