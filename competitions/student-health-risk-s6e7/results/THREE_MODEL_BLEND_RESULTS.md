# Three-model clean blend result — S6E7

## Models tested

- CatBoost on all target-encoded features;
- LightGBM on signal-focused features;
- XGBoost on all target-encoded features;
- five frozen stratified folds;
- balanced accuracy throughout.

The first run had a selection bug: the equal-weight baseline was absent from the
quarter-weight grid and the prior correction was compared to the wrong baseline.
The model outputs were preserved and reanalyzed correctly.

## Correct reanalysis

Best raw grid blend:

```text
CatBoost       : 70%
LightGBM signal: 10%
XGBoost        : 20%
```

Results:

```text
Raw blend       : 0.949724
With class bias : 0.949814
Protected core  : 0.949779
```

The class-bias variant improves the aggregate OOF score but regresses on two of
five folds, so it fails the stability gate. The raw three-model blend is also
below the protected core after accounting for the decision calibration.

## Decision

- No submission.
- No replacement of private score `0.94969`.
- XGBoost adds diversity but no validated gain in this configuration.
- The GPU MLP remains a separate optional experiment and should not use paid
  RunPod resources without explicit approval.
