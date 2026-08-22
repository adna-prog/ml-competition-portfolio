# Shift correction experiments — S6E7

## Protocol

A LightGBM adversarial classifier was fit on unlabeled train/test rows with `id`
excluded. Importance weights used the covariate-shift density ratio:

```text
p(test | x) / p(train | x)
```

Weights were clipped to `[0.25, 4.0]` and normalized. Three target models were
compared on three frozen stratified folds.

## Results

| Variant | Standard accuracy | Importance-weighted accuracy | Macro-F1 |
|---|---:|---:|---:|
| Standard | 0.967109 | 0.967090 | 0.910742 |
| Adversarial weighted | 0.967082 | 0.967066 | 0.910657 |
| Drop top shifted features | 0.947349 | 0.947323 | 0.843539 |

Adversarial weighting is slightly worse and the drop-shifted variant is clearly
unusable. No submission was made.

## Interpretation

The multivariate shift is real, but a generic density-ratio correction does not
recover the private score. The shift classifier's in-sample AUC was `0.666764`,
with most weights near 1 and a small extreme tail (`max=15.87`), so clipping or
weight variance is not the sole explanation.

The current evidence points to a conditional relationship problem: the mapping
from features to `health_condition` likely changes between the synthetic train
and test generators. Pure covariate-shift reweighting assumes that mapping is
stable, so it cannot fix this gap.

Current protected benchmark remains LightGBM private `0.874570`.
