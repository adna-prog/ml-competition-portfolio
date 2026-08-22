
## Post-correction improvement tests

### High-confidence pseudo-labeling

A fold-safe test used test predictions above `0.99` confidence, with pseudo-label
weight `0.25`:

```text
Base CV       : 0.949548867
Augmented CV  : 0.949411130
Difference    : -0.000137737
```

Approximately 28,000 test rows passed the confidence threshold per fold, but the
augmented model regressed on every fold. No submission was made.

### Public CatBoost parameter reproduction

The public bake-off settings (`iterations=1500`, `learning_rate=0.05`, `depth=6`,
balanced class weights) were reproduced using our fold-safe target encoding:

```text
Our corrected CatBoost : 0.949327219
Public-parameter run   : 0.949329980
Gain                   : +0.000002761
```

This is noise-level and does not justify a new submission.

## Protected final result

The corrected blend with OOF decision calibration remains the protected result:

```text
Kaggle private : 0.94969
Submission     : 55678486
```

The pseudo-label and public-parameter variants are rejected. Further work would
require a genuinely new signal or a carefully isolated public-method component,
not more repetitions of the same CatBoost/target-encoding search.
