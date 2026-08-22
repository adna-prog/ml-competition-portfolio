# Audit — Competition-Grade Student Health GPT

## Source

- Notebook: `raedennab/competition-grade-student-health-gpt`
- URL: https://www.kaggle.com/code/raedennab/competition-grade-student-health-gpt
- Pulled current notebook metadata: public, GPU T4, competition source only.
- No external dataset source, model source or internet dependency declared.

## Validated components

The notebook correctly optimizes `balanced_accuracy_score` and contains a strong
clean pipeline:

### Feature engineering

- missing count;
- categorical pair interactions;
- sleep distance/shortfall/excess features;
- BMI distance and tail features;
- heart-rate distance;
- log steps;
- activity ratios and mixtures;
- sleep × exercise and sleep × log steps.

The transformations are unsupervised and applied to train/test consistently.

### Tree ensemble

- LightGBM all-feature model;
- LightGBM signal-feature model;
- XGBoost;
- CatBoost;
- cross-validation and OOF probabilities;
- balanced class/sample weighting;
- prior-correction parameter `alpha`;
- geometric/log-probability blending;
- held-out OOF decision validation before accepting blend calibration.

### Neural extension

The second part adds a GPU MLP with:

- discretized numeric features;
- cross-fitted multiclass target encoding;
- categorical embeddings;
- robust scaling;
- periodic numeric bases;
- class weighting;
- label smoothing;
- EMA checkpoints;
- meta-blending with tree probabilities.

## Admissibility classification

### Clean and reproducible

- balanced accuracy;
- unsupervised interactions;
- fold-safe target encoding;
- balanced weights;
- seed blending;
- held-out OOF calibration;
- tree-family diversity.

### Must be isolated before use

- neural MLP: valid methodology, but expensive and requires GPU;
- `alpha` prior correction: must be validated against a held-out OOF decision
  split, not selected on full OOF alone;
- any public-score-tested anchor or hedge: not part of autonomous selection.

### Not found in the inspected notebook

- no external labels;
- no anchor CSV;
- no direct private leaderboard fitting;
- no test pseudo-labeling in the visible V1 tree pipeline.

## Main gap versus our current pipeline

Our corrected core already has:

- balanced accuracy;
- fold-safe target encoding;
- stress × sleep-bin interaction;
- CatBoost + LightGBM;
- OOF decision calibration.

The strongest untested clean components from this notebook are:

1. LightGBM signal-feature diversity model;
2. XGBoost model in the same target-encoded matrix;
3. held-out OOF weight calibration over a broader tree ensemble;
4. selected unsupervised interaction features;
5. only later, the GPU MLP.

The next reproduction should start with items 1–4. It should not jump directly to
the neural extension.
