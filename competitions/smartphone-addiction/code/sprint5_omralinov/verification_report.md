# Sprint 5 Code Verification — verified run record

## Scope

The candidate was audited and executed against the real S6E8 data (691,369 train rows, 296,302 test rows). The remote run used an NVIDIA RTX 3090 with CUDA-enabled PyTorch and saved intermediate checkpoints on a persistent volume.

## Findings

1. The initial pipeline failed on CatBoost categorical NaNs.
2. The published Kaggle script omitted `train_lightgbm`, so its first full run stopped after CatBoost.
3. The corrected remote pipeline completed CatBoost and LightGBM and saved their OOF/test arrays.
4. The NN initially failed on object-valued categoricals, then on incompatible numerical-token dimensions, then on NaN logits. These issues were fixed and covered by local smoke execution before the final NN run.
5. The final NN run completed all 11 folds and produced a fixed blend.

## Final verified metrics

- CatBoost OOF AUC: `0.9556661855`
- LightGBM OOF AUC: `0.9442184798`
- Neural network OOF AUC: `0.8872528428`
- Fixed rank blend OOF AUC: `0.9379862718`
- Autonomous S6E8 benchmark: `0.96970` public

## Decision

**NO-GO.** The candidate was not submitted to Kaggle and must not be described as an improvement. The code remains useful as an audited experiment and as a checkpointed-pipeline example, but the S6E8 cycle is closed.
