# Sprint 5 — Omralinov-inspired S6E8

## Verdict

**NO-GO — no public submission was made.** The run was executed on a RunPod RTX 3090 with the full S6E8 train/test data. CatBoost and LightGBM completed; the neural-network path required several fixes before completing and remained substantially weaker than the autonomous benchmark.

## Verified local and remote results

| Component | OOF AUC | Status |
|---|---:|---|
| CatBoost | 0.9556661855 | Completed on GPU; checkpoint saved |
| LightGBM | 0.9442184798 | Completed; checkpoint saved |
| Neural network | 0.8872528428 | Completed after fixes; best epochs selected per fold |
| Fixed rank blend (0.49 / 0.31 / 0.21) | 0.9379862718 | Reproducible locally from saved checkpoints |

The current autonomous benchmark remains **0.96970 public** (Sprint 4). Sprint 5's blend is 0.031714 below that benchmark in OOF AUC and was not submitted.

## Operational evidence

- Full data: 691,369 train rows and 296,302 test rows.
- GPU execution: NVIDIA RTX 3090, CUDA-enabled PyTorch.
- Checkpoints were written after CatBoost and LightGBM, so the later NN failures did not erase earlier work.
- The final remote output included `oof_nn.npy`, `test_nn.npy`, `nn_metrics.json`, `oof_blend.npy`, `test_blend.npy`, `blend_metrics.json` and `submission_sprint5.csv` on the persistent volume. No Kaggle submission was made.

## Bugs found and fixed

1. CatBoost categorical NaNs were not converted to strings.
2. The Kaggle script initially omitted `train_lightgbm`.
3. LightGBM received duplicate `slack` features.
4. The NN read original object-valued categoricals instead of mapped indices.
5. The NN numerical-token projection had incompatible dimensions and an incomplete mask.
6. Unknown categorical values lacked a reserved embedding slot.
7. Missing and badly scaled numerical values produced NaN logits; tensor inputs were made finite and bounded.

The regression tests in `tests/test_sprint5_pipeline.py` cover the first two data-boundary failures. The final model result is still a NO-GO: making the NN executable did not make it competitive.

## Reproducibility boundary

The RunPod output files are not committed as portfolio artifacts because they lived on the remote persistent volume and the candidate did not pass the promotion gate. The committed code and tests preserve the audit trail; no estimated score is recorded as a submission result.
