# S6E7 — Frozen status

## Final protected result

```text
Competition: Predicting Student Health Risk (Kaggle S6E7)
Metric: balanced accuracy
Best clean private score: 0.95029
Best clean submission: 55690196
Final calibration candidate: 55691824
```

The MLP/tree meta-blend improved the corrected tree core from `0.94969` to
`0.95029` on the private leaderboard.

## Frozen decisions

- No further public-leaderboard probing.
- No test-ID overrides or manual flips.
- No external submission voting.
- No more RunPod runs for this competition unless a materially new hypothesis
  is explicitly approved.
- The clean MLP/tree meta-blend remains the final portfolio benchmark.

## Lessons transferred to future competitions

- Verify the official metric before any modeling.
- Keep the metric contract in code and reports.
- Separate OOF calibration from OOF decision validation.
- Archive OOF/test probabilities before stopping remote compute.
- Treat public/private leaderboard divergence as a diagnostic, not an objective.
- Classify public artifacts separately from clean from-scratch results.
