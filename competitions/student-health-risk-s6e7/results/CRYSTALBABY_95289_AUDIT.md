# Audit — crystalbaby/lb-0-95289

## Verdict

The reported `LB 0.95289` is **not a clean model improvement**. It is an explicit
public-leaderboard probing experiment.

## Verified implementation

The notebook contains:

1. A large hard-coded `OVERRIDES` dictionary mapping individual test IDs to
   manually selected labels.
2. `SOURCE_CANDIDATES` pointing to external/public submission files, including
   files named around scores `0.95075`, `0.95086`, and other external ensemble
   outputs.
3. Majority voting across those external submissions.
4. Manual override application after the vote.
5. A markdown section explicitly describing the `Probed Track` as optimizing
   against the fixed public leaderboard subset.

The notebook itself states that it studies the difference between an honest
OOF-driven track and a public-leaderboard-probed track.

## Classification

```text
Clean from-scratch model       : NO
Public submission blending     : YES
Hard-coded test-ID overrides   : YES
Public leaderboard probing     : YES
External prediction artifacts  : YES
```

## Action

- Do not copy the override dictionary.
- Do not copy the external submissions.
- Do not count `0.95289` as a reproducible model score.
- Keep our RunPod MLP result `0.95029` as the best clean/private result.

The notebook is useful only as evidence that tiny private/public leaderboard
differences can be manipulated with test-ID overrides and external public
artifacts. It does not identify a new generalizable feature or model.
