# Audit — navazshfathi/predicting-student-health-risk

## Verdict

This notebook is not a clean standalone model. It is a public-submission
arbitration pipeline.

## Verified behavior

The notebook:

1. Searches for a pool of at least five submission CSVs.
2. Loads external predictions and extracts their claimed scores from filenames.
3. Sorts/deduplicates the external prediction vectors by those scores.
4. Builds committee votes for `k = 5, 8, 10` submissions.
5. Trains a CatBoost arbiter on the competition training data.
6. Finds test IDs where committee consensus disagrees with the best external
   submission and where the arbiter agrees with the committee.
7. Applies up to `MAX_FLIPS = 12` manual test-row changes.

The code explicitly searches paths such as:

```text
/kaggle/input/...submissions
external/...
```

and treats filenames containing values such as `0.95070` as score metadata.

## Classification

```text
From-scratch clean model       : NO
External submission pool       : YES
Public score used as signal    : YES
ID-specific test corrections   : YES
Leaderboard probing/arbitrage : YES
```

## Action

Do not reproduce the submission pool, score extraction, committee overrides or
ID flips. The result may be useful for studying public-leaderboard probing, but
it provides no clean generalization evidence.

Our RunPod MLP meta-blend remains the best clean result:

```text
Private score: 0.95029
```
