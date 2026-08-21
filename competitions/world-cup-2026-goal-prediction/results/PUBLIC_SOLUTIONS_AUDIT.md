# Audit des solutions publiques — World Cup 2026 Goal Prediction

## Sources inspected

### 1. `sashishjha/wc2026_prediction`

Revision inspected: `5eebf68`, 2026-06-22.

Verified method claims/code:

- Elo World-Cup-only with group/knockout K factors, goal-difference multiplier and inter-tournament decay;
- PageRank-like historical match graph;
- direct and rate/opportunity goals models;
- Dixon-Coles scoreline model;
- Monte Carlo simulation of 10,000+ complete tournaments;
- Hungarian constrained stage assignment;
- exact 48-team quotas.

Important compliance findings:

- `config.py` contains `LIVE_RESULTS_RAW` described as matches already played in June 2026;
- the pipeline accepts `--external-matches`;
- these are not allowed under the official closed-data rule.

Transferable methodology: Dixon-Coles, causal Elo, tournament simulation,
probability-matrix assignment, and quota reconciliation — only when trained on
the supplied historical database and without live/current results.

### 2. `emmanuel-123tech/2026-Fifa-World-cup-prediction`

Revision inspected: `e7cf1e0`, 2026-07-25. README claims fourth place.

Verified method claims/code:

- historical cumulative and rolling features;
- match-level aggregation;
- Random Forest, Extra Trees, Gradient Boosting and Ridge ensembles;
- constrained 48-team assignment;
- a manually curated `rank_order` and `attack_bonus` current-strength layer.

The current-strength layer is likely a major source of leaderboard lift, but it is
not learned from the supplied dataset. The notebook explicitly says these values
are judgement-based current football assumptions. Under the closed-data rule,
that layer is not admissible for our compliant benchmark.

Also, its validation reports model-component metrics but not the full post-processing
pipeline, and its README formula uses `0.60 * RMSE + 0.40 * (1-F1)`, which should
not be trusted as the official normalized score without reproducing Zindi's metric.

### 3. `Mustafa-elsherif/worldcup2026-prediction`

Revision inspected: `abfaac8`, 2026-06-20.

Verified method claims:

- Random Forest, XGBoost and LightGBM comparison;
- 17 historical/current-strength features;
- reported RMSE around 3.95 and F1 around 0.41;
- constrained-looking predictions and dashboard.

The README also describes a manually constructed current-strength layer. It is a
useful baseline reference but does not explain the large leaderboard gap in a
fully reproducible, data-only way.

## What we were missing

1. **Full tournament simulation** rather than a deterministic ranking by predicted goals.
2. **Scoreline model** (Dixon-Coles) rather than only team-level total-goals regression.
3. **Stage probability matrix + Hungarian assignment**, not only one scalar ranking score.
4. **Explicit match-opportunity modeling** through simulated group and knockout matches.
5. A clear distinction between compliant historical-only features and inadmissible
   current-strength/live-results inputs.

## Strategic conclusion

The biggest transferable gap is structural: our practice champion uses quotas, but
not a full probabilistic tournament simulation. The next compliant experiment, if
we continue, should implement Dixon-Coles + Monte Carlo using only the supplied
historical match data, then aggregate expected goals and stage probabilities and
apply Hungarian assignment.

Do not copy:

- `LIVE_RESULTS_RAW`;
- current 2026 match results;
- manual current rankings/attack bonuses;
- external rankings, odds or post-competition data.

The public repositories are therefore valid methodological inspiration, not
admissible artifacts for the historical-only benchmark.
