from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, f1_score

ROOT = Path(__file__).parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN = pd.read_csv(RAW / "Train.csv")
TEST = pd.read_csv(RAW / "Test.csv")
TOURN = pd.read_csv(RAW / "tournaments.csv")
WINNERS = dict(zip(TOURN.tournament_id, TOURN.winner))

STAGE_MAP = {
    "group stage": "group",
    "round of 16": "roundof16",
    "quarter-finals": "qf",
    "semi-finals": "sf",
    "third-place match": "sf",
}
VALID_YEARS = [2010, 2014, 2018, 2022]
STAGE_ORDER = ["group", "roundof32", "roundof16", "qf", "sf", "runnerup", "champion"]


def normalize_stage(row):
    raw = row["stage_reached"]
    if raw == "final":
        return "champion" if WINNERS.get(row["tournament_id"]) == row["country"] else "runnerup"
    return STAGE_MAP.get(raw, None)


def history_features(history, teams):
    """Causal team-history features: history contains only years before prediction."""
    global_goals = float(history.total_goals.mean()) if len(history) else 3.0
    global_matches = float(history.matches_played.mean()) if len(history) else 3.0
    rows = []
    for country in teams:
        h = history[history.country == country].sort_values("year")
        goals = h.total_goals.to_numpy(float)
        matches = h.matches_played.to_numpy(float)
        if len(h):
            recent = goals[-3:]
            goal_mean = float(goals.mean())
            recent_mean = float(recent.mean())
            last_goals = float(goals[-1])
            last_matches = float(matches[-1])
            prior_stage = normalize_stage(h.iloc[-1])
            prior_stage_rank = STAGE_ORDER.index(prior_stage) if prior_stage in STAGE_ORDER else 0
            stage_history = [normalize_stage(r) for _, r in h.iterrows()]
            stage_history = [s for s in stage_history if s in STAGE_ORDER]
            mode_stage = pd.Series(stage_history).mode().iloc[0] if stage_history else "group"
            mode_stage_rank = STAGE_ORDER.index(mode_stage)
            best_stage = [STAGE_ORDER.index(s) for s in stage_history]
            best_stage_rank = max(best_stage) if best_stage else 0
            last_year = int(h.year.iloc[-1])
        else:
            goal_mean = recent_mean = last_goals = global_goals
            last_matches = global_matches
            prior_stage_rank = best_stage_rank = 0
            last_year = 0
        rows.append({
            "country": country,
            "history_count": len(h),
            "goal_mean": goal_mean,
            "recent_goal_mean": recent_mean,
            "last_goals": last_goals,
            "last_matches": last_matches,
            "prior_stage_rank": prior_stage_rank,
            "mode_stage_rank": mode_stage_rank,
            "best_stage_rank": best_stage_rank,
            "years_since_last": 999 if not last_year else 2026 - last_year,
        })
    return pd.DataFrame(rows)


def predict_from_history(history, teams, stage_strategy="last"):
    f = history_features(history, teams)
    # Conservative shrinkage toward recent performance and historical global mean.
    global_goals = float(history.total_goals.mean()) if len(history) else 3.0
    f["pred_goals"] = 0.55 * f.recent_goal_mean + 0.30 * f.goal_mean + 0.15 * global_goals
    rank_col = {"last": "prior_stage_rank", "mode": "mode_stage_rank", "best": "best_stage_rank"}.get(stage_strategy)
    if rank_col is None:
        f["pred_stage"] = "group"
    else:
        f["pred_stage"] = [STAGE_ORDER[int(x)] if int(x) < len(STAGE_ORDER) else "group" for x in f[rank_col]]
    return f


def evaluate_year(year, stage_strategy):
    history = TRAIN[TRAIN.year < year].copy()
    target = TRAIN[TRAIN.year == year].copy()
    target["true_stage"] = target.apply(normalize_stage, axis=1)
    pred = predict_from_history(history, target.country.tolist(), stage_strategy)
    pred = pred.rename(columns={"country": "country_pred"})
    merged = target.reset_index(drop=True).join(pred.reset_index(drop=True))
    rmse = float(mean_squared_error(merged.total_goals, merged.pred_goals) ** 0.5)
    valid = merged.true_stage.notna()
    weighted = float(f1_score(merged.loc[valid, "true_stage"], merged.loc[valid, "pred_stage"], average="weighted", zero_division=0))
    macro = float(f1_score(merged.loc[valid, "true_stage"], merged.loc[valid, "pred_stage"], average="macro", zero_division=0))
    return {"year": year, "rows": len(merged), "rmse_goals": rmse, "f1_weighted_stage": weighted, "f1_macro_stage": macro}


if __name__ == "__main__":
    results = {}
    for strategy in ["group", "last", "mode", "best"]:
        vals = [evaluate_year(y, strategy) for y in VALID_YEARS]
        results[strategy] = {"validation": vals, "summary": {
            k: float(np.mean([r[k] for r in vals]))
            for k in ["rmse_goals", "f1_weighted_stage", "f1_macro_stage"]
        }}
    final = predict_from_history(TRAIN, TEST.country.tolist(), "last")
    submission = pd.DataFrame({
        "ID": TEST.ID,
        "total_goals": np.maximum(0, final.pred_goals.round(3)),
        "Target": final.pred_stage,
    })
    submission.to_csv(OUT / "practice_baseline_submission.csv", index=False)
    (OUT / "walk_forward_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(submission.head(10).to_string(index=False))
