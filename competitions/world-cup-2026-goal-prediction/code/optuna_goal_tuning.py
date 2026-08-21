from pathlib import Path
import json
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, f1_score
from joint_prediction import TRAIN, TEST, RAW, OUT, STAGES, stage, prep, assign
from enhanced_joint_prediction import enhanced_history_features

SEED = 20260845
OUT = OUT / "optuna"
OUT.mkdir(parents=True, exist_ok=True)
VALID_YEARS = [2010, 2014, 2018, 2022]


def causal_features(df):
    parts = []
    for year in sorted(df.year.unique()):
        hist = df[df.year < year].copy()
        cur = df[df.year == year].copy()
        parts.append(enhanced_history_features(hist, cur))
    return pd.concat(parts, ignore_index=True)


def prepare_folds():
    folds = []
    for year in VALID_YEARS:
        hist = TRAIN[TRAIN.year < year].copy()
        target = TRAIN[TRAIN.year == year].copy()
        train_feat = causal_features(hist)
        val_feat = enhanced_history_features(hist, target)
        X_train, cats = prep(train_feat)
        X_val, _ = prep(val_feat)
        folds.append((X_train, X_val, cats, hist.total_goals.to_numpy(), target))
    return folds

FOLDS = prepare_folds()


def evaluate(params):
    rmses, f1s = [], []
    for X_train, X_val, cats, y_train, target in FOLDS:
        model = CatBoostRegressor(
            iterations=params["iterations"], depth=params["depth"],
            learning_rate=params["learning_rate"], l2_leaf_reg=params["l2_leaf_reg"],
            random_strength=params["random_strength"],
            bagging_temperature=params["bagging_temperature"],
            loss_function="RMSE", random_seed=SEED,
            verbose=False, allow_writing_files=False, thread_count=4,
        )
        model.fit(X_train, y_train, cat_features=cats, verbose=False)
        goals = np.maximum(0, model.predict(X_val))
        stage_pred = assign(goals, len(target))
        true_stage = target.apply(stage, axis=1)
        valid = true_stage.notna().to_numpy()
        rmses.append(float(mean_squared_error(target.total_goals, goals) ** 0.5))
        f1s.append(float(f1_score(true_stage[valid], stage_pred[valid], average="weighted", zero_division=0)))
    return float(np.mean(rmses)), float(np.mean(f1s))


def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500, step=50),
        "depth": trial.suggest_int("depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
    }
    rmse, f1 = evaluate(params)
    trial.set_user_attr("rmse", rmse)
    trial.set_user_attr("f1_weighted_stage", f1)
    return rmse, f1


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="world_cup_goal_joint_multiobjective",
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    trials = []
    for t in study.trials:
        if t.values is None:
            continue
        trials.append({"number": t.number, "state": t.state.name, "params": t.params, "rmse": t.values[0], "f1_weighted_stage": t.values[1]})
    result = {"seed": SEED, "n_trials": len(trials), "trials": trials, "pareto_trials": [t.number for t in study.best_trials]}
    (OUT / "study_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
