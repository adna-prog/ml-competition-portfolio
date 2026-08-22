
from __future__ import annotations

import gc
import json
import os
import random
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

display = print

from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

warnings.filterwarnings('ignore')

SEED = 202607
FAST_MODE = False
FORCE_CPU = False
N_SPLITS = 2 if FAST_MODE else 5
FAST_TRAIN_ROWS = 120_000
CALIBRATION_FRACTION = 0.50

DATA_DIR = Path('/workspace/s6e7/data')
OUTPUT_DIR = Path('/workspace/s6e7/output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'health_condition'
ID_COL = 'id'

random.seed(SEED)
np.random.seed(SEED)

try:
    import torch
    GPU_AVAILABLE = bool(torch.cuda.is_available())
except Exception:
    GPU_AVAILABLE = False

USE_GPU = GPU_AVAILABLE and not FORCE_CPU

print({
    'FAST_MODE': FAST_MODE,
    'N_SPLITS': N_SPLITS,
    'GPU_AVAILABLE': GPU_AVAILABLE,
    'USE_GPU': USE_GPU,
    'DATA_DIR': str(DATA_DIR),
})



train_path = DATA_DIR / 'train.csv'
test_path = DATA_DIR / 'test.csv'
sample_path = DATA_DIR / 'sample_submission.csv'

for path in (train_path, test_path, sample_path):
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {path}. Attach the Kaggle competition dataset to the notebook.'
        )

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

assert {ID_COL, TARGET}.issubset(train.columns)
assert ID_COL in test.columns
assert TARGET not in test.columns
assert len(test) == len(sample_submission)
assert sample_submission[ID_COL].equals(test[ID_COL])

print('Train:', train.shape)
print('Test :', test.shape)
display(train.head())
display(train[TARGET].value_counts().to_frame('count').assign(
    proportion=lambda x: x['count'] / len(train)
))



def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    original_features = [c for c in out.columns if c not in (ID_COL, TARGET)]
    out['missing_count'] = out[original_features].isna().sum(axis=1).astype('int8')

    categorical_pairs = [
        ('stress_level', 'physical_activity_level'),
        ('stress_level', 'sleep_quality'),
        ('stress_level', 'smoking_alcohol'),
        ('physical_activity_level', 'sleep_quality'),
        ('diet_type', 'physical_activity_level'),
    ]
    for left, right in categorical_pairs:
        if left in out.columns and right in out.columns:
            out[f'{left}__{right}'] = (
                out[left].fillna('__MISSING__').astype(str)
                + '|'
                + out[right].fillna('__MISSING__').astype(str)
            )

    if 'sleep_duration' in out:
        out['sleep_distance_8h'] = (out['sleep_duration'] - 8.0).abs()
        out['sleep_shortfall_7h'] = (7.0 - out['sleep_duration']).clip(lower=0)
        out['sleep_excess_9h'] = (out['sleep_duration'] - 9.0).clip(lower=0)

    if 'bmi' in out:
        out['bmi_distance_22'] = (out['bmi'] - 22.0).abs()
        out['bmi_under_18_5'] = (18.5 - out['bmi']).clip(lower=0)
        out['bmi_over_25'] = (out['bmi'] - 25.0).clip(lower=0)

    if 'heart_rate' in out:
        out['heart_rate_distance_70'] = (out['heart_rate'] - 70.0).abs()

    if 'step_count' in out:
        out['log_step_count'] = np.log1p(out['step_count'].clip(lower=0))

    if {'step_count', 'exercise_duration'}.issubset(out.columns):
        out['activity_mix'] = out['step_count'] / 1000.0 + out['exercise_duration'] / 10.0
        out['exercise_per_1k_steps'] = out['exercise_duration'] / (1.0 + out['step_count'] / 1000.0)

    if {'sleep_duration', 'exercise_duration'}.issubset(out.columns):
        out['sleep_x_exercise'] = out['sleep_duration'] * out['exercise_duration']

    if {'sleep_duration', 'step_count'}.issubset(out.columns):
        out['sleep_x_log_steps'] = out['sleep_duration'] * np.log1p(out['step_count'].clip(lower=0))

    return out

train_fe = add_features(train)
test_fe = add_features(test)

if FAST_MODE:
    train_fe, _ = train_test_split(
        train_fe,
        train_size=min(FAST_TRAIN_ROWS, len(train_fe)),
        stratify=train_fe[TARGET],
        random_state=SEED,
    )
    train_fe = train_fe.reset_index(drop=True)

print('Engineered train:', train_fe.shape)
print('Engineered test :', test_fe.shape)



label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_fe[TARGET])
class_names = label_encoder.classes_
n_classes = len(class_names)

X_train_raw = train_fe.drop(columns=[TARGET, ID_COL])
X_test_raw = test_fe.drop(columns=[ID_COL])
assert list(X_train_raw.columns) == list(X_test_raw.columns)

categorical_columns = X_train_raw.select_dtypes(include=['object', 'category']).columns.tolist()
X_all_cat = pd.concat([
    X_train_raw[categorical_columns].fillna('__MISSING__').astype(str),
    X_test_raw[categorical_columns].fillna('__MISSING__').astype(str),
], axis=0, ignore_index=True)

category_encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1,
    encoded_missing_value=-1,
    dtype=np.int32,
)
encoded_all = category_encoder.fit_transform(X_all_cat)

X_train = X_train_raw.copy()
X_test = X_test_raw.copy()
for j, col in enumerate(categorical_columns):
    X_train[col] = encoded_all[:len(X_train), j].astype(np.int32)
    X_test[col] = encoded_all[len(X_train):, j].astype(np.int32)

class_priors = np.bincount(y, minlength=n_classes).astype(float)
class_priors /= class_priors.sum()

print('Class mapping:', dict(enumerate(class_names)))
print('Class priors :', dict(zip(class_names, class_priors.round(6))))
print('Categorical columns:', categorical_columns)



import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

BASE_SIGNAL_COLUMNS = [
    'sleep_duration', 'heart_rate', 'bmi', 'step_count', 'exercise_duration',
    'diet_type', 'stress_level', 'physical_activity_level', 'smoking_alcohol',
]

SIGNAL_COLUMNS = [
    c for c in X_train.columns
    if (
        c in BASE_SIGNAL_COLUMNS
        or c.startswith('stress_level__')
        or c.startswith('physical_activity_level__')
        or c in {
            'missing_count', 'sleep_distance_8h', 'sleep_shortfall_7h',
            'sleep_excess_9h', 'bmi_distance_22', 'bmi_under_18_5',
            'bmi_over_25', 'heart_rate_distance_70', 'log_step_count',
            'activity_mix', 'exercise_per_1k_steps', 'sleep_x_exercise',
            'sleep_x_log_steps',
        }
    )
]
if len(SIGNAL_COLUMNS) < 8:
    SIGNAL_COLUMNS = list(X_train.columns)

MODEL_SPECS = {
    'lgb_all': {
        'family': 'lightgbm',
        'feature_columns': list(X_train.columns),
        'params': {
            'objective': 'multiclass', 'num_class': n_classes,
            'n_estimators': 160 if FAST_MODE else 2600,
            'learning_rate': 0.06 if FAST_MODE else 0.025,
            'num_leaves': 31, 'max_depth': -1, 'min_child_samples': 90,
            'subsample': 0.90, 'colsample_bytree': 0.95,
            'reg_alpha': 0.03, 'reg_lambda': 2.0,
            'random_state': SEED, 'n_jobs': 4, 'verbosity': -1,
        },
    },
    'lgb_signal': {
        'family': 'lightgbm',
        'feature_columns': SIGNAL_COLUMNS,
        'params': {
            'objective': 'multiclass', 'num_class': n_classes,
            'n_estimators': 180 if FAST_MODE else 3000,
            'learning_rate': 0.055 if FAST_MODE else 0.022,
            'num_leaves': 47, 'max_depth': -1, 'min_child_samples': 110,
            'subsample': 0.92, 'colsample_bytree': 0.90,
            'reg_alpha': 0.06, 'reg_lambda': 2.5,
            'random_state': SEED + 11, 'n_jobs': 4, 'verbosity': -1,
        },
    },
    'xgb_all': {
        'family': 'xgboost',
        'feature_columns': list(X_train.columns),
        'params': {
            'objective': 'multi:softprob', 'num_class': n_classes,
            'n_estimators': 160 if FAST_MODE else 2600,
            'learning_rate': 0.06 if FAST_MODE else 0.025,
            'max_depth': 7, 'min_child_weight': 8,
            'subsample': 0.90, 'colsample_bytree': 0.90,
            'gamma': 0.0, 'reg_alpha': 0.04, 'reg_lambda': 2.2,
            'tree_method': 'hist', 'device': 'cpu',
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 120 if FAST_MODE else 220,
            'random_state': SEED + 23, 'n_jobs': 4,
        },
    },
    'cat_all': {
        'family': 'catboost',
        'feature_columns': list(X_train.columns),
        'params': {
            'loss_function': 'MultiClass', 'eval_metric': 'MultiClass',
            'iterations': 180 if FAST_MODE else 2800,
            'learning_rate': 0.055 if FAST_MODE else 0.025,
            'depth': 8, 'l2_leaf_reg': 5.0, 'random_strength': 0.30,
            'bootstrap_type': 'Bayesian', 'bagging_temperature': 0.7,
            'random_seed': SEED + 37,
            'task_type': 'GPU' if USE_GPU else 'CPU', 'devices': '0',
            'thread_count': 4, 'allow_writing_files': False, 'verbose': False,
        },
    },
}

MODEL_SPECS = {k: v for k, v in MODEL_SPECS.items() if k in ('lgb_all', 'lgb_signal', 'cat_all')}

print('Models:', list(MODEL_SPECS))
print('All features:', len(X_train.columns))
print('Signal-focused features:', len(SIGNAL_COLUMNS))



def balanced_prior_predict(probabilities, priors, alpha=1.0, log_bias=None):
    adjusted = np.clip(probabilities, 1e-15, 1.0) / np.power(priors[None, :], alpha)
    if log_bias is not None:
        adjusted = adjusted * np.exp(log_bias[None, :])
    return adjusted.argmax(axis=1)


def fast_balanced_accuracy(y_true, y_pred, n_classes):
    encoded = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    matrix = np.bincount(encoded, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    denominators = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix), denominators,
        out=np.zeros(n_classes, dtype=float),
        where=denominators != 0,
    )
    return float(recalls.mean())


def score_probabilities(y_true, probabilities, priors, alpha=1.0, log_bias=None):
    pred = balanced_prior_predict(probabilities, priors, alpha, log_bias)
    return fast_balanced_accuracy(y_true, pred, len(priors))

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_predictions = {
    name: np.zeros((len(X_train), n_classes), dtype=np.float32)
    for name in MODEL_SPECS
}
test_predictions = {
    name: np.zeros((len(X_test), n_classes), dtype=np.float32)
    for name in MODEL_SPECS
}
fold_records = []
feature_importance_records = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y), start=1):
    print(f'\n========== Fold {fold}/{N_SPLITS} ==========')
    y_train_fold = y[train_idx]
    y_valid_fold = y[valid_idx]

    for model_name, spec in MODEL_SPECS.items():
        family = spec['family']
        feature_columns = spec['feature_columns']
        X_tr = X_train.iloc[train_idx][feature_columns]
        X_va = X_train.iloc[valid_idx][feature_columns]
        X_te = X_test[feature_columns]

        if family == 'lightgbm':
            model = lgb.LGBMClassifier(**spec['params'])
            model.fit(
                X_tr, y_train_fold,
                eval_set=[(X_va, y_valid_fold)],
                eval_metric='multi_logloss',
                categorical_feature=[c for c in categorical_columns if c in feature_columns],
                callbacks=[lgb.early_stopping(80 if FAST_MODE else 220, verbose=False)],
            )
            valid_proba = model.predict_proba(X_va, num_iteration=model.best_iteration_)
            test_proba = model.predict_proba(X_te, num_iteration=model.best_iteration_)
            feature_importance_records.append(pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_,
                'model': model_name,
                'fold': fold,
            }))
            best_iteration = int(model.best_iteration_ or spec['params']['n_estimators'])

        elif family == 'xgboost':
            model = XGBClassifier(**spec['params'])
            model.fit(X_tr, y_train_fold, eval_set=[(X_va, y_valid_fold)], verbose=False)
            valid_proba = model.predict_proba(X_va)
            test_proba = model.predict_proba(X_te)
            best_iteration = int(getattr(model, 'best_iteration', spec['params']['n_estimators']))

        elif family == 'catboost':
            cat_indices = [feature_columns.index(c) for c in categorical_columns if c in feature_columns]
            model = CatBoostClassifier(**spec['params'])
            model.fit(
                X_tr, y_train_fold,
                cat_features=cat_indices,
                eval_set=(X_va, y_valid_fold),
                early_stopping_rounds=100 if FAST_MODE else 220,
                verbose=False,
            )
            valid_proba = model.predict_proba(X_va)
            test_proba = model.predict_proba(X_te)
            best_iteration = int(model.get_best_iteration())
        else:
            raise ValueError(f'Unknown model family: {family}')

        oof_predictions[model_name][valid_idx] = valid_proba.astype(np.float32)
        test_predictions[model_name] += test_proba.astype(np.float32) / N_SPLITS

        record = {
            'fold': fold,
            'model': model_name,
            'best_iteration': best_iteration,
            'raw_balanced_accuracy': balanced_accuracy_score(y_valid_fold, valid_proba.argmax(axis=1)),
            'prior_corrected_balanced_accuracy': score_probabilities(
                y_valid_fold, valid_proba, class_priors, alpha=1.0
            ),
        }
        fold_records.append(record)
        print(record)

        del model, X_tr, X_va, X_te, valid_proba, test_proba
        gc.collect()

fold_results = pd.DataFrame(fold_records)
display(fold_results)
display(fold_results.groupby('model')[[
    'raw_balanced_accuracy', 'prior_corrected_balanced_accuracy'
]].agg(['mean', 'std']))



model_oof_summary = []
alpha_grid = np.linspace(0.75, 1.25, 101)

for model_name, probabilities in oof_predictions.items():
    raw_score = balanced_accuracy_score(y, probabilities.argmax(axis=1))
    alpha_scores = [
        score_probabilities(y, probabilities, class_priors, alpha=float(alpha))
        for alpha in alpha_grid
    ]
    best_index = int(np.argmax(alpha_scores))
    model_oof_summary.append({
        'model': model_name,
        'raw_oof_bacc': raw_score,
        'best_alpha': float(alpha_grid[best_index]),
        'best_corrected_oof_bacc': float(alpha_scores[best_index]),
    })

model_oof_summary = pd.DataFrame(model_oof_summary).sort_values(
    'best_corrected_oof_bacc', ascending=False
)
display(model_oof_summary)



model_names = list(oof_predictions)
oof_stack = np.stack([oof_predictions[name] for name in model_names], axis=0)
test_stack = np.stack([test_predictions[name] for name in model_names], axis=0)

calibration_idx, decision_valid_idx = train_test_split(
    np.arange(len(y)),
    test_size=1.0 - CALIBRATION_FRACTION,
    stratify=y,
    random_state=SEED + 101,
)

inverse_prior_weight = 1.0 / class_priors[y[calibration_idx]]
inverse_prior_weight /= inverse_prior_weight.mean()


def blend_from_weights(stack, weights):
    return np.tensordot(weights, stack, axes=(0, 0))


def weighted_blend_logloss(weights):
    if np.any(weights < 0):
        return 1e9
    weights = weights / weights.sum()
    blended = blend_from_weights(oof_stack[:, calibration_idx, :], weights)
    return log_loss(
        y[calibration_idx],
        np.clip(blended, 1e-12, 1.0),
        labels=np.arange(n_classes),
        sample_weight=inverse_prior_weight,
    )

initial_weights = np.full(len(model_names), 1.0 / len(model_names))
weight_result = minimize(
    weighted_blend_logloss,
    x0=initial_weights,
    method='SLSQP',
    bounds=[(0.0, 1.0)] * len(model_names),
    constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}],
    options={'maxiter': 250, 'ftol': 1e-10, 'disp': False},
)

optimized_weights = np.clip(weight_result.x, 0, None)
optimized_weights /= optimized_weights.sum()
print('Weight optimization success:', weight_result.success)
print('Optimized weights:', dict(zip(model_names, optimized_weights.round(4))))

oof_blend_optimized = blend_from_weights(oof_stack, optimized_weights)
test_blend_optimized = blend_from_weights(test_stack, optimized_weights)
oof_blend_equal = blend_from_weights(oof_stack, initial_weights)
test_blend_equal = blend_from_weights(test_stack, initial_weights)



def tune_alpha_and_bias(y_true, probabilities, priors, indices) -> Tuple[float, np.ndarray, float]:
    # Low-dimensional, reproducible coordinate tuning for balanced accuracy.
    probs = probabilities[indices]
    labels = y_true[indices]

    alpha_candidates = np.linspace(0.75, 1.25, 101)
    alpha_scores = [
        score_probabilities(labels, probs, priors, alpha=float(a))
        for a in alpha_candidates
    ]
    best_alpha = float(alpha_candidates[int(np.argmax(alpha_scores))])
    log_bias = np.zeros(len(priors), dtype=float)
    bias_grid = np.linspace(-0.22, 0.22, 89)

    for _ in range(3):
        for class_index in range(1, len(priors)):
            best_value = log_bias[class_index]
            best_score = -np.inf
            for candidate in bias_grid:
                candidate_bias = log_bias.copy()
                candidate_bias[class_index] = float(candidate)
                score = score_probabilities(
                    labels, probs, priors,
                    alpha=best_alpha,
                    log_bias=candidate_bias,
                )
                if score > best_score:
                    best_score = score
                    best_value = float(candidate)
            log_bias[class_index] = best_value

    final_score = score_probabilities(
        labels, probs, priors,
        alpha=best_alpha,
        log_bias=log_bias,
    )
    return best_alpha, log_bias, final_score

best_alpha, best_log_bias, calibration_score = tune_alpha_and_bias(
    y, oof_blend_optimized, class_priors, calibration_idx
)

conservative_score = score_probabilities(
    y[decision_valid_idx], oof_blend_equal[decision_valid_idx],
    class_priors, alpha=1.0
)
optimized_validation_score = score_probabilities(
    y[decision_valid_idx], oof_blend_optimized[decision_valid_idx],
    class_priors, alpha=best_alpha, log_bias=best_log_bias
)

accept_optimized = optimized_validation_score >= conservative_score

if accept_optimized:
    final_weights = optimized_weights
    final_alpha = best_alpha
    final_log_bias = best_log_bias
    final_oof_blend = oof_blend_optimized
    final_test_blend = test_blend_optimized
    selected_strategy = 'optimized weights + held-out-validated decision calibration'
else:
    final_weights = initial_weights
    final_alpha = 1.0
    final_log_bias = np.zeros(n_classes)
    final_oof_blend = oof_blend_equal
    final_test_blend = test_blend_equal
    selected_strategy = 'conservative equal-weight prior-corrected blend'

summary = {
    'selected_strategy': selected_strategy,
    'model_names': model_names,
    'final_weights': dict(zip(model_names, final_weights.tolist())),
    'final_alpha': float(final_alpha),
    'final_log_bias': dict(zip(class_names, final_log_bias.tolist())),
    'calibration_score': float(calibration_score),
    'decision_validation_optimized': float(optimized_validation_score),
    'decision_validation_conservative': float(conservative_score),
    'accepted_optimized': bool(accept_optimized),
}
print(json.dumps(summary, indent=2))



final_oof_pred = balanced_prior_predict(
    final_oof_blend, class_priors,
    alpha=final_alpha, log_bias=final_log_bias
)
final_oof_score = balanced_accuracy_score(y, final_oof_pred)
final_confusion = confusion_matrix(y, final_oof_pred, normalize='true')

print('Final full-OOF balanced accuracy:', round(final_oof_score, 6))
display(pd.DataFrame(
    final_confusion,
    index=[f'true_{name}' for name in class_names],
    columns=[f'pred_{name}' for name in class_names],
).round(5))

display(pd.DataFrame({
    'class': class_names,
    'recall': np.diag(final_confusion),
    'train_prevalence': class_priors,
}).sort_values('class'))



if feature_importance_records:
    feature_importance = pd.concat(feature_importance_records, ignore_index=True)
    feature_importance_summary = (
        feature_importance
        .groupby(['model', 'feature'], as_index=False)['importance']
        .mean()
        .sort_values(['model', 'importance'], ascending=[True, False])
    )
    display(feature_importance_summary.groupby('model', group_keys=False).head(20).reset_index(drop=True))
else:
    print('No LightGBM feature importances were recorded.')



def make_submission(probabilities, filename, alpha, log_bias=None):
    encoded_prediction = balanced_prior_predict(
        probabilities, class_priors,
        alpha=alpha, log_bias=log_bias
    )
    prediction = label_encoder.inverse_transform(encoded_prediction)
    submission = sample_submission.copy()
    submission[TARGET] = prediction

    assert submission.shape == sample_submission.shape
    assert submission[ID_COL].equals(sample_submission[ID_COL])
    assert set(submission[TARGET].unique()).issubset(set(class_names))
    assert not submission.isna().any().any()

    output_path = OUTPUT_DIR / filename
    submission.to_csv(output_path, index=False)
    print(f'Saved {output_path}')
    print(submission[TARGET].value_counts(normalize=True).round(6).to_dict())
    return submission

submission_best = make_submission(
    final_test_blend, 'submission_best_cv.csv',
    alpha=final_alpha, log_bias=final_log_bias
)
submission_robust = make_submission(
    test_blend_equal, 'submission_robust_prior.csv',
    alpha=1.0, log_bias=np.zeros(n_classes)
)
submission_raw = make_submission(
    test_blend_equal, 'submission_raw_blend.csv',
    alpha=0.0, log_bias=np.zeros(n_classes)
)
submission_alpha_095 = make_submission(
    test_blend_equal, 'submission_alpha_095.csv',
    alpha=0.95, log_bias=np.zeros(n_classes)
)
submission_alpha_105 = make_submission(
    test_blend_equal, 'submission_alpha_105.csv',
    alpha=1.05, log_bias=np.zeros(n_classes)
)

display(submission_best.head())



metadata = {
    **summary,
    'seed': SEED,
    'n_splits': N_SPLITS,
    'fast_mode': FAST_MODE,
    'use_gpu': USE_GPU,
    'class_names': class_names.tolist(),
    'class_priors': class_priors.tolist(),
    'final_full_oof_balanced_accuracy': float(final_oof_score),
    'train_rows': int(len(X_train)),
    'test_rows': int(len(X_test)),
    'feature_count': int(X_train.shape[1]),
}
metadata_path = OUTPUT_DIR / 'ensemble_metadata.json'
metadata_path.write_text(json.dumps(metadata, indent=2))
print(metadata_path)


# ============================================================
# V2 neural configuration and feature domain
# ============================================================
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.class_weight import compute_class_weight

RUN_NEURAL = True
NEURAL_FAST_MODE = FAST_MODE
NEURAL_N_FOLDS = 2 if NEURAL_FAST_MODE else 5
NEURAL_EPOCHS = 2 if NEURAL_FAST_MODE else 4
NEURAL_SEEDS = [SEED + 701] if NEURAL_FAST_MODE else [SEED + 701, SEED + 1701]
NEURAL_BATCH_SIZE = 4096 if NEURAL_FAST_MODE else 2048
NEURAL_EVAL_BATCH_SIZE = 16384
NEURAL_EMBED_DIM = 6
NEURAL_LEARNING_RATE = 2.0e-3
NEURAL_LABEL_SMOOTHING = 0.03
NEURAL_EMA_DECAY = 0.997

if torch.cuda.is_available():
    neural_gpu_index = 1 if torch.cuda.device_count() > 1 else 0
    NEURAL_DEVICE = torch.device(f"cuda:{neural_gpu_index}")
else:
    NEURAL_DEVICE = torch.device("cpu")


def prepare_neural_domain(train_frame, test_frame):
    train_nn = train_frame.drop(columns=[ID_COL, TARGET]).copy()
    test_nn = test_frame.drop(columns=[ID_COL]).copy()

    original_categorical = train_nn.select_dtypes(include=["object", "category"]).columns.tolist()
    original_numeric = [c for c in train_nn.columns if c not in original_categorical]

    # Small transductive stabilization: levels absent from either side are
    # treated as missing. No target values are used.
    for col in original_numeric:
        common = set(train_nn[col].dropna().unique()) & set(test_nn[col].dropna().unique())
        train_nn.loc[train_nn[col].notna() & ~train_nn[col].isin(common), col] = np.nan
        test_nn.loc[test_nn[col].notna() & ~test_nn[col].isin(common), col] = np.nan
        train_nn[col] = train_nn[col].fillna(0.0)
        test_nn[col] = test_nn[col].fillna(0.0)

    discretized = []
    for col in original_numeric:
        if col == "step_count":
            fine_train = np.floor(train_nn[col] / 10.0)
            fine_test = np.floor(test_nn[col] / 10.0)
            coarse_train = np.floor(train_nn[col] / 20.0)
            coarse_test = np.floor(test_nn[col] / 20.0)
        elif col == "calorie_expenditure":
            fine_train = np.floor(train_nn[col] / 5.0)
            fine_test = np.floor(test_nn[col] / 5.0)
            coarse_train = np.floor(train_nn[col] / 50.0)
            coarse_test = np.floor(test_nn[col] / 50.0)
        elif col == "water_intake":
            fine_train = train_nn[col].round(4)
            fine_test = test_nn[col].round(4)
            coarse_train = np.floor(train_nn[col] * 50.0)
            coarse_test = np.floor(test_nn[col] * 50.0)
        elif col in {"heart_rate", "bmi"}:
            fine_train = train_nn[col].round(4)
            fine_test = test_nn[col].round(4)
            coarse_train = np.floor(train_nn[col] * 5.0)
            coarse_test = np.floor(test_nn[col] * 5.0)
        else:
            fine_train = train_nn[col].round(4)
            fine_test = test_nn[col].round(4)
            coarse_train = np.floor(train_nn[col] / 2.0)
            coarse_test = np.floor(test_nn[col] / 2.0)

        fine_name = f"{col}__fine"
        coarse_name = f"{col}__coarse"
        train_nn[fine_name] = fine_train.astype(str)
        test_nn[fine_name] = fine_test.astype(str)
        train_nn[coarse_name] = coarse_train.astype(str)
        test_nn[coarse_name] = coarse_test.astype(str)
        discretized.extend([fine_name, coarse_name])

    categorical_columns_nn = original_categorical + discretized

    for col in categorical_columns_nn:
        combined = pd.concat(
            [
                train_nn[col].fillna("__MISSING__").astype(str),
                test_nn[col].fillna("__MISSING__").astype(str),
            ],
            ignore_index=True,
        )
        mapping = {value: index + 1 for index, value in enumerate(pd.Index(combined.unique()))}
        train_nn[col] = train_nn[col].fillna("__MISSING__").astype(str).map(mapping).fillna(0).astype("int32")
        test_nn[col] = test_nn[col].fillna("__MISSING__").astype(str).map(mapping).fillna(0).astype("int32")

    category_dimensions = (
        pd.concat(
            [train_nn[categorical_columns_nn], test_nn[categorical_columns_nn]],
            ignore_index=True,
        )
        .max(axis=0)
        .astype("int64")
        .to_numpy()
        + 1
    )

    return (
        train_nn,
        test_nn,
        original_numeric,
        categorical_columns_nn,
        discretized,
        category_dimensions,
    )


(
    train_nn,
    test_nn,
    neural_raw_numeric_columns,
    neural_categorical_columns,
    neural_target_encoding_columns,
    neural_category_dimensions,
) = prepare_neural_domain(train, test)

print({
    "RUN_NEURAL": RUN_NEURAL,
    "NEURAL_DEVICE": str(NEURAL_DEVICE),
    "NEURAL_N_FOLDS": NEURAL_N_FOLDS,
    "NEURAL_EPOCHS": NEURAL_EPOCHS,
    "NEURAL_SEEDS": NEURAL_SEEDS,
    "raw_numeric": len(neural_raw_numeric_columns),
    "categorical": len(neural_categorical_columns),
    "target_encoded_bins": len(neural_target_encoding_columns),
})


# ============================================================
# V2 neural architecture
# ============================================================
class RobustSmoothScaler:
    def fit(self, values):
        self.median_ = np.median(values, axis=0)
        spread = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
        zero = spread == 0
        if np.any(zero):
            fallback = 0.5 * (np.max(values, axis=0) - np.min(values, axis=0))
            spread[zero] = fallback[zero]
        self.factor_ = 1.0 / (spread + 1e-30)
        self.factor_[spread == 0] = 0.0
        return self

    def transform(self, values):
        scaled = (values - self.median_[None, :]) * self.factor_[None, :]
        return scaled / np.sqrt(1.0 + (scaled / 3.0) ** 2)


def periodic_expand(values):
    pieces = [values]
    for frequency in (0.5, 1.0, 2.0):
        pieces.append(np.sin(math.pi * frequency * values))
        pieces.append(np.cos(math.pi * frequency * values))
    return np.concatenate(pieces, axis=1).astype(np.float32)


class ResidualMLPBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.normalization_1 = nn.LayerNorm(width)
        self.linear_1 = nn.Linear(width, width * 2)
        self.normalization_2 = nn.LayerNorm(width * 2)
        self.linear_2 = nn.Linear(width * 2, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        hidden = self.normalization_1(values)
        hidden = F.gelu(self.linear_1(hidden))
        hidden = self.dropout(hidden)
        hidden = self.normalization_2(hidden)
        hidden = self.linear_2(hidden)
        return values + self.dropout(hidden)


class StudentHealthSignalMLP(nn.Module):
    def __init__(self, numeric_features, category_dimensions, n_classes, embed_dim=6):
        super().__init__()
        self.embeddings = nn.ModuleList()
        embedding_output = 0
        for dimension in category_dimensions:
            dimension = int(dimension)
            current_dim = min(embed_dim, max(2, int(round(math.log2(max(dimension, 2))))))
            self.embeddings.append(nn.Embedding(dimension, current_dim, padding_idx=0))
            embedding_output += current_dim

        input_features = numeric_features + embedding_output
        width = 384
        self.input_norm = nn.LayerNorm(input_features)
        self.input_projection = nn.Linear(input_features, width)
        self.blocks = nn.ModuleList([ResidualMLPBlock(width, 0.08) for _ in range(3)])
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, n_classes)

    def forward(self, numeric_values, categorical_values):
        embedded = [
            embedding(categorical_values[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        hidden = torch.cat([numeric_values] + embedded, dim=1)
        hidden = self.input_projection(self.input_norm(hidden))
        hidden = F.gelu(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden))


def state_to_cpu(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


@torch.no_grad()
def update_ema_state(ema_state, model_state, decay):
    for key, value in model_state.items():
        if torch.is_floating_point(value):
            ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            ema_state[key].copy_(value)


@torch.no_grad()
def predict_signal_mlp(model, numeric_tensor, categorical_tensor, batch_size):
    model.eval()
    output = []
    for start in range(0, len(numeric_tensor), batch_size):
        stop = start + batch_size
        logits = model(numeric_tensor[start:stop], categorical_tensor[start:stop])
        output.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(output, axis=0)


# ============================================================
# V2 cross-validated neural training
# ============================================================
if RUN_NEURAL:
    neural_oof = np.zeros((len(train_nn), n_classes), dtype=np.float32)
    neural_test_by_fold = []

    all_indices = np.arange(len(y))
    outer_fold_id = all_indices % NEURAL_N_FOLDS

    train_numeric_base = train_nn[neural_raw_numeric_columns].to_numpy(dtype=np.float32)
    test_numeric_base = test_nn[neural_raw_numeric_columns].to_numpy(dtype=np.float32)
    train_categorical_base = train_nn[neural_categorical_columns].to_numpy(dtype=np.int64)
    test_categorical_base = test_nn[neural_categorical_columns].to_numpy(dtype=np.int64)

    for fold in range(NEURAL_N_FOLDS):
        print(f"\n========== Signal MLP fold {fold + 1}/{NEURAL_N_FOLDS} ==========")
        validation_indices = all_indices[outer_fold_id == fold]
        training_indices = all_indices[outer_fold_id != fold]
        y_fold_train = y[training_indices]
        y_fold_valid = y[validation_indices]

        target_encoder = TargetEncoder(
            target_type="multiclass",
            smooth="auto",
            cv=5,
            shuffle=True,
            random_state=SEED + fold,
        )
        encoded_train = target_encoder.fit_transform(
            train_nn.iloc[training_indices][neural_target_encoding_columns],
            y_fold_train,
        ).astype(np.float32)
        encoded_valid = target_encoder.transform(
            train_nn.iloc[validation_indices][neural_target_encoding_columns]
        ).astype(np.float32)
        encoded_test = target_encoder.transform(
            test_nn[neural_target_encoding_columns]
        ).astype(np.float32)

        numeric_train = np.concatenate([train_numeric_base[training_indices], encoded_train], axis=1)
        numeric_valid = np.concatenate([train_numeric_base[validation_indices], encoded_valid], axis=1)
        numeric_test = np.concatenate([test_numeric_base, encoded_test], axis=1)

        scaler = RobustSmoothScaler().fit(numeric_train)
        numeric_train = periodic_expand(scaler.transform(numeric_train))
        numeric_valid = periodic_expand(scaler.transform(numeric_valid))
        numeric_test = periodic_expand(scaler.transform(numeric_test))

        categorical_train = train_categorical_base[training_indices]
        categorical_valid = train_categorical_base[validation_indices]

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(n_classes),
            y=y_fold_train,
        ) * np.array([0.90, 1.10, 1.00])
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=NEURAL_DEVICE)

        numeric_train_tensor = torch.tensor(numeric_train, dtype=torch.float32, device=NEURAL_DEVICE)
        numeric_valid_tensor = torch.tensor(numeric_valid, dtype=torch.float32, device=NEURAL_DEVICE)
        numeric_test_tensor = torch.tensor(numeric_test, dtype=torch.float32, device=NEURAL_DEVICE)
        categorical_train_tensor = torch.tensor(categorical_train, dtype=torch.long, device=NEURAL_DEVICE)
        categorical_valid_tensor = torch.tensor(categorical_valid, dtype=torch.long, device=NEURAL_DEVICE)
        categorical_test_tensor = torch.tensor(test_categorical_base, dtype=torch.long, device=NEURAL_DEVICE)
        y_train_tensor = torch.tensor(y_fold_train, dtype=torch.long, device=NEURAL_DEVICE)

        seed_valid_predictions = []
        seed_test_predictions = []

        for neural_seed in NEURAL_SEEDS:
            torch.manual_seed(neural_seed + fold)
            np.random.seed(neural_seed + fold)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(neural_seed + fold)

            model = StudentHealthSignalMLP(
                numeric_features=numeric_train.shape[1],
                category_dimensions=neural_category_dimensions,
                n_classes=n_classes,
                embed_dim=NEURAL_EMBED_DIM,
            ).to(NEURAL_DEVICE)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=NEURAL_LEARNING_RATE,
                weight_decay=1.0e-4,
                betas=(0.9, 0.98),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(NEURAL_EPOCHS, 1),
                eta_min=NEURAL_LEARNING_RATE * 0.05,
            )

            ema_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_state = None
            best_score = -np.inf
            order = np.arange(len(training_indices))

            for epoch in range(NEURAL_EPOCHS):
                rng = np.random.default_rng(neural_seed + fold * 100 + epoch)
                rng.shuffle(order)
                model.train()

                for start in range(0, len(order), NEURAL_BATCH_SIZE):
                    selected = order[start:start + NEURAL_BATCH_SIZE]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(
                        numeric_train_tensor[selected],
                        categorical_train_tensor[selected],
                    )
                    loss = F.cross_entropy(
                        logits,
                        y_train_tensor[selected],
                        weight=class_weights_tensor,
                        label_smoothing=NEURAL_LABEL_SMOOTHING,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    update_ema_state(ema_state, model.state_dict(), NEURAL_EMA_DECAY)

                scheduler.step()

                live_state = state_to_cpu(model.state_dict())
                model.load_state_dict(ema_state, strict=True)
                valid_probability = predict_signal_mlp(
                    model,
                    numeric_valid_tensor,
                    categorical_valid_tensor,
                    NEURAL_EVAL_BATCH_SIZE,
                )
                epoch_score = balanced_accuracy_score(y_fold_valid, valid_probability.argmax(axis=1))
                print(
                    f"fold={fold + 1}, seed={neural_seed}, epoch={epoch + 1}, "
                    f"balanced_accuracy={epoch_score:.6f}"
                )
                if epoch_score > best_score:
                    best_score = epoch_score
                    best_state = state_to_cpu(model.state_dict())
                model.load_state_dict(live_state, strict=True)

            if best_state is None:
                raise RuntimeError("No neural checkpoint was captured.")

            model.load_state_dict(best_state, strict=True)
            model = model.to(NEURAL_DEVICE)
            seed_valid_predictions.append(
                predict_signal_mlp(model, numeric_valid_tensor, categorical_valid_tensor, NEURAL_EVAL_BATCH_SIZE)
            )
            seed_test_predictions.append(
                predict_signal_mlp(model, numeric_test_tensor, categorical_test_tensor, NEURAL_EVAL_BATCH_SIZE)
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        neural_oof[validation_indices] = np.mean(seed_valid_predictions, axis=0)
        neural_test_by_fold.append(np.mean(seed_test_predictions, axis=0))

        del (
            target_encoder,
            encoded_train,
            encoded_valid,
            encoded_test,
            numeric_train,
            numeric_valid,
            numeric_test,
            numeric_train_tensor,
            numeric_valid_tensor,
            numeric_test_tensor,
            categorical_train_tensor,
            categorical_valid_tensor,
            categorical_test_tensor,
            y_train_tensor,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    neural_test = np.mean(neural_test_by_fold, axis=0).astype(np.float32)
    neural_oof_score = balanced_accuracy_score(y, neural_oof.argmax(axis=1))
    print("Signal MLP full OOF balanced accuracy:", round(neural_oof_score, 6))

    np.save(OUTPUT_DIR / "signal_mlp_oof.npy", neural_oof)
    np.save(OUTPUT_DIR / "signal_mlp_test.npy", neural_test)
else:
    neural_oof = np.load(OUTPUT_DIR / "signal_mlp_oof.npy")
    neural_test = np.load(OUTPUT_DIR / "signal_mlp_test.npy")
    neural_oof_score = balanced_accuracy_score(y, neural_oof.argmax(axis=1))


# ============================================================
# V2 robust geometric meta-blend
# ============================================================
def safe_log_probability(probabilities):
    return np.log(np.clip(probabilities, 1e-12, 1.0))

TREE_REFERENCE_ALPHA = 0.95
# Equal tree weights were the strongest public V1 anchor.
tree_oof_probability = oof_blend_equal
tree_test_probability = test_blend_equal

meta_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 909)
meta_folds = list(meta_splitter.split(np.zeros(len(y)), y))

records = []
for tree_alpha in np.round(np.arange(0.88, 1.001, 0.02), 2):
    tree_oof_logits = safe_log_probability(tree_oof_probability) - tree_alpha * np.log(class_priors[None, :])
    for neural_weight in np.round(np.arange(0.00, 0.651, 0.05), 2):
        combined_logits = (
            (1.0 - neural_weight) * tree_oof_logits
            + neural_weight * safe_log_probability(neural_oof)
        )
        prediction = combined_logits.argmax(axis=1)
        fold_scores = [
            balanced_accuracy_score(y[valid_indices], prediction[valid_indices])
            for _, valid_indices in meta_folds
        ]
        mean_score = float(np.mean(fold_scores))
        score_std = float(np.std(fold_scores))
        records.append({
            "tree_alpha": float(tree_alpha),
            "neural_weight": float(neural_weight),
            "mean_meta_bacc": mean_score,
            "std_meta_bacc": score_std,
            "robust_meta_score": mean_score - 0.50 * score_std,
            "full_oof_bacc": float(balanced_accuracy_score(y, prediction)),
        })

meta_results = pd.DataFrame(records).sort_values(
    ["robust_meta_score", "full_oof_bacc"], ascending=False
).reset_index(drop=True)
display(meta_results.head(20))

selected = meta_results.iloc[0]
selected_tree_alpha = float(selected["tree_alpha"])
selected_neural_weight = float(selected["neural_weight"])

print({
    "selected_tree_alpha": selected_tree_alpha,
    "selected_neural_weight": selected_neural_weight,
    "selected_full_oof_bacc": float(selected["full_oof_bacc"]),
    "selected_robust_meta_score": float(selected["robust_meta_score"]),
    "signal_mlp_oof_bacc": float(neural_oof_score),
})

v2_test_logits = (
    (1.0 - selected_neural_weight)
    * (
        safe_log_probability(tree_test_probability)
        - selected_tree_alpha * np.log(class_priors[None, :])
    )
    + selected_neural_weight * safe_log_probability(neural_test)
)
v2_test_prediction = v2_test_logits.argmax(axis=1)


# ============================================================
# V2 submissions and probability archive
# ============================================================
def save_encoded_submission(encoded_prediction, filename):
    submission = sample_submission.copy()
    submission[TARGET] = label_encoder.inverse_transform(encoded_prediction.astype(int))
    assert submission[ID_COL].equals(sample_submission[ID_COL])
    assert not submission.isna().any().any()
    output_path = OUTPUT_DIR / filename
    submission.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
    print(submission[TARGET].value_counts(normalize=True).round(6).to_dict())
    return submission

submission_v2_primary = save_encoded_submission(v2_test_prediction, "submission_v2_primary.csv")
submission_v2_neural = save_encoded_submission(neural_test.argmax(axis=1), "submission_v2_signal_mlp_only.csv")

for neural_weight in (0.20, 0.30, 0.40):
    logits = (
        (1.0 - neural_weight)
        * (
            safe_log_probability(tree_test_probability)
            - TREE_REFERENCE_ALPHA * np.log(class_priors[None, :])
        )
        + neural_weight * safe_log_probability(neural_test)
    )
    save_encoded_submission(
        logits.argmax(axis=1),
        f"submission_v2_fixed_blend_w{int(neural_weight * 100):02d}.csv",
    )

np.savez_compressed(
    OUTPUT_DIR / "v2_probability_archive.npz",
    y=y,
    class_names=class_names,
    class_priors=class_priors,
    tree_oof=tree_oof_probability,
    tree_test=tree_test_probability,
    neural_oof=neural_oof,
    neural_test=neural_test,
)

v2_metadata = {
    "tree_reference_alpha": TREE_REFERENCE_ALPHA,
    "selected_tree_alpha": selected_tree_alpha,
    "selected_neural_weight": selected_neural_weight,
    "selected_full_oof_bacc": float(selected["full_oof_bacc"]),
    "selected_robust_meta_score": float(selected["robust_meta_score"]),
    "signal_mlp_oof_bacc": float(neural_oof_score),
    "neural_device": str(NEURAL_DEVICE),
    "neural_folds": NEURAL_N_FOLDS,
    "neural_epochs": NEURAL_EPOCHS,
    "neural_seeds": NEURAL_SEEDS,
}
(OUTPUT_DIR / "v2_metadata.json").write_text(json.dumps(v2_metadata, indent=2))
display(submission_v2_primary.head())
