import os
import glob
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TARGET = 'addicted_label'
NUM_COLS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time', 'slack', 'social_gaming_ratio']
CAT_COLS = ['gender', 'stress_level', 'academic_work_impact']

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    # First: check Kaggle input structure
    for root, _, files in os.walk('/kaggle/input/'):
        if 'train.csv' in files:
            train_path = os.path.join(root, 'train.csv')
            test_path = os.path.join(root, 'test.csv')
            sub_path = os.path.join(root, 'sample_submission.csv')
            print(f"Found Kaggle data at: {root}")
            return pd.read_csv(train_path), pd.read_csv(test_path), pd.read_csv(sub_path)

    # Second: check local development path
    local_base = os.environ.get(
        'S6E8_DATA',
        os.path.expanduser('~/kaggle-portfolio/competitions/smartphone-addiction/data/')
    )
    train_path = os.path.join(local_base, 'train.csv')
    test_path = os.path.join(local_base, 'test.csv')
    sub_path = os.path.join(local_base, 'sample_submission.csv')
    
    if all(os.path.exists(p) for p in [train_path, test_path, sub_path]):
        print(f"Using local data from: {local_base}")
        return pd.read_csv(train_path), pd.read_csv(test_path), pd.read_csv(sub_path)

    # Final: list everything for debugging
    print("ERROR: No data found in Kaggle or local paths")
    print("Checking Kaggle input structure:")
    for root, _, files in os.walk('/kaggle/input/'):
        for f in files:
            print(f"  {os.path.join(root, f)}")
    raise FileNotFoundError("No train.csv found in Kaggle or local paths")


# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def add_budget_features(df):
    """Finding 2: Generative constraint — daily_screen >= social + gaming + work.
    Create slack = daily_screen - (social + gaming + work) and flag violations."""
    # Use hardcoded budget columns to avoid NUM_COLS dependency
    budget_cols = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours']
    complete = df[budget_cols].notna().all(axis=1)
    df['slack'] = df.loc[complete, 'daily_screen_time_hours'] - df.loc[complete, budget_cols[1:]].sum(axis=1)
    df['slack_violation'] = (df['slack'] < -1e-10) & complete
    return df


def prepare_data(train, test):
    # Create features
    train = add_budget_features(train)
    test = add_budget_features(test)

    # Convert categorical columns to string to avoid CatBoost float errors
    cat_cols_to_convert = ['gender', 'stress_level', 'academic_work_impact', 'slack_violation', 'high_slack']
    for col in cat_cols_to_convert:
        if col in train.columns:
            train[col] = train[col].fillna('missing').astype(str)
        if col in test.columns:
            test[col] = test[col].fillna('missing').astype(str)

    # Add interaction features
    train['social_gaming_ratio'] = train['social_media_hours'] / (train['gaming_hours'] + 1e-5)
    test['social_gaming_ratio'] = test['social_media_hours'] / (test['gaming_hours'] + 1e-5)

    # Fill missing values
    for col in NUM_COLS:
        train[col] = train[col].fillna(train[col].median())
        test[col] = test[col].fillna(train[col].median())

    # Create categorical features
    train['high_slack'] = (train['slack'] > 1).astype(int)
    test['high_slack'] = (test['slack'] > 1).astype(int)

    return train, test


# ═══════════════════════════════════════════════════════════════════════════
# 3. CATBOOST MODEL
# ═══════════════════════════════════════════════════════════════════════════

def train_catboost(train, test, n_folds=11, seed=42):
    import catboost
    from catboost import CatBoostClassifier, Pool

    X = train.drop(columns=[TARGET])
    y = train[TARGET]
    X_test = test.copy()

    cat_features = [col for col in X.columns if col not in NUM_COLS + [TARGET]]

    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_aucs = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"  CatBoost fold {fold+1}/{n_folds}")
        t0 = time.time()

        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            eval_metric='AUC',
            task_type='GPU',
            devices='0',
            random_state=seed,
            early_stopping_rounds=50,
            verbose=0
        )
        model.fit(
            Pool(X_tr, y_tr, cat_features=cat_features),
            eval_set=Pool(X_va, y_va, cat_features=cat_features),
            use_best_model=True
        )

        va_preds = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = va_preds
        test_preds += model.predict_proba(X_test)[:, 1] / n_folds

        va_auc = roc_auc_score(y_va, va_preds)
        fold_aucs.append(va_auc)
        print(f"    Fold {fold+1} AUC: {va_auc:.6f} ({time.time()-t0:.1f}s)")

    overall_auc = roc_auc_score(y, oof)
    print(f"  CatBoost OOF AUC: {overall_auc:.6f}")
    return oof, test_preds, {'overall': overall_auc, 'folds': fold_aucs}


# ═══════════════════════════════════════════════════════════════════════════
# 3. LIGHTGBM MODEL
# ═══════════════════════════════════════════════════════════════════════════

def train_lightgbm(train, test, n_folds=11, seed=42):
    """LightGBM companion with fold-local exact-value target encoding."""
    import lightgbm as lgb

    feature_cols = NUM_COLS + CAT_COLS + [
        'slack_violation', 'high_slack'
    ]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    # Encode categoricals using a train+test vocabulary, without target leakage.
    work_train = train.copy()
    work_test = test.copy()
    for col in CAT_COLS + ['slack_violation', 'high_slack']:
        all_values = pd.concat([work_train[col], work_test[col]], ignore_index=True).astype(str)
        categories = pd.Categorical(all_values).categories
        work_train[col] = pd.Categorical(work_train[col].astype(str), categories=categories).codes
        work_test[col] = pd.Categorical(work_test[col].astype(str), categories=categories).codes

    te_cols = NUM_COLS
    for col in te_cols:
        work_train[f'{col}__te'] = 0.0
        work_test[f'{col}__te'] = 0.0
    all_features = feature_cols + [f'{c}__te' for c in te_cols]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(work_train, work_train[TARGET])):
        print(f"  LightGBM fold {fold+1}/{n_folds}")
        global_mean = work_train.iloc[tr_idx][TARGET].mean()
        for col in te_cols:
            stats = work_train.iloc[tr_idx].groupby(col)[TARGET].agg(['mean', 'count'])
            stats['value'] = (stats['count'] * stats['mean'] + 10 * global_mean) / (stats['count'] + 10)
            mapping = stats['value'].to_dict()
            work_train.loc[work_train.index[va_idx], f'{col}__te'] = work_train.iloc[va_idx][col].map(mapping).fillna(global_mean).to_numpy()
            work_train.loc[work_train.index[tr_idx], f'{col}__te'] = work_train.iloc[tr_idx][col].map(mapping).fillna(global_mean).to_numpy()
            work_test[f'{col}__te'] += work_test[col].map(mapping).fillna(global_mean).to_numpy() / n_folds

        model = lgb.LGBMClassifier(
            n_estimators=6000, learning_rate=0.03, num_leaves=63, max_depth=7,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=seed + fold,
            verbose=-1, n_jobs=-1,
        )
        model.fit(
            work_train.iloc[tr_idx][all_features], work_train.iloc[tr_idx][TARGET],
            eval_set=[(work_train.iloc[va_idx][all_features], work_train.iloc[va_idx][TARGET])],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va_idx] = model.predict_proba(work_train.iloc[va_idx][all_features])[:, 1]
        test_preds += model.predict_proba(work_test[all_features])[:, 1] / n_folds
        print(f"    Fold {fold+1} AUC: {roc_auc_score(work_train.iloc[va_idx][TARGET], oof[va_idx]):.6f}")

    overall_auc = roc_auc_score(work_train[TARGET], oof)
    print(f"  LightGBM OOF AUC: {overall_auc:.6f}")
    return oof, test_preds, {'overall': overall_auc}


# ═══════════════════════════════════════════════════════════════════════════
# 4. NEURAL NET MODEL
# ═══════════════════════════════════════════════════════════════════════════

class EMA:
    def __init__(self, model, decay):
        self.shadow = {}
        self.decay = decay
        self.model = model
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

class EmbeddingPLRTransformer(nn.Module):
    def __init__(self, vocab_sizes, embed_dim, n_heads, n_layers, plr_frequencies, dropout):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for vocab_size in vocab_sizes
        ])
        self.num_proj = nn.Linear(1, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, n_heads, dim_feedforward=embed_dim*4, dropout=dropout),
            n_layers
        )
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, cat_idx, num_vals, budget, mask):
        # Embed categorical features
        cat_embs = [emb(cat_idx[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embs = torch.stack(cat_embs, dim=1)  # [B, N_cat, D]

        # Project each numerical feature into the same token dimension.
        num_embs = self.num_proj(num_vals.unsqueeze(-1))  # [B, N_num, D]

        # Combine and mask
        x = torch.cat([cat_embs, num_embs], dim=1)  # [B, N_cat + N_num, D]
        cat_mask = torch.ones(
            (cat_idx.size(0), cat_embs.size(1)), dtype=mask.dtype, device=mask.device
        )
        combined_mask = torch.cat([cat_mask, mask], dim=1)
        x = x * combined_mask.unsqueeze(-1)

        # Transformer
        x = x.permute(1, 0, 2)  # [N, B, D]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, N, D]

        # Aggregate and predict
        x = x.mean(dim=1)  # [B, D]
        return self.fc(x).squeeze(-1)

def train_neural_net(train, test, n_folds=11, seed=42, epochs=32):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Preprocess
    X = train.drop(columns=[TARGET])
    y = train[TARGET]
    X_test = test.copy()

    # Create categorical mappings
    cat_cols = [col for col in X.columns if col not in NUM_COLS]
    vocab_maps = {}
    for col in cat_cols:
        vocab = {val: i for i, val in enumerate(X[col].unique())}
        vocab_maps[col] = vocab
        X[col] = X[col].map(vocab).fillna(len(vocab)).astype(int)
        X_test[col] = X_test[col].map(vocab).fillna(len(vocab)).astype(int)

    # Convert to tensors
    def df_to_tensors(df):
        encoded_cats = []
        for col in cat_cols:
            encoded_cats.append(
                df[col].map(vocab_maps[col]).fillna(len(vocab_maps[col])).astype(np.int64).to_numpy()
            )
        cat_idx = torch.tensor(np.column_stack(encoded_cats), dtype=torch.long)
        num_array = df[NUM_COLS].to_numpy(dtype=np.float32)
        num_array = np.nan_to_num(num_array, nan=0.0, posinf=0.0, neginf=0.0)
        num_array = np.clip(num_array, -10.0, 10.0)
        budget_array = df['daily_screen_time_hours'].to_numpy(dtype=np.float32)
        budget_array = np.nan_to_num(budget_array, nan=0.0, posinf=0.0, neginf=0.0)
        budget_array = np.clip(budget_array, -10.0, 10.0)
        num_vals = torch.tensor(num_array, dtype=torch.float32)
        budget = torch.tensor(budget_array, dtype=torch.float32).unsqueeze(1)
        mask = torch.ones_like(num_vals)
        return cat_idx, num_vals, budget, mask

    # GPU compatibility check
    cuda_available = False
    if torch.cuda.is_available():
        try:
            test_tensor = torch.randn(64, 64, device='cuda')
            _ = test_tensor @ test_tensor.T
            torch.cuda.synchronize()
            cuda_available = True
            del test_tensor
        except RuntimeError as e:
            if "no kernel image" in str(e) or "CUDA error" in str(e):
                print(f"  ⚠️ CUDA incompatible (likely sm_60 P100 + new PyTorch build)")
                print(f"  Falling back to CPU for Neural Net")
                cuda_available = False
            else:
                raise

    device = torch.device('cuda' if cuda_available else 'cpu')
    print(f"  NN device: {device}")

    # CPU optimizations
    if cuda_available:
        embed_dim = 16
        n_layers = 2
        epochs_nn = epochs
    else:
        embed_dim = 12
        n_layers = 2
        epochs_nn = min(epochs, 24)
        print(f"  CPU mode: embed_dim={embed_dim}, epochs={epochs_nn}")

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Training loop
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, train[TARGET])):
        t0 = time.time()
        print(f"  NN fold {fold+1}/{n_folds}")

        train_df = train.iloc[tr_idx].reset_index(drop=True)
        val_df = train.iloc[va_idx].reset_index(drop=True)
        test_df = test.reset_index(drop=True)

        # Convert to tensors
        tr_tensors = df_to_tensors(train_df)
        va_tensors = df_to_tensors(val_df)
        te_tensors = df_to_tensors(test_df)

        # Create datasets
        train_dataset = TensorDataset(*tr_tensors, torch.tensor(train_df[TARGET].values, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

        # Model setup
        vocab_sizes = [len(vocab_maps[col]) + 1 for col in cat_cols]
        model = EmbeddingPLRTransformer(
            vocab_sizes=vocab_sizes, embed_dim=embed_dim, n_heads=4,
            n_layers=n_layers, plr_frequencies=16, dropout=0.1
        ).to(device)

        ema = EMA(model, decay=0.999)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=epochs_nn, steps_per_epoch=len(train_loader)
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_auc = 0
        best_epoch = 0
        best_oof_preds = None
        best_test_preds = None

        for epoch in range(epochs_nn):
            model.train()
            for batch in train_loader:
                cat_idx_b, num_vals_b, budget_b, mask_b, labels_b = [x.to(device) for x in batch]

                # Re-apply random masking per batch
                mask_b = torch.ones_like(mask_b)
                rand_mask = torch.rand_like(mask_b)
                mask_b[rand_mask < 0.1] = 0.0

                optimizer.zero_grad()
                logits = model(cat_idx_b, num_vals_b, budget_b, mask_b)
                loss = criterion(logits, labels_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                ema.update(model)

            # Validate every 4 epochs or at end
            if (epoch + 1) % 4 == 0 or epoch == epochs_nn - 1:
                orig_state = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema.state_dict())
                model.eval()

                with torch.no_grad():
                    va_logits = []
                    for i in range(0, len(val_df), 2048):
                        cat_b = va_tensors[0][i:i+2048].to(device)
                        num_b = va_tensors[1][i:i+2048].to(device)
                        bud_b = va_tensors[2][i:i+2048].to(device)
                        msk_b = va_tensors[3][i:i+2048].to(device)
                        logits = model(cat_b, num_b, bud_b, msk_b)
                        va_logits.append(logits.cpu().numpy())
                    va_preds = 1 / (1 + np.exp(-np.concatenate(va_logits)))
                    va_auc = roc_auc_score(val_df[TARGET], va_preds)

                if va_auc > best_val_auc:
                    best_val_auc = va_auc
                    best_epoch = epoch + 1
                    with torch.no_grad():
                        te_logits = []
                        for i in range(0, len(test_df), 2048):
                            cat_b = te_tensors[0][i:i+2048].to(device)
                            num_b = te_tensors[1][i:i+2048].to(device)
                            bud_b = te_tensors[2][i:i+2048].to(device)
                            msk_b = te_tensors[3][i:i+2048].to(device)
                            logits = model(cat_b, num_b, bud_b, msk_b)
                            te_logits.append(logits.cpu().numpy())
                        best_test_preds = 1 / (1 + np.exp(-np.concatenate(te_logits)))
                    best_oof_preds = va_preds.copy()

                model.load_state_dict(orig_state)
                print(f"    Epoch {epoch+1}/{epochs_nn} — Val AUC: {va_auc:.6f} (best: {best_val_auc:.6f})")

        oof[va_idx] = best_oof_preds
        test_preds += best_test_preds / n_folds
        fold_aucs.append(best_val_auc)

        elapsed = time.time() - t0
        print(f"    Fold {fold+1} best AUC: {best_val_auc:.6f} (epoch {best_epoch}, {elapsed:.1f}s)")

    overall_auc = roc_auc_score(train[TARGET], oof)
    print(f"  NN OOF AUC: {overall_auc:.6f}")
    return oof, test_preds, {'overall': overall_auc, 'folds': fold_aucs}


# ═══════════════════════════════════════════════════════════════════════════
# 5. RANK-NORMALIZED BLEND
# ═══════════════════════════════════════════════════════════════════════════

def rank_normalize(preds):
    return rankdata(preds) / len(preds)


def blend_models(oof_dict, test_dict, weights=None):
    if weights is None:
        weights = {'nn': 0.49, 'catboost': 0.31, 'lightgbm': 0.21}

    oof_blend = sum(weights[k] * rank_normalize(v) for k, v in oof_dict.items())
    test_blend = sum(weights[k] * rank_normalize(v) for k, v in test_dict.items())
    return oof_blend, test_blend


# ═══════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    out_dir = Path(os.environ.get('S6E8_OUT', '/workspace/s6e8/output'))
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Sprint 5 — Omralinov-inspired pipeline (Kaggle GPU)")
    print("=" * 70)

    # Load data
    train, test, sub = load_data()
    print("Train columns after load:", train.columns.tolist())
    print(f"Train: {train.shape}, Test: {test.shape}")

    # Verify budget constraint on complete rows
    complete = train[['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours']].notna().all(axis=1)
    slack = train.loc[complete, 'daily_screen_time_hours'] - train.loc[complete, ['social_media_hours', 'gaming_hours', 'work_study_hours']].sum(axis=1)
    print(f"Budget constraint (complete rows): {(slack >= -1e-10).all()}, min slack: {slack.min():.4f}")

    # Feature engineering
    train, test = prepare_data(train, test)
    print("Train columns after prepare:", train.columns.tolist())

    n_folds = 11
    seed = 42

    # Train all 3 models
    print("\n── CatBoost ──")
    cb_oof, cb_test, cb_metrics = train_catboost(train, test, n_folds=n_folds, seed=seed)
    np.save(out_dir / 'oof_catboost.npy', cb_oof)
    np.save(out_dir / 'test_catboost.npy', cb_test)
    with open(out_dir / 'catboost_metrics.json', 'w') as f:
        json.dump(cb_metrics, f, indent=2)
    print(f"CatBoost checkpoint saved to {out_dir}")

    print("\n── LightGBM ──")
    lgb_oof, lgb_test, lgb_metrics = train_lightgbm(train, test, n_folds=n_folds, seed=seed)
    np.save(out_dir / 'oof_lightgbm.npy', lgb_oof)
    np.save(out_dir / 'test_lightgbm.npy', lgb_test)
    with open(out_dir / 'lightgbm_metrics.json', 'w') as f:
        json.dump(lgb_metrics, f, indent=2)
    print(f"LightGBM checkpoint saved to {out_dir}")

    print("\n── Neural Net (Embedding + PLR + Transformer) ──")
    nn_oof, nn_test, nn_metrics = train_neural_net(train, test, n_folds=n_folds, seed=seed, epochs=32)
    np.save(out_dir / 'oof_nn.npy', nn_oof)
    np.save(out_dir / 'test_nn.npy', nn_test)
    with open(out_dir / 'nn_metrics.json', 'w') as f:
        json.dump(nn_metrics, f, indent=2)
    print(f"NN checkpoint saved to {out_dir}")

    # Blend
    print("\n── Blending ──")
    oof_dict = {'nn': nn_oof, 'catboost': cb_oof, 'lightgbm': lgb_oof}
    test_dict = {'nn': nn_test, 'catboost': cb_test, 'lightgbm': lgb_test}

    # Default Omralinov weights
    oof_blend, test_blend = blend_models(oof_dict, test_dict,
                                          weights={'nn': 0.49, 'catboost': 0.31, 'lightgbm': 0.21})
    blend_auc = roc_auc_score(train[TARGET], oof_blend)
    print(f"Blend OOF AUC (0.49/0.31/0.21): {blend_auc:.6f}")

    # Also try equal weights
    oof_eq, test_eq = blend_models(oof_dict, test_dict,
                                    weights={'nn': 1/3, 'catboost': 1/3, 'lightgbm': 1/3})
    eq_auc = roc_auc_score(train[TARGET], oof_eq)
    print(f"Blend OOF AUC (equal 1/3):      {eq_auc:.6f}")

    # Create submission (use best blend)
    best_weights = {'nn': 0.49, 'catboost': 0.31, 'lightgbm': 0.21} if blend_auc >= eq_auc else {'nn': 1/3, 'catboost': 1/3, 'lightgbm': 1/3}
    best_oof = oof_blend if blend_auc >= eq_auc else oof_eq
    best_test = test_blend if blend_auc >= eq_auc else test_eq
    best_auc = max(blend_auc, eq_auc)

    sub[TARGET] = best_test
    sub.to_csv("submission.csv", index=False)

    # Save all predictions for potential re-blending
    np.save("oof_nn.npy", nn_oof)
    np.save("oof_catboost.npy", cb_oof)
    np.save("oof_lightgbm.npy", lgb_oof)
    np.save("oof_blend.npy", best_oof)
    np.save("test_nn.npy", nn_test)
    np.save("test_catboost.npy", cb_test)
    np.save("test_lightgbm.npy", lgb_test)

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS (Sprint 5 — Omralinov-inspired)")
    print(f"{'='*70}")
    print(f"CatBoost  OOF AUC: {cb_metrics['overall']:.6f}")
    print(f"LightGBM  OOF AUC: {lgb_metrics['overall']:.6f}")
    print(f"NN        OOF AUC: {nn_metrics['overall']:.6f}")
    print(f"Blend     OOF AUC: {best_auc:.6f}")
    print(f"Blend weights: {best_weights}")
    print(f"Total time: {elapsed/60:.1f} min")
    print(f"Target: beat autonomous 0.96970 → need +0.0008 minimum")


if __name__ == "__main__":
    main()
