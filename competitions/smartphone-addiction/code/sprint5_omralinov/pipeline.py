"""
Sprint 5 — Omralinov-inspired pipeline for S6E8 Smartphone Addiction
=====================================================================
Key innovations from Tamerlan Omralinov (0.97052 public, rank 353):
1. Budget constraint features (other_screen, other_frac) — +0.00096/fold ablation
2. Exact-value embedding tables per column (lookup signal from quantized data)
3. PLR (Periodic-Linear Representation) — learned Fourier + linear for smooth trends
4. Transformer attention on feature tokens for interactions
5. NaN as index 0 → learned embedding per column
6. Random masking during training (augmentation on missingness patterns)
7. 11 folds, 32 epochs, AdamW + OneCycle + EMA

Companions: CatBoost (numeric+categorical) + LightGBM (target-encoding)
Blend: NN 0.49 / CatBoost 0.31 / LightGBM 0.21 on rank-normalized OOF
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/home/hermes/kaggle-portfolio/competitions/smartphone-addiction")
DATA = BASE / "data"
OUT = BASE / "code" / "sprint5_omralinov" / "output"
OUT.mkdir(parents=True, exist_ok=True)

# ── Feature columns ────────────────────────────────────────────────────────
NUM_COLS = [
    'age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
    'work_study_hours', 'sleep_hours', 'notifications_per_day',
    'app_opens_per_day', 'weekend_screen_time'
]
CAT_COLS = ['gender', 'stress_level', 'academic_work_impact']
TARGET = 'addicted_label'

# Columns that compose the budget constraint
BUDGET_TOTAL = 'daily_screen_time_hours'
BUDGET_COMPONENTS = ['social_media_hours', 'gaming_hours', 'work_study_hours']


def load_data():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sub = pd.read_csv(DATA / "sample_submission.csv")
    return train, test, sub


def add_budget_features(df):
    """Finding 2: Generative constraint — daily_screen ≥ social + gaming + work.
    The residual 'unaccounted screen time' is a strong signal (AUC 0.765).
    GBDTs cannot construct this 4-term sum with axis-aligned splits."""
    total = df[BUDGET_TOTAL].fillna(0)
    components_sum = df[BUDGET_COMPONENTS].fillna(0).sum(axis=1)
    df['other_screen'] = total - components_sum
    df['other_frac'] = df['other_screen'] / df[BUDGET_TOTAL].replace(0, np.nan)
    df['other_frac'] = df['other_frac'].fillna(0)
    df['on_boundary'] = (df['other_screen'] == 0).astype(float)
    # Additional budget-derived tokens (per Omralinov)
    for comp in BUDGET_COMPONENTS:
        df[f'{comp}_frac'] = df[comp].fillna(0) / df[BUDGET_TOTAL].replace(0, np.nan)
        df[f'{comp}_frac'] = df[f'{comp}_frac'].fillna(0)
    return df


def add_frequency_features(df, fit_df=None):
    """Value frequency features — fit on train only."""
    freq_cols = NUM_COLS
    for col in freq_cols:
        if fit_df is not None:
            freq = fit_df[col].value_counts(normalize=True)
        else:
            freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq).fillna(0)
    return df


def add_missingness_features(df):
    """Missingness indicators — the NaN pattern carries signal."""
    for col in NUM_COLS:
        df[f'{col}_missing'] = df[col].isna().astype(float)
    df['n_missing'] = df[[f'{c}_missing' for c in NUM_COLS]].sum(axis=1)
    return df


def prepare_data(train, test):
    """Full feature engineering pipeline."""
    train = add_budget_features(train)
    test = add_budget_features(test)
    
    train = add_frequency_features(train)
    test = add_frequency_features(test, fit_df=train)
    
    train = add_missingness_features(train)
    test = add_missingness_features(test)
    
    return train, test


# ── CatBoost with dual numeric + exact-value categorical ──────────────────
def train_catboost(train, test, n_folds=11, seed=42):
    """CatBoost treating quantized numeric columns as both numeric AND categorical
    (exact-value categorical representation). This captures the lookup signal
    that Omralinov's embedding tables exploit."""
    from catboost import CatBoostClassifier, Pool
    
    feature_cols = NUM_COLS + CAT_COLS + [
        'other_screen', 'other_frac', 'on_boundary',
        'social_media_hours_frac', 'gaming_hours_frac', 'work_study_hours_frac',
        'n_missing'
    ]
    # Add frequency features
    freq_feature_cols = [f'{c}_freq' for c in NUM_COLS]
    feature_cols += freq_feature_cols
    
    # CatBoost categorical columns must contain only strings or integers.
    # Convert missing source categoricals before every fold so the full-data
    # pipeline is safe on the real S6E8 missingness pattern.
    for col in CAT_COLS:
        train[col] = train[col].fillna('missing').astype(str)
        test[col] = test[col].fillna('missing').astype(str)

    # Exact-value categorical columns — quantized numeric values as strings
    exact_cat_cols = []
    for col in NUM_COLS:
        ec = f'{col}__exact_cat'
        train[ec] = train[col].fillna('missing').astype(str)
        test[ec] = test[col].fillna('missing').astype(str)
        exact_cat_cols.append(ec)
    
    all_features = feature_cols + exact_cat_cols
    cat_features_idx = [all_features.index(c) for c in CAT_COLS + exact_cat_cols]
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, train[TARGET])):
        print(f"  CatBoost fold {fold+1}/{n_folds}")
        X_tr = train.iloc[tr_idx][all_features]
        y_tr = train.iloc[tr_idx][TARGET]
        X_va = train.iloc[va_idx][all_features]
        y_va = train.iloc[va_idx][TARGET]
        X_te = test[all_features]
        
        model = CatBoostClassifier(
            iterations=6000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=3.0,
            random_seed=seed + fold,
            eval_metric='AUC',
            verbose=0,
            early_stopping_rounds=200,
            cat_features=cat_features_idx,
            one_hot_max_size=50,
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            verbose=0,
        )
        
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(X_te)[:, 1] / n_folds
        
        fold_auc = roc_auc_score(y_va, oof[va_idx])
        print(f"    Fold {fold+1} AUC: {fold_auc:.6f}")
    
    overall_auc = roc_auc_score(train[TARGET], oof)
    print(f"  CatBoost OOF AUC: {overall_auc:.6f}")
    return oof, test_preds


# ── LightGBM with fold-safe target encoding ───────────────────────────────
def train_lightgbm(train, test, n_folds=11, seed=42):
    """LightGBM with target encoding on exact values (fold-safe).
    The TE captures the lookup signal similarly to embedding tables."""
    import lightgbm as lgb
    
    feature_cols = NUM_COLS + CAT_COLS + [
        'other_screen', 'other_frac', 'on_boundary',
        'social_media_hours_frac', 'gaming_hours_frac', 'work_study_hours_frac',
        'n_missing'
    ]
    freq_feature_cols = [f'{c}_freq' for c in NUM_COLS]
    feature_cols += freq_feature_cols
    
    # Target encoding columns (exact values, fold-fitted)
    te_cols = NUM_COLS.copy()
    for col in te_cols:
        train[f'{col}__te'] = np.nan
        test[f'{col}__te'] = 0.0
    
    te_feature_cols = [f'{c}__te' for c in te_cols]
    all_features = feature_cols + te_feature_cols
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    # Encode categoricals
    cat_mappings = {}
    for col in CAT_COLS:
        cats = train[col].astype('category').cat.categories
        cat_mappings[col] = cats
        train[col] = train[col].astype('category').cat.codes.astype(int)
        test[col] = test[col].astype('category')
        test[col] = test[col].cat.set_categories(cats).cat.codes.astype(int)
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, train[TARGET])):
        print(f"  LightGBM fold {fold+1}/{n_folds}")
        
        # Fold-safe target encoding
        for col in te_cols:
            global_mean = train.iloc[tr_idx][TARGET].mean()
            te_map = train.iloc[tr_idx].groupby(col)[TARGET].agg(['mean', 'count'])
            smoothing = 10
            te_map['smoothed'] = (
                te_map['count'] * te_map['mean'] + smoothing * global_mean
            ) / (te_map['count'] + smoothing)
            te_map_dict = te_map['smoothed'].to_dict()
            train.loc[train.index[va_idx], f'{col}__te'] = (
                train.iloc[va_idx][col].map(te_map_dict).fillna(global_mean)
            )
            test[f'{col}__te'] += test[col].map(te_map_dict).fillna(global_mean) / n_folds
        
        X_tr = train.iloc[tr_idx][all_features]
        y_tr = train.iloc[tr_idx][TARGET]
        X_va = train.iloc[va_idx][all_features]
        y_va = train.iloc[va_idx][TARGET]
        X_te = test[all_features]
        
        model = lgb.LGBMClassifier(
            n_estimators=6000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=7,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed + fold,
            verbose=-1,
            n_jobs=-1,
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(X_te)[:, 1] / n_folds
        
        fold_auc = roc_auc_score(y_va, oof[va_idx])
        print(f"    Fold {fold+1} AUC: {fold_auc:.6f}")
    
    overall_auc = roc_auc_score(train[TARGET], oof)
    print(f"  LightGBM OOF AUC: {overall_auc:.6f}")
    return oof, test_preds


# ── Neural Network: Embedding + PLR + Transformer ────────────────────────
def train_neural_net(train, test, n_folds=11, seed=42, epochs=32):
    """Omralinov-inspired NN with:
    - Per-column embedding table on exact values (NaN → index 0)
    - PLR (Periodic-Linear Representation) — learned Fourier + linear
    - Budget-derived tokens
    - Transformer attention on feature tokens
    - Random masking during training
    - EMA (Exponential Moving Average) model weights
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  NN device: {device}")
    
    # Build vocabulary for each numeric column (exact-value lookup)
    vocab_maps = {}
    for col in NUM_COLS:
        vals = train[col].dropna().unique()
        vals = np.sort(vals)
        # Index 0 = NaN/masked, index 1+ = actual values
        vocab = {v: i + 1 for i, v in enumerate(vals)}
        vocab_maps[col] = vocab
    
    # Budget feature columns
    budget_cols = ['other_screen', 'other_frac', 'on_boundary',
                   'social_media_hours_frac', 'gaming_hours_frac', 'work_study_hours_frac']
    missing_cols = [f'{c}_missing' for c in NUM_COLS] + ['n_missing']
    
    all_nn_features = NUM_COLS + budget_cols + missing_cols
    n_numeric = len(budget_cols) + len(missing_cols)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    class EmbeddingPLRTransformer(nn.Module):
        def __init__(self, vocab_sizes, embed_dim=16, n_heads=4, n_layers=2, 
                     plr_frequencies=16, dropout=0.1):
            super().__init__()
            self.n_features = len(vocab_sizes)
            self.embed_dim = embed_dim
            
            # Per-column embedding tables (index 0 = NaN/masked)
            self.embeddings = nn.ModuleList([
                nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
                for vocab_size in vocab_sizes
            ])
            
            # PLR: learned periodic (Fourier) + linear representation
            self.plr_linear = nn.ModuleList([
                nn.Linear(1, embed_dim) for _ in range(self.n_features)
            ])
            self.plr_freq = nn.ParameterList([
                nn.Parameter(torch.randn(plr_frequencies))
                for _ in range(self.n_features)
            ])
            self.plr_phase = nn.ParameterList([
                nn.Parameter(torch.randn(plr_frequencies))
                for _ in range(self.n_features)
            ])
            self.plr_fourier_proj = nn.ModuleList([
                nn.Linear(plr_frequencies * 2, embed_dim)
                for _ in range(self.n_features)
            ])
            
            # Budget-derived numeric features → embed_dim tokens
            self.numeric_proj = nn.Linear(n_numeric, embed_dim * 2)
            
            # Feature-type embedding (distinguishes emb/plr/budget tokens)
            self.type_embed = nn.Embedding(3, embed_dim)  # 0=emb, 1=plr, 2=budget
            
            # Transformer
            n_tokens = self.n_features * 2 + 2  # emb + plr per feature + 2 budget tokens
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            
            # Head
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim * n_tokens),
                nn.Linear(embed_dim * n_tokens, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            )
        
        def forward(self, cat_idx, numeric_vals, budget_numeric, mask):
            batch_size = cat_idx.size(0)
            tokens = []
            
            for i in range(self.n_features):
                # Embedding token
                emb = self.embeddings[i](cat_idx[:, i])  # (B, embed_dim)
                
                # PLR token
                x = numeric_vals[:, i:i+1]  # (B, 1)
                lin = self.plr_linear[i](x)  # (B, embed_dim)
                
                freq = self.plr_freq[i]  # (F,)
                phase = self.plr_phase[i]  # (F,)
                fourier = torch.cat([
                    torch.sin(x * freq + phase),
                    torch.cos(x * freq + phase)
                ], dim=-1)  # (B, 2F)
                fourier = self.plr_fourier_proj[i](fourier)  # (B, embed_dim)
                
                plr = lin + fourier
                
                # Apply masking (set to zero during training)
                if mask is not None:
                    m = mask[:, i:i+1].unsqueeze(-1)  # (B, 1, 1)
                    emb = emb * m
                    plr = plr * m
                
                tokens.append(emb + self.type_embed(torch.zeros(batch_size, dtype=torch.long, device=cat_idx.device)))
                tokens.append(plr + self.type_embed(torch.ones(batch_size, dtype=torch.long, device=cat_idx.device)))
            
            # Budget tokens
            budget_proj = self.numeric_proj(budget_numeric)  # (B, embed_dim*2)
            budget_proj = budget_proj.view(batch_size, 2, -1)  # (B, 2, embed_dim)
            for t in range(2):
                tokens.append(budget_proj[:, t] + self.type_embed(
                    torch.full((batch_size,), 2, dtype=torch.long, device=cat_idx.device)))
            
            # Stack tokens: (B, n_tokens, embed_dim)
            token_stack = torch.stack(tokens, dim=1)
            
            # Transformer
            out = self.transformer(token_stack)  # (B, n_tokens, embed_dim)
            out = out.view(batch_size, -1)  # (B, n_tokens * embed_dim)
            
            return self.head(out).squeeze(-1)
    
    class EMA:
        def __init__(self, model, decay=0.999):
            self.decay = decay
            self.shadow = {k: v.clone() for k, v in model.state_dict().items()}
        
        def update(self, model):
            for k, v in model.state_dict().items():
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v
        
        def state_dict(self):
            return self.shadow
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, train[TARGET])):
        print(f"  NN fold {fold+1}/{n_folds}")
        
        # Prepare tensors for this fold
        train_df = train.iloc[tr_idx].reset_index(drop=True)
        val_df = train.iloc[va_idx].reset_index(drop=True)
        test_df = test.reset_index(drop=True)
        
        def df_to_tensors(df, is_training=False):
            # Categorical indices for embeddings
            cat_idx = np.zeros((len(df), len(NUM_COLS)), dtype=np.int64)
            numeric_vals = np.zeros((len(df), len(NUM_COLS)), dtype=np.float32)
            for i, col in enumerate(NUM_COLS):
                vocab = vocab_maps[col]
                for j, v in enumerate(df[col].values):
                    if pd.isna(v):
                        cat_idx[j, i] = 0  # NaN index
                        numeric_vals[j, i] = 0.0
                    else:
                        cat_idx[j, i] = vocab.get(v, 0)
                        numeric_vals[j, i] = float(v)
            
            # Budget + missing features
            budget_features = df[budget_cols + missing_cols].fillna(0).values.astype(np.float32)
            
            # Mask for random masking augmentation
            mask = np.ones((len(df), len(NUM_COLS)), dtype=np.float32)
            if is_training:
                mask_rand = np.random.random(mask.shape)
                mask[mask_rand < 0.1] = 0.0  # 10% random masking
            
            labels = df[TARGET].values.astype(np.float32)
            
            return (
                torch.LongTensor(cat_idx),
                torch.FloatTensor(numeric_vals),
                torch.FloatTensor(budget_features),
                torch.FloatTensor(mask),
                torch.FloatTensor(labels)
            )
        
        tr_tensors = df_to_tensors(train_df, is_training=True)
        va_tensors = df_to_tensors(val_df)
        te_tensors = df_to_tensors(test_df)
        
        train_ds = TensorDataset(*tr_tensors)
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
        
        # Model
        vocab_sizes = [len(vocab_maps[col]) for col in NUM_COLS]
        model = EmbeddingPLRTransformer(
            vocab_sizes=vocab_sizes,
            embed_dim=16,
            n_heads=4,
            n_layers=2,
            plr_frequencies=16,
            dropout=0.1
        ).to(device)
        
        ema = EMA(model, decay=0.999)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader)
        )
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        best_val_auc = 0
        best_epoch = 0
        for epoch in range(epochs):
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
                optimizer.step()
                scheduler.step()
                ema.update(model)
            
            # Validation with EMA weights
            if (epoch + 1) % 4 == 0 or epoch == epochs - 1:
                # Temporarily load EMA weights
                orig_state = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema.state_dict())
                model.eval()
                
                with torch.no_grad():
                    va_logits = []
                    for i in range(0, len(val_df), 2048):
                        batch_tensors = [x[i:i+2048].to(device) for x in va_tensors[:4]]
                        logits = model(*batch_tensors)
                        va_logits.append(logits.cpu().numpy())
                    va_preds = np.concatenate(va_logits)
                    va_probs = 1 / (1 + np.exp(-va_preds))  # sigmoid
                    va_auc = roc_auc_score(val_df[TARGET], va_probs)
                
                if va_auc > best_val_auc:
                    best_val_auc = va_auc
                    best_epoch = epoch + 1
                    # Save best EMA predictions for this fold
                    with torch.no_grad():
                        te_logits = []
                        for i in range(0, len(test_df), 2048):
                            batch_tensors = [x[i:i+2048].to(device) for x in te_tensors[:4]]
                            logits = model(*batch_tensors)
                            te_logits.append(logits.cpu().numpy())
                        best_test_preds = 1 / (1 + np.exp(-np.concatenate(te_logits)))
                    best_oof_preds = va_probs.copy()
                
                # Restore original weights for continued training
                model.load_state_dict(orig_state)
                
                print(f"    Epoch {epoch+1}/{epochs} — Val AUC: {va_auc:.6f} (best: {best_val_auc:.6f})")
        
        oof[va_idx] = best_oof_preds
        test_preds += best_test_preds / n_folds
        
        print(f"    Fold {fold+1} best AUC: {best_val_auc:.6f} (epoch {best_epoch})")
    
    overall_auc = roc_auc_score(train[TARGET], oof)
    print(f"  NN OOF AUC: {overall_auc:.6f}")
    return oof, test_preds


# ── Rank-normalized blend ─────────────────────────────────────────────────
def rank_normalize(preds):
    return rankdata(preds) / len(preds)


def blend_models(oof_dict, test_dict, weights=None):
    """Rank-normalized weighted blend."""
    if weights is None:
        weights = {'nn': 0.49, 'catboost': 0.31, 'lightgbm': 0.21}
    
    oof_blend = sum(
        weights[k] * rank_normalize(v) for k, v in oof_dict.items()
    )
    test_blend = sum(
        weights[k] * rank_normalize(v) for k, v in test_dict.items()
    )
    return oof_blend, test_blend


# ── Main pipeline ─────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Sprint 5 — Omralinov-inspired pipeline")
    print("=" * 70)
    
    # Load data
    train, test, sub = load_data()
    print(f"Train: {train.shape}, Test: {test.shape}")
    
    # Feature engineering
    train, test = prepare_data(train, test)
    
    n_folds = 11
    seed = 42
    
    # Train all 3 models
    print("\n── CatBoost ──")
    cb_oof, cb_test = train_catboost(train, test, n_folds=n_folds, seed=seed)
    
    print("\n── LightGBM ──")
    lgb_oof, lgb_test = train_lightgbm(train, test, n_folds=n_folds, seed=seed)
    
    print("\n── Neural Net (Embedding + PLR + Transformer) ──")
    nn_oof, nn_test = train_neural_net(train, test, n_folds=n_folds, seed=seed, epochs=32)
    
    # Blend
    print("\n── Blending ──")
    oof_dict = {'nn': nn_oof, 'catboost': cb_oof, 'lightgbm': lgb_oof}
    test_dict = {'nn': nn_test, 'catboost': cb_test, 'lightgbm': lgb_test}
    
    # Default Omralinov weights
    oof_blend, test_blend = blend_models(oof_dict, test_dict, 
                                          weights={'nn': 0.49, 'catboost': 0.31, 'lightgbm': 0.21})
    blend_auc = roc_auc_score(train[TARGET], oof_blend)
    print(f"Blend OOF AUC (0.49/0.31/0.21): {blend_auc:.6f}")
    
    # Save OOF predictions
    np.save(OUT / "oof_nn.npy", nn_oof)
    np.save(OUT / "oof_catboost.npy", cb_oof)
    np.save(OUT / "oof_lightgbm.npy", lgb_oof)
    np.save(OUT / "oof_blend.npy", oof_blend)
    
    # Create submission
    sub['addicted_label'] = test_blend
    sub_path = OUT / "submission_sprint5.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission saved to {sub_path}")
    
    # Save metrics
    metrics = {
        'nn_oof_auc': float(roc_auc_score(train[TARGET], nn_oof)),
        'catboost_oof_auc': float(roc_auc_score(train[TARGET], cb_oof)),
        'lightgbm_oof_auc': float(roc_auc_score(train[TARGET], lgb_oof)),
        'blend_oof_auc': float(blend_auc),
        'n_folds': n_folds,
        'blend_weights': {'nn': 0.49, 'catboost': 0.31, 'lightgbm': 0.21},
    }
    with open(OUT / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    return metrics


if __name__ == "__main__":
    main()
