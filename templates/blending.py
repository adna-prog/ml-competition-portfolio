"""Blend pondéré des OOF : trouve les poids optimaux par recherche de grille.
Simple, robuste, complémentaire du stacking.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import roc_auc_score


def blend_2(oof_a, oof_b, y, step=0.01):
    """Meilleur poids pour blend de 2 OOF."""
    best = (0, 0)
    for w in np.arange(0, 1.001, step):
        auc = roc_auc_score(y, w*oof_a + (1-w)*oof_b)
        if auc > best[0]:
            best = (auc, w)
    return best  # (auc, w_a)


def blend_3(oof_a, oof_b, oof_c, y, step=0.05):
    """Meilleur triplet de poids pour blend de 3 OOF (grille)."""
    best = (0, None)
    for wa in np.arange(0, 1.001, step):
        for wb in np.arange(0, 1.001-wa, step):
            wc = 1 - wa - wb
            auc = roc_auc_score(y, wa*oof_a + wb*oof_b + wc*oof_c)
            if auc > best[0]:
                best = (auc, (wa, wb, wc))
    return best  # (auc, (wa, wb, wc))


def make_submission(test_ids, preds, out='submission.csv'):
    import pandas as pd
    sub = pd.DataFrame({'id': test_ids, 'addicted_label': preds})
    sub.to_csv(out, index=False)
    print(f"submission -> {out} ({len(sub)} lignes)")
