"""Construction des variables explicatives (strictement causales : aucune fuite du futur)."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "mom_5", "mom_21", "mom_63", "mom_126_skip21",
    "vol_21", "dist_ma50", "drawdown_126",
]

FEATURE_LABELS = {
    "mom_5": "Momentum 5j",
    "mom_21": "Momentum 1 mois",
    "mom_63": "Momentum 3 mois",
    "mom_126_skip21": "Momentum 6 mois (hors dernier mois)",
    "vol_21": "Volatilité 1 mois",
    "dist_ma50": "Écart à la moyenne mobile 50j",
    "drawdown_126": "Drawdown sur 6 mois",
}


def build_features(prices: pd.DataFrame):
    """Retourne (X, valid) avec X de forme (T, N, F) et valid de forme (T,).

    Les variables sont standardisées en coupe transversale à chaque date (z-score, tronqué à ±3) :
    le modèle apprend donc *quels actifs surperforment les autres*, pas le niveau du marché.
    Les lignes non valides (début d'historique) sont mises à 0 et marquées valid=False.
    """
    rets = np.log(prices).diff()
    feats = {
        "mom_5": rets.rolling(5).sum(),
        "mom_21": rets.rolling(21).sum(),
        "mom_63": rets.rolling(63).sum(),
        "mom_126_skip21": rets.shift(21).rolling(105).sum(),
        "vol_21": rets.rolling(21).std(),
        "dist_ma50": np.log(prices / prices.rolling(50).mean()),
        "drawdown_126": prices / prices.rolling(126).max() - 1.0,
    }
    x = np.stack([feats[name].values for name in FEATURE_NAMES], axis=-1)  # (T, N, F)
    valid = np.isfinite(x).all(axis=(1, 2))

    z = np.zeros_like(x)
    if valid.any():
        xv = x[valid]
        mean = xv.mean(axis=1, keepdims=True)
        std = xv.std(axis=1, keepdims=True)
        z[valid] = (xv - mean) / np.where(std < 1e-12, 1.0, std)
    z = np.clip(z, -3.0, 3.0)
    z[~valid] = 0.0
    return z, valid
