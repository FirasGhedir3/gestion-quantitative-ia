"""Modèle de prédiction des rendements en apprentissage continu.

On utilise les moindres carrés récursifs avec facteur d'oubli (RLS, « Recursive Least Squares »).
C'est de l'apprentissage en ligne exact : chaque nouvelle observation met à jour les coefficients
sans réentraîner sur tout l'historique, et le facteur d'oubli λ < 1 donne moins de poids aux
observations anciennes, ce qui permet de s'adapter aux changements de régime.

    w_t = w_{t-1} + K_t (y_t - X_t w_{t-1})
    P_t = (P_{t-1}/λ) - K_t X_t (P_{t-1}/λ)
"""
from __future__ import annotations

import numpy as np


class OnlineRLS:
    def __init__(self, n_features: int, forget: float = 0.995, ridge: float = 1.0):
        if not 0.0 < forget <= 1.0:
            raise ValueError("Le facteur d'oubli doit être dans ]0, 1].")
        if ridge <= 0:
            raise ValueError("ridge doit être > 0.")
        self.n_features = n_features
        self.forget = forget
        self.ridge = ridge
        self.w = np.zeros(n_features)
        self.P = np.eye(n_features) / ridge
        self.n_obs = 0

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.w

    def update(self, X, y) -> None:
        """Mise à jour par lot : X de forme (n, F), y de forme (n,)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X et y doivent avoir le même nombre de lignes.")

        P = self.P / self.forget
        PXt = P @ X.T                                   # (F, n)
        S = np.eye(X.shape[0]) + X @ PXt                # (n, n), symétrique
        K = np.linalg.solve(S, PXt.T).T                 # PXt @ S^{-1}
        self.w = self.w + K @ (y - X @ self.w)
        P = P - K @ PXt.T
        P = 0.5 * (P + P.T)

        # Garde-fou : l'incertitude ne dépasse jamais celle du prior (évite l'explosion de P).
        max_trace = self.n_features / self.ridge
        trace = np.trace(P)
        if trace > max_trace:
            P *= max_trace / trace
        self.P = P
        self.n_obs += len(y)
