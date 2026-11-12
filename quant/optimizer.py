"""Optimisation de portefeuille : maximisation du ratio de Sharpe sous contraintes."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def portfolio_stats(w, mu, cov, rf: float = 0.0):
    """(rendement, volatilité, Sharpe) annualisés."""
    w = np.asarray(w, dtype=float)
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 1e-16)))
    return ret, vol, (ret - rf) / vol


def _bounds(n: int, w_max: float):
    w_max = max(float(w_max), 1.0 / n + 1e-6)  # garantit que la contrainte est réalisable
    return [(0.0, w_max)] * n, w_max


def _start(n: int, w_prev, w_max: float) -> np.ndarray:
    if w_prev is None:
        return np.full(n, 1.0 / n)
    x0 = np.clip(np.asarray(w_prev, dtype=float), 0.0, w_max)
    total = x0.sum()
    return x0 / total if total > 0 else np.full(n, 1.0 / n)


def _clean(w: np.ndarray, w_max: float) -> np.ndarray:
    w = np.clip(w, 0.0, w_max)
    total = w.sum()
    return w / total if total > 0 else np.full(len(w), 1.0 / len(w))


def min_variance(cov, w_max: float = 1.0, w_prev=None) -> np.ndarray:
    n = cov.shape[0]
    bounds, w_max = _bounds(n, w_max)
    x0 = _start(n, w_prev, w_max)
    res = minimize(
        lambda w: float(w @ cov @ w), x0, jac=lambda w: 2.0 * cov @ w, method="SLSQP",
        bounds=bounds, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 200, "ftol": 1e-12},
    )
    return _clean(res.x, w_max) if np.all(np.isfinite(res.x)) else np.full(n, 1.0 / n)


def max_sharpe(mu, cov, rf: float = 0.0, w_max: float = 0.35, w_prev=None,
               turnover_penalty: float = 0.0) -> np.ndarray:
    """Poids long-only, pleinement investis, qui maximisent (μᵀw − rf) / √(wᵀΣw).

    - w_max : poids maximal par actif (diversification).
    - turnover_penalty : pénalité sur la rotation ‖w − w_prev‖₁ (lissée), pour éviter les
      allocations instables d'une date à l'autre.
    Si aucun actif n'a d'espérance supérieure au taux sans risque, le ratio de Sharpe n'est
    plus un critère pertinent : on bascule sur le portefeuille de variance minimale.
    """
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    bounds, w_max = _bounds(n, w_max)

    if np.max(mu) - rf <= 1e-6:
        return min_variance(cov, w_max, w_prev)

    x0 = _start(n, w_prev, w_max)
    anchor = x0 if w_prev is not None else None

    def objective(w):
        ret = w @ mu - rf
        vol = np.sqrt(max(w @ cov @ w, 1e-16))
        value = -ret / vol
        if anchor is not None and turnover_penalty > 0:
            value += turnover_penalty * np.sum(np.sqrt((w - anchor) ** 2 + 1e-6))
        return value

    res = minimize(
        objective, x0, method="SLSQP", bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not np.all(np.isfinite(res.x)):
        return min_variance(cov, w_max, w_prev)
    return _clean(res.x, w_max)


def random_portfolios(mu, cov, rf: float, n: int = 3000, w_max: float = 1.0, seed: int = 0):
    """Portefeuilles aléatoires (Dirichlet) respectant le plafond par actif."""
    rng = np.random.default_rng(seed)
    k = len(mu)
    _, w_max = _bounds(k, w_max)
    w = rng.dirichlet(np.ones(k), size=n * 3)
    w = w[(w <= w_max).all(axis=1)][:n]
    if len(w) == 0:
        w = np.full((1, k), 1.0 / k)
    rets = w @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    return vols, rets, (rets - rf) / vols


def efficient_frontier(mu, cov, w_max: float = 1.0, n_points: int = 30):
    """Frontière efficiente long-only : (volatilités, rendements)."""
    k = len(mu)
    bounds, w_max = _bounds(k, w_max)
    w_mv = min_variance(cov, w_max)
    lo = float(w_mv @ mu)
    hi = float(min(np.max(mu), _max_return(mu, w_max)))
    targets = np.linspace(lo, max(hi, lo + 1e-6), n_points)
    vols, rets = [], []
    x0 = w_mv
    for target in targets:
        res = minimize(
            lambda w: float(w @ cov @ w), x0, jac=lambda w: 2.0 * cov @ w, method="SLSQP",
            bounds=bounds,
            constraints=[
                {"type": "eq", "fun": lambda w: w.sum() - 1.0},
                {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t},
            ],
            options={"maxiter": 200, "ftol": 1e-12},
        )
        if res.success:
            x0 = res.x
            vols.append(float(np.sqrt(max(res.x @ cov @ res.x, 0.0))))
            rets.append(float(res.x @ mu))
    return np.array(vols), np.array(rets)


def _max_return(mu, w_max: float) -> float:
    """Rendement maximal atteignable sous le plafond par actif (remplissage glouton)."""
    remaining, total = 1.0, 0.0
    for m in np.sort(mu)[::-1]:
        take = min(w_max, remaining)
        total += take * m
        remaining -= take
        if remaining <= 1e-12:
            break
    return total
