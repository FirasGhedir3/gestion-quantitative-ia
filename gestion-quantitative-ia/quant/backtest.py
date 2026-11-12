"""Backtest walk-forward : apprentissage continu + allocation par maximisation du Sharpe.

Chronologie à la date t (aucune information future n'est utilisée) :
  1. Le rendement réalisé entre t-h et t est maintenant connu : on met à jour le modèle avec
     les variables observées en t-h (apprentissage continu, quotidien, labels qui se chevauchent).
  2. Si t est une date de rééquilibrage : prédiction de l'alpha de chaque actif sur h jours,
     construction de μ (rendement de marché de base + alpha de Grinold),
     covariance de Ledoit-Wolf sur la fenêtre glissante, puis maximisation du Sharpe.
  3. Les poids sont appliqués à partir du jour suivant ; coûts de transaction déduits à t.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .features import FEATURE_NAMES, build_features
from .metrics import TRADING_DAYS, performance_summary
from .model import OnlineRLS
from .optimizer import max_sharpe, portfolio_stats

ALPHA_CAP = 0.5  # plafond de l'alpha annualisé prédit (robustesse)


@dataclass(frozen=True)
class Config:
    horizon: int = 5              # jours de bourse entre deux rééquilibrages (= horizon de prédiction)
    lookback: int = 252           # fenêtre de la matrice de covariance
    warmup: int = 126             # jours d'apprentissage avant le premier rééquilibrage
    forget: float = 0.995         # facteur d'oubli du RLS
    ridge: float = 1.0            # régularisation initiale du RLS
    signal_strength: float = 0.10  # IC supposé (Grinold) : amplitude de l'alpha = IC x volatilité x score
    base_shrink: float = 0.5      # part du rendement de marché historique conservée
    rf: float = 0.02              # taux sans risque annuel
    w_max: float = 0.30           # poids maximal par actif
    tc_bps: float = 5.0           # coût de transaction (points de base, par unité échangée)
    turnover_penalty: float = 0.30


@dataclass
class WalkForwardResult:
    cfg: Config
    weights: pd.DataFrame
    alpha: pd.DataFrame           # alpha annualisé prédit, par date de rééquilibrage
    realized: pd.DataFrame        # rendement réalisé sur h jours (centré), NaN pour les dernières dates
    coefs: pd.DataFrame           # coefficients du modèle à chaque rééquilibrage
    ic: pd.Series                 # corrélation de rang (alpha prédit vs réalisé)
    model: OnlineRLS
    start_idx: int


@dataclass
class Allocation:
    weights: pd.Series
    mu: pd.Series
    alpha: pd.Series
    cov: pd.DataFrame
    exp_return: float
    exp_vol: float
    exp_sharpe: float
    as_of: pd.Timestamp


@dataclass
class Study:
    cfg: Config
    ai: WalkForwardResult
    frozen: WalkForwardResult
    sims: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    summary: pd.DataFrame | None = None


# ---------------------------------------------------------------- briques communes

def _covariance(returns_window: np.ndarray) -> np.ndarray:
    window = np.nan_to_num(returns_window)
    return LedoitWolf().fit(window).covariance_ * TRADING_DAYS


def _base_returns(rets: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Rendement de marché annualisé (EWMA causale, réduit par shrinkage)."""
    market = rets.mean(axis=1).ewm(halflife=126, min_periods=20).mean()
    return (market * TRADING_DAYS * cfg.base_shrink).fillna(0.0).values


def _signal_to_mu(model: OnlineRLS, x_t: np.ndarray, base_t: float, cov: np.ndarray, cfg: Config):
    """Alpha = IC supposé x volatilité x score (règle de Grinold).

    Le modèle fournit le *classement* des actifs (score = prédiction standardisée en coupe
    transversale) ; l'amplitude de l'alpha est bornée par l'IC supposé, ce qui évite qu'un
    signal bruité provoque des retournements de portefeuille.
    """
    pred = model.predict(x_t)
    std = pred.std()
    score = (pred - pred.mean()) / std if std > 1e-12 else np.zeros_like(pred)
    alpha_ann = np.clip(cfg.signal_strength * np.sqrt(np.diag(cov)) * score, -ALPHA_CAP, ALPHA_CAP)
    return base_t + alpha_ann, alpha_ann


def _check_length(t_len: int, first_valid: int, cfg: Config) -> int:
    if cfg.warmup <= cfg.horizon:
        raise ValueError("warmup doit être supérieur à l'horizon.")
    start_idx = first_valid + cfg.warmup
    if t_len < start_idx + cfg.horizon + 60:
        raise ValueError(
            "Historique trop court : augmentez la période ou réduisez le warm-up."
        )
    return start_idx


# ---------------------------------------------------------------- walk-forward IA

def run_walkforward(prices: pd.DataFrame, cfg: Config, adaptive: bool = True) -> WalkForwardResult:
    """adaptive=False : le modèle est figé à la fin du warm-up (référence sans apprentissage continu)."""
    x, valid = build_features(prices)
    if not valid.any():
        raise ValueError("Historique insuffisant pour calculer les variables explicatives.")
    p = prices.values
    t_len, n = p.shape
    rets = prices.pct_change()
    r = rets.values
    base = _base_returns(rets, cfg)
    h = cfg.horizon
    first_valid = int(np.argmax(valid))
    start_idx = _check_length(t_len, first_valid, cfg)

    model = OnlineRLS(len(FEATURE_NAMES), forget=cfg.forget, ridge=cfg.ridge)
    w_prev = np.full(n, 1.0 / n)
    dates, weights, alphas, coefs = [], [], [], []

    for t in range(first_valid + h, t_len):
        s = t - h
        if adaptive or t <= start_idx:
            y = p[t] / p[s] - 1.0
            model.update(x[s], y - y.mean())

        if t >= start_idx and (t - start_idx) % h == 0:
            cov = _covariance(r[max(1, t - cfg.lookback + 1): t + 1])
            mu, alpha_ann = _signal_to_mu(model, x[t], base[t], cov, cfg)
            w = max_sharpe(mu, cov, cfg.rf, cfg.w_max, w_prev, cfg.turnover_penalty)
            dates.append(prices.index[t])
            weights.append(w)
            alphas.append(alpha_ann)
            coefs.append(model.w.copy())
            w_prev = w

    idx = pd.DatetimeIndex(dates)
    cols = prices.columns
    weights_df = pd.DataFrame(weights, index=idx, columns=cols)
    alpha_df = pd.DataFrame(alphas, index=idx, columns=cols)
    coef_df = pd.DataFrame(coefs, index=idx, columns=FEATURE_NAMES)

    realized = np.full((len(idx), n), np.nan)
    for k, d in enumerate(idx):
        i = prices.index.get_loc(d)
        if i + h < t_len:
            fwd = p[i + h] / p[i] - 1.0
            realized[k] = fwd - fwd.mean()
    realized_df = pd.DataFrame(realized, index=idx, columns=cols)

    return WalkForwardResult(
        cfg=cfg, weights=weights_df, alpha=alpha_df, realized=realized_df, coefs=coef_df,
        ic=_information_coefficient(alpha_df, realized_df), model=model, start_idx=start_idx,
    )


def _information_coefficient(alpha: pd.DataFrame, realized: pd.DataFrame) -> pd.Series:
    out = {}
    for d in alpha.index:
        a, r = alpha.loc[d], realized.loc[d]
        if r.isna().any():
            continue
        out[d] = a.rank().corr(r.rank())
    return pd.Series(out, name="IC", dtype=float)


# ---------------------------------------------------------------- allocation courante (temps réel)

def current_allocation(prices: pd.DataFrame, model: OnlineRLS, cfg: Config, w_prev=None) -> Allocation:
    """Allocation recommandée à partir de la dernière ligne de prix (clôture ou cotation live)."""
    x, valid = build_features(prices)
    if not valid[-1]:
        raise ValueError("Variables explicatives indisponibles pour la dernière date.")
    rets = prices.pct_change()
    base = _base_returns(rets, cfg)
    cov = _covariance(rets.values[max(1, len(prices) - cfg.lookback):])
    mu, alpha_ann = _signal_to_mu(model, x[-1], base[-1], cov, cfg)
    w = max_sharpe(mu, cov, cfg.rf, cfg.w_max, w_prev, cfg.turnover_penalty)
    ret, vol, sharpe = portfolio_stats(w, mu, cov, cfg.rf)
    cols = prices.columns
    return Allocation(
        weights=pd.Series(w, index=cols), mu=pd.Series(mu, index=cols),
        alpha=pd.Series(alpha_ann, index=cols), cov=pd.DataFrame(cov, index=cols, columns=cols),
        exp_return=ret, exp_vol=vol, exp_sharpe=sharpe, as_of=prices.index[-1],
    )


# ---------------------------------------------------------------- références et simulation

def historical_sharpe_weights(prices: pd.DataFrame, dates: pd.DatetimeIndex, cfg: Config) -> pd.DataFrame:
    """Référence classique : Sharpe maximal avec moyennes et covariances historiques."""
    r = prices.pct_change().values
    n = prices.shape[1]
    w_prev = np.full(n, 1.0 / n)
    rows = []
    for d in dates:
        i = prices.index.get_loc(d)
        window = np.nan_to_num(r[max(1, i - cfg.lookback + 1): i + 1])
        mu = window.mean(axis=0) * TRADING_DAYS
        cov = _covariance(window)
        w_prev = max_sharpe(mu, cov, cfg.rf, cfg.w_max, w_prev, cfg.turnover_penalty)
        rows.append(w_prev)
    return pd.DataFrame(rows, index=dates, columns=prices.columns)


def simulate_portfolio(prices: pd.DataFrame, weights: pd.DataFrame, tc_bps: float) -> pd.DataFrame:
    """Rendements journaliers nets de coûts, avec dérive des poids entre rééquilibrages."""
    r = prices.pct_change().values
    idx = prices.index
    pos = {d: i for i, d in enumerate(idx)}
    reb = {pos[d]: weights.loc[d].values for d in weights.index}
    start = min(reb)
    w = reb[start].copy()
    rows, dates = [], []
    for t in range(start + 1, len(idx)):
        rt = np.nan_to_num(r[t])
        port = float(w @ rt)
        w = w * (1.0 + rt) / (1.0 + port)
        turnover = 0.0
        if t in reb:
            new_w = reb[t]
            turnover = float(np.abs(new_w - w).sum())
            port -= turnover * tc_bps / 1e4
            w = new_w.copy()
        rows.append((port, turnover))
        dates.append(idx[t])
    return pd.DataFrame(rows, index=dates, columns=["ret", "turnover"])


def run_study(prices: pd.DataFrame, cfg: Config) -> Study:
    ai = run_walkforward(prices, cfg, adaptive=True)
    frozen = run_walkforward(prices, cfg, adaptive=False)
    dates = ai.weights.index
    n = prices.shape[1]

    weights = {
        "IA continue": ai.weights,
        "IA figée": frozen.weights,
        "Max Sharpe historique": historical_sharpe_weights(prices, dates, cfg),
        "Équipondéré": pd.DataFrame(1.0 / n, index=dates, columns=prices.columns),
    }
    sims = {name: simulate_portfolio(prices, w, cfg.tc_bps) for name, w in weights.items()}
    summary = pd.DataFrame(
        {name: performance_summary(s["ret"], cfg.rf, s["turnover"]) for name, s in sims.items()}
    ).T
    return Study(cfg=cfg, ai=ai, frozen=frozen, sims=sims, weights=weights, summary=summary)
