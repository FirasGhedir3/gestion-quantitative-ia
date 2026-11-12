"""Indicateurs de performance."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def drawdown(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def performance_summary(returns: pd.Series, rf: float = 0.0, turnover: pd.Series | None = None) -> dict:
    r = returns.dropna()
    n = len(r)
    if n < 2:
        return {}
    years = n / TRADING_DAYS
    total = float((1.0 + r).prod() - 1.0)
    ann_ret = float((1.0 + total) ** (1.0 / years) - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    excess = float(r.mean() * TRADING_DAYS - rf)
    sharpe = excess / vol if vol > 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(r - rf / TRADING_DAYS, 0.0) ** 2)) * np.sqrt(TRADING_DAYS)
    sortino = excess / downside if downside > 0 else np.nan
    mdd = float(drawdown(r).min())
    return {
        "Rendement total": total,
        "Rendement annualisé": ann_ret,
        "Volatilité": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max drawdown": mdd,
        "Calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
        "Rotation annuelle": float(turnover.sum() / years) if turnover is not None else np.nan,
    }
