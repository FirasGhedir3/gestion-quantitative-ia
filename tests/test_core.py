import numpy as np
import pandas as pd
import pytest

from quant import backtest as bt
from quant.data import simulate_prices
from quant.features import build_features
from quant.model import OnlineRLS
from quant.optimizer import max_sharpe, min_variance, portfolio_stats

TICKERS = list("ABCDEFGH")


@pytest.fixture(scope="module")
def prices():
    return simulate_prices(TICKERS, "2019-01-01", "2024-12-31", seed=1)


def test_rls_recovers_coefficients():
    rng = np.random.default_rng(0)
    w_true = np.array([0.5, -1.0, 2.0])
    model = OnlineRLS(3, forget=1.0, ridge=1e-3)
    for _ in range(300):
        X = rng.normal(size=(8, 3))
        model.update(X, X @ w_true + rng.normal(0, 0.01, 8))
    assert np.allclose(model.w, w_true, atol=0.05)


def test_rls_forgetting_adapts_to_regime_change():
    rng = np.random.default_rng(1)
    adaptive = OnlineRLS(2, forget=0.97, ridge=1.0)
    static = OnlineRLS(2, forget=1.0, ridge=1.0)
    for step in range(600):
        w_true = np.array([1.0, 0.0]) if step < 300 else np.array([-1.0, 0.5])
        X = rng.normal(size=(6, 2))
        y = X @ w_true + rng.normal(0, 0.05, 6)
        adaptive.update(X, y)
        static.update(X, y)
    target = np.array([-1.0, 0.5])
    assert np.linalg.norm(adaptive.w - target) < 0.1
    assert np.linalg.norm(static.w - target) > np.linalg.norm(adaptive.w - target)


def test_features_have_no_lookahead(prices):
    cut = 700
    x_full, _ = build_features(prices)
    altered = prices.copy()
    altered.iloc[cut + 1:] *= 3.0  # on bouleverse le futur
    x_alt, _ = build_features(altered)
    assert np.allclose(x_full[: cut + 1], x_alt[: cut + 1])


def test_max_sharpe_constraints_and_quality():
    rng = np.random.default_rng(3)
    n = 6
    a = rng.normal(size=(n, n))
    cov = a @ a.T / n * 0.05 + np.eye(n) * 0.02
    mu = rng.uniform(0.02, 0.15, n)
    w = max_sharpe(mu, cov, rf=0.01, w_max=0.4)
    assert w.sum() == pytest.approx(1.0)
    assert w.min() >= -1e-9 and w.max() <= 0.4 + 1e-6
    sharpe = portfolio_stats(w, mu, cov, 0.01)[2]
    equal = portfolio_stats(np.full(n, 1 / n), mu, cov, 0.01)[2]
    assert sharpe >= equal - 1e-9


def test_max_sharpe_falls_back_to_min_variance_when_all_excess_negative():
    cov = np.diag([0.04, 0.09, 0.01])
    w = max_sharpe(np.array([-0.05, -0.02, -0.01]), cov, rf=0.02, w_max=1.0)
    assert np.allclose(w, min_variance(cov, 1.0), atol=1e-6)


def test_backtest_pipeline_runs_and_is_consistent(prices):
    cfg = bt.Config(horizon=10, warmup=126)
    study = bt.run_study(prices, cfg)
    for name, w in study.weights.items():
        assert np.allclose(w.sum(axis=1), 1.0), name
        assert (w.values <= cfg.w_max + 1e-6).all(), name
    assert np.isfinite(study.summary["Sharpe"]).all()
    assert study.ai.weights.index.equals(study.frozen.weights.index)
    assert study.ai.coefs.iloc[-1].abs().sum() > 0

    alloc = bt.current_allocation(prices, study.ai.model, cfg)
    assert alloc.weights.sum() == pytest.approx(1.0)
