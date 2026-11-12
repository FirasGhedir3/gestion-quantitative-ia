"""Graphiques Plotly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .features import FEATURE_LABELS
from .metrics import drawdown

TEMPLATE = "plotly_white"
STRATEGY_COLORS = {
    "IA continue": "#1d4ed8",
    "IA figée": "#7c3aed",
    "Max Sharpe historique": "#d97706",
    "Équipondéré": "#6b7280",
}
ASSET_COLORS = px.colors.qualitative.Light24


def _layout(fig: go.Figure, title: str, height: int = 420, **kwargs) -> go.Figure:
    options = dict(
        template=TEMPLATE, title=dict(text=title, x=0.0), height=height,
        margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
        legend=dict(orientation="h", y=-0.18),
    )
    options.update(kwargs)
    fig.update_layout(**options)
    return fig


def equity_fig(sims: dict, log_y: bool = False) -> go.Figure:
    fig = go.Figure()
    for name, df in sims.items():
        equity = (1.0 + df["ret"]).cumprod()
        fig.add_scatter(
            x=equity.index, y=equity.values, name=name, mode="lines",
            line=dict(color=STRATEGY_COLORS.get(name), width=3 if name == "IA continue" else 1.6),
        )
    fig.update_yaxes(type="log" if log_y else "linear")
    return _layout(fig, "Valeur du portefeuille (base 1, nette de coûts)")


def drawdown_fig(sims: dict) -> go.Figure:
    fig = go.Figure()
    for name, df in sims.items():
        dd = drawdown(df["ret"]) * 100
        fig.add_scatter(x=dd.index, y=dd.values, name=name, mode="lines",
                        line=dict(color=STRATEGY_COLORS.get(name), width=1.6))
    fig.update_yaxes(ticksuffix=" %")
    return _layout(fig, "Drawdown", height=320)


def weights_fig(weights: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(weights.columns):
        fig.add_scatter(
            x=weights.index, y=weights[col] * 100, name=col, stackgroup="w", mode="lines",
            line=dict(width=0.5, color=ASSET_COLORS[i % len(ASSET_COLORS)]),
        )
    fig.update_yaxes(ticksuffix=" %", range=[0, 100])
    return _layout(fig, "Allocation de la stratégie IA au fil du temps", height=380)


def coef_fig(coefs: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in coefs.columns:
        fig.add_scatter(x=coefs.index, y=coefs[col], name=FEATURE_LABELS.get(col, col), mode="lines")
    return _layout(fig, "Coefficients du modèle au fil de l'apprentissage", height=380)


def ic_fig(ic_by_name: dict, window: int = 26) -> go.Figure:
    fig = go.Figure()
    for name, ic in ic_by_name.items():
        rolling = ic.rolling(window, min_periods=max(5, window // 3)).mean()
        fig.add_scatter(x=rolling.index, y=rolling.values, name=name, mode="lines",
                        line=dict(color=STRATEGY_COLORS.get(name)))
    fig.add_hline(y=0, line_dash="dot", line_color="#9ca3af")
    return _layout(fig, f"Corrélation de rang (IC) prédiction / réalisé — moyenne glissante {window} périodes",
                   height=340)


def prediction_scatter_fig(alpha: pd.DataFrame, realized: pd.DataFrame) -> go.Figure:
    a = alpha.values.ravel()
    r = realized.values.ravel()
    mask = np.isfinite(a) & np.isfinite(r)
    fig = go.Figure(go.Scattergl(x=a[mask] * 100, y=r[mask] * 100, mode="markers",
                                 marker=dict(size=4, opacity=0.35, color="#1d4ed8")))
    fig.update_xaxes(title="Alpha prédit (annualisé, %)")
    fig.update_yaxes(title="Rendement réalisé relatif (%)")
    return _layout(fig, "Prédit vs réalisé (IA continue)", height=340, hovermode="closest")


def allocation_bar_fig(df: pd.DataFrame) -> go.Figure:
    """df : colonnes 'Optimal' et 'Précédent' indexées par actif."""
    df = df.sort_values("Optimal")
    fig = go.Figure()
    fig.add_bar(y=df.index, x=df["Précédent"] * 100, name="Dernier rééquilibrage", orientation="h",
                marker_color="#cbd5e1")
    fig.add_bar(y=df.index, x=df["Optimal"] * 100, name="Allocation recommandée", orientation="h",
                marker_color=STRATEGY_COLORS["IA continue"])
    fig.update_xaxes(ticksuffix=" %")
    return _layout(fig, "Poids par actif", height=max(320, 28 * len(df) + 120), barmode="group",
                   hovermode="closest")


def frontier_fig(rand_vols, rand_rets, rand_sharpe, front_vols, front_rets, points: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=rand_vols * 100, y=rand_rets * 100, mode="markers", name="Portefeuilles aléatoires",
        marker=dict(size=4, color=rand_sharpe, colorscale="Blues", showscale=True,
                    colorbar=dict(title="Sharpe")),
    )
    fig.add_scatter(x=front_vols * 100, y=front_rets * 100, mode="lines", name="Frontière efficiente",
                    line=dict(color="#111827", width=2))
    symbols = {"Sharpe max": "star", "Variance minimale": "diamond", "Équipondéré": "square"}
    for name, (vol, ret) in points.items():
        fig.add_scatter(x=[vol * 100], y=[ret * 100], mode="markers", name=name,
                        marker=dict(size=14, symbol=symbols.get(name, "circle"),
                                    line=dict(width=1, color="white")))
    fig.update_xaxes(title="Volatilité annualisée (%)")
    fig.update_yaxes(title="Rendement espéré annualisé (%)")
    return _layout(fig, "Frontière efficiente (μ prédit par l'IA, covariance Ledoit-Wolf)", height=480,
                   hovermode="closest")
