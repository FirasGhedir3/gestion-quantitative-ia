"""Dashboard de gestion quantitative IA — optimisation de portefeuille en temps réel (Streamlit)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from quant import backtest as bt
from quant import data as qdata
from quant import optimizer as opt
from quant import plots
from quant.features import FEATURE_LABELS

st.set_page_config(page_title="Gestion quantitative IA", layout="wide")

st.title("Gestion quantitative IA")
st.caption(
    "Prédiction des rendements par apprentissage continu (moindres carrés récursifs) "
    "et allocation par maximisation du ratio de Sharpe."
)

# ------------------------------------------------------------------ paramètres
with st.sidebar:
    st.header("Paramètres")
    with st.form("params"):
        source = st.radio("Source des données", [qdata.YAHOO, qdata.SIMULATED], index=0,
                          help="Yahoo Finance nécessite un accès Internet. En cas d'échec, "
                               "l'application bascule sur des données simulées.")
        tickers = st.multiselect("Univers d'actifs", qdata.DEFAULT_TICKERS,
                                 default=qdata.DEFAULT_TICKERS[:10])
        extra = st.text_input("Tickers additionnels (séparés par des virgules)", "")
        years = st.slider("Historique (années)", 3, 10, 6)
        horizon = st.select_slider("Horizon / rééquilibrage (jours de bourse)", [5, 10, 21], value=5)
        rf = st.slider("Taux sans risque annuel (%)", 0.0, 6.0, 2.0, 0.25) / 100
        w_max = st.slider("Poids maximal par actif (%)", 10, 100, 30, 5) / 100
        tc_bps = st.slider("Coûts de transaction (pb)", 0, 30, 5)

        with st.expander("Modèle et optimiseur"):
            forget = st.slider("Facteur d'oubli λ", 0.950, 0.9995, 0.995, 0.0005, format="%.4f",
                               help="Plus λ est petit, plus le modèle oublie vite le passé.")
            signal_strength = st.slider("Force du signal (IC supposé)", 0.0, 0.30, 0.10, 0.01,
                                        help="Amplitude de l'alpha = IC × volatilité × score. "
                                             "0 ignore totalement l'IA.")
            turnover_penalty = st.slider("Pénalité de rotation", 0.0, 1.0, 0.30, 0.05)
            lookback = st.select_slider("Fenêtre de covariance (jours)", [126, 252, 504], value=252)
            warmup = st.number_input("Warm-up avant le 1er rééquilibrage (jours)", 63, 504, 126, 21)

        st.form_submit_button("Calculer", type="primary", width="stretch")

all_tickers = list(dict.fromkeys(tickers + [t.strip().upper() for t in extra.split(",") if t.strip()]))
if len(all_tickers) < 4:
    st.warning("Sélectionnez au moins 4 actifs.")
    st.stop()

cfg = bt.Config(
    horizon=int(horizon), lookback=int(lookback), warmup=int(warmup), forget=float(forget),
    signal_strength=float(signal_strength), rf=float(rf), w_max=float(w_max),
    tc_bps=float(tc_bps), turnover_penalty=float(turnover_penalty),
)


# ------------------------------------------------------------------ calculs (mis en cache)
@st.cache_data(ttl=3600, show_spinner="Chargement des prix…")
def cached_prices(source: str, tickers: tuple, start: str, end: str):
    return qdata.load_prices(source, tickers, start, end)


@st.cache_data(show_spinner="Entraînement walk-forward et backtest…")
def cached_study(prices: pd.DataFrame, cfg_dict: dict):
    return bt.run_study(prices, bt.Config(**cfg_dict))


end = date.today()
start = end - timedelta(days=int(years * 365.25))
prices, source_used, warning = cached_prices(source, tuple(all_tickers), str(start), str(end))
if warning:
    st.warning(warning)
if source_used == qdata.SIMULATED:
    st.info("Données **simulées** : les résultats illustrent la mécanique, pas une performance réelle.")

try:
    study = cached_study(prices, cfg.__dict__)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

last_weights = study.ai.weights.iloc[-1]
alloc_close = bt.current_allocation(prices, study.ai.model, cfg, w_prev=last_weights.values)

tab_live, tab_bt, tab_model, tab_front, tab_doc = st.tabs(
    ["Allocation en temps réel", "Backtest", "Modèle (apprentissage continu)",
     "Frontière efficiente", "Méthodologie"]
)

# ------------------------------------------------------------------ temps réel
with tab_live:
    c1, c2 = st.columns([1, 1])
    auto = c1.toggle("Rafraîchissement automatique", value=False)
    every = c2.select_slider("Intervalle (secondes)", [15, 30, 60, 120], value=60, disabled=not auto)

    @st.fragment(run_every=f"{every}s" if auto else None)
    def live_panel():
        latest, ts, label = qdata.latest_prices(prices, source_used)
        live_prices = qdata.append_live_row(prices, latest, ts)
        try:
            alloc = bt.current_allocation(live_prices, study.ai.model, cfg, w_prev=last_weights.values)
        except ValueError as exc:
            st.error(str(exc))
            return

        st.caption(f"Mis à jour à {ts:%H:%M:%S} — {label}. Le modèle est entraîné jusqu'à la dernière "
                   f"clôture ({prices.index[-1]:%d/%m/%Y}) ; il se met à jour dès qu'un horizon se réalise.")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rendement espéré (annualisé)", f"{alloc.exp_return * 100:.1f} %")
        k2.metric("Volatilité espérée", f"{alloc.exp_vol * 100:.1f} %")
        k3.metric("Sharpe espéré", f"{alloc.exp_sharpe:.2f}")
        k4.metric("Rotation vs dernier rééquilibrage", f"{np.abs(alloc.weights - last_weights).sum() * 100:.0f} %")

        table = pd.DataFrame({
            "Dernier prix": latest,
            "Var. vs clôture (%)": (latest / prices.iloc[-1] - 1) * 100,
            "Alpha prédit (%/an)": alloc.alpha * 100,
            "μ total (%/an)": alloc.mu * 100,
            "Poids recommandé": alloc.weights,
            "Dernier rééquilibrage": last_weights,
        }).sort_values("Poids recommandé", ascending=False)
        left, right = st.columns([3, 2])
        left.dataframe(
            table, width="stretch",
            column_config={
                "Dernier prix": st.column_config.NumberColumn(format="%.2f"),
                "Var. vs clôture (%)": st.column_config.NumberColumn(format="%+.2f"),
                "Alpha prédit (%/an)": st.column_config.NumberColumn(format="%+.1f"),
                "μ total (%/an)": st.column_config.NumberColumn(format="%.1f"),
                "Poids recommandé": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
                "Dernier rééquilibrage": st.column_config.NumberColumn(format="percent"),
            },
        )
        right.plotly_chart(
            plots.allocation_bar_fig(pd.DataFrame({"Optimal": alloc.weights, "Précédent": last_weights})),
            width="stretch",
        )

    live_panel()

# ------------------------------------------------------------------ backtest
with tab_bt:
    summary = study.summary
    ai_row, eq_row = summary.loc["IA continue"], summary.loc["Équipondéré"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sharpe — IA continue", f"{ai_row['Sharpe']:.2f}", f"{ai_row['Sharpe'] - eq_row['Sharpe']:+.2f} vs équipondéré")
    k2.metric("Rendement annualisé", f"{ai_row['Rendement annualisé'] * 100:.1f} %",
              f"{(ai_row['Rendement annualisé'] - eq_row['Rendement annualisé']) * 100:+.1f} pt")
    k3.metric("Max drawdown", f"{ai_row['Max drawdown'] * 100:.1f} %")
    k4.metric("Rotation annuelle", f"{ai_row['Rotation annuelle'] * 100:.0f} %")

    first, last = study.sims["IA continue"].index[[0, -1]]
    st.caption(f"Période testée hors-échantillon : {first:%d/%m/%Y} → {last:%d/%m/%Y}. "
               f"Coûts de {cfg.tc_bps:.0f} pb par unité échangée inclus.")
    st.plotly_chart(plots.equity_fig(study.sims, log_y=st.toggle("Échelle logarithmique")), width="stretch")
    st.plotly_chart(plots.drawdown_fig(study.sims), width="stretch")
    st.plotly_chart(plots.weights_fig(study.ai.weights), width="stretch")
    st.subheader("Synthèse des performances")
    st.dataframe(
        summary.style.format({
            "Rendement total": "{:.1%}", "Rendement annualisé": "{:.1%}", "Volatilité": "{:.1%}",
            "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Max drawdown": "{:.1%}", "Calmar": "{:.2f}",
            "Rotation annuelle": "{:.0%}",
        }),
        width="stretch",
    )

# ------------------------------------------------------------------ modèle
with tab_model:
    ic_ai, ic_fz = study.ai.ic, study.frozen.ic
    k1, k2, k3 = st.columns(3)
    k1.metric("IC moyen — IA continue", f"{ic_ai.mean():+.3f}")
    k2.metric("IC moyen — IA figée", f"{ic_fz.mean():+.3f}")
    k3.metric("Part d'IC > 0 (continue)", f"{(ic_ai > 0).mean() * 100:.0f} %")
    st.caption("L'IC est la corrélation de rang entre l'alpha prédit et le rendement réalisé ensuite. "
               "En pratique, un IC moyen de 0,02 à 0,05 est déjà considéré comme utile ; "
               "une valeur proche de 0 ou négative signifie que le signal n'a pas de pouvoir prédictif.")
    st.plotly_chart(plots.ic_fig({"IA continue": ic_ai, "IA figée": ic_fz}), width="stretch")
    st.plotly_chart(plots.coef_fig(study.ai.coefs), width="stretch")
    st.plotly_chart(plots.prediction_scatter_fig(study.ai.alpha, study.ai.realized), width="stretch")
    coef_now = pd.Series(study.ai.model.w, index=[FEATURE_LABELS[c] for c in study.ai.coefs.columns])
    st.subheader("Coefficients actuels")
    st.dataframe(coef_now.rename("Coefficient").to_frame().style.format("{:+.5f}"), width="stretch")

# ------------------------------------------------------------------ frontière
with tab_front:
    mu_v, cov_m = alloc_close.mu.values, alloc_close.cov.values
    r_vol, r_ret, r_sh = opt.random_portfolios(mu_v, cov_m, cfg.rf, 3000, cfg.w_max)
    f_vol, f_ret = opt.efficient_frontier(mu_v, cov_m, cfg.w_max)
    w_mv = opt.min_variance(cov_m, cfg.w_max)
    w_eq = np.full(len(mu_v), 1.0 / len(mu_v))
    points = {}
    for name, w in {"Sharpe max": alloc_close.weights.values, "Variance minimale": w_mv, "Équipondéré": w_eq}.items():
        ret, vol, _ = opt.portfolio_stats(w, mu_v, cov_m, cfg.rf)
        points[name] = (vol, ret)
    st.plotly_chart(plots.frontier_fig(r_vol, r_ret, r_sh, f_vol, f_ret, points), width="stretch")
    st.caption(f"Situation à la dernière clôture ({alloc_close.as_of:%d/%m/%Y}), contrainte de poids maximal "
               f"{cfg.w_max:.0%} par actif.")

# ------------------------------------------------------------------ méthodologie
with tab_doc:
    st.markdown(
        """
**Pipeline** (à chaque date de rééquilibrage, sans aucune information future) :

1. **Variables** : momentum (5j, 1m, 3m, 6m hors dernier mois), volatilité, écart à la moyenne mobile 50j,
   drawdown 6 mois — standardisées en coupe transversale.
2. **Apprentissage continu** : un modèle linéaire mis à jour chaque jour par moindres carrés récursifs avec
   facteur d'oubli λ. Dès qu'un horizon *h* se réalise, le rendement observé met à jour les coefficients.
3. **Rendements espérés** : μ = rendement de marché (EWMA, réduit de moitié) + alpha, avec
   alpha = IC supposé × volatilité × score du modèle (règle de Grinold).
4. **Risque** : covariance de Ledoit-Wolf sur fenêtre glissante.
5. **Allocation** : maximisation du ratio de Sharpe, long-only, plafond par actif, pénalité de rotation.
   Si aucun actif ne dépasse le taux sans risque, bascule sur la variance minimale.
        """
    )
    st.latex(r"\max_{w}\ \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}\quad \text{s.c. } \mathbf{1}^\top w = 1,\ 0 \le w_i \le w_{\max}")
    st.latex(r"w_t = w_{t-1} + K_t\,(y_t - X_t w_{t-1}),\qquad P_t = \lambda^{-1}\!\left(I - K_t X_t\right)P_{t-1}")
    st.markdown(
        """
**Références du backtest** : *IA figée* (même modèle, entraîné uniquement pendant le warm-up),
*Max Sharpe historique* (moyennes/covariances historiques) et *Équipondéré*.

**Limites** : prix ajustés Yahoo Finance (biais du survivant possible sur l'univers choisi), pas de frais de
financement ni d'impôts, exécution supposée à la clôture, résultats sensibles aux hyper-paramètres.
Outil pédagogique : ne constitue pas un conseil en investissement.
        """
    )
