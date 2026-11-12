"""Chargement des prix : Yahoo Finance (API) ou données simulées.

Les données simulées servent à la démonstration et aux tests hors-ligne : modèle à un facteur
de marché + bruit idiosyncratique + dérive lentement variable (régimes), ce qui crée un peu
de momentum. Ce n'est PAS une preuve de performance : l'application l'indique clairement.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM",
    "XOM", "JNJ", "PG", "KO", "GLD", "TLT",
]

YAHOO = "Yahoo Finance"
SIMULATED = "Données simulées"


class DataError(RuntimeError):
    """Erreur de chargement des données de marché."""


def simulate_prices(tickers, start, end, seed: int = 42) -> pd.DataFrame:
    """Prix journaliers simulés (jours ouvrés) pour une liste de tickers."""
    idx = pd.bdate_range(start, end)
    t_len, n = len(idx), len(tickers)
    if t_len < 50:
        raise DataError("Période trop courte pour la simulation.")
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.6, 1.3, n)
    vol_idio = rng.uniform(0.008, 0.02, n)
    market = rng.normal(0.0003, 0.01, t_len)

    drift = np.zeros((t_len, n))
    d = rng.normal(0.0, 0.0004, n)
    for t in range(t_len):
        d = 0.98 * d + rng.normal(0.0, 0.00015, n)
        drift[t] = d

    rets = market[:, None] * beta + drift + rng.normal(0.0, 1.0, (t_len, n)) * vol_idio
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=list(tickers))


def fetch_yahoo(tickers, start, end) -> pd.DataFrame:
    """Prix de clôture ajustés via yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise DataError("Le paquet yfinance n'est pas installé.") from exc

    try:
        raw = yf.download(
            list(tickers), start=str(start), end=str(end),
            auto_adjust=True, progress=False, threads=True,
        )
    except Exception as exc:  # réseau, quota, etc.
        raise DataError(f"Échec du téléchargement Yahoo Finance : {exc}") from exc

    if raw is None or raw.empty or "Close" not in raw:
        raise DataError("Yahoo Finance n'a renvoyé aucune donnée.")

    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(list(tickers)[0])
    close = close.dropna(axis=1, how="all").ffill(limit=5).dropna(how="any")
    if close.shape[1] < 3 or len(close) < 300:
        raise DataError("Données insuffisantes (au moins 3 actifs et ~300 jours requis).")
    return close


def load_prices(source: str, tickers, start, end, seed: int = 42):
    """Retourne (prix, source_effective, message_d_avertissement ou None)."""
    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if source == YAHOO:
        try:
            return fetch_yahoo(tickers, start, end), YAHOO, None
        except DataError as exc:
            msg = f"{exc} Bascule sur des données simulées."
            return simulate_prices(tickers, start, end, seed), SIMULATED, msg
    return simulate_prices(tickers, start, end, seed), SIMULATED, None


def latest_prices(prices: pd.DataFrame, source: str):
    """Derniers prix « temps réel » : (Series, horodatage, libellé de la source).

    - Yahoo : dernière barre à 1 minute (repli sur la clôture journalière).
    - Simulé : petit mouvement aléatoire autour de la dernière clôture.
    Les tickers manquants retombent sur la dernière clôture connue.
    """
    last_close = prices.iloc[-1]
    now = pd.Timestamp.now()

    if source == YAHOO:
        try:
            import yfinance as yf

            raw = yf.download(
                list(prices.columns), period="1d", interval="1m",
                auto_adjust=True, progress=False, threads=True,
            )
            if raw is None or raw.empty:
                raw = yf.download(
                    list(prices.columns), period="5d", interval="1d",
                    auto_adjust=True, progress=False, threads=True,
                )
            close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(prices.columns[0])
            last = close.ffill().iloc[-1].reindex(prices.columns)
            return last.fillna(last_close), now, "Yahoo Finance (dernière cotation)"
        except Exception:
            return last_close, now, "Yahoo Finance indisponible : dernière clôture"

    rng = np.random.default_rng(int(time.time() // 5))
    shock = np.exp(rng.normal(0.0, 0.002, len(last_close)))
    return last_close * shock, now, "Simulation (tick aléatoire)"


def append_live_row(prices: pd.DataFrame, latest: pd.Series, ts: pd.Timestamp) -> pd.DataFrame:
    """Ajoute (ou remplace) la ligne du jour avec les derniers prix."""
    day = ts.normalize()
    out = prices.copy()
    if day <= out.index[-1]:
        out.iloc[-1] = latest.reindex(out.columns).values
    else:
        out.loc[day] = latest.reindex(out.columns).values
    return out
