# Gestion quantitative IA

Dashboard Streamlit d'optimisation de portefeuille en temps réel : un modèle en **apprentissage continu**
prédit les rendements relatifs des actifs, puis l'allocation est calculée par **maximisation du ratio de Sharpe**.

## Lancer

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Tests : `python -m pytest -q tests`

Sans accès Internet (ou si Yahoo Finance échoue), l'application bascule automatiquement sur des **données simulées**
et l'indique clairement.

## Structure

```
app.py                 Interface Streamlit (5 onglets)
quant/data.py          Prix Yahoo Finance (yfinance) ou simulés ; cotation « live »
quant/features.py      Variables explicatives causales, standardisées en coupe transversale
quant/model.py         Moindres carrés récursifs avec facteur d'oubli (apprentissage en ligne)
quant/optimizer.py     Max Sharpe (SLSQP), variance minimale, frontière efficiente
quant/backtest.py      Walk-forward, références, allocation courante
quant/metrics.py       Sharpe, Sortino, drawdown, Calmar, rotation
quant/plots.py         Graphiques Plotly
tests/test_core.py     Tests (absence de fuite du futur, RLS, contraintes de l'optimiseur, pipeline)
```

## Méthode

À chaque date *t*, uniquement avec l'information disponible en *t* :

1. **Mise à jour du modèle** : le rendement réalisé entre *t−h* et *t* devient connu ; il met à jour les coefficients
   (RLS, facteur d'oubli λ). Les variables : momentums, volatilité, écart à la MM50, drawdown 6 mois.
2. **Rendements espérés** : `μ = rendement de marché (EWMA, réduit) + alpha`, avec la règle de Grinold
   `alpha = IC supposé × volatilité × score`. Le modèle fournit le *classement* ; l'IC supposé borne l'amplitude.
3. **Risque** : covariance Ledoit-Wolf sur fenêtre glissante.
4. **Allocation** : `max (wᵀμ − r_f) / √(wᵀΣw)` sous `Σw = 1`, `0 ≤ wᵢ ≤ w_max`, pénalité de rotation.
   Repli sur la variance minimale si aucun actif ne dépasse le taux sans risque.
5. **Backtest** net de coûts de transaction, comparé à : IA figée (sans apprentissage continu),
   Max Sharpe historique, équipondéré.

## Temps réel

L'onglet « Allocation en temps réel » récupère la dernière cotation (Yahoo, barre 1 min), reconstruit les variables
avec le prix du moment et recalcule l'allocation ; rafraîchissement automatique optionnel (`st.fragment`).
Le modèle lui-même se met à jour à chaque nouvelle clôture / horizon réalisé.

## Limites

- Les prix de marché sont efficients : l'**IC mesuré** (onglet Modèle) est le juge de paix. Un IC proche de 0 signifie
  que le signal ne prédit rien, et la stratégie n'a alors aucune raison de battre l'équipondéré.
- Résultats sensibles aux hyper-paramètres (pénalité de rotation, IC supposé, λ) : ne pas les ajuster pour « embellir »
  un backtest sur la même période (risque de sur-apprentissage).
- Biais du survivant possible selon l'univers choisi ; exécution supposée à la clôture ; pas de fiscalité ni de financement.
- Outil pédagogique, pas un conseil en investissement.

## Pistes d'extension

Modèles non linéaires (LightGBM en ré-entraînement glissant), contraintes sectorielles, CVaR, rééquilibrage
déclenché par seuil de dérive, flux de prix réel (WebSocket) à la place du polling.
