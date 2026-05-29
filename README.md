# 📈 PRB222 — Calcul de Sensibilités (Grecques) par Monte Carlo

Projet numérique du cours **PRB222** portant sur le calcul des grecques d'options vanilles et d'options sur panier dans le modèle de Black-Scholes, via deux méthodes Monte Carlo.

---

## 🎯 Objectif

Calculer et comparer les sensibilités **Δ (Delta), Γ (Gamma) et Vega** par :
- **Différences finies** : `dP/dλ ≈ (P(λ+ε) − P(λ−ε)) / 2ε`
- **Méthode trajectorielle** : dérivation sous l'espérance

Avec réduction de variance par **variables antithétiques**.

---

## 📂 Structure

```
├── greques_commented.py   # Script principal commenté
└── figures/               # Graphiques générés (PDF)
    ├── fig_q6_*.pdf        # Convergence des 4 estimateurs (vanille)
    ├── fig_q7_*.pdf        # Grecques en fonction de S0
    ├── fig_q9_*.pdf        # Convergence grecques panier
    ├── fig_q10_*.pdf       # Grecques en fonction de S1,0
    ├── fig_q11_*.pdf       # Grecques en fonction de K
    └── fig_q12_*.pdf       # Grecques en fonction de ρ
```

---

## ⚙️ Paramètres

| Paramètre | Valeur |
|-----------|--------|
| `S0` | 1 |
| `K` | 1 |
| `T` | 2 ans |
| `r` | 0.01 |
| `σ` (vanille) | 0.30 |
| `σ = (σ1, σ2)` (panier) | (0.25, 0.30) |
| `α = β` | 0.5 |
| `ρ` | 0.5 |

> Les simulations utilisent uniquement des lois **Uniformes** via la méthode de Box-Muller.

---

## 📋 Questions traitées

| Q | Contenu |
|---|---------|
| Q2 | Prix du call vanille (MC + formule B&S) |
| Q3 | Grecques théoriques Δ, Γ, Vega |
| Q4-5 | Estimateurs différences finies et trajectoriels |
| Q6 | Variables antithétiques + comparaison des 4 estimateurs |
| Q7 | Grecques en fonction de S0 ∈ [0, 2] |
| Q8 | Grecques du call panier (méthode trajectorielle) |
| Q9 | Convergence des estimateurs panier en fonction de N |
| Q10 | Grecques en fonction de S1,0, comparaison B&S |
| Q11 | Grecques en fonction de K ∈ [0, 2] |
| Q12 | Grecques en fonction de ρ ∈ ]−1, 1[ |
| Q13 | Parité Call-Put sur panier (analytique + MC) |
| Q14 | Paramètres clés de la couverture Delta quotidienne |

---

## 🚀 Lancement

```bash
pip install numpy matplotlib scipy
python greques_commented.py
```

Les figures sont sauvegardées dans le dossier `figures/`.

---

## 📦 Dépendances

- `numpy`
- `matplotlib`
- `scipy`
