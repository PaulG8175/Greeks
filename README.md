# Greeks Computation via Monte Carlo — Black-Scholes

> 🎓 **Academic project** — Course PRB222, Financial Mathematics

Numerical computation of option sensitivities (**Greeks**) for vanilla and basket options in the Black-Scholes model, using two Monte Carlo methods.

---

## Methods

- **Finite differences**: `dP/dλ ≈ (P(λ+ε) − P(λ−ε)) / 2ε`
- **Pathwise differentiation**: differentiation under the expectation sign
- **Variance reduction**: antithetic variables

---

## Greeks computed

| Greek | Definition |
|-------|-----------|
| Δ (Delta) | `∂P/∂S₀` |
| Γ (Gamma) | `∂²P/∂S₀²` |
| Vega | `∂P/∂σ` |

---

## Parameters

| Parameter | Value |
|-----------|-------|
| `S₀` | 1 |
| `K` | 1 |
| `T` | 2 years |
| `r` | 0.01 |
| `σ` (vanilla) | 0.30 |
| `σ = (σ₁, σ₂)` (basket) | (0.25, 0.30) |
| `α = β` | 0.5 |
| `ρ` | 0.5 |

> All simulations use **Uniform random variables only** via Box-Muller transform.

---

## Topics covered

| Q | Content |
|---|---------|
| Q2 | Vanilla call price (MC + Black-Scholes formula) |
| Q3 | Theoretical Greeks Δ, Γ, Vega |
| Q4–5 | Finite differences and pathwise estimators |
| Q6 | Antithetic variables + comparison of 4 estimators |
| Q7 | Greeks as a function of S₀ ∈ [0, 2] |
| Q8 | Basket call Greeks (pathwise method) |
| Q9 | Convergence of basket estimators vs N |
| Q10 | Greeks vs S₁,₀, comparison with B&S |
| Q11 | Greeks vs K ∈ [0, 2] |
| Q12 | Greeks vs ρ ∈ ]−1, 1[ |
| Q13 | Call-Put parity on basket (analytical + MC) |
| Q14 | Key parameters for daily delta-hedging |

---

## Run

```bash
pip install numpy matplotlib scipy
python greques_commented.py
```

## Dependencies

`numpy` · `matplotlib` · `scipy`
