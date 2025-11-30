# Pre-Registro Endógeno NEO-EVA
## Fecha: 2025-11-30T16:14:16.894895

## Hipótesis

### H1: Transfer Entropy Condicional
TE_active / TE_sleep ≥ q95(null_ratio) condicionado por:
- state ∈ {SLEEP, WAKE, WORK, LEARN, SOCIAL}
- GW_on (deciles de intensidad)
- H (terciles de entropía)

### H2: Coeficiente κ en Regresión
β̂_κ > 0 en rank-regression: TE ~ κ + GW + H + state
Con p < 0.05 (bootstrap/permutation, n=200)

## Métricas GO/NO-GO (todas relativas a nulos)

1. **AUC_test** ≥ median(AUC_null) + IQR(AUC_null)
2. **r_real** ≥ q99(r_null) en rolling origin
3. **Warmup** ≤ 5%
4. **Endogeneity-lint**: PASS
5. **T-scaling**: PASS (τ, η, σ ∝ 1/√T)

## Diseño de Ventanas
w_estado = max(10, floor(√T_estado))

## Seeds
5 seeds independientes. Reportar mediana + IQR (no el mejor).

## Constantes Permitidas
- ε numérico (machine epsilon ≈ 2.2e-16)
- Prior uniforme simplex (1/3, 1/3, 1/3)

---
SHA256: 0251e0778967b721
