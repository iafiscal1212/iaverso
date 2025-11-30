# Auditoría de Endogeneidad NEO-EVA
## Fecha: 2025-11-30T16:14:19.256859

## 1. Lint Endógeno (Análisis Estático)
- **Resultado**: PASS ✓
- Violaciones: 0

## 2. T-Scaling (τ, η, σ ∝ 1/√T)
- **Resultado**: PASS ✓
- Coeficiente de variación: 0.0000
- Detalles:
  - T=100: η=0.099504, ratio=1.000
  - T=400: η=0.049938, ratio=1.000
  - T=900: η=0.033315, ratio=1.000
  - T=1600: η=0.024992, ratio=1.000
  - T=2500: η=0.019996, ratio=1.000

## 3. Warmup
- **Resultado**: PASS ✓
- Tasa de warmup: 0.90%
- Límite: 5%

## Resumen
- Lint: PASS
- T-Scaling: PASS
- Warmup: PASS
