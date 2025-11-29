# Phase 5 Report — NEO↔EVA Sistema Dual
_Generado: 2025-11-29T18:23:58.134348Z_

## 1. Estado Actual

### NEO
- **T** = 10176 ciclos
- **Última I**: t=10175, S=1.000000, N=2.96e-323, C=4.94e-324
- **Var (w=100)**: S=0.000e+00, N=0.000e+00, C=0.000e+00

### EVA
- **T** = 201 ciclos
- **Última I**: t=200, S=0.333333, N=0.333333, C=0.333333
- **Var (w=14)**: S=0.000e+00, N=0.000e+00, C=0.000e+00

## 2. Estabilidad Local (Jacobiano)
- **ρ(J)** = 0.994492
- **Eigenvalores**: [0.2500, 0.9944±0.0101i, 0.9944±0.0101i]
- **RMSE** = 7.82e-04
- **Estable**: Sí

## 3. Susceptibilidad
- **χ**: min=0.000e+00, max=0.000e+00, median=0.000e+00
- **τ**: min=4, max=4, median=4.0

## 4. Tests de Nulos
- **Métrica**: variance
- **Observado**: 0.010735
- **Nulo mediana**: 0.010735
- **p̂**: 0.0000
- **B**: 1008

## 5. IWVI (Inter-World Validation)
- **MI observado**: 0.000000
- **MI p̂**: 1.0000
- **TE observado**: 0.000000
- **TE p̂**: 1.0000
- **k (kNN)**: 12
- **B**: 100

## 6. Phase 4 (Variabilidad Endógena)
- **Activaciones**: 100
- **Tasa**: 100.0%
- **IWVI válido**: True
- **Gate activations**: 100

## 7. Ablaciones (Esperadas)
| Ablación | Factor | ρ esperado |
|----------|--------|-----------|
| baseline | x1.00 | 0.9945 |
| no_recall_eva | x1.05 | 1.0442 |
| no_gate | x1.02 | 1.0144 |
| no_bus | x1.01 | 1.0044 |
| no_pca | x1.10 | 1.0939 |

## 8. Artefactos Generados
- ablation_expected.json
- iwvi_null_tests.json
- jacobian_neo.json
- neo_eva_status.md
- neo_eva_status_timeseries.csv
- nulls_neo.json
- phase4_integration_test.json
- phase4_live_run.json
- preregistration.md
- reproducibility.json
- susceptibility_neo.json
- susceptibility_refined.json
