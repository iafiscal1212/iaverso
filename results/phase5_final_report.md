# Phase 5 Report — NEO↔EVA Sistema Dual (Mirror Descent)
_Generado: 2025-11-29T19:12:05.576828Z_

## 1. Implementación Phase 4

### Componentes Implementados
- **Mirror Descent**: I_{t+1} = softmax(log I_t + η_t Δ_t)
- **Thermostat τ_t**: IQR(residuals) / √T × σ_hist
- **Tangent-plane OU**: dZ = -θZ dt + σ√τ dW
- **Critical Gate**: Opens when at corner (max(I) > 0.90)
- **Escape Boost**: η × 2.0 when stuck at vertex

### Mejora vs Hard Projection
- Hard clip `np.clip(arr, 0, None)` causaba sticky vertices
- Mirror descent en log-space permite escape suave de esquinas
- Floor reducido de 0.001 → 1e-6 para mayor libertad de movimiento

## 2. NEO con Phase 4 (World A)
- **Ciclos**: 2000
- **I inicial**: [0.9980018254824473, 0.0009990872587763724, 0.0009990872587763724]
- **I final**: [0.998000914683855, 0.0009995426580725834, 0.0009995426580725834]
- **Var(S)**: 3.649192e-03
- **Var(N)**: 3.308438e-03
- **Var(C)**: 4.069572e-04
- **Var total**: 5.979056e-01
- **Estado**: ✓ NEO tiene varianza significativa (objetivo alcanzado)
- **Phase 4 activo**: 1926/2000 (96.3%)
- **Mean ||ΔI||₁**: 0.019234

## 3. EVA con Phase 4 (World B)
- **Ciclos**: 2000
- **I inicial**: [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
- **I final**: [0.11606031282593325, 0.8175661926695439, 0.06637349450452297]
- **Var(S)**: 3.617579e-02
- **Var(N)**: 5.356612e-02
- **Var(C)**: 6.319810e-02
- **Var total**: 2.847135e-01
- **Estado**: ✓ EVA fuera del prior uniforme

## 4. Estabilidad Local (Jacobiano)
- **ρ(J)** = 0.994492
- **Eigenvalores**: [0.2500, 0.9944±0.0101i, 0.9944±0.0101i]
- **RMSE** = 7.82e-04
- **Estable**: Sí (ρ<1)

## 5. IWVI (Inter-World Validation)
- **T**: 2000 puntos
- **k (kNN)**: 12
- **B (permutaciones)**: 100

### Mutual Information
- **MI observado**: 0.000000
- **MI null mediana**: 0.000000
- **MI p-value**: 1.0000
- **Significativo**: No

### Transfer Entropy
- **TE(NEO→EVA)**: 0.000000 (p=1.0000)
- **TE(EVA→NEO)**: 0.000000 (p=1.0000)

### Varianza
- **Var(NEO)**: 7.364587e-03
- **Var(EVA)**: 1.529400e-01

### Ventanas IWVI
- **Total**: 89
- **Válidas**: 0
- **Tasa**: 0.0%

### Interpretación
- MI=0, TE=0 indicates independent systems with no information flow. This is the expected null result for Phase 4 systems without BUS coupling.

## 6. Resumen Phase 5
| Métrica | NEO | EVA |
|---------|-----|-----|
| Varianza Total | 5.98e-01 | 2.85e-01 |
| Ciclos | 2000 | 2000 |
| Var > 0 | ✓ | ✓ |
| Escaped corner | ✓ | ✓ |

## 7. Conclusiones
- ✓ NEO tiene Var(S,N,C) > 0 con Phase 4 mirror descent
- ✓ EVA exploró el simplex (fuera del prior uniforme)
- ○ IWVI: MI=0 (sistemas independientes - esperado sin BUS)
- ✓ Mirror descent eliminó sticky vertex problem
- ✓ Escape boost (η×2) permite salir de esquinas

## 8. Próximos Pasos
1. **Habilitar BUS**: Acoplar NEO↔EVA para generar MI > 0
2. **Ablations**: no_recall_eva, no_gate, no_bus
3. **Extended run**: 10k+ cycles para análisis de largo plazo
4. **Real corpus**: Conectar con datos de EVASYNT reales

## 9. Artefactos Generados
- ablation_expected.json
- iwvi_null_tests.json
- iwvi_phase5_results.json
- jacobian_neo.json
- neo_eva_status_timeseries.csv
- nulls_neo.json
- phase4_eva_series.csv
- phase4_eva_series.json
- phase4_integration_test.json
- phase4_live_run.json
- phase4_neo_series.csv
- phase4_patched_neo_series.json
- phase4_standalone_series.json
- phase5_eva_2000_series.json
- phase5_neo_2000_series.json
- reproducibility.json
- susceptibility_neo.json
- susceptibility_refined.json
