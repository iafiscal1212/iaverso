# NEO-EVA Phase 12: Resumen Final de Robustez
## Fecha: 2025-11-30T16:14:19.264185

## GO/NO-GO Checklist

| Criterio | Resultado | Valor |
|----------|-----------|-------|
| Warmup ≤ 5% | ✓ PASS | 0.90% |
| Lint endógeno | ✓ PASS | - |
| T-scaling | ✓ PASS | CV=0.0000 |
| TE_active/TE_sleep ≥ 1.5 | ✓ PASS | 4.24x |
| β̂_κ > 0, p<0.05 | ✗ FAIL | β=-0.2121 |
| AUC_test ≥ threshold | ✓ PASS | 0.9543 |

## TE/MIT Condicionados

### Ratio TE Activo/Sleep
- **Observado**: 4.24x
- **IC 95%**: [3.39, 5.09]

### Por Condición (Top 5)
| Condición | TE | MIT | n |
|-----------|-----|-----|---|
| WORK_GW_on_mid | 1.5345 | 0.1890 | 23 |
| LEARN_GW_on_mid | 0.9864 | 0.2890 | 22 |
| SOCIAL_GW_on_mid | 0.8346 | 0.1188 | 36 |
| WORK_GW_off_low | 0.7065 | 0.1063 | 32 |
| SOCIAL_GW_off_mid | 0.3465 | 0.2075 | 41 |

## AUC vs Nulos

- **AUC Observado**: 0.9543
- **AUC Null (mediana ± IQR)**: 0.5105 ± 0.0798

## Calibración de π

- **ECE (mediana)**: 0.2289
- **ECE (IQR)**: 0.3268

## Regresión: TE ~ κ + GW + H + state

| Variable | β̂ | p-value | IC 95% |
|----------|-----|---------|--------|
| kappa | -0.2121* | 0.0000 | [-0.2514, -0.1777] |
| GW | -0.2121* | 0.0000 | [-0.2514, -0.1777] |
| entropy | -0.0766* | 0.0400 | [-0.1488, -0.0078] |

## Multi-Seed Analysis (n=5)

| Métrica | Mediana | IQR |
|---------|---------|-----|
| AUC | 0.8217 | 0.0169 |
| TE Ratio | 4.32 | 0.28 |
| β_κ | 0.5326 | 0.0982 |

## Limitations

- Observational signatures in an endogenous framework
- No implementational recipe disclosed
- Aggregated statistics only (no raw traces)

---
*Generated automatically by Phase 12 Robustness Pipeline*
