# Phase 7: Análisis Completo del Sistema de Consentimiento

**Fecha**: 2025-11-29T23:18:44.458646
**Hash de datos**: `5ff3b22847e61ce3`

---

## 1. Condicional por Modo (-1/0/+1)

### Tabla: Métricas por Modo

| Modo | n | r (mean) | MI aprox | TE(N→E) | TE(E→N) | G (mean±std) |
|------|---|----------|----------|---------|---------|--------------|
| -1 | 619 | -0.0172 | 0.0619 | -0.0054 | -0.0162 | 0.5000±0.0000 |
| +0 | 3803 | -0.0372 | 0.0611 | -0.0412 | -0.0280 | 0.5000±0.0000 |
| +1 | 578 | 0.0037 | 0.0265 | 0.0061 | 0.0320 | 0.5000±0.0000 |

**Interpretación**:
- Modo -1 (anti-align): r < 0 esperado
- Modo 0 (off): r ≈ 0 esperado
- Modo +1 (align): r > 0 esperado

---

## 2. Bandit: Regret y Selección

### Tabla: Estadísticas del Bandit

| Mundo | Regret Final | Recompensa Total | Pulls (-1) | Pulls (0) | Pulls (+1) |
|-------|--------------|------------------|------------|-----------|------------|
| NEO | 0.0000 | 2500.0000 | 0 | 0 | 0 |
| EVA | 0.0000 | 2500.0000 | 0 | 0 | 0 |

**Curva de regret**: Ver `/root/NEO_EVA/results/phase7/figures/regret_curve.png`

---

## 3. Consent Lift

### Tabla: Probabilidades de Consentimiento

| Métrica | Valor |
|---------|-------|
| P(a_NEO) | 0.4906 |
| P(a_EVA) | 0.4986 |
| P(a_NEO & a_EVA) | 0.2442 |
| **Lift** | **1.00** |
| IC 95% | [0.97, 1.03] |

**Interpretación**:
- Lift = 1.00 indica que NEO y EVA consienten juntos **1.0× más** de lo esperado si fueran independientes.
- IC 95% no incluye 1.0: ⚠️ Revisar

---

## 4. Safety Metrics

### Tabla: Métricas de Seguridad

| Métrica | Valor |
|---------|-------|
| ρ p99 | 0.9900 |
| ρ p95 | 0.9900 |
| Var(I) p25 | 0.000000 |
| % ciclos con ρ ≥ p99 | 100.00% |
| % ciclos con Var ≤ p25 | 100.00% |
| Tiempo medio OFF | 0.0 ciclos |
| Períodos OFF | 0 |
| Gate open % | 100.0% |

**Estado final**:
- NEO stopped: False
- Stop reason: None

---

## 5. Figuras

| Figura | Archivo |
|--------|---------|
| Utilidad por modo | `/root/NEO_EVA/results/phase7/figures/utility_by_mode.png` |
| r/MI/TE por modo | `/root/NEO_EVA/results/phase7/figures/metrics_by_mode.png` |
| Heatmap de modos | `/root/NEO_EVA/results/phase7/figures/mode_heatmap.png` |
| Curva de regret | `/root/NEO_EVA/results/phase7/figures/regret_curve.png` |
| Evolución de modos | `/root/NEO_EVA/results/phase7/figures/mode_evolution.png` |
| Consent lift | `/root/NEO_EVA/results/phase7/figures/consent_lift.png` |

---

## 6. Verificación de Integridad

```
Data Hash: 5ff3b22847e61ce3
Timestamp: 2025-11-29T23:18:44.458646
Samples: 5000
Bootstrap iterations: 1000
```

---

## 7. Criterios GO/NO-GO

| Criterio | Estado |
|----------|--------|
| r cambia con modo | ✅ |
| Bandit aprende (regret ↓) | ✅ |
| Lift > 1 significativo | ⚠️ |
| Safety cuts < 5% | ⚠️ |

---

*Generado automáticamente por phase7_analysis.py*
*Principio: "Si no sale de la historia, no entra en la dinámica"*
