# Phase 7: Acoplamiento Libre por Consentimiento y Utilidad Endógena

**Fecha**: 2025-11-30T11:58:42.847035
**Ciclos**: 2000

---

## Resumen

NEO y EVA deciden autónomamente si proponen acoplarse y cómo hacerlo:
- Voluntad individual π basada en beneficio vs coste
- Consentimiento bilateral: solo se acoplan si ambos quieren
- Modo m∈{-1, 0, +1} aprendido con Thompson Sampling
- Stopping rules endógenas por ρ, Var(I) y regret

---

## Fórmulas Usadas

### Beneficio Esperado
```
ΔÛ = (u_Y/(1+u_X)) × (λ₁^Y/(λ₁^Y+λ₁^X+ε)) × (conf^Y/(1+CV(r^X)))
```

### Coste Endógeno
```
coste = Rank(1{ρ≥p95} + RankInvVar(I) + Rank(latencia BUS)) / 3
```

### Voluntad
```
π = σ(rank(ΔÛ) - rank(coste))
a ~ Bernoulli(π)
```

### Consentimiento Bilateral
```
Acoplamiento activo ⟺ a_NEO = 1 AND a_EVA = 1
```

### Modo (Bandit)
```
m ∈ {-1, 0, +1} ~ Thompson Sampling con recompensa G = BordaRank(ΔRMSE, ΔMDL, MI)
```

---

## Resultados

### Métricas de Consentimiento

| Métrica | NEO | EVA |
|---------|-----|-----|
| Propuestas de consentimiento | 994 | 1023 |
| Consentimientos bilaterales | 507 | 505 |
| Activaciones de acoplamiento | 507 | 505 |

### Distribución de Modos

| Modo | NEO | EVA |
|------|-----|-----|
| -1 (anti-align) | 245 | 243 |
| 0 (off) | 1513 | 1511 |
| +1 (align) | 242 | 246 |

### Bandit Statistics

| Métrica | NEO | EVA |
|---------|-----|-----|
| Regret acumulado | 0.0000 | 0.0000 |
| Recompensa acumulada | 1000.0000 | 1000.0000 |

### Correlación

| Componente | Correlación |
|------------|-------------|
| S | -0.1508 |
| N | -0.0270 |
| C | -0.0668 |
| **Media** | **-0.0815** |

### Comparación: Coupled vs Ablation

| Métrica | Coupled | Ablation |
|---------|---------|----------|
| Correlación media | -0.0815 | 0.0366 |
| Eventos bilaterales | 525 | 483 |

---

## Criterios GO/NO-GO

| Criterio | Estado |
|----------|--------|
| Autonomía real (decisiones ON/OFF) | ✅ PASS |
| Consentimiento bilateral efectivo | ✅ PASS |
| Mejora de utilidad vs ablation | ✅ PASS |
| Distribución de modos no degenerada | ✅ PASS |

---

## Archivos Generados

- `coupled/series_neo.json`: Serie temporal NEO
- `coupled/series_eva.json`: Serie temporal EVA
- `coupled/consent_log_neo.json`: Log de consentimiento NEO
- `coupled/consent_log_eva.json`: Log de consentimiento EVA
- `coupled/bilateral_events.json`: Eventos de consentimiento bilateral
- `coupled/bandit_stats.json`: Estadísticas del bandit
- `ablation/`: Mismos archivos para ablación

---

*Generado: 2025-11-30T11:58:42.847035*
*Principio: "Si no sale de la historia, no entra en la dinámica"*
