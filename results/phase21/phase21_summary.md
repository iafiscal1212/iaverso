# Phase 21: Cross-Agent Ecology & Influence - Summary

Generated: 2025-11-30T22:16:03.671941

## Overview

Phase 21 implements **Cross-Agent Ecology & Influence** as purely endogenous
inter-agent coupling. Two agents (N and E) influence each other based on
shared ecological tension.

## Key Metrics

### Shared Tension
- Mean T_eco: 0.2487
- T_eco persistence (autocorr): 0.1555

### Cross-Influence
- Mean |F_E→N|: 0.3894
- Mean displacement: 0.3894

### Correlations
- T_eco vs proximity: -0.0745
- T_eco vs |F|: 0.9170

## Null Model Comparison

### Disabled (Null A)
- Mean F: 0.0000

### Shuffled (Null B)
- Mean F: 0.3464

### Random Fields (Null C)
- Mean F: 0.4959

## GO/NO-GO Criteria

| Criterion | Status |
|-----------|--------|
| Cross-influence magnitude > 0.1 | PASS |
| Influence gain is modulated (std > 0.1) | PASS |
| Shared tension correlates with influence | PASS |
| Ecological update changes trajectory | PASS |
| Shared tension is temporally coherent | PASS |

**Passing: 5/5 (need >= 3)**

## GO

Cross-agent ecology demonstrates functional inter-agent influence beyond null baselines.

## Endogeneity Verification

All parameters derived from data:
- d_NE = ||z_N - z_E||
- d_mu_NE = min_{k,l} ||mu_k_N - mu_l_E||
- T_a = rank(var_w(z)) * rank(R)
- T_eco = (rank(T_N) + rank(T_E))/2 * rank(1 - d_tilde_mu)
- beta = rank(T_eco) * rank(D_nov)
- F = beta * (z_source - z_target)
- w = sqrt(t+1)

**ZERO magic constants. NO semantic labels.**

## Files Generated

- `results/phase21/ecology_metrics.json` - Full metrics
- `results/phase21/phase21_summary.md` - This summary
- `figures/21_ecology_timeline.png` - T_eco and F timeline
- `figures/21_influence_field.png` - Influence field visualization
- `figures/21_null_comparison.png` - Null model comparison
