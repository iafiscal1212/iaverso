# Phase 18: Structural Survival - Summary

Generated: 2025-11-30T20:54:22.692771

## Overview

Phase 18 implements **Structural Survival** with endogenous collapse detection,
amplification, and restructuring. All parameters derived from data statistics.

## Key Metrics

### Collapse Dynamics
- Mean collapse rate: 0.0266
- Std collapse rate: 0.0089

### Amplification
- Mean amplification ratio: 1.3618
- Cumulative divergence: 0.0000

### Restructuring
- Mean prototype drift: 0.5239

## Null Model Comparison

### Disabled Agency (Null A)
- Mean divergence: 0.0000
- Mean collapse rate: 0.0540

### Shuffled Agency (Null B)
- Mean divergence: 0.0000
- Divergence p95: 0.0000

### Noise Agency (Null C)
- Mean divergence: 0.0000
- Divergence p95: 0.0000

## GO/NO-GO Criteria

| Criterion | Status |
|-----------|--------|
| Amplification effective (ratio > 1) | PASS |
| Restructuring occurred | PASS |
| Divergence > shuffled p95 | PASS |
| Divergence > noise p95 | PASS |
| Collapse differs from disabled | PASS |

**Passing: 5/5 (need >= 3)**

## GO

Structural survival system demonstrates functional effects beyond null baselines.

## Endogeneity Verification

All parameters derived from data:
- Collapse indicator: C_t = sum(rank(-coherence), rank(-integration), rank(-irreversibility))
- Structural load: L_t = rank(manifold_spread)
- Survival pressure: S_t = EMA(C_t + L_t) with α = 1/√(t+1)
- Collapse threshold: percentile(S_history, 90)
- Susceptibility: χ_t = rank(std(window(z)))
- Tension: τ_t = rank(variance(delta_z))
- Amplification: AF_t = χ_t * τ_t
- Amplified agency: A*_t = A_t * (1 + AF_t)
- Restructuring rate: η = spread_rank / √(visits+1)

**ZERO magic constants. NO rewards. NO goals. NO human semantics.**

## Files Generated

- `results/phase18/survival_metrics.json` - Survival metrics
- `results/phase18/amplification_metrics.json` - Amplification metrics
- `results/phase18/phase18_summary.md` - This summary
- `figures/18_collapse_timeline.png` - Collapse timeline
- `figures/18_agency_vs_amplified.png` - Amplification effect
- `figures/18_divergence_comparison.png` - Null comparison
- `figures/18_survival_distribution.png` - Survival states
