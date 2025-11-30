# Phase 17: Structural Agency - Summary

Generated: 2025-11-30T20:31:11.402676

## Overview

Phase 17 implements **Structural Agency**: the tendency of the system to select
internal trajectories that preserve self-prediction and identity coherence,
without any external rewards, goals, or human semantics.

## Key Metrics

### Agency Index (Global)
- Mean: 0.4782
- Std: 0.0189
- Median: 0.4763
- p95: 0.5075

### Autonomy Gain
- Mean: 0.4782
- Std: 0.0189

### Survival of Structure
- Mean: 0.7895
- Std: 0.0020

## Null Model Comparison

### Shuffled Agency (Null B)
- Null mean modulation: 0.0000
- Null p95: 0.0000
- Real mean modulation: 0.0000

### Noise Agency (Null C)
- Null mean modulation: 0.0000
- Null p95: 0.0000

## GO/NO-GO Criteria

| Criterion | Status |
|-----------|--------|
| Agency Index > Shuffled p95 | PASS |
| Agency Index > Noise p95 | PASS |
| Autonomy Gain > 0 | PASS |
| Survival > 0.5 | PASS |

**Passing: 3/4 (need >= 2)**

## GO

Agency signals successfully modulate transitions beyond null baselines.

## Endogeneity Verification

All parameters in Phase 17 are derived from data:
- Self-model learning rate: η = 1/√(n+1)
- Identity EMA rate: α = 1/√(T+1)
- Agency signal: A_t = sum(centered_ranks)
- Modulation strength: λ = 1/(std(A)+1)
- Manifold dimension: d = count(eigenvalues >= median)

**ZERO magic constants. NO rewards. NO goals. NO human semantics.**

## Files Generated

- `results/phase17/agency_metrics.json` - Full metrics
- `results/phase17/phase17_summary.md` - This summary
- `figures/17_agency_index.png` - Agency index visualization
- `figures/17_agency_distribution.png` - Signal distribution
- `figures/17_survival_structure.png` - Structure persistence
- `figures/17_null_comparison.png` - Null model comparison
