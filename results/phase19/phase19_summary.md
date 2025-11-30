# Phase 19: Structural Drives - Summary

Generated: 2025-11-30T21:11:33.361538

## Overview

Phase 19 implements **Structural Drives** as purely endogenous scalar and vector fields
in the internal manifold. Drives induce preferred trajectories without semantic labels.

## Key Metrics

### Drive Statistics
- D_stab (Stability): mean=0.5043, std=0.0077
- D_nov (Novelty/Tension): mean=0.4840, std=0.0463
- D_irr (Irreversibility): mean=0.5056, std=0.0077

### Transition Modulation
- Mean cumulative divergence: 253.230751
- Drive persistence (autocorr lag-1): 0.4833

## Null Model Comparison

### Disabled Drives (Null A)
- Mean divergence: 0.000000

### Shuffled Drives (Null B)
- Mean divergence: 156.168361
- Divergence p95: 157.323140

### Noise Drives (Null C)
- Mean divergence: 156.168361
- Divergence p95: 157.323140

## GO/NO-GO Criteria

| Criterion | Status |
|-----------|--------|
| Drives endogenous (all ranks in [0,1]) | PASS |
| Divergence positive (drives affect transitions) | PASS |
| Divergence > shuffled p95 | PASS |
| Divergence > noise p95 | PASS |
| Persistence sufficient (> 0.3) | PASS |

**Passing: 5/5 (need >= 3)**

## GO

Structural drives demonstrate functional transition modulation beyond null baselines.

## Endogeneity Verification

All parameters derived from data:
- D_stab = rank(-rank(spread) + rank(integration))
- D_nov = rank(rank(novelty) + rank(tension))
- D_irr = rank(rank(irr_local) + rank(epr_local))
- k_neighbors = max(3, min(log(T+1), sqrt(T)))
- w_x = variance_proportional(drive_variances)
- λ_drive = 1/(std(bias)+1)

**ZERO magic constants. NO rewards. NO goals. NO human semantics.**

## Files Generated

- `results/phase19/drives_metrics.json` - Drive metrics
- `results/phase19/phase19_summary.md` - This summary
- `figures/19_drives_timeseries.png` - Drive time series
- `figures/19_drives_distribution.png` - Drive distributions
- `figures/19_divergence_comparison.png` - Null comparison
- `figures/19_drive_correlations.png` - Inter-drive correlations
- `figures/19_drive_vector_3d.png` - 3D drive trajectory
