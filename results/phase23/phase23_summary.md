# Phase 23: Structural Self-Report - Summary

Generated: 2025-11-30T22:19:57.627188

## Overview

Phase 23 implements **Structural Self-Report** as purely endogenous feature
compression. System state is summarized without semantic interpretation.

## Key Metrics

### Report Statistics
- Mean |report|: 0.3844
- Report variance: 0.0153
- Report autocorrelation: 0.8629

### Compression
- Mean report dim: 4.0
- Compression ratio: 2.00

### Dynamics
- Mean SC: 0.6187
- Corr(SC, input): 0.0399

## Null Model Comparison

### Disabled (Null A)
- Mean |report|: 1.4142

### Shuffled (Null B)
- Mean |report|: 0.5314

### Random (Null C)
- Mean |report|: 0.9567

## GO/NO-GO Criteria

| Criterion | Status |
|-----------|--------|
| Report has meaningful variance | PASS |
| Compression reduces dimensionality | PASS |
| Report responds to input changes | PASS |
| Report shows temporal coherence | PASS |
| Report is non-trivial | PASS |

**Passing: 5/5 (need >= 3)**

## GO

Structural self-report demonstrates functional feature compression.

## Endogeneity Verification

All parameters derived from data:
- f_t = [rank(EPR), rank(D_nov), rank(T), rank(R), rank(spread), rank(v)]
- k = ceil(log2(feature_dim))
- c_t = project(f_t, V[:k])
- V = SVD(F_window)
- window = sqrt(t+1)
- report = c_t

**ZERO magic constants. NO semantic labels.**

## Files Generated

- `results/phase23/selfreport_metrics.json` - Full metrics
- `results/phase23/phase23_summary.md` - This summary
- `figures/23_selfreport_timeline.png` - Report timeline
- `figures/23_null_comparison.png` - Null model comparison
