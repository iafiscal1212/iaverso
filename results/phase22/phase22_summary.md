# Phase 22: Minimal Grounding - Summary

Generated: 2025-11-30T22:17:23.284141

## Overview

Phase 22 implements **Minimal Grounding** as purely endogenous external signal
integration. External signals are projected onto the internal state manifold.

## Key Metrics

### Grounding Field
- Mean |G|: 0.6381
- Std |G|: 0.3320
- G autocorrelation: 0.9816

### Projection
- Mean P_norm: 1.4134
- P_norm std: 0.0185

### Displacement
- Mean displacement: 0.6381

### Signal Response
- Corr(G, novelty): 0.5677

## Null Model Comparison

### Disabled (Null A)
- Mean G: 0.0000

### Shuffled (Null B)
- Mean G: 0.6405

### Random (Null C)
- Mean G: 0.4900

## GO/NO-GO Criteria

| Criterion | Status |
|-----------|--------|
| Grounding magnitude > 0.05 | PASS |
| Projection matrix adapts (std > 0.01) | PASS |
| Grounding has structure (responds to signal) | PASS |
| Grounding changes trajectory | PASS |
| Grounding shows temporal coherence | PASS |

**Passing: 5/5 (need >= 3)**

## GO

Minimal grounding demonstrates functional external signal integration.

## Endogeneity Verification

All parameters derived from data:
- s_tilde = normalize(s_ext - mu_s)
- mu_s = EMA(s), alpha = 1/sqrt(t+1)
- P = cov(z_window) / ||cov(z)||_F
- window = sqrt(t+1)
- G = rank(||P*s||) * P*s
- z_next = z_base + G

**ZERO magic constants. NO semantic labels.**

## Files Generated

- `results/phase22/grounding_metrics.json` - Full metrics
- `results/phase22/phase22_summary.md` - This summary
- `figures/22_grounding_timeline.png` - Grounding timeline
- `figures/22_null_comparison.png` - Null model comparison
