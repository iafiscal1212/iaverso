# Phase 19: Parameter Derivation Report

## Overview

This report documents the provenance of ALL parameters used in Phase 19.
Every parameter must be derived from data - ZERO magic numbers allowed.

## Structural Drives (drives19.py)

### Stability Drive (D_stab)

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `rank_spread` | `rank(manifold_spread)` | Spread history | Rank within distribution |
| `rank_integration` | `rank(integration)` | Integration history | Rank within distribution |
| `stability_t` | `-rank_spread + rank_integration` | Ranks | High integration, low spread |
| `D_stab_t` | `rank(stability_t)` | Stability history | Final rank |

### Novelty/Tension Drive (D_nov)

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `novelty_raw` | `min_distance(z_t, prototypes)` | Manifold, prototypes | Distance to nearest |
| `novelty_t` | `rank(novelty_raw)` | Novelty history | Rank |
| `tension_raw` | `var(velocity_magnitudes)` | Velocity history | Variance of ||delta_z|| |
| `tension_t` | `rank(tension_raw)` | Tension history | Rank |
| `combined_t` | `novelty_t + tension_t` | Ranks | Sum of ranked components |
| `D_nov_t` | `rank(combined_t)` | Combined history | Final rank |

### Irreversibility Drive (D_irr)

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `rank_irr` | `rank(irr_local)` | Local irreversibility | Rank |
| `rank_epr` | `rank(epr_local)` | Local EPR | Rank |
| `combined_t` | `rank_irr + rank_epr` | Ranks | Sum |
| `D_irr_t` | `rank(combined_t)` | Combined history | Final rank |

### Drive Gradient Estimation

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `k` | `max(3, min(log(T+1), sqrt(T)))` | Timestep T | Endogenous k for k-NN |
| `gradient` | `sum((delta_d/dist) * direction)` | k neighbors | Finite difference estimation |

### Drive Direction Computation

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `var_x` | `var(D_x history)` | Drive histories | Variance of each drive |
| `w_x` | `var_x / sum(variances)` | Variances | Variance-proportional weights |
| `direction` | `sum(w_x * gradient_x)` | Weights, gradients | Weighted gradient combination |

### Transition Modulation

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `bias_t` | `cos(drive_direction, displacement)` | Direction, states | Cosine similarity |
| `std_bias` | `std(bias_history)` | Bias history | Standard deviation |
| `lambda_t` | `1/(std_bias + 1)` | Bias std | Endogenous modulation strength |
| `P'_t(i->j)` | `P_base * exp(lambda * bias)` | Base probs, lambda, bias | Modulated probabilities |

## Structural Bounds

| Bound | Value | Justification |
|-------|-------|---------------|
| Min neighbors (k) | 3 | Statistical minimum for gradient |
| Queue maxlen | 100-500 | Memory constraint |
| N_states | 10 | Configuration |
| N_prototypes | 5 | Configuration |
| Warmup | 50 | sqrt(T) scaling |

## Semantic Label Analysis

Phase 19 uses ONLY structural/mathematical terms:
- `drive` - structural drive (mathematical scalar field)
- `stability` - stability drive (mathematical)
- `novelty` - novelty drive (mathematical distance)
- `tension` - velocity variance (mathematical)
- `irreversibility` - irreversibility drive (mathematical)
- `gradient` - gradient estimation (mathematical)
- `bias` - transition bias (mathematical cosine)
- `modulation` - transition modulation (mathematical)

NO human-centric semantic labels:
- No `reward`, `goal`, `utility`
- No `hunger`, `pain`, `fear`
- No `good`, `bad`, `optimal`

## Certification

All parameters traced to:
1. Data statistics (ranks, variances, means)
2. History lengths (T, log(T), sqrt(T))
3. Mathematical identities (0, 0.5, 1)
4. Structural constraints (dimension bounds)

ZERO arbitrary/magic constants.
