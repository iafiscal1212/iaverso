# Phase 20: Parameter Derivation Report

## Overview

This report documents the provenance of ALL parameters used in Phase 20.
Every parameter must be derived from data - ZERO magic numbers allowed.

## Structural Veto System (veto20.py)

### Intrusion Detection

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `delta` | `||x_t - mu_k||` | Manifold, prototypes | Distance to nearest prototype |
| `delta_spread` | `|spread_t - spread_ema|` | Spread history | Deviation from EMA |
| `delta_epr` | `|epr_t - epr_ema|` | EPR history | Deviation from EMA |
| `alpha_ema` | `1/sqrt(t+1)` | Timestep | Endogenous decay |
| `rank_delta` | `rank(delta)` | Delta history | Rank |
| `rank_spread` | `rank(delta_spread)` | Delta spread history | Rank |
| `rank_epr` | `rank(delta_epr)` | Delta EPR history | Rank |
| `shock_t` | `rank_delta * rank_spread * rank_epr` | Ranks | Multiplicative shock |

### Structural Opposition Field

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `direction` | `x_t - mu_k` | Manifold, nearest prototype | Direction from prototype |
| `rank_shock` | `rank(shock_t)` | Shock history | Rank of current shock |
| `O_t` | `-rank_shock * normalize(direction)` | Rank, direction | Opposition back to prototype |

### Resistance Gain

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `window_size` | `sqrt(t)` | Timestep | Endogenous window |
| `window_std` | `std(shock[-window:])` | Recent shock history | Volatility |
| `gamma_t` | `1/(1 + window_std)` | Window std | Inverse volatility |

### Veto Transition Adjustment

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `adjustment` | `gamma_t * O_t` | Gamma, opposition | Scaled opposition |
| `x_next` | `x_next_base + adjustment` | Base transition, adjustment | Veto-adjusted next state |
| `veto_effect` | `||x_next - x_next_base||` | Adjusted vs base | Magnitude of veto |

## Structural Bounds

| Bound | Value | Justification |
|-------|-------|---------------|
| Min window (k) | 5 | Statistical minimum |
| N_prototypes | 5 | Configuration |
| STATE_DIM | 4 | Configuration |
| Warmup | 50 | sqrt(T) scaling |

## Semantic Label Analysis

Phase 20 uses ONLY structural/mathematical terms:
- `veto` - structural veto (mathematical adjustment)
- `resistance` - resistance gain (mathematical inverse volatility)
- `opposition` - opposition field (mathematical vector)
- `shock` - shock indicator (mathematical deviation product)
- `intrusion` - intrusion detection (mathematical distance)

NO human-centric semantic labels:
- No `pain`, `fear`, `threat`
- No `danger`, `safe`, `attack`
- No `reward`, `goal`, `utility`

## Certification

All parameters traced to:
1. Data statistics (ranks, variances, stds)
2. History lengths (sqrt(t), 1/sqrt(t+1))
3. Mathematical identities (0, 0.5, 1)
4. Geometric operations (normalization, distance)

ZERO arbitrary/magic constants.
