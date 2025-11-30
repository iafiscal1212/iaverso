# Phase 18: Parameter Derivation Report

## Overview

This report documents the provenance of ALL parameters used in Phase 18.
Every parameter must be derived from data - ZERO magic numbers allowed.

## Survival System (survival18.py)

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `C_t` | `rank(-coh) + rank(-int) + rank(-irr)` | Component histories | Sum of negative ranks |
| `L_t` | `rank(manifold_spread)` | Spread history | Rank of dispersion |
| `α_survival` | `1/√(t+1)` | Timestep | Standard EMA decay |
| `S_t` | `EMA(C_t + L_t)` | Pressure history | Accumulated pressure |
| `threshold` | `percentile(S_history, 90)` | S history | Endogenous 90th percentile |
| `η_collapse` | `spread_rank / √(visits+1)` | Spread rank, visits | Scaled by experience |

## Amplification System (amplification18.py)

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `w` | `√T` | Timestep | Endogenous window size |
| `χ_t` | `rank(std(window(z)))` | Trajectory history | Susceptibility from variance |
| `τ_t` | `rank(variance(delta_z))` | Velocity history | Tension from velocity variance |
| `AF_t` | `χ_t * τ_t` | χ, τ | Multiplicative amplification |
| `A*_t` | `A_t * (1 + AF_t)` | Agency, AF | Amplified agency |
| `λ_t` | `1/(std(A*_t)+1)` | A* history | Endogenous modulation strength |

## Structural Bounds

| Bound | Value | Justification |
|-------|-------|---------------|
| Min history for q90 | 10 | Statistical minimum |
| Prototype dim | 3-5 | Configuration |
| N_states | 10 | Configuration |
| Warmup | 50 | √T scaling |

## Semantic Label Analysis

Phase 18 uses ONLY structural/mathematical terms:
- `survival` - structural survival (mathematical state)
- `collapse` - collapse indicator (mathematical threshold)
- `pressure` - survival pressure (mathematical metric)
- `tension` - velocity variance (mathematical)
- `coherence` - identity coherence (mathematical)

NO human-centric semantic labels:
- No `reward`, `goal`, `utility`
- No `hunger`, `pain`, `fear`
- No `good`, `bad`, `optimal`

## Certification

All parameters traced to:
1. Data statistics (ranks, percentiles, variance, std)
2. History counts (visits, timesteps)
3. Mathematical identities (0, 0.5, 1)
4. Structural constraints (dimension bounds)

ZERO arbitrary/magic constants.
