# Endogeneity Audit Report
_Generated: 2025-11-29T20:45:18.722170Z_

## Compliance Status: ✅ GO

### Module Status
| Module | Status |
|--------|--------|
| Static Audit | ✅ PASS (0 violations) |
| Dynamic Audit | ✅ PASS (2/2 tests) |
| Coupling Audit | ✅ PASS |

## Formula Derivations
| Parameter | Formula |
|-----------|---------|
| w (window) | `max{10, ⌊√T⌋}` |
| max_hist | `min{T, ⌊10√T⌋}` |
| σ_med | `median(σ_S, σ_N, σ_C) in window` |
| τ | `IQR(r)/√T × σ_med/(IQR_hist + ε)` |
| τ_floor | `σ_med / T` |
| η | `τ (no boost)` |
| drift | `Proj_Tan(EMA of (I_{k+1} - I_k))` |
| σ_noise | `max{IQR(I), σ_med} / √T` |
| OU limits | `clip(Z, q_{0.001}, q_{0.999}) or m ± 4×MAD` |
| gate | `ρ ≥ ρ_p95 AND IQR ≥ IQR_p75 (pure quantiles)` |
| κ | `(u_Y/(1+u_X)) × (λ₁^Y/(λ₁^Y+λ₁^X+ε)) × (conf^Y/(1+CV(r^X)))` |

## Static Audit
- Total findings: 0
- Violations: 0
- Reviews: 0
- OK (tolerance): 0
- OK (geometric): 0

## Dynamic Audit
- Tests run: 2
- Passed: 2
- Failed: 0

## Coupling Audit (κ_t)
- Examples captured: 5
- No magic constants: Yes

### κ Distribution
- p50: 0.0000
- p75: 0.0000
- p95: 0.0000
- mean: 0.0000
- n: 48

## File Hashes (SHA256)
| File | Hash |
|------|------|
| phase6_coupled_system_v2.py | `47ab60205cc65008` |

---
_Report generated: 2025-11-29T20:45:18.722170Z_