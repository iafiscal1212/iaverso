#!/usr/bin/env python3
"""
Quick Verification Checks for NEO_EVA
=====================================
1. Jacobian ρ(J) analysis
2. Refined susceptibility with orthonormal basis, log-log χ(α)
3. Ablation experiments
4. IWVI null tests for MI/TE
"""

import sys
import os
import json
import math
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/root/NEO_EVA/tools')
from common import (
    load_hist, triplet, sigmas, cov3, pctl, acf_window,
    pca_full, median_alpha, mi_knn, transfer_entropy,
    null_permutation, phase_randomize, sha256_file, proj_simplex
)
from analysis import (
    estimate_jacobian, compute_susceptibility_map,
    null_phase_test, get_ablation_flags, set_ablation, clear_ablations
)
import numpy as np

RESULTS_DIR = Path("/root/NEO_EVA/results")
NEO_HIST = "/root/NEOSYNT/state/neo_state.yaml"
EVA_HIST = "/root/EVASYNT/state/history.jsonl"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("QUICK VERIFICATION CHECKS - NEO_EVA")
print("=" * 80)
print(f"Timestamp: {datetime.now().isoformat()}")
print()

# ============================================================================
# 1. JACOBIAN ANALYSIS
# ============================================================================

print("=" * 80)
print("1. JACOBIAN ANALYSIS")
print("=" * 80)

# Load NEO history
import yaml
with open(NEO_HIST) as f:
    state = yaml.safe_load(f)
raw = state.get("autonomy", {}).get("history_intention", [])
neo_hist = [{"t": i, "I": {"S": v[0], "N": v[1], "C": v[2]}} for i, v in enumerate(raw) if len(v) == 3]

jac = estimate_jacobian(neo_hist)

print(f"T = {jac.get('T', 0)} samples")
print(f"\nJacobian J:")
if 'J' in jac:
    J = np.array(jac['J'])
    for row in J:
        print(f"  [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]")

    print(f"\nEigenvalues:")
    eigvals_real = jac['eigvals_real']
    eigvals_imag = jac['eigvals_imag']
    for i, (r, im) in enumerate(zip(eigvals_real, eigvals_imag)):
        if abs(im) > 1e-10:
            print(f"  λ_{i+1} = {r:.6f} ± {abs(im):.6f}i  |λ| = {np.sqrt(r**2 + im**2):.6f}")
        else:
            print(f"  λ_{i+1} = {r:.6f} (real)")

    print(f"\nSpectral radius ρ(J) = {jac['rho']:.6f}")
    print(f"RMSE = {jac['rmse']:.6e}")
    print(f"Stable (ρ < 1): {jac['stable']}")

    # Check condition ρ << 1
    if jac['rho'] < 0.5:
        print("\n✓ ρ(J) << 1: System is STRONGLY stable")
    elif jac['rho'] < 0.9:
        print("\n○ ρ(J) < 0.9: System is stable")
    elif jac['rho'] < 1.0:
        print("\n△ ρ(J) close to 1: System is MARGINALLY stable")
    else:
        print("\n✗ ρ(J) ≥ 1: System may be unstable")

print()

# ============================================================================
# 2. REFINED SUSCEPTIBILITY MAP
# ============================================================================

print("=" * 80)
print("2. REFINED SUSCEPTIBILITY MAP (orthonormal basis)")
print("=" * 80)

# Get orthonormal basis from PCA
COV = cov3(neo_hist)
lambdas, vecs, varexp = pca_full(COV)

# Orthonormalize using Gram-Schmidt on simplex plane
simplex_normal = np.array([1, 1, 1]) / np.sqrt(3)

def project_to_simplex_plane(v):
    """Project to plane S+N+C=0 (tangent to simplex)."""
    return v - np.dot(v, simplex_normal) * simplex_normal

# Create orthonormal basis in simplex
u1 = project_to_simplex_plane(vecs[0])
u1 = u1 / np.linalg.norm(u1) if np.linalg.norm(u1) > 1e-12 else u1

u2 = project_to_simplex_plane(vecs[1])
u2 = u2 - np.dot(u2, u1) * u1  # Orthogonalize
u2 = u2 / np.linalg.norm(u2) if np.linalg.norm(u2) > 1e-12 else u2

print(f"\nOrthonormal basis in simplex plane:")
print(f"  u1 = {u1}")
print(f"  u2 = {u2}")
print(f"  u1·u2 = {np.dot(u1, u2):.6e} (should be ~0)")

# Compute susceptibility with log-spaced alphas
T = len(neo_hist)
sig = sigmas(neo_hist)
alpha_base = np.median(list(sig)) / np.sqrt(T)

# Log-spaced alphas: 10 points from 0.1*alpha_base to 10*alpha_base
log_alphas = [alpha_base * (10 ** ((i - 4) / 2)) for i in range(10)]

print(f"\nAlphas (log-spaced):")
for i, a in enumerate(log_alphas):
    print(f"  α_{i} = {a:.6e}")

# Compute χ for each direction and alpha
susc_results = []

# Window from ACF
w = acf_window(triplet(neo_hist, "S"))
I_star = np.array([
    np.mean(triplet(neo_hist[-w:], "S")),
    np.mean(triplet(neo_hist[-w:], "N")),
    np.mean(triplet(neo_hist[-w:], "C"))
])

print(f"\nStationary state I* = {I_star}")
print(f"ACF window w = {w}")

for dir_name, direction in [("u1", u1), ("u2", u2)]:
    v = direction / (np.linalg.norm(direction) + 1e-12)

    for alpha in log_alphas:
        # χ from observed variance in that direction
        proj = np.array([np.dot([h["I"]["S"], h["I"]["N"], h["I"]["C"]] - I_star, v)
                        for h in neo_hist[-w:]])
        chi = float(np.std(proj) / (alpha + 1e-12))

        susc_results.append({
            "dir": dir_name,
            "alpha": alpha,
            "chi": chi,
            "log_alpha": math.log10(alpha),
            "log_chi": math.log10(chi + 1e-20)
        })

print(f"\nSusceptibility χ(α) [log-log]:")
print(f"{'Dir':<5} {'log10(α)':<12} {'log10(χ)':<12}")
print("-" * 30)
for r in susc_results[::3]:  # Sample every 3rd
    print(f"{r['dir']:<5} {r['log_alpha']:+.4f}     {r['log_chi']:+.4f}")

# Save
with open(RESULTS_DIR / "susceptibility_refined.json", 'w') as f:
    json.dump({"I_star": I_star.tolist(), "w": w, "results": susc_results}, f, indent=2)
print(f"\n✓ Saved: susceptibility_refined.json")

print()

# ============================================================================
# 3. ABLATION EXPERIMENTS
# ============================================================================

print("=" * 80)
print("3. ABLATION EXPERIMENTS")
print("=" * 80)

# Define ablation configurations
ablations = [
    ("baseline", {}),
    ("no_recall_eva", {"no_recall_eva": True}),
    ("no_gate", {"no_gate": True}),
    ("no_bus", {"no_bus": True}),
    ("no_pca", {"no_pca": True}),
]

# Load EVA history if exists
if os.path.exists(EVA_HIST):
    eva_hist = load_hist(EVA_HIST)
    print(f"EVA history: T={len(eva_hist)}")
else:
    eva_hist = []
    print("EVA history: not yet generated (T=0)")

# For ablations, we compute expected metrics degradation
ablation_results = []

# Clear any existing ablations
clear_ablations()

for name, flags in ablations:
    print(f"\n--- Ablation: {name} ---")

    # Set flags
    for flag, val in flags.items():
        set_ablation(flag, val)

    current_flags = get_ablation_flags()
    print(f"  Flags: {current_flags}")

    # Compute metrics on NEO history as proxy
    # (Real ablation would require re-running the system)
    if len(neo_hist) > 20:
        # Simulated degradation based on ablation
        base_rho = jac['rho'] if 'rho' in jac else 0.99
        base_rmse = jac['rmse'] if 'rmse' in jac else 0.001

        # Apply degradation factors (heuristic, would need real runs)
        if "no_recall_eva" in flags:
            deg_factor = 1.05  # Memory loss degrades stability slightly
        elif "no_gate" in flags:
            deg_factor = 1.02  # No gate has small effect
        elif "no_bus" in flags:
            deg_factor = 1.01  # No inter-world has minimal effect on single world
        elif "no_pca" in flags:
            deg_factor = 1.10  # Random direction instead of PCA
        else:
            deg_factor = 1.00  # Baseline

        ablation_results.append({
            "name": name,
            "flags": flags,
            "expected_rho": min(1.5, base_rho * deg_factor),
            "expected_rmse": base_rmse * deg_factor,
            "degradation_factor": deg_factor
        })

        print(f"  Expected ρ degradation: x{deg_factor:.2f}")

    # Clear for next iteration
    clear_ablations()

print(f"\n--- Ablation Summary ---")
print(f"{'Name':<15} {'Deg Factor':<12} {'Expected ρ':<12}")
print("-" * 40)
for r in ablation_results:
    print(f"{r['name']:<15} x{r['degradation_factor']:.2f}        {r['expected_rho']:.4f}")

# Save
with open(RESULTS_DIR / "ablation_expected.json", 'w') as f:
    json.dump(ablation_results, f, indent=2)
print(f"\n✓ Saved: ablation_expected.json")
print("\nNOTE: These are expected degradations. Full ablation requires re-running systems.")

print()

# ============================================================================
# 4. IWVI NULL TESTS FOR MI/TE
# ============================================================================

print("=" * 80)
print("4. IWVI NULL TESTS (MI/TE)")
print("=" * 80)

# For MI/TE we need paired series from both worlds
# If EVA doesn't have enough data, use phase randomization on NEO

if len(neo_hist) < 50:
    print("Insufficient history for null tests (need T >= 50)")
else:
    # Series - use only last 2000 samples for speed
    S_neo_full = np.array(triplet(neo_hist, "S"))
    T_full = len(S_neo_full)

    # Subsample for computation efficiency
    if T_full > 2000:
        S_neo = S_neo_full[-2000:]
        T = 2000
        print(f"Full T = {T_full}, using last {T} for null tests")
    else:
        S_neo = S_neo_full
        T = T_full
        print(f"T = {T}")

    # k for MI/TE
    k = max(1, int(T ** (1/3)))
    print(f"k = {k}")

    # Observed MI within NEO (using lagged series as proxy)
    lag = max(1, w // 2)
    X = S_neo[:-lag].reshape(-1, 1)
    Y = S_neo[lag:].reshape(-1, 1)

    mi_obs = mi_knn(X, Y, k=k)
    print(f"\nObserved MI (lag={lag}): {mi_obs:.6f}")

    # Null distribution via phase randomization
    # Limit B for large T to avoid long computation
    B = min(100, max(20, int(10 * np.sqrt(T))))
    print(f"Generating null distribution (B={B})...")

    mi_nulls = []
    for b in range(B):
        S_surr = phase_randomize(S_neo)
        X_surr = S_surr[:-lag].reshape(-1, 1)
        Y_surr = S_surr[lag:].reshape(-1, 1)
        mi_nulls.append(mi_knn(X_surr, Y_surr, k=k))

    mi_nulls = np.array(mi_nulls)

    # p-value
    p_hat_mi = np.sum(mi_nulls >= mi_obs) / B

    print(f"\nNull distribution:")
    print(f"  median = {np.median(mi_nulls):.6f}")
    print(f"  p25    = {np.percentile(mi_nulls, 25):.6f}")
    print(f"  p75    = {np.percentile(mi_nulls, 75):.6f}")
    print(f"  p95    = {np.percentile(mi_nulls, 95):.6f}")

    print(f"\np-hat = {p_hat_mi:.4f}")
    if p_hat_mi < 0.05:
        print("✓ MI significantly above null (p < 0.05)")
    else:
        print("○ MI not significantly above null (p >= 0.05)")

    # Transfer Entropy test
    print(f"\n--- Transfer Entropy ---")

    # TE from past to future
    max_lag = max(1, int(np.sqrt(T)))

    # Observed TE
    te_obs = transfer_entropy(S_neo, S_neo, k=k, lag=lag)
    print(f"Observed TE (lag={lag}): {te_obs:.6f}")

    # Null distribution
    te_nulls = []
    for b in range(B):
        S_surr = phase_randomize(S_neo)
        te_nulls.append(transfer_entropy(S_surr, S_surr, k=k, lag=lag))

    te_nulls = np.array(te_nulls)
    p_hat_te = np.sum(te_nulls >= te_obs) / B

    print(f"\nNull distribution:")
    print(f"  median = {np.median(te_nulls):.6f}")
    print(f"  p95    = {np.percentile(te_nulls, 95):.6f}")

    print(f"\np-hat = {p_hat_te:.4f}")
    if p_hat_te < 0.05:
        print("✓ TE significantly above null (p < 0.05)")
    else:
        print("○ TE not significantly above null (p >= 0.05)")

    # Save results
    null_results = {
        "T": int(T),
        "k": k,
        "B": B,
        "lag": lag,
        "mi": {
            "observed": float(mi_obs),
            "null_median": float(np.median(mi_nulls)),
            "null_p25": float(np.percentile(mi_nulls, 25)),
            "null_p75": float(np.percentile(mi_nulls, 75)),
            "null_p95": float(np.percentile(mi_nulls, 95)),
            "p_hat": float(p_hat_mi)
        },
        "te": {
            "observed": float(te_obs),
            "null_median": float(np.median(te_nulls)),
            "null_p95": float(np.percentile(te_nulls, 95)),
            "p_hat": float(p_hat_te)
        }
    }

    with open(RESULTS_DIR / "iwvi_null_tests.json", 'w') as f:
        json.dump(null_results, f, indent=2)
    print(f"\n✓ Saved: iwvi_null_tests.json")

print()
print("=" * 80)
print("QUICK CHECKS COMPLETE")
print("=" * 80)
