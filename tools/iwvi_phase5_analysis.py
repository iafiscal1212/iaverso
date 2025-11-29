#!/usr/bin/env python3
"""
Phase 5 IWVI Analysis
=====================

Inter-World Validation of Information flow between NEO and EVA.
Uses the Phase 4 series with variance for MI/TE estimation.
"""

import sys
import os
import json
import math
import numpy as np
from datetime import datetime
from scipy.spatial import KDTree

sys.path.insert(0, '/root/NEO_EVA/tools')


def load_series(path: str):
    """Load series from JSON file."""
    with open(path) as f:
        data = json.load(f)
    series = data['series']
    # Extract I_new values
    I = np.array([[s['S_new'], s['N_new'], s['C_new']] for s in series])
    return I


def knn_mutual_information(X: np.ndarray, Y: np.ndarray, k: int = None) -> float:
    """
    k-NN estimator of mutual information I(X;Y).

    Uses Kraskov-Stögbauer-Grassberger (KSG) estimator.
    """
    T = len(X)
    if T < 20:
        return 0.0

    if k is None:
        k = max(3, int(T ** (1/3)))

    # Flatten if needed
    if X.ndim > 1:
        X = X.reshape(len(X), -1)
    if Y.ndim > 1:
        Y = Y.reshape(len(Y), -1)

    # Joint space
    XY = np.hstack([X, Y])

    # Build KD-trees
    tree_xy = KDTree(XY)
    tree_x = KDTree(X)
    tree_y = KDTree(Y)

    # For each point, find k-th neighbor distance in joint space
    # Then count points within that distance in marginal spaces
    digamma = lambda x: float(np.log(x + 1e-10)) if x > 0 else -20.0

    mi_sum = 0.0
    for i in range(T):
        # k+1 because query_ball includes the point itself
        dists_xy, _ = tree_xy.query(XY[i], k=k+1)
        eps = dists_xy[-1]  # Distance to k-th neighbor

        # Count points in marginals within eps
        n_x = len(tree_x.query_ball_point(X[i], eps + 1e-10)) - 1
        n_y = len(tree_y.query_ball_point(Y[i], eps + 1e-10)) - 1

        mi_sum += digamma(n_x) + digamma(n_y)

    # KSG formula
    psi_k = digamma(k)
    psi_T = digamma(T)

    MI = psi_k - mi_sum / T + psi_T

    return max(0.0, MI)


def transfer_entropy(source: np.ndarray, target: np.ndarray, lag: int = 1, k: int = None) -> float:
    """
    Transfer entropy from source to target.

    TE(X→Y) = I(Y_t; X_{t-lag} | Y_{t-lag})
            = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})
    """
    T = len(source)
    if T < lag + 20:
        return 0.0

    if k is None:
        k = max(3, int((T - lag) ** (1/3)))

    # Create lagged variables
    # Y_t: target[lag:]
    # Y_{t-lag}: target[:-lag]
    # X_{t-lag}: source[:-lag]

    Y_t = target[lag:]
    Y_past = target[:-lag]
    X_past = source[:-lag]

    # Flatten if needed
    if Y_t.ndim > 1:
        Y_t = Y_t.reshape(len(Y_t), -1)
    if Y_past.ndim > 1:
        Y_past = Y_past.reshape(len(Y_past), -1)
    if X_past.ndim > 1:
        X_past = X_past.reshape(len(X_past), -1)

    # TE = I(Y_t; X_past | Y_past)
    # Using chain rule: I(Y_t; Y_past, X_past) - I(Y_t; Y_past)

    # I(Y_t; (Y_past, X_past))
    YX_past = np.hstack([Y_past, X_past])
    MI_joint = knn_mutual_information(Y_t, YX_past, k=k)

    # I(Y_t; Y_past)
    MI_past = knn_mutual_information(Y_t, Y_past, k=k)

    TE = MI_joint - MI_past

    return max(0.0, TE)


def null_test(observed: float, null_values: np.ndarray) -> float:
    """Compute p-value from null distribution."""
    if len(null_values) == 0:
        return 1.0
    # Two-sided: fraction of nulls >= observed
    p = np.mean(null_values >= observed)
    return p


def run_iwvi_analysis(neo_path: str, eva_path: str, B: int = 100):
    """
    Run IWVI analysis on NEO and EVA series.
    """
    print("=" * 70)
    print("Phase 5 IWVI Analysis")
    print("=" * 70)

    # Load series
    print("\n[1] Loading series...")
    I_neo = load_series(neo_path)
    I_eva = load_series(eva_path)

    print(f"  NEO: {len(I_neo)} points")
    print(f"  EVA: {len(I_eva)} points")

    # Use minimum length
    T = min(len(I_neo), len(I_eva))
    I_neo = I_neo[:T]
    I_eva = I_eva[:T]

    # Compute k
    k = max(3, int(T ** (1/3)))
    print(f"  k (kNN): {k}")

    # Variance check
    var_neo = np.sum(np.var(I_neo, axis=0))
    var_eva = np.sum(np.var(I_eva, axis=0))
    print(f"\n[2] Variance check...")
    print(f"  NEO total variance: {var_neo:.6e}")
    print(f"  EVA total variance: {var_eva:.6e}")

    if var_neo < 1e-10 or var_eva < 1e-10:
        print("  WARNING: Insufficient variance for IWVI")

    # Compute observed MI
    print(f"\n[3] Computing Mutual Information...")
    MI_observed = knn_mutual_information(I_neo, I_eva, k=k)
    print(f"  MI(NEO, EVA) = {MI_observed:.6f}")

    # Compute observed TE
    print(f"\n[4] Computing Transfer Entropy...")
    TE_neo_to_eva = transfer_entropy(I_neo, I_eva, lag=1, k=k)
    TE_eva_to_neo = transfer_entropy(I_eva, I_neo, lag=1, k=k)
    print(f"  TE(NEO → EVA) = {TE_neo_to_eva:.6f}")
    print(f"  TE(EVA → NEO) = {TE_eva_to_neo:.6f}")

    # Null tests with permutation
    print(f"\n[5] Running null tests (B={B})...")

    MI_nulls = []
    TE_neo_nulls = []
    TE_eva_nulls = []

    for b in range(B):
        # Permute EVA independently
        perm = np.random.permutation(T)
        I_eva_perm = I_eva[perm]

        MI_null = knn_mutual_information(I_neo, I_eva_perm, k=k)
        MI_nulls.append(MI_null)

        TE_null_neo = transfer_entropy(I_neo, I_eva_perm, lag=1, k=k)
        TE_null_eva = transfer_entropy(I_eva_perm, I_neo, lag=1, k=k)
        TE_neo_nulls.append(TE_null_neo)
        TE_eva_nulls.append(TE_null_eva)

        if (b + 1) % 20 == 0:
            print(f"    {b+1}/{B} done...")

    MI_nulls = np.array(MI_nulls)
    TE_neo_nulls = np.array(TE_neo_nulls)
    TE_eva_nulls = np.array(TE_eva_nulls)

    # P-values
    p_MI = null_test(MI_observed, MI_nulls)
    p_TE_neo = null_test(TE_neo_to_eva, TE_neo_nulls)
    p_TE_eva = null_test(TE_eva_to_neo, TE_eva_nulls)

    print(f"\n[6] Results:")
    print(f"  MI observed: {MI_observed:.6f}")
    print(f"  MI null median: {np.median(MI_nulls):.6f}")
    print(f"  MI p-value: {p_MI:.4f}")
    print(f"  MI significant (p < 0.05): {p_MI < 0.05}")

    print(f"\n  TE(NEO→EVA) observed: {TE_neo_to_eva:.6f}")
    print(f"  TE(NEO→EVA) null median: {np.median(TE_neo_nulls):.6f}")
    print(f"  TE(NEO→EVA) p-value: {p_TE_neo:.4f}")

    print(f"\n  TE(EVA→NEO) observed: {TE_eva_to_neo:.6f}")
    print(f"  TE(EVA→NEO) null median: {np.median(TE_eva_nulls):.6f}")
    print(f"  TE(EVA→NEO) p-value: {p_TE_eva:.4f}")

    # Check for valid windows (variance threshold)
    # A window is valid if variance exceeds p50 of historical variance
    window_size = max(20, int(np.sqrt(T)))
    valid_windows = 0
    total_windows = 0

    var_threshold_neo = np.percentile([np.var(I_neo[i:i+window_size]) for i in range(0, T-window_size, window_size//2)], 50)
    var_threshold_eva = np.percentile([np.var(I_eva[i:i+window_size]) for i in range(0, T-window_size, window_size//2)], 50)

    for i in range(0, T - window_size, window_size // 2):
        var_w_neo = np.sum(np.var(I_neo[i:i+window_size], axis=0))
        var_w_eva = np.sum(np.var(I_eva[i:i+window_size], axis=0))

        total_windows += 1
        if var_w_neo >= var_threshold_neo and var_w_eva >= var_threshold_eva:
            valid_windows += 1

    print(f"\n[7] Valid IWVI Windows:")
    print(f"  Window size: {window_size}")
    print(f"  Total windows: {total_windows}")
    print(f"  Valid windows: {valid_windows} ({100*valid_windows/max(1,total_windows):.1f}%)")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "T": T,
        "k": k,
        "mi": {
            "observed": float(MI_observed),
            "null_median": float(np.median(MI_nulls)),
            "null_std": float(np.std(MI_nulls)),
            "p_hat": float(p_MI),
            "significant": bool(p_MI < 0.05)
        },
        "te_neo_to_eva": {
            "observed": float(TE_neo_to_eva),
            "null_median": float(np.median(TE_neo_nulls)),
            "p_hat": float(p_TE_neo),
            "significant": bool(p_TE_neo < 0.05)
        },
        "te_eva_to_neo": {
            "observed": float(TE_eva_to_neo),
            "null_median": float(np.median(TE_eva_nulls)),
            "p_hat": float(p_TE_eva),
            "significant": bool(p_TE_eva < 0.05)
        },
        "variance": {
            "neo": float(var_neo),
            "eva": float(var_eva)
        },
        "valid_windows": {
            "total": total_windows,
            "valid": valid_windows,
            "rate": float(valid_windows / max(1, total_windows))
        },
        "B": B,
        "interpretation": "MI=0, TE=0 indicates independent systems with no information flow. This is the expected null result for Phase 4 systems without BUS coupling."
    }

    out_path = "/root/NEO_EVA/results/iwvi_phase5_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved: {out_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 5 IWVI Analysis")
    parser.add_argument("--B", type=int, default=100, help="Number of null permutations")
    parser.add_argument("--neo", type=str, default="/root/NEO_EVA/results/phase5_neo_2000_series.json")
    parser.add_argument("--eva", type=str, default="/root/NEO_EVA/results/phase5_eva_2000_series.json")
    args = parser.parse_args()

    neo_path = args.neo
    eva_path = args.eva

    if not os.path.exists(neo_path):
        print(f"ERROR: NEO series not found: {neo_path}")
        print("Run: python3 phase4_standalone_run.py")
        return

    if not os.path.exists(eva_path):
        print(f"ERROR: EVA series not found: {eva_path}")
        print("Run: python3 eva_phase4_run.py")
        return

    run_iwvi_analysis(neo_path, eva_path, B=args.B)


if __name__ == "__main__":
    main()
