#!/usr/bin/env python3
"""
Phase 6 IWVI Analysis
=====================

Analyzes the coupled NEO↔EVA series for:
- Mutual Information (MI)
- Transfer Entropy (TE)
- Valid windows with variance threshold
- Phase null tests

Success: MI or TE > null in ≥1 window (p̂ ≤ 0.05)
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from scipy.spatial import KDTree

sys.path.insert(0, '/root/NEO_EVA/tools')


def load_coupled_series(neo_path: str, eva_path: str):
    """Load coupled series from JSON files."""
    with open(neo_path) as f:
        neo_data = json.load(f)
    with open(eva_path) as f:
        eva_data = json.load(f)

    I_neo = np.array([[s['S_new'], s['N_new'], s['C_new']] for s in neo_data['series']])
    I_eva = np.array([[s['S_new'], s['N_new'], s['C_new']] for s in eva_data['series']])

    return I_neo, I_eva, neo_data, eva_data


def knn_mutual_information(X: np.ndarray, Y: np.ndarray, k: int = None) -> float:
    """k-NN estimator of mutual information I(X;Y) using KSG."""
    T = len(X)
    if T < 20:
        return 0.0

    if k is None:
        k = max(3, int(T ** (1/3)))

    if X.ndim > 1:
        X = X.reshape(len(X), -1)
    if Y.ndim > 1:
        Y = Y.reshape(len(Y), -1)

    XY = np.hstack([X, Y])

    tree_xy = KDTree(XY)
    tree_x = KDTree(X)
    tree_y = KDTree(Y)

    digamma = lambda x: float(np.log(x + 1e-10)) if x > 0 else -20.0

    mi_sum = 0.0
    for i in range(T):
        dists_xy, _ = tree_xy.query(XY[i], k=k+1)
        eps = dists_xy[-1]

        n_x = len(tree_x.query_ball_point(X[i], eps + 1e-10)) - 1
        n_y = len(tree_y.query_ball_point(Y[i], eps + 1e-10)) - 1

        mi_sum += digamma(n_x) + digamma(n_y)

    psi_k = digamma(k)
    psi_T = digamma(T)

    MI = psi_k - mi_sum / T + psi_T

    return max(0.0, MI)


def transfer_entropy(source: np.ndarray, target: np.ndarray, lag: int = 1, k: int = None) -> float:
    """Transfer entropy from source to target: TE(X→Y)."""
    T = len(source)
    if T < lag + 20:
        return 0.0

    if k is None:
        k = max(3, int((T - lag) ** (1/3)))

    Y_t = target[lag:]
    Y_past = target[:-lag]
    X_past = source[:-lag]

    if Y_t.ndim > 1:
        Y_t = Y_t.reshape(len(Y_t), -1)
    if Y_past.ndim > 1:
        Y_past = Y_past.reshape(len(Y_past), -1)
    if X_past.ndim > 1:
        X_past = X_past.reshape(len(X_past), -1)

    YX_past = np.hstack([Y_past, X_past])
    MI_joint = knn_mutual_information(Y_t, YX_past, k=k)
    MI_past = knn_mutual_information(Y_t, Y_past, k=k)

    TE = MI_joint - MI_past
    return max(0.0, TE)


def phase_randomize(series: np.ndarray) -> np.ndarray:
    """
    Phase randomization for null test.

    Preserves power spectrum but destroys phase relationships.
    """
    if series.ndim == 1:
        series = series.reshape(-1, 1)

    T, d = series.shape
    randomized = np.zeros_like(series)

    for col in range(d):
        fft = np.fft.rfft(series[:, col])
        phases = np.angle(fft)
        magnitudes = np.abs(fft)

        # Randomize phases
        random_phases = np.random.uniform(-np.pi, np.pi, len(phases))
        random_phases[0] = 0  # Keep DC component real
        if T % 2 == 0:
            random_phases[-1] = 0  # Keep Nyquist real for even T

        new_fft = magnitudes * np.exp(1j * random_phases)
        randomized[:, col] = np.fft.irfft(new_fft, n=T)

    return randomized


def compute_valid_windows(I_neo: np.ndarray, I_eva: np.ndarray, window_size: int = None):
    """
    Compute valid IWVI windows.

    A window is valid if Var(I) ≥ p50(Var_hist) for both worlds.
    """
    T = len(I_neo)
    if window_size is None:
        window_size = max(20, int(np.sqrt(T)))

    # Compute variance for all windows
    step = window_size // 2
    windows = []

    for i in range(0, T - window_size, step):
        neo_win = I_neo[i:i+window_size]
        eva_win = I_eva[i:i+window_size]

        var_neo = np.var(neo_win, axis=0).sum()
        var_eva = np.var(eva_win, axis=0).sum()

        windows.append({
            'start': i,
            'end': i + window_size,
            'var_neo': var_neo,
            'var_eva': var_eva,
        })

    if not windows:
        return [], 0

    # Compute p50 thresholds
    all_var_neo = [w['var_neo'] for w in windows]
    all_var_eva = [w['var_eva'] for w in windows]

    p50_neo = np.percentile(all_var_neo, 50)
    p50_eva = np.percentile(all_var_eva, 50)

    # Mark valid windows
    valid_windows = []
    for w in windows:
        if w['var_neo'] >= p50_neo and w['var_eva'] >= p50_eva:
            w['valid'] = True
            valid_windows.append(w)
        else:
            w['valid'] = False

    return windows, len(valid_windows)


def run_iwvi_analysis(neo_path: str, eva_path: str, B: int = None):
    """Run IWVI analysis on coupled series."""
    print("=" * 70)
    print("Phase 6 IWVI Analysis")
    print("=" * 70)

    # Load series
    print("\n[1] Loading coupled series...")
    I_neo, I_eva, neo_data, eva_data = load_coupled_series(neo_path, eva_path)
    T = min(len(I_neo), len(I_eva))
    I_neo = I_neo[:T]
    I_eva = I_eva[:T]

    print(f"  NEO: {len(I_neo)} points")
    print(f"  EVA: {len(I_eva)} points")

    # B = floor(10√T)
    if B is None:
        B = int(10 * np.sqrt(T))
    k = max(3, int(T ** (1/3)))

    print(f"  k (kNN): {k}")
    print(f"  B (nulls): {B}")

    # Variance check
    var_neo = np.var(I_neo, axis=0).sum()
    var_eva = np.var(I_eva, axis=0).sum()
    print(f"\n[2] Variance check...")
    print(f"  NEO total variance: {var_neo:.6e}")
    print(f"  EVA total variance: {var_eva:.6e}")

    # Compute valid windows
    print(f"\n[3] Computing valid windows...")
    windows, n_valid = compute_valid_windows(I_neo, I_eva)
    print(f"  Total windows: {len(windows)}")
    print(f"  Valid windows: {n_valid} ({100*n_valid/max(1,len(windows)):.1f}%)")

    # Compute observed MI and TE
    print(f"\n[4] Computing MI and TE...")
    MI_observed = knn_mutual_information(I_neo, I_eva, k=k)
    TE_neo_to_eva = transfer_entropy(I_neo, I_eva, lag=1, k=k)
    TE_eva_to_neo = transfer_entropy(I_eva, I_neo, lag=1, k=k)

    print(f"  MI(NEO, EVA) = {MI_observed:.6f}")
    print(f"  TE(NEO → EVA) = {TE_neo_to_eva:.6f}")
    print(f"  TE(EVA → NEO) = {TE_eva_to_neo:.6f}")

    # Phase null tests
    print(f"\n[5] Running phase null tests (B={B})...")
    MI_nulls = []
    TE_neo_nulls = []
    TE_eva_nulls = []

    for b in range(B):
        # Phase randomize EVA
        I_eva_rand = phase_randomize(I_eva)

        MI_null = knn_mutual_information(I_neo, I_eva_rand, k=k)
        MI_nulls.append(MI_null)

        TE_null_neo = transfer_entropy(I_neo, I_eva_rand, lag=1, k=k)
        TE_null_eva = transfer_entropy(I_eva_rand, I_neo, lag=1, k=k)
        TE_neo_nulls.append(TE_null_neo)
        TE_eva_nulls.append(TE_null_eva)

        if (b + 1) % 20 == 0:
            print(f"    {b+1}/{B} done...")

    MI_nulls = np.array(MI_nulls)
    TE_neo_nulls = np.array(TE_neo_nulls)
    TE_eva_nulls = np.array(TE_eva_nulls)

    # Compute p-values (one-sided: observed > null)
    p_MI = np.mean(MI_nulls >= MI_observed)
    p_TE_neo = np.mean(TE_neo_nulls >= TE_neo_to_eva)
    p_TE_eva = np.mean(TE_eva_nulls >= TE_eva_to_neo)

    print(f"\n[6] Results:")
    print(f"  MI observed: {MI_observed:.6f}")
    print(f"  MI null median: {np.median(MI_nulls):.6f}")
    print(f"  MI null std: {np.std(MI_nulls):.6f}")
    print(f"  MI p-value: {p_MI:.4f}")
    print(f"  MI significant (p ≤ 0.05): {p_MI <= 0.05}")

    print(f"\n  TE(NEO→EVA) observed: {TE_neo_to_eva:.6f}")
    print(f"  TE(NEO→EVA) null median: {np.median(TE_neo_nulls):.6f}")
    print(f"  TE(NEO→EVA) p-value: {p_TE_neo:.4f}")
    print(f"  TE(NEO→EVA) significant: {p_TE_neo <= 0.05}")

    print(f"\n  TE(EVA→NEO) observed: {TE_eva_to_neo:.6f}")
    print(f"  TE(EVA→NEO) null median: {np.median(TE_eva_nulls):.6f}")
    print(f"  TE(EVA→NEO) p-value: {p_TE_eva:.4f}")
    print(f"  TE(EVA→NEO) significant: {p_TE_eva <= 0.05}")

    # Window-by-window analysis
    print(f"\n[7] Window-by-window MI analysis...")
    window_results = []
    significant_windows = 0

    for w in windows:
        if not w.get('valid', False):
            continue

        neo_win = I_neo[w['start']:w['end']]
        eva_win = I_eva[w['start']:w['end']]

        mi_win = knn_mutual_information(neo_win, eva_win, k=max(3, len(neo_win)//5))

        # Quick null for this window
        mi_nulls_win = []
        for _ in range(min(B, 50)):
            eva_rand = phase_randomize(eva_win)
            mi_nulls_win.append(knn_mutual_information(neo_win, eva_rand, k=max(3, len(neo_win)//5)))

        p_win = np.mean(np.array(mi_nulls_win) >= mi_win)

        window_results.append({
            'start': w['start'],
            'end': w['end'],
            'mi': float(mi_win),
            'p_value': float(p_win),
            'significant': bool(p_win <= 0.05),
        })

        if p_win <= 0.05:
            significant_windows += 1

    print(f"  Valid windows analyzed: {len(window_results)}")
    print(f"  Significant windows (p ≤ 0.05): {significant_windows}")

    # Summary
    any_significant = bool((p_MI <= 0.05) or (p_TE_neo <= 0.05) or (p_TE_eva <= 0.05) or (significant_windows > 0))

    results = {
        'timestamp': datetime.now().isoformat(),
        'T': T,
        'k': k,
        'B': B,
        'mi': {
            'observed': float(MI_observed),
            'null_median': float(np.median(MI_nulls)),
            'null_std': float(np.std(MI_nulls)),
            'p_hat': float(p_MI),
            'significant': bool(p_MI <= 0.05),
        },
        'te_neo_to_eva': {
            'observed': float(TE_neo_to_eva),
            'null_median': float(np.median(TE_neo_nulls)),
            'p_hat': float(p_TE_neo),
            'significant': bool(p_TE_neo <= 0.05),
        },
        'te_eva_to_neo': {
            'observed': float(TE_eva_to_neo),
            'null_median': float(np.median(TE_eva_nulls)),
            'p_hat': float(p_TE_eva),
            'significant': bool(p_TE_eva <= 0.05),
        },
        'variance': {
            'neo': float(var_neo),
            'eva': float(var_eva),
        },
        'valid_windows': {
            'total': len(windows),
            'valid': n_valid,
            'rate': float(n_valid / max(1, len(windows))),
            'significant': significant_windows,
        },
        'window_results': window_results,
        'success': any_significant,
        'interpretation': (
            f"MI={MI_observed:.4f} (p={p_MI:.3f}), "
            f"TE(N→E)={TE_neo_to_eva:.4f} (p={p_TE_neo:.3f}), "
            f"TE(E→N)={TE_eva_to_neo:.4f} (p={p_TE_eva:.3f}). "
            + ("SUCCESS: Information flow detected!" if any_significant else "No significant signal detected.")
        ),
    }

    # Save results
    out_path = "/root/NEO_EVA/results/phase6_iwvi_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved: {out_path}")

    print(f"\n[8] Summary:")
    print(f"  {results['interpretation']}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6 IWVI Analysis")
    parser.add_argument("--neo", default="/root/NEO_EVA/results/phase6_coupled_neo.json")
    parser.add_argument("--eva", default="/root/NEO_EVA/results/phase6_coupled_eva.json")
    parser.add_argument("--B", type=int, default=None)
    args = parser.parse_args()

    run_iwvi_analysis(args.neo, args.eva, B=args.B)


if __name__ == "__main__":
    main()
