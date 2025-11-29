#!/usr/bin/env python3
"""
NEO Phase 4 Injector
====================

Runs NEO cycles with Phase 4 perturbations ACTUALLY INJECTED via the /say endpoint.
This modifies the intention vector to generate real variance.

Strategy:
1. Compute Phase 4 perturbation (δ in tangent plane)
2. Convert δ to influence command via /say
3. NEO applies the influence, generating variance
4. Record series for IWVI analysis
"""

import sys
import os
import json
import math
import time
import urllib.request
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, '/root/NEO_EVA/tools')
from phase4_variability import Phase4Controller

# Config
NEO_URL = "http://127.0.0.1:7777"


def server_request(endpoint, method='GET', data=None, timeout=10):
    """Make request to NEO server."""
    url = f'{NEO_URL}{endpoint}'
    try:
        if data:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode(),
                headers={'Content-Type': 'application/json'},
                method=method
            )
        else:
            req = urllib.request.Request(url, method=method)
        response = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(response.read().decode())
    except Exception as e:
        print(f"  [Error] {endpoint}: {e}")
        return None


def load_neo_history():
    """Load NEO intention history from state file."""
    import yaml
    state_path = '/root/NEOSYNT/state/neo_state.yaml'
    if not os.path.exists(state_path):
        return []

    with open(state_path) as f:
        state = yaml.safe_load(f)

    raw = state.get("autonomy", {}).get("history_intention", [])
    return [np.array(v) for v in raw if len(v) == 3]


def compute_sigmas(I_history):
    """Compute historical sigmas from intention history."""
    if len(I_history) < 10:
        return (1e-6, 1e-6, 1e-6)

    I_arr = np.array(I_history)
    sig_S = float(np.std(I_arr[:, 0]))
    sig_N = float(np.std(I_arr[:, 1]))
    sig_C = float(np.std(I_arr[:, 2]))

    # Floor for extreme equilibrium
    floor = 1e-6
    return (max(sig_S, floor), max(sig_N, floor), max(sig_C, floor))


def delta_to_influence_message(delta, I_current):
    """
    Convert Phase 4 delta to an influence message for /say endpoint.

    Maps delta components to keywords that the server recognizes.
    """
    # Analyze delta direction
    d_S, d_N, d_C = delta[0], delta[1], delta[2]

    # Build message based on dominant delta component
    messages = []

    if d_S > 0.001:
        messages.append("sorprende explora descubre")
    elif d_S < -0.001:
        messages.append("calma equilibrio")

    if d_N > 0.001:
        messages.append("nuevo diferente crea")
    elif d_N < -0.001:
        messages.append("estable calma")

    if d_C > 0.001:
        messages.append("estable equilibrio calma")
    elif d_C < -0.001:
        messages.append("explora busca")

    if not messages:
        # Use generic perturbation
        if np.random.random() > 0.5:
            messages.append("explora nuevo")
        else:
            messages.append("crea diferente")

    return " ".join(messages)


def direct_intention_perturbation(I_current, delta, scale=0.1):
    """
    Create a new intention by applying delta with proper simplex projection.

    Uses mirror descent: I_new = softmax(log(I) + η*δ)
    """
    eps = 1e-10
    I_safe = np.clip(I_current, eps, 1 - eps)

    # Mirror descent step
    log_I = np.log(I_safe)
    log_I_new = log_I + scale * delta

    # Softmax projection back to simplex
    exp_log = np.exp(log_I_new - np.max(log_I_new))
    I_new = exp_log / np.sum(exp_log)

    return I_new


def run_phase4_injection(cycles=100, sample_every=5):
    """Run NEO with Phase 4 perturbations injected."""

    print("=" * 70)
    print("Phase 4 Injection Runner")
    print("=" * 70)
    print(f"Cycles: {cycles}")
    print(f"Sample every: {sample_every}")
    print()

    # Check server
    health = server_request('/health')
    if not health:
        print("ERROR: NEO server not running")
        return None

    print(f"Server OK: t={health.get('t', '?')}")

    # Load history and create controller
    I_history = load_neo_history()
    print(f"History loaded: {len(I_history)} points")

    sigmas = compute_sigmas(I_history)
    print(f"Sigmas: S={sigmas[0]:.2e}, N={sigmas[1]:.2e}, C={sigmas[2]:.2e}")

    # Create Phase 4 controller
    controller = Phase4Controller()

    # Get initial status
    status = server_request('/status')
    I_initial = np.array([
        status['intention']['S'],
        status['intention']['N'],
        status['intention']['C']
    ])
    print(f"\nInitial I: [{I_initial[0]:.6f}, {I_initial[1]:.6f}, {I_initial[2]:.6f}]")

    # Series for recording
    series = []
    injections = []

    print(f"\n--- Running {cycles} cycles with Phase 4 injection ---\n")

    for i in range(cycles):
        # Get current state
        status = server_request('/status')
        if not status:
            print(f"Cycle {i}: Server error")
            break

        I_current = np.array([
            status['intention']['S'],
            status['intention']['N'],
            status['intention']['C']
        ])
        t = status.get('t', i)

        # Build residuals from recent changes
        if len(I_history) >= 10:
            recent = I_history[-10:]
            diffs = [np.linalg.norm(recent[j+1] - recent[j]) for j in range(len(recent)-1)]
            residuals = np.array(diffs)
        else:
            residuals = np.array([1e-6])

        # Phase 4 step
        I_new, diag = controller.step(
            I_current=I_current,
            residuals_window=residuals,
            residuals_history=residuals,
            sigmas_triplet=sigmas,
            T=len(I_history) + i,
            rho=0.9945,  # From Jacobian analysis
            rho_history=[0.9945],
            iqr_history=[float(np.std(residuals))]
        )

        delta = I_new - I_current
        delta_norm = np.linalg.norm(delta)

        # Apply perturbation via direct message OR compute target
        if diag.get('phase4_active', False) and delta_norm > 1e-10:
            # Compute perturbation scale based on sigma
            scale = sigmas[1] * 100  # Scale by N component (novelty)

            # Create message to influence intention
            msg = delta_to_influence_message(delta, I_current)

            # Send influence
            say_result = server_request('/say', method='POST', data={'message': msg})

            injections.append({
                't': t,
                'delta_norm': delta_norm,
                'message': msg,
                'influence': say_result.get('influence_applied', [0,0,0]) if say_result else [0,0,0]
            })

        # Run a step
        step_result = server_request('/step', method='POST')

        # Update history
        I_history.append(I_current)

        # Record sample
        if i % sample_every == 0:
            # Get post-step state
            post_status = server_request('/status')
            I_post = np.array([
                post_status['intention']['S'],
                post_status['intention']['N'],
                post_status['intention']['C']
            ]) if post_status else I_current

            actual_delta = np.linalg.norm(I_post - I_current)

            sample = {
                "t": t,
                "S": float(I_current[0]),
                "N": float(I_current[1]),
                "C": float(I_current[2]),
                "S_post": float(I_post[0]),
                "N_post": float(I_post[1]),
                "C_post": float(I_post[2]),
                "phase4_active": diag.get("phase4_active", False),
                "tau": diag.get("tau", 0),
                "delta_planned": delta_norm,
                "delta_actual": actual_delta,
                "eci": status.get('eci', 0)
            }
            series.append(sample)

            print(f"  t={t}: I=[{I_current[0]:.4f}, {I_current[1]:.4f}, {I_current[2]:.4f}] "
                  f"P4={'ON' if diag.get('phase4_active') else 'off'} "
                  f"δ_plan={delta_norm:.2e} δ_act={actual_delta:.2e}")

    # Final status
    final_status = server_request('/status')
    if final_status:
        I_final = [
            final_status['intention']['S'],
            final_status['intention']['N'],
            final_status['intention']['C']
        ]
        print(f"\nFinal I: [{I_final[0]:.6f}, {I_final[1]:.6f}, {I_final[2]:.6f}]")

    # Compute variance from series
    if len(series) > 5:
        S_vals = [s['S'] for s in series]
        N_vals = [s['N'] for s in series]
        C_vals = [s['C'] for s in series]

        print(f"\n--- Variance Analysis ---")
        print(f"Var(S): {np.var(S_vals):.6e}")
        print(f"Var(N): {np.var(N_vals):.6e}")
        print(f"Var(C): {np.var(C_vals):.6e}")

    # Diagnostics
    print(f"\n--- Phase 4 Diagnostics ---")
    full_diag = controller.get_full_diagnostics()
    print(f"Total calls: {full_diag['total_calls']}")
    print(f"Activations: {full_diag['activations']}")
    print(f"Rate: {full_diag['activation_rate']*100:.1f}%")
    print(f"Injections sent: {len(injections)}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "cycles": cycles,
        "initial_I": I_initial.tolist(),
        "final_I": I_final if final_status else None,
        "series": series,
        "injections": injections,
        "diagnostics": full_diag,
        "variance": {
            "S": float(np.var([s['S'] for s in series])) if series else 0,
            "N": float(np.var([s['N'] for s in series])) if series else 0,
            "C": float(np.var([s['C'] for s in series])) if series else 0,
        }
    }

    out_path = "/root/NEO_EVA/results/phase4_injection_run.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Saved: {out_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run NEO with Phase 4 injection")
    parser.add_argument("--cycles", type=int, default=200, help="Number of cycles")
    parser.add_argument("--sample-every", type=int, default=5, help="Sample frequency")
    args = parser.parse_args()

    run_phase4_injection(cycles=args.cycles, sample_every=args.sample_every)


if __name__ == "__main__":
    main()
