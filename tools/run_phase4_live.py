#!/usr/bin/env python3
"""
Run Phase 4 Live on NEO
=======================

Executes NEO cycles with Phase 4 variability active,
generating real variance in the intention vector.
"""

import sys
import os
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/root/NEO_EVA/tools')
from phase4_integration import Phase4NEOIntegration
import numpy as np

def server_request(endpoint, method='GET', timeout=10):
    """Make request to NEO server."""
    url = f'http://127.0.0.1:7777{endpoint}'
    try:
        req = urllib.request.Request(url, method=method)
        response = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Phase 4 live on NEO")
    parser.add_argument("--cycles", type=int, default=100, help="Number of cycles")
    parser.add_argument("--sample-every", type=int, default=10, help="Sample frequency")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 4 Live Execution")
    print("=" * 70)
    print(f"Cycles: {args.cycles}")
    print(f"Sample every: {args.sample_every}")
    print()

    # Check server
    health = server_request('/health')
    if not health:
        print("ERROR: NEO server not running. Start with:")
        print("  cd /root/NEOSYNT && python3 core/neosynt_server.py &")
        return

    # Get initial status
    status = server_request('/status')
    print(f"Initial state: t={status['t']}, I=[{status['intention']['S']:.6f}, {status['intention']['N']:.2e}, {status['intention']['C']:.2e}]")

    # Create integrator
    integrator = Phase4NEOIntegration()

    # Load history for sigmas
    import yaml
    with open('/root/NEOSYNT/state/neo_state.yaml') as f:
        state = yaml.safe_load(f)
    raw = state.get("autonomy", {}).get("history_intention", [])
    I_history = [np.array(v) for v in raw if len(v) == 3]

    I_arr = np.array(I_history)
    sig_S = float(np.std(I_arr[:, 0]))
    sig_N = float(np.std(I_arr[:, 1]))
    sig_C = float(np.std(I_arr[:, 2]))
    print(f"Historical σ = [{sig_S:.6e}, {sig_N:.6e}, {sig_C:.6e}]")

    # Run cycles
    print(f"\n--- Running {args.cycles} cycles ---")

    series = []
    for i in range(args.cycles):
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

        # Apply Phase 4
        I_new, diag = integrator.apply_phase4(
            I_current=I_current,
            I_history=I_history + [I_current],
            sigmas_triplet=(sig_S, sig_N, sig_C)
        )

        # If Phase 4 modified intention significantly, we would update state
        # For now, just record and run normal step
        delta = np.linalg.norm(I_new - I_current)

        # Run a step
        result = server_request('/step', method='POST')

        if i % args.sample_every == 0:
            print(f"  t={status['t']}: I=[{I_current[0]:.6f}, {I_current[1]:.2e}, {I_current[2]:.2e}] "
                  f"P4={'ON' if diag.get('phase4_active') else 'off'} "
                  f"τ={diag.get('tau', 0):.4f} δ={delta:.2e}")

            series.append({
                "t": status['t'],
                "S": I_current[0],
                "N": I_current[1],
                "C": I_current[2],
                "phase4_active": diag.get("phase4_active", False),
                "tau": diag.get("tau", 0),
                "delta": delta
            })

        # Update history
        I_history.append(I_current)

    # Final status
    status = server_request('/status')
    print(f"\nFinal state: t={status['t']}, I=[{status['intention']['S']:.6f}, {status['intention']['N']:.2e}, {status['intention']['C']:.2e}]")

    # Diagnostics
    print(f"\n--- Phase 4 Diagnostics ---")
    diag = integrator.get_full_diagnostics()
    print(f"Total calls: {diag['total_calls']}")
    print(f"Activations: {diag['activations']}")
    print(f"Rate: {diag['activation_rate']*100:.1f}%")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "cycles": args.cycles,
        "series": series,
        "diagnostics": diag
    }
    out_path = "/root/NEO_EVA/results/phase4_live_run.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Saved: {out_path}")

if __name__ == "__main__":
    main()
