#!/usr/bin/env python3
"""
Phase 4 Integration for NEOSYNT
===============================

Hooks Phase 4 variability into NEO's autonomy loop.

Integration points:
1. After intention update in _update_intention()
2. Jacobian computed periodically from history_intention
3. Critical gate checks before applying variability

100% endogenous. No hardcoded values.
"""

import sys
import os
import math
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add paths
sys.path.insert(0, '/root/NEO_EVA/tools')
sys.path.insert(0, '/root/NEOSYNT')

from phase4_variability import (
    Phase4Controller,
    compute_thermostat_tau,
    compute_prediction_residuals,
    acf_zero_crossing,
    is_valid_iwvi_window,
)
from common import acf_window, sigmas


class Phase4NEOIntegration:
    """
    Integrates Phase 4 variability into NEO's autonomy loop.

    Usage:
        integrator = Phase4NEOIntegration()

        # In autonomy loop, after computing new_intention:
        new_intention = integrator.apply_phase4(
            I_current=new_intention,
            I_history=history_intention,
            sigmas_triplet=(sig_S, sig_N, sig_C)
        )
    """

    def __init__(self, jacobian_update_interval: Optional[int] = None):
        """
        Initialize integrator.

        jacobian_update_interval: if None, computed endogenously as √T
        """
        self.controller = Phase4Controller()
        self._jacobian_update_interval = jacobian_update_interval
        self._last_jacobian_t = 0
        self._jacobian_history: List[np.ndarray] = []

        # Diagnostics
        self.activations = 0
        self.total_calls = 0
        self.last_diag: Dict = {}

    def _compute_jacobian_from_history(
        self,
        I_history: List[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Estimate local Jacobian from intention history.

        J_ij ≈ ∂I_i(t+1)/∂I_j(t)

        Returns (J, rho) or (None, 0.0) if insufficient data.
        """
        T = len(I_history)
        if T < 10:
            return None, 0.0

        # Build transition matrices
        X = np.array(I_history[:-1])  # I_t
        Y = np.array(I_history[1:])   # I_{t+1}

        # Augment with bias
        X_aug = np.column_stack([X, np.ones(len(X))])

        try:
            # Least squares: [J | c] = (X^T X)^{-1} X^T Y
            beta = np.linalg.lstsq(X_aug, Y, rcond=None)[0]
            J = beta[:3, :].T  # 3x3

            # Spectral radius
            eigvals = np.linalg.eigvals(J)
            rho = float(np.max(np.abs(eigvals)))

            return J, rho

        except Exception:
            return None, 0.0

    def _should_update_jacobian(self, T: int) -> bool:
        """Check if we should recompute Jacobian."""
        if self._jacobian_update_interval is not None:
            interval = self._jacobian_update_interval
        else:
            # Endogenous: √T
            interval = max(10, int(math.sqrt(T)))

        return (T - self._last_jacobian_t) >= interval

    def apply_phase4(
        self,
        I_current: np.ndarray,
        I_history: List[np.ndarray],
        sigmas_triplet: Tuple[float, float, float],
        I_predicted: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply Phase 4 variability to intention.

        Parameters
        ----------
        I_current : np.ndarray
            Current intention after normal update
        I_history : List[np.ndarray]
            History of intentions
        sigmas_triplet : Tuple[float, float, float]
            (σ_S, σ_N, σ_C) from history
        I_predicted : Optional[np.ndarray]
            Predicted intention (if None, uses persistence I_{t-1})

        Returns
        -------
        I_new : np.ndarray
            Modified intention (unchanged if gate closed)
        diagnostics : Dict
            Phase 4 diagnostics
        """
        self.total_calls += 1
        T = len(I_history)

        # Default prediction: persistence
        if I_predicted is None and len(I_history) >= 1:
            I_predicted = I_history[-1]
        elif I_predicted is None:
            I_predicted = I_current

        # Maybe update Jacobian
        if T >= 10 and self._should_update_jacobian(T):
            J, rho = self._compute_jacobian_from_history(I_history)
            if J is not None:
                self.controller.update_jacobian(J, rho)
                self._jacobian_history.append(J)
                self._last_jacobian_t = T

        # Compute ACF lag for OU parameters
        if T >= 8:
            # Use S component for ACF
            S_series = np.array([I[0] for I in I_history])
            acf_lag = acf_zero_crossing(S_series)
        else:
            acf_lag = max(1, int(math.log1p(T) + 1))

        # Window size from ACF
        window = acf_lag * 2

        # Apply Phase 4
        I_new, diag = self.controller.step(
            I_current=I_current,
            I_predicted=I_predicted,
            sigmas_triplet=sigmas_triplet,
            acf_lag=acf_lag,
            window_size=window
        )

        # Update stats
        if diag.get("phase4_active", False):
            self.activations += 1

        self.last_diag = diag
        diag["total_calls"] = self.total_calls
        diag["activation_rate"] = self.activations / max(1, self.total_calls)

        return I_new, diag

    def is_iwvi_valid(
        self,
        I_history: List[np.ndarray],
        window: Optional[int] = None
    ) -> bool:
        """
        Check if current window is valid for IWVI evaluation.

        Returns True if variance is sufficient.
        """
        if len(I_history) < 20:
            return False

        I_arr = np.array(I_history)

        if window is None:
            window = max(10, int(math.sqrt(len(I_history))))

        I_window = I_arr[-window:]

        return is_valid_iwvi_window(I_window, I_arr)

    def get_full_diagnostics(self) -> Dict:
        """Get comprehensive diagnostics."""
        return {
            "total_calls": self.total_calls,
            "activations": self.activations,
            "activation_rate": self.activations / max(1, self.total_calls),
            "jacobians_computed": len(self._jacobian_history),
            "controller_state": self.controller.get_diagnostics(),
            "last_diag": self.last_diag
        }


# =============================================================================
# Patch for AutonomyLoop
# =============================================================================

def patch_autonomy_loop_phase4(autonomy_loop, integrator: Phase4NEOIntegration):
    """
    Monkey-patch the AutonomyLoop to include Phase 4.

    This wraps the _update_intention method to apply Phase 4 after
    the normal intention update.
    """
    original_update = autonomy_loop._update_intention

    def patched_update_intention(prediction):
        # Normal update
        new_intention = original_update(prediction)

        # Get sigmas from history
        if len(autonomy_loop.state.history_intention) >= 2:
            I_arr = np.array([
                I.tolist() if hasattr(I, 'tolist') else I
                for I in autonomy_loop.state.history_intention
            ])
            sig_S = float(np.std(I_arr[:, 0]))
            sig_N = float(np.std(I_arr[:, 1]))
            sig_C = float(np.std(I_arr[:, 2]))
        else:
            sig_S = sig_N = sig_C = 0.001

        # Apply Phase 4
        I_new, diag = integrator.apply_phase4(
            I_current=new_intention,
            I_history=[
                I.tolist() if hasattr(I, 'tolist') else I
                for I in autonomy_loop.state.history_intention
            ],
            sigmas_triplet=(sig_S, sig_N, sig_C)
        )

        # Store diagnostics
        autonomy_loop.state.phase4_diag = diag

        # Update intention with Phase 4 result
        autonomy_loop.state.current_intention = I_new

        return I_new

    autonomy_loop._update_intention = patched_update_intention
    return autonomy_loop


# =============================================================================
# Standalone runner for testing
# =============================================================================

def run_phase4_test():
    """Run Phase 4 on actual NEO history."""
    import yaml

    print("=" * 70)
    print("Phase 4 Integration Test")
    print("=" * 70)

    # Load NEO state
    neo_state_path = "/root/NEOSYNT/state/neo_state.yaml"
    if not os.path.exists(neo_state_path):
        print("NEO state not found")
        return

    with open(neo_state_path) as f:
        state = yaml.safe_load(f)

    raw = state.get("autonomy", {}).get("history_intention", [])
    I_history = [np.array(v) for v in raw if len(v) == 3]

    T = len(I_history)
    print(f"\nLoaded T={T} intention samples")

    if T < 50:
        print("Insufficient history for Phase 4 test")
        return

    # Compute sigmas
    I_arr = np.array(I_history)
    sig_S = float(np.std(I_arr[:, 0]))
    sig_N = float(np.std(I_arr[:, 1]))
    sig_C = float(np.std(I_arr[:, 2]))

    print(f"σ = [{sig_S:.6e}, {sig_N:.6e}, {sig_C:.6e}]")

    # Create integrator
    integrator = Phase4NEOIntegration()

    # Simulate stepping through last 100 samples
    print("\n--- Simulating Phase 4 on last 100 samples ---")

    start_idx = max(0, T - 100)
    activations = 0

    for i in range(start_idx, T):
        I_curr = I_history[i]
        I_hist_so_far = I_history[:i+1]

        I_new, diag = integrator.apply_phase4(
            I_current=I_curr,
            I_history=I_hist_so_far,
            sigmas_triplet=(sig_S, sig_N, sig_C)
        )

        if diag.get("phase4_active", False):
            activations += 1
            delta = np.linalg.norm(I_new - I_curr)
            print(f"  t={i}: ACTIVE τ={diag['tau']:.6f}, δ={delta:.6e}")

    print(f"\n--- Results ---")
    print(f"Total samples: {T - start_idx}")
    print(f"Activations: {activations}")
    print(f"Activation rate: {activations / (T - start_idx):.2%}")

    # Check IWVI validity
    valid = integrator.is_iwvi_valid(I_history)
    print(f"\nIWVI valid window: {valid}")

    # Full diagnostics
    print("\n--- Full Diagnostics ---")
    print(json.dumps(integrator.get_full_diagnostics(), indent=2, default=str))

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "T": T,
        "activations": activations,
        "activation_rate": activations / (T - start_idx),
        "iwvi_valid": valid,
        "diagnostics": integrator.get_full_diagnostics()
    }

    results_path = "/root/NEO_EVA/results/phase4_integration_test.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Saved: {results_path}")


if __name__ == "__main__":
    run_phase4_test()
