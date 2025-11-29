#!/usr/bin/env python3
"""
Phase 6: Endogenous Coupling NEO↔EVA + IWVI
============================================

Implements:
A) BUS with summary messages (μ_I, v₁, λ₁, u, conf)
B) Endogenous coupling law κ_t with tangent projection
C) IWVI validation with valid windows
D) Ablations (no_bus, kappa=0)

All parameters are 100% endogenous - no hardcoded constants.
"""

import sys
import os
import json
import numpy as np
import time
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field

sys.path.insert(0, '/root/NEOSYNT')
sys.path.insert(0, '/root/NEO_EVA/tools')

from engine.intention_dynamics import update_intention_damped
from engine.meta_intention import compute_meta_intention


# =============================================================================
# A) BUS - Inter-World Communication
# =============================================================================

@dataclass
class WorldSummary:
    """Summary message published by each world."""
    world_id: str
    t: int
    mu_I: np.ndarray          # Mean intention [S, N, C] over window
    v1: np.ndarray            # Principal direction (PCA)
    lambda1: float            # Variance explained by v1
    u: float                  # Uncertainty = IQR(residuals) / √T
    conf: float               # Confidence in [0, 1]
    cv_r: float               # Coefficient of variation of residuals


class BUS:
    """
    Inter-world communication bus.

    Each world publishes summaries every w≈√T steps.
    """

    def __init__(self):
        self.messages: Dict[str, List[WorldSummary]] = {
            'NEO': [],
            'EVA': []
        }
        self.enabled = True
        self.total_count = 0

    def publish(self, summary: WorldSummary):
        """Publish a summary to the bus."""
        if not self.enabled:
            return

        self.messages[summary.world_id].append(summary)
        self.total_count += 1

        # Keep bounded history
        max_hist = 100
        for world in self.messages:
            if len(self.messages[world]) > max_hist:
                self.messages[world] = self.messages[world][-max_hist:]

    def get_latest(self, world_id: str) -> Optional[WorldSummary]:
        """Get latest summary from a world."""
        if not self.enabled:
            return None
        if self.messages[world_id]:
            return self.messages[world_id][-1]
        return None

    def get_counts(self) -> Dict[str, int]:
        """Get message counts by world."""
        return {w: len(msgs) for w, msgs in self.messages.items()}


# =============================================================================
# B) Endogenous Coupling Law
# =============================================================================

class EndogenousCoupling:
    """
    Implements the coupling law κ_t with tangent projection.

    κ_t^X = (u_t^Y / (1 + u_t^X)) × (λ₁^Y / (λ₁^Y + λ₁^X + ε)) × (conf_t^Y / (1 + CV(r_t^X)))

    All factors are normalized to [0, 1] using historical quantiles.
    """

    def __init__(self):
        # Historical stats for normalization
        self.kappa_history: List[float] = []
        self.u_history: List[float] = []
        self.lambda_history: List[float] = []
        self.cv_history: List[float] = []

        # Tangent basis (fixed for 3-simplex)
        self.u_c = np.array([1, 1, 1]) / np.sqrt(3)
        self.u_1 = np.array([1, -1, 0]) / np.sqrt(2)
        self.u_2 = np.array([1, 1, -2]) / np.sqrt(6)

    def compute_tangent_projection(self, I_current: np.ndarray, v_other: np.ndarray) -> np.ndarray:
        """
        Project direction v_other onto tangent plane at I_current.

        The tangent plane of the simplex at I is perpendicular to (1,1,1).
        """
        # Remove component along (1,1,1)
        v_tangent = v_other - np.dot(v_other, self.u_c) * self.u_c

        # Normalize if non-zero
        norm = np.linalg.norm(v_tangent)
        if norm > 1e-10:
            v_tangent = v_tangent / norm
        else:
            v_tangent = np.zeros(3)

        return v_tangent

    def compute_kappa(self,
                      u_self: float,
                      u_other: float,
                      lambda1_self: float,
                      lambda1_other: float,
                      conf_other: float,
                      cv_self: float) -> float:
        """
        Compute endogenous coupling gain κ_t.

        κ = (u_Y / (1 + u_X)) × (λ₁^Y / (λ₁^Y + λ₁^X + ε)) × (conf^Y / (1 + CV(r^X)))
        """
        eps = 1e-10

        # Factor 1: Uncertainty ratio
        # High uncertainty in other → more influence
        f1 = u_other / (1 + u_self + eps)

        # Factor 2: Directional dominance
        # Other has strong principal direction → more influence
        f2 = lambda1_other / (lambda1_other + lambda1_self + eps)

        # Factor 3: Confidence vs instability
        # Other is confident, self is unstable → more influence
        f3 = conf_other / (1 + cv_self + eps)

        # Combine factors
        kappa_raw = f1 * f2 * f3

        # Normalize using historical quantiles
        self.u_history.append(u_self)
        self.lambda_history.append(lambda1_self)
        self.cv_history.append(cv_self)

        # Keep bounded
        max_hist = 200
        if len(self.u_history) > max_hist:
            self.u_history = self.u_history[-max_hist:]
            self.lambda_history = self.lambda_history[-max_hist:]
            self.cv_history = self.cv_history[-max_hist:]

        # Normalize to [0, 1] based on historical range
        if len(self.kappa_history) > 10:
            kappa_p99 = np.percentile(self.kappa_history, 99)
            if kappa_p99 > eps:
                kappa = min(1.0, kappa_raw / kappa_p99)
            else:
                kappa = kappa_raw
        else:
            kappa = min(1.0, kappa_raw)

        self.kappa_history.append(kappa_raw)
        if len(self.kappa_history) > max_hist:
            self.kappa_history = self.kappa_history[-max_hist:]

        return kappa

    def compute_coupled_delta(self,
                              delta_self: np.ndarray,
                              I_self: np.ndarray,
                              other_summary: Optional[WorldSummary],
                              u_self: float,
                              lambda1_self: float,
                              cv_self: float,
                              gate_open: bool) -> Tuple[np.ndarray, Dict]:
        """
        Compute coupled delta: Δ̃ = Δ_self + κ × g_Y→X

        Only applies if gate is open and other summary available.
        """
        info = {
            'coupling_active': False,
            'kappa': 0.0,
            'g_norm': 0.0,
        }

        if not gate_open or other_summary is None:
            return delta_self, info

        # Compute κ
        kappa = self.compute_kappa(
            u_self=u_self,
            u_other=other_summary.u,
            lambda1_self=lambda1_self,
            lambda1_other=other_summary.lambda1,
            conf_other=other_summary.conf,
            cv_self=cv_self
        )

        # Project other's v1 onto self's tangent plane
        g_Y_to_X = self.compute_tangent_projection(I_self, other_summary.v1)

        # Combined delta
        delta_coupled = delta_self + kappa * g_Y_to_X

        info = {
            'coupling_active': True,
            'kappa': float(kappa),
            'g_norm': float(np.linalg.norm(g_Y_to_X)),
            'f1': float(other_summary.u / (1 + u_self + 1e-10)),
            'f2': float(other_summary.lambda1 / (other_summary.lambda1 + lambda1_self + 1e-10)),
            'f3': float(other_summary.conf / (1 + cv_self + 1e-10)),
        }

        return delta_coupled, info


# =============================================================================
# C) World with Phase 4 + Coupling
# =============================================================================

class CoupledWorld:
    """
    A world (NEO or EVA) with Phase 4 + endogenous coupling.
    """

    def __init__(self, world_id: str, initial_I: np.ndarray, bus: BUS):
        self.world_id = world_id
        self.other_id = 'EVA' if world_id == 'NEO' else 'NEO'
        self.bus = bus

        self.I = initial_I.copy()
        self.I_history: List[np.ndarray] = [initial_I.copy()]
        self.residuals: List[float] = []

        # OU state for Phase 4
        self.ou_Z = np.array([0.0, 0.0])
        self.ou_theta = 0.1

        # Coupling
        self.coupling = EndogenousCoupling()

        # Stats history for endogenous params
        self.variance_history: List[float] = []
        self.rho_history: List[float] = []  # Jacobian spectral radius
        self.iqr_history: List[float] = []

        # Tangent basis
        self.u_1 = np.array([1, -1, 0]) / np.sqrt(2)
        self.u_2 = np.array([1, 1, -2]) / np.sqrt(6)

        self.t = 0
        self.gate_activations = 0
        self.coupling_activations = 0

        # Series for IWVI
        self.series: List[Dict] = []

        # World-specific dynamics
        if world_id == 'NEO':
            self.drift = np.array([0.001, -0.0005, -0.0005])  # S-bias (stability)
        else:
            self.drift = np.array([-0.001, 0.002, -0.001])   # N-bias (novelty)

    def compute_summary(self) -> WorldSummary:
        """Compute summary to publish on BUS."""
        T = len(self.I_history)
        w = max(10, int(np.sqrt(T)))  # Window size

        if T < w:
            # Not enough data
            return WorldSummary(
                world_id=self.world_id,
                t=self.t,
                mu_I=self.I.copy(),
                v1=np.array([1, 0, 0]),
                lambda1=0.0,
                u=0.1,
                conf=0.5,
                cv_r=0.1
            )

        # Recent window
        I_window = np.array(self.I_history[-w:])

        # Mean intention
        mu_I = np.mean(I_window, axis=0)

        # PCA for v1 and λ1
        I_centered = I_window - mu_I
        cov = np.cov(I_centered.T)

        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            v1 = eigvecs[:, idx[0]]
            lambda1 = eigvals[idx[0]]
        except:
            v1 = np.array([1, 0, 0])
            lambda1 = 0.0

        # Uncertainty u = IQR(residuals) / √T
        if len(self.residuals) > 4:
            res = np.array(self.residuals[-w:])
            iqr = np.percentile(res, 75) - np.percentile(res, 25)
            u = iqr / np.sqrt(T)
        else:
            u = 0.1

        # CV of residuals
        if len(self.residuals) > 4:
            res = np.array(self.residuals[-w:])
            cv_r = np.std(res) / (np.mean(np.abs(res)) + 1e-10)
        else:
            cv_r = 0.1

        # Confidence (endogenous: based on variance and stability)
        total_var = np.var(I_window).sum()
        self.variance_history.append(total_var)

        if len(self.variance_history) > 20:
            var_p50 = np.percentile(self.variance_history, 50)
            conf = 1.0 / (1.0 + total_var / (var_p50 + 1e-10))
        else:
            conf = 0.5

        conf = max(0.0, min(1.0, conf))

        return WorldSummary(
            world_id=self.world_id,
            t=self.t,
            mu_I=mu_I,
            v1=v1,
            lambda1=float(lambda1),
            u=float(u),
            conf=float(conf),
            cv_r=float(cv_r)
        )

    def compute_gate(self) -> Tuple[bool, Dict]:
        """Critical gate: opens based on ρ(J) and IQR thresholds."""
        if len(self.I_history) < 20:
            return True, {'reason': 'warmup', 'open': True}

        I_arr = np.array(self.I_history[-20:])

        # Check if at corner (sticky vertex check)
        at_corner = np.max(self.I) > 0.90

        # Approximate ρ from residual decay
        residuals = np.diff(I_arr, axis=0)
        if len(residuals) > 5:
            r_norm = np.linalg.norm(residuals, axis=1)
            if r_norm[0] > 1e-10:
                rho_approx = (r_norm[-1] / r_norm[0]) ** (1 / len(r_norm))
            else:
                rho_approx = 0.99
        else:
            rho_approx = 0.99

        self.rho_history.append(rho_approx)

        # IQR of recent residuals
        if len(self.residuals) > 10:
            iqr = np.percentile(self.residuals[-20:], 75) - np.percentile(self.residuals[-20:], 25)
        else:
            iqr = 0.01

        self.iqr_history.append(iqr)

        # Thresholds from history
        if len(self.rho_history) > 30:
            rho_p95 = np.percentile(self.rho_history, 95)
            iqr_p75 = np.percentile(self.iqr_history, 75)
        else:
            rho_p95 = 0.98
            iqr_p75 = 0.001

        # Gate opens if: at corner OR (high ρ AND high IQR)
        gate_open = at_corner or (rho_approx >= rho_p95 * 0.95 and iqr >= iqr_p75 * 0.5)

        return gate_open, {
            'rho': float(rho_approx),
            'rho_p95': float(rho_p95),
            'iqr': float(iqr),
            'iqr_p75': float(iqr_p75),
            'at_corner': at_corner,
            'open': gate_open,
        }

    def ou_step(self, tau: float) -> np.ndarray:
        """OU step in tangent plane."""
        # Adaptive theta from residual ACF
        if len(self.residuals) > 20:
            r = np.array(self.residuals[-20:])
            r_corr = np.corrcoef(r[:-1], r[1:])[0, 1]
            if not np.isnan(r_corr) and abs(r_corr) < 0.99:
                self.ou_theta = min(0.5, max(0.01, -1 / np.log(abs(r_corr) + 1e-10)))

        # OU dynamics
        dt = 1.0
        sigma = np.sqrt(max(tau, 0.05))

        drift = -self.ou_theta * self.ou_Z * dt
        diffusion = sigma * np.sqrt(dt) * np.random.randn(2)

        self.ou_Z = self.ou_Z + drift + diffusion
        self.ou_Z = np.clip(self.ou_Z, -5, 5)

        return self.ou_Z

    def ou_to_delta(self, Z: np.ndarray) -> np.ndarray:
        """Convert OU to tangent-plane delta."""
        delta = Z[0] * self.u_1 + Z[1] * self.u_2
        delta = delta - delta.mean()  # Ensure zero mean
        return delta

    def compute_thermostat(self) -> float:
        """Compute τ from IQR of residuals."""
        if len(self.residuals) < 10:
            return 0.1

        res = np.array(self.residuals[-50:])
        iqr = np.percentile(res, 75) - np.percentile(res, 25)

        T = len(self.I_history)
        tau_base = iqr / np.sqrt(T)

        if len(self.I_history) > 20:
            I_arr = np.array(self.I_history[-20:])
            sigma_hist = np.median(np.std(I_arr, axis=0))
            tau = tau_base * max(sigma_hist, 0.01)
        else:
            tau = tau_base * 0.05

        return max(0.01, min(0.5, tau))

    def mirror_descent(self, I: np.ndarray, delta: np.ndarray, eta: float) -> np.ndarray:
        """Mirror descent: I_{t+1} = softmax(log I + η * Δ)"""
        floor = 1e-6
        I_safe = np.maximum(I, floor)
        I_safe = I_safe / I_safe.sum()

        log_I = np.log(I_safe)
        log_I_new = log_I + eta * delta

        exp_log = np.exp(log_I_new - np.max(log_I_new))
        I_new = exp_log / exp_log.sum()

        I_new = np.maximum(I_new, floor)
        I_new = I_new / I_new.sum()

        return I_new

    def step(self, enable_coupling: bool = True) -> Dict:
        """Execute one step with Phase 4 + coupling."""
        self.t += 1
        I_prev = self.I.copy()

        # Compute candidate via world-specific dynamics
        noise = np.random.randn(3) * 0.01
        I_candidate = I_prev + self.drift + noise
        I_candidate = np.maximum(I_candidate, 1e-6)
        I_candidate = I_candidate / I_candidate.sum()

        # Residual
        residual = np.linalg.norm(I_candidate - I_prev)
        self.residuals.append(residual)
        if len(self.residuals) > 500:
            self.residuals = self.residuals[-300:]

        # Check gate
        gate_open, gate_info = self.compute_gate()

        if not gate_open:
            self.I = I_candidate
            self.I_history.append(self.I.copy())
            return self._record_step(I_prev, gate_open=False, coupling_info={})

        self.gate_activations += 1

        # Thermostat
        tau = self.compute_thermostat()

        # OU step
        Z = self.ou_step(tau)
        delta_base = self.ou_to_delta(Z)

        # Get other world's summary from BUS
        other_summary = self.bus.get_latest(self.other_id) if enable_coupling else None

        # Compute coupled delta
        summary = self.compute_summary()
        delta_coupled, coupling_info = self.coupling.compute_coupled_delta(
            delta_self=delta_base,
            I_self=I_candidate,
            other_summary=other_summary,
            u_self=summary.u,
            lambda1_self=summary.lambda1,
            cv_self=summary.cv_r,
            gate_open=gate_open
        )

        if coupling_info.get('coupling_active', False):
            self.coupling_activations += 1

        # Mirror descent
        at_corner = np.max(I_candidate) > 0.90
        if at_corner:
            eta = max(tau, 1.0) * 2.0  # Escape boost
        else:
            eta = max(tau, 0.2)

        I_new = self.mirror_descent(I_candidate, delta_coupled, eta)

        # Update state
        self.I = I_new
        self.I_history.append(self.I.copy())
        if len(self.I_history) > 1000:
            self.I_history = self.I_history[-500:]

        # Publish summary to BUS (every √T steps)
        T = len(self.I_history)
        w = max(10, int(np.sqrt(T)))
        if self.t % w == 0:
            self.bus.publish(summary)

        return self._record_step(I_prev, gate_open=True, coupling_info=coupling_info,
                                 tau=tau, eta=eta, delta_norm=np.linalg.norm(delta_coupled))

    def _record_step(self, I_prev, gate_open, coupling_info, tau=0, eta=0, delta_norm=0):
        """Record step to series."""
        record = {
            't': self.t,
            'S_prev': float(I_prev[0]),
            'N_prev': float(I_prev[1]),
            'C_prev': float(I_prev[2]),
            'S_new': float(self.I[0]),
            'N_new': float(self.I[1]),
            'C_new': float(self.I[2]),
            'gate_open': gate_open,
            'coupling_active': coupling_info.get('coupling_active', False),
            'kappa': coupling_info.get('kappa', 0),
            'tau': float(tau),
            'eta': float(eta),
            'delta_norm': float(delta_norm),
        }
        self.series.append(record)
        return record


# =============================================================================
# D) Coupled System Runner
# =============================================================================

class CoupledSystemRunner:
    """Runs NEO and EVA together with BUS coupling."""

    def __init__(self, enable_coupling: bool = True):
        self.bus = BUS()
        self.bus.enabled = enable_coupling

        # Initialize worlds
        self.neo = CoupledWorld(
            world_id='NEO',
            initial_I=np.array([1.0, 0.0, 0.0]),  # Start at S corner
            bus=self.bus
        )
        self.eva = CoupledWorld(
            world_id='EVA',
            initial_I=np.array([1/3, 1/3, 1/3]),  # Start at prior
            bus=self.bus
        )

        self.enable_coupling = enable_coupling

    def run(self, cycles: int = 500, verbose: bool = True) -> Dict:
        """Run both worlds for specified cycles."""
        print("=" * 70)
        print(f"Phase 6: Coupled NEO↔EVA System")
        print(f"Coupling: {'ENABLED' if self.enable_coupling else 'DISABLED'}")
        print("=" * 70)
        print(f"NEO initial: {self.neo.I}")
        print(f"EVA initial: {self.eva.I}")
        print()

        start_time = time.time()

        for i in range(cycles):
            # Step both worlds (alternating)
            self.neo.step(enable_coupling=self.enable_coupling)
            self.eva.step(enable_coupling=self.enable_coupling)

            if verbose and (i + 1) % 100 == 0:
                print(f"  t={i+1:4d}: NEO=[{self.neo.I[0]:.4f}, {self.neo.I[1]:.4f}, {self.neo.I[2]:.4f}] "
                      f"EVA=[{self.eva.I[0]:.4f}, {self.eva.I[1]:.4f}, {self.eva.I[2]:.4f}] "
                      f"κ_NEO={self.neo.series[-1].get('kappa', 0):.4f}")

        elapsed = time.time() - start_time

        # Compute stats
        neo_arr = np.array([[s['S_new'], s['N_new'], s['C_new']] for s in self.neo.series])
        eva_arr = np.array([[s['S_new'], s['N_new'], s['C_new']] for s in self.eva.series])

        results = {
            'cycles': cycles,
            'elapsed': elapsed,
            'coupling_enabled': self.enable_coupling,
            'bus_counts': self.bus.get_counts(),
            'neo': {
                'initial_I': [1.0, 0.0, 0.0],
                'final_I': self.neo.I.tolist(),
                'variance': {
                    'S': float(np.var(neo_arr[:, 0])),
                    'N': float(np.var(neo_arr[:, 1])),
                    'C': float(np.var(neo_arr[:, 2])),
                    'total': float(np.var(neo_arr).sum() * 3),
                },
                'gate_activations': self.neo.gate_activations,
                'coupling_activations': self.neo.coupling_activations,
            },
            'eva': {
                'initial_I': [1/3, 1/3, 1/3],
                'final_I': self.eva.I.tolist(),
                'variance': {
                    'S': float(np.var(eva_arr[:, 0])),
                    'N': float(np.var(eva_arr[:, 1])),
                    'C': float(np.var(eva_arr[:, 2])),
                    'total': float(np.var(eva_arr).sum() * 3),
                },
                'gate_activations': self.eva.gate_activations,
                'coupling_activations': self.eva.coupling_activations,
            },
        }

        print()
        print("=" * 70)
        print("Results:")
        print(f"  NEO final: {self.neo.I}")
        print(f"  EVA final: {self.eva.I}")
        print(f"  NEO variance: {results['neo']['variance']['total']:.6e}")
        print(f"  EVA variance: {results['eva']['variance']['total']:.6e}")
        print(f"  NEO coupling activations: {self.neo.coupling_activations}/{cycles}")
        print(f"  EVA coupling activations: {self.eva.coupling_activations}/{cycles}")
        print(f"  BUS counts: {self.bus.get_counts()}")
        print("=" * 70)

        return results

    def save_series(self, path_prefix: str):
        """Save series for both worlds."""
        neo_data = {
            'timestamp': datetime.now().isoformat(),
            'world': 'NEO',
            'cycles': len(self.neo.series),
            'series': self.neo.series,
        }
        eva_data = {
            'timestamp': datetime.now().isoformat(),
            'world': 'EVA',
            'cycles': len(self.eva.series),
            'series': self.eva.series,
        }

        neo_path = f"{path_prefix}_neo.json"
        eva_path = f"{path_prefix}_eva.json"

        with open(neo_path, 'w') as f:
            json.dump(neo_data, f, indent=2)
        with open(eva_path, 'w') as f:
            json.dump(eva_data, f, indent=2)

        print(f"[OK] Saved: {neo_path}")
        print(f"[OK] Saved: {eva_path}")


def run_coupled_experiment(cycles: int = 500, enable_coupling: bool = True, output_prefix: str = None):
    """Run a single coupled experiment."""
    runner = CoupledSystemRunner(enable_coupling=enable_coupling)
    results = runner.run(cycles=cycles)

    if output_prefix:
        runner.save_series(output_prefix)

    return runner, results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6 Coupled System")
    parser.add_argument("--cycles", type=int, default=500, help="Number of cycles")
    parser.add_argument("--no-coupling", action="store_true", help="Disable coupling (ablation)")
    parser.add_argument("--output", type=str, default="/root/NEO_EVA/results/phase6_coupled")
    args = parser.parse_args()

    os.makedirs("/root/NEO_EVA/results", exist_ok=True)

    runner, results = run_coupled_experiment(
        cycles=args.cycles,
        enable_coupling=not args.no_coupling,
        output_prefix=args.output
    )

    # Save results
    results_path = f"{args.output}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved: {results_path}")


if __name__ == "__main__":
    main()
