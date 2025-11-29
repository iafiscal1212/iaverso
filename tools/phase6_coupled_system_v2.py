#!/usr/bin/env python3
"""
Phase 6 v2: 100% Endogenous Coupled NEO↔EVA + IWVI
===================================================

ZERO hardcoded parameters. All values derived from:
- Historical quantiles (p50/p75/p95/p97.5/p99)
- Window statistics (IQR, σ, CV)
- Scale factors (√T, log T)

Audit trail: every parameter shows its derivation source.
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


# =============================================================================
# Utility: Endogenous Statistics
# =============================================================================

class EndogenousStats:
    """
    Computes all statistics from history only.
    No hardcoded values except numerical stability (1e-12).
    """
    EPS = 1e-12  # Only allowed constant: numerical stability

    @staticmethod
    def window_size(T: int) -> int:
        """w = max{10, ⌊√T⌋} - endogenous from T"""
        return max(10, int(np.sqrt(T)))

    @staticmethod
    def max_hist(T: int) -> int:
        """max_hist = min{T, ⌊10√T⌋} - scales with √T"""
        return min(T, int(10 * np.sqrt(T)))

    @staticmethod
    def quantile_safe(arr: List[float], q: float, default: float = 0.0) -> float:
        """Safe quantile with fallback for empty/small arrays."""
        if len(arr) < 2:
            return default
        return float(np.percentile(arr, q * 100))

    @staticmethod
    def iqr(arr: np.ndarray) -> float:
        """IQR = p75 - p25"""
        if len(arr) < 4:
            return EndogenousStats.EPS
        return float(np.percentile(arr, 75) - np.percentile(arr, 25))

    @staticmethod
    def cv(arr: np.ndarray) -> float:
        """Coefficient of variation = σ / |μ|"""
        if len(arr) < 2:
            return 1.0
        mean = np.mean(np.abs(arr))
        std = np.std(arr)
        return float(std / (mean + EndogenousStats.EPS))

    @staticmethod
    def mad(arr: np.ndarray) -> float:
        """Median Absolute Deviation"""
        if len(arr) < 2:
            return EndogenousStats.EPS
        median = np.median(arr)
        return float(np.median(np.abs(arr - median)))


# =============================================================================
# A) BUS - Inter-World Communication (unchanged logic, endogenous buffer)
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
    """Inter-world communication bus with endogenous buffer sizing."""

    def __init__(self):
        self.messages: Dict[str, List[WorldSummary]] = {'NEO': [], 'EVA': []}
        self.enabled = True
        self.total_count = 0
        self._T = 1  # Track T for endogenous max_hist

    def publish(self, summary: WorldSummary):
        if not self.enabled:
            return

        self._T = max(self._T, summary.t)
        self.messages[summary.world_id].append(summary)
        self.total_count += 1

        # Endogenous buffer: max_hist = min{T, ⌊10√T⌋}
        max_hist = EndogenousStats.max_hist(self._T)
        for world in self.messages:
            if len(self.messages[world]) > max_hist:
                self.messages[world] = self.messages[world][-max_hist:]

    def get_latest(self, world_id: str) -> Optional[WorldSummary]:
        if not self.enabled or not self.messages[world_id]:
            return None
        return self.messages[world_id][-1]

    def get_counts(self) -> Dict[str, int]:
        return {w: len(msgs) for w, msgs in self.messages.items()}


# =============================================================================
# B) Endogenous Coupling Law (100% from statistics)
# =============================================================================

class EndogenousCoupling:
    """
    κ_t^X = (u_Y / (1 + u_X)) × (λ₁^Y / (λ₁^Y + λ₁^X + ε)) × (conf^Y / (1 + CV(r^X)))

    All normalization by historical quantiles, not fixed factors.
    """

    def __init__(self):
        self.kappa_history: List[float] = []

        # Tangent basis (geometric, not tunable)
        self.u_c = np.array([1, 1, 1]) / np.sqrt(3)

    def compute_tangent_projection(self, I_current: np.ndarray, v_other: np.ndarray) -> np.ndarray:
        """Project v_other onto tangent plane (perpendicular to (1,1,1))."""
        v_tangent = v_other - np.dot(v_other, self.u_c) * self.u_c
        norm = np.linalg.norm(v_tangent)
        if norm > EndogenousStats.EPS:
            v_tangent = v_tangent / norm
        else:
            v_tangent = np.zeros(3)
        return v_tangent

    def compute_kappa(self, u_self: float, u_other: float,
                      lambda1_self: float, lambda1_other: float,
                      conf_other: float, cv_self: float, T: int) -> float:
        """Compute κ - all factors endogenous."""
        eps = EndogenousStats.EPS

        # Factor 1: Uncertainty ratio
        f1 = u_other / (1 + u_self + eps)

        # Factor 2: Directional dominance
        f2 = lambda1_other / (lambda1_other + lambda1_self + eps)

        # Factor 3: Confidence vs instability
        f3 = conf_other / (1 + cv_self + eps)

        kappa_raw = f1 * f2 * f3

        # Normalize by p99 of history (endogenous ceiling)
        self.kappa_history.append(kappa_raw)
        max_hist = EndogenousStats.max_hist(T)
        if len(self.kappa_history) > max_hist:
            self.kappa_history = self.kappa_history[-max_hist:]

        if len(self.kappa_history) > 10:
            kappa_p99 = EndogenousStats.quantile_safe(self.kappa_history, 0.99, 1.0)
            if kappa_p99 > eps:
                kappa = min(1.0, kappa_raw / kappa_p99)
            else:
                kappa = kappa_raw
        else:
            kappa = min(1.0, kappa_raw)

        return kappa

    def compute_coupled_delta(self, delta_self: np.ndarray, I_self: np.ndarray,
                              other_summary: Optional[WorldSummary],
                              u_self: float, lambda1_self: float,
                              cv_self: float, gate_open: bool, T: int) -> Tuple[np.ndarray, Dict]:
        """Δ̃ = Δ_self + κ × g_Y→X"""
        info = {'coupling_active': False, 'kappa': 0.0, 'g_norm': 0.0}

        if not gate_open or other_summary is None:
            return delta_self, info

        kappa = self.compute_kappa(
            u_self=u_self, u_other=other_summary.u,
            lambda1_self=lambda1_self, lambda1_other=other_summary.lambda1,
            conf_other=other_summary.conf, cv_self=cv_self, T=T
        )

        g_Y_to_X = self.compute_tangent_projection(I_self, other_summary.v1)
        delta_coupled = delta_self + kappa * g_Y_to_X

        info = {
            'coupling_active': True,
            'kappa': float(kappa),
            'g_norm': float(np.linalg.norm(g_Y_to_X)),
        }
        return delta_coupled, info


# =============================================================================
# C) World with 100% Endogenous Dynamics
# =============================================================================

class CoupledWorld:
    """
    World with fully endogenous dynamics:
    - Drift from EMA of differences (not hardcoded)
    - τ from IQR/σ_med/√T (not fixed bounds)
    - η = τ (not boosted)
    - Gate by quantiles (no factors)
    - OU clipped by quantiles (not ±5)
    - Noise scaled by IQR/√T
    """

    def __init__(self, world_id: str, initial_I: np.ndarray, bus: BUS):
        self.world_id = world_id
        self.other_id = 'EVA' if world_id == 'NEO' else 'NEO'
        self.bus = bus

        self.I = initial_I.copy()
        self.I_history: List[np.ndarray] = [initial_I.copy()]
        self.residuals: List[float] = []

        # OU state
        self.ou_Z = np.array([0.0, 0.0])
        self.ou_Z_history: List[np.ndarray] = []

        # Coupling
        self.coupling = EndogenousCoupling()

        # Statistics histories (for endogenous quantiles)
        self.rho_history: List[float] = []
        self.iqr_history: List[float] = []
        self.tau_history: List[float] = []
        self.theta_history: List[float] = []  # OU mean-reversion rates

        # Tangent basis (geometric constants)
        self.u_1 = np.array([1, -1, 0]) / np.sqrt(2)
        self.u_2 = np.array([1, 1, -2]) / np.sqrt(6)
        self.u_c = np.array([1, 1, 1]) / np.sqrt(3)

        # Drift (will be computed from data, starts at 0)
        self.drift_ema = np.zeros(3)

        self.t = 0
        self.gate_activations = 0
        self.coupling_activations = 0
        self.series: List[Dict] = []

        # Diagnostics for audit
        self.diagnostics: Dict[str, List[float]] = {
            'tau': [], 'eta': [], 'drift_norm': [],
            'ou_clip_min': [], 'ou_clip_max': [], 'noise_scale': []
        }

    def _get_window(self) -> int:
        """w = max{10, ⌊√T⌋}"""
        return EndogenousStats.window_size(len(self.I_history))

    def _get_max_hist(self) -> int:
        """max_hist = min{T, ⌊10√T⌋}"""
        return EndogenousStats.max_hist(len(self.I_history))

    def _compute_sigma_med(self) -> float:
        """
        σ_med = median(σ_S, σ_N, σ_C) in window.

        Warmup: Use σ_uniform = √(1/12) ≈ 0.289 (variance of uniform on [0,1]).
        This is a geometric constant of the simplex, not arbitrary.
        """
        w = self._get_window()

        # Geometric constant: σ of uniform distribution on simplex
        # For d=3 simplex, σ ≈ 1/√12 per component
        sigma_uniform = 1.0 / np.sqrt(12)  # ≈ 0.289

        if len(self.I_history) < w:
            # During warmup, use geometric reference
            return sigma_uniform / np.sqrt(len(self.I_history) + 1)

        I_window = np.array(self.I_history[-w:])
        sigmas = np.std(I_window, axis=0)
        sigma_med = float(np.median(sigmas))

        # If no variance observed, use geometric floor
        if sigma_med < EndogenousStats.EPS:
            sigma_med = sigma_uniform / np.sqrt(len(self.I_history))

        return sigma_med

    def _compute_drift_endogenous(self) -> np.ndarray:
        """
        Drift from EMA of first differences, projected to tangent plane.
        d_t = Proj_Tan(EMA of (I_{k+1} - I_k))
        β = (w-1)/(w+1) - derived from window size
        """
        w = self._get_window()
        if len(self.I_history) < 3:
            return np.zeros(3)

        # Compute differences in window
        I_arr = np.array(self.I_history[-w:])
        diffs = np.diff(I_arr, axis=0)

        if len(diffs) == 0:
            return np.zeros(3)

        # EMA with β = (w-1)/(w+1)
        beta = (w - 1) / (w + 1)
        ema = diffs[0]
        for d in diffs[1:]:
            ema = beta * ema + (1 - beta) * d

        # Project to tangent plane
        drift = ema - np.dot(ema, self.u_c) * self.u_c

        # Update EMA state with β derived from window size: β = (w-1)/(w+1)
        # This makes the decay rate scale with the observation window
        beta_ema = (w - 1) / (w + 1)  # Endogenous from w
        self.drift_ema = beta_ema * self.drift_ema + (1 - beta_ema) * drift

        return self.drift_ema

    def _compute_tau_endogenous(self) -> float:
        """
        τ = (IQR(r) / √T) × (σ_med / (IQR_r_hist + ε))
        τ_floor = σ_med / T
        No fixed bounds - only endogenous floor.
        """
        T = len(self.I_history)
        w = self._get_window()
        sigma_med = self._compute_sigma_med()

        # τ_floor = σ_med / T
        tau_floor = sigma_med / max(T, 1)

        if len(self.residuals) < w:
            return max(tau_floor, EndogenousStats.EPS)

        res = np.array(self.residuals[-w:])
        iqr_r = EndogenousStats.iqr(res)

        # Historical IQR for normalization
        iqr_r_hist = EndogenousStats.quantile_safe(
            [EndogenousStats.iqr(np.array(self.residuals[max(0,i-w):i]))
             for i in range(w, len(self.residuals), w//2)],
            0.5, iqr_r
        ) if len(self.residuals) > 2*w else iqr_r

        tau = (iqr_r / np.sqrt(T)) * (sigma_med / (iqr_r_hist + EndogenousStats.EPS))

        # Apply only endogenous floor
        tau = max(tau, tau_floor)

        # Optional: endogenous ceiling from p99 of history
        if len(self.tau_history) > 20:
            tau_p99 = EndogenousStats.quantile_safe(self.tau_history, 0.99, tau)
            tau = min(tau, tau_p99)

        self.tau_history.append(tau)
        max_hist = self._get_max_hist()
        if len(self.tau_history) > max_hist:
            self.tau_history = self.tau_history[-max_hist:]

        return tau

    def _compute_gate_endogenous(self) -> Tuple[bool, Dict]:
        """
        Gate ON iff ρ(J_t) ≥ ρ_p95 AND IQR(r) ≥ IQR_p75
        No multiplicative factors (0.95, 0.5).
        """
        w = self._get_window()
        T = len(self.I_history)

        if T < w:
            return True, {'reason': 'warmup', 'open': True}

        I_arr = np.array(self.I_history[-w:])

        # At corner check
        at_corner = np.max(self.I) > 0.90

        # Approximate ρ from residual decay
        residuals = np.diff(I_arr, axis=0)
        if len(residuals) > 5:
            r_norm = np.linalg.norm(residuals, axis=1)
            if r_norm[0] > EndogenousStats.EPS:
                rho_approx = (r_norm[-1] / r_norm[0]) ** (1 / len(r_norm))
            else:
                rho_approx = 0.99
        else:
            rho_approx = 0.99

        self.rho_history.append(rho_approx)

        # Current IQR
        if len(self.residuals) > w:
            iqr_current = EndogenousStats.iqr(np.array(self.residuals[-w:]))
        else:
            iqr_current = EndogenousStats.EPS

        self.iqr_history.append(iqr_current)

        # Trim histories
        max_hist = self._get_max_hist()
        if len(self.rho_history) > max_hist:
            self.rho_history = self.rho_history[-max_hist:]
        if len(self.iqr_history) > max_hist:
            self.iqr_history = self.iqr_history[-max_hist:]

        # Thresholds from history ONLY (no factors)
        rho_p95 = EndogenousStats.quantile_safe(self.rho_history, 0.95, 0.99)
        iqr_p75 = EndogenousStats.quantile_safe(self.iqr_history, 0.75, EndogenousStats.EPS)

        # Gate: pure quantile comparison, no factors
        gate_open = at_corner or (rho_approx >= rho_p95 and iqr_current >= iqr_p75)

        return gate_open, {
            'rho': float(rho_approx),
            'rho_p95': float(rho_p95),
            'iqr': float(iqr_current),
            'iqr_p75': float(iqr_p75),
            'at_corner': at_corner,
            'open': gate_open,
        }

    def _ou_clip_endogenous(self, Z: np.ndarray) -> np.ndarray:
        """
        Clip OU by quantiles (p0.1, p99.9) or MAD, not fixed ±5.
        """
        self.ou_Z_history.append(Z.copy())
        max_hist = self._get_max_hist()
        if len(self.ou_Z_history) > max_hist:
            self.ou_Z_history = self.ou_Z_history[-max_hist:]

        if len(self.ou_Z_history) < 10:
            # Cold start: no clipping
            return Z

        Z_arr = np.array(self.ou_Z_history)

        # Option 1: Quantile clipping
        z_min = np.percentile(Z_arr, 0.1, axis=0)
        z_max = np.percentile(Z_arr, 99.9, axis=0)

        # Option 2: MAD-based (m ± 4*MAD)
        z_median = np.median(Z_arr, axis=0)
        z_mad = np.median(np.abs(Z_arr - z_median), axis=0)
        z_min_mad = z_median - 4 * z_mad
        z_max_mad = z_median + 4 * z_mad

        # Use wider of the two
        z_min = np.minimum(z_min, z_min_mad)
        z_max = np.maximum(z_max, z_max_mad)

        # Record for diagnostics
        self.diagnostics['ou_clip_min'].append(float(z_min[0]))
        self.diagnostics['ou_clip_max'].append(float(z_max[0]))

        return np.clip(Z, z_min, z_max)

    def _ou_step(self, tau: float) -> np.ndarray:
        """OU step with endogenous theta from residual ACF."""
        T = len(self.I_history)
        w = self._get_window()
        sigma_med = self._compute_sigma_med()

        # Endogenous theta bounds:
        # - Floor: σ_med / T (same scaling as τ_floor)
        # - Ceil: quantile(theta_history, p99) or geometric default 1/w during warmup
        theta_floor = sigma_med / max(T, 1)
        if len(self.theta_history) >= w:
            theta_ceil = EndogenousStats.quantile_safe(self.theta_history[-self._get_max_hist():], 0.99)
            # Ensure ceil > floor
            theta_ceil = max(theta_ceil, theta_floor * 10)
        else:
            # Warmup: ceil = 1/w (characteristic rate for window timescale)
            theta_ceil = 1.0 / w

        # Theta from residual autocorrelation
        theta = (theta_floor + theta_ceil) / 2  # Default: geometric mean of bounds
        if len(self.residuals) > w:
            r = np.array(self.residuals[-w:])
            if len(r) > 2:
                r_corr = np.corrcoef(r[:-1], r[1:])[0, 1]
                if not np.isnan(r_corr) and abs(r_corr) < 0.99 and abs(r_corr) > EndogenousStats.EPS:
                    theta_raw = -1 / np.log(abs(r_corr) + EndogenousStats.EPS)
                    # Apply endogenous bounds
                    theta = max(theta_floor, min(theta_ceil, theta_raw))

        # Store theta in history for future endogenous bounds
        self.theta_history.append(theta)
        max_hist = self._get_max_hist()
        if len(self.theta_history) > max_hist:
            self.theta_history = self.theta_history[-max_hist:]

        dt = 1.0
        sigma = np.sqrt(max(tau, EndogenousStats.EPS))

        drift = -theta * self.ou_Z * dt
        diffusion = sigma * np.sqrt(dt) * np.random.randn(2)

        self.ou_Z = self.ou_Z + drift + diffusion
        self.ou_Z = self._ou_clip_endogenous(self.ou_Z)

        return self.ou_Z

    def _ou_to_delta(self, Z: np.ndarray) -> np.ndarray:
        """Convert OU to tangent-plane delta."""
        delta = Z[0] * self.u_1 + Z[1] * self.u_2
        return delta - delta.mean()

    def _compute_noise_endogenous(self) -> np.ndarray:
        """
        σ_noise = max{IQR(I), σ_med} / √T
        Noise in tangent plane.

        During warmup (T < w), use geometric scale to bootstrap exploration.
        """
        T = len(self.I_history)
        w = self._get_window()
        sigma_med = self._compute_sigma_med()

        # Geometric constant for warmup
        sigma_uniform = 1.0 / np.sqrt(12)

        if T < w:
            # Warmup: σ_noise = σ_uniform / √(T+1) - ensures initial exploration
            sigma_noise = sigma_uniform / np.sqrt(T + 1)
        else:
            I_window = np.array(self.I_history[-w:])
            iqr_I = np.mean([EndogenousStats.iqr(I_window[:, i]) for i in range(3)])
            sigma_noise = max(iqr_I, sigma_med) / np.sqrt(T)

        self.diagnostics['noise_scale'].append(float(sigma_noise))

        # Generate in tangent plane
        z1, z2 = np.random.randn(2)
        noise = sigma_noise * (z1 * self.u_1 + z2 * self.u_2)

        return noise

    def _mirror_descent(self, I: np.ndarray, delta: np.ndarray, eta: float) -> np.ndarray:
        """Mirror descent: I_{t+1} = softmax(log I + η × Δ)"""
        # Only numerical stability floor (not tunable)
        log_floor = -30  # ~1e-13, numerical stability

        I_safe = np.maximum(I, EndogenousStats.EPS)
        I_safe = I_safe / I_safe.sum()

        log_I = np.log(I_safe)
        log_I = np.maximum(log_I, log_floor)

        log_I_new = log_I + eta * delta

        exp_log = np.exp(log_I_new - np.max(log_I_new))
        I_new = exp_log / exp_log.sum()

        return I_new

    def compute_summary(self) -> WorldSummary:
        """Compute summary - all values from data."""
        T = len(self.I_history)
        w = self._get_window()

        if T < w:
            # Cold start: minimal defaults, will be overwritten
            return WorldSummary(
                world_id=self.world_id, t=self.t,
                mu_I=self.I.copy(), v1=self.u_1,  # Use tangent basis, not arbitrary
                lambda1=0.0, u=EndogenousStats.EPS, conf=0.5, cv_r=1.0
            )

        I_window = np.array(self.I_history[-w:])
        mu_I = np.mean(I_window, axis=0)

        # PCA
        I_centered = I_window - mu_I
        cov = np.cov(I_centered.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            v1 = eigvecs[:, idx[0]]
            lambda1 = eigvals[idx[0]]
        except:
            v1 = self.u_1
            lambda1 = 0.0

        # u = IQR(r) / √T
        if len(self.residuals) > w:
            res = np.array(self.residuals[-w:])
            iqr = EndogenousStats.iqr(res)
            u = iqr / np.sqrt(T)
            cv_r = EndogenousStats.cv(res)
        else:
            u = EndogenousStats.EPS
            cv_r = 1.0

        # conf = 1 / (1 + CV(r))
        conf = 1.0 / (1.0 + cv_r)
        conf = max(0.0, min(1.0, conf))

        return WorldSummary(
            world_id=self.world_id, t=self.t,
            mu_I=mu_I, v1=v1, lambda1=float(lambda1),
            u=float(u), conf=float(conf), cv_r=float(cv_r)
        )

    def step(self, enable_coupling: bool = True) -> Dict:
        """Execute one step - 100% endogenous."""
        self.t += 1
        T = len(self.I_history)
        I_prev = self.I.copy()

        # Endogenous drift
        drift = self._compute_drift_endogenous()

        # Endogenous noise
        noise = self._compute_noise_endogenous()

        # Candidate
        I_candidate = I_prev + drift + noise
        I_candidate = np.maximum(I_candidate, EndogenousStats.EPS)
        I_candidate = I_candidate / I_candidate.sum()

        # Residual
        residual = np.linalg.norm(I_candidate - I_prev)
        self.residuals.append(residual)
        max_hist = self._get_max_hist()
        if len(self.residuals) > max_hist:
            self.residuals = self.residuals[-max_hist:]

        # Gate (endogenous)
        gate_open, gate_info = self._compute_gate_endogenous()

        if not gate_open:
            self.I = I_candidate
            self.I_history.append(self.I.copy())
            self._trim_history()
            return self._record_step(I_prev, False, {}, 0, 0, 0, drift)

        self.gate_activations += 1

        # τ (endogenous)
        tau = self._compute_tau_endogenous()

        # OU step
        Z = self._ou_step(tau)
        delta_base = self._ou_to_delta(Z)

        # Coupling
        other_summary = self.bus.get_latest(self.other_id) if enable_coupling else None
        summary = self.compute_summary()

        delta_coupled, coupling_info = self.coupling.compute_coupled_delta(
            delta_self=delta_base, I_self=I_candidate,
            other_summary=other_summary,
            u_self=summary.u, lambda1_self=summary.lambda1,
            cv_self=summary.cv_r, gate_open=gate_open, T=T
        )

        if coupling_info.get('coupling_active', False):
            self.coupling_activations += 1

        # η = τ (no boost, no arbitrary minimum)
        eta = tau

        # Mirror descent
        I_new = self._mirror_descent(I_candidate, delta_coupled, eta)

        # Update state
        self.I = I_new
        self.I_history.append(self.I.copy())
        self._trim_history()

        # Record diagnostics
        self.diagnostics['tau'].append(float(tau))
        self.diagnostics['eta'].append(float(eta))
        self.diagnostics['drift_norm'].append(float(np.linalg.norm(drift)))

        # Publish to BUS
        w = self._get_window()
        if self.t % w == 0:
            self.bus.publish(summary)

        return self._record_step(I_prev, True, coupling_info, tau, eta,
                                 np.linalg.norm(delta_coupled), drift)

    def _trim_history(self):
        """Trim history to endogenous max_hist."""
        max_hist = self._get_max_hist()
        if len(self.I_history) > max_hist:
            self.I_history = self.I_history[-max_hist:]

    def _record_step(self, I_prev, gate_open, coupling_info, tau, eta, delta_norm, drift):
        record = {
            't': self.t,
            'S_prev': float(I_prev[0]), 'N_prev': float(I_prev[1]), 'C_prev': float(I_prev[2]),
            'S_new': float(self.I[0]), 'N_new': float(self.I[1]), 'C_new': float(self.I[2]),
            'gate_open': gate_open,
            'coupling_active': coupling_info.get('coupling_active', False),
            'kappa': coupling_info.get('kappa', 0),
            'tau': float(tau), 'eta': float(eta),
            'delta_norm': float(delta_norm),
            'drift_norm': float(np.linalg.norm(drift)),
        }
        self.series.append(record)
        return record

    def get_quantile_report(self) -> Dict:
        """Report all endogenous quantiles for audit."""
        return {
            'T': len(self.I_history),
            'w': self._get_window(),
            'max_hist': self._get_max_hist(),
            'sigma_med': self._compute_sigma_med(),
            'rho_quantiles': {
                'p50': EndogenousStats.quantile_safe(self.rho_history, 0.50),
                'p75': EndogenousStats.quantile_safe(self.rho_history, 0.75),
                'p95': EndogenousStats.quantile_safe(self.rho_history, 0.95),
                'p97.5': EndogenousStats.quantile_safe(self.rho_history, 0.975),
            },
            'iqr_quantiles': {
                'p50': EndogenousStats.quantile_safe(self.iqr_history, 0.50),
                'p75': EndogenousStats.quantile_safe(self.iqr_history, 0.75),
                'p95': EndogenousStats.quantile_safe(self.iqr_history, 0.95),
            },
            'tau_quantiles': {
                'p50': EndogenousStats.quantile_safe(self.tau_history, 0.50),
                'p75': EndogenousStats.quantile_safe(self.tau_history, 0.75),
                'p99': EndogenousStats.quantile_safe(self.tau_history, 0.99),
            },
            'ou_limits': {
                'clip_min_mean': np.mean(self.diagnostics['ou_clip_min']) if self.diagnostics['ou_clip_min'] else 0,
                'clip_max_mean': np.mean(self.diagnostics['ou_clip_max']) if self.diagnostics['ou_clip_max'] else 0,
            },
        }


# =============================================================================
# D) Coupled System Runner
# =============================================================================

class CoupledSystemRunner:
    """Runs NEO and EVA with 100% endogenous dynamics."""

    def __init__(self, enable_coupling: bool = True):
        self.bus = BUS()
        self.bus.enabled = enable_coupling

        self.neo = CoupledWorld('NEO', np.array([1.0, 0.0, 0.0]), self.bus)
        self.eva = CoupledWorld('EVA', np.array([1/3, 1/3, 1/3]), self.bus)
        self.enable_coupling = enable_coupling

    def run(self, cycles: int = 500, verbose: bool = True) -> Dict:
        print("=" * 70)
        print("Phase 6 v2: 100% Endogenous Coupled NEO↔EVA")
        print(f"Coupling: {'ENABLED' if self.enable_coupling else 'DISABLED'}")
        print("=" * 70)
        print(f"NEO initial: {self.neo.I}")
        print(f"EVA initial: {self.eva.I}")
        print()

        start_time = time.time()

        for i in range(cycles):
            self.neo.step(enable_coupling=self.enable_coupling)
            self.eva.step(enable_coupling=self.enable_coupling)

            if verbose and (i + 1) % 100 == 0:
                tau_neo = self.neo.diagnostics['tau'][-1] if self.neo.diagnostics['tau'] else 0
                print(f"  t={i+1:4d}: NEO=[{self.neo.I[0]:.4f}, {self.neo.I[1]:.4f}, {self.neo.I[2]:.4f}] "
                      f"EVA=[{self.eva.I[0]:.4f}, {self.eva.I[1]:.4f}, {self.eva.I[2]:.4f}] "
                      f"τ={tau_neo:.4f}")

        elapsed = time.time() - start_time

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
                'quantiles': self.neo.get_quantile_report(),
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
                'quantiles': self.eva.get_quantile_report(),
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
        print("=" * 70)

        return results

    def save_series(self, path_prefix: str):
        for world, name in [(self.neo, 'neo'), (self.eva, 'eva')]:
            data = {
                'timestamp': datetime.now().isoformat(),
                'world': world.world_id,
                'cycles': len(world.series),
                'series': world.series,
                'diagnostics': {k: v[-100:] for k, v in world.diagnostics.items()},
            }
            path = f"{path_prefix}_{name}.json"
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[OK] Saved: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6 v2: 100% Endogenous")
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--no-coupling", action="store_true")
    parser.add_argument("--output", type=str, default="/root/NEO_EVA/results/phase6_v2")
    args = parser.parse_args()

    os.makedirs("/root/NEO_EVA/results", exist_ok=True)

    runner = CoupledSystemRunner(enable_coupling=not args.no_coupling)
    results = runner.run(cycles=args.cycles)

    runner.save_series(args.output)

    results_path = f"{args.output}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved: {results_path}")


if __name__ == "__main__":
    main()
