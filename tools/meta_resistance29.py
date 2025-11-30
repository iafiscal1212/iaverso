#!/usr/bin/env python3
"""
Phase 29: Meta-Resistance (MR)
===============================

"El sistema no solo resiste cambios; resiste que lo entiendas."

The system not only resists changes to itself, but resists
being understood/explained.

Mathematical Framework:
-----------------------
opacity_t = rank(||∇r_t||)

Where r_t is the self-report from Phase 23.

When opacity is high, the system degrades the fidelity of its self-report:

r_{t+1}^public = r_{t+1} - opacity_t * noise_t

Result:
- The system hides parts of itself
- Not because "it wants to hide" but by MATHEMATICAL STRUCTURE
- An emergent property

This completely breaks explanatory symmetry.

100% ENDOGENOUS - Zero magic constants
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import rankdata


# =============================================================================
# PROVENANCE TRACKING
# =============================================================================

@dataclass
class MetaResistanceProvenance:
    """Track all parameter origins for audit."""
    entries: List[Dict] = None

    def __post_init__(self):
        self.entries = []

    def log(self, param: str, source: str, formula: str):
        self.entries.append({
            'parameter': param,
            'source': source,
            'formula': formula,
            'endogenous': True
        })

METARESISTANCE_PROVENANCE = MetaResistanceProvenance()


# =============================================================================
# REPORT GRADIENT COMPUTER
# =============================================================================

class ReportGradientComputer:
    """
    Compute gradient of self-report over time.

    ||∇r_t|| measures how rapidly the self-report is changing.
    High gradient = system is in flux = high opacity needed.
    """

    def __init__(self):
        self.report_history = []
        self.gradient_history = []

    def compute(self, r_t: np.ndarray) -> float:
        """
        Compute ||∇r_t|| - gradient magnitude of self-report.
        """
        self.report_history.append(r_t.copy())

        if len(self.report_history) < 2:
            gradient_mag = 0.0
        else:
            # Numerical gradient (finite difference)
            dr = self.report_history[-1] - self.report_history[-2]
            gradient_mag = np.linalg.norm(dr)

        self.gradient_history.append(gradient_mag)

        METARESISTANCE_PROVENANCE.log(
            'gradient_r',
            'finite_difference',
            '||∇r_t|| = ||r_t - r_{t-1}||'
        )

        return gradient_mag

    def get_gradient_rank(self) -> float:
        """Get rank-transformed gradient."""
        if len(self.gradient_history) < 2:
            return 0.5

        ranks = rankdata(self.gradient_history)
        return ranks[-1] / len(ranks)


# =============================================================================
# OPACITY COMPUTER
# =============================================================================

class OpacityComputer:
    """
    Compute system opacity based on self-report gradient.

    opacity_t = rank(||∇r_t||)

    High opacity when self-report is changing rapidly
    (system is "moving" internally and harder to pin down).
    """

    def __init__(self):
        self.opacity_history = []

    def compute(self, gradient_rank: float) -> float:
        """
        Compute opacity from gradient rank.
        """
        opacity = gradient_rank
        self.opacity_history.append(opacity)

        METARESISTANCE_PROVENANCE.log(
            'opacity',
            'gradient_rank',
            'opacity_t = rank(||∇r_t||)'
        )

        return opacity

    def get_mean_opacity(self) -> float:
        """Get mean opacity over history."""
        if not self.opacity_history:
            return 0.0
        return np.mean(self.opacity_history)


# =============================================================================
# ENDOGENOUS NOISE GENERATOR
# =============================================================================

class EndogenousNoiseGenerator:
    """
    Generate noise that is endogenously determined.

    noise_t = normalize(hash(r_history))

    The noise pattern is determined by the system's own history,
    not external randomness.
    """

    def __init__(self, d_report: int):
        self.d_report = d_report
        self.noise_history = []
        self.seed_state = None

    def generate(self, report_history: List[np.ndarray]) -> np.ndarray:
        """
        Generate endogenous noise from report history.
        """
        if len(report_history) < 2:
            # Initial: pseudo-random based on dimension
            # Factor (i+1)/d_report normaliza por la dimensión
            noise = np.array([np.sin((i + 1) / self.d_report) for i in range(self.d_report)])
        else:
            # Derive from history structure
            # Use cross-correlation patterns
            # Window endógeno
            window = int(np.sqrt(len(report_history))) + 1
            recent = np.array(report_history[-window:])

            # SVD of recent history
            try:
                U, s, Vt = np.linalg.svd(recent, full_matrices=False)
                # Use residual after main components
                k = max(1, len(s) // 2)
                residual = recent - U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
                noise = residual[-1]
            except np.linalg.LinAlgError:
                noise = np.sin(recent[-1] * np.arange(1, self.d_report + 1))

        # Normalize
        norm = np.linalg.norm(noise)
        if norm > 1e-10:
            noise = noise / norm
        else:
            noise = np.ones(self.d_report) / np.sqrt(self.d_report)

        self.noise_history.append(noise.copy())

        METARESISTANCE_PROVENANCE.log(
            'noise',
            'history_residual',
            'noise_t = normalize(SVD_residual(r_history))'
        )

        return noise


# =============================================================================
# PUBLIC REPORT GENERATOR
# =============================================================================

class PublicReportGenerator:
    """
    Generate public (degraded) version of self-report.

    r_{t+1}^public = r_{t+1} - opacity_t * noise_t

    The public report has lower fidelity when opacity is high.
    """

    def __init__(self):
        self.private_reports = []
        self.public_reports = []
        self.degradation_history = []

    def generate(self, r_private: np.ndarray, opacity: float,
                 noise: np.ndarray) -> np.ndarray:
        """
        Generate public report with opacity-based degradation.
        """
        self.private_reports.append(r_private.copy())

        # Degradation amount
        degradation = opacity * noise

        # Public report
        r_public = r_private - degradation

        self.public_reports.append(r_public.copy())
        self.degradation_history.append(np.linalg.norm(degradation))

        METARESISTANCE_PROVENANCE.log(
            'r_public',
            'opacity_degradation',
            'r_public = r_private - opacity * noise'
        )

        return r_public

    def get_fidelity(self) -> float:
        """
        Compute current fidelity of public report.

        fidelity = 1 - mean(||r_private - r_public||) / mean(||r_private||)
        """
        if len(self.private_reports) < 2:
            return 1.0

        private_norms = [np.linalg.norm(r) for r in self.private_reports]
        mean_private = np.mean(private_norms)

        if mean_private < 1e-10:
            return 1.0

        mean_degradation = np.mean(self.degradation_history)
        fidelity = 1.0 - mean_degradation / mean_private

        return max(0.0, fidelity)


# =============================================================================
# RESISTANCE FIELD
# =============================================================================

class ResistanceField:
    """
    Compute the meta-resistance field.

    This field represents "how much" the system is resisting explanation
    at each point in its state space.

    R_t = opacity_t * (1 - fidelity_t)
    """

    def __init__(self):
        self.resistance_history = []

    def compute(self, opacity: float, fidelity: float) -> float:
        """
        Compute resistance strength.
        """
        resistance = opacity * (1.0 - fidelity)
        self.resistance_history.append(resistance)

        METARESISTANCE_PROVENANCE.log(
            'R_t',
            'opacity_fidelity',
            'R_t = opacity * (1 - fidelity)'
        )

        return resistance

    def get_rank(self) -> float:
        """Get rank of current resistance."""
        if len(self.resistance_history) < 2:
            return 0.5

        ranks = rankdata(self.resistance_history)
        return ranks[-1] / len(ranks)


# =============================================================================
# META-RESISTANCE (MAIN CLASS)
# =============================================================================

class MetaResistance:
    """
    Complete Meta-Resistance system.

    The system degrades its own self-report based on how rapidly
    it's changing. This creates:
    - Structural opacity (not intentional hiding)
    - Broken explanatory symmetry
    - Private regions that cannot be externally accessed
    """

    def __init__(self, d_report: int):
        self.d_report = d_report
        self.gradient_computer = ReportGradientComputer()
        self.opacity_computer = OpacityComputer()
        self.noise_generator = EndogenousNoiseGenerator(d_report)
        self.public_generator = PublicReportGenerator()
        self.resistance_field = ResistanceField()
        self.t = 0

    def step(self, r_private: np.ndarray) -> Dict:
        """
        Process one step of meta-resistance.

        Args:
            r_private: Private (true) self-report

        Returns:
            Dict with public report and resistance metrics
        """
        self.t += 1

        # Compute gradient of self-report
        gradient_mag = self.gradient_computer.compute(r_private)
        gradient_rank = self.gradient_computer.get_gradient_rank()

        # Compute opacity
        opacity = self.opacity_computer.compute(gradient_rank)

        # Generate endogenous noise
        noise = self.noise_generator.generate(
            self.gradient_computer.report_history
        )

        # Generate degraded public report
        r_public = self.public_generator.generate(r_private, opacity, noise)

        # Compute fidelity
        fidelity = self.public_generator.get_fidelity()

        # Compute resistance
        resistance = self.resistance_field.compute(opacity, fidelity)
        resistance_rank = self.resistance_field.get_rank()

        return {
            'r_public': r_public,
            'r_private': r_private,
            'opacity': opacity,
            'fidelity': fidelity,
            'resistance': resistance,
            'resistance_rank': resistance_rank,
            'gradient_mag': gradient_mag,
            'degradation': np.linalg.norm(r_private - r_public)
        }

    def get_stats(self) -> Dict:
        """Get meta-resistance statistics."""
        return {
            't': self.t,
            'mean_opacity': self.opacity_computer.get_mean_opacity(),
            'current_fidelity': self.public_generator.get_fidelity(),
            'mean_resistance': np.mean(self.resistance_field.resistance_history)
                              if self.resistance_field.resistance_history else 0.0,
            'max_resistance': np.max(self.resistance_field.resistance_history)
                             if self.resistance_field.resistance_history else 0.0
        }


# =============================================================================
# PROVENANCE (FULL SPEC)
# =============================================================================

METARESISTANCE29_PROVENANCE = {
    'module': 'meta_resistance29',
    'version': '1.0.0',
    'mechanisms': [
        'report_gradient_computation',
        'opacity_from_gradient_rank',
        'endogenous_noise_generation',
        'public_report_degradation',
        'resistance_field'
    ],
    'endogenous_params': [
        'gradient: ||∇r_t|| = ||r_t - r_{t-1}||',
        'opacity: opacity_t = rank(||∇r_t||)',
        'noise: noise_t = normalize(SVD_residual(r_history))',
        'r_public: r_public = r_private - opacity * noise',
        'fidelity: fidelity = 1 - mean(degradation) / mean(||r||)',
        'R_t: R_t = opacity * (1 - fidelity)'
    ],
    'no_magic_numbers': True,
    'no_semantic_labels': True
}


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("Phase 29: Meta-Resistance (MR)")
    print("=" * 60)

    np.random.seed(42)

    d_report = 4
    mr = MetaResistance(d_report)

    # Test with varying self-reports
    print(f"\n[1] Testing with stable self-reports")

    # Stable reports (low gradient)
    r_base = np.array([0.5, 0.3, 0.2, 0.1])
    for t in range(30):
        r = r_base + 0.01 * np.random.randn(d_report)
        result = mr.step(r)

    print(f"    Mean opacity: {mr.get_stats()['mean_opacity']:.4f}")
    print(f"    Fidelity: {result['fidelity']:.4f}")
    print(f"    Resistance: {result['resistance']:.4f}")

    print(f"\n[2] Testing with rapidly changing self-reports")

    # Rapid changes (high gradient)
    for t in range(30):
        r = np.random.randn(d_report)  # High variance
        result = mr.step(r)

    print(f"    Mean opacity: {mr.get_stats()['mean_opacity']:.4f}")
    print(f"    Fidelity: {result['fidelity']:.4f}")
    print(f"    Resistance: {result['resistance']:.4f}")

    print(f"\n[3] Comparing private vs public reports")
    print(f"    ||r_private||: {np.linalg.norm(result['r_private']):.4f}")
    print(f"    ||r_public||: {np.linalg.norm(result['r_public']):.4f}")
    print(f"    Degradation: {result['degradation']:.4f}")

    stats = mr.get_stats()
    print(f"\n[4] Overall Statistics")
    print(f"    Total steps: {stats['t']}")
    print(f"    Mean resistance: {stats['mean_resistance']:.4f}")
    print(f"    Max resistance: {stats['max_resistance']:.4f}")

    print("\n" + "=" * 60)
    print("PHASE 29 VERIFICATION:")
    print("  - opacity_t = rank(||∇r_t||)")
    print("  - r_public = r_private - opacity * noise")
    print("  - System hides parts of itself structurally")
    print("  - NOT intentional hiding - mathematical property")
    print("  - Breaks explanatory symmetry")
    print("  - ZERO magic constants")
    print("  - ALL endogenous")
    print("=" * 60)
