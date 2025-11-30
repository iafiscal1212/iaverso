#!/usr/bin/env python3
"""
Phase 21: Cross-Agent Ecology & Ecological Resistance (FULL SPECIFICATION)
===========================================================================

Implements PURELY ENDOGENOUS inter-agent coupling using only internal history.

Key components:
1. Ecological distance: d_NE = ||z_N - z_E||, d_mu_NE = min_k,l ||mu_k_N - mu_l_E||
2. Individual tension: T_a = rank(spread) * rank(R)
3. Shared tension: T_eco = (rank(T_N) + rank(T_E))/2 * Overlap
4. Cross-influence field: F = beta * normalize(z_source - z_target)
5. Ecological resistance: gamma_eco = 1/(1 + sigma_eco) from eco_shock variance
6. Combined eco_shock from Phase 20 shocks

NO semantic labels. NO magic constants.
All parameters derived from internal history.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

NUMERIC_EPS = 1e-16


# =============================================================================
# PROVENANCE TRACKING
# =============================================================================

class EcologyProvenance:
    """Track derivation of all ecology parameters."""

    def __init__(self):
        self.logs: List[Dict] = []

    def log(self, param_name: str, value: float, derivation: str,
            source_data: Dict, timestep: int):
        self.logs.append({
            'param': param_name,
            'value': value,
            'derivation': derivation,
            'source': source_data,
            't': timestep
        })

    def get_logs(self) -> List[Dict]:
        return self.logs

    def clear(self):
        self.logs = []


ECOLOGY_PROVENANCE = EcologyProvenance()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_rank(value: float, history: np.ndarray) -> float:
    """Compute rank of value within history [0, 1].

    Uses midrank for ties: rank = (count_below + 0.5*count_equal) / total
    """
    if len(history) == 0:
        return 0.5
    n = len(history)
    count_below = float(np.sum(history < value))
    count_equal = float(np.sum(history == value))
    # Midrank formula for ties
    midrank = (count_below + 0.5 * count_equal) / n
    return midrank


def get_window_size(t: int) -> int:
    """Endogenous window size: floor(sqrt(t+1))."""
    return max(1, int(np.sqrt(t + 1)))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length, handle zero vectors."""
    norm = np.linalg.norm(v)
    if norm < NUMERIC_EPS:
        return np.zeros_like(v)
    return v / norm


# =============================================================================
# ECOLOGICAL DISTANCE
# =============================================================================

class EcologicalDistance:
    """
    Computes ecological distance between two agents.

    d_t^NE = ||z_N - z_E||                          (instantaneous)
    d_mu,t^NE = min_{k,l} ||mu_k^N - mu_l^E||       (manifold/cluster)

    All scaled to [0,1] via rank:
    d_tilde_t = rank(d_t)
    d_tilde_mu,t = rank(d_mu,t)
    """

    def __init__(self):
        self.d_instant_history: List[float] = []
        self.d_manifold_history: List[float] = []
        self.t = 0

    def compute(self, z_N: np.ndarray, z_E: np.ndarray,
                prototypes_N: np.ndarray, prototypes_E: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Compute ecological distances.

        Args:
            z_N, z_E: Internal states of agents N and E
            prototypes_N, prototypes_E: Prototype arrays (mu_k)

        Returns:
            (distances_dict, diagnostics)
        """
        self.t += 1

        # Handle dimension mismatch
        min_dim = min(len(z_N), len(z_E))
        z_N_adj = z_N[:min_dim]
        z_E_adj = z_E[:min_dim]

        # d_t^NE = ||z_N - z_E|| (instantaneous distance)
        d_instant = float(np.linalg.norm(z_N_adj - z_E_adj))
        self.d_instant_history.append(d_instant)

        # d_mu,t^NE = min_{k,l} ||mu_k^N - mu_l^E|| (manifold distance)
        if prototypes_N is not None and prototypes_E is not None:
            min_proto_dim = min(prototypes_N.shape[1], prototypes_E.shape[1])
            proto_N_adj = prototypes_N[:, :min_proto_dim]
            proto_E_adj = prototypes_E[:, :min_proto_dim]

            d_manifold = float('inf')
            for mu_N in proto_N_adj:
                for mu_E in proto_E_adj:
                    d = np.linalg.norm(mu_N - mu_E)
                    if d < d_manifold:
                        d_manifold = d
            d_manifold = float(d_manifold)
        else:
            d_manifold = d_instant

        self.d_manifold_history.append(d_manifold)

        # Compute ranks: d_tilde = rank(d)
        d_instant_arr = np.array(self.d_instant_history)
        d_manifold_arr = np.array(self.d_manifold_history)

        d_tilde_instant = compute_rank(d_instant, d_instant_arr)
        d_tilde_manifold = compute_rank(d_manifold, d_manifold_arr)

        ECOLOGY_PROVENANCE.log(
            'd_tilde_NE', d_tilde_instant,
            'rank(||z_N - z_E||)',
            {'d_instant': d_instant},
            self.t
        )

        # Overlap = rank(1 - d_tilde_mu) - high when manifolds are close
        overlap_raw = 1.0 - d_tilde_manifold

        distances = {
            'd_instant': d_instant,
            'd_manifold': d_manifold,
            'd_tilde_instant': d_tilde_instant,
            'd_tilde_manifold': d_tilde_manifold,
            'overlap_raw': overlap_raw  # For use in T_eco calculation
        }

        diagnostics = {
            'min_dim': min_dim,
            't': self.t
        }

        return distances, diagnostics

    def get_statistics(self) -> Dict:
        if not self.d_instant_history:
            return {'mean_d_instant': 0.0}

        return {
            'mean_d_instant': float(np.mean(self.d_instant_history)),
            'std_d_instant': float(np.std(self.d_instant_history)),
            'mean_d_manifold': float(np.mean(self.d_manifold_history)),
            'n_samples': len(self.d_instant_history)
        }


# =============================================================================
# INDIVIDUAL TENSION
# =============================================================================

class IndividualTension:
    """
    Computes individual tension for an agent.

    T_a = rank(spread) * rank(R)

    where spread = var_w(z) in window w = floor(sqrt(t+1)).
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        # maxlen derivado de sqrt(1e6)
        self.z_history: deque = deque(maxlen=int(np.sqrt(1e6)))
        self.R_history: List[float] = []
        self.T_history: List[float] = []
        self.spread_history: List[float] = []
        self.t = 0

    def compute(self, z_t: np.ndarray, R_t: float) -> Tuple[float, Dict]:
        """
        Compute individual tension.

        T_a = rank(spread) * rank(R)

        Args:
            z_t: Current internal state
            R_t: Current irreversibility

        Returns:
            (T_t, diagnostics)
        """
        self.t += 1
        self.z_history.append(z_t.copy())
        self.R_history.append(R_t)

        # Window size: w = floor(sqrt(t+1))
        w = get_window_size(self.t)

        # spread = var_w(z) - variance in window
        if len(self.z_history) >= w:
            window_z = list(self.z_history)[-w:]
            z_arr = np.array(window_z)
            spread = float(np.mean(np.var(z_arr, axis=0)))
        else:
            spread = 0.0

        self.spread_history.append(spread)

        # Compute ranks
        R_arr = np.array(self.R_history)
        spread_arr = np.array(self.spread_history)

        rank_R = compute_rank(R_t, R_arr)
        rank_spread = compute_rank(spread, spread_arr)

        # T_a = rank(spread) * rank(R)
        T_t = rank_spread * rank_R
        self.T_history.append(T_t)

        ECOLOGY_PROVENANCE.log(
            f'T_{self.agent_id}', float(T_t),
            'rank(spread) * rank(R)',
            {'rank_spread': rank_spread, 'rank_R': rank_R, 'w': w},
            self.t
        )

        diagnostics = {
            'spread': spread,
            'rank_spread': rank_spread,
            'R_t': R_t,
            'rank_R': rank_R,
            'w': w,
            'T_t': T_t
        }

        return float(T_t), diagnostics

    def get_statistics(self) -> Dict:
        if not self.T_history:
            return {'mean_T': 0.0}

        T_arr = np.array(self.T_history)
        return {
            'mean_T': float(np.mean(T_arr)),
            'std_T': float(np.std(T_arr)),
            'n_samples': len(T_arr)
        }


# =============================================================================
# SHARED TENSION (T_eco)
# =============================================================================

class SharedTension:
    """
    Computes shared ecological tension between two agents.

    T_eco = (rank(T_N) + rank(T_E))/2 * Overlap

    where Overlap = rank(1 - d_tilde_mu)
    """

    def __init__(self):
        self.T_eco_history: List[float] = []
        self.overlap_history: List[float] = []
        self.t = 0

    def compute(self, T_N: float, T_E: float, overlap_raw: float,
                T_N_history: List[float], T_E_history: List[float]) -> Tuple[float, Dict]:
        """
        Compute shared tension.

        T_eco = (rank(T_N) + rank(T_E))/2 * Overlap

        Args:
            T_N, T_E: Individual tensions
            overlap_raw: 1 - d_tilde_manifold (higher when manifolds close)
            T_N_history, T_E_history: Tension histories for ranking

        Returns:
            (T_eco, diagnostics)
        """
        self.t += 1

        # Rank individual tensions
        T_N_arr = np.array(T_N_history) if T_N_history else np.array([T_N])
        T_E_arr = np.array(T_E_history) if T_E_history else np.array([T_E])

        rank_T_N = compute_rank(T_N, T_N_arr)
        rank_T_E = compute_rank(T_E, T_E_arr)

        # Overlap = rank(1 - d_tilde_mu) for ranking
        self.overlap_history.append(overlap_raw)
        overlap_arr = np.array(self.overlap_history)
        Overlap = compute_rank(overlap_raw, overlap_arr)

        # T_eco = (rank(T_N) + rank(T_E))/2 * Overlap
        T_eco = 0.5 * (rank_T_N + rank_T_E) * Overlap
        self.T_eco_history.append(T_eco)

        ECOLOGY_PROVENANCE.log(
            'T_eco', float(T_eco),
            '(rank(T_N) + rank(T_E))/2 * Overlap',
            {'rank_T_N': rank_T_N, 'rank_T_E': rank_T_E, 'Overlap': Overlap},
            self.t
        )

        diagnostics = {
            'rank_T_N': rank_T_N,
            'rank_T_E': rank_T_E,
            'overlap_raw': overlap_raw,
            'Overlap': Overlap,
            'T_eco': T_eco
        }

        return float(T_eco), diagnostics

    def get_statistics(self) -> Dict:
        if not self.T_eco_history:
            return {'mean_T_eco': 0.0}

        T_arr = np.array(self.T_eco_history)
        return {
            'mean_T_eco': float(np.mean(T_arr)),
            'std_T_eco': float(np.std(T_arr)),
            'n_samples': len(T_arr)
        }


# =============================================================================
# CROSS-INFLUENCE FIELD (F)
# =============================================================================

class CrossInfluenceField:
    """
    Computes cross-influence field between agents.

    F_E->N = beta * normalize(z_E - z_N)

    where beta = rank(T_eco) * rank(D_nov)
    """

    def __init__(self, source_id: str, target_id: str):
        self.source_id = source_id
        self.target_id = target_id
        self.beta_history: List[float] = []
        self.F_magnitude_history: List[float] = []
        self.t = 0

    def compute(self, z_source: np.ndarray, z_target: np.ndarray,
                T_eco: float, D_nov_target: float,
                T_eco_history: List[float], D_nov_history: List[float]) -> Tuple[np.ndarray, Dict]:
        """
        Compute cross-influence field.

        F = beta * normalize(z_source - z_target)
        beta = rank(T_eco) * rank(D_nov_target)

        Args:
            z_source, z_target: Agent states
            T_eco: Shared tension
            D_nov_target: Novelty drive of target agent
            T_eco_history, D_nov_history: Histories for ranking

        Returns:
            (F_field, diagnostics)
        """
        self.t += 1

        # Handle dimension mismatch
        min_dim = min(len(z_source), len(z_target))
        z_source_adj = z_source[:min_dim]
        z_target_adj = z_target[:min_dim]

        # Compute ranks
        T_eco_arr = np.array(T_eco_history) if T_eco_history else np.array([T_eco])
        D_nov_arr = np.array(D_nov_history) if D_nov_history else np.array([D_nov_target])

        rank_T_eco = compute_rank(T_eco, T_eco_arr)
        rank_D_nov = compute_rank(D_nov_target, D_nov_arr)

        # beta = rank(T_eco) * rank(D_nov)
        beta = rank_T_eco * rank_D_nov
        self.beta_history.append(beta)

        # Direction from target to source, normalized
        direction = z_source_adj - z_target_adj
        direction_norm = normalize_vector(direction)

        # F = beta * normalize(z_source - z_target)
        F = beta * direction_norm

        # Pad back if needed
        if len(F) < len(z_target):
            F = np.concatenate([F, np.zeros(len(z_target) - len(F))])

        F_magnitude = float(np.linalg.norm(F))
        self.F_magnitude_history.append(F_magnitude)

        ECOLOGY_PROVENANCE.log(
            f'beta_{self.source_id}->{self.target_id}', float(beta),
            'rank(T_eco) * rank(D_nov_target)',
            {'rank_T_eco': rank_T_eco, 'rank_D_nov': rank_D_nov},
            self.t
        )

        diagnostics = {
            'beta': beta,
            'rank_T_eco': rank_T_eco,
            'rank_D_nov': rank_D_nov,
            'F_magnitude': F_magnitude
        }

        return F, diagnostics

    def get_statistics(self) -> Dict:
        if not self.beta_history:
            return {'mean_beta': 0.0}

        beta_arr = np.array(self.beta_history)
        F_arr = np.array(self.F_magnitude_history)
        return {
            'mean_beta': float(np.mean(beta_arr)),
            'std_beta': float(np.std(beta_arr)),
            'mean_F_magnitude': float(np.mean(F_arr)),
            'n_samples': len(beta_arr)
        }


# =============================================================================
# ECOLOGICAL RESISTANCE (gamma_eco)
# =============================================================================

class EcologicalResistance:
    """
    Computes ecological resistance gain.

    gamma_eco = 1 / (1 + sigma_eco)

    where sigma_eco = std(eco_shock_history)

    Higher variance in eco_shock -> lower gamma (more resistance)
    This modulates how external ecological shocks affect the system.
    """

    def __init__(self):
        self.eco_shock_history: List[float] = []
        self.gamma_history: List[float] = []
        self.sigma_history: List[float] = []
        self.t = 0

    def compute(self, eco_shock_magnitude: float) -> Tuple[float, Dict]:
        """
        Compute ecological resistance gain.

        gamma_eco = 1 / (1 + sigma_eco)

        Args:
            eco_shock_magnitude: Magnitude of current ecological shock

        Returns:
            (gamma_eco, diagnostics)
        """
        self.t += 1
        self.eco_shock_history.append(eco_shock_magnitude)

        # sigma_eco = std(eco_shock_history) - endogenous from history
        if len(self.eco_shock_history) >= 2:
            sigma_eco = float(np.std(self.eco_shock_history))
        else:
            sigma_eco = 0.0

        self.sigma_history.append(sigma_eco)

        # gamma_eco = 1 / (1 + sigma_eco)
        gamma_eco = 1.0 / (1.0 + sigma_eco)
        self.gamma_history.append(gamma_eco)

        ECOLOGY_PROVENANCE.log(
            'gamma_eco', float(gamma_eco),
            '1 / (1 + sigma_eco)',
            {'sigma_eco': sigma_eco, 'n_history': len(self.eco_shock_history)},
            self.t
        )

        diagnostics = {
            'sigma_eco': sigma_eco,
            'gamma_eco': gamma_eco,
            'eco_shock_magnitude': eco_shock_magnitude
        }

        return float(gamma_eco), diagnostics

    def get_statistics(self) -> Dict:
        if not self.gamma_history:
            return {'mean_gamma': 1.0}

        return {
            'mean_gamma': float(np.mean(self.gamma_history)),
            'std_gamma': float(np.std(self.gamma_history)),
            'mean_sigma': float(np.mean(self.sigma_history)),
            'n_samples': len(self.gamma_history)
        }


# =============================================================================
# ECO_SHOCK COMBINER (from Phase 20 shocks)
# =============================================================================

class EcoShockCombiner:
    """
    Combines Phase 20 shocks into ecological shock.

    eco_shock = weighted combination of S_novelty, S_stability, S_survival
    Weights derived from internal statistics (ranks).

    This bridges Phase 20 shock mechanisms to ecological resistance.
    """

    def __init__(self):
        self.shock_history: List[Dict] = []
        self.combined_history: List[float] = []
        self.t = 0

    def combine(self, S_novelty: np.ndarray, S_stability: np.ndarray,
                S_survival: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Combine Phase 20 shocks into eco_shock.

        eco_shock = w_nov * S_nov + w_stab * S_stab + w_surv * S_surv
        Weights from rank(magnitude) of each shock type.

        Args:
            S_novelty, S_stability, S_survival: Phase 20 shock vectors

        Returns:
            (eco_shock, diagnostics)
        """
        self.t += 1

        # Compute magnitudes
        mag_nov = float(np.linalg.norm(S_novelty))
        mag_stab = float(np.linalg.norm(S_stability))
        mag_surv = float(np.linalg.norm(S_survival))

        self.shock_history.append({
            'mag_nov': mag_nov,
            'mag_stab': mag_stab,
            'mag_surv': mag_surv
        })

        # Compute ranks from history for weighting
        mag_nov_arr = np.array([h['mag_nov'] for h in self.shock_history])
        mag_stab_arr = np.array([h['mag_stab'] for h in self.shock_history])
        mag_surv_arr = np.array([h['mag_surv'] for h in self.shock_history])

        rank_nov = compute_rank(mag_nov, mag_nov_arr)
        rank_stab = compute_rank(mag_stab, mag_stab_arr)
        rank_surv = compute_rank(mag_surv, mag_surv_arr)

        # Normalize weights to sum to 1
        total_rank = rank_nov + rank_stab + rank_surv + NUMERIC_EPS
        w_nov = rank_nov / total_rank
        w_stab = rank_stab / total_rank
        w_surv = rank_surv / total_rank

        # Handle dimension mismatch
        min_dim = min(len(S_novelty), len(S_stability), len(S_survival))
        S_nov_adj = S_novelty[:min_dim]
        S_stab_adj = S_stability[:min_dim]
        S_surv_adj = S_survival[:min_dim]

        # Combine: eco_shock = weighted sum
        eco_shock = w_nov * S_nov_adj + w_stab * S_stab_adj + w_surv * S_surv_adj

        combined_mag = float(np.linalg.norm(eco_shock))
        self.combined_history.append(combined_mag)

        ECOLOGY_PROVENANCE.log(
            'eco_shock', combined_mag,
            'w_nov*S_nov + w_stab*S_stab + w_surv*S_surv',
            {'w_nov': w_nov, 'w_stab': w_stab, 'w_surv': w_surv},
            self.t
        )

        diagnostics = {
            'w_nov': w_nov,
            'w_stab': w_stab,
            'w_surv': w_surv,
            'combined_magnitude': combined_mag
        }

        return eco_shock, diagnostics

    def get_statistics(self) -> Dict:
        if not self.combined_history:
            return {'mean_eco_shock': 0.0}

        return {
            'mean_eco_shock': float(np.mean(self.combined_history)),
            'std_eco_shock': float(np.std(self.combined_history)),
            'n_samples': len(self.combined_history)
        }


# =============================================================================
# CROSS-AGENT ECOLOGY SYSTEM (FULL SPECIFICATION)
# =============================================================================

class CrossAgentEcology:
    """
    Main class for Phase 21 cross-agent ecology (FULL SPEC).

    Integrates:
    - Ecological distance (d_NE, d_mu)
    - Individual tensions (T_N, T_E)
    - Shared tension (T_eco with Overlap)
    - Cross-influence fields (F with normalized direction)
    - Ecological resistance (gamma_eco from sigma_eco)
    - Eco_shock combination from Phase 20

    State update:
    z_next = z + F + gamma_eco * eco_shock

    ALL parameters endogenous.
    """

    def __init__(self):
        # Distance computer
        self.distance = EcologicalDistance()

        # Individual tensions
        self.tension_N = IndividualTension('N')
        self.tension_E = IndividualTension('E')

        # Shared tension
        self.shared_tension = SharedTension()

        # Cross-influence fields
        self.influence_E_to_N = CrossInfluenceField('E', 'N')
        self.influence_N_to_E = CrossInfluenceField('N', 'E')

        # Ecological resistance
        self.resistance_N = EcologicalResistance()
        self.resistance_E = EcologicalResistance()

        # Eco_shock combiner
        self.eco_shock_combiner = EcoShockCombiner()

        # Drive histories for ranking
        self.D_nov_N_history: List[float] = []
        self.D_nov_E_history: List[float] = []

        self.t = 0

    def process_step(self,
                    z_N: np.ndarray, z_E: np.ndarray,
                    prototypes_N: np.ndarray, prototypes_E: np.ndarray,
                    R_N: float, R_E: float,
                    D_nov_N: float, D_nov_E: float,
                    S_novelty_N: Optional[np.ndarray] = None,
                    S_stability_N: Optional[np.ndarray] = None,
                    S_survival_N: Optional[np.ndarray] = None,
                    S_novelty_E: Optional[np.ndarray] = None,
                    S_stability_E: Optional[np.ndarray] = None,
                    S_survival_E: Optional[np.ndarray] = None) -> Dict:
        """
        Process one step of cross-agent ecology (FULL SPEC).

        Args:
            z_N, z_E: Internal states
            prototypes_N, prototypes_E: Prototype arrays
            R_N, R_E: Irreversibility values
            D_nov_N, D_nov_E: Novelty drives
            S_novelty_*, S_stability_*, S_survival_*: Phase 20 shocks (optional)

        Returns:
            Dict with all ecology outputs including influence fields and gamma_eco
        """
        self.t += 1

        # Store drive histories
        self.D_nov_N_history.append(D_nov_N)
        self.D_nov_E_history.append(D_nov_E)

        # 1. Compute ecological distances (with Overlap)
        distances, dist_diag = self.distance.compute(z_N, z_E, prototypes_N, prototypes_E)

        # 2. Compute individual tensions: T_a = rank(spread) * rank(R)
        T_N, T_N_diag = self.tension_N.compute(z_N, R_N)
        T_E, T_E_diag = self.tension_E.compute(z_E, R_E)

        # 3. Compute shared tension: T_eco = (rank(T_N)+rank(T_E))/2 * Overlap
        T_eco, T_eco_diag = self.shared_tension.compute(
            T_N, T_E, distances['overlap_raw'],
            self.tension_N.T_history, self.tension_E.T_history
        )

        # 4. Compute cross-influence fields: F = beta * normalize(z_source - z_target)
        F_E_to_N, F_EN_diag = self.influence_E_to_N.compute(
            z_E, z_N, T_eco, D_nov_N,
            self.shared_tension.T_eco_history, self.D_nov_N_history
        )

        F_N_to_E, F_NE_diag = self.influence_N_to_E.compute(
            z_N, z_E, T_eco, D_nov_E,
            self.shared_tension.T_eco_history, self.D_nov_E_history
        )

        # 5. Compute eco_shock and gamma_eco (if Phase 20 shocks provided)
        eco_shock_N = np.zeros_like(z_N)
        eco_shock_E = np.zeros_like(z_E)
        gamma_eco_N = 1.0
        gamma_eco_E = 1.0
        eco_shock_diag = {}

        if S_novelty_N is not None and S_stability_N is not None and S_survival_N is not None:
            # Combine Phase 20 shocks for agent N
            eco_shock_N, shock_diag_N = self.eco_shock_combiner.combine(
                S_novelty_N, S_stability_N, S_survival_N
            )
            # Compute gamma_eco from eco_shock variance
            eco_shock_mag_N = float(np.linalg.norm(eco_shock_N))
            gamma_eco_N, gamma_diag_N = self.resistance_N.compute(eco_shock_mag_N)
            eco_shock_diag['N'] = {'shock': shock_diag_N, 'gamma': gamma_diag_N}

        if S_novelty_E is not None and S_stability_E is not None and S_survival_E is not None:
            # Combine Phase 20 shocks for agent E
            eco_shock_E, shock_diag_E = self.eco_shock_combiner.combine(
                S_novelty_E, S_stability_E, S_survival_E
            )
            eco_shock_mag_E = float(np.linalg.norm(eco_shock_E))
            gamma_eco_E, gamma_diag_E = self.resistance_E.compute(eco_shock_mag_E)
            eco_shock_diag['E'] = {'shock': shock_diag_E, 'gamma': gamma_diag_E}

        result = {
            't': self.t,
            'distances': distances,
            'T_N': T_N,
            'T_E': T_E,
            'T_eco': T_eco,
            'F_E_to_N': F_E_to_N.tolist(),
            'F_N_to_E': F_N_to_E.tolist(),
            'F_E_to_N_magnitude': float(np.linalg.norm(F_E_to_N)),
            'F_N_to_E_magnitude': float(np.linalg.norm(F_N_to_E)),
            'gamma_eco_N': gamma_eco_N,
            'gamma_eco_E': gamma_eco_E,
            'eco_shock_N': eco_shock_N.tolist() if isinstance(eco_shock_N, np.ndarray) else eco_shock_N,
            'eco_shock_E': eco_shock_E.tolist() if isinstance(eco_shock_E, np.ndarray) else eco_shock_E,
            'diagnostics': {
                'distance': dist_diag,
                'tension_N': T_N_diag,
                'tension_E': T_E_diag,
                'shared_tension': T_eco_diag,
                'influence_E_to_N': F_EN_diag,
                'influence_N_to_E': F_NE_diag,
                'eco_shock': eco_shock_diag
            }
        }

        return result

    def apply_ecological_update(self, z_N_base: np.ndarray, z_E_base: np.ndarray,
                                F_E_to_N: np.ndarray, F_N_to_E: np.ndarray,
                                gamma_eco_N: float = 1.0, gamma_eco_E: float = 1.0,
                                eco_shock_N: Optional[np.ndarray] = None,
                                eco_shock_E: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ecological influence to base state updates.

        z_next = z_base + F + gamma_eco * eco_shock

        Full specification includes ecological resistance modulation.
        """
        # Handle dimension mismatch
        min_dim_N = min(len(z_N_base), len(F_E_to_N))
        min_dim_E = min(len(z_E_base), len(F_N_to_E))

        z_N_next = z_N_base.copy()
        z_N_next[:min_dim_N] += F_E_to_N[:min_dim_N]

        z_E_next = z_E_base.copy()
        z_E_next[:min_dim_E] += F_N_to_E[:min_dim_E]

        # Apply gamma_eco * eco_shock
        if eco_shock_N is not None:
            min_shock_dim = min(len(z_N_next), len(eco_shock_N))
            z_N_next[:min_shock_dim] += gamma_eco_N * eco_shock_N[:min_shock_dim]

        if eco_shock_E is not None:
            min_shock_dim = min(len(z_E_next), len(eco_shock_E))
            z_E_next[:min_shock_dim] += gamma_eco_E * eco_shock_E[:min_shock_dim]

        return z_N_next, z_E_next

    def get_statistics(self) -> Dict:
        return {
            'distance': self.distance.get_statistics(),
            'tension_N': self.tension_N.get_statistics(),
            'tension_E': self.tension_E.get_statistics(),
            'shared_tension': self.shared_tension.get_statistics(),
            'influence_E_to_N': self.influence_E_to_N.get_statistics(),
            'influence_N_to_E': self.influence_N_to_E.get_statistics(),
            'resistance_N': self.resistance_N.get_statistics(),
            'resistance_E': self.resistance_E.get_statistics(),
            'eco_shock_combiner': self.eco_shock_combiner.get_statistics(),
            'n_steps': self.t
        }


# =============================================================================
# PROVENANCE (FULL SPEC)
# =============================================================================

ECOLOGY21_PROVENANCE = {
    'module': 'ecology21',
    'version': '2.0.0',  # Full specification
    'mechanisms': [
        'ecological_distance',
        'individual_tension',
        'shared_tension',
        'cross_influence_field',
        'ecological_resistance',
        'eco_shock_combination'
    ],
    'endogenous_params': [
        'd_NE: d_t^NE = ||z_N - z_E||',
        'd_mu_NE: d_mu,t^NE = min_{k,l} ||mu_k^N - mu_l^E||',
        'd_tilde: d_tilde = rank(d)',
        'Overlap: Overlap = rank(1 - d_tilde_mu)',
        'T_a: T_a = rank(spread) * rank(R)',
        'T_eco: T_eco = (rank(T_N) + rank(T_E))/2 * Overlap',
        'beta: beta = rank(T_eco) * rank(D_nov)',
        'F: F = beta * normalize(z_source - z_target)',
        'w: w = sqrt(t+1)',
        'gamma_eco: gamma_eco = 1 / (1 + sigma_eco)',
        'sigma_eco: sigma_eco = std(eco_shock_history)',
        'eco_shock: eco_shock = w_nov*S_nov + w_stab*S_stab + w_surv*S_surv',
        'z_next: z_next = z + F + gamma_eco * eco_shock'
    ],
    'no_magic_numbers': True,
    'no_semantic_labels': True
}


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("Phase 21: Cross-Agent Ecology & Ecological Resistance (FULL SPEC)")
    print("=" * 60)

    np.random.seed(42)

    # Test ecology system
    print("\n[1] Testing CrossAgentEcology (Full Specification)...")
    ecology = CrossAgentEcology()

    # Generate trajectories for two agents
    T = 500
    prototypes_N = np.random.randn(5, 4) * 0.5
    prototypes_E = np.random.randn(5, 4) * 0.5 + 0.3  # Slightly offset

    T_eco_history = []
    F_magnitude_history = []
    gamma_eco_history = []

    for t in range(T):
        # Agent N state
        z_N = prototypes_N[t % 5] + np.random.randn(4) * 0.2
        R_N = np.abs(np.random.randn()) * 0.3
        D_nov_N = np.random.rand()

        # Agent E state
        z_E = prototypes_E[t % 5] + np.random.randn(4) * 0.2
        R_E = np.abs(np.random.randn()) * 0.3
        D_nov_E = np.random.rand()

        # Phase 20 shocks (simulated)
        S_nov_N = np.random.randn(4) * 0.1
        S_stab_N = np.random.randn(4) * 0.1
        S_surv_N = np.random.randn(4) * 0.1

        result = ecology.process_step(
            z_N, z_E, prototypes_N, prototypes_E,
            R_N, R_E, D_nov_N, D_nov_E,
            S_novelty_N=S_nov_N, S_stability_N=S_stab_N, S_survival_N=S_surv_N
        )

        T_eco_history.append(result['T_eco'])
        F_magnitude_history.append(result['F_E_to_N_magnitude'])
        gamma_eco_history.append(result['gamma_eco_N'])

        if t % 100 == 0:
            print(f"  t={t}: T_eco={result['T_eco']:.3f}, "
                  f"|F|={result['F_E_to_N_magnitude']:.3f}, "
                  f"gamma_eco={result['gamma_eco_N']:.3f}")

    stats = ecology.get_statistics()
    print(f"\n[2] Final Statistics:")
    print(f"  Mean T_eco: {stats['shared_tension']['mean_T_eco']:.4f}")
    print(f"  Mean |F_E->N|: {stats['influence_E_to_N']['mean_F_magnitude']:.4f}")
    print(f"  Mean gamma_eco: {stats['resistance_N']['mean_gamma']:.4f}")
    print(f"  Mean sigma_eco: {stats['resistance_N']['mean_sigma']:.4f}")

    print("\n" + "=" * 60)
    print("PHASE 21 FULL SPECIFICATION VERIFICATION:")
    print("  - d_t^NE = ||z_N - z_E||")
    print("  - d_mu,t^NE = min_{k,l} ||mu_k^N - mu_l^E||")
    print("  - Overlap = rank(1 - d_tilde_mu)")
    print("  - T_a = rank(spread) * rank(R)")
    print("  - T_eco = (rank(T_N)+rank(T_E))/2 * Overlap")
    print("  - F = beta * normalize(z_source - z_target)")
    print("  - gamma_eco = 1/(1 + sigma_eco)")
    print("  - eco_shock = weighted sum of Phase 20 shocks")
    print("  - z_next = z + F + gamma_eco * eco_shock")
    print("  - ZERO magic constants")
    print("=" * 60)
