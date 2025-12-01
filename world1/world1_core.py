"""
WORLD-1 Core Dynamics v2

Endogenous world dynamics without magic constants.
w_{t+1} = A_t @ w_t + B_t @ sigma(w_t) + xi_t + F(actions)

Where:
- A_t, B_t derived from covariance statistics
- F(actions) = sensitive field responses with nonlinear kicks

New in v2:
- Sensitive fields that respond strongly to coordinated actions
- Rest regime: low variance when total action is low
- Action amplification zones for clear causal attribution

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy import linalg


@dataclass
class WorldState:
    """Complete state of WORLD-1."""
    # Sub-blocks of the world state
    fields: np.ndarray      # w^(f): continuous fields (temperatures, pressures)
    entities: np.ndarray    # w^(e): entity states
    resources: np.ndarray   # w^(r): resource/energy levels
    modes: np.ndarray       # w^(m): regime encoding (soft, not one-hot)

    @property
    def full_vector(self) -> np.ndarray:
        """Concatenated world state vector."""
        return np.concatenate([self.fields, self.entities, self.resources, self.modes])

    @classmethod
    def from_vector(cls, v: np.ndarray, d_fields: int, d_entities: int,
                   d_resources: int, d_modes: int) -> 'WorldState':
        """Reconstruct from flat vector."""
        idx = 0
        fields = v[idx:idx+d_fields]; idx += d_fields
        entities = v[idx:idx+d_entities]; idx += d_entities
        resources = v[idx:idx+d_resources]; idx += d_resources
        modes = v[idx:idx+d_modes]
        return cls(fields=fields, entities=entities, resources=resources, modes=modes)


class World1Core:
    """
    Core dynamics of WORLD-1.

    A mathematical mini-universe with:
    - Endogenous transition matrices (from covariance)
    - Regime detection (from clustering history)
    - Noise that decreases with experience
    """

    def __init__(self, n_fields: int = 4, n_entities: int = 5,
                 n_resources: int = 3, n_modes: int = 3):
        """
        Initialize WORLD-1.

        Dimensions derived from parameters, not hardcoded.
        """
        self.n_fields = n_fields
        self.n_entities = n_entities
        self.n_resources = n_resources
        self.n_modes = n_modes

        # Total dimension
        self.D = n_fields + n_entities + n_resources + n_modes

        # Dimension indices
        self.idx_fields = (0, n_fields)
        self.idx_entities = (n_fields, n_fields + n_entities)
        self.idx_resources = (n_fields + n_entities, n_fields + n_entities + n_resources)
        self.idx_modes = (n_fields + n_entities + n_resources, self.D)

        # History
        self.history: List[np.ndarray] = []
        self.t = 0

        # Current state
        self.w = self._initialize_state()
        self.history.append(self.w.copy())

        # Matrices (will be computed endogenously)
        self.A: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None

        # Effective dimension (endogenous)
        self.d_eff: int = self.D

        # Regime information
        self.current_regime: int = 0
        self.regime_history: List[int] = [0]

        # === NEW v2: Sensitive fields and action tracking ===
        # Sensitive field indices (first half of fields are "sensitive")
        self.n_sensitive = max(1, n_fields // 2)

        # Action history for endogenous thresholds
        self.action_history: List[float] = []  # ||total_action||
        self.action_effects: List[float] = []  # ||ΔW|| when action applied

        # Coordination tracking
        self.coordination_history: List[Dict] = []  # {agents: set, magnitude: float}

        # Rest regime tracking
        self.rest_regime_active: bool = False
        self.rest_delta_norms: List[float] = []  # ||ΔW|| during rest

    def _initialize_state(self) -> np.ndarray:
        """Initialize world state endogenously."""
        # Start near center with small random perturbation
        w = np.zeros(self.D)

        # Fields: slightly positive (like temperatures)
        w[self.idx_fields[0]:self.idx_fields[1]] = 0.5 + np.random.randn(self.n_fields) * 0.1

        # Entities: random positions in unit cube
        w[self.idx_entities[0]:self.idx_entities[1]] = np.random.rand(self.n_entities)

        # Resources: start at half capacity
        w[self.idx_resources[0]:self.idx_resources[1]] = 0.5 + np.random.randn(self.n_resources) * 0.05

        # Modes: uniform (no dominant mode initially)
        w[self.idx_modes[0]:self.idx_modes[1]] = 1.0 / self.n_modes

        return w

    def _compute_window_size(self) -> int:
        """Endogenous window size: floor(sqrt(t+1))."""
        return max(2, int(np.floor(np.sqrt(self.t + 1))))

    def _compute_covariance(self) -> np.ndarray:
        """Compute empirical covariance from recent history."""
        W = self._compute_window_size()
        if len(self.history) < W:
            # Not enough history, return identity
            return np.eye(self.D)

        recent = np.array(self.history[-W:])
        return np.cov(recent.T) + 1e-8 * np.eye(self.D)

    def _compute_effective_dimension(self, eigenvalues: np.ndarray) -> int:
        """
        Endogenous effective dimension.
        d_eff = #{lambda_i >= median(lambda)}
        """
        median_lambda = np.median(eigenvalues)
        d_eff = np.sum(eigenvalues >= median_lambda)
        return max(1, int(d_eff))

    def _update_transition_matrices(self):
        """
        Update A_t and B_t from covariance statistics.

        A_t = V[:, :d_eff] @ V[:, :d_eff].T
        B_t = weighted by mode vector m_t
        """
        Sigma = self._compute_covariance()

        # Eigen-decomposition
        eigenvalues, V = linalg.eigh(Sigma)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        V = V[:, idx]

        # Effective dimension
        self.d_eff = self._compute_effective_dimension(eigenvalues)

        # A_t: projection onto high-variance subspace
        V_eff = V[:, :self.d_eff]
        self.A = V_eff @ V_eff.T

        # B_t: weighted by current mode
        mode_weights = self.w[self.idx_modes[0]:self.idx_modes[1]]
        mode_weights = np.abs(mode_weights) / (np.sum(np.abs(mode_weights)) + 1e-8)

        # Scale B by mode-weighted eigenvalues
        scaled_eigenvalues = eigenvalues.copy()
        for i in range(min(len(mode_weights), self.d_eff)):
            scaled_eigenvalues[i] *= (1 + mode_weights[i % len(mode_weights)])

        Lambda_scaled = np.diag(np.sqrt(np.abs(scaled_eigenvalues[:self.d_eff]) + 1e-8))
        self.B = V_eff @ Lambda_scaled @ V_eff.T

    def _compute_noise(self) -> np.ndarray:
        """
        Endogenous noise: variance proportional to 1/(t+1).
        More experience = less noise.

        v2: Noise further reduced during rest regime.
        """
        # Base noise scale: endogenous 1/√t
        noise_scale = 1.0 / np.sqrt(self.t + 1)

        # Endogenous noise magnitude from history variance
        if len(self.history) > 10:
            recent = np.array(self.history[-10:])
            hist_std = np.std(recent)
            # Scale noise relative to historical variability
            noise_magnitude = hist_std / np.sqrt(self.t + 1)
        else:
            noise_magnitude = noise_scale

        # Rest regime: further reduce noise
        if self.rest_regime_active:
            # Endogenous reduction: Q25% of normal noise
            if self.action_effects:
                noise_magnitude *= np.percentile(self.action_effects, 25) / (np.median(self.action_effects) + 1e-8)
            else:
                noise_magnitude *= 0.5  # Bootstrap only

        return np.random.randn(self.D) * noise_magnitude

    def _nonlinearity(self, w: np.ndarray) -> np.ndarray:
        """Smooth nonlinearity (tanh is structurally justified)."""
        return np.tanh(w)

    def _compute_sensitive_field_response(
        self,
        total_action: np.ndarray,
        agent_actions: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute sensitive field responses to actions.

        Sensitive fields respond nonlinearly to:
        1. High total action magnitude
        2. Coordinated actions (multiple agents acting similarly)

        Response = kick * sigmoid(action_strength / threshold)

        All thresholds derived endogenously from action history.
        """
        response = np.zeros(self.D)
        action_norm = np.linalg.norm(total_action)

        if action_norm < 1e-10:
            return response

        # Endogenous threshold: median of action history
        if len(self.action_history) > 5:
            action_threshold = np.median(self.action_history) + 1e-8
        else:
            action_threshold = 1.0  # Bootstrap

        # === Sensitive field kick ===
        # Sigmoid activation: how strong is current action relative to history?
        activation = action_norm / (action_norm + action_threshold)

        # Kick direction: aligned with action projection onto sensitive fields
        sensitive_action = total_action[self.idx_fields[0]:self.idx_fields[0] + self.n_sensitive]

        # Kick magnitude: endogenous from Q75% of effects
        if len(self.action_effects) > 5:
            kick_scale = np.percentile(self.action_effects, 75)
        else:
            kick_scale = 1.0

        # Apply nonlinear kick to sensitive fields
        kick = activation * kick_scale * np.sign(sensitive_action) * np.abs(sensitive_action) ** 0.5
        response[self.idx_fields[0]:self.idx_fields[0] + self.n_sensitive] = kick

        # === Coordination bonus ===
        if len(agent_actions) >= 2:
            # Check if agents are acting in similar directions
            actions_list = list(agent_actions.values())
            if len(actions_list) >= 2:
                # Compute pairwise alignment
                alignments = []
                for i in range(len(actions_list)):
                    for j in range(i + 1, len(actions_list)):
                        a1, a2 = actions_list[i], actions_list[j]
                        norm1, norm2 = np.linalg.norm(a1), np.linalg.norm(a2)
                        if norm1 > 1e-8 and norm2 > 1e-8:
                            alignment = np.dot(a1, a2) / (norm1 * norm2)
                            alignments.append(alignment)

                if alignments:
                    mean_alignment = np.mean(alignments)
                    # Coordination bonus: extra kick when agents align
                    if mean_alignment > 0.5:  # Structural threshold: positive alignment
                        coord_bonus = mean_alignment * kick_scale
                        # Apply to resources (coordination affects shared resources)
                        response[self.idx_resources[0]:self.idx_resources[1]] += coord_bonus * np.sign(
                            total_action[self.idx_resources[0]:self.idx_resources[1]]
                        )

                        # Record coordination event
                        self.coordination_history.append({
                            'agents': set(agent_actions.keys()),
                            'alignment': mean_alignment,
                            'magnitude': action_norm
                        })

        return response

    def _detect_rest_regime(self, action_norm: float) -> bool:
        """
        Detect if we're in rest regime (low action).

        Rest regime: action_norm ≤ Q25%(action_history)
        """
        if len(self.action_history) < 5:
            return action_norm < 1e-6

        threshold = np.percentile(self.action_history, 25)
        return action_norm <= threshold

    def _apply_rest_contraction(self) -> np.ndarray:
        """
        Apply contractive dynamics during rest regime.

        During rest, world contracts toward a stable attractor:
        - Fields decay toward their historical mean
        - Resources stabilize
        - Variance decreases

        This makes Stab_{A=0} in CI much clearer.
        """
        contraction = np.zeros(self.D)

        if len(self.history) < 10:
            return contraction

        # Historical mean as attractor
        recent = np.array(self.history[-20:])
        attractor = np.mean(recent, axis=0)

        # Contraction rate: endogenous from 1/√t
        rate = 1.0 / np.sqrt(self.t + 1)

        # Contract toward attractor
        contraction = rate * (attractor - self.w)

        # Stronger contraction for fields (they should stabilize more)
        contraction[self.idx_fields[0]:self.idx_fields[1]] *= 2.0

        return contraction

    def step(self, agent_perturbations: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Advance WORLD-1 by one timestep.

        v2 dynamics:
        w_{t+1} = A_t @ w_t + B_t @ sigma(w_t) + xi_t + F(actions) + [rest_contraction]

        Where F(actions) includes:
        - Sensitive field responses (nonlinear kicks)
        - Coordination bonuses

        Returns:
            New world state vector.
        """
        self.t += 1
        prev_w = self.w.copy()

        # Update transition matrices from history
        if self.t > 2:
            self._update_transition_matrices()
        else:
            # Bootstrap: simple dynamics
            self.A = 0.95 * np.eye(self.D)
            self.B = 0.05 * np.eye(self.D)

        # === Compute total action and detect rest regime ===
        total_perturbation = np.zeros(self.D)
        agent_actions = {}
        if agent_perturbations:
            for agent_name, delta_w in agent_perturbations.items():
                if len(delta_w) == self.D:
                    total_perturbation += delta_w
                    agent_actions[agent_name] = delta_w.copy()

        action_norm = np.linalg.norm(total_perturbation)
        self.action_history.append(action_norm)

        # Limit action history
        max_hist = int(50 + 5 * np.sqrt(self.t + 1))  # Endogenous
        if len(self.action_history) > max_hist:
            self.action_history = self.action_history[-max_hist:]

        # Detect rest regime
        self.rest_regime_active = self._detect_rest_regime(action_norm)

        # === Core dynamics ===
        linear_term = self.A @ self.w
        nonlinear_term = self.B @ self._nonlinearity(self.w)
        noise = self._compute_noise()

        # === v2: Sensitive field response ===
        sensitive_response = self._compute_sensitive_field_response(total_perturbation, agent_actions)

        # === v2: Rest contraction ===
        rest_contraction = np.zeros(self.D)
        if self.rest_regime_active:
            rest_contraction = self._apply_rest_contraction()

        # Update
        self.w = linear_term + nonlinear_term + noise + total_perturbation + sensitive_response + rest_contraction

        # Soft constraints
        self._apply_soft_constraints()

        # Update mode encoding based on recent dynamics
        self._update_modes()

        # === Track action effects for endogenous thresholds ===
        delta_w = np.linalg.norm(self.w - prev_w)
        self.action_effects.append(delta_w)

        if len(self.action_effects) > max_hist:
            self.action_effects = self.action_effects[-max_hist:]

        # Track rest regime stability
        if self.rest_regime_active:
            self.rest_delta_norms.append(delta_w)
            if len(self.rest_delta_norms) > max_hist:
                self.rest_delta_norms = self.rest_delta_norms[-max_hist:]

        # Record history
        self.history.append(self.w.copy())

        # Limit history
        if len(self.history) > max_hist:
            self.history = self.history[-max_hist:]

        return self.w.copy()

    def _apply_soft_constraints(self):
        """Apply soft constraints to keep world bounded but not rigid."""
        # Fields: can be negative or positive, but bounded
        fields = self.w[self.idx_fields[0]:self.idx_fields[1]]
        self.w[self.idx_fields[0]:self.idx_fields[1]] = np.tanh(fields)

        # Entities: in [0, 1]
        entities = self.w[self.idx_entities[0]:self.idx_entities[1]]
        self.w[self.idx_entities[0]:self.idx_entities[1]] = np.clip(entities, 0, 1)

        # Resources: non-negative, soft cap at 1
        resources = self.w[self.idx_resources[0]:self.idx_resources[1]]
        self.w[self.idx_resources[0]:self.idx_resources[1]] = np.clip(resources, 0, 2)

        # Modes: sum to 1 (probability distribution)
        modes = self.w[self.idx_modes[0]:self.idx_modes[1]]
        modes = np.abs(modes) + 1e-8
        self.w[self.idx_modes[0]:self.idx_modes[1]] = modes / modes.sum()

    def _update_modes(self):
        """
        Update mode encoding based on recent dynamics.
        Modes emerge from clustering world history features.
        """
        if self.t < 10:
            return

        W = self._compute_window_size()
        if len(self.history) < W:
            return

        recent = np.array(self.history[-W:])

        # Features for mode detection
        variance = np.var(recent, axis=0).mean()
        mean_change = np.mean(np.abs(np.diff(recent, axis=0)))

        # Update mode weights based on features
        modes = self.w[self.idx_modes[0]:self.idx_modes[1]]

        # Mode 0: stable (low variance)
        # Mode 1: volatile (high variance)
        # Mode 2: transitional (high change rate)

        variance_percentile = self._compute_percentile(variance, 'variance')
        change_percentile = self._compute_percentile(mean_change, 'change')

        new_modes = np.zeros(self.n_modes)
        new_modes[0] = 1 - variance_percentile  # stable
        new_modes[1] = variance_percentile  # volatile
        new_modes[2] = change_percentile  # transitional

        # Smooth update
        alpha = 1.0 / np.sqrt(self.t + 1)
        modes = (1 - alpha) * modes + alpha * new_modes
        modes = modes / (modes.sum() + 1e-8)

        self.w[self.idx_modes[0]:self.idx_modes[1]] = modes

        # Record dominant regime
        self.current_regime = int(np.argmax(modes))
        self.regime_history.append(self.current_regime)

    def _compute_percentile(self, value: float, metric_name: str) -> float:
        """Compute percentile of value in history of that metric."""
        if not hasattr(self, '_metric_history'):
            self._metric_history = {}

        if metric_name not in self._metric_history:
            self._metric_history[metric_name] = []

        self._metric_history[metric_name].append(value)

        # Keep bounded history
        max_history = 1000
        if len(self._metric_history[metric_name]) > max_history:
            self._metric_history[metric_name] = self._metric_history[metric_name][-max_history:]

        hist = self._metric_history[metric_name]
        percentile = np.sum(np.array(hist) <= value) / len(hist)
        return percentile

    def get_state(self) -> WorldState:
        """Get current state as WorldState object."""
        return WorldState(
            fields=self.w[self.idx_fields[0]:self.idx_fields[1]].copy(),
            entities=self.w[self.idx_entities[0]:self.idx_entities[1]].copy(),
            resources=self.w[self.idx_resources[0]:self.idx_resources[1]].copy(),
            modes=self.w[self.idx_modes[0]:self.idx_modes[1]].copy()
        )

    def get_statistics(self) -> Dict:
        """Get current world statistics."""
        state = self.get_state()

        # v2 statistics
        mean_action = np.mean(self.action_history) if self.action_history else 0.0
        mean_effect = np.mean(self.action_effects) if self.action_effects else 0.0
        rest_stability = 0.0
        if self.rest_delta_norms and self.action_effects:
            # Rest stability: how much smaller are deltas during rest?
            rest_mean = np.mean(self.rest_delta_norms)
            action_mean = np.mean(self.action_effects)
            rest_stability = 1.0 - rest_mean / (action_mean + 1e-8)
            rest_stability = float(np.clip(rest_stability, 0, 1))

        return {
            't': self.t,
            'D': self.D,
            'd_eff': self.d_eff,
            'current_regime': self.current_regime,
            'fields_mean': float(np.mean(state.fields)),
            'fields_std': float(np.std(state.fields)),
            'entities_mean': float(np.mean(state.entities)),
            'resources_mean': float(np.mean(state.resources)),
            'dominant_mode': int(np.argmax(state.modes)),
            'mode_entropy': float(-np.sum(state.modes * np.log(state.modes + 1e-8))),
            # v2 additions
            'n_sensitive_fields': self.n_sensitive,
            'mean_action_norm': float(mean_action),
            'mean_action_effect': float(mean_effect),
            'rest_regime_active': self.rest_regime_active,
            'rest_stability': rest_stability,
            'n_coordination_events': len(self.coordination_history)
        }


def test_world1_core():
    """Test WORLD-1 core dynamics."""
    print("=" * 60)
    print("WORLD-1 CORE DYNAMICS TEST")
    print("=" * 60)

    world = World1Core(n_fields=4, n_entities=5, n_resources=3, n_modes=3)

    print(f"\nInitialized WORLD-1:")
    print(f"  Total dimension D = {world.D}")
    print(f"  Fields: {world.n_fields}")
    print(f"  Entities: {world.n_entities}")
    print(f"  Resources: {world.n_resources}")
    print(f"  Modes: {world.n_modes}")

    # Run for 500 steps
    stats_history = []
    for t in range(500):
        world.step()
        if (t + 1) % 100 == 0:
            stats = world.get_statistics()
            stats_history.append(stats)
            print(f"\n  t={t+1}:")
            print(f"    d_eff = {stats['d_eff']}")
            print(f"    Regime = {stats['current_regime']}")
            print(f"    Fields mean = {stats['fields_mean']:.3f}")
            print(f"    Mode entropy = {stats['mode_entropy']:.3f}")

    # Check endogenous properties
    print("\n" + "=" * 60)
    print("ENDOGENOUS VERIFICATION")
    print("=" * 60)

    # Variance not degenerate
    history_array = np.array(world.history)
    total_variance = np.var(history_array)
    print(f"\n  Total variance: {total_variance:.4f}")
    print(f"  Status: {'PASS' if 0.001 < total_variance < 10 else 'FAIL'}")

    # Regime changes detected
    regime_changes = np.sum(np.diff(world.regime_history) != 0)
    print(f"\n  Regime changes: {regime_changes}")
    print(f"  Status: {'PASS' if regime_changes > 0 else 'WARN (no changes)'}")

    # d_eff is meaningful
    print(f"\n  Final d_eff: {world.d_eff} / {world.D}")
    print(f"  Status: {'PASS' if 1 < world.d_eff < world.D else 'WARN'}")

    return world


if __name__ == "__main__":
    test_world1_core()
