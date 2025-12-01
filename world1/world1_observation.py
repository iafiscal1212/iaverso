"""
WORLD-1 Observation System

Each agent gets an endogenous projection of the world state.
Projections derived from covariance with agent's history.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import linalg


class ObservationProjector:
    """
    Creates endogenous observation projections for each agent.

    Each agent A sees: o_t^A = P_t^A @ w_t

    Where P_t^A is derived from:
    - Principal components relevant to that agent
    - Weighted by agent's drive dominance
    """

    def __init__(self, world_dim: int, agent_dim: int = 6):
        """
        Initialize observation projector.

        Args:
            world_dim: Dimension of world state
            agent_dim: Dimension of agent internal state (z)
        """
        self.world_dim = world_dim
        self.agent_dim = agent_dim

        # Per-agent projection matrices
        self.projections: Dict[str, np.ndarray] = {}

        # Per-agent observation dimensions
        self.obs_dims: Dict[str, int] = {}

        # History for computing projections
        self.world_history: List[np.ndarray] = []
        self.agent_histories: Dict[str, List[np.ndarray]] = {}

        self.t = 0

    def register_agent(self, agent_name: str, initial_z: np.ndarray):
        """Register an agent for observation."""
        self.agent_histories[agent_name] = [initial_z.copy()]

        # Initial projection: identity-like (observe everything equally)
        obs_dim = min(self.world_dim, self.agent_dim)
        self.obs_dims[agent_name] = obs_dim

        # Random orthogonal projection initially
        random_matrix = np.random.randn(obs_dim, self.world_dim)
        Q, _ = linalg.qr(random_matrix.T)
        self.projections[agent_name] = Q[:, :obs_dim].T

    def record_world_state(self, w: np.ndarray):
        """Record world state for projection computation."""
        self.world_history.append(w.copy())
        self.t += 1

        # Keep bounded
        max_hist = 500
        if len(self.world_history) > max_hist:
            self.world_history = self.world_history[-max_hist:]

    def record_agent_state(self, agent_name: str, z: np.ndarray):
        """Record agent internal state."""
        if agent_name not in self.agent_histories:
            self.register_agent(agent_name, z)
        else:
            self.agent_histories[agent_name].append(z.copy())

            # Keep bounded
            max_hist = 500
            if len(self.agent_histories[agent_name]) > max_hist:
                self.agent_histories[agent_name] = self.agent_histories[agent_name][-max_hist:]

    def _compute_window_size(self) -> int:
        """Endogenous window size."""
        return max(5, int(np.sqrt(self.t + 1)))

    def update_projection(self, agent_name: str):
        """
        Update projection matrix for agent endogenously.

        Uses covariance between world states and agent states
        to find relevant directions.
        """
        if agent_name not in self.agent_histories:
            return

        W = self._compute_window_size()

        if len(self.world_history) < W or len(self.agent_histories[agent_name]) < W:
            return

        # Get recent histories
        world_recent = np.array(self.world_history[-W:])
        agent_recent = np.array(self.agent_histories[agent_name][-W:])

        # Compute cross-covariance
        world_centered = world_recent - world_recent.mean(axis=0)
        agent_centered = agent_recent - agent_recent.mean(axis=0)

        # Cross-covariance: which world directions predict agent states?
        cross_cov = world_centered.T @ agent_centered / W

        # SVD to find relevant directions
        try:
            U, S, Vt = linalg.svd(cross_cov, full_matrices=False)
        except:
            return

        # Effective observation dimension
        if len(S) > 0:
            median_s = np.median(S)
            d_obs = max(1, np.sum(S >= median_s))
        else:
            d_obs = 1

        self.obs_dims[agent_name] = min(d_obs, self.world_dim, self.agent_dim)

        # Projection: top singular vectors of world
        new_projection = U[:, :self.obs_dims[agent_name]].T

        # Smooth update
        alpha = 1.0 / np.sqrt(self.t + 1)

        if agent_name in self.projections:
            old_shape = self.projections[agent_name].shape
            new_shape = new_projection.shape

            if old_shape == new_shape:
                self.projections[agent_name] = (
                    (1 - alpha) * self.projections[agent_name] +
                    alpha * new_projection
                )
            else:
                self.projections[agent_name] = new_projection
        else:
            self.projections[agent_name] = new_projection

    def get_observation(self, agent_name: str, world_state: np.ndarray) -> np.ndarray:
        """
        Get agent's observation of the world.

        Args:
            agent_name: Name of agent
            world_state: Current world state vector

        Returns:
            Projected observation vector
        """
        if agent_name not in self.projections:
            # Return full state if agent not registered
            return world_state.copy()

        P = self.projections[agent_name]
        return P @ world_state

    def get_observation_with_bias(self, agent_name: str, world_state: np.ndarray,
                                   agent_z: np.ndarray) -> np.ndarray:
        """
        Get observation biased by agent's current drives.

        Different agents "see" different aspects based on their drives:
        - High stability drive → focus on predictable components
        - High novelty drive → focus on high-variance components
        - High otherness drive → focus on interaction-relevant components
        """
        base_obs = self.get_observation(agent_name, world_state)

        if len(agent_z) < 6:
            return base_obs

        # Drive indices: [entropy, neg_surprise, novelty, stability, integration, otherness]
        stability_drive = agent_z[3]
        novelty_drive = agent_z[2]
        entropy_drive = agent_z[0]

        # Compute local variance of observation
        obs_variance = np.var(base_obs)

        # Bias based on drives
        bias = np.zeros_like(base_obs)

        # Stability seekers: dampen high-variance components
        if stability_drive > 0.2:
            obs_centered = base_obs - np.mean(base_obs)
            bias -= stability_drive * obs_centered * 0.1

        # Novelty seekers: amplify high-variance components
        if novelty_drive > 0.2:
            obs_centered = base_obs - np.mean(base_obs)
            bias += novelty_drive * obs_centered * 0.1

        return base_obs + bias

    def get_statistics(self, agent_name: str) -> Dict:
        """Get observation statistics for an agent."""
        if agent_name not in self.projections:
            return {'status': 'not_registered'}

        P = self.projections[agent_name]

        return {
            'agent': agent_name,
            'obs_dim': self.obs_dims.get(agent_name, 0),
            'world_dim': self.world_dim,
            'projection_norm': float(np.linalg.norm(P)),
            'projection_rank': int(np.linalg.matrix_rank(P))
        }


def test_observation():
    """Test observation projector."""
    print("=" * 60)
    print("OBSERVATION PROJECTOR TEST")
    print("=" * 60)

    world_dim = 15
    agent_dim = 6

    projector = ObservationProjector(world_dim, agent_dim)

    # Register agents
    agents = {
        'NEO': np.array([0.15, 0.15, 0.15, 0.25, 0.15, 0.15]),  # stability bias
        'EVA': np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.25]),  # otherness bias
        'ALEX': np.array([0.20, 0.10, 0.25, 0.15, 0.15, 0.15]), # novelty bias
        'ADAM': np.array([0.15, 0.15, 0.15, 0.15, 0.25, 0.15]), # integration bias
        'IRIS': np.array([0.18, 0.18, 0.18, 0.18, 0.18, 0.10])  # balanced/connection
    }

    for name, z in agents.items():
        projector.register_agent(name, z)

    print(f"\nRegistered {len(agents)} agents")

    # Simulate world evolution
    for t in range(200):
        # Random world state
        w = np.sin(np.arange(world_dim) * 0.1 + t * 0.05) + np.random.randn(world_dim) * 0.1
        projector.record_world_state(w)

        # Agents evolve slightly
        for name, z in agents.items():
            z_new = z + np.random.randn(agent_dim) * 0.01
            z_new = np.abs(z_new)
            z_new = z_new / z_new.sum()
            agents[name] = z_new
            projector.record_agent_state(name, z_new)

        # Update projections periodically
        if (t + 1) % 20 == 0:
            for name in agents.keys():
                projector.update_projection(name)

        if (t + 1) % 50 == 0:
            print(f"\n  t={t+1}:")
            for name in agents.keys():
                obs = projector.get_observation(name, w)
                stats = projector.get_statistics(name)
                print(f"    {name}: obs_dim={stats['obs_dim']}, obs_norm={np.linalg.norm(obs):.3f}")

    return projector


if __name__ == "__main__":
    test_observation()
