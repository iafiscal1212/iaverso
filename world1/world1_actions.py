"""
WORLD-1 Action System

Maps agent internal actions to world perturbations.
All mappings learned endogenously from covariance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class ActionMapper:
    """
    Maps agent actions to world perturbations.

    delta_w^A = M_t^A @ a_t^A

    Where M_t^A is learned from covariance between
    agent states and world states.
    """

    def __init__(self, world_dim: int, agent_dim: int = 6):
        """
        Initialize action mapper.

        Args:
            world_dim: Dimension of world state
            agent_dim: Dimension of agent internal state
        """
        self.world_dim = world_dim
        self.agent_dim = agent_dim

        # Per-agent mapping matrices
        self.mappings: Dict[str, np.ndarray] = {}

        # Per-agent influence weights (how much they affect the world)
        self.influence_weights: Dict[str, float] = {}

        # History for learning mappings
        self.world_history: List[np.ndarray] = []
        self.agent_state_histories: Dict[str, List[np.ndarray]] = {}
        self.agent_action_histories: Dict[str, List[np.ndarray]] = {}

        self.t = 0

    def register_agent(self, agent_name: str, initial_z: np.ndarray):
        """Register an agent for action mapping."""
        # Initial mapping: random orthogonal
        M = np.random.randn(self.world_dim, self.agent_dim) * 0.1
        self.mappings[agent_name] = M

        self.influence_weights[agent_name] = 1.0

        self.agent_state_histories[agent_name] = [initial_z.copy()]
        self.agent_action_histories[agent_name] = []

    def record_world_state(self, w: np.ndarray):
        """Record world state."""
        self.world_history.append(w.copy())
        self.t += 1

        max_hist = 500
        if len(self.world_history) > max_hist:
            self.world_history = self.world_history[-max_hist:]

    def record_agent_state(self, agent_name: str, z: np.ndarray):
        """Record agent state."""
        if agent_name not in self.agent_state_histories:
            self.register_agent(agent_name, z)
        else:
            self.agent_state_histories[agent_name].append(z.copy())

            max_hist = 500
            if len(self.agent_state_histories[agent_name]) > max_hist:
                self.agent_state_histories[agent_name] = \
                    self.agent_state_histories[agent_name][-max_hist:]

    def record_action(self, agent_name: str, action: np.ndarray):
        """Record agent action."""
        if agent_name not in self.agent_action_histories:
            self.agent_action_histories[agent_name] = []

        self.agent_action_histories[agent_name].append(action.copy())

        max_hist = 500
        if len(self.agent_action_histories[agent_name]) > max_hist:
            self.agent_action_histories[agent_name] = \
                self.agent_action_histories[agent_name][-max_hist:]

    def _compute_window_size(self) -> int:
        """Endogenous window size."""
        return max(5, int(np.sqrt(self.t + 1)))

    def update_mapping(self, agent_name: str):
        """
        Update action mapping endogenously.

        Learn M_t^A from: argmin_M MSE(w_t - M @ z_t^A)
        Using gradient descent with endogenous learning rate.
        """
        if agent_name not in self.agent_state_histories:
            return

        W = self._compute_window_size()

        if len(self.world_history) < W:
            return
        if len(self.agent_state_histories[agent_name]) < W:
            return

        # Get aligned histories
        world_recent = np.array(self.world_history[-W:])
        agent_recent = np.array(self.agent_state_histories[agent_name][-W:])

        # Align lengths
        min_len = min(len(world_recent), len(agent_recent))
        if min_len < 2:
            return

        world_recent = world_recent[-min_len:]
        agent_recent = agent_recent[-min_len:]

        # Compute optimal mapping via least squares
        # w = M @ z  =>  M = w @ pinv(z)
        try:
            M_new = world_recent.T @ np.linalg.pinv(agent_recent.T)
        except:
            return

        # Smooth update with endogenous rate
        eta = 1.0 / np.sqrt(self.t + 1)

        if agent_name in self.mappings:
            self.mappings[agent_name] = (
                (1 - eta) * self.mappings[agent_name] +
                eta * M_new
            )
        else:
            self.mappings[agent_name] = M_new

        # Update influence weight based on mapping stability
        self._update_influence_weight(agent_name)

    def _update_influence_weight(self, agent_name: str):
        """
        Update influence weight endogenously.

        gamma_t^A = 1 / (1 + std(delta_w_history^A))
        Less variable action effects = more influence
        """
        if agent_name not in self.agent_action_histories:
            self.influence_weights[agent_name] = 1.0
            return

        action_hist = self.agent_action_histories[agent_name]
        if len(action_hist) < 5:
            self.influence_weights[agent_name] = 1.0
            return

        # Compute effect variability
        effects = []
        for action in action_hist[-20:]:
            effect = self.mappings[agent_name] @ action
            effects.append(effect)

        effects_array = np.array(effects)
        effect_std = np.std(effects_array)

        # Influence weight: lower variance = higher influence
        self.influence_weights[agent_name] = 1.0 / (1.0 + effect_std)

    def get_world_perturbation(self, agent_name: str, action: np.ndarray) -> np.ndarray:
        """
        Get world perturbation from agent action.

        Args:
            agent_name: Name of agent
            action: Agent's action vector (typically based on z or delta_z)

        Returns:
            World perturbation vector
        """
        if agent_name not in self.mappings:
            return np.zeros(self.world_dim)

        M = self.mappings[agent_name]
        gamma = self.influence_weights.get(agent_name, 1.0)

        # Ensure action has right dimension
        if len(action) != self.agent_dim:
            # Pad or truncate
            action_padded = np.zeros(self.agent_dim)
            action_padded[:min(len(action), self.agent_dim)] = \
                action[:min(len(action), self.agent_dim)]
            action = action_padded

        perturbation = gamma * M @ action

        # Record action
        self.record_action(agent_name, action)

        return perturbation

    def compute_action_from_drives(self, agent_name: str, z: np.ndarray,
                                    target_change: str = 'exploration') -> np.ndarray:
        """
        Compute action from agent drives and goal.

        Different strategies based on dominant drives:
        - 'exploration': amplify novelty-seeking directions
        - 'stability': dampen high-variance directions
        - 'integration': balance all directions
        """
        # Action is a function of current drives
        action = z.copy()

        if target_change == 'exploration':
            # Amplify entropy and novelty drives
            action[0] *= 1.5  # entropy
            action[2] *= 1.5  # novelty

        elif target_change == 'stability':
            # Amplify stability, dampen exploration
            action[3] *= 1.5  # stability
            action[0] *= 0.5  # entropy
            action[2] *= 0.5  # novelty

        elif target_change == 'integration':
            # Balance all drives
            action = action / (np.linalg.norm(action) + 1e-8)

        elif target_change == 'connection':
            # Amplify otherness
            action[5] *= 2.0  # otherness

        # Normalize
        action = action / (np.sum(np.abs(action)) + 1e-8)

        return action

    def get_statistics(self, agent_name: str) -> Dict:
        """Get action mapping statistics for an agent."""
        if agent_name not in self.mappings:
            return {'status': 'not_registered'}

        M = self.mappings[agent_name]

        return {
            'agent': agent_name,
            'mapping_norm': float(np.linalg.norm(M)),
            'mapping_rank': int(np.linalg.matrix_rank(M)),
            'influence_weight': float(self.influence_weights.get(agent_name, 1.0)),
            'n_actions_recorded': len(self.agent_action_histories.get(agent_name, []))
        }


def test_actions():
    """Test action mapper."""
    print("=" * 60)
    print("ACTION MAPPER TEST")
    print("=" * 60)

    world_dim = 15
    agent_dim = 6

    mapper = ActionMapper(world_dim, agent_dim)

    # Register agents
    agents = {
        'NEO': np.array([0.15, 0.15, 0.15, 0.25, 0.15, 0.15]),
        'EVA': np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.25]),
        'ALEX': np.array([0.20, 0.10, 0.25, 0.15, 0.15, 0.15]),
        'ADAM': np.array([0.15, 0.15, 0.15, 0.15, 0.25, 0.15]),
        'IRIS': np.array([0.18, 0.18, 0.18, 0.18, 0.18, 0.10])
    }

    for name, z in agents.items():
        mapper.register_agent(name, z)

    print(f"\nRegistered {len(agents)} agents")

    # Simulate
    for t in range(200):
        # Simulated world state
        w = np.sin(np.arange(world_dim) * 0.1 + t * 0.05)
        mapper.record_world_state(w)

        # Each agent acts
        total_perturbation = np.zeros(world_dim)
        for name, z in agents.items():
            # Evolve agent slightly
            z_new = z + np.random.randn(agent_dim) * 0.01
            z_new = np.abs(z_new)
            z_new = z_new / z_new.sum()
            agents[name] = z_new

            mapper.record_agent_state(name, z_new)

            # Compute action
            action = mapper.compute_action_from_drives(name, z_new, 'exploration')
            perturbation = mapper.get_world_perturbation(name, action)
            total_perturbation += perturbation

        # Update mappings periodically
        if (t + 1) % 20 == 0:
            for name in agents.keys():
                mapper.update_mapping(name)

        if (t + 1) % 50 == 0:
            print(f"\n  t={t+1}:")
            print(f"    Total perturbation norm: {np.linalg.norm(total_perturbation):.3f}")
            for name in agents.keys():
                stats = mapper.get_statistics(name)
                print(f"    {name}: influence={stats['influence_weight']:.3f}, "
                      f"actions={stats['n_actions_recorded']}")

    return mapper


if __name__ == "__main__":
    test_actions()
