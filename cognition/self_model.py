"""
Self-Model and Theory of Mind

Explicit self-modeling and prediction of other agents.
All learned endogenously from internal data.

"NEO believes NEO is X" = self-state estimate with specific structure
"EVA anticipates ADAM will do Y" = other-state prediction
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SelfBelief:
    """What an agent believes about itself."""
    state_estimate: np.ndarray
    coherence: float  # How consistent with history
    confidence: float  # Based on prediction error
    dominant_type: int  # Cluster index of self-type


class SelfModel:
    """
    Explicit self-model for an agent.

    s_t^A = [z_t^A, φ_t^A, D_t^A]
    ŝ_t^A = f_self^A(H_{t-1}^A)

    Model is linear regression updated with endogenous gradient.
    """

    def __init__(self, agent_name: str, state_dim: int):
        """
        Initialize self-model.

        Args:
            agent_name: Name of the agent
            state_dim: Dimension of combined state [z, φ, D]
        """
        self.agent_name = agent_name
        self.state_dim = state_dim

        # Model weights
        self.W_self = np.eye(state_dim) * 0.1

        # State history
        self.state_history: List[np.ndarray] = []

        # Prediction errors for learning
        self.error_history: List[float] = []

        # Self-belief tracking
        self.belief_history: List[SelfBelief] = []

        # For self-type clustering
        self.state_estimates: List[np.ndarray] = []
        self.self_types: Optional[np.ndarray] = None
        self.n_types: int = 3

        self.t = 0

    def _compute_window_size(self) -> int:
        """Endogenous window size."""
        return max(5, int(np.sqrt(self.t + 1)))

    def record_state(self, s: np.ndarray):
        """Record actual state."""
        self.state_history.append(s.copy())
        self.t += 1

        max_hist = 500
        if len(self.state_history) > max_hist:
            self.state_history = self.state_history[-max_hist:]

    def predict_self(self) -> np.ndarray:
        """
        Predict own next state.

        ŝ_t^A = W_self^A · h̄_{t-1}^A

        where h̄ is mean of recent states.
        """
        if len(self.state_history) < 2:
            return np.zeros(self.state_dim)

        W = self._compute_window_size()
        recent = np.array(self.state_history[-W:])
        h_bar = np.mean(recent, axis=0)

        s_hat = self.W_self @ h_bar
        return s_hat

    def update_model(self, s_actual: np.ndarray):
        """
        Update self-model from prediction error.

        e_t = s_t - ŝ_t
        g_t = ∂||e_t||² / ∂W_self
        η_t = 1/√(t+1)
        ω_t = rank(||e_t||)
        W_new = W - η_t * ω_t * g_t / ||g_t||
        """
        s_hat = self.predict_self()
        e = s_actual - s_hat
        error_norm = np.linalg.norm(e)

        self.error_history.append(error_norm)
        if len(self.error_history) > 500:
            self.error_history = self.error_history[-500:]

        # Compute rank of error
        if len(self.error_history) > 5:
            omega = np.sum(np.array(self.error_history) <= error_norm) / len(self.error_history)
        else:
            omega = 0.5

        # Learning rate
        eta = 1.0 / np.sqrt(self.t + 1)

        # Gradient (simplified: outer product)
        W = self._compute_window_size()
        if len(self.state_history) >= W:
            h_bar = np.mean(np.array(self.state_history[-W:]), axis=0)
            g = np.outer(e, h_bar)
            g_norm = np.linalg.norm(g) + 1e-16

            self.W_self -= eta * omega * g / g_norm

        # Record state and estimate
        self.record_state(s_actual)
        self.state_estimates.append(s_hat)

        if len(self.state_estimates) > 500:
            self.state_estimates = self.state_estimates[-500:]

    def compute_self_coherence(self) -> float:
        """
        Compute self-coherence.

        SC_t = cos(ŝ_t, s̄) where s̄ is historical mean
        """
        if len(self.state_history) < 5:
            return 0.5

        s_hat = self.predict_self()
        s_bar = np.mean(np.array(self.state_history), axis=0)

        norm_hat = np.linalg.norm(s_hat)
        norm_bar = np.linalg.norm(s_bar)

        if norm_hat < 1e-8 or norm_bar < 1e-8:
            return 0.0

        coherence = np.dot(s_hat, s_bar) / (norm_hat * norm_bar)
        return float(coherence)

    def get_self_belief(self) -> SelfBelief:
        """Get current self-belief."""
        s_hat = self.predict_self()
        coherence = self.compute_self_coherence()

        # Confidence from prediction accuracy
        if len(self.error_history) > 0:
            mean_error = np.mean(self.error_history)
            confidence = 1.0 / (1.0 + mean_error)
        else:
            confidence = 0.5

        # Self-type (placeholder - would use clustering)
        dominant_type = 0

        belief = SelfBelief(
            state_estimate=s_hat,
            coherence=coherence,
            confidence=confidence,
            dominant_type=dominant_type
        )

        self.belief_history.append(belief)
        return belief

    def get_statistics(self) -> Dict:
        """Get self-model statistics."""
        if len(self.error_history) == 0:
            return {'status': 'no_data'}

        return {
            'agent': self.agent_name,
            't': self.t,
            'mean_error': float(np.mean(self.error_history)),
            'recent_error': float(np.mean(self.error_history[-10:])),
            'coherence': self.compute_self_coherence(),
            'model_norm': float(np.linalg.norm(self.W_self))
        }


class TheoryOfMind:
    """
    Theory of Mind: modeling other agents.

    ŝ_{t+1}^{B|A} = f_other^A(H_{t-1}^B)

    Each agent A maintains models of all other agents B.
    """

    def __init__(self, agent_name: str, other_names: List[str], state_dim: int):
        """
        Initialize Theory of Mind.

        Args:
            agent_name: Name of this agent
            other_names: Names of other agents to model
            state_dim: Dimension of state vector
        """
        self.agent_name = agent_name
        self.other_names = other_names
        self.state_dim = state_dim

        # Model weights for each other agent
        self.W_other: Dict[str, np.ndarray] = {
            name: np.eye(state_dim) * 0.1 for name in other_names
        }

        # State histories for each other agent
        self.other_histories: Dict[str, List[np.ndarray]] = {
            name: [] for name in other_names
        }

        # Prediction errors
        self.other_errors: Dict[str, List[float]] = {
            name: [] for name in other_names
        }

        self.t = 0

    def _compute_window_size(self) -> int:
        """Endogenous window size."""
        return max(5, int(np.sqrt(self.t + 1)))

    def record_other_state(self, other_name: str, s: np.ndarray):
        """Record observed state of another agent."""
        if other_name not in self.other_histories:
            return

        self.other_histories[other_name].append(s.copy())
        self.t += 1

        max_hist = 500
        if len(self.other_histories[other_name]) > max_hist:
            self.other_histories[other_name] = self.other_histories[other_name][-max_hist:]

    def predict_other(self, other_name: str) -> np.ndarray:
        """
        Predict other agent's next state.

        ŝ_{t+1}^{B|A} = W_other^{A->B} · h̄_B
        """
        if other_name not in self.W_other:
            return np.zeros(self.state_dim)

        history = self.other_histories.get(other_name, [])
        if len(history) < 2:
            return np.zeros(self.state_dim)

        W = self._compute_window_size()
        recent = np.array(history[-W:])
        h_bar = np.mean(recent, axis=0)

        s_hat = self.W_other[other_name] @ h_bar
        return s_hat

    def update_model(self, other_name: str, s_actual: np.ndarray):
        """
        Update model of other agent.

        e_{t+1}^{B|A} = s_{t+1}^B - ŝ_{t+1}^{B|A}
        """
        if other_name not in self.W_other:
            return

        s_hat = self.predict_other(other_name)
        e = s_actual - s_hat
        error_norm = np.linalg.norm(e)

        self.other_errors[other_name].append(error_norm)
        if len(self.other_errors[other_name]) > 500:
            self.other_errors[other_name] = self.other_errors[other_name][-500:]

        # Rank of error
        errors = self.other_errors[other_name]
        if len(errors) > 5:
            omega = np.sum(np.array(errors) <= error_norm) / len(errors)
        else:
            omega = 0.5

        # Learning rate
        eta = 1.0 / np.sqrt(self.t + 1)

        # Gradient update
        W = self._compute_window_size()
        history = self.other_histories.get(other_name, [])
        if len(history) >= W:
            h_bar = np.mean(np.array(history[-W:]), axis=0)
            g = np.outer(e, h_bar)
            g_norm = np.linalg.norm(g) + 1e-16

            self.W_other[other_name] -= eta * omega * g / g_norm

        # Record state
        self.record_other_state(other_name, s_actual)

    def get_tom_accuracy(self, other_name: str) -> float:
        """
        Get Theory of Mind accuracy for another agent.

        ToM_{A->B} = 1 - rank(MSE(e^{B|A}))
        """
        if other_name not in self.other_errors:
            return 0.5

        errors = self.other_errors[other_name]
        if len(errors) < 5:
            return 0.5

        mse = np.mean(np.array(errors) ** 2)
        all_mses = [np.mean(np.array(e) ** 2) for e in self.other_errors.values() if len(e) > 0]

        if len(all_mses) == 0:
            return 0.5

        rank = np.sum(np.array(all_mses) <= mse) / len(all_mses)
        return 1.0 - rank

    def get_statistics(self) -> Dict:
        """Get ToM statistics."""
        stats = {
            'agent': self.agent_name,
            't': self.t,
            'tom_accuracy': {}
        }

        for other in self.other_names:
            if len(self.other_errors.get(other, [])) > 0:
                stats['tom_accuracy'][other] = self.get_tom_accuracy(other)
                stats[f'mean_error_{other}'] = float(np.mean(self.other_errors[other]))

        return stats


def test_self_model_and_tom():
    """Test self-model and Theory of Mind."""
    print("=" * 60)
    print("SELF-MODEL AND THEORY OF MIND TEST")
    print("=" * 60)

    state_dim = 17  # z(6) + phi(5) + D(6)

    # Create agents
    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']

    # Self-models
    self_models = {name: SelfModel(name, state_dim) for name in agents}

    # Theory of Mind
    toms = {
        name: TheoryOfMind(name, [a for a in agents if a != name], state_dim)
        for name in agents
    }

    print(f"\nSimulating {len(agents)} agents for 200 steps...")

    # Simulate
    agent_states = {name: np.random.randn(state_dim) * 0.1 for name in agents}

    for t in range(200):
        for name in agents:
            # Evolve state
            s = agent_states[name]
            s = 0.95 * s + np.random.randn(state_dim) * 0.05
            agent_states[name] = s

            # Update self-model
            self_models[name].update_model(s)

            # Update ToM
            for other in agents:
                if other != name:
                    toms[name].update_model(other, agent_states[other])

        if (t + 1) % 50 == 0:
            print(f"\n  t={t+1}:")
            for name in agents[:2]:  # Just show first 2
                sm_stats = self_models[name].get_statistics()
                tom_stats = toms[name].get_statistics()
                print(f"    {name}:")
                print(f"      Self-model error: {sm_stats['mean_error']:.3f}, "
                      f"coherence: {sm_stats['coherence']:.3f}")
                print(f"      ToM accuracy: {tom_stats['tom_accuracy']}")

    # Final self-beliefs
    print("\n" + "=" * 60)
    print("FINAL SELF-BELIEFS:")
    for name in agents:
        belief = self_models[name].get_self_belief()
        print(f"  {name}: coherence={belief.coherence:.3f}, "
              f"confidence={belief.confidence:.3f}")

    return self_models, toms


if __name__ == "__main__":
    test_self_model_and_tom()
