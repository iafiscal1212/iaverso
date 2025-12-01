"""
Episodic Memory System

Endogenous episode segmentation, encoding, and persistence.
Episodes are real experiences, not logs.

Notation:
- t = simulator time
- τ_t = subjective internal time
- z_t ∈ R^d = structural state (manifold)
- φ_t ∈ R^k = phenomenological vector
- D_t = drive vector
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy import linalg


@dataclass
class Episode:
    """A single episode of experience."""
    idx: int
    t_start: int
    t_end: int
    tau_start: float  # Subjective time start
    tau_end: float    # Subjective time end

    # Compressed features (averages within episode)
    z_bar: np.ndarray           # Mean structural state
    phi_bar: np.ndarray         # Mean phenomenological vector
    D_bar: np.ndarray           # Mean drives

    # Variance within episode
    z_var: float = 0.0
    phi_var: float = 0.0

    # Projected representation (episodic token)
    y: Optional[np.ndarray] = None

    # Importance metrics
    importance: float = 0.0
    persistence_weight: float = 0.0

    @property
    def length(self) -> int:
        """Duration in simulator steps."""
        return self.t_end - self.t_start + 1

    @property
    def subjective_duration(self) -> float:
        """Duration in subjective time."""
        return self.tau_end - self.tau_start


class EpisodicMemory:
    """
    Episodic memory with endogenous segmentation and encoding.

    Detects episode boundaries via structural shocks.
    Encodes episodes as compressed vectors.
    Maintains persistence weights based on importance and age.
    """

    def __init__(self, z_dim: int = 6, phi_dim: int = 5, D_dim: int = 6):
        """
        Initialize episodic memory.

        Args:
            z_dim: Dimension of structural state
            phi_dim: Dimension of phenomenological vector
            D_dim: Dimension of drive vector
        """
        self.z_dim = z_dim
        self.phi_dim = phi_dim
        self.D_dim = D_dim
        self.feature_dim = z_dim + phi_dim + D_dim

        # Time tracking
        self.t = 0
        self.tau = 0.0  # Subjective time

        # State histories
        self.z_history: List[np.ndarray] = []
        self.phi_history: List[np.ndarray] = []
        self.D_history: List[np.ndarray] = []
        self.tau_history: List[float] = []

        # Shock tracking for segmentation
        self.shock_history: List[float] = []
        self.shock_threshold: float = 0.9  # Will be endogenous (90th percentile)

        # Episodes
        self.episodes: List[Episode] = []
        self.current_episode_start: int = 0

        # Projection matrix for encoding (learned endogenously)
        self.V_epi: Optional[np.ndarray] = None
        self.d_epi: int = 3  # Effective episodic dimension

        # Episode feature covariance
        self.episode_features: List[np.ndarray] = []

    def _compute_shock(self, z: np.ndarray, phi: np.ndarray) -> float:
        """
        Compute structural shock at current timestep.

        s_t = rank(Δz_t) + rank(Δφ_t)
        """
        if len(self.z_history) < 2:
            return 0.0

        # Compute deltas
        delta_z = np.linalg.norm(z - self.z_history[-1])
        delta_phi = np.linalg.norm(phi - self.phi_history[-1])

        # Compute ranks from history
        if len(self.z_history) > 10:
            z_deltas = [np.linalg.norm(self.z_history[i] - self.z_history[i-1])
                       for i in range(1, len(self.z_history))]
            phi_deltas = [np.linalg.norm(self.phi_history[i] - self.phi_history[i-1])
                         for i in range(1, len(self.phi_history))]

            rank_z = np.sum(np.array(z_deltas) <= delta_z) / len(z_deltas)
            rank_phi = np.sum(np.array(phi_deltas) <= delta_phi) / len(phi_deltas)
        else:
            rank_z = 0.5
            rank_phi = 0.5

        shock = rank_z + rank_phi
        return shock

    def _update_shock_threshold(self):
        """
        Update shock threshold endogenously.

        θ_s(t) = percentile_95(H_s)

        Using 95th percentile for more selective cuts.
        """
        if len(self.shock_history) < 20:
            self.shock_threshold = 1.6  # Initial conservative threshold
        else:
            self.shock_threshold = np.percentile(self.shock_history, 95)

    def _should_cut_episode(self, shock: float) -> bool:
        """Check if current shock warrants episode cut."""
        return shock >= self.shock_threshold

    def _create_episode(self, t_start: int, t_end: int) -> Episode:
        """
        Create episode from time interval.

        Computes compressed features:
        - z̄(e) = mean of z over episode
        - φ̄(e) = mean of φ over episode
        - D̄(e) = mean of D over episode
        """
        # Extract data for episode
        z_data = np.array(self.z_history[t_start:t_end+1])
        phi_data = np.array(self.phi_history[t_start:t_end+1])
        D_data = np.array(self.D_history[t_start:t_end+1])

        # Compute means
        z_bar = np.mean(z_data, axis=0)
        phi_bar = np.mean(phi_data, axis=0)
        D_bar = np.mean(D_data, axis=0)

        # Compute variances
        z_var = float(np.var(z_data))
        phi_var = float(np.var(phi_data))

        # Get subjective times
        tau_start = self.tau_history[t_start] if t_start < len(self.tau_history) else 0.0
        tau_end = self.tau_history[t_end] if t_end < len(self.tau_history) else self.tau

        episode = Episode(
            idx=len(self.episodes),
            t_start=t_start,
            t_end=t_end,
            tau_start=tau_start,
            tau_end=tau_end,
            z_bar=z_bar,
            phi_bar=phi_bar,
            D_bar=D_bar,
            z_var=z_var,
            phi_var=phi_var
        )

        # Compute importance
        episode.importance = self._compute_importance(episode)

        return episode

    def _compute_importance(self, episode: Episode) -> float:
        """
        Compute structural importance of episode.

        r_e = rank(||φ̄(e)||) + rank(var(φ in e)) + rank(L_e)
        """
        if len(self.episodes) < 2:
            return 1.0

        # Collect historical values
        phi_norms = [np.linalg.norm(e.phi_bar) for e in self.episodes]
        phi_vars = [e.phi_var for e in self.episodes]
        lengths = [e.length for e in self.episodes]

        # Current values
        phi_norm = np.linalg.norm(episode.phi_bar)
        phi_var = episode.phi_var
        length = episode.length

        # Compute ranks
        rank_phi = np.sum(np.array(phi_norms) <= phi_norm) / max(1, len(phi_norms))
        rank_var = np.sum(np.array(phi_vars) <= phi_var) / max(1, len(phi_vars))
        rank_len = np.sum(np.array(lengths) <= length) / max(1, len(lengths))

        importance = rank_phi + rank_var + rank_len
        return importance

    def _update_projection_matrix(self):
        """
        Update episodic projection matrix from covariance.

        Σ_x = cov({x(e)})
        V_epi = eigenvectors with λ >= median(λ)
        """
        if len(self.episode_features) < 5:
            return

        X = np.array(self.episode_features)
        Sigma = np.cov(X.T)

        try:
            eigenvalues, eigenvectors = linalg.eigh(Sigma)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Effective dimension
            median_lambda = np.median(eigenvalues[eigenvalues > 0])
            self.d_epi = max(1, np.sum(eigenvalues >= median_lambda))

            # Projection matrix
            self.V_epi = eigenvectors[:, :self.d_epi]

        except:
            pass

    def _project_episode(self, episode: Episode):
        """
        Project episode to compressed representation.

        y(e) = V_epi^T @ x(e)
        """
        # Construct feature vector
        x = np.concatenate([episode.z_bar, episode.phi_bar, episode.D_bar])

        # Store for covariance computation
        self.episode_features.append(x)

        # Project if matrix available
        if self.V_epi is not None:
            episode.y = self.V_epi.T @ x
        else:
            episode.y = x[:self.d_epi]

    def _update_persistence_weights(self):
        """
        Update persistence weights for all episodes.

        w_e(t) = (r_e / Σr_j) * 1/(1 + Δτ_e(t))

        where Δτ_e(t) = τ_t - τ_end^e
        """
        if len(self.episodes) == 0:
            return

        # Total importance
        total_importance = sum(e.importance for e in self.episodes) + 1e-8

        for episode in self.episodes:
            # Age in subjective time
            delta_tau = self.tau - episode.tau_end

            # Normalized importance
            norm_importance = episode.importance / total_importance

            # Decay with subjective age
            decay = 1.0 / (1.0 + delta_tau)

            episode.persistence_weight = norm_importance * decay

    def is_persistent(self, episode: Episode) -> bool:
        """
        Check if episode is persistent (above 75th percentile weight).

        persistent(e, t) = I[w_e(t) >= percentile_75({w_j(t)})]
        """
        if len(self.episodes) < 4:
            return True

        weights = [e.persistence_weight for e in self.episodes]
        threshold = np.percentile(weights, 75)
        return episode.persistence_weight >= threshold

    def record(self, z: np.ndarray, phi: np.ndarray, D: np.ndarray,
               tau: Optional[float] = None):
        """
        Record new state and potentially create episode.

        Args:
            z: Structural state
            phi: Phenomenological vector
            D: Drive vector
            tau: Subjective time (if None, increments by 1)
        """
        self.t += 1

        # Update subjective time
        if tau is not None:
            self.tau = tau
        else:
            self.tau += 1.0

        # Store state
        self.z_history.append(z.copy())
        self.phi_history.append(phi.copy())
        self.D_history.append(D.copy())
        self.tau_history.append(self.tau)

        # Keep bounded
        max_hist = 10000
        if len(self.z_history) > max_hist:
            offset = len(self.z_history) - max_hist
            self.z_history = self.z_history[-max_hist:]
            self.phi_history = self.phi_history[-max_hist:]
            self.D_history = self.D_history[-max_hist:]
            self.tau_history = self.tau_history[-max_hist:]
            self.current_episode_start = max(0, self.current_episode_start - offset)

        # Compute shock
        shock = self._compute_shock(z, phi)
        self.shock_history.append(shock)

        # Update threshold
        self._update_shock_threshold()

        # Check for episode cut (minimum episode length = 25 steps)
        if self._should_cut_episode(shock) and self.t - self.current_episode_start > 25:
            # Create episode
            episode = self._create_episode(self.current_episode_start, self.t - 1)
            self._project_episode(episode)
            self.episodes.append(episode)

            # Update projection matrix
            if len(self.episodes) % 5 == 0:
                self._update_projection_matrix()

            # Start new episode
            self.current_episode_start = self.t

        # Update persistence weights
        if self.t % 10 == 0:
            self._update_persistence_weights()

    def get_persistent_episodes(self) -> List[Episode]:
        """Get all persistent episodes."""
        return [e for e in self.episodes if self.is_persistent(e)]

    def get_recent_episodes(self, n: int = 10) -> List[Episode]:
        """Get n most recent episodes."""
        return self.episodes[-n:] if len(self.episodes) >= n else self.episodes

    def get_episode_by_tau(self, tau: float) -> Optional[Episode]:
        """Get episode containing given subjective time."""
        for episode in self.episodes:
            if episode.tau_start <= tau <= episode.tau_end:
                return episode
        return None

    def similarity(self, e1: Episode, e2: Episode) -> float:
        """Compute similarity between two episodes."""
        # Always use raw features for consistent dimensions
        x1 = np.concatenate([e1.z_bar, e1.phi_bar, e1.D_bar])
        x2 = np.concatenate([e2.z_bar, e2.phi_bar, e2.D_bar])

        # Cosine similarity
        norm1 = np.linalg.norm(x1)
        norm2 = np.linalg.norm(x2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(x1, x2) / (norm1 * norm2))

    def get_statistics(self) -> Dict:
        """Get memory statistics."""
        if len(self.episodes) == 0:
            return {'n_episodes': 0}

        persistent = self.get_persistent_episodes()

        return {
            't': self.t,
            'tau': self.tau,
            'n_episodes': len(self.episodes),
            'n_persistent': len(persistent),
            'd_epi': self.d_epi,
            'mean_episode_length': float(np.mean([e.length for e in self.episodes])),
            'mean_importance': float(np.mean([e.importance for e in self.episodes])),
            'shock_threshold': float(self.shock_threshold)
        }


def test_episodic_memory():
    """Test episodic memory system."""
    print("=" * 60)
    print("EPISODIC MEMORY TEST")
    print("=" * 60)

    memory = EpisodicMemory(z_dim=6, phi_dim=5, D_dim=6)

    print(f"\nSimulating 500 steps with shocks...")

    # Simulate agent life with occasional shocks
    z = np.random.randn(6) * 0.1
    phi = np.random.randn(5) * 0.1
    D = np.abs(np.random.randn(6))
    D = D / D.sum()

    tau = 0.0

    for t in range(500):
        # Normal evolution
        z = 0.95 * z + 0.05 * np.tanh(z) + np.random.randn(6) * 0.02
        phi = 0.9 * phi + 0.1 * np.random.randn(5) * 0.1
        D = D + np.random.randn(6) * 0.01
        D = np.abs(D)
        D = D / D.sum()

        # Occasional shocks
        if t % 80 == 40:
            z += np.random.randn(6) * 0.5
            phi += np.random.randn(5) * 0.3

        # Subjective time (varies with phi magnitude)
        tau += 1.0 + 0.5 * np.linalg.norm(phi)

        memory.record(z, phi, D, tau)

        if (t + 1) % 100 == 0:
            stats = memory.get_statistics()
            print(f"\n  t={t+1}:")
            print(f"    Episodes: {stats['n_episodes']}")
            print(f"    Persistent: {stats['n_persistent']}")
            print(f"    Mean length: {stats['mean_episode_length']:.1f}")

    # Final stats
    print("\n" + "=" * 60)
    stats = memory.get_statistics()
    print(f"Final: {stats['n_episodes']} episodes, {stats['n_persistent']} persistent")
    print(f"d_epi = {stats['d_epi']}")

    # Show persistent episodes
    persistent = memory.get_persistent_episodes()
    print(f"\nPersistent episodes:")
    for e in persistent[:5]:
        print(f"  Episode {e.idx}: t=[{e.t_start}, {e.t_end}], "
              f"importance={e.importance:.2f}, weight={e.persistence_weight:.3f}")

    return memory


if __name__ == "__main__":
    test_episodic_memory()
