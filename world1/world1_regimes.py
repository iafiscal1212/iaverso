"""
WORLD-1 Regime Detection

Detects structural regimes (modes) without human labels.
All clustering and detection endogenous.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Regime:
    """A structural regime in WORLD-1."""
    idx: int
    centroid: np.ndarray        # Center of regime in feature space
    variance: float             # Internal variance
    frequency: float            # How often it occurs
    transitions_to: Dict[int, float]  # Transition probabilities to other regimes


class RegimeDetector:
    """
    Detects and tracks structural regimes in WORLD-1.

    Regimes emerge from clustering world history features.
    Number of regimes (k) determined endogenously.
    """

    def __init__(self, world_dim: int):
        """Initialize regime detector."""
        self.world_dim = world_dim
        self.t = 0

        # Feature history (not raw states)
        self.feature_history: List[np.ndarray] = []

        # Detected regimes
        self.regimes: Dict[int, Regime] = {}
        self.n_regimes: int = 2  # Start with 2, will adapt

        # Regime sequence
        self.regime_sequence: List[int] = []

        # Current regime
        self.current_regime: int = 0

        # For clustering
        self.centroids: Optional[np.ndarray] = None

    def _extract_features(self, w: np.ndarray) -> np.ndarray:
        """
        Extract features from world state for regime detection.

        Features are structural, not semantic.
        Always returns features of size 3 * len(w) for consistency.
        """
        # Features:
        # 1. Current state (normalized)
        feat_state = w / (np.linalg.norm(w) + 1e-8)

        # 2. Variance of recent window
        if len(self.feature_history) > 1:
            W = min(10, len(self.feature_history))
            # Extract just the state part (first len(w) elements) from each feature
            recent_states = []
            for f in self.feature_history[-W:]:
                recent_states.append(f[:len(w)])
            recent_states = np.array(recent_states)
            feat_var = np.var(recent_states, axis=0)
        else:
            feat_var = np.zeros_like(w)

        # 3. Change rate
        if len(self.feature_history) > 0:
            prev_state = self.feature_history[-1][:len(w)]
            feat_change = w - prev_state
        else:
            feat_change = np.zeros_like(w)

        # Combine - always same size: 3 * len(w)
        features = np.concatenate([
            feat_state,
            feat_var,
            feat_change
        ])

        return features

    def _compute_optimal_k(self) -> int:
        """
        Determine optimal number of regimes endogenously.

        k = max(2, min(6, floor(sqrt(d_eff))))
        where d_eff is effective dimension of features.
        """
        if len(self.feature_history) < 20:
            return 2

        # Compute effective dimension from feature covariance
        recent = np.array(self.feature_history[-100:])
        cov = np.cov(recent.T)

        try:
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[eigenvalues > 0]

            if len(eigenvalues) > 0:
                median_eig = np.median(eigenvalues)
                d_eff = np.sum(eigenvalues >= median_eig)
            else:
                d_eff = 2
        except:
            d_eff = 2

        k = max(2, min(6, int(np.floor(np.sqrt(d_eff)))))
        return k

    def _simple_kmeans(self, data: np.ndarray, k: int, max_iter: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple k-means clustering.

        Returns:
            centroids: (k, d) array
            labels: (n,) array
        """
        n, d = data.shape

        # Initialize centroids from data points
        indices = np.random.choice(n, min(k, n), replace=False)
        centroids = data[indices].copy()

        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            # Assign points to nearest centroid
            for i in range(n):
                distances = [np.linalg.norm(data[i] - c) for c in centroids]
                labels[i] = np.argmin(distances)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if np.sum(mask) > 0:
                    new_centroids[j] = data[mask].mean(axis=0)
                else:
                    new_centroids[j] = data[np.random.randint(n)]

            # Check convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return centroids, labels

    def update_regimes(self):
        """
        Update regime detection from feature history.

        Performs clustering and updates regime definitions.
        """
        if len(self.feature_history) < 20:
            return

        # Determine k endogenously
        self.n_regimes = self._compute_optimal_k()

        # Get recent features
        data = np.array(self.feature_history[-200:])

        # Cluster
        self.centroids, labels = self._simple_kmeans(data, self.n_regimes)

        # Update regime definitions
        self.regimes = {}
        for j in range(self.n_regimes):
            mask = labels == j
            if np.sum(mask) > 0:
                cluster_data = data[mask]
                variance = float(np.var(cluster_data))
                frequency = float(np.mean(mask))
            else:
                variance = 0.0
                frequency = 0.0

            self.regimes[j] = Regime(
                idx=j,
                centroid=self.centroids[j],
                variance=variance,
                frequency=frequency,
                transitions_to={}
            )

        # Compute transition probabilities
        self._compute_transitions(labels)

    def _compute_transitions(self, labels: np.ndarray):
        """Compute transition probabilities between regimes."""
        if len(labels) < 2:
            return

        for i in range(len(labels) - 1):
            from_regime = labels[i]
            to_regime = labels[i + 1]

            if from_regime not in self.regimes:
                continue

            if to_regime not in self.regimes[from_regime].transitions_to:
                self.regimes[from_regime].transitions_to[to_regime] = 0

            self.regimes[from_regime].transitions_to[to_regime] += 1

        # Normalize to probabilities
        for regime in self.regimes.values():
            total = sum(regime.transitions_to.values())
            if total > 0:
                for k in regime.transitions_to:
                    regime.transitions_to[k] /= total

    def detect_regime(self, w: np.ndarray) -> int:
        """
        Detect current regime from world state.

        Returns:
            Regime index
        """
        self.t += 1

        # Extract features
        features = self._extract_features(w)
        self.feature_history.append(features)

        # Keep bounded
        max_hist = 500
        if len(self.feature_history) > max_hist:
            self.feature_history = self.feature_history[-max_hist:]

        # Update regimes periodically
        if self.t % 20 == 0:
            self.update_regimes()

        # Find nearest regime
        if self.centroids is None or len(self.centroids) == 0:
            self.current_regime = 0
        else:
            distances = [np.linalg.norm(features[:len(c)] - c[:len(features)])
                        for c in self.centroids]
            self.current_regime = int(np.argmin(distances))

        self.regime_sequence.append(self.current_regime)
        if len(self.regime_sequence) > max_hist:
            self.regime_sequence = self.regime_sequence[-max_hist:]

        return self.current_regime

    def get_regime_encoding(self) -> np.ndarray:
        """
        Get soft encoding of current regime.

        Returns probability distribution over regimes.
        """
        if self.centroids is None:
            return np.ones(max(2, self.n_regimes)) / max(2, self.n_regimes)

        features = self.feature_history[-1] if self.feature_history else np.zeros(self.world_dim)

        # Distances to each centroid
        distances = np.array([
            np.linalg.norm(features[:len(c)] - c[:len(features)])
            for c in self.centroids
        ])

        # Convert to soft assignment (softmax-like)
        inv_distances = 1.0 / (distances + 1e-8)
        probs = inv_distances / inv_distances.sum()

        return probs

    def get_regime_stability(self) -> float:
        """
        Get stability of current regime.

        High stability = same regime for many steps.
        """
        if len(self.regime_sequence) < 10:
            return 0.5

        recent = self.regime_sequence[-20:]
        same_as_current = np.sum(np.array(recent) == self.current_regime)
        stability = same_as_current / len(recent)

        return float(stability)

    def predict_next_regime(self) -> Tuple[int, float]:
        """
        Predict next regime based on transition probabilities.

        Returns:
            (predicted_regime, confidence)
        """
        if self.current_regime not in self.regimes:
            return (self.current_regime, 0.5)

        transitions = self.regimes[self.current_regime].transitions_to

        if not transitions:
            return (self.current_regime, 0.5)

        # Most likely transition
        next_regime = max(transitions, key=transitions.get)
        confidence = transitions[next_regime]

        return (next_regime, confidence)

    def get_statistics(self) -> Dict:
        """Get regime statistics."""
        return {
            't': self.t,
            'n_regimes': self.n_regimes,
            'current_regime': self.current_regime,
            'regime_stability': self.get_regime_stability(),
            'regime_frequencies': {r.idx: r.frequency for r in self.regimes.values()},
            'regime_changes': np.sum(np.diff(self.regime_sequence) != 0) if len(self.regime_sequence) > 1 else 0
        }


def test_regimes():
    """Test regime detector."""
    print("=" * 60)
    print("REGIME DETECTOR TEST")
    print("=" * 60)

    world_dim = 15
    detector = RegimeDetector(world_dim)

    print(f"\nInitialized detector for world_dim={world_dim}")

    # Simulate world with regime changes
    w = np.random.randn(world_dim) * 0.5
    regime_mode = 0

    for t in range(300):
        # Simulate regime changes
        if t % 80 == 0:
            regime_mode = (regime_mode + 1) % 3

        # Different dynamics per regime
        if regime_mode == 0:
            # Stable regime
            w = 0.98 * w + np.random.randn(world_dim) * 0.05
        elif regime_mode == 1:
            # Volatile regime
            w = 0.8 * w + np.random.randn(world_dim) * 0.3
        else:
            # Oscillating regime
            w = 0.9 * w + 0.1 * np.sin(np.arange(world_dim) * t * 0.1)

        # Detect regime
        detected = detector.detect_regime(w)

        if (t + 1) % 50 == 0:
            stats = detector.get_statistics()
            encoding = detector.get_regime_encoding()
            next_pred, conf = detector.predict_next_regime()

            print(f"\n  t={t+1}:")
            print(f"    True regime mode: {regime_mode}")
            print(f"    Detected regime: {detected}")
            print(f"    N regimes: {stats['n_regimes']}")
            print(f"    Stability: {stats['regime_stability']:.3f}")
            print(f"    Encoding: {encoding.round(2)}")
            print(f"    Next prediction: {next_pred} (conf={conf:.2f})")

    return detector


if __name__ == "__main__":
    test_regimes()
