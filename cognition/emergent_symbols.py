"""
Emergent Symbols from Consequences

Symbols emerge from clustering state transitions.
Each symbol represents a category of consequence patterns.

All endogenous - semantic is purely statistical.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy import linalg


@dataclass
class Symbol:
    """An emergent symbol representing a consequence pattern."""
    idx: int
    center: np.ndarray          # Cluster center in transition space
    frequency: float            # How often this pattern appears
    mean_D_effect: np.ndarray   # Average effect on drives
    mean_phi_effect: np.ndarray # Average effect on phenomenology
    semantic_strength: float    # How predictive of future


class EmergentSymbols:
    """
    Emergent symbol system from consequence clustering.

    Δs_t = [z_{t+1} - z_t, φ_{t+1} - φ_t, D_{t+1} - D_t]
    u_t = V_Δ^T @ Δs_t
    σ_k = cluster(u_t)

    Semantic = statistical effect on future drives and φ.
    """

    def __init__(self, z_dim: int = 6, phi_dim: int = 5, D_dim: int = 6):
        """
        Initialize emergent symbol system.

        Args:
            z_dim: Dimension of structural state
            phi_dim: Dimension of phenomenological vector
            D_dim: Dimension of drive vector
        """
        self.z_dim = z_dim
        self.phi_dim = phi_dim
        self.D_dim = D_dim
        self.delta_dim = z_dim + phi_dim + D_dim

        # State histories
        self.z_history: List[np.ndarray] = []
        self.phi_history: List[np.ndarray] = []
        self.D_history: List[np.ndarray] = []

        # Transition vectors
        self.delta_history: List[np.ndarray] = []
        self.projected_history: List[np.ndarray] = []

        # Projection matrix (from covariance)
        self.V_delta: Optional[np.ndarray] = None
        self.d_delta: int = 3  # Effective dimension

        # Symbols
        self.symbols: List[Symbol] = []
        self.n_symbols: int = 0
        self.symbol_centers: Optional[np.ndarray] = None

        # Symbol assignment history
        self.symbol_assignments: List[int] = []

        self.t = 0

    def record_state(self, z: np.ndarray, phi: np.ndarray, D: np.ndarray):
        """
        Record new state and compute transition.

        Δs_t = [z_{t+1} - z_t, φ_{t+1} - φ_t, D_{t+1} - D_t]
        """
        self.z_history.append(z.copy())
        self.phi_history.append(phi.copy())
        self.D_history.append(D.copy())
        self.t += 1

        # Compute transition if we have previous state
        if len(self.z_history) >= 2:
            delta_z = z - self.z_history[-2]
            delta_phi = phi - self.phi_history[-2]
            delta_D = D - self.D_history[-2]

            delta = np.concatenate([delta_z, delta_phi, delta_D])
            self.delta_history.append(delta)

            # Project if matrix available
            if self.V_delta is not None:
                u = self.V_delta.T @ delta
                self.projected_history.append(u)
            else:
                self.projected_history.append(delta[:self.d_delta])

        # Keep bounded
        max_hist = 1000
        if len(self.z_history) > max_hist:
            self.z_history = self.z_history[-max_hist:]
            self.phi_history = self.phi_history[-max_hist:]
            self.D_history = self.D_history[-max_hist:]
            self.delta_history = self.delta_history[-max_hist:]
            self.projected_history = self.projected_history[-max_hist:]

        # Update projection matrix periodically
        if self.t % 50 == 0 and len(self.delta_history) > 20:
            self._update_projection_matrix()

    def _update_projection_matrix(self):
        """
        Update projection matrix from transition covariance.

        Σ_Δ = cov({Δs_t})
        V_Δ = eigenvectors with λ >= median(λ)
        """
        if len(self.delta_history) < 10:
            return

        Delta = np.array(self.delta_history)
        Sigma = np.cov(Delta.T)

        try:
            eigenvalues, eigenvectors = linalg.eigh(Sigma)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Effective dimension
            positive_eigenvalues = eigenvalues[eigenvalues > 0]
            if len(positive_eigenvalues) > 0:
                median_lambda = np.median(positive_eigenvalues)
                self.d_delta = max(2, min(8, np.sum(eigenvalues >= median_lambda)))
            else:
                self.d_delta = 3

            # Projection matrix
            self.V_delta = eigenvectors[:, :self.d_delta]

            # Re-project history
            self.projected_history = []
            for delta in self.delta_history:
                u = self.V_delta.T @ delta
                self.projected_history.append(u)

        except:
            pass

    def discover_symbols(self):
        """
        Discover symbols via clustering of projected transitions.

        k = d_Δ (effective dimension)
        σ_k = cluster_center_k
        """
        if len(self.projected_history) < 20:
            return

        U = np.array(self.projected_history)

        # Number of symbols = effective dimension
        self.n_symbols = min(self.d_delta, 7)

        # Simple k-means
        self.symbol_centers = self._simple_kmeans(U, self.n_symbols)

        # Assign all points to symbols
        self.symbol_assignments = []
        for u in self.projected_history:
            distances = [np.linalg.norm(u - c) for c in self.symbol_centers]
            self.symbol_assignments.append(int(np.argmin(distances)))

        # Create symbol objects with semantics
        self._compute_symbol_semantics()

    def _simple_kmeans(self, data: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
        """Simple k-means clustering."""
        n = len(data)
        if n < k:
            return data[:k] if len(data) >= k else np.zeros((k, data.shape[1]))

        # Initialize from data points
        indices = np.random.choice(n, k, replace=False)
        centers = data[indices].copy()

        for _ in range(max_iter):
            # Assign points
            labels = np.zeros(n, dtype=int)
            for i in range(n):
                distances = [np.linalg.norm(data[i] - c) for c in centers]
                labels[i] = np.argmin(distances)

            # Update centers
            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = labels == j
                if np.sum(mask) > 0:
                    new_centers[j] = data[mask].mean(axis=0)
                else:
                    new_centers[j] = data[np.random.randint(n)]

            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        return centers

    def _compute_symbol_semantics(self):
        """
        Compute semantic content of each symbol.

        Semantic = mean effect on future D and φ.
        """
        if len(self.symbol_assignments) == 0:
            return

        self.symbols = []

        for k in range(self.n_symbols):
            # Find all transitions assigned to this symbol
            indices = [i for i, s in enumerate(self.symbol_assignments) if s == k]

            if len(indices) == 0:
                continue

            # Frequency
            frequency = len(indices) / len(self.symbol_assignments)

            # Mean effects
            D_effects = []
            phi_effects = []

            for i in indices:
                if i < len(self.delta_history):
                    delta = self.delta_history[i]
                    # Extract components
                    delta_phi = delta[self.z_dim:self.z_dim + self.phi_dim]
                    delta_D = delta[self.z_dim + self.phi_dim:]

                    phi_effects.append(delta_phi)
                    D_effects.append(delta_D)

            if len(D_effects) > 0:
                mean_D_effect = np.mean(D_effects, axis=0)
                mean_phi_effect = np.mean(phi_effects, axis=0)
            else:
                mean_D_effect = np.zeros(self.D_dim)
                mean_phi_effect = np.zeros(self.phi_dim)

            # Semantic strength = predictiveness
            # High variance in effects = low predictiveness
            if len(D_effects) > 1:
                var_D = np.var(D_effects)
                var_phi = np.var(phi_effects)
                semantic_strength = 1.0 / (1.0 + var_D + var_phi)
            else:
                semantic_strength = 0.5

            symbol = Symbol(
                idx=k,
                center=self.symbol_centers[k],
                frequency=frequency,
                mean_D_effect=mean_D_effect,
                mean_phi_effect=mean_phi_effect,
                semantic_strength=semantic_strength
            )
            self.symbols.append(symbol)

    def get_current_symbol(self) -> Optional[Symbol]:
        """Get symbol of most recent transition."""
        if len(self.projected_history) == 0 or len(self.symbols) == 0:
            return None

        u = self.projected_history[-1]
        distances = [np.linalg.norm(u - s.center) for s in self.symbols]
        nearest_idx = np.argmin(distances)

        return self.symbols[nearest_idx]

    def predict_from_symbol(self, symbol: Symbol) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict future effects from a symbol.

        Returns expected D and φ changes.
        """
        return symbol.mean_D_effect.copy(), symbol.mean_phi_effect.copy()

    def symbol_distance(self, s1: Symbol, s2: Symbol) -> float:
        """Distance between two symbols in semantic space."""
        d_effect = np.linalg.norm(s1.mean_D_effect - s2.mean_D_effect)
        phi_effect = np.linalg.norm(s1.mean_phi_effect - s2.mean_phi_effect)
        return d_effect + phi_effect

    def get_symbol_sequence(self, length: int = 10) -> List[int]:
        """Get recent symbol sequence."""
        if len(self.symbol_assignments) == 0:
            return []
        return self.symbol_assignments[-length:]

    def get_statistics(self) -> Dict:
        """Get symbol system statistics."""
        if len(self.symbols) == 0:
            return {'n_symbols': 0, 'd_delta': self.d_delta}

        return {
            'n_symbols': len(self.symbols),
            'd_delta': self.d_delta,
            'n_transitions': len(self.delta_history),
            'symbol_frequencies': [s.frequency for s in self.symbols],
            'semantic_strengths': [s.semantic_strength for s in self.symbols],
            'mean_semantic_strength': float(np.mean([s.semantic_strength for s in self.symbols]))
        }


class SymbolGrounding:
    """
    Ground symbols to phenomenological states.

    Maps symbols to characteristic φ patterns.
    """

    def __init__(self, symbol_system: EmergentSymbols, phi_dim: int = 5):
        """
        Initialize symbol grounding.

        Args:
            symbol_system: The emergent symbol system
            phi_dim: Dimension of phenomenological vector
        """
        self.symbols = symbol_system
        self.phi_dim = phi_dim

        # Grounding maps symbol -> typical phi
        self.grounding: Dict[int, np.ndarray] = {}

        # History of (symbol, phi) pairs
        self.grounding_history: List[Tuple[int, np.ndarray]] = []

    def record_grounding(self, symbol_idx: int, phi: np.ndarray):
        """Record a symbol-phi association."""
        self.grounding_history.append((symbol_idx, phi.copy()))

        # Keep bounded
        if len(self.grounding_history) > 1000:
            self.grounding_history = self.grounding_history[-1000:]

    def update_grounding(self):
        """
        Update grounding from history.

        φ̂(σ_k) = mean({φ_t : symbol(t) = k})
        """
        if len(self.grounding_history) < 10:
            return

        # Collect phi for each symbol
        symbol_phis: Dict[int, List[np.ndarray]] = {}

        for symbol_idx, phi in self.grounding_history:
            if symbol_idx not in symbol_phis:
                symbol_phis[symbol_idx] = []
            symbol_phis[symbol_idx].append(phi)

        # Compute mean phi for each symbol
        for symbol_idx, phis in symbol_phis.items():
            self.grounding[symbol_idx] = np.mean(phis, axis=0)

    def get_grounded_phi(self, symbol_idx: int) -> Optional[np.ndarray]:
        """Get grounded phi for a symbol."""
        return self.grounding.get(symbol_idx)

    def grounding_confidence(self, symbol_idx: int) -> float:
        """
        Compute confidence in grounding.

        confidence = 1 / (1 + var(φ for symbol))
        """
        phis = [phi for s, phi in self.grounding_history if s == symbol_idx]

        if len(phis) < 2:
            return 0.5

        var = np.var(phis)
        return 1.0 / (1.0 + var)

    def get_statistics(self) -> Dict:
        """Get grounding statistics."""
        if len(self.grounding) == 0:
            return {'status': 'no_grounding'}

        confidences = {k: self.grounding_confidence(k) for k in self.grounding.keys()}

        return {
            'n_grounded': len(self.grounding),
            'confidences': confidences,
            'mean_confidence': float(np.mean(list(confidences.values())))
        }


def test_emergent_symbols():
    """Test emergent symbol system."""
    print("=" * 60)
    print("EMERGENT SYMBOLS TEST")
    print("=" * 60)

    symbol_system = EmergentSymbols(z_dim=6, phi_dim=5, D_dim=6)

    print("\nSimulating 300 state transitions...")

    z = np.random.randn(6) * 0.1
    phi = np.random.randn(5) * 0.1
    D = np.abs(np.random.randn(6))
    D = D / D.sum()

    for t in range(300):
        # Different transition patterns
        if t % 50 < 15:
            # Pattern A: exploration
            z += np.random.randn(6) * 0.2
            phi += np.array([0.1, 0, 0, 0.1, 0])
            D[0] *= 1.1
        elif t % 50 < 30:
            # Pattern B: stabilization
            z = 0.9 * z
            phi += np.array([0, 0.1, 0.1, 0, 0])
            D[3] *= 1.1
        else:
            # Pattern C: integration
            z = 0.95 * z + 0.05 * np.mean(z)
            phi += np.array([0, 0, 0, 0, 0.1])
            D[4] *= 1.1

        D = np.clip(D, 0.05, None)
        D = D / D.sum()

        symbol_system.record_state(z, phi, D)

    # Discover symbols
    print("\nDiscovering symbols...")
    symbol_system.discover_symbols()

    stats = symbol_system.get_statistics()
    print(f"\nSymbol Statistics:")
    print(f"  Discovered {stats['n_symbols']} symbols")
    print(f"  Effective dimension: {stats['d_delta']}")
    print(f"  Frequencies: {stats['symbol_frequencies']}")
    print(f"  Mean semantic strength: {stats['mean_semantic_strength']:.3f}")

    # Show symbol details
    print("\nSymbol Details:")
    for s in symbol_system.symbols:
        print(f"  Symbol {s.idx}:")
        print(f"    Frequency: {s.frequency:.3f}")
        print(f"    D effect: {s.mean_D_effect}")
        print(f"    φ effect: {s.mean_phi_effect}")
        print(f"    Semantic strength: {s.semantic_strength:.3f}")

    # Test grounding
    print("\nTesting symbol grounding...")
    grounding = SymbolGrounding(symbol_system)

    for i, (s_idx, phi) in enumerate(zip(
        symbol_system.symbol_assignments[-50:],
        symbol_system.phi_history[-50:]
    )):
        grounding.record_grounding(s_idx, phi)

    grounding.update_grounding()
    grounding_stats = grounding.get_statistics()
    print(f"  Grounded symbols: {grounding_stats['n_grounded']}")
    print(f"  Mean confidence: {grounding_stats['mean_confidence']:.3f}")

    return symbol_system, grounding


if __name__ == "__main__":
    test_emergent_symbols()
