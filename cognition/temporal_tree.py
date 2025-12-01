"""
Temporal Tree: Proto-Future Simulation

Generates possible futures by applying internal operators.
Evaluates branches endogenously without external rewards.

Subjective time unit U = median(v_t) where v_t = τ_t - τ_{t-1}
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable


@dataclass
class TreeNode:
    """A node in the temporal tree."""
    depth: int
    branch: int
    z: np.ndarray           # Structural state
    D: np.ndarray           # Drives at this node
    phi: np.ndarray         # Phenomenological state
    operator_used: str      # Which operator created this node
    parent: Optional['TreeNode'] = None

    # Evaluations
    delta_D: Optional[np.ndarray] = None
    delta_phi: Optional[np.ndarray] = None
    crisis_risk: float = 0.0
    value: float = 0.0
    probability: float = 0.0


class TemporalTree:
    """
    Generates and evaluates future trajectories.

    Uses internal operators (homeostasis, exploration, etc.)
    to simulate possible futures and evaluate them endogenously.
    """

    def __init__(self, z_dim: int = 6, phi_dim: int = 5, D_dim: int = 6):
        """
        Initialize temporal tree.

        Args:
            z_dim: Dimension of structural state
            phi_dim: Dimension of phenomenological vector
            D_dim: Dimension of drive vector
        """
        self.z_dim = z_dim
        self.phi_dim = phi_dim
        self.D_dim = D_dim

        # Operators (will be populated)
        self.operators: Dict[str, Callable] = {}
        self._register_default_operators()

        # Operator performance history (for weighting)
        self.operator_performance: Dict[str, List[float]] = {
            name: [] for name in self.operators
        }

        # Subjective time tracking
        self.tau_history: List[float] = []
        self.subjective_unit: float = 1.0  # U = median(v_t)

        # State history for crisis prediction
        self.z_history: List[np.ndarray] = []
        self.crisis_history: List[bool] = []

        # Current tree
        self.root: Optional[TreeNode] = None
        self.leaves: List[TreeNode] = []

        self.t = 0

    def _register_default_operators(self):
        """Register default internal operators."""

        def homeostasis(z: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Pull toward stable mean."""
            z_new = 0.9 * z + 0.1 * np.tanh(z)
            D_new = D.copy()
            D_new[3] *= 1.1  # Boost stability drive
            D_new = D_new / D_new.sum()
            return z_new, D_new

        def exploration(z: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Increase variance, explore new states."""
            z_new = z + np.random.randn(len(z)) * 0.2
            D_new = D.copy()
            D_new[0] *= 1.2  # Boost entropy drive
            D_new[2] *= 1.2  # Boost novelty drive
            D_new = D_new / D_new.sum()
            return z_new, D_new

        def integration(z: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Increase coherence between dimensions."""
            mean_z = np.mean(z)
            z_new = 0.8 * z + 0.2 * mean_z
            D_new = D.copy()
            D_new[4] *= 1.2  # Boost integration drive
            D_new = D_new / D_new.sum()
            return z_new, D_new

        def connection(z: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Focus on otherness/connection."""
            z_new = z.copy()
            z_new += np.random.randn(len(z)) * 0.1
            D_new = D.copy()
            D_new[5] *= 1.3  # Boost otherness drive
            D_new = D_new / D_new.sum()
            return z_new, D_new

        def consolidation(z: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Reduce variance, consolidate gains."""
            z_new = 0.95 * z
            D_new = D.copy()
            D_new[1] *= 1.1  # Reduce surprise sensitivity
            D_new[3] *= 1.1  # Boost stability
            D_new = D_new / D_new.sum()
            return z_new, D_new

        self.operators = {
            'homeostasis': homeostasis,
            'exploration': exploration,
            'integration': integration,
            'connection': connection,
            'consolidation': consolidation
        }

    def update_subjective_unit(self, tau: float):
        """
        Update subjective time unit.

        U = median({v_t}) where v_t = τ_t - τ_{t-1}
        """
        self.tau_history.append(tau)

        if len(self.tau_history) > 2:
            velocities = np.diff(self.tau_history)
            self.subjective_unit = float(np.median(velocities))

    def record_performance(self, operator_name: str, delta_metric: float):
        """Record performance of an operator for weighting."""
        if operator_name in self.operator_performance:
            self.operator_performance[operator_name].append(delta_metric)

            # Keep bounded
            max_hist = 100
            if len(self.operator_performance[operator_name]) > max_hist:
                self.operator_performance[operator_name] = \
                    self.operator_performance[operator_name][-max_hist:]

    def record_state(self, z: np.ndarray, in_crisis: bool):
        """Record state for crisis prediction."""
        self.z_history.append(z.copy())
        self.crisis_history.append(in_crisis)
        self.t += 1

        max_hist = 500
        if len(self.z_history) > max_hist:
            self.z_history = self.z_history[-max_hist:]
            self.crisis_history = self.crisis_history[-max_hist:]

    def _get_operator_weights(self) -> Dict[str, float]:
        """
        Get operator weights from historical performance.

        p_k(t) = rank(performance_k) / Σ rank(performance_j)
        """
        weights = {}

        for name, history in self.operator_performance.items():
            if len(history) > 0:
                mean_perf = np.mean(history)
            else:
                mean_perf = 0.0
            weights[name] = mean_perf

        # Convert to ranks
        all_perfs = list(weights.values())
        if len(all_perfs) == 0 or max(all_perfs) == min(all_perfs):
            # Uniform weights
            n = len(self.operators)
            return {name: 1.0 / n for name in self.operators}

        # Rank-based weights
        ranked = {}
        for name, perf in weights.items():
            rank = np.sum(np.array(all_perfs) <= perf) / len(all_perfs)
            ranked[name] = max(0.1, rank)  # Minimum weight

        total = sum(ranked.values())
        return {name: w / total for name, w in ranked.items()}

    def _predict_crisis_risk(self, z: np.ndarray) -> float:
        """
        Predict crisis risk for a state.

        R(h,j) = rank(prob_crisis(z^{(h,j)}))
        """
        if len(self.z_history) < 20:
            return 0.5

        # Find similar historical states
        distances = [np.linalg.norm(z - hz) for hz in self.z_history]
        sorted_idx = np.argsort(distances)

        # Look at k nearest neighbors
        k = min(10, len(self.z_history))
        nearest_idx = sorted_idx[:k]

        # Crisis rate in nearest neighbors
        crisis_rate = np.mean([self.crisis_history[i] for i in nearest_idx])

        return float(crisis_rate)

    def _compute_phi(self, z: np.ndarray) -> np.ndarray:
        """Compute phenomenological vector from structural state."""
        # Simplified phi computation
        phi = np.zeros(self.phi_dim)

        # Integration (coherence of z)
        phi[0] = 1.0 - np.std(z) / (np.mean(np.abs(z)) + 1e-8)

        # Temporal (momentum)
        if len(self.z_history) > 1:
            delta = z - self.z_history[-1]
            phi[1] = np.linalg.norm(delta)
        else:
            phi[1] = 0.0

        # Cross-modal (correlation structure)
        if len(z) > 2:
            phi[2] = np.abs(np.corrcoef(z[:len(z)//2], z[len(z)//2:])[0, 1])
            if np.isnan(phi[2]):
                phi[2] = 0.0

        # Modal diversity
        z_norm = np.abs(z) / (np.sum(np.abs(z)) + 1e-8)
        phi[3] = -np.sum(z_norm * np.log(z_norm + 1e-8))

        # Depth (recursion proxy)
        phi[4] = float(len(self.z_history)) / 100.0

        return phi

    def _compute_D(self, z: np.ndarray, D_prev: np.ndarray) -> np.ndarray:
        """Compute drives from structural state."""
        D = D_prev.copy()

        # Adjust based on z
        D[0] += np.std(z) * 0.1  # Entropy
        D[3] += (1 - np.std(z)) * 0.1  # Stability

        D = np.clip(D, 0.05, None)
        D = D / D.sum()
        return D

    def generate_tree(self, z_current: np.ndarray, D_current: np.ndarray,
                     depth: int = 3, branching: int = 3) -> TreeNode:
        """
        Generate temporal tree of possible futures.

        Args:
            z_current: Current structural state
            D_current: Current drives
            depth: Tree depth (H)
            branching: Branches per node

        Returns:
            Root node of tree
        """
        phi_current = self._compute_phi(z_current)

        self.root = TreeNode(
            depth=0,
            branch=0,
            z=z_current.copy(),
            D=D_current.copy(),
            phi=phi_current,
            operator_used='root'
        )

        self.leaves = []
        self._expand_node(self.root, depth, branching)

        # Evaluate all leaves
        self._evaluate_leaves()

        return self.root

    def _expand_node(self, node: TreeNode, remaining_depth: int, branching: int):
        """Recursively expand a node."""
        if remaining_depth == 0:
            self.leaves.append(node)
            return

        # Get operator weights
        weights = self._get_operator_weights()
        operator_names = list(self.operators.keys())

        # Select operators for branches
        probs = [weights[name] for name in operator_names]
        probs = np.array(probs) / sum(probs)

        selected = np.random.choice(
            operator_names,
            size=min(branching, len(operator_names)),
            replace=False,
            p=probs
        )

        for i, op_name in enumerate(selected):
            operator = self.operators[op_name]

            # Apply operator
            z_new, D_new = operator(node.z, node.D)
            phi_new = self._compute_phi(z_new)

            child = TreeNode(
                depth=node.depth + 1,
                branch=i,
                z=z_new,
                D=D_new,
                phi=phi_new,
                operator_used=op_name,
                parent=node
            )

            # Compute deltas
            child.delta_D = D_new - node.D
            child.delta_phi = phi_new - node.phi

            # Recursively expand
            self._expand_node(child, remaining_depth - 1, branching)

    def _evaluate_leaves(self):
        """
        Evaluate all leaf nodes.

        V(h,j) = rank(-||ΔD||) + rank(-R) + rank(||Δφ||)
        """
        if len(self.leaves) == 0:
            return

        # Collect metrics
        delta_D_norms = []
        risks = []
        delta_phi_norms = []

        for leaf in self.leaves:
            # Compute deltas from root
            delta_D = leaf.D - self.root.D
            delta_phi = leaf.phi - self.root.phi

            leaf.delta_D = delta_D
            leaf.delta_phi = delta_phi
            leaf.crisis_risk = self._predict_crisis_risk(leaf.z)

            delta_D_norms.append(np.linalg.norm(delta_D))
            risks.append(leaf.crisis_risk)
            delta_phi_norms.append(np.linalg.norm(delta_phi))

        # Compute ranks and values
        values = []
        for i, leaf in enumerate(self.leaves):
            # Lower delta_D is better (stability)
            rank_D = 1 - np.sum(np.array(delta_D_norms) <= delta_D_norms[i]) / len(delta_D_norms)

            # Lower risk is better
            rank_R = 1 - np.sum(np.array(risks) <= risks[i]) / len(risks)

            # Higher delta_phi is better (growth)
            rank_phi = np.sum(np.array(delta_phi_norms) <= delta_phi_norms[i]) / len(delta_phi_norms)

            value = rank_D + rank_R + rank_phi
            leaf.value = value
            values.append(value)

        # Compute probabilities via endogenous softmax
        values = np.array(values)
        sigma_V = np.std(values) + 1e-8
        beta = 1.0 / (sigma_V + 1)

        exp_values = np.exp(beta * values)
        probs = exp_values / np.sum(exp_values)

        for i, leaf in enumerate(self.leaves):
            leaf.probability = float(probs[i])

    def get_best_branch(self) -> Optional[TreeNode]:
        """Get the leaf with highest probability."""
        if len(self.leaves) == 0:
            return None
        return max(self.leaves, key=lambda x: x.probability)

    def get_branch_distribution(self) -> Dict[str, float]:
        """Get probability distribution over operators at first level."""
        if self.root is None:
            return {}

        dist = {}
        for leaf in self.leaves:
            # Trace back to first operator
            node = leaf
            while node.parent is not None and node.parent.parent is not None:
                node = node.parent

            op = node.operator_used
            if op not in dist:
                dist[op] = 0.0
            dist[op] += leaf.probability

        return dist

    def get_statistics(self) -> Dict:
        """Get tree statistics."""
        if self.root is None or len(self.leaves) == 0:
            return {'status': 'no_tree'}

        best = self.get_best_branch()

        return {
            'n_leaves': len(self.leaves),
            'subjective_unit': float(self.subjective_unit),
            'best_operator': best.operator_used if best else None,
            'best_probability': float(best.probability) if best else 0.0,
            'best_value': float(best.value) if best else 0.0,
            'mean_crisis_risk': float(np.mean([l.crisis_risk for l in self.leaves])),
            'operator_distribution': self.get_branch_distribution()
        }


def test_temporal_tree():
    """Test temporal tree."""
    print("=" * 60)
    print("TEMPORAL TREE TEST")
    print("=" * 60)

    tree = TemporalTree(z_dim=6, phi_dim=5, D_dim=6)

    # Simulate some history
    print("\nBuilding history...")
    z = np.random.randn(6) * 0.1
    for t in range(100):
        z = 0.95 * z + np.random.randn(6) * 0.05
        in_crisis = np.random.random() < 0.1
        tree.record_state(z, in_crisis)
        tree.update_subjective_unit(t * 1.1)

        # Random operator performance
        for op in tree.operators:
            tree.record_performance(op, np.random.randn() * 0.1)

    # Generate tree
    print("\nGenerating temporal tree...")
    z_current = np.random.randn(6) * 0.1
    D_current = np.abs(np.random.randn(6))
    D_current = D_current / D_current.sum()

    root = tree.generate_tree(z_current, D_current, depth=3, branching=3)

    stats = tree.get_statistics()
    print(f"\nTree Statistics:")
    print(f"  Leaves: {stats['n_leaves']}")
    print(f"  Best operator: {stats['best_operator']}")
    print(f"  Best probability: {stats['best_probability']:.3f}")
    print(f"  Mean crisis risk: {stats['mean_crisis_risk']:.3f}")
    print(f"  Operator distribution: {stats['operator_distribution']}")

    return tree


if __name__ == "__main__":
    test_temporal_tree()
