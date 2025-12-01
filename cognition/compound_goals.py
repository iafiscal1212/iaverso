"""
Compound Goals and Planning

Goals emerge from successful drive combinations.
Planning uses temporal tree to approach goals.

All endogenous - no human-defined objectives.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy import linalg


@dataclass
class CompoundGoal:
    """A compound goal learned from successful episodes."""
    idx: int
    center: np.ndarray      # Center of cluster in drive space
    frequency: float        # How often it appears in success
    effectiveness: float    # Average SAGI when achieved


class CompoundGoals:
    """
    Discovers compound goals from drive patterns in successful episodes.

    G_k = C_k (cluster center of successful drive combinations)
    """

    def __init__(self, D_dim: int = 6):
        """
        Initialize compound goal system.

        Args:
            D_dim: Dimension of drive vector
        """
        self.D_dim = D_dim

        # Episode drive patterns
        self.episode_drives: List[np.ndarray] = []
        self.episode_metrics: List[float] = []  # SAGI or success metric

        # Discovered goals
        self.goals: List[CompoundGoal] = []
        self.n_goals: int = 0

        # For clustering
        self.goal_centers: Optional[np.ndarray] = None

        self.t = 0

    def record_episode(self, D_bar: np.ndarray, metric: float):
        """
        Record drive pattern from an episode.

        Args:
            D_bar: Mean drives during episode
            metric: Success metric (e.g., SAGI)
        """
        self.episode_drives.append(D_bar.copy())
        self.episode_metrics.append(metric)
        self.t += 1

        max_hist = 500
        if len(self.episode_drives) > max_hist:
            self.episode_drives = self.episode_drives[-max_hist:]
            self.episode_metrics = self.episode_metrics[-max_hist:]

    def _identify_successful_episodes(self) -> List[int]:
        """
        Identify successful episodes endogenously.

        success(e) = I[SAGI_e >= percentile_75({SAGI_j})]
        """
        if len(self.episode_metrics) < 10:
            return list(range(len(self.episode_metrics)))

        threshold = np.percentile(self.episode_metrics, 75)
        return [i for i, m in enumerate(self.episode_metrics) if m >= threshold]

    def discover_goals(self):
        """
        Discover compound goals from successful episodes.

        Uses clustering on drive patterns of successful episodes.
        k = d_D (effective dimension)
        """
        successful_idx = self._identify_successful_episodes()

        if len(successful_idx) < 5:
            return

        # Get drives from successful episodes
        D_success = np.array([self.episode_drives[i] for i in successful_idx])

        # Compute covariance
        Sigma_D = np.cov(D_success.T)

        try:
            eigenvalues, eigenvectors = linalg.eigh(Sigma_D)
            eigenvalues = eigenvalues[::-1]
            eigenvectors = eigenvectors[:, ::-1]

            # Effective dimension
            median_lambda = np.median(eigenvalues[eigenvalues > 0])
            d_D = max(2, min(5, np.sum(eigenvalues >= median_lambda)))
            self.n_goals = d_D

        except:
            self.n_goals = 3

        # Simple k-means clustering
        self.goal_centers = self._simple_kmeans(D_success, self.n_goals)

        # Create goal objects
        self.goals = []
        for k, center in enumerate(self.goal_centers):
            # Compute frequency and effectiveness
            distances = np.array([np.linalg.norm(d - center) for d in D_success])
            threshold = np.median(distances)
            near_idx = distances < threshold

            frequency = np.sum(near_idx) / len(D_success)
            effectiveness = np.mean([self.episode_metrics[successful_idx[i]]
                                    for i, is_near in enumerate(near_idx) if is_near])

            goal = CompoundGoal(
                idx=k,
                center=center,
                frequency=frequency,
                effectiveness=effectiveness
            )
            self.goals.append(goal)

    def _simple_kmeans(self, data: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
        """Simple k-means clustering."""
        n = len(data)
        if n < k:
            return data[:k] if len(data) >= k else np.zeros((k, self.D_dim))

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

    def get_nearest_goal(self, D: np.ndarray) -> Optional[CompoundGoal]:
        """Get the goal nearest to current drives."""
        if len(self.goals) == 0:
            return None

        distances = [np.linalg.norm(D - g.center) for g in self.goals]
        return self.goals[np.argmin(distances)]

    def distance_to_goal(self, D: np.ndarray, goal: CompoundGoal) -> float:
        """Compute distance from current drives to goal."""
        return float(np.linalg.norm(D - goal.center))

    def get_statistics(self) -> Dict:
        """Get goal statistics."""
        if len(self.goals) == 0:
            return {'n_goals': 0}

        return {
            'n_goals': len(self.goals),
            'goal_effectiveness': [g.effectiveness for g in self.goals],
            'goal_frequency': [g.frequency for g in self.goals]
        }


class GoalPlanner:
    """
    Plans toward compound goals using temporal tree.

    Integrates goal distance with branch value for planning.
    """

    def __init__(self, goal_system: CompoundGoals):
        """
        Initialize goal planner.

        Args:
            goal_system: Compound goals system
        """
        self.goals = goal_system
        self.t = 0

        # Planning history
        self.plan_history: List[Dict] = []

    def evaluate_branch(self, D_branch: np.ndarray, V_branch: float) -> float:
        """
        Evaluate a branch considering both value and goal proximity.

        U(h,j) = α_t * rank(V) + (1-α_t) * rank(P)
        P(h,j) = -min_k rank(δ_k)

        where α_t = 1/√(t+1) (more planning over time)
        """
        self.t += 1

        # α decreases over time (more goal-directed)
        alpha = 1.0 / np.sqrt(self.t + 1)

        # Value component (exploration/immediate utility)
        V_component = V_branch

        # Planning component (goal proximity)
        if len(self.goals.goals) == 0:
            P_component = 0.5
        else:
            distances = [self.goals.distance_to_goal(D_branch, g)
                        for g in self.goals.goals]
            min_dist = min(distances)

            # Rank of minimum distance (lower is better)
            all_dists = [self.goals.distance_to_goal(D_branch, g)
                        for g in self.goals.goals]
            rank_dist = np.sum(np.array(all_dists) <= min_dist) / len(all_dists)
            P_component = 1 - rank_dist  # Higher is better

        U = alpha * V_component + (1 - alpha) * P_component
        return float(U)

    def select_best_action(self, branches: List[Tuple[str, np.ndarray, float]]) -> str:
        """
        Select best action from branches.

        Args:
            branches: List of (operator_name, D_result, V_value)

        Returns:
            Best operator name
        """
        if len(branches) == 0:
            return 'exploration'

        utilities = []
        for op_name, D_branch, V_branch in branches:
            U = self.evaluate_branch(D_branch, V_branch)
            utilities.append((op_name, U))

        best = max(utilities, key=lambda x: x[1])

        # Record
        self.plan_history.append({
            't': self.t,
            'selected': best[0],
            'utility': best[1],
            'alpha': 1.0 / np.sqrt(self.t + 1)
        })

        return best[0]

    def get_current_goal(self, D: np.ndarray) -> Optional[CompoundGoal]:
        """Get the goal currently being pursued."""
        return self.goals.get_nearest_goal(D)

    def get_statistics(self) -> Dict:
        """Get planner statistics."""
        if len(self.plan_history) == 0:
            return {'status': 'no_plans'}

        recent = self.plan_history[-20:]

        return {
            't': self.t,
            'total_plans': len(self.plan_history),
            'current_alpha': 1.0 / np.sqrt(self.t + 1),
            'recent_mean_utility': float(np.mean([p['utility'] for p in recent])),
            'action_distribution': self._get_action_distribution(recent)
        }

    def _get_action_distribution(self, plans: List[Dict]) -> Dict[str, float]:
        """Get distribution of actions in plans."""
        dist = {}
        for p in plans:
            op = p['selected']
            if op not in dist:
                dist[op] = 0
            dist[op] += 1

        total = sum(dist.values())
        return {k: v / total for k, v in dist.items()}


def test_compound_goals():
    """Test compound goals and planning."""
    print("=" * 60)
    print("COMPOUND GOALS AND PLANNING TEST")
    print("=" * 60)

    D_dim = 6
    goal_system = CompoundGoals(D_dim)

    print("\nSimulating 100 episodes...")

    # Simulate episodes
    for e in range(100):
        # Random drive pattern
        D = np.abs(np.random.randn(D_dim))
        D = D / D.sum()

        # Success metric (higher for certain patterns)
        metric = np.random.random()
        if D[3] > 0.2:  # Stability helps
            metric += 0.3
        if D[4] > 0.2:  # Integration helps
            metric += 0.2

        goal_system.record_episode(D, metric)

    # Discover goals
    print("\nDiscovering compound goals...")
    goal_system.discover_goals()

    stats = goal_system.get_statistics()
    print(f"  Discovered {stats['n_goals']} goals")
    for i, g in enumerate(goal_system.goals):
        print(f"    Goal {i}: effectiveness={g.effectiveness:.3f}, "
              f"frequency={g.frequency:.3f}")

    # Test planner
    print("\nTesting planner...")
    planner = GoalPlanner(goal_system)

    for t in range(50):
        # Simulate branches
        branches = []
        for op in ['homeostasis', 'exploration', 'integration']:
            D = np.abs(np.random.randn(D_dim))
            D = D / D.sum()
            V = np.random.random()
            branches.append((op, D, V))

        action = planner.select_best_action(branches)

    planner_stats = planner.get_statistics()
    print(f"\nPlanner Statistics:")
    print(f"  Total plans: {planner_stats['total_plans']}")
    print(f"  Current alpha: {planner_stats['current_alpha']:.3f}")
    print(f"  Action distribution: {planner_stats['action_distribution']}")

    return goal_system, planner


if __name__ == "__main__":
    test_compound_goals()
