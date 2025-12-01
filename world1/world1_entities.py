"""
WORLD-1 Entities

Internal entities without human semantics.
Entities have positions, internal states, and interact structurally.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Entity:
    """
    An internal entity in WORLD-1.

    No semantic meaning (not "creature", "object", etc.)
    Just structural state that evolves.
    """
    idx: int                    # Unique identifier
    position: np.ndarray        # Position in internal space
    internal_state: np.ndarray  # Internal degrees of freedom
    activity: float = 0.5       # Activity level [0, 1]
    age: int = 0                # Timesteps since creation

    # History for endogenous computations
    position_history: List[np.ndarray] = field(default_factory=list)
    state_history: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        if len(self.position_history) == 0:
            self.position_history.append(self.position.copy())
        if len(self.state_history) == 0:
            self.state_history.append(self.internal_state.copy())

    def update(self, new_position: np.ndarray, new_state: np.ndarray):
        """Update entity state and record history."""
        self.position = new_position
        self.internal_state = new_state
        self.age += 1

        # Bounded history
        max_hist = 100
        self.position_history.append(self.position.copy())
        self.state_history.append(self.internal_state.copy())

        if len(self.position_history) > max_hist:
            self.position_history = self.position_history[-max_hist:]
            self.state_history = self.state_history[-max_hist:]

    def get_velocity(self) -> np.ndarray:
        """Compute velocity from recent history."""
        if len(self.position_history) < 2:
            return np.zeros_like(self.position)
        return self.position_history[-1] - self.position_history[-2]

    def get_state_change(self) -> np.ndarray:
        """Compute internal state change."""
        if len(self.state_history) < 2:
            return np.zeros_like(self.internal_state)
        return self.state_history[-1] - self.state_history[-2]


class EntityPopulation:
    """
    Population of entities in WORLD-1.

    Manages creation, evolution, and interaction of entities.
    All parameters endogenous.
    """

    def __init__(self, n_entities: int = 5, position_dim: int = 3,
                 state_dim: int = 4):
        """
        Initialize entity population.

        Args:
            n_entities: Number of entities (structural parameter)
            position_dim: Dimension of position space
            state_dim: Dimension of internal state
        """
        self.n_entities = n_entities
        self.position_dim = position_dim
        self.state_dim = state_dim

        # Create entities
        self.entities: Dict[int, Entity] = {}
        for i in range(n_entities):
            self.entities[i] = Entity(
                idx=i,
                position=np.random.rand(position_dim),
                internal_state=np.random.randn(state_dim) * 0.1
            )

        self.t = 0

        # Interaction matrix (endogenously learned)
        self.interaction_strengths: np.ndarray = np.zeros((n_entities, n_entities))

        # History for metrics
        self.population_state_history: List[np.ndarray] = []

    def _compute_pairwise_distances(self) -> np.ndarray:
        """Compute distances between all entity pairs."""
        distances = np.zeros((self.n_entities, self.n_entities))
        for i in range(self.n_entities):
            for j in range(i + 1, self.n_entities):
                dist = np.linalg.norm(
                    self.entities[i].position - self.entities[j].position
                )
                distances[i, j] = dist
                distances[j, i] = dist
        return distances

    def _compute_state_correlations(self) -> np.ndarray:
        """Compute state correlations between entities."""
        correlations = np.zeros((self.n_entities, self.n_entities))

        for i in range(self.n_entities):
            for j in range(i + 1, self.n_entities):
                # Use recent history
                hist_i = np.array(self.entities[i].state_history[-20:])
                hist_j = np.array(self.entities[j].state_history[-20:])

                if len(hist_i) > 2 and len(hist_j) > 2:
                    # Flatten and correlate
                    flat_i = hist_i.flatten()
                    flat_j = hist_j.flatten()
                    min_len = min(len(flat_i), len(flat_j))
                    if min_len > 1:
                        corr = np.corrcoef(flat_i[:min_len], flat_j[:min_len])[0, 1]
                        if not np.isnan(corr):
                            correlations[i, j] = corr
                            correlations[j, i] = corr

        return correlations

    def _update_interaction_strengths(self):
        """Update interaction strengths endogenously."""
        distances = self._compute_pairwise_distances()
        correlations = self._compute_state_correlations()

        # Interaction = inverse distance * correlation
        # Closer + correlated = stronger interaction
        for i in range(self.n_entities):
            for j in range(self.n_entities):
                if i != j:
                    dist_factor = 1.0 / (distances[i, j] + 0.1)
                    corr_factor = (correlations[i, j] + 1) / 2  # Map to [0, 1]

                    new_strength = dist_factor * corr_factor

                    # Smooth update with endogenous rate
                    alpha = 1.0 / np.sqrt(self.t + 1)
                    self.interaction_strengths[i, j] = (
                        (1 - alpha) * self.interaction_strengths[i, j] +
                        alpha * new_strength
                    )

    def step(self, external_field: Optional[np.ndarray] = None):
        """
        Advance all entities by one timestep.

        Args:
            external_field: Optional field affecting all entities
        """
        self.t += 1

        # Update interaction strengths
        if self.t > 5:
            self._update_interaction_strengths()

        # Compute forces/influences
        new_positions = {}
        new_states = {}

        for i, entity in self.entities.items():
            # Position dynamics: drift + interaction + noise
            drift = entity.get_velocity() * 0.9  # Momentum

            # Interaction with others
            interaction = np.zeros(self.position_dim)
            for j, other in self.entities.items():
                if i != j:
                    direction = other.position - entity.position
                    dist = np.linalg.norm(direction) + 1e-8
                    direction = direction / dist

                    # Attraction/repulsion based on interaction strength
                    strength = self.interaction_strengths[i, j]
                    interaction += strength * direction * 0.1

            # External field effect
            field_effect = np.zeros(self.position_dim)
            if external_field is not None and len(external_field) >= self.position_dim:
                field_effect = external_field[:self.position_dim] * 0.05

            # Noise (decreases with time)
            noise_scale = 0.1 / np.sqrt(self.t + 1)
            noise = np.random.randn(self.position_dim) * noise_scale

            new_pos = entity.position + drift + interaction + field_effect + noise
            new_pos = np.clip(new_pos, 0, 1)  # Bounded space
            new_positions[i] = new_pos

            # Internal state dynamics
            state_drift = entity.get_state_change() * 0.8

            # State coupling to position change
            pos_change = new_pos - entity.position
            state_coupling = np.zeros(self.state_dim)
            for d in range(min(self.position_dim, self.state_dim)):
                state_coupling[d] = pos_change[d] * 0.2

            state_noise = np.random.randn(self.state_dim) * noise_scale
            new_state = entity.internal_state + state_drift + state_coupling + state_noise
            new_state = np.tanh(new_state)  # Bounded
            new_states[i] = new_state

            # Update activity based on state magnitude
            entity.activity = float(np.mean(np.abs(new_state)))

        # Apply updates
        for i, entity in self.entities.items():
            entity.update(new_positions[i], new_states[i])

        # Record population state
        pop_state = self.get_population_vector()
        self.population_state_history.append(pop_state)
        if len(self.population_state_history) > 500:
            self.population_state_history = self.population_state_history[-500:]

    def get_population_vector(self) -> np.ndarray:
        """Get flattened vector of all entity states."""
        vectors = []
        for i in range(self.n_entities):
            entity = self.entities[i]
            vectors.append(entity.position)
            vectors.append(entity.internal_state)
            vectors.append(np.array([entity.activity]))
        return np.concatenate(vectors)

    def get_total_dimension(self) -> int:
        """Get total dimension of population vector."""
        return self.n_entities * (self.position_dim + self.state_dim + 1)

    def get_statistics(self) -> Dict:
        """Get population statistics."""
        positions = np.array([e.position for e in self.entities.values()])
        states = np.array([e.internal_state for e in self.entities.values()])
        activities = np.array([e.activity for e in self.entities.values()])

        # Clustering measure
        centroid = positions.mean(axis=0)
        dispersion = np.mean([np.linalg.norm(p - centroid) for p in positions])

        return {
            't': self.t,
            'n_entities': self.n_entities,
            'mean_activity': float(np.mean(activities)),
            'dispersion': float(dispersion),
            'state_variance': float(np.var(states)),
            'max_interaction': float(np.max(self.interaction_strengths)),
            'mean_interaction': float(np.mean(self.interaction_strengths))
        }


def test_entities():
    """Test entity population."""
    print("=" * 60)
    print("ENTITY POPULATION TEST")
    print("=" * 60)

    pop = EntityPopulation(n_entities=5, position_dim=3, state_dim=4)

    print(f"\nInitialized {pop.n_entities} entities")
    print(f"  Position dim: {pop.position_dim}")
    print(f"  State dim: {pop.state_dim}")
    print(f"  Total vector dim: {pop.get_total_dimension()}")

    for t in range(200):
        # Random external field
        field = np.random.randn(pop.position_dim) * 0.1 if t % 50 == 0 else None
        pop.step(field)

        if (t + 1) % 50 == 0:
            stats = pop.get_statistics()
            print(f"\n  t={t+1}:")
            print(f"    Mean activity: {stats['mean_activity']:.3f}")
            print(f"    Dispersion: {stats['dispersion']:.3f}")
            print(f"    Max interaction: {stats['max_interaction']:.3f}")

    return pop


if __name__ == "__main__":
    test_entities()
