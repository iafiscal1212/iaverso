"""
Narrative Memory System

Builds narrative structure from episodic memory.
Tracks episode transitions and finds dominant narrative chains.

All endogenous - no human semantic labels.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .episodic_memory import Episode, EpisodicMemory


@dataclass
class NarrativeChain:
    """A chain of episodes forming a narrative."""
    episodes: List[int]  # Episode indices
    probability: float   # Chain probability
    total_weight: float  # Sum of episode weights


class NarrativeMemory:
    """
    Narrative memory built on top of episodic memory.

    Constructs transition matrix between episodes.
    Finds dominant narrative chains.
    """

    def __init__(self, episodic_memory: EpisodicMemory):
        """
        Initialize narrative memory.

        Args:
            episodic_memory: Underlying episodic memory system
        """
        self.episodic = episodic_memory

        # Transition counts P[e -> e']
        self.transition_counts: Dict[Tuple[int, int], int] = {}
        self.outgoing_counts: Dict[int, int] = {}

        # Weighted transitions P̃[e -> e']
        self.weighted_transitions: Dict[Tuple[int, int], float] = {}

        # Narrative chains
        self.dominant_chains: List[NarrativeChain] = []

        # Last recorded episode for tracking transitions
        self.last_episode_idx: Optional[int] = None

    def update(self):
        """Update narrative structure from episodic memory."""
        episodes = self.episodic.episodes

        if len(episodes) < 2:
            return

        # Check for new episode
        current_last = episodes[-1].idx

        if self.last_episode_idx is not None and current_last > self.last_episode_idx:
            # Record transition from previous to current
            for prev_idx in range(self.last_episode_idx, current_last):
                if prev_idx + 1 < len(episodes):
                    self._record_transition(prev_idx, prev_idx + 1)

        self.last_episode_idx = current_last

        # Update weighted transitions
        self._update_weighted_transitions()

    def _record_transition(self, from_idx: int, to_idx: int):
        """Record a transition between episodes."""
        key = (from_idx, to_idx)

        if key not in self.transition_counts:
            self.transition_counts[key] = 0
        self.transition_counts[key] += 1

        if from_idx not in self.outgoing_counts:
            self.outgoing_counts[from_idx] = 0
        self.outgoing_counts[from_idx] += 1

    def get_transition_probability(self, from_idx: int, to_idx: int) -> float:
        """
        Get transition probability P[e -> e'].

        P_{e->e'} = freq(e followed by e') / Σ_k freq(e -> k)
        """
        key = (from_idx, to_idx)
        count = self.transition_counts.get(key, 0)
        total = self.outgoing_counts.get(from_idx, 0)

        if total == 0:
            return 0.0
        return count / total

    def _update_weighted_transitions(self):
        """
        Update weighted transition probabilities.

        P̃_{e->e'} = P_{e->e'} * rank(w_{e'})
        """
        episodes = self.episodic.episodes

        if len(episodes) < 2:
            return

        # Get all weights
        weights = [e.persistence_weight for e in episodes]

        for key, count in self.transition_counts.items():
            from_idx, to_idx = key

            # Base probability
            prob = self.get_transition_probability(from_idx, to_idx)

            # Weight rank of target episode
            if to_idx < len(episodes):
                target_weight = episodes[to_idx].persistence_weight
                weight_rank = np.sum(np.array(weights) <= target_weight) / len(weights)
            else:
                weight_rank = 0.5

            self.weighted_transitions[key] = prob * weight_rank

    def find_dominant_chain(self, max_length: int = 10) -> NarrativeChain:
        """
        Find dominant narrative chain.

        N = argmax_{(e_1,...,e_K)} Π_k P̃_{e_k -> e_{k+1}}

        Uses greedy search for efficiency.
        """
        episodes = self.episodic.episodes

        if len(episodes) < 2:
            return NarrativeChain(episodes=[], probability=0.0, total_weight=0.0)

        # Start from most important episode
        importances = [e.importance for e in episodes]
        start_idx = int(np.argmax(importances))

        chain = [start_idx]
        total_prob = 1.0
        total_weight = episodes[start_idx].persistence_weight

        current = start_idx

        for _ in range(max_length - 1):
            # Find best next episode
            best_next = None
            best_prob = 0.0

            for to_idx in range(len(episodes)):
                if to_idx in chain:
                    continue

                prob = self.weighted_transitions.get((current, to_idx), 0.0)
                if prob > best_prob:
                    best_prob = prob
                    best_next = to_idx

            if best_next is None or best_prob < 1e-8:
                break

            chain.append(best_next)
            total_prob *= best_prob
            total_weight += episodes[best_next].persistence_weight
            current = best_next

        return NarrativeChain(
            episodes=chain,
            probability=total_prob,
            total_weight=total_weight
        )

    def find_narrative_from(self, start_idx: int, length: int = 5) -> NarrativeChain:
        """Find narrative chain starting from specific episode."""
        episodes = self.episodic.episodes

        if start_idx >= len(episodes):
            return NarrativeChain(episodes=[], probability=0.0, total_weight=0.0)

        chain = [start_idx]
        total_prob = 1.0
        total_weight = episodes[start_idx].persistence_weight

        current = start_idx

        for _ in range(length - 1):
            # Find most likely next
            best_next = None
            best_prob = 0.0

            for (from_idx, to_idx), prob in self.weighted_transitions.items():
                if from_idx == current and to_idx not in chain:
                    if prob > best_prob:
                        best_prob = prob
                        best_next = to_idx

            if best_next is None:
                break

            chain.append(best_next)
            total_prob *= best_prob
            total_weight += episodes[best_next].persistence_weight
            current = best_next

        return NarrativeChain(
            episodes=chain,
            probability=total_prob,
            total_weight=total_weight
        )

    def get_episode_context(self, episode_idx: int) -> Dict:
        """Get narrative context for an episode."""
        episodes = self.episodic.episodes

        if episode_idx >= len(episodes):
            return {}

        # Predecessors
        predecessors = []
        for (from_idx, to_idx), prob in self.weighted_transitions.items():
            if to_idx == episode_idx:
                predecessors.append((from_idx, prob))

        # Successors
        successors = []
        for (from_idx, to_idx), prob in self.weighted_transitions.items():
            if from_idx == episode_idx:
                successors.append((to_idx, prob))

        # Sort by probability
        predecessors.sort(key=lambda x: x[1], reverse=True)
        successors.sort(key=lambda x: x[1], reverse=True)

        return {
            'episode_idx': episode_idx,
            'predecessors': predecessors[:3],
            'successors': successors[:3],
            'importance': episodes[episode_idx].importance,
            'weight': episodes[episode_idx].persistence_weight
        }

    def get_narrative_summary(self) -> Dict:
        """Get summary of narrative structure."""
        episodes = self.episodic.episodes

        if len(episodes) < 2:
            return {'status': 'insufficient_episodes'}

        # Find dominant chain
        dominant = self.find_dominant_chain()
        self.dominant_chains = [dominant]

        # Compute narrative density (how connected are episodes)
        n_transitions = len(self.transition_counts)
        max_transitions = len(episodes) * (len(episodes) - 1)
        density = n_transitions / max(1, max_transitions)

        return {
            'n_episodes': len(episodes),
            'n_transitions': n_transitions,
            'narrative_density': float(density),
            'dominant_chain_length': len(dominant.episodes),
            'dominant_chain_prob': float(dominant.probability),
            'dominant_chain': dominant.episodes
        }


def test_narrative_memory():
    """Test narrative memory system."""
    print("=" * 60)
    print("NARRATIVE MEMORY TEST")
    print("=" * 60)

    # Create episodic memory and populate
    episodic = EpisodicMemory(z_dim=6, phi_dim=5, D_dim=6)

    print("\nPopulating episodic memory...")

    z = np.random.randn(6) * 0.1
    phi = np.random.randn(5) * 0.1
    D = np.abs(np.random.randn(6))
    D = D / D.sum()
    tau = 0.0

    for t in range(400):
        z = 0.95 * z + np.random.randn(6) * 0.05
        phi = 0.9 * phi + np.random.randn(5) * 0.1

        if t % 60 == 30:
            z += np.random.randn(6) * 0.4
            phi += np.random.randn(5) * 0.3

        tau += 1.0 + 0.3 * np.linalg.norm(phi)
        episodic.record(z, phi, D, tau)

    print(f"Created {len(episodic.episodes)} episodes")

    # Create narrative memory
    narrative = NarrativeMemory(episodic)

    # Update multiple times to build transitions
    for _ in range(5):
        narrative.update()

    # Get summary
    summary = narrative.get_narrative_summary()
    print(f"\nNarrative Summary:")
    print(f"  Transitions: {summary['n_transitions']}")
    print(f"  Density: {summary['narrative_density']:.3f}")
    print(f"  Dominant chain: {summary['dominant_chain']}")

    # Get context for an episode
    if len(episodic.episodes) > 3:
        context = narrative.get_episode_context(2)
        print(f"\nContext for episode 2:")
        print(f"  Predecessors: {context['predecessors']}")
        print(f"  Successors: {context['successors']}")

    return narrative


if __name__ == "__main__":
    test_narrative_memory()
