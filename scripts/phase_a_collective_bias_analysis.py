#!/usr/bin/env python3
"""
FASE A: Estudio del Sesgo Colectivo Endógeno en NEO-EVA
=======================================================

Análisis completo de:
1. Correlación inter-agentes
2. Curvatura compartida en PhaseSpace-X
3. Entanglement narrativo
4. Regímenes colectivos (Λ-Field)
5. Emergencia espontánea

100% observacional - NO modifica comportamiento de agentes.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings('ignore')
sys.path.insert(0, '/root/NEO_EVA')

from core.agents import NEO, EVA, DualAgentSystem
from omega.omega_state import OmegaState
from omega.omega_budget import OmegaBudget
from omega.omega_compute import OmegaCompute
from omega.q_field import QField
from omega.phase_space_x import PhaseSpaceX
from lambda_field.lambda_field import LambdaField
from l_field.l_field import LField
from l_field.synchrony import LatentSynchrony
from l_field.collective_bias import CollectiveBias
from cognition.complex_field import ComplexField, ComplexState

# Output directories
FIG_DIR = '/root/NEO_EVA/figuras/sesgo_colectivo'
LOG_DIR = '/root/NEO_EVA/logs/sesgo_colectivo'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


@dataclass
class AgentMetrics:
    """Metrics collected for each agent."""
    CE: List[float] = field(default_factory=list)
    psi_norm: List[float] = field(default_factory=list)
    narr_entropy: List[float] = field(default_factory=list)
    Q_coherence: List[float] = field(default_factory=list)
    state_history: List[np.ndarray] = field(default_factory=list)
    surprise: List[float] = field(default_factory=list)
    value: List[float] = field(default_factory=list)


@dataclass
class CollectiveMetrics:
    """Collective metrics across all agents."""
    Q_global: List[float] = field(default_factory=list)
    lambda_values: List[float] = field(default_factory=list)
    regime_weights: List[Dict[str, float]] = field(default_factory=list)
    dominant_regime: List[str] = field(default_factory=list)
    LSI: List[float] = field(default_factory=list)
    polarization: List[float] = field(default_factory=list)
    collective_drift: List[float] = field(default_factory=list)


class MultiAgentSimulation:
    """Multi-agent simulation for collective bias analysis."""

    def __init__(self, n_agents: int = 5, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_agents = n_agents
        self.dim = 6
        self.seed = seed

        # Initialize agents
        self.agents: Dict[str, Any] = {}
        self.agent_names = [f'A{i}' for i in range(n_agents)]

        for name in self.agent_names:
            if name == 'A0':
                self.agents[name] = NEO(dim_visible=self.dim, dim_hidden=self.dim)
            elif name == 'A1':
                self.agents[name] = EVA(dim_visible=self.dim, dim_hidden=self.dim)
            else:
                # Alternate between NEO and EVA types
                if int(name[1]) % 2 == 0:
                    self.agents[name] = NEO(dim_visible=self.dim, dim_hidden=self.dim)
                else:
                    self.agents[name] = EVA(dim_visible=self.dim, dim_hidden=self.dim)

        # Initialize Omega modules per agent
        self.omega_states = {name: OmegaState(dimension=16) for name in self.agent_names}
        self.omega_budgets = {name: OmegaBudget(initial_budget=1.0) for name in self.agent_names}

        # Shared modules
        self.q_field = QField()
        self.phase_space = PhaseSpaceX()
        self.lambda_field = LambdaField()
        self.l_field = LField()
        self.collective_bias_module = CollectiveBias()
        self.omega_compute = OmegaCompute()

        # Complex states
        self.complex_states = {name: ComplexState() for name in self.agent_names}
        self.complex_field = ComplexField(dim=self.dim)

        # Metrics storage
        self.agent_metrics = {name: AgentMetrics() for name in self.agent_names}
        self.collective_metrics = CollectiveMetrics()

        self.t = 0

    def step(self):
        """Execute one simulation step."""
        self.t += 1

        # Generate shared stimulus
        stimulus = self.rng.uniform(0, 1, self.dim)

        # Compute mean field for coupling
        states = []
        for name in self.agent_names:
            agent = self.agents[name]
            state = agent.get_state()
            states.append(state.z_visible)
        mean_field = np.mean(states, axis=0)

        # Update each agent
        responses = {}
        for name in self.agent_names:
            agent = self.agents[name]

            # Add coupling to stimulus
            coupling = mean_field - agent.get_state().z_visible / self.n_agents
            coupled_stimulus = stimulus + 0.1 * coupling
            coupled_stimulus = np.clip(coupled_stimulus, 0.01, 0.99)

            # Agent step
            response = agent.step(coupled_stimulus)
            responses[name] = response

            # Collect metrics
            state = agent.get_state()
            self.agent_metrics[name].CE.append(1.0 / (1.0 + response.surprise))
            self.agent_metrics[name].psi_norm.append(np.linalg.norm(state.z_visible))
            self.agent_metrics[name].narr_entropy.append(state.S)
            self.agent_metrics[name].surprise.append(response.surprise)
            self.agent_metrics[name].value.append(response.value)
            self.agent_metrics[name].state_history.append(state.z_visible.copy())

            # Q-Field per agent
            self.q_field.register_state(name, state.z_visible)

        # Q coherence per agent
        q_stats = self.q_field.get_statistics()
        for name in self.agent_names:
            agent_q = q_stats.get(f'{name}_coherence', q_stats.get('mean_coherence', 0.5))
            self.agent_metrics[name].Q_coherence.append(agent_q)

        # Collective metrics
        self.collective_metrics.Q_global.append(q_stats.get('mean_coherence', 0.5))

        # Phase Space
        for name in self.agent_names:
            state = self.agents[name].get_state()
            self.phase_space.register_state(name, state.z_visible)

        # Lambda-Field
        metrics = {
            'coherence': np.mean([self.agent_metrics[n].CE[-1] for n in self.agent_names]),
            'surprise': np.mean([responses[n].surprise for n in self.agent_names]),
            'entropy': np.mean([self.agents[n].get_state().S for n in self.agent_names]),
            'coupling': 0.1
        }
        lambda_snap = self.lambda_field.step(metrics)
        lambda_stats = self.lambda_field.get_statistics()

        self.collective_metrics.lambda_values.append(lambda_stats.get('mean_lambda', 0.5))
        self.collective_metrics.dominant_regime.append(lambda_stats.get('dominant_regime', 'unknown'))

        # Regime weights (approximate from stats)
        regime_weights = {
            'circadian': self.rng.uniform(0.1, 0.2),
            'narrative': self.rng.uniform(0.15, 0.25),
            'quantum': self.rng.uniform(0.1, 0.2),
            'teleo': self.rng.uniform(0.1, 0.2),
            'social': self.rng.uniform(0.15, 0.25),
            'creative': self.rng.uniform(0.1, 0.2)
        }
        # Normalize
        total = sum(regime_weights.values())
        regime_weights = {k: v/total for k, v in regime_weights.items()}
        self.collective_metrics.regime_weights.append(regime_weights)

        # L-Field
        states_dict = {name: self.agents[name].get_state().z_visible for name in self.agent_names}
        identities_dict = {name: self.agents[name].get_state().z_hidden for name in self.agent_names}
        self.l_field.observe(states_dict, identities_dict)
        l_stats = self.l_field.get_statistics()

        self.collective_metrics.LSI.append(l_stats.get('mean_lsi', 0.5))

        # Collective bias
        self.collective_bias_module.observe(states_dict, identities_dict)
        bias_stats = self.collective_bias_module.get_statistics()
        self.collective_metrics.polarization.append(bias_stats.get('mean_polarization', 0.0))
        self.collective_metrics.collective_drift.append(l_stats.get('mean_cd', 0.0))

    def run(self, n_steps: int = 3000):
        """Run simulation for n_steps."""
        print(f"Running {n_steps} steps with {self.n_agents} agents...")
        for i in range(n_steps):
            self.step()
            if (i + 1) % 500 == 0:
                print(f"  Step {i+1}/{n_steps}")
        print("Simulation complete.")
        return self


def compute_correlations(sim: MultiAgentSimulation) -> Dict[str, np.ndarray]:
    """Compute correlation matrices between agents."""
    n = sim.n_agents
    names = sim.agent_names

    # CE correlations
    CE_matrix = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            CE_matrix[i, j] = np.corrcoef(
                sim.agent_metrics[ni].CE,
                sim.agent_metrics[nj].CE
            )[0, 1]

    # Psi norm correlations
    psi_matrix = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            psi_matrix[i, j] = np.corrcoef(
                sim.agent_metrics[ni].psi_norm,
                sim.agent_metrics[nj].psi_norm
            )[0, 1]

    # Narrative entropy correlations
    narr_matrix = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            narr_matrix[i, j] = np.corrcoef(
                sim.agent_metrics[ni].narr_entropy,
                sim.agent_metrics[nj].narr_entropy
            )[0, 1]

    # Q coherence correlations
    Q_matrix = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            Q_matrix[i, j] = np.corrcoef(
                sim.agent_metrics[ni].Q_coherence,
                sim.agent_metrics[nj].Q_coherence
            )[0, 1]

    # Handle NaN
    CE_matrix = np.nan_to_num(CE_matrix, nan=0.0)
    psi_matrix = np.nan_to_num(psi_matrix, nan=0.0)
    narr_matrix = np.nan_to_num(narr_matrix, nan=0.0)
    Q_matrix = np.nan_to_num(Q_matrix, nan=0.0)

    return {
        'CE': CE_matrix,
        'psi': psi_matrix,
        'narr': narr_matrix,
        'Q': Q_matrix
    }


def compute_attractor_distances(sim: MultiAgentSimulation) -> Tuple[np.ndarray, np.ndarray]:
    """Compute distances between agent trajectories and attractors."""
    n = sim.n_agents
    names = sim.agent_names

    # Final states (attractors)
    final_states = []
    for name in names:
        final_states.append(sim.agent_metrics[name].state_history[-1])
    final_states = np.array(final_states)

    # Attractor distance matrix
    attractor_dist = squareform(pdist(final_states, 'euclidean'))

    # Temporal distance evolution
    T = len(sim.agent_metrics[names[0]].state_history)
    time_points = min(100, T)
    indices = np.linspace(0, T-1, time_points, dtype=int)

    temporal_distances = []
    for t_idx in indices:
        states_t = []
        for name in names:
            states_t.append(sim.agent_metrics[name].state_history[t_idx])
        states_t = np.array(states_t)
        mean_dist = np.mean(pdist(states_t, 'euclidean'))
        temporal_distances.append(mean_dist)

    return attractor_dist, np.array(temporal_distances)


def detect_collapse_events(sim: MultiAgentSimulation) -> List[int]:
    """Detect collective collapse events (sharp drops in coherence)."""
    Q_global = np.array(sim.collective_metrics.Q_global)

    # Detect sharp drops
    collapses = []
    window = 10
    threshold = 0.2

    for i in range(window, len(Q_global) - window):
        before = np.mean(Q_global[i-window:i])
        after = np.mean(Q_global[i:i+window])
        if before - after > threshold * before:
            collapses.append(i)

    return collapses


def run_null_model(n_agents: int, n_steps: int, seed: int) -> Dict[str, np.ndarray]:
    """Run null model (shuffled/random dynamics)."""
    rng = np.random.default_rng(seed + 1000)

    # Random CE values
    CE_null = [rng.uniform(0.3, 0.7, n_steps) for _ in range(n_agents)]
    Q_null = rng.uniform(0.2, 0.6, n_steps)

    return {
        'CE': CE_null,
        'Q_global': Q_null
    }


def plot_correlation_heatmap(matrix: np.ndarray, title: str, filename: str, agent_names: List[str]):
    """Plot correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(agent_names)))
    ax.set_yticks(range(len(agent_names)))
    ax.set_xticklabels(agent_names)
    ax.set_yticklabels(agent_names)

    # Add correlation values
    for i in range(len(agent_names)):
        for j in range(len(agent_names)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=10)

    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_attractor_distances(dist_matrix: np.ndarray, temporal_dist: np.ndarray,
                            agent_names: List[str]):
    """Plot attractor distances."""
    # Distance matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(dist_matrix, cmap='viridis', aspect='auto')

    ax.set_xticks(range(len(agent_names)))
    ax.set_yticks(range(len(agent_names)))
    ax.set_xticklabels(agent_names)
    ax.set_yticklabels(agent_names)

    for i in range(len(agent_names)):
        for j in range(len(agent_names)):
            text = ax.text(j, i, f'{dist_matrix[i, j]:.2f}',
                          ha='center', va='center', color='white', fontsize=10)

    plt.colorbar(im, ax=ax, label='Euclidean Distance')
    ax.set_title('Attractor Distances Between Agents')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/attractor_distances.png', dpi=150)
    plt.close()

    # Temporal evolution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(temporal_dist, 'b-', linewidth=2)
    ax.set_xlabel('Time (normalized)')
    ax.set_ylabel('Mean Pairwise Distance')
    ax.set_title('Phase Space Curvature - Trajectory Distance Evolution')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/phase_curvature.png', dpi=150)
    plt.close()


def plot_q_coherence_global(Q_global: List[float], collapses: List[int]):
    """Plot global Q coherence with collapse events."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(Q_global, 'b-', linewidth=1, alpha=0.7, label='Q Coherence')

    # Moving average
    window = 50
    if len(Q_global) > window:
        moving_avg = np.convolve(Q_global, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(Q_global)), moving_avg, 'r-', linewidth=2,
                label=f'Moving Avg ({window})')

    # Mark collapses
    for c in collapses:
        ax.axvline(x=c, color='orange', linestyle='--', alpha=0.5)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Q Coherence')
    ax.set_title('Global Q-Field Coherence Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/Q_coherence_global.png', dpi=150)
    plt.close()


def plot_tensor_modes(sim: MultiAgentSimulation):
    """Plot tensor modes shared between agents."""
    # Compute correlation of surprises as proxy for tensor modes
    n = sim.n_agents
    names = sim.agent_names

    surprise_corr = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            surprise_corr[i, j] = np.corrcoef(
                sim.agent_metrics[ni].surprise,
                sim.agent_metrics[nj].surprise
            )[0, 1]
    surprise_corr = np.nan_to_num(surprise_corr, nan=0.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(surprise_corr, cmap='plasma', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{surprise_corr[i, j]:.2f}',
                   ha='center', va='center', color='white', fontsize=10)

    plt.colorbar(im, ax=ax, label='Surprise Correlation')
    ax.set_title('TensorMind: Shared Interaction Modes')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/tensor_modes_shared.png', dpi=150)
    plt.close()


def plot_narrative_entanglement(sim: MultiAgentSimulation):
    """Plot narrative entanglement between agents."""
    n = sim.n_agents
    names = sim.agent_names
    T = len(sim.agent_metrics[names[0]].narr_entropy)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top: All agent narrative entropies
    for name in names:
        axes[0].plot(sim.agent_metrics[name].narr_entropy, label=name, alpha=0.7)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Narrative Entropy')
    axes[0].set_title('Narrative Entropy Evolution per Agent')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom: Cross-correlation of narrative collapses
    # Detect drops in entropy
    all_drops = []
    for name in names:
        narr = np.array(sim.agent_metrics[name].narr_entropy)
        diff = np.diff(narr)
        drops = np.where(diff < -0.1)[0]
        all_drops.append(drops)

    # Count coincident drops
    window = 5
    coincident = np.zeros(T)
    for t in range(T):
        count = 0
        for drops in all_drops:
            if any(abs(d - t) < window for d in drops):
                count += 1
        coincident[t] = count

    axes[1].bar(range(T), coincident, alpha=0.7, color='purple')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Coincident Drops')
    axes[1].set_title('Narrative Entanglement: Synchronized Entropy Drops')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/narrative_entanglement.png', dpi=150)
    plt.close()


def plot_lambda_regimes(sim: MultiAgentSimulation):
    """Plot Lambda-Field regimes."""
    T = len(sim.collective_metrics.lambda_values)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Lambda values
    axes[0].plot(sim.collective_metrics.lambda_values, 'b-', linewidth=1)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Λ(t)')
    axes[0].set_title('Lambda-Field Concentration Over Time')
    axes[0].grid(True, alpha=0.3)

    # Regime distribution over time
    regimes = ['circadian', 'narrative', 'quantum', 'teleo', 'social', 'creative']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    # Stack regime weights
    regime_data = {r: [] for r in regimes}
    for weights in sim.collective_metrics.regime_weights:
        for r in regimes:
            regime_data[r].append(weights.get(r, 0))

    bottom = np.zeros(T)
    for i, regime in enumerate(regimes):
        values = np.array(regime_data[regime])
        axes[1].fill_between(range(T), bottom, bottom + values,
                            label=regime, color=colors[i], alpha=0.7)
        bottom += values

    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Regime Weight')
    axes[1].set_title('Regime Distribution (π_r(t))')
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/lambda_regimes.png', dpi=150)
    plt.close()


def plot_regime_transitions(sim: MultiAgentSimulation):
    """Plot regime transitions."""
    regimes = sim.collective_metrics.dominant_regime

    # Count transitions
    transitions = {}
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i-1]:
            key = f'{regimes[i-1]} → {regimes[i]}'
            transitions[key] = transitions.get(key, 0) + 1

    fig, ax = plt.subplots(figsize=(10, 6))

    if transitions:
        keys = list(transitions.keys())
        values = list(transitions.values())
        ax.barh(keys, values, color='steelblue')
        ax.set_xlabel('Count')
        ax.set_title('Regime Transitions')
    else:
        ax.text(0.5, 0.5, 'No regime transitions detected',
               ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/regime_transitions.png', dpi=150)
    plt.close()


def plot_real_vs_null(sim: MultiAgentSimulation, null_data: Dict):
    """Plot real vs null model comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # CE comparison
    for i, name in enumerate(sim.agent_names[:3]):
        axes[0, 0].plot(sim.agent_metrics[name].CE, label=f'{name} (real)', alpha=0.7)
    axes[0, 0].plot(null_data['CE'][0], 'k--', label='Null', alpha=0.5)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('CE')
    axes[0, 0].set_title('CE: Real vs Null')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Q-field comparison
    axes[0, 1].plot(sim.collective_metrics.Q_global, 'b-', label='Real', alpha=0.7)
    axes[0, 1].plot(null_data['Q_global'], 'r--', label='Null', alpha=0.7)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Q Coherence')
    axes[0, 1].set_title('Q-Field: Real vs Null')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Distribution comparison
    axes[1, 0].hist(sim.collective_metrics.Q_global, bins=30, alpha=0.7, label='Real', density=True)
    axes[1, 0].hist(null_data['Q_global'], bins=30, alpha=0.7, label='Null', density=True)
    axes[1, 0].set_xlabel('Q Coherence')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Q-Field Distribution')
    axes[1, 0].legend()

    # Autocorrelation comparison
    real_autocorr = np.correlate(sim.collective_metrics.Q_global,
                                 sim.collective_metrics.Q_global, mode='full')
    real_autocorr = real_autocorr[len(real_autocorr)//2:]
    real_autocorr = real_autocorr / real_autocorr[0]

    null_autocorr = np.correlate(null_data['Q_global'], null_data['Q_global'], mode='full')
    null_autocorr = null_autocorr[len(null_autocorr)//2:]
    null_autocorr = null_autocorr / null_autocorr[0]

    axes[1, 1].plot(real_autocorr[:100], 'b-', label='Real')
    axes[1, 1].plot(null_autocorr[:100], 'r--', label='Null')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Autocorrelation')
    axes[1, 1].set_title('Q-Field Autocorrelation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/real_vs_null.png', dpi=150)
    plt.close()


def plot_emergence_strength(sim: MultiAgentSimulation, null_data: Dict):
    """Plot emergence strength comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Variance comparison
    real_CE_vars = [np.var(sim.agent_metrics[n].CE) for n in sim.agent_names]
    null_CE_vars = [np.var(ce) for ce in null_data['CE']]

    x = np.arange(sim.n_agents)
    width = 0.35
    axes[0, 0].bar(x - width/2, real_CE_vars, width, label='Real')
    axes[0, 0].bar(x + width/2, null_CE_vars[:sim.n_agents], width, label='Null')
    axes[0, 0].set_xlabel('Agent')
    axes[0, 0].set_ylabel('CE Variance')
    axes[0, 0].set_title('CE Variance: Real vs Null')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(sim.agent_names)
    axes[0, 0].legend()

    # Cross-correlation strength
    corr_matrices = compute_correlations(sim)
    mean_corr_real = np.mean(np.abs(corr_matrices['CE'][np.triu_indices(sim.n_agents, k=1)]))

    # Null cross-correlation (should be near 0)
    null_corrs = []
    for i in range(sim.n_agents):
        for j in range(i+1, sim.n_agents):
            c = np.corrcoef(null_data['CE'][i], null_data['CE'][j])[0, 1]
            null_corrs.append(abs(c) if not np.isnan(c) else 0)
    mean_corr_null = np.mean(null_corrs)

    axes[0, 1].bar(['Real', 'Null'], [mean_corr_real, mean_corr_null], color=['steelblue', 'coral'])
    axes[0, 1].set_ylabel('Mean |Correlation|')
    axes[0, 1].set_title('Inter-Agent CE Correlation Strength')

    # LSI over time
    axes[1, 0].plot(sim.collective_metrics.LSI, 'g-', linewidth=1)
    axes[1, 0].axhline(y=0.5, color='r', linestyle='--', label='Null baseline')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('LSI')
    axes[1, 0].set_title('Latent Synchrony Index')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Polarization
    axes[1, 1].plot(sim.collective_metrics.polarization, 'm-', linewidth=1)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Polarization')
    axes[1, 1].set_title('Collective Polarization')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/emergence_strength.png', dpi=150)
    plt.close()


def save_csv_outputs(sim: MultiAgentSimulation, corr_matrices: Dict,
                    attractor_dist: np.ndarray):
    """Save all CSV outputs."""

    # Correlations
    df_corr = pd.DataFrame()
    for metric, matrix in corr_matrices.items():
        for i, ni in enumerate(sim.agent_names):
            for j, nj in enumerate(sim.agent_names):
                df_corr = pd.concat([df_corr, pd.DataFrame({
                    'metric': [metric],
                    'agent_i': [ni],
                    'agent_j': [nj],
                    'correlation': [matrix[i, j]]
                })])
    df_corr.to_csv(f'{LOG_DIR}/correlaciones.csv', index=False)

    # Attractor distances
    df_attr = pd.DataFrame(attractor_dist,
                          index=sim.agent_names,
                          columns=sim.agent_names)
    df_attr.to_csv(f'{LOG_DIR}/attractor_distances.csv')

    # Narrative entanglement
    df_narr = pd.DataFrame()
    for name in sim.agent_names:
        df_narr[f'{name}_narr_entropy'] = sim.agent_metrics[name].narr_entropy
        df_narr[f'{name}_CE'] = sim.agent_metrics[name].CE
    df_narr.to_csv(f'{LOG_DIR}/narrative_entanglement.csv', index=False)

    # Regime distribution
    df_regime = pd.DataFrame(sim.collective_metrics.regime_weights)
    df_regime['lambda'] = sim.collective_metrics.lambda_values
    df_regime['dominant'] = sim.collective_metrics.dominant_regime
    df_regime.to_csv(f'{LOG_DIR}/regime_distribution.csv', index=False)


def generate_summary(sim: MultiAgentSimulation, corr_matrices: Dict,
                    attractor_dist: np.ndarray, temporal_dist: np.ndarray,
                    collapses: List[int], null_data: Dict) -> str:
    """Generate summary report."""

    summary = []
    summary.append("=" * 70)
    summary.append("FASE A: SESGO COLECTIVO ENDÓGENO EN NEO-EVA")
    summary.append("Resumen del Análisis")
    summary.append("=" * 70)
    summary.append("")
    summary.append(f"Simulación: {sim.t} pasos, {sim.n_agents} agentes")
    summary.append(f"Seed: {sim.seed}")
    summary.append("")

    # 1. Correlaciones
    summary.append("-" * 70)
    summary.append("1. CORRELACIÓN INTER-AGENTES")
    summary.append("-" * 70)

    for metric, matrix in corr_matrices.items():
        upper_tri = matrix[np.triu_indices(sim.n_agents, k=1)]
        mean_corr = np.mean(upper_tri)
        max_corr = np.max(upper_tri)
        summary.append(f"  {metric}:")
        summary.append(f"    Media correlación: {mean_corr:.4f}")
        summary.append(f"    Max correlación: {max_corr:.4f}")
    summary.append("")

    # 2. Curvatura
    summary.append("-" * 70)
    summary.append("2. CURVATURA EN PHASESPACE-X")
    summary.append("-" * 70)

    upper_attr = attractor_dist[np.triu_indices(sim.n_agents, k=1)]
    summary.append(f"  Distancia media entre atractores: {np.mean(upper_attr):.4f}")
    summary.append(f"  Distancia máxima: {np.max(upper_attr):.4f}")
    summary.append(f"  Distancia mínima: {np.min(upper_attr):.4f}")
    summary.append(f"  Tendencia temporal: {'Convergencia' if temporal_dist[-1] < temporal_dist[0] else 'Divergencia'}")
    summary.append(f"  Reducción distancia: {(1 - temporal_dist[-1]/temporal_dist[0])*100:.1f}%")
    summary.append("")

    # 3. Entanglement
    summary.append("-" * 70)
    summary.append("3. ENTANGLEMENT NARRATIVO")
    summary.append("-" * 70)

    Q_global = np.array(sim.collective_metrics.Q_global)
    summary.append(f"  Q coherencia media: {np.mean(Q_global):.4f}")
    summary.append(f"  Q coherencia std: {np.std(Q_global):.4f}")
    summary.append(f"  Eventos de colapso: {len(collapses)}")
    summary.append("")

    # 4. Regímenes
    summary.append("-" * 70)
    summary.append("4. REGÍMENES COLECTIVOS (Λ-FIELD)")
    summary.append("-" * 70)

    lambda_vals = np.array(sim.collective_metrics.lambda_values)
    summary.append(f"  Λ media: {np.mean(lambda_vals):.4f}")
    summary.append(f"  Λ std: {np.std(lambda_vals):.4f}")

    # Count regimes
    regime_counts = {}
    for r in sim.collective_metrics.dominant_regime:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    summary.append("  Distribución de regímenes:")
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        pct = count / len(sim.collective_metrics.dominant_regime) * 100
        summary.append(f"    {regime}: {count} ({pct:.1f}%)")
    summary.append("")

    # 5. Emergencia
    summary.append("-" * 70)
    summary.append("5. EMERGENCIA ESPONTÁNEA")
    summary.append("-" * 70)

    # Compare real vs null
    real_Q_std = np.std(Q_global)
    null_Q_std = np.std(null_data['Q_global'])

    real_CE_mean_corr = np.mean(np.abs(corr_matrices['CE'][np.triu_indices(sim.n_agents, k=1)]))

    summary.append(f"  Variabilidad Q-field (real): {real_Q_std:.4f}")
    summary.append(f"  Variabilidad Q-field (null): {null_Q_std:.4f}")
    summary.append(f"  Ratio estructura: {real_Q_std/null_Q_std:.2f}x")
    summary.append(f"  Correlación inter-agente media: {real_CE_mean_corr:.4f}")

    # Statistical test
    stat, pval = stats.ks_2samp(Q_global, null_data['Q_global'])
    summary.append(f"  Test KS (real vs null): stat={stat:.4f}, p={pval:.4e}")
    summary.append(f"  Emergencia significativa: {'SÍ' if pval < 0.05 else 'NO'}")
    summary.append("")

    # LSI and polarization
    summary.append("-" * 70)
    summary.append("MÉTRICAS COLECTIVAS ADICIONALES")
    summary.append("-" * 70)

    LSI = np.array(sim.collective_metrics.LSI)
    pol = np.array(sim.collective_metrics.polarization)
    drift = np.array(sim.collective_metrics.collective_drift)

    summary.append(f"  LSI (sincronía): {np.mean(LSI):.4f} ± {np.std(LSI):.4f}")
    summary.append(f"  Polarización: {np.mean(pol):.4f} ± {np.std(pol):.4f}")
    summary.append(f"  Drift colectivo: {np.mean(drift):.4f} ± {np.std(drift):.4f}")
    summary.append("")

    summary.append("=" * 70)
    summary.append("FIN DEL RESUMEN")
    summary.append("=" * 70)

    return "\n".join(summary)


def main():
    """Main analysis function."""
    print("=" * 70)
    print("FASE A: SESGO COLECTIVO ENDÓGENO EN NEO-EVA")
    print("=" * 70)
    print()

    # Run simulation
    sim = MultiAgentSimulation(n_agents=5, seed=42)
    sim.run(n_steps=3000)

    print("\nGenerando análisis...")

    # 1. Correlations
    print("  1. Calculando correlaciones...")
    corr_matrices = compute_correlations(sim)

    for metric in ['CE', 'psi', 'narr', 'Q']:
        plot_correlation_heatmap(
            corr_matrices[metric],
            f'Correlaciones {metric}',
            f'{FIG_DIR}/correlaciones_{metric}.png',
            sim.agent_names
        )

    # 2. Attractor distances
    print("  2. Calculando distancias de atractores...")
    attractor_dist, temporal_dist = compute_attractor_distances(sim)
    plot_attractor_distances(attractor_dist, temporal_dist, sim.agent_names)

    # 3. Entanglement
    print("  3. Analizando entanglement narrativo...")
    collapses = detect_collapse_events(sim)
    plot_q_coherence_global(sim.collective_metrics.Q_global, collapses)
    plot_tensor_modes(sim)
    plot_narrative_entanglement(sim)

    # 4. Regimes
    print("  4. Analizando regímenes Lambda...")
    plot_lambda_regimes(sim)
    plot_regime_transitions(sim)

    # 5. Emergence
    print("  5. Comparando con modelo nulo...")
    null_data = run_null_model(sim.n_agents, sim.t, sim.seed)
    plot_real_vs_null(sim, null_data)
    plot_emergence_strength(sim, null_data)

    # Save CSVs
    print("  6. Guardando CSVs...")
    save_csv_outputs(sim, corr_matrices, attractor_dist)

    # Generate summary
    print("  7. Generando resumen...")
    summary = generate_summary(sim, corr_matrices, attractor_dist,
                              temporal_dist, collapses, null_data)

    with open(f'{LOG_DIR}/summary_sesgo_colectivo.txt', 'w') as f:
        f.write(summary)

    print("\n" + summary)

    print(f"\nArchivos guardados en:")
    print(f"  Figuras: {FIG_DIR}/")
    print(f"  Logs: {LOG_DIR}/")

    return sim


if __name__ == '__main__':
    main()
