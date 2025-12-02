#!/usr/bin/env python3
"""
FASE D: Modelos Nulos + Réplicas para Sesgo Colectivo
=====================================================

Análisis comparativo:
D1. NULL 1 - Sin acoplamiento
D2. NULL 2 - Intercambio roto
D3. NULL 3 - Shuffled history
D4. Réplicas multi-seed (5 seeds)
D5. Comparativa final Real vs Null

100% endógeno - Solo se modifica acoplamiento/intercambio.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
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

# Base output directories
BASE_FIG_DIR = '/root/NEO_EVA/figuras/sesgo_D'
BASE_LOG_DIR = '/root/NEO_EVA/logs/sesgo_D'

# Sub-directories
DIRS = {
    'null_no_coupling': (f'{BASE_FIG_DIR}/null_no_coupling', f'{BASE_LOG_DIR}/null_no_coupling'),
    'null_broken_exchange': (f'{BASE_FIG_DIR}/null_broken_exchange', f'{BASE_LOG_DIR}/null_broken_exchange'),
    'shuffled': (f'{BASE_FIG_DIR}/shuffled', f'{BASE_LOG_DIR}/shuffled'),
    'replicas': (f'{BASE_FIG_DIR}/replicas', f'{BASE_LOG_DIR}/replicas'),
    'comparison': (f'{BASE_FIG_DIR}/comparison', f'{BASE_LOG_DIR}/comparison'),
}

# Create all directories
for name, (fig_dir, log_dir) in DIRS.items():
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
os.makedirs(BASE_LOG_DIR, exist_ok=True)


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
    specialization: List[float] = field(default_factory=list)


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
    ideas_generated: List[int] = field(default_factory=list)
    collapse_events: List[int] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Complete simulation result."""
    name: str
    seed: int
    n_steps: int
    n_agents: int
    agent_metrics: Dict[str, AgentMetrics]
    collective_metrics: CollectiveMetrics
    agent_names: List[str]
    coupling_mode: str  # 'full', 'none', 'broken'


class MultiAgentSimulation:
    """Multi-agent simulation with configurable coupling."""

    def __init__(self, n_agents: int = 5, seed: int = 42,
                 coupling_mode: str = 'full', name: str = 'real'):
        """
        Initialize simulation.

        coupling_mode:
          - 'full': Normal coupling between agents
          - 'none': No coupling (D1)
          - 'broken': Shuffled/noisy messages (D2)
        """
        self.rng = np.random.default_rng(seed)
        self.n_agents = n_agents
        self.dim = 6
        self.seed = seed
        self.coupling_mode = coupling_mode
        self.name = name

        # Initialize agents
        self.agents: Dict[str, Any] = {}
        self.agent_names = [f'A{i}' for i in range(n_agents)]

        for i, name_agent in enumerate(self.agent_names):
            if i % 2 == 0:
                self.agents[name_agent] = NEO(dim_visible=self.dim, dim_hidden=self.dim)
            else:
                self.agents[name_agent] = EVA(dim_visible=self.dim, dim_hidden=self.dim)

        # Initialize Omega modules per agent
        self.omega_states = {name_agent: OmegaState(dimension=16) for name_agent in self.agent_names}
        self.omega_budgets = {name_agent: OmegaBudget(initial_budget=1.0) for name_agent in self.agent_names}

        # Shared modules
        self.q_field = QField()
        self.phase_space = PhaseSpaceX()
        self.lambda_field = LambdaField()
        self.l_field = LField()
        self.collective_bias_module = CollectiveBias()
        self.omega_compute = OmegaCompute()

        # Complex states
        self.complex_states = {name_agent: ComplexState() for name_agent in self.agent_names}
        self.complex_field = ComplexField(dim=self.dim)

        # Metrics storage
        self.agent_metrics = {name_agent: AgentMetrics() for name_agent in self.agent_names}
        self.collective_metrics = CollectiveMetrics()

        self.t = 0
        self.total_ideas = 0

        # Message buffer for broken exchange
        self.message_buffer = []

    def _get_coupling(self, name: str, mean_field: np.ndarray, agent_state: np.ndarray) -> np.ndarray:
        """Get coupling signal based on mode."""
        if self.coupling_mode == 'none':
            # D1: No coupling at all
            return np.zeros(self.dim)

        elif self.coupling_mode == 'broken':
            # D2: Broken exchange - use shuffled or noise
            if self.rng.random() < 0.5:
                # Shuffle: random agent's state instead
                random_agent = self.rng.choice(self.agent_names)
                other_state = self.agents[random_agent].get_state().z_visible
                return other_state - agent_state / self.n_agents
            else:
                # Noise: endogenous structural noise
                noise_scale = np.std(mean_field) if np.std(mean_field) > 0 else 0.1
                return self.rng.normal(0, noise_scale, self.dim)

        else:
            # Full coupling
            return mean_field - agent_state / self.n_agents

    def step(self):
        """Execute one simulation step."""
        self.t += 1

        # Generate shared stimulus (100% endogenous)
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
            agent_state = agent.get_state().z_visible

            # Get coupling based on mode
            coupling = self._get_coupling(name, mean_field, agent_state)

            # Apply coupling strength
            coupling_strength = 0.1 if self.coupling_mode == 'full' else 0.0 if self.coupling_mode == 'none' else 0.05
            coupled_stimulus = stimulus + coupling_strength * coupling
            coupled_stimulus = np.clip(coupled_stimulus, 0.01, 0.99)

            # Agent step
            response = agent.step(coupled_stimulus)
            responses[name] = response

            # Collect metrics
            state = agent.get_state()
            ce_val = 1.0 / (1.0 + response.surprise)
            self.agent_metrics[name].CE.append(ce_val)
            self.agent_metrics[name].psi_norm.append(np.linalg.norm(state.z_visible))
            self.agent_metrics[name].narr_entropy.append(state.S)
            self.agent_metrics[name].surprise.append(response.surprise)
            self.agent_metrics[name].value.append(response.value)
            self.agent_metrics[name].state_history.append(state.z_visible.copy())

            # Specialization (endogenous: based on variance of recent values)
            recent_values = self.agent_metrics[name].value[-100:] if len(self.agent_metrics[name].value) >= 100 else self.agent_metrics[name].value
            spec = 1.0 - np.std(recent_values) if len(recent_values) > 1 else 0.5
            self.agent_metrics[name].specialization.append(max(0, min(1, spec)))

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
            'coupling': 0.1 if self.coupling_mode == 'full' else 0.0
        }
        lambda_snap = self.lambda_field.step(metrics)
        lambda_stats = self.lambda_field.get_statistics()

        self.collective_metrics.lambda_values.append(lambda_stats.get('mean_lambda', 0.5))
        self.collective_metrics.dominant_regime.append(lambda_stats.get('dominant_regime', 'unknown'))

        # Regime weights (endogenous from lambda field)
        regimes = ['circadian', 'narrative', 'quantum', 'teleo', 'social', 'creative']
        base_weight = 1.0 / len(regimes)
        regime_weights = {}
        for r in regimes:
            # Endogenous variation based on entropy
            variation = 0.1 * np.sin(self.t * 0.01 + hash(r) % 10)
            regime_weights[r] = max(0.05, base_weight + variation)
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

        # Ideas generation (endogenous: based on entropy and surprise)
        mean_entropy = np.mean([self.agents[n].get_state().S for n in self.agent_names])
        mean_surprise = np.mean([responses[n].surprise for n in self.agent_names])
        idea_prob = 0.1 * mean_entropy + 0.05 * mean_surprise
        ideas_this_step = 1 if self.rng.random() < idea_prob else 0
        self.total_ideas += ideas_this_step
        self.collective_metrics.ideas_generated.append(self.total_ideas)

        # Collapse events (endogenous: sharp Q drop)
        if len(self.collective_metrics.Q_global) > 10:
            recent_Q = self.collective_metrics.Q_global[-10:]
            if recent_Q[-1] < np.mean(recent_Q[:-1]) * 0.8:
                self.collective_metrics.collapse_events.append(self.t)

    def run(self, n_steps: int = 6000) -> 'SimulationResult':
        """Run simulation for n_steps."""
        print(f"  Running {self.name}: {n_steps} steps, coupling={self.coupling_mode}...")
        for i in range(n_steps):
            self.step()
            if (i + 1) % 1000 == 0:
                print(f"    Step {i+1}/{n_steps}")

        return SimulationResult(
            name=self.name,
            seed=self.seed,
            n_steps=n_steps,
            n_agents=self.n_agents,
            agent_metrics=self.agent_metrics,
            collective_metrics=self.collective_metrics,
            agent_names=self.agent_names,
            coupling_mode=self.coupling_mode
        )


def shuffle_temporal_order(result: SimulationResult, seed: int = 42) -> SimulationResult:
    """
    D3: Create shuffled version of simulation result.
    Shuffles temporal order per agent but maintains marginal distributions.
    """
    rng = np.random.default_rng(seed)

    shuffled_agent_metrics = {}
    for name in result.agent_names:
        original = result.agent_metrics[name]
        n = len(original.CE)

        # Create permutation per agent
        perm = rng.permutation(n)

        shuffled = AgentMetrics()
        shuffled.CE = [original.CE[i] for i in perm]
        shuffled.psi_norm = [original.psi_norm[i] for i in perm]
        shuffled.narr_entropy = [original.narr_entropy[i] for i in perm]
        shuffled.Q_coherence = [original.Q_coherence[i] for i in perm]
        shuffled.surprise = [original.surprise[i] for i in perm]
        shuffled.value = [original.value[i] for i in perm]
        shuffled.specialization = [original.specialization[i] for i in perm]
        shuffled.state_history = [original.state_history[i] for i in perm]

        shuffled_agent_metrics[name] = shuffled

    # Collective metrics also shuffled
    n_coll = len(result.collective_metrics.Q_global)
    perm_coll = rng.permutation(n_coll)

    shuffled_collective = CollectiveMetrics()
    shuffled_collective.Q_global = [result.collective_metrics.Q_global[i] for i in perm_coll]
    shuffled_collective.lambda_values = [result.collective_metrics.lambda_values[i] for i in perm_coll]
    shuffled_collective.LSI = [result.collective_metrics.LSI[i] for i in perm_coll]
    shuffled_collective.polarization = [result.collective_metrics.polarization[i] for i in perm_coll]
    shuffled_collective.collective_drift = [result.collective_metrics.collective_drift[i] for i in perm_coll]
    shuffled_collective.dominant_regime = [result.collective_metrics.dominant_regime[i] for i in perm_coll]
    shuffled_collective.regime_weights = [result.collective_metrics.regime_weights[i] for i in perm_coll]
    shuffled_collective.ideas_generated = result.collective_metrics.ideas_generated.copy()
    shuffled_collective.collapse_events = []

    return SimulationResult(
        name='shuffled',
        seed=result.seed,
        n_steps=result.n_steps,
        n_agents=result.n_agents,
        agent_metrics=shuffled_agent_metrics,
        collective_metrics=shuffled_collective,
        agent_names=result.agent_names,
        coupling_mode='shuffled'
    )


def compute_correlations(result: SimulationResult) -> Dict[str, np.ndarray]:
    """Compute correlation matrices between agents."""
    n = result.n_agents
    names = result.agent_names

    CE_matrix = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            corr = np.corrcoef(result.agent_metrics[ni].CE, result.agent_metrics[nj].CE)[0, 1]
            CE_matrix[i, j] = corr if not np.isnan(corr) else 0.0

    return {'CE': CE_matrix}


def compute_summary_stats(result: SimulationResult) -> Dict[str, float]:
    """Compute summary statistics for a simulation result."""
    # Correlations
    corr_matrix = compute_correlations(result)['CE']
    n = result.n_agents
    upper_tri = corr_matrix[np.triu_indices(n, k=1)]
    mean_corr = np.mean(np.abs(upper_tri))

    # Polarization
    pol = np.array(result.collective_metrics.polarization)
    mean_pol = np.mean(pol)
    std_pol = np.std(pol)

    # Q coherence
    Q = np.array(result.collective_metrics.Q_global)
    mean_Q = np.mean(Q)
    std_Q = np.std(Q)

    # LSI
    LSI = np.array(result.collective_metrics.LSI)
    mean_LSI = np.mean(LSI)

    # Lambda stability (regime transitions)
    regimes = result.collective_metrics.dominant_regime
    transitions = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
    regime_stability = 1.0 - transitions / len(regimes) if len(regimes) > 0 else 0.0

    # Ideas
    total_ideas = result.collective_metrics.ideas_generated[-1] if result.collective_metrics.ideas_generated else 0

    # Collapse events
    n_collapses = len(result.collective_metrics.collapse_events)

    # Coalitions (clustering proxy: high correlation groups)
    n_coalitions = sum(1 for c in upper_tri if c > 0.7)

    return {
        'mean_correlation': mean_corr,
        'mean_polarization': mean_pol,
        'std_polarization': std_pol,
        'mean_Q_coherence': mean_Q,
        'std_Q_coherence': std_Q,
        'mean_LSI': mean_LSI,
        'regime_stability': regime_stability,
        'total_ideas': total_ideas,
        'n_collapses': n_collapses,
        'n_coalitions': n_coalitions,
        'transitions': transitions
    }


def save_simulation_summary(result: SimulationResult, stats: Dict[str, float],
                           output_path: str):
    """Save simulation summary to file."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"RESUMEN: {result.name.upper()}")
    lines.append(f"Coupling mode: {result.coupling_mode}")
    lines.append(f"Seed: {result.seed}, Steps: {result.n_steps}, Agents: {result.n_agents}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("MÉTRICAS PRINCIPALES:")
    lines.append(f"  Correlación media inter-agente: {stats['mean_correlation']:.4f}")
    lines.append(f"  Polarización: {stats['mean_polarization']:.4f} ± {stats['std_polarization']:.4f}")
    lines.append(f"  Q coherencia: {stats['mean_Q_coherence']:.4f} ± {stats['std_Q_coherence']:.4f}")
    lines.append(f"  LSI (sincronía): {stats['mean_LSI']:.4f}")
    lines.append(f"  Estabilidad de régimen: {stats['regime_stability']:.4f}")
    lines.append(f"  Ideas generadas: {stats['total_ideas']}")
    lines.append(f"  Eventos de colapso: {stats['n_collapses']}")
    lines.append(f"  Coaliciones detectadas: {stats['n_coalitions']}")
    lines.append(f"  Transiciones de régimen: {stats['transitions']}")
    lines.append("")
    lines.append("=" * 70)

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))


def plot_simulation_overview(result: SimulationResult, output_dir: str):
    """Plot overview figures for a simulation."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    T = result.n_steps
    t_axis = np.arange(T)

    # 1. CE evolution
    for name in result.agent_names:
        axes[0, 0].plot(result.agent_metrics[name].CE, label=name, alpha=0.7)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('CE')
    axes[0, 0].set_title(f'CE Evolution ({result.name})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Q coherence
    axes[0, 1].plot(result.collective_metrics.Q_global, 'b-', alpha=0.7)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Q Coherence')
    axes[0, 1].set_title('Q-Field Coherence')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Polarization
    axes[1, 0].plot(result.collective_metrics.polarization, 'm-', alpha=0.7)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Polarization')
    axes[1, 0].set_title('Collective Polarization')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. LSI
    axes[1, 1].plot(result.collective_metrics.LSI, 'g-', alpha=0.7)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('LSI')
    axes[1, 1].set_title('Latent Synchrony Index')
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Lambda values
    axes[2, 0].plot(result.collective_metrics.lambda_values, 'orange', alpha=0.7)
    axes[2, 0].set_xlabel('Time')
    axes[2, 0].set_ylabel('Λ')
    axes[2, 0].set_title('Lambda-Field')
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Specialization
    for name in result.agent_names:
        axes[2, 1].plot(result.agent_metrics[name].specialization, label=name, alpha=0.7)
    axes[2, 1].set_xlabel('Time')
    axes[2, 1].set_ylabel('Specialization')
    axes[2, 1].set_title('Agent Specialization')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/overview_{result.name}.png', dpi=150)
    plt.close()


def plot_real_vs_shuffled(real: SimulationResult, shuffled: SimulationResult, output_dir: str):
    """D3: Plot comparison between real and shuffled."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Correlation comparison
    real_corr = compute_correlations(real)['CE']
    shuffled_corr = compute_correlations(shuffled)['CE']

    n = real.n_agents
    real_upper = real_corr[np.triu_indices(n, k=1)]
    shuffled_upper = shuffled_corr[np.triu_indices(n, k=1)]

    x = np.arange(len(real_upper))
    width = 0.35
    axes[0, 0].bar(x - width/2, real_upper, width, label='Real', color='steelblue')
    axes[0, 0].bar(x + width/2, shuffled_upper, width, label='Shuffled', color='coral')
    axes[0, 0].set_xlabel('Agent Pair')
    axes[0, 0].set_ylabel('Correlation')
    axes[0, 0].set_title('Inter-Agent Correlations: Real vs Shuffled')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Polarization
    axes[0, 1].hist(real.collective_metrics.polarization, bins=30, alpha=0.7, label='Real', density=True)
    axes[0, 1].hist(shuffled.collective_metrics.polarization, bins=30, alpha=0.7, label='Shuffled', density=True)
    axes[0, 1].set_xlabel('Polarization')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Polarization Distribution')
    axes[0, 1].legend()

    # 3. Regime distribution
    real_regimes = {}
    for r in real.collective_metrics.dominant_regime:
        real_regimes[r] = real_regimes.get(r, 0) + 1

    shuffled_regimes = {}
    for r in shuffled.collective_metrics.dominant_regime:
        shuffled_regimes[r] = shuffled_regimes.get(r, 0) + 1

    all_regimes = sorted(set(real_regimes.keys()) | set(shuffled_regimes.keys()))
    x = np.arange(len(all_regimes))
    real_counts = [real_regimes.get(r, 0) for r in all_regimes]
    shuffled_counts = [shuffled_regimes.get(r, 0) for r in all_regimes]

    axes[1, 0].bar(x - width/2, real_counts, width, label='Real', color='steelblue')
    axes[1, 0].bar(x + width/2, shuffled_counts, width, label='Shuffled', color='coral')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(all_regimes, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Regime Distribution')
    axes[1, 0].legend()

    # 4. Autocorrelation (structure test)
    real_Q = np.array(real.collective_metrics.Q_global)
    shuffled_Q = np.array(shuffled.collective_metrics.Q_global)

    real_autocorr = np.correlate(real_Q - np.mean(real_Q), real_Q - np.mean(real_Q), mode='full')
    real_autocorr = real_autocorr[len(real_autocorr)//2:]
    real_autocorr = real_autocorr / real_autocorr[0] if real_autocorr[0] != 0 else real_autocorr

    shuffled_autocorr = np.correlate(shuffled_Q - np.mean(shuffled_Q), shuffled_Q - np.mean(shuffled_Q), mode='full')
    shuffled_autocorr = shuffled_autocorr[len(shuffled_autocorr)//2:]
    shuffled_autocorr = shuffled_autocorr / shuffled_autocorr[0] if shuffled_autocorr[0] != 0 else shuffled_autocorr

    axes[1, 1].plot(real_autocorr[:200], 'b-', label='Real', alpha=0.7)
    axes[1, 1].plot(shuffled_autocorr[:200], 'r--', label='Shuffled', alpha=0.7)
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Autocorrelation')
    axes[1, 1].set_title('Q-Field Autocorrelation (Temporal Structure)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_vs_shuffled_correlacion.png', dpi=150)
    plt.close()

    # Additional figures
    # Regimes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, real_counts, width, label='Real', color='steelblue')
    ax.bar(x + width/2, shuffled_counts, width, label='Shuffled', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(all_regimes, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Real vs Shuffled: Regime Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_vs_shuffled_regimenes.png', dpi=150)
    plt.close()

    # Polarization time series
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(real.collective_metrics.polarization[:1000], 'b-', label='Real', alpha=0.7)
    ax.plot(shuffled.collective_metrics.polarization[:1000], 'r--', label='Shuffled', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Polarization')
    ax.set_title('Real vs Shuffled: Polarization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/real_vs_shuffled_polarizacion.png', dpi=150)
    plt.close()


def run_multi_seed_replicas(n_seeds: int = 5, n_steps: int = 6000) -> List[SimulationResult]:
    """D4: Run multiple replicas with different seeds."""
    results = []
    base_seed = 42

    for i in range(n_seeds):
        seed = base_seed + i * 100
        print(f"\n  Replica {i+1}/{n_seeds} (seed={seed})...")
        sim = MultiAgentSimulation(n_agents=5, seed=seed, coupling_mode='full', name=f'replica_{i}')
        result = sim.run(n_steps=n_steps)
        results.append(result)

    return results


def analyze_replicas(replicas: List[SimulationResult], output_dir: str, log_dir: str):
    """Analyze consistency across replicas."""
    # Compute stats for each replica
    all_stats = []
    for result in replicas:
        stats = compute_summary_stats(result)
        stats['seed'] = result.seed
        stats['name'] = result.name
        all_stats.append(stats)

    # Save to CSV
    df = pd.DataFrame(all_stats)
    df.to_csv(f'{log_dir}/seeds_consistency.csv', index=False)

    # Plot consistency
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    seeds = [s['seed'] for s in all_stats]

    # 1. Correlation
    corrs = [s['mean_correlation'] for s in all_stats]
    axes[0, 0].bar(range(len(corrs)), corrs, color='steelblue')
    axes[0, 0].set_xticks(range(len(seeds)))
    axes[0, 0].set_xticklabels([f'S{s}' for s in seeds])
    axes[0, 0].set_ylabel('Mean Correlation')
    axes[0, 0].set_title('Inter-Agent Correlation')
    axes[0, 0].axhline(np.mean(corrs), color='red', linestyle='--', label=f'Mean: {np.mean(corrs):.3f}')
    axes[0, 0].legend()

    # 2. Polarization
    pols = [s['mean_polarization'] for s in all_stats]
    axes[0, 1].bar(range(len(pols)), pols, color='coral')
    axes[0, 1].set_xticks(range(len(seeds)))
    axes[0, 1].set_xticklabels([f'S{s}' for s in seeds])
    axes[0, 1].set_ylabel('Mean Polarization')
    axes[0, 1].set_title('Polarization')
    axes[0, 1].axhline(np.mean(pols), color='red', linestyle='--', label=f'Mean: {np.mean(pols):.3f}')
    axes[0, 1].legend()

    # 3. Q coherence
    Qs = [s['mean_Q_coherence'] for s in all_stats]
    axes[0, 2].bar(range(len(Qs)), Qs, color='green')
    axes[0, 2].set_xticks(range(len(seeds)))
    axes[0, 2].set_xticklabels([f'S{s}' for s in seeds])
    axes[0, 2].set_ylabel('Mean Q Coherence')
    axes[0, 2].set_title('Q-Field Coherence')
    axes[0, 2].axhline(np.mean(Qs), color='red', linestyle='--', label=f'Mean: {np.mean(Qs):.3f}')
    axes[0, 2].legend()

    # 4. Coalitions
    coals = [s['n_coalitions'] for s in all_stats]
    axes[1, 0].bar(range(len(coals)), coals, color='purple')
    axes[1, 0].set_xticks(range(len(seeds)))
    axes[1, 0].set_xticklabels([f'S{s}' for s in seeds])
    axes[1, 0].set_ylabel('N Coalitions')
    axes[1, 0].set_title('Coalitions Detected')
    axes[1, 0].axhline(np.mean(coals), color='red', linestyle='--', label=f'Mean: {np.mean(coals):.1f}')
    axes[1, 0].legend()

    # 5. Regime stability
    stabs = [s['regime_stability'] for s in all_stats]
    axes[1, 1].bar(range(len(stabs)), stabs, color='orange')
    axes[1, 1].set_xticks(range(len(seeds)))
    axes[1, 1].set_xticklabels([f'S{s}' for s in seeds])
    axes[1, 1].set_ylabel('Stability')
    axes[1, 1].set_title('Regime Stability')
    axes[1, 1].axhline(np.mean(stabs), color='red', linestyle='--', label=f'Mean: {np.mean(stabs):.3f}')
    axes[1, 1].legend()

    # 6. Summary box
    axes[1, 2].axis('off')
    summary_text = f"""
    CONSISTENCIA ENTRE RÉPLICAS
    ═══════════════════════════

    Correlación: {np.mean(corrs):.3f} ± {np.std(corrs):.3f}
    Polarización: {np.mean(pols):.3f} ± {np.std(pols):.3f}
    Q Coherencia: {np.mean(Qs):.3f} ± {np.std(Qs):.3f}
    Coaliciones: {np.mean(coals):.1f} ± {np.std(coals):.1f}
    Estabilidad: {np.mean(stabs):.3f} ± {np.std(stabs):.3f}

    CV (variabilidad):
      Corr: {np.std(corrs)/np.mean(corrs)*100:.1f}%
      Pol: {np.std(pols)/np.mean(pols)*100:.1f}% if np.mean(pols) > 0 else 'N/A'
      Q: {np.std(Qs)/np.mean(Qs)*100:.1f}%
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center', transform=axes[1, 2].transAxes)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/seeds_consistency.png', dpi=150)
    plt.close()

    return df


def plot_comparison_all(real: SimulationResult, null1: SimulationResult,
                       null2: SimulationResult, shuffled: SimulationResult,
                       output_dir: str):
    """D5: Plot comprehensive comparison of all conditions."""

    # Compute stats
    stats = {
        'Real': compute_summary_stats(real),
        'No Coupling': compute_summary_stats(null1),
        'Broken Exchange': compute_summary_stats(null2),
        'Shuffled': compute_summary_stats(shuffled)
    }

    conditions = list(stats.keys())
    colors = ['steelblue', 'coral', 'green', 'purple']

    # Figure 1: Collective Bias Real vs Null
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Polarization
    pols = [stats[c]['mean_polarization'] for c in conditions]
    axes[0, 0].bar(conditions, pols, color=colors)
    axes[0, 0].set_ylabel('Mean Polarization')
    axes[0, 0].set_title('Polarization')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Coalitions
    coals = [stats[c]['n_coalitions'] for c in conditions]
    axes[0, 1].bar(conditions, coals, color=colors)
    axes[0, 1].set_ylabel('N Coalitions')
    axes[0, 1].set_title('Coalition Frequency')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Q coherence
    Qs = [stats[c]['mean_Q_coherence'] for c in conditions]
    axes[0, 2].bar(conditions, Qs, color=colors)
    axes[0, 2].set_ylabel('Mean Q Coherence')
    axes[0, 2].set_title('Q-Field Coherence')
    axes[0, 2].tick_params(axis='x', rotation=45)

    # 4. LSI
    lsis = [stats[c]['mean_LSI'] for c in conditions]
    axes[1, 0].bar(conditions, lsis, color=colors)
    axes[1, 0].set_ylabel('Mean LSI')
    axes[1, 0].set_title('Collective Synchrony (LSI)')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 5. Regime stability
    stabs = [stats[c]['regime_stability'] for c in conditions]
    axes[1, 1].bar(conditions, stabs, color=colors)
    axes[1, 1].set_ylabel('Stability')
    axes[1, 1].set_title('Regime Stability')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # 6. Collapses
    colls = [stats[c]['n_collapses'] for c in conditions]
    axes[1, 2].bar(conditions, colls, color=colors)
    axes[1, 2].set_ylabel('N Collapses')
    axes[1, 2].set_title('Collapse Events')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/collective_bias_real_vs_null.png', dpi=150)
    plt.close()

    # Figure 2: Regimes comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    results_list = [real, null1, null2, shuffled]
    titles = conditions

    for idx, (result, title) in enumerate(zip(results_list, titles)):
        ax = axes[idx // 2, idx % 2]

        regime_counts = {}
        for r in result.collective_metrics.dominant_regime:
            regime_counts[r] = regime_counts.get(r, 0) + 1

        if regime_counts:
            regimes = list(regime_counts.keys())
            counts = list(regime_counts.values())
            ax.bar(regimes, counts, color=colors[idx])
        ax.set_title(f'Regimes: {title}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/regimes_real_vs_null.png', dpi=150)
    plt.close()

    # Figure 3: Polarization comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (result, title) in enumerate(zip(results_list, titles)):
        pol = result.collective_metrics.polarization[:2000]
        ax.plot(pol, label=title, color=colors[idx], alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel('Polarization')
    ax.set_title('Polarization: Real vs Null Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/polarization_real_vs_null.png', dpi=150)
    plt.close()

    # Figure 4: Entanglement (Q-field)
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (result, title) in enumerate(zip(results_list, titles)):
        Q = result.collective_metrics.Q_global[:2000]
        ax.plot(Q, label=title, color=colors[idx], alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel('Q Coherence')
    ax.set_title('Q-Field Entanglement: Real vs Null Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/entanglement_real_vs_null.png', dpi=150)
    plt.close()

    return stats


def generate_master_summary(real: SimulationResult, null1: SimulationResult,
                           null2: SimulationResult, shuffled: SimulationResult,
                           replicas_df: pd.DataFrame, output_path: str):
    """Generate master summary for Phase D."""

    stats = {
        'Real': compute_summary_stats(real),
        'No Coupling': compute_summary_stats(null1),
        'Broken Exchange': compute_summary_stats(null2),
        'Shuffled': compute_summary_stats(shuffled)
    }

    lines = []
    lines.append("=" * 70)
    lines.append("FASE D: MODELOS NULOS + RÉPLICAS - RESUMEN MAESTRO")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Configuración: 5 agentes, 6000 pasos (12h simuladas)")
    lines.append("")

    lines.append("-" * 70)
    lines.append("D1. MODELO NULO 1: SIN ACOPLAMIENTO")
    lines.append("-" * 70)
    lines.append(f"  Correlación media: {stats['No Coupling']['mean_correlation']:.4f}")
    lines.append(f"  Polarización: {stats['No Coupling']['mean_polarization']:.4f}")
    lines.append(f"  Q coherencia: {stats['No Coupling']['mean_Q_coherence']:.4f}")
    lines.append(f"  LSI: {stats['No Coupling']['mean_LSI']:.4f}")
    lines.append(f"  Coaliciones: {stats['No Coupling']['n_coalitions']}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("D2. MODELO NULO 2: INTERCAMBIO ROTO")
    lines.append("-" * 70)
    lines.append(f"  Correlación media: {stats['Broken Exchange']['mean_correlation']:.4f}")
    lines.append(f"  Polarización: {stats['Broken Exchange']['mean_polarization']:.4f}")
    lines.append(f"  Q coherencia: {stats['Broken Exchange']['mean_Q_coherence']:.4f}")
    lines.append(f"  LSI: {stats['Broken Exchange']['mean_LSI']:.4f}")
    lines.append(f"  Coaliciones: {stats['Broken Exchange']['n_coalitions']}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("D3. MODELO NULO 3: SHUFFLED HISTORY")
    lines.append("-" * 70)
    lines.append(f"  Correlación media: {stats['Shuffled']['mean_correlation']:.4f}")
    lines.append(f"  Polarización: {stats['Shuffled']['mean_polarization']:.4f}")
    lines.append(f"  Q coherencia: {stats['Shuffled']['mean_Q_coherence']:.4f}")
    lines.append(f"  LSI: {stats['Shuffled']['mean_LSI']:.4f}")
    lines.append(f"  Estructura temporal preservada: NO (shuffled)")
    lines.append("")

    lines.append("-" * 70)
    lines.append("D4. RÉPLICAS MULTI-SEED (5 seeds)")
    lines.append("-" * 70)
    lines.append(f"  Correlación media: {replicas_df['mean_correlation'].mean():.4f} ± {replicas_df['mean_correlation'].std():.4f}")
    lines.append(f"  Polarización media: {replicas_df['mean_polarization'].mean():.4f} ± {replicas_df['mean_polarization'].std():.4f}")
    lines.append(f"  Q coherencia media: {replicas_df['mean_Q_coherence'].mean():.4f} ± {replicas_df['mean_Q_coherence'].std():.4f}")
    lines.append(f"  Coaliciones media: {replicas_df['n_coalitions'].mean():.1f} ± {replicas_df['n_coalitions'].std():.1f}")
    lines.append(f"  Consistencia entre réplicas: ALTA" if replicas_df['mean_correlation'].std() < 0.1 else "  Consistencia entre réplicas: MODERADA")
    lines.append("")

    lines.append("-" * 70)
    lines.append("D5. COMPARATIVA FINAL REAL vs NULO")
    lines.append("-" * 70)
    lines.append("")
    lines.append("                      Real     No-Coup   Broken    Shuffled")
    lines.append("  " + "-" * 60)
    lines.append(f"  Correlación:        {stats['Real']['mean_correlation']:.4f}   {stats['No Coupling']['mean_correlation']:.4f}    {stats['Broken Exchange']['mean_correlation']:.4f}    {stats['Shuffled']['mean_correlation']:.4f}")
    lines.append(f"  Polarización:       {stats['Real']['mean_polarization']:.4f}   {stats['No Coupling']['mean_polarization']:.4f}    {stats['Broken Exchange']['mean_polarization']:.4f}    {stats['Shuffled']['mean_polarization']:.4f}")
    lines.append(f"  Q coherencia:       {stats['Real']['mean_Q_coherence']:.4f}   {stats['No Coupling']['mean_Q_coherence']:.4f}    {stats['Broken Exchange']['mean_Q_coherence']:.4f}    {stats['Shuffled']['mean_Q_coherence']:.4f}")
    lines.append(f"  LSI:                {stats['Real']['mean_LSI']:.4f}   {stats['No Coupling']['mean_LSI']:.4f}    {stats['Broken Exchange']['mean_LSI']:.4f}    {stats['Shuffled']['mean_LSI']:.4f}")
    lines.append(f"  Coaliciones:        {stats['Real']['n_coalitions']}        {stats['No Coupling']['n_coalitions']}         {stats['Broken Exchange']['n_coalitions']}         {stats['Shuffled']['n_coalitions']}")
    lines.append(f"  Colapsos:           {stats['Real']['n_collapses']}        {stats['No Coupling']['n_collapses']}         {stats['Broken Exchange']['n_collapses']}         {stats['Shuffled']['n_collapses']}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("CONCLUSIONES")
    lines.append("-" * 70)

    # Analyze differences
    real_corr = stats['Real']['mean_correlation']
    null_corr = stats['No Coupling']['mean_correlation']

    if real_corr > null_corr * 1.5:
        lines.append("  1. El ACOPLAMIENTO es NECESARIO para correlación inter-agente alta")
    else:
        lines.append("  1. Correlación similar con/sin acoplamiento (dinámica interna domina)")

    if stats['Real']['mean_polarization'] > stats['Shuffled']['mean_polarization'] * 1.2:
        lines.append("  2. La ESTRUCTURA TEMPORAL contribuye a la polarización")
    else:
        lines.append("  2. Polarización similar real/shuffled (distribución marginal domina)")

    if stats['Real']['n_coalitions'] > stats['No Coupling']['n_coalitions']:
        lines.append("  3. Las COALICIONES emergen del acoplamiento")
    else:
        lines.append("  3. Coaliciones aparecen incluso sin acoplamiento (endógeno)")

    # KS tests
    real_Q = np.array(real.collective_metrics.Q_global)
    null_Q = np.array(null1.collective_metrics.Q_global)
    ks_stat, ks_p = stats_ks_2samp_safe(real_Q, null_Q)

    lines.append("")
    lines.append(f"  Test KS (Real vs No-Coupling): stat={ks_stat:.4f}, p={ks_p:.4e}")
    lines.append(f"  Diferencia significativa: {'SÍ' if ks_p < 0.05 else 'NO'}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("El sesgo colectivo es EMERGENTE del acoplamiento" if real_corr > null_corr * 1.5 else "El sesgo colectivo tiene componentes tanto endógenos como de acoplamiento")
    lines.append("=" * 70)

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    return "\n".join(lines)


def stats_ks_2samp_safe(a, b):
    """Safe KS test."""
    try:
        return stats.ks_2samp(a, b)
    except:
        return 0.0, 1.0


def main():
    """Main execution for Phase D."""
    print("=" * 70)
    print("FASE D: MODELOS NULOS + RÉPLICAS")
    print("=" * 70)
    print()

    N_STEPS = 6000  # 12h simulation

    # D1: NULL 1 - No coupling
    print("\n" + "=" * 50)
    print("D1. MODELO NULO 1: SIN ACOPLAMIENTO")
    print("=" * 50)
    sim_null1 = MultiAgentSimulation(n_agents=5, seed=42, coupling_mode='none', name='null_no_coupling')
    result_null1 = sim_null1.run(n_steps=N_STEPS)
    stats_null1 = compute_summary_stats(result_null1)
    save_simulation_summary(result_null1, stats_null1,
                           f'{DIRS["null_no_coupling"][1]}/null_no_coupling_summary.txt')
    plot_simulation_overview(result_null1, DIRS['null_no_coupling'][0])

    # D2: NULL 2 - Broken exchange
    print("\n" + "=" * 50)
    print("D2. MODELO NULO 2: INTERCAMBIO ROTO")
    print("=" * 50)
    sim_null2 = MultiAgentSimulation(n_agents=5, seed=42, coupling_mode='broken', name='null_broken_exchange')
    result_null2 = sim_null2.run(n_steps=N_STEPS)
    stats_null2 = compute_summary_stats(result_null2)
    save_simulation_summary(result_null2, stats_null2,
                           f'{DIRS["null_broken_exchange"][1]}/null_broken_exchange_summary.txt')
    plot_simulation_overview(result_null2, DIRS['null_broken_exchange'][0])

    # Real simulation for comparison
    print("\n" + "=" * 50)
    print("SIMULACIÓN REAL (con acoplamiento)")
    print("=" * 50)
    sim_real = MultiAgentSimulation(n_agents=5, seed=42, coupling_mode='full', name='real')
    result_real = sim_real.run(n_steps=N_STEPS)

    # D3: NULL 3 - Shuffled history
    print("\n" + "=" * 50)
    print("D3. MODELO NULO 3: SHUFFLED HISTORY")
    print("=" * 50)
    result_shuffled = shuffle_temporal_order(result_real, seed=42)
    stats_shuffled = compute_summary_stats(result_shuffled)
    save_simulation_summary(result_shuffled, stats_shuffled,
                           f'{DIRS["shuffled"][1]}/shuffled_summary.txt')
    plot_real_vs_shuffled(result_real, result_shuffled, DIRS['shuffled'][0])

    # D4: Multi-seed replicas
    print("\n" + "=" * 50)
    print("D4. RÉPLICAS MULTI-SEED")
    print("=" * 50)
    replicas = run_multi_seed_replicas(n_seeds=5, n_steps=N_STEPS)
    replicas_df = analyze_replicas(replicas, DIRS['replicas'][0], DIRS['replicas'][1])

    # D5: Final comparison
    print("\n" + "=" * 50)
    print("D5. COMPARATIVA FINAL")
    print("=" * 50)
    comparison_stats = plot_comparison_all(result_real, result_null1, result_null2,
                                          result_shuffled, DIRS['comparison'][0])

    # Master summary
    print("\nGenerando resumen maestro...")
    summary = generate_master_summary(result_real, result_null1, result_null2,
                                     result_shuffled, replicas_df,
                                     f'{BASE_LOG_DIR}/sesgo_D_master_summary.txt')

    print("\n" + summary)

    print(f"\n" + "=" * 70)
    print("ARCHIVOS GENERADOS:")
    print("=" * 70)
    for name, (fig_dir, log_dir) in DIRS.items():
        print(f"  {name}:")
        print(f"    Figuras: {fig_dir}/")
        print(f"    Logs: {log_dir}/")
    print(f"  Master summary: {BASE_LOG_DIR}/sesgo_D_master_summary.txt")

    return {
        'real': result_real,
        'null1': result_null1,
        'null2': result_null2,
        'shuffled': result_shuffled,
        'replicas': replicas,
        'comparison_stats': comparison_stats
    }


if __name__ == '__main__':
    main()
