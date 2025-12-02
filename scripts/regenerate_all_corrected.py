#!/usr/bin/env python3
"""
REGENERACIÓN DE FIGURAS Y LOGS SIN ARTEFACTOS
==============================================

Regenera todas las figuras y logs del sistema usando las métricas
corregidas que son 100% endógenas y específicas por agente.

Verifica:
1. Métricas diferentes entre agentes
2. Q_coherence por agente (no global)
3. psi_norm, H_narr, etc. con variabilidad real
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from typing import Dict, List, Any
from dataclasses import dataclass, field

sys.path.insert(0, '/root/NEO_EVA')

from core.agents import NEO, EVA, BaseAgent
from omega.q_field import QField

# Output directories
FIG_DIR = '/root/NEO_EVA/figuras/sesgo_colectivo_corrected'
LOG_DIR = '/root/NEO_EVA/logs/sesgo_colectivo_corrected'
AUDIT_DIR = '/root/NEO_EVA/AUDIT'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(AUDIT_DIR, exist_ok=True)


@dataclass
class AgentMetrics:
    """Metrics per agent."""
    CE: List[float] = field(default_factory=list)
    psi_norm: List[float] = field(default_factory=list)
    H_narr: List[float] = field(default_factory=list)
    Q_coherence: List[float] = field(default_factory=list)
    surprise: List[float] = field(default_factory=list)
    value: List[float] = field(default_factory=list)


def run_corrected_simulation(n_steps: int = 3000, n_agents: int = 5, seed: int = 42):
    """Run simulation with corrected metrics."""
    print(f"Running corrected simulation: {n_steps} steps, {n_agents} agents")

    # Reset agent counter for reproducibility
    BaseAgent._agent_counter = 0

    rng = np.random.default_rng(seed)
    dim = 6

    # Create agents - each with unique RNG
    agents = {}
    agent_names = [f'A{i}' for i in range(n_agents)]

    for i, name in enumerate(agent_names):
        if i % 2 == 0:
            agents[name] = NEO(dim_visible=dim, dim_hidden=dim)
        else:
            agents[name] = EVA(dim_visible=dim, dim_hidden=dim)

    # Q-Field for coherence
    q_field = QField()

    # Metrics storage
    metrics = {name: AgentMetrics() for name in agent_names}

    for t in range(n_steps):
        # Shared stimulus
        stimulus = rng.uniform(0, 1, dim)

        # Mean field for coupling
        states = [agents[name].get_state().z_visible for name in agent_names]
        mean_field = np.mean(states, axis=0)

        for name in agent_names:
            agent = agents[name]
            agent_state = agent.get_state()

            # Coupling
            coupling = mean_field - agent_state.z_visible / n_agents
            coupled_stimulus = stimulus + 0.1 * coupling
            coupled_stimulus = np.clip(coupled_stimulus, 0.01, 0.99)

            # Agent step
            response = agent.step(coupled_stimulus)
            state = agent.get_state()

            # Collect metrics - EACH FROM ITS OWN STATE
            metrics[name].CE.append(1.0 / (1.0 + response.surprise))
            metrics[name].psi_norm.append(np.linalg.norm(state.z_visible))
            metrics[name].H_narr.append(state.S)  # Entropy from individual state
            metrics[name].surprise.append(response.surprise)
            metrics[name].value.append(response.value)

            # Q-Field registration - individual state
            q_field.register_state(name, state.z_visible)

        # Q coherence - NOW PER AGENT
        q_stats = q_field.get_statistics()
        for name in agent_names:
            # Use agent-specific coherence (now available)
            agent_q = q_stats.get(f'{name}_coherence', q_stats.get('mean_coherence', 0.5))
            metrics[name].Q_coherence.append(agent_q)

        if (t + 1) % 500 == 0:
            print(f"  Step {t+1}/{n_steps}")

    return metrics, agent_names


def verify_uniqueness(metrics: Dict[str, AgentMetrics], agent_names: List[str]) -> Dict[str, Any]:
    """Verify that metrics are unique per agent."""
    results = {
        'psi_norm': {'unique': True, 'max_corr': 0.0},
        'H_narr': {'unique': True, 'max_corr': 0.0},
        'Q_coherence': {'unique': True, 'max_corr': 0.0},
        'CE': {'unique': True, 'max_corr': 0.0},
    }

    for metric_name in ['psi_norm', 'H_narr', 'Q_coherence', 'CE']:
        max_corr = 0.0
        all_identical = True

        for i, name_i in enumerate(agent_names):
            for j, name_j in enumerate(agent_names):
                if i < j:
                    series_i = getattr(metrics[name_i], metric_name)
                    series_j = getattr(metrics[name_j], metric_name)

                    # Check identical
                    if not np.array_equal(series_i, series_j):
                        all_identical = False

                    # Check correlation
                    corr = np.corrcoef(series_i, series_j)[0, 1]
                    if not np.isnan(corr):
                        max_corr = max(max_corr, abs(corr))

        results[metric_name]['unique'] = not all_identical
        results[metric_name]['max_corr'] = max_corr

    return results


def compute_correlations(metrics: Dict[str, AgentMetrics], agent_names: List[str], metric_name: str) -> np.ndarray:
    """Compute correlation matrix for a metric."""
    n = len(agent_names)
    matrix = np.zeros((n, n))

    for i, ni in enumerate(agent_names):
        for j, nj in enumerate(agent_names):
            series_i = getattr(metrics[ni], metric_name)
            series_j = getattr(metrics[nj], metric_name)
            corr = np.corrcoef(series_i, series_j)[0, 1]
            matrix[i, j] = corr if not np.isnan(corr) else 0.0

    return matrix


def plot_correlation_heatmap(matrix: np.ndarray, title: str, filename: str, agent_names: List[str]):
    """Plot correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(agent_names)))
    ax.set_yticks(range(len(agent_names)))
    ax.set_xticklabels(agent_names)
    ax.set_yticklabels(agent_names)

    for i in range(len(agent_names)):
        for j in range(len(agent_names)):
            ax.text(j, i, f'{matrix[i, j]:.3f}',
                   ha='center', va='center', color='black', fontsize=10)

    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_time_series(metrics: Dict[str, AgentMetrics], agent_names: List[str],
                     metric_name: str, filename: str):
    """Plot time series for all agents."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for name in agent_names:
        series = getattr(metrics[name], metric_name)[:500]  # First 500 for clarity
        ax.plot(series, label=name, alpha=0.7)

    ax.set_xlabel('Time Step')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name}: Series por Agente (CORREGIDO)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def generate_comparison_report(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    """Generate comparison report before/after fix."""
    lines = []
    lines.append("=" * 70)
    lines.append("COMPARACIÓN: ANTES vs DESPUÉS DE CORRECCIÓN")
    lines.append("=" * 70)
    lines.append("")

    lines.append("ANTES (Artefactos):")
    lines.append("-" * 40)
    lines.append("  psi_norm: Series IDÉNTICAS (corr = 1.0000)")
    lines.append("  H_narr: Series IDÉNTICAS (corr = 1.0000)")
    lines.append("  Q_coherence: Series IDÉNTICAS (corr = 1.0000)")
    lines.append("  CE: Parcialmente diferentes (NEO≠EVA)")
    lines.append("")

    lines.append("DESPUÉS (Corregido):")
    lines.append("-" * 40)
    for metric, info in after.items():
        status = "✓ ÚNICO" if info['unique'] else "✗ IDÉNTICO"
        lines.append(f"  {metric}: {status} (max_corr = {info['max_corr']:.4f})")

    lines.append("")
    lines.append("=" * 70)
    lines.append("CORRECCIONES APLICADAS:")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. Agentes: RNG individual por agente (self._rng)")
    lines.append("2. Agentes: Estado inicial con perturbación endógena individual")
    lines.append("3. Agentes: Ruido endógeno en _update_state()")
    lines.append("4. Q-Field: get_statistics() incluye valores por agente")
    lines.append("5. Simulaciones: Uso de Q_coherence específico por agente")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main execution."""
    print("=" * 70)
    print("REGENERACIÓN DE FIGURAS Y LOGS SIN ARTEFACTOS")
    print("=" * 70)

    # Run corrected simulation
    metrics, agent_names = run_corrected_simulation(n_steps=3000, n_agents=5, seed=42)

    # Verify uniqueness
    print("\nVerificando unicidad de métricas...")
    uniqueness_results = verify_uniqueness(metrics, agent_names)

    for metric, info in uniqueness_results.items():
        status = "✓ ÚNICO" if info['unique'] else "✗ IDÉNTICO"
        print(f"  {metric}: {status} (max_corr = {info['max_corr']:.4f})")

    # Generate figures
    print("\nGenerando figuras...")

    # Correlation heatmaps
    for metric_name in ['CE', 'psi_norm', 'H_narr', 'Q_coherence']:
        matrix = compute_correlations(metrics, agent_names, metric_name)
        plot_correlation_heatmap(
            matrix,
            f'Correlaciones {metric_name} (CORREGIDO)',
            f'{FIG_DIR}/correlaciones_{metric_name}.png',
            agent_names
        )
        print(f"  ✓ correlaciones_{metric_name}.png")

    # Time series
    for metric_name in ['CE', 'psi_norm', 'H_narr', 'Q_coherence']:
        plot_time_series(metrics, agent_names, metric_name,
                        f'{FIG_DIR}/timeseries_{metric_name}.png')
        print(f"  ✓ timeseries_{metric_name}.png")

    # Save CSVs
    print("\nGuardando CSVs...")

    # Correlations
    df_corr = pd.DataFrame()
    for metric_name in ['CE', 'psi_norm', 'H_narr', 'Q_coherence']:
        matrix = compute_correlations(metrics, agent_names, metric_name)
        for i, ni in enumerate(agent_names):
            for j, nj in enumerate(agent_names):
                df_corr = pd.concat([df_corr, pd.DataFrame({
                    'metric': [metric_name],
                    'agent_i': [ni],
                    'agent_j': [nj],
                    'correlation': [matrix[i, j]]
                })])
    df_corr.to_csv(f'{LOG_DIR}/correlaciones_corrected.csv', index=False)
    print(f"  ✓ correlaciones_corrected.csv")

    # Stats per agent
    stats = []
    for name in agent_names:
        for metric_name in ['CE', 'psi_norm', 'H_narr', 'Q_coherence']:
            series = getattr(metrics[name], metric_name)
            stats.append({
                'agent': name,
                'metric': metric_name,
                'min': np.min(series),
                'max': np.max(series),
                'mean': np.mean(series),
                'std': np.std(series)
            })
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(f'{LOG_DIR}/stats_per_agent_corrected.csv', index=False)
    print(f"  ✓ stats_per_agent_corrected.csv")

    # Generate comparison report
    print("\nGenerando reporte de comparación...")
    report = generate_comparison_report({}, uniqueness_results)
    with open(f'{AUDIT_DIR}/post_fix_differences.txt', 'w') as f:
        f.write(report)
    print(f"  ✓ post_fix_differences.txt")

    print("\n" + report)

    print(f"\n" + "=" * 70)
    print("ARCHIVOS GENERADOS:")
    print(f"  Figuras: {FIG_DIR}/")
    print(f"  Logs: {LOG_DIR}/")
    print(f"  Audit: {AUDIT_DIR}/")
    print("=" * 70)

    return metrics, uniqueness_results


if __name__ == '__main__':
    main()
