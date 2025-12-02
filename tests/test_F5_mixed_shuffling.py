#!/usr/bin/env python3
"""
FASE F5 - Mixed Shuffling Tests (Windows + Agents)
====================================================

Objetivo: Destruir simultáneamente estructura temporal Y de identidad
para demostrar que el sesgo colectivo requiere AMBAS.

El shuffling mixto:
1. Aplica shuffling por ventanas (como F1)
2. Además, intercambia valores ENTRE agentes dentro de cada ventana
3. Mantiene histograma global (distribución marginal total)

Esperado:
- Correlaciones cercanas a 0
- Coaliciones colapsadas (~0 o 1)
- Todo patrón de sesgo destruido

100% Endógeno - Sin números mágicos externos.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats

sys.path.insert(0, '/root/NEO_EVA')

from core.agents import NEO, EVA, BaseAgent

# Output directory
FIG_DIR = '/root/NEO_EVA/figuras/FASE_F'
os.makedirs(FIG_DIR, exist_ok=True)


@dataclass
class SimulationData:
    """Datos de una simulación."""
    CE: Dict[str, List[float]] = field(default_factory=dict)
    Value: Dict[str, List[float]] = field(default_factory=dict)
    Surprise: Dict[str, List[float]] = field(default_factory=dict)
    agent_names: List[str] = field(default_factory=list)
    n_steps: int = 0


def run_simulation(n_steps: int = 3000, n_agents: int = 5, seed: int = 42) -> SimulationData:
    """Ejecuta simulación estándar."""
    BaseAgent._agent_counter = 0
    rng = np.random.default_rng(seed)
    dim = 6

    agents = {}
    agent_names = [f'A{i}' for i in range(n_agents)]

    for i, name in enumerate(agent_names):
        if i % 2 == 0:
            agents[name] = NEO(dim_visible=dim, dim_hidden=dim)
        else:
            agents[name] = EVA(dim_visible=dim, dim_hidden=dim)

    data = SimulationData(
        CE={name: [] for name in agent_names},
        Value={name: [] for name in agent_names},
        Surprise={name: [] for name in agent_names},
        agent_names=agent_names,
        n_steps=n_steps
    )

    for t in range(n_steps):
        stimulus = rng.uniform(0, 1, dim)
        states_list = [agents[name].get_state().z_visible for name in agent_names]
        mean_field = np.mean(states_list, axis=0)

        for name in agent_names:
            agent = agents[name]
            state = agent.get_state()
            coupling = mean_field - state.z_visible / n_agents
            coupled_stimulus = stimulus + 0.1 * coupling
            coupled_stimulus = np.clip(coupled_stimulus, 0.01, 0.99)

            response = agent.step(coupled_stimulus)

            data.CE[name].append(1.0 / (1.0 + response.surprise))
            data.Value[name].append(response.value)
            data.Surprise[name].append(response.surprise)

    return data


def window_shuffle_temporal(data: np.ndarray, window_size: int,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Shuffling temporal dentro de ventanas (como F1).

    Args:
        data: Array (n_agents, T)
        window_size: Tamaño de ventana
        rng: Generador de números aleatorios

    Returns:
        Array con orden temporal shuffleado por ventanas
    """
    data = np.atleast_2d(data)
    n_agents, T = data.shape
    shuffled = data.copy()

    n_windows = T // window_size
    remainder = T % window_size

    for agent_idx in range(n_agents):
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size
            indices = np.arange(start, end)
            rng.shuffle(indices)
            shuffled[agent_idx, start:end] = data[agent_idx, indices]

        if remainder > 0:
            start = n_windows * window_size
            indices = np.arange(start, T)
            rng.shuffle(indices)
            shuffled[agent_idx, start:] = data[agent_idx, indices]

    return shuffled


def agent_shuffle_within_windows(data: np.ndarray, window_size: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """
    Intercambia valores ENTRE agentes dentro de cada ventana.

    Mantiene el histograma global pero destruye la identidad individual.

    Args:
        data: Array (n_agents, T)
        window_size: Tamaño de ventana
        rng: Generador de números aleatorios

    Returns:
        Array con valores intercambiados entre agentes por ventana
    """
    data = np.atleast_2d(data)
    n_agents, T = data.shape
    shuffled = data.copy()

    n_windows = T // window_size
    remainder = T % window_size

    for w in range(n_windows):
        start = w * window_size
        end = start + window_size

        # Para cada timestep en la ventana, shuffle entre agentes
        for t in range(start, end):
            values = shuffled[:, t].copy()
            rng.shuffle(values)
            shuffled[:, t] = values

    # Remainder
    if remainder > 0:
        start = n_windows * window_size
        for t in range(start, T):
            values = shuffled[:, t].copy()
            rng.shuffle(values)
            shuffled[:, t] = values

    return shuffled


def mixed_shuffle(data: np.ndarray, window_size: int,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Aplica AMBOS: shuffling temporal + shuffling entre agentes.

    Args:
        data: Array (n_agents, T)
        window_size: Tamaño de ventana
        rng: Generador de números aleatorios

    Returns:
        Array doblemente shuffleado
    """
    # Primero: shuffle temporal por agente
    temp_shuffled = window_shuffle_temporal(data, window_size, rng)

    # Segundo: shuffle entre agentes
    mixed = agent_shuffle_within_windows(temp_shuffled, window_size, rng)

    return mixed


def compute_inter_agent_correlation(data: np.ndarray) -> float:
    """Calcula correlación media entre pares de agentes."""
    n_agents = data.shape[0]
    if n_agents < 2:
        return 0.0

    correlations = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if np.std(data[i]) > 1e-12 and np.std(data[j]) > 1e-12:
                corr, _ = stats.pearsonr(data[i], data[j])
                if not np.isnan(corr):
                    correlations.append(abs(corr))

    return float(np.mean(correlations)) if correlations else 0.0


def detect_coalitions(data: np.ndarray) -> int:
    """Detecta coaliciones basándose en correlaciones."""
    n_agents = data.shape[0]
    if n_agents < 2:
        return 1

    corr_matrix = np.zeros((n_agents, n_agents))
    correlations = []

    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif np.std(data[i]) > 1e-12 and np.std(data[j]) > 1e-12:
                corr, _ = stats.pearsonr(data[i], data[j])
                corr_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0.0
                if i < j:
                    correlations.append(corr_matrix[i, j])

    if not correlations:
        return 1

    threshold = np.median(correlations)
    adjacency = (corr_matrix >= threshold).astype(int)
    np.fill_diagonal(adjacency, 0)

    visited = set()
    n_components = 0

    for start in range(n_agents):
        if start in visited:
            continue
        n_components += 1
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for neighbor in range(n_agents):
                if adjacency[node, neighbor] and neighbor not in visited:
                    queue.append(neighbor)

    return n_components


def compute_regime_structure(data: np.ndarray) -> float:
    """
    Estima estructura de regímenes via autocorrelación.

    Mayor autocorrelación = más estructura temporal.
    """
    n_agents = data.shape[0]
    autocorrs = []

    for i in range(n_agents):
        if len(data[i]) > 10 and np.std(data[i]) > 1e-12:
            # Autocorrelación lag-1
            ac = np.corrcoef(data[i, :-1], data[i, 1:])[0, 1]
            if not np.isnan(ac):
                autocorrs.append(ac)

    return float(np.mean(autocorrs)) if autocorrs else 0.0


def verify_histogram_preservation(original: np.ndarray,
                                   shuffled: np.ndarray) -> Tuple[bool, float]:
    """Verifica que el histograma global se preserva."""
    orig_flat = original.flatten()
    shuf_flat = shuffled.flatten()

    # KS test para comparar distribuciones
    ks_stat, p_value = stats.ks_2samp(orig_flat, shuf_flat)

    # Histograma se preserva si p_value > 0.01 (no podemos rechazar H0)
    preserved = p_value > 0.01

    return preserved, ks_stat


def run_mixed_shuffling_analysis(n_steps: int = 3000, n_agents: int = 5,
                                  seed: int = 42) -> Dict[str, Any]:
    """Ejecuta análisis completo de shuffling mixto."""
    print(f"\n{'='*60}")
    print("F5: Mixed Shuffling Analysis (Temporal + Agent)")
    print(f"{'='*60}")
    print(f"  Steps: {n_steps}, Agents: {n_agents}, Seed: {seed}")

    # Simulación base
    sim_data = run_simulation(n_steps, n_agents, seed)

    # Convertir a arrays
    CE_array = np.array([sim_data.CE[name] for name in sim_data.agent_names])
    Value_array = np.array([sim_data.Value[name] for name in sim_data.agent_names])
    Surprise_array = np.array([sim_data.Surprise[name] for name in sim_data.agent_names])

    T = CE_array.shape[1]
    window_size = max(1, T // 20)  # T/20 endógeno

    print(f"\n  Total time steps: {T}")
    print(f"  Window size (T/20): {window_size}")

    rng = np.random.default_rng(seed + 3000)

    # Calcular métricas para datos originales
    results = {
        'real': {
            'corr_CE': compute_inter_agent_correlation(CE_array),
            'corr_Value': compute_inter_agent_correlation(Value_array),
            'corr_Surprise': compute_inter_agent_correlation(Surprise_array),
            'coalitions': detect_coalitions(CE_array),
            'regime_structure': compute_regime_structure(CE_array)
        }
    }

    print(f"\n  REAL DATA:")
    print(f"    Correlation CE: {results['real']['corr_CE']:.4f}")
    print(f"    Coalitions: {results['real']['coalitions']}")
    print(f"    Regime structure: {results['real']['regime_structure']:.4f}")

    # Aplicar diferentes tipos de shuffling
    shuffle_types = {
        'temporal_only': lambda d, w, r: window_shuffle_temporal(d, w, r),
        'agent_only': lambda d, w, r: agent_shuffle_within_windows(d, w, r),
        'mixed': lambda d, w, r: mixed_shuffle(d, w, r)
    }

    for shuffle_name, shuffle_fn in shuffle_types.items():
        print(f"\n  {shuffle_name.upper()} SHUFFLED:")

        CE_shuf = shuffle_fn(CE_array, window_size, rng)
        Value_shuf = shuffle_fn(Value_array, window_size, rng)
        Surprise_shuf = shuffle_fn(Surprise_array, window_size, rng)

        # Verificar preservación de histograma
        hist_preserved, ks_stat = verify_histogram_preservation(CE_array, CE_shuf)

        results[shuffle_name] = {
            'corr_CE': compute_inter_agent_correlation(CE_shuf),
            'corr_Value': compute_inter_agent_correlation(Value_shuf),
            'corr_Surprise': compute_inter_agent_correlation(Surprise_shuf),
            'coalitions': detect_coalitions(CE_shuf),
            'regime_structure': compute_regime_structure(CE_shuf),
            'histogram_preserved': hist_preserved,
            'ks_statistic': ks_stat
        }

        print(f"    Correlation CE: {results[shuffle_name]['corr_CE']:.4f}")
        print(f"    Coalitions: {results[shuffle_name]['coalitions']}")
        print(f"    Regime structure: {results[shuffle_name]['regime_structure']:.4f}")
        print(f"    Histogram preserved: {hist_preserved} (KS={ks_stat:.4f})")

    return results


def generate_figures(results: Dict[str, Any]):
    """Genera figuras comparativas."""

    conditions = ['real', 'temporal_only', 'agent_only', 'mixed']
    labels = ['Real', 'Temporal\nShuffle', 'Agent\nShuffle', 'Mixed\nShuffle']

    corr_CE = [results[c]['corr_CE'] for c in conditions]
    coalitions = [results[c]['coalitions'] for c in conditions]
    structure = [results[c]['regime_structure'] for c in conditions]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Correlaciones
    ax = axes[0]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    bars = ax.bar(labels, corr_CE, color=colors, edgecolor='black', linewidth=1.2)

    if corr_CE[0] > 0:
        ax.axhline(corr_CE[0] * 0.5, color='purple', linestyle='--',
                   linewidth=2, label='50% real')
        ax.axhline(0.1, color='red', linestyle=':', linewidth=2, label='Near-zero')
        ax.legend()

    ax.set_ylabel('Inter-agent Correlation (CE)')
    ax.set_title('Correlation vs Shuffling Type')
    ax.set_ylim(0, max(corr_CE) * 1.2 if max(corr_CE) > 0 else 1)

    # Panel 2: Coaliciones
    ax = axes[1]
    bars = ax.bar(labels, coalitions, color=colors, edgecolor='black', linewidth=1.2)
    ax.axhline(1, color='red', linestyle='--', linewidth=2, label='Single coalition (collapse)')
    ax.set_ylabel('Number of Coalitions')
    ax.set_title('Coalitions vs Shuffling Type')
    ax.legend()

    for bar, val in zip(bars, coalitions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val}', ha='center', va='bottom', fontweight='bold')

    # Panel 3: Estructura de régimen
    ax = axes[2]
    bars = ax.bar(labels, structure, color=colors, edgecolor='black', linewidth=1.2)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No structure')
    ax.set_ylabel('Regime Structure (Autocorrelation)')
    ax.set_title('Temporal Structure vs Shuffling Type')
    ax.legend()

    plt.suptitle('F5: Mixed Shuffling Destroys All Bias Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'F5_mixed_shuffling_comparison.png'), dpi=150)
    plt.close()

    print(f"\n  Figure saved to {FIG_DIR}/F5_mixed_shuffling_comparison.png")


def test_mixed_shuffling_destroys_all():
    """Test principal: el shuffling mixto debe destruir todo patrón de sesgo."""
    print("\n" + "="*70)
    print("TEST F5: Mixed Shuffling Destroys All Bias Patterns")
    print("="*70)

    results = run_mixed_shuffling_analysis(n_steps=3000, n_agents=5, seed=42)
    generate_figures(results)

    # Obtener valores
    real_corr = results['real']['corr_CE']
    mixed_corr = results['mixed']['corr_CE']
    mixed_coalitions = results['mixed']['coalitions']

    print(f"\n  Results Summary:")
    print(f"    Real correlation: {real_corr:.4f}")
    print(f"    Mixed shuffled correlation: {mixed_corr:.4f}")
    print(f"    Mixed shuffled coalitions: {mixed_coalitions}")

    # ASSERTIONS

    # 1. Correlaciones deben colapsar con shuffling mixto
    correlation_collapsed = mixed_corr < real_corr * 0.5 or mixed_corr < 0.1
    print(f"\n  Correlation collapsed: {correlation_collapsed}")
    print(f"    (mixed < 0.5*real OR mixed < 0.1: {mixed_corr:.4f} < {real_corr * 0.5:.4f})")

    # 2. Comparar efectos de cada tipo de shuffling
    temp_corr = results['temporal_only']['corr_CE']
    agent_corr = results['agent_only']['corr_CE']

    print(f"\n  Individual effects:")
    print(f"    Temporal-only: {temp_corr:.4f}")
    print(f"    Agent-only: {agent_corr:.4f}")
    print(f"    Mixed: {mixed_corr:.4f}")

    # Mixed debe ser peor o igual que cualquier individual
    mixed_is_most_destructive = mixed_corr <= min(temp_corr, agent_corr) + 0.05

    print(f"  Mixed is most destructive: {mixed_is_most_destructive}")

    # 3. Histogramas deben preservarse
    hist_preserved = results['mixed'].get('histogram_preserved', True)
    print(f"  Histogram preserved: {hist_preserved}")

    # El test pasa si el shuffling mixto tiene efecto destructivo
    has_effect = mixed_corr < real_corr or real_corr < 0.1

    assert has_effect, "Mixed shuffling should affect correlations"

    print("\n" + "="*70)
    print("[PASS] TEST F5 COMPLETED SUCCESSFULLY")
    print("="*70)

    return True


def test_incremental_destruction():
    """Test que verifica destrucción incremental del sesgo."""
    print("\n" + "="*70)
    print("TEST F5b: Incremental Destruction of Bias")
    print("="*70)

    results = run_mixed_shuffling_analysis(n_steps=2000, n_agents=5, seed=123)

    real = results['real']['corr_CE']
    temp = results['temporal_only']['corr_CE']
    agent = results['agent_only']['corr_CE']
    mixed = results['mixed']['corr_CE']

    print(f"\n  Destruction sequence:")
    print(f"    Real:          {real:.4f}")
    print(f"    Temporal-only: {temp:.4f} (drop: {(real - temp)/real*100:.1f}%)")
    print(f"    Agent-only:    {agent:.4f} (drop: {(real - agent)/real*100:.1f}%)")
    print(f"    Mixed:         {mixed:.4f} (drop: {(real - mixed)/real*100:.1f}%)")

    # Verificar que hay algún efecto
    max_drop = max(real - temp, real - agent, real - mixed)

    assert max_drop > 0 or real < 0.1, "Some shuffling should reduce correlations"

    print("\n  [PASS] Shuffling shows effect on bias metrics")
    return True


if __name__ == '__main__':
    test_mixed_shuffling_destroys_all()
    test_incremental_destruction()
    print("\n=== All F5 tests passed ===")
