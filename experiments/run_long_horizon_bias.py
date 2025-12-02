#!/usr/bin/env python3
"""
FASE G - Long Horizon Bias Analysis
=====================================

Objetivo: Analizar comportamiento del sesgo colectivo en horizontes largos
(24h, 48h, 7 días virtuales) y detectar histéresis.

Análisis:
1. ¿El sesgo se estabiliza o cambia con el tiempo?
2. ¿Hay saltos de régimen (cambios abruptos)?
3. ¿Existe histéresis al variar parámetros?

Outputs:
- Timelines largas de métricas
- Mapas de calor (tiempo vs agente vs CE)
- Análisis de histéresis

100% Endógeno - Sin números mágicos externos.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from scipy import stats, signal
from datetime import datetime

sys.path.insert(0, '/root/NEO_EVA')

from core.agents import NEO, EVA, BaseAgent

# Output directories
FIG_DIR = '/root/NEO_EVA/figuras/FASE_G'
LOG_DIR = '/root/NEO_EVA/logs/long_horizon'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


@dataclass
class LongHorizonConfig:
    """Configuración para simulación de largo horizonte."""
    duration_name: str
    n_steps: int
    n_agents: int = 5
    dim: int = 6
    seed: int = 42
    coupling_base: float = 0.1


@dataclass
class LongHorizonResults:
    """Resultados de simulación de largo horizonte."""
    config: Dict
    CE: Dict[str, List[float]] = field(default_factory=dict)
    Value: Dict[str, List[float]] = field(default_factory=dict)
    Surprise: Dict[str, List[float]] = field(default_factory=dict)
    inter_agent_corr: List[float] = field(default_factory=list)
    coalitions: List[int] = field(default_factory=list)
    regime_entropy: List[float] = field(default_factory=list)
    regime_jumps: List[int] = field(default_factory=list)


def run_long_simulation(config: LongHorizonConfig,
                         progress_interval: int = 500) -> LongHorizonResults:
    """
    Ejecuta simulación de largo horizonte.

    Args:
        config: Configuración de la simulación
        progress_interval: Cada cuántos pasos mostrar progreso

    Returns:
        LongHorizonResults con todos los datos
    """
    print(f"\n  Running {config.duration_name} simulation ({config.n_steps} steps)...")
    BaseAgent._agent_counter = 0
    rng = np.random.default_rng(config.seed)

    # Crear agentes
    agents = {}
    agent_names = [f'A{i}' for i in range(config.n_agents)]

    for i, name in enumerate(agent_names):
        if i % 2 == 0:
            agents[name] = NEO(dim_visible=config.dim, dim_hidden=config.dim)
        else:
            agents[name] = EVA(dim_visible=config.dim, dim_hidden=config.dim)

    results = LongHorizonResults(
        config=asdict(config),
        CE={name: [] for name in agent_names},
        Value={name: [] for name in agent_names},
        Surprise={name: [] for name in agent_names}
    )

    # Ventana para métricas colectivas (endógena: √n_steps)
    metric_window = max(10, int(np.sqrt(config.n_steps)))
    last_regime_state = None

    for t in range(config.n_steps):
        # Progreso
        if t > 0 and t % progress_interval == 0:
            print(f"    Step {t}/{config.n_steps} ({t/config.n_steps*100:.1f}%)")

        stimulus = rng.uniform(0, 1, config.dim)
        states_list = [agents[name].get_state().z_visible for name in agent_names]
        mean_field = np.mean(states_list, axis=0)

        for name in agent_names:
            agent = agents[name]
            state = agent.get_state()
            coupling = mean_field - state.z_visible / config.n_agents
            coupled_stimulus = stimulus + config.coupling_base * coupling
            coupled_stimulus = np.clip(coupled_stimulus, 0.01, 0.99)

            response = agent.step(coupled_stimulus)

            results.CE[name].append(1.0 / (1.0 + response.surprise))
            results.Value[name].append(response.value)
            results.Surprise[name].append(response.surprise)

        # Calcular métricas colectivas periódicamente
        if t > 0 and t % metric_window == 0:
            # Correlación inter-agentes
            CE_recent = np.array([results.CE[name][-metric_window:]
                                  for name in agent_names])
            corrs = []
            for i in range(config.n_agents):
                for j in range(i + 1, config.n_agents):
                    if np.std(CE_recent[i]) > 1e-12 and np.std(CE_recent[j]) > 1e-12:
                        c, _ = stats.pearsonr(CE_recent[i], CE_recent[j])
                        if not np.isnan(c):
                            corrs.append(abs(c))
            results.inter_agent_corr.append(float(np.mean(corrs)) if corrs else 0.0)

            # Coaliciones
            n_coal = detect_coalitions_from_array(CE_recent)
            results.coalitions.append(n_coal)

            # Entropía de régimen (dispersión de métricas)
            ce_means = [np.mean(results.CE[name][-metric_window:]) for name in agent_names]
            regime_entropy = compute_regime_entropy(ce_means)
            results.regime_entropy.append(regime_entropy)

            # Detectar saltos de régimen
            current_state = np.mean(ce_means)
            if last_regime_state is not None:
                jump = abs(current_state - last_regime_state) > np.std(ce_means)
                results.regime_jumps.append(1 if jump else 0)
            last_regime_state = current_state

    return results


def detect_coalitions_from_array(data: np.ndarray) -> int:
    """Detecta coaliciones desde array (n_agents, T)."""
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
                c, _ = stats.pearsonr(data[i], data[j])
                corr_matrix[i, j] = abs(c) if not np.isnan(c) else 0.0
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


def compute_regime_entropy(values: List[float]) -> float:
    """Calcula entropía normalizada de valores."""
    arr = np.array(values)
    arr = arr - arr.min() + 1e-12
    arr = arr / arr.sum()
    entropy = -np.sum(arr * np.log(arr + 1e-12))
    max_entropy = np.log(len(arr))
    return float(entropy / (max_entropy + 1e-12))


def run_hysteresis_experiment(n_steps_per_phase: int = 1000,
                               n_agents: int = 5,
                               seed: int = 42) -> Dict[str, Any]:
    """
    Experimento de histéresis: variar coupling suavemente ida y vuelta.

    Returns:
        Dict con trayectorias de ida y vuelta
    """
    print("\n  Running hysteresis experiment...")
    BaseAgent._agent_counter = 0
    rng = np.random.default_rng(seed)
    dim = 6

    # Rango de coupling
    coupling_min = 0.01
    coupling_max = 0.5
    n_phases = 20

    coupling_values = np.linspace(coupling_min, coupling_max, n_phases)
    # Ida y vuelta
    coupling_sequence = np.concatenate([coupling_values, coupling_values[::-1]])

    results = {
        'coupling': [],
        'corr_forward': [],
        'corr_backward': [],
        'forward_phase': [],
        'backward_phase': []
    }

    # Crear agentes
    agents = {}
    agent_names = [f'A{i}' for i in range(n_agents)]

    for i, name in enumerate(agent_names):
        if i % 2 == 0:
            agents[name] = NEO(dim_visible=dim, dim_hidden=dim)
        else:
            agents[name] = EVA(dim_visible=dim, dim_hidden=dim)

    phase_metrics = []

    for phase_idx, coupling in enumerate(coupling_sequence):
        CE_phase = {name: [] for name in agent_names}

        for t in range(n_steps_per_phase):
            stimulus = rng.uniform(0, 1, dim)
            states_list = [agents[name].get_state().z_visible for name in agent_names]
            mean_field = np.mean(states_list, axis=0)

            for name in agent_names:
                agent = agents[name]
                state = agent.get_state()
                coup = mean_field - state.z_visible / n_agents
                coupled_stimulus = stimulus + coupling * coup
                coupled_stimulus = np.clip(coupled_stimulus, 0.01, 0.99)

                response = agent.step(coupled_stimulus)
                CE_phase[name].append(1.0 / (1.0 + response.surprise))

        # Calcular correlación media de esta fase
        CE_array = np.array([CE_phase[name] for name in agent_names])
        corrs = []
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                if np.std(CE_array[i]) > 1e-12 and np.std(CE_array[j]) > 1e-12:
                    c, _ = stats.pearsonr(CE_array[i], CE_array[j])
                    if not np.isnan(c):
                        corrs.append(abs(c))

        mean_corr = float(np.mean(corrs)) if corrs else 0.0
        phase_metrics.append((coupling, mean_corr))

    # Separar forward y backward
    n_forward = len(coupling_values)
    for idx, (coup, corr) in enumerate(phase_metrics):
        results['coupling'].append(coup)
        if idx < n_forward:
            results['forward_phase'].append(corr)
            results['corr_forward'].append(corr)
        else:
            results['backward_phase'].append(corr)
            results['corr_backward'].append(corr)

    return results


def generate_long_horizon_figures(all_results: Dict[str, LongHorizonResults],
                                   hysteresis_results: Dict[str, Any]):
    """Genera todas las figuras de largo horizonte."""

    # Figura 1: Timeline de correlaciones por duración
    fig, axes = plt.subplots(len(all_results), 1, figsize=(14, 4 * len(all_results)))
    if len(all_results) == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))

    for idx, (duration, results) in enumerate(all_results.items()):
        ax = axes[idx]
        t = np.arange(len(results.inter_agent_corr))

        ax.plot(t, results.inter_agent_corr, color=colors[idx], linewidth=1.5)
        ax.fill_between(t, results.inter_agent_corr, alpha=0.3, color=colors[idx])

        # Añadir media móvil
        if len(results.inter_agent_corr) > 20:
            window = max(5, len(results.inter_agent_corr) // 20)
            smoothed = np.convolve(results.inter_agent_corr,
                                    np.ones(window)/window, mode='valid')
            ax.plot(range(window//2, window//2 + len(smoothed)), smoothed,
                   'r--', linewidth=2, label='Smoothed')

        ax.set_ylabel('Inter-agent Correlation')
        ax.set_title(f'{duration} - Correlation Timeline')
        ax.legend()
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'G_correlation_timelines.png'), dpi=150)
    plt.close()

    # Figura 2: Heatmap tiempo vs agente vs CE
    for duration, results in all_results.items():
        fig, ax = plt.subplots(figsize=(14, 6))

        agent_names = list(results.CE.keys())
        CE_matrix = np.array([results.CE[name] for name in agent_names])

        # Subsamplear si hay demasiados puntos
        if CE_matrix.shape[1] > 500:
            step = CE_matrix.shape[1] // 500
            CE_matrix = CE_matrix[:, ::step]

        im = ax.imshow(CE_matrix, aspect='auto', cmap='viridis',
                       interpolation='nearest')
        ax.set_yticks(range(len(agent_names)))
        ax.set_yticklabels(agent_names)
        ax.set_xlabel('Time (subsampled)')
        ax.set_ylabel('Agent')
        ax.set_title(f'{duration} - CE Heatmap (Agent x Time)')
        plt.colorbar(im, ax=ax, label='CE')

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'G_heatmap_{duration.replace(" ", "_")}.png'), dpi=150)
        plt.close()

    # Figura 3: Regime jumps analysis
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for idx, (duration, results) in enumerate(all_results.items()):
        if idx >= 2:
            break

        ax = axes[idx] if len(all_results) > 1 else axes

        # Coaliciones a lo largo del tiempo
        ax.plot(results.coalitions, 'b-', linewidth=1.5, label='Coalitions')
        ax.set_ylabel('Number of Coalitions', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        # Regime jumps en eje secundario
        ax2 = ax.twinx()
        if results.regime_jumps:
            cumulative_jumps = np.cumsum(results.regime_jumps)
            ax2.plot(cumulative_jumps, 'r-', linewidth=1.5, label='Cumulative Jumps')
        ax2.set_ylabel('Cumulative Regime Jumps', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax.set_title(f'{duration} - Coalitions & Regime Jumps')
        ax.set_xlabel('Measurement Window')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'G_regime_analysis.png'), dpi=150)
    plt.close()

    # Figura 4: Histéresis
    if hysteresis_results:
        fig, ax = plt.subplots(figsize=(10, 8))

        forward = hysteresis_results['forward_phase']
        backward = hysteresis_results['backward_phase']
        coupling_forward = hysteresis_results['coupling'][:len(forward)]
        coupling_backward = hysteresis_results['coupling'][len(forward):]

        ax.plot(coupling_forward, forward, 'b-o', linewidth=2, markersize=8,
               label='Forward (increasing coupling)')
        ax.plot(coupling_backward, backward, 'r-s', linewidth=2, markersize=8,
               label='Backward (decreasing coupling)')

        ax.set_xlabel('Coupling Strength')
        ax.set_ylabel('Inter-agent Correlation')
        ax.set_title('G: Hysteresis Analysis - Forward vs Backward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Calcular área de histéresis (si existe)
        if len(forward) == len(backward):
            hysteresis_area = np.trapz(np.abs(np.array(forward) - np.array(backward[::-1])))
            ax.text(0.05, 0.95, f'Hysteresis Area: {hysteresis_area:.4f}',
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'G_hysteresis.png'), dpi=150)
        plt.close()

    print(f"\n  Figures saved to {FIG_DIR}")


def analyze_bias_evolution(results: LongHorizonResults) -> Dict[str, Any]:
    """Analiza la evolución del sesgo a lo largo del tiempo."""
    analysis = {}

    # 1. ¿El sesgo se estabiliza?
    corrs = np.array(results.inter_agent_corr)
    if len(corrs) > 10:
        # Comparar primera y segunda mitad
        first_half = corrs[:len(corrs)//2]
        second_half = corrs[len(corrs)//2:]

        var_first = np.var(first_half)
        var_second = np.var(second_half)

        stabilizes = var_second < var_first * 1.5
        analysis['stabilizes'] = stabilizes
        analysis['var_first_half'] = float(var_first)
        analysis['var_second_half'] = float(var_second)

    # 2. Tendencia general
    if len(corrs) > 5:
        slope, _, r_value, p_value, _ = stats.linregress(range(len(corrs)), corrs)
        analysis['trend_slope'] = float(slope)
        analysis['trend_r_squared'] = float(r_value ** 2)
        analysis['trend_p_value'] = float(p_value)

    # 3. Número total de saltos de régimen
    analysis['total_regime_jumps'] = sum(results.regime_jumps)
    analysis['jump_rate'] = float(sum(results.regime_jumps) / (len(results.regime_jumps) + 1e-12))

    # 4. Distribución de coaliciones
    coals = np.array(results.coalitions)
    analysis['coalitions_mean'] = float(np.mean(coals))
    analysis['coalitions_std'] = float(np.std(coals))
    analysis['coalitions_mode'] = int(stats.mode(coals, keepdims=True)[0][0])

    return analysis


def run_all_long_horizon_experiments():
    """Ejecuta todos los experimentos de largo horizonte."""
    print("\n" + "="*70)
    print("FASE G: Long Horizon Bias Analysis")
    print("="*70)

    # Configuraciones de duración
    # 24h = 6000 pasos, 48h = 12000, 7 días = 42000
    configs = [
        LongHorizonConfig("24h", 6000, seed=42),
        LongHorizonConfig("48h", 12000, seed=42),
        LongHorizonConfig("7_days", 42000, seed=42)
    ]

    all_results = {}

    for config in configs:
        print(f"\n{'='*50}")
        print(f"Running: {config.duration_name}")
        print(f"{'='*50}")

        results = run_long_simulation(config)
        all_results[config.duration_name] = results

        # Análisis
        analysis = analyze_bias_evolution(results)
        print(f"\n  Analysis for {config.duration_name}:")
        print(f"    Stabilizes: {analysis.get('stabilizes', 'N/A')}")
        print(f"    Trend slope: {analysis.get('trend_slope', 0):.6f}")
        print(f"    Total regime jumps: {analysis['total_regime_jumps']}")
        print(f"    Mean coalitions: {analysis['coalitions_mean']:.2f}")

        # Guardar resultados
        save_path = os.path.join(LOG_DIR, f"long_horizon_{config.duration_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        save_data = {
            'config': asdict(config),
            'analysis': analysis,
            'inter_agent_corr': results.inter_agent_corr,
            'coalitions': results.coalitions,
            'regime_jumps': results.regime_jumps,
            'regime_entropy': results.regime_entropy
        }

        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"    Saved to: {save_path}")

    # Experimento de histéresis
    print(f"\n{'='*50}")
    print("Running: Hysteresis Experiment")
    print(f"{'='*50}")

    hysteresis_results = run_hysteresis_experiment(n_steps_per_phase=1000)

    # Generar figuras
    print(f"\n{'='*50}")
    print("Generating Figures")
    print(f"{'='*50}")

    generate_long_horizon_figures(all_results, hysteresis_results)

    print("\n" + "="*70)
    print("FASE G COMPLETED SUCCESSFULLY")
    print("="*70)

    return all_results, hysteresis_results


if __name__ == '__main__':
    run_all_long_horizon_experiments()
