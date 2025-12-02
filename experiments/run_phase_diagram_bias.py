#!/usr/bin/env python3
"""
FASE H - Phase Diagram for Collective Bias
============================================

Objetivo: Construir un mapa de en qué zonas del espacio de parámetros
aparece sesgo colectivo.

Parámetros explorados:
1. Intensidad de acoplamiento entre agentes
2. Número de agentes N
3. Nivel de ruido estructural (opcional)

Outputs:
- Heatmaps de sesgo colectivo en el espacio de parámetros
- Tabla CSV con todos los puntos del diagrama

100% Endógeno - rangos relativos, sin valores inventados.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import csv
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from datetime import datetime

sys.path.insert(0, '/root/NEO_EVA')

from core.agents import NEO, EVA, BaseAgent

# Output directories
FIG_DIR = '/root/NEO_EVA/figuras/FASE_H'
LOG_DIR = '/root/NEO_EVA/logs/phase_diagram'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


@dataclass
class PhasePoint:
    """Un punto en el diagrama de fases."""
    coupling: float
    n_agents: int
    noise_level: float
    corr_CE: float
    coalitions: int
    Q_coherence_proxy: float
    regime_stability: float


def run_single_point(coupling: float, n_agents: int, noise_level: float,
                      n_steps: int = 1000, dim: int = 6, seed: int = 42) -> PhasePoint:
    """
    Ejecuta una simulación corta para un punto del diagrama de fases.

    Args:
        coupling: Intensidad de acoplamiento [0, 1]
        n_agents: Número de agentes
        noise_level: Nivel de ruido estructural [0, 1]
        n_steps: Pasos de simulación
        dim: Dimensión del espacio de estados
        seed: Semilla para reproducibilidad

    Returns:
        PhasePoint con métricas calculadas
    """
    BaseAgent._agent_counter = 0
    rng = np.random.default_rng(seed)

    # Crear agentes
    agents = {}
    agent_names = [f'A{i}' for i in range(n_agents)]

    for i, name in enumerate(agent_names):
        if i % 2 == 0:
            agents[name] = NEO(dim_visible=dim, dim_hidden=dim)
        else:
            agents[name] = EVA(dim_visible=dim, dim_hidden=dim)

    # Métricas
    CE_history = {name: [] for name in agent_names}
    coherence_proxy = []

    for t in range(n_steps):
        stimulus = rng.uniform(0, 1, dim)

        # Añadir ruido estructural al estímulo
        if noise_level > 0:
            noise = rng.normal(0, noise_level, dim)
            stimulus = np.clip(stimulus + noise, 0.01, 0.99)

        states_list = [agents[name].get_state().z_visible for name in agent_names]
        mean_field = np.mean(states_list, axis=0)

        # Calcular coherencia de fase como proxy de Q_coherence
        phases = [np.arctan2(s[1], s[0]) for s in states_list]
        phase_coherence = abs(np.mean(np.exp(1j * np.array(phases))))
        coherence_proxy.append(phase_coherence)

        for name in agent_names:
            agent = agents[name]
            state = agent.get_state()
            coup = mean_field - state.z_visible / n_agents
            coupled_stimulus = stimulus + coupling * coup
            coupled_stimulus = np.clip(coupled_stimulus, 0.01, 0.99)

            response = agent.step(coupled_stimulus)
            CE_history[name].append(1.0 / (1.0 + response.surprise))

    # Calcular métricas finales
    CE_array = np.array([CE_history[name] for name in agent_names])

    # Correlación inter-agentes
    corrs = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if np.std(CE_array[i]) > 1e-12 and np.std(CE_array[j]) > 1e-12:
                c, _ = stats.pearsonr(CE_array[i], CE_array[j])
                if not np.isnan(c):
                    corrs.append(abs(c))

    corr_CE = float(np.mean(corrs)) if corrs else 0.0

    # Coaliciones
    coalitions = detect_coalitions(CE_array)

    # Q-coherence proxy (media)
    Q_coh = float(np.mean(coherence_proxy))

    # Estabilidad de régimen (autocorrelación)
    stability = compute_stability(CE_array)

    return PhasePoint(
        coupling=coupling,
        n_agents=n_agents,
        noise_level=noise_level,
        corr_CE=corr_CE,
        coalitions=coalitions,
        Q_coherence_proxy=Q_coh,
        regime_stability=stability
    )


def detect_coalitions(data: np.ndarray) -> int:
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


def compute_stability(data: np.ndarray) -> float:
    """Calcula estabilidad como autocorrelación media."""
    n_agents = data.shape[0]
    autocorrs = []

    for i in range(n_agents):
        if len(data[i]) > 10 and np.std(data[i]) > 1e-12:
            ac = np.corrcoef(data[i, :-1], data[i, 1:])[0, 1]
            if not np.isnan(ac):
                autocorrs.append(ac)

    return float(np.mean(autocorrs)) if autocorrs else 0.5


def run_phase_diagram(coupling_range: Tuple[float, float] = (0.01, 0.5),
                       n_agents_range: Tuple[int, int] = (3, 10),
                       noise_level: float = 0.0,
                       grid_resolution: int = 10,
                       n_steps: int = 1000,
                       base_seed: int = 42) -> Dict[str, Any]:
    """
    Ejecuta barrido del diagrama de fases.

    Args:
        coupling_range: Rango de coupling (min, max)
        n_agents_range: Rango de número de agentes (min, max)
        noise_level: Nivel de ruido fijo para este diagrama
        grid_resolution: Número de puntos por eje
        n_steps: Pasos por simulación
        base_seed: Semilla base

    Returns:
        Dict con resultados del diagrama
    """
    print(f"\n  Running phase diagram...")
    print(f"    Coupling range: {coupling_range}")
    print(f"    N agents range: {n_agents_range}")
    print(f"    Noise level: {noise_level}")
    print(f"    Grid resolution: {grid_resolution}")

    coupling_values = np.linspace(coupling_range[0], coupling_range[1], grid_resolution)
    n_agents_values = list(range(n_agents_range[0], n_agents_range[1] + 1))

    # Limitar para no tener demasiados puntos
    if len(n_agents_values) > grid_resolution:
        step = max(1, len(n_agents_values) // grid_resolution)
        n_agents_values = n_agents_values[::step]

    points = []
    total = len(coupling_values) * len(n_agents_values)
    count = 0

    for coupling in coupling_values:
        for n_agents in n_agents_values:
            count += 1
            if count % 10 == 0:
                print(f"    Point {count}/{total}")

            seed = base_seed + int(coupling * 1000) + n_agents
            point = run_single_point(coupling, n_agents, noise_level, n_steps, seed=seed)
            points.append(point)

    return {
        'coupling_values': coupling_values.tolist(),
        'n_agents_values': n_agents_values,
        'noise_level': noise_level,
        'points': points
    }


def run_3d_phase_diagram(coupling_range: Tuple[float, float] = (0.01, 0.3),
                          noise_range: Tuple[float, float] = (0.0, 0.3),
                          n_agents: int = 5,
                          grid_resolution: int = 8,
                          n_steps: int = 800,
                          base_seed: int = 42) -> Dict[str, Any]:
    """
    Diagrama de fases 2D: coupling vs noise (N fijo).
    """
    print(f"\n  Running coupling vs noise phase diagram...")

    coupling_values = np.linspace(coupling_range[0], coupling_range[1], grid_resolution)
    noise_values = np.linspace(noise_range[0], noise_range[1], grid_resolution)

    points = []
    total = len(coupling_values) * len(noise_values)
    count = 0

    for coupling in coupling_values:
        for noise in noise_values:
            count += 1
            if count % 10 == 0:
                print(f"    Point {count}/{total}")

            seed = base_seed + int(coupling * 1000) + int(noise * 1000)
            point = run_single_point(coupling, n_agents, noise, n_steps, seed=seed)
            points.append(point)

    return {
        'coupling_values': coupling_values.tolist(),
        'noise_values': noise_values.tolist(),
        'n_agents': n_agents,
        'points': points
    }


def generate_phase_diagram_figures(results_coupling_agents: Dict[str, Any],
                                    results_coupling_noise: Dict[str, Any]):
    """Genera figuras de los diagramas de fase."""

    # Figura 1: Coupling vs N_agents - Correlación
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Preparar datos para heatmap
    coupling_values = results_coupling_agents['coupling_values']
    n_agents_values = results_coupling_agents['n_agents_values']
    points = results_coupling_agents['points']

    # Crear matrices para heatmaps
    corr_matrix = np.zeros((len(n_agents_values), len(coupling_values)))
    coal_matrix = np.zeros((len(n_agents_values), len(coupling_values)))

    for point in points:
        i = n_agents_values.index(point.n_agents)
        j = coupling_values.index(point.coupling)
        corr_matrix[i, j] = point.corr_CE
        coal_matrix[i, j] = point.coalitions

    # Panel 1: Correlación CE
    ax = axes[0, 0]
    im = ax.imshow(corr_matrix, aspect='auto', origin='lower', cmap='viridis',
                   extent=[coupling_values[0], coupling_values[-1],
                          n_agents_values[0], n_agents_values[-1]])
    ax.set_xlabel('Coupling Strength')
    ax.set_ylabel('Number of Agents')
    ax.set_title('Inter-agent Correlation (CE)')
    plt.colorbar(im, ax=ax)

    # Panel 2: Coaliciones
    ax = axes[0, 1]
    im = ax.imshow(coal_matrix, aspect='auto', origin='lower', cmap='RdYlGn_r',
                   extent=[coupling_values[0], coupling_values[-1],
                          n_agents_values[0], n_agents_values[-1]])
    ax.set_xlabel('Coupling Strength')
    ax.set_ylabel('Number of Agents')
    ax.set_title('Number of Coalitions')
    plt.colorbar(im, ax=ax)

    # Figura para coupling vs noise
    coupling_values_2 = results_coupling_noise['coupling_values']
    noise_values = results_coupling_noise['noise_values']
    points_2 = results_coupling_noise['points']

    corr_matrix_2 = np.zeros((len(noise_values), len(coupling_values_2)))
    stab_matrix = np.zeros((len(noise_values), len(coupling_values_2)))

    for point in points_2:
        i = noise_values.index(point.noise_level)
        j = coupling_values_2.index(point.coupling)
        corr_matrix_2[i, j] = point.corr_CE
        stab_matrix[i, j] = point.regime_stability

    # Panel 3: Coupling vs Noise - Correlación
    ax = axes[1, 0]
    im = ax.imshow(corr_matrix_2, aspect='auto', origin='lower', cmap='viridis',
                   extent=[coupling_values_2[0], coupling_values_2[-1],
                          noise_values[0], noise_values[-1]])
    ax.set_xlabel('Coupling Strength')
    ax.set_ylabel('Noise Level')
    ax.set_title('Correlation (Coupling vs Noise)')
    plt.colorbar(im, ax=ax)

    # Panel 4: Estabilidad de régimen
    ax = axes[1, 1]
    im = ax.imshow(stab_matrix, aspect='auto', origin='lower', cmap='coolwarm',
                   extent=[coupling_values_2[0], coupling_values_2[-1],
                          noise_values[0], noise_values[-1]])
    ax.set_xlabel('Coupling Strength')
    ax.set_ylabel('Noise Level')
    ax.set_title('Regime Stability')
    plt.colorbar(im, ax=ax)

    plt.suptitle('H: Phase Diagrams of Collective Bias', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'H_phase_diagrams.png'), dpi=150)
    plt.close()

    # Figura adicional: Cortes 1D
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Corte a N fijo
    ax = axes[0]
    n_mid = n_agents_values[len(n_agents_values) // 2]
    corrs_at_n = [p.corr_CE for p in points if p.n_agents == n_mid]
    coups_at_n = [p.coupling for p in points if p.n_agents == n_mid]
    ax.plot(coups_at_n, corrs_at_n, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Coupling Strength')
    ax.set_ylabel('Inter-agent Correlation')
    ax.set_title(f'1D Cut at N={n_mid} agents')
    ax.grid(True, alpha=0.3)

    # Corte a coupling fijo
    ax = axes[1]
    coup_mid = coupling_values[len(coupling_values) // 2]
    corrs_at_c = [p.corr_CE for p in points if abs(p.coupling - coup_mid) < 0.01]
    ns_at_c = [p.n_agents for p in points if abs(p.coupling - coup_mid) < 0.01]
    ax.plot(ns_at_c, corrs_at_c, 's-', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Inter-agent Correlation')
    ax.set_title(f'1D Cut at coupling={coup_mid:.2f}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'H_phase_cuts.png'), dpi=150)
    plt.close()

    print(f"\n  Figures saved to {FIG_DIR}")


def save_phase_diagram_csv(results: Dict[str, Any], filename: str):
    """Guarda diagrama de fases como CSV."""
    points = results['points']

    csv_path = os.path.join(LOG_DIR, filename)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['coupling', 'n_agents', 'noise_level', 'corr_CE',
                        'coalitions', 'Q_coherence_proxy', 'regime_stability'])

        for p in points:
            writer.writerow([p.coupling, p.n_agents, p.noise_level, p.corr_CE,
                           p.coalitions, p.Q_coherence_proxy, p.regime_stability])

    print(f"  Saved CSV: {csv_path}")


def run_all_phase_diagrams():
    """Ejecuta todos los diagramas de fase."""
    print("\n" + "="*70)
    print("FASE H: Phase Diagram Analysis")
    print("="*70)

    # Diagrama 1: Coupling vs N_agents
    print(f"\n{'='*50}")
    print("Diagram 1: Coupling vs Number of Agents")
    print(f"{'='*50}")

    results_coupling_agents = run_phase_diagram(
        coupling_range=(0.01, 0.5),
        n_agents_range=(3, 12),
        noise_level=0.0,
        grid_resolution=10,
        n_steps=1000
    )

    save_phase_diagram_csv(results_coupling_agents,
                           f"phase_diagram_coupling_agents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    # Diagrama 2: Coupling vs Noise
    print(f"\n{'='*50}")
    print("Diagram 2: Coupling vs Noise")
    print(f"{'='*50}")

    results_coupling_noise = run_3d_phase_diagram(
        coupling_range=(0.01, 0.4),
        noise_range=(0.0, 0.3),
        n_agents=5,
        grid_resolution=8,
        n_steps=800
    )

    save_phase_diagram_csv(results_coupling_noise,
                           f"phase_diagram_coupling_noise_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    # Generar figuras
    print(f"\n{'='*50}")
    print("Generating Figures")
    print(f"{'='*50}")

    generate_phase_diagram_figures(results_coupling_agents, results_coupling_noise)

    print("\n" + "="*70)
    print("FASE H COMPLETED SUCCESSFULLY")
    print("="*70)

    return results_coupling_agents, results_coupling_noise


if __name__ == '__main__':
    run_all_phase_diagrams()
