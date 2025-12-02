#!/usr/bin/env python3
"""
BÚSQUEDA DEL SWEET SPOT: Diversidad sin Colapso
=================================================

Objetivo: Encontrar la región del espacio de parámetros donde:
- Hay múltiples coaliciones (diversidad, no homogeneización)
- Las coaliciones son estables (no caos)
- No hay polarización extrema (las coaliciones interactúan)

Métricas clave:
- n_coaliciones: queremos > 1 pero no = N (fragmentación total)
- correlación intra-coalición: alta (cohesión interna)
- correlación inter-coalición: media (no aislamiento total, no fusión)
- estabilidad temporal: baja varianza en el tiempo

El "sweet spot" es donde la sociedad es diversa pero funcional.

100% Endógeno.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from scipy.ndimage import gaussian_filter
import json
from datetime import datetime

sys.path.insert(0, '/root/NEO_EVA')

from core.agents import NEO, EVA, BaseAgent

# Output
FIG_DIR = '/root/NEO_EVA/figuras/sweet_spot'
LOG_DIR = '/root/NEO_EVA/logs/sweet_spot'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


@dataclass
class SweetSpotMetrics:
    """Métricas para evaluar el sweet spot."""
    coupling: float
    noise: float
    n_agents: int

    n_coalitions: float  # Promedio temporal
    coalition_stability: float  # 1 - varianza normalizada
    intra_coalition_corr: float  # Cohesión interna
    inter_coalition_corr: float  # Conexión entre coaliciones

    diversity_index: float  # Derivado: qué tan diverso sin fragmentar
    health_index: float  # Índice compuesto de "sociedad funcional"


def run_simulation_for_sweet_spot(coupling: float, noise: float, n_agents: int = 7,
                                   n_steps: int = 1500, dim: int = 6,
                                   seed: int = 42) -> SweetSpotMetrics:
    """Ejecuta simulación y calcula métricas de sweet spot."""
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

    # Historiales
    CE_history = {name: [] for name in agent_names}
    coalition_history = []

    # Ventana para análisis
    analysis_window = max(50, n_steps // 10)

    for t in range(n_steps):
        # Estímulo con ruido
        stimulus = rng.uniform(0, 1, dim)
        if noise > 0:
            stimulus = stimulus + rng.normal(0, noise, dim)
            stimulus = np.clip(stimulus, 0.01, 0.99)

        states_list = [agents[name].get_state().z_visible for name in agent_names]
        mean_field = np.mean(states_list, axis=0)

        for name in agent_names:
            agent = agents[name]
            state = agent.get_state()
            coup = mean_field - state.z_visible / n_agents
            coupled_stimulus = stimulus + coupling * coup
            coupled_stimulus = np.clip(coupled_stimulus, 0.01, 0.99)

            response = agent.step(coupled_stimulus)
            CE_history[name].append(1.0 / (1.0 + response.surprise))

        # Detectar coaliciones periódicamente
        if t > analysis_window and t % (analysis_window // 5) == 0:
            CE_recent = np.array([CE_history[name][-analysis_window:]
                                  for name in agent_names])
            n_coal = detect_coalitions_detailed(CE_recent)
            coalition_history.append(n_coal)

    # Calcular métricas finales
    CE_array = np.array([CE_history[name] for name in agent_names])

    # Número medio de coaliciones y estabilidad
    if coalition_history:
        n_coalitions_mean = np.mean(coalition_history)
        coalition_stability = 1.0 - (np.std(coalition_history) / (np.mean(coalition_history) + 1e-12))
        coalition_stability = max(0, min(1, coalition_stability))
    else:
        n_coalitions_mean = 1.0
        coalition_stability = 1.0

    # Correlaciones intra e inter coalición
    intra_corr, inter_corr = compute_intra_inter_correlation(CE_array)

    # Índice de diversidad: queremos coaliciones > 1 pero no fragmentación total
    # Óptimo sería n_coaliciones ~ n_agents / 2 o 3
    optimal_coalitions = n_agents / 3
    diversity_index = 1.0 - abs(n_coalitions_mean - optimal_coalitions) / n_agents
    diversity_index = max(0, min(1, diversity_index))

    # Índice de salud: combinación de métricas
    # - Diversidad alta (no homogeneización)
    # - Estabilidad alta (no caos)
    # - Inter-correlación media (conexión sin fusión)
    # - Intra-correlación alta (cohesión interna)

    inter_corr_score = 1.0 - abs(inter_corr - 0.3)  # Óptimo ~0.3

    health_index = (
        0.3 * diversity_index +
        0.2 * coalition_stability +
        0.25 * inter_corr_score +
        0.25 * intra_corr
    )

    return SweetSpotMetrics(
        coupling=coupling,
        noise=noise,
        n_agents=n_agents,
        n_coalitions=n_coalitions_mean,
        coalition_stability=coalition_stability,
        intra_coalition_corr=intra_corr,
        inter_coalition_corr=inter_corr,
        diversity_index=diversity_index,
        health_index=health_index
    )


def detect_coalitions_detailed(data: np.ndarray) -> int:
    """Detecta coaliciones con más detalle."""
    n_agents = data.shape[0]
    if n_agents < 2:
        return 1

    # Matriz de correlaciones
    corr_matrix = np.zeros((n_agents, n_agents))
    correlations = []

    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif np.std(data[i]) > 1e-12 and np.std(data[j]) > 1e-12:
                c, _ = stats.pearsonr(data[i], data[j])
                corr_matrix[i, j] = c if not np.isnan(c) else 0.0
                if i < j:
                    correlations.append(abs(c) if not np.isnan(c) else 0.0)

    if not correlations:
        return 1

    # Threshold basado en percentil 60 (más estricto para detectar coaliciones reales)
    threshold = np.percentile(correlations, 60)

    adjacency = (np.abs(corr_matrix) >= threshold).astype(int)
    np.fill_diagonal(adjacency, 0)

    # BFS
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


def compute_intra_inter_correlation(CE_array: np.ndarray) -> Tuple[float, float]:
    """
    Calcula correlación intra-coalición e inter-coalición.

    Usa clustering simple para separar coaliciones primero.
    """
    n_agents = CE_array.shape[0]

    # Primero detectar coaliciones
    corr_matrix = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif np.std(CE_array[i]) > 1e-12 and np.std(CE_array[j]) > 1e-12:
                c, _ = stats.pearsonr(CE_array[i], CE_array[j])
                corr_matrix[i, j] = abs(c) if not np.isnan(c) else 0.0

    # Clustering simple: umbral en mediana
    threshold = np.median(corr_matrix[np.triu_indices(n_agents, k=1)])

    # Asignar coaliciones via componentes conectadas
    adjacency = (corr_matrix >= threshold).astype(int)
    np.fill_diagonal(adjacency, 0)

    coalition_assignment = [-1] * n_agents
    current_coalition = 0

    for start in range(n_agents):
        if coalition_assignment[start] != -1:
            continue
        queue = [start]
        while queue:
            node = queue.pop(0)
            if coalition_assignment[node] != -1:
                continue
            coalition_assignment[node] = current_coalition
            for neighbor in range(n_agents):
                if adjacency[node, neighbor] and coalition_assignment[neighbor] == -1:
                    queue.append(neighbor)
        current_coalition += 1

    # Calcular correlaciones intra e inter
    intra_corrs = []
    inter_corrs = []

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if coalition_assignment[i] == coalition_assignment[j]:
                intra_corrs.append(corr_matrix[i, j])
            else:
                inter_corrs.append(corr_matrix[i, j])

    intra_mean = float(np.mean(intra_corrs)) if intra_corrs else 0.5
    inter_mean = float(np.mean(inter_corrs)) if inter_corrs else 0.5

    return intra_mean, inter_mean


def run_sweet_spot_search(coupling_range: Tuple[float, float] = (0.01, 0.6),
                          noise_range: Tuple[float, float] = (0.0, 0.4),
                          grid_resolution: int = 15,
                          n_agents: int = 7,
                          n_steps: int = 1500) -> List[SweetSpotMetrics]:
    """Barrido completo del espacio de parámetros."""

    print(f"\n{'='*70}")
    print("BÚSQUEDA DEL SWEET SPOT")
    print(f"{'='*70}")
    print(f"  Coupling range: {coupling_range}")
    print(f"  Noise range: {noise_range}")
    print(f"  Grid: {grid_resolution}x{grid_resolution}")
    print(f"  Agents: {n_agents}")
    print(f"  Steps per point: {n_steps}")

    coupling_values = np.linspace(coupling_range[0], coupling_range[1], grid_resolution)
    noise_values = np.linspace(noise_range[0], noise_range[1], grid_resolution)

    results = []
    total = grid_resolution * grid_resolution
    count = 0

    for coupling in coupling_values:
        for noise in noise_values:
            count += 1
            if count % 20 == 0:
                print(f"  Progress: {count}/{total} ({count/total*100:.1f}%)")

            seed = 42 + int(coupling * 1000) + int(noise * 1000)
            metrics = run_simulation_for_sweet_spot(
                coupling, noise, n_agents, n_steps, seed=seed
            )
            results.append(metrics)

    return results


def find_optimal_zones(results: List[SweetSpotMetrics]) -> Dict[str, Any]:
    """Encuentra las zonas óptimas del espacio de parámetros."""

    # Ordenar por health_index
    sorted_by_health = sorted(results, key=lambda x: x.health_index, reverse=True)

    # Top 10%
    top_n = max(1, len(results) // 10)
    top_results = sorted_by_health[:top_n]

    # Estadísticas de la zona óptima
    optimal_zone = {
        'coupling_range': (
            min(r.coupling for r in top_results),
            max(r.coupling for r in top_results)
        ),
        'noise_range': (
            min(r.noise for r in top_results),
            max(r.noise for r in top_results)
        ),
        'coupling_mean': np.mean([r.coupling for r in top_results]),
        'noise_mean': np.mean([r.noise for r in top_results]),
        'health_mean': np.mean([r.health_index for r in top_results]),
        'n_coalitions_mean': np.mean([r.n_coalitions for r in top_results]),
        'diversity_mean': np.mean([r.diversity_index for r in top_results]),
        'top_points': [(r.coupling, r.noise, r.health_index) for r in top_results[:5]]
    }

    # Zonas problemáticas
    worst_results = sorted_by_health[-top_n:]

    danger_zone = {
        'coupling_range': (
            min(r.coupling for r in worst_results),
            max(r.coupling for r in worst_results)
        ),
        'noise_range': (
            min(r.noise for r in worst_results),
            max(r.noise for r in worst_results)
        ),
        'health_mean': np.mean([r.health_index for r in worst_results]),
    }

    return {
        'optimal': optimal_zone,
        'danger': danger_zone,
        'all_results': results
    }


def generate_sweet_spot_figures(results: List[SweetSpotMetrics],
                                 analysis: Dict[str, Any]):
    """Genera visualizaciones del sweet spot."""

    # Extraer datos para heatmaps
    coupling_vals = sorted(set(r.coupling for r in results))
    noise_vals = sorted(set(r.noise for r in results))

    n_coupling = len(coupling_vals)
    n_noise = len(noise_vals)

    # Crear matrices
    health_matrix = np.zeros((n_noise, n_coupling))
    diversity_matrix = np.zeros((n_noise, n_coupling))
    coalitions_matrix = np.zeros((n_noise, n_coupling))
    inter_corr_matrix = np.zeros((n_noise, n_coupling))

    for r in results:
        i = noise_vals.index(r.noise)
        j = coupling_vals.index(r.coupling)
        health_matrix[i, j] = r.health_index
        diversity_matrix[i, j] = r.diversity_index
        coalitions_matrix[i, j] = r.n_coalitions
        inter_corr_matrix[i, j] = r.inter_coalition_corr

    # Suavizar para mejor visualización
    health_smooth = gaussian_filter(health_matrix, sigma=0.8)

    # Figura principal: 4 paneles
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    extent = [coupling_vals[0], coupling_vals[-1], noise_vals[0], noise_vals[-1]]

    # Panel 1: Health Index (el sweet spot)
    ax = axes[0, 0]
    im = ax.imshow(health_smooth, origin='lower', extent=extent,
                   aspect='auto', cmap='RdYlGn')
    ax.set_xlabel('Coupling (acoplamiento)')
    ax.set_ylabel('Noise (ruido)')
    ax.set_title('SWEET SPOT: Índice de Salud Social')
    plt.colorbar(im, ax=ax, label='Health Index')

    # Marcar zona óptima
    opt = analysis['optimal']
    ax.axvline(opt['coupling_mean'], color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(opt['noise_mean'], color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.plot(opt['coupling_mean'], opt['noise_mean'], 'b*', markersize=20, label='Óptimo')
    ax.legend()

    # Panel 2: Número de coaliciones
    ax = axes[0, 1]
    im = ax.imshow(coalitions_matrix, origin='lower', extent=extent,
                   aspect='auto', cmap='viridis')
    ax.set_xlabel('Coupling')
    ax.set_ylabel('Noise')
    ax.set_title('Número de Coaliciones')
    plt.colorbar(im, ax=ax, label='N coaliciones')

    # Contorno donde hay 2-3 coaliciones (diversidad ideal)
    try:
        ax.contour(coalitions_matrix, levels=[1.5, 2.5, 3.5],
                   extent=extent, colors=['red', 'white', 'red'], linewidths=2)
    except:
        pass

    # Panel 3: Correlación inter-coalición
    ax = axes[1, 0]
    im = ax.imshow(inter_corr_matrix, origin='lower', extent=extent,
                   aspect='auto', cmap='coolwarm')
    ax.set_xlabel('Coupling')
    ax.set_ylabel('Noise')
    ax.set_title('Correlación Inter-Coalición\n(conexión entre grupos)')
    plt.colorbar(im, ax=ax, label='Correlación')

    # Panel 4: Índice de diversidad
    ax = axes[1, 1]
    im = ax.imshow(diversity_matrix, origin='lower', extent=extent,
                   aspect='auto', cmap='plasma')
    ax.set_xlabel('Coupling')
    ax.set_ylabel('Noise')
    ax.set_title('Índice de Diversidad')
    plt.colorbar(im, ax=ax, label='Diversidad')

    plt.suptitle('MAPA DEL SWEET SPOT: Dónde la Sociedad es Diversa pero Funcional',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'sweet_spot_map.png'), dpi=150)
    plt.close()

    # Figura 2: Interpretación para humanos
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(health_smooth, origin='lower', extent=extent,
                   aspect='auto', cmap='RdYlGn', alpha=0.8)

    # Añadir zonas con etiquetas
    ax.text(coupling_vals[-1]*0.8, noise_vals[0]*0.2,
            'HOMOGENEIZACIÓN\n(pensamiento único)',
            fontsize=11, ha='center', color='darkred', fontweight='bold')

    ax.text(coupling_vals[0]*1.5, noise_vals[-1]*0.8,
            'FRAGMENTACIÓN\n(sin conexión)',
            fontsize=11, ha='center', color='darkred', fontweight='bold')

    ax.text(opt['coupling_mean'], opt['noise_mean'] + 0.03,
            'SWEET SPOT\n(diversidad funcional)',
            fontsize=12, ha='center', color='darkgreen', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.plot(opt['coupling_mean'], opt['noise_mean'], 'g*', markersize=25)

    ax.set_xlabel('Acoplamiento Social\n(redes, medios, interacción)', fontsize=12)
    ax.set_ylabel('Ruido / Diversidad de Inputs\n(fuentes diversas, asincronía)', fontsize=12)
    ax.set_title('¿Dónde Queremos Estar Como Sociedad?', fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Salud del Sistema')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'sweet_spot_interpretacion.png'), dpi=150)
    plt.close()

    print(f"\n  Figuras guardadas en {FIG_DIR}")


def print_recommendations(analysis: Dict[str, Any]):
    """Imprime recomendaciones basadas en el análisis."""

    opt = analysis['optimal']

    print(f"\n{'='*70}")
    print("RESULTADOS: EL SWEET SPOT")
    print(f"{'='*70}")

    print(f"\n  ZONA ÓPTIMA ENCONTRADA:")
    print(f"    Coupling: {opt['coupling_range'][0]:.3f} - {opt['coupling_range'][1]:.3f}")
    print(f"    Noise: {opt['noise_range'][0]:.3f} - {opt['noise_range'][1]:.3f}")
    print(f"    Health Index medio: {opt['health_mean']:.3f}")
    print(f"    Coaliciones promedio: {opt['n_coalitions_mean']:.1f}")

    print(f"\n  PUNTO ÓPTIMO:")
    print(f"    Coupling = {opt['coupling_mean']:.3f}")
    print(f"    Noise = {opt['noise_mean']:.3f}")

    print(f"\n  INTERPRETACIÓN PARA SISTEMAS SOCIALES:")
    print(f"  " + "-"*60)

    if opt['coupling_mean'] < 0.2:
        print(f"    → Acoplamiento BAJO óptimo")
        print(f"      Implicación: Menos redes sociales, menos medios masivos")
    elif opt['coupling_mean'] < 0.4:
        print(f"    → Acoplamiento MEDIO óptimo")
        print(f"      Implicación: Conexión moderada, no híper-conectividad")
    else:
        print(f"    → Acoplamiento ALTO óptimo")
        print(f"      Implicación: Alta interacción necesaria")

    if opt['noise_mean'] < 0.1:
        print(f"    → Ruido BAJO óptimo")
        print(f"      Implicación: Inputs relativamente coherentes")
    elif opt['noise_mean'] < 0.25:
        print(f"    → Ruido MEDIO óptimo")
        print(f"      Implicación: Diversidad de fuentes necesaria")
    else:
        print(f"    → Ruido ALTO óptimo")
        print(f"      Implicación: Mucha diversidad de inputs necesaria")

    print(f"\n  RECETA PARA DIVERSIDAD FUNCIONAL:")
    print(f"    1. Acoplamiento moderado (no hiper-conectividad)")
    print(f"    2. Fuentes de información diversas (ruido estructural)")
    print(f"    3. Asincronía en procesamiento (no tiempo real)")
    print(f"    4. Espacio para coaliciones múltiples estables")


def run_complete_sweet_spot_analysis():
    """Ejecuta análisis completo."""

    # Búsqueda
    results = run_sweet_spot_search(
        coupling_range=(0.01, 0.6),
        noise_range=(0.0, 0.4),
        grid_resolution=15,
        n_agents=7,
        n_steps=1500
    )

    # Análisis
    analysis = find_optimal_zones(results)

    # Figuras
    generate_sweet_spot_figures(results, analysis)

    # Recomendaciones
    print_recommendations(analysis)

    # Guardar resultados
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'optimal_zone': analysis['optimal'],
        'danger_zone': analysis['danger'],
        'all_points': [
            {
                'coupling': r.coupling,
                'noise': r.noise,
                'health_index': r.health_index,
                'n_coalitions': r.n_coalitions,
                'diversity_index': r.diversity_index,
                'intra_corr': r.intra_coalition_corr,
                'inter_corr': r.inter_coalition_corr
            }
            for r in results
        ]
    }

    save_path = os.path.join(LOG_DIR, f"sweet_spot_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Datos guardados en: {save_path}")

    return results, analysis


if __name__ == '__main__':
    results, analysis = run_complete_sweet_spot_analysis()
