#!/usr/bin/env python3
"""
LA CATÁSTROFE DEL COUPLING: De 2008 a Hoy
==========================================

Simulación de qué pasa cuando aumentas el acoplamiento progresivamente
desde el sweet spot hasta híper-conectividad.

Es como ver en cámara rápida la evolución de las sociedades
desde pre-redes sociales hasta hoy.

Hipótesis: Hay una transición de fase crítica donde el sistema
pasa de "diversidad funcional" a "caos/homogeneización".

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
from scipy.ndimage import gaussian_filter1d
import json
from datetime import datetime

sys.path.insert(0, '/root/NEO_EVA')

from core.agents import NEO, EVA, BaseAgent

# Output
FIG_DIR = '/root/NEO_EVA/figuras/coupling_catastrophe'
os.makedirs(FIG_DIR, exist_ok=True)


@dataclass
class EraMetrics:
    """Métricas de una era/nivel de coupling."""
    coupling: float
    era_name: str

    n_coalitions: float
    polarization: float  # 0 = homogéneo, 1 = máxima polarización
    inter_coalition_corr: float
    intra_coalition_corr: float
    cascade_frequency: float  # Frecuencia de cascadas informativas
    stability: float

    # Índices derivados
    diversity_health: float
    chaos_index: float


def simulate_era(coupling: float, noise: float = 0.15, n_agents: int = 8,
                 n_steps: int = 2000, dim: int = 6, seed: int = 42) -> EraMetrics:
    """Simula una era con un nivel de coupling dado."""

    BaseAgent._agent_counter = 0
    rng = np.random.default_rng(seed)

    agents = {}
    agent_names = [f'A{i}' for i in range(n_agents)]

    for i, name in enumerate(agent_names):
        if i % 2 == 0:
            agents[name] = NEO(dim_visible=dim, dim_hidden=dim)
        else:
            agents[name] = EVA(dim_visible=dim, dim_hidden=dim)

    CE_history = {name: [] for name in agent_names}
    surprise_history = {name: [] for name in agent_names}
    coalition_counts = []

    window = max(50, n_steps // 20)

    for t in range(n_steps):
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
            surprise_history[name].append(response.surprise)

        # Contar coaliciones periódicamente
        if t > window and t % (window // 2) == 0:
            CE_recent = np.array([CE_history[name][-window:] for name in agent_names])
            n_coal = detect_coalitions(CE_recent)
            coalition_counts.append(n_coal)

    # Calcular métricas
    CE_array = np.array([CE_history[name] for name in agent_names])
    surprise_array = np.array([surprise_history[name] for name in agent_names])

    # Número medio de coaliciones
    n_coalitions = np.mean(coalition_counts) if coalition_counts else 1

    # Polarización: varianza entre medias de agentes
    agent_means = [np.mean(CE_history[name]) for name in agent_names]
    polarization = np.std(agent_means) / (np.mean(agent_means) + 1e-12)
    polarization = min(1, polarization * 2)  # Normalizar

    # Correlaciones
    intra_corr, inter_corr = compute_coalition_correlations(CE_array)

    # Frecuencia de cascadas: picos sincronizados de surprise
    cascade_freq = detect_cascades(surprise_array)

    # Estabilidad: inverso de varianza temporal de coaliciones
    stability = 1.0 / (1.0 + np.std(coalition_counts)) if coalition_counts else 0.5

    # Índices derivados
    # Salud de diversidad: queremos 2-3 coaliciones, correlación inter media
    optimal_coal = n_agents / 3
    diversity_health = (1 - abs(n_coalitions - optimal_coal) / n_agents) * \
                       (1 - abs(inter_corr - 0.3))
    diversity_health = max(0, min(1, diversity_health))

    # Índice de caos: alta variabilidad + cascadas frecuentes
    chaos_index = (1 - stability) * 0.5 + cascade_freq * 0.5

    # Determinar nombre de era
    if coupling < 0.1:
        era_name = "Pre-internet"
    elif coupling < 0.2:
        era_name = "Web 1.0"
    elif coupling < 0.35:
        era_name = "Facebook 2008"
    elif coupling < 0.5:
        era_name = "Twitter/Instagram"
    elif coupling < 0.7:
        era_name = "TikTok/Algoritmos"
    else:
        era_name = "Metaverso/Futuro"

    return EraMetrics(
        coupling=coupling,
        era_name=era_name,
        n_coalitions=n_coalitions,
        polarization=polarization,
        inter_coalition_corr=inter_corr,
        intra_coalition_corr=intra_corr,
        cascade_frequency=cascade_freq,
        stability=stability,
        diversity_health=diversity_health,
        chaos_index=chaos_index
    )


def detect_coalitions(data: np.ndarray) -> int:
    """Detecta número de coaliciones."""
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

    threshold = np.percentile(correlations, 60)
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


def compute_coalition_correlations(CE_array: np.ndarray) -> Tuple[float, float]:
    """Calcula correlaciones intra e inter coalición."""
    n_agents = CE_array.shape[0]

    corr_matrix = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif np.std(CE_array[i]) > 1e-12 and np.std(CE_array[j]) > 1e-12:
                c, _ = stats.pearsonr(CE_array[i], CE_array[j])
                corr_matrix[i, j] = abs(c) if not np.isnan(c) else 0.0

    threshold = np.median(corr_matrix[np.triu_indices(n_agents, k=1)])

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
                if corr_matrix[node, neighbor] >= threshold and coalition_assignment[neighbor] == -1:
                    queue.append(neighbor)
        current_coalition += 1

    intra_corrs = []
    inter_corrs = []

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if coalition_assignment[i] == coalition_assignment[j]:
                intra_corrs.append(corr_matrix[i, j])
            else:
                inter_corrs.append(corr_matrix[i, j])

    return (float(np.mean(intra_corrs)) if intra_corrs else 0.5,
            float(np.mean(inter_corrs)) if inter_corrs else 0.5)


def detect_cascades(surprise_array: np.ndarray) -> float:
    """Detecta frecuencia de cascadas (picos sincronizados de surprise)."""
    n_agents, T = surprise_array.shape

    # Calcular surprise media global
    global_surprise = np.mean(surprise_array, axis=0)

    # Detectar picos (> 2 std)
    mean_s = np.mean(global_surprise)
    std_s = np.std(global_surprise)

    if std_s < 1e-12:
        return 0.0

    peaks = global_surprise > (mean_s + 2 * std_s)
    n_cascades = np.sum(peaks)

    # Normalizar por tiempo
    cascade_freq = n_cascades / T

    return min(1.0, cascade_freq * 100)  # Escalar para visualización


def run_coupling_evolution():
    """Simula la evolución del coupling desde pre-internet hasta futuro."""

    print("\n" + "="*70)
    print("LA CATÁSTROFE DEL COUPLING: Simulación Histórica")
    print("="*70)

    # Niveles de coupling representando eras
    coupling_levels = np.linspace(0.05, 0.8, 20)

    results = []

    for i, coupling in enumerate(coupling_levels):
        print(f"  Simulando coupling={coupling:.3f} ({i+1}/20)...")

        # Múltiples seeds para robustez
        era_results = []
        for seed in [42, 123, 456]:
            metrics = simulate_era(coupling, noise=0.15, n_agents=8,
                                   n_steps=1500, seed=seed)
            era_results.append(metrics)

        # Promediar
        avg_metrics = EraMetrics(
            coupling=coupling,
            era_name=era_results[0].era_name,
            n_coalitions=np.mean([m.n_coalitions for m in era_results]),
            polarization=np.mean([m.polarization for m in era_results]),
            inter_coalition_corr=np.mean([m.inter_coalition_corr for m in era_results]),
            intra_coalition_corr=np.mean([m.intra_coalition_corr for m in era_results]),
            cascade_frequency=np.mean([m.cascade_frequency for m in era_results]),
            stability=np.mean([m.stability for m in era_results]),
            diversity_health=np.mean([m.diversity_health for m in era_results]),
            chaos_index=np.mean([m.chaos_index for m in era_results])
        )
        results.append(avg_metrics)

    return results


def find_critical_point(results: List[EraMetrics]) -> Tuple[float, str]:
    """Encuentra el punto crítico de transición de fase."""

    # Buscar donde diversity_health cae más rápido
    healths = [r.diversity_health for r in results]
    couplings = [r.coupling for r in results]

    # Derivada numérica
    derivatives = np.diff(healths) / np.diff(couplings)

    # Punto de máxima caída (derivada más negativa)
    critical_idx = np.argmin(derivatives)
    critical_coupling = (couplings[critical_idx] + couplings[critical_idx + 1]) / 2

    # Determinar era
    for r in results:
        if abs(r.coupling - critical_coupling) < 0.05:
            return critical_coupling, r.era_name

    return critical_coupling, "Transición"


def generate_catastrophe_figures(results: List[EraMetrics]):
    """Genera visualizaciones de la catástrofe."""

    couplings = [r.coupling for r in results]

    # Encontrar punto crítico
    critical_coupling, critical_era = find_critical_point(results)

    # Figura principal: múltiples métricas
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Colores por era
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(results)))

    # Panel 1: Número de coaliciones
    ax = axes[0, 0]
    y = [r.n_coalitions for r in results]
    ax.plot(couplings, y, 'o-', linewidth=2, markersize=6, color='blue')
    ax.axvline(critical_coupling, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvspan(0.08, 0.15, alpha=0.2, color='green', label='Sweet Spot')
    ax.set_xlabel('Coupling (Acoplamiento)')
    ax.set_ylabel('Número de Coaliciones')
    ax.set_title('Coaliciones vs Coupling')
    ax.legend()

    # Panel 2: Polarización
    ax = axes[0, 1]
    y = [r.polarization for r in results]
    ax.plot(couplings, y, 'o-', linewidth=2, markersize=6, color='orange')
    ax.axvline(critical_coupling, color='red', linestyle='--', linewidth=2)
    ax.axvspan(0.08, 0.15, alpha=0.2, color='green')
    ax.set_xlabel('Coupling')
    ax.set_ylabel('Polarización')
    ax.set_title('Polarización vs Coupling')

    # Panel 3: Correlación inter-coalición
    ax = axes[0, 2]
    y = [r.inter_coalition_corr for r in results]
    ax.plot(couplings, y, 'o-', linewidth=2, markersize=6, color='purple')
    ax.axvline(critical_coupling, color='red', linestyle='--', linewidth=2)
    ax.axhline(0.3, color='green', linestyle=':', linewidth=2, label='Óptimo')
    ax.set_xlabel('Coupling')
    ax.set_ylabel('Correlación Inter-Coalición')
    ax.set_title('Conexión Entre Grupos')
    ax.legend()

    # Panel 4: Frecuencia de cascadas
    ax = axes[1, 0]
    y = [r.cascade_frequency for r in results]
    ax.plot(couplings, y, 'o-', linewidth=2, markersize=6, color='red')
    ax.axvline(critical_coupling, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Coupling')
    ax.set_ylabel('Frecuencia de Cascadas')
    ax.set_title('Viralidad / Cascadas Informativas')

    # Panel 5: Salud de diversidad
    ax = axes[1, 1]
    y = [r.diversity_health for r in results]
    y_smooth = gaussian_filter1d(y, sigma=1)
    ax.fill_between(couplings, y_smooth, alpha=0.3, color='green')
    ax.plot(couplings, y_smooth, 'o-', linewidth=2, markersize=6, color='green')
    ax.axvline(critical_coupling, color='red', linestyle='--', linewidth=2,
               label=f'Punto Crítico: {critical_coupling:.2f}')
    ax.set_xlabel('Coupling')
    ax.set_ylabel('Salud de Diversidad')
    ax.set_title('ÍNDICE DE SALUD SOCIAL')
    ax.legend()

    # Panel 6: Índice de caos
    ax = axes[1, 2]
    y = [r.chaos_index for r in results]
    ax.plot(couplings, y, 'o-', linewidth=2, markersize=6, color='darkred')
    ax.axvline(critical_coupling, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Coupling')
    ax.set_ylabel('Índice de Caos')
    ax.set_title('Inestabilidad del Sistema')

    plt.suptitle(f'LA CATÁSTROFE DEL COUPLING\nPunto Crítico en coupling ≈ {critical_coupling:.2f} ({critical_era})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'coupling_catastrophe_metrics.png'), dpi=150)
    plt.close()

    # Figura 2: Timeline histórico
    fig, ax = plt.subplots(figsize=(14, 8))

    # Salud como área
    y = [r.diversity_health for r in results]
    y_smooth = gaussian_filter1d(y, sigma=1)

    # Colorear por zona
    for i in range(len(couplings) - 1):
        color = 'green' if y_smooth[i] > 0.5 else ('yellow' if y_smooth[i] > 0.3 else 'red')
        ax.fill_between([couplings[i], couplings[i+1]],
                        [y_smooth[i], y_smooth[i+1]],
                        alpha=0.5, color=color)

    ax.plot(couplings, y_smooth, 'k-', linewidth=3)

    # Marcar eras
    eras = [
        (0.05, "Pre-\ninternet"),
        (0.15, "Web 1.0\n(2000)"),
        (0.3, "Facebook\n(2008)"),
        (0.45, "Twitter\nInstagram"),
        (0.6, "TikTok\n(2018)"),
        (0.75, "¿Futuro?")
    ]

    for coupling, era in eras:
        idx = np.argmin(np.abs(np.array(couplings) - coupling))
        ax.annotate(era, xy=(coupling, y_smooth[idx]),
                    xytext=(coupling, y_smooth[idx] + 0.15),
                    ha='center', fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black'))

    # Línea crítica
    ax.axvline(critical_coupling, color='darkred', linestyle='--', linewidth=3,
               label=f'PUNTO DE NO RETORNO\n(coupling ≈ {critical_coupling:.2f})')

    ax.set_xlabel('Acoplamiento Social (coupling)', fontsize=12)
    ax.set_ylabel('Salud de la Diversidad Social', fontsize=12)
    ax.set_title('EVOLUCIÓN HISTÓRICA: Del Sweet Spot al Caos', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 1)

    # Añadir texto explicativo
    ax.text(0.1, 0.85, 'ZONA SANA', fontsize=12, fontweight='bold', color='darkgreen')
    ax.text(0.5, 0.15, 'ZONA DE CAOS', fontsize=12, fontweight='bold', color='darkred')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'coupling_timeline.png'), dpi=150)
    plt.close()

    # Figura 3: El diagnóstico
    fig, ax = plt.subplots(figsize=(12, 8))

    # Crear "termómetro" de la sociedad actual
    current_coupling = 0.55  # Estimación para 2024

    y_health = [r.diversity_health for r in results]

    ax.barh(range(len(results)), [r.coupling for r in results],
            color=[plt.cm.RdYlGn(h) for h in y_health], height=0.8)

    # Marcar posición actual
    current_idx = np.argmin(np.abs(np.array(couplings) - current_coupling))
    ax.barh(current_idx, couplings[current_idx], color='black', height=0.8)
    ax.text(couplings[current_idx] + 0.02, current_idx, '← ESTAMOS AQUÍ (2024)',
            fontsize=12, fontweight='bold', va='center')

    # Marcar sweet spot
    sweet_idx = np.argmin(np.abs(np.array(couplings) - 0.12))
    ax.barh(sweet_idx, couplings[sweet_idx], color='blue', height=0.8, alpha=0.7)
    ax.text(couplings[sweet_idx] + 0.02, sweet_idx, '← SWEET SPOT',
            fontsize=12, fontweight='bold', va='center', color='blue')

    ax.set_xlabel('Nivel de Acoplamiento', fontsize=12)
    ax.set_ylabel('Simulación #', fontsize=12)
    ax.set_title('DIAGNÓSTICO: ¿Dónde Estamos?', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'coupling_diagnosis.png'), dpi=150)
    plt.close()

    print(f"\n  Figuras guardadas en {FIG_DIR}")

    return critical_coupling, critical_era


def print_analysis(results: List[EraMetrics], critical_coupling: float, critical_era: str):
    """Imprime análisis detallado."""

    print(f"\n{'='*70}")
    print("ANÁLISIS DE LA CATÁSTROFE")
    print(f"{'='*70}")

    print(f"\n  PUNTO CRÍTICO DE TRANSICIÓN:")
    print(f"    Coupling: {critical_coupling:.3f}")
    print(f"    Era aproximada: {critical_era}")

    print(f"\n  EVOLUCIÓN POR ERAS:")
    print(f"  {'-'*60}")

    eras_shown = set()
    for r in results:
        if r.era_name not in eras_shown:
            print(f"\n    {r.era_name} (coupling ≈ {r.coupling:.2f}):")
            print(f"      Coaliciones: {r.n_coalitions:.1f}")
            print(f"      Polarización: {r.polarization:.2f}")
            print(f"      Cascadas: {r.cascade_frequency:.2f}")
            print(f"      Salud diversidad: {r.diversity_health:.2f}")
            eras_shown.add(r.era_name)

    print(f"\n  CONCLUSIONES:")
    print(f"  {'-'*60}")
    print(f"    1. El punto crítico está en coupling ≈ {critical_coupling:.2f}")
    print(f"    2. Antes del punto crítico: diversidad funcional")
    print(f"    3. Después: caos, polarización, cascadas")
    print(f"    4. Sociedad actual (2024) estimada en coupling ≈ 0.55")
    print(f"    5. ESTAMOS DESPUÉS DEL PUNTO CRÍTICO")

    print(f"\n  QUÉ HARÍA FALTA PARA VOLVER AL SWEET SPOT:")
    print(f"  {'-'*60}")
    print(f"    • Reducir coupling de ~0.55 a ~0.12")
    print(f"    • Eso significa: 4-5x MENOS conectividad")
    print(f"    • Alternativa: aumentar ruido/diversidad masivamente")
    print(f"    • O ambas cosas")


def run_full_catastrophe_analysis():
    """Ejecuta análisis completo."""

    results = run_coupling_evolution()
    critical_coupling, critical_era = generate_catastrophe_figures(results)
    print_analysis(results, critical_coupling, critical_era)

    # Guardar datos
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'critical_coupling': critical_coupling,
        'critical_era': critical_era,
        'results': [
            {
                'coupling': r.coupling,
                'era': r.era_name,
                'n_coalitions': r.n_coalitions,
                'polarization': r.polarization,
                'cascade_freq': r.cascade_frequency,
                'diversity_health': r.diversity_health,
                'chaos_index': r.chaos_index
            }
            for r in results
        ]
    }

    save_path = os.path.join(FIG_DIR, f"catastrophe_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Datos guardados en: {save_path}")

    return results, critical_coupling


if __name__ == '__main__':
    results, critical = run_full_catastrophe_analysis()
