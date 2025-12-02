"""
Visualizaciones de Omega Spaces
================================

Genera gráficos de las métricas de simulación Omega.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

sys.path.insert(0, '/root/NEO_EVA')


def load_simulation_data(json_path: str) -> Dict[str, Any]:
    """Carga datos de simulación desde JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_omega_modes_by_agent(data: Dict, output_dir: str):
    """Gráfico de modos Ω activos por agente."""
    fig, ax = plt.subplots(figsize=(14, 6))

    agents = data['metadata']['agents']
    agent_metrics = data['agent_metrics']

    colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))

    for i, agent in enumerate(agents):
        agent_data = [m for m in agent_metrics if m['agent_id'] == agent]
        t = [m['t'] for m in agent_data]
        modes = [m['n_active_modes'] for m in agent_data]

        ax.plot(t, modes, label=agent, color=colors[i], alpha=0.7, linewidth=1)

    ax.set_xlabel('Tiempo (pasos)')
    ax.set_ylabel('Modos Ω Activos')
    ax.set_title('Modos Ω-Compute Activos por Agente')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'omega_modes_by_agent.png'), dpi=150)
    plt.close()


def plot_qfield_coherence_energy(data: Dict, output_dir: str):
    """Gráfico de C_Q(t) y E_Q(t) del Q-Field."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    agents = data['metadata']['agents']
    agent_metrics = data['agent_metrics']

    colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))

    # Coherencia C_Q(t)
    ax1 = axes[0]
    for i, agent in enumerate(agents):
        agent_data = [m for m in agent_metrics if m['agent_id'] == agent]
        t = [m['t'] for m in agent_data]
        cq = [m['coherence_cq'] for m in agent_data]

        ax1.plot(t, cq, label=agent, color=colors[i], alpha=0.7, linewidth=1)

    ax1.set_ylabel('C_Q(t) Coherencia')
    ax1.set_title('Q-Field: Coherencia C_Q(t) por Agente')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Energía de superposición E_Q(t)
    ax2 = axes[1]
    for i, agent in enumerate(agents):
        agent_data = [m for m in agent_metrics if m['agent_id'] == agent]
        t = [m['t'] for m in agent_data]
        eq = [m['superposition_energy_eq'] for m in agent_data]

        ax2.plot(t, eq, label=agent, color=colors[i], alpha=0.7, linewidth=1)

    ax2.set_xlabel('Tiempo (pasos)')
    ax2.set_ylabel('E_Q(t) Energía Superposición')
    ax2.set_title('Q-Field: Energía de Superposición E_Q(t) por Agente')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qfield_coherence_energy.png'), dpi=150)
    plt.close()


def plot_phase_space_curvature(data: Dict, output_dir: str):
    """Gráfico de curvatura de PhaseSpace-X."""
    fig, ax = plt.subplots(figsize=(14, 6))

    agents = data['metadata']['agents']
    agent_metrics = data['agent_metrics']

    colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))

    for i, agent in enumerate(agents):
        agent_data = [m for m in agent_metrics if m['agent_id'] == agent]
        t = [m['t'] for m in agent_data]
        curvature = [m['curvature'] for m in agent_data]

        ax.plot(t, curvature, label=agent, color=colors[i], alpha=0.7, linewidth=1)

    ax.set_xlabel('Tiempo (pasos)')
    ax.set_ylabel('Curvatura')
    ax.set_title('PhaseSpace-X: Curvatura de Trayectorias por Agente')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_space_curvature.png'), dpi=150)
    plt.close()


def plot_tensor_mind_global(data: Dict, output_dir: str):
    """Gráfico de métricas globales de TensorMind."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    global_metrics = data['global_metrics']

    t = [m['t'] for m in global_metrics]
    n_strong_modes = [m['n_strong_modes'] for m in global_metrics]
    interaction_strength = [m['mean_interaction_strength'] for m in global_metrics]

    # Modos fuertes
    ax1 = axes[0]
    ax1.plot(t, n_strong_modes, color='purple', linewidth=1.5)
    ax1.fill_between(t, n_strong_modes, alpha=0.3, color='purple')
    ax1.set_ylabel('Modos Fuertes')
    ax1.set_title('TensorMind: Número de Modos Tensoriales Fuertes')
    ax1.grid(True, alpha=0.3)

    # Fuerza de interacción
    ax2 = axes[1]
    ax2.plot(t, interaction_strength, color='green', linewidth=1.5)
    ax2.fill_between(t, interaction_strength, alpha=0.3, color='green')
    ax2.set_xlabel('Tiempo (pasos)')
    ax2.set_ylabel('Fuerza Media')
    ax2.set_title('TensorMind: Fuerza de Interacción Media (Orden 2)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tensor_mind_global.png'), dpi=150)
    plt.close()


def plot_global_summary(data: Dict, output_dir: str):
    """Gráfico resumen de métricas globales."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    global_metrics = data['global_metrics']
    t = [m['t'] for m in global_metrics]

    # Ω-Compute: Modos totales
    ax1 = axes[0, 0]
    total_modes = [m['total_modes'] for m in global_metrics]
    ax1.plot(t, total_modes, color='blue', linewidth=1.5)
    ax1.set_ylabel('Modos Totales')
    ax1.set_title('Ω-Compute: Modos Totales')
    ax1.grid(True, alpha=0.3)

    # Q-Field: Coherencia media
    ax2 = axes[0, 1]
    mean_coherence = [m['field_mean_coherence'] for m in global_metrics]
    ax2.plot(t, mean_coherence, color='red', linewidth=1.5)
    ax2.set_ylabel('Coherencia Media')
    ax2.set_title('Q-Field: Coherencia Media del Campo')
    ax2.grid(True, alpha=0.3)

    # PhaseSpace-X: Velocidad media
    ax3 = axes[1, 0]
    mean_speed = [m['mean_speed'] for m in global_metrics]
    ax3.plot(t, mean_speed, color='orange', linewidth=1.5)
    ax3.set_xlabel('Tiempo (pasos)')
    ax3.set_ylabel('Velocidad Media')
    ax3.set_title('PhaseSpace-X: Velocidad Media')
    ax3.grid(True, alpha=0.3)

    # TensorMind: Comunidades
    ax4 = axes[1, 1]
    n_communities = [m['n_communities'] for m in global_metrics]
    ax4.plot(t, n_communities, color='purple', linewidth=1.5)
    ax4.set_xlabel('Tiempo (pasos)')
    ax4.set_ylabel('Comunidades')
    ax4.set_title('TensorMind: Número de Comunidades')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'omega_global_summary.png'), dpi=150)
    plt.close()


def plot_agent_comparison(data: Dict, output_dir: str):
    """Comparación de métricas finales por agente."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    agents = data['metadata']['agents']
    agent_metrics = data['agent_metrics']

    # Calcular promedios por agente
    agent_stats = {}
    for agent in agents:
        agent_data = [m for m in agent_metrics if m['agent_id'] == agent]
        agent_stats[agent] = {
            'mean_modes': np.mean([m['n_active_modes'] for m in agent_data]),
            'mean_cq': np.mean([m['coherence_cq'] for m in agent_data]),
            'mean_eq': np.mean([m['superposition_energy_eq'] for m in agent_data]),
            'mean_curvature': np.mean([m['curvature'] for m in agent_data]),
        }

    x = np.arange(len(agents))
    width = 0.6
    colors = plt.cm.Set2(np.linspace(0, 1, len(agents)))

    # Modos activos
    ax1 = axes[0, 0]
    values = [agent_stats[a]['mean_modes'] for a in agents]
    ax1.bar(x, values, width, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents)
    ax1.set_ylabel('Media')
    ax1.set_title('Modos Ω Activos (Media)')
    for i, v in enumerate(values):
        ax1.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=9)

    # Coherencia C_Q
    ax2 = axes[0, 1]
    values = [agent_stats[a]['mean_cq'] for a in agents]
    ax2.bar(x, values, width, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(agents)
    ax2.set_ylabel('Media')
    ax2.set_title('Coherencia C_Q (Media)')
    for i, v in enumerate(values):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    # Energía E_Q
    ax3 = axes[1, 0]
    values = [agent_stats[a]['mean_eq'] for a in agents]
    ax3.bar(x, values, width, color=colors)
    ax3.set_xticks(x)
    ax3.set_xticklabels(agents)
    ax3.set_ylabel('Media')
    ax3.set_title('Energía E_Q (Media)')
    for i, v in enumerate(values):
        ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    # Curvatura
    ax4 = axes[1, 1]
    values = [agent_stats[a]['mean_curvature'] for a in agents]
    ax4.bar(x, values, width, color=colors)
    ax4.set_xticks(x)
    ax4.set_xticklabels(agents)
    ax4.set_ylabel('Media')
    ax4.set_title('Curvatura PhaseSpace (Media)')
    for i, v in enumerate(values):
        ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'omega_agent_comparison.png'), dpi=150)
    plt.close()


def generate_all_plots(json_path: str, output_dir: str = None):
    """Genera todos los gráficos."""
    data = load_simulation_data(json_path)

    if output_dir is None:
        output_dir = os.path.dirname(json_path)

    os.makedirs(output_dir, exist_ok=True)

    print("Generando gráficos de Omega Spaces...")

    print("  - Modos Ω por agente")
    plot_omega_modes_by_agent(data, output_dir)

    print("  - Q-Field coherencia y energía")
    plot_qfield_coherence_energy(data, output_dir)

    print("  - PhaseSpace-X curvatura")
    plot_phase_space_curvature(data, output_dir)

    print("  - TensorMind global")
    plot_tensor_mind_global(data, output_dir)

    print("  - Resumen global")
    plot_global_summary(data, output_dir)

    print("  - Comparación por agente")
    plot_agent_comparison(data, output_dir)

    print(f"\nGráficos guardados en: {output_dir}")

    return [
        'omega_modes_by_agent.png',
        'qfield_coherence_energy.png',
        'phase_space_curvature.png',
        'tensor_mind_global.png',
        'omega_global_summary.png',
        'omega_agent_comparison.png',
    ]


if __name__ == "__main__":
    # Buscar el JSON más reciente
    log_dir = '/root/NEO_EVA/logs/omega_simulation'
    json_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]

    if json_files:
        latest = sorted(json_files)[-1]
        json_path = os.path.join(log_dir, latest)
        output_dir = '/root/NEO_EVA/visualizations/omega_plots'

        generate_all_plots(json_path, output_dir)
    else:
        print("No se encontraron archivos JSON de simulación")
