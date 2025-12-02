#!/usr/bin/env python3
"""
AUDITORÍA DE CORRELACIONES PERFECTAS
=====================================

Verificar si las correlaciones = 1.0000 son:
1) Resultado genuino de dinámicas acopladas
2) Artefacto de implementación (campo global, series duplicadas, etc.)

100% diagnóstico - NO modifica comportamiento.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, '/root/NEO_EVA')

from core.agents import NEO, EVA
from omega.omega_state import OmegaState
from omega.q_field import QField

# Output
OUTPUT_DIR = '/root/NEO_EVA/logs/audit_correlations'
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_diagnostic_simulation(n_steps=3000, seed=42):
    """Run simulation and collect per-agent metrics with full traceability."""
    rng = np.random.default_rng(seed)
    n_agents = 5
    dim = 6

    # Initialize agents
    agents = {}
    agent_names = [f'A{i}' for i in range(n_agents)]

    for i, name in enumerate(agent_names):
        if i % 2 == 0:
            agents[name] = NEO(dim_visible=dim, dim_hidden=dim)
        else:
            agents[name] = EVA(dim_visible=dim, dim_hidden=dim)

    # Initialize modules
    q_field = QField()
    omega_states = {name: OmegaState(dimension=16) for name in agent_names}

    # Storage - SEPARATE arrays per agent
    psi_norm = {name: [] for name in agent_names}
    H_narr = {name: [] for name in agent_names}
    Q_coherence = {name: [] for name in agent_names}
    CE = {name: [] for name in agent_names}

    print(f"Running diagnostic simulation: {n_steps} steps, {n_agents} agents")
    print(f"Seed: {seed}")
    print()

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

            # Collect metrics - EACH from its own source
            psi_norm[name].append(np.linalg.norm(state.z_visible))
            H_narr[name].append(state.S)  # Entropy from agent state
            CE[name].append(1.0 / (1.0 + response.surprise))

            # Q-Field registration
            q_field.register_state(name, state.z_visible)

        # Q coherence - check what q_field returns
        q_stats = q_field.get_statistics()

        # DIAGNOSTIC: What keys are in q_stats?
        if t == 0:
            print("Q-Field statistics keys:", list(q_stats.keys()))
            print("Q-Field stats sample:", q_stats)
            print()

        # Assign Q coherence per agent
        for name in agent_names:
            # Try agent-specific key first
            agent_q = q_stats.get(f'{name}_coherence', None)
            if agent_q is None:
                # Fallback to global - THIS IS THE PROBLEM if it happens
                agent_q = q_stats.get('mean_coherence', 0.5)
            Q_coherence[name].append(agent_q)

        if (t + 1) % 1000 == 0:
            print(f"  Step {t+1}/{n_steps}")

    return {
        'psi_norm': psi_norm,
        'H_narr': H_narr,
        'Q_coherence': Q_coherence,
        'CE': CE,
        'agent_names': agent_names
    }


def analyze_series_identity(data, metric_name):
    """Check if all agent series are identical."""
    agent_names = data['agent_names']
    series = data[metric_name]

    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE IDENTIDAD: {metric_name}")
    print(f"{'='*60}")

    # Convert to arrays
    arrays = {name: np.array(series[name]) for name in agent_names}

    # Check pairwise identity
    identity_matrix = {}
    for i, ni in enumerate(agent_names):
        for j, nj in enumerate(agent_names):
            if i < j:
                is_close = np.allclose(arrays[ni], arrays[nj])
                is_identical = np.array_equal(arrays[ni], arrays[nj])
                max_diff = np.max(np.abs(arrays[ni] - arrays[nj]))
                identity_matrix[(ni, nj)] = {
                    'allclose': is_close,
                    'identical': is_identical,
                    'max_diff': max_diff
                }
                print(f"  {ni} vs {nj}: identical={is_identical}, allclose={is_close}, max_diff={max_diff:.10f}")

    # Check if ALL are identical (global field)
    all_identical = all(v['identical'] for v in identity_matrix.values())
    all_close = all(v['allclose'] for v in identity_matrix.values())

    if all_identical:
        print(f"\n  ⚠️  ALERTA: {metric_name} es IDÉNTICO para todos los agentes")
        print(f"      → Esto indica un CAMPO GLOBAL, no una variable por agente")
        return 'GLOBAL_IDENTICAL'
    elif all_close:
        print(f"\n  ⚠️  ALERTA: {metric_name} es CASI IDÉNTICO (allclose=True)")
        print(f"      → Las diferencias son < 1e-8")
        return 'GLOBAL_CLOSE'
    else:
        print(f"\n  ✓ {metric_name} tiene series DISTINTAS por agente")
        return 'DISTINCT'


def compute_stats_table(data, metric_name):
    """Compute per-agent statistics."""
    agent_names = data['agent_names']
    series = data[metric_name]

    print(f"\n{'='*60}")
    print(f"ESTADÍSTICAS POR AGENTE: {metric_name}")
    print(f"{'='*60}")
    print(f"{'Agente':<8} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-" * 60)

    stats = []
    for name in agent_names:
        arr = np.array(series[name])
        s = {
            'agent': name,
            'metric': metric_name,
            'min': np.min(arr),
            'max': np.max(arr),
            'mean': np.mean(arr),
            'std': np.std(arr)
        }
        stats.append(s)
        print(f"{name:<8} {s['min']:>12.6f} {s['max']:>12.6f} {s['mean']:>12.6f} {s['std']:>12.6f}")

    return stats


def compute_correlations_detailed(data, metric_name):
    """Compute correlations with full precision."""
    agent_names = data['agent_names']
    series = data[metric_name]

    print(f"\n{'='*60}")
    print(f"CORRELACIONES (6 decimales): {metric_name}")
    print(f"{'='*60}")

    n = len(agent_names)
    arrays = {name: np.array(series[name]) for name in agent_names}

    # Header
    print(f"{'':>8}", end='')
    for name in agent_names:
        print(f"{name:>12}", end='')
    print()

    corr_matrix = np.zeros((n, n))
    for i, ni in enumerate(agent_names):
        print(f"{ni:<8}", end='')
        for j, nj in enumerate(agent_names):
            corr = np.corrcoef(arrays[ni], arrays[nj])[0, 1]
            corr_matrix[i, j] = corr
            print(f"{corr:>12.6f}", end='')
        print()

    return corr_matrix


def plot_scatter_pairs(data, metric_name, output_dir):
    """Plot scatter for agent pairs."""
    agent_names = data['agent_names']
    series = data[metric_name]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    pairs = [('A0', 'A1'), ('A0', 'A2'), ('A1', 'A2'), ('A2', 'A3')]

    for idx, (a1, a2) in enumerate(pairs):
        ax = axes[idx // 2, idx % 2]
        x = np.array(series[a1])
        y = np.array(series[a2])

        ax.scatter(x, y, alpha=0.3, s=1)
        ax.set_xlabel(f'{a1}')
        ax.set_ylabel(f'{a2}')

        corr = np.corrcoef(x, y)[0, 1]
        ax.set_title(f'{metric_name}: {a1} vs {a2}\nCorr = {corr:.6f}')

        # Add diagonal line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='y=x')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_{metric_name}.png', dpi=150)
    plt.close()


def plot_time_series_overlay(data, metric_name, output_dir):
    """Plot all agent series overlaid."""
    agent_names = data['agent_names']
    series = data[metric_name]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot first 500 steps for clarity
    n_show = 500

    for name in agent_names:
        arr = np.array(series[name])[:n_show]
        ax.plot(arr, label=name, alpha=0.7)

    ax.set_xlabel('Time Step')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name}: Series por Agente (primeros {n_show} pasos)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/timeseries_{metric_name}.png', dpi=150)
    plt.close()


def audit_q_field_implementation():
    """Check Q-Field implementation for global vs per-agent values."""
    print(f"\n{'='*60}")
    print("AUDITORÍA DE IMPLEMENTACIÓN: Q-Field")
    print(f"{'='*60}")

    # Read Q-Field source
    try:
        with open('/root/NEO_EVA/omega/q_field.py', 'r') as f:
            source = f.read()

        # Check for per-agent tracking
        has_per_agent = 'agent' in source.lower() or 'name' in source.lower()
        has_dict = 'dict' in source or '{}' in source

        print(f"  Q-Field tiene tracking por agente: {has_per_agent}")
        print(f"  Q-Field usa diccionarios: {has_dict}")

        # Look for register_state signature
        if 'def register_state' in source:
            # Extract function signature
            import re
            match = re.search(r'def register_state\((.*?)\):', source)
            if match:
                print(f"  register_state signature: ({match.group(1)})")

        # Look for get_statistics
        if 'def get_statistics' in source:
            match = re.search(r'def get_statistics\((.*?)\):', source)
            if match:
                print(f"  get_statistics signature: ({match.group(1)})")

    except Exception as e:
        print(f"  Error leyendo Q-Field: {e}")


def audit_agent_state():
    """Check what state.S (entropy) actually represents."""
    print(f"\n{'='*60}")
    print("AUDITORÍA DE IMPLEMENTACIÓN: Agent State")
    print(f"{'='*60}")

    # Create test agents
    neo = NEO(dim_visible=6, dim_hidden=6)
    eva = EVA(dim_visible=6, dim_hidden=6)

    # Get initial states
    neo_state = neo.get_state()
    eva_state = eva.get_state()

    print(f"  NEO initial state.S: {neo_state.S}")
    print(f"  EVA initial state.S: {eva_state.S}")
    print(f"  NEO z_visible: {neo_state.z_visible}")
    print(f"  EVA z_visible: {eva_state.z_visible}")
    print(f"  z_visible identical: {np.array_equal(neo_state.z_visible, eva_state.z_visible)}")

    # Step and check
    rng = np.random.default_rng(42)
    stim = rng.uniform(0, 1, 6)

    neo.step(stim)
    eva.step(stim)

    neo_state2 = neo.get_state()
    eva_state2 = eva.get_state()

    print(f"\n  After step with same stimulus:")
    print(f"  NEO state.S: {neo_state2.S}")
    print(f"  EVA state.S: {eva_state2.S}")
    print(f"  S identical: {neo_state2.S == eva_state2.S}")


def check_phase_c_data():
    """Check if Phase C data exists and analyze it."""
    print(f"\n{'='*60}")
    print("VERIFICACIÓN DE DATOS FASE C")
    print(f"{'='*60}")

    phase_c_paths = [
        '/root/NEO_EVA/logs/sesgo_C/',
        '/root/NEO_EVA/logs/sesgo_colectivo/',
    ]

    for path in phase_c_paths:
        if os.path.exists(path):
            print(f"\n  Encontrado: {path}")
            files = os.listdir(path)
            for f in files:
                print(f"    - {f}")

            # Check correlaciones.csv
            corr_file = os.path.join(path, 'correlaciones.csv')
            if os.path.exists(corr_file):
                print(f"\n  Analizando {corr_file}:")
                df = pd.read_csv(corr_file)
                print(df.head(20).to_string())


def main():
    """Run complete audit."""
    print("=" * 70)
    print("AUDITORÍA DE CORRELACIONES PERFECTAS")
    print("=" * 70)

    # 1. Check implementations
    audit_q_field_implementation()
    audit_agent_state()

    # 2. Run diagnostic simulation
    print("\n" + "=" * 70)
    print("SIMULACIÓN DIAGNÓSTICA")
    print("=" * 70)
    data = run_diagnostic_simulation(n_steps=3000, seed=42)

    # 3. Analyze each metric
    metrics = ['psi_norm', 'H_narr', 'Q_coherence', 'CE']
    identity_results = {}
    all_stats = []

    for metric in metrics:
        # Identity check
        identity_results[metric] = analyze_series_identity(data, metric)

        # Statistics
        stats = compute_stats_table(data, metric)
        all_stats.extend(stats)

        # Correlations
        compute_correlations_detailed(data, metric)

        # Plots
        plot_scatter_pairs(data, metric, OUTPUT_DIR)
        plot_time_series_overlay(data, metric, OUTPUT_DIR)

    # 4. Save stats table
    df_stats = pd.DataFrame(all_stats)
    df_stats.to_csv(f'{OUTPUT_DIR}/stats_per_agent.csv', index=False)

    # 5. Check Phase C data
    check_phase_c_data()

    # 6. Summary
    print("\n" + "=" * 70)
    print("RESUMEN DE AUDITORÍA")
    print("=" * 70)

    for metric, result in identity_results.items():
        status = "⚠️ GLOBAL" if 'GLOBAL' in result else "✓ DISTINTO"
        print(f"  {metric}: {status} ({result})")

    print(f"\n  Archivos guardados en: {OUTPUT_DIR}/")

    # 7. Recommendations
    print("\n" + "=" * 70)
    print("DIAGNÓSTICO Y RECOMENDACIONES")
    print("=" * 70)

    global_metrics = [m for m, r in identity_results.items() if 'GLOBAL' in r]

    if global_metrics:
        print(f"\n  ⚠️  MÉTRICAS GLOBALES DETECTADAS: {global_metrics}")
        print("      Estas métricas NO son evidencia de 'entanglement entre agentes'")
        print("      Son campos globales compartidos, no variables independientes")
        print("\n      CAUSA PROBABLE:")
        print("      - Q_coherence: q_field.get_statistics() retorna valor global, no por agente")
        print("      - H_narr: Los agentes pueden compartir el mismo cálculo de entropía")
        print("      - psi_norm: Si z_visible es idéntico, psi_norm también lo será")
    else:
        print("\n  ✓ Todas las métricas son DISTINTAS por agente")
        print("    Las correlaciones altas son resultado de dinámica acoplada genuina")

    return data, identity_results


if __name__ == '__main__':
    main()
