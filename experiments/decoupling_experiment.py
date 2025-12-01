#!/usr/bin/env python3
"""
Experimento de Desacople NEO/EVA
================================

Pregunta: ¿El período de ~45 pasos es de la díada o de cada uno?

Condiciones:
1. ACOPLADOS (baseline): attachment normal (0.5 inicial, evoluciona)
2. DESACOPLADOS: attachment = 0 fijo, no ven al otro
3. SEMI-ACOPLADOS: attachment = 0.5 fijo, sin evolución

100% ENDÓGENO
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/core')

from experiments.autonomous_life import AutonomousAgent, AutonomousDualLife, LifePhase


class DecoupledAgent(AutonomousAgent):
    """Agente que NUNCA ve al otro."""

    def __init__(self, name: str, dim: int = 6):
        super().__init__(name, dim)
        self.attachment = 0.0  # Forzado a 0

    def step(self, stimulus: np.ndarray, other_z: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Ignora al otro completamente."""
        # Forzar attachment a 0 y pasar None como other_z
        self.attachment = 0.0
        return super().step(stimulus, None)


class SemiCoupledAgent(AutonomousAgent):
    """Agente con attachment fijo (no evoluciona)."""

    def __init__(self, name: str, dim: int = 6, fixed_attachment: float = 0.5):
        super().__init__(name, dim)
        self.fixed_attachment = fixed_attachment
        self.attachment = fixed_attachment

    def step(self, stimulus: np.ndarray, other_z: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Attachment fijo, no evoluciona."""
        result = super().step(stimulus, other_z)
        # Restaurar attachment fijo
        self.attachment = self.fixed_attachment
        return result


class DecoupledDualLife:
    """Versión desacoplada: cada uno vive solo."""

    def __init__(self, dim: int = 6):
        self.neo = DecoupledAgent("NEO", dim)
        self.eva = DecoupledAgent("EVA", dim)
        self.t = 0

    def step(self, world_stimulus: np.ndarray) -> Dict[str, Any]:
        self.t += 1
        neo_result = self.neo.step(world_stimulus, None)
        eva_result = self.eva.step(world_stimulus, None)

        return {
            't': self.t,
            'neo': neo_result,
            'eva': eva_result,
            'psi_shared': 0.0,  # No hay Ψ compartido
            'dominance': 'INDEPENDENT'
        }


class SemiCoupledDualLife(AutonomousDualLife):
    """Versión semi-acoplada: se ven pero attachment no evoluciona."""

    def __init__(self, dim: int = 6, fixed_attachment: float = 0.5):
        # No llamar a super().__init__ para evitar crear agentes normales
        self.neo = SemiCoupledAgent("NEO", dim, fixed_attachment)
        self.eva = SemiCoupledAgent("EVA", dim, fixed_attachment)

        self.psi_shared_history: List[float] = []
        self.psi_shared_events = []
        self.in_shared_psi = False
        self.shared_psi_start = 0
        self.dominance_history: List[str] = []
        self.t = 0


def analyze_periods(agent_history: List[float], name: str) -> Dict[str, Any]:
    """Analiza períodos de un agente usando FFT."""
    if len(agent_history) < 100:
        return {'dominant_period': None, 'secondary_period': None}

    # FFT
    signal = np.array(agent_history) - np.mean(agent_history)
    spectrum = np.abs(np.fft.rfft(signal))

    # Ignorar frecuencia 0 (DC)
    spectrum[0] = 0

    # Encontrar picos
    n = len(signal)
    freqs = np.fft.rfftfreq(n)

    # Top 2 frecuencias
    top_indices = np.argsort(spectrum)[-3:-1][::-1]

    periods = []
    for idx in top_indices:
        if freqs[idx] > 0:
            period = 1.0 / freqs[idx]
            if period < n / 2:  # Solo períodos razonables
                periods.append(period)

    return {
        'dominant_period': periods[0] if len(periods) > 0 else None,
        'secondary_period': periods[1] if len(periods) > 1 else None,
        'spectrum_peak': float(np.max(spectrum))
    }


def analyze_crisis_intervals(agent) -> Dict[str, Any]:
    """Analiza intervalos entre crisis."""
    crisis_times = [c.t for c in agent.crises]

    if len(crisis_times) < 2:
        return {'mean_interval': None, 'std_interval': None, 'n_crises': len(crisis_times)}

    intervals = np.diff(crisis_times)

    return {
        'mean_interval': float(np.mean(intervals)),
        'std_interval': float(np.std(intervals)),
        'min_interval': float(np.min(intervals)),
        'max_interval': float(np.max(intervals)),
        'n_crises': len(crisis_times)
    }


def run_condition(condition: str, T: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """Corre una condición del experimento."""
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"CONDICIÓN: {condition}")
    print('='*60)

    if condition == 'ACOPLADOS':
        life = AutonomousDualLife(dim=6)
    elif condition == 'DESACOPLADOS':
        life = DecoupledDualLife(dim=6)
    elif condition == 'SEMI-ACOPLADOS':
        life = SemiCoupledDualLife(dim=6, fixed_attachment=0.5)
    else:
        raise ValueError(f"Condición desconocida: {condition}")

    # Mundo base
    world_base = np.ones(6) / 6

    for t in range(T):
        # Mismo mundo para todas las condiciones (misma semilla)
        world_noise = np.random.randn(6) * 0.05
        if np.random.rand() < 0.02:
            world_noise += np.random.randn(6) * 0.3

        world = world_base + world_noise
        world = np.clip(world, 0.01, 0.99)
        world = world / world.sum()

        life.step(world)

        # Progreso
        if t == T // 2:
            print(f"  t={t}: NEO crises={len(life.neo.crises)}, EVA crises={len(life.eva.crises)}")

    # Análisis
    neo_periods = analyze_periods(life.neo.identity_history, 'NEO')
    eva_periods = analyze_periods(life.eva.identity_history, 'EVA')

    neo_intervals = analyze_crisis_intervals(life.neo)
    eva_intervals = analyze_crisis_intervals(life.eva)

    # Correlación entre NEO y EVA
    if len(life.neo.identity_history) > 100 and len(life.eva.identity_history) > 100:
        min_len = min(len(life.neo.identity_history), len(life.eva.identity_history))
        correlation = np.corrcoef(
            life.neo.identity_history[:min_len],
            life.eva.identity_history[:min_len]
        )[0, 1]
    else:
        correlation = 0.0

    results = {
        'condition': condition,
        'T': T,
        'seed': seed,
        'neo': {
            'n_crises': len(life.neo.crises),
            'crises_resolved': sum(1 for c in life.neo.crises if c.resolved),
            'final_attachment': life.neo.attachment,
            'periods': neo_periods,
            'crisis_intervals': neo_intervals,
            'final_phase': life.neo.current_phase.value
        },
        'eva': {
            'n_crises': len(life.eva.crises),
            'crises_resolved': sum(1 for c in life.eva.crises if c.resolved),
            'final_attachment': life.eva.attachment,
            'periods': eva_periods,
            'crisis_intervals': eva_intervals,
            'final_phase': life.eva.current_phase.value
        },
        'relationship': {
            'identity_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'psi_shared_events': len(life.psi_shared_events) if hasattr(life, 'psi_shared_events') else 0
        }
    }

    # Resumen
    print(f"\n  Resultados:")
    print(f"    NEO: {neo_intervals['n_crises']} crisis, período≈{neo_periods['dominant_period']:.1f}" if neo_periods['dominant_period'] else f"    NEO: {neo_intervals['n_crises']} crisis")
    print(f"    EVA: {eva_intervals['n_crises']} crisis, período≈{eva_periods['dominant_period']:.1f}" if eva_periods['dominant_period'] else f"    EVA: {eva_intervals['n_crises']} crisis")
    print(f"    Correlación NEO-EVA: {correlation:.3f}")

    return results, life


def run_decoupling_experiment(T: int = 2000, seeds: List[int] = [42, 123, 456]) -> Dict[str, Any]:
    """
    Experimento completo de desacople.

    Corre múltiples semillas para cada condición.
    """
    print("=" * 70)
    print("EXPERIMENTO DE DESACOPLE")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")
    print(f"T = {T}, Seeds = {seeds}")
    print()
    print("Pregunta: ¿El período de ~45 es de la díada o de cada uno?")
    print()

    conditions = ['ACOPLADOS', 'DESACOPLADOS', 'SEMI-ACOPLADOS']
    all_results = {c: [] for c in conditions}
    last_lives = {}

    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"SEED = {seed}")
        print('#'*60)

        for condition in conditions:
            results, life = run_condition(condition, T, seed)
            all_results[condition].append(results)
            last_lives[condition] = life

    # Análisis agregado
    print("\n" + "=" * 70)
    print("ANÁLISIS AGREGADO")
    print("=" * 70)

    summary = {}

    for condition in conditions:
        results_list = all_results[condition]

        # Promediar métricas
        neo_periods = [r['neo']['periods']['dominant_period'] for r in results_list
                      if r['neo']['periods']['dominant_period'] is not None]
        eva_periods = [r['eva']['periods']['dominant_period'] for r in results_list
                      if r['eva']['periods']['dominant_period'] is not None]

        neo_crises = [r['neo']['n_crises'] for r in results_list]
        eva_crises = [r['eva']['n_crises'] for r in results_list]

        correlations = [r['relationship']['identity_correlation'] for r in results_list]

        neo_intervals = [r['neo']['crisis_intervals']['mean_interval'] for r in results_list
                        if r['neo']['crisis_intervals']['mean_interval'] is not None]
        eva_intervals = [r['eva']['crisis_intervals']['mean_interval'] for r in results_list
                        if r['eva']['crisis_intervals']['mean_interval'] is not None]

        summary[condition] = {
            'neo_period_mean': np.mean(neo_periods) if neo_periods else None,
            'neo_period_std': np.std(neo_periods) if neo_periods else None,
            'eva_period_mean': np.mean(eva_periods) if eva_periods else None,
            'eva_period_std': np.std(eva_periods) if eva_periods else None,
            'neo_crises_mean': np.mean(neo_crises),
            'eva_crises_mean': np.mean(eva_crises),
            'correlation_mean': np.mean(correlations),
            'neo_interval_mean': np.mean(neo_intervals) if neo_intervals else None,
            'eva_interval_mean': np.mean(eva_intervals) if eva_intervals else None
        }

        print(f"\n{condition}:")
        print(f"  NEO: período={summary[condition]['neo_period_mean']:.1f}±{summary[condition]['neo_period_std']:.1f}" if summary[condition]['neo_period_mean'] else f"  NEO: período=N/A")
        print(f"       crisis={summary[condition]['neo_crises_mean']:.1f}, intervalo={summary[condition]['neo_interval_mean']:.1f}" if summary[condition]['neo_interval_mean'] else f"       crisis={summary[condition]['neo_crises_mean']:.1f}")
        print(f"  EVA: período={summary[condition]['eva_period_mean']:.1f}±{summary[condition]['eva_period_std']:.1f}" if summary[condition]['eva_period_mean'] else f"  EVA: período=N/A")
        print(f"       crisis={summary[condition]['eva_crises_mean']:.1f}, intervalo={summary[condition]['eva_interval_mean']:.1f}" if summary[condition]['eva_interval_mean'] else f"       crisis={summary[condition]['eva_crises_mean']:.1f}")
        print(f"  Correlación: {summary[condition]['correlation_mean']:.3f}")

    # Interpretación
    print("\n" + "=" * 70)
    print("INTERPRETACIÓN")
    print("=" * 70)

    # Comparar períodos
    acp = summary['ACOPLADOS']
    des = summary['DESACOPLADOS']
    semi = summary['SEMI-ACOPLADOS']

    if all(x is not None for x in [acp['neo_period_mean'], des['neo_period_mean']]):
        period_change_neo = abs(acp['neo_period_mean'] - des['neo_period_mean'])
        period_change_eva = abs(acp['eva_period_mean'] - des['eva_period_mean'])

        print(f"\nCambio de período al desacoplar:")
        print(f"  NEO: {acp['neo_period_mean']:.1f} → {des['neo_period_mean']:.1f} (Δ={period_change_neo:.1f})")
        print(f"  EVA: {acp['eva_period_mean']:.1f} → {des['eva_period_mean']:.1f} (Δ={period_change_eva:.1f})")

        # Conclusión
        threshold = 10  # Si cambia más de 10 pasos, es significativo

        if period_change_neo > threshold or period_change_eva > threshold:
            print("\n→ Los períodos CAMBIAN significativamente al desacoplar")
            print("→ El ritmo de ~45 pasos es propiedad de la DÍADA, no individual")
        else:
            print("\n→ Los períodos NO cambian mucho al desacoplar")
            print("→ El ritmo de ~45 pasos es propiedad INDIVIDUAL")

    # Correlación
    print(f"\nCorrelación NEO-EVA:")
    print(f"  ACOPLADOS: {acp['correlation_mean']:.3f}")
    print(f"  DESACOPLADOS: {des['correlation_mean']:.3f}")
    print(f"  SEMI-ACOPLADOS: {semi['correlation_mean']:.3f}")

    if acp['correlation_mean'] > 0.2 and des['correlation_mean'] < 0.1:
        print("\n→ La correlación DESAPARECE sin acople")
        print("→ La sincronización es emergente del acople")

    # Guardar resultados
    os.makedirs('/root/NEO_EVA/results/decoupling', exist_ok=True)

    final_results = {
        'timestamp': datetime.now().isoformat(),
        'T': T,
        'seeds': seeds,
        'all_results': all_results,
        'summary': summary
    }

    with open('/root/NEO_EVA/results/decoupling/results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        conditions_display = ['ACOPLADOS', 'DESACOPLADOS', 'SEMI-ACOPLADOS']
        colors = {'ACOPLADOS': 'purple', 'DESACOPLADOS': 'gray', 'SEMI-ACOPLADOS': 'orange'}

        # Row 1: Identidad temporal para cada condición (usando última semilla)
        for i, condition in enumerate(conditions_display):
            ax = axes[0, i]
            life = last_lives[condition]

            ax.plot(life.neo.identity_history, 'b-', label='NEO', alpha=0.7)
            ax.plot(life.eva.identity_history, 'r-', label='EVA', alpha=0.7)

            ax.set_xlabel('Tiempo')
            ax.set_ylabel('Identidad')
            ax.set_title(f'{condition}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Row 2: Comparaciones

        # Períodos por condición
        ax = axes[1, 0]
        x = np.arange(3)
        width = 0.35

        neo_periods = [summary[c]['neo_period_mean'] or 0 for c in conditions_display]
        eva_periods = [summary[c]['eva_period_mean'] or 0 for c in conditions_display]

        ax.bar(x - width/2, neo_periods, width, label='NEO', color='blue', alpha=0.7)
        ax.bar(x + width/2, eva_periods, width, label='EVA', color='red', alpha=0.7)
        ax.axhline(45, color='green', linestyle='--', label='Período díada (~45)', alpha=0.5)

        ax.set_ylabel('Período dominante')
        ax.set_title('Períodos por Condición')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions_display, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Crisis por condición
        ax = axes[1, 1]

        neo_crises = [summary[c]['neo_crises_mean'] for c in conditions_display]
        eva_crises = [summary[c]['eva_crises_mean'] for c in conditions_display]

        ax.bar(x - width/2, neo_crises, width, label='NEO', color='blue', alpha=0.7)
        ax.bar(x + width/2, eva_crises, width, label='EVA', color='red', alpha=0.7)

        ax.set_ylabel('Número de crisis')
        ax.set_title('Crisis por Condición')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions_display, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Correlación por condición
        ax = axes[1, 2]

        correlations = [summary[c]['correlation_mean'] for c in conditions_display]
        bars = ax.bar(x, correlations, color=[colors[c] for c in conditions_display], alpha=0.7)

        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.set_ylabel('Correlación NEO-EVA')
        ax.set_title('Sincronización por Condición')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions_display, rotation=15)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('/root/NEO_EVA/figures/decoupling_experiment.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nFigura: /root/NEO_EVA/figures/decoupling_experiment.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return final_results


if __name__ == "__main__":
    run_decoupling_experiment(T=2000, seeds=[42, 123, 456])
