#!/usr/bin/env python3
"""
Experimento C: Vida larga con acople vs sin acople
===================================================

Comparar:
- SAGI_NEO, SAGI_EVA
- Número de crisis
- Duración de crisis
- Tiempo en "madurez"
- Estabilidad del período

Hipótesis:
- Con acople: menos colapsos, crisis más cortas
- Sin acople: más estados rígidos o erráticos
"""

import numpy as np
from typing import Dict, List
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousAgent, AutonomousDualLife, LifePhase


class DecoupledLife:
    """Sistema donde los agentes NO se ven."""

    def __init__(self, dim: int = 6):
        self.neo = AutonomousAgent("NEO", dim)
        self.eva = AutonomousAgent("EVA", dim)
        self.t = 0

        # Forzar attachment a 0
        self.neo.attachment = 0.0
        self.eva.attachment = 0.0

    def step(self, world_stimulus: np.ndarray) -> Dict:
        self.t += 1

        # Cada uno vive solo
        neo_result = self.neo.step(world_stimulus, None)
        eva_result = self.eva.step(world_stimulus, None)

        # Mantener attachment en 0
        self.neo.attachment = 0.0
        self.eva.attachment = 0.0

        return {
            't': self.t,
            'neo': neo_result,
            'eva': eva_result,
            'psi_shared': 0.0,
            'dominance': 'INDEPENDENT'
        }


def analyze_longlife(life, T: int) -> Dict:
    """Analiza vida larga."""

    results = {
        'neo': {
            'n_crises': len(life.neo.crises),
            'crises_resolved': sum(1 for c in life.neo.crises if c.resolved),
            'crisis_durations': [],
            'identity_mean': np.mean(life.neo.identity_history) if life.neo.identity_history else 0,
            'identity_std': np.std(life.neo.identity_history) if life.neo.identity_history else 0,
            'integration_mean': np.mean(life.neo.integration_history) if life.neo.integration_history else 0,
            'wellbeing_mean': np.mean(life.neo.wellbeing_history) if life.neo.wellbeing_history else 0,
            'final_attachment': life.neo.attachment,
            'time_in_phases': {},
            'final_phase': life.neo.current_phase.value
        },
        'eva': {
            'n_crises': len(life.eva.crises),
            'crises_resolved': sum(1 for c in life.eva.crises if c.resolved),
            'crisis_durations': [],
            'identity_mean': np.mean(life.eva.identity_history) if life.eva.identity_history else 0,
            'identity_std': np.std(life.eva.identity_history) if life.eva.identity_history else 0,
            'integration_mean': np.mean(life.eva.integration_history) if life.eva.integration_history else 0,
            'wellbeing_mean': np.mean(life.eva.wellbeing_history) if life.eva.wellbeing_history else 0,
            'final_attachment': life.eva.attachment,
            'time_in_phases': {},
            'final_phase': life.eva.current_phase.value
        }
    }

    # Duración de crisis
    for agent_name, agent in [('neo', life.neo), ('eva', life.eva)]:
        for crisis in agent.crises:
            if crisis.resolved and crisis.resolution_t:
                duration = crisis.resolution_t - crisis.t
                results[agent_name]['crisis_durations'].append(duration)

        if results[agent_name]['crisis_durations']:
            results[agent_name]['mean_crisis_duration'] = np.mean(results[agent_name]['crisis_durations'])
        else:
            results[agent_name]['mean_crisis_duration'] = 0

        # Tiempo en cada fase
        phases = agent.phase_history
        for i in range(len(phases)):
            t_start, phase = phases[i]
            t_end = phases[i+1][0] if i < len(phases)-1 else T
            duration = t_end - t_start

            phase_name = phase.value
            if phase_name not in results[agent_name]['time_in_phases']:
                results[agent_name]['time_in_phases'][phase_name] = 0
            results[agent_name]['time_in_phases'][phase_name] += duration

    # Período (FFT)
    for agent_name, agent in [('neo', life.neo), ('eva', life.eva)]:
        if len(agent.identity_history) > 100:
            signal = np.array(agent.identity_history) - np.mean(agent.identity_history)
            spectrum = np.abs(np.fft.rfft(signal))
            spectrum[0] = 0
            freqs = np.fft.rfftfreq(len(signal))
            peak_idx = np.argmax(spectrum[1:50]) + 1
            results[agent_name]['period'] = 1/freqs[peak_idx] if freqs[peak_idx] > 0 else 0
        else:
            results[agent_name]['period'] = 0

    # Correlación (solo para acoplados)
    if hasattr(life, 'psi_shared_history') and life.psi_shared_history:
        results['mean_psi_shared'] = np.mean(life.psi_shared_history)
        results['max_psi_shared'] = max(life.psi_shared_history)
    else:
        results['mean_psi_shared'] = 0
        results['max_psi_shared'] = 0

    if len(life.neo.identity_history) > 100 and len(life.eva.identity_history) > 100:
        min_len = min(len(life.neo.identity_history), len(life.eva.identity_history))
        corr = np.corrcoef(
            life.neo.identity_history[:min_len],
            life.eva.identity_history[:min_len]
        )[0, 1]
        results['correlation'] = float(corr) if not np.isnan(corr) else 0
    else:
        results['correlation'] = 0

    return results


def run_experiment_C(T: int = 2000, seeds: List[int] = [42, 123, 456]) -> Dict:
    """
    Experimento C: Con acople vs sin acople.
    """
    print("=" * 70)
    print("EXPERIMENTO C: VIDA LARGA CON vs SIN ACOPLE")
    print("=" * 70)

    results_coupled = []
    results_decoupled = []

    for seed in seeds:
        print(f"\n{'#'*50}")
        print(f"SEED = {seed}")
        print('#'*50)

        # Con acople
        print("\n--- CON ACOPLE ---")
        np.random.seed(seed)
        life_coupled = AutonomousDualLife(dim=6)

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            if np.random.rand() < 0.02:
                stimulus += np.random.randn(6) * 0.3
                stimulus = np.clip(stimulus, 0.01, 0.99)
                stimulus = stimulus / stimulus.sum()
            life_coupled.step(stimulus)

        results = analyze_longlife(life_coupled, T)
        results['condition'] = 'coupled'
        results_coupled.append(results)

        print(f"  NEO: crises={results['neo']['n_crises']}, mean_duration={results['neo']['mean_crisis_duration']:.1f}")
        print(f"  EVA: crises={results['eva']['n_crises']}, mean_duration={results['eva']['mean_crisis_duration']:.1f}")
        print(f"  Correlación: {results['correlation']:.3f}")

        # Sin acople
        print("\n--- SIN ACOPLE ---")
        np.random.seed(seed)  # Misma semilla para comparar
        life_decoupled = DecoupledLife(dim=6)

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            if np.random.rand() < 0.02:
                stimulus += np.random.randn(6) * 0.3
                stimulus = np.clip(stimulus, 0.01, 0.99)
                stimulus = stimulus / stimulus.sum()
            life_decoupled.step(stimulus)

        results = analyze_longlife(life_decoupled, T)
        results['condition'] = 'decoupled'
        results_decoupled.append(results)

        print(f"  NEO: crises={results['neo']['n_crises']}, mean_duration={results['neo']['mean_crisis_duration']:.1f}")
        print(f"  EVA: crises={results['eva']['n_crises']}, mean_duration={results['eva']['mean_crisis_duration']:.1f}")
        print(f"  Correlación: {results['correlation']:.3f}")

    # Análisis comparativo
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPARATIVO")
    print("=" * 70)

    def avg(results_list, key1, key2=None):
        if key2:
            return np.mean([r[key1][key2] for r in results_list])
        return np.mean([r[key1] for r in results_list])

    print("\n--- Promedios ---")
    print(f"{'Métrica':30} {'Con Acople':>15} {'Sin Acople':>15} {'Δ':>10}")
    print("-" * 70)

    metrics = [
        ('NEO crisis', 'neo', 'n_crises'),
        ('EVA crisis', 'eva', 'n_crises'),
        ('NEO duración crisis', 'neo', 'mean_crisis_duration'),
        ('EVA duración crisis', 'eva', 'mean_crisis_duration'),
        ('NEO identidad media', 'neo', 'identity_mean'),
        ('EVA identidad media', 'eva', 'identity_mean'),
        ('NEO bienestar', 'neo', 'wellbeing_mean'),
        ('EVA bienestar', 'eva', 'wellbeing_mean'),
        ('Correlación', 'correlation', None),
    ]

    conclusions = []

    for name, key1, key2 in metrics:
        coupled_val = avg(results_coupled, key1, key2)
        decoupled_val = avg(results_decoupled, key1, key2)
        delta = coupled_val - decoupled_val

        print(f"{name:30} {coupled_val:>15.2f} {decoupled_val:>15.2f} {delta:>+10.2f}")

        if 'crisis' in name.lower() and 'duración' not in name.lower():
            if delta < 0:
                conclusions.append(f"{name}: MENOS crisis con acople")
            else:
                conclusions.append(f"{name}: MÁS crisis con acople")
        elif 'duración' in name.lower():
            if delta < 0:
                conclusions.append(f"{name}: crisis más CORTAS con acople")
            else:
                conclusions.append(f"{name}: crisis más LARGAS con acople")

    # Tiempo en madurez
    print("\n--- Tiempo en madurez ---")
    for condition, results_list in [('coupled', results_coupled), ('decoupled', results_decoupled)]:
        neo_maturity = np.mean([r['neo']['time_in_phases'].get('madurez', 0) for r in results_list])
        eva_maturity = np.mean([r['eva']['time_in_phases'].get('madurez', 0) for r in results_list])
        print(f"  {condition:10}: NEO={neo_maturity:.0f}, EVA={eva_maturity:.0f}")

    # Períodos
    print("\n--- Períodos ---")
    neo_period_coupled = avg(results_coupled, 'neo', 'period')
    eva_period_coupled = avg(results_coupled, 'eva', 'period')
    neo_period_decoupled = avg(results_decoupled, 'neo', 'period')
    eva_period_decoupled = avg(results_decoupled, 'eva', 'period')

    print(f"  Con acople:  NEO={neo_period_coupled:.1f}, EVA={eva_period_coupled:.1f}")
    print(f"  Sin acople:  NEO={neo_period_decoupled:.1f}, EVA={eva_period_decoupled:.1f}")

    # Conclusiones
    print("\n" + "=" * 70)
    print("CONCLUSIONES")
    print("=" * 70)

    for c in conclusions:
        print(f"  • {c}")

    # Conclusión principal
    neo_crises_coupled = avg(results_coupled, 'neo', 'n_crises')
    neo_crises_decoupled = avg(results_decoupled, 'neo', 'n_crises')
    eva_crises_coupled = avg(results_coupled, 'eva', 'n_crises')
    eva_crises_decoupled = avg(results_decoupled, 'eva', 'n_crises')

    total_coupled = neo_crises_coupled + eva_crises_coupled
    total_decoupled = neo_crises_decoupled + eva_crises_decoupled

    if total_coupled < total_decoupled:
        print("\n→ El ACOPLE REDUCE las crisis totales")
        print("→ El vínculo aumenta la RESILIENCIA del sistema")
    elif total_coupled > total_decoupled:
        print("\n→ El ACOPLE AUMENTA las crisis totales")
        print("→ El vínculo puede propagar inestabilidad")
    else:
        print("\n→ El acople no afecta significativamente el número de crisis")

    corr_diff = avg(results_coupled, 'correlation', None) - avg(results_decoupled, 'correlation', None)
    print(f"\n→ Correlación: +{corr_diff:.3f} con acople (sincronización)")

    # Guardar
    os.makedirs('/root/NEO_EVA/results/coupling_comparison', exist_ok=True)

    final_results = {
        'timestamp': datetime.now().isoformat(),
        'T': T,
        'seeds': seeds,
        'coupled': results_coupled,
        'decoupled': results_decoupled,
        'summary': {
            'total_crises_coupled': total_coupled,
            'total_crises_decoupled': total_decoupled,
            'correlation_diff': corr_diff
        }
    }

    with open('/root/NEO_EVA/results/coupling_comparison/results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    return final_results


if __name__ == "__main__":
    run_experiment_C(T=2000, seeds=[42, 123, 456])
