#!/usr/bin/env python3
"""
Experimento B: ¿Quién sufre más si el otro cambia de carácter?
==============================================================

Mantenemos acople normal.
Distorsionamos drives de UNO solo:
- NEO: más novelty, menos integration
- EVA: más otherness, menos neg_surprise

Miramos:
- Período de crisis
- Sincronía (correlación)
- Número de crisis y % superadas

Hipótesis:
- Si uno es "marcapasos", el otro se arrastra
- Si NEO depende más de EVA (attachment=1.0), sufrirá más cuando EVA mute
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousAgent, AutonomousDualLife


class MutatedDualLife(AutonomousDualLife):
    """Sistema dual donde podemos mutar drives de un agente."""

    def __init__(self, mutate_agent: str = None, mutation_type: str = None):
        super().__init__(dim=6)
        self.mutate_agent = mutate_agent
        self.mutation_type = mutation_type

        # Aplicar mutación inicial
        if mutate_agent and mutation_type:
            self._apply_mutation()

    def _apply_mutation(self):
        """Aplica mutación a los pesos del drive."""
        if self.mutate_agent == "NEO":
            agent = self.neo
        elif self.mutate_agent == "EVA":
            agent = self.eva
        else:
            return

        # Obtener pesos actuales
        # component_names = ['entropy', 'neg_surprise', 'novelty', 'stability', 'integration', 'otherness']
        weights = agent.meta_drive.weights.copy()

        if self.mutation_type == "neo_novelty_up":
            # NEO: más novelty (idx 2), menos integration (idx 4)
            weights[2] *= 2.0  # novelty up
            weights[4] *= 0.3  # integration down

        elif self.mutation_type == "eva_otherness_up":
            # EVA: más otherness (idx 5), menos neg_surprise (idx 1)
            weights[5] *= 2.5  # otherness up
            weights[1] *= 0.3  # neg_surprise down

        elif self.mutation_type == "neo_stability_up":
            # NEO: más stability (idx 3), menos entropy (idx 0)
            weights[3] *= 2.0  # stability up
            weights[0] *= 0.3  # entropy down

        elif self.mutation_type == "eva_entropy_up":
            # EVA: más entropy (idx 0), menos stability (idx 3)
            weights[0] *= 2.5  # entropy up
            weights[3] *= 0.3  # stability down

        # Normalizar
        weights = np.clip(weights, 0.01, None)
        weights = weights / weights.sum()

        agent.meta_drive.weights = weights


def analyze_results(life: AutonomousDualLife, T: int) -> Dict:
    """Analiza resultados de una simulación."""

    # Crisis
    neo_crises = len(life.neo.crises)
    eva_crises = len(life.eva.crises)
    neo_resolved = sum(1 for c in life.neo.crises if c.resolved)
    eva_resolved = sum(1 for c in life.eva.crises if c.resolved)

    # Intervalos entre crisis
    if len(life.neo.crises) > 1:
        neo_intervals = [life.neo.crises[i+1].t - life.neo.crises[i].t
                        for i in range(len(life.neo.crises)-1)]
        neo_mean_interval = np.mean(neo_intervals)
    else:
        neo_mean_interval = T

    if len(life.eva.crises) > 1:
        eva_intervals = [life.eva.crises[i+1].t - life.eva.crises[i].t
                        for i in range(len(life.eva.crises)-1)]
        eva_mean_interval = np.mean(eva_intervals)
    else:
        eva_mean_interval = T

    # Correlación de identidades
    if len(life.neo.identity_history) > 100 and len(life.eva.identity_history) > 100:
        min_len = min(len(life.neo.identity_history), len(life.eva.identity_history))
        corr = np.corrcoef(
            life.neo.identity_history[:min_len],
            life.eva.identity_history[:min_len]
        )[0, 1]
    else:
        corr = 0

    # Período dominante (FFT)
    def get_period(history):
        if len(history) < 100:
            return 0
        signal = np.array(history) - np.mean(history)
        spectrum = np.abs(np.fft.rfft(signal))
        spectrum[0] = 0
        freqs = np.fft.rfftfreq(len(signal))
        peak_idx = np.argmax(spectrum[1:50]) + 1
        return 1/freqs[peak_idx] if freqs[peak_idx] > 0 else 0

    neo_period = get_period(life.neo.identity_history)
    eva_period = get_period(life.eva.identity_history)

    # Bienestar promedio
    neo_wellbeing = np.mean(life.neo.wellbeing_history) if life.neo.wellbeing_history else 0
    eva_wellbeing = np.mean(life.eva.wellbeing_history) if life.eva.wellbeing_history else 0

    # Attachment final
    neo_attachment = life.neo.attachment
    eva_attachment = life.eva.attachment

    return {
        'neo_crises': neo_crises,
        'eva_crises': eva_crises,
        'neo_resolved': neo_resolved,
        'eva_resolved': eva_resolved,
        'neo_interval': neo_mean_interval,
        'eva_interval': eva_mean_interval,
        'correlation': float(corr) if not np.isnan(corr) else 0,
        'neo_period': neo_period,
        'eva_period': eva_period,
        'neo_wellbeing': neo_wellbeing,
        'eva_wellbeing': eva_wellbeing,
        'neo_attachment': neo_attachment,
        'eva_attachment': eva_attachment
    }


def run_experiment_B(T: int = 2000, seeds: List[int] = [42, 123, 456]) -> Dict:
    """
    Experimento B: ¿Quién sufre más si el otro cambia?
    """
    print("=" * 70)
    print("EXPERIMENTO B: ¿QUIÉN SUFRE MÁS SI EL OTRO CAMBIA?")
    print("=" * 70)

    conditions = [
        ('baseline', None, None),
        ('neo_mutated', 'NEO', 'neo_novelty_up'),
        ('eva_mutated', 'EVA', 'eva_otherness_up'),
        ('neo_stability', 'NEO', 'neo_stability_up'),
        ('eva_entropy', 'EVA', 'eva_entropy_up'),
    ]

    all_results = {name: [] for name, _, _ in conditions}

    for seed in seeds:
        print(f"\n{'#'*50}")
        print(f"SEED = {seed}")
        print('#'*50)

        np.random.seed(seed)

        for name, mutate_agent, mutation_type in conditions:
            print(f"\n--- {name} ---")

            life = MutatedDualLife(mutate_agent, mutation_type)

            for t in range(T):
                stimulus = np.random.dirichlet(np.ones(6) * 2)
                if np.random.rand() < 0.02:
                    stimulus += np.random.randn(6) * 0.3
                    stimulus = np.clip(stimulus, 0.01, 0.99)
                    stimulus = stimulus / stimulus.sum()

                life.step(stimulus)

            results = analyze_results(life, T)
            results['condition'] = name
            all_results[name].append(results)

            print(f"  NEO: crises={results['neo_crises']}, period={results['neo_period']:.1f}")
            print(f"  EVA: crises={results['eva_crises']}, period={results['eva_period']:.1f}")
            print(f"  Correlación: {results['correlation']:.3f}")

    # Análisis comparativo
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPARATIVO")
    print("=" * 70)

    print("\n--- Promedios por condición ---")
    print(f"{'Condición':20} {'NEO crisis':>12} {'EVA crisis':>12} {'Corr':>8} {'NEO period':>12} {'EVA period':>12}")
    print("-" * 80)

    for name, _, _ in conditions:
        results_list = all_results[name]
        avg_neo_crises = np.mean([r['neo_crises'] for r in results_list])
        avg_eva_crises = np.mean([r['eva_crises'] for r in results_list])
        avg_corr = np.mean([r['correlation'] for r in results_list])
        avg_neo_period = np.mean([r['neo_period'] for r in results_list])
        avg_eva_period = np.mean([r['eva_period'] for r in results_list])

        print(f"{name:20} {avg_neo_crises:>12.1f} {avg_eva_crises:>12.1f} {avg_corr:>8.3f} {avg_neo_period:>12.1f} {avg_eva_period:>12.1f}")

    # Análisis de impacto
    print("\n--- Impacto de mutaciones ---")

    baseline = all_results['baseline']
    baseline_neo_crises = np.mean([r['neo_crises'] for r in baseline])
    baseline_eva_crises = np.mean([r['eva_crises'] for r in baseline])

    for name, mutate_agent, _ in conditions[1:]:
        results_list = all_results[name]
        neo_crises = np.mean([r['neo_crises'] for r in results_list])
        eva_crises = np.mean([r['eva_crises'] for r in results_list])

        delta_neo = neo_crises - baseline_neo_crises
        delta_eva = eva_crises - baseline_eva_crises

        print(f"\n{name}:")
        print(f"  ΔNEO crisis: {delta_neo:+.1f} ({100*delta_neo/baseline_neo_crises:+.1f}%)")
        print(f"  ΔEVA crisis: {delta_eva:+.1f} ({100*delta_eva/baseline_eva_crises:+.1f}%)")

        if mutate_agent == "NEO":
            # ¿EVA sufrió por la mutación de NEO?
            if delta_eva > delta_neo:
                print(f"  → EVA sufrió MÁS que NEO por la mutación de NEO")
            elif delta_eva < delta_neo:
                print(f"  → NEO sufrió más sus propios cambios")
        else:
            # ¿NEO sufrió por la mutación de EVA?
            if delta_neo > delta_eva:
                print(f"  → NEO sufrió MÁS que EVA por la mutación de EVA")
            elif delta_neo < delta_eva:
                print(f"  → EVA sufrió más sus propios cambios")

    # Determinar quién es más dependiente
    print("\n" + "-" * 50)
    print("CONCLUSIÓN: ¿QUIÉN ES MÁS DEPENDIENTE?")
    print("-" * 50)

    # Calcular impacto cruzado promedio
    neo_impact_when_eva_mutates = []
    eva_impact_when_neo_mutates = []

    for name, mutate_agent, _ in conditions[1:]:
        results_list = all_results[name]
        neo_crises = np.mean([r['neo_crises'] for r in results_list])
        eva_crises = np.mean([r['eva_crises'] for r in results_list])

        if mutate_agent == "NEO":
            eva_impact_when_neo_mutates.append((eva_crises - baseline_eva_crises) / baseline_eva_crises)
        else:
            neo_impact_when_eva_mutates.append((neo_crises - baseline_neo_crises) / baseline_neo_crises)

    avg_neo_impact = np.mean(neo_impact_when_eva_mutates) if neo_impact_when_eva_mutates else 0
    avg_eva_impact = np.mean(eva_impact_when_neo_mutates) if eva_impact_when_neo_mutates else 0

    print(f"\nImpacto cruzado promedio:")
    print(f"  Cuando EVA muta, NEO cambia: {100*avg_neo_impact:+.1f}%")
    print(f"  Cuando NEO muta, EVA cambia: {100*avg_eva_impact:+.1f}%")

    if avg_neo_impact > avg_eva_impact:
        print(f"\n→ NEO es MÁS DEPENDIENTE de EVA")
        print(f"→ EVA es el 'marcapasos' emocional del sistema")
    elif avg_eva_impact > avg_neo_impact:
        print(f"\n→ EVA es MÁS DEPENDIENTE de NEO")
        print(f"→ NEO es el 'marcapasos' emocional del sistema")
    else:
        print(f"\n→ Dependencia SIMÉTRICA")

    # Guardar
    os.makedirs('/root/NEO_EVA/results/who_suffers', exist_ok=True)

    final_results = {
        'timestamp': datetime.now().isoformat(),
        'T': T,
        'seeds': seeds,
        'all_results': all_results,
        'conclusion': {
            'neo_impact_when_eva_mutates': avg_neo_impact,
            'eva_impact_when_neo_mutates': avg_eva_impact,
            'more_dependent': 'NEO' if avg_neo_impact > avg_eva_impact else 'EVA'
        }
    }

    with open('/root/NEO_EVA/results/who_suffers/results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    return final_results


if __name__ == "__main__":
    run_experiment_B(T=2000, seeds=[42, 123, 456])
