#!/usr/bin/env python3
"""
RUN Q COALITION GAME - Experimentos del Juego Cuántico Endógeno
===============================================================

Este script:
1. Ejecuta el juego de coalición cuántico
2. Calcula payoffs endógenos
3. Analiza emergencia de patrones
4. Verifica endogeneidad antes de correr

Experimentos:
- 2 agentes (NEO-EVA): ¿emerge coordinación?
- 3 agentes (NEO-EVA-ALEX): ¿coaliciones emergentes?
- Perturbaciones: ¿resiliencia del sistema?
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from datetime import datetime
from collections import defaultdict

# Agregar path
sys.path.insert(0, '/root/NEO_EVA')

from quantum_game.endogenous.coalition_game_qg1 import CoalitionGameQG1
from quantum_game.endogenous.payoff_endogenous import PayoffCalculator, CooperationMetric, PayoffMatrix
from quantum_game.endogenous.audit_q_endogenous import EndogeneityAuditor, run_audit


def ensure_results_dir():
    """Crea directorio de resultados si no existe."""
    results_dir = '/root/NEO_EVA/results/quantum_game_endogenous'
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def run_pre_audit():
    """Ejecuta auditoría antes de experimentos."""
    print("=" * 70)
    print("PRE-AUDITORÍA DE ENDOGENEIDAD")
    print("=" * 70)

    auditor = EndogeneityAuditor()
    summary = auditor.run_full_audit()
    auditor.print_report()

    if summary['errors'] > 0:
        print("\n❌ ABORTANDO: El sistema tiene errores de endogeneidad")
        return False

    print("\n✅ Sistema aprobado para experimentos")
    return True


def experiment_two_agents(num_rounds: int = 500) -> dict:
    """
    Experimento 1: 2 agentes (NEO-EVA)

    Preguntas:
    - ¿Emerge coordinación sin instrucciones explícitas?
    - ¿Se estabiliza el entanglement?
    - ¿Cómo evoluciona el payoff del sistema?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 1: DOS AGENTES (NEO-EVA)")
    print("=" * 70)

    # Crear juego
    game = CoalitionGameQG1(agent_names=['NEO', 'EVA'])

    # Calculadoras de payoff
    payoff_calc = PayoffCalculator(agent_names=['NEO', 'EVA'])
    coop_metric = CooperationMetric(agent_names=['NEO', 'EVA'])
    payoff_matrix = PayoffMatrix(agent_names=['NEO', 'EVA'])

    # Historias para análisis
    histories = {
        'entanglement': [],
        'system_payoff': [],
        'fairness': [],
        'crisis_neo': [],
        'crisis_eva': [],
        'phi_neo': [],
        'phi_eva': [],
        'entropy_neo': [],
        'entropy_eva': [],
    }

    print(f"\nJugando {num_rounds} rondas...")

    for i in range(num_rounds):
        # Jugar ronda
        round_data = game.play_round()

        # Calcular payoffs
        payoffs = payoff_calc.compute_all_payoffs(round_data)
        coop_metric.update(payoffs)
        payoff_matrix.update_from_round(round_data, payoffs)

        # Guardar historias
        histories['entanglement'].append(game.get_entanglement_matrix()[0, 1])
        histories['system_payoff'].append(coop_metric.system_payoff_history[-1])
        histories['fairness'].append(coop_metric.get_fairness_index())
        histories['crisis_neo'].append(game.agents['NEO'].in_crisis)
        histories['crisis_eva'].append(game.agents['EVA'].in_crisis)
        histories['phi_neo'].append(game.agents['NEO'].phi)
        histories['phi_eva'].append(game.agents['EVA'].phi)
        histories['entropy_neo'].append(game.agents['NEO'].entropy)
        histories['entropy_eva'].append(game.agents['EVA'].entropy)

        if (i + 1) % 100 == 0:
            print(f"  Ronda {i+1}/{num_rounds} - Entanglement: {histories['entanglement'][-1]:.3f}")

    # Resultados
    results = {
        'num_rounds': num_rounds,
        'final_entanglement': histories['entanglement'][-1],
        'mean_entanglement': np.mean(histories['entanglement'][-100:]),
        'mean_system_payoff': np.mean(histories['system_payoff'][-100:]),
        'mean_fairness': np.mean(histories['fairness'][-100:]),
        'crisis_rate_neo': np.mean(histories['crisis_neo']),
        'crisis_rate_eva': np.mean(histories['crisis_eva']),
        'operator_stats': game.get_statistics(),
        'nash_approximation': payoff_matrix.get_nash_approximation(),
        'histories': histories
    }

    # Análisis
    print("\n--- RESULTADOS ---")
    print(f"Entanglement final: {results['final_entanglement']:.3f}")
    print(f"Entanglement medio (últimas 100): {results['mean_entanglement']:.3f}")
    print(f"Payoff sistema medio: {results['mean_system_payoff']:.3f}")
    print(f"Fairness media: {results['mean_fairness']:.3f}")
    print(f"Tasa de crisis NEO: {results['crisis_rate_neo']*100:.1f}%")
    print(f"Tasa de crisis EVA: {results['crisis_rate_eva']*100:.1f}%")

    print("\n--- Nash Emergente ---")
    for agent, probs in results['nash_approximation'].items():
        print(f"  {agent}: {probs}")

    return results


def experiment_three_agents(num_rounds: int = 500) -> dict:
    """
    Experimento 2: 3 agentes (NEO-EVA-ALEX)

    Preguntas:
    - ¿Emergen coaliciones (2 vs 1)?
    - ¿Cambia la dinámica con un tercero?
    - ¿ALEX desestabiliza o estabiliza?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 2: TRES AGENTES (NEO-EVA-ALEX)")
    print("=" * 70)

    # Crear juego
    game = CoalitionGameQG1(agent_names=['NEO', 'EVA', 'ALEX'])

    # Calculadoras
    payoff_calc = PayoffCalculator(agent_names=['NEO', 'EVA', 'ALEX'])
    coop_metric = CooperationMetric(agent_names=['NEO', 'EVA', 'ALEX'])

    # Historias
    histories = {
        'ent_neo_eva': [],
        'ent_neo_alex': [],
        'ent_eva_alex': [],
        'system_payoff': [],
        'fairness': [],
        'inequality': [],
    }

    print(f"\nJugando {num_rounds} rondas...")

    for i in range(num_rounds):
        round_data = game.play_round()
        payoffs = payoff_calc.compute_all_payoffs(round_data)
        coop_metric.update(payoffs)

        ent_matrix = game.get_entanglement_matrix()
        histories['ent_neo_eva'].append(ent_matrix[0, 1])
        histories['ent_neo_alex'].append(ent_matrix[0, 2])
        histories['ent_eva_alex'].append(ent_matrix[1, 2])
        histories['system_payoff'].append(coop_metric.system_payoff_history[-1])
        histories['fairness'].append(coop_metric.get_fairness_index())
        histories['inequality'].append(coop_metric.inequality_history[-1])

        if (i + 1) % 100 == 0:
            print(f"  Ronda {i+1} - NEO-EVA: {histories['ent_neo_eva'][-1]:.2f}, "
                  f"NEO-ALEX: {histories['ent_neo_alex'][-1]:.2f}, "
                  f"EVA-ALEX: {histories['ent_eva_alex'][-1]:.2f}")

    results = {
        'num_rounds': num_rounds,
        'final_entanglements': {
            'NEO-EVA': histories['ent_neo_eva'][-1],
            'NEO-ALEX': histories['ent_neo_alex'][-1],
            'EVA-ALEX': histories['ent_eva_alex'][-1]
        },
        'mean_entanglements': {
            'NEO-EVA': np.mean(histories['ent_neo_eva'][-100:]),
            'NEO-ALEX': np.mean(histories['ent_neo_alex'][-100:]),
            'EVA-ALEX': np.mean(histories['ent_eva_alex'][-100:])
        },
        'mean_fairness': np.mean(histories['fairness'][-100:]),
        'mean_inequality': np.mean(histories['inequality'][-100:]),
        'operator_stats': game.get_statistics(),
        'histories': histories
    }

    # Detectar coaliciones
    ents = results['mean_entanglements']
    max_pair = max(ents.keys(), key=lambda k: ents[k])
    min_pair = min(ents.keys(), key=lambda k: ents[k])

    results['strongest_bond'] = max_pair
    results['weakest_bond'] = min_pair
    results['coalition_detected'] = ents[max_pair] > 1.5 * ents[min_pair]

    print("\n--- RESULTADOS ---")
    print(f"Entanglements finales: {results['final_entanglements']}")
    print(f"Vínculo más fuerte: {max_pair} ({ents[max_pair]:.3f})")
    print(f"Vínculo más débil: {min_pair} ({ents[min_pair]:.3f})")
    print(f"¿Coalición detectada?: {'SÍ' if results['coalition_detected'] else 'NO'}")
    print(f"Fairness media: {results['mean_fairness']:.3f}")

    return results


def experiment_perturbation(num_rounds: int = 500) -> dict:
    """
    Experimento 3: Perturbación del sistema

    Introduce crisis artificiales para ver resiliencia.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 3: PERTURBACIÓN Y RESILIENCIA")
    print("=" * 70)

    game = CoalitionGameQG1(agent_names=['NEO', 'EVA'])
    payoff_calc = PayoffCalculator(agent_names=['NEO', 'EVA'])
    coop_metric = CooperationMetric(agent_names=['NEO', 'EVA'])

    histories = {
        'entanglement': [],
        'system_payoff': [],
        'perturbation_points': [],
        'recovery_times': []
    }

    perturbation_interval = 100
    last_perturbation = 0
    pre_perturbation_entanglement = None

    print(f"\nJugando {num_rounds} rondas con perturbaciones cada {perturbation_interval}...")

    for i in range(num_rounds):
        # Perturbación cada N rondas
        if i > 0 and i % perturbation_interval == 0:
            pre_perturbation_entanglement = histories['entanglement'][-1] if histories['entanglement'] else 0

            # Perturbar: resetear drives de NEO a alta varianza
            game.agents['NEO'].drives = np.random.dirichlet(np.ones(6) * 0.1)
            game.agents['NEO'].in_crisis = True

            histories['perturbation_points'].append(i)
            last_perturbation = i
            print(f"  ⚡ Perturbación en ronda {i}")

        round_data = game.play_round()
        payoffs = payoff_calc.compute_all_payoffs(round_data)
        coop_metric.update(payoffs)

        histories['entanglement'].append(game.get_entanglement_matrix()[0, 1])
        histories['system_payoff'].append(coop_metric.system_payoff_history[-1])

        # Detectar recuperación
        if pre_perturbation_entanglement is not None:
            if histories['entanglement'][-1] >= pre_perturbation_entanglement * 0.9:
                recovery_time = i - last_perturbation
                histories['recovery_times'].append(recovery_time)
                pre_perturbation_entanglement = None
                print(f"    ↩ Recuperación en {recovery_time} rondas")

    results = {
        'num_rounds': num_rounds,
        'num_perturbations': len(histories['perturbation_points']),
        'mean_recovery_time': np.mean(histories['recovery_times']) if histories['recovery_times'] else None,
        'recovery_rate': len(histories['recovery_times']) / len(histories['perturbation_points']) if histories['perturbation_points'] else 0,
        'final_entanglement': histories['entanglement'][-1],
        'histories': histories
    }

    print("\n--- RESULTADOS ---")
    print(f"Perturbaciones: {results['num_perturbations']}")
    print(f"Tiempo medio de recuperación: {results['mean_recovery_time']}")
    print(f"Tasa de recuperación: {results['recovery_rate']*100:.1f}%")
    print(f"Entanglement final: {results['final_entanglement']:.3f}")

    return results


def generate_plots(results_2ag, results_3ag, results_pert, results_dir):
    """Genera visualizaciones."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Entanglement 2 agentes
    ax = axes[0, 0]
    ax.plot(results_2ag['histories']['entanglement'], alpha=0.7)
    ax.axhline(y=results_2ag['mean_entanglement'], color='r', linestyle='--',
               label=f'Media: {results_2ag["mean_entanglement"]:.3f}')
    ax.set_title('Entanglement NEO-EVA (2 agentes)')
    ax.set_xlabel('Ronda')
    ax.set_ylabel('Entanglement')
    ax.legend()

    # Plot 2: Entanglements 3 agentes
    ax = axes[0, 1]
    ax.plot(results_3ag['histories']['ent_neo_eva'], label='NEO-EVA', alpha=0.7)
    ax.plot(results_3ag['histories']['ent_neo_alex'], label='NEO-ALEX', alpha=0.7)
    ax.plot(results_3ag['histories']['ent_eva_alex'], label='EVA-ALEX', alpha=0.7)
    ax.set_title('Entanglements (3 agentes)')
    ax.set_xlabel('Ronda')
    ax.set_ylabel('Entanglement')
    ax.legend()

    # Plot 3: Resiliencia
    ax = axes[0, 2]
    ax.plot(results_pert['histories']['entanglement'], alpha=0.7)
    for pp in results_pert['histories']['perturbation_points']:
        ax.axvline(x=pp, color='r', alpha=0.3, linestyle='--')
    ax.set_title('Resiliencia ante perturbaciones')
    ax.set_xlabel('Ronda')
    ax.set_ylabel('Entanglement')

    # Plot 4: φ evolution (2 agentes)
    ax = axes[1, 0]
    ax.plot(results_2ag['histories']['phi_neo'], label='NEO φ', alpha=0.7)
    ax.plot(results_2ag['histories']['phi_eva'], label='EVA φ', alpha=0.7)
    ax.set_title('Integración de Información (φ)')
    ax.set_xlabel('Ronda')
    ax.set_ylabel('φ')
    ax.legend()

    # Plot 5: System payoff
    ax = axes[1, 1]
    ax.plot(results_2ag['histories']['system_payoff'], alpha=0.7, label='2 agentes')
    ax.plot(results_3ag['histories']['system_payoff'], alpha=0.7, label='3 agentes')
    ax.set_title('Payoff del Sistema')
    ax.set_xlabel('Ronda')
    ax.set_ylabel('Payoff')
    ax.legend()

    # Plot 6: Fairness
    ax = axes[1, 2]
    ax.plot(results_2ag['histories']['fairness'], alpha=0.7, label='2 agentes')
    ax.plot(results_3ag['histories']['fairness'], alpha=0.7, label='3 agentes')
    ax.set_title('Fairness del Sistema')
    ax.set_xlabel('Ronda')
    ax.set_ylabel('Fairness')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'quantum_game_analysis.png'), dpi=150)
    plt.close()

    print(f"\n📊 Gráficos guardados en {results_dir}/quantum_game_analysis.png")


def save_results(results_2ag, results_3ag, results_pert, results_dir):
    """Guarda resultados en JSON."""
    # Limpiar historias para JSON (eliminar arrays numpy)
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()
                    if k != 'histories' and k != 'operator_stats'}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment_2_agents': clean_for_json(results_2ag),
        'experiment_3_agents': clean_for_json(results_3ag),
        'experiment_perturbation': clean_for_json(results_pert),
    }

    filepath = os.path.join(results_dir, 'quantum_game_results.json')
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"📄 Resultados guardados en {filepath}")


def main():
    """Ejecuta todos los experimentos."""
    print("=" * 70)
    print("QUANTUM COALITION GAME - EXPERIMENTOS ENDÓGENOS")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Pre-auditoría
    if not run_pre_audit():
        return

    # 2. Crear directorio de resultados
    results_dir = ensure_results_dir()

    # 3. Ejecutar experimentos
    results_2ag = experiment_two_agents(num_rounds=500)
    results_3ag = experiment_three_agents(num_rounds=500)
    results_pert = experiment_perturbation(num_rounds=500)

    # 4. Generar visualizaciones
    generate_plots(results_2ag, results_3ag, results_pert, results_dir)

    # 5. Guardar resultados
    save_results(results_2ag, results_3ag, results_pert, results_dir)

    # 6. Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN DE HALLAZGOS")
    print("=" * 70)

    print("\n1. COORDINACIÓN EMERGENTE (2 agentes):")
    if results_2ag['mean_entanglement'] > 0.5:
        print(f"   ✓ Entanglement significativo: {results_2ag['mean_entanglement']:.3f}")
        print("   → Los agentes se coordinan sin instrucciones explícitas")
    else:
        print(f"   ? Entanglement bajo: {results_2ag['mean_entanglement']:.3f}")

    print("\n2. COALICIONES (3 agentes):")
    if results_3ag['coalition_detected']:
        print(f"   ✓ Coalición detectada: {results_3ag['strongest_bond']}")
        print(f"   → Vínculo excluido: {results_3ag['weakest_bond']}")
    else:
        print("   → No se detectaron coaliciones claras")

    print("\n3. RESILIENCIA:")
    if results_pert['recovery_rate'] > 0.5:
        print(f"   ✓ Sistema resiliente: {results_pert['recovery_rate']*100:.0f}% de recuperaciones")
        print(f"   → Tiempo medio de recuperación: {results_pert['mean_recovery_time']:.0f} rondas")
    else:
        print(f"   ? Baja resiliencia: {results_pert['recovery_rate']*100:.0f}% de recuperaciones")

    print("\n" + "=" * 70)
    print("✓ EXPERIMENTOS COMPLETADOS - TODO ENDÓGENO")
    print("=" * 70)


if __name__ == "__main__":
    main()
