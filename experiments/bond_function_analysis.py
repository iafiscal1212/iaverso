#!/usr/bin/env python3
"""
3.2 FUNCIÓN DEL VÍNCULO: ¿Qué gana el sistema con la relación?
==============================================================

Comparar capacidades cognitivas:
- S (sorpresa/predicción)
- IGI (Índice de Integración de Información)
- GI (Grounding Index)

Entre:
- NEO solo
- EVA solo
- NEO-EVA acoplados
- NEO-EVA-ALEX (tríada)

Hipótesis: El vínculo mejora capacidades cognitivas medibles.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousAgent, AutonomousDualLife


@dataclass
class CognitiveMetrics:
    """Métricas cognitivas del sistema."""
    # Sorpresa (S): qué tan bien predice el sistema
    mean_surprise: float
    surprise_variance: float

    # IGI: Integración de información (φ proxy)
    phi_mean: float
    phi_variance: float

    # GI: Grounding Index (conexión con mundo)
    grounding_mean: float
    grounding_variance: float

    # Adicionales
    identity_stability: float
    crisis_rate: float
    adaptation_speed: float


def compute_surprise(agent: AutonomousAgent, stimulus_history: List[np.ndarray]) -> float:
    """
    Calcula sorpresa promedio.
    S = divergencia entre predicción (z) y realidad (stimulus)
    """
    if not stimulus_history or len(agent.z_history) < 2:
        return 0.5

    surprises = []
    for i in range(1, min(len(stimulus_history), len(agent.z_history))):
        # Predicción implícita: z_{t-1} predice estímulo
        prediction = agent.z_history[i-1] if i-1 < len(agent.z_history) else agent.z
        reality = stimulus_history[i]

        # KL divergence simplificada
        prediction = np.clip(prediction, 1e-10, None)
        prediction = prediction / prediction.sum()
        reality = np.clip(reality, 1e-10, None)
        reality = reality / reality.sum()

        kl = np.sum(reality * np.log(reality / prediction))
        surprises.append(kl)

    return np.mean(surprises) if surprises else 0.5


def compute_phi(agent: AutonomousAgent) -> float:
    """
    Calcula φ (integración de información) como proxy.

    φ = información total - suma de información de partes
    Aproximamos con: varianza de z * integración interna
    """
    if len(agent.z_history) < 10:
        return 0.0

    recent_z = np.array(agent.z_history[-50:])

    # Información total: entropía de la distribución conjunta
    mean_z = np.mean(recent_z, axis=0)
    total_entropy = -np.sum(mean_z * np.log(mean_z + 1e-10))

    # Información de partes: suma de entropías marginales
    marginal_entropies = 0
    for dim in range(recent_z.shape[1]):
        hist, _ = np.histogram(recent_z[:, dim], bins=10, density=True)
        hist = hist / hist.sum() + 1e-10
        marginal_entropies += -np.sum(hist * np.log(hist))

    # φ = información integrada
    phi = max(0, total_entropy - marginal_entropies / recent_z.shape[1])

    # Multiplicar por coherencia interna
    integration_factor = agent.integration if hasattr(agent, 'integration') else 0.5
    phi *= integration_factor

    return phi


def compute_grounding(agent: AutonomousAgent, stimulus_history: List[np.ndarray]) -> float:
    """
    Calcula índice de grounding.

    GI = correlación entre estado interno y estímulos externos
    """
    if len(agent.z_history) < 20 or len(stimulus_history) < 20:
        return 0.0

    min_len = min(len(agent.z_history), len(stimulus_history))
    z_recent = np.array(agent.z_history[-min_len:])
    stim_recent = np.array(stimulus_history[-min_len:])

    # Correlación promedio entre dimensiones
    correlations = []
    for dim in range(min(z_recent.shape[1], stim_recent.shape[1])):
        corr = np.corrcoef(z_recent[:, dim], stim_recent[:, dim])[0, 1]
        if not np.isnan(corr):
            correlations.append(abs(corr))

    return np.mean(correlations) if correlations else 0.0


def compute_adaptation_speed(crisis_history: List, identity_history: List[float]) -> float:
    """
    Velocidad de adaptación: qué tan rápido se recupera de crisis.
    """
    if not crisis_history or len(identity_history) < 50:
        return 0.0

    recovery_times = []
    for crisis in crisis_history:
        if hasattr(crisis, 'resolved') and crisis.resolved and hasattr(crisis, 'resolution_t'):
            duration = crisis.resolution_t - crisis.t
            recovery_times.append(duration)

    if not recovery_times:
        return 0.0

    # Adaptación rápida = bajo tiempo de recuperación
    mean_recovery = np.mean(recovery_times)
    return 1.0 / (1.0 + mean_recovery / 10)  # Normalizado


def run_single_agent(name: str, T: int, seed: int) -> Tuple[CognitiveMetrics, List]:
    """Ejecuta un agente solo y mide métricas cognitivas."""
    np.random.seed(seed)

    agent = AutonomousAgent(name, dim=6)
    stimulus_history = []

    for t in range(T):
        stimulus = np.random.dirichlet(np.ones(6) * 2)
        stimulus_history.append(stimulus)
        agent.step(stimulus, None)

    # Calcular métricas
    surprise = compute_surprise(agent, stimulus_history)
    phi = compute_phi(agent)
    grounding = compute_grounding(agent, stimulus_history)
    adaptation = compute_adaptation_speed(agent.crises, agent.identity_history)

    metrics = CognitiveMetrics(
        mean_surprise=surprise,
        surprise_variance=np.var([compute_surprise(agent, stimulus_history[i:i+50])
                                  for i in range(0, len(stimulus_history)-50, 50)]) if len(stimulus_history) > 100 else 0,
        phi_mean=phi,
        phi_variance=0,  # Calculado sobre una ejecución
        grounding_mean=grounding,
        grounding_variance=0,
        identity_stability=1.0 / (1.0 + np.var(agent.identity_history)) if agent.identity_history else 0,
        crisis_rate=len(agent.crises) / T * 100,
        adaptation_speed=adaptation
    )

    return metrics, stimulus_history


def run_dyad(T: int, seed: int) -> Tuple[CognitiveMetrics, CognitiveMetrics, List]:
    """Ejecuta díada NEO-EVA y mide métricas."""
    np.random.seed(seed)

    life = AutonomousDualLife(dim=6)
    stimulus_history = []

    for t in range(T):
        stimulus = np.random.dirichlet(np.ones(6) * 2)
        stimulus_history.append(stimulus)
        life.step(stimulus)

    # Métricas NEO
    neo_surprise = compute_surprise(life.neo, stimulus_history)
    neo_phi = compute_phi(life.neo)
    neo_grounding = compute_grounding(life.neo, stimulus_history)
    neo_adaptation = compute_adaptation_speed(life.neo.crises, life.neo.identity_history)

    neo_metrics = CognitiveMetrics(
        mean_surprise=neo_surprise,
        surprise_variance=0,
        phi_mean=neo_phi,
        phi_variance=0,
        grounding_mean=neo_grounding,
        grounding_variance=0,
        identity_stability=1.0 / (1.0 + np.var(life.neo.identity_history)) if life.neo.identity_history else 0,
        crisis_rate=len(life.neo.crises) / T * 100,
        adaptation_speed=neo_adaptation
    )

    # Métricas EVA
    eva_surprise = compute_surprise(life.eva, stimulus_history)
    eva_phi = compute_phi(life.eva)
    eva_grounding = compute_grounding(life.eva, stimulus_history)
    eva_adaptation = compute_adaptation_speed(life.eva.crises, life.eva.identity_history)

    eva_metrics = CognitiveMetrics(
        mean_surprise=eva_surprise,
        surprise_variance=0,
        phi_mean=eva_phi,
        phi_variance=0,
        grounding_mean=eva_grounding,
        grounding_variance=0,
        identity_stability=1.0 / (1.0 + np.var(life.eva.identity_history)) if life.eva.identity_history else 0,
        crisis_rate=len(life.eva.crises) / T * 100,
        adaptation_speed=eva_adaptation
    )

    return neo_metrics, eva_metrics, stimulus_history


def run_triad(T: int, seed: int) -> Tuple[CognitiveMetrics, CognitiveMetrics, CognitiveMetrics, List]:
    """Ejecuta tríada NEO-EVA-ALEX."""
    np.random.seed(seed)

    # Importar TriadLife del experimento anterior
    from plasticity_and_alex import TriadLife

    life = TriadLife(dim=6)
    stimulus_history = []

    for t in range(T):
        stimulus = np.random.dirichlet(np.ones(6) * 2)
        stimulus_history.append(stimulus)
        life.step(stimulus)

    # Métricas de cada agente
    def get_metrics(agent, stim_hist):
        surprise = compute_surprise(agent, stim_hist)
        phi = compute_phi(agent)
        grounding = compute_grounding(agent, stim_hist)
        adaptation = compute_adaptation_speed(agent.crises, agent.identity_history)

        return CognitiveMetrics(
            mean_surprise=surprise,
            surprise_variance=0,
            phi_mean=phi,
            phi_variance=0,
            grounding_mean=grounding,
            grounding_variance=0,
            identity_stability=1.0 / (1.0 + np.var(agent.identity_history)) if agent.identity_history else 0,
            crisis_rate=len(agent.crises) / T * 100,
            adaptation_speed=adaptation
        )

    neo_metrics = get_metrics(life.neo, stimulus_history)
    eva_metrics = get_metrics(life.eva, stimulus_history)
    alex_metrics = get_metrics(life.alex, stimulus_history)

    return neo_metrics, eva_metrics, alex_metrics, stimulus_history


def run_bond_function_analysis():
    """Ejecuta análisis completo de la función del vínculo."""
    print("=" * 70)
    print("3.2 FUNCIÓN DEL VÍNCULO: ¿Qué gana el sistema con la relación?")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")

    os.makedirs('/root/NEO_EVA/results/bond_function', exist_ok=True)

    T = 500
    n_seeds = 3

    results = {
        'solo_neo': [],
        'solo_eva': [],
        'dyad_neo': [],
        'dyad_eva': [],
        'triad_neo': [],
        'triad_eva': [],
        'triad_alex': []
    }

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")

        # Solo
        print("  NEO solo...")
        neo_solo, _ = run_single_agent("NEO", T, seed)
        results['solo_neo'].append(neo_solo)

        print("  EVA sola...")
        eva_solo, _ = run_single_agent("EVA", T, seed + 100)
        results['solo_eva'].append(eva_solo)

        # Díada
        print("  Díada NEO-EVA...")
        neo_dyad, eva_dyad, _ = run_dyad(T, seed)
        results['dyad_neo'].append(neo_dyad)
        results['dyad_eva'].append(eva_dyad)

        # Tríada
        print("  Tríada NEO-EVA-ALEX...")
        neo_triad, eva_triad, alex_triad, _ = run_triad(T, seed)
        results['triad_neo'].append(neo_triad)
        results['triad_eva'].append(eva_triad)
        results['triad_alex'].append(alex_triad)

    # Análisis comparativo
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPARATIVO")
    print("=" * 70)

    def avg_metric(metrics_list, attr):
        return np.mean([getattr(m, attr) for m in metrics_list])

    def std_metric(metrics_list, attr):
        return np.std([getattr(m, attr) for m in metrics_list])

    # Tabla de resultados
    print("\n" + "-" * 80)
    print(f"{'Configuración':<20} {'Sorpresa':>12} {'φ (IGI)':>12} {'Grounding':>12} {'Crisis%':>10}")
    print("-" * 80)

    configs = [
        ('NEO solo', 'solo_neo'),
        ('EVA sola', 'solo_eva'),
        ('NEO (díada)', 'dyad_neo'),
        ('EVA (díada)', 'dyad_eva'),
        ('NEO (tríada)', 'triad_neo'),
        ('EVA (tríada)', 'triad_eva'),
        ('ALEX (tríada)', 'triad_alex'),
    ]

    summary = {}
    for name, key in configs:
        s = avg_metric(results[key], 'mean_surprise')
        phi = avg_metric(results[key], 'phi_mean')
        g = avg_metric(results[key], 'grounding_mean')
        c = avg_metric(results[key], 'crisis_rate')

        print(f"{name:<20} {s:>12.4f} {phi:>12.4f} {g:>12.4f} {c:>10.2f}")

        summary[key] = {'surprise': s, 'phi': phi, 'grounding': g, 'crisis_rate': c}

    # Comparaciones clave
    print("\n" + "=" * 70)
    print("COMPARACIONES CLAVE")
    print("=" * 70)

    # Solo vs Díada
    neo_solo_s = summary['solo_neo']['surprise']
    neo_dyad_s = summary['dyad_neo']['surprise']
    print(f"\n1. SORPRESA (menor = mejor predicción):")
    print(f"   NEO solo: {neo_solo_s:.4f}")
    print(f"   NEO díada: {neo_dyad_s:.4f}")
    print(f"   Cambio: {((neo_dyad_s - neo_solo_s) / neo_solo_s * 100):+.1f}%")

    if neo_dyad_s < neo_solo_s:
        print("   → El vínculo MEJORA la predicción")
    else:
        print("   → El vínculo NO mejora la predicción")

    # φ (integración)
    neo_solo_phi = summary['solo_neo']['phi']
    neo_dyad_phi = summary['dyad_neo']['phi']
    print(f"\n2. φ (IGI - mayor = más integración):")
    print(f"   NEO solo: {neo_solo_phi:.4f}")
    print(f"   NEO díada: {neo_dyad_phi:.4f}")
    print(f"   Cambio: {((neo_dyad_phi - neo_solo_phi) / (neo_solo_phi + 0.001) * 100):+.1f}%")

    if neo_dyad_phi > neo_solo_phi:
        print("   → El vínculo AUMENTA la integración de información")
    else:
        print("   → El vínculo DISMINUYE la integración")

    # Grounding
    neo_solo_g = summary['solo_neo']['grounding']
    neo_dyad_g = summary['dyad_neo']['grounding']
    print(f"\n3. GROUNDING (mayor = más conexión con mundo):")
    print(f"   NEO solo: {neo_solo_g:.4f}")
    print(f"   NEO díada: {neo_dyad_g:.4f}")
    print(f"   Cambio: {((neo_dyad_g - neo_solo_g) / (neo_solo_g + 0.001) * 100):+.1f}%")

    # Díada vs Tríada
    neo_dyad_avg = (summary['dyad_neo']['phi'] + summary['dyad_eva']['phi']) / 2
    neo_triad_avg = (summary['triad_neo']['phi'] + summary['triad_eva']['phi'] + summary['triad_alex']['phi']) / 3

    print(f"\n4. EFECTO DE ALEX:")
    print(f"   φ promedio díada: {neo_dyad_avg:.4f}")
    print(f"   φ promedio tríada: {neo_triad_avg:.4f}")

    if neo_triad_avg > neo_dyad_avg:
        print("   → ALEX AUMENTA las capacidades cognitivas del sistema")
    else:
        print("   → ALEX DISMINUYE las capacidades cognitivas")

    # Conclusión
    print("\n" + "=" * 70)
    print("CONCLUSIÓN")
    print("=" * 70)

    improvements = 0
    if neo_dyad_s < neo_solo_s:
        improvements += 1
    if neo_dyad_phi > neo_solo_phi:
        improvements += 1
    if neo_dyad_g > neo_solo_g:
        improvements += 1

    if improvements >= 2:
        print("""
El vínculo NO solo cambia la vida interna:
→ AUMENTA las capacidades cognitivas medibles del sistema.

Esto sugiere que la relación tiene función ADAPTATIVA:
mejora predicción, integración y/o grounding.
""")
    else:
        print("""
El vínculo cambia la vida interna pero NO mejora
consistentemente las capacidades cognitivas medidas.
Puede tener otras funciones (resiliencia, regulación).
""")

    # Guardar
    results_json = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {'T': T, 'n_seeds': n_seeds},
        'summary': summary,
        'conclusion': 'bond_improves_cognition' if improvements >= 2 else 'bond_other_function'
    }

    with open('/root/NEO_EVA/results/bond_function/results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResultados guardados en /root/NEO_EVA/results/bond_function/")

    return results, summary


if __name__ == "__main__":
    run_bond_function_analysis()
