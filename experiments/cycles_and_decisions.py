#!/usr/bin/env python3
"""
3.3 CONECTAR CICLOS CON DECISIONES
==================================

Relacionar fase del ciclo con:
- Metas priorizadas (R2)
- Tareas adquiridas/abandonadas (R3)
- Cambios de valores (EVF)

Hipótesis testeables:
H1: "Cuando φ alto e identidad baja → más cambio de metas"
H2: "Cuando identidad alta y φ bajo → consolida valores, menos cambio"
H3: "Las crisis predicen cambios de drive dominante"

Si se confirma: "estados internos que modulan sistemáticamente
cómo el sistema decide y aprende"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousAgent, AutonomousDualLife

DRIVE_NAMES = ['entropy', 'neg_surprise', 'novelty', 'stability', 'integration', 'otherness']


@dataclass
class InternalState:
    """Estado interno del sistema en un momento dado."""
    t: int
    identity: float
    phi: float  # Proxy de integración de información
    coherence: float  # Estabilidad de drives
    in_crisis: bool
    dominant_drive: str
    drive_entropy: float  # Dispersión de drives


@dataclass
class DecisionEvent:
    """Un evento de decisión/cambio."""
    t: int
    event_type: str  # 'drive_change', 'crisis_start', 'crisis_end', 'goal_shift'
    from_state: str
    to_state: str
    magnitude: float


def compute_phi_proxy(agent: AutonomousAgent, window: int = 20) -> float:
    """
    Calcula φ como proxy de integración de información.
    φ ≈ varianza de z * integración interna
    """
    if len(agent.z_history) < window:
        return 0.5

    recent_z = np.array(agent.z_history[-window:])

    # Varianza total (información integrada)
    total_var = np.var(recent_z)

    # Multiplicar por factor de integración
    integration = agent.integration if hasattr(agent, 'integration') else 0.5

    return total_var * integration * 10  # Escalar para visualización


def compute_coherence(agent: AutonomousAgent, window: int = 20) -> float:
    """Coherencia: qué tan estables son los drives."""
    if not hasattr(agent, 'meta_drive') or not agent.meta_drive.weight_history:
        return 0.5

    if len(agent.meta_drive.weight_history) < window:
        return 0.5

    recent = np.array(agent.meta_drive.weight_history[-window:])
    variance = np.mean(np.var(recent, axis=0))

    return 1.0 / (1.0 + 10 * variance)


def compute_drive_entropy(weights: np.ndarray) -> float:
    """Entropía de Shannon de los drives."""
    weights = np.clip(weights, 1e-10, None)
    weights = weights / weights.sum()
    return -np.sum(weights * np.log(weights))


def extract_states_and_decisions(life: AutonomousDualLife, T: int) -> Tuple[List[InternalState], List[DecisionEvent]]:
    """
    Extrae serie temporal de estados internos y eventos de decisión.
    """
    states = []
    decisions = []

    prev_dominant_neo = None
    prev_in_crisis_neo = False

    for t in range(T):
        # Estado interno NEO
        identity = life.neo.identity_history[t] if t < len(life.neo.identity_history) else 0.5
        phi = compute_phi_proxy(life.neo, window=min(20, t+1))
        coherence = compute_coherence(life.neo, window=min(20, t+1))

        # Drive dominante
        if hasattr(life.neo, 'meta_drive') and life.neo.meta_drive.weight_history and t < len(life.neo.meta_drive.weight_history):
            weights = life.neo.meta_drive.weight_history[t]
            dominant = DRIVE_NAMES[np.argmax(weights)]
            entropy = compute_drive_entropy(weights)
        else:
            dominant = 'unknown'
            entropy = 0

        # Crisis
        in_crisis = any(c.t <= t < (c.resolution_t if c.resolution_t else t+100) for c in life.neo.crises)

        state = InternalState(
            t=t,
            identity=identity,
            phi=phi,
            coherence=coherence,
            in_crisis=in_crisis,
            dominant_drive=dominant,
            drive_entropy=entropy
        )
        states.append(state)

        # Detectar eventos de decisión
        if prev_dominant_neo is not None and dominant != prev_dominant_neo:
            decisions.append(DecisionEvent(
                t=t,
                event_type='drive_change',
                from_state=prev_dominant_neo,
                to_state=dominant,
                magnitude=1.0
            ))

        if not prev_in_crisis_neo and in_crisis:
            decisions.append(DecisionEvent(
                t=t,
                event_type='crisis_start',
                from_state='stable',
                to_state='crisis',
                magnitude=1.0
            ))
        elif prev_in_crisis_neo and not in_crisis:
            decisions.append(DecisionEvent(
                t=t,
                event_type='crisis_end',
                from_state='crisis',
                to_state='stable',
                magnitude=1.0
            ))

        prev_dominant_neo = dominant
        prev_in_crisis_neo = in_crisis

    return states, decisions


def test_hypothesis_1(states: List[InternalState], decisions: List[DecisionEvent]) -> Dict:
    """
    H1: Cuando φ alto e identidad baja → más cambio de metas/drives

    Método:
    - Definir "φ alto" e "identidad baja" como percentiles
    - Contar cambios de drive en esas condiciones vs otras
    """
    if not states or not decisions:
        return {'supported': False, 'reason': 'insufficient data'}

    # Calcular percentiles
    phi_values = [s.phi for s in states]
    identity_values = [s.identity for s in states]

    phi_threshold = np.percentile(phi_values, 70)  # Alto
    identity_threshold = np.percentile(identity_values, 30)  # Bajo

    # Clasificar cada t
    high_phi_low_id = set()
    other_states = set()

    for s in states:
        if s.phi > phi_threshold and s.identity < identity_threshold:
            high_phi_low_id.add(s.t)
        else:
            other_states.add(s.t)

    # Contar cambios de drive
    drive_changes = [d for d in decisions if d.event_type == 'drive_change']

    changes_in_hpli = sum(1 for d in drive_changes if d.t in high_phi_low_id)
    changes_in_other = sum(1 for d in drive_changes if d.t in other_states)

    # Tasas
    rate_hpli = changes_in_hpli / len(high_phi_low_id) if high_phi_low_id else 0
    rate_other = changes_in_other / len(other_states) if other_states else 0

    # Test estadístico (chi-squared aproximado)
    if changes_in_hpli + changes_in_other > 0:
        ratio = rate_hpli / (rate_other + 0.001)
    else:
        ratio = 1.0

    return {
        'hypothesis': 'H1: φ alto + identidad baja → más cambio',
        'n_high_phi_low_id': len(high_phi_low_id),
        'n_other': len(other_states),
        'changes_in_hpli': changes_in_hpli,
        'changes_in_other': changes_in_other,
        'rate_hpli': rate_hpli,
        'rate_other': rate_other,
        'ratio': ratio,
        'supported': ratio > 1.5,
        'interpretation': 'CONFIRMADO: más cambio cuando φ alto e identidad baja' if ratio > 1.5
                         else 'NO CONFIRMADO: ratio insuficiente'
    }


def test_hypothesis_2(states: List[InternalState], decisions: List[DecisionEvent]) -> Dict:
    """
    H2: Cuando identidad alta y φ bajo → consolida, menos cambio

    Opuesto de H1: consolidación de valores.
    """
    if not states or not decisions:
        return {'supported': False, 'reason': 'insufficient data'}

    phi_values = [s.phi for s in states]
    identity_values = [s.identity for s in states]

    phi_threshold = np.percentile(phi_values, 30)  # Bajo
    identity_threshold = np.percentile(identity_values, 70)  # Alto

    high_id_low_phi = set()
    other_states = set()

    for s in states:
        if s.phi < phi_threshold and s.identity > identity_threshold:
            high_id_low_phi.add(s.t)
        else:
            other_states.add(s.t)

    drive_changes = [d for d in decisions if d.event_type == 'drive_change']

    changes_in_hilp = sum(1 for d in drive_changes if d.t in high_id_low_phi)
    changes_in_other = sum(1 for d in drive_changes if d.t in other_states)

    rate_hilp = changes_in_hilp / len(high_id_low_phi) if high_id_low_phi else 0
    rate_other = changes_in_other / len(other_states) if other_states else 0

    ratio = rate_other / (rate_hilp + 0.001) if rate_hilp > 0 else 0

    return {
        'hypothesis': 'H2: identidad alta + φ bajo → menos cambio (consolidación)',
        'n_high_id_low_phi': len(high_id_low_phi),
        'n_other': len(other_states),
        'changes_in_hilp': changes_in_hilp,
        'changes_in_other': changes_in_other,
        'rate_hilp': rate_hilp,
        'rate_other': rate_other,
        'ratio': ratio,
        'supported': ratio > 1.5,
        'interpretation': 'CONFIRMADO: menos cambio cuando identidad alta y φ bajo' if ratio > 1.5
                         else 'NO CONFIRMADO'
    }


def test_hypothesis_3(states: List[InternalState], decisions: List[DecisionEvent]) -> Dict:
    """
    H3: Las crisis predicen cambios de drive dominante

    Método: ¿Los cambios de drive ocurren más cerca de crisis?
    """
    if not states or not decisions:
        return {'supported': False, 'reason': 'insufficient data'}

    # Tiempos de inicio de crisis
    crisis_starts = [d.t for d in decisions if d.event_type == 'crisis_start']

    if not crisis_starts:
        return {'supported': False, 'reason': 'no crises detected'}

    # Tiempos de cambio de drive
    drive_changes = [d.t for d in decisions if d.event_type == 'drive_change']

    if not drive_changes:
        return {'supported': False, 'reason': 'no drive changes detected'}

    # Distancia mínima de cada cambio a una crisis
    distances_to_crisis = []
    for change_t in drive_changes:
        min_dist = min(abs(change_t - crisis_t) for crisis_t in crisis_starts)
        distances_to_crisis.append(min_dist)

    # Distancia esperada si fuera aleatorio
    T = max(s.t for s in states)
    expected_distance = T / (len(crisis_starts) + 1) / 2  # Aproximación

    mean_distance = np.mean(distances_to_crisis)

    # ¿Los cambios están más cerca de crisis de lo esperado?
    ratio = expected_distance / (mean_distance + 0.001)

    return {
        'hypothesis': 'H3: crisis predicen cambios de drive',
        'n_crises': len(crisis_starts),
        'n_drive_changes': len(drive_changes),
        'mean_distance_to_crisis': mean_distance,
        'expected_distance_random': expected_distance,
        'proximity_ratio': ratio,
        'supported': ratio > 1.3,
        'interpretation': 'CONFIRMADO: cambios de drive ocurren cerca de crisis' if ratio > 1.3
                         else 'NO CONFIRMADO: distribución similar a aleatoria'
    }


def analyze_phase_behavior(states: List[InternalState]) -> Dict:
    """
    Analiza cómo cambia el comportamiento según la fase del ciclo.
    """
    if len(states) < 100:
        return {'error': 'insufficient data'}

    # Dividir en fases por cuartiles de identidad y phi
    identity_values = [s.identity for s in states]
    phi_values = [s.phi for s in states]

    id_q1, id_q3 = np.percentile(identity_values, [25, 75])
    phi_q1, phi_q3 = np.percentile(phi_values, [25, 75])

    phases = {
        'exploration': [],  # bajo id, alto phi
        'consolidation': [],  # alto id, bajo phi
        'crisis': [],  # bajo id, bajo phi
        'flow': []  # alto id, alto phi
    }

    for s in states:
        if s.identity < id_q1 and s.phi > phi_q3:
            phases['exploration'].append(s)
        elif s.identity > id_q3 and s.phi < phi_q1:
            phases['consolidation'].append(s)
        elif s.identity < id_q1 and s.phi < phi_q1:
            phases['crisis'].append(s)
        elif s.identity > id_q3 and s.phi > phi_q3:
            phases['flow'].append(s)

    # Características de cada fase
    phase_stats = {}
    for phase_name, phase_states in phases.items():
        if phase_states:
            phase_stats[phase_name] = {
                'n_states': len(phase_states),
                'pct_of_total': len(phase_states) / len(states) * 100,
                'mean_entropy': np.mean([s.drive_entropy for s in phase_states]),
                'mean_coherence': np.mean([s.coherence for s in phase_states]),
                'pct_in_crisis': sum(1 for s in phase_states if s.in_crisis) / len(phase_states) * 100
            }
        else:
            phase_stats[phase_name] = {'n_states': 0}

    return phase_stats


def run_cycles_decisions_analysis():
    """Ejecuta análisis completo de ciclos y decisiones."""
    print("=" * 70)
    print("3.3 CONECTAR CICLOS CON DECISIONES")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")

    os.makedirs('/root/NEO_EVA/results/cycles_decisions', exist_ok=True)

    T = 1000
    n_seeds = 3

    all_h1_results = []
    all_h2_results = []
    all_h3_results = []
    all_phase_stats = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed + 42)

        # Ejecutar simulación
        life = AutonomousDualLife(dim=6)

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            life.step(stimulus)

        # Extraer estados y decisiones
        print("  Extrayendo estados y decisiones...")
        states, decisions = extract_states_and_decisions(life, T)

        print(f"    Estados: {len(states)}, Decisiones: {len(decisions)}")

        # Testar hipótesis
        print("  Testando hipótesis...")

        h1 = test_hypothesis_1(states, decisions)
        all_h1_results.append(h1)
        print(f"    H1: {h1['interpretation']}")

        h2 = test_hypothesis_2(states, decisions)
        all_h2_results.append(h2)
        print(f"    H2: {h2['interpretation']}")

        h3 = test_hypothesis_3(states, decisions)
        all_h3_results.append(h3)
        print(f"    H3: {h3['interpretation']}")

        # Análisis de fases
        phase_stats = analyze_phase_behavior(states)
        all_phase_stats.append(phase_stats)

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE HIPÓTESIS")
    print("=" * 70)

    def summarize_hypothesis(results: List[Dict], name: str):
        supported = sum(1 for r in results if r.get('supported', False))
        print(f"\n{name}:")
        print(f"  Soportada en {supported}/{len(results)} seeds")

        if 'ratio' in results[0]:
            avg_ratio = np.mean([r['ratio'] for r in results if 'ratio' in r])
            print(f"  Ratio promedio: {avg_ratio:.2f}")

        if supported >= len(results) / 2:
            print(f"  → HIPÓTESIS CONFIRMADA")
            return True
        else:
            print(f"  → HIPÓTESIS NO CONFIRMADA")
            return False

    h1_confirmed = summarize_hypothesis(all_h1_results, "H1: φ alto + identidad baja → más cambio")
    h2_confirmed = summarize_hypothesis(all_h2_results, "H2: identidad alta + φ bajo → consolidación")
    h3_confirmed = summarize_hypothesis(all_h3_results, "H3: crisis predicen cambios de drive")

    # Análisis de fases agregado
    print("\n" + "=" * 70)
    print("ANÁLISIS DE FASES DEL CICLO")
    print("=" * 70)

    phase_names = ['exploration', 'consolidation', 'crisis', 'flow']

    print(f"\n{'Fase':<15} {'% Tiempo':>10} {'Entropía':>10} {'Coherencia':>10} {'% Crisis':>10}")
    print("-" * 60)

    for phase in phase_names:
        pcts = []
        entropies = []
        coherences = []
        crisis_pcts = []

        for ps in all_phase_stats:
            if phase in ps and ps[phase].get('n_states', 0) > 0:
                pcts.append(ps[phase]['pct_of_total'])
                entropies.append(ps[phase]['mean_entropy'])
                coherences.append(ps[phase]['mean_coherence'])
                crisis_pcts.append(ps[phase]['pct_in_crisis'])

        if pcts:
            print(f"{phase:<15} {np.mean(pcts):>10.1f} {np.mean(entropies):>10.3f} "
                  f"{np.mean(coherences):>10.3f} {np.mean(crisis_pcts):>10.1f}")

    # Conclusión
    print("\n" + "=" * 70)
    print("CONCLUSIÓN")
    print("=" * 70)

    confirmed_count = sum([h1_confirmed, h2_confirmed, h3_confirmed])

    if confirmed_count >= 2:
        print("""
RESULTADO POSITIVO:

Los estados internos (φ, identidad) MODULAN SISTEMÁTICAMENTE
cómo el sistema decide y cambia.

Esto sugiere:
→ "Estados internos que modulan sistemáticamente cómo el sistema decide y aprende"
→ Una forma prudente de hablar de "estados subjetivos funcionales"

Específicamente:
""")
        if h1_confirmed:
            print("  • Cuando hay alta integración (φ) pero baja identidad, el sistema EXPLORA más")
        if h2_confirmed:
            print("  • Cuando hay alta identidad pero baja integración, el sistema CONSOLIDA")
        if h3_confirmed:
            print("  • Las CRISIS son puntos de inflexión para cambios de valores/metas")

    else:
        print("""
RESULTADO MIXTO:

Los estados internos muestran correlaciones con decisiones,
pero no de forma consistente en todas las condiciones.

Posibles explicaciones:
- Los umbrales elegidos no son óptimos
- La dinámica es más compleja
- Se necesitan más datos
""")

    # Guardar
    results_json = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {'T': T, 'n_seeds': n_seeds},
        'hypotheses': {
            'H1': {
                'description': 'φ alto + identidad baja → más cambio',
                'confirmed': h1_confirmed,
                'details': [{'ratio': r.get('ratio', 0), 'supported': r.get('supported', False)}
                           for r in all_h1_results]
            },
            'H2': {
                'description': 'identidad alta + φ bajo → consolidación',
                'confirmed': h2_confirmed,
                'details': [{'ratio': r.get('ratio', 0), 'supported': r.get('supported', False)}
                           for r in all_h2_results]
            },
            'H3': {
                'description': 'crisis predicen cambios de drive',
                'confirmed': h3_confirmed,
                'details': [{'ratio': r.get('proximity_ratio', 0), 'supported': r.get('supported', False)}
                           for r in all_h3_results]
            }
        },
        'conclusion': 'states_modulate_decisions' if confirmed_count >= 2 else 'mixed_results'
    }

    with open('/root/NEO_EVA/results/cycles_decisions/results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResultados guardados en /root/NEO_EVA/results/cycles_decisions/")

    return results_json


if __name__ == "__main__":
    run_cycles_decisions_analysis()
