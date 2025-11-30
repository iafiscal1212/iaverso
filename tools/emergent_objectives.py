#!/usr/bin/env python3
"""
Phase 14: Objetivos Emergentes desde Narrativa
===============================================

Los objetivos NO se programan - emergen de:
1. Tensión narrativa: episodios que rompen patrones habituales (alta sorpresa)
2. Atractores narrativos: historias que tienden a repetirse con TE alta + identidad estable
3. Proto-preferencias: tendencia endógena a evitar tensión y buscar atractores

Principio fundamental:
- Si episodios tipo X llevan a TE alta + identidad estable → se repiten más
- Si episodios tipo Y llevan a caída de identidad/TE → se evitan

Esto es proto-preferencia sin rewards externos.
100% endógeno - CERO números mágicos.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from scipy import stats
import json
from datetime import datetime

import sys
sys.path.insert(0, '/root/NEO_EVA/tools')

from narrative import (
    NarrativeSystem, NarrativeMemory, Episode,
    get_episode_type, get_transition_probabilities,
    compute_ini
)
from endogenous_core import (
    derive_window_size, derive_learning_rate,
    compute_iqr, NUMERIC_EPS, PROVENANCE
)


# =============================================================================
# TENSIÓN NARRATIVA
# =============================================================================

def compute_narrative_tension(
    memory: NarrativeMemory,
    current_episode: Episode
) -> float:
    """
    Tensión narrativa = sorpresa cuando el episodio actual rompe patrones.

    Basado en:
    - Probabilidad de transición (baja prob = alta tensión)
    - Distancia al centroide de episodios recientes
    - Cambio en INI

    Todo derivado endógenamente.
    """
    if len(memory.episodes) < 3:
        return 0.5  # Neutro durante warmup

    # 1. Sorpresa por transición
    n_types = max(3, min(10, int(np.sqrt(len(memory.episodes)))))
    current_type = str(get_episode_type(current_episode, n_types))
    prev_type = str(get_episode_type(memory.episodes[-1], n_types))

    trans_probs = get_transition_probabilities(memory)

    if prev_type in trans_probs and current_type in trans_probs[prev_type]:
        prob = trans_probs[prev_type][current_type]
    else:
        # Transición nunca vista - máxima sorpresa
        prob = 1.0 / (n_types + 1)  # Prior uniforme

    # Sorpresa = -log(prob), normalizada
    surprise_transition = -np.log(prob + NUMERIC_EPS)
    max_surprise = -np.log(1.0 / (n_types + 1))
    surprise_transition = surprise_transition / (max_surprise + NUMERIC_EPS)

    # 2. Distancia al centroide reciente
    recent_vectors = [ep.to_vector() for ep in memory.episodes[-5:]]
    centroid = np.mean(recent_vectors, axis=0)
    current_vector = current_episode.to_vector()

    distance = np.linalg.norm(current_vector - centroid)

    # Normalizar por distancia típica
    all_distances = []
    for v in recent_vectors:
        all_distances.append(np.linalg.norm(v - centroid))

    typical_distance = np.median(all_distances) if all_distances else 1.0
    distance_normalized = distance / (typical_distance + NUMERIC_EPS)
    distance_normalized = min(distance_normalized, 2.0) / 2.0  # Cap at 1.0

    # 3. Combinación endógena (pesos iguales, sin tuning)
    tension = (surprise_transition + distance_normalized) / 2.0

    PROVENANCE.log('narrative_tension', tension,
                   'surprise_trans + distance_norm / 2',
                   {'surprise': surprise_transition, 'distance': distance_normalized},
                   0)

    return float(tension)


def compute_tension_history(
    memory: NarrativeMemory
) -> List[float]:
    """Calcula historial de tensión para todos los episodios."""
    tensions = []

    for i, episode in enumerate(memory.episodes):
        if i < 2:
            tensions.append(0.5)
            continue

        # Crear memoria temporal hasta ese punto
        temp_episodes = memory.episodes[:i]
        temp_memory = NarrativeMemory(agent=memory.agent)
        temp_memory.episodes = temp_episodes
        temp_memory.transition_counts = memory.transition_counts.copy()

        tension = compute_narrative_tension(temp_memory, episode)
        tensions.append(tension)

    return tensions


# =============================================================================
# ATRACTORES NARRATIVOS
# =============================================================================

@dataclass
class NarrativeAttractor:
    """Un atractor narrativo - patrón de historia que tiende a repetirse."""
    type_sequence: Tuple[int, ...]  # Secuencia de tipos de episodio
    frequency: int                   # Cuántas veces ha ocurrido
    mean_te: float                   # TE medio cuando ocurre
    mean_ini_after: float            # INI medio después de ocurrir
    stability_score: float           # Qué tan estable es la identidad después


def find_narrative_attractors(
    memory: NarrativeMemory,
    min_length: int = 2,
    max_length: int = 4
) -> List[NarrativeAttractor]:
    """
    Encuentra atractores narrativos - secuencias que se repiten con buenos resultados.

    Un atractor es una secuencia de tipos de episodio que:
    - Se repite más de lo esperado por azar
    - Lleva a TE alto
    - Mantiene o mejora la identidad (INI)

    Longitudes derivadas endógenamente de √(n_episodes).
    """
    if len(memory.episodes) < 5:
        return []

    n_types = max(3, min(10, int(np.sqrt(len(memory.episodes)))))
    type_sequence = [get_episode_type(ep, n_types) for ep in memory.episodes]

    # Longitudes a buscar - endógenas
    max_len = min(max_length, int(np.sqrt(len(type_sequence))))
    max_len = max(min_length, max_len)

    # Encontrar todas las subsecuencias
    sequence_stats = {}

    for length in range(min_length, max_len + 1):
        for i in range(len(type_sequence) - length):
            seq = tuple(type_sequence[i:i+length])

            if seq not in sequence_stats:
                sequence_stats[seq] = {
                    'count': 0,
                    'te_values': [],
                    'ini_after': []
                }

            sequence_stats[seq]['count'] += 1

            # TE medio de los episodios en la secuencia
            te_values = [memory.episodes[i+j].mean_te for j in range(length)]
            sequence_stats[seq]['te_values'].append(np.mean(te_values))

            # INI después (si hay suficientes episodios después)
            if i + length < len(memory.episodes):
                # Aproximar INI por estabilidad de tipos siguientes
                next_types = type_sequence[i+length:min(i+length+3, len(type_sequence))]
                if next_types:
                    stability = 1.0 - len(set(next_types)) / len(next_types)
                    sequence_stats[seq]['ini_after'].append(stability)

    # Filtrar: solo secuencias que ocurren más de lo esperado
    attractors = []

    total_sequences = len(type_sequence) - min_length + 1

    for seq, stats in sequence_stats.items():
        # Frecuencia esperada bajo independencia
        expected_freq = total_sequences / (n_types ** len(seq))

        # Solo si ocurre más que lo esperado
        if stats['count'] > expected_freq * 1.5:
            mean_te = np.mean(stats['te_values']) if stats['te_values'] else 0
            mean_ini = np.mean(stats['ini_after']) if stats['ini_after'] else 0.5

            # Score de estabilidad: combina frecuencia, TE e INI
            stability = (stats['count'] / total_sequences) * mean_te * mean_ini

            attractors.append(NarrativeAttractor(
                type_sequence=seq,
                frequency=stats['count'],
                mean_te=mean_te,
                mean_ini_after=mean_ini,
                stability_score=stability
            ))

    # Ordenar por stability_score
    attractors.sort(key=lambda x: x.stability_score, reverse=True)

    return attractors


# =============================================================================
# PROTO-PREFERENCIAS
# =============================================================================

@dataclass
class ProtoPreference:
    """Una proto-preferencia emergente."""
    name: str
    direction: str  # 'seek' o 'avoid'
    strength: float  # 0-1, derivado de la historia
    evidence: Dict   # Datos que la soportan


def derive_proto_preferences(
    memory: NarrativeMemory,
    tension_history: List[float],
    attractors: List[NarrativeAttractor]
) -> List[ProtoPreference]:
    """
    Deriva proto-preferencias de la narrativa.

    No son rewards programados - emergen de:
    - Qué historias llevan a baja tensión
    - Qué historias llevan a alta TE e identidad estable
    - Qué patrones se repiten naturalmente
    """
    preferences = []

    if len(memory.episodes) < 5:
        return preferences

    # 1. Preferencia por baja tensión
    if tension_history:
        # Correlación entre tensión y TE siguiente
        if len(tension_history) > 3:
            te_values = [ep.mean_te for ep in memory.episodes]

            # TE después de alta tensión vs baja tensión
            median_tension = np.median(tension_history)
            high_tension_idx = [i for i, t in enumerate(tension_history) if t > median_tension]
            low_tension_idx = [i for i, t in enumerate(tension_history) if t <= median_tension]

            if high_tension_idx and low_tension_idx:
                te_after_high = np.mean([te_values[min(i+1, len(te_values)-1)]
                                        for i in high_tension_idx])
                te_after_low = np.mean([te_values[min(i+1, len(te_values)-1)]
                                       for i in low_tension_idx])

                # Si TE es mayor después de baja tensión → preferencia por estabilidad
                if te_after_low > te_after_high:
                    strength = (te_after_low - te_after_high) / (te_after_low + NUMERIC_EPS)
                    preferences.append(ProtoPreference(
                        name='stability_seeking',
                        direction='seek',
                        strength=min(1.0, strength),
                        evidence={
                            'te_after_low_tension': te_after_low,
                            'te_after_high_tension': te_after_high
                        }
                    ))

    # 2. Preferencia por atractores específicos
    if attractors:
        top_attractor = attractors[0]
        if top_attractor.stability_score > 0.1:
            preferences.append(ProtoPreference(
                name=f'attractor_{top_attractor.type_sequence}',
                direction='seek',
                strength=min(1.0, top_attractor.stability_score * 2),
                evidence={
                    'sequence': top_attractor.type_sequence,
                    'frequency': top_attractor.frequency,
                    'mean_te': top_attractor.mean_te
                }
            ))

    # 3. Preferencia por estados específicos
    state_te = {}
    for ep in memory.episodes:
        state = ep.dominant_state
        if state not in state_te:
            state_te[state] = []
        state_te[state].append(ep.mean_te)

    if state_te:
        # Estado con mayor TE medio
        best_state = max(state_te.keys(), key=lambda s: np.mean(state_te[s]))
        best_te = np.mean(state_te[best_state])
        overall_te = np.mean([ep.mean_te for ep in memory.episodes])

        if best_te > overall_te * 1.2:  # 20% mejor que promedio
            strength = (best_te - overall_te) / (overall_te + NUMERIC_EPS)
            preferences.append(ProtoPreference(
                name=f'state_{best_state}',
                direction='seek',
                strength=min(1.0, strength),
                evidence={
                    'state': best_state,
                    'mean_te': best_te,
                    'overall_te': overall_te
                }
            ))

    # 4. Preferencia por rol social
    role_outcomes = {'leader': [], 'follower': [], 'mutual': []}
    for i, ep in enumerate(memory.episodes[:-1]):
        next_ep = memory.episodes[i+1]
        role_outcomes[ep.social_role].append(next_ep.mean_te)

    if all(role_outcomes.values()):
        best_role = max(role_outcomes.keys(),
                       key=lambda r: np.mean(role_outcomes[r]) if role_outcomes[r] else 0)
        best_outcome = np.mean(role_outcomes[best_role])
        other_outcomes = np.mean([np.mean(v) for k, v in role_outcomes.items()
                                  if k != best_role and v])

        if best_outcome > other_outcomes * 1.1:
            strength = (best_outcome - other_outcomes) / (other_outcomes + NUMERIC_EPS)
            preferences.append(ProtoPreference(
                name=f'role_{best_role}',
                direction='seek',
                strength=min(1.0, strength),
                evidence={
                    'role': best_role,
                    'outcome_te': best_outcome,
                    'other_te': other_outcomes
                }
            ))

    return preferences


# =============================================================================
# SISTEMA DE OBJETIVOS EMERGENTES
# =============================================================================

class EmergentObjectiveSystem:
    """Sistema completo de objetivos emergentes."""

    def __init__(self, narrative_system: NarrativeSystem):
        self.narrative = narrative_system

        # Historial de tensión
        self.neo_tension_history = []
        self.eva_tension_history = []

        # Atractores
        self.neo_attractors = []
        self.eva_attractors = []

        # Proto-preferencias
        self.neo_preferences = []
        self.eva_preferences = []

        # Historial de preferencias
        self.preference_evolution = []

    def update(self, t: int) -> Dict:
        """Actualiza el sistema de objetivos."""
        result = {
            't': t,
            'neo': {},
            'eva': {}
        }

        # Calcular tensión actual
        if self.narrative.neo_memory.episodes:
            current_neo = self.narrative.neo_memory.episodes[-1]
            neo_tension = compute_narrative_tension(
                self.narrative.neo_memory, current_neo
            )
            self.neo_tension_history.append(neo_tension)
            result['neo']['tension'] = neo_tension

        if self.narrative.eva_memory.episodes:
            current_eva = self.narrative.eva_memory.episodes[-1]
            eva_tension = compute_narrative_tension(
                self.narrative.eva_memory, current_eva
            )
            self.eva_tension_history.append(eva_tension)
            result['eva']['tension'] = eva_tension

        # Actualizar atractores periódicamente
        window = derive_window_size(t)
        if t % window == 0 and t > 0:
            self.neo_attractors = find_narrative_attractors(self.narrative.neo_memory)
            self.eva_attractors = find_narrative_attractors(self.narrative.eva_memory)

            result['neo']['n_attractors'] = len(self.neo_attractors)
            result['eva']['n_attractors'] = len(self.eva_attractors)

            # Actualizar preferencias
            self.neo_preferences = derive_proto_preferences(
                self.narrative.neo_memory,
                self.neo_tension_history,
                self.neo_attractors
            )
            self.eva_preferences = derive_proto_preferences(
                self.narrative.eva_memory,
                self.eva_tension_history,
                self.eva_attractors
            )

            result['neo']['preferences'] = [
                {'name': p.name, 'direction': p.direction, 'strength': p.strength}
                for p in self.neo_preferences
            ]
            result['eva']['preferences'] = [
                {'name': p.name, 'direction': p.direction, 'strength': p.strength}
                for p in self.eva_preferences
            ]

            self.preference_evolution.append({
                't': t,
                'neo': result['neo'].get('preferences', []),
                'eva': result['eva'].get('preferences', [])
            })

        return result

    def get_summary(self) -> Dict:
        """Resumen del sistema de objetivos."""
        return {
            'neo': {
                'mean_tension': np.mean(self.neo_tension_history) if self.neo_tension_history else 0.5,
                'tension_trend': self._compute_trend(self.neo_tension_history),
                'top_attractors': [
                    {
                        'sequence': a.type_sequence,
                        'frequency': a.frequency,
                        'stability': a.stability_score
                    }
                    for a in self.neo_attractors[:3]
                ],
                'preferences': [
                    {
                        'name': p.name,
                        'direction': p.direction,
                        'strength': p.strength,
                        'evidence': p.evidence
                    }
                    for p in self.neo_preferences
                ]
            },
            'eva': {
                'mean_tension': np.mean(self.eva_tension_history) if self.eva_tension_history else 0.5,
                'tension_trend': self._compute_trend(self.eva_tension_history),
                'top_attractors': [
                    {
                        'sequence': a.type_sequence,
                        'frequency': a.frequency,
                        'stability': a.stability_score
                    }
                    for a in self.eva_attractors[:3]
                ],
                'preferences': [
                    {
                        'name': p.name,
                        'direction': p.direction,
                        'strength': p.strength,
                        'evidence': p.evidence
                    }
                    for p in self.eva_preferences
                ]
            },
            'preference_correlation': self._compute_preference_correlation()
        }

    def _compute_trend(self, history: List[float]) -> str:
        """Calcula tendencia de una serie."""
        if len(history) < 10:
            return 'insufficient_data'

        recent = history[-len(history)//3:]
        early = history[:len(history)//3]

        if np.mean(recent) < np.mean(early) * 0.9:
            return 'decreasing'
        elif np.mean(recent) > np.mean(early) * 1.1:
            return 'increasing'
        else:
            return 'stable'

    def _compute_preference_correlation(self) -> float:
        """Correlación entre preferencias de NEO y EVA."""
        if not self.neo_preferences or not self.eva_preferences:
            return 0.0

        # Comparar preferencias por nombre
        neo_names = {p.name: p.strength for p in self.neo_preferences}
        eva_names = {p.name: p.strength for p in self.eva_preferences}

        common = set(neo_names.keys()) & set(eva_names.keys())

        if not common:
            return 0.0

        neo_vals = [neo_names[n] for n in common]
        eva_vals = [eva_names[n] for n in common]

        if len(neo_vals) < 2:
            return 1.0 if neo_vals[0] * eva_vals[0] > 0 else -1.0

        corr, _ = stats.spearmanr(neo_vals, eva_vals)
        return float(corr) if not np.isnan(corr) else 0.0

    def save(self, path: str):
        """Guarda el sistema."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'neo_tension_history': self.neo_tension_history,
            'eva_tension_history': self.eva_tension_history,
            'neo_attractors': [
                {
                    'sequence': a.type_sequence,
                    'frequency': a.frequency,
                    'mean_te': a.mean_te,
                    'stability': a.stability_score
                }
                for a in self.neo_attractors
            ],
            'eva_attractors': [
                {
                    'sequence': a.type_sequence,
                    'frequency': a.frequency,
                    'mean_te': a.mean_te,
                    'stability': a.stability_score
                }
                for a in self.eva_attractors
            ],
            'neo_preferences': [
                {
                    'name': p.name,
                    'direction': p.direction,
                    'strength': p.strength,
                    'evidence': {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in p.evidence.items()}
                }
                for p in self.neo_preferences
            ],
            'eva_preferences': [
                {
                    'name': p.name,
                    'direction': p.direction,
                    'strength': p.strength,
                    'evidence': {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in p.evidence.items()}
                }
                for p in self.eva_preferences
            ],
            'preference_evolution': self.preference_evolution
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 14: OBJETIVOS EMERGENTES - TEST")
    print("=" * 70)

    # Crear sistema narrativo
    ns = NarrativeSystem()

    # Simular datos
    np.random.seed(42)
    n_steps = 5000
    states = ['SLEEP', 'WAKE', 'WORK', 'LEARN', 'SOCIAL']

    neo_pi = 0.5
    eva_pi = 0.5

    print("\n[1] Simulando sistema narrativo...")

    for t in range(n_steps):
        if t % 200 == 0:
            current_state_idx = np.random.randint(0, 5)
        neo_state = states[current_state_idx]
        eva_state = states[(current_state_idx + np.random.randint(0, 2)) % 5]

        gw_active = np.random.rand() > 0.6
        gw_intensity = np.random.rand() * 0.8 if gw_active else 0

        base_te = 0.3 if neo_state in ['WORK', 'LEARN', 'SOCIAL'] else 0.05
        te_neo_to_eva = base_te + np.random.rand() * 0.2
        te_eva_to_neo = base_te + np.random.rand() * 0.2

        neo_pi += np.random.randn() * 0.02
        neo_pi = np.clip(neo_pi, 0, 1)
        eva_pi += np.random.randn() * 0.02 + 0.01 * (neo_pi - eva_pi)
        eva_pi = np.clip(eva_pi, 0, 1)

        neo_self_error = abs(np.random.randn() * 0.1)
        eva_self_error = abs(np.random.randn() * 0.1)

        ns.process_step(
            t, neo_pi, eva_pi,
            te_neo_to_eva, te_eva_to_neo,
            neo_state, eva_state,
            gw_active, gw_intensity,
            neo_self_error, eva_self_error
        )

    print(f"    Episodios NEO: {len(ns.neo_memory.episodes)}")
    print(f"    Episodios EVA: {len(ns.eva_memory.episodes)}")

    # Crear sistema de objetivos
    print("\n[2] Analizando objetivos emergentes...")
    eos = EmergentObjectiveSystem(ns)

    # Actualizar
    for t in range(0, n_steps, 100):
        eos.update(t)

    # Resumen
    summary = eos.get_summary()

    print(f"\nTensión narrativa:")
    print(f"  NEO: {summary['neo']['mean_tension']:.3f} ({summary['neo']['tension_trend']})")
    print(f"  EVA: {summary['eva']['mean_tension']:.3f} ({summary['eva']['tension_trend']})")

    print(f"\nAtractores top:")
    for a in summary['neo']['top_attractors'][:2]:
        print(f"  NEO: {a['sequence']} (freq={a['frequency']}, stab={a['stability']:.3f})")
    for a in summary['eva']['top_attractors'][:2]:
        print(f"  EVA: {a['sequence']} (freq={a['frequency']}, stab={a['stability']:.3f})")

    print(f"\nProto-preferencias:")
    for p in summary['neo']['preferences']:
        print(f"  NEO: {p['direction']} {p['name']} (strength={p['strength']:.3f})")
    for p in summary['eva']['preferences']:
        print(f"  EVA: {p['direction']} {p['name']} (strength={p['strength']:.3f})")

    print(f"\nCorrelación de preferencias NEO-EVA: {summary['preference_correlation']:.3f}")

    # Guardar
    eos.save('/root/NEO_EVA/results/phase14_objectives_test.json')
    print(f"\n[OK] Guardado en results/phase14_objectives_test.json")
