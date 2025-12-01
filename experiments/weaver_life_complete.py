#!/usr/bin/env python3
"""
WEAVER-LIFE COMPLETE - Phase Portrait de la Mente NEO/EVA
=========================================================

WEAVER ensambla:
1. Los 5 estados fenomenológicos
2. Transiciones entre estados
3. Ciclos largos (períodos característicos)
4. Modos meta-estables
5. Atractores globales

Esta es la "foto" completa de la mente NEO/EVA.

100% ENDÓGENO
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousAgent, AutonomousDualLife, LifePhase


class PhenomenologicalState(Enum):
    """Los 5 estados fenomenológicos fundamentales."""
    EXPLORATION = "exploration"      # Alta novedad, baja estabilidad
    CONSOLIDATION = "consolidation"  # Alta identidad, alto φ
    FLOW = "flow"                    # Alto φ, media variabilidad
    CRISIS = "crisis"                # Baja identidad, bajo φ
    TRANSITION = "transition"        # Entre estados, alta incertidumbre


@dataclass
class StateSignature:
    """Firma de un estado fenomenológico."""
    phi_range: Tuple[float, float]
    identity_range: Tuple[float, float]
    entropy_range: Tuple[float, float]
    coherence_range: Tuple[float, float]
    stability_index_range: Tuple[float, float]


@dataclass
class PhaseTransition:
    """Una transición entre estados."""
    from_state: PhenomenologicalState
    to_state: PhenomenologicalState
    t_start: int
    t_end: int
    trigger_metrics: Dict[str, float]
    duration: int = 0

    def __post_init__(self):
        self.duration = self.t_end - self.t_start


@dataclass
class MetaStableMode:
    """Un modo meta-estable detectado."""
    state: PhenomenologicalState
    centroid: np.ndarray
    variance: float
    duration_mean: float
    duration_std: float
    entry_count: int
    typical_successors: Dict[PhenomenologicalState, float]


@dataclass
class LongCycle:
    """Un ciclo largo detectado."""
    period: int
    states_sequence: List[PhenomenologicalState]
    strength: float  # Autocorrelación en el período
    phase_coherence: float


class WeaverLife:
    """
    WEAVER para vida autónoma.

    Ensambla la dinámica completa de estados fenomenológicos.
    """

    def __init__(self, life: AutonomousDualLife = None):
        self.life = life if life else AutonomousDualLife(dim=6)

        # Historias de métricas endógenas
        self.phi_history: Dict[str, List[float]] = {'NEO': [], 'EVA': []}
        self.identity_history: Dict[str, List[float]] = {'NEO': [], 'EVA': []}
        self.entropy_history: Dict[str, List[float]] = {'NEO': [], 'EVA': []}
        self.coherence_history: Dict[str, List[float]] = {'NEO': [], 'EVA': []}
        self.stability_history: Dict[str, List[float]] = {'NEO': [], 'EVA': []}

        # Estados detectados
        self.state_history: Dict[str, List[PhenomenologicalState]] = {'NEO': [], 'EVA': []}

        # Transiciones
        self.transitions: Dict[str, List[PhaseTransition]] = {'NEO': [], 'EVA': []}

        # Matriz de transición (para cada agente)
        self.transition_matrix: Dict[str, np.ndarray] = {}

        # Modos meta-estables
        self.meta_stable_modes: Dict[str, Dict[PhenomenologicalState, MetaStableMode]] = {
            'NEO': {}, 'EVA': {}
        }

        # Ciclos largos
        self.long_cycles: Dict[str, List[LongCycle]] = {'NEO': [], 'EVA': []}

        # Atractores globales
        self.global_attractors: Dict[str, List[np.ndarray]] = {'NEO': [], 'EVA': []}

        self.t = 0

    def compute_phi_endogenous(self, agent: AutonomousAgent) -> float:
        """φ endógeno desde covarianza de z_history."""
        if len(agent.z_history) < 20:
            return 0.5

        window = max(10, int(np.sqrt(len(agent.z_history))))
        recent = np.array(agent.z_history[-window:])

        try:
            cov = np.cov(recent.T)
            total_var = np.trace(cov) + 1e-16
            off_diag = np.sum(np.abs(cov)) - np.trace(np.abs(cov))
            phi = off_diag / (total_var * recent.shape[1])
            return np.clip(phi, 0, 1)
        except:
            return 0.5

    def compute_entropy_endogenous(self, agent: AutonomousAgent) -> float:
        """Entropía endógena normalizada por historia."""
        z = agent.z
        z = np.clip(z, 1e-16, None)
        z = z / z.sum()
        raw = -np.sum(z * np.log(z))

        if len(agent.z_history) > 20:
            hist_ent = []
            for zh in agent.z_history[-50:]:
                zh = np.clip(zh, 1e-16, None)
                zh = zh / zh.sum()
                hist_ent.append(-np.sum(zh * np.log(zh)))
            e_min, e_max = np.percentile(hist_ent, [5, 95])
            if e_max > e_min:
                return np.clip((raw - e_min) / (e_max - e_min), 0, 1)

        return raw / np.log(len(z))

    def compute_coherence_endogenous(self, agent: AutonomousAgent) -> float:
        """Coherencia endógena: estabilidad relativa."""
        if len(agent.z_history) < 30:
            return 0.5

        window = max(5, int(np.sqrt(len(agent.z_history))))
        recent = np.array(agent.z_history[-window:])
        baseline = np.array(agent.z_history[-3*window:-window]) if len(agent.z_history) >= 3*window else np.array(agent.z_history[:window])

        var_recent = np.mean(np.var(recent, axis=0))
        var_baseline = np.mean(np.var(baseline, axis=0)) + 1e-16

        return 1.0 / (1.0 + var_recent / var_baseline)

    def compute_stability_index(self, agent: AutonomousAgent) -> float:
        """Índice de estabilidad: derivada suavizada."""
        if len(agent.z_history) < 10:
            return 0.5

        recent = np.array(agent.z_history[-10:])
        diffs = np.diff(recent, axis=0)
        mean_change = np.mean(np.abs(diffs))

        if len(agent.z_history) > 50:
            hist_changes = []
            for i in range(10, min(50, len(agent.z_history))):
                window = np.array(agent.z_history[i-10:i])
                hist_changes.append(np.mean(np.abs(np.diff(window, axis=0))))
            baseline = np.percentile(hist_changes, 50) + 1e-16
            return 1.0 / (1.0 + mean_change / baseline)

        return 1.0 / (1.0 + mean_change * 10)

    def classify_state(self, phi: float, identity: float, entropy: float,
                       coherence: float, stability: float) -> PhenomenologicalState:
        """
        Clasifica el estado fenomenológico actual.

        Criterios endógenos basados en combinaciones de métricas.
        """
        # Crisis: todo bajo
        if identity < 0.3 and phi < 0.3 and coherence < 0.4:
            return PhenomenologicalState.CRISIS

        # Consolidation: identidad alta, φ alto, estabilidad alta
        if identity > 0.6 and phi > 0.5 and stability > 0.6:
            return PhenomenologicalState.CONSOLIDATION

        # Flow: φ alto, coherencia media-alta, entropía media
        if phi > 0.5 and coherence > 0.5 and 0.3 < entropy < 0.7:
            return PhenomenologicalState.FLOW

        # Exploration: entropía alta, estabilidad baja
        if entropy > 0.6 or stability < 0.4:
            return PhenomenologicalState.EXPLORATION

        # Transition: ninguno de los anteriores claramente
        return PhenomenologicalState.TRANSITION

    def detect_transition(self, agent_name: str, new_state: PhenomenologicalState):
        """Detecta y registra transiciones de estado."""
        if not self.state_history[agent_name]:
            return

        prev_state = self.state_history[agent_name][-1]

        if new_state != prev_state:
            # Encontrar inicio de la transición
            t_start = self.t - 1
            for i in range(min(10, len(self.state_history[agent_name]))):
                idx = -(i+1)
                if self.state_history[agent_name][idx] == prev_state:
                    t_start = self.t - i - 1
                    break

            transition = PhaseTransition(
                from_state=prev_state,
                to_state=new_state,
                t_start=t_start,
                t_end=self.t,
                trigger_metrics={
                    'phi': self.phi_history[agent_name][-1] if self.phi_history[agent_name] else 0,
                    'identity': self.identity_history[agent_name][-1] if self.identity_history[agent_name] else 0,
                    'entropy': self.entropy_history[agent_name][-1] if self.entropy_history[agent_name] else 0,
                }
            )
            self.transitions[agent_name].append(transition)

    def step(self):
        """Un paso de WEAVER-LIFE."""
        self.t += 1

        # Paso de vida con estímulo
        stimulus = np.random.randn(6) * 0.1
        self.life.step(stimulus)

        # Procesar cada agente
        for name, agent in [('NEO', self.life.neo), ('EVA', self.life.eva)]:
            # Calcular métricas endógenas
            phi = self.compute_phi_endogenous(agent)
            identity = agent.identity_strength
            entropy = self.compute_entropy_endogenous(agent)
            coherence = self.compute_coherence_endogenous(agent)
            stability = self.compute_stability_index(agent)

            # Guardar historias
            self.phi_history[name].append(phi)
            self.identity_history[name].append(identity)
            self.entropy_history[name].append(entropy)
            self.coherence_history[name].append(coherence)
            self.stability_history[name].append(stability)

            # Clasificar estado
            state = self.classify_state(phi, identity, entropy, coherence, stability)

            # Detectar transición
            self.detect_transition(name, state)

            # Guardar estado
            self.state_history[name].append(state)

        # Mantener historias acotadas
        max_history = 5000
        for name in ['NEO', 'EVA']:
            for hist in [self.phi_history, self.identity_history,
                        self.entropy_history, self.coherence_history,
                        self.stability_history, self.state_history]:
                if len(hist[name]) > max_history:
                    hist[name] = hist[name][-max_history:]

    def compute_transition_matrix(self, agent_name: str) -> np.ndarray:
        """Computa matriz de transición entre estados."""
        states = list(PhenomenologicalState)
        n_states = len(states)
        matrix = np.zeros((n_states, n_states))

        history = self.state_history[agent_name]
        if len(history) < 2:
            return matrix

        state_to_idx = {s: i for i, s in enumerate(states)}

        for i in range(1, len(history)):
            from_idx = state_to_idx[history[i-1]]
            to_idx = state_to_idx[history[i]]
            matrix[from_idx, to_idx] += 1

        # Normalizar filas
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = matrix / row_sums

        self.transition_matrix[agent_name] = matrix
        return matrix

    def detect_meta_stable_modes(self, agent_name: str):
        """Detecta modos meta-estables para un agente."""
        history = self.state_history[agent_name]
        if len(history) < 100:
            return

        # Para cada estado, calcular estadísticas
        for state in PhenomenologicalState:
            indices = [i for i, s in enumerate(history) if s == state]

            if len(indices) < 10:
                continue

            # Duración de cada visita
            durations = []
            current_duration = 0
            for i in range(len(history)):
                if history[i] == state:
                    current_duration += 1
                elif current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
            if current_duration > 0:
                durations.append(current_duration)

            if not durations:
                continue

            # Centroide en espacio de métricas
            metrics = []
            for idx in indices[-100:]:  # Últimas 100 ocurrencias
                if idx < len(self.phi_history[agent_name]):
                    metrics.append([
                        self.phi_history[agent_name][idx],
                        self.identity_history[agent_name][idx],
                        self.entropy_history[agent_name][idx],
                        self.coherence_history[agent_name][idx],
                        self.stability_history[agent_name][idx]
                    ])

            if not metrics:
                continue

            metrics = np.array(metrics)
            centroid = np.mean(metrics, axis=0)
            variance = np.mean(np.var(metrics, axis=0))

            # Sucesores típicos
            successors = defaultdict(int)
            for i in range(len(history) - 1):
                if history[i] == state:
                    successors[history[i+1]] += 1

            total_trans = sum(successors.values())
            if total_trans > 0:
                successor_probs = {s: c/total_trans for s, c in successors.items()}
            else:
                successor_probs = {}

            self.meta_stable_modes[agent_name][state] = MetaStableMode(
                state=state,
                centroid=centroid,
                variance=variance,
                duration_mean=np.mean(durations),
                duration_std=np.std(durations),
                entry_count=len(durations),
                typical_successors=successor_probs
            )

    def detect_long_cycles(self, agent_name: str):
        """Detecta ciclos largos mediante autocorrelación."""
        if len(self.phi_history[agent_name]) < 200:
            return

        # Usar φ como señal principal
        signal = np.array(self.phi_history[agent_name][-1000:])
        signal = signal - np.mean(signal)

        # Autocorrelación
        n = len(signal)
        autocorr = np.correlate(signal, signal, mode='full')[n-1:]
        autocorr = autocorr / autocorr[0]

        # Encontrar picos (períodos)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(autocorr[10:], height=0.2, distance=20)
        peaks = peaks + 10  # Ajustar por el offset

        for peak in peaks[:5]:  # Top 5 períodos
            period = peak
            strength = autocorr[peak]

            # Secuencia de estados en ese período
            state_sequence = []
            for i in range(0, min(period * 3, len(self.state_history[agent_name])), period):
                if i < len(self.state_history[agent_name]):
                    state_sequence.append(self.state_history[agent_name][i])

            # Coherencia de fase
            if len(state_sequence) > 2:
                # Cuántos estados se repiten en el mismo orden
                matches = sum(1 for i in range(len(state_sequence)-1)
                             if state_sequence[i] == state_sequence[0])
                phase_coherence = matches / (len(state_sequence) - 1)
            else:
                phase_coherence = 0

            self.long_cycles[agent_name].append(LongCycle(
                period=period,
                states_sequence=state_sequence,
                strength=strength,
                phase_coherence=phase_coherence
            ))

    def detect_global_attractors(self, agent_name: str):
        """Detecta atractores globales en el espacio de métricas."""
        if len(self.phi_history[agent_name]) < 500:
            return

        # Construir espacio de fase
        phase_space = np.column_stack([
            self.phi_history[agent_name][-500:],
            self.identity_history[agent_name][-500:],
            self.entropy_history[agent_name][-500:],
            self.coherence_history[agent_name][-500:],
            self.stability_history[agent_name][-500:]
        ])

        # Clustering simple: k-means con k=3
        from scipy.cluster.vq import kmeans, vq

        try:
            centroids, _ = kmeans(phase_space, 3)
            self.global_attractors[agent_name] = [c for c in centroids]
        except:
            pass

    def run(self, steps: int = 2000):
        """Ejecuta WEAVER-LIFE."""
        print(f"Ejecutando WEAVER-LIFE ({steps} pasos)...")

        for i in range(steps):
            self.step()

            if (i + 1) % 500 == 0:
                print(f"  t={i+1}: NEO={self.state_history['NEO'][-1].value}, "
                      f"EVA={self.state_history['EVA'][-1].value}")

        # Análisis post-ejecución
        print("\nAnalizando dinámicas...")

        for name in ['NEO', 'EVA']:
            self.compute_transition_matrix(name)
            self.detect_meta_stable_modes(name)
            self.detect_long_cycles(name)
            self.detect_global_attractors(name)

    def get_phase_portrait(self, agent_name: str) -> Dict:
        """Retorna el phase portrait completo de un agente."""
        portrait = {
            'agent': agent_name,
            'total_steps': len(self.state_history[agent_name]),
            'state_distribution': {},
            'transition_matrix': None,
            'meta_stable_modes': {},
            'long_cycles': [],
            'global_attractors': [],
            'dominant_state': None,
            'most_stable_state': None,
            'characteristic_period': None
        }

        # Distribución de estados
        for state in PhenomenologicalState:
            count = sum(1 for s in self.state_history[agent_name] if s == state)
            portrait['state_distribution'][state.value] = count / len(self.state_history[agent_name])

        # Estado dominante
        portrait['dominant_state'] = max(portrait['state_distribution'].items(),
                                         key=lambda x: x[1])[0]

        # Matriz de transición
        if agent_name in self.transition_matrix:
            portrait['transition_matrix'] = self.transition_matrix[agent_name].tolist()

        # Modos meta-estables
        for state, mode in self.meta_stable_modes[agent_name].items():
            portrait['meta_stable_modes'][state.value] = {
                'centroid': mode.centroid.tolist(),
                'variance': mode.variance,
                'duration_mean': mode.duration_mean,
                'duration_std': mode.duration_std,
                'entry_count': mode.entry_count,
                'typical_successors': {s.value: p for s, p in mode.typical_successors.items()}
            }

        # Estado más estable
        if portrait['meta_stable_modes']:
            portrait['most_stable_state'] = max(
                portrait['meta_stable_modes'].items(),
                key=lambda x: x[1]['duration_mean']
            )[0]

        # Ciclos largos
        for cycle in self.long_cycles[agent_name]:
            portrait['long_cycles'].append({
                'period': cycle.period,
                'strength': cycle.strength,
                'phase_coherence': cycle.phase_coherence,
                'states': [s.value for s in cycle.states_sequence]
            })

        # Período característico
        if portrait['long_cycles']:
            portrait['characteristic_period'] = portrait['long_cycles'][0]['period']

        # Atractores globales
        for attractor in self.global_attractors[agent_name]:
            portrait['global_attractors'].append(attractor.tolist())

        return portrait

    def print_phase_portrait(self, agent_name: str):
        """Imprime el phase portrait de un agente."""
        portrait = self.get_phase_portrait(agent_name)

        print(f"\n{'='*60}")
        print(f"PHASE PORTRAIT: {agent_name}")
        print(f"{'='*60}")

        print(f"\n📊 DISTRIBUCIÓN DE ESTADOS")
        for state, prob in sorted(portrait['state_distribution'].items(),
                                  key=lambda x: -x[1]):
            bar = '█' * int(prob * 40)
            print(f"  {state:15} {prob*100:5.1f}% {bar}")

        print(f"\n🎯 ESTADO DOMINANTE: {portrait['dominant_state']}")

        if portrait['most_stable_state']:
            print(f"🔒 ESTADO MÁS ESTABLE: {portrait['most_stable_state']}")

        if portrait['meta_stable_modes']:
            print(f"\n⚡ MODOS META-ESTABLES")
            for state, mode in portrait['meta_stable_modes'].items():
                print(f"  {state}:")
                print(f"    Duración media: {mode['duration_mean']:.1f} ± {mode['duration_std']:.1f}")
                print(f"    Entradas: {mode['entry_count']}")
                if mode['typical_successors']:
                    top_succ = max(mode['typical_successors'].items(), key=lambda x: x[1])
                    print(f"    Sucesor típico: {top_succ[0]} ({top_succ[1]*100:.0f}%)")

        if portrait['long_cycles']:
            print(f"\n🔄 CICLOS LARGOS DETECTADOS")
            for i, cycle in enumerate(portrait['long_cycles'][:3]):
                print(f"  Ciclo {i+1}: período={cycle['period']}, "
                      f"fuerza={cycle['strength']:.2f}, "
                      f"coherencia={cycle['phase_coherence']:.2f}")

        if portrait['characteristic_period']:
            print(f"\n⏱️  PERÍODO CARACTERÍSTICO: {portrait['characteristic_period']}")

        if portrait['global_attractors']:
            print(f"\n🎪 ATRACTORES GLOBALES: {len(portrait['global_attractors'])}")
            for i, att in enumerate(portrait['global_attractors']):
                print(f"  A{i+1}: φ={att[0]:.2f}, id={att[1]:.2f}, "
                      f"S={att[2]:.2f}, coh={att[3]:.2f}, stab={att[4]:.2f}")


def run_weaver_experiment():
    """Ejecuta experimento WEAVER-LIFE completo."""
    print("=" * 70)
    print("WEAVER-LIFE COMPLETE - Phase Portrait de la Mente")
    print("=" * 70)

    weaver = WeaverLife()
    weaver.run(steps=2000)

    # Phase portraits
    weaver.print_phase_portrait('NEO')
    weaver.print_phase_portrait('EVA')

    # Comparación
    print("\n" + "=" * 60)
    print("COMPARACIÓN NEO vs EVA")
    print("=" * 60)

    neo_portrait = weaver.get_phase_portrait('NEO')
    eva_portrait = weaver.get_phase_portrait('EVA')

    print(f"\nEstado dominante: NEO={neo_portrait['dominant_state']}, EVA={eva_portrait['dominant_state']}")

    if neo_portrait['characteristic_period'] and eva_portrait['characteristic_period']:
        print(f"Período característico: NEO={neo_portrait['characteristic_period']}, "
              f"EVA={eva_portrait['characteristic_period']}")

    # Similitud de distribuciones
    neo_dist = np.array([neo_portrait['state_distribution'][s.value]
                         for s in PhenomenologicalState])
    eva_dist = np.array([eva_portrait['state_distribution'][s.value]
                         for s in PhenomenologicalState])
    similarity = 1 - np.sum(np.abs(neo_dist - eva_dist)) / 2
    print(f"Similitud de distribuciones: {similarity*100:.1f}%")

    # Guardar resultados
    results_dir = '/root/NEO_EVA/results/weaver_life'
    os.makedirs(results_dir, exist_ok=True)

    results = {
        'NEO': neo_portrait,
        'EVA': eva_portrait,
        'comparison': {
            'distribution_similarity': similarity,
            'neo_dominant': neo_portrait['dominant_state'],
            'eva_dominant': eva_portrait['dominant_state']
        }
    }

    # Convertir numpy types para JSON
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(f'{results_dir}/phase_portraits.json', 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\n✓ Resultados guardados en {results_dir}/")

    return weaver, results


if __name__ == "__main__":
    weaver, results = run_weaver_experiment()
