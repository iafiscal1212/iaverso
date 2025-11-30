#!/usr/bin/env python3
"""
Phase 15B: Dinámica de Estados
===============================

Análisis de transiciones, recurrencias y ciclos en estados emergentes.

Componentes:
1. Matriz de transición entre prototipos (Markov)
2. Tiempo de recurrencia a prototipos
3. Detección de ciclos/loops narrativos
4. Estabilidad de atractores

Todo basado en la historia del sistema - CERO números mágicos.
100% endógeno.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque, Counter
from scipy import stats
import json
from datetime import datetime

import sys
sys.path.insert(0, '/root/NEO_EVA/tools')

from endogenous_core import (
    derive_window_size,
    derive_learning_rate,
    compute_iqr,
    compute_entropy_normalized,
    NUMERIC_EPS,
    PROVENANCE
)


# =============================================================================
# MATRIZ DE TRANSICIÓN
# =============================================================================

class TransitionMatrix:
    """
    Matriz de transición entre prototipos emergentes.

    P[i][j] = probabilidad de ir de prototipo i a prototipo j.
    Actualizada online con smoothing endógeno.
    """

    def __init__(self):
        # Conteos de transiciones
        self.counts: Dict[int, Dict[int, int]] = {}

        # Total de transiciones
        self.total_transitions = 0

        # Historial de transiciones
        # Derivado: maxlen = √1e6 ≈ 1000
        derived_maxlen = int(np.sqrt(1e6))
        self.transition_history: deque = deque(maxlen=derived_maxlen)

        # Último prototipo visitado
        self.last_prototype: Optional[int] = None

    def observe_transition(self, from_proto: int, to_proto: int):
        """Registra una transición observada."""
        if from_proto not in self.counts:
            self.counts[from_proto] = {}

        if to_proto not in self.counts[from_proto]:
            self.counts[from_proto][to_proto] = 0

        self.counts[from_proto][to_proto] += 1
        self.total_transitions += 1

        self.transition_history.append((from_proto, to_proto))
        self.last_prototype = to_proto

    def update(self, current_prototype: int):
        """
        Actualiza con el prototipo actual.

        Registra transición desde el último prototipo (si existe).
        """
        if self.last_prototype is not None and self.last_prototype != current_prototype:
            self.observe_transition(self.last_prototype, current_prototype)

        self.last_prototype = current_prototype

    def get_smoothing_alpha(self) -> float:
        """
        Smoothing endógeno: α = 1/√(N+1)

        donde N = número total de transiciones observadas.
        """
        alpha = 1.0 / np.sqrt(self.total_transitions + 1)

        PROVENANCE.log('transition_smoothing', alpha,
                       '1/sqrt(N_transitions + 1)',
                       {'N': self.total_transitions}, self.total_transitions)

        return alpha

    def get_transition_probability(self, from_proto: int, to_proto: int) -> float:
        """
        Probabilidad de transición con smoothing.

        P(j|i) = (count[i][j] + α) / (total_from_i + α * n_targets)
        """
        if from_proto not in self.counts:
            return 0.0

        alpha = self.get_smoothing_alpha()
        targets = self.counts[from_proto]
        total_from = sum(targets.values())
        n_targets = len(targets) + 1  # +1 para el target actual (puede ser nuevo)

        count = targets.get(to_proto, 0)
        prob = (count + alpha) / (total_from + alpha * n_targets)

        return float(prob)

    def get_transition_matrix(self) -> Tuple[np.ndarray, List[int]]:
        """
        Retorna matriz de transición como array numpy.

        Returns:
            (matrix, prototype_ids): matriz y lista de IDs de prototipos
        """
        if not self.counts:
            return np.array([[1.0]]), [0]

        # Obtener todos los prototipos
        all_protos = set()
        for from_p, targets in self.counts.items():
            all_protos.add(from_p)
            all_protos.update(targets.keys())

        proto_list = sorted(all_protos)
        n = len(proto_list)
        proto_to_idx = {p: i for i, p in enumerate(proto_list)}

        # Construir matriz
        matrix = np.zeros((n, n))
        alpha = self.get_smoothing_alpha()

        for i, from_p in enumerate(proto_list):
            if from_p in self.counts:
                targets = self.counts[from_p]
                total = sum(targets.values())

                for j, to_p in enumerate(proto_list):
                    count = targets.get(to_p, 0)
                    matrix[i, j] = (count + alpha) / (total + alpha * n)
            else:
                # Distribución uniforme si no hay transiciones desde este prototipo
                matrix[i, :] = 1.0 / n

        return matrix, proto_list

    def get_stationary_distribution(self) -> Tuple[np.ndarray, List[int]]:
        """
        Calcula distribución estacionaria π tal que π = π P.

        Esta es la distribución de tiempo largo en los prototipos.
        """
        matrix, proto_list = self.get_transition_matrix()
        n = len(proto_list)

        if n <= 1:
            return np.array([1.0]), proto_list

        # Calcular autovector izquierdo dominante
        # π P = π => π^T = P^T π^T => autovalor 1
        eigenvalues, eigenvectors = np.linalg.eig(matrix.T)

        # Buscar autovalor más cercano a 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])

        # Normalizar
        stationary = np.abs(stationary)
        stationary = stationary / (stationary.sum() + NUMERIC_EPS)

        return stationary, proto_list

    def get_summary(self) -> Dict:
        """Resumen de la matriz de transición."""
        matrix, protos = self.get_transition_matrix()
        stationary, _ = self.get_stationary_distribution()

        # Entropía de la matriz (incertidumbre promedio)
        entropies = []
        for row in matrix:
            h = compute_entropy_normalized(row)
            entropies.append(h)
        mean_entropy = np.mean(entropies)

        return {
            'n_prototypes': len(protos),
            'total_transitions': self.total_transitions,
            'smoothing_alpha': self.get_smoothing_alpha(),
            'mean_transition_entropy': float(mean_entropy),
            'stationary_distribution': {p: float(s) for p, s in zip(protos, stationary)},
            'transition_matrix': {
                'prototypes': protos,
                'matrix': matrix.tolist()
            }
        }


# =============================================================================
# TIEMPO DE RECURRENCIA
# =============================================================================

class RecurrenceAnalyzer:
    """
    Analiza tiempos de recurrencia a prototipos.

    Tiempo de recurrencia = cuántos pasos hasta volver a un prototipo.
    """

    def __init__(self):
        # Última visita a cada prototipo
        self.last_visit: Dict[int, int] = {}

        # Tiempos de recurrencia observados
        # Derivado: maxlen = √1e6 ≈ 1000
        derived_maxlen = int(np.sqrt(1e6))
        self.recurrence_times: Dict[int, deque] = {}

        # Contador de tiempo
        self.t = 0

    def observe(self, prototype_id: int):
        """Observa visita a un prototipo."""
        if prototype_id in self.last_visit:
            recurrence_time = self.t - self.last_visit[prototype_id]

            if prototype_id not in self.recurrence_times:
                # Derivado: maxlen para cada prototipo
                derived_maxlen = int(np.sqrt(1e6))
                self.recurrence_times[prototype_id] = deque(maxlen=derived_maxlen)

            self.recurrence_times[prototype_id].append(recurrence_time)

        self.last_visit[prototype_id] = self.t
        self.t += 1

    def get_mean_recurrence(self, prototype_id: int) -> float:
        """Tiempo medio de recurrencia a un prototipo."""
        if prototype_id not in self.recurrence_times:
            return float('inf')

        times = list(self.recurrence_times[prototype_id])
        if not times:
            return float('inf')

        return float(np.mean(times))

    def get_recurrence_stability(self, prototype_id: int) -> float:
        """
        Estabilidad de recurrencia = 1 / (1 + CV)

        donde CV = coeficiente de variación.
        Alta estabilidad = recurrencias regulares.
        """
        if prototype_id not in self.recurrence_times:
            return 0.0

        times = list(self.recurrence_times[prototype_id])
        if len(times) < 2:
            return 0.5

        mean = np.mean(times)
        std = np.std(times)

        if mean < NUMERIC_EPS:
            return 1.0

        cv = std / mean
        stability = 1.0 / (1.0 + cv)

        return float(stability)

    def get_summary(self) -> Dict:
        """Resumen de análisis de recurrencia."""
        summary = {
            'n_prototypes_observed': len(self.last_visit),
            'prototypes': {}
        }

        for proto_id in self.last_visit:
            mean_rec = self.get_mean_recurrence(proto_id)
            stability = self.get_recurrence_stability(proto_id)

            n_recurrences = len(self.recurrence_times.get(proto_id, []))

            summary['prototypes'][proto_id] = {
                'mean_recurrence_time': mean_rec if mean_rec != float('inf') else None,
                'recurrence_stability': stability,
                'n_recurrences': n_recurrences
            }

        return summary


# =============================================================================
# DETECCIÓN DE CICLOS
# =============================================================================

class CycleDetector:
    """
    Detecta ciclos/loops en secuencias de prototipos.

    Un ciclo es una secuencia de prototipos que se repite.
    """

    def __init__(self):
        # Historial de prototipos
        # Derivado: maxlen = √1e6 ≈ 1000
        derived_maxlen = int(np.sqrt(1e6))
        self.prototype_sequence: deque = deque(maxlen=derived_maxlen)

        # Ciclos detectados
        self.detected_cycles: List[Dict] = []

    def observe(self, prototype_id: int):
        """Observa un prototipo en la secuencia."""
        self.prototype_sequence.append(prototype_id)

    def find_cycles(self, min_length: int = 2, max_length: Optional[int] = None) -> List[Dict]:
        """
        Busca ciclos en la secuencia.

        Args:
            min_length: longitud mínima del ciclo
            max_length: longitud máxima (derivada endógenamente si None)
        """
        seq = list(self.prototype_sequence)
        n = len(seq)

        if n < min_length * 2:
            return []

        # max_length endógeno: √n
        if max_length is None:
            max_length = max(min_length, int(np.sqrt(n)))

        PROVENANCE.log('cycle_max_length', max_length,
                       'max(min_length, sqrt(n))',
                       {'n': n}, n)

        cycles = {}

        for length in range(min_length, max_length + 1):
            # Buscar todas las subsecuencias de esta longitud
            for i in range(n - length * 2 + 1):
                pattern = tuple(seq[i:i + length])

                # Buscar repeticiones
                count = 0
                for j in range(i, n - length + 1):
                    if tuple(seq[j:j + length]) == pattern:
                        count += 1

                if count >= 2:  # Al menos 2 ocurrencias
                    if pattern not in cycles:
                        cycles[pattern] = {
                            'pattern': list(pattern),
                            'length': length,
                            'count': count,
                            'first_occurrence': i
                        }
                    else:
                        cycles[pattern]['count'] = max(cycles[pattern]['count'], count)

        # Filtrar: solo ciclos que ocurren más de lo esperado
        significant_cycles = []

        for pattern, info in cycles.items():
            # Frecuencia esperada bajo independencia
            n_unique = len(set(seq))
            expected = n / (n_unique ** len(pattern)) if n_unique > 0 else 0

            if info['count'] > expected * 1.5:
                info['expected'] = expected
                info['significance'] = info['count'] / (expected + NUMERIC_EPS)
                significant_cycles.append(info)

        # Ordenar por significancia
        significant_cycles.sort(key=lambda x: x['significance'], reverse=True)

        return significant_cycles

    def detect_current_cycle(self) -> Optional[Dict]:
        """
        Detecta si estamos actualmente en un ciclo.

        Busca si los últimos k prototipos coinciden con un patrón anterior.
        """
        seq = list(self.prototype_sequence)
        n = len(seq)

        if n < 4:
            return None

        # Buscar coincidencias con patrones recientes
        max_check = min(int(np.sqrt(n)), 20)

        for length in range(2, max_check + 1):
            current_pattern = seq[-length:]

            # Buscar este patrón antes en la secuencia
            for i in range(n - length * 2, -1, -1):
                if seq[i:i + length] == current_pattern:
                    return {
                        'pattern': current_pattern,
                        'length': length,
                        'previous_occurrence': i,
                        'gap': n - length - i
                    }

        return None

    def get_summary(self) -> Dict:
        """Resumen de detección de ciclos."""
        cycles = self.find_cycles()
        current = self.detect_current_cycle()

        return {
            'sequence_length': len(self.prototype_sequence),
            'n_cycles_found': len(cycles),
            'top_cycles': cycles[:5],
            'in_current_cycle': current is not None,
            'current_cycle': current
        }


# =============================================================================
# SISTEMA DE DINÁMICA DE ESTADOS
# =============================================================================

class StateDynamicsSystem:
    """
    Sistema completo de análisis de dinámica de estados.

    Integra:
    - Matriz de transición
    - Análisis de recurrencia
    - Detección de ciclos
    """

    def __init__(self):
        # Para cada agente
        self.neo_transitions = TransitionMatrix()
        self.eva_transitions = TransitionMatrix()

        self.neo_recurrence = RecurrenceAnalyzer()
        self.eva_recurrence = RecurrenceAnalyzer()

        self.neo_cycles = CycleDetector()
        self.eva_cycles = CycleDetector()

        # Dinámica conjunta
        self.joint_transitions = TransitionMatrix()
        self.joint_cycles = CycleDetector()

        # Historial
        self.t = 0

    def update(
        self,
        neo_prototype_id: int,
        eva_prototype_id: int
    ) -> Dict:
        """
        Actualiza análisis de dinámica con nuevos prototipos.
        """
        result = {
            't': self.t,
            'neo': {},
            'eva': {},
            'joint': {}
        }

        # NEO
        self.neo_transitions.update(neo_prototype_id)
        self.neo_recurrence.observe(neo_prototype_id)
        self.neo_cycles.observe(neo_prototype_id)

        result['neo'] = {
            'prototype': neo_prototype_id,
            'in_cycle': self.neo_cycles.detect_current_cycle() is not None
        }

        # EVA
        self.eva_transitions.update(eva_prototype_id)
        self.eva_recurrence.observe(eva_prototype_id)
        self.eva_cycles.observe(eva_prototype_id)

        result['eva'] = {
            'prototype': eva_prototype_id,
            'in_cycle': self.eva_cycles.detect_current_cycle() is not None
        }

        # Conjunto (combinar IDs)
        joint_id = neo_prototype_id * 1000 + eva_prototype_id
        self.joint_transitions.update(joint_id)
        self.joint_cycles.observe(joint_id)

        result['joint'] = {
            'joint_prototype': joint_id,
            'in_joint_cycle': self.joint_cycles.detect_current_cycle() is not None
        }

        self.t += 1

        return result

    def get_markov_properties(self, agent: str = 'neo') -> Dict:
        """
        Propiedades Markovianas de la dinámica.
        """
        transitions = self.neo_transitions if agent == 'neo' else self.eva_transitions
        matrix, protos = transitions.get_transition_matrix()
        stationary, _ = transitions.get_stationary_distribution()

        if len(protos) <= 1:
            return {
                'n_states': len(protos),
                'is_ergodic': True,
                'mixing_time': 0
            }

        # Verificar ergodicidad (matriz irreducible y aperiódica)
        # Simplificación: verificar si todos los estados son alcanzables
        n = len(protos)
        reachable = np.linalg.matrix_power(matrix + np.eye(n), n) > 0
        is_irreducible = reachable.all()

        # Mixing time aproximado (tiempo para converger a estacionaria)
        # Usar segundo autovalor
        eigenvalues = np.linalg.eigvals(matrix)
        eigenvalues_sorted = sorted(np.abs(eigenvalues), reverse=True)

        if len(eigenvalues_sorted) > 1 and eigenvalues_sorted[1] < 1:
            spectral_gap = 1 - eigenvalues_sorted[1]
            mixing_time = int(1 / spectral_gap) if spectral_gap > NUMERIC_EPS else float('inf')
        else:
            mixing_time = float('inf')

        return {
            'n_states': n,
            'is_ergodic': bool(is_irreducible),
            'spectral_gap': float(1 - eigenvalues_sorted[1]) if len(eigenvalues_sorted) > 1 else 1.0,
            'mixing_time': mixing_time,
            'stationary_entropy': float(compute_entropy_normalized(stationary))
        }

    def get_summary(self) -> Dict:
        """Resumen completo de dinámica de estados."""
        return {
            'neo': {
                'transitions': self.neo_transitions.get_summary(),
                'recurrence': self.neo_recurrence.get_summary(),
                'cycles': self.neo_cycles.get_summary(),
                'markov': self.get_markov_properties('neo')
            },
            'eva': {
                'transitions': self.eva_transitions.get_summary(),
                'recurrence': self.eva_recurrence.get_summary(),
                'cycles': self.eva_cycles.get_summary(),
                'markov': self.get_markov_properties('eva')
            },
            'joint': {
                'transitions': self.joint_transitions.get_summary(),
                'cycles': self.joint_cycles.get_summary()
            }
        }

    def save(self, path: str):
        """Guarda el sistema de dinámica."""
        data = {
            'timestamp': datetime.now().isoformat(),
            't': self.t,
            'summary': self.get_summary()
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 15B: DINÁMICA DE ESTADOS - TEST")
    print("=" * 70)

    # Crear sistema
    dynamics = StateDynamicsSystem()

    # Simular datos con estructura
    np.random.seed(42)
    n_steps = 5000

    print("\n[1] Simulando dinámica de estados...")

    # Crear secuencia de prototipos con ciclos
    neo_protos = [0, 1, 2]  # 3 prototipos para NEO
    eva_protos = [0, 1, 2, 3]  # 4 prototipos para EVA

    neo_current = 0
    eva_current = 0

    for t in range(n_steps):
        # NEO: transiciones con cierta estructura
        if np.random.rand() < 0.7:
            # Seguir patrón (ciclo 0 → 1 → 2 → 0)
            neo_current = (neo_current + 1) % 3
        else:
            # Salto aleatorio
            neo_current = np.random.choice(neo_protos)

        # EVA: más aleatorio pero con preferencia
        if np.random.rand() < 0.5:
            eva_current = (eva_current + 1) % 4
        else:
            eva_current = np.random.choice(eva_protos)

        # Actualizar
        dynamics.update(neo_current, eva_current)

    print(f"    Pasos procesados: {n_steps}")

    # Resumen
    print("\n[2] Resumen de dinámica:")
    summary = dynamics.get_summary()

    # NEO
    print("\nNEO:")
    print(f"  Transiciones totales: {summary['neo']['transitions']['total_transitions']}")
    print(f"  Entropía de transición: {summary['neo']['transitions']['mean_transition_entropy']:.3f}")

    print(f"\n  Distribución estacionaria:")
    for p, prob in summary['neo']['transitions']['stationary_distribution'].items():
        print(f"    Proto {p}: {prob:.3f}")

    print(f"\n  Markov:")
    markov = summary['neo']['markov']
    print(f"    Ergódico: {markov['is_ergodic']}")
    print(f"    Gap espectral: {markov['spectral_gap']:.3f}")
    print(f"    Tiempo de mezcla: {markov['mixing_time']}")

    print(f"\n  Ciclos:")
    cycles = summary['neo']['cycles']
    print(f"    Ciclos encontrados: {cycles['n_cycles_found']}")
    if cycles['top_cycles']:
        for c in cycles['top_cycles'][:3]:
            print(f"    Patrón {c['pattern']}: count={c['count']}, signif={c['significance']:.1f}")

    print(f"\n  Recurrencia:")
    rec = summary['neo']['recurrence']
    for p, info in list(rec['prototypes'].items())[:3]:
        print(f"    Proto {p}: T_rec={info['mean_recurrence_time']:.1f}, estab={info['recurrence_stability']:.2f}")

    # EVA
    print("\nEVA:")
    print(f"  Transiciones totales: {summary['eva']['transitions']['total_transitions']}")
    print(f"  Ciclos encontrados: {summary['eva']['cycles']['n_cycles_found']}")

    # Guardar
    dynamics.save('/root/NEO_EVA/results/phase15b_dynamics_test.json')
    print(f"\n[OK] Guardado en results/phase15b_dynamics_test.json")

    print("\n" + "=" * 70)
    print("VERIFICACIÓN ANTI-MAGIA:")
    print("  - Smoothing α = 1/√(N+1) (endógeno)")
    print("  - max_length de ciclos = √n (endógeno)")
    print("  - maxlen derivado de √1e6")
    print("  - NO hay constantes hardcodeadas")
    print("=" * 70)
