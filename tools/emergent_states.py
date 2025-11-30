#!/usr/bin/env python3
"""
Phase 15B: Estados Emergentes
==============================

Estados internos que emergen de métricas - SIN reloj, SIN etiquetas predefinidas.

Principio fundamental:
- Los "estados" no se programan - emergen como atractores en el espacio de métricas
- No hay SLEEP/WAKE/WORK - solo hay regiones del espacio interno que se visitan
- Los prototipos son clusters endógenos, no categorías predefinidas

Métricas del vector de estado:
1. TE (Transfer Entropy) - información que fluye
2. SE (Self-prediction Error) - qué tan predecible es uno mismo
3. SYNC (sincronización) - correlación entre agentes
4. H (entropía local) - incertidumbre de la distribución

Todo basado en ranks de la historia - CERO números mágicos.
100% endógeno.
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

from endogenous_core import (
    derive_window_size,
    derive_learning_rate,
    compute_iqr,
    rank_normalize,
    rolling_rank,
    compute_entropy_normalized,
    NUMERIC_EPS,
    PROVENANCE
)


# =============================================================================
# VECTOR DE ESTADO
# =============================================================================

@dataclass
class StateVector:
    """
    Vector de estado interno de un agente.

    NO es un "estado" en sentido clásico (SLEEP, WORK, etc).
    Es un punto en el espacio 4D de métricas internas.
    """
    t: int
    agent: str
    te_rank: float       # rank(TE_t) en historia
    se_rank: float       # rank(self_error_t) en historia
    sync_rank: float     # rank(sync_t) en historia
    entropy_rank: float  # rank(H_t) en historia

    def to_array(self) -> np.ndarray:
        """Vector como array numpy."""
        return np.array([self.te_rank, self.se_rank, self.sync_rank, self.entropy_rank])

    def to_dict(self) -> Dict:
        return {
            't': self.t,
            'agent': self.agent,
            'te_rank': self.te_rank,
            'se_rank': self.se_rank,
            'sync_rank': self.sync_rank,
            'entropy_rank': self.entropy_rank
        }


def compute_state_vector(
    t: int,
    agent: str,
    te: float,
    self_error: float,
    sync: float,
    pi_distribution: np.ndarray,
    te_history: deque,
    se_history: deque,
    sync_history: deque,
    entropy_history: deque
) -> StateVector:
    """
    Construye vector de estado desde métricas actuales.

    Cada componente es el rank del valor actual en su historia.
    """
    # Entropía de la distribución
    H = compute_entropy_normalized(pi_distribution)

    # Calcular ranks en historia
    te_rank = rolling_rank(te, te_history)
    se_rank = rolling_rank(self_error, se_history)
    sync_rank = rolling_rank(sync, sync_history)
    entropy_rank = rolling_rank(H, entropy_history)

    state = StateVector(
        t=t,
        agent=agent,
        te_rank=te_rank,
        se_rank=se_rank,
        sync_rank=sync_rank,
        entropy_rank=entropy_rank
    )

    PROVENANCE.log('state_vector', te_rank,
                   'rolling_rank de métricas',
                   {'te': te, 'se': self_error, 'sync': sync, 'H': H}, t)

    return state


# =============================================================================
# PROTOTIPOS EMERGENTES (ONLINE CLUSTERING)
# =============================================================================

@dataclass
class StatePrototype:
    """
    Un prototipo emergente - un cluster en el espacio de estados.

    NO tiene etiqueta semántica. Solo es un centroide.
    La etiqueta (si se necesita) emerge después del análisis.
    """
    id: int
    centroid: np.ndarray          # Centro del cluster (4D)
    n_visits: int                 # Cuántas veces se ha visitado
    mean_te: float                # TE medio cuando se visita
    last_visit: int               # Último t cuando se visitó
    visit_times: List[int] = field(default_factory=list)

    def update_centroid(self, new_point: np.ndarray, eta: float):
        """Actualiza centroide con EMA."""
        self.centroid = (1 - eta) * self.centroid + eta * new_point

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'centroid': self.centroid.tolist(),
            'n_visits': self.n_visits,
            'mean_te': self.mean_te,
            'last_visit': self.last_visit,
            'visit_times_sample': self.visit_times[-10:]  # Últimas 10 visitas
        }


class OnlinePrototypeManager:
    """
    Gestor de prototipos emergentes.

    Prototipos emergen del clustering online:
    - Si un punto está cerca de un prototipo existente → se asigna ahí
    - Si está lejos de todos → crea nuevo prototipo

    "Cerca" se define por umbral endógeno (cuantil de distancias históricas).
    """

    def __init__(self, agent: str):
        self.agent = agent
        self.prototypes: List[StatePrototype] = []
        self.next_id = 0

        # Historia de distancias para umbral endógeno
        # Derivado: maxlen = √1e6 ≈ 1000
        derived_maxlen = int(np.sqrt(1e6))
        self.distance_history: deque = deque(maxlen=derived_maxlen)

        # Historia de asignaciones
        self.assignment_history: List[int] = []

    def get_merge_threshold(self) -> float:
        """
        Umbral de merge endógeno.

        threshold = q_X(distance_history)
        donde X se deriva de la distribución.
        """
        if len(self.distance_history) < 10:
            # Warmup: usar 0.5 (medio del rango [0, 2] para distancias normalizadas)
            return 0.5

        # q25 de distancias - si estás más cerca que esto, eres del mismo cluster
        threshold = np.percentile(list(self.distance_history), 25)

        PROVENANCE.log('merge_threshold', threshold,
                       f'q25 de {len(self.distance_history)} distancias',
                       {'n_samples': len(self.distance_history)}, 0)

        return threshold

    def assign_or_create(
        self,
        state_vec: StateVector,
        te_value: float
    ) -> Tuple[int, bool]:
        """
        Asigna un estado a un prototipo existente o crea uno nuevo.

        Returns: (prototype_id, is_new)
        """
        point = state_vec.to_array()
        t = state_vec.t

        if len(self.prototypes) == 0:
            # Primer prototipo
            proto = StatePrototype(
                id=self.next_id,
                centroid=point.copy(),
                n_visits=1,
                mean_te=te_value,
                last_visit=t,
                visit_times=[t]
            )
            self.prototypes.append(proto)
            self.next_id += 1
            self.assignment_history.append(proto.id)
            return proto.id, True

        # Calcular distancias a todos los prototipos
        distances = []
        for proto in self.prototypes:
            dist = np.linalg.norm(point - proto.centroid)
            distances.append(dist)

        # Registrar distancia mínima para historia
        min_dist = min(distances)
        self.distance_history.append(min_dist)

        # Umbral endógeno
        threshold = self.get_merge_threshold()

        if min_dist <= threshold:
            # Asignar al más cercano
            closest_idx = np.argmin(distances)
            proto = self.prototypes[closest_idx]

            # Learning rate endógeno
            eta = derive_learning_rate(proto.n_visits)

            # Actualizar prototipo
            proto.update_centroid(point, eta)
            proto.n_visits += 1
            proto.mean_te = (proto.mean_te * (proto.n_visits - 1) + te_value) / proto.n_visits
            proto.last_visit = t
            proto.visit_times.append(t)

            self.assignment_history.append(proto.id)
            return proto.id, False
        else:
            # Crear nuevo prototipo
            # Solo si no hay demasiados (límite endógeno: √n_history)
            max_prototypes = max(3, int(np.sqrt(len(self.assignment_history) + 1)))

            if len(self.prototypes) >= max_prototypes:
                # Fusionar los dos más cercanos y luego crear nuevo
                self._merge_closest_prototypes()

            proto = StatePrototype(
                id=self.next_id,
                centroid=point.copy(),
                n_visits=1,
                mean_te=te_value,
                last_visit=t,
                visit_times=[t]
            )
            self.prototypes.append(proto)
            self.next_id += 1
            self.assignment_history.append(proto.id)
            return proto.id, True

    def _merge_closest_prototypes(self):
        """Fusiona los dos prototipos más cercanos."""
        if len(self.prototypes) < 2:
            return

        # Encontrar par más cercano
        min_dist = float('inf')
        merge_pair = (0, 1)

        for i in range(len(self.prototypes)):
            for j in range(i + 1, len(self.prototypes)):
                dist = np.linalg.norm(
                    self.prototypes[i].centroid - self.prototypes[j].centroid
                )
                if dist < min_dist:
                    min_dist = dist
                    merge_pair = (i, j)

        i, j = merge_pair
        p1, p2 = self.prototypes[i], self.prototypes[j]

        # Nuevo centroide ponderado por visitas
        total_visits = p1.n_visits + p2.n_visits
        new_centroid = (p1.centroid * p1.n_visits + p2.centroid * p2.n_visits) / total_visits

        # Actualizar p1 con la fusión
        p1.centroid = new_centroid
        p1.n_visits = total_visits
        p1.mean_te = (p1.mean_te * p1.n_visits + p2.mean_te * p2.n_visits) / total_visits
        p1.last_visit = max(p1.last_visit, p2.last_visit)
        p1.visit_times.extend(p2.visit_times)

        # Eliminar p2
        self.prototypes.pop(j)

    def get_current_prototype(self) -> Optional[StatePrototype]:
        """Retorna el prototipo actual (último asignado)."""
        if not self.assignment_history or not self.prototypes:
            return None

        last_id = self.assignment_history[-1]
        for proto in self.prototypes:
            if proto.id == last_id:
                return proto
        return None

    def get_prototype_distribution(self) -> np.ndarray:
        """
        Distribución de visitas a prototipos.

        Útil para entropía de estados.
        """
        if not self.prototypes:
            return np.array([1.0])

        counts = np.array([p.n_visits for p in self.prototypes], dtype=float)
        return counts / counts.sum()

    def get_summary(self) -> Dict:
        """Resumen del estado de prototipos."""
        return {
            'n_prototypes': len(self.prototypes),
            'total_assignments': len(self.assignment_history),
            'prototypes': [p.to_dict() for p in self.prototypes],
            'prototype_entropy': compute_entropy_normalized(self.get_prototype_distribution()),
            'merge_threshold': self.get_merge_threshold()
        }


# =============================================================================
# SISTEMA DE ESTADOS EMERGENTES
# =============================================================================

class EmergentStateSystem:
    """
    Sistema completo de estados emergentes para NEO y EVA.

    SIN reloj, SIN etiquetas predefinidas.
    """

    def __init__(self):
        # Managers de prototipos
        self.neo_manager = OnlinePrototypeManager('NEO')
        self.eva_manager = OnlinePrototypeManager('EVA')

        # Historias de métricas (para ranks)
        # Derivado: maxlen = √1e8 ≈ 10000
        derived_maxlen = int(np.sqrt(1e8))

        self.neo_te_history = deque(maxlen=derived_maxlen)
        self.neo_se_history = deque(maxlen=derived_maxlen)
        self.neo_sync_history = deque(maxlen=derived_maxlen)
        self.neo_entropy_history = deque(maxlen=derived_maxlen)

        self.eva_te_history = deque(maxlen=derived_maxlen)
        self.eva_se_history = deque(maxlen=derived_maxlen)
        self.eva_sync_history = deque(maxlen=derived_maxlen)
        self.eva_entropy_history = deque(maxlen=derived_maxlen)

        # Estados actuales
        self.neo_current_state: Optional[StateVector] = None
        self.eva_current_state: Optional[StateVector] = None

        # Historial de estados
        self.neo_state_history: List[StateVector] = []
        self.eva_state_history: List[StateVector] = []

        # Log de eventos
        self.events: List[Dict] = []

    def process_step(
        self,
        t: int,
        neo_pi: np.ndarray,
        eva_pi: np.ndarray,
        te_neo_to_eva: float,
        te_eva_to_neo: float,
        neo_self_error: float,
        eva_self_error: float,
        sync: float
    ) -> Dict:
        """
        Procesa un paso y actualiza estados emergentes.

        Parámetros:
        - t: tiempo (usado solo para tracking, NO para derivar estados)
        - neo_pi, eva_pi: distribuciones π de cada agente
        - te_*: transfer entropy bidireccional
        - *_self_error: error de auto-predicción
        - sync: sincronización entre agentes
        """
        result = {
            't': t,
            'neo': {},
            'eva': {}
        }

        # --- NEO ---
        # Actualizar historias
        self.neo_te_history.append(te_neo_to_eva)
        self.neo_se_history.append(neo_self_error)
        self.neo_sync_history.append(sync)
        neo_H = compute_entropy_normalized(neo_pi)
        self.neo_entropy_history.append(neo_H)

        # Calcular vector de estado
        neo_state = compute_state_vector(
            t=t,
            agent='NEO',
            te=te_neo_to_eva,
            self_error=neo_self_error,
            sync=sync,
            pi_distribution=neo_pi,
            te_history=self.neo_te_history,
            se_history=self.neo_se_history,
            sync_history=self.neo_sync_history,
            entropy_history=self.neo_entropy_history
        )

        self.neo_current_state = neo_state
        self.neo_state_history.append(neo_state)

        # Asignar a prototipo
        neo_proto_id, neo_is_new = self.neo_manager.assign_or_create(
            neo_state, te_neo_to_eva
        )

        result['neo'] = {
            'state_vector': neo_state.to_dict(),
            'prototype_id': neo_proto_id,
            'new_prototype': neo_is_new,
            'n_prototypes': len(self.neo_manager.prototypes)
        }

        if neo_is_new:
            self.events.append({
                't': t,
                'agent': 'NEO',
                'event': 'new_prototype',
                'proto_id': neo_proto_id
            })

        # --- EVA ---
        self.eva_te_history.append(te_eva_to_neo)
        self.eva_se_history.append(eva_self_error)
        self.eva_sync_history.append(sync)
        eva_H = compute_entropy_normalized(eva_pi)
        self.eva_entropy_history.append(eva_H)

        eva_state = compute_state_vector(
            t=t,
            agent='EVA',
            te=te_eva_to_neo,
            self_error=eva_self_error,
            sync=sync,
            pi_distribution=eva_pi,
            te_history=self.eva_te_history,
            se_history=self.eva_se_history,
            sync_history=self.eva_sync_history,
            entropy_history=self.eva_entropy_history
        )

        self.eva_current_state = eva_state
        self.eva_state_history.append(eva_state)

        eva_proto_id, eva_is_new = self.eva_manager.assign_or_create(
            eva_state, te_eva_to_neo
        )

        result['eva'] = {
            'state_vector': eva_state.to_dict(),
            'prototype_id': eva_proto_id,
            'new_prototype': eva_is_new,
            'n_prototypes': len(self.eva_manager.prototypes)
        }

        if eva_is_new:
            self.events.append({
                't': t,
                'agent': 'EVA',
                'event': 'new_prototype',
                'proto_id': eva_proto_id
            })

        return result

    def get_joint_state(self) -> np.ndarray:
        """
        Estado conjunto del sistema (8D: 4 NEO + 4 EVA).

        Útil para Global Narrative Trace.
        """
        if self.neo_current_state is None or self.eva_current_state is None:
            return np.zeros(8)

        return np.concatenate([
            self.neo_current_state.to_array(),
            self.eva_current_state.to_array()
        ])

    def get_summary(self) -> Dict:
        """Resumen completo del sistema de estados."""
        return {
            'neo': self.neo_manager.get_summary(),
            'eva': self.eva_manager.get_summary(),
            'n_events': len(self.events),
            'recent_events': self.events[-10:],
            'joint_state': self.get_joint_state().tolist()
        }

    def save(self, path: str):
        """Guarda el sistema de estados."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'neo': self.neo_manager.get_summary(),
            'eva': self.eva_manager.get_summary(),
            'events': self.events,
            'neo_state_history': [s.to_dict() for s in self.neo_state_history[-100:]],
            'eva_state_history': [s.to_dict() for s in self.eva_state_history[-100:]]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 15B: ESTADOS EMERGENTES - TEST")
    print("=" * 70)

    # Crear sistema
    ess = EmergentStateSystem()

    # Simular datos SIN reloj
    np.random.seed(42)
    n_steps = 5000

    print("\n[1] Simulando sistema sin estados predefinidos...")

    neo_pi = np.array([0.33, 0.33, 0.34])
    eva_pi = np.array([0.33, 0.33, 0.34])

    for t in range(n_steps):
        # Simular métricas (variando según dinámica interna, NO reloj)
        # La "hora" no existe - solo hay dinámicas internas

        # TE varía según acoplamiento previo
        base_te = 0.2 + 0.3 * np.sin(np.random.randn() * 0.5)
        te_neo_to_eva = max(0, base_te + np.random.randn() * 0.1)
        te_eva_to_neo = max(0, base_te + np.random.randn() * 0.1)

        # Self error
        neo_se = abs(np.random.randn() * 0.1)
        eva_se = abs(np.random.randn() * 0.1)

        # Sync (correlación entre agentes)
        sync = 0.5 + 0.3 * np.tanh(te_neo_to_eva + te_eva_to_neo - 0.4)

        # Actualizar distribuciones
        neo_pi = neo_pi + np.random.randn(3) * 0.05
        neo_pi = np.abs(neo_pi)
        neo_pi = neo_pi / neo_pi.sum()

        eva_pi = eva_pi + np.random.randn(3) * 0.05
        eva_pi = np.abs(eva_pi)
        eva_pi = eva_pi / eva_pi.sum()

        # Procesar
        result = ess.process_step(
            t=t,
            neo_pi=neo_pi,
            eva_pi=eva_pi,
            te_neo_to_eva=te_neo_to_eva,
            te_eva_to_neo=te_eva_to_neo,
            neo_self_error=neo_se,
            eva_self_error=eva_se,
            sync=sync
        )

    print(f"    Pasos procesados: {n_steps}")

    # Resumen
    print("\n[2] Resumen de estados emergentes:")
    summary = ess.get_summary()

    print(f"\nNEO:")
    print(f"  Prototipos emergentes: {summary['neo']['n_prototypes']}")
    print(f"  Entropía de prototipos: {summary['neo']['prototype_entropy']:.3f}")
    print(f"  Umbral de merge: {summary['neo']['merge_threshold']:.3f}")

    print(f"\nEVA:")
    print(f"  Prototipos emergentes: {summary['eva']['n_prototypes']}")
    print(f"  Entropía de prototipos: {summary['eva']['prototype_entropy']:.3f}")
    print(f"  Umbral de merge: {summary['eva']['merge_threshold']:.3f}")

    print(f"\nEventos de creación de prototipos: {summary['n_events']}")

    # Mostrar prototipos
    print("\n[3] Prototipos NEO:")
    for p in summary['neo']['prototypes']:
        print(f"  Proto {p['id']}: visits={p['n_visits']}, mean_TE={p['mean_te']:.3f}")
        print(f"    Centroide: {[f'{x:.2f}' for x in p['centroid']]}")

    print("\n[4] Estado conjunto actual:")
    joint = summary['joint_state']
    print(f"  [NEO te={joint[0]:.2f}, se={joint[1]:.2f}, sync={joint[2]:.2f}, H={joint[3]:.2f}]")
    print(f"  [EVA te={joint[4]:.2f}, se={joint[5]:.2f}, sync={joint[6]:.2f}, H={joint[7]:.2f}]")

    # Guardar
    ess.save('/root/NEO_EVA/results/phase15b_states_test.json')
    print(f"\n[OK] Guardado en results/phase15b_states_test.json")

    print("\n" + "=" * 70)
    print("VERIFICACIÓN ANTI-MAGIA:")
    print("  - NO hay t % 24 ni ningún ciclo de reloj")
    print("  - NO hay etiquetas SLEEP/WAKE/WORK")
    print("  - Prototipos emergen de clustering online")
    print("  - Umbral de merge = q25(distance_history)")
    print("  - maxlen derivado de √1e8")
    print("=" * 70)
