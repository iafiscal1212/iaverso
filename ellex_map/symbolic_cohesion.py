"""
Symbolic Cohesion: Complemento para L2
=======================================

Analiza la cohesion del sistema simbolico:
    - Densidad de conexiones
    - Estabilidad de conceptos
    - Emergencia de patrones

100% endogeno.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Any, Tuple

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class CohesionState:
    """Estado de cohesion simbolica."""
    cohesion: float             # [0, 1] cohesion global
    concept_stability: float    # Estabilidad de conceptos
    connection_density: float   # Densidad de conexiones
    pattern_emergence: float    # Emergencia de patrones
    semantic_depth: float       # Profundidad semantica
    t: int


class SymbolicCohesion:
    """
    Analiza la cohesion del sistema simbolico del agente.

    Un sistema simbolico cohesivo tiene:
        - Conceptos estables (no cambian constantemente)
        - Conexiones densas (conceptos relacionados)
        - Patrones emergentes (estructuras que se repiten)
        - Profundidad semantica (jerarquias de significado)
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.t = 0

        # Conceptos activos y sus activaciones
        self._concept_history: Dict[str, List[float]] = {}
        self._connection_history: List[Set[Tuple[str, str]]] = []
        self._pattern_counts: Dict[str, int] = {}
        self._cohesion_history: List[float] = []

    def observe_concepts(
        self,
        active_concepts: Dict[str, float],
        connections: List[Tuple[str, str]] = None
    ):
        """
        Observa conceptos activos y conexiones.

        Args:
            active_concepts: {concepto: activacion}
            connections: Lista de pares conectados
        """
        # Actualizar historial de conceptos
        for concept, activation in active_concepts.items():
            if concept not in self._concept_history:
                self._concept_history[concept] = []
            self._concept_history[concept].append(activation)

            # Recortar
            max_len = max_history(self.t)
            if len(self._concept_history[concept]) > max_len:
                self._concept_history[concept] = \
                    self._concept_history[concept][-max_len:]

        # Actualizar conexiones
        if connections:
            conn_set = set(tuple(sorted(c)) for c in connections)
            self._connection_history.append(conn_set)

            max_len = max_history(self.t)
            if len(self._connection_history) > max_len:
                self._connection_history = self._connection_history[-max_len:]

    def _compute_concept_stability(self) -> float:
        """
        Calcula estabilidad de conceptos.

        Conceptos estables = activaciones consistentes.
        """
        if not self._concept_history:
            return 0.5

        stabilities = []
        window = L_t(self.t)

        for concept, history in self._concept_history.items():
            if len(history) >= 3:
                recent = history[-window:]
                std = np.std(recent)
                mean = np.mean(recent)
                cv = std / (mean + 1e-8)
                stability = 1 / (1 + cv)
                stabilities.append(stability)

        if not stabilities:
            return 0.5

        return float(np.mean(stabilities))

    def _compute_connection_density(self) -> float:
        """
        Calcula densidad de conexiones.

        Alta densidad = muchas conexiones estables.
        """
        if not self._connection_history:
            return 0.5

        # Contar conexiones que persisten
        window = min(L_t(self.t), len(self._connection_history))
        recent = self._connection_history[-window:]

        if not recent:
            return 0.5

        # Conexiones que aparecen en multiples pasos
        all_connections = set()
        connection_counts: Dict[Tuple, int] = {}

        for conn_set in recent:
            all_connections.update(conn_set)
            for conn in conn_set:
                connection_counts[conn] = connection_counts.get(conn, 0) + 1

        if not all_connections:
            return 0.5

        # Densidad = proporcion de conexiones persistentes
        persistent = sum(1 for count in connection_counts.values() if count > 1)
        density = persistent / len(all_connections)

        return float(np.clip(density, 0, 1))

    def _compute_pattern_emergence(self) -> float:
        """
        Calcula emergencia de patrones.

        Patrones = combinaciones de conceptos que se repiten.
        """
        if not self._concept_history:
            return 0.5

        # Identificar conceptos co-activos
        window = L_t(self.t)
        active_sets = []

        # Reconstruir conjuntos de conceptos activos por paso
        for t in range(max(0, self.t - window), self.t):
            active = set()
            for concept, history in self._concept_history.items():
                idx = t - (self.t - len(history))
                if 0 <= idx < len(history) and history[idx] > 0.5:
                    active.add(concept)
            if active:
                active_sets.append(frozenset(active))

        if len(active_sets) < 3:
            return 0.5

        # Contar patrones repetidos
        from collections import Counter
        pattern_counts = Counter(active_sets)

        # Emergencia = proporcion de patrones que se repiten
        repeated = sum(1 for count in pattern_counts.values() if count > 1)
        total = len(pattern_counts)

        if total == 0:
            return 0.5

        emergence = repeated / total

        return float(np.clip(emergence, 0, 1))

    def _compute_semantic_depth(self) -> float:
        """
        Calcula profundidad semantica.

        Profundidad = niveles de abstraccion detectados.
        """
        if not self._concept_history:
            return 0.5

        # Aproximacion: conceptos con alta activacion sostenida = abstractos
        # Conceptos con activacion variable = concretos
        concepts_by_abstraction = []

        for concept, history in self._concept_history.items():
            if len(history) >= 3:
                mean = np.mean(history)
                std = np.std(history)

                # Alta media + baja varianza = abstracto
                abstraction = mean * (1 - std)
                concepts_by_abstraction.append((concept, abstraction))

        if not concepts_by_abstraction:
            return 0.5

        # Profundidad = varianza de niveles de abstraccion
        abstractions = [a for _, a in concepts_by_abstraction]
        variance = np.var(abstractions)

        # Normalizar: alta varianza = mas niveles = mas profundidad
        depth = 2 / (1 + np.exp(-5 * variance)) - 1

        return float(np.clip(depth, 0, 1))

    def compute(self, observations: Dict[str, Any] = None) -> CohesionState:
        """
        Calcula estado de cohesion simbolica.

        observations opcionales:
            - active_concepts: Dict[str, float]
            - connections: List[Tuple[str, str]]
        """
        self.t += 1

        # Agregar observaciones si hay
        if observations:
            concepts = observations.get('active_concepts', {})
            connections = observations.get('connections', [])
            self.observe_concepts(concepts, connections)

        # Calcular componentes
        stability = self._compute_concept_stability()
        density = self._compute_connection_density()
        emergence = self._compute_pattern_emergence()
        depth = self._compute_semantic_depth()

        # Cohesion general con pesos endogenos
        components = [stability, density, emergence, depth]
        cohesion = np.mean(components)  # Simplificado: igual peso

        self._cohesion_history.append(cohesion)
        max_len = max_history(self.t)
        if len(self._cohesion_history) > max_len:
            self._cohesion_history = self._cohesion_history[-max_len:]

        return CohesionState(
            cohesion=float(cohesion),
            concept_stability=float(stability),
            connection_density=float(density),
            pattern_emergence=float(emergence),
            semantic_depth=float(depth),
            t=self.t
        )
