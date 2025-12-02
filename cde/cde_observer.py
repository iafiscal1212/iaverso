"""
CDE Observer - Recolector de Señales Internas
==============================================

Observa decisiones, eventos y recursos del sistema.
Solo registra, no interviene.

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ObservationState:
    """Estado de observación en un instante."""
    t: int
    decisions: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    resources: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CDEObserver:
    """
    Recolector de señales internas del sistema.

    Observa:
    - Decisiones de módulos
    - Eventos del sistema
    - Estado de recursos

    No interviene, solo registra.
    """

    def __init__(self):
        self.t = 0
        self._decisions: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []
        self._resources: Dict[str, float] = {}
        self._history: List[ObservationState] = []

    def observe_decision(
        self,
        module: str,
        decision_vector: np.ndarray,
        confidence: float
    ) -> None:
        """
        Registra una decisión de un módulo.

        Args:
            module: Nombre del módulo que decide
            decision_vector: Vector de la decisión
            confidence: Confianza en la decisión [0, 1]
        """
        self._decisions.append({
            't': self.t,
            'module': module,
            'decision_vector': decision_vector.copy(),
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })

    def observe_event(
        self,
        event_type: str,
        payload: Dict[str, Any]
    ) -> None:
        """
        Registra un evento del sistema.

        Args:
            event_type: Tipo de evento
            payload: Datos del evento
        """
        self._events.append({
            't': self.t,
            'event_type': event_type,
            'payload': payload,
            'timestamp': datetime.now().isoformat()
        })

    def observe_resources(
        self,
        cpu: float,
        ram: float,
        load: float,
        queue_length: int
    ) -> None:
        """
        Registra estado de recursos.

        Args:
            cpu: Uso de CPU [0, 1]
            ram: Uso de RAM [0, 1]
            load: Carga general [0, 1]
            queue_length: Longitud de cola
        """
        self._resources = {
            'cpu': cpu,
            'ram': ram,
            'load': load,
            'queue_length': queue_length,
            'timestamp': datetime.now().isoformat()
        }

    def step(self) -> ObservationState:
        """
        Finaliza un paso de observación.

        Returns:
            Estado de observación actual
        """
        self.t += 1

        state = ObservationState(
            t=self.t,
            decisions=self._decisions.copy(),
            events=self._events.copy(),
            resources=self._resources.copy()
        )

        self._history.append(state)

        # Limpiar para siguiente paso
        self._decisions = []
        self._events = []

        return state

    def get_observation_state(self) -> np.ndarray:
        """
        Convierte estado de observación a vector numérico.

        Returns:
            Vector de estado para WorldX
        """
        # Métricas de decisiones
        n_decisions = len(self._decisions)
        avg_confidence = np.mean([d['confidence'] for d in self._decisions]) if self._decisions else 1/2

        # Métricas de eventos
        n_events = len(self._events)

        # Métricas de recursos
        cpu = self._resources.get('cpu', 1/2)
        ram = self._resources.get('ram', 1/2)
        load = self._resources.get('load', 1/2)
        queue = self._resources.get('queue_length', 0)

        # Normalizar queue endógenamente
        if len(self._history) > 5:
            queue_hist = [h.resources.get('queue_length', 0) for h in self._history[-10:]]
            queue_max = max(queue_hist) if queue_hist else 1
            queue_norm = queue / (queue_max + 1)
        else:
            queue_norm = queue / (queue + 1)

        return np.array([
            n_decisions / (n_decisions + 1),  # Normalizado
            avg_confidence,
            n_events / (n_events + 1),
            cpu,
            ram,
            load,
            queue_norm
        ])

    def get_history(self) -> List[ObservationState]:
        """Retorna historial de observaciones."""
        return self._history

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de observación."""
        return {
            't': self.t,
            'total_decisions': sum(len(h.decisions) for h in self._history),
            'total_events': sum(len(h.events) for h in self._history),
            'history_length': len(self._history)
        }
