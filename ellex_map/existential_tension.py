"""
Existential Tension: L7 - Tension Existencial
==============================================

La cantidad de "vida" que atraviesa el agente.

Tension NO es malo. Es la fuerza vital, el conflicto productivo,
la energia de cambio. Tension muy baja = estancamiento.
Tension muy alta = crisis.

T = Var(drives) + Var(goals) + Stress + Entropy_of_transitions

Todo normalizado por percentiles endogenos.

100% endogeno.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import sys
sys.path.insert(0, '/root/NEO_EVA')

from ellex_map.layer_emergence import ExistentialLayer, LayerType
from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class TensionState:
    """Estado de tension existencial."""
    tension: float              # [0, 1] tension total
    drive_variance: float       # Varianza de drives
    goal_variance: float        # Varianza de metas
    stress: float               # Estres actual
    transition_entropy: float   # Entropia de transiciones
    zone: str                   # 'stagnant', 'healthy', 'crisis'
    t: int


class ExistentialTension(ExistentialLayer):
    """
    L7: Tension Existencial

    Mide la cantidad de "vida" que atraviesa el agente.
    No es coherencia - es la fuerza de cambio.

    Zonas:
        - stagnant: tension < percentil_30 (muy baja)
        - healthy: percentil_30 <= tension <= percentil_70
        - crisis: tension > percentil_70 (muy alta)
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id, LayerType.TENSION)

        # Historiales para normalizacion endogena
        self._drive_variance_history: List[float] = []
        self._goal_variance_history: List[float] = []
        self._stress_history: List[float] = []
        self._entropy_history: List[float] = []
        self._tension_history: List[float] = []

    def _compute_transition_entropy(
        self,
        transitions: List[str]
    ) -> float:
        """
        Calcula entropia de transiciones de fase.

        Alta entropia = muchos cambios diferentes.
        Baja entropia = patron estable.
        """
        if not transitions:
            return 0.0

        # Contar transiciones
        from collections import Counter
        counts = Counter(transitions)
        total = sum(counts.values())

        if total == 0:
            return 0.0

        # Entropia de Shannon
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs)

        # Normalizar por log del numero de tipos
        max_entropy = np.log2(len(counts) + 1)
        normalized = entropy / max_entropy if max_entropy > 0 else 0

        return float(np.clip(normalized, 0, 1))

    def _get_percentile_threshold(
        self,
        history: List[float],
        percentile: float
    ) -> float:
        """Obtiene threshold de percentil endogeno."""
        if len(history) < 3:
            return 0.5
        return np.percentile(history, percentile)

    def _get_zone(self, tension: float) -> str:
        """Determina zona de tension."""
        if len(self._tension_history) < 10:
            # Sin historial suficiente, usar rangos fijos temporalmente
            if tension < 0.3:
                return 'stagnant'
            elif tension > 0.7:
                return 'crisis'
            return 'healthy'

        # Thresholds endogenos
        low = self._get_percentile_threshold(self._tension_history, 30)
        high = self._get_percentile_threshold(self._tension_history, 70)

        if tension < low:
            return 'stagnant'
        elif tension > high:
            return 'crisis'
        return 'healthy'

    def calcular(self, observaciones: Dict[str, Any]) -> float:
        """Alias para compute."""
        return self.compute(observaciones)

    def compute(self, observations: Dict[str, Any]) -> float:
        """
        Calcula tension existencial.

        observations esperadas:
            - drives: List[float] o Dict[str, float] - valores de drives
            - goals: List[float] o Dict[str, float] - valores de metas
            - stress: float [0,1] - estres actual
            - transitions: List[str] - transiciones recientes (opcional)
        """
        # Extraer observaciones
        drives = observations.get('drives', [0.5])
        goals = observations.get('goals', [0.5])
        stress = observations.get('stress', 0.0)
        transitions = observations.get('transitions', [])

        # Convertir dicts a lists si es necesario
        if isinstance(drives, dict):
            drives = list(drives.values())
        if isinstance(goals, dict):
            goals = list(goals.values())

        # Calcular componentes
        drive_var = np.var(drives) if len(drives) > 1 else 0.0
        goal_var = np.var(goals) if len(goals) > 1 else 0.0
        trans_entropy = self._compute_transition_entropy(transitions)

        # Guardar en historiales
        self._drive_variance_history.append(drive_var)
        self._goal_variance_history.append(goal_var)
        self._stress_history.append(stress)
        self._entropy_history.append(trans_entropy)

        # Recortar historiales
        max_len = max_history(self.t)
        for hist in [self._drive_variance_history, self._goal_variance_history,
                     self._stress_history, self._entropy_history]:
            if len(hist) > max_len:
                hist[:] = hist[-max_len:]

        # Normalizar cada componente por su distribucion historica
        def normalize(value: float, history: List[float]) -> float:
            if len(history) < 3:
                return value
            min_val = np.percentile(history, 5)
            max_val = np.percentile(history, 95)
            if max_val - min_val < 1e-8:
                return 0.5
            return (value - min_val) / (max_val - min_val)

        norm_drive = normalize(drive_var, self._drive_variance_history)
        norm_goal = normalize(goal_var, self._goal_variance_history)
        norm_stress = normalize(stress, self._stress_history)
        norm_entropy = normalize(trans_entropy, self._entropy_history)

        # Agregar con pesos endogenos (varianza inversa)
        components = [norm_drive, norm_goal, norm_stress, norm_entropy]
        histories = [
            self._drive_variance_history,
            self._goal_variance_history,
            self._stress_history,
            self._entropy_history
        ]

        weights = []
        for hist in histories:
            if len(hist) < 3:
                weights.append(1.0)
            else:
                var = np.var(hist[-L_t(self.t):])
                weights.append(1.0 / (var + 0.01))

        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        tension = sum(c * w for c, w in zip(components, weights))
        tension = float(np.clip(tension, 0, 1))

        # Guardar tension
        self._tension_history.append(tension)
        if len(self._tension_history) > max_len:
            self._tension_history = self._tension_history[-max_len:]

        # Actualizar componentes para reporte
        self._current_components = {
            'drive_variance': drive_var,
            'goal_variance': goal_var,
            'stress': stress,
            'transition_entropy': trans_entropy,
            'zone': self._get_zone(tension)
        }

        return tension

    def get_tension_state(self) -> TensionState:
        """Obtiene estado detallado de tension."""
        return TensionState(
            tension=self._current_value,
            drive_variance=self._current_components.get('drive_variance', 0),
            goal_variance=self._current_components.get('goal_variance', 0),
            stress=self._current_components.get('stress', 0),
            transition_entropy=self._current_components.get('transition_entropy', 0),
            zone=self._current_components.get('zone', 'healthy'),
            t=self.t
        )
