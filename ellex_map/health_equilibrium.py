"""
Health Equilibrium: L5 - Salud Interior
========================================

Mide el estado de salud del agente segun MED-X:
    - Diagnostico emergente (M1)
    - No-iatrogenia (M3)
    - Rotacion sana del rol (M4)
    - Eficacia del tratamiento (M2)
    - Resiliencia medica

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
class HealthState:
    """Estado de salud interior."""
    health: float               # [0, 1] salud global
    diagnosis_quality: float    # M1: calidad diagnostica
    treatment_efficacy: float   # M2: eficacia tratamiento
    iatrogenesis_free: float    # M3: 1 - tasa iatrogenia
    rotation_health: float      # M4: salud de rotacion
    resilience: float           # Resiliencia medica
    is_healthy: bool            # Si esta en zona sana
    t: int


class HealthEquilibrium(ExistentialLayer):
    """
    L5: Salud Interior (MED-X)

    Integra las metricas del sistema medico para
    dar un indicador de salud general del agente.
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id, LayerType.HEALTH)

        # Historiales por componente
        self._diagnosis_history: List[float] = []
        self._efficacy_history: List[float] = []
        self._iatrogenesis_history: List[float] = []
        self._rotation_history: List[float] = []
        self._resilience_history: List[float] = []

    def _compute_resilience(
        self,
        health_history: List[float],
        stress_events: List[float]
    ) -> float:
        """
        Calcula resiliencia medica.

        Resiliencia = capacidad de recuperar salud despues de estres.
        """
        if len(health_history) < 5 or not stress_events:
            return 0.5

        # Calcular recuperaciones despues de cada evento de estres
        recoveries = []
        window = L_t(self.t)

        for i, stress in enumerate(stress_events):
            if stress > 0.5 and i + window < len(health_history):
                # Hubo estres, ver recuperacion
                pre_stress = health_history[max(0, i-3):i]
                post_stress = health_history[i:i+window]

                if pre_stress and post_stress:
                    pre_mean = np.mean(pre_stress)
                    post_mean = np.mean(post_stress)

                    # Recuperacion = que tanto se acerco al nivel pre-estres
                    if pre_mean > 0:
                        recovery = post_mean / pre_mean
                        recoveries.append(min(1.0, recovery))

        if not recoveries:
            return 0.5

        return float(np.mean(recoveries))

    def compute(self, observations: Dict[str, Any]) -> float:
        """
        Calcula salud interior.

        observations esperadas:
            - diagnosis_quality: [0,1] M1
            - treatment_efficacy: [0,1] M2
            - iatrogenesis_rate: [0,1] M3 (se invierte)
            - rotation_health: [0,1] M4
            - health_history: List[float] (para resiliencia)
            - stress_events: List[float] (para resiliencia)
            - medx_score: [0,1] (opcional, agregado MED-X)
        """
        # Extraer observaciones
        diagnosis = observations.get('diagnosis_quality', 0.5)
        efficacy = observations.get('treatment_efficacy', 0.5)
        iatrogenesis = observations.get('iatrogenesis_rate', 0.0)
        rotation = observations.get('rotation_health', 0.5)
        health_hist = observations.get('health_history', [])
        stress_events = observations.get('stress_events', [])

        # Invertir iatrogenia (alta iatrogenia = baja salud)
        iatrogen_free = 1 - iatrogenesis

        # Calcular resiliencia
        resilience = self._compute_resilience(health_hist, stress_events)

        # Guardar en historiales
        self._diagnosis_history.append(diagnosis)
        self._efficacy_history.append(efficacy)
        self._iatrogenesis_history.append(iatrogen_free)
        self._rotation_history.append(rotation)
        self._resilience_history.append(resilience)

        # Recortar historiales
        max_len = max_history(self.t)
        for hist in [self._diagnosis_history, self._efficacy_history,
                     self._iatrogenesis_history, self._rotation_history,
                     self._resilience_history]:
            if len(hist) > max_len:
                hist[:] = hist[-max_len:]

        # Si hay score MED-X agregado, usarlo como ancla
        medx_score = observations.get('medx_score')
        if medx_score is not None:
            # Combinar MED-X con componentes individuales
            components = [diagnosis, efficacy, iatrogen_free, rotation, resilience]
            component_mean = np.mean(components)
            health = 0.6 * medx_score + 0.4 * component_mean
        else:
            # Agregar con pesos endogenos
            components = [diagnosis, efficacy, iatrogen_free, rotation, resilience]
            histories = [
                self._diagnosis_history,
                self._efficacy_history,
                self._iatrogenesis_history,
                self._rotation_history,
                self._resilience_history
            ]

            weights = []
            for hist in histories:
                if len(hist) < 3:
                    weights.append(1.0)
                else:
                    window = L_t(self.t)
                    var = np.var(hist[-window:])
                    weights.append(1.0 / (var + 0.01))

            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            health = sum(c * w for c, w in zip(components, weights))

        health = float(np.clip(health, 0, 1))

        # Actualizar componentes
        self._current_components = {
            'diagnosis_quality': diagnosis,
            'treatment_efficacy': efficacy,
            'iatrogenesis_free': iatrogen_free,
            'rotation_health': rotation,
            'resilience': resilience,
            'is_healthy': health > 0.5
        }

        return health

    def get_health_state(self) -> HealthState:
        """Obtiene estado detallado de salud."""
        return HealthState(
            health=self._current_value,
            diagnosis_quality=self._current_components.get('diagnosis_quality', 0.5),
            treatment_efficacy=self._current_components.get('treatment_efficacy', 0.5),
            iatrogenesis_free=self._current_components.get('iatrogenesis_free', 1.0),
            rotation_health=self._current_components.get('rotation_health', 0.5),
            resilience=self._current_components.get('resilience', 0.5),
            is_healthy=self._current_components.get('is_healthy', True),
            t=self.t
        )
