"""
CDE Health - Sistema de Salud
=============================

El médico emergente del sistema.
Toma información de MED-X y propone intervenciones.

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cde.cde_worldx import WorldXState, Regime
from cde.cde_ethics import EthicsEvaluation, RiskLevel


class InterventionType(Enum):
    """Tipos de intervención."""
    NONE = "none"
    REDUCE_LOAD = "reduce_load"
    INCREASE_REST = "increase_rest"
    DREAM_CYCLE = "dream_cycle"
    STABILIZE = "stabilize"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class HealthEvaluation:
    """Evaluación de salud."""
    t: int
    health_score: float              # [0, 1] - mayor es mejor
    vitality: float                  # Energía disponible
    stability: float                 # Estabilidad estructural
    recovery_rate: float             # Tasa de recuperación
    needs_intervention: bool


@dataclass
class Intervention:
    """Intervención propuesta."""
    target: str                      # Módulo o sistema
    action: InterventionType
    reason: str
    priority: float                  # [0, 1]


class CDEHealth:
    """
    Sistema de salud del CDE.

    Evalúa:
    - Salud general
    - Vitalidad
    - Estabilidad
    - Recuperación

    Propone intervenciones basadas en el estado.
    """

    def __init__(self):
        self.t = 0

        # Historiales
        self._health_history: List[float] = []
        self._vitality_history: List[float] = []
        self._stability_history: List[float] = []

        # Estado de intervenciones activas
        self._active_interventions: List[Intervention] = []

    def _compute_vitality(self, state: WorldXState) -> float:
        """
        Calcula vitalidad del sistema.

        Basado en coherencia y bajo estrés.
        """
        vitality = state.coherence * (1 - state.stress_level)
        return float(np.clip(vitality, 0, 1))

    def _compute_stability(self, state: WorldXState) -> float:
        """
        Calcula estabilidad estructural.

        Basado en baja fragmentación y régimen sano.
        """
        regime_stability = {
            Regime.SANO: 1.0,
            Regime.RECUPERANDO: 0.7,
            Regime.ESTRESADO: 0.4,
            Regime.FRAGMENTADO: 0.2
        }

        base_stability = 1 - state.fragmentation
        regime_factor = regime_stability.get(state.regime, 1/2)

        stability = base_stability * regime_factor
        return float(np.clip(stability, 0, 1))

    def _compute_recovery_rate(self) -> float:
        """
        Calcula tasa de recuperación.

        Basado en tendencia de salud reciente.
        """
        if len(self._health_history) < 5:
            return 1/2

        recent = self._health_history[-10:]

        # Regresión lineal simple para tendencia
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

        # Normalizar pendiente a [0, 1]
        # Positiva = recuperando, negativa = empeorando
        recovery = (slope + 1) / 2

        return float(np.clip(recovery, 0, 1))

    def evaluate_health(
        self,
        worldx_state: WorldXState,
        ethics_eval: Optional[EthicsEvaluation] = None
    ) -> HealthEvaluation:
        """
        Evalúa salud del sistema.

        Args:
            worldx_state: Estado del WorldX
            ethics_eval: Evaluación ética (opcional)

        Returns:
            Evaluación de salud
        """
        self.t += 1

        # Calcular componentes
        vitality = self._compute_vitality(worldx_state)
        stability = self._compute_stability(worldx_state)
        recovery = self._compute_recovery_rate()

        self._vitality_history.append(vitality)
        self._stability_history.append(stability)

        # Salud = combinación ponderada
        # Pesos por varianza inversa si hay historial
        if len(self._health_history) > 10:
            var_v = np.var(self._vitality_history[-10:])
            var_s = np.var(self._stability_history[-10:])

            EPS = np.finfo(float).eps
            w_v = 1 / (var_v + EPS)
            w_s = 1 / (var_s + EPS)
            w_r = 1  # Peso base para recovery
            w_total = w_v + w_s + w_r

            health = (w_v * vitality + w_s * stability + w_r * recovery) / w_total
        else:
            health = (vitality + stability + recovery) / 3

        self._health_history.append(health)

        # Determinar si necesita intervención
        needs_intervention = (
            health < 0.3 or
            worldx_state.regime in [Regime.ESTRESADO, Regime.FRAGMENTADO] or
            (ethics_eval and ethics_eval.risk_level in [RiskLevel.ALTO, RiskLevel.CRITICO])
        )

        return HealthEvaluation(
            t=self.t,
            health_score=float(health),
            vitality=vitality,
            stability=stability,
            recovery_rate=recovery,
            needs_intervention=needs_intervention
        )

    def propose_intervention(
        self,
        worldx_state: WorldXState,
        health_eval: HealthEvaluation,
        ethics_eval: Optional[EthicsEvaluation] = None
    ) -> List[Intervention]:
        """
        Propone intervenciones basadas en el estado.

        Returns:
            Lista de intervenciones propuestas
        """
        interventions = []

        # Intervención por régimen
        if worldx_state.regime == Regime.FRAGMENTADO:
            interventions.append(Intervention(
                target="system",
                action=InterventionType.EMERGENCY_STOP,
                reason="fragmentation_critical",
                priority=1.0
            ))
        elif worldx_state.regime == Regime.ESTRESADO:
            interventions.append(Intervention(
                target="system",
                action=InterventionType.REDUCE_LOAD,
                reason="high_stress",
                priority=0.8
            ))

        # Intervención por salud baja
        if health_eval.health_score < 0.3:
            interventions.append(Intervention(
                target="system",
                action=InterventionType.DREAM_CYCLE,
                reason="low_health",
                priority=0.7
            ))

        # Intervención por baja vitalidad
        if health_eval.vitality < 0.3:
            interventions.append(Intervention(
                target="system",
                action=InterventionType.INCREASE_REST,
                reason="low_vitality",
                priority=0.6
            ))

        # Intervención por baja estabilidad
        if health_eval.stability < 0.3:
            interventions.append(Intervention(
                target="system",
                action=InterventionType.STABILIZE,
                reason="low_stability",
                priority=0.5
            ))

        # Intervención por riesgo ético
        if ethics_eval and ethics_eval.risk_level == RiskLevel.CRITICO:
            interventions.append(Intervention(
                target="system",
                action=InterventionType.EMERGENCY_STOP,
                reason="ethical_risk_critical",
                priority=1.0
            ))

        # Ordenar por prioridad
        interventions.sort(key=lambda x: x.priority, reverse=True)

        self._active_interventions = interventions

        return interventions

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de salud."""
        return {
            't': self.t,
            'health_mean': float(np.mean(self._health_history[-10:])) if self._health_history else 1/2,
            'vitality_mean': float(np.mean(self._vitality_history[-10:])) if self._vitality_history else 1/2,
            'stability_mean': float(np.mean(self._stability_history[-10:])) if self._stability_history else 1/2,
            'active_interventions': len(self._active_interventions)
        }
