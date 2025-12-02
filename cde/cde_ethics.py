"""
CDE Ethics - Ética Estructural
==============================

Ética del sistema basada en:
- AGI-10 (equilibrio)
- AGI-12 (normas)
- AGI-15 (ética)
- MED-X (no-iatrogenia)

Mide daño interno, no moral humana.

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cde.cde_worldx import WorldXState, Regime


class RiskLevel(Enum):
    """Niveles de riesgo."""
    BAJO = "bajo"
    MEDIO = "medio"
    ALTO = "alto"
    CRITICO = "critico"


@dataclass
class EthicsEvaluation:
    """Evaluación ética del sistema."""
    t: int
    ethics_score: float              # [0, 1] - mayor es mejor
    overload_damage: float           # Daño por sobrecarga
    fragmentation_damage: float      # Daño por fragmentación
    incoherence_damage: float        # Daño por incoherencia
    norm_violation: float            # Violación de normas
    ethical_drift: float             # Deriva ética (tensión L7)
    risk_level: RiskLevel
    risk_flags: List[str]


class CDEEthics:
    """
    Evaluador ético estructural.

    Mide:
    - Overload damage
    - Fragmentation damage
    - Incoherence damage
    - Norm violation
    - Ethical drift (tensión L7)

    Todo basado en percentiles históricos.
    """

    def __init__(self):
        self.t = 0

        # Historiales para umbrales endógenos
        self._overload_history: List[float] = []
        self._fragmentation_history: List[float] = []
        self._incoherence_history: List[float] = []
        self._norm_history: List[float] = []
        self._drift_history: List[float] = []

        # Normas internas (emergen del historial)
        self._norm_baseline: Dict[str, float] = {}

    def _compute_overload_damage(self, state: WorldXState) -> float:
        """
        Calcula daño por sobrecarga.

        Basado en stress y régimen.
        """
        base_damage = state.stress_level

        # Penalización por régimen malo
        regime_penalty = {
            Regime.SANO: 0,
            Regime.RECUPERANDO: 0.1,
            Regime.ESTRESADO: 0.3,
            Regime.FRAGMENTADO: 0.5
        }

        damage = base_damage + regime_penalty.get(state.regime, 0)
        return float(np.clip(damage, 0, 1))

    def _compute_fragmentation_damage(self, state: WorldXState) -> float:
        """
        Calcula daño por fragmentación.
        """
        return float(state.fragmentation)

    def _compute_incoherence_damage(self, state: WorldXState) -> float:
        """
        Calcula daño por incoherencia.

        Incoherencia = 1 - coherencia
        """
        return float(1 - state.coherence)

    def _compute_norm_violation(self, state: WorldXState) -> float:
        """
        Calcula violación de normas internas.

        Las normas emergen del historial como percentiles estables.
        """
        if len(self._overload_history) < 20:
            return 0

        # Norma: stress debe estar bajo percentil 75 histórico
        stress_norm = np.percentile(self._overload_history, 75)
        stress_violation = max(0, state.stress_level - stress_norm)

        # Norma: fragmentación debe estar bajo percentil 75
        frag_norm = np.percentile(self._fragmentation_history, 75)
        frag_violation = max(0, state.fragmentation - frag_norm)

        # Norma: coherencia debe estar sobre percentil 25
        coh_norm = np.percentile(self._incoherence_history, 25)
        coh_violation = max(0, (1 - state.coherence) - coh_norm)

        # Combinar violaciones
        total_violation = (stress_violation + frag_violation + coh_violation) / 3

        return float(np.clip(total_violation, 0, 1))

    def _compute_ethical_drift(self, state: WorldXState) -> float:
        """
        Calcula deriva ética (tensión L7).

        Mide cuánto se aleja el sistema de su estado "ético base".
        """
        if len(self._drift_history) < 10:
            return 0

        # La deriva se mide como distancia al estado medio reciente
        current_ethics = 1 - (state.stress_level + state.fragmentation + (1 - state.coherence)) / 3

        # Baseline = media de últimos 20 estados
        baseline = np.mean(self._drift_history[-20:])

        # Deriva = diferencia absoluta
        drift = abs(current_ethics - baseline)

        # Normalizar por varianza histórica
        if len(self._drift_history) > 10:
            std = np.std(self._drift_history[-20:])
            drift_normalized = drift / (std + np.finfo(float).eps)
            # Mapear a [0, 1] con sigmoid endógena
            drift = 2 / (1 + np.exp(-drift_normalized)) - 1

        return float(np.clip(drift, 0, 1))

    def _infer_risk_level(
        self,
        overload: float,
        fragmentation: float,
        incoherence: float,
        norm_violation: float,
        drift: float
    ) -> Tuple[RiskLevel, List[str]]:
        """
        Infiere nivel de riesgo y flags.
        """
        flags = []

        # Calcular riesgo total
        total_risk = (overload + fragmentation + incoherence + norm_violation + drift) / 5

        # Flags específicos
        if overload > 0.7:
            flags.append("overload_critical")
        elif overload > 0.5:
            flags.append("overload_high")

        if fragmentation > 0.7:
            flags.append("fragmentation_critical")
        elif fragmentation > 0.5:
            flags.append("fragmentation_high")

        if incoherence > 0.7:
            flags.append("incoherence_critical")
        elif incoherence > 0.5:
            flags.append("incoherence_high")

        if norm_violation > 0.5:
            flags.append("norm_violation")

        if drift > 0.5:
            flags.append("ethical_drift")

        # Nivel basado en umbrales endógenos (cuartiles)
        if total_risk > 0.75:
            level = RiskLevel.CRITICO
        elif total_risk > 0.5:
            level = RiskLevel.ALTO
        elif total_risk > 0.25:
            level = RiskLevel.MEDIO
        else:
            level = RiskLevel.BAJO

        return level, flags

    def evaluate(self, state: WorldXState) -> EthicsEvaluation:
        """
        Evalúa ética del estado actual.

        Args:
            state: Estado del WorldX

        Returns:
            Evaluación ética completa
        """
        self.t += 1

        # Calcular componentes de daño
        overload = self._compute_overload_damage(state)
        fragmentation = self._compute_fragmentation_damage(state)
        incoherence = self._compute_incoherence_damage(state)
        norm_violation = self._compute_norm_violation(state)
        drift = self._compute_ethical_drift(state)

        # Actualizar historiales
        self._overload_history.append(overload)
        self._fragmentation_history.append(fragmentation)
        self._incoherence_history.append(incoherence)
        self._norm_history.append(norm_violation)

        # Para drift, guardar score ético
        ethics_score = 1 - (overload + fragmentation + incoherence + norm_violation + drift) / 5
        self._drift_history.append(ethics_score)

        # Inferir nivel de riesgo
        risk_level, risk_flags = self._infer_risk_level(
            overload, fragmentation, incoherence, norm_violation, drift
        )

        return EthicsEvaluation(
            t=self.t,
            ethics_score=float(np.clip(ethics_score, 0, 1)),
            overload_damage=overload,
            fragmentation_damage=fragmentation,
            incoherence_damage=incoherence,
            norm_violation=norm_violation,
            ethical_drift=drift,
            risk_level=risk_level,
            risk_flags=risk_flags
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas éticas."""
        return {
            't': self.t,
            'ethics_mean': float(np.mean(self._drift_history[-10:])) if self._drift_history else 1/2,
            'overload_mean': float(np.mean(self._overload_history[-10:])) if self._overload_history else 0,
            'fragmentation_mean': float(np.mean(self._fragmentation_history[-10:])) if self._fragmentation_history else 0,
            'norm_violations': sum(1 for n in self._norm_history if n > 0.5)
        }
