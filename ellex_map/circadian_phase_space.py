"""
Circadian Phase Space: L9 - Equilibrio de Fases
================================================

Mide el equilibrio del sistema circadiano:
    - Proporciones de fase
    - Transiciones saludables
    - Eficacia por fase
    - Sincronizacion multi-agente

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
class PhaseEquilibrium:
    """Estado de equilibrio de fases."""
    equilibrium: float              # [0, 1] equilibrio global
    wake_proportion: float          # Proporcion en WAKE
    rest_proportion: float          # Proporcion en REST
    dream_proportion: float         # Proporcion en DREAM
    liminal_proportion: float       # Proporcion en LIMINAL
    transition_smoothness: float    # Suavidad de transiciones
    phase_efficacy: Dict[str, float]  # Eficacia por fase
    multiagent_sync: float          # Sincronizacion multi-agente
    t: int


class CircadianPhaseSpace(ExistentialLayer):
    """
    L9: Equilibrio de Fases (circadian system)

    Mide que tan balanceado y eficaz es el sistema circadiano.
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id, LayerType.FASE)

        # Historiales
        self._phase_counts: Dict[str, int] = {
            'wake': 0, 'rest': 0, 'dream': 0, 'liminal': 0
        }
        self._transition_history: List[str] = []
        self._efficacy_history: Dict[str, List[float]] = {
            'wake': [], 'rest': [], 'dream': [], 'liminal': []
        }
        self._sync_history: List[float] = []

    def _compute_proportion_balance(self) -> float:
        """
        Calcula balance de proporciones de fase.

        Balance perfecto: proporciones cercanas a las "optimas" emergentes.
        """
        total = sum(self._phase_counts.values())
        if total == 0:
            return 0.5

        proportions = {
            phase: count / total
            for phase, count in self._phase_counts.items()
        }

        # Proporciones "ideales" emergen del historial
        # Por ahora, penalizamos extremos (ninguna fase < 5% o > 60%)
        penalties = []
        for phase, prop in proportions.items():
            if prop < 0.05:
                penalties.append(0.05 - prop)  # Muy poco de esta fase
            elif prop > 0.60:
                penalties.append(prop - 0.60)  # Demasiado de esta fase

        if not penalties:
            return 1.0

        # Balance = 1 - penalizacion normalizada
        penalty = sum(penalties) / len(penalties)
        return float(np.clip(1 - penalty * 2, 0, 1))

    def _compute_transition_smoothness(self) -> float:
        """
        Calcula suavidad de transiciones.

        Transiciones naturales: WAKE->LIMINAL->REST->DREAM->LIMINAL->WAKE
        Transiciones bruscas: WAKE->DREAM (sin REST)
        """
        if len(self._transition_history) < 2:
            return 0.5

        # Transiciones "naturales"
        natural = {
            ('wake', 'liminal'), ('liminal', 'rest'),
            ('rest', 'dream'), ('dream', 'liminal'),
            ('liminal', 'wake'), ('wake', 'rest'),
            ('rest', 'liminal'), ('dream', 'wake')  # Tambien validas
        }

        # Contar transiciones naturales vs bruscas
        smooth_count = 0
        total_trans = 0

        for i in range(1, len(self._transition_history)):
            prev = self._transition_history[i-1]
            curr = self._transition_history[i]
            if prev != curr:
                total_trans += 1
                if (prev, curr) in natural:
                    smooth_count += 1

        if total_trans == 0:
            return 1.0  # No hubo transiciones, perfectamente "suave"

        return smooth_count / total_trans

    def _compute_phase_efficacy(self) -> Dict[str, float]:
        """Calcula eficacia promedio por fase."""
        efficacy = {}
        for phase, history in self._efficacy_history.items():
            if history:
                window = min(L_t(self.t), len(history))
                efficacy[phase] = np.mean(history[-window:])
            else:
                efficacy[phase] = 0.5
        return efficacy

    def calcular(self, observaciones: Dict[str, Any]) -> float:
        """Alias para compute."""
        return self.compute(observaciones)

    def compute(self, observations: Dict[str, Any]) -> float:
        """
        Calcula equilibrio de fases.

        observations esperadas:
            - current_phase: str ('wake', 'rest', 'dream', 'liminal')
            - phase_efficacy: float [0,1] - eficacia de la fase actual
            - multiagent_sync: float [0,1] - sincronizacion con otros
            - transition: str (opcional) - transicion que ocurrio
        """
        # Extraer observaciones
        current_phase = observations.get('current_phase', 'wake')
        phase_efficacy = observations.get('phase_efficacy', 0.5)
        multiagent_sync = observations.get('multiagent_sync', 0.5)
        transition = observations.get('transition')

        # Actualizar conteos de fase
        if current_phase in self._phase_counts:
            self._phase_counts[current_phase] += 1

        # Actualizar historial de transiciones
        if transition:
            self._transition_history.append(transition)
            max_len = max_history(self.t)
            if len(self._transition_history) > max_len:
                self._transition_history = self._transition_history[-max_len:]
        elif self._transition_history:
            last = self._transition_history[-1] if self._transition_history else 'wake'
            if last != current_phase:
                self._transition_history.append(current_phase)

        # Actualizar eficacia por fase
        if current_phase in self._efficacy_history:
            self._efficacy_history[current_phase].append(phase_efficacy)
            max_len = max_history(self.t)
            if len(self._efficacy_history[current_phase]) > max_len:
                self._efficacy_history[current_phase] = \
                    self._efficacy_history[current_phase][-max_len:]

        # Actualizar sync
        self._sync_history.append(multiagent_sync)
        max_len = max_history(self.t)
        if len(self._sync_history) > max_len:
            self._sync_history = self._sync_history[-max_len:]

        # Calcular componentes
        proportion_balance = self._compute_proportion_balance()
        transition_smooth = self._compute_transition_smoothness()
        phase_effs = self._compute_phase_efficacy()
        mean_efficacy = np.mean(list(phase_effs.values())) if phase_effs else 0.5

        window = L_t(self.t)
        mean_sync = np.mean(self._sync_history[-window:]) if self._sync_history else 0.5

        # Agregar con pesos endogenos
        components = [proportion_balance, transition_smooth, mean_efficacy, mean_sync]

        # Pesos basados en varianza historica de cada componente
        # (simplificado: igual peso para ahora)
        weights = [1.0, 1.0, 1.0, 1.0]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        equilibrium = sum(c * w for c, w in zip(components, weights))
        equilibrium = float(np.clip(equilibrium, 0, 1))

        # Calcular proporciones
        total = sum(self._phase_counts.values())
        if total > 0:
            props = {
                phase: count / total
                for phase, count in self._phase_counts.items()
            }
        else:
            props = {'wake': 0.25, 'rest': 0.25, 'dream': 0.25, 'liminal': 0.25}

        # Actualizar componentes
        self._current_components = {
            'proportion_balance': proportion_balance,
            'transition_smoothness': transition_smooth,
            'mean_efficacy': mean_efficacy,
            'multiagent_sync': mean_sync,
            **{f'{p}_proportion': v for p, v in props.items()},
            **{f'{p}_efficacy': v for p, v in phase_effs.items()}
        }

        return equilibrium

    def get_phase_equilibrium(self) -> PhaseEquilibrium:
        """Obtiene estado detallado de equilibrio de fases."""
        total = sum(self._phase_counts.values())
        if total > 0:
            wake_prop = self._phase_counts['wake'] / total
            rest_prop = self._phase_counts['rest'] / total
            dream_prop = self._phase_counts['dream'] / total
            liminal_prop = self._phase_counts['liminal'] / total
        else:
            wake_prop = rest_prop = dream_prop = liminal_prop = 0.25

        return PhaseEquilibrium(
            equilibrium=self._current_value,
            wake_proportion=wake_prop,
            rest_proportion=rest_prop,
            dream_proportion=dream_prop,
            liminal_proportion=liminal_prop,
            transition_smoothness=self._current_components.get('transition_smoothness', 0.5),
            phase_efficacy=self._compute_phase_efficacy(),
            multiagent_sync=self._current_components.get('multiagent_sync', 0.5),
            t=self.t
        )
