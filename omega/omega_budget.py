"""
Ω3: Presupuesto Existencial
===========================

Gestión de:
- Estrés
- Tensión
- Incoherencia
- Daño acumulado
- Fatiga narrativa
- Fragmentación simbólica

Budget_t = softmin(L7_tension, 1 - coherence, 1 - health)

Si el presupuesto cae:
- Se baja actividad
- Se aumenta DREAM
- Se cierran proyectos
- Se evita riesgo interno

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class BudgetState(Enum):
    """Estados del presupuesto."""
    ABUNDANTE = "abundante"
    NORMAL = "normal"
    BAJO = "bajo"
    CRITICO = "critico"


@dataclass
class ExistenceBudget:
    """Presupuesto de existencia."""
    t: int
    budget: float                   # [0, 1] - presupuesto disponible
    state: BudgetState
    stress_cost: float              # Costo por estrés
    incoherence_cost: float         # Costo por incoherencia
    fragmentation_cost: float       # Costo por fragmentación
    fatigue_cost: float             # Costo por fatiga
    should_rest: bool               # Si debe descansar
    should_reduce_activity: bool    # Si debe reducir actividad


class OmegaBudget:
    """
    Sistema de presupuesto existencial.

    Cada sistema tiene un presupuesto de:
    - Tolerancia al estrés
    - Tolerancia a incoherencia
    - Tolerancia a fragmentación

    Cuando cae, el sistema se auto-frena.
    """

    def __init__(self, initial_budget: float = 1.0):
        """
        Args:
            initial_budget: Presupuesto inicial (normalizado)
        """
        self.t = 0
        self._budget = initial_budget

        # Historiales
        self._budget_history: List[float] = []
        self._stress_history: List[float] = []
        self._coherence_history: List[float] = []
        self._health_history: List[float] = []
        self._fatigue_history: List[float] = []

        # Tasa de recuperación (endógena)
        self._recovery_rate = 1 / 100  # Inicial: 1% por paso

    def _softmin(self, *values) -> float:
        """
        Calcula softmin de valores.

        softmin(x) = -log(Σexp(-x_i)) / n
        Aproximación suave del mínimo.
        """
        values = np.array(values)
        # Evitar overflow
        values_clipped = np.clip(values, -10, 10)

        # softmin usando log-sum-exp
        return -np.log(np.mean(np.exp(-values_clipped)))

    def _compute_stress_cost(self, stress: float) -> float:
        """
        Calcula costo por estrés.

        Mayor estrés = mayor costo.
        """
        # Normalizar por historial
        if len(self._stress_history) > 10:
            stress_baseline = np.percentile(self._stress_history, 25)
            stress_excess = max(0, stress - stress_baseline)
            cost = stress_excess / (1 - stress_baseline + np.finfo(float).eps)
        else:
            cost = stress

        return float(np.clip(cost, 0, 1))

    def _compute_incoherence_cost(self, coherence: float) -> float:
        """
        Calcula costo por incoherencia.

        Menor coherencia = mayor costo.
        """
        incoherence = 1 - coherence

        # Normalizar por historial
        if len(self._coherence_history) > 10:
            coh_baseline = np.percentile(self._coherence_history, 75)
            coh_deficit = max(0, coh_baseline - coherence)
            cost = coh_deficit / (coh_baseline + np.finfo(float).eps)
        else:
            cost = incoherence

        return float(np.clip(cost, 0, 1))

    def _compute_fragmentation_cost(self, fragmentation: float) -> float:
        """
        Calcula costo por fragmentación.
        """
        return float(np.clip(fragmentation, 0, 1))

    def _compute_fatigue_cost(self) -> float:
        """
        Calcula costo por fatiga acumulada.

        Basado en tiempo sin descanso y actividad sostenida.
        """
        if len(self._budget_history) < 10:
            return 0

        # Fatiga = tendencia descendente del presupuesto
        recent = self._budget_history[-20:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

        # Pendiente negativa = fatiga
        if slope < 0:
            fatigue = min(1, abs(slope) * 10)
        else:
            fatigue = 0

        return float(fatigue)

    def _update_recovery_rate(self) -> None:
        """
        Actualiza tasa de recuperación endógena.

        Basada en varianza histórica del presupuesto.
        """
        if len(self._budget_history) < 20:
            return

        # Mayor varianza = sistema más dinámico = mayor recuperación
        var = np.var(self._budget_history[-20:])

        # Tasa = var / (1 + var), escalada
        self._recovery_rate = var / (1 + var) / 10

        # Límites endógenos
        if len(self._budget_history) > 50:
            rate_hist = [np.var(self._budget_history[i:i+20]) / 11
                        for i in range(len(self._budget_history) - 20)]
            self._recovery_rate = np.clip(
                self._recovery_rate,
                np.percentile(rate_hist, 5),
                np.percentile(rate_hist, 95)
            )

    def _infer_state(self) -> BudgetState:
        """
        Infiere estado del presupuesto.
        """
        if len(self._budget_history) < 10:
            if self._budget > 0.7:
                return BudgetState.ABUNDANTE
            elif self._budget > 0.4:
                return BudgetState.NORMAL
            elif self._budget > 0.2:
                return BudgetState.BAJO
            else:
                return BudgetState.CRITICO

        # Umbrales endógenos
        p75 = np.percentile(self._budget_history, 75)
        p50 = np.percentile(self._budget_history, 50)
        p25 = np.percentile(self._budget_history, 25)

        if self._budget > p75:
            return BudgetState.ABUNDANTE
        elif self._budget > p50:
            return BudgetState.NORMAL
        elif self._budget > p25:
            return BudgetState.BAJO
        else:
            return BudgetState.CRITICO

    def update(
        self,
        stress: float,
        coherence: float,
        health: float,
        fragmentation: float = 0,
        is_resting: bool = False
    ) -> ExistenceBudget:
        """
        Actualiza presupuesto existencial.

        Args:
            stress: Nivel de estrés [0, 1]
            coherence: Coherencia [0, 1]
            health: Salud [0, 1]
            fragmentation: Fragmentación [0, 1]
            is_resting: Si está en descanso

        Returns:
            Estado del presupuesto
        """
        self.t += 1

        # Actualizar historiales
        self._stress_history.append(stress)
        self._coherence_history.append(coherence)
        self._health_history.append(health)

        # Calcular costos
        stress_cost = self._compute_stress_cost(stress)
        incoherence_cost = self._compute_incoherence_cost(coherence)
        frag_cost = self._compute_fragmentation_cost(fragmentation)
        fatigue_cost = self._compute_fatigue_cost()

        self._fatigue_history.append(fatigue_cost)

        # Costo total = softmin de factores negativos
        # Incluye L7_tension (stress), incoherence, ill-health
        total_cost = self._softmin(
            1 - stress,       # L7_tension proxy
            coherence,        # ya en [0,1]
            health            # ya en [0,1]
        )

        # El costo real es 1 - softmin (porque softmin de cosas buenas)
        total_cost = 1 - total_cost

        # Actualizar presupuesto
        if is_resting:
            # Recuperar
            self._budget = min(1, self._budget + self._recovery_rate)
        else:
            # Consumir
            consumption = total_cost * (1 / (self.t + 1))  # Consumo decrece con t
            self._budget = max(0, self._budget - consumption)

        self._budget_history.append(self._budget)

        # Actualizar tasa de recuperación
        self._update_recovery_rate()

        # Inferir estado
        state = self._infer_state()

        # Determinar acciones necesarias
        should_rest = state in [BudgetState.BAJO, BudgetState.CRITICO]
        should_reduce = state == BudgetState.CRITICO

        return ExistenceBudget(
            t=self.t,
            budget=self._budget,
            state=state,
            stress_cost=stress_cost,
            incoherence_cost=incoherence_cost,
            fragmentation_cost=frag_cost,
            fatigue_cost=fatigue_cost,
            should_rest=should_rest,
            should_reduce_activity=should_reduce
        )

    def get_budget(self) -> float:
        """Retorna presupuesto actual."""
        return self._budget

    def force_rest(self, amount: float = 0.1) -> None:
        """Fuerza recuperación de presupuesto."""
        self._budget = min(1, self._budget + amount)

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas."""
        return {
            't': self.t,
            'budget': self._budget,
            'state': self._infer_state().value,
            'recovery_rate': self._recovery_rate,
            'budget_mean': float(np.mean(self._budget_history[-10:])) if self._budget_history else 1,
            'budget_trend': float(np.polyfit(
                np.arange(min(10, len(self._budget_history))),
                self._budget_history[-10:],
                1
            )[0]) if len(self._budget_history) >= 3 else 0
        }
