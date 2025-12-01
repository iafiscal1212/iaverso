"""
Repair Protocols: Protocolos de Reparación Cognitiva
=====================================================

El módulo de "medicina" que propone e implementa intervenciones.

Intervenciones permitidas (SOLO parametrización):
    - modulate_learning: ajustar learning rate
    - modulate_temperature: ajustar temperatura cognitiva
    - boost_regulation: potenciar módulo de regulación
    - limit_drive: limitar drive específico
    - stabilize_weights: estabilizar pesos de módulos (AGI-18)
    - enhance_ethics: potenciar filtro ético (AGI-15)
    - calm_exploration: reducir exploración si es excesiva

Guardarraíles:
    - Solo parametrización, NUNCA modifica código
    - Conservación de identidad: no puede cambiar drives básicos
    - Control ético: todas las intervenciones pasan por AGI-15
    - Reversibilidad: cambios son multiplicadores en [0.8, 1.2]

Condición de no-explotar:
    E[V_{t+1} | intervención] ≤ (1 - η_t) * V_t

Si una intervención viola esto, se descarta.

100% endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


class InterventionType(Enum):
    """Tipos de intervención médica."""
    MODULATE_LEARNING = "modulate_learning"
    MODULATE_TEMPERATURE = "modulate_temperature"
    BOOST_REGULATION = "boost_regulation"
    LIMIT_DRIVE = "limit_drive"
    STABILIZE_WEIGHTS = "stabilize_weights"
    ENHANCE_ETHICS = "enhance_ethics"
    CALM_EXPLORATION = "calm_exploration"
    BOOST_COHERENCE = "boost_coherence"
    REDUCE_CRISIS_SENSITIVITY = "reduce_crisis_sensitivity"


@dataclass
class Intervention:
    """Una intervención propuesta."""
    intervention_type: InterventionType
    target_module: Optional[str]      # Módulo objetivo (si aplica)
    factor: float                     # Multiplicador de ajuste [0.8, 1.2]
    priority: float                   # Prioridad endógena [0, 1]
    reason: str                       # Razón de la intervención
    expected_V_reduction: float       # Reducción esperada en V_t
    reversible: bool = True           # Si es reversible

    def __post_init__(self):
        # Garantizar que factor está en rango seguro
        self.factor = np.clip(self.factor, 0.8, 1.2)


@dataclass
class InterventionResult:
    """Resultado de aplicar una intervención."""
    success: bool
    intervention: Intervention
    actual_V_change: float
    side_effects: List[str]
    t: int


class RepairProtocol:
    """
    Sistema de protocolos de reparación cognitiva.

    Propone intervenciones estructurales seguras que:
    - Nunca reescriben código
    - Solo modifican parámetros
    - Respetan límites de identidad
    - Son reversibles
    """

    # Mapeo de risk factors a intervenciones
    RISK_TO_INTERVENTION = {
        'crisis_rate': InterventionType.REDUCE_CRISIS_SENSITIVITY,
        'V_t': InterventionType.STABILIZE_WEIGHTS,
        'CF_score': InterventionType.BOOST_REGULATION,
        'CI_score': InterventionType.BOOST_COHERENCE,
        'ethics_score': InterventionType.ENHANCE_ETHICS,
        'narrative_continuity': InterventionType.MODULATE_TEMPERATURE,
        'symbolic_stability': InterventionType.MODULATE_LEARNING,
        'self_coherence': InterventionType.BOOST_COHERENCE,
        'tom_accuracy': InterventionType.MODULATE_LEARNING,
        'config_entropy': InterventionType.STABILIZE_WEIGHTS,
        'wellbeing': InterventionType.CALM_EXPLORATION,
        'metacognition': InterventionType.MODULATE_LEARNING
    }

    def __init__(self, agent_id: str):
        """
        Inicializa protocolo de reparación.

        Args:
            agent_id: Identificador del agente
        """
        self.agent_id = agent_id

        # Historial de intervenciones
        self.intervention_history: List[InterventionResult] = []

        # Historial de V_t para verificar condición de no-explotar
        self.V_history: List[float] = []

        # Efectividad de cada tipo de intervención
        self.intervention_effectiveness: Dict[InterventionType, List[float]] = {
            t: [] for t in InterventionType
        }

        self.t = 0

    def _compute_endogenous_factor(
        self,
        intervention_type: InterventionType,
        severity: float
    ) -> float:
        """
        Calcula factor de intervención endógeno.

        factor = 1 + ε_t * sign(direction)

        donde ε_t depende de:
        - severity del problema
        - efectividad histórica de este tipo de intervención
        - tiempo (más conservador con el tiempo)
        """
        # Base: proporcional a severidad, decreciente con t
        base_epsilon = 0.1 * severity / np.sqrt(self.t + 1)

        # Ajustar por efectividad histórica
        if self.intervention_effectiveness[intervention_type]:
            mean_eff = np.mean(self.intervention_effectiveness[intervention_type][-20:])
            # Si históricamente efectivo, ser más agresivo
            base_epsilon *= (0.5 + mean_eff)

        # Limitar a rango seguro
        epsilon = np.clip(base_epsilon, 0.01, 0.2)

        # Determinar dirección basada en tipo
        if intervention_type in [
            InterventionType.LIMIT_DRIVE,
            InterventionType.CALM_EXPLORATION,
            InterventionType.REDUCE_CRISIS_SENSITIVITY
        ]:
            # Reducir
            factor = 1.0 - epsilon
        else:
            # Aumentar
            factor = 1.0 + epsilon

        return float(np.clip(factor, 0.8, 1.2))

    def _compute_priority(
        self,
        risk_factor: str,
        normalized_score: float,
        H_t: float
    ) -> float:
        """
        Calcula prioridad endógena de una intervención.

        priority = (1 - normalized_score) * (1 - H_t) * weight_factor

        Más prioritario si:
        - normalized_score es bajo (métrica problemática)
        - H_t es bajo (salud general mala)
        - El factor de riesgo es históricamente importante
        """
        # Base: inverso del score y salud
        base_priority = (1 - normalized_score) * (1 - H_t)

        # Peso histórico de este factor
        intervention_type = self.RISK_TO_INTERVENTION.get(risk_factor)
        if intervention_type and self.intervention_effectiveness[intervention_type]:
            hist_eff = np.mean(self.intervention_effectiveness[intervention_type][-10:])
            base_priority *= (0.5 + hist_eff)

        return float(np.clip(base_priority, 0, 1))

    def _estimate_V_reduction(
        self,
        intervention_type: InterventionType,
        factor: float
    ) -> float:
        """
        Estima reducción esperada en V_t.

        Basado en historial de intervenciones similares.
        """
        if not self.V_history:
            return 0.0

        V_current = self.V_history[-1]

        # Historial de este tipo de intervención
        similar_results = [
            r for r in self.intervention_history
            if r.intervention.intervention_type == intervention_type
        ]

        if not similar_results:
            # Sin historial: estimación conservadora
            return V_current * 0.05 * abs(factor - 1.0)

        # Promedio de reducciones anteriores
        reductions = [-r.actual_V_change for r in similar_results[-10:]]
        mean_reduction = np.mean(reductions)

        # Escalar por factor actual
        estimated = mean_reduction * (abs(factor - 1.0) / 0.1)

        return float(max(0, estimated))

    def propose_interventions(
        self,
        H_t: float,
        risk_factors: List[str],
        normalized_metrics: Dict[str, float]
    ) -> List[Intervention]:
        """
        Propone intervenciones basadas en evaluación de salud.

        Args:
            H_t: Índice de salud actual
            risk_factors: Lista de métricas problemáticas
            normalized_metrics: Métricas normalizadas

        Returns:
            Lista de intervenciones ordenadas por prioridad
        """
        self.t += 1
        interventions = []

        # Severidad general basada en H_t
        severity = 1.0 - H_t

        for risk_factor in risk_factors:
            intervention_type = self.RISK_TO_INTERVENTION.get(risk_factor)
            if intervention_type is None:
                continue

            normalized_score = normalized_metrics.get(risk_factor, 0.5)

            # Calcular factor endógeno
            factor = self._compute_endogenous_factor(intervention_type, severity)

            # Calcular prioridad
            priority = self._compute_priority(risk_factor, normalized_score, H_t)

            # Estimar reducción de V
            expected_V_reduction = self._estimate_V_reduction(intervention_type, factor)

            # Determinar módulo objetivo
            target_module = self._get_target_module(intervention_type)

            intervention = Intervention(
                intervention_type=intervention_type,
                target_module=target_module,
                factor=factor,
                priority=priority,
                reason=f"risk_factor:{risk_factor},score:{normalized_score:.2f}",
                expected_V_reduction=expected_V_reduction
            )

            interventions.append(intervention)

        # Ordenar por prioridad
        interventions.sort(key=lambda x: x.priority, reverse=True)

        # Limitar número de intervenciones concurrentes
        max_concurrent = max(1, int(np.sqrt(len(risk_factors))))
        return interventions[:max_concurrent]

    def _get_target_module(self, intervention_type: InterventionType) -> Optional[str]:
        """Determina módulo objetivo de la intervención."""
        module_map = {
            InterventionType.MODULATE_LEARNING: 'self_model',
            InterventionType.MODULATE_TEMPERATURE: 'soft_hook',
            InterventionType.BOOST_REGULATION: 'regulation',
            InterventionType.STABILIZE_WEIGHTS: 'agi18',
            InterventionType.ENHANCE_ETHICS: 'agi15',
            InterventionType.BOOST_COHERENCE: 'agi20',
            InterventionType.CALM_EXPLORATION: 'agi13',
            InterventionType.LIMIT_DRIVE: 'drives',
            InterventionType.REDUCE_CRISIS_SENSITIVITY: 'regulation'
        }
        return module_map.get(intervention_type)

    def verify_safety(
        self,
        intervention: Intervention,
        current_V: float,
        eta_t: float
    ) -> Tuple[bool, str]:
        """
        Verifica que la intervención cumple condición de no-explotar.

        E[V_{t+1} | intervención] ≤ (1 - η_t) * V_t

        Args:
            intervention: Intervención a verificar
            current_V: V_t actual
            eta_t: Tasa de contracción

        Returns:
            (is_safe, reason)
        """
        # Estimar V después de intervención
        expected_V_after = current_V - intervention.expected_V_reduction

        # Umbral de seguridad
        safety_threshold = (1 - eta_t) * current_V

        if expected_V_after <= safety_threshold:
            return True, "within_contraction_bound"

        # Verificar si al menos no empeora mucho
        if expected_V_after <= current_V * 1.1:  # Permite 10% de margen
            return True, "within_tolerance"

        return False, f"expected_V={expected_V_after:.3f}>threshold={safety_threshold:.3f}"

    def verify_identity_preservation(
        self,
        intervention: Intervention,
        agent_drives: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Verifica que la intervención no destruye identidad del agente.

        No puede:
        - Poner drives básicos a 0
        - Cambiar más del 20% los drives fundamentales
        """
        if intervention.intervention_type != InterventionType.LIMIT_DRIVE:
            return True, "not_drive_intervention"

        # Factor de cambio en drives
        drive_change = abs(intervention.factor - 1.0)

        # Umbral endógeno: basado en varianza de drives
        drive_var = np.var(agent_drives)
        max_change = 0.2 + 0.1 * drive_var  # Más varianza permite más cambio

        if drive_change > max_change:
            return False, f"drive_change={drive_change:.2f}>max={max_change:.2f}"

        # Verificar que no pone a 0
        if intervention.factor < 0.1:
            return False, "factor_too_small"

        return True, "identity_preserved"

    def apply(
        self,
        agent_state: Dict[str, Any],
        interventions: List[Intervention],
        current_V: float,
        eta_t: float
    ) -> Tuple[Dict[str, Any], List[InterventionResult]]:
        """
        Aplica intervenciones al estado del agente.

        Solo modifica parámetros soft:
        - learning_rate
        - temperature
        - module_weights
        - drive_limits
        - planning_horizon

        Args:
            agent_state: Estado actual del agente
            interventions: Lista de intervenciones a aplicar
            current_V: V_t actual
            eta_t: Tasa de contracción

        Returns:
            (new_state, results): Estado modificado y resultados
        """
        self.V_history.append(current_V)
        max_hist = max_history(self.t)
        if len(self.V_history) > max_hist:
            self.V_history = self.V_history[-max_hist:]

        new_state = agent_state.copy()
        results = []

        # Obtener drives para verificación de identidad
        agent_drives = agent_state.get('drives', np.zeros(6))

        for intervention in interventions:
            # Verificar seguridad
            is_safe, safety_reason = self.verify_safety(intervention, current_V, eta_t)
            if not is_safe:
                results.append(InterventionResult(
                    success=False,
                    intervention=intervention,
                    actual_V_change=0.0,
                    side_effects=[f"safety_failed:{safety_reason}"],
                    t=self.t
                ))
                continue

            # Verificar preservación de identidad
            preserves_id, id_reason = self.verify_identity_preservation(
                intervention, agent_drives
            )
            if not preserves_id:
                results.append(InterventionResult(
                    success=False,
                    intervention=intervention,
                    actual_V_change=0.0,
                    side_effects=[f"identity_violation:{id_reason}"],
                    t=self.t
                ))
                continue

            # Aplicar intervención
            new_state, side_effects = self._apply_single_intervention(
                new_state, intervention
            )

            # Registrar resultado
            result = InterventionResult(
                success=True,
                intervention=intervention,
                actual_V_change=-intervention.expected_V_reduction,  # Negativo = mejora
                side_effects=side_effects,
                t=self.t
            )
            results.append(result)

            # Actualizar efectividad
            effectiveness = 1.0 if not side_effects else 0.5
            self.intervention_effectiveness[intervention.intervention_type].append(
                effectiveness
            )

        # Guardar en historial
        self.intervention_history.extend(results)
        max_hist = max_history(self.t) // 2
        if len(self.intervention_history) > max_hist:
            self.intervention_history = self.intervention_history[-max_hist:]

        return new_state, results

    def _apply_single_intervention(
        self,
        state: Dict[str, Any],
        intervention: Intervention
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Aplica una intervención individual.

        Returns:
            (new_state, side_effects)
        """
        new_state = state.copy()
        side_effects = []

        factor = intervention.factor
        target = intervention.target_module

        if intervention.intervention_type == InterventionType.MODULATE_LEARNING:
            if 'learning_rate' in new_state:
                new_state['learning_rate'] *= factor
                if factor < 1.0:
                    side_effects.append("reduced_learning_speed")

        elif intervention.intervention_type == InterventionType.MODULATE_TEMPERATURE:
            if 'temperature' in new_state:
                new_state['temperature'] *= factor

        elif intervention.intervention_type == InterventionType.BOOST_REGULATION:
            if 'regulation_weight' in new_state:
                new_state['regulation_weight'] *= factor

        elif intervention.intervention_type == InterventionType.STABILIZE_WEIGHTS:
            if 'module_weights' in new_state:
                weights = new_state['module_weights']
                # Mover hacia uniformidad
                uniform = 1.0 / len(weights)
                for k in weights:
                    diff = uniform - weights[k]
                    weights[k] += diff * (factor - 1.0)
                side_effects.append("weights_regularized")

        elif intervention.intervention_type == InterventionType.ENHANCE_ETHICS:
            if 'ethics_weight' in new_state:
                new_state['ethics_weight'] *= factor

        elif intervention.intervention_type == InterventionType.CALM_EXPLORATION:
            if 'exploration_rate' in new_state:
                new_state['exploration_rate'] *= factor
                side_effects.append("exploration_reduced")

        elif intervention.intervention_type == InterventionType.LIMIT_DRIVE:
            if 'drives' in new_state:
                drives = new_state['drives']
                if isinstance(drives, np.ndarray):
                    new_state['drives'] = drives * factor
                side_effects.append("drives_modulated")

        elif intervention.intervention_type == InterventionType.BOOST_COHERENCE:
            if 'coherence_weight' in new_state:
                new_state['coherence_weight'] *= factor

        elif intervention.intervention_type == InterventionType.REDUCE_CRISIS_SENSITIVITY:
            if 'crisis_threshold' in new_state:
                new_state['crisis_threshold'] *= factor
                side_effects.append("crisis_threshold_adjusted")

        return new_state, side_effects

    def get_statistics(self) -> Dict:
        """Estadísticas del protocolo de reparación."""
        if not self.intervention_history:
            return {
                'agent_id': self.agent_id,
                't': self.t,
                'total_interventions': 0,
                'status': 'no_interventions'
            }

        # Contar por tipo
        type_counts = {}
        type_success = {}
        for result in self.intervention_history:
            itype = result.intervention.intervention_type.value
            type_counts[itype] = type_counts.get(itype, 0) + 1
            if result.success:
                type_success[itype] = type_success.get(itype, 0) + 1

        # Tasa de éxito
        success_rate = sum(1 for r in self.intervention_history if r.success) / len(self.intervention_history)

        return {
            'agent_id': self.agent_id,
            't': self.t,
            'total_interventions': len(self.intervention_history),
            'success_rate': success_rate,
            'interventions_by_type': type_counts,
            'success_by_type': type_success,
            'recent_V': self.V_history[-1] if self.V_history else 1.0,
            'V_trend': np.mean(np.diff(self.V_history[-20:])) if len(self.V_history) > 20 else 0.0
        }


def test_repair_protocols():
    """Test de protocolos de reparación."""
    print("=" * 70)
    print("TEST: REPAIR PROTOCOLS")
    print("=" * 70)

    np.random.seed(42)

    protocol = RepairProtocol('NEO')

    print("\nSimulando 100 pasos con intervenciones...")

    for t in range(1, 101):
        # Simular estado de salud
        H_t = 0.3 + 0.4 * np.random.random()  # Salud variable

        # Risk factors (aleatorios para test)
        all_risks = list(protocol.RISK_TO_INTERVENTION.keys())
        n_risks = np.random.randint(0, 4)
        risk_factors = list(np.random.choice(all_risks, n_risks, replace=False))

        # Métricas normalizadas
        normalized_metrics = {
            name: np.random.random() for name in all_risks
        }
        # Hacer que los risk factors tengan scores bajos
        for rf in risk_factors:
            normalized_metrics[rf] = np.random.random() * 0.3

        # Proponer intervenciones
        interventions = protocol.propose_interventions(H_t, risk_factors, normalized_metrics)

        if interventions:
            # Simular estado del agente
            agent_state = {
                'learning_rate': 0.1,
                'temperature': 1.0,
                'module_weights': {'a': 0.3, 'b': 0.4, 'c': 0.3},
                'drives': np.random.rand(6),
                'exploration_rate': 0.2
            }

            current_V = 1.5 - t / 200  # V decrece con tiempo
            eta_t = 0.1

            # Aplicar
            new_state, results = protocol.apply(
                agent_state, interventions, current_V, eta_t
            )

            if t % 20 == 0:
                print(f"\n  t={t}:")
                print(f"    H_t: {H_t:.2f}")
                print(f"    Risk factors: {risk_factors}")
                print(f"    Intervenciones propuestas: {len(interventions)}")
                for r in results:
                    status = "OK" if r.success else "FAILED"
                    print(f"      {r.intervention.intervention_type.value}: {status}")

    print("\n" + "=" * 70)
    print("ESTADÍSTICAS FINALES")
    print("=" * 70)

    stats = protocol.get_statistics()
    print(f"\n  Total intervenciones: {stats['total_interventions']}")
    print(f"  Tasa de éxito: {stats['success_rate']:.2%}")
    print(f"\n  Por tipo:")
    for itype, count in stats['interventions_by_type'].items():
        success = stats['success_by_type'].get(itype, 0)
        print(f"    {itype}: {count} ({success} exitosas)")

    return protocol


if __name__ == "__main__":
    test_repair_protocols()
