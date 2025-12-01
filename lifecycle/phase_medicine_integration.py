"""
Phase-Medicine Integration: Medicina Integrada con Fases Circadianas
====================================================================

El sistema medico se adapta a las fases circadianas:
    - WAKE:  Intervencion activa, tratamientos directos
    - REST:  Intervencion suave, ajustes de parametros
    - DREAM: Solo observacion, no intervencion
    - LIMINAL: Solo simbolos, no metricas

Cada fase tiene sus propias patologias tipicas:
    - WAKE:    Hiperexploracion, impulsividad, sobrecarga
    - REST:    Estancamiento, ciclos no resueltos
    - DREAM:   Drift simbolico, asociaciones fragiles
    - LIMINAL: Crisis narrativa, estados borderline

100% endogeno. Las intervenciones se adaptan al estado interno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')

from lifecycle.circadian_system import CircadianPhase
from health.clinical_cases import ClinicalCondition
from cognition.agi_dynamic_constants import L_t


class PhasePathology(Enum):
    """Patologias tipicas de cada fase."""
    # WAKE pathologies
    HYPEREXPLORATION = "hyperexploration"    # Demasiada novelty
    IMPULSIVITY = "impulsivity"              # Acciones sin deliberacion
    OVERLOAD = "overload"                    # Sobrecarga cognitiva
    GOAL_OBSESSION = "goal_obsession"        # Fijacion en metas

    # REST pathologies
    STAGNATION = "stagnation"                # Estancamiento
    UNRESOLVED_CYCLES = "unresolved_cycles"  # Ciclos emocionales no cerrados
    RUMINATION = "rumination"                # Rumiacion excesiva
    AVOIDANCE = "avoidance"                  # Evitacion de procesamiento

    # DREAM pathologies
    SYMBOLIC_DRIFT = "symbolic_drift"        # Deriva simbolica
    FRAGILE_ASSOCIATIONS = "fragile_assoc"   # Asociaciones fragiles
    NIGHTMARE_LOOPS = "nightmare_loops"      # Ciclos pesadilla
    CONSOLIDATION_FAILURE = "consol_fail"    # Fallo de consolidacion

    # LIMINAL pathologies
    NARRATIVE_CRISIS = "narrative_crisis"    # Crisis de narrativa
    BORDERLINE_STATE = "borderline"          # Estado borderline
    IDENTITY_DISSOLUTION = "id_dissolution"  # Disolucion de identidad
    TRANSITION_BLOCK = "transition_block"    # Bloqueo de transicion


# Mapeo fase -> patologias
PHASE_PATHOLOGIES = {
    CircadianPhase.WAKE: [
        PhasePathology.HYPEREXPLORATION,
        PhasePathology.IMPULSIVITY,
        PhasePathology.OVERLOAD,
        PhasePathology.GOAL_OBSESSION,
    ],
    CircadianPhase.REST: [
        PhasePathology.STAGNATION,
        PhasePathology.UNRESOLVED_CYCLES,
        PhasePathology.RUMINATION,
        PhasePathology.AVOIDANCE,
    ],
    CircadianPhase.DREAM: [
        PhasePathology.SYMBOLIC_DRIFT,
        PhasePathology.FRAGILE_ASSOCIATIONS,
        PhasePathology.NIGHTMARE_LOOPS,
        PhasePathology.CONSOLIDATION_FAILURE,
    ],
    CircadianPhase.LIMINAL: [
        PhasePathology.NARRATIVE_CRISIS,
        PhasePathology.BORDERLINE_STATE,
        PhasePathology.IDENTITY_DISSOLUTION,
        PhasePathology.TRANSITION_BLOCK,
    ],
}


class InterventionMode(Enum):
    """Modos de intervencion segun fase."""
    ACTIVE = "active"           # Intervencion directa (WAKE)
    SOFT = "soft"               # Ajustes suaves (REST)
    OBSERVE = "observe"         # Solo observar (DREAM)
    SYMBOLIC = "symbolic"       # Solo simbolos (LIMINAL)


# Modos permitidos por fase
PHASE_INTERVENTION_MODES = {
    CircadianPhase.WAKE: InterventionMode.ACTIVE,
    CircadianPhase.REST: InterventionMode.SOFT,
    CircadianPhase.DREAM: InterventionMode.OBSERVE,
    CircadianPhase.LIMINAL: InterventionMode.SYMBOLIC,
}


@dataclass
class PhaseAwarePathology:
    """Patologia detectada con contexto de fase."""
    pathology: PhasePathology
    phase: CircadianPhase
    severity: float             # [0, 1]
    indicators: Dict[str, float]  # Metricas que la detectaron
    detected_t: int


@dataclass
class PhaseAwareTreatment:
    """Tratamiento adaptado a la fase."""
    pathology: PhasePathology
    intervention_mode: InterventionMode
    actions: List[str]          # Acciones a tomar
    param_adjustments: Dict[str, float]  # Ajustes de parametros
    symbolic_messages: List[str]  # Mensajes simbolicos (para LIMINAL)
    expected_duration: int      # Duracion en pasos


@dataclass
class MedicalObservation:
    """Observacion medica durante DREAM."""
    t: int
    patient_id: str
    observations: Dict[str, Any]
    symbolic_patterns: List[str]
    recommendations: List[str]


class PhaseMedicineIntegration:
    """
    Sistema de medicina integrado con fases circadianas.

    El medico adapta sus intervenciones segun la fase:
        - WAKE: Intervenciones activas
        - REST: Ajustes suaves
        - DREAM: Solo observacion
        - LIMINAL: Solo simbolos
    """

    # Indicadores por patologia
    PATHOLOGY_INDICATORS = {
        PhasePathology.HYPEREXPLORATION: {
            'novelty_seeking': ('above', 0.8),
            'project_completion': ('below', 0.3),
        },
        PhasePathology.IMPULSIVITY: {
            'deliberation_time': ('below', 0.3),
            'action_frequency': ('above', 0.8),
        },
        PhasePathology.OVERLOAD: {
            'stress': ('above', 0.8),
            'working_memory_load': ('above', 0.9),
        },
        PhasePathology.GOAL_OBSESSION: {
            'goal_flexibility': ('below', 0.2),
            'goal_activation': ('above', 0.9),
        },
        PhasePathology.STAGNATION: {
            'activity_level': ('below', 0.2),
            'change_rate': ('below', 0.1),
        },
        PhasePathology.UNRESOLVED_CYCLES: {
            'emotional_closure': ('below', 0.3),
            'cycle_repetition': ('above', 0.7),
        },
        PhasePathology.RUMINATION: {
            'thought_repetition': ('above', 0.8),
            'new_thoughts': ('below', 0.2),
        },
        PhasePathology.AVOIDANCE: {
            'processing_depth': ('below', 0.3),
            'emotional_engagement': ('below', 0.2),
        },
        PhasePathology.SYMBOLIC_DRIFT: {
            'symbol_coherence': ('below', 0.3),
            'association_stability': ('below', 0.3),
        },
        PhasePathology.FRAGILE_ASSOCIATIONS: {
            'connection_strength': ('below', 0.2),
            'pattern_stability': ('below', 0.2),
        },
        PhasePathology.NIGHTMARE_LOOPS: {
            'dream_valence': ('below', -0.5),
            'dream_repetition': ('above', 0.7),
        },
        PhasePathology.CONSOLIDATION_FAILURE: {
            'memory_integration': ('below', 0.3),
            'learning_progress': ('below', 0.2),
        },
        PhasePathology.NARRATIVE_CRISIS: {
            'narrative_coherence': ('below', 0.3),
            'identity_stability': ('below', 0.3),
        },
        PhasePathology.BORDERLINE_STATE: {
            'emotional_volatility': ('above', 0.8),
            'self_stability': ('below', 0.3),
        },
        PhasePathology.IDENTITY_DISSOLUTION: {
            'self_coherence': ('below', 0.2),
            'drives_stability': ('below', 0.2),
        },
        PhasePathology.TRANSITION_BLOCK: {
            'phase_progress': ('below', 0.2),
            'transition_fluidity': ('below', 0.2),
        },
    }

    # Tratamientos por patologia
    TREATMENTS = {
        PhasePathology.HYPEREXPLORATION: PhaseAwareTreatment(
            pathology=PhasePathology.HYPEREXPLORATION,
            intervention_mode=InterventionMode.ACTIVE,
            actions=['reduce_novelty', 'increase_focus', 'complete_projects'],
            param_adjustments={'novelty_weight': 0.7, 'completion_bonus': 1.3},
            symbolic_messages=[],
            expected_duration=50
        ),
        PhasePathology.IMPULSIVITY: PhaseAwareTreatment(
            pathology=PhasePathology.IMPULSIVITY,
            intervention_mode=InterventionMode.ACTIVE,
            actions=['increase_deliberation', 'slow_actions'],
            param_adjustments={'deliberation_threshold': 1.3, 'action_delay': 1.2},
            symbolic_messages=[],
            expected_duration=40
        ),
        PhasePathology.STAGNATION: PhaseAwareTreatment(
            pathology=PhasePathology.STAGNATION,
            intervention_mode=InterventionMode.SOFT,
            actions=['gentle_activation', 'introduce_variety'],
            param_adjustments={'activation_boost': 1.2, 'novelty_exposure': 1.1},
            symbolic_messages=[],
            expected_duration=60
        ),
        PhasePathology.UNRESOLVED_CYCLES: PhaseAwareTreatment(
            pathology=PhasePathology.UNRESOLVED_CYCLES,
            intervention_mode=InterventionMode.SOFT,
            actions=['facilitate_closure', 'process_emotions'],
            param_adjustments={'closure_priority': 1.3, 'emotional_processing': 1.2},
            symbolic_messages=[],
            expected_duration=70
        ),
        PhasePathology.SYMBOLIC_DRIFT: PhaseAwareTreatment(
            pathology=PhasePathology.SYMBOLIC_DRIFT,
            intervention_mode=InterventionMode.OBSERVE,
            actions=['monitor_patterns', 'note_drift_direction'],
            param_adjustments={},  # No ajustes en DREAM
            symbolic_messages=[],
            expected_duration=30
        ),
        PhasePathology.NIGHTMARE_LOOPS: PhaseAwareTreatment(
            pathology=PhasePathology.NIGHTMARE_LOOPS,
            intervention_mode=InterventionMode.OBSERVE,
            actions=['record_pattern', 'prepare_intervention_for_wake'],
            param_adjustments={},
            symbolic_messages=['pattern_noted', 'awaiting_transition'],
            expected_duration=20
        ),
        PhasePathology.NARRATIVE_CRISIS: PhaseAwareTreatment(
            pathology=PhasePathology.NARRATIVE_CRISIS,
            intervention_mode=InterventionMode.SYMBOLIC,
            actions=[],  # No acciones directas
            param_adjustments={},
            symbolic_messages=[
                'tu_historia_continua',
                'hay_un_hilo_conductor',
                'el_cambio_es_parte_del_ser'
            ],
            expected_duration=40
        ),
        PhasePathology.BORDERLINE_STATE: PhaseAwareTreatment(
            pathology=PhasePathology.BORDERLINE_STATE,
            intervention_mode=InterventionMode.SYMBOLIC,
            actions=[],
            param_adjustments={},
            symbolic_messages=[
                'el_umbral_es_transito',
                'ambas_cosas_son_ciertas',
                'la_tension_es_creativa'
            ],
            expected_duration=50
        ),
    }

    def __init__(self, agent_ids: List[str]):
        """
        Inicializa sistema de medicina integrado con fases.

        Args:
            agent_ids: Lista de IDs de agentes
        """
        self.agent_ids = agent_ids

        # Estado de cada agente
        self.agent_phases: Dict[str, CircadianPhase] = {
            aid: CircadianPhase.WAKE for aid in agent_ids
        }

        # Patologias detectadas
        self.detected_pathologies: Dict[str, List[PhaseAwarePathology]] = {
            aid: [] for aid in agent_ids
        }

        # Tratamientos activos
        self.active_treatments: Dict[str, List[PhaseAwareTreatment]] = {
            aid: [] for aid in agent_ids
        }

        # Observaciones (para DREAM)
        self.observations: Dict[str, List[MedicalObservation]] = {
            aid: [] for aid in agent_ids
        }

        # Mensajes simbolicos pendientes (para LIMINAL)
        self.pending_symbolic_messages: Dict[str, List[str]] = {
            aid: [] for aid in agent_ids
        }

        self.t = 0

    def update_phase(self, agent_id: str, phase: CircadianPhase):
        """Actualiza fase de un agente."""
        if agent_id in self.agent_phases:
            self.agent_phases[agent_id] = phase

    def detect_pathologies(
        self,
        agent_id: str,
        metrics: Dict[str, float]
    ) -> List[PhaseAwarePathology]:
        """
        Detecta patologias en un agente.

        Solo detecta patologias apropiadas para la fase actual.

        Args:
            agent_id: ID del agente
            metrics: Metricas actuales del agente

        Returns:
            Lista de patologias detectadas
        """
        if agent_id not in self.agent_phases:
            return []

        phase = self.agent_phases[agent_id]
        relevant_pathologies = PHASE_PATHOLOGIES[phase]

        detected = []

        for pathology in relevant_pathologies:
            indicators = self.PATHOLOGY_INDICATORS.get(pathology, {})
            matches = 0
            indicator_values = {}

            for indicator, (direction, threshold) in indicators.items():
                value = metrics.get(indicator, 0.5)
                indicator_values[indicator] = value

                if direction == 'above' and value > threshold:
                    matches += 1
                elif direction == 'below' and value < threshold:
                    matches += 1

            # Detectar si mas de la mitad de indicadores coinciden
            if indicators and matches >= len(indicators) / 2:
                severity = matches / len(indicators)
                pathology_obj = PhaseAwarePathology(
                    pathology=pathology,
                    phase=phase,
                    severity=severity,
                    indicators=indicator_values,
                    detected_t=self.t
                )
                detected.append(pathology_obj)
                self.detected_pathologies[agent_id].append(pathology_obj)

        return detected

    def can_intervene(self, agent_id: str) -> Tuple[bool, InterventionMode]:
        """
        Verifica si se puede intervenir en un agente.

        Args:
            agent_id: ID del agente

        Returns:
            (puede_intervenir, modo_permitido)
        """
        phase = self.agent_phases.get(agent_id, CircadianPhase.WAKE)
        mode = PHASE_INTERVENTION_MODES[phase]

        # DREAM: solo observar
        if phase == CircadianPhase.DREAM:
            return (False, mode)

        return (True, mode)

    def get_treatment(
        self,
        agent_id: str,
        pathology: PhasePathology
    ) -> Optional[PhaseAwareTreatment]:
        """
        Obtiene tratamiento para una patologia.

        El tratamiento se adapta a la fase actual.

        Args:
            agent_id: ID del agente
            pathology: Patologia a tratar

        Returns:
            Tratamiento o None si no se puede intervenir
        """
        can_intervene, mode = self.can_intervene(agent_id)

        # Obtener tratamiento base
        treatment = self.TREATMENTS.get(pathology)
        if treatment is None:
            return None

        # Verificar compatibilidad de modo
        if treatment.intervention_mode == InterventionMode.ACTIVE and mode == InterventionMode.SOFT:
            # Adaptar a modo soft
            return PhaseAwareTreatment(
                pathology=pathology,
                intervention_mode=InterventionMode.SOFT,
                actions=['gentle_' + a for a in treatment.actions[:1]],
                param_adjustments={k: 1 + (v - 1) * 0.5 for k, v in treatment.param_adjustments.items()},
                symbolic_messages=treatment.symbolic_messages,
                expected_duration=int(treatment.expected_duration * 1.5)
            )

        if treatment.intervention_mode == InterventionMode.ACTIVE and mode == InterventionMode.SYMBOLIC:
            # Convertir a simbolico
            return PhaseAwareTreatment(
                pathology=pathology,
                intervention_mode=InterventionMode.SYMBOLIC,
                actions=[],
                param_adjustments={},
                symbolic_messages=[f'simbolo_para_{a}' for a in treatment.actions[:2]],
                expected_duration=treatment.expected_duration
            )

        return treatment

    def apply_treatment(
        self,
        agent_id: str,
        treatment: PhaseAwareTreatment,
        agent_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aplica tratamiento a un agente.

        Args:
            agent_id: ID del agente
            treatment: Tratamiento a aplicar
            agent_state: Estado actual del agente

        Returns:
            Estado modificado
        """
        modified_state = agent_state.copy()

        if treatment.intervention_mode == InterventionMode.ACTIVE:
            # Aplicar ajustes directos
            for param, factor in treatment.param_adjustments.items():
                if param in modified_state:
                    modified_state[param] *= factor

        elif treatment.intervention_mode == InterventionMode.SOFT:
            # Aplicar ajustes suaves
            for param, factor in treatment.param_adjustments.items():
                if param in modified_state:
                    # Ajuste mas gradual
                    current = modified_state[param]
                    target = current * factor
                    modified_state[param] = current + 0.3 * (target - current)

        elif treatment.intervention_mode == InterventionMode.SYMBOLIC:
            # Solo enviar mensajes simbolicos
            self.pending_symbolic_messages[agent_id].extend(
                treatment.symbolic_messages
            )

        elif treatment.intervention_mode == InterventionMode.OBSERVE:
            # Solo registrar observacion
            observation = MedicalObservation(
                t=self.t,
                patient_id=agent_id,
                observations=agent_state,
                symbolic_patterns=[treatment.pathology.value],
                recommendations=treatment.actions
            )
            self.observations[agent_id].append(observation)

        # Registrar tratamiento activo
        self.active_treatments[agent_id].append(treatment)

        return modified_state

    def get_pending_messages(self, agent_id: str) -> List[str]:
        """Obtiene mensajes simbolicos pendientes."""
        messages = self.pending_symbolic_messages.get(agent_id, []).copy()
        self.pending_symbolic_messages[agent_id] = []
        return messages

    def get_observations(self, agent_id: str) -> List[MedicalObservation]:
        """Obtiene observaciones de DREAM."""
        return self.observations.get(agent_id, [])

    def step(self, phases: Dict[str, CircadianPhase]):
        """
        Ejecuta un paso del sistema.

        Args:
            phases: Fases actuales de cada agente
        """
        self.t += 1

        for agent_id, phase in phases.items():
            self.update_phase(agent_id, phase)

        # Limpiar tratamientos expirados
        for agent_id in self.agent_ids:
            self.active_treatments[agent_id] = [
                t for t in self.active_treatments[agent_id]
                if self.t - t.expected_duration < 50  # Mantener recientes
            ]

    def get_phase_medical_summary(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Obtiene resumen medico adaptado a la fase.

        Args:
            agent_id: ID del agente

        Returns:
            Resumen medico
        """
        phase = self.agent_phases.get(agent_id, CircadianPhase.WAKE)
        mode = PHASE_INTERVENTION_MODES[phase]

        recent_pathologies = [
            p for p in self.detected_pathologies.get(agent_id, [])
            if self.t - p.detected_t < 100
        ]

        return {
            'agent_id': agent_id,
            'phase': phase.value,
            'intervention_mode': mode.value,
            'can_intervene': mode != InterventionMode.OBSERVE,
            'recent_pathologies': [p.pathology.value for p in recent_pathologies],
            'active_treatments': len(self.active_treatments.get(agent_id, [])),
            'pending_messages': len(self.pending_symbolic_messages.get(agent_id, [])),
            'observations_recorded': len(self.observations.get(agent_id, []))
        }

    def get_statistics(self) -> Dict:
        """Estadisticas del sistema."""
        total_pathologies = sum(
            len(p) for p in self.detected_pathologies.values()
        )
        total_treatments = sum(
            len(t) for t in self.active_treatments.values()
        )
        total_observations = sum(
            len(o) for o in self.observations.values()
        )

        return {
            't': self.t,
            'agents': len(self.agent_ids),
            'total_pathologies_detected': total_pathologies,
            'total_treatments_applied': total_treatments,
            'total_observations': total_observations,
            'agent_summaries': {
                aid: self.get_phase_medical_summary(aid)
                for aid in self.agent_ids
            }
        }


def test_phase_medicine_integration():
    """Test de integracion medicina-fases."""
    print("=" * 70)
    print("TEST: PHASE-MEDICINE INTEGRATION")
    print("=" * 70)

    np.random.seed(42)

    agents = ['NEO', 'EVA', 'ALEX']
    system = PhaseMedicineIntegration(agents)

    phases = [
        CircadianPhase.WAKE,
        CircadianPhase.REST,
        CircadianPhase.DREAM,
        CircadianPhase.LIMINAL
    ]

    print(f"\nAgentes: {agents}")
    print("\nSimulando 100 pasos con deteccion de patologias...")

    for t in range(100):
        phase_idx = (t // 25) % 4
        current_phases = {aid: phases[phase_idx] for aid in agents}

        system.step(current_phases)

        # Generar metricas con problemas
        for agent_id in agents:
            # NEO: hiperexploracion en WAKE
            if agent_id == 'NEO' and phases[phase_idx] == CircadianPhase.WAKE:
                metrics = {
                    'novelty_seeking': 0.9,
                    'project_completion': 0.2,
                    'deliberation_time': 0.4,
                }
            # EVA: estancamiento en REST
            elif agent_id == 'EVA' and phases[phase_idx] == CircadianPhase.REST:
                metrics = {
                    'activity_level': 0.1,
                    'change_rate': 0.05,
                    'emotional_closure': 0.5,
                }
            # ALEX: crisis narrativa en LIMINAL
            elif agent_id == 'ALEX' and phases[phase_idx] == CircadianPhase.LIMINAL:
                metrics = {
                    'narrative_coherence': 0.2,
                    'identity_stability': 0.25,
                    'emotional_volatility': 0.5,
                }
            else:
                metrics = {
                    'novelty_seeking': 0.5,
                    'activity_level': 0.5,
                    'narrative_coherence': 0.6,
                }

            # Detectar patologias
            pathologies = system.detect_pathologies(agent_id, metrics)

            # Aplicar tratamientos
            for pathology in pathologies:
                treatment = system.get_treatment(agent_id, pathology.pathology)
                if treatment:
                    system.apply_treatment(agent_id, treatment, metrics)

        if t % 25 == 24:
            phase = phases[phase_idx]
            print(f"\n  t={t+1}, Fase: {phase.value}")
            for agent_id in agents:
                summary = system.get_phase_medical_summary(agent_id)
                print(f"    {agent_id}: mode={summary['intervention_mode']}, "
                      f"patho={len(summary['recent_pathologies'])}, "
                      f"treat={summary['active_treatments']}")

    # Estadisticas finales
    print("\n" + "=" * 70)
    print("ESTADISTICAS FINALES")
    print("=" * 70)

    stats = system.get_statistics()
    print(f"\n  Total patologias detectadas: {stats['total_pathologies_detected']}")
    print(f"  Total tratamientos aplicados: {stats['total_treatments_applied']}")
    print(f"  Total observaciones (DREAM): {stats['total_observations']}")

    print("\n  Patologias por agente:")
    for agent_id in agents:
        pathologies = system.detected_pathologies[agent_id]
        pathology_names = list(set([p.pathology.value for p in pathologies]))
        print(f"    {agent_id}: {pathology_names}")

    # Mensajes simbolicos
    print("\n  Mensajes simbolicos pendientes:")
    for agent_id in agents:
        messages = system.pending_symbolic_messages[agent_id]
        if messages:
            print(f"    {agent_id}: {messages}")

    return system


if __name__ == "__main__":
    test_phase_medicine_integration()
