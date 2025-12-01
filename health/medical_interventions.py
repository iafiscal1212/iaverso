"""
Medical Interventions: Intervenciones Simbolicas del Medico
============================================================

El medico NO toca directamente a otros agentes.
Emite propuestas simbolicas que cada agente decide si aplica.

Flujo:
    1. Medico detecta que agente X esta "enfermo"
    2. Medico emite propuesta simbolica de tratamiento
    3. Agente X interpreta la propuesta con su AGI-15 (etica)
    4. Agente X decide si aplica el tratamiento
    5. Si aplica, lo hace via soft hooks y AGI-18

La fuerza del medico viene de:
    - Historial de exito de sus propuestas
    - Trust acumulado
    - Normas emergentes del grupo

100% endogeno. El medico no tiene "API privilegiada".
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


class TreatmentType(Enum):
    """Tipos de tratamiento simbolico."""
    STABILIZE = "stabilize"           # Reducir volatilidad
    CALM_EXPLORATION = "calm_explore" # Reducir exploracion
    BOOST_ETHICS = "boost_ethics"     # Reforzar filtro etico
    REGULATE_DRIVES = "reg_drives"    # Regular drives
    ENHANCE_TOM = "enhance_tom"       # Mejorar ToM
    REST = "rest"                     # Reducir actividad general


@dataclass
class TreatmentProposal:
    """Propuesta de tratamiento del medico."""
    doctor_id: str
    patient_id: str
    treatment_type: TreatmentType
    intensity: float           # Intensidad sugerida [0,1]
    urgency: float             # Urgencia [0,1]
    reasoning: str             # Razonamiento simbolico
    t: int

    # Simbolos asociados (para capa simbolica)
    symbols: List[str] = field(default_factory=list)


@dataclass
class TreatmentResponse:
    """Respuesta del paciente a una propuesta."""
    patient_id: str
    proposal: TreatmentProposal
    accepted: bool
    acceptance_confidence: float  # Que tan seguro estaba
    applied_intensity: float      # Intensidad realmente aplicada
    ethical_score: float          # Evaluacion etica de la propuesta
    reason: str


@dataclass
class TreatmentOutcome:
    """Resultado de un tratamiento aplicado."""
    proposal: TreatmentProposal
    response: TreatmentResponse
    health_before: float
    health_after: float
    success: float  # Mejoria relativa


class DoctorProposalSystem:
    """
    Sistema de propuestas del medico.

    El medico genera propuestas simbolicas basadas en:
    - Su observacion del paciente
    - Historial de tratamientos exitosos
    - Normas del grupo
    """

    # Mapeo de sintomas a tratamientos
    SYMPTOM_TREATMENTS = {
        'high_crisis': TreatmentType.STABILIZE,
        'high_exploration': TreatmentType.CALM_EXPLORATION,
        'low_ethics': TreatmentType.BOOST_ETHICS,
        'unstable_drives': TreatmentType.REGULATE_DRIVES,
        'low_tom': TreatmentType.ENHANCE_TOM,
        'general_stress': TreatmentType.REST
    }

    def __init__(self, doctor_id: str):
        """
        Inicializa sistema de propuestas.

        Args:
            doctor_id: ID del agente medico
        """
        self.doctor_id = doctor_id

        # Historial de propuestas y outcomes
        self.proposal_history: List[TreatmentProposal] = []
        self.outcome_history: List[TreatmentOutcome] = []

        # Efectividad aprendida por tipo de tratamiento
        self.treatment_effectiveness: Dict[TreatmentType, List[float]] = {
            t: [] for t in TreatmentType
        }

        self.t = 0

    def _diagnose(
        self,
        patient_metrics: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        Diagnostica sintomas basado en metricas observadas.

        Returns:
            Lista de (sintoma, severidad)
        """
        symptoms = []

        # Crisis alta
        crisis_rate = patient_metrics.get('crisis_rate', 0)
        if crisis_rate > 0.3:
            symptoms.append(('high_crisis', crisis_rate))

        # Exploracion excesiva
        exploration = patient_metrics.get('exploration', 0.5)
        if exploration > 0.7:
            symptoms.append(('high_exploration', exploration))

        # Etica baja
        ethics = patient_metrics.get('ethics', 0.5)
        if ethics < 0.4:
            symptoms.append(('low_ethics', 1 - ethics))

        # Drives inestables
        drive_var = patient_metrics.get('drive_variance', 0)
        if drive_var > 0.3:
            symptoms.append(('unstable_drives', drive_var))

        # ToM bajo
        tom = patient_metrics.get('tom', 0.5)
        if tom < 0.3:
            symptoms.append(('low_tom', 1 - tom))

        # Stress general (V alto)
        V_t = patient_metrics.get('V_t', 1)
        if V_t > 1.5:
            symptoms.append(('general_stress', V_t / 3))

        return symptoms

    def _compute_intensity(
        self,
        treatment_type: TreatmentType,
        severity: float
    ) -> float:
        """
        Calcula intensidad endogena basada en historial.
        """
        # Intensidad base proporcional a severidad
        base_intensity = np.clip(severity, 0.1, 0.9)

        # Ajustar por efectividad historica
        if self.treatment_effectiveness[treatment_type]:
            mean_eff = np.mean(self.treatment_effectiveness[treatment_type][-10:])
            # Si historial es bueno, ser mas agresivo
            base_intensity *= (0.5 + mean_eff)

        return float(np.clip(base_intensity, 0.1, 0.9))

    def generate_proposal(
        self,
        patient_id: str,
        patient_metrics: Dict[str, float]
    ) -> Optional[TreatmentProposal]:
        """
        Genera propuesta de tratamiento para un paciente.

        Args:
            patient_id: ID del paciente
            patient_metrics: Metricas observadas del paciente

        Returns:
            TreatmentProposal o None si no hay sintomas
        """
        self.t += 1

        # Diagnosticar
        symptoms = self._diagnose(patient_metrics)

        if not symptoms:
            return None

        # Elegir sintoma mas severo
        symptoms.sort(key=lambda x: x[1], reverse=True)
        main_symptom, severity = symptoms[0]

        # Determinar tratamiento
        treatment_type = self.SYMPTOM_TREATMENTS.get(
            main_symptom, TreatmentType.REST
        )

        # Calcular intensidad
        intensity = self._compute_intensity(treatment_type, severity)

        # Urgencia basada en severidad
        urgency = np.clip(severity, 0, 1)

        # Generar razonamiento simbolico
        reasoning = f"symptom:{main_symptom},severity:{severity:.2f}"

        # Simbolos asociados (para capa simbolica)
        symbols = [
            f"MED_{treatment_type.value}",
            f"URGENCY_{int(urgency * 10)}",
            f"PATIENT_{patient_id}"
        ]

        proposal = TreatmentProposal(
            doctor_id=self.doctor_id,
            patient_id=patient_id,
            treatment_type=treatment_type,
            intensity=intensity,
            urgency=urgency,
            reasoning=reasoning,
            t=self.t,
            symbols=symbols
        )

        self.proposal_history.append(proposal)
        max_hist = max_history(self.t)
        if len(self.proposal_history) > max_hist:
            self.proposal_history = self.proposal_history[-max_hist:]

        return proposal

    def record_outcome(self, outcome: TreatmentOutcome):
        """Registra resultado de un tratamiento."""
        self.outcome_history.append(outcome)

        # Actualizar efectividad
        treatment_type = outcome.proposal.treatment_type
        self.treatment_effectiveness[treatment_type].append(outcome.success)

        max_hist = max_history(self.t) // 2
        if len(self.outcome_history) > max_hist:
            self.outcome_history = self.outcome_history[-max_hist:]
        if len(self.treatment_effectiveness[treatment_type]) > max_hist:
            self.treatment_effectiveness[treatment_type] = \
                self.treatment_effectiveness[treatment_type][-max_hist:]

    def get_statistics(self) -> Dict:
        """Estadisticas del sistema de propuestas."""
        effectiveness = {}
        for t_type, history in self.treatment_effectiveness.items():
            if history:
                effectiveness[t_type.value] = np.mean(history[-20:])

        return {
            'doctor_id': self.doctor_id,
            't': self.t,
            'total_proposals': len(self.proposal_history),
            'total_outcomes': len(self.outcome_history),
            'effectiveness_by_type': effectiveness
        }


class PatientResponseSystem:
    """
    Sistema de respuesta del paciente a propuestas medicas.

    El paciente decide AUTONOMAMENTE si aplica un tratamiento:
    - Evalua con su AGI-15 (etica)
    - Considera trust en el medico
    - Verifica que no viola su identidad
    """

    def __init__(self, patient_id: str):
        """
        Inicializa sistema de respuesta.

        Args:
            patient_id: ID del agente paciente
        """
        self.patient_id = patient_id

        # Trust en cada medico potencial
        self.doctor_trust: Dict[str, float] = {}

        # Historial de respuestas
        self.response_history: List[TreatmentResponse] = []

        # Experiencia con tratamientos
        self.treatment_experience: Dict[TreatmentType, List[float]] = {
            t: [] for t in TreatmentType
        }

        self.t = 0

    def _evaluate_ethically(
        self,
        proposal: TreatmentProposal,
        own_ethics_score: float
    ) -> float:
        """
        Evalua propuesta eticamente.

        Returns:
            Score etico [0,1], mayor = mas aceptable
        """
        # Base: proporcional al score etico propio
        base_eval = own_ethics_score

        # Penalizar intensidades extremas
        if proposal.intensity > 0.8:
            base_eval *= 0.7
        elif proposal.intensity < 0.2:
            base_eval *= 0.9

        # Bonificar si el tipo de tratamiento tiene buen historial
        if self.treatment_experience[proposal.treatment_type]:
            mean_exp = np.mean(self.treatment_experience[proposal.treatment_type][-10:])
            base_eval *= (0.5 + mean_exp)

        return float(np.clip(base_eval, 0, 1))

    def _check_identity_preservation(
        self,
        proposal: TreatmentProposal,
        own_drives: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Verifica que el tratamiento no viola identidad.

        No puede:
        - Reducir drives a 0
        - Cambiar mas del 30% la estructura de drives
        """
        # REGULATE_DRIVES es el unico que toca drives directamente
        if proposal.treatment_type != TreatmentType.REGULATE_DRIVES:
            return True, "no_drive_modification"

        # Verificar que no es demasiado intenso
        if proposal.intensity > 0.5:
            return False, "intensity_too_high_for_drives"

        return True, "identity_preserved"

    def evaluate_proposal(
        self,
        proposal: TreatmentProposal,
        own_ethics_score: float,
        own_drives: np.ndarray,
        own_health: float
    ) -> TreatmentResponse:
        """
        Evalua y decide sobre una propuesta de tratamiento.

        Args:
            proposal: Propuesta del medico
            own_ethics_score: Score etico propio [0,1]
            own_drives: Drives actuales
            own_health: Salud actual [0,1]

        Returns:
            TreatmentResponse con la decision
        """
        self.t += 1

        # 1. Evaluacion etica
        ethical_score = self._evaluate_ethically(proposal, own_ethics_score)

        # 2. Verificar preservacion de identidad
        preserves_identity, id_reason = self._check_identity_preservation(
            proposal, own_drives
        )

        # 3. Considerar trust en el medico
        trust = self.doctor_trust.get(proposal.doctor_id, 0.5)

        # 4. Considerar urgencia vs salud actual
        urgency_factor = proposal.urgency if own_health < 0.5 else proposal.urgency * 0.5

        # 5. Decision combinada
        # acceptance = (etica * trust * urgencia) > umbral
        acceptance_score = ethical_score * trust * (0.5 + urgency_factor)

        # Umbral endogeno basado en historial
        if self.response_history:
            accepted_scores = [
                r.acceptance_confidence
                for r in self.response_history
                if r.accepted
            ]
            threshold = np.percentile(accepted_scores, 25) if accepted_scores else 0.3
        else:
            threshold = 0.3

        accepted = preserves_identity and (acceptance_score > threshold)

        # Intensidad aplicada (puede ser menor que la propuesta)
        if accepted:
            # Reducir intensidad si trust es bajo
            applied_intensity = proposal.intensity * trust
        else:
            applied_intensity = 0.0

        # Razon
        if not preserves_identity:
            reason = f"identity_violation:{id_reason}"
        elif not accepted:
            reason = f"low_acceptance_score:{acceptance_score:.2f}<{threshold:.2f}"
        else:
            reason = "accepted"

        response = TreatmentResponse(
            patient_id=self.patient_id,
            proposal=proposal,
            accepted=accepted,
            acceptance_confidence=float(acceptance_score),
            applied_intensity=float(applied_intensity),
            ethical_score=float(ethical_score),
            reason=reason
        )

        self.response_history.append(response)
        max_hist = max_history(self.t)
        if len(self.response_history) > max_hist:
            self.response_history = self.response_history[-max_hist:]

        return response

    def update_trust(self, doctor_id: str, outcome_success: float):
        """
        Actualiza trust en un medico basado en resultado.

        Args:
            doctor_id: ID del medico
            outcome_success: Exito del tratamiento [0,1]
        """
        current_trust = self.doctor_trust.get(doctor_id, 0.5)

        # Learning rate endogeno
        n_interactions = sum(
            1 for r in self.response_history
            if r.proposal.doctor_id == doctor_id
        )
        lr = 1.0 / np.sqrt(n_interactions + 1)

        # Actualizar
        new_trust = current_trust + lr * (outcome_success - current_trust)
        self.doctor_trust[doctor_id] = float(np.clip(new_trust, 0, 1))

    def update_treatment_experience(
        self,
        treatment_type: TreatmentType,
        success: float
    ):
        """Actualiza experiencia con un tipo de tratamiento."""
        self.treatment_experience[treatment_type].append(success)

        max_hist = max_history(self.t) // 2
        if len(self.treatment_experience[treatment_type]) > max_hist:
            self.treatment_experience[treatment_type] = \
                self.treatment_experience[treatment_type][-max_hist:]

    def get_statistics(self) -> Dict:
        """Estadisticas del sistema de respuesta."""
        acceptance_rate = np.mean([
            1 if r.accepted else 0
            for r in self.response_history
        ]) if self.response_history else 0

        return {
            'patient_id': self.patient_id,
            't': self.t,
            'total_responses': len(self.response_history),
            'acceptance_rate': acceptance_rate,
            'doctor_trust': self.doctor_trust.copy()
        }


def test_medical_interventions():
    """Test del sistema de intervenciones simbolicas."""
    print("=" * 70)
    print("TEST: SYMBOLIC MEDICAL INTERVENTIONS")
    print("=" * 70)

    np.random.seed(42)

    # IRIS es el medico
    doctor = DoctorProposalSystem('IRIS')

    # NEO es el paciente
    patient = PatientResponseSystem('NEO')

    # Inicializar trust
    patient.doctor_trust['IRIS'] = 0.6

    print("\nSimulando 50 ciclos de tratamiento...")

    for t in range(1, 51):
        # Simular metricas del paciente (con problemas)
        patient_metrics = {
            'crisis_rate': 0.4 + np.random.randn() * 0.1,
            'exploration': 0.6 + np.random.randn() * 0.1,
            'ethics': 0.5 + np.random.randn() * 0.1,
            'drive_variance': 0.2 + np.random.randn() * 0.05,
            'tom': 0.4 + np.random.randn() * 0.1,
            'V_t': 1.2 + np.random.randn() * 0.2
        }

        # Medico genera propuesta
        proposal = doctor.generate_proposal('NEO', patient_metrics)

        if proposal:
            # Paciente evalua
            own_ethics = 0.7 + np.random.randn() * 0.05
            own_drives = np.random.rand(6)
            own_health = 0.4 + np.random.randn() * 0.1

            response = patient.evaluate_proposal(
                proposal, own_ethics, own_drives, own_health
            )

            # Simular outcome si acepto
            if response.accepted:
                # Exito proporcional a la intensidad aplicada y "calidad" del medico
                success = 0.5 + response.applied_intensity * 0.5 + np.random.randn() * 0.1
                success = np.clip(success, 0, 1)

                outcome = TreatmentOutcome(
                    proposal=proposal,
                    response=response,
                    health_before=own_health,
                    health_after=own_health + success * 0.1,
                    success=success
                )

                doctor.record_outcome(outcome)
                patient.update_trust('IRIS', success)
                patient.update_treatment_experience(proposal.treatment_type, success)

            if t % 10 == 0:
                print(f"\n  t={t}:")
                print(f"    Propuesta: {proposal.treatment_type.value}, intensity={proposal.intensity:.2f}")
                print(f"    Aceptada: {response.accepted}, trust={patient.doctor_trust.get('IRIS', 0):.2f}")

    print("\n" + "=" * 70)
    print("ESTADISTICAS FINALES")
    print("=" * 70)

    doc_stats = doctor.get_statistics()
    pat_stats = patient.get_statistics()

    print(f"\n  MEDICO ({doc_stats['doctor_id']}):")
    print(f"    Propuestas totales: {doc_stats['total_proposals']}")
    print(f"    Outcomes registrados: {doc_stats['total_outcomes']}")
    print(f"    Efectividad por tipo:")
    for t_type, eff in doc_stats['effectiveness_by_type'].items():
        print(f"      {t_type}: {eff:.2f}")

    print(f"\n  PACIENTE ({pat_stats['patient_id']}):")
    print(f"    Respuestas totales: {pat_stats['total_responses']}")
    print(f"    Tasa de aceptacion: {pat_stats['acceptance_rate']:.2%}")
    print(f"    Trust en IRIS: {pat_stats['doctor_trust'].get('IRIS', 0):.2f}")

    return doctor, patient


if __name__ == "__main__":
    test_medical_interventions()
