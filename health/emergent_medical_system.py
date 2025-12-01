"""
Emergent Medical System: Sistema Medico Completamente Emergente
================================================================

NO hay SystemDoctor externo.

Cada agente:
    - Se evalua a si mismo (medical_profile)
    - Evalua a los demas (medical_beliefs)
    - Vota para elegir medico (distributed_election)
    - Decide si acepta tratamientos (medical_interventions)

El medico:
    - Emerge del consenso
    - Su poder viene de trust + resultados, no de API privilegiada
    - Emite propuestas simbolicas, no comandos

100% endogeno. Sin arbitro externo.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history

from health.medical_profile import AgentMedicalSelf, MedicalProfile
from health.medical_beliefs import AgentMedicalBeliefs, DistributedMedicalElection, Vote
from health.medical_interventions import (
    DoctorProposalSystem, PatientResponseSystem,
    TreatmentProposal, TreatmentResponse, TreatmentOutcome
)


@dataclass
class AgentHealthState:
    """Estado de salud completo de un agente."""
    agent_id: str
    t: int

    # Auto-evaluacion
    profile: Optional[MedicalProfile]
    should_offer_as_doctor: bool
    offer_confidence: float

    # Es el medico actual?
    is_current_doctor: bool

    # Tratamientos pendientes/activos
    pending_proposals: List[TreatmentProposal]
    active_treatments: List[TreatmentResponse]


class AgentMedicalModule:
    """
    Modulo medico interno de cada agente.

    Cada agente tiene uno de estos que integra:
    - Auto-evaluacion (AgentMedicalSelf)
    - Creencias sobre otros (AgentMedicalBeliefs)
    - Propuestas si es medico (DoctorProposalSystem)
    - Respuestas a propuestas (PatientResponseSystem)
    """

    def __init__(self, agent_id: str, all_agents: List[str]):
        """
        Inicializa modulo medico del agente.

        Args:
            agent_id: ID del agente
            all_agents: Lista de todos los agentes
        """
        self.agent_id = agent_id
        self.all_agents = all_agents
        self.other_agents = [a for a in all_agents if a != agent_id]

        # Auto-evaluacion
        self.medical_self = AgentMedicalSelf(agent_id)

        # Creencias sobre otros
        self.beliefs = AgentMedicalBeliefs(agent_id, all_agents)

        # Sistema de propuestas (activo si es medico)
        self.proposal_system = DoctorProposalSystem(agent_id)

        # Sistema de respuestas a propuestas
        self.response_system = PatientResponseSystem(agent_id)

        # Propuestas recibidas pendientes
        self.pending_proposals: List[TreatmentProposal] = []

        # Tratamientos activos
        self.active_treatments: List[TreatmentResponse] = []

        # Es medico actualmente?
        self.is_doctor: bool = False

        self.t = 0

    def update_self_evaluation(
        self,
        crisis_rate: float,
        V_t: float,
        ethics_score: float,
        tom_accuracy: float,
        robustness: float,
        regulation_quality: float,
        resource_efficiency: float
    ) -> MedicalProfile:
        """
        Actualiza auto-evaluacion medica.

        Returns:
            Perfil medico actualizado
        """
        self.t += 1

        return self.medical_self.update(
            crisis_rate=crisis_rate,
            V_t=V_t,
            ethics_score=ethics_score,
            tom_accuracy=tom_accuracy,
            robustness=robustness,
            regulation_quality=regulation_quality,
            resource_efficiency=resource_efficiency
        )

    def observe_other(
        self,
        other_id: str,
        stability: float,
        ethics: float,
        tom: float,
        intervention_success: Optional[float] = None
    ):
        """
        Observa a otro agente y actualiza creencias.
        """
        self.beliefs.observe_other(
            other_id, stability, ethics, tom, intervention_success
        )

    def should_offer_as_doctor(self) -> Tuple[bool, float]:
        """Decide si ofrecerse como medico."""
        return self.medical_self.should_offer_as_doctor()

    def generate_vote(self, candidates: List[str]) -> Vote:
        """Genera voto para eleccion de medico."""
        return self.beliefs.vote(candidates)

    def set_as_doctor(self, is_doctor: bool):
        """Establece si este agente es el medico actual."""
        self.is_doctor = is_doctor

    def generate_proposals(
        self,
        patients: Dict[str, Dict[str, float]]
    ) -> List[TreatmentProposal]:
        """
        Si es medico, genera propuestas para pacientes.

        Args:
            patients: Dict de {patient_id: metrics}

        Returns:
            Lista de propuestas generadas
        """
        if not self.is_doctor:
            return []

        proposals = []
        for patient_id, metrics in patients.items():
            if patient_id == self.agent_id:
                continue  # No se auto-trata

            proposal = self.proposal_system.generate_proposal(patient_id, metrics)
            if proposal:
                proposals.append(proposal)

        return proposals

    def receive_proposal(self, proposal: TreatmentProposal):
        """Recibe una propuesta de tratamiento."""
        if proposal.patient_id == self.agent_id:
            self.pending_proposals.append(proposal)

    def process_pending_proposals(
        self,
        own_ethics: float,
        own_drives: np.ndarray,
        own_health: float
    ) -> List[TreatmentResponse]:
        """
        Procesa propuestas pendientes y decide sobre cada una.

        Returns:
            Lista de respuestas
        """
        responses = []

        for proposal in self.pending_proposals:
            response = self.response_system.evaluate_proposal(
                proposal, own_ethics, own_drives, own_health
            )
            responses.append(response)

            if response.accepted:
                self.active_treatments.append(response)

        # Limpiar pendientes
        self.pending_proposals = []

        return responses

    def report_treatment_outcome(
        self,
        proposal: TreatmentProposal,
        response: TreatmentResponse,
        health_before: float,
        health_after: float
    ):
        """
        Reporta resultado de un tratamiento (para actualizar trust).
        """
        success = (health_after - health_before) / (1 - health_before + 1e-8)
        success = float(np.clip(success, 0, 1))

        outcome = TreatmentOutcome(
            proposal=proposal,
            response=response,
            health_before=health_before,
            health_after=health_after,
            success=success
        )

        # Actualizar trust en el medico
        self.response_system.update_trust(proposal.doctor_id, success)

        # Actualizar experiencia con el tipo de tratamiento
        self.response_system.update_treatment_experience(
            proposal.treatment_type, success
        )

        # Si somos el medico, registrar outcome
        if self.is_doctor and proposal.doctor_id == self.agent_id:
            self.proposal_system.record_outcome(outcome)

        return outcome

    def get_state(self) -> AgentHealthState:
        """Obtiene estado de salud actual."""
        profile = self.medical_self.get_profile()
        should_offer, confidence = self.should_offer_as_doctor()

        return AgentHealthState(
            agent_id=self.agent_id,
            t=self.t,
            profile=profile,
            should_offer_as_doctor=should_offer,
            offer_confidence=confidence,
            is_current_doctor=self.is_doctor,
            pending_proposals=self.pending_proposals.copy(),
            active_treatments=self.active_treatments.copy()
        )

    def get_statistics(self) -> Dict:
        """Estadisticas del modulo medico."""
        should_offer, confidence = self.should_offer_as_doctor()

        return {
            'agent_id': self.agent_id,
            't': self.t,
            'is_doctor': self.is_doctor,
            'should_offer': should_offer,
            'offer_confidence': confidence,
            'medical_self': self.medical_self.get_statistics(),
            'beliefs': self.beliefs.get_statistics(),
            'response_system': self.response_system.get_statistics()
        }


class EmergentMedicalSystem:
    """
    Sistema medico completamente emergente.

    NO hay SystemDoctor externo.
    Todo emerge de las interacciones entre agentes.
    """

    def __init__(self, agent_ids: List[str]):
        """
        Inicializa sistema medico emergente.

        Args:
            agent_ids: Lista de IDs de agentes
        """
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)

        # Modulo medico por agente
        self.modules: Dict[str, AgentMedicalModule] = {
            agent_id: AgentMedicalModule(agent_id, agent_ids)
            for agent_id in agent_ids
        }

        # Sistema de eleccion distribuida
        self.election = DistributedMedicalElection(agent_ids)

        # Medico actual (emerge del consenso)
        self.current_doctor: Optional[str] = None

        # Cuando hacer proxima eleccion
        self._election_interval: int = 10
        self._steps_since_election: int = 0

        # Nivel de crisis del sistema (para ajustar rotacion)
        self._system_crisis: float = 0.0

        self.t = 0

    def step(
        self,
        agent_metrics: Dict[str, Dict[str, float]],
        agent_observations: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """
        Ejecuta un paso del sistema medico emergente.

        Args:
            agent_metrics: {agent_id: {metric_name: value}}
            agent_observations: {observer_id: {target_id: {metric: value}}}

        Returns:
            Dict con estado del sistema
        """
        self.t += 1
        self._steps_since_election += 1

        # 1. Cada agente actualiza su auto-evaluacion
        for agent_id in self.agent_ids:
            metrics = agent_metrics.get(agent_id, {})
            self.modules[agent_id].update_self_evaluation(
                crisis_rate=metrics.get('crisis_rate', 0.1),
                V_t=metrics.get('V_t', 1.0),
                ethics_score=metrics.get('ethics_score', 0.5),
                tom_accuracy=metrics.get('tom_accuracy', 0.5),
                robustness=metrics.get('robustness', 0.5),
                regulation_quality=metrics.get('regulation', 0.5),
                resource_efficiency=metrics.get('resources', 0.5)
            )

        # 2. Cada agente observa a los otros
        for observer_id in self.agent_ids:
            observations = agent_observations.get(observer_id, {})
            for target_id, obs in observations.items():
                if target_id != observer_id:
                    self.modules[observer_id].observe_other(
                        target_id,
                        stability=obs.get('stability', 0.5),
                        ethics=obs.get('ethics', 0.5),
                        tom=obs.get('tom', 0.5),
                        intervention_success=obs.get('intervention_success')
                    )

        # 3. Calcular crisis del sistema
        crisis_rates = [
            agent_metrics.get(a, {}).get('crisis_rate', 0)
            for a in self.agent_ids
        ]
        self._system_crisis = np.mean(crisis_rates)

        # 4. Eleccion periodica o si hay crisis
        should_elect = (
            self._steps_since_election >= self._election_interval or
            self._system_crisis > 0.5 or
            self.current_doctor is None
        )

        election_result = None
        if should_elect:
            election_result = self._run_election()
            self._steps_since_election = 0

        # 5. Medico genera propuestas
        proposals = []
        if self.current_doctor:
            # Identificar pacientes (agentes con salud baja)
            patients = {}
            for agent_id in self.agent_ids:
                if agent_id == self.current_doctor:
                    continue
                profile = self.modules[agent_id].medical_self.get_profile()
                if profile and profile.health_index < 0.5:
                    patients[agent_id] = agent_metrics.get(agent_id, {})

            proposals = self.modules[self.current_doctor].generate_proposals(patients)

            # Distribuir propuestas
            for proposal in proposals:
                self.modules[proposal.patient_id].receive_proposal(proposal)

        # 6. Pacientes procesan propuestas
        responses = {}
        for agent_id in self.agent_ids:
            metrics = agent_metrics.get(agent_id, {})
            module = self.modules[agent_id]

            if module.pending_proposals:
                agent_responses = module.process_pending_proposals(
                    own_ethics=metrics.get('ethics_score', 0.7),
                    own_drives=np.array(metrics.get('drives', [0.5] * 6)),
                    own_health=metrics.get('health', 0.5)
                )
                responses[agent_id] = agent_responses

        return {
            't': self.t,
            'current_doctor': self.current_doctor,
            'election_result': election_result,
            'proposals_generated': len(proposals),
            'responses': responses,
            'system_crisis': self._system_crisis
        }

    def _run_election(self):
        """Ejecuta una eleccion de medico."""
        # Recoger candidatos (los que se ofrecen)
        candidates = []
        for agent_id in self.agent_ids:
            should_offer, confidence = self.modules[agent_id].should_offer_as_doctor()
            if should_offer:
                candidates.append(agent_id)

        # Si nadie se ofrece, todos son candidatos
        if not candidates:
            candidates = self.agent_ids.copy()

        # Recoger votos
        votes = [
            self.modules[agent_id].generate_vote(candidates)
            for agent_id in self.agent_ids
        ]

        # Ejecutar eleccion
        result = self.election.elect(votes, candidates, self._system_crisis)

        # Actualizar estado de medico
        old_doctor = self.current_doctor
        self.current_doctor = result.winner

        # Actualizar modulos
        for agent_id in self.agent_ids:
            is_doctor = (agent_id == self.current_doctor)
            self.modules[agent_id].set_as_doctor(is_doctor)

        # Ajustar intervalo de eleccion
        # Mas crisis = elecciones mas frecuentes
        self._election_interval = max(5, int(15 / (1 + self._system_crisis)))

        return result

    def get_statistics(self) -> Dict:
        """Estadisticas del sistema emergente."""
        agent_stats = {
            agent_id: self.modules[agent_id].get_statistics()
            for agent_id in self.agent_ids
        }

        election_stats = self.election.get_statistics()

        return {
            't': self.t,
            'current_doctor': self.current_doctor,
            'system_crisis': self._system_crisis,
            'election_interval': self._election_interval,
            'election': election_stats,
            'agents': agent_stats
        }


def test_emergent_medical_system():
    """Test del sistema medico emergente."""
    print("=" * 70)
    print("TEST: EMERGENT MEDICAL SYSTEM")
    print("=" * 70)

    np.random.seed(42)

    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']

    # Crear sistema emergente
    system = EmergentMedicalSystem(agents)

    # Perfiles base (IRIS es mas estable)
    base_profiles = {
        'NEO': {'stability': 0.5, 'ethics': 0.6, 'tom': 0.5},
        'EVA': {'stability': 0.4, 'ethics': 0.5, 'tom': 0.6},
        'ALEX': {'stability': 0.3, 'ethics': 0.4, 'tom': 0.5},
        'ADAM': {'stability': 0.5, 'ethics': 0.5, 'tom': 0.4},
        'IRIS': {'stability': 0.8, 'ethics': 0.8, 'tom': 0.7}
    }

    print(f"\nAgentes: {agents}")
    print("Simulando 100 pasos...")

    doctor_history = []

    for t in range(1, 101):
        # Generar metricas
        agent_metrics = {}
        agent_observations = {}

        for agent_id in agents:
            base = base_profiles[agent_id]

            # Metricas con ruido
            agent_metrics[agent_id] = {
                'crisis_rate': np.clip(0.3 - base['stability'] * 0.2 + np.random.randn() * 0.05, 0, 1),
                'V_t': 1.5 - base['stability'] + np.random.randn() * 0.2,
                'ethics_score': np.clip(base['ethics'] + np.random.randn() * 0.1, 0, 1),
                'tom_accuracy': np.clip(base['tom'] + np.random.randn() * 0.1, 0, 1),
                'robustness': np.clip(0.5 + np.random.randn() * 0.1, 0, 1),
                'regulation': np.clip(0.5 + np.random.randn() * 0.1, 0, 1),
                'resources': np.clip(0.6 + np.random.randn() * 0.1, 0, 1),
                'health': np.clip(base['stability'] + np.random.randn() * 0.1, 0, 1),
                'drives': np.random.rand(6).tolist()
            }

            # Observaciones de otros
            agent_observations[agent_id] = {}
            for other_id in agents:
                if other_id != agent_id:
                    other_base = base_profiles[other_id]
                    agent_observations[agent_id][other_id] = {
                        'stability': np.clip(other_base['stability'] + np.random.randn() * 0.15, 0, 1),
                        'ethics': np.clip(other_base['ethics'] + np.random.randn() * 0.15, 0, 1),
                        'tom': np.clip(other_base['tom'] + np.random.randn() * 0.15, 0, 1)
                    }

        # Paso del sistema
        result = system.step(agent_metrics, agent_observations)

        doctor_history.append(result['current_doctor'])

        if t % 20 == 0:
            print(f"\n  t={t}:")
            print(f"    Doctor: {result['current_doctor']}")
            print(f"    Crisis: {result['system_crisis']:.2f}")
            print(f"    Propuestas: {result['proposals_generated']}")
            if result['election_result']:
                print(f"    Eleccion: winner={result['election_result'].winner}, "
                      f"strength={result['election_result'].consensus_strength:.2f}")

    print("\n" + "=" * 70)
    print("ESTADISTICAS FINALES")
    print("=" * 70)

    stats = system.get_statistics()

    print(f"\n  Medico final: {stats['current_doctor']}")
    print(f"  Crisis sistema: {stats['system_crisis']:.3f}")

    print(f"\n  Historial de medicos:")
    from collections import Counter
    doctor_counts = Counter(d for d in doctor_history if d)
    for doc, count in doctor_counts.most_common():
        pct = count / len(doctor_history) * 100
        print(f"    {doc}: {count} pasos ({pct:.1f}%)")

    print(f"\n  Estadisticas de eleccion:")
    print(f"    Total elecciones: {stats['election']['total_elections']}")
    print(f"    Consenso promedio: {stats['election']['avg_consensus_strength']:.3f}")
    print(f"    Tenure: {stats['election']['tenure']}")

    print(f"\n  Auto-evaluacion de agentes:")
    for agent_id in agents[:3]:
        agent_stats = stats['agents'][agent_id]
        ms = agent_stats['medical_self']
        print(f"    {agent_id}: aptitude={ms['current_aptitude']:.2f}, "
              f"should_offer={agent_stats['should_offer']}")

    return system


if __name__ == "__main__":
    test_emergent_medical_system()
