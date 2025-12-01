"""
Health Module: Emergent Cognitive Health System
================================================

Health system for cognitive agents - FULLY EMERGENT.

NO external SystemDoctor. Each agent:
    - Evaluates itself (medical_profile)
    - Evaluates others (medical_beliefs)
    - Votes for doctor (distributed_election)
    - Decides on treatments (medical_interventions)

The doctor role emerges from consensus, not assignment.
Doctor's power comes from trust + results, not privileged API.

Components:
    - AgentMedicalSelf: Self-evaluation of medical aptitude
    - AgentMedicalBeliefs: Beliefs about others as doctors
    - DistributedMedicalElection: Consensus-based election
    - DoctorProposalSystem: Symbolic treatment proposals
    - PatientResponseSystem: Autonomous treatment decisions
    - EmergentMedicalSystem: Integration of all components

100% endogenous. No external arbiter.
"""

# New emergent system (preferred)
from .medical_profile import (
    AgentMedicalSelf,
    MedicalProfile
)

from .medical_beliefs import (
    AgentMedicalBeliefs,
    DistributedMedicalElection,
    MedicalBelief,
    Vote,
    ElectionResult
)

from .medical_interventions import (
    DoctorProposalSystem,
    PatientResponseSystem,
    TreatmentType,
    TreatmentProposal,
    TreatmentResponse,
    TreatmentOutcome
)

from .emergent_medical_system import (
    EmergentMedicalSystem,
    AgentMedicalModule,
    AgentHealthState
)

# Legacy components (for backwards compatibility)
from .health_monitor import (
    HealthMonitor,
    HealthMetrics,
    HealthAssessment,
    HealthLevel
)

from .repair_protocols import (
    RepairProtocol,
    Intervention,
    InterventionResult,
    InterventionType
)

# MED-X Benchmark
from .medx_benchmark import (
    MedXBenchmark,
    MedXResults,
    AgentHealthSnapshot,
    TreatmentEvent
)

# Clinical Cases
from .clinical_cases import (
    ClinicalCaseSimulator,
    ClinicalCondition,
    ClinicalCaseReport,
    TreatmentProtocol
)

__all__ = [
    # Emergent System (new)
    'EmergentMedicalSystem',
    'AgentMedicalModule',
    'AgentHealthState',
    # Medical Profile
    'AgentMedicalSelf',
    'MedicalProfile',
    # Medical Beliefs
    'AgentMedicalBeliefs',
    'DistributedMedicalElection',
    'MedicalBelief',
    'Vote',
    'ElectionResult',
    # Medical Interventions
    'DoctorProposalSystem',
    'PatientResponseSystem',
    'TreatmentType',
    'TreatmentProposal',
    'TreatmentResponse',
    'TreatmentOutcome',
    # Legacy (for compatibility)
    'HealthMonitor',
    'HealthMetrics',
    'HealthAssessment',
    'HealthLevel',
    'RepairProtocol',
    'Intervention',
    'InterventionResult',
    'InterventionType',
    # MED-X Benchmark
    'MedXBenchmark',
    'MedXResults',
    'AgentHealthSnapshot',
    'TreatmentEvent',
    # Clinical Cases
    'ClinicalCaseSimulator',
    'ClinicalCondition',
    'ClinicalCaseReport',
    'TreatmentProtocol',
]
