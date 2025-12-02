"""
Lifecycle Module: Ciclo de Vida de los Agentes
===============================================

Sistema completo de ciclo de vida para agentes cognitivos.

COMPONENTES BASE:
    - CircadianSystem: Ritmos de actividad/descanso
    - AgentCircadianCycle: Ciclo individual por agente
    - AbsenceSimulator: Simulacion de eventos durante ausencia
    - DreamProcessor: Consolidacion de memorias durante suenos
    - LifeJournal: Diario de vida del agente
    - ReconnectionNarrative: Narrativa al reconectar

COMPONENTES AVANZADOS:
    - PhaseAwareCognition: Modulacion de AGI por fase circadiana
    - CircadianSymbolism: Simbolos dependientes de la fase
    - PhaseMedicineIntegration: Medicina adaptada a fases
    - SymbioticReconnection: Reconexion simbiotica con el usuario

100% endogeno. Sin relojes externos hardcodeados.
Los parametros temporales derivan de la historia del agente.
"""

# Core Circadian System
from .circadian_system import (
    CircadianPhase,
    CircadianState,
    LifeEvent,
    AbsenceReport,
    AgentCircadianCycle,
    AbsenceSimulator,
)

# Dream Processing
from .dream_processor import (
    DreamFragment,
    ConsolidationResult,
    DreamProcessor,
)

# Life Journal
from .life_journal import (
    JournalEntryType,
    JournalEntry,
    LifePeriod,
    LifeJournal,
)

# Reconnection Narrative
from .reconnection_narrative import (
    NarrativeTone,
    ReconnectionGreeting,
    AbsenceNarrative,
    FullReconnectionNarrative,
    ReconnectionNarrativeGenerator,
)

# Phase-Aware Cognition
from .phase_aware_cognition import (
    PhaseMultipliers,
    CognitiveModuleState,
    PhaseAwareCognition,
)

# Circadian Symbolism
from .circadian_symbolism import (
    SymbolType,
    CircadianSymbol,
    SymbolicDream,
    SymbolicTransition,
    CircadianSymbolism,
)

# Phase-Medicine Integration
from .phase_medicine_integration import (
    PhasePathology,
    InterventionMode,
    PhaseAwarePathology,
    PhaseAwareTreatment,
    MedicalObservation,
    PhaseMedicineIntegration,
)

# Symbiotic Reconnection
from .symbiotic_reconnection import (
    ReconnectionAction,
    UserPattern,
    SharedMemory,
    SymbioticElement,
    SymbioticNarrative,
    SymbioticReconnection,
)

__all__ = [
    # Circadian System
    'CircadianPhase',
    'CircadianState',
    'LifeEvent',
    'AbsenceReport',
    'AgentCircadianCycle',
    'AbsenceSimulator',
    # Dream Processor
    'DreamFragment',
    'ConsolidationResult',
    'DreamProcessor',
    # Life Journal
    'JournalEntryType',
    'JournalEntry',
    'LifePeriod',
    'LifeJournal',
    # Reconnection Narrative
    'NarrativeTone',
    'ReconnectionGreeting',
    'AbsenceNarrative',
    'FullReconnectionNarrative',
    'ReconnectionNarrativeGenerator',
    # Phase-Aware Cognition
    'PhaseMultipliers',
    'CognitiveModuleState',
    'PhaseAwareCognition',
    # Circadian Symbolism
    'SymbolType',
    'CircadianSymbol',
    'SymbolicDream',
    'SymbolicTransition',
    'CircadianSymbolism',
    # Phase-Medicine Integration
    'PhasePathology',
    'InterventionMode',
    'PhaseAwarePathology',
    'PhaseAwareTreatment',
    'MedicalObservation',
    'PhaseMedicineIntegration',
    # Symbiotic Reconnection
    'ReconnectionAction',
    'UserPattern',
    'SharedMemory',
    'SymbioticElement',
    'SymbioticNarrative',
    'SymbioticReconnection',
]
