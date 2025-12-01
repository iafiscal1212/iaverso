"""
Lifecycle Module: Ciclo de Vida de los Agentes
===============================================

Sistema de ciclo de vida para agentes cognitivos.

Componentes:
    - CircadianSystem: Ritmos de actividad/descanso
    - AgentCircadianCycle: Ciclo individual por agente
    - AbsenceSimulator: Simulacion de eventos durante ausencia
    - DreamProcessor: Consolidacion de memorias durante suenos
    - LifeJournal: Diario de vida del agente
    - ReconnectionNarrative: Narrativa al reconectar

100% endogeno. Sin relojes externos hardcodeados.
"""

from .circadian_system import (
    CircadianPhase,
    CircadianState,
    LifeEvent,
    AbsenceReport,
    AgentCircadianCycle,
    AbsenceSimulator,
)

from .dream_processor import (
    DreamFragment,
    ConsolidationResult,
    DreamProcessor,
)

from .life_journal import (
    JournalEntryType,
    JournalEntry,
    LifePeriod,
    LifeJournal,
)

from .reconnection_narrative import (
    NarrativeTone,
    ReconnectionGreeting,
    AbsenceNarrative,
    FullReconnectionNarrative,
    ReconnectionNarrativeGenerator,
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
]
