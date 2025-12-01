"""
AGI-X v2.0: Sistema de Vida Cognitiva Completa
==============================================

Ciclo de vida endógeno completo:
percepción → cognición → acción → mundo → memoria → narrativa → reconfiguración → metas → acción

Todos los agentes son teleológicos por diseño.
Sin números mágicos. Todo endógeno.

Módulos:
- architecture: Estructuras de datos y sistemas base (CausalModel, MetaMemory, Antifragility)
- cognitive_life_cycle: Ciclo de vida cognitivo unificado
"""

from .architecture import (
    LifeCyclePhase,
    CognitiveState,
    WorldState,
    Action,
    Memory,
    Goal,
    CausalModel,
    MetaMemory,
    AntifragilitySystem,
    endogenous_goal_priority,
    endogenous_value_update
)

from .cognitive_life_cycle import (
    TeleologicalAgent,
    CognitiveLifeCycle
)

__all__ = [
    # Architecture
    'LifeCyclePhase',
    'CognitiveState',
    'WorldState',
    'Action',
    'Memory',
    'Goal',
    'CausalModel',
    'MetaMemory',
    'AntifragilitySystem',
    'endogenous_goal_priority',
    'endogenous_value_update',
    # Life Cycle
    'TeleologicalAgent',
    'CognitiveLifeCycle'
]
