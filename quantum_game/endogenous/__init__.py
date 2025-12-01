"""
QUANTUM GAME ENDOGENOUS - Juego Cuántico 100% Endógeno
=====================================================

Módulo de juego de coalición cuántico donde TODO es endógeno:

- Estados: ranks de drives, no valores absolutos
- Operadores: desde Phase R1 (homeostasis, exploración, momentum...)
- Payoffs: Δ de métricas internas (ΔS, Δφ, Δattachment, Δcrisis)
- Sin palabras semánticas (cooperar/defeccionar)
- Sin magic numbers

Componentes:
- state_encoding: Estados cuánticos desde ranks
- operators_qg: Operadores estructurales
- coalition_game_qg1: Lógica del juego
- payoff_endogenous: Sistema de payoffs
- audit_q_endogenous: Auditoría de endogeneidad
- run_q_coalition_game: Experimentos
"""

from .state_encoding import QuantumStateEncoding, EntangledStateEncoding
from .operators_qg import (
    StructuralOperator,
    OperatorSelector,
    create_homeostasis_operator,
    create_exploration_operator,
    create_momentum_operator,
    create_integration_operator,
    create_crisis_operator,
    create_attachment_operator
)
from .coalition_game_qg1 import CoalitionGameQG1, AgentGameState
from .payoff_endogenous import PayoffCalculator, CooperationMetric, PayoffMatrix
from .audit_q_endogenous import EndogeneityAuditor, run_audit

__all__ = [
    'QuantumStateEncoding',
    'EntangledStateEncoding',
    'StructuralOperator',
    'OperatorSelector',
    'CoalitionGameQG1',
    'AgentGameState',
    'PayoffCalculator',
    'CooperationMetric',
    'PayoffMatrix',
    'EndogeneityAuditor',
    'run_audit'
]
