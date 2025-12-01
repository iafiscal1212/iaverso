"""
WORLD-1: Internal Universe for Autonomous Agents

A mathematical mini-universe where NEO, EVA, ALEX, ADAM, and IRIS live,
predict, adapt, and make decisions - all within code.

No human semantics (no "food", "enemy", etc.)
All parameters endogenous (from percentiles, covariances, eigenvalues)
Connection only via: observations o_t, actions a_t, and existing metrics
"""

from .world1_core import World1Core, WorldState
from .world1_entities import Entity, EntityPopulation
from .world1_observation import ObservationProjector
from .world1_actions import ActionMapper
from .world1_metrics import WorldMetrics
from .world1_regimes import RegimeDetector

__all__ = [
    'World1Core', 'WorldState',
    'Entity', 'EntityPopulation',
    'ObservationProjector',
    'ActionMapper',
    'WorldMetrics',
    'RegimeDetector'
]
