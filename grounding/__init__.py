"""
Grounding: Conexión con el mundo estructurado
=============================================

Módulos:
- phaseG1_world_channel: Canal sensorial estructurado
- phaseG2_grounding: Tests de grounding (predictive, symbolic, value)
"""

from .phaseG1_world_channel import StructuredWorldChannel
from .phaseG2_grounding import GroundingTests

__all__ = ['StructuredWorldChannel', 'GroundingTests']
