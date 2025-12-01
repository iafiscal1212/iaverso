"""
Integración Global Certificada (IGC)
====================================

Módulos:
- phaseI1_subsystems: Descomposición en sub-sistemas + matriz de acoplos
- phaseI2_igi: Índice de Integración Global
"""

from .phaseI1_subsystems import SubsystemDecomposition
from .phaseI2_igi import GlobalIntegrationIndex

__all__ = ['SubsystemDecomposition', 'GlobalIntegrationIndex']
