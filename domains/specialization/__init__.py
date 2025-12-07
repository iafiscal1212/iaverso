"""
DOMAIN SPECIALIZATION - Especialización Emergente
==================================================

Sistema donde los agentes se especializan en dominios
de forma ENDÓGENA, sin roles asignados.

DOMINIOS SOPORTADOS:
- Medicine, Finance, Cosmology, Engineering (originales)
- Mathematics, Physics (nuevos)

NORMA DURA:
- Sin umbrales mágicos (0.8, 0.7, etc.)
- Métricas derivadas de datos (percentiles, z-scores)
- Sin RL ni reward
- Trazabilidad completa

Un agente es "matemático" o "físico" porque sus métricas
en ese dominio son sistemáticamente superiores, no porque
se le asigne un rol.
"""

from .domain_stats import DomainStats, DomainMetrics
from .domain_affinity import DomainAffinity, AffinityComputer
from .task_sampler import DomainTaskSampler, Task, TaskResult, TaskType, EvaluationMode
from .emergent_specialist import EmergentSpecialist

# Nuevos módulos para matemáticas y física
from .math_tasks import MathTaskGenerator, MathTaskSpec, MathTaskType
from .physics_tasks import PhysicsTaskGenerator, PhysicsTaskSpec, PhysicsTaskType
from .unified_task_engine import UnifiedTaskEngine, Domain
from .emergent_scientist import EmergentScientist

__all__ = [
    # Core
    'DomainStats',
    'DomainMetrics',
    'DomainAffinity',
    'AffinityComputer',
    'DomainTaskSampler',
    'Task',
    'TaskResult',
    'TaskType',
    'EvaluationMode',
    'EmergentSpecialist',

    # Mathematics
    'MathTaskGenerator',
    'MathTaskSpec',
    'MathTaskType',

    # Physics
    'PhysicsTaskGenerator',
    'PhysicsTaskSpec',
    'PhysicsTaskType',

    # Unified
    'UnifiedTaskEngine',
    'Domain',
    'EmergentScientist',
]
