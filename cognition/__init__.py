"""
COGNITION: Internal Cognitive Architecture for Autonomous Agents
================================================================

Complete cognitive system with:

BASE MODULES:
- Episodic memory (segmentation, encoding, persistence)
- Narrative memory (transitions, dominant chains)
- Temporal tree (proto-future simulation)
- Self-model and Theory of Mind
- Compound goals and planning
- Emergent symbols from consequences
- Long-term regulation and metacognition

AGI MODULES (1-4):
- AGI-1: Global Workspace (acceso global, competencia, broadcasting)
- AGI-2: Self Narrative Loop (identidad continua, bucle autorreferente)
- AGI-3: Persistent Goals (teleología interna, metas estables)
- AGI-4: Life Trajectory (regulación teleológica, evaluación vital)

AGI MODULES (5-10):
- AGI-5: Dynamic Metacognition (auto-evaluación de procesos cognitivos)
- AGI-6: Structural Skills (habilidades reutilizables emergentes)
- AGI-7: Cross-World Generalization (generalización entre regímenes)
- AGI-8: Internal Concepts (grafo de co-ocurrencias)
- AGI-9: Long-Term Projects (cadenas narrativas como proyectos)
- AGI-10: Reflexive Equilibrium (zonas prohibidas, auto-restricciones)

AGI MODULES (11-15):
- AGI-11: Counterfactual Selves (yos alternativos simulados)
- AGI-12: Norm Emergence (normas emergentes multi-agente)
- AGI-13: Structural Curiosity (curiosidad endógena)
- AGI-14: Introspective Uncertainty (calibración de confianza)
- AGI-15: Structural Ethics (minimización de daño estructural)

All 100% ENDOGENOUS - no magic constants.
Parameters derive from: percentiles, ranks, √t, covariances.
"""

# Base modules
from .episodic_memory import EpisodicMemory, Episode
from .narrative_memory import NarrativeMemory
from .temporal_tree import TemporalTree
from .self_model import SelfModel, TheoryOfMind
from .compound_goals import CompoundGoals, GoalPlanner
from .emergent_symbols import EmergentSymbols, SymbolGrounding
from .regulation import LongTermRegulation, Metacognition, IntegratedRegulation

# AGI modules (1-4)
from .global_workspace import GlobalWorkspace, MultiAgentGlobalWorkspace, ContentType
from .self_narrative_loop import SelfNarrativeLoop, SelfState
from .persistent_goals import PersistentGoals, TeleologicalAgent, GoalStatus
from .life_trajectory import LifeTrajectory, LifePhase
from .soft_hook import SoftHook, DifferentiatedSoftHook, EpisodeRegion, EpisodeCharacterization

# AGI modules (5-10)
from .agi5_metacognition import DynamicMetacognition, CognitiveProcess, MetacognitiveState
from .agi6_skills import StructuralSkills, Skill, SkillActivation
from .agi7_generalization import CrossWorldGeneralization, WorldRegime, GeneralizableItem
from .agi8_concepts import ConceptGraph, EmergentConcept, ItemType
from .agi9_projects import LongTermProjects, Project, ProjectStatus
from .agi10_equilibrium import ReflexiveEquilibrium, PolicyType, NoGoZone

# AGI modules (11-15)
from .agi11_counterfactual import CounterfactualSelves, CounterfactualSelf, CounterfactualAnalysis
from .agi12_norms import NormEmergence, EmergentNorm
from .agi13_curiosity import StructuralCuriosity, CuriosityState, CuriosityTarget
from .agi14_uncertainty import IntrospectiveUncertainty, PredictionChannel, UncertaintyState
from .agi15_ethics import StructuralEthics, HarmMetrics, NoGoConfiguration

__all__ = [
    # Base
    'EpisodicMemory', 'Episode',
    'NarrativeMemory',
    'TemporalTree',
    'SelfModel', 'TheoryOfMind',
    'CompoundGoals', 'GoalPlanner',
    'EmergentSymbols', 'SymbolGrounding',
    'LongTermRegulation', 'Metacognition', 'IntegratedRegulation',
    # AGI 1-4
    'GlobalWorkspace', 'MultiAgentGlobalWorkspace', 'ContentType',
    'SelfNarrativeLoop', 'SelfState',
    'PersistentGoals', 'TeleologicalAgent', 'GoalStatus',
    'LifeTrajectory', 'LifePhase',
    'SoftHook', 'DifferentiatedSoftHook', 'EpisodeRegion', 'EpisodeCharacterization',
    # AGI 5-10
    'DynamicMetacognition', 'CognitiveProcess', 'MetacognitiveState',
    'StructuralSkills', 'Skill', 'SkillActivation',
    'CrossWorldGeneralization', 'WorldRegime', 'GeneralizableItem',
    'ConceptGraph', 'EmergentConcept', 'ItemType',
    'LongTermProjects', 'Project', 'ProjectStatus',
    'ReflexiveEquilibrium', 'PolicyType', 'NoGoZone',
    # AGI 11-15
    'CounterfactualSelves', 'CounterfactualSelf', 'CounterfactualAnalysis',
    'NormEmergence', 'EmergentNorm',
    'StructuralCuriosity', 'CuriosityState', 'CuriosityTarget',
    'IntrospectiveUncertainty', 'PredictionChannel', 'UncertaintyState',
    'StructuralEthics', 'HarmMetrics', 'NoGoConfiguration'
]
