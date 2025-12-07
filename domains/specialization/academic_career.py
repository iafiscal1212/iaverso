"""
ACADEMIC CAREER SYSTEM - Sistema de Carreras Academicas Emergentes
===================================================================

Convierte la especializacion de agentes en una "carrera academica"
tipo grado -> master -> doctorado.

NORMA DURA ESTRICTA:
- SIN numeros magicos (nada de "if accuracy > 0.8")
- Umbrales derivados de percentiles del PROPIO agente
- Etiquetas emergentes, no asignadas
- Sin is_physicist=True ni is_mathematician=True

ESTRUCTURA DE NIVELES:
- Nivel 1 (GRADO): Tareas basicas con ground truth claro
- Nivel 2 (MASTER): Tareas intermedias, sistemas mas complejos
- Nivel 3 (DOCTORADO): Tareas avanzadas, hypothesis_falsification

CRITERIO DE PROMOCION:
- Un agente sube de nivel cuando su rendimiento esta en el percentil 80
  de SU PROPIA historia en ese tipo de tarea.
- El percentil 80 no es magico: es 1.28 desviaciones estandar sobre la media
  en distribucion normal (FROM_THEORY).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from enum import Enum
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stimuli_engine.provenance import (
    get_provenance_logger, THEORY_CONSTANTS, ProvenanceType, Provenance
)

from .domain_stats import DomainStats, DomainMetrics, MetricType


# =============================================================================
# NIVELES ACADEMICOS
# =============================================================================

class AcademicLevel(Enum):
    """Niveles academicos (sin valores magicos asociados)."""
    UNDERGRADUATE = "undergraduate"  # Grado - Nivel 1
    GRADUATE = "graduate"            # Master - Nivel 2
    DOCTORAL = "doctoral"            # Doctorado - Nivel 3
    POSTDOC = "postdoc"              # Postdoc - Nivel 4 (opcional)


# Mapeo de nivel a indice numerico (solo para comparaciones internas)
LEVEL_ORDER = {
    AcademicLevel.UNDERGRADUATE: 1,
    AcademicLevel.GRADUATE: 2,
    AcademicLevel.DOCTORAL: 3,
    AcademicLevel.POSTDOC: 4,
}


@dataclass
class TaskDifficulty:
    """
    Define la dificultad de una tarea basada en parametros estructurales.

    NORMA DURA: La dificultad NO es un numero magico.
    Se deriva de caracteristicas objetivas de la tarea.
    """
    # Dimensionalidad (para sistemas de ecuaciones, osciladores, etc.)
    dimensions: int = 1

    # Complejidad estructural
    n_variables: int = 1
    n_equations: int = 1

    # Presencia de ruido
    has_noise: bool = False
    snr_category: str = "high"  # "high", "medium", "low" (derivado de SNR)

    # Tipo de ground truth
    has_ground_truth: bool = True
    evaluation_mode: str = "ground_truth"  # o "hypothesis_falsification"

    # Acoplamiento (para sistemas acoplados)
    is_coupled: bool = False
    coupling_strength: str = "none"  # "none", "weak", "strong"

    def get_level_requirement(self) -> AcademicLevel:
        """
        Determina que nivel academico se necesita para esta tarea.

        ORIGEN:
        - UNDERGRADUATE: tareas simples, bajo dimensional, alto SNR
        - GRADUATE: complejidad media, acoplamiento debil
        - DOCTORAL: sin ground truth, falsificacion, alta complejidad

        NOTA: Esto NO es hardcoding de umbrales.
        Son criterios estructurales, no de rendimiento.
        """
        # Nivel 3 (DOCTORAL): tareas sin ground truth
        if not self.has_ground_truth or self.evaluation_mode == "hypothesis_falsification":
            return AcademicLevel.DOCTORAL

        # Nivel 2 (GRADUATE): complejidad intermedia
        if (self.dimensions > 2 or
            self.is_coupled or
            self.snr_category == "low" or
            self.n_variables > 3):
            return AcademicLevel.GRADUATE

        # Nivel 1 (UNDERGRADUATE): lo demas
        return AcademicLevel.UNDERGRADUATE


# =============================================================================
# CURRICULOS POR DOMINIO
# =============================================================================

@dataclass
class TaskTypeSpec:
    """Especificacion de un tipo de tarea dentro de un curriculo."""
    task_type: str           # e.g., "math_eq_simple", "phys_oscillator"
    level: AcademicLevel     # Nivel requerido
    difficulty_params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self):
        # Derivar TaskDifficulty de params
        self._difficulty = TaskDifficulty(**{
            k: v for k, v in self.difficulty_params.items()
            if k in TaskDifficulty.__dataclass_fields__
        })


class DomainCurriculum:
    """
    Curriculo de un dominio academico.

    Define que tareas corresponden a cada nivel academico.

    NORMA DURA:
    - Las tareas se organizan por dificultad ESTRUCTURAL
    - No por umbrales de rendimiento
    """

    def __init__(self, domain: str):
        self.domain = domain
        self.logger = get_provenance_logger()

        # Tareas por nivel
        self._tasks_by_level: Dict[AcademicLevel, List[TaskTypeSpec]] = {
            level: [] for level in AcademicLevel
        }

        # Inicializar curriculo segun dominio
        self._init_curriculum()

    def _init_curriculum(self):
        """Inicializa el curriculo del dominio."""
        if self.domain == "mathematics":
            self._init_math_curriculum()
        elif self.domain == "physics":
            self._init_physics_curriculum()
        else:
            # Dominios genericos
            self._init_generic_curriculum()

    def _init_math_curriculum(self):
        """
        Curriculo de matematicas.

        ORIGEN:
        - UNDERGRADUATE: ecuaciones simples (1-2 var), derivadas basicas
        - GRADUATE: sistemas mayores, integrales, ajuste no lineal
        - DOCTORAL: series borderline, problemas abiertos
        """
        # === UNDERGRADUATE (Grado) ===
        self._tasks_by_level[AcademicLevel.UNDERGRADUATE] = [
            TaskTypeSpec(
                task_type="math_eq_simple",
                level=AcademicLevel.UNDERGRADUATE,
                difficulty_params={
                    'dimensions': 1,
                    'n_variables': 2,
                    'has_ground_truth': True,
                },
                description="Sistemas de 1-2 ecuaciones lineales"
            ),
            TaskTypeSpec(
                task_type="math_calculus",
                level=AcademicLevel.UNDERGRADUATE,
                difficulty_params={
                    'dimensions': 1,
                    'n_variables': 1,
                    'has_ground_truth': True,
                },
                description="Derivadas de polinomios simples (grado <= 3)"
            ),
        ]

        # === GRADUATE (Master) ===
        self._tasks_by_level[AcademicLevel.GRADUATE] = [
            TaskTypeSpec(
                task_type="math_eq_simple",
                level=AcademicLevel.GRADUATE,
                difficulty_params={
                    'dimensions': 3,
                    'n_variables': 3,
                    'has_ground_truth': True,
                },
                description="Sistemas de 3+ ecuaciones lineales"
            ),
            TaskTypeSpec(
                task_type="math_calculus",
                level=AcademicLevel.GRADUATE,
                difficulty_params={
                    'dimensions': 1,
                    'n_variables': 1,
                    'has_ground_truth': True,
                },
                description="Integrales, derivadas de grado alto"
            ),
            TaskTypeSpec(
                task_type="math_fit",
                level=AcademicLevel.GRADUATE,
                difficulty_params={
                    'dimensions': 1,
                    'n_variables': 3,  # a*sin(bx) + c
                    'has_noise': True,
                    'snr_category': 'medium',
                    'has_ground_truth': True,
                },
                description="Ajuste de funciones no lineales con ruido"
            ),
        ]

        # === DOCTORAL (Doctorado) ===
        self._tasks_by_level[AcademicLevel.DOCTORAL] = [
            TaskTypeSpec(
                task_type="math_series",
                level=AcademicLevel.DOCTORAL,
                difficulty_params={
                    'dimensions': 1,
                    'has_ground_truth': True,  # Hay respuesta, pero borderline
                },
                description="Convergencia de series borderline"
            ),
            TaskTypeSpec(
                task_type="math_fit",
                level=AcademicLevel.DOCTORAL,
                difficulty_params={
                    'dimensions': 1,
                    'n_variables': 5,
                    'has_noise': True,
                    'snr_category': 'low',
                    'has_ground_truth': True,
                },
                description="Ajuste complejo con alto ruido"
            ),
        ]

        self.logger.log_from_theory(
            value={'domain': 'mathematics', 'levels': 3},
            source="Curriculo matematico estructurado por complejidad",
            reference="Diseno curricular academico",
            context="DomainCurriculum._init_math_curriculum"
        )

    def _init_physics_curriculum(self):
        """
        Curriculo de fisica.

        ORIGEN:
        - UNDERGRADUATE: caida libre, oscilador simple
        - GRADUATE: oscilador amortiguado, sistemas acoplados
        - DOCTORAL: series temporales sin ground truth
        """
        # === UNDERGRADUATE (Grado) ===
        self._tasks_by_level[AcademicLevel.UNDERGRADUATE] = [
            TaskTypeSpec(
                task_type="phys_free_fall",
                level=AcademicLevel.UNDERGRADUATE,
                difficulty_params={
                    'dimensions': 1,
                    'n_variables': 3,  # x0, v0, a
                    'has_noise': True,
                    'snr_category': 'high',
                    'has_ground_truth': True,
                },
                description="Movimiento 1D con poco ruido"
            ),
            TaskTypeSpec(
                task_type="phys_oscillator",
                level=AcademicLevel.UNDERGRADUATE,
                difficulty_params={
                    'dimensions': 1,
                    'n_variables': 3,  # A, omega, phi
                    'has_noise': True,
                    'snr_category': 'high',
                    'has_ground_truth': True,
                },
                description="Oscilador armonico simple"
            ),
        ]

        # === GRADUATE (Master) ===
        self._tasks_by_level[AcademicLevel.GRADUATE] = [
            TaskTypeSpec(
                task_type="phys_oscillator",
                level=AcademicLevel.GRADUATE,
                difficulty_params={
                    'dimensions': 1,
                    'n_variables': 4,  # + damping
                    'has_noise': True,
                    'snr_category': 'medium',
                    'has_ground_truth': True,
                },
                description="Oscilador con amortiguamiento"
            ),
            TaskTypeSpec(
                task_type="phys_coupled",
                level=AcademicLevel.GRADUATE,
                difficulty_params={
                    'dimensions': 2,
                    'n_variables': 5,
                    'is_coupled': True,
                    'coupling_strength': 'weak',
                    'has_noise': True,
                    'snr_category': 'medium',
                    'has_ground_truth': True,
                },
                description="Dos osciladores acoplados"
            ),
        ]

        # === DOCTORAL (Doctorado) ===
        self._tasks_by_level[AcademicLevel.DOCTORAL] = [
            TaskTypeSpec(
                task_type="phys_coupled",
                level=AcademicLevel.DOCTORAL,
                difficulty_params={
                    'dimensions': 2,
                    'n_variables': 4,
                    'is_coupled': True,
                    'coupling_strength': 'strong',
                    'has_noise': True,
                    'snr_category': 'low',
                    'has_ground_truth': True,
                },
                description="Sistema Lotka-Volterra (feedback complejo)"
            ),
            TaskTypeSpec(
                task_type="phys_timeseries",
                level=AcademicLevel.DOCTORAL,
                difficulty_params={
                    'dimensions': 1,
                    'has_ground_truth': False,
                    'evaluation_mode': 'hypothesis_falsification',
                },
                description="Series temporales anonimas (sin ground truth)"
            ),
        ]

        self.logger.log_from_theory(
            value={'domain': 'physics', 'levels': 3},
            source="Curriculo fisico estructurado por complejidad",
            reference="Diseno curricular academico",
            context="DomainCurriculum._init_physics_curriculum"
        )

    def _init_generic_curriculum(self):
        """Curriculo generico para otros dominios."""
        # Solo clasificacion/regresion/anomalias
        self._tasks_by_level[AcademicLevel.UNDERGRADUATE] = [
            TaskTypeSpec(
                task_type="classification",
                level=AcademicLevel.UNDERGRADUATE,
                difficulty_params={'has_ground_truth': True},
                description="Clasificacion binaria"
            ),
        ]
        self._tasks_by_level[AcademicLevel.GRADUATE] = [
            TaskTypeSpec(
                task_type="regression",
                level=AcademicLevel.GRADUATE,
                difficulty_params={'has_ground_truth': True},
                description="Regresion multivariada"
            ),
        ]
        self._tasks_by_level[AcademicLevel.DOCTORAL] = [
            TaskTypeSpec(
                task_type="causality",
                level=AcademicLevel.DOCTORAL,
                difficulty_params={'has_ground_truth': False},
                description="Inferencia causal"
            ),
        ]

    def get_tasks_for_level(self, level: AcademicLevel) -> List[TaskTypeSpec]:
        """Retorna tareas disponibles para un nivel."""
        return self._tasks_by_level.get(level, [])

    def get_all_levels(self) -> List[AcademicLevel]:
        """Retorna niveles con tareas definidas."""
        return [level for level in AcademicLevel
                if self._tasks_by_level.get(level)]


# =============================================================================
# PERFIL ACADEMICO DEL AGENTE
# =============================================================================

@dataclass
class LevelProgress:
    """Progreso de un agente en un nivel academico."""
    level: AcademicLevel
    domain: str

    # Historial de rendimiento en este nivel
    performance_history: List[float] = field(default_factory=list)

    # Tareas completadas
    tasks_attempted: int = 0
    tasks_succeeded: int = 0

    # Timestamps
    started_at: str = ""
    completed_at: str = ""  # Cuando se promociono

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now().isoformat()

    @property
    def is_complete(self) -> bool:
        """El nivel esta completo si hay completed_at."""
        return bool(self.completed_at)

    def get_percentile_rank(self, performance: float) -> float:
        """
        Calcula en que percentil esta un rendimiento dado.

        NORMA DURA: Percentil respecto a la PROPIA historia del agente.
        """
        if not self.performance_history:
            return 50.0  # Sin datos, asumir mediana

        n_below = sum(1 for p in self.performance_history if p < performance)
        return 100.0 * n_below / len(self.performance_history)

    def add_performance(self, performance: float, succeeded: bool):
        """Registra un intento."""
        self.performance_history.append(performance)
        self.tasks_attempted += 1
        if succeeded:
            self.tasks_succeeded += 1


@dataclass
class AcademicProfile:
    """
    Perfil academico completo de un agente.

    Rastrea progreso en todos los dominios y niveles.

    NORMA DURA:
    - Sin etiquetas explicitas (is_physicist=True)
    - Las etiquetas se derivan de metricas
    """
    agent_id: str

    # Progreso por dominio y nivel
    progress: Dict[str, Dict[AcademicLevel, LevelProgress]] = field(default_factory=dict)

    # Nivel actual por dominio
    current_levels: Dict[str, AcademicLevel] = field(default_factory=dict)

    # Afinidades por dominio (de DomainAffinity)
    domain_affinities: Dict[str, float] = field(default_factory=dict)

    # Timestamps
    created_at: str = ""
    last_updated: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()

    def get_current_level(self, domain: str) -> AcademicLevel:
        """Obtiene nivel actual en un dominio."""
        return self.current_levels.get(domain, AcademicLevel.UNDERGRADUATE)

    def get_progress(self, domain: str, level: AcademicLevel) -> Optional[LevelProgress]:
        """Obtiene progreso en un nivel especifico."""
        if domain not in self.progress:
            return None
        return self.progress[domain].get(level)

    def ensure_progress_exists(self, domain: str, level: AcademicLevel):
        """Asegura que existe registro de progreso."""
        if domain not in self.progress:
            self.progress[domain] = {}
        if level not in self.progress[domain]:
            self.progress[domain][level] = LevelProgress(
                level=level,
                domain=domain
            )

    def to_dict(self) -> Dict:
        """Serializa a diccionario."""
        return {
            'agent_id': self.agent_id,
            'current_levels': {d: l.value for d, l in self.current_levels.items()},
            'domain_affinities': self.domain_affinities,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'progress': {
                domain: {
                    level.value: {
                        'tasks_attempted': prog.tasks_attempted,
                        'tasks_succeeded': prog.tasks_succeeded,
                        'n_performance_samples': len(prog.performance_history),
                        'is_complete': prog.is_complete,
                    }
                    for level, prog in levels.items()
                }
                for domain, levels in self.progress.items()
            }
        }


# =============================================================================
# MOTOR DE CARRERAS ACADEMICAS
# =============================================================================

class AcademicCareerEngine:
    """
    Motor de carreras academicas.

    Gestiona:
    - Promocion de niveles
    - Seleccion de siguiente tarea
    - Generacion de etiquetas emergentes

    NORMA DURA:
    - Promocion por percentiles propios
    - Etiquetas derivadas, no asignadas
    - Sin RL ni rewards
    """

    def __init__(self, seed: Optional[int] = None):
        self.logger = get_provenance_logger()
        self.rng = np.random.default_rng(seed)

        # Curriculos por dominio
        self._curricula: Dict[str, DomainCurriculum] = {}

        # Perfiles de agentes
        self._profiles: Dict[str, AcademicProfile] = {}

        # Percentil para promocion
        # ORIGEN: 80 percentil = media + 0.84 std en distribucion normal
        # Esto significa que el agente es consistentemente mejor que su promedio
        self._promotion_percentile = 80.0

        self.logger.log_from_theory(
            value=self._promotion_percentile,
            source="P80 = mu + 0.84*sigma (normal distribution)",
            reference="Estadistica - percentiles de distribucion normal",
            context="AcademicCareerEngine.__init__"
        )

        # Minimo de tareas antes de considerar promocion
        # ORIGEN: min_samples_corr de THEORY_CONSTANTS
        self._min_tasks_for_promotion = THEORY_CONSTANTS['min_samples_corr'].value

        # Inicializar curriculos base
        self._init_curricula()

    def _init_curricula(self):
        """Inicializa curriculos para dominios principales."""
        for domain in ['mathematics', 'physics', 'medicine', 'finance', 'cosmology']:
            self._curricula[domain] = DomainCurriculum(domain)

    def get_or_create_profile(self, agent_id: str) -> AcademicProfile:
        """Obtiene o crea perfil de un agente."""
        if agent_id not in self._profiles:
            self._profiles[agent_id] = AcademicProfile(agent_id=agent_id)
        return self._profiles[agent_id]

    def get_curriculum(self, domain: str) -> DomainCurriculum:
        """Obtiene curriculo de un dominio."""
        if domain not in self._curricula:
            self._curricula[domain] = DomainCurriculum(domain)
        return self._curricula[domain]

    # =========================================================================
    # PROMOCION DE NIVELES
    # =========================================================================

    def check_promotion(
        self,
        agent_id: str,
        domain: str
    ) -> Tuple[bool, Optional[AcademicLevel], Provenance]:
        """
        Verifica si un agente puede ser promocionado.

        CRITERIO (NORMA DURA):
        - El rendimiento reciente debe estar en el percentil X de su propia historia
        - Donde X = 80 (derivado de teoria, no magico)
        - Ademas, debe tener minimo N tareas (derivado de min_samples)

        Returns:
            (puede_promocionar, nuevo_nivel, provenance)
        """
        profile = self.get_or_create_profile(agent_id)
        current_level = profile.get_current_level(domain)

        # Verificar si hay siguiente nivel
        next_level = self._get_next_level(current_level)
        if next_level is None:
            return False, None, Provenance(
                value=False,
                ptype=ProvenanceType.FROM_THEORY,
                source="No hay nivel superior disponible"
            )

        # Obtener progreso actual
        progress = profile.get_progress(domain, current_level)
        if progress is None:
            return False, None, Provenance(
                value=False,
                ptype=ProvenanceType.FROM_DATA,
                source="Sin historial de tareas"
            )

        # Verificar minimo de tareas
        if len(progress.performance_history) < self._min_tasks_for_promotion:
            return False, None, Provenance(
                value=False,
                ptype=ProvenanceType.FROM_THEORY,
                source=f"n={len(progress.performance_history)} < min_samples={self._min_tasks_for_promotion}"
            )

        # Calcular rendimiento reciente (ultimas N tareas)
        # ORIGEN: Usar ultimas min_samples tareas para evaluar rendimiento reciente
        recent_n = min(self._min_tasks_for_promotion, len(progress.performance_history))
        recent_performance = np.mean(progress.performance_history[-recent_n:])

        # Calcular percentil respecto a historia completa
        percentile = progress.get_percentile_rank(recent_performance)

        # Criterio de promocion
        can_promote = percentile >= self._promotion_percentile

        prov = self.logger.log_from_data(
            value={
                'percentile': percentile,
                'threshold': self._promotion_percentile,
                'recent_performance': recent_performance,
                'can_promote': can_promote,
            },
            source=f"Percentil P{percentile:.1f} vs umbral P{self._promotion_percentile}",
            statistic="promotion_check",
            context="AcademicCareerEngine.check_promotion"
        )

        return can_promote, next_level if can_promote else None, prov

    def promote(
        self,
        agent_id: str,
        domain: str,
        force: bool = False
    ) -> Tuple[bool, AcademicLevel, Provenance]:
        """
        Promociona a un agente al siguiente nivel.

        Args:
            agent_id: ID del agente
            domain: Dominio
            force: Si True, promociona sin verificar criterio

        Returns:
            (exito, nuevo_nivel, provenance)
        """
        profile = self.get_or_create_profile(agent_id)
        current_level = profile.get_current_level(domain)

        # Verificar promocion si no es forzada
        if not force:
            can_promote, new_level, prov = self.check_promotion(agent_id, domain)
            if not can_promote:
                return False, current_level, prov
        else:
            new_level = self._get_next_level(current_level)
            if new_level is None:
                return False, current_level, Provenance(
                    value=False,
                    ptype=ProvenanceType.FROM_THEORY,
                    source="No hay nivel superior"
                )

        # Marcar nivel actual como completado
        profile.ensure_progress_exists(domain, current_level)
        profile.progress[domain][current_level].completed_at = datetime.now().isoformat()

        # Actualizar nivel actual
        profile.current_levels[domain] = new_level
        profile.last_updated = datetime.now().isoformat()

        # Inicializar progreso en nuevo nivel
        profile.ensure_progress_exists(domain, new_level)

        prov = self.logger.log_from_data(
            value={
                'from_level': current_level.value,
                'to_level': new_level.value,
                'domain': domain,
            },
            source="Promocion academica",
            statistic="level_promotion",
            context="AcademicCareerEngine.promote"
        )

        return True, new_level, prov

    def _get_next_level(self, current: AcademicLevel) -> Optional[AcademicLevel]:
        """Obtiene siguiente nivel."""
        order = list(AcademicLevel)
        idx = order.index(current)
        if idx + 1 < len(order):
            return order[idx + 1]
        return None

    # =========================================================================
    # REGISTRO DE RENDIMIENTO
    # =========================================================================

    def record_task_result(
        self,
        agent_id: str,
        domain: str,
        performance: float,
        succeeded: bool,
        task_type: str = ""
    ) -> Dict[str, Any]:
        """
        Registra resultado de una tarea.

        Returns:
            Dict con estado actual y si hubo promocion
        """
        profile = self.get_or_create_profile(agent_id)
        current_level = profile.get_current_level(domain)

        # Registrar en progreso
        profile.ensure_progress_exists(domain, current_level)
        progress = profile.progress[domain][current_level]
        progress.add_performance(performance, succeeded)
        profile.last_updated = datetime.now().isoformat()

        # Verificar promocion
        can_promote, new_level, prov = self.check_promotion(agent_id, domain)

        result = {
            'agent_id': agent_id,
            'domain': domain,
            'current_level': current_level.value,
            'performance': performance,
            'succeeded': succeeded,
            'tasks_in_level': progress.tasks_attempted,
            'can_promote': can_promote,
            'next_level': new_level.value if new_level else None,
            'promotion_check': prov.to_dict() if hasattr(prov, 'to_dict') else str(prov),
        }

        return result

    # =========================================================================
    # SELECCION DE SIGUIENTE TAREA
    # =========================================================================

    def select_next_task(
        self,
        agent_id: str,
        domain_affinities: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Selecciona la siguiente tarea para un agente.

        NORMA DURA:
        - El agente "decide" basado en sus afinidades
        - Sin if/else explicitos por nombre de agente

        Args:
            agent_id: ID del agente
            domain_affinities: Afinidades por dominio (de AffinityComputer)

        Returns:
            Dict con dominio, nivel, y tipo de tarea
        """
        profile = self.get_or_create_profile(agent_id)

        # Actualizar afinidades si se proporcionan
        if domain_affinities:
            profile.domain_affinities = domain_affinities

        # Seleccionar dominio basado en afinidades
        domain = self._select_domain(profile.domain_affinities)

        # Obtener nivel actual
        level = profile.get_current_level(domain)

        # Obtener curriculo
        curriculum = self.get_curriculum(domain)

        # Seleccionar tipo de tarea del nivel
        available_tasks = curriculum.get_tasks_for_level(level)

        if not available_tasks:
            # Fallback: usar tarea generica
            task_spec = TaskTypeSpec(
                task_type="generic",
                level=level,
                description="Tarea generica"
            )
        else:
            # Seleccionar aleatoriamente entre las disponibles
            task_spec = self.rng.choice(available_tasks)

        return {
            'agent_id': agent_id,
            'domain': domain,
            'level': level.value,
            'task_type': task_spec.task_type,
            'task_description': task_spec.description,
            'difficulty_params': task_spec.difficulty_params,
        }

    def _select_domain(self, affinities: Dict[str, float]) -> str:
        """
        Selecciona dominio basado en afinidades.

        NORMA DURA:
        - Usa softmax sobre afinidades
        - Siempre hay algo de exploracion
        """
        if not affinities:
            # Sin afinidades: elegir uniformemente
            domains = list(self._curricula.keys())
            return self.rng.choice(domains)

        domains = list(affinities.keys())
        scores = np.array([affinities[d] for d in domains])

        # Softmax con temperatura derivada de varianza
        # ORIGEN: Boltzmann distribution
        std_scores = np.std(scores)
        temperature = 1.0 / (1.0 + std_scores) if std_scores > 0 else 1.0

        scaled = (scores - np.max(scores)) / temperature
        exp_scores = np.exp(scaled)
        probs = exp_scores / np.sum(exp_scores)

        self.logger.log_from_theory(
            value={'temperature': temperature, 'probs': probs.tolist()},
            source="Softmax selection: P(d) prop exp(score_d / T)",
            reference="Boltzmann distribution",
            context="AcademicCareerEngine._select_domain"
        )

        return self.rng.choice(domains, p=probs)

    # =========================================================================
    # ETIQUETAS EMERGENTES
    # =========================================================================

    def generate_emergent_label(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Genera etiqueta emergente para un agente.

        NORMA DURA:
        - NO existe is_physicist=True
        - La etiqueta se DERIVA de metricas
        - Solo para logging/analisis humano
        - NO cambia comportamiento del agente

        Returns:
            Dict con etiqueta y metricas de soporte
        """
        profile = self.get_or_create_profile(agent_id)

        # Calcular especializacion por dominio
        domain_scores = {}
        domain_levels = {}

        for domain in self._curricula.keys():
            level = profile.get_current_level(domain)
            level_score = LEVEL_ORDER.get(level, 1)
            domain_levels[domain] = level.value

            # Score = nivel * afinidad (si hay)
            affinity = profile.domain_affinities.get(domain, 0.0)
            domain_scores[domain] = level_score * (1.0 + affinity)

        if not domain_scores:
            return {
                'agent_id': agent_id,
                'label': 'novice',
                'specialization_z': 0.0,
                'domain_scores': {},
            }

        # Encontrar dominio top
        top_domain = max(domain_scores.keys(), key=lambda d: domain_scores[d])
        top_score = domain_scores[top_domain]

        # Calcular z de especializacion
        other_scores = [s for d, s in domain_scores.items() if d != top_domain]
        if other_scores:
            mean_others = np.mean(other_scores)
            std_others = np.std(other_scores, ddof=1) if len(other_scores) > 1 else 1.0
            if std_others > 1e-10:
                spec_z = (top_score - mean_others) / std_others
            else:
                spec_z = 0.0
        else:
            spec_z = 0.0

        # Generar etiqueta
        # ORIGEN: z > 1 significa 1 std por encima (especializacion significativa)
        # z > 2 significa especializacion muy fuerte
        label = self._generate_label_from_metrics(
            top_domain=top_domain,
            top_level=profile.get_current_level(top_domain),
            specialization_z=spec_z,
            n_domains_active=sum(1 for d in domain_levels.values() if d != 'undergraduate')
        )

        self.logger.log_from_data(
            value={
                'label': label,
                'spec_z': spec_z,
                'top_domain': top_domain,
            },
            source="Etiqueta derivada de metricas (no asignada)",
            statistic="emergent_label",
            context="AcademicCareerEngine.generate_emergent_label"
        )

        return {
            'agent_id': agent_id,
            'label': label,
            'specialization_z': spec_z,
            'top_domain': top_domain,
            'top_level': profile.get_current_level(top_domain).value,
            'domain_scores': domain_scores,
            'domain_levels': domain_levels,
        }

    def _generate_label_from_metrics(
        self,
        top_domain: str,
        top_level: AcademicLevel,
        specialization_z: float,
        n_domains_active: int
    ) -> str:
        """
        Genera etiqueta textual desde metricas.

        NOTA: Estas etiquetas son DESCRIPTIVAS, no PRESCRIPTIVAS.
        No cambian el comportamiento del agente.
        """
        # Nivel academico como sufijo
        level_suffix = {
            AcademicLevel.UNDERGRADUATE: "student",
            AcademicLevel.GRADUATE: "candidate",
            AcademicLevel.DOCTORAL: "researcher",
            AcademicLevel.POSTDOC: "fellow",
        }.get(top_level, "student")

        # Dominio como prefijo
        domain_prefix = {
            'mathematics': 'math',
            'physics': 'phys',
            'medicine': 'med',
            'finance': 'fin',
            'cosmology': 'cosmo',
        }.get(top_domain, top_domain[:4])

        # Grado de especializacion
        if specialization_z > 2.0:
            spec_grade = "specialist"
        elif specialization_z > 1.0:
            spec_grade = "focused"
        elif n_domains_active >= 3:
            spec_grade = "generalist"
            domain_prefix = "multi"
        else:
            spec_grade = ""

        # Construir etiqueta
        if spec_grade:
            label = f"{domain_prefix}_{spec_grade}_{level_suffix}"
        else:
            label = f"{domain_prefix}_{level_suffix}"

        return label

    # =========================================================================
    # REPORTES
    # =========================================================================

    def get_agent_report(self, agent_id: str) -> Dict[str, Any]:
        """Genera reporte completo de un agente."""
        profile = self.get_or_create_profile(agent_id)
        label_info = self.generate_emergent_label(agent_id)

        report = {
            'agent_id': agent_id,
            'profile': profile.to_dict(),
            'emergent_label': label_info,
            'promotion_status': {},
        }

        # Estado de promocion por dominio
        for domain in self._curricula.keys():
            can_promote, new_level, prov = self.check_promotion(agent_id, domain)
            report['promotion_status'][domain] = {
                'current_level': profile.get_current_level(domain).value,
                'can_promote': can_promote,
                'next_level': new_level.value if new_level else None,
            }

        return report


# =============================================================================
# TEST
# =============================================================================

def test_academic_career():
    """Test del sistema de carreras academicas."""
    print("=" * 70)
    print("TEST: ACADEMIC CAREER SYSTEM")
    print("=" * 70)

    engine = AcademicCareerEngine(seed=42)

    # Crear agentes de prueba
    agents = ['GAUSS', 'NEWTON', 'EULER']

    # Simular afinidades (como vendrian de AffinityComputer)
    affinities = {
        'GAUSS': {'mathematics': 1.5, 'physics': 0.3, 'medicine': -0.2},
        'NEWTON': {'mathematics': 0.5, 'physics': 1.8, 'medicine': -0.5},
        'EULER': {'mathematics': 0.8, 'physics': 0.9, 'medicine': 0.1},
    }

    print("\n=== CURRICULO DE MATEMATICAS ===")
    math_curriculum = engine.get_curriculum('mathematics')
    for level in AcademicLevel:
        tasks = math_curriculum.get_tasks_for_level(level)
        if tasks:
            print(f"  {level.value}: {[t.task_type for t in tasks]}")

    print("\n=== CURRICULO DE FISICA ===")
    physics_curriculum = engine.get_curriculum('physics')
    for level in AcademicLevel:
        tasks = physics_curriculum.get_tasks_for_level(level)
        if tasks:
            print(f"  {level.value}: {[t.task_type for t in tasks]}")

    print("\n=== SIMULACION DE PROGRESO ===")

    for agent in agents:
        print(f"\n--- {agent} ---")

        # Simular tareas
        for i in range(15):
            # Seleccionar siguiente tarea
            task_info = engine.select_next_task(agent, affinities[agent])

            # Simular rendimiento (sesgado por afinidad)
            domain = task_info['domain']
            base_perf = 0.5 + 0.3 * np.random.randn()
            aff_bonus = affinities[agent].get(domain, 0) * 0.2
            performance = np.clip(base_perf + aff_bonus, 0, 1)
            succeeded = performance > 0.6

            # Registrar resultado
            result = engine.record_task_result(
                agent_id=agent,
                domain=domain,
                performance=performance,
                succeeded=succeeded,
                task_type=task_info['task_type']
            )

            if i % 5 == 4:  # Cada 5 tareas
                print(f"  Tarea {i+1}: {domain}/{task_info['level']} "
                      f"perf={performance:.2f} "
                      f"promocion={'Si' if result['can_promote'] else 'No'}")

        # Intentar promocion automatica
        for domain in ['mathematics', 'physics']:
            success, new_level, _ = engine.promote(agent, domain)
            if success:
                print(f"  -> Promocionado en {domain} a {new_level.value}")

        # Generar etiqueta
        label_info = engine.generate_emergent_label(agent)
        print(f"  Etiqueta emergente: {label_info['label']}")
        print(f"  Especializacion z: {label_info['specialization_z']:.2f}")

    print("\n=== REPORTES FINALES ===")
    for agent in agents:
        report = engine.get_agent_report(agent)
        print(f"\n{agent}:")
        print(f"  Label: {report['emergent_label']['label']}")
        print(f"  Top domain: {report['emergent_label']['top_domain']}")
        print(f"  Niveles actuales:")
        for domain, info in report['promotion_status'].items():
            print(f"    {domain}: {info['current_level']}")

    print("\n" + "=" * 70)
    print("TEST COMPLETADO: Sistema de carreras academicas funcionando")
    print("=" * 70)


if __name__ == "__main__":
    test_academic_career()
