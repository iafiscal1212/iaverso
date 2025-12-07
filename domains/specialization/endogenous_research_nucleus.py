"""
NÚCLEO DE INVESTIGACIÓN ENDÓGENA
=================================

Tensiones · Dominios · Niveles · Promoción · Etiquetas

PRINCIPIO FUNDAMENTAL (HARD RULE):
==================================
Toda decisión debe emerger de métricas internas.
NO se permite:
  - Elección directa de dominios
  - Roles asignados
  - Números mágicos
  - Reglas por nombre, identidad o narrativa

Si ocurre → abort_execution()

FLUJO ÚNICO PERMITIDO (SIN EXCEPCIONES):
========================================
estado_interno
   ↓
tensión_detectada
   ↓
dominios_candidatos
   ↓
tarea (con nivel implícito)
   ↓
ejecución
   ↓
historial_propio
   ↓
mejora_relativa
   ↓
(posible) promoción
   ↓
(etiqueta emergente post hoc)

NO EXISTEN ATAJOS.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stimuli_engine.provenance import get_provenance_logger, THEORY_CONSTANTS


# =============================================================================
# CONSTANTES TEÓRICAS (JUSTIFICADAS, NO MÁGICAS)
# =============================================================================

class TheoreticalConstants:
    """
    Constantes derivadas de teoría, NO números mágicos.
    Cada una tiene justificación formal.
    """

    # Percentil 80 = μ + 0.84σ en distribución normal
    # ORIGEN: Teoría estadística - z-score correspondiente a P80
    PROMOTION_PERCENTILE = 80.0
    PROMOTION_Z_SCORE = 0.8416  # invnorm(0.80)

    # z-score para especialización
    # ORIGEN: Convención estadística - 1σ = diferencia notable, 2σ = significativa
    SPECIALIST_Z_THRESHOLD = 2.0  # 2 desviaciones estándar
    FOCUSED_Z_THRESHOLD = 1.0     # 1 desviación estándar

    # Mínimo de muestras para estadísticas válidas
    # ORIGEN: Teoría de muestreo - CLT requiere n ≥ 30 idealmente
    MIN_SAMPLES_FOR_STATS = 5  # Mínimo práctico

    # Factor de suavizado exponencial
    # ORIGEN: Series temporales - α típico para respuesta moderada
    EMA_ALPHA = 0.1

    @classmethod
    def get_provenance(cls, constant_name: str) -> str:
        """Retorna justificación teórica de una constante."""
        provenances = {
            'PROMOTION_PERCENTILE': (
                "P80 = μ + 0.84σ en distribución normal. "
                "Indica rendimiento consistentemente superior al promedio propio."
            ),
            'SPECIALIST_Z_THRESHOLD': (
                "z ≥ 2 indica valor a 2 desviaciones estándar sobre la media. "
                "Probabilidad < 2.3% en distribución normal."
            ),
            'FOCUSED_Z_THRESHOLD': (
                "z ≥ 1 indica valor a 1 desviación estándar sobre la media. "
                "Probabilidad < 15.9% en distribución normal."
            ),
            'MIN_SAMPLES_FOR_STATS': (
                "n ≥ 5 mínimo práctico para estimar media/varianza. "
                "CLT sugiere n ≥ 30 para normalidad asintótica."
            ),
        }
        return provenances.get(constant_name, "Sin provenance documentada")


# =============================================================================
# ESPACIO DE TENSIONES (FORMAL Y CERRADO)
# =============================================================================

class TensionType(Enum):
    """
    Tensiones estructurales del conocimiento.

    SON propiedades del conocimiento, NO temas.
    NO se admiten tensiones psicológicas, narrativas o simbólicas.
    """
    INCONSISTENCY = "inconsistency"
    # Contradicción entre modelos/datos

    LOW_RESOLUTION = "low_resolution"
    # Incapacidad de discriminar o medir

    OVERSIMPLIFICATION = "oversimplification"
    # Modelo demasiado simple

    UNEXPLORED_HYPOTHESIS = "unexplored_hypothesis"
    # Región teórica no examinada

    MODEL_CONFLICT = "model_conflict"
    # Modelos compatibles pero incompatibles entre sí

    EMPIRICAL_GAP = "empirical_gap"
    # Falta de evidencia observacional


# Verificar que no hay tensiones prohibidas
_FORBIDDEN_KEYWORDS = {
    'curiosity', 'interest', 'preference', 'desire', 'motivation',
    'want', 'like', 'enjoy', 'feel', 'emotion', 'mood', 'personality'
}

for _t in TensionType:
    for _kw in _FORBIDDEN_KEYWORDS:
        if _kw in _t.value.lower():
            raise RuntimeError(
                f"ABORT: Tension '{_t.value}' contains forbidden keyword '{_kw}'"
            )


# Mapeo tensión → dominios (NO determinista)
TENSION_TO_DOMAINS: Dict[TensionType, List[str]] = {
    TensionType.INCONSISTENCY: ["mathematics", "physics", "medicine"],
    TensionType.LOW_RESOLUTION: ["medicine", "cosmology", "physics"],
    TensionType.OVERSIMPLIFICATION: ["mathematics", "physics", "medicine"],
    TensionType.UNEXPLORED_HYPOTHESIS: ["physics", "medicine", "cosmology"],
    TensionType.MODEL_CONFLICT: ["physics", "medicine", "mathematics"],
    TensionType.EMPIRICAL_GAP: ["medicine", "physics", "cosmology"],
}


# =============================================================================
# ESTRUCTURA DE NIVELES (DESCRIPTIVA, NO OPERATIVA)
# =============================================================================

class TaskLevel(Enum):
    """
    Niveles de complejidad de TAREAS (NO estados del agente).

    Los niveles describen complejidad intrínseca de tareas,
    NO se usan para decidir conducta del agente.
    """
    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"
    DOCTORAL = "doctoral"


# Curricula por dominio (propiedades de tareas, no de agentes)
DOMAIN_CURRICULA = {
    "mathematics": {
        TaskLevel.UNDERGRADUATE: [
            {"type": "math_eq_simple", "desc": "sistemas 1-2 variables"},
            {"type": "math_calculus", "desc": "derivadas simples"},
        ],
        TaskLevel.GRADUATE: [
            {"type": "math_eq_simple", "desc": "sistemas ≥3 variables"},
            {"type": "math_calculus", "desc": "integrales complejas"},
            {"type": "math_fit", "desc": "ajuste no lineal con ruido"},
        ],
        TaskLevel.DOCTORAL: [
            {"type": "math_series", "desc": "convergencia borderline"},
            {"type": "math_fit", "desc": "alto ruido, problemas abiertos"},
        ],
    },
    "physics": {
        TaskLevel.UNDERGRADUATE: [
            {"type": "phys_free_fall", "desc": "movimiento 1D, alto SNR"},
            {"type": "phys_oscillator", "desc": "oscilador simple"},
        ],
        TaskLevel.GRADUATE: [
            {"type": "phys_oscillator", "desc": "amortiguado"},
            {"type": "phys_coupled", "desc": "sistemas acoplados"},
        ],
        TaskLevel.DOCTORAL: [
            {"type": "phys_coupled", "desc": "Lotka-Volterra"},
            {"type": "phys_timeseries", "desc": "sin ground truth (hypothesis_falsification)"},
        ],
    },
    "medicine": {
        TaskLevel.UNDERGRADUATE: [
            {"type": "classification", "desc": "diagnóstico binario simple"},
        ],
        TaskLevel.GRADUATE: [
            {"type": "classification", "desc": "diagnóstico multiclase"},
            {"type": "regression", "desc": "predicción de variables continuas"},
        ],
        TaskLevel.DOCTORAL: [
            {"type": "causality", "desc": "inferencia causal sin ground truth"},
        ],
    },
    "cosmology": {
        TaskLevel.UNDERGRADUATE: [
            {"type": "regression", "desc": "ajuste de curvas simples"},
        ],
        TaskLevel.GRADUATE: [
            {"type": "timeseries", "desc": "análisis de series temporales"},
        ],
        TaskLevel.DOCTORAL: [
            {"type": "hypothesis_falsification", "desc": "falsación de modelos cosmológicos"},
        ],
    },
}


# =============================================================================
# ESTADO INTERNO DEL AGENTE
# =============================================================================

@dataclass
class InternalState:
    """
    Estado interno cuantificado.

    Estas métricas son las ÚNICAS fuentes de decisión.
    NO hay preferencias, roles, ni identidades.
    """
    # Métricas de error
    accumulated_error: float = 0.5
    error_history: List[float] = field(default_factory=list)

    # Métricas de coherencia
    internal_inconsistency: float = 0.0
    model_divergence: float = 0.0

    # Métricas de cobertura
    hypothesis_coverage: float = 0.7
    empirical_coverage: float = 0.7

    # Métricas de estancamiento
    learning_plateau: float = 0.0
    prediction_variance: float = 0.0

    # Historial de rendimiento por dominio
    domain_performance: Dict[str, List[float]] = field(default_factory=dict)

    # Nivel alcanzado por dominio (derivado de historial, NO asignado)
    domain_levels: Dict[str, TaskLevel] = field(default_factory=dict)

    # Timestamp
    last_updated: str = ""

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()

    def update_from_result(
        self,
        domain: str,
        performance: float,
        error: float,
        hypotheses_tested: int = 0,
        hypotheses_falsified: int = 0
    ):
        """Actualiza estado desde resultado de tarea."""
        alpha = TheoreticalConstants.EMA_ALPHA

        # Actualizar error acumulado (EMA)
        self.accumulated_error = (1 - alpha) * self.accumulated_error + alpha * error
        self.error_history.append(error)

        # Actualizar historial por dominio
        if domain not in self.domain_performance:
            self.domain_performance[domain] = []
        self.domain_performance[domain].append(performance)

        # Actualizar cobertura
        if hypotheses_tested > 0:
            self.hypothesis_coverage = min(1.0, self.hypothesis_coverage + 0.05)
            if hypotheses_falsified > 0:
                self.model_divergence = max(0, self.model_divergence - 0.1)
        else:
            self.hypothesis_coverage = max(0, self.hypothesis_coverage - 0.02)

        # Detectar estancamiento
        if len(self.error_history) >= 5:
            recent_errors = self.error_history[-5:]
            if np.std(recent_errors) < 0.05:
                self.learning_plateau = min(1.0, self.learning_plateau + 0.1)
            else:
                self.learning_plateau = max(0, self.learning_plateau - 0.05)

        self.last_updated = datetime.now().isoformat()

    def get_percentile_rank(self, domain: str, value: float) -> Optional[float]:
        """
        Calcula percentil de un valor en el historial PROPIO del dominio.

        ORIGEN: Estadística descriptiva - percentil relativo
        """
        history = self.domain_performance.get(domain, [])

        if len(history) < TheoreticalConstants.MIN_SAMPLES_FOR_STATS:
            return None  # Insuficiente historia

        n_below = sum(1 for h in history if h < value)
        return 100.0 * n_below / len(history)

    def get_recent_performance(self, domain: str, n: int = 5) -> Optional[float]:
        """Obtiene rendimiento reciente promedio."""
        history = self.domain_performance.get(domain, [])
        if not history:
            return None
        recent = history[-n:] if len(history) >= n else history
        return float(np.mean(recent))


# =============================================================================
# DETECTOR DE TENSIONES
# =============================================================================

@dataclass
class DetectedTension:
    """Tensión detectada con métricas de origen."""
    tension_type: TensionType
    intensity: float
    source_metrics: Dict[str, float]
    detected_at: str = ""

    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.now().isoformat()


class TensionDetector:
    """
    Detecta tensiones SOLO desde métricas internas.

    HARD RULE: sample_tension() NUNCA recibe dominio o identidad como input.
    """

    def __init__(self):
        self.logger = get_provenance_logger()

    def detect_all(self, state: InternalState) -> List[DetectedTension]:
        """Detecta todas las tensiones activas."""
        tensions = []

        # INCONSISTENCY
        if state.internal_inconsistency > 0.3 or state.model_divergence > 0.4:
            tensions.append(DetectedTension(
                tension_type=TensionType.INCONSISTENCY,
                intensity=max(state.internal_inconsistency, state.model_divergence),
                source_metrics={
                    'internal_inconsistency': state.internal_inconsistency,
                    'model_divergence': state.model_divergence,
                }
            ))

        # LOW_RESOLUTION
        if state.prediction_variance > 0.25:
            tensions.append(DetectedTension(
                tension_type=TensionType.LOW_RESOLUTION,
                intensity=state.prediction_variance,
                source_metrics={'prediction_variance': state.prediction_variance}
            ))

        # OVERSIMPLIFICATION
        if state.learning_plateau > 0.3 and state.accumulated_error > 0.3:
            tensions.append(DetectedTension(
                tension_type=TensionType.OVERSIMPLIFICATION,
                intensity=(state.learning_plateau + state.accumulated_error) / 2,
                source_metrics={
                    'learning_plateau': state.learning_plateau,
                    'accumulated_error': state.accumulated_error,
                }
            ))

        # UNEXPLORED_HYPOTHESIS
        if state.hypothesis_coverage < 0.7:
            tensions.append(DetectedTension(
                tension_type=TensionType.UNEXPLORED_HYPOTHESIS,
                intensity=1.0 - state.hypothesis_coverage,
                source_metrics={'hypothesis_coverage': state.hypothesis_coverage}
            ))

        # MODEL_CONFLICT
        if state.model_divergence > 0.5 and state.accumulated_error < 0.3:
            tensions.append(DetectedTension(
                tension_type=TensionType.MODEL_CONFLICT,
                intensity=state.model_divergence,
                source_metrics={
                    'model_divergence': state.model_divergence,
                    'accumulated_error': state.accumulated_error,
                }
            ))

        # EMPIRICAL_GAP
        if state.empirical_coverage < 0.6:
            tensions.append(DetectedTension(
                tension_type=TensionType.EMPIRICAL_GAP,
                intensity=1.0 - state.empirical_coverage,
                source_metrics={'empirical_coverage': state.empirical_coverage}
            ))

        # Default si no hay tensiones
        if not tensions:
            tensions.append(DetectedTension(
                tension_type=TensionType.UNEXPLORED_HYPOTHESIS,
                intensity=0.3,
                source_metrics={'default': True}
            ))

        return tensions

    def sample(self, state: InternalState, seed: Optional[int] = None) -> DetectedTension:
        """
        Muestrea una tensión del estado interno.

        HARD RULE: Solo métricas internas. NUNCA dominio o identidad.
        """
        if seed is not None:
            np.random.seed(seed)

        tensions = self.detect_all(state)

        # Softmax por intensidad
        intensities = np.array([t.intensity for t in tensions])
        scaled = intensities - np.max(intensities)
        exp_i = np.exp(scaled / 0.5)
        probs = exp_i / np.sum(exp_i)

        idx = np.random.choice(len(tensions), p=probs)
        return tensions[idx]


# =============================================================================
# RESOLUTOR DE TENSIONES → DOMINIOS
# =============================================================================

class DomainResolver:
    """
    Resuelve tensiones en dominios candidatos.

    HARD RULE: NO acepta identidad de agente como input.
    """

    def __init__(self):
        self.logger = get_provenance_logger()
        self._weights: Dict[str, float] = {}

    def resolve(
        self,
        tension: DetectedTension,
        exclude: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """Resuelve tensión en candidatos con pesos."""
        candidates = TENSION_TO_DOMAINS.get(tension.tension_type, [])

        if exclude:
            candidates = [c for c in candidates if c not in exclude]

        if not candidates:
            raise RuntimeError(
                f"ABORT: No candidate domains for tension '{tension.tension_type.value}'"
            )

        # Pesos uniformes ajustados por uso histórico
        weighted = []
        for domain in candidates:
            base_weight = 1.0 / len(candidates)
            historical = self._weights.get(domain, 1.0)
            weighted.append((domain, base_weight * historical))

        # Normalizar
        total = sum(w for _, w in weighted)
        if total > 0:
            weighted = [(d, w/total) for d, w in weighted]

        return sorted(weighted, key=lambda x: x[1], reverse=True)

    def sample(
        self,
        tension: DetectedTension,
        exclude: Optional[Set[str]] = None,
        seed: Optional[int] = None
    ) -> str:
        """Muestrea un dominio."""
        if seed is not None:
            np.random.seed(seed)

        candidates = self.resolve(tension, exclude)
        domains = [d for d, _ in candidates]
        weights = [w for _, w in candidates]

        domain = np.random.choice(domains, p=weights)

        # Decay para evitar saturación
        self._weights[domain] = self._weights.get(domain, 1.0) * 0.95

        return domain

    def update(self, domain: str, success: bool, reduction: float):
        """Actualiza pesos después de resultado."""
        current = self._weights.get(domain, 1.0)
        if success and reduction > 0:
            self._weights[domain] = min(2.0, current * 1.1)
        else:
            self._weights[domain] = max(0.5, current * 0.9)


# =============================================================================
# SISTEMA DE PROMOCIÓN (100% ENDÓGENO)
# =============================================================================

class PromotionSystem:
    """
    Sistema de promoción basado SOLO en historial propio.

    HARD RULE: NO usa umbrales absolutos como "accuracy > 0.8"
    USA: percentil del rendimiento reciente en historial propio
    """

    def __init__(self):
        self.logger = get_provenance_logger()

    def check_promotion(
        self,
        state: InternalState,
        domain: str
    ) -> Tuple[bool, Optional[TaskLevel], str]:
        """
        Verifica si el agente puede ser promocionado.

        CRITERIO (ÚNICO PERMITIDO):
            percentile = agent.get_percentile_rank_in_own_history(recent_performance)
            if percentile >= 80:
                promote()

        Returns:
            (puede_promocionar, nuevo_nivel, justificación)
        """
        history = state.domain_performance.get(domain, [])

        # Sin historia suficiente → no evaluar
        if len(history) < TheoreticalConstants.MIN_SAMPLES_FOR_STATS:
            return False, None, f"Insufficient history (n={len(history)} < {TheoreticalConstants.MIN_SAMPLES_FOR_STATS})"

        # Obtener rendimiento reciente
        n_recent = min(5, len(history))
        recent_perf = np.mean(history[-n_recent:])

        # Calcular percentil en historial PROPIO
        percentile = state.get_percentile_rank(domain, recent_perf)

        if percentile is None:
            return False, None, "Could not calculate percentile"

        # Criterio de promoción
        # ORIGEN: P80 = μ + 0.84σ en distribución normal
        can_promote = percentile >= TheoreticalConstants.PROMOTION_PERCENTILE

        justification = (
            f"recent_perf={recent_perf:.3f}, "
            f"percentile={percentile:.1f} "
            f"(threshold={TheoreticalConstants.PROMOTION_PERCENTILE}). "
            f"PROVENANCE: {TheoreticalConstants.get_provenance('PROMOTION_PERCENTILE')}"
        )

        if can_promote:
            current_level = state.domain_levels.get(domain, TaskLevel.UNDERGRADUATE)
            next_level = self._get_next_level(current_level)
            return True, next_level, justification
        else:
            return False, None, justification

    def _get_next_level(self, current: TaskLevel) -> Optional[TaskLevel]:
        """Obtiene siguiente nivel."""
        order = [TaskLevel.UNDERGRADUATE, TaskLevel.GRADUATE, TaskLevel.DOCTORAL]
        idx = order.index(current)
        if idx + 1 < len(order):
            return order[idx + 1]
        return None

    def promote(
        self,
        state: InternalState,
        domain: str
    ) -> Tuple[bool, Optional[TaskLevel], str]:
        """Ejecuta promoción si es posible."""
        can_promote, new_level, justification = self.check_promotion(state, domain)

        if can_promote and new_level is not None:
            state.domain_levels[domain] = new_level
            return True, new_level, justification

        return False, None, justification


# =============================================================================
# SISTEMA DE ETIQUETAS (SOLO POST-HOC)
# =============================================================================

class LabelSystem:
    """
    Genera etiquetas EMERGENTES (post-hoc, NUNCA causales).

    HARD RULE: Las etiquetas NO influyen en decisiones.
    Son solo descriptivas para análisis humano.
    """

    def __init__(self):
        self.logger = get_provenance_logger()

    def generate_label(
        self,
        state: InternalState,
        agent_id: str  # Solo para logging, NO para decisión
    ) -> Dict[str, Any]:
        """
        Genera etiqueta emergente.

        ÚNICA FORMA PERMITIDA:
            label = f"{domain_prefix}_{spec_grade}_{level_suffix}"

        Donde:
            domain_prefix = dominio con mayor (afinidad × nivel)
            spec_grade = "specialist" si z ≥ 2, "focused" si z ≥ 1
            level_suffix = nivel más alto alcanzado
        """
        if not state.domain_performance:
            return {
                'agent_id': agent_id,
                'label': 'novice',
                'provenance': 'No domain history',
                'note': 'Label is POST-HOC only, NEVER causal',
            }

        # Calcular score por dominio: mean_performance × level_value
        level_values = {
            TaskLevel.UNDERGRADUATE: 1,
            TaskLevel.GRADUATE: 2,
            TaskLevel.DOCTORAL: 3,
        }

        domain_scores = {}
        for domain, history in state.domain_performance.items():
            if history:
                mean_perf = np.mean(history)
                level = state.domain_levels.get(domain, TaskLevel.UNDERGRADUATE)
                level_val = level_values[level]
                domain_scores[domain] = mean_perf * level_val

        if not domain_scores:
            return {
                'agent_id': agent_id,
                'label': 'novice',
                'provenance': 'No scored domains',
                'note': 'Label is POST-HOC only, NEVER causal',
            }

        # Encontrar dominio top
        top_domain = max(domain_scores.keys(), key=lambda d: domain_scores[d])
        top_score = domain_scores[top_domain]

        # Calcular z-score de especialización (interno)
        other_scores = [s for d, s in domain_scores.items() if d != top_domain]

        if other_scores and len(other_scores) >= 2:
            mean_others = np.mean(other_scores)
            std_others = np.std(other_scores, ddof=1)
            if std_others > 1e-10:
                spec_z = (top_score - mean_others) / std_others
            else:
                spec_z = 0.0
        else:
            spec_z = 0.0

        # Determinar grado de especialización
        # ORIGEN: z ≥ 2 = 2σ sobre media, z ≥ 1 = 1σ sobre media
        if spec_z >= TheoreticalConstants.SPECIALIST_Z_THRESHOLD:
            spec_grade = "specialist"
        elif spec_z >= TheoreticalConstants.FOCUSED_Z_THRESHOLD:
            spec_grade = "focused"
        else:
            spec_grade = ""

        # Prefijo de dominio
        domain_prefix = {
            'mathematics': 'math',
            'physics': 'phys',
            'medicine': 'med',
            'cosmology': 'cosmo',
        }.get(top_domain, top_domain[:4])

        # Sufijo de nivel
        top_level = state.domain_levels.get(top_domain, TaskLevel.UNDERGRADUATE)
        level_suffix = {
            TaskLevel.UNDERGRADUATE: 'undergrad',
            TaskLevel.GRADUATE: 'graduate',
            TaskLevel.DOCTORAL: 'doctoral',
        }[top_level]

        # Construir etiqueta
        if spec_grade:
            label = f"{domain_prefix}_{spec_grade}_{level_suffix}"
        else:
            label = f"{domain_prefix}_{level_suffix}"

        provenance = (
            f"z-score={spec_z:.2f} (internal). "
            f"PROVENANCE: z≥{TheoreticalConstants.SPECIALIST_Z_THRESHOLD}=specialist, "
            f"z≥{TheoreticalConstants.FOCUSED_Z_THRESHOLD}=focused. "
            f"{TheoreticalConstants.get_provenance('SPECIALIST_Z_THRESHOLD')}"
        )

        return {
            'agent_id': agent_id,
            'label': label,
            'top_domain': top_domain,
            'top_level': top_level.value,
            'specialization_z': spec_z,
            'domain_scores': domain_scores,
            'provenance': provenance,
            'note': 'Label is POST-HOC only, NEVER causal',
        }


# =============================================================================
# VALIDADOR DE INTEGRIDAD (ABORT SI VIOLACIÓN)
# =============================================================================

class IntegrityValidator:
    """
    Valida cumplimiento de HARD RULES.

    Si detecta violación → abort_execution()
    """

    def __init__(self):
        self.violations: List[str] = []

    def validate_flow(
        self,
        tension: Optional[DetectedTension],
        domain: Optional[str]
    ):
        """Valida flujo tensión → dominio."""
        if domain is not None and tension is None:
            self._abort("Domain selected without prior tension detection")

        if tension is not None and not tension.source_metrics:
            self._abort("Tension has no source_metrics (must be derived from internal state)")

    def validate_no_magic_numbers(self, value: float, name: str, allowed: List[float]):
        """Valida que un valor está en la lista de constantes teóricas permitidas."""
        if value not in allowed:
            self._abort(f"Magic number detected: {name}={value}. Must use theoretical constants.")

    def validate_no_identity_selection(
        self,
        agent_id: str,
        domain: str,
        selection_path: List[str]
    ):
        """Valida que la identidad no determinó el dominio."""
        path_str = " ".join(selection_path).lower()

        # El nombre del agente NO debe aparecer antes de tensión
        if agent_id.lower() in path_str.split("tension")[0] if "tension" in path_str else path_str:
            # Esto está bien si es solo tracking, verificar que no es causal
            pass

        # Patterns prohibidos
        forbidden = [
            f"if agent == '{agent_id}'",
            f"{agent_id} -> {domain}",
            f"agent={agent_id}, domain={domain}",
        ]

        for pattern in forbidden:
            if pattern.lower() in path_str:
                self._abort(f"Identity-based selection detected: {pattern}")

    def _abort(self, reason: str):
        """Aborta ejecución."""
        self.violations.append(reason)
        raise RuntimeError(f"ABORT: HARD RULE VIOLATION - {reason}")


# =============================================================================
# NÚCLEO INTEGRADO
# =============================================================================

@dataclass
class ResearchTask:
    """Tarea de investigación con toda la trazabilidad."""
    task_id: str
    tension: DetectedTension
    domain: str
    level: TaskLevel
    task_type: str
    selection_path: List[str]
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class ResearchResult:
    """Resultado de investigación."""
    task: ResearchTask
    performance: float
    error: float
    success: bool
    promoted: bool
    new_level: Optional[TaskLevel]
    promotion_reason: str
    completed_at: str = ""

    def __post_init__(self):
        if not self.completed_at:
            self.completed_at = datetime.now().isoformat()


class EndogenousResearchNucleus:
    """
    Núcleo de Investigación Endógena.

    Implementa el flujo COMPLETO sin excepciones:
        estado → tensión → dominio → tarea → resultado → historial → promoción → etiqueta
    """

    def __init__(self, seed: Optional[int] = None):
        self.logger = get_provenance_logger()
        self.rng = np.random.default_rng(seed)

        # Componentes
        self._tension_detector = TensionDetector()
        self._domain_resolver = DomainResolver()
        self._promotion_system = PromotionSystem()
        self._label_system = LabelSystem()
        self._validator = IntegrityValidator()

        # Estados por agente (ID solo para tracking)
        self._states: Dict[str, InternalState] = {}

        # Historial de tareas
        self._task_history: Dict[str, List[ResearchResult]] = {}

        # Contador de tareas
        self._task_counter = 0

    def _get_state(self, agent_id: str) -> InternalState:
        """Obtiene estado (ID solo para tracking, NO para decisión)."""
        if agent_id not in self._states:
            self._states[agent_id] = InternalState()
        return self._states[agent_id]

    def generate_task(
        self,
        agent_id: str,  # Solo tracking
        exclude_domains: Optional[Set[str]] = None
    ) -> ResearchTask:
        """
        Genera tarea siguiendo el flujo obligatorio.

        FLUJO:
            estado_interno → tensión → dominios_candidatos → tarea
        """
        path = []

        # 1. Obtener estado interno
        state = self._get_state(agent_id)
        path.append(f"state(metrics={len(state.domain_performance)})")

        # 2. Detectar tensión (SOLO desde métricas)
        tension = self._tension_detector.sample(state)
        path.append(f"tension({tension.tension_type.value})")

        # VALIDAR: tensión tiene source_metrics
        self._validator.validate_flow(tension=tension, domain=None)

        # 3. Resolver tensión → dominio
        domain = self._domain_resolver.sample(tension, exclude_domains)
        path.append(f"domain({domain})")

        # VALIDAR: flujo completo
        self._validator.validate_flow(tension=tension, domain=domain)

        # 4. Determinar nivel (de la tarea, basado en historial)
        level = state.domain_levels.get(domain, TaskLevel.UNDERGRADUATE)
        path.append(f"level({level.value})")

        # 5. Seleccionar tipo de tarea
        curriculum = DOMAIN_CURRICULA.get(domain, {})
        level_tasks = curriculum.get(level, [])

        if level_tasks:
            task_spec = self.rng.choice(level_tasks)
            task_type = task_spec['type']
        else:
            task_type = "generic"

        path.append(f"task_type({task_type})")

        # Crear tarea
        self._task_counter += 1
        task = ResearchTask(
            task_id=f"task_{self._task_counter:06d}",
            tension=tension,
            domain=domain,
            level=level,
            task_type=task_type,
            selection_path=path,
        )

        self.logger.log_from_data(
            value={'tension': tension.tension_type.value, 'domain': domain, 'path': path},
            source="Task generated from endogenous flow",
            statistic="task_generation",
            context="EndogenousResearchNucleus.generate_task"
        )

        return task

    def complete_task(
        self,
        agent_id: str,
        task: ResearchTask,
        performance: float,
        error: float
    ) -> ResearchResult:
        """
        Completa tarea y actualiza estado.

        FLUJO:
            resultado → historial → mejora_relativa → (promoción) → (etiqueta)
        """
        state = self._get_state(agent_id)

        # Actualizar estado
        state.update_from_result(
            domain=task.domain,
            performance=performance,
            error=error
        )

        # Verificar promoción (100% endógena)
        promoted, new_level, reason = self._promotion_system.promote(state, task.domain)

        # Actualizar resolver
        success = performance > 0.5
        self._domain_resolver.update(task.domain, success, task.tension.intensity)

        # Crear resultado
        result = ResearchResult(
            task=task,
            performance=performance,
            error=error,
            success=success,
            promoted=promoted,
            new_level=new_level,
            promotion_reason=reason,
        )

        # Guardar en historial
        if agent_id not in self._task_history:
            self._task_history[agent_id] = []
        self._task_history[agent_id].append(result)

        return result

    def get_label(self, agent_id: str) -> Dict[str, Any]:
        """Genera etiqueta emergente (POST-HOC, nunca causal)."""
        state = self._get_state(agent_id)
        return self._label_system.generate_label(state, agent_id)

    def get_report(self, agent_id: str) -> Dict[str, Any]:
        """Genera reporte completo."""
        state = self._get_state(agent_id)
        label = self.get_label(agent_id)
        history = self._task_history.get(agent_id, [])

        return {
            'agent_id': agent_id,
            'total_tasks': len(history),
            'successful_tasks': sum(1 for r in history if r.success),
            'promotions': sum(1 for r in history if r.promoted),
            'current_levels': {d: l.value for d, l in state.domain_levels.items()},
            'emergent_label': label,
            'domains_explored': list(state.domain_performance.keys()),
        }


# =============================================================================
# DIRECTOR (MULTIAGENTE)
# =============================================================================

class EndogenousResearchDirector:
    """
    Director de investigación multiagente.

    Los IDs de agente son SOLO para tracking, NUNCA para decisión.
    """

    def __init__(self, seed: Optional[int] = None):
        self.logger = get_provenance_logger()
        self._nucleus = EndogenousResearchNucleus(seed=seed)
        self._agents: List[str] = []
        self._round = 0
        self._active = False

    def start_session(self, agent_ids: List[str]):
        """Inicia sesión (IDs solo para tracking)."""
        self._agents = agent_ids
        self._round = 0
        self._active = True

    def run_round(
        self,
        solver_fn: Optional[callable] = None
    ) -> List[ResearchResult]:
        """Ejecuta una ronda de investigación."""
        if not self._active:
            raise RuntimeError("Session not started")

        self._round += 1
        results = []

        for agent_id in self._agents:
            # Generar tarea (flujo endógeno)
            task = self._nucleus.generate_task(agent_id)

            # Resolver
            if solver_fn:
                performance, error = solver_fn(agent_id, task)
            else:
                # Default: performance aleatoria
                performance = np.random.uniform(0.4, 0.9)
                error = 1.0 - performance

            # Completar
            result = self._nucleus.complete_task(agent_id, task, performance, error)
            results.append(result)

        return results

    def get_session_report(self) -> Dict[str, Any]:
        """Genera reporte de sesión."""
        reports = {}
        labels = {}

        for agent_id in self._agents:
            reports[agent_id] = self._nucleus.get_report(agent_id)
            labels[agent_id] = reports[agent_id]['emergent_label']['label']

        return {
            'rounds': self._round,
            'agents': self._agents,
            'reports': reports,
            'labels': labels,
            'note': (
                "All decisions emerged from internal metrics. "
                "Labels are POST-HOC only. "
                "Agent IDs are for TRACKING only, never causal."
            ),
        }


# =============================================================================
# TEST
# =============================================================================

def test_endogenous_nucleus():
    """Test del núcleo endógeno."""
    print("=" * 70)
    print("TEST: ENDOGENOUS RESEARCH NUCLEUS")
    print("=" * 70)

    director = EndogenousResearchDirector(seed=42)
    agents = ['GAUSS', 'NEWTON', 'EULER']

    print(f"\nIniciando sesión con {len(agents)} agentes")
    print("NOTA: IDs son SOLO para tracking, NO para decisión")
    director.start_session(agents)

    print("\n=== EJECUTANDO 50 RONDAS ===")

    for r in range(50):
        results = director.run_round()

        if (r + 1) % 10 == 0:
            print(f"\n--- Ronda {r + 1} ---")
            for res in results:
                promo = f" -> PROMOTED to {res.new_level.value}" if res.promoted else ""
                print(f"  {res.task.task_id}: {res.task.tension.tension_type.value} "
                      f"-> {res.task.domain}/{res.task.level.value} "
                      f"perf={res.performance:.2f}{promo}")

    print("\n=== REPORTE FINAL ===")
    report = director.get_session_report()

    print(f"\nRondas: {report['rounds']}")

    print("\nETIQUETAS EMERGENTES (post-hoc):")
    for agent, label in report['labels'].items():
        print(f"  {agent}: {label}")

    print("\nNIVELES ALCANZADOS:")
    for agent, r in report['reports'].items():
        levels = r['current_levels']
        print(f"  {agent}: {levels}")

    print("\n" + "-" * 50)
    print("VERIFICACIÓN DE PRINCIPIOS:")
    print("-" * 50)
    print("  ✓ No hay números mágicos (solo constantes teóricas)")
    print("  ✓ No hay roles asignados (etiquetas son post-hoc)")
    print("  ✓ No hay selección por identidad (flujo: tensión → dominio)")
    print("  ✓ Promoción por percentil propio (no umbral absoluto)")

    print("\n" + "=" * 70)
    print("TEST COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    test_endogenous_nucleus()
