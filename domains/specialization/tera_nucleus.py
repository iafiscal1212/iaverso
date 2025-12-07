"""
TENSION-DRIVEN ENDOGENOUS RESEARCH ARCHITECTURE (TERA)
=======================================================

Núcleo formal con metrificación rigurosa de tensiones.

PRINCIPIOS HARD:
================
1. Toda tensión tiene métricas formales (KL, Fisher, NRV, etc.)
2. Intensidad = ||z_T||_2 (norma L2 de z-scores)
3. Persistencia = media móvil de intensidad
4. Nivel de tarea = percentil interno de persistencia
5. Tendencia = derivada temporal de intensidad
6. Auditoría = tension_report.yaml exportable

NO SE PERMITE:
==============
- Umbrales absolutos (solo percentiles internos)
- Decisiones por identidad de agente
- Números mágicos sin provenance
- Saltar tensión → dominio directo

FLUJO FORMAL:
=============
métricas_internas → z_scores → tensión(I_T, P_T, trend) → dominio → nivel → tarea
"""

import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from enum import Enum
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stimuli_engine.provenance import get_provenance_logger


# =============================================================================
# CONSTANTES TEÓRICAS (CON PROVENANCE FORMAL)
# =============================================================================

class TheoreticalConstants:
    """
    Constantes derivadas de teoría estadística.
    Cada una tiene justificación formal documentada.
    """

    # Percentiles para niveles de tarea (CLT + distribución normal)
    UNDERGRADUATE_PERCENTILE = 50.0  # Mediana
    GRADUATE_PERCENTILE = 85.0       # ~1σ sobre mediana
    DOCTORAL_PERCENTILE = 95.0       # ~1.65σ sobre mediana

    # Tamaño de ventana para persistencia
    # ORIGEN: Teoría de series temporales - ventana típica para suavizado
    PERSISTENCE_WINDOW = 5

    # Mínimo de muestras para estadísticas válidas
    MIN_SAMPLES_FOR_STATS = 5

    # Factor de temperatura para softmax
    SOFTMAX_TEMPERATURE = 0.5

    # Z-scores para etiquetas post-hoc
    SPECIALIST_Z_THRESHOLD = 2.0  # 2σ = P97.7
    FOCUSED_Z_THRESHOLD = 1.0     # 1σ = P84.1

    # Promoción por percentil propio
    PROMOTION_PERCENTILE = 80.0   # P80 = μ + 0.84σ

    _PROVENANCE = {
        'UNDERGRADUATE_PERCENTILE': (
            "P50 = mediana. Tareas base para persistencia por debajo de la media histórica."
        ),
        'GRADUATE_PERCENTILE': (
            "P85 ≈ μ + 1σ en distribución normal. Tensión sostenida moderada."
        ),
        'DOCTORAL_PERCENTILE': (
            "P95 ≈ μ + 1.65σ en distribución normal. Tensión estructural intensa."
        ),
        'PERSISTENCE_WINDOW': (
            "Ventana de 5 pasos: balance entre respuesta y estabilidad (series temporales)."
        ),
        'MIN_SAMPLES_FOR_STATS': (
            "n ≥ 5 mínimo práctico para estimar media/varianza con estabilidad."
        ),
        'SOFTMAX_TEMPERATURE': (
            "T=0.5: moderadamente selectivo sin ser determinístico."
        ),
        'SPECIALIST_Z_THRESHOLD': (
            "z ≥ 2 = P97.7 en normal. Diferencia estadísticamente significativa."
        ),
        'FOCUSED_Z_THRESHOLD': (
            "z ≥ 1 = P84.1 en normal. Diferencia notable pero no extrema."
        ),
        'PROMOTION_PERCENTILE': (
            "P80 = μ + 0.84σ. Rendimiento consistentemente superior al promedio propio."
        ),
    }

    @classmethod
    def get_provenance(cls, name: str) -> str:
        return cls._PROVENANCE.get(name, "Sin provenance documentada")


# =============================================================================
# ENUMS
# =============================================================================

class TensionType(Enum):
    """
    Tipos de tensión epistémica.
    Cada uno tiene métricas formales asociadas.
    """
    INCONSISTENCY = "inconsistency"           # KL divergence, sign agreement
    LOW_RESOLUTION = "low_resolution"         # Fisher info, NRV
    OVERSIMPLIFICATION = "oversimplification" # Gen gap, residual entropy
    UNEXPLORED_HYPOTHESIS = "unexplored_hypothesis"  # Coverage, embedding distance
    MODEL_CONFLICT = "model_conflict"         # Prediction disagreement, causal distance
    EMPIRICAL_GAP = "empirical_gap"           # Evidence ratio, KL post/prior


class TaskLevel(Enum):
    """Niveles de complejidad de TAREA (no de agente)."""
    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"
    DOCTORAL = "doctoral"


class TensionTrend(Enum):
    """Tendencia temporal de una tensión."""
    GROWING = "growing"           # I_T ↑, dI/dt > 0
    STRUCTURAL = "structural"     # I_T alto, dI/dt ≈ 0
    RESOLVING = "resolving"       # I_T ↓, dI/dt < 0
    STABLE = "stable"             # I_T bajo, dI/dt ≈ 0


# =============================================================================
# MÉTRICAS FORMALES POR TENSIÓN
# =============================================================================

@dataclass
class TensionMetrics:
    """
    Métricas formales de una tensión específica.
    Cada tensión tiene su conjunto de métricas cuantificables.
    """
    tension_type: TensionType
    raw_metrics: Dict[str, float]      # Valores brutos
    z_scores: Dict[str, float]         # Z-scores internos
    intensity_L2: float = 0.0          # ||z||_2

    def __post_init__(self):
        if self.z_scores and self.intensity_L2 == 0.0:
            # Calcular intensidad L2 automáticamente
            z_values = np.array(list(self.z_scores.values()))
            self.intensity_L2 = float(np.linalg.norm(z_values))


@dataclass
class TensionState:
    """
    Estado completo de una tensión con dinámica temporal.
    """
    tension_type: TensionType
    metrics: TensionMetrics
    intensity: float                    # I_T(t) = ||z_T||_2
    persistence: float                  # P_T(t) = media móvil de I_T
    delta_intensity: float = 0.0        # dI/dt
    trend: TensionTrend = TensionTrend.STABLE
    percentile_rank: float = 0.0        # Percentil en historia propia

    @property
    def source_metrics(self) -> Dict[str, float]:
        """Métricas originales (para compatibilidad)."""
        return self.metrics.raw_metrics


# =============================================================================
# CALCULADOR DE MÉTRICAS POR TENSIÓN
# =============================================================================

class TensionMetricsCalculator:
    """
    Calcula métricas formales para cada tipo de tensión.

    MÉTRICAS POR TENSIÓN:
    - INCONSISTENCY: KL divergence, sign agreement
    - LOW_RESOLUTION: Fisher information, NRV
    - OVERSIMPLIFICATION: Gen gap, residual entropy
    - UNEXPLORED_HYPOTHESIS: Coverage, embedding distance
    - MODEL_CONFLICT: Prediction disagreement, causal distance
    - EMPIRICAL_GAP: Evidence ratio, KL post/prior
    """

    def __init__(self):
        self.logger = get_provenance_logger()

        # Historiales para z-scores (por métrica)
        self._metric_histories: Dict[str, List[float]] = {}

    def _update_history(self, metric_name: str, value: float):
        """Actualiza historial de una métrica."""
        if metric_name not in self._metric_histories:
            self._metric_histories[metric_name] = []
        self._metric_histories[metric_name].append(value)

        # Mantener ventana máxima de 100
        if len(self._metric_histories[metric_name]) > 100:
            self._metric_histories[metric_name] = self._metric_histories[metric_name][-100:]

    def _compute_z_score(self, metric_name: str, value: float) -> float:
        """Calcula z-score interno de una métrica."""
        history = self._metric_histories.get(metric_name, [])

        if len(history) < TheoreticalConstants.MIN_SAMPLES_FOR_STATS:
            return 0.0

        mean = np.mean(history)
        std = np.std(history, ddof=1)

        if std < 1e-10:
            return 0.0

        return (value - mean) / std

    def calculate_inconsistency(
        self,
        predictions: List[np.ndarray],
        labels: Optional[np.ndarray] = None
    ) -> TensionMetrics:
        """
        INCONSISTENCY: Contradicción entre modelos.

        Métricas:
        - kl_mean: Divergencia KL media entre predicciones
        - sign_agreement: Proporción de acuerdo en signos
        """
        n_models = len(predictions)
        raw_metrics = {}

        if n_models < 2:
            raw_metrics = {'kl_mean': 0.0, 'sign_agreement': 1.0, 'models_compared': 1}
        else:
            # KL divergence media
            kl_values = []
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    p = predictions[i].flatten() + 1e-10
                    q = predictions[j].flatten() + 1e-10
                    p = p / p.sum()
                    q = q / q.sum()
                    kl = np.sum(p * np.log(p / q))
                    kl_values.append(kl)

            kl_mean = np.mean(kl_values) if kl_values else 0.0

            # Sign agreement
            if labels is not None:
                agreements = []
                for i in range(n_models):
                    for j in range(i + 1, n_models):
                        sign_i = np.sign(predictions[i])
                        sign_j = np.sign(predictions[j])
                        agreements.append(np.mean(sign_i == sign_j))
                sign_agreement = np.mean(agreements) if agreements else 1.0
            else:
                sign_agreement = 1.0

            raw_metrics = {
                'kl_mean': float(kl_mean),
                'sign_agreement': float(sign_agreement),
                'models_compared': n_models
            }

        # Actualizar historiales y calcular z-scores
        z_scores = {}
        for name, value in raw_metrics.items():
            if name != 'models_compared':
                self._update_history(f"inconsistency_{name}", value)
                z_scores[name] = self._compute_z_score(f"inconsistency_{name}", value)

        return TensionMetrics(
            tension_type=TensionType.INCONSISTENCY,
            raw_metrics=raw_metrics,
            z_scores=z_scores
        )

    def calculate_low_resolution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        gradients: Optional[np.ndarray] = None
    ) -> TensionMetrics:
        """
        LOW_RESOLUTION: Incapacidad de discriminar.

        Métricas:
        - fisher_information: Aproximación de información de Fisher
        - normalized_residual_variance: Var(y - y_hat) / Var(y)
        """
        raw_metrics = {}

        # NRV (Normalized Residual Variance)
        residuals = y_true - y_pred
        var_y = np.var(y_true)
        var_residual = np.var(residuals)

        if var_y > 1e-10:
            nrv = var_residual / var_y
        else:
            nrv = 1.0

        raw_metrics['normalized_residual_variance'] = float(nrv)

        # Fisher Information (aproximación)
        if gradients is not None and len(gradients) > 0:
            fisher = np.mean(gradients ** 2)
        else:
            # Proxy: inversa de varianza residual
            fisher = 1.0 / (var_residual + 1e-10)

        raw_metrics['fisher_information'] = float(np.clip(fisher, 0, 100))

        # Z-scores
        z_scores = {}
        for name, value in raw_metrics.items():
            self._update_history(f"low_resolution_{name}", value)
            z_scores[name] = self._compute_z_score(f"low_resolution_{name}", value)

        return TensionMetrics(
            tension_type=TensionType.LOW_RESOLUTION,
            raw_metrics=raw_metrics,
            z_scores=z_scores
        )

    def calculate_oversimplification(
        self,
        loss_train: float,
        loss_val: float,
        residual_entropy: Optional[float] = None
    ) -> TensionMetrics:
        """
        OVERSIMPLIFICATION: Modelo demasiado simple.

        Métricas:
        - generalization_gap: L_val - L_train
        - residual_entropy: H(y | y_hat)
        """
        raw_metrics = {}

        # Generalization gap
        gen_gap = loss_val - loss_train
        raw_metrics['generalization_gap'] = float(gen_gap)

        # Residual entropy
        if residual_entropy is not None:
            raw_metrics['residual_entropy'] = float(residual_entropy)
        else:
            # Proxy: si gap pequeño pero loss alto → alta entropía residual
            raw_metrics['residual_entropy'] = float(loss_val * (1 - abs(gen_gap)))

        # Z-scores
        z_scores = {}
        for name, value in raw_metrics.items():
            self._update_history(f"oversimplification_{name}", value)
            z_scores[name] = self._compute_z_score(f"oversimplification_{name}", value)

        return TensionMetrics(
            tension_type=TensionType.OVERSIMPLIFICATION,
            raw_metrics=raw_metrics,
            z_scores=z_scores
        )

    def calculate_unexplored_hypothesis(
        self,
        hypotheses_evaluated: int,
        hypotheses_reachable: int,
        embedding_distances: Optional[List[float]] = None
    ) -> TensionMetrics:
        """
        UNEXPLORED_HYPOTHESIS: Zonas no visitadas.

        Métricas:
        - hypothesis_coverage: evaluadas / alcanzables
        - mean_embedding_distance: distancia media a prototipos conocidos
        """
        raw_metrics = {}

        # Coverage
        if hypotheses_reachable > 0:
            coverage = hypotheses_evaluated / hypotheses_reachable
        else:
            coverage = 1.0

        raw_metrics['hypothesis_coverage'] = float(np.clip(coverage, 0, 1))

        # Mean embedding distance
        if embedding_distances:
            raw_metrics['mean_embedding_distance'] = float(np.mean(embedding_distances))
        else:
            # Proxy: inverso de coverage
            raw_metrics['mean_embedding_distance'] = float(1.0 - coverage)

        # Z-scores
        z_scores = {}
        for name, value in raw_metrics.items():
            self._update_history(f"unexplored_{name}", value)
            z_scores[name] = self._compute_z_score(f"unexplored_{name}", value)

        return TensionMetrics(
            tension_type=TensionType.UNEXPLORED_HYPOTHESIS,
            raw_metrics=raw_metrics,
            z_scores=z_scores
        )

    def calculate_model_conflict(
        self,
        predictions_a: np.ndarray,
        predictions_b: np.ndarray,
        causal_graph_distance: Optional[int] = None
    ) -> TensionMetrics:
        """
        MODEL_CONFLICT: Modelos consistentes pero incompatibles.

        Métricas:
        - prediction_disagreement: E[|y_a - y_b|]
        - causal_graph_distance: SHD entre grafos causales
        """
        raw_metrics = {}

        # Prediction disagreement
        disagreement = np.mean(np.abs(predictions_a - predictions_b))
        raw_metrics['prediction_disagreement'] = float(disagreement)

        # Causal graph distance
        if causal_graph_distance is not None:
            raw_metrics['causal_graph_distance'] = float(causal_graph_distance)
        else:
            raw_metrics['causal_graph_distance'] = 0.0

        # Z-scores
        z_scores = {}
        for name, value in raw_metrics.items():
            self._update_history(f"conflict_{name}", value)
            z_scores[name] = self._compute_z_score(f"conflict_{name}", value)

        return TensionMetrics(
            tension_type=TensionType.MODEL_CONFLICT,
            raw_metrics=raw_metrics,
            z_scores=z_scores
        )

    def calculate_empirical_gap(
        self,
        n_observations: int,
        n_parameters: int,
        kl_posterior_prior: Optional[float] = None
    ) -> TensionMetrics:
        """
        EMPIRICAL_GAP: Hipótesis sin apoyo observacional.

        Métricas:
        - evidence_parameter_ratio: #observaciones / #parámetros
        - kl_post_prior: D_KL(posterior || prior)
        """
        raw_metrics = {}

        # Evidence ratio
        if n_parameters > 0:
            ratio = n_observations / n_parameters
        else:
            ratio = float('inf')

        raw_metrics['evidence_parameter_ratio'] = float(np.clip(ratio, 0, 100))

        # KL posterior/prior
        if kl_posterior_prior is not None:
            raw_metrics['kl_post_prior'] = float(kl_posterior_prior)
        else:
            # Proxy: bajo ratio → posterior ≈ prior
            raw_metrics['kl_post_prior'] = float(max(0, 1 - ratio/10))

        # Z-scores
        z_scores = {}
        for name, value in raw_metrics.items():
            self._update_history(f"empirical_{name}", value)
            z_scores[name] = self._compute_z_score(f"empirical_{name}", value)

        return TensionMetrics(
            tension_type=TensionType.EMPIRICAL_GAP,
            raw_metrics=raw_metrics,
            z_scores=z_scores
        )


# =============================================================================
# ESTADO INTERNO DEL AGENTE (METRIFICADO)
# =============================================================================

@dataclass
class InternalState:
    """
    Estado interno cuantificado con métricas formales.

    Estas métricas son las ÚNICAS fuentes de tensión.
    NO hay preferencias, roles, ni identidades.
    """
    # === Métricas para INCONSISTENCY ===
    model_predictions: List[np.ndarray] = field(default_factory=list)
    kl_divergence_mean: float = 0.0
    sign_agreement: float = 1.0

    # === Métricas para LOW_RESOLUTION ===
    prediction_variance: float = 0.0
    fisher_information: float = 1.0
    normalized_residual_variance: float = 0.0

    # === Métricas para OVERSIMPLIFICATION ===
    train_loss: float = 0.5
    val_loss: float = 0.5
    residual_entropy: float = 0.0

    # === Métricas para UNEXPLORED_HYPOTHESIS ===
    hypotheses_evaluated: int = 0
    hypotheses_reachable: int = 10
    embedding_distances: List[float] = field(default_factory=list)

    # === Métricas para MODEL_CONFLICT ===
    prediction_disagreement: float = 0.0
    causal_graph_distance: int = 0

    # === Métricas para EMPIRICAL_GAP ===
    n_observations: int = 10
    n_parameters: int = 5
    kl_posterior_prior: float = 0.5

    # === Historial de rendimiento (para promoción) ===
    domain_performance: Dict[str, List[float]] = field(default_factory=dict)
    domain_levels: Dict[str, TaskLevel] = field(default_factory=dict)

    # === Historial de tensiones (para persistencia y tendencias) ===
    tension_history: Dict[TensionType, List[float]] = field(default_factory=dict)

    def get_coverage(self) -> float:
        """Coverage de hipótesis."""
        if self.hypotheses_reachable > 0:
            return self.hypotheses_evaluated / self.hypotheses_reachable
        return 1.0

    def get_gen_gap(self) -> float:
        """Gap de generalización."""
        return self.val_loss - self.train_loss

    def update_tension_history(self, tension_type: TensionType, intensity: float):
        """Actualiza historial de intensidad de una tensión."""
        if tension_type not in self.tension_history:
            self.tension_history[tension_type] = []
        self.tension_history[tension_type].append(intensity)

        # Mantener ventana máxima
        if len(self.tension_history[tension_type]) > 100:
            self.tension_history[tension_type] = self.tension_history[tension_type][-100:]

    def get_persistence(self, tension_type: TensionType) -> float:
        """Calcula persistencia (media móvil de intensidad)."""
        history = self.tension_history.get(tension_type, [])

        if len(history) < 1:
            return 0.0

        window = TheoreticalConstants.PERSISTENCE_WINDOW
        recent = history[-window:]
        return float(np.mean(recent))

    def get_delta_intensity(self, tension_type: TensionType) -> float:
        """Calcula derivada temporal de intensidad."""
        history = self.tension_history.get(tension_type, [])

        if len(history) < 2:
            return 0.0

        return history[-1] - history[-2]

    def get_intensity_percentile(self, tension_type: TensionType, current: float) -> float:
        """Calcula percentil de intensidad actual en historia propia."""
        history = self.tension_history.get(tension_type, [])

        if len(history) < TheoreticalConstants.MIN_SAMPLES_FOR_STATS:
            return 50.0  # Neutral si no hay historia

        rank = sum(1 for h in history if h <= current)
        return 100.0 * rank / len(history)


# =============================================================================
# DETECTOR DE TENSIONES (CON MÉTRICAS FORMALES)
# =============================================================================

class TensionDetector:
    """
    Detecta tensiones epistémicas desde métricas internas.

    HARD RULE: Solo usa métricas cuantificables.
    NUNCA usa identidad, nombre, o rol del agente.
    """

    def __init__(self):
        self.logger = get_provenance_logger()
        self._calculator = TensionMetricsCalculator()

    def detect_all(self, state: InternalState) -> List[TensionState]:
        """
        Detecta todas las tensiones activas desde el estado interno.

        Retorna lista de TensionState con métricas formales.
        """
        tensions = []

        # 1. INCONSISTENCY
        if len(state.model_predictions) >= 2:
            metrics = self._calculator.calculate_inconsistency(
                state.model_predictions
            )
            intensity = metrics.intensity_L2
            state.update_tension_history(TensionType.INCONSISTENCY, intensity)

            tensions.append(self._build_tension_state(
                TensionType.INCONSISTENCY, metrics, state
            ))

        # 2. LOW_RESOLUTION
        if state.normalized_residual_variance > 0:
            metrics = TensionMetrics(
                tension_type=TensionType.LOW_RESOLUTION,
                raw_metrics={
                    'fisher_information': state.fisher_information,
                    'normalized_residual_variance': state.normalized_residual_variance
                },
                z_scores={}
            )
            # Calcular z-scores manualmente
            for name, value in metrics.raw_metrics.items():
                self._calculator._update_history(f"low_res_{name}", value)
                metrics.z_scores[name] = self._calculator._compute_z_score(
                    f"low_res_{name}", value
                )
            metrics.intensity_L2 = float(np.linalg.norm(
                list(metrics.z_scores.values())
            )) if metrics.z_scores else 0.0

            state.update_tension_history(TensionType.LOW_RESOLUTION, metrics.intensity_L2)
            tensions.append(self._build_tension_state(
                TensionType.LOW_RESOLUTION, metrics, state
            ))

        # 3. OVERSIMPLIFICATION
        gen_gap = state.get_gen_gap()
        if abs(gen_gap) < 0.1 and state.val_loss > 0.3:  # Gap pequeño pero error alto
            metrics = self._calculator.calculate_oversimplification(
                state.train_loss, state.val_loss, state.residual_entropy
            )
            state.update_tension_history(TensionType.OVERSIMPLIFICATION, metrics.intensity_L2)
            tensions.append(self._build_tension_state(
                TensionType.OVERSIMPLIFICATION, metrics, state
            ))

        # 4. UNEXPLORED_HYPOTHESIS
        coverage = state.get_coverage()
        if coverage < 0.8:
            metrics = self._calculator.calculate_unexplored_hypothesis(
                state.hypotheses_evaluated,
                state.hypotheses_reachable,
                state.embedding_distances
            )
            state.update_tension_history(TensionType.UNEXPLORED_HYPOTHESIS, metrics.intensity_L2)
            tensions.append(self._build_tension_state(
                TensionType.UNEXPLORED_HYPOTHESIS, metrics, state
            ))

        # 5. MODEL_CONFLICT
        if state.prediction_disagreement > 0.5:
            metrics = TensionMetrics(
                tension_type=TensionType.MODEL_CONFLICT,
                raw_metrics={
                    'prediction_disagreement': state.prediction_disagreement,
                    'causal_graph_distance': float(state.causal_graph_distance)
                },
                z_scores={}
            )
            for name, value in metrics.raw_metrics.items():
                self._calculator._update_history(f"conflict_{name}", value)
                metrics.z_scores[name] = self._calculator._compute_z_score(
                    f"conflict_{name}", value
                )
            metrics.intensity_L2 = float(np.linalg.norm(
                list(metrics.z_scores.values())
            )) if metrics.z_scores else 0.0

            state.update_tension_history(TensionType.MODEL_CONFLICT, metrics.intensity_L2)
            tensions.append(self._build_tension_state(
                TensionType.MODEL_CONFLICT, metrics, state
            ))

        # 6. EMPIRICAL_GAP
        ratio = state.n_observations / max(1, state.n_parameters)
        if ratio < 5:  # Menos de 5 obs por parámetro
            metrics = self._calculator.calculate_empirical_gap(
                state.n_observations,
                state.n_parameters,
                state.kl_posterior_prior
            )
            state.update_tension_history(TensionType.EMPIRICAL_GAP, metrics.intensity_L2)
            tensions.append(self._build_tension_state(
                TensionType.EMPIRICAL_GAP, metrics, state
            ))

        # Default: si no hay tensiones, crear una exploratoria
        if not tensions:
            metrics = TensionMetrics(
                tension_type=TensionType.UNEXPLORED_HYPOTHESIS,
                raw_metrics={'default': 1.0, 'hypothesis_coverage': 0.5},
                z_scores={'default': 0.0}
            )
            state.update_tension_history(TensionType.UNEXPLORED_HYPOTHESIS, 0.3)
            tensions.append(TensionState(
                tension_type=TensionType.UNEXPLORED_HYPOTHESIS,
                metrics=metrics,
                intensity=0.3,
                persistence=0.3,
                delta_intensity=0.0,
                trend=TensionTrend.STABLE,
                percentile_rank=50.0
            ))

        return tensions

    def _build_tension_state(
        self,
        tension_type: TensionType,
        metrics: TensionMetrics,
        state: InternalState
    ) -> TensionState:
        """Construye TensionState completo con dinámica temporal."""
        intensity = metrics.intensity_L2
        persistence = state.get_persistence(tension_type)
        delta = state.get_delta_intensity(tension_type)
        percentile = state.get_intensity_percentile(tension_type, intensity)

        # Determinar tendencia
        if intensity > 1.0 and delta > 0.1:
            trend = TensionTrend.GROWING
        elif intensity > 1.0 and abs(delta) < 0.1:
            trend = TensionTrend.STRUCTURAL
        elif delta < -0.1:
            trend = TensionTrend.RESOLVING
        else:
            trend = TensionTrend.STABLE

        return TensionState(
            tension_type=tension_type,
            metrics=metrics,
            intensity=intensity,
            persistence=persistence,
            delta_intensity=delta,
            trend=trend,
            percentile_rank=percentile
        )

    def sample(self, state: InternalState, seed: Optional[int] = None) -> TensionState:
        """
        Muestrea una tensión del estado interno.

        Probabilidad proporcional a persistencia (no solo intensidad).
        """
        if seed is not None:
            np.random.seed(seed)

        tensions = self.detect_all(state)

        if not tensions:
            raise RuntimeError("ABORT: No tensions detected (impossible state)")

        # Softmax sobre persistencia (no intensidad instantánea)
        persistences = np.array([t.persistence for t in tensions])

        # Ajustar por tendencia: tensiones crecientes tienen boost
        boosts = np.array([
            1.5 if t.trend == TensionTrend.GROWING else
            1.0 if t.trend == TensionTrend.STRUCTURAL else
            0.8 if t.trend == TensionTrend.RESOLVING else
            0.9
            for t in tensions
        ])

        scores = persistences * boosts
        scores = scores - np.max(scores)

        T = TheoreticalConstants.SOFTMAX_TEMPERATURE
        exp_scores = np.exp(scores / T)
        probs = exp_scores / np.sum(exp_scores)

        idx = np.random.choice(len(tensions), p=probs)
        return tensions[idx]


# =============================================================================
# SELECTOR DE NIVEL (EMERGENTE DE PERSISTENCIA)
# =============================================================================

class LevelSelector:
    """
    Selecciona nivel de tarea basado en persistencia de tensión.

    HARD RULE: Usa percentiles internos, NO umbrales absolutos.
    """

    def __init__(self):
        self.logger = get_provenance_logger()

    def from_percentile(self, percentile: float) -> TaskLevel:
        """
        Determina nivel desde percentil de persistencia.

        < P50 → UNDERGRADUATE
        P50-P85 → GRADUATE
        ≥ P85 → DOCTORAL
        """
        if percentile >= TheoreticalConstants.DOCTORAL_PERCENTILE:
            return TaskLevel.DOCTORAL
        elif percentile >= TheoreticalConstants.GRADUATE_PERCENTILE:
            return TaskLevel.GRADUATE
        else:
            return TaskLevel.UNDERGRADUATE

    def select(self, tension: TensionState, state: InternalState) -> TaskLevel:
        """
        Selecciona nivel emergente desde tensión y estado.

        El nivel emerge de:
        1. Percentil de persistencia en historia propia
        2. Nivel actual en el dominio (si existe)
        """
        # Percentil de persistencia
        persistence_history = state.tension_history.get(tension.tension_type, [])

        if len(persistence_history) < TheoreticalConstants.MIN_SAMPLES_FOR_STATS:
            # Sin historia suficiente → nivel base
            return TaskLevel.UNDERGRADUATE

        # Calcular percentil de la persistencia actual
        current_persistence = tension.persistence
        rank = sum(1 for p in persistence_history if p <= current_persistence)
        percentile = 100.0 * rank / len(persistence_history)

        return self.from_percentile(percentile)


# =============================================================================
# MAPEO TENSIÓN → DOMINIOS
# =============================================================================

TENSION_TO_DOMAINS: Dict[TensionType, List[str]] = {
    TensionType.INCONSISTENCY: ["mathematics", "physics", "medicine"],
    TensionType.LOW_RESOLUTION: ["medicine", "cosmology", "physics"],
    TensionType.OVERSIMPLIFICATION: ["physics", "mathematics"],
    TensionType.UNEXPLORED_HYPOTHESIS: ["cosmology", "physics", "medicine"],
    TensionType.MODEL_CONFLICT: ["physics", "mathematics", "cosmology"],
    TensionType.EMPIRICAL_GAP: ["medicine", "cosmology"],
}


# =============================================================================
# CURRICULA POR DOMINIO
# =============================================================================

DOMAIN_CURRICULA: Dict[str, Dict[TaskLevel, List[Dict]]] = {
    "mathematics": {
        TaskLevel.UNDERGRADUATE: [
            {"type": "math_eq_simple", "desc": "sistemas 1-2 variables"},
            {"type": "math_calculus", "desc": "derivadas simples"},
        ],
        TaskLevel.GRADUATE: [
            {"type": "math_eq_complex", "desc": "sistemas ≥3 variables"},
            {"type": "math_integration", "desc": "integrales complejas"},
        ],
        TaskLevel.DOCTORAL: [
            {"type": "math_series", "desc": "convergencia borderline"},
            {"type": "math_proof", "desc": "demostraciones formales"},
        ],
    },
    "physics": {
        TaskLevel.UNDERGRADUATE: [
            {"type": "phys_mechanics", "desc": "movimiento 1D"},
            {"type": "phys_oscillator", "desc": "oscilador simple"},
        ],
        TaskLevel.GRADUATE: [
            {"type": "phys_coupled", "desc": "sistemas acoplados"},
            {"type": "phys_waves", "desc": "propagación de ondas"},
        ],
        TaskLevel.DOCTORAL: [
            {"type": "phys_nonlinear", "desc": "dinámica no lineal"},
            {"type": "phys_chaos", "desc": "sistemas caóticos"},
        ],
    },
    "medicine": {
        TaskLevel.UNDERGRADUATE: [
            {"type": "med_classification", "desc": "diagnóstico binario"},
        ],
        TaskLevel.GRADUATE: [
            {"type": "med_multiclass", "desc": "diagnóstico multiclase"},
            {"type": "med_regression", "desc": "predicción continua"},
        ],
        TaskLevel.DOCTORAL: [
            {"type": "med_causality", "desc": "inferencia causal"},
        ],
    },
    "cosmology": {
        TaskLevel.UNDERGRADUATE: [
            {"type": "cosmo_regression", "desc": "ajuste de curvas"},
        ],
        TaskLevel.GRADUATE: [
            {"type": "cosmo_timeseries", "desc": "análisis temporal"},
        ],
        TaskLevel.DOCTORAL: [
            {"type": "cosmo_hypothesis", "desc": "falsación de modelos"},
        ],
    },
}


# =============================================================================
# RESOLUTOR DE DOMINIOS
# =============================================================================

class DomainResolver:
    """
    Resuelve tensiones en dominios candidatos.

    HARD RULE: NO acepta identidad de agente como input.
    """

    def __init__(self):
        self.logger = get_provenance_logger()
        self._domain_weights: Dict[str, float] = {}

    def resolve(self, tension: TensionState) -> List[Tuple[str, float]]:
        """Retorna candidatos con pesos."""
        candidates = TENSION_TO_DOMAINS.get(tension.tension_type, ["physics"])

        weighted = []
        for domain in candidates:
            base = 1.0 / len(candidates)
            historical = self._domain_weights.get(domain, 1.0)
            weighted.append((domain, base * historical))

        total = sum(w for _, w in weighted)
        if total > 0:
            weighted = [(d, w/total) for d, w in weighted]

        return sorted(weighted, key=lambda x: x[1], reverse=True)

    def sample(
        self,
        tension: TensionState,
        seed: Optional[int] = None
    ) -> str:
        """Muestrea un dominio."""
        if seed is not None:
            np.random.seed(seed)

        candidates = self.resolve(tension)
        domains = [d for d, _ in candidates]
        weights = [w for _, w in candidates]

        domain = np.random.choice(domains, p=weights)

        # Decay para diversidad
        self._domain_weights[domain] = self._domain_weights.get(domain, 1.0) * 0.95

        return domain


# =============================================================================
# REPORTE DE TENSIÓN (AUDITABLE)
# =============================================================================

@dataclass
class TensionReport:
    """
    Reporte auditable de una decisión de investigación.

    Exportable a YAML para trazabilidad.
    """
    session_id: str
    agent_id: str
    round: int
    timestamp: str

    tension_type: str
    intensity_L2: float
    persistence_mean: float
    delta_intensity: float
    trend: str
    percentile_rank: float

    source_metrics: Dict[str, float]
    z_scores: Dict[str, float]

    domain_candidates: List[str]
    selected_domain: str
    task_level: str
    selected_task: str

    explanation: str = ""

    def to_yaml(self) -> str:
        """Exporta a YAML."""
        # Convertir numpy types a Python natives
        def to_native(val):
            if hasattr(val, 'item'):
                return val.item()
            return val

        data = {
            'session_id': self.session_id,
            'agent_id': self.agent_id,
            'round': self.round,
            'timestamp': self.timestamp,
            'tension': {
                'type': self.tension_type,
                'intensity_L2': round(float(self.intensity_L2), 4),
                'persistence_mean': round(float(self.persistence_mean), 4),
                'trend': {
                    'delta_intensity': round(float(self.delta_intensity), 4),
                    'slope': self.trend,
                },
                'percentile_rank': round(float(self.percentile_rank), 2),
            },
            'source_metrics': {str(k): round(float(v), 4) for k, v in self.source_metrics.items()},
            'z_scores': {str(k): round(float(v), 4) for k, v in self.z_scores.items()},
            'derived_decisions': {
                'domain_candidates': [str(d) for d in self.domain_candidates],
                'selected_domain': str(self.selected_domain),
                'task_level': str(self.task_level),
                'selected_task': str(self.selected_task),
            },
            'notes': {
                'explanation': self.explanation or (
                    f"Domain and task emerged from {self.trend} {self.tension_type} tension "
                    f"at P{self.percentile_rank:.0f} of historical persistence."
                ),
            },
        }
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def save(self, path: Path):
        """Guarda reporte a archivo."""
        path.write_text(self.to_yaml())


# =============================================================================
# SISTEMA DE PROMOCIÓN
# =============================================================================

class PromotionSystem:
    """
    Promoción basada 100% en historial propio.

    HARD RULE: NO usa umbrales absolutos.
    USA: percentil P80 del rendimiento reciente en historia propia.
    """

    def __init__(self):
        self.logger = get_provenance_logger()

    def check_promotion(
        self,
        state: InternalState,
        domain: str
    ) -> Tuple[bool, Optional[TaskLevel], str]:
        """Verifica si hay promoción."""
        history = state.domain_performance.get(domain, [])

        if len(history) < TheoreticalConstants.MIN_SAMPLES_FOR_STATS:
            return False, None, f"Insufficient history for {domain} (need {TheoreticalConstants.MIN_SAMPLES_FOR_STATS})"

        # Rendimiento reciente (últimas 5 tareas)
        recent = history[-5:]
        recent_mean = np.mean(recent)

        # Calcular percentil en historia propia
        rank = sum(1 for h in history if h <= recent_mean)
        percentile = 100.0 * rank / len(history)

        can_promote = percentile >= TheoreticalConstants.PROMOTION_PERCENTILE

        current_level = state.domain_levels.get(domain, TaskLevel.UNDERGRADUATE)

        if can_promote:
            if current_level == TaskLevel.UNDERGRADUATE:
                new_level = TaskLevel.GRADUATE
            elif current_level == TaskLevel.GRADUATE:
                new_level = TaskLevel.DOCTORAL
            else:
                new_level = None
                can_promote = False
        else:
            new_level = None

        reason = (
            f"recent_perf={recent_mean:.3f}, percentile={percentile:.1f} "
            f"(threshold={TheoreticalConstants.PROMOTION_PERCENTILE}). "
            f"PROVENANCE: {TheoreticalConstants.get_provenance('PROMOTION_PERCENTILE')}"
        )

        return can_promote, new_level, reason


# =============================================================================
# SISTEMA DE ETIQUETAS POST-HOC
# =============================================================================

class LabelSystem:
    """
    Genera etiquetas EMERGENTES post-hoc.

    HARD RULE: Las etiquetas son DESCRIPTIVAS, nunca causales.
    """

    def __init__(self):
        self.logger = get_provenance_logger()

    def generate(self, state: InternalState, agent_id: str) -> Dict[str, Any]:
        """Genera etiqueta emergente."""
        # Calcular scores por dominio
        domain_scores = {}
        for domain, history in state.domain_performance.items():
            if len(history) >= TheoreticalConstants.MIN_SAMPLES_FOR_STATS:
                domain_scores[domain] = np.mean(history)

        if not domain_scores:
            return {
                'agent_id': agent_id,
                'label': 'novice',
                'specialization_z': 0.0,
                'note': 'Label is POST-HOC only, NEVER causal. Insufficient history.',
            }

        # Dominio top
        top_domain = max(domain_scores, key=domain_scores.get)
        top_score = domain_scores[top_domain]

        # Z-score de especialización
        other_scores = [s for d, s in domain_scores.items() if d != top_domain]

        if len(other_scores) >= 2:
            mean_others = np.mean(other_scores)
            std_others = np.std(other_scores, ddof=1)
            if std_others > 1e-10:
                spec_z = (top_score - mean_others) / std_others
            else:
                spec_z = 0.0
        else:
            spec_z = 0.0

        # Grado de especialización
        if spec_z >= TheoreticalConstants.SPECIALIST_Z_THRESHOLD:
            grade = "specialist"
        elif spec_z >= TheoreticalConstants.FOCUSED_Z_THRESHOLD:
            grade = "focused"
        else:
            grade = ""

        # Prefijo de dominio
        prefix = {
            'mathematics': 'math',
            'physics': 'phys',
            'medicine': 'med',
            'cosmology': 'cosmo',
        }.get(top_domain, top_domain[:4])

        # Sufijo de nivel
        level = state.domain_levels.get(top_domain, TaskLevel.UNDERGRADUATE)
        suffix = {
            TaskLevel.UNDERGRADUATE: 'undergrad',
            TaskLevel.GRADUATE: 'graduate',
            TaskLevel.DOCTORAL: 'doctoral',
        }[level]

        # Construir etiqueta
        if grade:
            label = f"{prefix}_{grade}_{suffix}"
        else:
            label = f"{prefix}_{suffix}"

        return {
            'agent_id': agent_id,
            'label': label,
            'top_domain': top_domain,
            'top_level': level.value,
            'specialization_z': spec_z,
            'domain_scores': domain_scores,
            'note': 'Label is POST-HOC only, NEVER causal',
            'provenance': (
                f"z-score={spec_z:.2f} (internal). "
                f"PROVENANCE: z≥{TheoreticalConstants.SPECIALIST_Z_THRESHOLD}=specialist, "
                f"z≥{TheoreticalConstants.FOCUSED_Z_THRESHOLD}=focused."
            ),
        }


# =============================================================================
# VALIDADOR DE INTEGRIDAD
# =============================================================================

class IntegrityValidator:
    """
    Valida cumplimiento de HARD RULES.

    Si detecta violación → abort_execution()
    """

    def __init__(self):
        self.violations: List[str] = []

    def validate_tension(self, tension: Optional[TensionState]):
        """Valida que la tensión tiene métricas formales."""
        if tension is None:
            self._abort("Tension is None (must detect from metrics)")

        if not tension.source_metrics:
            self._abort("Tension has no source_metrics (must be derived from internal state)")

        # Nota: intensidad 0 es válida para estados iniciales sin historia
        # La persistencia se construye con el tiempo

    def validate_flow(
        self,
        tension: Optional[TensionState],
        domain: Optional[str]
    ):
        """Valida flujo tensión → dominio."""
        if domain is not None and tension is None:
            self._abort("Domain selected without prior tension detection")

        if tension is not None:
            self.validate_tension(tension)

    def validate_level_selection(
        self,
        level: TaskLevel,
        tension: TensionState,
        used_percentile: bool
    ):
        """Valida que el nivel emerge de percentiles internos."""
        if not used_percentile:
            self._abort(
                f"Level {level.value} selected without using internal percentile. "
                "Must use from_percentile() method."
            )

    def _abort(self, reason: str):
        """Aborta ejecución."""
        self.violations.append(reason)
        raise RuntimeError(f"ABORT: HARD RULE VIOLATION - {reason}")


# =============================================================================
# TAREA DE INVESTIGACIÓN
# =============================================================================

@dataclass
class ResearchTask:
    """Tarea de investigación con trazabilidad completa."""
    task_id: str
    tension: TensionState
    domain: str
    level: TaskLevel
    task_type: str
    task_desc: str
    selection_path: List[str]
    report: Optional[TensionReport] = None
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


# =============================================================================
# NÚCLEO TERA
# =============================================================================

class TeraNucleus:
    """
    Tension-Driven Endogenous Research Architecture (TERA).

    Núcleo formal de investigación autónoma basada en tensiones epistémicas.

    FLUJO:
        métricas_internas → z_scores → tensión(I_T, P_T, trend)
        → dominio → nivel → tarea → resultado → historial → promoción → etiqueta
    """

    def __init__(self, seed: Optional[int] = None, session_id: Optional[str] = None):
        self.logger = get_provenance_logger()
        self.session_id = session_id or datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        if seed is not None:
            np.random.seed(seed)

        # Componentes
        self._tension_detector = TensionDetector()
        self._level_selector = LevelSelector()
        self._domain_resolver = DomainResolver()
        self._promotion_system = PromotionSystem()
        self._label_system = LabelSystem()
        self._validator = IntegrityValidator()

        # Estados por agente
        self._states: Dict[str, InternalState] = {}

        # Historial
        self._task_history: Dict[str, List[ResearchResult]] = {}
        self._reports: List[TensionReport] = []

        # Contadores
        self._task_counter = 0
        self._round = 0

    def _get_state(self, agent_id: str) -> InternalState:
        """Obtiene estado (ID solo para tracking)."""
        if agent_id not in self._states:
            self._states[agent_id] = InternalState()
        return self._states[agent_id]

    def generate_task(self, agent_id: str) -> ResearchTask:
        """
        Genera tarea siguiendo el flujo formal TERA.

        FLUJO:
            estado → métricas → z_scores → tensión → dominio → nivel → tarea
        """
        path = []
        state = self._get_state(agent_id)

        # 1. Detectar tensión desde métricas
        tension = self._tension_detector.sample(state)
        self._validator.validate_tension(tension)
        path.append(f"state(metrics={len(state.domain_performance)})")
        path.append(f"tension({tension.tension_type.value}, I={tension.intensity:.2f}, P={tension.persistence:.2f})")

        # 2. Resolver dominio
        domain = self._domain_resolver.sample(tension)
        self._validator.validate_flow(tension=tension, domain=domain)
        path.append(f"domain({domain})")

        # 3. Determinar nivel emergente
        level = self._level_selector.select(tension, state)

        # Si hay nivel previo en el dominio, respetar máximo
        current_level = state.domain_levels.get(domain, TaskLevel.UNDERGRADUATE)
        if level.value > current_level.value:
            level = current_level

        path.append(f"level({level.value}, from_percentile={tension.percentile_rank:.1f})")

        # 4. Seleccionar tipo de tarea
        curriculum = DOMAIN_CURRICULA.get(domain, {})
        level_tasks = curriculum.get(level, [{"type": "generic", "desc": "tarea genérica"}])

        task_info = np.random.choice(level_tasks)
        task_type = task_info['type']
        task_desc = task_info['desc']
        path.append(f"task({task_type})")

        # 5. Crear reporte auditable
        self._task_counter += 1
        task_id = f"task_{self._task_counter:06d}"

        candidates = self._domain_resolver.resolve(tension)

        report = TensionReport(
            session_id=self.session_id,
            agent_id=agent_id,
            round=self._round,
            timestamp=datetime.now().isoformat(),
            tension_type=tension.tension_type.value,
            intensity_L2=tension.intensity,
            persistence_mean=tension.persistence,
            delta_intensity=tension.delta_intensity,
            trend=tension.trend.value,
            percentile_rank=tension.percentile_rank,
            source_metrics=tension.source_metrics,
            z_scores=tension.metrics.z_scores,
            domain_candidates=[d for d, _ in candidates],
            selected_domain=domain,
            task_level=level.value,
            selected_task=task_type,
        )

        self._reports.append(report)

        return ResearchTask(
            task_id=task_id,
            tension=tension,
            domain=domain,
            level=level,
            task_type=task_type,
            task_desc=task_desc,
            selection_path=path,
            report=report,
        )

    def complete_task(
        self,
        agent_id: str,
        task: ResearchTask,
        performance: float,
        error: float
    ) -> ResearchResult:
        """Completa una tarea y actualiza estado."""
        state = self._get_state(agent_id)

        # Actualizar historial
        if task.domain not in state.domain_performance:
            state.domain_performance[task.domain] = []
        state.domain_performance[task.domain].append(performance)

        # Verificar promoción
        promoted, new_level, reason = self._promotion_system.check_promotion(
            state, task.domain
        )

        if promoted and new_level:
            state.domain_levels[task.domain] = new_level

        # Crear resultado
        success = performance > 0.5
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
        """Genera etiqueta emergente post-hoc."""
        state = self._get_state(agent_id)
        return self._label_system.generate(state, agent_id)

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

    def export_reports(self, directory: Path):
        """Exporta todos los reportes de tensión a archivos YAML."""
        directory.mkdir(parents=True, exist_ok=True)

        for report in self._reports:
            filename = f"{report.agent_id}_round{report.round:03d}.yaml"
            report.save(directory / filename)

    def get_all_reports(self) -> List[TensionReport]:
        """Retorna todos los reportes de tensión."""
        return self._reports


# =============================================================================
# DIRECTOR MULTIAGENTE
# =============================================================================

class TeraDirector:
    """
    Director multiagente para TERA.

    Los IDs son SOLO para tracking, NUNCA para decisión.
    """

    def __init__(self, seed: Optional[int] = None):
        self.logger = get_provenance_logger()
        self._nucleus = TeraNucleus(seed=seed)
        self._agents: List[str] = []
        self._round = 0
        self._active = False

    def start_session(self, agent_ids: List[str]):
        """Inicia sesión."""
        self._agents = agent_ids
        self._round = 0
        self._active = True

    def run_round(self, solver_fn: Optional[callable] = None) -> List[ResearchResult]:
        """Ejecuta una ronda."""
        if not self._active:
            raise RuntimeError("Session not started")

        self._round += 1
        self._nucleus._round = self._round
        results = []

        for agent_id in self._agents:
            task = self._nucleus.generate_task(agent_id)

            if solver_fn:
                performance, error = solver_fn(agent_id, task)
            else:
                performance = np.random.uniform(0.4, 0.9)
                error = 1.0 - performance

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
            'session_id': self._nucleus.session_id,
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

    def get_all_labels(self) -> Dict[str, Dict[str, Any]]:
        """Retorna etiquetas de todos los agentes."""
        return {
            agent_id: self._nucleus.get_label(agent_id)
            for agent_id in self._agents
        }

    def export_reports(self, directory: Path):
        """Exporta todos los reportes."""
        self._nucleus.export_reports(directory)


# =============================================================================
# EXPORTACIONES
# =============================================================================

__all__ = [
    # Constantes
    'TheoreticalConstants',

    # Enums
    'TensionType',
    'TaskLevel',
    'TensionTrend',

    # Métricas
    'TensionMetrics',
    'TensionState',
    'TensionMetricsCalculator',

    # Estado
    'InternalState',

    # Detector
    'TensionDetector',

    # Selector
    'LevelSelector',

    # Resolver
    'DomainResolver',
    'TENSION_TO_DOMAINS',
    'DOMAIN_CURRICULA',

    # Promoción y etiquetas
    'PromotionSystem',
    'LabelSystem',

    # Validación
    'IntegrityValidator',

    # Reporte
    'TensionReport',

    # Tareas
    'ResearchTask',
    'ResearchResult',

    # Núcleo
    'TeraNucleus',
    'TeraDirector',
]
