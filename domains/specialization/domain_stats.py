"""
DOMAIN STATS - Estadísticas de Rendimiento por Dominio
=======================================================

Registra métricas de rendimiento de un agente en un dominio.

NORMA DURA:
- Todas las métricas derivadas de datos
- Sin umbrales fijos
- Trazabilidad de procedencia
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stimuli_engine.provenance import (
    Provenance, ProvenanceType, get_provenance_logger, THEORY_CONSTANTS
)


class MetricType(Enum):
    """Tipos de métricas permitidas."""
    AUROC = "auroc"                 # Área bajo curva ROC
    ACCURACY = "accuracy"           # Precisión
    BRIER = "brier"                 # Brier score (calibración)
    MSE = "mse"                     # Error cuadrático medio
    MAE = "mae"                     # Error absoluto medio
    FPR = "fpr"                     # Tasa de falsos positivos
    FNR = "fnr"                     # Tasa de falsos negativos
    FALSIFICATION = "falsification" # Tasa de hipótesis falsadas
    PREDICTION_ERROR = "pred_error" # Error de predicción en series


@dataclass
class DomainMetrics:
    """
    Métricas de una evaluación en un dominio.

    NORMA DURA: Solo métricas derivadas de datos.
    """
    task_id: str
    domain: str
    timestamp: str = ""

    # Métricas de clasificación (cuando aplica)
    auroc: Optional[float] = None
    accuracy: Optional[float] = None
    brier_score: Optional[float] = None
    fpr: Optional[float] = None
    fnr: Optional[float] = None

    # Métricas de regresión (cuando aplica)
    mse: Optional[float] = None
    mae: Optional[float] = None
    r_squared: Optional[float] = None

    # Métricas de hipótesis
    n_hypotheses_generated: int = 0
    n_hypotheses_confirmed: int = 0
    n_hypotheses_falsified: int = 0

    # Metadatos
    n_samples: int = 0
    task_type: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @property
    def falsification_rate(self) -> float:
        """Tasa de hipótesis falsadas."""
        total = self.n_hypotheses_generated
        if total == 0:
            return 0.0
        return self.n_hypotheses_falsified / total

    @property
    def confirmation_rate(self) -> float:
        """Tasa de hipótesis confirmadas."""
        total = self.n_hypotheses_generated
        if total == 0:
            return 0.0
        return self.n_hypotheses_confirmed / total

    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'domain': self.domain,
            'timestamp': self.timestamp,
            'auroc': self.auroc,
            'accuracy': self.accuracy,
            'brier_score': self.brier_score,
            'fpr': self.fpr,
            'fnr': self.fnr,
            'mse': self.mse,
            'mae': self.mae,
            'r_squared': self.r_squared,
            'n_hypotheses_generated': self.n_hypotheses_generated,
            'n_hypotheses_confirmed': self.n_hypotheses_confirmed,
            'n_hypotheses_falsified': self.n_hypotheses_falsified,
            'falsification_rate': self.falsification_rate,
            'confirmation_rate': self.confirmation_rate,
            'n_samples': self.n_samples,
            'task_type': self.task_type,
        }


@dataclass
class DomainStats:
    """
    Estadísticas acumuladas de un agente en un dominio.

    NORMA DURA:
    - Todos los umbrales derivados de la distribución de datos
    - Sin valores mágicos
    - Trazabilidad completa
    """
    domain: str
    metrics_history: List[DomainMetrics] = field(default_factory=list)

    # Caches de distribuciones (para percentiles)
    _auroc_distribution: List[float] = field(default_factory=list)
    _accuracy_distribution: List[float] = field(default_factory=list)
    _falsification_distribution: List[float] = field(default_factory=list)
    _mse_distribution: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.logger = get_provenance_logger()

    def add_metrics(self, metrics: DomainMetrics):
        """Añade métricas de una tarea."""
        self.metrics_history.append(metrics)

        # Actualizar distribuciones
        if metrics.auroc is not None:
            self._auroc_distribution.append(metrics.auroc)
        if metrics.accuracy is not None:
            self._accuracy_distribution.append(metrics.accuracy)
        if metrics.n_hypotheses_generated > 0:
            self._falsification_distribution.append(metrics.falsification_rate)
        if metrics.mse is not None:
            self._mse_distribution.append(metrics.mse)

    @property
    def n_tasks(self) -> int:
        """Número de tareas intentadas."""
        return len(self.metrics_history)

    def get_mean_metric(self, metric_name: str) -> Tuple[Optional[float], Provenance]:
        """
        Obtiene media de una métrica.

        Returns:
            (valor, procedencia)
        """
        if metric_name == 'auroc':
            dist = self._auroc_distribution
        elif metric_name == 'accuracy':
            dist = self._accuracy_distribution
        elif metric_name == 'falsification':
            dist = self._falsification_distribution
        elif metric_name == 'mse':
            dist = self._mse_distribution
        else:
            return None, Provenance(
                value=None,
                ptype=ProvenanceType.UNKNOWN,
                source=f"Unknown metric: {metric_name}"
            )

        if not dist:
            return None, Provenance(
                value=None,
                ptype=ProvenanceType.FROM_DATA,
                source=f"No data for {metric_name}"
            )

        mean_val = float(np.mean(dist))
        prov = self.logger.log_from_data(
            value=mean_val,
            source=f"mean({metric_name})",
            dataset=f"domain={self.domain}, n={len(dist)}",
            statistic="arithmetic_mean",
            context="DomainStats.get_mean_metric"
        )

        return mean_val, prov

    def get_std_metric(self, metric_name: str) -> Tuple[Optional[float], Provenance]:
        """Obtiene desviación estándar de una métrica."""
        if metric_name == 'auroc':
            dist = self._auroc_distribution
        elif metric_name == 'accuracy':
            dist = self._accuracy_distribution
        elif metric_name == 'falsification':
            dist = self._falsification_distribution
        elif metric_name == 'mse':
            dist = self._mse_distribution
        else:
            return None, Provenance(
                value=None,
                ptype=ProvenanceType.UNKNOWN,
                source=f"Unknown metric: {metric_name}"
            )

        min_samples = THEORY_CONSTANTS['min_samples_corr'].value

        if len(dist) < min_samples:
            return None, Provenance(
                value=None,
                ptype=ProvenanceType.FROM_THEORY,
                source=f"n={len(dist)} < min_samples={min_samples}"
            )

        # ORIGEN: ddof=1 para sample std
        std_val = float(np.std(dist, ddof=1))
        prov = self.logger.log_from_data(
            value=std_val,
            source=f"std({metric_name}, ddof=1)",
            dataset=f"domain={self.domain}, n={len(dist)}",
            statistic="sample_std",
            context="DomainStats.get_std_metric"
        )

        return std_val, prov

    def get_percentile_metric(
        self,
        metric_name: str,
        percentile: float
    ) -> Tuple[Optional[float], Provenance]:
        """
        Obtiene percentil de una métrica.

        Args:
            metric_name: Nombre de la métrica
            percentile: Percentil (0-100)

        NORMA DURA: percentile viene de fuera (configurable),
        pero su uso está documentado.
        """
        if metric_name == 'auroc':
            dist = self._auroc_distribution
        elif metric_name == 'accuracy':
            dist = self._accuracy_distribution
        elif metric_name == 'falsification':
            dist = self._falsification_distribution
        elif metric_name == 'mse':
            dist = self._mse_distribution
        else:
            return None, Provenance(
                value=None,
                ptype=ProvenanceType.UNKNOWN,
                source=f"Unknown metric: {metric_name}"
            )

        min_samples = THEORY_CONSTANTS['min_samples_corr'].value

        if len(dist) < min_samples:
            return None, Provenance(
                value=None,
                ptype=ProvenanceType.FROM_THEORY,
                source=f"n={len(dist)} < min_samples={min_samples}"
            )

        p_val = float(np.percentile(dist, percentile))
        prov = self.logger.log_from_data(
            value=p_val,
            source=f"percentile({metric_name}, {percentile})",
            dataset=f"domain={self.domain}, n={len(dist)}",
            statistic=f"percentile_{percentile}",
            context="DomainStats.get_percentile_metric"
        )

        return p_val, prov

    def get_z_score(
        self,
        metric_name: str,
        value: float
    ) -> Tuple[Optional[float], Provenance]:
        """
        Calcula z-score de un valor respecto a la distribución.

        ORIGEN: z = (x - μ) / σ
        """
        mean_val, _ = self.get_mean_metric(metric_name)
        std_val, _ = self.get_std_metric(metric_name)

        if mean_val is None or std_val is None:
            return None, Provenance(
                value=None,
                ptype=ProvenanceType.FROM_DATA,
                source="Insufficient data for z-score"
            )

        eps = np.finfo(float).eps
        if std_val < eps:
            z = 0.0
        else:
            z = (value - mean_val) / std_val

        prov = self.logger.log_from_theory(
            value=z,
            source="z-score = (x - μ) / σ",
            reference="Estadística estándar",
            context="DomainStats.get_z_score"
        )

        return z, prov

    def get_stability(self) -> Tuple[Optional[float], Provenance]:
        """
        Calcula estabilidad del rendimiento.

        ORIGEN: stability = 1 / (1 + CV)
        donde CV = σ / μ (coeficiente de variación)

        Alta estabilidad = bajo CV = rendimiento consistente
        """
        # Usar AUROC si hay, sino accuracy, sino MSE
        if self._auroc_distribution:
            dist = self._auroc_distribution
            metric = 'auroc'
        elif self._accuracy_distribution:
            dist = self._accuracy_distribution
            metric = 'accuracy'
        elif self._mse_distribution:
            dist = self._mse_distribution
            metric = 'mse'
        else:
            return None, Provenance(
                value=None,
                ptype=ProvenanceType.FROM_DATA,
                source="No metrics for stability"
            )

        min_samples = THEORY_CONSTANTS['min_samples_corr'].value
        if len(dist) < min_samples:
            return None, Provenance(
                value=None,
                ptype=ProvenanceType.FROM_THEORY,
                source=f"n={len(dist)} < min_samples"
            )

        mean_val = np.mean(dist)
        std_val = np.std(dist, ddof=1)

        eps = np.finfo(float).eps
        if abs(mean_val) < eps:
            cv = float('inf')
        else:
            cv = std_val / abs(mean_val)

        # ORIGEN: Transformación para mapear CV a [0, 1]
        stability = 1.0 / (1.0 + cv)

        prov = self.logger.log_from_theory(
            value=stability,
            source="stability = 1 / (1 + CV), CV = σ/μ",
            reference="Coeficiente de variación",
            context="DomainStats.get_stability"
        )

        return stability, prov

    def get_summary(self) -> Dict[str, Any]:
        """Resumen de estadísticas del dominio."""
        auroc_mean, _ = self.get_mean_metric('auroc')
        accuracy_mean, _ = self.get_mean_metric('accuracy')
        falsification_mean, _ = self.get_mean_metric('falsification')
        stability, _ = self.get_stability()

        return {
            'domain': self.domain,
            'n_tasks': self.n_tasks,
            'auroc_mean': auroc_mean,
            'accuracy_mean': accuracy_mean,
            'falsification_rate_mean': falsification_mean,
            'stability': stability,
            'n_auroc_samples': len(self._auroc_distribution),
            'n_accuracy_samples': len(self._accuracy_distribution),
        }
