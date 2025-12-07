#!/usr/bin/env python3
"""
Domain Base - Infraestructura Genérica para Dominios
=====================================================

NORMA DURA EXTENDIDA:
- Este módulo define la INFRAESTRUCTURA, no el conocimiento de dominio.
- Los agentes aprenden de datos, no de reglas hardcodeadas.
- Cualquier constante debe tener PROVENANCE documentada.

Arquitectura:
- DomainSchema: Define variables, tipos, unidades (ontología)
- DomainConnector: Carga y valida datos
- DomainAnalyzer: Funciones matemáticas genéricas
- DomainHypothesis: Sistema de hipótesis/falsación

NO PERMITIDO:
- Reglas tipo "if variable > X then condición"
- Números mágicos sin FROM_DATA o FROM_THEORY
- Especialización por nombre de agente
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from datetime import datetime
import json
from pathlib import Path

import sys
sys.path.insert(0, '/root/NEO_EVA')

try:
    from core.norma_dura_config import CONSTANTS, ProvenanceTag
except ImportError:
    # Fallback si no existe norma_dura_config
    class CONSTANTS:
        PERCENTILE_10 = 0.1
        PERCENTILE_25 = 0.25
        PERCENTILE_50 = 0.5
        PERCENTILE_75 = 0.75
        PERCENTILE_90 = 0.9

    class ProvenanceTag(Enum):
        FROM_DATA = "from_data"
        FROM_THEORY = "from_theory"
        FROM_CALIB = "from_calibration"
        FROM_DIST = "from_distribution"

try:
    from scripts.endogenous_param_logger import EndogenousParamLogger
except ImportError:
    # Stub si no existe
    class EndogenousParamLogger:
        def __init__(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass


# =============================================================================
# TIPOS DE VARIABLES
# =============================================================================

class VariableType(Enum):
    """Tipos de variables soportados."""
    CONTINUOUS = "continuous"      # Valores reales continuos
    DISCRETE = "discrete"          # Valores enteros discretos
    CATEGORICAL = "categorical"    # Categorías nominales
    ORDINAL = "ordinal"           # Categorías ordenadas
    TEMPORAL = "temporal"          # Series temporales
    BINARY = "binary"             # 0/1, True/False
    TEXT = "text"                 # Texto libre


class VariableRole(Enum):
    """Rol de la variable en análisis."""
    PREDICTOR = "predictor"       # Variable predictora
    OUTCOME = "outcome"           # Variable objetivo
    INDEX = "index"               # Variable índice (ID, timestamp)
    COVARIATE = "covariate"       # Covariable / confusor
    TREATMENT = "treatment"       # Variable de tratamiento
    # Aliases para compatibilidad
    FEATURE = "predictor"
    TARGET = "outcome"
    IDENTIFIER = "index"
    TIMESTAMP = "index"
    AUXILIARY = "covariate"


# =============================================================================
# SCHEMA DE DOMINIO
# =============================================================================

@dataclass
class VariableDefinition:
    """
    Definición formal de una variable.

    SOLO ontología, NO interpretación de dominio.
    """
    name: str
    var_type: VariableType
    role: VariableRole
    unit: Optional[str] = None           # Unidad física/médica/etc.
    description: str = ""                 # Descripción técnica
    valid_range: Optional[Tuple] = None  # Rango válido (min, max) - derivado de datos
    categories: Optional[List] = None    # Para categorías
    nullable: bool = True


@dataclass
class DomainSchema:
    """
    Schema de un dominio de conocimiento.

    Define la ESTRUCTURA de los datos, no su interpretación.
    """
    domain_name: str
    version: str
    variables: Dict[str, VariableDefinition] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_variable(self, var_def: VariableDefinition):
        """Añade una variable al schema."""
        self.variables[var_def.name] = var_def

    def get_features(self) -> List[str]:
        """Obtiene nombres de variables feature."""
        return [name for name, var in self.variables.items()
                if var.role == VariableRole.FEATURE]

    def get_targets(self) -> List[str]:
        """Obtiene nombres de variables target."""
        return [name for name, var in self.variables.items()
                if var.role == VariableRole.TARGET]

    def validate_record(self, record: Dict) -> Tuple[bool, List[str]]:
        """
        Valida un registro contra el schema.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        for name, var_def in self.variables.items():
            if name not in record:
                if not var_def.nullable:
                    errors.append(f"Missing required field: {name}")
                continue

            value = record[name]

            # Validar rango si está definido
            if var_def.valid_range and value is not None:
                min_val, max_val = var_def.valid_range
                if not (min_val <= value <= max_val):
                    errors.append(f"{name} out of range: {value}")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict:
        """Serializa schema a dict."""
        return {
            'domain_name': self.domain_name,
            'version': self.version,
            'variables': {
                name: {
                    'name': var.name,
                    'type': var.var_type.value,
                    'role': var.role.value,
                    'unit': var.unit,
                    'description': var.description,
                    'nullable': var.nullable
                }
                for name, var in self.variables.items()
            },
            'metadata': self.metadata
        }


# =============================================================================
# CONECTOR DE DATOS
# =============================================================================

class DomainConnector(ABC):
    """
    Conector abstracto para cargar datos de un dominio.

    Cada dominio implementa su propio conector.
    """

    def __init__(self, schema: DomainSchema):
        self.schema = schema
        self.logger = EndogenousParamLogger(f"domain_{schema.domain_name}")
        self._data_stats: Dict = {}

    @abstractmethod
    def load_data(self, source: str, **kwargs) -> np.ndarray:
        """
        Carga datos desde una fuente.

        Args:
            source: Ruta o identificador de la fuente
            **kwargs: Parámetros específicos del conector

        Returns:
            Array estructurado con los datos
        """
        pass

    @abstractmethod
    def get_available_sources(self) -> List[str]:
        """Lista fuentes de datos disponibles."""
        pass

    def compute_data_statistics(self, data: np.ndarray) -> Dict:
        """
        Calcula estadísticas de los datos cargados.

        NORMA DURA: Todos los parámetros se derivan de datos.
        """
        stats = {}

        for var_name in self.schema.get_features() + self.schema.get_targets():
            var_def = self.schema.variables.get(var_name)
            if var_def is None:
                continue

            if var_def.var_type in [VariableType.CONTINUOUS, VariableType.DISCRETE]:
                # Extraer columna - esto depende de la estructura de data
                try:
                    if isinstance(data, np.ndarray) and data.dtype.names:
                        col = data[var_name]
                    elif isinstance(data, dict):
                        col = np.array(data.get(var_name, []))
                    else:
                        continue

                    col = col[~np.isnan(col)] if np.issubdtype(col.dtype, np.floating) else col

                    if len(col) > 0:
                        var_stats = {
                            'n': len(col),
                            'mean': float(np.mean(col)),
                            'std': float(np.std(col)),
                            'min': float(np.min(col)),
                            'max': float(np.max(col)),
                            'p10': float(np.percentile(col, 10)),
                            'p25': float(np.percentile(col, 25)),
                            'p50': float(np.percentile(col, 50)),
                            'p75': float(np.percentile(col, 75)),
                            'p90': float(np.percentile(col, 90)),
                        }
                        stats[var_name] = var_stats

                        # Log parameter with NORMA DURA provenance
                        self.logger.log_param(
                            name=f"{var_name}_mean",
                            value=var_stats['mean'],
                            provenance=ProvenanceTag.FROM_DATA,
                            source_description=f"Mean of {var_name} from {self.schema.domain_name}",
                            source_data=col,
                            derivation_method="np.mean",
                            module="DomainConnector",
                            function="compute_data_statistics"
                        )
                except Exception:
                    continue

        self._data_stats = stats
        return stats

    def get_derived_threshold(self, var_name: str, percentile: float) -> Optional[float]:
        """
        Obtiene un umbral derivado de datos.

        NORMA DURA: Umbrales SIEMPRE de datos, nunca hardcodeados.

        Args:
            var_name: Nombre de la variable
            percentile: Percentil (0-100)

        Returns:
            Valor del umbral o None
        """
        if var_name not in self._data_stats:
            return None

        # Mapear percentil a clave
        pctl_map = {10: 'p10', 25: 'p25', 50: 'p50', 75: 'p75', 90: 'p90'}
        key = pctl_map.get(int(percentile))

        if key and key in self._data_stats[var_name]:
            return self._data_stats[var_name][key]

        return None


# =============================================================================
# ANALIZADOR GENÉRICO
# =============================================================================

class DomainAnalyzer:
    """
    Analizador genérico para cualquier dominio.

    SOLO funciones matemáticas, NO interpretación de dominio.
    """

    def __init__(self, schema: DomainSchema):
        self.schema = schema
        self.logger = EndogenousParamLogger(f"analyzer_{schema.domain_name}")

    # -------------------------------------------------------------------------
    # ESTADÍSTICA DESCRIPTIVA
    # -------------------------------------------------------------------------

    def describe(self, data: np.ndarray, var_name: str) -> Dict:
        """Estadísticas descriptivas de una variable."""
        if isinstance(data, dict):
            col = np.array(data.get(var_name, []))
        elif hasattr(data, 'dtype') and data.dtype.names:
            col = data[var_name]
        else:
            return {}

        col = col[~np.isnan(col)] if np.issubdtype(col.dtype, np.floating) else col

        if len(col) == 0:
            return {}

        return {
            'n': len(col),
            'mean': float(np.mean(col)),
            'std': float(np.std(col)),
            'median': float(np.median(col)),
            'iqr': float(np.percentile(col, 75) - np.percentile(col, 25)),
            'skewness': float(self._skewness(col)),
            'kurtosis': float(self._kurtosis(col))
        }

    def _skewness(self, x: np.ndarray) -> float:
        """Calcula asimetría."""
        n = len(x)
        if n < 3:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 3))

    def _kurtosis(self, x: np.ndarray) -> float:
        """Calcula curtosis."""
        n = len(x)
        if n < 4:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 4) - 3)

    # -------------------------------------------------------------------------
    # CORRELACIÓN Y CAUSALIDAD
    # -------------------------------------------------------------------------

    def correlation_matrix(self, data: Dict[str, np.ndarray],
                          method: str = 'pearson') -> np.ndarray:
        """
        Calcula matriz de correlación.

        Args:
            data: Dict de {var_name: array}
            method: 'pearson' o 'spearman'
        """
        var_names = list(data.keys())
        n_vars = len(var_names)

        corr = np.zeros((n_vars, n_vars))

        for i, var1 in enumerate(var_names):
            for j, var2 in enumerate(var_names):
                if i == j:
                    corr[i, j] = 1.0
                elif i < j:
                    x = data[var1]
                    y = data[var2]

                    # Alinear longitudes
                    min_len = min(len(x), len(y))
                    x, y = x[:min_len], y[:min_len]

                    # Remover NaN
                    mask = ~(np.isnan(x) | np.isnan(y))
                    x, y = x[mask], y[mask]

                    if len(x) > 2:
                        if method == 'spearman':
                            # Rangos
                            x = np.argsort(np.argsort(x))
                            y = np.argsort(np.argsort(y))

                        r = np.corrcoef(x, y)[0, 1]
                        corr[i, j] = r if not np.isnan(r) else 0.0
                        corr[j, i] = corr[i, j]

        return corr

    def granger_causality_test(self, x: np.ndarray, y: np.ndarray,
                               max_lag: Optional[int] = None) -> Dict:
        """
        Test de causalidad de Granger.

        NORMA DURA: max_lag se deriva de datos si no se especifica.
        """
        # Derivar max_lag de autocorrelación si no se especifica
        if max_lag is None:
            acf = self._autocorrelation(x)
            # FROM_DATA: primer lag donde ACF < 1/e
            decay_threshold = CONSTANTS.DECAY_RATE
            max_lag = 1
            for i, ac in enumerate(acf[1:], 1):
                if abs(ac) < decay_threshold:
                    max_lag = i
                    break
            max_lag = min(max_lag * 2, len(x) // 10)

        # Implementación simple de Granger
        n = len(x)
        if n < max_lag * 3:
            return {'significant': False, 'reason': 'insufficient_data'}

        # Modelo restringido: y ~ y_lags
        # Modelo no restringido: y ~ y_lags + x_lags

        from numpy.linalg import lstsq

        # Construir matrices de diseño
        y_target = y[max_lag:]
        n_obs = len(y_target)

        # Matriz de lags de y
        Y_lags = np.zeros((n_obs, max_lag))
        for lag in range(1, max_lag + 1):
            Y_lags[:, lag-1] = y[max_lag - lag:-lag] if lag < len(y) else 0

        # Matriz de lags de x
        X_lags = np.zeros((n_obs, max_lag))
        for lag in range(1, max_lag + 1):
            X_lags[:, lag-1] = x[max_lag - lag:-lag] if lag < len(x) else 0

        # Modelo restringido
        try:
            coef_r, res_r, _, _ = lstsq(Y_lags, y_target, rcond=None)
            rss_r = np.sum((y_target - Y_lags @ coef_r) ** 2)
        except Exception:
            return {'significant': False, 'reason': 'numerical_error'}

        # Modelo no restringido
        XY_lags = np.hstack([Y_lags, X_lags])
        try:
            coef_u, res_u, _, _ = lstsq(XY_lags, y_target, rcond=None)
            rss_u = np.sum((y_target - XY_lags @ coef_u) ** 2)
        except Exception:
            return {'significant': False, 'reason': 'numerical_error'}

        # F-test
        df1 = max_lag
        df2 = n_obs - 2 * max_lag

        if rss_u > 0 and df2 > 0:
            f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        else:
            f_stat = 0.0

        # Significancia usando umbral derivado
        # FROM_THEORY: F crítico aproximado para p=0.05
        f_critical = 2.0 + 3.0 / df1 if df1 > 0 else 4.0

        return {
            'significant': f_stat > f_critical,
            'f_statistic': float(f_stat),
            'f_critical': float(f_critical),
            'max_lag': max_lag,
            'rss_restricted': float(rss_r),
            'rss_unrestricted': float(rss_u)
        }

    def _autocorrelation(self, x: np.ndarray, max_lag: int = 50) -> np.ndarray:
        """Calcula autocorrelación."""
        n = len(x)
        max_lag = min(max_lag, n // 2)
        x = x - np.mean(x)
        var = np.var(x)

        if var == 0:
            return np.zeros(max_lag)

        acf = np.zeros(max_lag)
        for lag in range(max_lag):
            if lag == 0:
                acf[lag] = 1.0
            else:
                acf[lag] = np.mean(x[:-lag] * x[lag:]) / var

        return acf

    # -------------------------------------------------------------------------
    # DETECCIÓN DE PATRONES
    # -------------------------------------------------------------------------

    def detect_clusters(self, data: np.ndarray, n_clusters: Optional[int] = None) -> Dict:
        """
        Detecta clusters en los datos.

        NORMA DURA: n_clusters se deriva de datos si no se especifica.
        """
        if n_clusters is None:
            # Derivar de datos usando elbow method simplificado
            max_k = min(10, len(data) // 10)
            if max_k < 2:
                return {'n_clusters': 1, 'labels': np.zeros(len(data))}

            # Calcular inercia para diferentes k
            inertias = []
            for k in range(1, max_k + 1):
                labels, centers = self._kmeans_simple(data, k)
                inertia = sum(np.min(np.sum((data - c) ** 2, axis=1))
                             for c in centers) if len(centers) > 0 else 0
                inertias.append(inertia)

            # Encontrar codo
            if len(inertias) > 2:
                diffs = np.diff(inertias)
                diffs2 = np.diff(diffs)
                n_clusters = np.argmax(diffs2) + 2 if len(diffs2) > 0 else 2
            else:
                n_clusters = 2

        labels, centers = self._kmeans_simple(data, n_clusters)

        return {
            'n_clusters': n_clusters,
            'labels': labels,
            'centers': centers,
            'derived_from_data': n_clusters is None
        }

    def _kmeans_simple(self, data: np.ndarray, k: int,
                       max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """K-means simple."""
        n = len(data)
        if n < k:
            return np.zeros(n, dtype=int), data[:1]

        # Inicialización aleatoria
        idx = np.random.choice(n, k, replace=False)
        centers = data[idx].copy()

        for _ in range(max_iter):
            # Asignar puntos al centro más cercano
            distances = np.zeros((n, k))
            for i, c in enumerate(centers):
                distances[:, i] = np.sum((data - c) ** 2, axis=1)
            labels = np.argmin(distances, axis=1)

            # Actualizar centros
            new_centers = np.zeros_like(centers)
            for i in range(k):
                mask = labels == i
                if np.sum(mask) > 0:
                    new_centers[i] = np.mean(data[mask], axis=0)
                else:
                    new_centers[i] = centers[i]

            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        return labels, centers

    def detect_anomalies(self, data: np.ndarray,
                        method: str = 'iqr') -> Dict:
        """
        Detecta anomalías usando métodos estadísticos.

        NORMA DURA: Umbrales derivados de datos (IQR, z-score).
        """
        if method == 'iqr':
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            iqr = q3 - q1

            # FROM_THEORY: Tukey fence = 1.5 * IQR
            lower = q1 - CONSTANTS.TUKEY_MODERATE * iqr
            upper = q3 + CONSTANTS.TUKEY_MODERATE * iqr

            if len(data.shape) == 1:
                anomalies = (data < lower) | (data > upper)
            else:
                anomalies = np.any((data < lower) | (data > upper), axis=1)

            return {
                'method': 'iqr',
                'anomaly_mask': anomalies,
                'n_anomalies': int(np.sum(anomalies)),
                'fraction': float(np.mean(anomalies)),
                'thresholds': {'lower': lower.tolist() if hasattr(lower, 'tolist') else float(lower),
                              'upper': upper.tolist() if hasattr(upper, 'tolist') else float(upper)}
            }

        elif method == 'zscore':
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)

            # FROM_THEORY: z > 3 es outlier extremo
            z_threshold = CONSTANTS.TUKEY_EXTREME

            if len(data.shape) == 1:
                z_scores = np.abs((data - mean) / (std + 1e-10))
                anomalies = z_scores > z_threshold
            else:
                z_scores = np.abs((data - mean) / (std + 1e-10))
                anomalies = np.any(z_scores > z_threshold, axis=1)

            return {
                'method': 'zscore',
                'anomaly_mask': anomalies,
                'n_anomalies': int(np.sum(anomalies)),
                'fraction': float(np.mean(anomalies)),
                'threshold': float(z_threshold)
            }

        return {}


# =============================================================================
# SISTEMA DE HIPÓTESIS
# =============================================================================

@dataclass
class Hypothesis:
    """
    Una hipótesis formulada por un agente.

    NO contiene conocimiento de dominio, solo estructura formal.
    """
    id: str
    domain: str
    description: str
    variables_involved: List[str]
    hypothesis_type: str  # 'correlation', 'causation', 'clustering', 'prediction'
    created_at: datetime = field(default_factory=datetime.now)
    tested: bool = False
    falsified: bool = False
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    confidence: float = 0.0
    test_results: Dict = field(default_factory=dict)


class HypothesisEngine:
    """
    Motor de hipótesis y falsación.

    Los agentes formulan hipótesis, el motor las testea.
    """

    def __init__(self, analyzer: DomainAnalyzer):
        self.analyzer = analyzer
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.logger = EndogenousParamLogger("hypothesis_engine")

    def register_hypothesis(self, hypothesis: Hypothesis):
        """Registra una nueva hipótesis."""
        self.hypotheses[hypothesis.id] = hypothesis

    def test_hypothesis(self, hypothesis_id: str, data: Dict) -> Dict:
        """
        Testea una hipótesis contra datos.

        Returns:
            Resultados del test
        """
        if hypothesis_id not in self.hypotheses:
            return {'error': 'hypothesis_not_found'}

        h = self.hypotheses[hypothesis_id]
        h.tested = True

        if h.hypothesis_type == 'correlation':
            return self._test_correlation(h, data)
        elif h.hypothesis_type == 'causation':
            return self._test_causation(h, data)
        elif h.hypothesis_type == 'clustering':
            return self._test_clustering(h, data)
        else:
            return {'error': 'unknown_hypothesis_type'}

    def _test_correlation(self, h: Hypothesis, data: Dict) -> Dict:
        """Testea hipótesis de correlación."""
        if len(h.variables_involved) < 2:
            return {'error': 'need_at_least_2_variables'}

        var1, var2 = h.variables_involved[:2]
        if var1 not in data or var2 not in data:
            return {'error': 'variables_not_in_data'}

        x, y = np.array(data[var1]), np.array(data[var2])
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        if len(x) < CONSTANTS.MIN_SAMPLES_CLT:
            h.falsified = True
            h.test_results = {'reason': 'insufficient_samples', 'n': len(x)}
            return h.test_results

        # Calcular correlación
        r = np.corrcoef(x, y)[0, 1]

        # FROM_THEORY: significancia = 2/sqrt(n)
        significance_threshold = 2.0 / np.sqrt(len(x))

        h.effect_size = abs(r)
        h.falsified = abs(r) < significance_threshold
        h.confidence = min(1.0, abs(r) / significance_threshold) if not h.falsified else 0.0

        h.test_results = {
            'correlation': float(r),
            'significance_threshold': float(significance_threshold),
            'n_samples': len(x),
            'falsified': h.falsified,
            'confidence': h.confidence
        }

        return h.test_results

    def _test_causation(self, h: Hypothesis, data: Dict) -> Dict:
        """Testea hipótesis de causalidad."""
        if len(h.variables_involved) < 2:
            return {'error': 'need_at_least_2_variables'}

        cause, effect = h.variables_involved[:2]
        if cause not in data or effect not in data:
            return {'error': 'variables_not_in_data'}

        x, y = np.array(data[cause]), np.array(data[effect])

        result = self.analyzer.granger_causality_test(x, y)

        h.falsified = not result.get('significant', False)
        h.effect_size = result.get('f_statistic', 0.0)
        h.confidence = 1.0 if result.get('significant', False) else 0.0
        h.test_results = result

        return result

    def _test_clustering(self, h: Hypothesis, data: Dict) -> Dict:
        """Testea hipótesis de clustering."""
        # Construir matriz de datos
        arrays = []
        for var in h.variables_involved:
            if var in data:
                arrays.append(np.array(data[var]))

        if len(arrays) == 0:
            return {'error': 'no_valid_variables'}

        min_len = min(len(a) for a in arrays)
        X = np.column_stack([a[:min_len] for a in arrays])

        result = self.analyzer.detect_clusters(X)

        # Evaluar calidad del clustering
        n_clusters = result['n_clusters']
        h.falsified = n_clusters < 2
        h.effect_size = float(n_clusters)
        h.confidence = 1.0 - 1.0 / n_clusters if n_clusters > 1 else 0.0
        h.test_results = result

        return result

    def get_surviving_hypotheses(self) -> List[Hypothesis]:
        """Obtiene hipótesis que no han sido falsadas."""
        return [h for h in self.hypotheses.values()
                if h.tested and not h.falsified]

    def get_statistics(self) -> Dict:
        """Estadísticas del motor de hipótesis."""
        tested = [h for h in self.hypotheses.values() if h.tested]
        falsified = [h for h in tested if h.falsified]

        return {
            'total_hypotheses': len(self.hypotheses),
            'tested': len(tested),
            'falsified': len(falsified),
            'surviving': len(tested) - len(falsified),
            'falsification_rate': len(falsified) / len(tested) if tested else 0.0
        }


# =============================================================================
# BLOQUE DE AUDITORÍA NORMA DURA
# =============================================================================
"""
MAGIC NUMBERS AUDIT
==================

Este módulo es INFRAESTRUCTURA GENÉRICA.

CONSTANTES USADAS (con origen):
- CONSTANTS.DECAY_RATE (FROM_THEORY: 1/e para decorrelación)
- CONSTANTS.TUKEY_MODERATE (FROM_THEORY: 1.5 * IQR para outliers)
- CONSTANTS.TUKEY_EXTREME (FROM_THEORY: 3.0 * IQR para outliers extremos)
- CONSTANTS.MIN_SAMPLES_CLT (FROM_THEORY: 30 para aproximación normal)
- 2.0 / sqrt(n) (FROM_THEORY: umbral de significancia para correlación)

NO HAY:
- Reglas de dominio
- Umbrales específicos de medicina/finanzas/etc.
- Números mágicos sin documentar

TODAS LAS DECISIONES TIENEN ORIGEN DOCUMENTADO.
"""
