"""
DOMAIN TASK SAMPLER - Generador de Tareas por Dominio
======================================================

Genera tareas de los diferentes dominios para que los agentes
las intenten resolver.

NORMA DURA:
- NO codifica reglas específicas del dominio
- Solo genera datos + oracle para evaluación
- Sin umbrales hardcodeados
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from enum import Enum
import hashlib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stimuli_engine.provenance import get_provenance_logger


class TaskType(Enum):
    """Tipos de tareas genéricas (no específicas de dominio)."""
    CLASSIFICATION = "classification"   # Clasificar instancias
    REGRESSION = "regression"           # Predecir valor continuo
    ANOMALY = "anomaly"                 # Detectar anomalías
    CAUSALITY = "causality"             # Inferir relaciones causales
    TIMESERIES = "timeseries"           # Predecir series temporales
    CLUSTERING = "clustering"           # Agrupar instancias


class EvaluationMode(Enum):
    """Modo de evaluación de tareas."""
    GROUND_TRUTH = "ground_truth"                    # Comparar con solución conocida
    HYPOTHESIS_FALSIFICATION = "hypothesis_falsification"  # Evaluar por falsación


@dataclass
class Task:
    """
    Una tarea genérica para que un agente intente resolver.

    NO contiene lógica específica del dominio.
    Solo datos + metadatos + oracle para evaluación.

    NORMA DURA:
    - has_ground_truth: indica si hay solución conocida
    - evaluation_mode: cómo evaluar al agente
    - ground_truth_provenance: documenta origen de la solución
    """
    task_id: str
    domain: str
    task_type: TaskType

    # Datos de entrada (anónimos)
    X: np.ndarray                       # Features
    y: Optional[np.ndarray] = None      # Labels (si supervisado)

    # Metadatos estructurales (no semánticos)
    n_samples: int = 0
    n_features: int = 0
    feature_names: List[str] = field(default_factory=list)  # f_001, f_002, ...

    # === NUEVO: Ground truth y modo de evaluación ===
    has_ground_truth: bool = True                    # ¿Tiene solución conocida?
    ground_truth_provenance: str = ""                # Origen de la solución
    evaluation_mode: EvaluationMode = EvaluationMode.GROUND_TRUTH

    # Parámetros adicionales de la tarea
    params: Dict[str, Any] = field(default_factory=dict)

    # Solución oracle (solo si has_ground_truth=True)
    oracle_solution: Optional[Any] = None

    # Oracle para evaluación (función que evalúa predicciones)
    # El agente NO tiene acceso directo a esto
    _oracle: Optional[Callable] = None

    # Timestamps
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.n_samples == 0 and self.X is not None:
            self.n_samples = self.X.shape[0]
        if self.n_features == 0 and self.X is not None and len(self.X.shape) > 1:
            self.n_features = self.X.shape[1]
        if not self.feature_names and self.n_features > 0:
            self.feature_names = [f"f_{i:03d}" for i in range(self.n_features)]

    def get_train_test_split(
        self,
        test_fraction: float = 0.2,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide datos en train/test.

        ORIGEN: test_fraction viene de fuera (configurable).
        """
        if seed is not None:
            np.random.seed(seed)

        n = self.n_samples
        indices = np.random.permutation(n)

        # ORIGEN: División basada en fracción especificada
        split_idx = int(n * (1 - test_fraction))

        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        X_train = self.X[train_idx]
        X_test = self.X[test_idx]

        if self.y is not None:
            y_train = self.y[train_idx]
            y_test = self.y[test_idx]
        else:
            y_train = None
            y_test = None

        return X_train, X_test, y_train, y_test


@dataclass
class TaskResult:
    """
    Resultado de un agente en una tarea.

    Contiene predicciones y métricas de evaluación.

    NORMA DURA:
    - Para tareas con ground_truth: usa predictions + métricas estándar
    - Para tareas sin ground_truth: usa hipótesis + falsificación
    """
    task_id: str
    agent_id: str

    # Predicciones del agente (para tareas con ground truth)
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None  # Para clasificación

    # Respuesta genérica (para tareas de física/matemáticas)
    solution: Optional[Any] = None

    # Hipótesis generadas (para tareas sin ground truth)
    hypotheses_generated: List[Dict] = field(default_factory=list)
    hypotheses_confirmed: List[Dict] = field(default_factory=list)
    hypotheses_falsified: List[Dict] = field(default_factory=list)

    # Métricas de estabilidad bajo surrogates
    surrogate_stability: Optional[float] = None

    # Métricas calculadas por el oracle
    metrics: Dict[str, float] = field(default_factory=dict)

    # Timestamps
    started_at: str = ""
    completed_at: str = ""

    @property
    def falsification_rate(self) -> float:
        """Tasa de hipótesis falsadas."""
        n_gen = len(self.hypotheses_generated)
        if n_gen == 0:
            return 0.0
        return len(self.hypotheses_falsified) / n_gen

    @property
    def has_predictions(self) -> bool:
        """¿Tiene predicciones estándar?"""
        return self.predictions is not None or self.solution is not None


class DomainTaskSampler:
    """
    Generador de tareas desde los conectores de dominio.

    RESPONSABILIDADES:
    - Conectar con los domain connectors existentes
    - Generar tareas genéricas (sin lógica específica)
    - Proporcionar oracles para evaluación
    - NO interpretar los datos

    El agente recibe datos anónimos y decide cómo analizarlos.
    """

    def __init__(self, domains: Optional[List[str]] = None):
        """
        Args:
            domains: Lista de dominios disponibles
        """
        self.logger = get_provenance_logger()
        self.domains = domains or ['medicine', 'finance', 'cosmology', 'engineering']
        self._task_counter = 0
        self._connectors = {}

        # Inicializar conectores
        self._init_connectors()

    def _init_connectors(self):
        """Inicializa conectores de dominio."""
        # Importar conectores existentes
        try:
            from domains.medicine.medicine_connector import MedicineConnector
            self._connectors['medicine'] = MedicineConnector()
        except ImportError:
            pass

        try:
            from domains.finance.finance_connector import FinanceConnector
            self._connectors['finance'] = FinanceConnector()
        except ImportError:
            pass

        try:
            from domains.cosmology.cosmology_connector import CosmologyConnector
            self._connectors['cosmology'] = CosmologyConnector()
        except ImportError:
            pass

        try:
            from domains.engineering.engineering_connector import EngineeringConnector
            self._connectors['engineering'] = EngineeringConnector()
        except ImportError:
            pass

    def _next_task_id(self) -> str:
        """Genera siguiente ID de tarea."""
        self._task_counter += 1
        return f"task_{self._task_counter:06d}"

    def get_available_domains(self) -> List[str]:
        """Retorna dominios disponibles."""
        return list(self._connectors.keys())

    def sample_task(
        self,
        domain: str,
        task_type: Optional[TaskType] = None,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Task:
        """
        Genera una tarea de un dominio.

        Args:
            domain: Dominio (medicine, finance, etc.)
            task_type: Tipo de tarea (opcional, se elige automáticamente)
            n_samples: Número de muestras (opcional)
            seed: Semilla para reproducibilidad

        Returns:
            Task con datos anónimos
        """
        if seed is not None:
            np.random.seed(seed)

        if domain not in self._connectors:
            # Si no hay conector, generar tarea sintética
            return self._sample_synthetic_task(domain, task_type, n_samples)

        connector = self._connectors[domain]

        # Usar el conector para obtener datos
        # Los conectores ya implementan load_synthetic_for_testing
        if hasattr(connector, 'load_synthetic_for_testing'):
            data_result = connector.load_synthetic_for_testing(
                n_samples=n_samples or 500,
                seed=seed
            )
            data = data_result.get('data')

            if data is not None:
                return self._create_task_from_data(domain, data, task_type)

        # Fallback a tarea sintética
        return self._sample_synthetic_task(domain, task_type, n_samples)

    def _sample_synthetic_task(
        self,
        domain: str,
        task_type: Optional[TaskType] = None,
        n_samples: Optional[int] = None
    ) -> Task:
        """
        Genera tarea sintética cuando no hay conector.

        NOTA: Los datos sintéticos NO representan el dominio real.
        Solo sirven para que el agente practique.
        """
        n = n_samples or 500

        # Elegir tipo de tarea si no se especifica
        if task_type is None:
            task_type = np.random.choice([
                TaskType.CLASSIFICATION,
                TaskType.REGRESSION,
                TaskType.ANOMALY,
            ])

        # Número de features basado en hash del dominio
        # ORIGEN: Determinístico por dominio pero no hardcodeado
        domain_hash = int(hashlib.sha256(domain.encode()).hexdigest()[:8], 16)
        n_features = 5 + (domain_hash % 10)  # 5-14 features

        # Generar datos sintéticos
        X = np.random.randn(n, n_features)

        if task_type == TaskType.CLASSIFICATION:
            # Clasificación binaria
            # ORIGEN: Separación lineal + ruido
            weights = np.random.randn(n_features)
            logits = X @ weights
            probs = 1 / (1 + np.exp(-logits))
            y = (probs > 0.5).astype(int)
            # Añadir ruido
            flip_idx = np.random.choice(n, size=int(n * 0.1), replace=False)
            y[flip_idx] = 1 - y[flip_idx]

        elif task_type == TaskType.REGRESSION:
            # Regresión
            weights = np.random.randn(n_features)
            y = X @ weights + np.random.randn(n) * 0.5

        elif task_type == TaskType.ANOMALY:
            # Anomalías
            y = np.zeros(n)
            # 5% anomalías
            n_anomalies = max(1, int(n * 0.05))
            anomaly_idx = np.random.choice(n, size=n_anomalies, replace=False)
            X[anomaly_idx] += np.random.randn(n_anomalies, n_features) * 3
            y[anomaly_idx] = 1

        else:
            y = None

        # Crear oracle
        oracle = self._create_oracle(task_type, y)

        task = Task(
            task_id=self._next_task_id(),
            domain=domain,
            task_type=task_type,
            X=X,
            y=y,
            _oracle=oracle
        )

        self.logger.log_from_data(
            value={'domain': domain, 'type': task_type.value, 'n': n},
            source="synthetic_task_generation",
            dataset=f"task_{task.task_id}",
            context="DomainTaskSampler.sample_synthetic_task"
        )

        return task

    def _create_task_from_data(
        self,
        domain: str,
        data: Any,
        task_type: Optional[TaskType] = None
    ) -> Task:
        """
        Crea tarea desde datos del conector.

        Convierte DataFrame o similar a arrays anónimos.
        """
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            # Separar features numéricas
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) < 2:
                # No hay suficientes columnas numéricas
                return self._sample_synthetic_task(domain, task_type)

            # Usar última columna como target (convención)
            feature_cols = numeric_cols[:-1]
            target_col = numeric_cols[-1]

            X = data[feature_cols].values
            y = data[target_col].values

            # Determinar tipo de tarea basado en target
            unique_values = np.unique(y[~np.isnan(y)])

            if len(unique_values) <= 10:
                # Clasificación
                if task_type is None:
                    task_type = TaskType.CLASSIFICATION
                # Binarizar si hay más de 2 clases
                if len(unique_values) > 2:
                    median = np.median(y[~np.isnan(y)])
                    y = (y > median).astype(int)
            else:
                # Regresión
                if task_type is None:
                    task_type = TaskType.REGRESSION

            # Limpiar NaN
            valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[valid_idx]
            y = y[valid_idx]

            if len(X) < 10:
                return self._sample_synthetic_task(domain, task_type)

            oracle = self._create_oracle(task_type, y)

            return Task(
                task_id=self._next_task_id(),
                domain=domain,
                task_type=task_type,
                X=X,
                y=y,
                _oracle=oracle
            )

        else:
            # Tipo de datos no soportado
            return self._sample_synthetic_task(domain, task_type)

    def _create_oracle(
        self,
        task_type: TaskType,
        y_true: Optional[np.ndarray]
    ) -> Callable:
        """
        Crea función oracle para evaluar predicciones.

        El oracle calcula métricas objetivas.

        NOTA: El oracle almacena y_true completo.
        Cuando se llama, ajusta a la longitud de predictions usando
        la última porción (que corresponde al test set).
        """
        _y_full = y_true  # Almacenar referencia

        def oracle(predictions: np.ndarray, probabilities: Optional[np.ndarray] = None) -> Dict[str, float]:
            """Evalúa predicciones y retorna métricas."""
            metrics = {}

            if _y_full is None:
                return metrics

            # Ajustar y_true a la longitud de predictions
            # Las predicciones son sobre el test set (última porción de datos)
            n_pred = len(predictions)
            n_full = len(_y_full)

            if n_pred < n_full:
                # Usar la última porción de y_true que corresponde al test set
                y_true_subset = _y_full[-n_pred:]
            elif n_pred > n_full:
                # Truncar predicciones si son más largas
                predictions = predictions[:n_full]
                if probabilities is not None:
                    probabilities = probabilities[:n_full]
                y_true_subset = _y_full
            else:
                y_true_subset = _y_full

            if task_type == TaskType.CLASSIFICATION:
                # Accuracy
                correct = (predictions == y_true_subset).sum()
                metrics['accuracy'] = correct / len(y_true_subset)

                # FPR / FNR
                if len(np.unique(y_true_subset)) == 2:
                    positives = y_true_subset == 1
                    negatives = y_true_subset == 0

                    if positives.sum() > 0:
                        fn = ((predictions == 0) & positives).sum()
                        metrics['fnr'] = fn / positives.sum()

                    if negatives.sum() > 0:
                        fp = ((predictions == 1) & negatives).sum()
                        metrics['fpr'] = fp / negatives.sum()

                # AUROC si hay probabilidades
                if probabilities is not None and len(np.unique(y_true_subset)) == 2:
                    try:
                        from sklearn.metrics import roc_auc_score
                        metrics['auroc'] = roc_auc_score(y_true_subset, probabilities)
                    except:
                        pass

                # Brier score si hay probabilidades
                if probabilities is not None:
                    metrics['brier_score'] = np.mean((probabilities - y_true_subset) ** 2)

            elif task_type == TaskType.REGRESSION:
                # MSE
                metrics['mse'] = np.mean((predictions - y_true_subset) ** 2)
                # MAE
                metrics['mae'] = np.mean(np.abs(predictions - y_true_subset))
                # R²
                ss_res = np.sum((y_true_subset - predictions) ** 2)
                ss_tot = np.sum((y_true_subset - np.mean(y_true_subset)) ** 2)
                if ss_tot > 0:
                    metrics['r_squared'] = 1 - (ss_res / ss_tot)

            elif task_type == TaskType.ANOMALY:
                # Tratar como clasificación binaria
                correct = (predictions == y_true_subset).sum()
                metrics['accuracy'] = correct / len(y_true_subset)

                # Precision/Recall para anomalías
                true_positives = ((predictions == 1) & (y_true_subset == 1)).sum()
                false_positives = ((predictions == 1) & (y_true_subset == 0)).sum()
                false_negatives = ((predictions == 0) & (y_true_subset == 1)).sum()

                if true_positives + false_positives > 0:
                    metrics['precision'] = true_positives / (true_positives + false_positives)
                if true_positives + false_negatives > 0:
                    metrics['recall'] = true_positives / (true_positives + false_negatives)

            return metrics

        return oracle

    def evaluate_result(
        self,
        task: Task,
        result: TaskResult
    ) -> Dict[str, float]:
        """
        Evalúa el resultado de un agente en una tarea.

        Usa el oracle de la tarea para calcular métricas.
        El oracle maneja internamente el ajuste de dimensiones.
        """
        if task._oracle is None:
            return {}

        if result.predictions is None:
            return {}

        # El oracle maneja el ajuste de dimensiones internamente
        metrics = task._oracle(result.predictions, result.probabilities)
        result.metrics = metrics

        return metrics
