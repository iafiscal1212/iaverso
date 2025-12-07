"""
UNIFIED TASK ENGINE - Motor Unificado de Tareas por Dominio
============================================================

Integra todos los generadores de tareas de todos los dominios:
- Medicine, Finance, Cosmology, Engineering (originales)
- Mathematics, Physics (nuevos)

NORMA DURA:
- Sin roles asignados
- Especialización emerge de métricas
- Umbrales por percentiles
- Provenance documentada
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stimuli_engine.provenance import get_provenance_logger, THEORY_CONSTANTS

from .task_sampler import Task, TaskResult, TaskType, EvaluationMode
from .math_tasks import MathTaskGenerator, MathTaskSpec, MathTaskType
from .physics_tasks import PhysicsTaskGenerator, PhysicsTaskSpec, PhysicsTaskType


class Domain(Enum):
    """Dominios disponibles para especialización."""
    # Dominios originales
    MEDICINE = "medicine"
    FINANCE = "finance"
    COSMOLOGY = "cosmology"
    ENGINEERING = "engineering"

    # Dominios nuevos
    MATHEMATICS = "mathematics"
    PHYSICS = "physics"


# Mapeo de tipos de tarea específicos a TaskType genérico
MATH_TO_GENERIC = {
    MathTaskType.MATH_EQ_SIMPLE: TaskType.REGRESSION,
    MathTaskType.MATH_CALCULUS: TaskType.REGRESSION,
    MathTaskType.MATH_FIT: TaskType.REGRESSION,
    MathTaskType.MATH_SERIES: TaskType.CLASSIFICATION,
}

PHYSICS_TO_GENERIC = {
    PhysicsTaskType.PHYS_FREE_FALL: TaskType.REGRESSION,
    PhysicsTaskType.PHYS_OSCILLATOR: TaskType.REGRESSION,
    PhysicsTaskType.PHYS_COUPLED: TaskType.CAUSALITY,
    PhysicsTaskType.PHYS_TIMESERIES: TaskType.TIMESERIES,
}


class UnifiedTaskEngine:
    """
    Motor unificado de tareas para todos los dominios.

    Integra:
    - Generadores específicos de dominio
    - Conversión a formato Task unificado
    - Oracles para evaluación

    NORMA DURA:
    - No asigna roles
    - Todos los dominios tratados igual
    - Métricas derivadas de datos
    """

    def __init__(self, seed: Optional[int] = None):
        self.logger = get_provenance_logger()
        self.rng = np.random.default_rng(seed)

        # Generadores por dominio
        self._math_generator = MathTaskGenerator(seed=seed)
        self._physics_generator = PhysicsTaskGenerator(seed=seed)

        # Contador de tareas
        self._task_counter = 0

        # Historial de errores por dominio (para umbrales)
        self._error_history: Dict[str, List[float]] = {
            d.value: [] for d in Domain
        }

    def _next_task_id(self) -> str:
        """Genera siguiente ID de tarea."""
        self._task_counter += 1
        return f"task_{self._task_counter:06d}"

    def get_available_domains(self) -> List[str]:
        """Retorna dominios disponibles."""
        return [d.value for d in Domain]

    def sample_task(
        self,
        domain: str,
        task_subtype: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Task:
        """
        Genera una tarea de cualquier dominio.

        Args:
            domain: Nombre del dominio
            task_subtype: Subtipo específico (e.g., "math_eq_simple")
            seed: Semilla para reproducibilidad

        Returns:
            Task unificada
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        domain_lower = domain.lower()

        if domain_lower == "mathematics":
            return self._sample_math_task(task_subtype, seed)
        elif domain_lower == "physics":
            return self._sample_physics_task(task_subtype, seed)
        else:
            # Dominios originales: usar generador sintético
            return self._sample_synthetic_task(domain_lower, seed)

    def _sample_math_task(
        self,
        task_subtype: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Task:
        """Genera tarea de matemáticas."""
        # Determinar tipo de tarea
        if task_subtype:
            try:
                math_type = MathTaskType(task_subtype)
            except ValueError:
                math_type = None
        else:
            math_type = None

        # Generar tarea específica
        spec = self._math_generator.generate_task(math_type, seed)

        # Convertir a Task unificada
        return self._convert_math_spec(spec)

    def _convert_math_spec(self, spec: MathTaskSpec) -> Task:
        """Convierte MathTaskSpec a Task unificada."""
        # Mapear tipo
        generic_type = MATH_TO_GENERIC.get(spec.task_type, TaskType.REGRESSION)

        # Determinar modo de evaluación
        eval_mode = EvaluationMode.GROUND_TRUTH if spec.has_ground_truth else \
                    EvaluationMode.HYPOTHESIS_FALSIFICATION

        # Crear oracle wrapper
        def oracle_wrapper(predictions, probabilities=None):
            if spec.oracle_function is None:
                return {}

            # Las predicciones para math vienen como solution
            result = spec.oracle_function(predictions)

            # Extraer métrica principal
            metrics = {}
            if 'relative_error' in result:
                metrics['error'] = result['relative_error']
                metrics['accuracy'] = 1.0 - min(1.0, result['relative_error'])
            elif 'normalized_error' in result:
                metrics['error'] = result['normalized_error']
                metrics['accuracy'] = 1.0 - min(1.0, result['normalized_error'])
            elif 'mean_param_error' in result:
                metrics['error'] = result['mean_param_error']
                metrics['accuracy'] = 1.0 - min(1.0, result['mean_param_error'])
            elif 'accuracy' in result:
                metrics['accuracy'] = result['accuracy']
                metrics['error'] = 1.0 - result['accuracy']

            metrics.update(result)
            return metrics

        return Task(
            task_id=self._next_task_id(),
            domain="mathematics",
            task_type=generic_type,
            X=spec.X if spec.X is not None else np.array([]),
            y=spec.y,
            has_ground_truth=spec.has_ground_truth,
            ground_truth_provenance=spec.ground_truth_provenance,
            evaluation_mode=eval_mode,
            params={
                'math_task_type': spec.task_type.value,
                **spec.params
            },
            oracle_solution=spec.oracle_solution,
            _oracle=oracle_wrapper
        )

    def _sample_physics_task(
        self,
        task_subtype: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Task:
        """Genera tarea de física."""
        # Determinar tipo de tarea
        if task_subtype:
            try:
                physics_type = PhysicsTaskType(task_subtype)
            except ValueError:
                physics_type = None
        else:
            physics_type = None

        # Generar tarea específica
        spec = self._physics_generator.generate_task(physics_type, seed)

        # Convertir a Task unificada
        return self._convert_physics_spec(spec)

    def _convert_physics_spec(self, spec: PhysicsTaskSpec) -> Task:
        """Convierte PhysicsTaskSpec a Task unificada."""
        # Mapear tipo
        generic_type = PHYSICS_TO_GENERIC.get(spec.task_type, TaskType.REGRESSION)

        # Determinar modo de evaluación
        eval_mode = EvaluationMode.GROUND_TRUTH if spec.has_ground_truth else \
                    EvaluationMode.HYPOTHESIS_FALSIFICATION

        # Combinar t y X si es necesario
        if spec.t is not None and spec.X is not None:
            if spec.X.ndim == 1:
                X = np.column_stack([spec.t, spec.X])
            else:
                X = np.column_stack([spec.t, spec.X])
        elif spec.X is not None:
            X = spec.X
        else:
            X = spec.t if spec.t is not None else np.array([])

        # Crear oracle wrapper
        def oracle_wrapper(predictions, probabilities=None):
            if spec.oracle_function is None:
                return {}

            result = spec.oracle_function(predictions)

            # Extraer métricas principales
            metrics = {}

            # Para tareas con ground truth
            if spec.has_ground_truth:
                if 'mean_param_error' in result:
                    metrics['error'] = result['mean_param_error']
                    metrics['accuracy'] = 1.0 - min(1.0, result['mean_param_error'])
                elif 'omega_error' in result:
                    metrics['error'] = result['omega_error']
                    metrics['accuracy'] = 1.0 - min(1.0, result['omega_error'])
                elif 'coupling_detection' in result:
                    metrics['accuracy'] = result.get('coupling_detection', 0.0)
                    metrics['error'] = 1.0 - metrics['accuracy']
            else:
                # Para tareas sin ground truth
                if 'falsification_rate' in result:
                    metrics['falsification_rate'] = result['falsification_rate']
                if 'surrogate_stability' in result:
                    metrics['stability'] = result['surrogate_stability']

            metrics.update(result)
            return metrics

        return Task(
            task_id=self._next_task_id(),
            domain="physics",
            task_type=generic_type,
            X=X,
            y=spec.y,
            has_ground_truth=spec.has_ground_truth,
            ground_truth_provenance=spec.ground_truth_provenance,
            evaluation_mode=eval_mode,
            params={
                'physics_task_type': spec.task_type.value,
                't': spec.t,
                **spec.params
            },
            oracle_solution=spec.oracle_solution,
            _oracle=oracle_wrapper
        )

    def _sample_synthetic_task(
        self,
        domain: str,
        seed: Optional[int] = None
    ) -> Task:
        """
        Genera tarea sintética para dominios originales.

        Similar al DomainTaskSampler original pero usando
        la estructura Task extendida.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        n = 500

        # Elegir tipo de tarea
        task_type = self.rng.choice([
            TaskType.CLASSIFICATION,
            TaskType.REGRESSION,
            TaskType.ANOMALY,
        ])

        # Número de features basado en hash del dominio
        import hashlib
        domain_hash = int(hashlib.sha256(domain.encode()).hexdigest()[:8], 16)
        n_features = 5 + (domain_hash % 10)

        # Generar datos sintéticos
        X = self.rng.standard_normal((n, n_features))

        if task_type == TaskType.CLASSIFICATION:
            weights = self.rng.standard_normal(n_features)
            logits = X @ weights
            probs = 1 / (1 + np.exp(-logits))
            y = (probs > 0.5).astype(int)
            # Añadir ruido
            flip_idx = self.rng.choice(n, size=int(n * 0.1), replace=False)
            y[flip_idx] = 1 - y[flip_idx]

        elif task_type == TaskType.REGRESSION:
            weights = self.rng.standard_normal(n_features)
            y = X @ weights + self.rng.standard_normal(n) * 0.5

        elif task_type == TaskType.ANOMALY:
            y = np.zeros(n)
            n_anomalies = max(1, int(n * 0.05))
            anomaly_idx = self.rng.choice(n, size=n_anomalies, replace=False)
            X[anomaly_idx] += self.rng.standard_normal((n_anomalies, n_features)) * 3
            y[anomaly_idx] = 1

        else:
            y = None

        # Oracle para evaluación
        _y_full = y

        def oracle(predictions, probabilities=None):
            metrics = {}
            if _y_full is None:
                return metrics

            n_pred = len(predictions)
            n_full = len(_y_full)
            y_subset = _y_full[-n_pred:] if n_pred < n_full else _y_full

            if task_type == TaskType.CLASSIFICATION:
                correct = (predictions == y_subset).sum()
                metrics['accuracy'] = correct / len(y_subset)
                metrics['error'] = 1.0 - metrics['accuracy']

            elif task_type == TaskType.REGRESSION:
                metrics['mse'] = float(np.mean((predictions - y_subset) ** 2))
                metrics['mae'] = float(np.mean(np.abs(predictions - y_subset)))
                metrics['error'] = metrics['mse']
                ss_res = np.sum((y_subset - predictions) ** 2)
                ss_tot = np.sum((y_subset - np.mean(y_subset)) ** 2)
                if ss_tot > 0:
                    metrics['r_squared'] = 1 - (ss_res / ss_tot)
                    metrics['accuracy'] = max(0, metrics['r_squared'])

            elif task_type == TaskType.ANOMALY:
                correct = (predictions == y_subset).sum()
                metrics['accuracy'] = correct / len(y_subset)
                metrics['error'] = 1.0 - metrics['accuracy']

            return metrics

        return Task(
            task_id=self._next_task_id(),
            domain=domain,
            task_type=task_type,
            X=X,
            y=y,
            has_ground_truth=True,
            ground_truth_provenance="FROM_DATA: synthetic task",
            evaluation_mode=EvaluationMode.GROUND_TRUTH,
            params={'n_features': n_features},
            _oracle=oracle
        )

    def evaluate_result(
        self,
        task: Task,
        result: TaskResult
    ) -> Dict[str, float]:
        """
        Evalúa el resultado de un agente en una tarea.

        NORMA DURA:
        - Si has_ground_truth: compara con oracle
        - Si no: evalúa falsificación y consistencia
        """
        if task._oracle is None:
            return {}

        # Determinar qué evaluar según el modo
        if task.evaluation_mode == EvaluationMode.GROUND_TRUTH:
            # Usar predicciones o solución
            if result.solution is not None:
                predictions = result.solution
            elif result.predictions is not None:
                predictions = result.predictions
            else:
                return {}

            metrics = task._oracle(predictions, result.probabilities)

        else:  # HYPOTHESIS_FALSIFICATION
            # Evaluar hipótesis
            analysis = {
                'hypotheses': result.hypotheses_generated,
                'falsified': result.hypotheses_falsified,
                'surrogate_stability': result.surrogate_stability
            }
            metrics = task._oracle(analysis)

            # Añadir métricas derivadas
            metrics['falsification_rate'] = result.falsification_rate

        result.metrics = metrics

        # Registrar error para umbrales
        if 'error' in metrics:
            self._error_history[task.domain].append(metrics['error'])

        return metrics

    def get_success_threshold(
        self,
        domain: str,
        percentile: float = 75
    ) -> float:
        """
        Obtiene umbral de éxito derivado de datos.

        NORMA DURA: Umbral por percentil, no constante mágica.
        """
        errors = self._error_history.get(domain, [])

        min_samples = THEORY_CONSTANTS['min_samples_corr'].value

        if len(errors) < min_samples:
            return float('inf')

        threshold = float(np.percentile(errors, percentile))

        self.logger.log_from_data(
            value=threshold,
            source=f"percentile(errors_{domain}, {percentile})",
            dataset=f"n={len(errors)}",
            statistic=f"percentile_{percentile}",
            context="UnifiedTaskEngine.get_success_threshold"
        )

        return threshold


# =============================================================================
# TEST
# =============================================================================

def test_unified_engine():
    """Test del motor unificado."""
    print("=" * 70)
    print("TEST: UNIFIED TASK ENGINE")
    print("=" * 70)

    engine = UnifiedTaskEngine(seed=42)

    print(f"\nDominios disponibles: {engine.get_available_domains()}")

    # Probar cada dominio
    for domain in Domain:
        print(f"\n=== {domain.value.upper()} ===")

        for i in range(3):
            task = engine.sample_task(domain.value)
            print(f"  Task {i+1}:")
            print(f"    ID: {task.task_id}")
            print(f"    Type: {task.task_type.value}")
            print(f"    has_ground_truth: {task.has_ground_truth}")
            print(f"    evaluation_mode: {task.evaluation_mode.value}")

            if task.params:
                key_params = {k: v for k, v in task.params.items()
                             if not isinstance(v, (np.ndarray, list)) or
                             (isinstance(v, list) and len(v) < 5)}
                print(f"    params: {key_params}")

            # Probar evaluación con oracle
            if task.oracle_solution is not None:
                result = TaskResult(
                    task_id=task.task_id,
                    agent_id="test_agent",
                    solution=task.oracle_solution
                )
                metrics = engine.evaluate_result(task, result)
                print(f"    Oracle metrics: {metrics}")

    print("\n" + "=" * 70)
    print("TEST COMPLETADO: Motor unificado funcionando")
    print("=" * 70)


if __name__ == "__main__":
    test_unified_engine()
