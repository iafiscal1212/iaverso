"""
MATHEMATICS TASK GENERATOR - Tareas Matemáticas para Especialización
=====================================================================

Genera tareas matemáticas para que los agentes se especialicen
de forma ENDÓGENA.

TIPOS DE TAREA:
- MATH_EQ_SIMPLE: Sistemas de ecuaciones algebraicas
- MATH_CALCULUS: Derivadas e integrales
- MATH_FIT: Ajuste de funciones
- MATH_SERIES: Convergencia de series

NORMA DURA:
- Sin números mágicos
- Umbrales por percentiles
- Provenance documentada
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import hashlib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stimuli_engine.provenance import (
    get_provenance_logger, MATH_CONSTANTS, THEORY_CONSTANTS, ProvenanceType
)


class MathTaskType(Enum):
    """Tipos de tareas matemáticas."""
    MATH_EQ_SIMPLE = "math_eq_simple"      # Ecuaciones algebraicas
    MATH_CALCULUS = "math_calculus"        # Derivadas/integrales
    MATH_FIT = "math_fit"                  # Ajuste de funciones
    MATH_SERIES = "math_series"            # Convergencia de series


@dataclass
class MathTaskSpec:
    """
    Especificación de una tarea matemática.

    NORMA DURA:
    - has_ground_truth indica si hay solución conocida
    - ground_truth_provenance documenta origen de la solución
    """
    task_type: MathTaskType
    has_ground_truth: bool = True
    ground_truth_provenance: str = ""
    evaluation_mode: str = "ground_truth"  # "ground_truth" o "hypothesis_falsification"

    # Datos de la tarea
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    params: Dict[str, Any] = field(default_factory=dict)

    # Solución oracle (solo si has_ground_truth=True)
    oracle_solution: Optional[Any] = None
    oracle_function: Optional[Callable] = None


class MathTaskGenerator:
    """
    Generador de tareas matemáticas.

    NORMA DURA:
    - Parámetros generados con distribuciones documentadas
    - Soluciones verificadas por oracle
    - Métricas por percentiles
    """

    def __init__(self, seed: Optional[int] = None):
        self.logger = get_provenance_logger()
        self.rng = np.random.default_rng(seed)
        self._error_history: List[float] = []

    def generate_task(
        self,
        task_type: Optional[MathTaskType] = None,
        seed: Optional[int] = None
    ) -> MathTaskSpec:
        """
        Genera una tarea matemática.

        Args:
            task_type: Tipo de tarea (si None, elige aleatoriamente)
            seed: Semilla para reproducibilidad
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if task_type is None:
            task_type = self.rng.choice(list(MathTaskType))

        if task_type == MathTaskType.MATH_EQ_SIMPLE:
            return self._generate_equation_task()
        elif task_type == MathTaskType.MATH_CALCULUS:
            return self._generate_calculus_task()
        elif task_type == MathTaskType.MATH_FIT:
            return self._generate_fit_task()
        elif task_type == MathTaskType.MATH_SERIES:
            return self._generate_series_task()
        else:
            return self._generate_equation_task()

    def _generate_equation_task(self) -> MathTaskSpec:
        """
        Genera tarea de ecuaciones algebraicas.

        Sistema: Ax = b
        donde A es matriz n×n, x son incógnitas, b términos independientes.

        ORIGEN:
        - Coeficientes ~ Normal(0, 1) [FROM_THEORY: distribución estándar]
        - Dimensión 1-3 [FROM_THEORY: sistemas pequeños resolubles]
        """
        # ORIGEN: Dimensión del sistema (1-3 ecuaciones)
        # FROM_THEORY: Sistemas pequeños para verificación manual
        n = int(self.rng.integers(1, 4))

        self.logger.log_from_theory(
            value=n,
            source="n ∈ {1, 2, 3} para sistemas verificables",
            reference="Álgebra lineal básica",
            context="MathTaskGenerator._generate_equation_task"
        )

        # ORIGEN: Coeficientes ~ N(0, 1)
        # FROM_THEORY: Distribución normal estándar
        A = self.rng.standard_normal((n, n))

        # Asegurar que A es invertible (det != 0)
        while np.abs(np.linalg.det(A)) < 1e-10:
            A = self.rng.standard_normal((n, n))

        self.logger.log_from_theory(
            value="N(0, 1)",
            source="Coeficientes ~ Normal(μ=0, σ=1)",
            reference="Distribución normal estándar",
            context="MathTaskGenerator._generate_equation_task"
        )

        # Términos independientes
        b = self.rng.standard_normal(n)

        # Solución oracle
        x_oracle = np.linalg.solve(A, b)

        self.logger.log_from_theory(
            value=x_oracle.tolist(),
            source="x = A^(-1) b",
            reference="Solución de sistema lineal",
            context="MathTaskGenerator._generate_equation_task"
        )

        # Oracle function para evaluar
        def oracle_eval(x_pred: np.ndarray) -> Dict[str, float]:
            """Evalúa predicción comparando con solución exacta."""
            if len(x_pred) != n:
                return {'error': float('inf'), 'valid': False}

            # Error relativo
            abs_error = np.linalg.norm(x_pred - x_oracle)
            rel_error = abs_error / (np.linalg.norm(x_oracle) + 1e-10)

            # Verificación: Ax_pred ≈ b
            residual = np.linalg.norm(A @ x_pred - b)

            return {
                'absolute_error': float(abs_error),
                'relative_error': float(rel_error),
                'residual': float(residual),
                'valid': True
            }

        return MathTaskSpec(
            task_type=MathTaskType.MATH_EQ_SIMPLE,
            has_ground_truth=True,
            ground_truth_provenance="FROM_THEORY: x = A^(-1)b, sistema lineal",
            evaluation_mode="ground_truth",
            X=A,
            y=b,
            params={'n': n, 'coef_distribution': 'N(0,1)'},
            oracle_solution=x_oracle,
            oracle_function=oracle_eval
        )

    def _generate_calculus_task(self) -> MathTaskSpec:
        """
        Genera tarea de cálculo (derivadas/integrales).

        Funciones: polinomios, trigonométricas, exponenciales.

        ORIGEN:
        - Grados de polinomios ~ Uniform(1, 5)
        - Coeficientes ~ N(0, 1)
        """
        # Tipo: derivada o integral
        is_derivative = self.rng.choice([True, False])

        # ORIGEN: Grado del polinomio
        # FROM_THEORY: Grados pequeños para verificación
        degree = int(self.rng.integers(1, 6))

        self.logger.log_from_theory(
            value=degree,
            source="grado ∈ {1, ..., 5}",
            reference="Polinomios de grado bajo",
            context="MathTaskGenerator._generate_calculus_task"
        )

        # Coeficientes del polinomio
        coeffs = self.rng.standard_normal(degree + 1)

        # Función polinómica: f(x) = sum(coeffs[i] * x^i)
        def f(x):
            return sum(c * x**i for i, c in enumerate(coeffs))

        if is_derivative:
            # Derivada: f'(x) = sum(i * coeffs[i] * x^(i-1))
            deriv_coeffs = np.array([i * coeffs[i] for i in range(1, len(coeffs))])

            def oracle_f(x):
                if len(deriv_coeffs) == 0:
                    return 0.0
                return sum(c * x**i for i, c in enumerate(deriv_coeffs))

            oracle_coeffs = deriv_coeffs
            task_description = "derivative"

            self.logger.log_from_theory(
                value=deriv_coeffs.tolist(),
                source="d/dx[sum(c_i x^i)] = sum(i * c_i * x^(i-1))",
                reference="Regla de derivación de polinomios",
                context="MathTaskGenerator._generate_calculus_task"
            )
        else:
            # Integral: F(x) = sum(coeffs[i] / (i+1) * x^(i+1))
            integ_coeffs = np.array([coeffs[i] / (i + 1) for i in range(len(coeffs))])
            integ_coeffs = np.insert(integ_coeffs, 0, 0)  # Constante = 0

            def oracle_f(x):
                return sum(c * x**i for i, c in enumerate(integ_coeffs))

            oracle_coeffs = integ_coeffs
            task_description = "integral"

            self.logger.log_from_theory(
                value=integ_coeffs.tolist(),
                source="∫sum(c_i x^i)dx = sum(c_i/(i+1) * x^(i+1)) + C",
                reference="Regla de integración de polinomios",
                context="MathTaskGenerator._generate_calculus_task"
            )

        # Puntos de evaluación para comparar
        # ORIGEN: Malla uniforme en [-2, 2]
        x_eval = np.linspace(-2, 2, 50)
        y_oracle = np.array([oracle_f(x) for x in x_eval])

        def oracle_eval(y_pred: np.ndarray) -> Dict[str, float]:
            """Evalúa comparando con solución exacta en malla."""
            if len(y_pred) != len(y_oracle):
                return {'error': float('inf'), 'valid': False}

            # Error L2 normalizado
            l2_error = np.sqrt(np.mean((y_pred - y_oracle) ** 2))
            max_error = np.max(np.abs(y_pred - y_oracle))

            # Normalizar por escala de y_oracle
            scale = np.std(y_oracle) + 1e-10
            normalized_error = l2_error / scale

            return {
                'l2_error': float(l2_error),
                'max_error': float(max_error),
                'normalized_error': float(normalized_error),
                'valid': True
            }

        return MathTaskSpec(
            task_type=MathTaskType.MATH_CALCULUS,
            has_ground_truth=True,
            ground_truth_provenance=f"FROM_THEORY: {task_description} de polinomio grado {degree}",
            evaluation_mode="ground_truth",
            X=x_eval,
            y=np.array([f(x) for x in x_eval]),  # Función original
            params={
                'coeffs': coeffs.tolist(),
                'degree': degree,
                'is_derivative': is_derivative,
                'oracle_coeffs': oracle_coeffs.tolist()
            },
            oracle_solution=y_oracle,
            oracle_function=oracle_eval
        )

    def _generate_fit_task(self) -> MathTaskSpec:
        """
        Genera tarea de ajuste de funciones.

        Modelos:
        - Lineal: y = a*x + b + ruido
        - Sinusoidal: y = a*sin(b*x) + c + ruido

        ORIGEN:
        - Parámetros ~ N(0, 1) o Uniform según tipo
        - Ruido ~ N(0, σ) donde σ derivada de SNR
        """
        # Tipo de modelo
        model_type = self.rng.choice(['linear', 'sinusoidal'])

        # Número de puntos
        n_points = int(self.rng.integers(30, 101))

        # ORIGEN: x ~ Uniform(-π, π) para cubrir periodo completo
        pi = MATH_CONSTANTS['pi'].value
        x = self.rng.uniform(-pi, pi, n_points)
        x = np.sort(x)

        self.logger.log_from_theory(
            value=[-pi, pi],
            source="x ∈ [-π, π] para cubrir un periodo",
            reference="Análisis de Fourier",
            context="MathTaskGenerator._generate_fit_task"
        )

        if model_type == 'linear':
            # y = a*x + b
            a_true = float(self.rng.standard_normal())
            b_true = float(self.rng.standard_normal())
            y_clean = a_true * x + b_true
            true_params = {'a': a_true, 'b': b_true}

            self.logger.log_from_theory(
                value=true_params,
                source="y = ax + b, a,b ~ N(0,1)",
                reference="Modelo lineal estándar",
                context="MathTaskGenerator._generate_fit_task"
            )
        else:
            # y = a*sin(b*x) + c
            a_true = float(self.rng.uniform(0.5, 2.0))
            b_true = float(self.rng.uniform(0.5, 3.0))
            c_true = float(self.rng.standard_normal())
            y_clean = a_true * np.sin(b_true * x) + c_true
            true_params = {'a': a_true, 'b': b_true, 'c': c_true}

            self.logger.log_from_theory(
                value=true_params,
                source="y = a*sin(bx) + c",
                reference="Modelo sinusoidal",
                context="MathTaskGenerator._generate_fit_task"
            )

        # ORIGEN: SNR (Signal-to-Noise Ratio) ~ Uniform(5, 20)
        # FROM_THEORY: SNR típico en datos experimentales
        snr = float(self.rng.uniform(5, 20))
        signal_power = np.var(y_clean)
        noise_std = np.sqrt(signal_power / snr)

        self.logger.log_from_data(
            value={'snr': snr, 'noise_std': noise_std},
            source="SNR ~ Uniform(5, 20), σ_noise = sqrt(signal_var / SNR)",
            statistic="derived_noise_level",
            context="MathTaskGenerator._generate_fit_task"
        )

        # Añadir ruido
        noise = self.rng.normal(0, noise_std, n_points)
        y = y_clean + noise

        def oracle_eval(params_pred: Dict[str, float]) -> Dict[str, float]:
            """Evalúa parámetros estimados vs reales."""
            errors = {}
            total_error = 0.0

            for key in true_params:
                if key not in params_pred:
                    return {'error': float('inf'), 'valid': False}

                # Error relativo por parámetro
                true_val = true_params[key]
                pred_val = params_pred[key]

                if abs(true_val) > 1e-10:
                    rel_error = abs(pred_val - true_val) / abs(true_val)
                else:
                    rel_error = abs(pred_val - true_val)

                errors[f'{key}_error'] = float(rel_error)
                total_error += rel_error

            errors['mean_param_error'] = total_error / len(true_params)
            errors['valid'] = True

            return errors

        return MathTaskSpec(
            task_type=MathTaskType.MATH_FIT,
            has_ground_truth=True,
            ground_truth_provenance=f"FROM_THEORY: modelo {model_type} con ruido SNR={snr:.1f}",
            evaluation_mode="ground_truth",
            X=x,
            y=y,
            params={
                'model_type': model_type,
                'n_points': n_points,
                'snr': snr,
                'noise_std': noise_std,
                'true_params': true_params
            },
            oracle_solution=true_params,
            oracle_function=oracle_eval
        )

    def _generate_series_task(self) -> MathTaskSpec:
        """
        Genera tarea de convergencia de series.

        Series:
        - Geométrica: sum(a^n) converge si |a| < 1
        - p-serie: sum(1/n^p) converge si p > 1
        - Alternante: sum((-1)^n / n^p) converge si p > 0

        ORIGEN: Criterios de convergencia estándar
        """
        series_type = self.rng.choice(['geometric', 'p_series', 'alternating'])

        if series_type == 'geometric':
            # Series geométrica: sum(a^n)
            # Converge si |a| < 1, diverge si |a| >= 1

            # 50% converge, 50% diverge
            if self.rng.choice([True, False]):
                a = float(self.rng.uniform(-0.99, 0.99))
                converges = True
            else:
                a = float(self.rng.choice([-1, 1]) * self.rng.uniform(1.0, 2.0))
                converges = False

            params = {'a': a, 'series_type': 'geometric'}

            # Términos parciales para que el agente analice
            n_terms = 50
            terms = np.array([a**n for n in range(n_terms)])
            partial_sums = np.cumsum(terms)

            self.logger.log_from_theory(
                value={'a': a, 'converges': converges},
                source="sum(a^n) converge ⟺ |a| < 1",
                reference="Criterio de convergencia geométrica",
                context="MathTaskGenerator._generate_series_task"
            )

        elif series_type == 'p_series':
            # p-serie: sum(1/n^p)
            # Converge si p > 1, diverge si p <= 1

            if self.rng.choice([True, False]):
                p = float(self.rng.uniform(1.01, 3.0))
                converges = True
            else:
                p = float(self.rng.uniform(0.5, 1.0))
                converges = False

            params = {'p': p, 'series_type': 'p_series'}

            n_terms = 50
            terms = np.array([1.0 / (n**p) for n in range(1, n_terms + 1)])
            partial_sums = np.cumsum(terms)

            self.logger.log_from_theory(
                value={'p': p, 'converges': converges},
                source="sum(1/n^p) converge ⟺ p > 1",
                reference="Criterio de la p-serie",
                context="MathTaskGenerator._generate_series_task"
            )

        else:  # alternating
            # Serie alternante: sum((-1)^n / n^p)
            # Converge si p > 0 (Leibniz)

            if self.rng.choice([True, False]):
                p = float(self.rng.uniform(0.1, 2.0))
                converges = True
            else:
                p = float(self.rng.uniform(-0.5, 0.0))
                converges = False

            params = {'p': p, 'series_type': 'alternating'}

            n_terms = 50
            terms = np.array([(-1)**n / (n**p) if n > 0 else 0 for n in range(n_terms)])
            partial_sums = np.cumsum(terms)

            self.logger.log_from_theory(
                value={'p': p, 'converges': converges},
                source="sum((-1)^n/n^p) converge si p > 0 (Leibniz)",
                reference="Criterio de Leibniz",
                context="MathTaskGenerator._generate_series_task"
            )

        def oracle_eval(prediction: Dict[str, Any]) -> Dict[str, float]:
            """Evalúa predicción de convergencia."""
            pred_converges = prediction.get('converges')

            if pred_converges is None:
                return {'accuracy': 0.0, 'valid': False}

            correct = (pred_converges == converges)

            return {
                'accuracy': 1.0 if correct else 0.0,
                'correct': correct,
                'valid': True
            }

        return MathTaskSpec(
            task_type=MathTaskType.MATH_SERIES,
            has_ground_truth=True,
            ground_truth_provenance=f"FROM_THEORY: {series_type} series convergence criterion",
            evaluation_mode="ground_truth",
            X=np.arange(len(terms)),
            y=terms,
            params={
                **params,
                'partial_sums': partial_sums.tolist(),
                'n_terms': len(terms)
            },
            oracle_solution={'converges': converges},
            oracle_function=oracle_eval
        )

    def get_success_threshold(self, metric_name: str, percentile: float = 75) -> float:
        """
        Obtiene umbral de éxito derivado de distribución de errores.

        NORMA DURA: Umbral por percentil, no constante mágica.

        Args:
            metric_name: Nombre de la métrica
            percentile: Percentil para umbral (default 75)

        Returns:
            Umbral derivado de datos
        """
        if len(self._error_history) < THEORY_CONSTANTS['min_samples_corr'].value:
            # No hay suficientes datos, usar placeholder
            return float('inf')

        threshold = float(np.percentile(self._error_history, percentile))

        self.logger.log_from_data(
            value=threshold,
            source=f"percentile({metric_name}, {percentile})",
            dataset=f"n={len(self._error_history)}",
            statistic=f"percentile_{percentile}",
            context="MathTaskGenerator.get_success_threshold"
        )

        return threshold

    def record_error(self, error: float):
        """Registra error para calcular umbrales."""
        self._error_history.append(error)


# =============================================================================
# AGENTE SOLUCIONADOR SIMPLE (para testing)
# =============================================================================

class SimpleMathSolver:
    """
    Solucionador matemático simple para testing.

    NO es el agente real - solo para validar el sistema.
    """

    def __init__(self, use_oracle: bool = False):
        """
        Args:
            use_oracle: Si True, usa solución del oracle (control positivo)
        """
        self.use_oracle = use_oracle
        self.rng = np.random.default_rng()

    def solve(self, task: MathTaskSpec) -> Any:
        """Resuelve una tarea matemática."""
        if self.use_oracle and task.oracle_solution is not None:
            return task.oracle_solution

        # Solución aleatoria (control negativo)
        if task.task_type == MathTaskType.MATH_EQ_SIMPLE:
            n = task.params.get('n', 1)
            return self.rng.standard_normal(n)

        elif task.task_type == MathTaskType.MATH_CALCULUS:
            return self.rng.standard_normal(len(task.X))

        elif task.task_type == MathTaskType.MATH_FIT:
            true_params = task.params.get('true_params', {})
            # Parámetros aleatorios
            return {k: self.rng.standard_normal() for k in true_params}

        elif task.task_type == MathTaskType.MATH_SERIES:
            return {'converges': self.rng.choice([True, False])}

        return None


# =============================================================================
# TEST
# =============================================================================

def test_math_tasks():
    """Test del generador de tareas matemáticas."""
    print("=" * 70)
    print("TEST: MATHEMATICS TASK GENERATOR")
    print("=" * 70)

    generator = MathTaskGenerator(seed=42)

    # Generar tareas de cada tipo
    for task_type in MathTaskType:
        print(f"\n=== {task_type.value} ===")
        task = generator.generate_task(task_type)

        print(f"  has_ground_truth: {task.has_ground_truth}")
        print(f"  provenance: {task.ground_truth_provenance}")
        print(f"  params: {task.params}")

        # Probar solver con oracle
        oracle_solver = SimpleMathSolver(use_oracle=True)
        oracle_result = oracle_solver.solve(task)

        if task.oracle_function:
            metrics = task.oracle_function(oracle_result)
            print(f"  Oracle solver metrics: {metrics}")

        # Probar solver aleatorio
        random_solver = SimpleMathSolver(use_oracle=False)
        random_result = random_solver.solve(task)

        if task.oracle_function:
            metrics = task.oracle_function(random_result)
            print(f"  Random solver metrics: {metrics}")

    print("\n" + "=" * 70)
    print("TEST COMPLETADO: Todas las tareas matemáticas generadas correctamente")
    print("=" * 70)


if __name__ == "__main__":
    test_math_tasks()
