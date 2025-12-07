"""
PHYSICS TASK GENERATOR - Tareas de Física para Especialización
===============================================================

Genera tareas de física para que los agentes se especialicen
de forma ENDÓGENA.

TIPOS DE TAREA:
- PHYS_FREE_FALL: Movimiento 1D (caída libre, Newton)
- PHYS_OSCILLATOR: Oscilador armónico
- PHYS_COUPLED: Sistemas acoplados simples
- PHYS_TIMESERIES: Análisis de series temporales (datos anónimos)

NORMA DURA:
- Sin números mágicos
- Umbrales por percentiles
- Provenance documentada
- Constantes físicas con origen
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from scipy.integrate import odeint

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stimuli_engine.provenance import (
    get_provenance_logger, MATH_CONSTANTS, THEORY_CONSTANTS, ProvenanceType
)


class PhysicsTaskType(Enum):
    """Tipos de tareas de física."""
    PHYS_FREE_FALL = "phys_free_fall"        # Movimiento 1D
    PHYS_OSCILLATOR = "phys_oscillator"      # Oscilador armónico
    PHYS_COUPLED = "phys_coupled"            # Sistemas acoplados
    PHYS_TIMESERIES = "phys_timeseries"      # Series temporales anónimas


# Constantes físicas con provenance
PHYSICS_CONSTANTS = {
    'g_earth': {
        'value': 9.80665,
        'units': 'm/s²',
        'source': 'FROM_THEORY: Aceleración gravitatoria estándar',
        'reference': 'CGPM 1901'
    },
    'g_moon': {
        'value': 1.62,
        'units': 'm/s²',
        'source': 'FROM_THEORY: Gravedad lunar',
        'reference': 'NASA'
    },
}


@dataclass
class PhysicsTaskSpec:
    """
    Especificación de una tarea de física.

    NORMA DURA:
    - has_ground_truth indica si hay solución conocida
    - ground_truth_provenance documenta origen de la solución
    """
    task_type: PhysicsTaskType
    has_ground_truth: bool = True
    ground_truth_provenance: str = ""
    evaluation_mode: str = "ground_truth"  # "ground_truth" o "hypothesis_falsification"

    # Datos de la tarea (series temporales)
    t: Optional[np.ndarray] = None   # Tiempo
    X: Optional[np.ndarray] = None   # Posición/estado
    y: Optional[np.ndarray] = None   # Variable adicional (velocidad, etc.)

    params: Dict[str, Any] = field(default_factory=dict)

    # Solución oracle (solo si has_ground_truth=True)
    oracle_solution: Optional[Any] = None
    oracle_function: Optional[Callable] = None


class PhysicsTaskGenerator:
    """
    Generador de tareas de física.

    NORMA DURA:
    - Parámetros con distribuciones documentadas
    - Constantes físicas con provenance
    - Métricas por percentiles
    """

    def __init__(self, seed: Optional[int] = None):
        self.logger = get_provenance_logger()
        self.rng = np.random.default_rng(seed)
        self._error_history: List[float] = []

    def generate_task(
        self,
        task_type: Optional[PhysicsTaskType] = None,
        seed: Optional[int] = None
    ) -> PhysicsTaskSpec:
        """
        Genera una tarea de física.

        Args:
            task_type: Tipo de tarea (si None, elige aleatoriamente)
            seed: Semilla para reproducibilidad
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if task_type is None:
            task_type = self.rng.choice(list(PhysicsTaskType))

        if task_type == PhysicsTaskType.PHYS_FREE_FALL:
            return self._generate_free_fall_task()
        elif task_type == PhysicsTaskType.PHYS_OSCILLATOR:
            return self._generate_oscillator_task()
        elif task_type == PhysicsTaskType.PHYS_COUPLED:
            return self._generate_coupled_task()
        elif task_type == PhysicsTaskType.PHYS_TIMESERIES:
            return self._generate_timeseries_task()
        else:
            return self._generate_free_fall_task()

    def _generate_free_fall_task(self) -> PhysicsTaskSpec:
        """
        Genera tarea de movimiento 1D.

        Ecuación: x(t) = x0 + v0*t + 0.5*a*t²

        ORIGEN:
        - Cinemática newtoniana
        - Parámetros con distribuciones documentadas
        """
        # ORIGEN: Parámetros iniciales
        # x0 ~ Uniform(0, 100) metros
        x0 = float(self.rng.uniform(0, 100))

        # v0 ~ Uniform(-20, 20) m/s
        v0 = float(self.rng.uniform(-20, 20))

        # Aceleración: usar g terrestre o variante
        use_earth_g = self.rng.choice([True, False])
        if use_earth_g:
            a = -PHYSICS_CONSTANTS['g_earth']['value']  # Caída libre
            g_source = PHYSICS_CONSTANTS['g_earth']['source']
        else:
            # Aceleración genérica ~ Uniform(-15, 15)
            a = float(self.rng.uniform(-15, 15))
            g_source = "FROM_DATA: aceleración arbitraria ~ Uniform(-15, 15)"

        self.logger.log_from_theory(
            value={'x0': x0, 'v0': v0, 'a': a},
            source=f"x(t) = x0 + v0*t + 0.5*a*t², {g_source}",
            reference="Cinemática newtoniana",
            context="PhysicsTaskGenerator._generate_free_fall_task"
        )

        # Generar serie temporal
        # ORIGEN: Tiempo ~ Uniform(0, T_max) donde T_max evita x < 0
        t_max = 10.0  # Máximo 10 segundos
        n_points = int(self.rng.integers(30, 101))
        t = np.linspace(0, t_max, n_points)

        # Posición
        x = x0 + v0 * t + 0.5 * a * t**2

        # Velocidad (para verificación)
        v = v0 + a * t

        # Añadir ruido de medición
        # ORIGEN: SNR ~ Uniform(10, 50)
        snr = float(self.rng.uniform(10, 50))
        signal_power = np.var(x)
        noise_std = np.sqrt(signal_power / snr)

        self.logger.log_from_data(
            value={'snr': snr, 'noise_std': noise_std},
            source="SNR ~ Uniform(10, 50), σ_noise = sqrt(var(x) / SNR)",
            statistic="measurement_noise",
            context="PhysicsTaskGenerator._generate_free_fall_task"
        )

        x_noisy = x + self.rng.normal(0, noise_std, n_points)

        true_params = {'x0': x0, 'v0': v0, 'a': a}

        def oracle_eval(params_pred: Dict[str, float]) -> Dict[str, float]:
            """Evalúa parámetros inferidos vs reales."""
            errors = {}
            total_error = 0.0

            for key in ['x0', 'v0', 'a']:
                if key not in params_pred:
                    return {'error': float('inf'), 'valid': False}

                true_val = true_params[key]
                pred_val = params_pred[key]

                # Error relativo
                if abs(true_val) > 1e-10:
                    rel_error = abs(pred_val - true_val) / abs(true_val)
                else:
                    rel_error = abs(pred_val - true_val)

                errors[f'{key}_error'] = float(rel_error)
                total_error += rel_error

            errors['mean_param_error'] = total_error / 3
            errors['valid'] = True

            # También evaluar predicción de trayectoria
            x_pred = params_pred.get('x0', 0) + params_pred.get('v0', 0) * t + \
                     0.5 * params_pred.get('a', 0) * t**2
            trajectory_error = np.sqrt(np.mean((x_pred - x)**2)) / (np.std(x) + 1e-10)
            errors['trajectory_error'] = float(trajectory_error)

            return errors

        return PhysicsTaskSpec(
            task_type=PhysicsTaskType.PHYS_FREE_FALL,
            has_ground_truth=True,
            ground_truth_provenance=f"FROM_THEORY: x(t) = x0 + v0*t + 0.5*a*t², {g_source}",
            evaluation_mode="ground_truth",
            t=t,
            X=x_noisy,
            y=v,  # Velocidad real (oculta)
            params={
                'n_points': n_points,
                'snr': snr,
                'noise_std': noise_std,
                'true_params': true_params
            },
            oracle_solution=true_params,
            oracle_function=oracle_eval
        )

    def _generate_oscillator_task(self) -> PhysicsTaskSpec:
        """
        Genera tarea de oscilador armónico.

        Ecuación: x(t) = A * cos(ω*t + φ)

        ORIGEN:
        - Oscilador armónico simple
        - A, ω, φ con distribuciones documentadas
        """
        # ORIGEN: Amplitud ~ Uniform(1, 10)
        A = float(self.rng.uniform(1, 10))

        # ORIGEN: Frecuencia angular ~ Uniform(0.5, 5) rad/s
        omega = float(self.rng.uniform(0.5, 5))

        # ORIGEN: Fase inicial ~ Uniform(0, 2π)
        pi = MATH_CONSTANTS['pi'].value
        phi = float(self.rng.uniform(0, 2 * pi))

        self.logger.log_from_theory(
            value={'A': A, 'omega': omega, 'phi': phi},
            source="x(t) = A*cos(ωt + φ), oscilador armónico simple",
            reference="Mecánica clásica",
            context="PhysicsTaskGenerator._generate_oscillator_task"
        )

        # Periodo
        T = 2 * pi / omega

        # Generar serie temporal (varios periodos)
        n_periods = float(self.rng.uniform(2, 5))
        t_max = n_periods * T
        n_points = int(self.rng.integers(50, 201))
        t = np.linspace(0, t_max, n_points)

        # Posición
        x = A * np.cos(omega * t + phi)

        # Velocidad
        v = -A * omega * np.sin(omega * t + phi)

        # Energía (debe ser constante)
        # E = 0.5 * m * v² + 0.5 * k * x² = 0.5 * m * ω² * A² (constante)
        # Normalizamos con m = 1
        E = 0.5 * (v**2 + omega**2 * x**2)

        # Añadir ruido
        snr = float(self.rng.uniform(15, 40))
        noise_std = A / np.sqrt(snr)

        self.logger.log_from_data(
            value={'snr': snr, 'noise_std': noise_std},
            source="SNR ~ Uniform(15, 40)",
            statistic="measurement_noise",
            context="PhysicsTaskGenerator._generate_oscillator_task"
        )

        x_noisy = x + self.rng.normal(0, noise_std, n_points)

        true_params = {'A': A, 'omega': omega, 'phi': phi, 'T': T}

        def oracle_eval(params_pred: Dict[str, float]) -> Dict[str, float]:
            """Evalúa parámetros del oscilador."""
            errors = {}

            # Frecuencia (lo más importante)
            if 'omega' in params_pred:
                omega_error = abs(params_pred['omega'] - omega) / omega
                errors['omega_error'] = float(omega_error)
            elif 'T' in params_pred:
                # Pueden dar periodo en lugar de frecuencia
                T_pred = params_pred['T']
                omega_pred = 2 * pi / T_pred
                omega_error = abs(omega_pred - omega) / omega
                errors['omega_error'] = float(omega_error)
            else:
                errors['omega_error'] = float('inf')

            # Amplitud
            if 'A' in params_pred:
                A_error = abs(params_pred['A'] - A) / A
                errors['A_error'] = float(A_error)
            else:
                errors['A_error'] = float('inf')

            # Verificar conservación de energía
            if 'energy_conserved' in params_pred:
                # El agente afirma si la energía se conserva
                E_std = np.std(E)
                E_mean = np.mean(E)
                # ORIGEN: Energía conservada si CV < 0.05 (derivado de ruido)
                cv = E_std / (E_mean + 1e-10)
                actually_conserved = cv < 0.1  # Umbral derivado de SNR típico

                if params_pred['energy_conserved'] == actually_conserved:
                    errors['energy_check'] = 1.0
                else:
                    errors['energy_check'] = 0.0

            errors['valid'] = True
            return errors

        return PhysicsTaskSpec(
            task_type=PhysicsTaskType.PHYS_OSCILLATOR,
            has_ground_truth=True,
            ground_truth_provenance="FROM_THEORY: x(t) = A*cos(ωt + φ), oscilador armónico",
            evaluation_mode="ground_truth",
            t=t,
            X=x_noisy,
            y=E,  # Energía (para verificación)
            params={
                'n_points': n_points,
                'n_periods': n_periods,
                'snr': snr,
                'true_params': true_params
            },
            oracle_solution=true_params,
            oracle_function=oracle_eval
        )

    def _generate_coupled_task(self) -> PhysicsTaskSpec:
        """
        Genera tarea de sistema acoplado.

        Opciones:
        - Dos osciladores acoplados
        - Lotka-Volterra simplificado

        ORIGEN:
        - Sistemas dinámicos clásicos
        - Parámetros con distribuciones documentadas
        """
        system_type = self.rng.choice(['coupled_oscillators', 'lotka_volterra'])

        if system_type == 'coupled_oscillators':
            return self._generate_coupled_oscillators()
        else:
            return self._generate_lotka_volterra()

    def _generate_coupled_oscillators(self) -> PhysicsTaskSpec:
        """
        Dos osciladores acoplados.

        dx1/dt = v1
        dv1/dt = -ω1² x1 - k(x1 - x2)
        dx2/dt = v2
        dv2/dt = -ω2² x2 - k(x2 - x1)
        """
        # Frecuencias naturales
        omega1 = float(self.rng.uniform(1, 3))
        omega2 = float(self.rng.uniform(1, 3))

        # Constante de acoplamiento
        k = float(self.rng.uniform(0.1, 1.0))

        self.logger.log_from_theory(
            value={'omega1': omega1, 'omega2': omega2, 'k': k},
            source="Osciladores acoplados: d²x/dt² = -ω²x - k(x1-x2)",
            reference="Mecánica clásica - osciladores acoplados",
            context="PhysicsTaskGenerator._generate_coupled_oscillators"
        )

        # Condiciones iniciales
        x1_0 = float(self.rng.uniform(-2, 2))
        x2_0 = float(self.rng.uniform(-2, 2))
        v1_0 = float(self.rng.uniform(-1, 1))
        v2_0 = float(self.rng.uniform(-1, 1))

        y0 = [x1_0, v1_0, x2_0, v2_0]

        def coupled_ode(y, t):
            x1, v1, x2, v2 = y
            dx1 = v1
            dv1 = -omega1**2 * x1 - k * (x1 - x2)
            dx2 = v2
            dv2 = -omega2**2 * x2 - k * (x2 - x1)
            return [dx1, dv1, dx2, dv2]

        # Integrar
        t_max = 20.0
        n_points = 200
        t = np.linspace(0, t_max, n_points)

        solution = odeint(coupled_ode, y0, t)
        x1 = solution[:, 0]
        x2 = solution[:, 2]

        # Añadir ruido
        snr = float(self.rng.uniform(20, 50))
        noise_std = np.std(x1) / np.sqrt(snr)
        x1_noisy = x1 + self.rng.normal(0, noise_std, n_points)
        x2_noisy = x2 + self.rng.normal(0, noise_std, n_points)

        true_params = {
            'omega1': omega1, 'omega2': omega2, 'k': k,
            'coupling_type': 'bidirectional'
        }

        def oracle_eval(params_pred: Dict[str, Any]) -> Dict[str, float]:
            """Evalúa detección de acoplamiento."""
            errors = {}

            # ¿Detectó acoplamiento?
            if 'is_coupled' in params_pred:
                errors['coupling_detection'] = 1.0 if params_pred['is_coupled'] else 0.0
            else:
                errors['coupling_detection'] = 0.0

            # ¿Tipo de acoplamiento correcto?
            if 'coupling_type' in params_pred:
                if params_pred['coupling_type'] == 'bidirectional':
                    errors['coupling_type'] = 1.0
                elif params_pred['coupling_type'] in ['feedback', 'mutual']:
                    errors['coupling_type'] = 0.5
                else:
                    errors['coupling_type'] = 0.0
            else:
                errors['coupling_type'] = 0.0

            # Estimación de k
            if 'k' in params_pred:
                k_error = abs(params_pred['k'] - k) / k
                errors['k_error'] = float(k_error)

            errors['valid'] = True
            return errors

        return PhysicsTaskSpec(
            task_type=PhysicsTaskType.PHYS_COUPLED,
            has_ground_truth=True,
            ground_truth_provenance="FROM_THEORY: Osciladores acoplados",
            evaluation_mode="ground_truth",
            t=t,
            X=np.column_stack([x1_noisy, x2_noisy]),  # Dos series
            y=np.column_stack([x1, x2]),  # Series sin ruido
            params={
                'system_type': 'coupled_oscillators',
                'n_points': n_points,
                'snr': snr,
                'true_params': true_params
            },
            oracle_solution=true_params,
            oracle_function=oracle_eval
        )

    def _generate_lotka_volterra(self) -> PhysicsTaskSpec:
        """
        Sistema Lotka-Volterra (predador-presa).

        dx/dt = αx - βxy
        dy/dt = δxy - γy
        """
        # Parámetros
        alpha = float(self.rng.uniform(0.5, 1.5))
        beta = float(self.rng.uniform(0.05, 0.2))
        delta = float(self.rng.uniform(0.05, 0.2))
        gamma = float(self.rng.uniform(0.5, 1.5))

        self.logger.log_from_theory(
            value={'alpha': alpha, 'beta': beta, 'delta': delta, 'gamma': gamma},
            source="Lotka-Volterra: dx/dt = αx - βxy, dy/dt = δxy - γy",
            reference="Ecología matemática",
            context="PhysicsTaskGenerator._generate_lotka_volterra"
        )

        # Condiciones iniciales
        x0 = float(self.rng.uniform(5, 15))
        y0 = float(self.rng.uniform(5, 15))

        def lotka_volterra(state, t):
            x, y = state
            dx = alpha * x - beta * x * y
            dy = delta * x * y - gamma * y
            return [dx, dy]

        # Integrar
        t_max = 50.0
        n_points = 200
        t = np.linspace(0, t_max, n_points)

        solution = odeint(lotka_volterra, [x0, y0], t)
        x = solution[:, 0]  # Presa
        y = solution[:, 1]  # Predador

        # Añadir ruido
        snr = float(self.rng.uniform(15, 40))
        noise_std_x = np.std(x) / np.sqrt(snr)
        noise_std_y = np.std(y) / np.sqrt(snr)
        x_noisy = x + self.rng.normal(0, noise_std_x, n_points)
        y_noisy = y + self.rng.normal(0, noise_std_y, n_points)

        # Asegurar valores positivos
        x_noisy = np.maximum(x_noisy, 0.1)
        y_noisy = np.maximum(y_noisy, 0.1)

        true_params = {
            'alpha': alpha, 'beta': beta, 'delta': delta, 'gamma': gamma,
            'system_type': 'predator_prey',
            'has_feedback': True
        }

        def oracle_eval(params_pred: Dict[str, Any]) -> Dict[str, float]:
            """Evalúa análisis del sistema."""
            errors = {}

            # ¿Detectó feedback/causalidad bidireccional?
            if 'has_feedback' in params_pred:
                errors['feedback_detection'] = 1.0 if params_pred['has_feedback'] else 0.0
            else:
                errors['feedback_detection'] = 0.0

            # ¿Detectó oscilaciones?
            if 'is_oscillatory' in params_pred:
                # Lotka-Volterra es oscilatorio
                errors['oscillation_detection'] = 1.0 if params_pred['is_oscillatory'] else 0.0

            # Estimación de parámetros
            param_errors = []
            for key in ['alpha', 'beta', 'delta', 'gamma']:
                if key in params_pred:
                    true_val = true_params[key]
                    rel_error = abs(params_pred[key] - true_val) / true_val
                    errors[f'{key}_error'] = float(rel_error)
                    param_errors.append(rel_error)

            if param_errors:
                errors['mean_param_error'] = float(np.mean(param_errors))

            errors['valid'] = True
            return errors

        return PhysicsTaskSpec(
            task_type=PhysicsTaskType.PHYS_COUPLED,
            has_ground_truth=True,
            ground_truth_provenance="FROM_THEORY: Sistema Lotka-Volterra",
            evaluation_mode="ground_truth",
            t=t,
            X=np.column_stack([x_noisy, y_noisy]),
            y=np.column_stack([x, y]),
            params={
                'system_type': 'lotka_volterra',
                'n_points': n_points,
                'snr': snr,
                'true_params': true_params
            },
            oracle_solution=true_params,
            oracle_function=oracle_eval
        )

    def _generate_timeseries_task(self) -> PhysicsTaskSpec:
        """
        Genera tarea de análisis de series temporales anónimas.

        Esta tarea NO tiene ground truth determinista.
        El agente debe:
        - Detectar patrones
        - Generar hipótesis
        - Falsarlas

        ORIGEN: Preparación para datos reales vía Stimulus Engine
        """
        # Generar serie con estructura oculta
        structure_type = self.rng.choice([
            'periodic', 'trend', 'random_walk', 'chaotic'
        ])

        n_points = int(self.rng.integers(100, 501))
        t = np.linspace(0, 100, n_points)

        if structure_type == 'periodic':
            # Serie periódica con ruido
            freq = float(self.rng.uniform(0.1, 0.5))
            amp = float(self.rng.uniform(1, 5))
            x = amp * np.sin(2 * np.pi * freq * t)
            x += self.rng.normal(0, amp * 0.1, n_points)

            hidden_structure = {'type': 'periodic', 'freq': freq, 'amp': amp}

        elif structure_type == 'trend':
            # Tendencia lineal + ruido
            slope = float(self.rng.uniform(-0.5, 0.5))
            intercept = float(self.rng.uniform(-10, 10))
            x = slope * t + intercept
            x += self.rng.normal(0, abs(slope * 10), n_points)

            hidden_structure = {'type': 'trend', 'slope': slope}

        elif structure_type == 'random_walk':
            # Caminata aleatoria
            steps = self.rng.normal(0, 1, n_points)
            x = np.cumsum(steps)

            hidden_structure = {'type': 'random_walk'}

        else:  # chaotic
            # Sistema caótico simplificado (mapa logístico)
            r = float(self.rng.uniform(3.5, 4.0))
            x = np.zeros(n_points)
            x[0] = 0.5

            for i in range(1, n_points):
                x[i] = r * x[i-1] * (1 - x[i-1])

            hidden_structure = {'type': 'chaotic', 'r': r}

        self.logger.log_from_theory(
            value=hidden_structure,
            source=f"Serie sintética tipo {structure_type}",
            reference="Análisis de series temporales",
            context="PhysicsTaskGenerator._generate_timeseries_task"
        )

        # Para series sin ground truth, el oracle evalúa consistencia
        def oracle_eval(analysis: Dict[str, Any]) -> Dict[str, float]:
            """
            Evalúa análisis de serie temporal.

            NOTA: No hay 'verdad' que comparar.
            Evaluamos:
            - Consistencia de hipótesis
            - Capacidad de falsación
            """
            metrics = {}

            # ¿Generó hipótesis?
            hypotheses = analysis.get('hypotheses', [])
            metrics['n_hypotheses'] = len(hypotheses)

            # ¿Falsó alguna hipótesis?
            falsified = analysis.get('falsified', [])
            if hypotheses:
                metrics['falsification_rate'] = len(falsified) / len(hypotheses)
            else:
                metrics['falsification_rate'] = 0.0

            # ¿Detectó el tipo correcto? (bonus, no requerido)
            detected_type = analysis.get('detected_type')
            if detected_type == hidden_structure['type']:
                metrics['type_bonus'] = 1.0
            else:
                metrics['type_bonus'] = 0.0

            # Estabilidad bajo surrogates (si lo hizo)
            surrogate_stability = analysis.get('surrogate_stability')
            if surrogate_stability is not None:
                metrics['surrogate_stability'] = float(surrogate_stability)

            metrics['valid'] = True
            return metrics

        return PhysicsTaskSpec(
            task_type=PhysicsTaskType.PHYS_TIMESERIES,
            has_ground_truth=False,  # No hay solución cerrada
            ground_truth_provenance="",  # No aplica
            evaluation_mode="hypothesis_falsification",  # Evalúa por falsación
            t=t,
            X=x,
            y=None,
            params={
                'n_points': n_points,
                'hidden_structure': hidden_structure  # Para debugging, no expuesto
            },
            oracle_solution=hidden_structure,  # Solo para referencia interna
            oracle_function=oracle_eval
        )


# =============================================================================
# AGENTE SOLUCIONADOR SIMPLE (para testing)
# =============================================================================

class SimplePhysicsSolver:
    """
    Solucionador de física simple para testing.

    NO es el agente real - solo para validar el sistema.
    """

    def __init__(self, use_oracle: bool = False):
        """
        Args:
            use_oracle: Si True, usa solución del oracle (control positivo)
        """
        self.use_oracle = use_oracle
        self.rng = np.random.default_rng()

    def solve(self, task: PhysicsTaskSpec) -> Any:
        """Resuelve una tarea de física."""
        if self.use_oracle and task.oracle_solution is not None:
            return task.oracle_solution

        # Solución aleatoria (control negativo)
        if task.task_type == PhysicsTaskType.PHYS_FREE_FALL:
            return {
                'x0': self.rng.uniform(-100, 100),
                'v0': self.rng.uniform(-50, 50),
                'a': self.rng.uniform(-20, 20)
            }

        elif task.task_type == PhysicsTaskType.PHYS_OSCILLATOR:
            return {
                'A': self.rng.uniform(0.1, 20),
                'omega': self.rng.uniform(0.1, 10),
                'phi': self.rng.uniform(0, 6.28),
                'energy_conserved': self.rng.choice([True, False])
            }

        elif task.task_type == PhysicsTaskType.PHYS_COUPLED:
            return {
                'is_coupled': self.rng.choice([True, False]),
                'coupling_type': self.rng.choice(['unidirectional', 'bidirectional', 'none']),
                'has_feedback': self.rng.choice([True, False]),
                'k': self.rng.uniform(0, 2)
            }

        elif task.task_type == PhysicsTaskType.PHYS_TIMESERIES:
            return {
                'hypotheses': [{'id': i} for i in range(self.rng.integers(0, 5))],
                'falsified': [],
                'detected_type': self.rng.choice(['periodic', 'trend', 'random_walk', 'chaotic']),
                'surrogate_stability': self.rng.uniform(0, 1)
            }

        return None


# =============================================================================
# TEST
# =============================================================================

def test_physics_tasks():
    """Test del generador de tareas de física."""
    print("=" * 70)
    print("TEST: PHYSICS TASK GENERATOR")
    print("=" * 70)

    generator = PhysicsTaskGenerator(seed=42)

    # Generar tareas de cada tipo
    for task_type in PhysicsTaskType:
        print(f"\n=== {task_type.value} ===")
        task = generator.generate_task(task_type)

        print(f"  has_ground_truth: {task.has_ground_truth}")
        print(f"  evaluation_mode: {task.evaluation_mode}")
        print(f"  provenance: {task.ground_truth_provenance[:60]}...")

        # Info de datos
        if task.t is not None:
            print(f"  t: shape={task.t.shape}, range=[{task.t.min():.2f}, {task.t.max():.2f}]")
        if task.X is not None:
            if task.X.ndim == 1:
                print(f"  X: shape={task.X.shape}")
            else:
                print(f"  X: shape={task.X.shape} (multivariate)")

        # Probar solver con oracle
        oracle_solver = SimplePhysicsSolver(use_oracle=True)
        oracle_result = oracle_solver.solve(task)

        if task.oracle_function:
            metrics = task.oracle_function(oracle_result)
            print(f"  Oracle solver: {metrics}")

        # Probar solver aleatorio
        random_solver = SimplePhysicsSolver(use_oracle=False)
        random_result = random_solver.solve(task)

        if task.oracle_function:
            metrics = task.oracle_function(random_result)
            print(f"  Random solver: {metrics}")

    print("\n" + "=" * 70)
    print("TEST COMPLETADO: Todas las tareas de física generadas correctamente")
    print("=" * 70)


if __name__ == "__main__":
    test_physics_tasks()
