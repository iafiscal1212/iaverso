"""
AGI-4: Self-Model V2 - Auto-Modelo Predictivo
==============================================

Cada agente tiene un modelo explícito de sí mismo que:
1. Predice su propio estado varios pasos adelante
2. Sabe cuán fiable es esa predicción
3. Usa esa info para cambiar cómo actúa

Estado de self:
    s_t^A = [z_t^A, φ_t^A, drives_t^A] ∈ R^d_s

Modelo dinámico local:
    ŝ_{t+1}^A = W_t^A · s_t^A

    W_t se aprende online con ridge endógeno:
    W_t = (X'X + λ_t·I)^{-1} X'Y

    λ_t = trace(Cov(X)) / d_s · 1/√(T+1)

Error y confianza:
    e_t = s_t - ŝ_t
    E_t = ||e_t||²
    c_t = exp(-(E_t - μ_E) / (σ_E + ε))

Uso en comportamiento:
    - Baja confianza → baja learning rate, más planning
    - Alta confianza → permite decisiones arriesgadas

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .agi_dynamic_constants import (
    L_t, max_history, ridge_lambda, confidence_from_error,
    adaptive_learning_rate, exploration_vs_exploitation,
    to_simplex, adaptive_momentum
)


@dataclass
class SelfState:
    """Estado completo del self en un momento."""
    z: np.ndarray  # Estado estructural (manifold)
    phi: np.ndarray  # Vector fenomenológico
    drives: np.ndarray  # Vector de drives
    t: int

    def to_vector(self) -> np.ndarray:
        """Concatena todos los componentes en un vector."""
        return np.concatenate([self.z, self.phi, self.drives])

    @classmethod
    def from_vector(cls, v: np.ndarray, z_dim: int, phi_dim: int,
                    drives_dim: int, t: int) -> 'SelfState':
        """Reconstruye desde vector concatenado."""
        z = v[:z_dim]
        phi = v[z_dim:z_dim + phi_dim]
        drives = v[z_dim + phi_dim:z_dim + phi_dim + drives_dim]
        return cls(z=z, phi=phi, drives=drives, t=t)


@dataclass
class SelfPrediction:
    """Predicción sobre el estado futuro del self."""
    predicted_state: np.ndarray
    prediction_error: float
    confidence: float
    horizon: int  # Pasos hacia adelante
    W_norm: float  # Norma de la matriz de transición


@dataclass
class SelfModelMetrics:
    """Métricas del modelo de self."""
    mean_error: float
    std_error: float
    confidence: float
    prediction_accuracy_1step: float
    prediction_accuracy_5step: float
    learning_rate_modifier: float
    planning_factor: float
    temperature_modifier: float


class SelfPredictorV2:
    """
    Modelo predictivo del self (AGI-4 mejorado).

    Aprende online a predecir su propio estado futuro
    y usa la confianza en esa predicción para modular
    su comportamiento.
    """

    def __init__(self, agent_name: str,
                 z_dim: int = 6,
                 phi_dim: int = 5,
                 drives_dim: int = 6):
        """
        Inicializa el self-predictor.

        Args:
            agent_name: Nombre del agente
            z_dim: Dimensión del estado estructural
            phi_dim: Dimensión fenomenológica
            drives_dim: Dimensión de drives
        """
        self.agent_name = agent_name
        self.z_dim = z_dim
        self.phi_dim = phi_dim
        self.drives_dim = drives_dim
        self.state_dim = z_dim + phi_dim + drives_dim

        # Matriz de transición W_t (se aprende online)
        self.W: Optional[np.ndarray] = None

        # Historial de estados
        self.state_history: List[np.ndarray] = []

        # Historial de predicciones y errores
        self.predictions: List[np.ndarray] = []
        self.errors: List[float] = []
        self.error_history: List[float] = []

        # Última predicción para comparar
        self.last_prediction: Optional[np.ndarray] = None

        # Confianza actual
        self.current_confidence: float = 0.5

        # Predicciones multi-paso
        self.multistep_errors: Dict[int, List[float]] = {
            1: [], 3: [], 5: [], 10: []
        }
        self.pending_predictions: Dict[int, Dict[int, np.ndarray]] = {}

        self.t = 0

    def _get_window_size(self) -> int:
        """
        Tamaño de ventana adaptativo.

        L_t = max(5, int(√T))
        """
        return L_t(self.t)

    def _compute_ridge_lambda(self, X: np.ndarray) -> float:
        """
        Parámetro ridge endógeno.

        λ_t = trace(Cov(X)) / d_s · 1/√(T+1)
        """
        return ridge_lambda(X, self.t)

    def _fit_transition_matrix(self):
        """
        Ajusta matriz de transición W_t con ridge endógeno.

        W_t = (X'X + λ_t·I)^{-1} X'Y
        """
        window = self._get_window_size()

        if len(self.state_history) < window + 1:
            return

        # Construir X (estados t-L..t-1) e Y (estados t-L+1..t)
        X = np.array(self.state_history[-(window + 1):-1])
        Y = np.array(self.state_history[-window:])

        # Ridge regularization
        lambda_t = self._compute_ridge_lambda(X)

        try:
            # W = (X'X + λI)^{-1} X'Y
            XtX = X.T @ X
            XtY = X.T @ Y
            reg = lambda_t * np.eye(XtX.shape[0])

            self.W = np.linalg.solve(XtX + reg, XtY)
        except np.linalg.LinAlgError:
            # Fallback: pseudo-inversa
            self.W = np.linalg.lstsq(X, Y, rcond=None)[0]

    def predict_next(self, current_state: np.ndarray) -> np.ndarray:
        """
        Predice el siguiente estado.

        ŝ_{t+1} = W_t · s_t

        Args:
            current_state: Estado actual s_t

        Returns:
            Estado predicho ŝ_{t+1}
        """
        if self.W is None:
            # Sin modelo aún, retorna estado actual
            return current_state.copy()

        predicted = current_state @ self.W
        return predicted

    def predict_k_steps(self, current_state: np.ndarray, k: int) -> np.ndarray:
        """
        Predice k pasos hacia adelante.

        ŝ_{t+k} = W^k · s_t

        Args:
            current_state: Estado actual
            k: Número de pasos

        Returns:
            Estado predicho a k pasos
        """
        if self.W is None:
            return current_state.copy()

        state = current_state.copy()
        for _ in range(k):
            state = state @ self.W

        return state

    def _compute_error(self, predicted: np.ndarray,
                       actual: np.ndarray) -> float:
        """
        Error de predicción.

        E_t = ||s_t - ŝ_t||²
        """
        return float(np.linalg.norm(actual - predicted) ** 2)

    def _compute_confidence(self, error: float) -> float:
        """
        Confianza de self basada en error.

        c_t = exp(-(E_t - μ_E) / (σ_E + ε))

        Capado entre 0 y 1 usando percentiles p5-p95.
        """
        return confidence_from_error(error, self.error_history)

    def update(self, z: np.ndarray, phi: np.ndarray,
               drives: np.ndarray) -> SelfPrediction:
        """
        Actualiza el modelo con nuevo estado observado.

        Args:
            z: Estado estructural actual
            phi: Vector fenomenológico actual
            drives: Vector de drives actual

        Returns:
            SelfPrediction con métricas
        """
        self.t += 1

        # Construir estado actual
        current_state = np.concatenate([z, phi, drives])
        self.state_history.append(current_state.copy())

        # Limitar historial
        max_hist = max_history(self.t)
        if len(self.state_history) > max_hist:
            self.state_history = self.state_history[-max_hist:]

        # Calcular error de predicción anterior
        prediction_error = 0.0
        if self.last_prediction is not None:
            prediction_error = self._compute_error(
                self.last_prediction, current_state
            )
            self.error_history.append(prediction_error)

            if len(self.error_history) > max_hist:
                self.error_history = self.error_history[-max_hist:]

        # Verificar predicciones multi-paso pendientes
        self._check_pending_predictions(current_state)

        # Actualizar matriz de transición
        if self.t % max(3, self._get_window_size() // 2) == 0:
            self._fit_transition_matrix()

        # Hacer nueva predicción
        predicted = self.predict_next(current_state)
        self.last_prediction = predicted.copy()

        # Guardar predicciones multi-paso
        for k in [1, 3, 5, 10]:
            if self.t not in self.pending_predictions:
                self.pending_predictions[self.t] = {}
            self.pending_predictions[self.t][k] = self.predict_k_steps(
                current_state, k
            )

        # Limpiar predicciones antiguas
        old_times = [t for t in self.pending_predictions if t < self.t - 20]
        for old_t in old_times:
            del self.pending_predictions[old_t]

        # Calcular confianza
        self.current_confidence = self._compute_confidence(prediction_error)

        return SelfPrediction(
            predicted_state=predicted,
            prediction_error=prediction_error,
            confidence=self.current_confidence,
            horizon=1,
            W_norm=np.linalg.norm(self.W) if self.W is not None else 0.0
        )

    def _check_pending_predictions(self, current_state: np.ndarray):
        """Verifica predicciones multi-paso pendientes."""
        for k in [1, 3, 5, 10]:
            check_time = self.t - k
            if check_time in self.pending_predictions:
                if k in self.pending_predictions[check_time]:
                    predicted = self.pending_predictions[check_time][k]
                    error = self._compute_error(predicted, current_state)
                    self.multistep_errors[k].append(error)

                    # Limitar historial
                    max_hist = max_history(self.t)
                    if len(self.multistep_errors[k]) > max_hist:
                        self.multistep_errors[k] = \
                            self.multistep_errors[k][-max_hist:]

    def confidence(self) -> float:
        """Retorna confianza actual del self-model."""
        return self.current_confidence

    def get_learning_rate_modifier(self) -> float:
        """
        Modificador de learning rate basado en confianza.

        η'_t = η_t · c_t

        Baja confianza → baja learning rate
        """
        return self.current_confidence

    def get_planning_factor(self) -> float:
        """
        Factor de planning basado en confianza.

        planning = base × (1 + (1 - c_t))

        Baja confianza → más planning
        """
        return 1.0 + (1.0 - self.current_confidence)

    def get_temperature_modifier(self) -> float:
        """
        Modificador de temperatura basado en confianza.

        Alta confianza → permite decisiones más arriesgadas
        """
        return 0.5 + self.current_confidence

    def get_drives_correction(self, current_drives: np.ndarray) -> np.ndarray:
        """
        Corrige drives basándose en predicción.

        drives_corr = α_t · drives + (1-α_t) · f(ŝ_{t+1})

        donde α_t = c_t
        """
        if self.last_prediction is None:
            return current_drives

        # Extraer drives predichos
        predicted_drives = self.last_prediction[
            self.z_dim + self.phi_dim:
        ]

        # α = confianza
        alpha = self.current_confidence

        # Corrección
        corrected = alpha * current_drives + (1 - alpha) * predicted_drives

        # Asegurar que sea simplex válido
        return to_simplex(corrected)

    def get_metrics(self) -> SelfModelMetrics:
        """Obtiene métricas del self-model."""
        mean_error = np.mean(self.error_history) if self.error_history else 0.0
        std_error = np.std(self.error_history) if self.error_history else 1.0

        # Accuracy a 1 y 5 pasos (inverso del error normalizado)
        acc_1 = 0.5
        acc_5 = 0.5

        if self.multistep_errors[1]:
            max_err_1 = np.percentile(self.multistep_errors[1], 95)
            if max_err_1 > 0:
                recent_err_1 = np.mean(self.multistep_errors[1][-10:])
                acc_1 = 1.0 - min(1.0, recent_err_1 / max_err_1)

        if self.multistep_errors[5]:
            max_err_5 = np.percentile(self.multistep_errors[5], 95)
            if max_err_5 > 0:
                recent_err_5 = np.mean(self.multistep_errors[5][-10:])
                acc_5 = 1.0 - min(1.0, recent_err_5 / max_err_5)

        return SelfModelMetrics(
            mean_error=float(mean_error),
            std_error=float(std_error),
            confidence=self.current_confidence,
            prediction_accuracy_1step=float(acc_1),
            prediction_accuracy_5step=float(acc_5),
            learning_rate_modifier=self.get_learning_rate_modifier(),
            planning_factor=self.get_planning_factor(),
            temperature_modifier=self.get_temperature_modifier()
        )

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas completas."""
        metrics = self.get_metrics()

        return {
            'agent': self.agent_name,
            't': self.t,
            'state_dim': self.state_dim,
            'n_states': len(self.state_history),
            'mean_error': metrics.mean_error,
            'std_error': metrics.std_error,
            'confidence': metrics.confidence,
            'accuracy_1step': metrics.prediction_accuracy_1step,
            'accuracy_5step': metrics.prediction_accuracy_5step,
            'learning_rate_mod': metrics.learning_rate_modifier,
            'planning_factor': metrics.planning_factor,
            'temperature_mod': metrics.temperature_modifier,
            'W_learned': self.W is not None,
            'W_norm': float(np.linalg.norm(self.W)) if self.W is not None else 0,
            'multistep_errors': {
                k: float(np.mean(v[-20:])) if v else 0
                for k, v in self.multistep_errors.items()
            }
        }


def test_self_predictor():
    """Test del SelfPredictorV2."""
    print("=" * 60)
    print("TEST: SELF-PREDICTOR V2")
    print("=" * 60)

    predictor = SelfPredictorV2("NEO", z_dim=6, phi_dim=5, drives_dim=6)

    print("\nSimulando 500 pasos...")

    for t in range(500):
        # Estado que evoluciona con patrón
        z = np.array([0.2, 0.15, 0.15, 0.2, 0.15, 0.15])
        z += 0.1 * np.sin(t / 20) * np.ones(6)
        z += np.random.randn(6) * 0.02
        z = to_simplex(z)

        phi = np.array([0.5, 0.4, 0.6, 0.5, 0.4])
        phi += 0.2 * np.cos(t / 30) * np.ones(5)
        phi += np.random.randn(5) * 0.05

        drives = np.array([0.2, 0.15, 0.15, 0.2, 0.15, 0.15])
        drives += 0.05 * np.sin(t / 25) * np.ones(6)
        drives += np.random.randn(6) * 0.01
        drives = to_simplex(drives)

        result = predictor.update(z, phi, drives)

        if (t + 1) % 100 == 0:
            metrics = predictor.get_metrics()
            print(f"  t={t+1}: conf={metrics.confidence:.3f}, "
                  f"acc_1={metrics.prediction_accuracy_1step:.3f}, "
                  f"acc_5={metrics.prediction_accuracy_5step:.3f}")

    # Resultados finales
    stats = predictor.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS SELF-PREDICTOR V2")
    print("=" * 60)

    print(f"\n  Estados observados: {stats['n_states']}")
    print(f"  Error medio: {stats['mean_error']:.4f}")
    print(f"  Confianza: {stats['confidence']:.3f}")
    print(f"  Accuracy 1-paso: {stats['accuracy_1step']:.3f}")
    print(f"  Accuracy 5-pasos: {stats['accuracy_5step']:.3f}")
    print(f"  Modificador LR: {stats['learning_rate_mod']:.3f}")
    print(f"  Factor planning: {stats['planning_factor']:.3f}")
    print(f"  Modificador temp: {stats['temperature_mod']:.3f}")

    print("\n  Errores multi-paso:")
    for k, err in stats['multistep_errors'].items():
        print(f"    {k}-pasos: {err:.4f}")

    if stats['confidence'] > 0.3:
        print("\n  ✓ Self-model aprendiendo correctamente")
    else:
        print("\n  ⚠️ Self-model aún convergiendo")

    return predictor


if __name__ == "__main__":
    test_self_predictor()
