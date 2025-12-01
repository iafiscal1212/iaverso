"""
AGI-5: Theory of Mind V2 - Modelo del Otro
===========================================

Cada agente tiene un modelo explícito del otro:
"NEO predice cómo será EVA dentro de k pasos"
"EVA anticipa cómo reaccionará ALEX si yo hago Z"

Estado observable del otro:
    o_t^{A←B} = [z_t^B, φ_t^B, drives_t^B] visible ∈ R^{d_o}

Creencia interna sobre el otro:
    b_t^{A→B} ∈ R^{d_b}

Modelo dinámico del otro:
    ô_{t+1}^{A←B} = F_t^{A→B} · o_t^{A←B}

    F_t se aprende online con ridge endógeno.

ToM-accuracy:
    ToMAcc_t^{AB} = 1 - ||e_t^{AB}|| / percentile_95(||e_i^{AB}||)

Uso en conducta:
    - Selección de partner: U'(A,B) = U(A,B) + β_t · ToMAcc
    - Acciones anticipatorias: simula reacción de B

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .agi_dynamic_constants import (
    L_t, max_history, ridge_lambda, tom_accuracy,
    adaptive_momentum, dynamic_dim_from_covariance,
    to_simplex, softmax
)


@dataclass
class OtherState:
    """Estado observable de otro agente."""
    agent_id: str
    z: np.ndarray
    phi: np.ndarray
    drives: np.ndarray
    t: int

    def to_vector(self) -> np.ndarray:
        """Concatena componentes en vector."""
        return np.concatenate([self.z, self.phi, self.drives])

    @classmethod
    def from_vector(cls, v: np.ndarray, agent_id: str,
                    z_dim: int, phi_dim: int, drives_dim: int,
                    t: int) -> 'OtherState':
        """Reconstruye desde vector."""
        z = v[:z_dim]
        phi = v[z_dim:z_dim + phi_dim]
        drives = v[z_dim + phi_dim:]
        return cls(agent_id=agent_id, z=z, phi=phi, drives=drives, t=t)


@dataclass
class ToMPrediction:
    """Predicción sobre otro agente."""
    target_agent: str
    predicted_state: np.ndarray
    prediction_error: float
    tom_accuracy: float
    confidence: float


@dataclass
class BeliefState:
    """Creencia interna sobre otro agente."""
    target_agent: str
    belief_vector: np.ndarray
    uncertainty: float
    last_update: int


class OtherModel:
    """
    Modelo de otro agente (Theory of Mind).

    Aprende a predecir el comportamiento de otro agente
    basándose en observaciones.
    """

    def __init__(self, observer_name: str, target_name: str,
                 z_dim: int = 6, phi_dim: int = 5, drives_dim: int = 6):
        """
        Inicializa modelo del otro.

        Args:
            observer_name: Nombre del agente observador (A)
            target_name: Nombre del agente objetivo (B)
            z_dim: Dimensión de z
            phi_dim: Dimensión de φ
            drives_dim: Dimensión de drives
        """
        self.observer_name = observer_name
        self.target_name = target_name
        self.z_dim = z_dim
        self.phi_dim = phi_dim
        self.drives_dim = drives_dim
        self.state_dim = z_dim + phi_dim + drives_dim

        # Matriz de transición F_t (se aprende online)
        self.F: Optional[np.ndarray] = None

        # Historial de observaciones
        self.observation_history: List[np.ndarray] = []

        # Historial de predicciones y errores
        self.predictions: List[np.ndarray] = []
        self.errors: List[float] = []
        self.error_history: List[float] = []

        # Última predicción
        self.last_prediction: Optional[np.ndarray] = None

        # Creencia actual sobre el otro
        self.belief: Optional[BeliefState] = None

        # ToM accuracy
        self.current_tom_accuracy: float = 0.5

        # Predicciones multi-paso
        self.multistep_errors: Dict[int, List[float]] = {
            1: [], 3: [], 5: []
        }
        self.pending_predictions: Dict[int, Dict[int, np.ndarray]] = {}

        # Historial de interacciones (para partner selection)
        self.interaction_values: List[float] = []

        self.t = 0
        self.T_AB = 0  # Observaciones específicas de este par

    def _get_window_size(self) -> int:
        """Ventana adaptativa basada en observaciones del par."""
        return L_t(self.T_AB)

    def _compute_ridge_lambda(self, X: np.ndarray) -> float:
        """Ridge lambda endógeno."""
        return ridge_lambda(X, self.T_AB)

    def _fit_transition_matrix(self):
        """
        Ajusta matriz de transición F_t con ridge.

        F_t = (O'O + λI)^{-1} O'Y
        """
        window = self._get_window_size()

        if len(self.observation_history) < window + 1:
            return

        # Construir matrices
        O = np.array(self.observation_history[-(window + 1):-1])
        Y = np.array(self.observation_history[-window:])

        lambda_t = self._compute_ridge_lambda(O)

        try:
            OtO = O.T @ O
            OtY = O.T @ Y
            reg = lambda_t * np.eye(OtO.shape[0])

            self.F = np.linalg.solve(OtO + reg, OtY)
        except np.linalg.LinAlgError:
            self.F = np.linalg.lstsq(O, Y, rcond=None)[0]

    def observe(self, z: np.ndarray, phi: np.ndarray,
                drives: np.ndarray) -> ToMPrediction:
        """
        Observa el estado actual del otro agente.

        Args:
            z: Estado estructural observado de B
            phi: Vector fenomenológico de B
            drives: Drives de B

        Returns:
            ToMPrediction con métricas
        """
        self.t += 1
        self.T_AB += 1

        # Construir observación
        observation = np.concatenate([z, phi, drives])
        self.observation_history.append(observation.copy())

        # Limitar historial
        max_hist = max_history(self.t)
        if len(self.observation_history) > max_hist:
            self.observation_history = self.observation_history[-max_hist:]

        # Calcular error de predicción anterior
        prediction_error = 0.0
        if self.last_prediction is not None:
            prediction_error = float(
                np.linalg.norm(observation - self.last_prediction)
            )
            self.error_history.append(prediction_error)

            if len(self.error_history) > max_hist:
                self.error_history = self.error_history[-max_hist:]

        # Verificar predicciones multi-paso
        self._check_pending_predictions(observation)

        # Actualizar matriz de transición
        window = self._get_window_size()
        if self.T_AB % max(3, window // 2) == 0:
            self._fit_transition_matrix()

        # Nueva predicción
        predicted = self.predict_next(observation)
        self.last_prediction = predicted.copy()

        # Guardar predicciones multi-paso
        for k in [1, 3, 5]:
            if self.t not in self.pending_predictions:
                self.pending_predictions[self.t] = {}
            self.pending_predictions[self.t][k] = self.predict_k_steps(
                observation, k
            )

        # Limpiar predicciones antiguas
        old_times = [t for t in self.pending_predictions if t < self.t - 10]
        for old_t in old_times:
            del self.pending_predictions[old_t]

        # Calcular ToM accuracy
        self.current_tom_accuracy = tom_accuracy(
            prediction_error, self.error_history
        )

        # Actualizar creencia
        self._update_belief(observation)

        return ToMPrediction(
            target_agent=self.target_name,
            predicted_state=predicted,
            prediction_error=prediction_error,
            tom_accuracy=self.current_tom_accuracy,
            confidence=self.current_tom_accuracy
        )

    def _check_pending_predictions(self, current_observation: np.ndarray):
        """Verifica predicciones multi-paso."""
        for k in [1, 3, 5]:
            check_time = self.t - k
            if check_time in self.pending_predictions:
                if k in self.pending_predictions[check_time]:
                    predicted = self.pending_predictions[check_time][k]
                    error = float(np.linalg.norm(
                        current_observation - predicted
                    ))
                    self.multistep_errors[k].append(error)

                    max_hist = max_history(self.t)
                    if len(self.multistep_errors[k]) > max_hist:
                        self.multistep_errors[k] = \
                            self.multistep_errors[k][-max_hist:]

    def predict_next(self, current_observation: np.ndarray) -> np.ndarray:
        """
        Predice siguiente estado del otro.

        ô_{t+1} = F_t · o_t
        """
        if self.F is None:
            return current_observation.copy()

        return current_observation @ self.F

    def predict_k_steps(self, observation: np.ndarray, k: int) -> np.ndarray:
        """Predice k pasos hacia adelante."""
        if self.F is None:
            return observation.copy()

        state = observation.copy()
        for _ in range(k):
            state = state @ self.F

        return state

    def _update_belief(self, observation: np.ndarray):
        """Actualiza creencia interna sobre el otro."""
        if self.belief is None:
            self.belief = BeliefState(
                target_agent=self.target_name,
                belief_vector=observation.copy(),
                uncertainty=1.0,
                last_update=self.t
            )
        else:
            # Actualización con momentum adaptativo
            beta = adaptive_momentum(self.error_history)
            self.belief.belief_vector = (
                beta * self.belief.belief_vector +
                (1 - beta) * observation
            )
            self.belief.uncertainty = 1.0 - self.current_tom_accuracy
            self.belief.last_update = self.t

    def tom_accuracy_score(self) -> float:
        """Retorna ToM accuracy actual."""
        return self.current_tom_accuracy

    def record_interaction_value(self, value: float):
        """Registra valor de una interacción con este agente."""
        self.interaction_values.append(value)
        max_hist = max_history(self.t)
        if len(self.interaction_values) > max_hist:
            self.interaction_values = self.interaction_values[-max_hist:]

    def get_partner_selection_bonus(self) -> float:
        """
        Bonus para selección de partner.

        β_t · ToMAcc_t

        donde β_t deriva de la varianza histórica de ToMAcc.
        """
        if len(self.error_history) < 5:
            return 0.0

        # β basado en estabilidad del ToM accuracy
        tom_scores = [
            tom_accuracy(e, self.error_history)
            for e in self.error_history[-20:]
        ]

        if len(tom_scores) < 3:
            beta = 0.5
        else:
            variance = np.var(tom_scores)
            # Menor varianza → mayor β (más confiable)
            beta = 1.0 / (1.0 + variance)

        return beta * self.current_tom_accuracy

    def simulate_reaction(self, my_action: np.ndarray,
                          current_observation: np.ndarray) -> np.ndarray:
        """
        Simula cómo reaccionará el otro a mi acción.

        Simple: predice siguiente estado dado observación actual.
        En implementación completa, usaría modelo de reacción.

        Args:
            my_action: Mi acción propuesta
            current_observation: Observación actual del otro

        Returns:
            Estado predicho del otro
        """
        # Por ahora, predicción simple
        # En versión completa: incorporar my_action en la predicción
        return self.predict_next(current_observation)

    def evaluate_action(self, my_action: np.ndarray,
                        current_observation: np.ndarray,
                        my_value_function: callable) -> float:
        """
        Evalúa acción anticipando reacción del otro.

        Simula 1 paso de WORLD-1 usando predicción de B.

        Args:
            my_action: Acción propuesta
            current_observation: Observación actual de B
            my_value_function: Función que evalúa valor dado estado

        Returns:
            Valor esperado de la acción
        """
        # Predecir reacción del otro
        predicted_other = self.simulate_reaction(my_action, current_observation)

        # Evaluar mi valor dado la reacción predicha
        # (Esto requiere el value function del agente)
        expected_value = my_value_function(predicted_other)

        # Ponderar por confianza en la predicción
        return expected_value * self.current_tom_accuracy

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del modelo."""
        mean_error = np.mean(self.error_history) if self.error_history else 0.0

        multistep_accuracies = {}
        for k, errors in self.multistep_errors.items():
            if errors:
                p95 = np.percentile(errors, 95)
                recent = np.mean(errors[-10:]) if errors else 0
                acc = 1 - min(1, recent / (p95 + 1e-8))
                multistep_accuracies[k] = float(acc)
            else:
                multistep_accuracies[k] = 0.5

        return {
            'observer': self.observer_name,
            'target': self.target_name,
            't': self.t,
            'T_AB': self.T_AB,
            'n_observations': len(self.observation_history),
            'mean_error': float(mean_error),
            'tom_accuracy': self.current_tom_accuracy,
            'partner_bonus': self.get_partner_selection_bonus(),
            'F_learned': self.F is not None,
            'F_norm': float(np.linalg.norm(self.F)) if self.F is not None else 0,
            'belief_uncertainty': self.belief.uncertainty if self.belief else 1.0,
            'multistep_accuracy': multistep_accuracies,
            'mean_interaction_value': float(np.mean(self.interaction_values))
                if self.interaction_values else 0
        }


class TheoryOfMindSystem:
    """
    Sistema completo de Theory of Mind para múltiples agentes.

    Mantiene un OtherModel para cada par (observer, target).
    """

    def __init__(self, agent_names: List[str],
                 z_dim: int = 6, phi_dim: int = 5, drives_dim: int = 6):
        """
        Inicializa sistema ToM.

        Args:
            agent_names: Lista de nombres de agentes
            z_dim, phi_dim, drives_dim: Dimensiones
        """
        self.agent_names = agent_names
        self.z_dim = z_dim
        self.phi_dim = phi_dim
        self.drives_dim = drives_dim

        # Crear modelo para cada par
        self.models: Dict[Tuple[str, str], OtherModel] = {}
        for observer in agent_names:
            for target in agent_names:
                if observer != target:
                    key = (observer, target)
                    self.models[key] = OtherModel(
                        observer, target, z_dim, phi_dim, drives_dim
                    )

    def observe(self, observer: str, target: str,
                z: np.ndarray, phi: np.ndarray,
                drives: np.ndarray) -> ToMPrediction:
        """
        Observer observa estado de target.

        Args:
            observer: Nombre del observador
            target: Nombre del objetivo
            z, phi, drives: Estado de target

        Returns:
            ToMPrediction
        """
        key = (observer, target)
        if key not in self.models:
            raise ValueError(f"Par ({observer}, {target}) no existe")

        return self.models[key].observe(z, phi, drives)

    def get_model(self, observer: str, target: str) -> OtherModel:
        """Obtiene modelo específico."""
        return self.models[(observer, target)]

    def get_partner_utilities(self, agent: str) -> Dict[str, float]:
        """
        Obtiene utilidades ajustadas por ToM para selección de partner.

        U'(A,B) = U_base + bonus_ToM
        """
        utilities = {}
        for other in self.agent_names:
            if other != agent:
                model = self.models[(agent, other)]
                utilities[other] = model.get_partner_selection_bonus()

        return utilities

    def select_best_partner(self, agent: str,
                            base_utilities: Dict[str, float]) -> str:
        """
        Selecciona mejor partner combinando utilidad base y ToM.

        Args:
            agent: Agente que selecciona
            base_utilities: Utilidades base U(A,B)

        Returns:
            Nombre del mejor partner
        """
        tom_bonuses = self.get_partner_utilities(agent)

        combined = {}
        for other, base_u in base_utilities.items():
            tom_bonus = tom_bonuses.get(other, 0)
            combined[other] = base_u + tom_bonus

        return max(combined, key=combined.get)

    def get_statistics(self) -> Dict:
        """Estadísticas globales del sistema ToM."""
        all_accuracies = []
        pair_stats = []

        for (observer, target), model in self.models.items():
            stats = model.get_statistics()
            all_accuracies.append(stats['tom_accuracy'])
            pair_stats.append({
                'pair': f"{observer}→{target}",
                'accuracy': stats['tom_accuracy'],
                'n_obs': stats['n_observations']
            })

        return {
            'n_agents': len(self.agent_names),
            'n_pairs': len(self.models),
            'mean_tom_accuracy': float(np.mean(all_accuracies))
                if all_accuracies else 0,
            'std_tom_accuracy': float(np.std(all_accuracies))
                if all_accuracies else 0,
            'pairs': sorted(pair_stats, key=lambda x: x['accuracy'],
                           reverse=True)
        }


def test_theory_of_mind():
    """Test del sistema Theory of Mind."""
    print("=" * 60)
    print("TEST: THEORY OF MIND V2")
    print("=" * 60)

    agents = ['NEO', 'EVA', 'ALEX']
    tom = TheoryOfMindSystem(agents)

    print(f"\nSimulando {len(agents)} agentes con observaciones mutuas...")

    for t in range(300):
        # Generar estados para cada agente
        states = {}
        for agent in agents:
            z = np.random.dirichlet(np.ones(6))
            phi = 0.5 + 0.2 * np.sin(t / 20) * np.ones(5) + np.random.randn(5) * 0.05
            drives = np.random.dirichlet(np.ones(6))
            states[agent] = (z, phi, drives)

        # Cada agente observa a los demás
        for observer in agents:
            for target in agents:
                if observer != target:
                    z, phi, drives = states[target]
                    tom.observe(observer, target, z, phi, drives)

        if (t + 1) % 50 == 0:
            stats = tom.get_statistics()
            print(f"  t={t+1}: mean_ToM={stats['mean_tom_accuracy']:.3f}")

    # Resultados finales
    stats = tom.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS THEORY OF MIND V2")
    print("=" * 60)

    print(f"\n  Agentes: {stats['n_agents']}")
    print(f"  Pares: {stats['n_pairs']}")
    print(f"  ToM accuracy media: {stats['mean_tom_accuracy']:.3f}")
    print(f"  ToM accuracy std: {stats['std_tom_accuracy']:.3f}")

    print("\n  Top 5 pares:")
    for pair in stats['pairs'][:5]:
        print(f"    {pair['pair']}: acc={pair['accuracy']:.3f}, "
              f"obs={pair['n_obs']}")

    # Test partner selection
    print("\n  Test selección de partner:")
    base_utilities = {'EVA': 0.6, 'ALEX': 0.5}
    best = tom.select_best_partner('NEO', base_utilities)
    print(f"    NEO elige: {best}")

    if stats['mean_tom_accuracy'] > 0.3:
        print("\n  ✓ Theory of Mind funcionando")
    else:
        print("\n  ⚠️ ToM aún convergiendo")

    return tom


if __name__ == "__main__":
    test_theory_of_mind()
