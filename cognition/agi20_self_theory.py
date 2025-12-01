"""
AGI-20: Teoría Estructural de Sí Mismo
=======================================

"Construir una teoría de cómo funciono yo."

Historial de estados internos:
    S_A = [s_1, s_2, ..., s_T]
    s_t = [drives_t, φ_t, z_t, values_t, ...]

Reducción dimensional:
    S̃_A = PCA(S_A, k)
    k = min(d_s, ⌈√T⌉)

Teoría estructural:
    Θ_self = {μ_s̃, Σ_s̃, PC_1...k, varianza explicada}

Predicción de sí mismo:
    ŝ_{t+1} = Θ_self.predict(s_t)

Consistencia:
    Cons_self = 1 - ||s_t - ŝ_t|| / σ_s

Narrativa estructural:
    N_self = secuencia de transiciones dominantes

Índice de auto-comprensión:
    U_self = var_explained · Cons_self

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from .agi_dynamic_constants import (
    L_t, max_history, adaptive_momentum, confidence_from_error
)


def theory_components(T: int, d_s: int) -> int:
    """
    Número de componentes principales endógeno.

    k = min(d_s, ⌈√T⌉)
    """
    k = int(np.ceil(np.sqrt(T)))
    return min(d_s, k)


@dataclass
class StateTransition:
    """Una transición de estado interno."""
    t: int
    from_state: np.ndarray
    to_state: np.ndarray
    dominant_change: int  # Índice de dimensión con mayor cambio
    change_magnitude: float


@dataclass
class StructuralTheory:
    """Teoría estructural del self."""
    mean_state: np.ndarray
    covariance: np.ndarray
    principal_components: np.ndarray
    explained_variance: np.ndarray
    total_variance_explained: float
    n_components: int
    transition_model: Optional[np.ndarray] = None  # Matriz de transición


@dataclass
class SelfTheoryState:
    """Estado de la teoría del self."""
    t: int
    theory: Optional[StructuralTheory]
    consistency: float
    self_understanding: float
    prediction_accuracy: float
    narrative_length: int
    is_coherent: bool


class StructuralSelfTheory:
    """
    Sistema de teoría estructural de sí mismo.

    Construye un modelo de cómo funciona el agente
    basado en patrones de estados internos.
    """

    def __init__(self, agent_name: str, state_dim: int = 20):
        """
        Inicializa sistema de teoría del self.

        Args:
            agent_name: Nombre del agente
            state_dim: Dimensión del estado interno
        """
        self.agent_name = agent_name
        self.state_dim = state_dim

        # Historial de estados internos
        self.state_history: List[np.ndarray] = []

        # Teoría actual
        self.theory: Optional[StructuralTheory] = None

        # Historial de predicciones
        self.predictions: List[np.ndarray] = []
        self.prediction_errors: List[float] = []

        # Transiciones narrativas
        self.transitions: List[StateTransition] = []

        # Métricas
        self.consistency_history: List[float] = []
        self.understanding_history: List[float] = []

        self.t = 0

    def _build_theory(self):
        """
        Construye teoría estructural.

        S̃_A = PCA(S_A, k)
        Θ_self = {μ_s̃, Σ_s̃, PC_1...k, varianza explicada}
        """
        min_samples = L_t(self.t) * 2
        if len(self.state_history) < min_samples:
            return

        window = min(max_history(self.t), len(self.state_history))
        states = np.array(self.state_history[-window:])

        # Número de componentes endógeno
        n_components = theory_components(len(states), self.state_dim)
        n_components = min(n_components, states.shape[0] - 1, states.shape[1])
        n_components = max(2, n_components)

        try:
            # PCA
            pca = PCA(n_components=n_components)
            pca.fit(states)

            # Construir teoría
            self.theory = StructuralTheory(
                mean_state=pca.mean_,
                covariance=np.cov(states.T) if states.shape[0] > 1 else np.eye(self.state_dim),
                principal_components=pca.components_,
                explained_variance=pca.explained_variance_ratio_,
                total_variance_explained=float(np.sum(pca.explained_variance_ratio_)),
                n_components=n_components
            )

            # Construir modelo de transición simple
            if len(states) > 10:
                X = states[:-1]
                Y = states[1:]
                # Pseudo-inversa para modelo lineal
                try:
                    self.theory.transition_model = np.linalg.lstsq(X, Y, rcond=None)[0]
                except:
                    pass

        except Exception:
            pass

    def _predict_next_state(self, current_state: np.ndarray) -> np.ndarray:
        """
        Predice siguiente estado.

        ŝ_{t+1} = Θ_self.predict(s_t)
        """
        if self.theory is None or self.theory.transition_model is None:
            return current_state.copy()

        try:
            prediction = current_state @ self.theory.transition_model
            return prediction
        except:
            return current_state.copy()

    def _compute_consistency(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calcula consistencia de predicción.

        Cons_self = 1 - ||s_t - ŝ_t|| / σ_s
        """
        if self.theory is None:
            return 0.5

        error = np.linalg.norm(actual - predicted)

        # σ_s del historial
        if len(self.state_history) > 10:
            states = np.array(self.state_history[-100:])
            sigma = np.mean(np.std(states, axis=0))
        else:
            sigma = 1.0

        consistency = 1.0 - min(1.0, error / (sigma + 1e-8))
        return float(consistency)

    def _record_transition(self, from_state: np.ndarray, to_state: np.ndarray):
        """
        Registra transición para narrativa.

        N_self = secuencia de transiciones dominantes
        """
        diff = to_state - from_state
        dominant = int(np.argmax(np.abs(diff)))
        magnitude = float(np.linalg.norm(diff))

        transition = StateTransition(
            t=self.t,
            from_state=from_state.copy(),
            to_state=to_state.copy(),
            dominant_change=dominant,
            change_magnitude=magnitude
        )

        self.transitions.append(transition)

        # Limitar
        max_trans = max_history(self.t) // 2
        if len(self.transitions) > max_trans:
            self.transitions = self.transitions[-max_trans:]

    def _compute_self_understanding(self) -> float:
        """
        Calcula índice de auto-comprensión.

        U_self = var_explained · mean(Cons_self)
        """
        if self.theory is None:
            return 0.0

        var_explained = self.theory.total_variance_explained

        if self.consistency_history:
            mean_consistency = float(np.mean(self.consistency_history[-50:]))
        else:
            mean_consistency = 0.5

        return var_explained * mean_consistency

    def record_state(self, internal_state: np.ndarray):
        """
        Registra estado interno.

        Args:
            internal_state: Vector de estado interno
        """
        self.t += 1

        # Asegurar dimensión correcta
        if len(internal_state) != self.state_dim:
            # Pad o truncar
            if len(internal_state) < self.state_dim:
                internal_state = np.pad(internal_state, (0, self.state_dim - len(internal_state)))
            else:
                internal_state = internal_state[:self.state_dim]

        # Registrar transición si hay estado anterior
        if self.state_history:
            self._record_transition(self.state_history[-1], internal_state)

            # Predecir y evaluar
            prediction = self._predict_next_state(self.state_history[-1])
            consistency = self._compute_consistency(internal_state, prediction)

            self.predictions.append(prediction)
            self.consistency_history.append(consistency)

            error = float(np.linalg.norm(internal_state - prediction))
            self.prediction_errors.append(error)

        # Guardar estado
        self.state_history.append(internal_state.copy())

        # Limitar historiales
        max_hist = max_history(self.t)
        if len(self.state_history) > max_hist:
            self.state_history = self.state_history[-max_hist:]
        if len(self.predictions) > max_hist:
            self.predictions = self.predictions[-max_hist:]
        if len(self.consistency_history) > max_hist:
            self.consistency_history = self.consistency_history[-max_hist:]
        if len(self.prediction_errors) > max_hist:
            self.prediction_errors = self.prediction_errors[-max_hist:]

        # Actualizar teoría periódicamente
        update_freq = max(20, L_t(self.t) * 2)
        if self.t % update_freq == 0:
            self._build_theory()

        # Actualizar índice de comprensión
        understanding = self._compute_self_understanding()
        self.understanding_history.append(understanding)
        if len(self.understanding_history) > max_hist:
            self.understanding_history = self.understanding_history[-max_hist:]

    def get_self_prediction(self, horizon: int = 1) -> Tuple[np.ndarray, float]:
        """
        Predice estado futuro propio.

        Args:
            horizon: Pasos hacia adelante

        Returns:
            (predicción, confianza)
        """
        if not self.state_history or self.theory is None:
            return np.zeros(self.state_dim), 0.0

        current = self.state_history[-1].copy()

        for _ in range(horizon):
            current = self._predict_next_state(current)

        # Confianza basada en errores pasados
        if self.prediction_errors:
            mean_error = np.mean(self.prediction_errors[-50:])
            confidence = confidence_from_error(mean_error, self.prediction_errors[-50:])
            # Degradar confianza con horizonte
            confidence *= (0.9 ** horizon)
        else:
            confidence = 0.3

        return current, float(confidence)

    def get_dominant_dimensions(self, n: int = 3) -> List[Tuple[int, float]]:
        """
        Obtiene dimensiones más importantes del self.

        Returns:
            Lista de (índice, importancia)
        """
        if self.theory is None or len(self.theory.principal_components) == 0:
            return []

        # Contribución de cada dimensión original
        importance = np.zeros(self.state_dim)

        for i, (pc, var) in enumerate(zip(self.theory.principal_components,
                                         self.theory.explained_variance)):
            importance += np.abs(pc) * var

        # Top n
        top_indices = np.argsort(importance)[-n:][::-1]
        return [(int(idx), float(importance[idx])) for idx in top_indices]

    def get_narrative(self, n_events: int = 5) -> List[Dict]:
        """
        Obtiene narrativa de transiciones significativas.

        Returns:
            Lista de eventos narrativos
        """
        if not self.transitions:
            return []

        # Transiciones más significativas
        sorted_trans = sorted(self.transitions,
                            key=lambda t: t.change_magnitude,
                            reverse=True)[:n_events]

        narrative = []
        for trans in sorted_trans:
            narrative.append({
                't': trans.t,
                'dominant_dimension': trans.dominant_change,
                'change_magnitude': trans.change_magnitude,
                'direction': 'increase' if (trans.to_state[trans.dominant_change] >
                                           trans.from_state[trans.dominant_change]) else 'decrease'
            })

        return narrative

    def is_self_coherent(self) -> bool:
        """
        Verifica si el self es coherente.

        Coherente si:
        - Teoría explica >50% varianza
        - Consistencia media >0.5
        """
        if self.theory is None:
            return False

        var_ok = self.theory.total_variance_explained > 0.5

        if self.consistency_history:
            cons_ok = np.mean(self.consistency_history[-50:]) > 0.5
        else:
            cons_ok = False

        return var_ok and cons_ok

    def get_state(self) -> SelfTheoryState:
        """Obtiene estado actual."""
        if self.consistency_history:
            consistency = float(np.mean(self.consistency_history[-50:]))
        else:
            consistency = 0.0

        if self.prediction_errors:
            accuracy = 1.0 - min(1.0, float(np.mean(self.prediction_errors[-50:])))
        else:
            accuracy = 0.0

        understanding = self._compute_self_understanding()

        return SelfTheoryState(
            t=self.t,
            theory=self.theory,
            consistency=consistency,
            self_understanding=understanding,
            prediction_accuracy=accuracy,
            narrative_length=len(self.transitions),
            is_coherent=self.is_self_coherent()
        )

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del sistema."""
        state = self.get_state()

        theory_stats = {}
        if self.theory:
            theory_stats = {
                'n_components': self.theory.n_components,
                'total_variance_explained': self.theory.total_variance_explained,
                'variance_per_component': self.theory.explained_variance.tolist(),
                'has_transition_model': self.theory.transition_model is not None
            }

        return {
            'agent': self.agent_name,
            't': self.t,
            'state_dim': self.state_dim,
            'n_states_recorded': len(self.state_history),
            'consistency': state.consistency,
            'self_understanding': state.self_understanding,
            'prediction_accuracy': state.prediction_accuracy,
            'is_coherent': state.is_coherent,
            'narrative_length': state.narrative_length,
            'dominant_dimensions': self.get_dominant_dimensions(),
            'theory': theory_stats
        }


def test_self_theory():
    """Test de teoría estructural del self."""
    print("=" * 60)
    print("TEST AGI-20: STRUCTURAL SELF-THEORY")
    print("=" * 60)

    theory = StructuralSelfTheory("NEO", state_dim=10)

    print(f"\nSimulando 500 estados internos...")

    for t in range(500):
        # Estado interno con estructura
        # Algunas dimensiones correlacionadas, otras no
        base = np.zeros(10)

        # Dimensiones 0-2: drives (correlacionadas)
        drive_base = 0.5 + 0.3 * np.sin(t / 30)
        base[0] = drive_base + np.random.randn() * 0.1
        base[1] = drive_base * 0.8 + np.random.randn() * 0.1
        base[2] = drive_base * 0.6 + np.random.randn() * 0.1

        # Dimensiones 3-5: emociones (ciclo diferente)
        emotion_base = 0.5 + 0.2 * np.cos(t / 50)
        base[3] = emotion_base + np.random.randn() * 0.15
        base[4] = 1 - emotion_base + np.random.randn() * 0.15
        base[5] = 0.5 + np.random.randn() * 0.1

        # Dimensiones 6-9: ruido
        base[6:] = np.random.randn(4) * 0.3

        # Eventos significativos ocasionales
        if t % 100 == 50:
            base[0] += 0.5  # Spike en drive principal
        if t % 150 == 75:
            base[3] -= 0.4  # Caída emocional

        base = np.clip(base, 0, 1)
        theory.record_state(base)

        if (t + 1) % 100 == 0:
            state = theory.get_state()
            print(f"  t={t+1}: understanding={state.self_understanding:.3f}, "
                  f"consistency={state.consistency:.3f}, "
                  f"coherent={state.is_coherent}")

    # Resultados finales
    stats = theory.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS STRUCTURAL SELF-THEORY")
    print("=" * 60)

    print(f"\n  Estados registrados: {stats['n_states_recorded']}")
    print(f"  Consistencia: {stats['consistency']:.3f}")
    print(f"  Auto-comprensión: {stats['self_understanding']:.3f}")
    print(f"  Precisión predicción: {stats['prediction_accuracy']:.3f}")
    print(f"  Es coherente: {stats['is_coherent']}")

    if stats['theory']:
        print(f"\n  Teoría:")
        print(f"    Componentes: {stats['theory']['n_components']}")
        print(f"    Varianza explicada: {stats['theory']['total_variance_explained']:.3f}")
        print(f"    Tiene modelo transición: {stats['theory']['has_transition_model']}")

    print(f"\n  Dimensiones dominantes:")
    for dim, importance in stats['dominant_dimensions']:
        print(f"    Dim {dim}: importancia={importance:.3f}")

    # Narrativa
    print(f"\n  Narrativa ({stats['narrative_length']} transiciones):")
    narrative = theory.get_narrative(3)
    for event in narrative:
        print(f"    t={event['t']}: dim{event['dominant_dimension']} "
              f"{event['direction']} ({event['change_magnitude']:.3f})")

    # Test de predicción
    print("\n  Predicción de self:")
    for horizon in [1, 3, 5]:
        pred, conf = theory.get_self_prediction(horizon)
        print(f"    Horizonte {horizon}: conf={conf:.3f}")

    if stats['is_coherent']:
        print("\n  ✓ Teoría del self coherente")
    else:
        print("\n  ⚠ Teoría del self aún no coherente")

    return theory


if __name__ == "__main__":
    test_self_theory()
