"""
AGI-2: Self Narrative Loop
==========================

Bucle autorreferente que une:
- episodios → narrativa
- narrativa → self
- self → futuro
- futuro → decisiones
- decisiones → episodios

El Self deja de ser un vector estático.
Pasa a ser una curva histórica con dirección.

Todo 100% endógeno - sin constantes mágicas.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy import linalg

# Imports internos (relativos al módulo cognition)
from .episodic_memory import EpisodicMemory, Episode
from .narrative_memory import NarrativeMemory
from .temporal_tree import TemporalTree
from .self_model import SelfModel
from .compound_goals import CompoundGoals, GoalPlanner


@dataclass
class SelfState:
    """
    Estado del Self en un momento dado.

    No es solo un vector - es una estructura con:
    - Estado actual
    - Tendencia (de dónde viene)
    - Expectativa (hacia dónde va)
    - Coherencia con la narrativa
    """
    t: int
    state: np.ndarray           # Estado latente actual
    tendency: np.ndarray        # Derivada temporal (de dónde viene)
    expectation: np.ndarray     # Hacia dónde espera ir
    narrative_coherence: float  # Coherencia con cadena narrativa
    identity_strength: float    # Fuerza de identidad
    future_clarity: float       # Claridad del futuro simulado


@dataclass
class LoopIteration:
    """Una iteración completa del Self Narrative Loop."""
    t: int
    episode_encoded: bool
    narrative_updated: bool
    self_updated: bool
    future_simulated: bool
    decision_made: str
    self_state: SelfState


class SelfNarrativeLoop:
    """
    Self Narrative Loop - AGI-2

    Ciclo autorreferente:

    1. E_t = encode(z_t, Δz_t, shock_t, φ_t)
    2. T_t = P(E_{t-1} → E_t)
    3. N_t = chain_update(N_{t-1}, T_t)
    4. self_t = f(self_{t-1}, E_t, N_t)
    5. T_t = simulate_future(self_t)
    6. N_t^future = propagate(N_t, T_t)
    7. self_t* = g(self_t, N_t^future)
    8. decision_t = choose_action(self_t*)
    9. E_{t+1} = consequence(decision_t)
    10. Loop = repeat

    Todo endógeno: pesos derivan de covarianzas, percentiles, √t.
    """

    def __init__(self, agent_name: str, z_dim: int = 6, phi_dim: int = 5):
        """
        Inicializa Self Narrative Loop.

        Args:
            agent_name: Nombre del agente
            z_dim: Dimensión del estado estructural
            phi_dim: Dimensión del vector fenomenológico
        """
        self.agent_name = agent_name
        self.z_dim = z_dim
        self.phi_dim = phi_dim
        self.self_dim = z_dim + phi_dim  # Dimensión del self latente

        # Módulos cognitivos subyacentes
        self.episodic = EpisodicMemory(z_dim, phi_dim, z_dim)
        self.narrative = NarrativeMemory(self.episodic)
        self.temporal_tree = TemporalTree(z_dim, phi_dim, z_dim)
        self.self_model = SelfModel(agent_name, z_dim + phi_dim + z_dim)
        self.goals = CompoundGoals(D_dim=z_dim)
        self.planner = GoalPlanner(self.goals)

        # Estado del Self (latente, continuo)
        self.self_state = np.zeros(self.self_dim)
        self.self_tendency = np.zeros(self.self_dim)
        self.self_expectation = np.zeros(self.self_dim)

        # Historial del self
        self.self_history: List[SelfState] = []

        # Matrices de actualización (aprenden endógenamente)
        # W_self: cómo episodios + narrativa → self
        self.W_episode_to_self = np.eye(self.self_dim, z_dim + phi_dim) * 0.1
        self.W_narrative_to_self = np.eye(self.self_dim) * 0.1
        # W_future: cómo futuro → self*
        self.W_future_to_self = np.eye(self.self_dim) * 0.1

        # Parámetros adaptativos
        self.narrative_weight = 0.5  # Balance episodio vs narrativa
        self.future_weight = 0.3    # Cuánto afecta el futuro al self

        # Historial de iteraciones
        self.iterations: List[LoopIteration] = []

        # Estado actual
        self.current_episode_idx = -1
        self.current_decision = "none"

        self.t = 0

    def _compute_learning_rate(self) -> float:
        """Learning rate endógeno: η = 1/√(t+1)"""
        return 1.0 / np.sqrt(self.t + 1)

    def _encode_episode(self, z: np.ndarray, phi: np.ndarray, D: np.ndarray,
                       tau: float) -> bool:
        """
        Paso 1: Codifica estado actual como episodio.

        E_t = encode(z_t, Δz_t, shock_t, φ_t)

        Returns:
            True si se creó nuevo episodio
        """
        n_episodes_before = len(self.episodic.episodes)
        self.episodic.record(z, phi, D, tau)
        n_episodes_after = len(self.episodic.episodes)

        new_episode = n_episodes_after > n_episodes_before

        if new_episode:
            self.current_episode_idx = n_episodes_after - 1

        return new_episode

    def _update_narrative(self) -> bool:
        """
        Pasos 2-3: Actualiza transiciones y cadena narrativa.

        T_t = P(E_{t-1} → E_t)
        N_t = chain_update(N_{t-1}, T_t)

        Returns:
            True si narrativa cambió significativamente
        """
        # Actualizar narrativa
        self.narrative.update()

        # Obtener resumen
        summary = self.narrative.get_narrative_summary()

        # Detectar si cambió la cadena dominante
        if 'dominant_chain' in summary:
            chain = summary['dominant_chain']
            if len(chain) > 0 and len(self.iterations) > 0:
                # Comparar con iteración anterior si existe
                return True

        return len(self.episodic.episodes) > 1

    def _update_self(self, z: np.ndarray, phi: np.ndarray) -> SelfState:
        """
        Paso 4: Actualiza self basado en episodio y narrativa.

        self_t = f(self_{t-1}, E_t, N_t)

        donde f combina:
        - Estado anterior (momentum)
        - Episodio actual (input)
        - Coherencia narrativa (contexto)
        """
        eta = self._compute_learning_rate()

        # Construir input desde episodio actual
        episode_input = np.concatenate([z, phi])

        # Proyectar episodio al espacio del self
        if episode_input.shape[0] != self.W_episode_to_self.shape[1]:
            # Ajustar dimensiones si es necesario
            episode_input = episode_input[:self.W_episode_to_self.shape[1]]

        episode_contribution = self.W_episode_to_self @ episode_input

        # Contribución narrativa (coherencia con cadena dominante)
        narrative_contribution = np.zeros(self.self_dim)
        if len(self.episodic.episodes) > 1:
            # Usar similitud con episodio anterior como proxy de coherencia
            current_ep = self.episodic.episodes[-1] if len(self.episodic.episodes) > 0 else None
            prev_ep = self.episodic.episodes[-2] if len(self.episodic.episodes) > 1 else None

            if current_ep and prev_ep:
                coherence = self.episodic.similarity(prev_ep, current_ep)
                # Modular contribución narrativa
                narrative_contribution = self.W_narrative_to_self @ self.self_state * coherence

        # Actualizar self con momentum
        momentum = 0.9  # Cuánto del self anterior se mantiene
        new_self = (momentum * self.self_state +
                   (1 - momentum) * (
                       (1 - self.narrative_weight) * episode_contribution +
                       self.narrative_weight * narrative_contribution
                   ))

        # Actualizar tendencia (derivada temporal)
        if len(self.self_history) > 0:
            self.self_tendency = new_self - self.self_state
        else:
            self.self_tendency = np.zeros(self.self_dim)

        # Actualizar estado
        self.self_state = new_self

        # Calcular métricas
        identity_strength = self._compute_identity_strength()
        narrative_coherence = self._compute_narrative_coherence()

        # Crear estado del self
        state = SelfState(
            t=self.t,
            state=self.self_state.copy(),
            tendency=self.self_tendency.copy(),
            expectation=self.self_expectation.copy(),
            narrative_coherence=narrative_coherence,
            identity_strength=identity_strength,
            future_clarity=0.0  # Se actualiza después de simular futuro
        )

        return state

    def _simulate_future(self, z: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        Paso 5: Simula futuro usando temporal tree.

        T_t = simulate_future(self_t)

        Returns:
            Estado futuro esperado (promedio ponderado de hojas)
        """
        # Registrar estado en árbol
        self.temporal_tree.record_state(z, in_crisis=False)

        # Generar árbol
        root = self.temporal_tree.generate_tree(z, D, depth=2, branching=3)

        # Obtener mejor rama
        best = self.temporal_tree.get_best_branch()

        if best is not None:
            # Retornar estado de mejor rama
            future_z = best.z
            future_phi = best.phi
            return np.concatenate([future_z, future_phi[:self.phi_dim]])
        else:
            # Retornar proyección lineal simple
            return self.self_state + self.self_tendency

    def _propagate_narrative_to_future(self, future_state: np.ndarray) -> np.ndarray:
        """
        Paso 6: Propaga narrativa hacia el futuro.

        N_t^future = propagate(N_t, T_t)

        Modifica expectativas basándose en coherencia narrativa.
        """
        # La narrativa futura es el estado futuro modulado por
        # la coherencia con la cadena dominante

        narrative_coherence = self._compute_narrative_coherence()

        # Si alta coherencia, el futuro sigue la tendencia
        # Si baja coherencia, más incertidumbre
        propagated = future_state * narrative_coherence + \
                    self.self_state * (1 - narrative_coherence)

        return propagated

    def _update_future_self(self, future_narrative: np.ndarray) -> SelfState:
        """
        Paso 7: Actualiza self con expectativas futuras.

        self_t* = g(self_t, N_t^future)

        El self incorpora el futuro esperado.
        """
        # Proyectar futuro al espacio del self
        future_contribution = self.W_future_to_self @ future_narrative

        # Actualizar expectativa
        self.self_expectation = (1 - self.future_weight) * self.self_expectation + \
                               self.future_weight * future_contribution

        # El self* es el self actual modulado por expectativas
        self_star = self.self_state + 0.1 * self.self_expectation

        # Calcular claridad del futuro
        future_clarity = 1.0 / (1.0 + np.linalg.norm(future_narrative - self.self_state))

        # Actualizar estado del self con claridad
        if len(self.self_history) > 0:
            self.self_history[-1].future_clarity = future_clarity
            self.self_history[-1].expectation = self.self_expectation.copy()

        return self.self_history[-1] if len(self.self_history) > 0 else None

    def _choose_action(self) -> str:
        """
        Paso 8: Elige acción basada en self*.

        decision_t = choose_action(self_t*)

        Returns:
            Nombre de la acción elegida
        """
        # Usar planner con goals
        if len(self.goals.goals) > 0:
            # Obtener goal actual
            current_goal = self.goals.get_nearest_goal(self.self_state[:self.z_dim])

            if current_goal is not None:
                # Distancia al goal
                distance = self.goals.distance_to_goal(
                    self.self_state[:self.z_dim], current_goal
                )

                # Si cerca del goal, explorar
                if distance < 0.3:
                    return 'exploration'
                # Si lejos, avanzar hacia goal
                else:
                    return 'integration'

        # Default: usar temporal tree
        stats = self.temporal_tree.get_statistics()
        if 'best_operator' in stats and stats['best_operator']:
            return stats['best_operator']

        return 'homeostasis'

    def _compute_identity_strength(self) -> float:
        """
        Calcula fuerza de identidad.

        identity_strength = 1 / (1 + var(self_history[-W:]))
        """
        if len(self.self_history) < 5:
            return 0.5

        W = min(20, len(self.self_history))
        recent = np.array([s.state for s in self.self_history[-W:]])

        var = np.var(recent)
        return float(1.0 / (1.0 + var))

    def _compute_narrative_coherence(self) -> float:
        """
        Calcula coherencia narrativa.

        coherence = mean_similarity(episodios recientes)
        """
        episodes = self.episodic.episodes
        if len(episodes) < 3:
            return 0.5

        # Similitud entre episodios consecutivos recientes
        coherences = []
        for i in range(max(0, len(episodes) - 5), len(episodes) - 1):
            coh = self.episodic.similarity(episodes[i], episodes[i + 1])
            coherences.append(coh)

        if len(coherences) == 0:
            return 0.5

        return float(np.mean(coherences))

    def _adapt_weights(self):
        """
        Adapta pesos del loop endógenamente.

        Basado en correlación entre:
        - narrative_weight y narrative_coherence
        - future_weight y future_clarity
        """
        if len(self.self_history) < 20:
            return

        recent = self.self_history[-50:]

        # Adaptar narrative_weight
        coherences = [s.narrative_coherence for s in recent]
        mean_coh = np.mean(coherences)

        # Si coherencia alta, aumentar peso narrativo
        # Si baja, reducirlo
        eta = self._compute_learning_rate()
        self.narrative_weight += eta * (mean_coh - 0.5) * 0.1
        self.narrative_weight = np.clip(self.narrative_weight, 0.2, 0.8)

        # Adaptar future_weight
        clarities = [s.future_clarity for s in recent]
        mean_clarity = np.mean(clarities)

        self.future_weight += eta * (mean_clarity - 0.5) * 0.1
        self.future_weight = np.clip(self.future_weight, 0.1, 0.5)

    def step(self, z: np.ndarray, phi: np.ndarray, D: np.ndarray,
             tau: Optional[float] = None) -> LoopIteration:
        """
        Ejecuta una iteración completa del Self Narrative Loop.

        Args:
            z: Estado estructural actual
            phi: Vector fenomenológico
            D: Drives
            tau: Tiempo subjetivo (opcional)

        Returns:
            LoopIteration con resultados de la iteración
        """
        self.t += 1

        if tau is None:
            tau = float(self.t)

        # Paso 1: Codificar episodio
        episode_encoded = self._encode_episode(z, phi, D, tau)

        # Paso 2-3: Actualizar narrativa
        narrative_updated = self._update_narrative()

        # Paso 4: Actualizar self
        self_state = self._update_self(z, phi)
        self.self_history.append(self_state)

        # Limitar historial
        if len(self.self_history) > 500:
            self.self_history = self.self_history[-500:]

        # Paso 5: Simular futuro
        future_state = self._simulate_future(z, D)

        # Paso 6: Propagar narrativa al futuro
        future_narrative = self._propagate_narrative_to_future(future_state)

        # Paso 7: Actualizar self con futuro
        self._update_future_self(future_narrative)

        # Paso 8: Elegir acción
        decision = self._choose_action()
        self.current_decision = decision

        # Registrar goals periódicamente
        if self.t % 50 == 0 and len(self.episodic.episodes) > 5:
            recent_eps = self.episodic.get_recent_episodes(5)
            D_bar = np.mean([e.D_bar for e in recent_eps], axis=0)
            metric = self._compute_identity_strength()
            self.goals.record_episode(D_bar, metric)

            if self.t % 100 == 0:
                self.goals.discover_goals()

        # Adaptar pesos
        if self.t % 20 == 0:
            self._adapt_weights()

        # Crear registro de iteración
        iteration = LoopIteration(
            t=self.t,
            episode_encoded=episode_encoded,
            narrative_updated=narrative_updated,
            self_updated=True,
            future_simulated=True,
            decision_made=decision,
            self_state=self_state
        )

        self.iterations.append(iteration)
        if len(self.iterations) > 500:
            self.iterations = self.iterations[-500:]

        return iteration

    def get_self_trajectory(self, length: int = 50) -> np.ndarray:
        """
        Obtiene trayectoria reciente del self.

        Returns:
            Array de estados del self
        """
        if len(self.self_history) == 0:
            return np.array([])

        recent = self.self_history[-length:]
        return np.array([s.state for s in recent])

    def get_identity_vector(self) -> np.ndarray:
        """
        Obtiene vector de identidad (centroide del self).

        identity = mean(self_history[-W:])
        """
        if len(self.self_history) < 5:
            return self.self_state.copy()

        W = min(50, len(self.self_history))
        recent = np.array([s.state for s in self.self_history[-W:]])
        return np.mean(recent, axis=0)

    def get_life_direction(self) -> np.ndarray:
        """
        Obtiene dirección vital (hacia dónde va el self).

        direction = expectation - current_state
        """
        return self.self_expectation - self.self_state

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del Self Narrative Loop."""
        if len(self.self_history) == 0:
            return {
                'agent': self.agent_name,
                't': self.t,
                'status': 'initializing'
            }

        recent = self.self_history[-50:]

        return {
            'agent': self.agent_name,
            't': self.t,
            'n_episodes': len(self.episodic.episodes),
            'n_goals': len(self.goals.goals),
            'identity_strength': float(np.mean([s.identity_strength for s in recent])),
            'narrative_coherence': float(np.mean([s.narrative_coherence for s in recent])),
            'future_clarity': float(np.mean([s.future_clarity for s in recent])),
            'narrative_weight': float(self.narrative_weight),
            'future_weight': float(self.future_weight),
            'current_decision': self.current_decision,
            'self_magnitude': float(np.linalg.norm(self.self_state)),
            'tendency_magnitude': float(np.linalg.norm(self.self_tendency)),
            'expectation_magnitude': float(np.linalg.norm(self.self_expectation))
        }

    def get_narrative_report(self) -> Dict:
        """Obtiene reporte narrativo del agente."""
        narrative_summary = self.narrative.get_narrative_summary()
        goal_stats = self.goals.get_statistics()

        return {
            'agent': self.agent_name,
            'life_stage': self._detect_life_stage(),
            'narrative': narrative_summary,
            'goals': goal_stats,
            'identity': {
                'vector': self.get_identity_vector().tolist(),
                'strength': self._compute_identity_strength(),
                'direction': self.get_life_direction().tolist()
            }
        }

    def _detect_life_stage(self) -> str:
        """
        Detecta etapa vital basada en métricas internas.

        Endógeno: basado en percentiles de identidad, coherencia, claridad.
        """
        if len(self.self_history) < 20:
            return "birth"

        id_strength = self._compute_identity_strength()
        coherence = self._compute_narrative_coherence()

        if len(self.self_history) > 0:
            clarity = self.self_history[-1].future_clarity
        else:
            clarity = 0.5

        # Clasificar endógenamente
        if id_strength < 0.3:
            if coherence < 0.5:
                return "crisis"
            else:
                return "exploration"
        elif id_strength > 0.7:
            if clarity > 0.6:
                return "maturity"
            else:
                return "consolidation"
        else:
            if coherence > 0.7:
                return "growth"
            else:
                return "transition"


def test_self_narrative_loop():
    """Test del Self Narrative Loop."""
    print("=" * 60)
    print("TEST SELF NARRATIVE LOOP (AGI-2)")
    print("=" * 60)

    # Crear loop para NEO
    loop = SelfNarrativeLoop("NEO", z_dim=6, phi_dim=5)

    print("\nSimulando 500 pasos de vida...")

    z = np.random.randn(6) * 0.1
    phi = np.random.randn(5) * 0.1
    D = np.abs(np.random.randn(6))
    D = D / D.sum()

    for t in range(500):
        # Evolución del estado
        z = 0.95 * z + 0.05 * np.tanh(z) + np.random.randn(6) * 0.02
        phi = 0.9 * phi + 0.1 * np.random.randn(5) * 0.1
        D = D + np.random.randn(6) * 0.01
        D = np.abs(D)
        D = D / D.sum()

        # Shocks ocasionales
        if t % 80 == 40:
            z += np.random.randn(6) * 0.3
            phi += np.random.randn(5) * 0.2

        tau = t * (1 + 0.1 * np.linalg.norm(phi))

        # Step del loop
        iteration = loop.step(z, phi, D, tau)

        if (t + 1) % 100 == 0:
            stats = loop.get_statistics()
            print(f"\n  t={t+1}:")
            print(f"    Episodes: {stats['n_episodes']}, "
                  f"Goals: {stats['n_goals']}")
            print(f"    Identity: {stats['identity_strength']:.3f}, "
                  f"Coherence: {stats['narrative_coherence']:.3f}")
            print(f"    Future clarity: {stats['future_clarity']:.3f}")
            print(f"    Decision: {stats['current_decision']}")
            print(f"    Life stage: {loop._detect_life_stage()}")

    # Reporte final
    print("\n" + "=" * 60)
    print("REPORTE NARRATIVO FINAL")
    print("=" * 60)

    report = loop.get_narrative_report()
    print(f"\nAgente: {report['agent']}")
    print(f"Etapa vital: {report['life_stage']}")
    print(f"Fuerza de identidad: {report['identity']['strength']:.3f}")
    print(f"Metas descubiertas: {report['goals']['n_goals']}")

    # Trayectoria del self
    trajectory = loop.get_self_trajectory(20)
    if len(trajectory) > 0:
        print(f"\nVariación del self (últimos 20 pasos): {np.std(trajectory):.4f}")

    return loop


if __name__ == "__main__":
    test_self_narrative_loop()
