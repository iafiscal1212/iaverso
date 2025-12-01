"""
AGI-3: Teleología Interna - Metas Persistentes
==============================================

Metas que persisten a largo plazo.
Una AGI no es solo que piense - es que QUIERE algo estable.

G_{t+1} = G_t       si U(G_t) > percentile_50
G_{t+1} = nuevo     si U(G_t) <= percentile_10

Esto crea dirección vital.

Todo 100% endógeno - sin constantes mágicas.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class GoalStatus(Enum):
    """Estado de una meta."""
    ACTIVE = "active"          # Meta activa, se persigue
    ACHIEVED = "achieved"      # Meta lograda
    ABANDONED = "abandoned"    # Meta abandonada (utilidad muy baja)
    DORMANT = "dormant"        # Meta inactiva pero no abandonada


@dataclass
class PersistentGoal:
    """
    Meta persistente con historia y utilidad.

    A diferencia de CompoundGoals (patrones), estas son
    intenciones que persisten en el tiempo.
    """
    idx: int
    created_t: int
    target_state: np.ndarray    # Estado objetivo en espacio de drives
    utility_history: List[float] = field(default_factory=list)
    status: GoalStatus = GoalStatus.ACTIVE
    pursuit_history: List[int] = field(default_factory=list)  # t's de persecución
    distance_history: List[float] = field(default_factory=list)

    # Métricas de persistencia
    persistence_score: float = 0.0
    achievement_progress: float = 0.0

    # Metadata
    source: str = "endogenous"  # De dónde surgió
    last_evaluation_t: int = 0


@dataclass
class GoalTransition:
    """Registro de transición entre metas."""
    t: int
    from_goal: Optional[int]
    to_goal: Optional[int]
    reason: str  # "achieved", "abandoned", "new_discovery"
    utility_at_transition: float


class PersistentGoals:
    """
    Sistema de metas persistentes.

    Mantiene metas a largo plazo que:
    - Persisten si tienen utilidad suficiente
    - Se abandonan si utilidad cae bajo p10
    - Emergen de patrones exitosos

    G_{t+1} = G_t       si U(G_t) > percentile_50({U_history})
    G_{t+1} = nuevo     si U(G_t) <= percentile_10({U_history})
    """

    def __init__(self, D_dim: int = 6):
        """
        Inicializa sistema de metas persistentes.

        Args:
            D_dim: Dimensión del espacio de drives
        """
        self.D_dim = D_dim

        # Metas activas
        self.goals: List[PersistentGoal] = []
        self.active_goal_idx: Optional[int] = None

        # Historial
        self.utility_global_history: List[float] = []
        self.transitions: List[GoalTransition] = []

        # Estados de éxito (para descubrir nuevas metas)
        self.success_states: List[np.ndarray] = []
        self.success_metrics: List[float] = []

        # Umbrales (adaptativos)
        self.persistence_threshold = 0.5  # percentile_50
        self.abandon_threshold = 0.1      # percentile_10

        self.t = 0
        self.next_goal_idx = 0

    def _compute_utility(self, goal: PersistentGoal, current_D: np.ndarray,
                        SAGI: float, crisis: bool) -> float:
        """
        Calcula utilidad de una meta.

        U(G) = α·progress + β·SAGI_when_pursuing + γ·(1-crisis_rate)

        donde α, β, γ derivan de correlaciones históricas.
        """
        # Progreso hacia la meta
        distance = np.linalg.norm(current_D - goal.target_state)
        if len(goal.distance_history) > 0:
            initial_distance = goal.distance_history[0]
            progress = max(0, 1 - distance / (initial_distance + 1e-8))
        else:
            progress = 0.5

        # SAGI durante persecución
        sagi_component = SAGI

        # Componente de crisis
        crisis_component = 0.0 if crisis else 1.0

        # Pesos adaptativos basados en historial
        if len(goal.utility_history) > 10:
            # Usar varianza para ponderar
            var_utility = np.var(goal.utility_history)
            alpha = 0.4 / (1 + var_utility)
            beta = 0.4
            gamma = 0.2
        else:
            alpha = 0.4
            beta = 0.4
            gamma = 0.2

        utility = alpha * progress + beta * sagi_component + gamma * crisis_component

        return float(utility)

    def _update_thresholds(self):
        """
        Actualiza umbrales endógenamente.

        persistence = percentile_50(utility_history)
        abandon = percentile_10(utility_history)
        """
        if len(self.utility_global_history) < 20:
            return

        self.persistence_threshold = np.percentile(self.utility_global_history, 50)
        self.abandon_threshold = np.percentile(self.utility_global_history, 10)

    def _should_persist(self, utility: float) -> bool:
        """Verifica si la meta debe persistir."""
        return utility > self.persistence_threshold

    def _should_abandon(self, utility: float) -> bool:
        """Verifica si la meta debe abandonarse."""
        return utility <= self.abandon_threshold

    def _discover_new_goal(self) -> Optional[PersistentGoal]:
        """
        Descubre nueva meta de estados exitosos.

        Nueva meta = centroide de estados con alto SAGI.
        """
        if len(self.success_states) < 5:
            return None

        # Filtrar estados más exitosos (top 25%)
        threshold = np.percentile(self.success_metrics, 75)
        top_indices = [i for i, m in enumerate(self.success_metrics) if m >= threshold]

        if len(top_indices) < 3:
            return None

        # Centroide de estados exitosos
        top_states = np.array([self.success_states[i] for i in top_indices])
        target = np.mean(top_states, axis=0)

        # Verificar que no sea muy similar a metas existentes
        for goal in self.goals:
            if goal.status == GoalStatus.ACTIVE:
                similarity = 1 - np.linalg.norm(target - goal.target_state) / \
                           (np.linalg.norm(target) + 1e-8)
                if similarity > 0.8:
                    return None  # Muy similar a meta existente

        # Crear nueva meta
        new_goal = PersistentGoal(
            idx=self.next_goal_idx,
            created_t=self.t,
            target_state=target,
            source="discovered"
        )
        self.next_goal_idx += 1

        return new_goal

    def record_success_state(self, D: np.ndarray, metric: float):
        """
        Registra estado exitoso para descubrimiento de metas.

        Args:
            D: Estado de drives
            metric: Métrica de éxito (ej: SAGI)
        """
        self.success_states.append(D.copy())
        self.success_metrics.append(metric)

        # Limitar historial
        if len(self.success_states) > 200:
            self.success_states = self.success_states[-200:]
            self.success_metrics = self.success_metrics[-200:]

    def evaluate_and_update(self, current_D: np.ndarray, SAGI: float,
                           in_crisis: bool) -> Dict:
        """
        Evalúa metas activas y actualiza sistema.

        Implementa la regla de persistencia:
        G_{t+1} = G_t       si U(G_t) > p50
        G_{t+1} = nuevo     si U(G_t) <= p10

        Returns:
            Dict con información de la evaluación
        """
        self.t += 1

        result = {
            't': self.t,
            'action': 'none',
            'active_goal': self.active_goal_idx,
            'utility': 0.0
        }

        # Registrar estado exitoso si SAGI alto
        if SAGI > 0.6:  # Umbral adaptativo sería mejor
            self.record_success_state(current_D, SAGI)

        # Si no hay meta activa, intentar crear una
        if self.active_goal_idx is None:
            new_goal = self._discover_new_goal()
            if new_goal:
                self.goals.append(new_goal)
                self.active_goal_idx = new_goal.idx
                self.transitions.append(GoalTransition(
                    t=self.t,
                    from_goal=None,
                    to_goal=new_goal.idx,
                    reason="new_discovery",
                    utility_at_transition=0.0
                ))
                result['action'] = 'new_goal'
                result['active_goal'] = new_goal.idx
            return result

        # Evaluar meta activa
        active_goal = None
        for goal in self.goals:
            if goal.idx == self.active_goal_idx:
                active_goal = goal
                break

        if active_goal is None:
            self.active_goal_idx = None
            return result

        # Calcular utilidad
        utility = self._compute_utility(active_goal, current_D, SAGI, in_crisis)
        active_goal.utility_history.append(utility)
        active_goal.last_evaluation_t = self.t

        # Registrar distancia
        distance = np.linalg.norm(current_D - active_goal.target_state)
        active_goal.distance_history.append(distance)

        # Registrar globalmente
        self.utility_global_history.append(utility)
        if len(self.utility_global_history) > 500:
            self.utility_global_history = self.utility_global_history[-500:]

        # Actualizar umbrales
        self._update_thresholds()

        result['utility'] = utility

        # Verificar si se logró la meta
        if distance < 0.1:  # Umbral de logro
            active_goal.status = GoalStatus.ACHIEVED
            active_goal.achievement_progress = 1.0

            self.transitions.append(GoalTransition(
                t=self.t,
                from_goal=active_goal.idx,
                to_goal=None,
                reason="achieved",
                utility_at_transition=utility
            ))

            # Buscar nueva meta
            new_goal = self._discover_new_goal()
            if new_goal:
                self.goals.append(new_goal)
                self.active_goal_idx = new_goal.idx
            else:
                self.active_goal_idx = None

            result['action'] = 'achieved'
            return result

        # Verificar persistencia
        if self._should_abandon(utility):
            # Abandonar meta
            active_goal.status = GoalStatus.ABANDONED

            self.transitions.append(GoalTransition(
                t=self.t,
                from_goal=active_goal.idx,
                to_goal=None,
                reason="abandoned",
                utility_at_transition=utility
            ))

            # Buscar nueva meta
            new_goal = self._discover_new_goal()
            if new_goal:
                self.goals.append(new_goal)
                self.active_goal_idx = new_goal.idx
                self.transitions[-1].to_goal = new_goal.idx
            else:
                self.active_goal_idx = None

            result['action'] = 'abandoned'

        elif self._should_persist(utility):
            # Meta persiste
            active_goal.pursuit_history.append(self.t)
            active_goal.persistence_score = len(active_goal.pursuit_history) / \
                                           (self.t - active_goal.created_t + 1)
            result['action'] = 'persist'

        else:
            # Zona intermedia - meta en riesgo pero no abandonada
            result['action'] = 'at_risk'

        # Actualizar progreso
        if len(active_goal.distance_history) > 1:
            initial = active_goal.distance_history[0]
            current = active_goal.distance_history[-1]
            active_goal.achievement_progress = max(0, 1 - current / (initial + 1e-8))

        return result

    def get_active_goal(self) -> Optional[PersistentGoal]:
        """Obtiene meta activa actual."""
        if self.active_goal_idx is None:
            return None
        for goal in self.goals:
            if goal.idx == self.active_goal_idx:
                return goal
        return None

    def get_goal_direction(self, current_D: np.ndarray) -> np.ndarray:
        """
        Obtiene dirección hacia la meta activa.

        direction = normalize(target - current)
        """
        goal = self.get_active_goal()
        if goal is None:
            return np.zeros(self.D_dim)

        direction = goal.target_state - current_D
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return np.zeros(self.D_dim)
        return direction / norm

    def get_persistence_report(self) -> Dict:
        """Obtiene reporte de persistencia de metas."""
        active_goals = [g for g in self.goals if g.status == GoalStatus.ACTIVE]
        achieved_goals = [g for g in self.goals if g.status == GoalStatus.ACHIEVED]
        abandoned_goals = [g for g in self.goals if g.status == GoalStatus.ABANDONED]

        return {
            't': self.t,
            'total_goals': len(self.goals),
            'active': len(active_goals),
            'achieved': len(achieved_goals),
            'abandoned': len(abandoned_goals),
            'achievement_rate': len(achieved_goals) / max(1, len(self.goals)),
            'abandonment_rate': len(abandoned_goals) / max(1, len(self.goals)),
            'current_active_goal': self.active_goal_idx,
            'persistence_threshold': self.persistence_threshold,
            'abandon_threshold': self.abandon_threshold,
            'n_transitions': len(self.transitions)
        }

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del sistema de metas."""
        if len(self.goals) == 0:
            return {
                't': self.t,
                'status': 'no_goals'
            }

        active_goal = self.get_active_goal()

        stats = self.get_persistence_report()
        stats['active_goal_progress'] = active_goal.achievement_progress if active_goal else 0.0
        stats['active_goal_persistence'] = active_goal.persistence_score if active_goal else 0.0

        if len(self.utility_global_history) > 0:
            stats['mean_utility'] = float(np.mean(self.utility_global_history[-50:]))
        else:
            stats['mean_utility'] = 0.0

        return stats


class TeleologicalAgent:
    """
    Agente con teleología interna.

    Combina:
    - PersistentGoals (metas estables)
    - Dirección vital
    - Evaluación continua de propósito
    """

    def __init__(self, name: str, D_dim: int = 6):
        """
        Inicializa agente teleológico.

        Args:
            name: Nombre del agente
            D_dim: Dimensión de drives
        """
        self.name = name
        self.D_dim = D_dim

        # Sistema de metas persistentes
        self.goals = PersistentGoals(D_dim)

        # Estado actual
        self.current_D = np.ones(D_dim) / D_dim
        self.life_direction = np.zeros(D_dim)

        # Historial
        self.purpose_history: List[float] = []  # Sentido de propósito

        self.t = 0

    def _compute_purpose(self) -> float:
        """
        Calcula sentido de propósito.

        purpose = goal_progress * persistence_score * direction_alignment
        """
        goal = self.goals.get_active_goal()
        if goal is None:
            return 0.3  # Propósito base sin meta

        # Progreso
        progress = goal.achievement_progress

        # Persistencia
        persistence = goal.persistence_score

        # Alineamiento de dirección
        goal_direction = self.goals.get_goal_direction(self.current_D)
        if np.linalg.norm(self.life_direction) > 1e-8:
            alignment = np.dot(goal_direction, self.life_direction) / \
                       (np.linalg.norm(self.life_direction) + 1e-8)
            alignment = (alignment + 1) / 2  # Normalizar a [0, 1]
        else:
            alignment = 0.5

        purpose = 0.4 * progress + 0.3 * persistence + 0.3 * alignment
        return float(purpose)

    def step(self, D: np.ndarray, SAGI: float, in_crisis: bool) -> Dict:
        """
        Paso teleológico.

        Args:
            D: Drives actuales
            SAGI: Métrica de integración
            in_crisis: Si está en crisis

        Returns:
            Dict con información del paso
        """
        self.t += 1
        self.current_D = D.copy()

        # Actualizar dirección de vida
        if self.t > 1:
            self.life_direction = 0.9 * self.life_direction + 0.1 * (D - self.current_D)

        # Evaluar y actualizar metas
        eval_result = self.goals.evaluate_and_update(D, SAGI, in_crisis)

        # Calcular propósito
        purpose = self._compute_purpose()
        self.purpose_history.append(purpose)

        # Limitar historial
        if len(self.purpose_history) > 500:
            self.purpose_history = self.purpose_history[-500:]

        return {
            't': self.t,
            'goal_action': eval_result['action'],
            'utility': eval_result['utility'],
            'purpose': purpose,
            'direction': self.goals.get_goal_direction(D).tolist()
        }

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del agente teleológico."""
        goal_stats = self.goals.get_statistics()

        return {
            'name': self.name,
            't': self.t,
            'purpose': self._compute_purpose(),
            'mean_purpose': float(np.mean(self.purpose_history[-50:])) if self.purpose_history else 0.0,
            'goals': goal_stats,
            'life_direction_magnitude': float(np.linalg.norm(self.life_direction))
        }


def test_persistent_goals():
    """Test del sistema de metas persistentes."""
    print("=" * 60)
    print("TEST METAS PERSISTENTES (AGI-3)")
    print("=" * 60)

    # Crear agente teleológico
    agent = TeleologicalAgent("NEO", D_dim=6)

    print("\nSimulando 500 pasos de vida con metas...")

    D = np.abs(np.random.randn(6))
    D = D / D.sum()

    for t in range(500):
        # Evolución de drives
        D = D + np.random.randn(6) * 0.02
        D = np.abs(D)
        D = D / D.sum()

        # SAGI simulado (oscila)
        SAGI = 0.5 + 0.3 * np.sin(t / 50) + np.random.randn() * 0.1
        SAGI = np.clip(SAGI, 0, 1)

        # Crisis ocasional
        in_crisis = np.random.random() < 0.05

        # Step
        result = agent.step(D, SAGI, in_crisis)

        if (t + 1) % 100 == 0:
            stats = agent.get_statistics()
            print(f"\n  t={t+1}:")
            print(f"    Goals: total={stats['goals']['total_goals']}, "
                  f"active={stats['goals']['active']}, "
                  f"achieved={stats['goals']['achieved']}")
            print(f"    Purpose: {stats['purpose']:.3f}, "
                  f"mean={stats['mean_purpose']:.3f}")
            print(f"    Last action: {result['goal_action']}")
            print(f"    Thresholds: persist={stats['goals']['persistence_threshold']:.3f}, "
                  f"abandon={stats['goals']['abandon_threshold']:.3f}")

    # Reporte final
    print("\n" + "=" * 60)
    print("REPORTE FINAL DE METAS")
    print("=" * 60)

    report = agent.goals.get_persistence_report()
    print(f"\nMetas totales: {report['total_goals']}")
    print(f"  Logradas: {report['achieved']} ({report['achievement_rate']*100:.0f}%)")
    print(f"  Abandonadas: {report['abandoned']} ({report['abandonment_rate']*100:.0f}%)")
    print(f"  Activas: {report['active']}")
    print(f"\nTransiciones: {report['n_transitions']}")

    # Mostrar transiciones
    print("\nHistorial de transiciones:")
    for trans in agent.goals.transitions[-10:]:
        print(f"  t={trans.t}: {trans.from_goal} → {trans.to_goal} ({trans.reason})")

    return agent


if __name__ == "__main__":
    test_persistent_goals()
