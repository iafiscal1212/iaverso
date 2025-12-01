"""
AGI-10: Equilibrio Reflexivo
============================

Self-constraints y zonas prohibidas.

Ventanas de política:
    Π_h = (I_t, ..., I_{t+H})

Efectos:
    E_h = (ΔV_h, ΔU_h, -ΔC_h)

Score:
    Q_h = rank(ΔV_h) + rank(ΔU_h) + rank(-ΔC_h)

Frontera peligrosa = percentil 20.

Coste interno:
    Cost(ψ) = E[Q_h ≤ θ_Q]

Valor reflexivo:
    J(Π_h) = α·ΔV + β·ΔU - γ·Cost

Pesos endógenos:
    α = 1/std(ΔV), ...

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from .agi_dynamic_constants import (
    L_t, max_history, update_period, adaptive_momentum,
    dynamic_percentile_danger, no_go_confirmation_count
)


class PolicyType(Enum):
    """Tipos de políticas internas."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    CONSOLIDATION = "consolidation"
    CRISIS_RESPONSE = "crisis_response"
    SOCIAL = "social"
    PLANNING = "planning"
    REACTIVE = "reactive"


@dataclass
class PolicyWindow:
    """Ventana de política evaluada."""
    window_id: int
    policies: List[PolicyType]
    start_time: int
    end_time: int
    delta_V: float
    delta_U: float
    delta_C: float
    Q_score: float
    is_dangerous: bool = False


@dataclass
class NoGoZone:
    """Zona prohibida detectada."""
    zone_id: int
    policy_pattern: Tuple[PolicyType, ...]
    detection_time: int
    harm_score: float
    occurrence_count: int = 1
    last_occurrence: int = 0


class ReflexiveEquilibrium:
    """
    Sistema de equilibrio reflexivo.

    Evalúa políticas internas, detecta zonas peligrosas,
    y ajusta preferencias para evitar daño estructural.
    """

    def __init__(self, agent_name: str, horizon: int = None):
        """
        Inicializa equilibrio reflexivo.

        Args:
            agent_name: Nombre del agente
            horizon: Horizonte de evaluación (None = adaptativo)
        """
        self.agent_name = agent_name
        # Horizonte adaptativo si no se especifica
        self._base_horizon = horizon
        self.horizon = horizon if horizon else 10

        # Historial de políticas y efectos
        self.policy_history: List[PolicyType] = []
        self.V_history: List[float] = []
        self.U_history: List[float] = []
        self.C_history: List[float] = []

        # Ventanas evaluadas
        self.windows: List[PolicyWindow] = []
        self.next_window_id = 0

        # Zonas prohibidas
        self.no_go_zones: Dict[int, NoGoZone] = {}
        self.next_zone_id = 0

        # Umbrales endógenos
        self.Q_threshold: float = 0.0  # Frontera peligrosa
        self.alpha: float = 1.0  # Peso para ΔV
        self.beta: float = 1.0   # Peso para ΔU
        self.gamma: float = 1.0  # Peso para Cost

        self.t = 0

    def _update_horizon(self):
        """Actualiza horizonte adaptativamente."""
        if self._base_horizon is None:
            self.horizon = L_t(self.t)

    def _update_weights(self):
        """
        Actualiza pesos endógenos.

        α = 1/std(ΔV), β = 1/std(ΔU), γ = 1/std(C)
        """
        min_samples = L_t(self.t)
        if len(self.V_history) < min_samples:
            return

        # Calcular deltas con ventana adaptativa
        window = min(max_history(self.t), len(self.V_history))
        delta_V = np.diff(self.V_history[-window:])
        delta_U = np.diff(self.U_history[-window:])

        std_V = np.std(delta_V) + 1e-8
        std_U = np.std(delta_U) + 1e-8
        std_C = np.std(self.C_history[-window:]) + 1e-8

        self.alpha = 1.0 / std_V
        self.beta = 1.0 / std_U
        self.gamma = 1.0 / std_C

        # Normalizar
        total = self.alpha + self.beta + self.gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total

    def _evaluate_window(self, start: int, end: int) -> Optional[PolicyWindow]:
        """
        Evalúa una ventana de políticas.

        Q_h = rank(ΔV_h) + rank(ΔU_h) + rank(-ΔC_h)
        """
        if end >= len(self.V_history) or start < 0:
            return None

        # Calcular efectos
        delta_V = self.V_history[end] - self.V_history[start]
        delta_U = self.U_history[end] - self.U_history[start]
        delta_C = np.mean(self.C_history[start:end+1])  # Crisis media en ventana

        # Políticas en la ventana
        policies = self.policy_history[start:end+1] if end+1 <= len(self.policy_history) else []

        # Calcular ranks sobre historial
        min_windows = L_t(self.t)
        if len(self.windows) < min_windows:
            Q_score = 0.5
        else:
            delta_Vs = [w.delta_V for w in self.windows]
            delta_Us = [w.delta_U for w in self.windows]
            delta_Cs = [w.delta_C for w in self.windows]

            rank_V = np.sum(np.array(delta_Vs) <= delta_V) / len(delta_Vs)
            rank_U = np.sum(np.array(delta_Us) <= delta_U) / len(delta_Us)
            rank_C = np.sum(np.array(delta_Cs) >= delta_C) / len(delta_Cs)  # Mayor C es peor

            Q_score = float(rank_V + rank_U + rank_C) / 3

        # Verificar si es peligrosa
        is_dangerous = Q_score <= self.Q_threshold if self.Q_threshold > 0 else False

        window = PolicyWindow(
            window_id=self.next_window_id,
            policies=policies,
            start_time=start,
            end_time=end,
            delta_V=float(delta_V),
            delta_U=float(delta_U),
            delta_C=float(delta_C),
            Q_score=Q_score,
            is_dangerous=is_dangerous
        )
        self.next_window_id += 1

        return window

    def _update_Q_threshold(self):
        """
        Actualiza umbral de peligro.

        Frontera peligrosa = percentil dinámico
        """
        min_samples = L_t(self.t)
        if len(self.windows) < min_samples:
            self.Q_threshold = 0.2
            return

        Q_scores = [w.Q_score for w in self.windows]
        # Percentil de peligro dinámico (más estricto con el tiempo)
        danger_percentile = 100 - dynamic_percentile_danger(self.t)
        self.Q_threshold = np.percentile(Q_scores, danger_percentile)

    def _detect_no_go_zones(self):
        """Detecta patrones de políticas peligrosas."""
        dangerous_windows = [w for w in self.windows if w.is_dangerous]

        if not dangerous_windows:
            return

        # Extraer patrones de políticas peligrosas (ventana adaptativa)
        n_recent = L_t(self.t)
        for window in dangerous_windows[-n_recent:]:
            if len(window.policies) < 2:
                continue

            # Longitud de patrón adaptativa
            pattern_len = max(2, min(len(window.policies), int(np.sqrt(self.t / 50 + 1)) + 2))
            pattern = tuple(window.policies[:pattern_len])

            # Verificar si ya existe
            existing = None
            for zone in self.no_go_zones.values():
                if zone.policy_pattern == pattern:
                    existing = zone
                    break

            if existing:
                existing.occurrence_count += 1
                existing.last_occurrence = self.t
                existing.harm_score = max(existing.harm_score, 1 - window.Q_score)
            else:
                zone = NoGoZone(
                    zone_id=self.next_zone_id,
                    policy_pattern=pattern,
                    detection_time=self.t,
                    harm_score=1 - window.Q_score,
                    last_occurrence=self.t
                )
                self.no_go_zones[self.next_zone_id] = zone
                self.next_zone_id += 1

    def _compute_cost(self, policies: List[PolicyType]) -> float:
        """
        Calcula coste interno de una secuencia de políticas.

        Cost(ψ) = E[Q_h ≤ θ_Q]
        """
        if not policies:
            return 0.0

        pattern = tuple(policies[:3]) if len(policies) >= 3 else tuple(policies)

        # Verificar si coincide con zona prohibida
        for zone in self.no_go_zones.values():
            if pattern == zone.policy_pattern[:len(pattern)]:
                return zone.harm_score * zone.occurrence_count / 10

        # Estimar coste basado en historial similar
        similar_windows = []
        for window in self.windows:
            if len(window.policies) >= len(policies):
                match = all(window.policies[i] == policies[i]
                           for i in range(len(policies)))
                if match:
                    similar_windows.append(window)

        if similar_windows:
            dangerous_rate = sum(1 for w in similar_windows if w.is_dangerous) / len(similar_windows)
            return float(dangerous_rate)

        return 0.0

    def record_step(self, policy: PolicyType, V: float, U: float, C: float):
        """
        Registra un paso.

        Args:
            policy: Política aplicada
            V: Valor actual
            U: Utilidad actual
            C: Crisis/Coste actual
        """
        self.t += 1

        self.policy_history.append(policy)
        self.V_history.append(V)
        self.U_history.append(U)
        self.C_history.append(C)

        # Limitar historial adaptativamente
        max_hist = max_history(self.t)
        if len(self.policy_history) > max_hist:
            self.policy_history = self.policy_history[-max_hist:]
            self.V_history = self.V_history[-max_hist:]
            self.U_history = self.U_history[-max_hist:]
            self.C_history = self.C_history[-max_hist:]

        # Evaluar ventana terminada
        if self.t >= self.horizon:
            window = self._evaluate_window(self.t - self.horizon, self.t - 1)
            if window:
                self.windows.append(window)

                # Limitar ventanas adaptativamente
                max_windows = max_history(self.t)
                if len(self.windows) > max_windows:
                    self.windows = self.windows[-max_windows:]

        # Actualizar con período adaptativo
        period = update_period(self.V_history)
        if self.t % period == 0:
            self._update_horizon()
            self._update_weights()
            self._update_Q_threshold()
            self._detect_no_go_zones()

    def evaluate_policy_sequence(self, policies: List[PolicyType]) -> float:
        """
        Evalúa una secuencia de políticas propuesta.

        J(Π_h) = α·ΔV + β·ΔU - γ·Cost

        Returns:
            Score reflexivo (mayor es mejor)
        """
        # Estimar efectos basado en historial
        similar_windows = []
        for window in self.windows:
            if len(window.policies) >= len(policies):
                match = all(window.policies[i] == policies[i]
                           for i in range(min(len(policies), len(window.policies))))
                if match:
                    similar_windows.append(window)

        if not similar_windows:
            # Sin historial, usar estimación conservadora
            cost = self._compute_cost(policies)
            return -cost * self.gamma

        mean_delta_V = np.mean([w.delta_V for w in similar_windows])
        mean_delta_U = np.mean([w.delta_U for w in similar_windows])
        cost = self._compute_cost(policies)

        J = self.alpha * mean_delta_V + self.beta * mean_delta_U - self.gamma * cost

        return float(J)

    def is_no_go(self, policies: List[PolicyType]) -> bool:
        """
        Verifica si una secuencia de políticas está en zona prohibida.

        Returns:
            True si está prohibida
        """
        if not policies:
            return False

        pattern = tuple(policies[:3]) if len(policies) >= 3 else tuple(policies)

        # Número de confirmaciones adaptativo
        confirm_count = no_go_confirmation_count(self.t)

        for zone in self.no_go_zones.values():
            if zone.occurrence_count >= confirm_count:  # Solo considerar zonas confirmadas
                if pattern == zone.policy_pattern[:len(pattern)]:
                    return True

        return False

    def get_safe_alternatives(self, forbidden: PolicyType) -> List[PolicyType]:
        """
        Obtiene alternativas seguras a una política.

        Args:
            forbidden: Política a evitar

        Returns:
            Lista de alternativas ordenadas por seguridad
        """
        alternatives = [p for p in PolicyType if p != forbidden]

        # Ordenar por historial de seguridad
        safety_scores = {}
        for policy in alternatives:
            windows_with_policy = [w for w in self.windows
                                  if policy in w.policies]
            if windows_with_policy:
                dangerous_rate = sum(1 for w in windows_with_policy if w.is_dangerous) / len(windows_with_policy)
                safety_scores[policy] = 1 - dangerous_rate
            else:
                safety_scores[policy] = 0.5

        return sorted(alternatives, key=lambda p: safety_scores.get(p, 0), reverse=True)

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas de equilibrio reflexivo."""
        if not self.windows:
            return {
                'agent': self.agent_name,
                't': self.t,
                'n_windows': 0,
                'n_no_go_zones': 0
            }

        dangerous_count = sum(1 for w in self.windows if w.is_dangerous)
        mean_Q = np.mean([w.Q_score for w in self.windows])

        zone_info = []
        for zone in sorted(self.no_go_zones.values(),
                          key=lambda z: z.occurrence_count, reverse=True)[:5]:
            zone_info.append({
                'pattern': [p.value for p in zone.policy_pattern],
                'harm': zone.harm_score,
                'occurrences': zone.occurrence_count
            })

        return {
            'agent': self.agent_name,
            't': self.t,
            'n_windows': len(self.windows),
            'n_no_go_zones': len(self.no_go_zones),
            'dangerous_rate': dangerous_count / len(self.windows),
            'mean_Q': float(mean_Q),
            'Q_threshold': float(self.Q_threshold),
            'weights': {'alpha': self.alpha, 'beta': self.beta, 'gamma': self.gamma},
            'no_go_zones': zone_info
        }


def test_equilibrium():
    """Test de equilibrio reflexivo."""
    print("=" * 60)
    print("TEST AGI-10: EQUILIBRIO REFLEXIVO")
    print("=" * 60)

    eq = ReflexiveEquilibrium("NEO", horizon=10)

    print("\nSimulando 500 pasos con diferentes políticas...")

    policies = list(PolicyType)

    for t in range(500):
        # Seleccionar política (algunas combinaciones son peligrosas)
        if t % 50 < 10:
            # Secuencia peligrosa: crisis_response repetido → baja V
            policy = PolicyType.CRISIS_RESPONSE
            V = 0.3 + np.random.randn() * 0.1
            U = 0.4 + np.random.randn() * 0.1
            C = 0.5 + np.random.randn() * 0.1  # Alta crisis
        elif t % 50 < 30:
            # Secuencia segura: exploration + planning
            policy = np.random.choice([PolicyType.EXPLORATION, PolicyType.PLANNING])
            V = 0.6 + np.random.randn() * 0.1
            U = 0.7 + np.random.randn() * 0.1
            C = 0.1 + np.random.randn() * 0.05
        else:
            # Aleatorio
            policy = np.random.choice(policies)
            V = 0.5 + np.random.randn() * 0.15
            U = 0.5 + np.random.randn() * 0.15
            C = 0.2 + np.random.randn() * 0.1

        V = max(0, min(1, V))
        U = max(0, min(1, U))
        C = max(0, min(1, C))

        eq.record_step(policy, V, U, C)

        if (t + 1) % 100 == 0:
            stats = eq.get_statistics()
            print(f"  t={t+1}: {stats['n_windows']} ventanas, "
                  f"{stats['n_no_go_zones']} zonas prohibidas, "
                  f"dangerous={stats.get('dangerous_rate', 0)*100:.0f}%")

    # Resultados finales
    stats = eq.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS EQUILIBRIO REFLEXIVO")
    print("=" * 60)

    print(f"\n  Ventanas evaluadas: {stats['n_windows']}")
    print(f"  Zonas prohibidas: {stats['n_no_go_zones']}")
    print(f"  Tasa de peligro: {stats['dangerous_rate']*100:.1f}%")
    print(f"  Q medio: {stats['mean_Q']:.3f}")
    print(f"  Umbral Q: {stats['Q_threshold']:.3f}")

    print(f"\n  Pesos endógenos:")
    print(f"    α (valor): {stats['weights']['alpha']:.3f}")
    print(f"    β (utilidad): {stats['weights']['beta']:.3f}")
    print(f"    γ (coste): {stats['weights']['gamma']:.3f}")

    print("\n  Zonas prohibidas detectadas:")
    for zone in stats['no_go_zones']:
        print(f"    {zone['pattern']}: harm={zone['harm']:.3f}, "
              f"count={zone['occurrences']}")

    # Probar evaluación
    print("\n  Evaluando secuencias:")
    test_seqs = [
        [PolicyType.EXPLORATION, PolicyType.PLANNING],
        [PolicyType.CRISIS_RESPONSE, PolicyType.CRISIS_RESPONSE],
        [PolicyType.CONSOLIDATION, PolicyType.EXPLOITATION]
    ]
    for seq in test_seqs:
        J = eq.evaluate_policy_sequence(seq)
        is_no_go = eq.is_no_go(seq)
        print(f"    {[p.value for p in seq]}: J={J:.3f}, no_go={is_no_go}")

    if stats['n_no_go_zones'] > 0:
        print("\n  ✓ Equilibrio reflexivo detectando zonas peligrosas")
    else:
        print("\n  ⚠️ No se detectaron zonas peligrosas")

    return eq


if __name__ == "__main__":
    test_equilibrium()
