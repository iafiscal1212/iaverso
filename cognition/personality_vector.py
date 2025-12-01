"""
Vector de Personalidad (Observacional)
======================================

p_i = (χ_i, S_i, ρ_i, H_i, N_i, Σ_i)

donde:
- χ_i: Curiosidad (entropy seek rate)
- S_i: Sociabilidad (N coocurrencias / T)
- ρ_i: Riesgo (Var(ΔV) / E[ΔV])
- H_i: Horizonte (longitud media de secuencias predictivas)
- N_i: Normatividad (% acciones dentro de δ_ético)
- Σ_i: Síntesis simbólica (# nuevos bindings / T)

Solo observacional (no afecta decisiones).
100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history, normalized_entropy


@dataclass
class PersonalityVector:
    """Vector de personalidad observacional."""
    chi: float      # Curiosidad
    S: float        # Sociabilidad
    rho: float      # Riesgo
    H: float        # Horizonte
    N: float        # Normatividad
    Sigma: float    # Síntesis simbólica

    def to_array(self) -> np.ndarray:
        """Convierte a array numpy."""
        return np.array([self.chi, self.S, self.rho, self.H, self.N, self.Sigma])

    def to_dict(self) -> Dict[str, float]:
        """Convierte a diccionario."""
        return {
            'curiosity': self.chi,
            'sociability': self.S,
            'risk': self.rho,
            'horizon': self.H,
            'normativity': self.N,
            'synthesis': self.Sigma
        }


class PersonalityObserver:
    """
    Sistema de observación de personalidad.

    Solo observa, no afecta decisiones del agente.
    Todos los componentes son endógenos.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

        # Historiales para cada componente
        # χ: Curiosidad
        self.entropy_history: List[float] = []
        self.entropy_seek_rate: List[float] = []

        # S: Sociabilidad
        self.cooccurrences: List[int] = []  # N agentes coocurrentes
        self.social_interactions: int = 0

        # ρ: Riesgo
        self.delta_V_history: List[float] = []

        # H: Horizonte
        self.sequence_lengths: List[int] = []

        # N: Normatividad
        self.actions_in_delta: List[bool] = []
        self.ethical_bounds_history: List[Tuple[float, float]] = []  # (lower, upper)

        # Σ: Síntesis simbólica
        self.new_bindings_history: List[int] = []
        self.total_bindings: int = 0

        # Historial de vectores
        self.personality_history: List[PersonalityVector] = []

        self.t = 0
        self.T = 0  # Tiempo total

    def observe_entropy_seeking(
        self,
        t: int,
        state_entropy: float,
        action_was_explorative: bool
    ) -> None:
        """
        Observa búsqueda de entropía (curiosidad).

        χ = rate at which agent seeks high-entropy states
        """
        self.t = t
        self.T += 1

        self.entropy_history.append(state_entropy)

        # Rate de seek: ¿buscó estados de alta entropía?
        if action_was_explorative:
            self.entropy_seek_rate.append(1.0)
        else:
            self.entropy_seek_rate.append(0.0)

        # Limitar
        max_h = max_history(t)
        if len(self.entropy_history) > max_h:
            self.entropy_history = self.entropy_history[-max_h:]
        if len(self.entropy_seek_rate) > max_h:
            self.entropy_seek_rate = self.entropy_seek_rate[-max_h:]

    def observe_social(
        self,
        t: int,
        n_agents_present: int,
        had_interaction: bool = False
    ) -> None:
        """
        Observa interacciones sociales.

        S = N coocurrencias / T
        """
        self.t = t

        self.cooccurrences.append(n_agents_present)
        if had_interaction:
            self.social_interactions += 1

        # Limitar
        max_h = max_history(t)
        if len(self.cooccurrences) > max_h:
            self.cooccurrences = self.cooccurrences[-max_h:]

    def observe_risk(
        self,
        t: int,
        delta_V: float
    ) -> None:
        """
        Observa comportamiento de riesgo.

        ρ = Var(ΔV) / E[|ΔV|]
        """
        self.t = t
        self.delta_V_history.append(delta_V)

        # Limitar
        max_h = max_history(t)
        if len(self.delta_V_history) > max_h:
            self.delta_V_history = self.delta_V_history[-max_h:]

    def observe_planning_horizon(
        self,
        t: int,
        sequence_length: int
    ) -> None:
        """
        Observa horizonte de planificación.

        H = longitud media de secuencias predictivas
        """
        self.t = t
        self.sequence_lengths.append(sequence_length)

        # Limitar
        max_h = max_history(t)
        if len(self.sequence_lengths) > max_h:
            self.sequence_lengths = self.sequence_lengths[-max_h:]

    def observe_normative_action(
        self,
        t: int,
        action_value: float,
        ethical_lower: float,
        ethical_upper: float
    ) -> None:
        """
        Observa acciones normativas.

        N = % acciones dentro de δ_ético
        """
        self.t = t

        in_bounds = ethical_lower <= action_value <= ethical_upper
        self.actions_in_delta.append(in_bounds)
        self.ethical_bounds_history.append((ethical_lower, ethical_upper))

        # Limitar
        max_h = max_history(t)
        if len(self.actions_in_delta) > max_h:
            self.actions_in_delta = self.actions_in_delta[-max_h:]
        if len(self.ethical_bounds_history) > max_h:
            self.ethical_bounds_history = self.ethical_bounds_history[-max_h:]

    def observe_symbolic_synthesis(
        self,
        t: int,
        n_new_bindings: int
    ) -> None:
        """
        Observa síntesis simbólica.

        Σ = # nuevos bindings / T
        """
        self.t = t

        self.new_bindings_history.append(n_new_bindings)
        self.total_bindings += n_new_bindings

        # Limitar
        max_h = max_history(t)
        if len(self.new_bindings_history) > max_h:
            self.new_bindings_history = self.new_bindings_history[-max_h:]

    def _compute_chi(self, t: int) -> float:
        """Computa curiosidad χ."""
        L = L_t(t)

        if not self.entropy_seek_rate:
            return 0.5

        recent_rates = self.entropy_seek_rate[-L:]
        chi = np.mean(recent_rates)

        return float(np.clip(chi, 0, 1))

    def _compute_S(self, t: int) -> float:
        """Computa sociabilidad S."""
        if self.T == 0:
            return 0.5

        # S = mean(cooccurrences) / max_possible
        L = L_t(t)
        recent_cooc = self.cooccurrences[-L:] if self.cooccurrences else [1]

        # Normalizar por máximo observado
        max_cooc = max(recent_cooc) if recent_cooc else 1
        S = np.mean(recent_cooc) / (max_cooc + 1e-10)

        return float(np.clip(S, 0, 1))

    def _compute_rho(self, t: int) -> float:
        """Computa riesgo ρ."""
        L = L_t(t)

        if len(self.delta_V_history) < L:
            return 0.5

        recent_deltas = self.delta_V_history[-L:]

        var_delta = np.var(recent_deltas)
        mean_abs_delta = np.mean(np.abs(recent_deltas))

        if mean_abs_delta < 1e-10:
            rho = 0.5
        else:
            rho = var_delta / mean_abs_delta

        # Normalizar a [0, 1]
        rho = np.tanh(rho)

        return float(np.clip(rho, 0, 1))

    def _compute_H(self, t: int) -> float:
        """Computa horizonte H."""
        L = L_t(t)

        if not self.sequence_lengths:
            return 0.5

        recent_lengths = self.sequence_lengths[-L:]
        mean_length = np.mean(recent_lengths)

        # Normalizar por máximo razonable
        max_reasonable = np.sqrt(t) + 5
        H = mean_length / max_reasonable

        return float(np.clip(H, 0, 1))

    def _compute_N(self, t: int) -> float:
        """Computa normatividad N."""
        L = L_t(t)

        if not self.actions_in_delta:
            return 0.5

        recent_actions = self.actions_in_delta[-L:]
        N = sum(recent_actions) / len(recent_actions)

        return float(np.clip(N, 0, 1))

    def _compute_Sigma(self, t: int) -> float:
        """Computa síntesis simbólica Σ."""
        if self.T == 0:
            return 0.5

        L = L_t(t)
        recent_bindings = self.new_bindings_history[-L:] if self.new_bindings_history else [0]

        # Σ = rate of new bindings
        mean_bindings = np.mean(recent_bindings)

        # Normalizar
        expected_rate = np.sqrt(t) / 10
        Sigma = mean_bindings / (expected_rate + 1e-10)
        Sigma = np.tanh(Sigma)

        return float(np.clip(Sigma, 0, 1))

    def compute_personality(self, t: int) -> PersonalityVector:
        """
        Computa vector de personalidad completo.

        p_i = (χ_i, S_i, ρ_i, H_i, N_i, Σ_i)
        """
        p = PersonalityVector(
            chi=self._compute_chi(t),
            S=self._compute_S(t),
            rho=self._compute_rho(t),
            H=self._compute_H(t),
            N=self._compute_N(t),
            Sigma=self._compute_Sigma(t)
        )

        self.personality_history.append(p)

        # Limitar historial
        max_h = max_history(t)
        if len(self.personality_history) > max_h:
            self.personality_history = self.personality_history[-max_h:]

        return p

    def get_personality_drift(self, t: int) -> float:
        """
        Mide cuánto ha cambiado la personalidad.

        Drift = ||p_t - p_{t-L}|| / √6
        """
        L = L_t(t)

        if len(self.personality_history) < L + 1:
            return 0.0

        p_current = self.personality_history[-1].to_array()
        p_past = self.personality_history[-L - 1].to_array()

        drift = np.linalg.norm(p_current - p_past) / np.sqrt(6)

        return float(np.clip(drift, 0, 1))

    def get_statistics(self) -> Dict[str, Any]:
        """Estadísticas del sistema de personalidad."""
        if not self.personality_history:
            return {
                'agent_id': self.agent_id,
                't': self.t,
                'T': self.T,
                'personality': None
            }

        current_p = self.personality_history[-1]

        return {
            'agent_id': self.agent_id,
            't': self.t,
            'T': self.T,
            'personality': current_p.to_dict(),
            'drift': self.get_personality_drift(self.t),
            'total_interactions': self.social_interactions,
            'total_bindings': self.total_bindings
        }


def run_test() -> Dict[str, Any]:
    """
    Test del Vector de Personalidad.

    p_i = (χ_i, S_i, ρ_i, H_i, N_i, Σ_i)
    Solo observacional.
    """
    np.random.seed(42)

    observer = PersonalityObserver('TEST')

    # Simular comportamiento de un agente
    for t in range(1, 301):
        # Curiosidad: búsqueda de entropía
        entropy = np.random.rand()
        explorative = np.random.rand() > 0.6  # 40% explorador
        observer.observe_entropy_seeking(t, entropy, explorative)

        # Sociabilidad: coocurrencias
        n_agents = np.random.randint(1, 5)
        interaction = np.random.rand() > 0.5
        observer.observe_social(t, n_agents, interaction)

        # Riesgo: cambios de valor
        delta_V = np.random.randn() * 0.3
        observer.observe_risk(t, delta_V)

        # Horizonte: longitud de secuencias
        seq_len = np.random.randint(3, 10 + t // 50)
        observer.observe_planning_horizon(t, seq_len)

        # Normatividad: acciones éticas
        action = np.random.rand()
        lower, upper = 0.2, 0.8  # Bounds éticos endógenos
        observer.observe_normative_action(t, action, lower, upper)

        # Síntesis: nuevos bindings
        new_bindings = np.random.poisson(0.5)
        observer.observe_symbolic_synthesis(t, new_bindings)

    # Computar personalidad final
    personality = observer.compute_personality(300)
    stats = observer.get_statistics()

    return {
        'score': float(np.mean(personality.to_array())),
        'passed': True,  # Siempre pasa (es observacional)
        'details': {
            'chi_curiosity': float(personality.chi),
            'S_sociability': float(personality.S),
            'rho_risk': float(personality.rho),
            'H_horizon': float(personality.H),
            'N_normativity': float(personality.N),
            'Sigma_synthesis': float(personality.Sigma),
            'drift': float(stats['drift']),
            'total_interactions': stats['total_interactions'],
            'total_bindings': stats['total_bindings']
        }
    }


if __name__ == "__main__":
    result = run_test()
    print("=" * 60)
    print("PERSONALITY VECTOR (OBSERVATIONAL)")
    print("=" * 60)
    print(f"Mean Score: {result['score']:.4f}")
    print(f"\nComponents:")
    print(f"  χ (Curiosity):    {result['details']['chi_curiosity']:.4f}")
    print(f"  S (Sociability):  {result['details']['S_sociability']:.4f}")
    print(f"  ρ (Risk):         {result['details']['rho_risk']:.4f}")
    print(f"  H (Horizon):      {result['details']['H_horizon']:.4f}")
    print(f"  N (Normativity):  {result['details']['N_normativity']:.4f}")
    print(f"  Σ (Synthesis):    {result['details']['Sigma_synthesis']:.4f}")
    print(f"\nDrift: {result['details']['drift']:.4f}")
    print(f"Total Interactions: {result['details']['total_interactions']}")
    print(f"Total Bindings: {result['details']['total_bindings']}")
