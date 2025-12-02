"""
Ω2: Teleología Extensa
======================

Metas de segundo orden:
- Mantener coherencia a largo plazo
- Preservar valores incluso si el mundo cambia
- Maximizar ELLEX
- Estabilizar identidad

Functional Telos Index (FTI):
    U_Ω = α·U_local + β·ΔELLEX + γ·ΔNorms + δ·IdentityStability

Todos los pesos α, β, γ, δ endógenos.

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FunctionalTelosIndex:
    """Índice de teleología funcional."""
    t: int
    fti: float                      # Functional Telos Index
    utility_local: float            # Utilidad local
    delta_ellex: float              # Cambio en ELLEX
    delta_norms: float              # Cambio en normas
    identity_stability: float       # Estabilidad de identidad
    weights: Dict[str, float]       # Pesos usados


class OmegaTeleology:
    """
    Sistema de teleología extensa.

    Metas de segundo orden que preservan:
    - Coherencia
    - Valores
    - Identidad
    - ELLEX

    Pesos α, β, γ, δ son endógenos por varianza inversa.
    """

    def __init__(self):
        self.t = 0

        # Historiales para pesos endógenos
        self._utility_history: List[float] = []
        self._ellex_history: List[float] = []
        self._norms_history: List[float] = []
        self._identity_history: List[float] = []

        # Pesos actuales (inicialmente uniformes)
        self._weights = {
            'alpha': 1/4,  # utility_local
            'beta': 1/4,   # delta_ellex
            'gamma': 1/4,  # delta_norms
            'delta': 1/4   # identity_stability
        }

    def _update_weights(self) -> None:
        """
        Actualiza pesos por varianza inversa.

        w_k = 1/var_k / Σ(1/var_i)
        """
        if len(self._utility_history) < 10:
            return

        EPS = np.finfo(float).eps

        var_u = np.var(self._utility_history[-20:]) + EPS
        var_e = np.var(self._ellex_history[-20:]) + EPS
        var_n = np.var(self._norms_history[-20:]) + EPS
        var_i = np.var(self._identity_history[-20:]) + EPS

        inv_vars = [1/var_u, 1/var_e, 1/var_n, 1/var_i]
        total = sum(inv_vars)

        self._weights = {
            'alpha': inv_vars[0] / total,
            'beta': inv_vars[1] / total,
            'gamma': inv_vars[2] / total,
            'delta': inv_vars[3] / total
        }

    def _compute_utility_local(
        self,
        state: np.ndarray,
        action_result: Dict[str, Any]
    ) -> float:
        """
        Calcula utilidad local de una decisión.

        Basado en resultado inmediato.
        """
        if 'reward' in action_result:
            utility = action_result['reward']
        elif 'success' in action_result:
            utility = 1 if action_result['success'] else 0
        else:
            # Sin info, usar norma del estado como proxy
            utility = 1 - (np.var(state) / (np.var(state) + 1))

        return float(np.clip(utility, 0, 1))

    def _compute_delta_ellex(
        self,
        ellex_current: float,
        ellex_previous: float
    ) -> float:
        """
        Calcula cambio en ELLEX.

        Positivo = mejora, Negativo = deterioro
        """
        delta = ellex_current - ellex_previous

        # Normalizar a [-1, 1]
        if len(self._ellex_history) > 5:
            max_delta = max(abs(self._ellex_history[i] - self._ellex_history[i-1])
                           for i in range(1, len(self._ellex_history)))
            delta_norm = delta / (max_delta + np.finfo(float).eps)
        else:
            delta_norm = delta

        # Mapear a [0, 1] donde 0.5 = sin cambio
        return float((delta_norm + 1) / 2)

    def _compute_delta_norms(
        self,
        norms_current: Dict[str, float],
        norms_previous: Dict[str, float]
    ) -> float:
        """
        Calcula adherencia a normas.

        Mayor = mejor adherencia
        """
        if not norms_current:
            return 1/2

        # Calcular desviación de normas
        deviations = []
        for k, v_curr in norms_current.items():
            v_prev = norms_previous.get(k, v_curr)
            deviation = abs(v_curr - v_prev)
            deviations.append(deviation)

        if not deviations:
            return 1/2

        # Menor desviación = mejor
        mean_dev = np.mean(deviations)

        # Normalizar por historial
        if len(self._norms_history) > 5:
            max_dev = np.percentile(self._norms_history, 95)
            adherence = 1 - (mean_dev / (max_dev + np.finfo(float).eps))
        else:
            adherence = 1 / (1 + mean_dev)

        return float(np.clip(adherence, 0, 1))

    def _compute_identity_stability(
        self,
        identity_current: np.ndarray,
        identity_previous: np.ndarray
    ) -> float:
        """
        Calcula estabilidad de identidad.

        Mayor = identidad más estable
        """
        # Distancia entre identidades
        dist = np.linalg.norm(identity_current - identity_previous)

        # Normalizar
        if len(self._identity_history) > 5:
            max_dist = np.percentile(self._identity_history, 95)
            stability = 1 - (dist / (max_dist + np.finfo(float).eps))
        else:
            stability = 1 / (1 + dist)

        return float(np.clip(stability, 0, 1))

    def compute_fti(
        self,
        state: np.ndarray,
        action_result: Dict[str, Any],
        ellex_current: float,
        ellex_previous: float,
        norms_current: Dict[str, float],
        norms_previous: Dict[str, float],
        identity_current: np.ndarray,
        identity_previous: np.ndarray
    ) -> FunctionalTelosIndex:
        """
        Calcula Functional Telos Index.

        U_Ω = α·U_local + β·ΔELLEX + γ·ΔNorms + δ·IdentityStability

        Returns:
            FunctionalTelosIndex con todos los componentes
        """
        self.t += 1

        # Calcular componentes
        u_local = self._compute_utility_local(state, action_result)
        delta_ellex = self._compute_delta_ellex(ellex_current, ellex_previous)
        delta_norms = self._compute_delta_norms(norms_current, norms_previous)
        id_stability = self._compute_identity_stability(identity_current, identity_previous)

        # Actualizar historiales
        self._utility_history.append(u_local)
        self._ellex_history.append(ellex_current)
        self._norms_history.append(delta_norms)
        self._identity_history.append(id_stability)

        # Actualizar pesos
        self._update_weights()

        # Calcular FTI
        fti = (
            self._weights['alpha'] * u_local +
            self._weights['beta'] * delta_ellex +
            self._weights['gamma'] * delta_norms +
            self._weights['delta'] * id_stability
        )

        return FunctionalTelosIndex(
            t=self.t,
            fti=float(fti),
            utility_local=u_local,
            delta_ellex=delta_ellex,
            delta_norms=delta_norms,
            identity_stability=id_stability,
            weights=self._weights.copy()
        )

    def evaluate_decision(
        self,
        candidate_actions: List[Dict[str, Any]],
        current_state: Dict[str, Any]
    ) -> int:
        """
        Evalúa decisiones candidatas por FTI.

        Returns:
            Índice de la mejor acción
        """
        if not candidate_actions:
            return -1

        best_idx = 0
        best_fti = -float('inf')

        for i, action in enumerate(candidate_actions):
            # Estimar FTI para cada acción
            estimated_fti = self._estimate_action_fti(action, current_state)
            if estimated_fti > best_fti:
                best_fti = estimated_fti
                best_idx = i

        return best_idx

    def _estimate_action_fti(
        self,
        action: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> float:
        """Estima FTI para una acción candidata."""
        # Estimación basada en impacto declarado
        impact_ellex = action.get('impact_ellex', 0)
        impact_norms = action.get('impact_norms', 0)
        impact_identity = action.get('impact_identity', 0)
        reward = action.get('expected_reward', 1/2)

        # FTI estimado
        fti = (
            self._weights['alpha'] * reward +
            self._weights['beta'] * (impact_ellex + 1) / 2 +
            self._weights['gamma'] * (1 - abs(impact_norms)) +
            self._weights['delta'] * (1 - abs(impact_identity))
        )

        return fti

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas."""
        return {
            't': self.t,
            'weights': self._weights,
            'fti_mean': float(np.mean([
                self._weights['alpha'] * u +
                self._weights['beta'] * e +
                self._weights['gamma'] * n +
                self._weights['delta'] * i
                for u, e, n, i in zip(
                    self._utility_history[-10:] or [1/2],
                    self._ellex_history[-10:] or [1/2],
                    self._norms_history[-10:] or [1/2],
                    self._identity_history[-10:] or [1/2]
                )
            ])) if self._utility_history else 1/2
        }
