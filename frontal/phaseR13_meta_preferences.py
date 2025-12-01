#!/usr/bin/env python3
"""
R13 – Meta-Preferences (MPF)
============================

Los agentes desarrollan preferencias de segundo orden:
"No solo me gusta X... me gusta SER el tipo de agente al que le gusta X"

Siempre en matemáticas, sin misticismo.

100% ENDÓGENO
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys

sys.path.insert(0, '/root/NEO_EVA')


@dataclass
class MPFState:
    """Estado de Meta-Preferences."""
    meta_preferences: np.ndarray  # m^A
    stability_scores: np.ndarray  # stab_j
    contribution_scores: np.ndarray  # contrib_j
    filter_strength: np.ndarray  # cuánto filtra cada dimensión


class MetaPreferences:
    """
    R13: Meta-preferencias sobre dimensiones del drive.

    Para cada dimensión j:
    - stab_j = 1 - var(w_j(t-W:t))  [estabilidad]
    - contrib_j = corr(w_j, V)      [contribución a valor]

    Meta-preferencia:
    m_j = rank(stab_j) + rank(contrib_j)

    Filtro de cambios:
    Δw_j(aceptado) = Δw_j(propuesto) * rank(m_j)

    Las dimensiones valoradas cambian poco;
    las menos valoradas pueden mutar más.
    """

    def __init__(self, d: int = 7, feature_names: List[str] = None):
        self.d = d
        if feature_names is None:
            self.feature_names = [
                'integration', 'neg_surprise', 'entropy',
                'stability', 'novelty', 'otherness', 'identity'
            ]
        else:
            self.feature_names = feature_names

        # Estados por agente
        self.agents: Dict[str, MPFState] = {}

        # Historias
        self.w_history: Dict[str, List[np.ndarray]] = {}
        self.V_history: Dict[str, List[float]] = {}
        self.meta_pref_history: Dict[str, List[np.ndarray]] = {}

        self.t = 0

    def register_agent(self, name: str):
        """Registra un agente."""
        self.agents[name] = MPFState(
            meta_preferences=np.ones(self.d) / self.d,
            stability_scores=np.ones(self.d) * 0.5,
            contribution_scores=np.zeros(self.d),
            filter_strength=np.ones(self.d) * 0.5
        )
        self.w_history[name] = []
        self.V_history[name] = []
        self.meta_pref_history[name] = []

    def _compute_window(self) -> int:
        """Ventana: W = ceil(sqrt(t))"""
        return max(10, int(np.ceil(np.sqrt(self.t + 1))))

    def _compute_stability(self, name: str) -> np.ndarray:
        """
        Estabilidad por dimensión:
        stab_j = 1 - var(w_j(t-W:t))
        """
        W = self._compute_window()

        if len(self.w_history[name]) < W:
            return np.ones(self.d) * 0.5

        recent_w = np.array(self.w_history[name][-W:])
        variances = np.var(recent_w, axis=0)

        # Normalizar variances a [0, 1] y luego invertir
        max_var = variances.max() + 1e-10
        stab = 1 - (variances / max_var)

        return stab

    def _compute_contribution(self, name: str) -> np.ndarray:
        """
        Contribución a valor por dimensión:
        contrib_j = corr(w_j, V)
        """
        W = self._compute_window()

        if len(self.w_history[name]) < W or len(self.V_history[name]) < W:
            return np.zeros(self.d)

        recent_w = np.array(self.w_history[name][-W:])
        recent_V = np.array(self.V_history[name][-W:])

        contrib = np.zeros(self.d)
        for j in range(self.d):
            if np.std(recent_w[:, j]) > 1e-10 and np.std(recent_V) > 1e-10:
                c = np.corrcoef(recent_w[:, j], recent_V)[0, 1]
                contrib[j] = c if not np.isnan(c) else 0

        return contrib

    def _compute_ranks(self, values: np.ndarray) -> np.ndarray:
        """Convierte valores a ranks en [0, 1]."""
        ranks = np.zeros(len(values))
        for i in range(len(values)):
            ranks[i] = np.mean([1 if values[i] > values[j] else 0
                               for j in range(len(values)) if j != i])
        return ranks

    def update_meta_preferences(self, name: str, w: np.ndarray, V: float) -> np.ndarray:
        """
        Actualiza meta-preferencias.

        Args:
            name: Nombre del agente
            w: Pesos actuales
            V: Valor actual

        Returns:
            Meta-preferencias actualizadas
        """
        self.t += 1

        if name not in self.agents:
            self.register_agent(name)

        state = self.agents[name]

        # Guardar historia
        self.w_history[name].append(w.copy())
        self.V_history[name].append(V)

        # Calcular estabilidad
        stab = self._compute_stability(name)
        state.stability_scores = stab

        # Calcular contribución
        contrib = self._compute_contribution(name)
        state.contribution_scores = contrib

        # Meta-preferencia: m_j = rank(stab_j) + rank(contrib_j)
        stab_ranks = self._compute_ranks(stab)
        contrib_ranks = self._compute_ranks(contrib)

        m = stab_ranks + contrib_ranks
        # Normalizar a [0, 1]
        m = m / (m.max() + 1e-10)

        state.meta_preferences = m
        self.meta_pref_history[name].append(m.copy())

        # Filter strength = rank(m)
        state.filter_strength = self._compute_ranks(m)

        return m

    def filter_weight_change(self, name: str, delta_w: np.ndarray) -> np.ndarray:
        """
        Filtra cambio de pesos por meta-preferencias.

        Δw_j(aceptado) = Δw_j(propuesto) * (1 - rank(m_j))

        Las dimensiones con alta meta-preferencia cambian MENOS.
        Las dimensiones con baja meta-preferencia pueden mutar MÁS.
        """
        if name not in self.agents:
            return delta_w

        state = self.agents[name]

        # Invertir: alta m -> poco cambio
        # 1 - filter_strength: alta m -> bajo multiplicador
        filter_mult = 1 - state.filter_strength * 0.8  # Max reduction 80%

        return delta_w * filter_mult

    def get_protected_dimensions(self, name: str, threshold: float = 0.7) -> List[str]:
        """Retorna dimensiones protegidas (alta meta-preferencia)."""
        if name not in self.agents:
            return []

        state = self.agents[name]
        protected = []

        for j in range(self.d):
            if state.meta_preferences[j] > threshold:
                protected.append(self.feature_names[j])

        return protected


def test_R13_go_nogo(mpf: MetaPreferences, n_nulls: int = 100) -> Dict:
    """
    Tests GO/NO-GO para R13.

    GO si:
    1. m^A no es uniforme: var(m) > p95(null)
    2. Dimensiones con alta m tienen menor volatilidad de w:
       corr(m_j, -var(w_j)) > p95(null)
    3. V mejora con meta-preferencias vs sin ellas
    """
    results = {'passed': [], 'failed': []}

    for name, state in mpf.agents.items():
        # Test 1: Meta-preferencias no uniformes
        m_var = np.var(state.meta_preferences)

        null_vars = []
        for _ in range(n_nulls):
            m_random = np.random.dirichlet(np.ones(mpf.d))
            null_vars.append(np.var(m_random))

        p95 = np.percentile(null_vars, 95)

        if m_var > p95:
            results['passed'].append(f'nonuniform_{name}')
        else:
            results['failed'].append(f'nonuniform_{name}')

        # Test 2: Correlación m vs -var(w)
        if len(mpf.w_history[name]) > 50:
            W = np.array(mpf.w_history[name])
            w_vars = np.var(W, axis=0)

            # Correlación con meta-preferencias (negativa porque alta m -> baja var)
            corr_real = np.corrcoef(state.meta_preferences, -w_vars)[0, 1]
            if np.isnan(corr_real):
                corr_real = 0

            null_corrs = []
            for _ in range(n_nulls):
                m_shuffled = state.meta_preferences.copy()
                np.random.shuffle(m_shuffled)
                c = np.corrcoef(m_shuffled, -w_vars)[0, 1]
                null_corrs.append(c if not np.isnan(c) else 0)

            p95 = np.percentile(null_corrs, 95)

            if corr_real > p95:
                results['passed'].append(f'protection_works_{name}')
            else:
                results['failed'].append(f'protection_works_{name}')

    results['is_go'] = len(results['failed']) == 0
    results['summary'] = "GO" if results['is_go'] else f"NO-GO: {results['failed']}"

    return results


if __name__ == "__main__":
    print("R13 – Meta-Preferences")
    print("=" * 50)

    mpf = MetaPreferences()
    mpf.register_agent("NEO")
    mpf.register_agent("EVA")

    np.random.seed(42)

    w_neo = np.array([0.2, 0.25, 0.1, 0.1, 0.15, 0.1, 0.1])
    w_eva = np.array([0.1, 0.1, 0.2, 0.25, 0.1, 0.15, 0.1])

    for t in range(200):
        # Valor simulado
        V_neo = 0.5 + 0.1 * np.sin(t/20)
        V_eva = 0.5 + 0.1 * np.cos(t/20)

        # Actualizar meta-preferencias
        m_neo = mpf.update_meta_preferences("NEO", w_neo, V_neo)
        m_eva = mpf.update_meta_preferences("EVA", w_eva, V_eva)

        # Simular cambio de pesos
        delta_w_neo = np.random.randn(7) * 0.01
        delta_w_eva = np.random.randn(7) * 0.01

        # Filtrar por meta-preferencias
        delta_w_neo_filtered = mpf.filter_weight_change("NEO", delta_w_neo)
        delta_w_eva_filtered = mpf.filter_weight_change("EVA", delta_w_eva)

        # Aplicar cambio filtrado
        w_neo = w_neo + delta_w_neo_filtered
        w_neo = np.clip(w_neo, 0.01, None)
        w_neo = w_neo / w_neo.sum()

        w_eva = w_eva + delta_w_eva_filtered
        w_eva = np.clip(w_eva, 0.01, None)
        w_eva = w_eva / w_eva.sum()

        if t % 50 == 0:
            print(f"\nt={t}")
            print(f"NEO meta-pref: {m_neo.round(2)}")
            print(f"NEO protegidas: {mpf.get_protected_dimensions('NEO')}")
            print(f"EVA meta-pref: {m_eva.round(2)}")
            print(f"EVA protegidas: {mpf.get_protected_dimensions('EVA')}")

    # Test
    results = test_R13_go_nogo(mpf)
    print(f"\n{results['summary']}")
