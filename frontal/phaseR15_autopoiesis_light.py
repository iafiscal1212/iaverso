#!/usr/bin/env python3
"""
R15 – Autopoiesis Light (APL)
=============================

El sistema ajusta su propia arquitectura funcional...
pero solo en términos de pesos entre módulos/fases.

No se reescribe como Skynet; se reorganiza como un cerebro
que regula qué zonas mandan más o menos.

100% ENDÓGENO
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys

sys.path.insert(0, '/root/NEO_EVA')


@dataclass
class ModuleHealth:
    """Salud de un módulo."""
    name: str
    delta_sagi: float  # Cambio en SAGI
    stability: float  # Estabilidad
    collapses: int  # Número de colapsos
    score: float  # H_i(t) = rank(ΔSAGI) + rank(stability) - rank(collapses)


@dataclass
class APLState:
    """Estado de Autopoiesis Light."""
    W_mod: np.ndarray  # Matriz de influencia entre módulos
    module_health: Dict[str, ModuleHealth]
    history: List[np.ndarray]


class AutopoiesisLight:
    """
    R15: Auto-reorganización ligera.

    Grafo de módulos con Transfer Entropy:
    W_mod(t) ∈ R^{n×n}
    W_mod,ij(t) = rank(TE_{i→j}(t))

    Salud de cada módulo:
    H_i(t) = rank(ΔSAGI_i) + rank(stability_i) - rank(collapses_i)

    Regla autopoética:
    η_arch,t = 1/√(t+1)
    ΔW_mod,ij = η_arch * rank(H_i) * (W_mod,ij - mean_j W_mod,ij)

    Normalización por filas.
    """

    # Módulos/fases clave
    DEFAULT_MODULES = [
        'drives',           # R11 - Self-Drive Redefinition
        'coherence',        # R12 - Drive-Coherence
        'meta_pref',        # R13 - Meta-Preferences
        'intentionality',   # R14 - Structural Intentionality
        'irreversibility',  # R4 - Irreversibility
        'grounding',        # G1/G2 - Grounding
        'counterfactuals',  # R3 - Counterfactuals
        'phenomenology',    # S1/S2 - Subjectivity
        'integration',      # I1/I2 - Integration
        'causality'         # WEAVER - Causal structure
    ]

    def __init__(self, modules: List[str] = None):
        if modules is None:
            self.modules = self.DEFAULT_MODULES
        else:
            self.modules = modules

        self.n = len(self.modules)
        self.module_idx = {m: i for i, m in enumerate(self.modules)}

        # Matriz de influencia inicial (uniforme)
        self.W_mod = np.ones((self.n, self.n)) / self.n

        # Salud de módulos
        self.module_health: Dict[str, ModuleHealth] = {
            m: ModuleHealth(name=m, delta_sagi=0, stability=0.5, collapses=0, score=0.5)
            for m in self.modules
        }

        # Historias
        self.W_history: List[np.ndarray] = [self.W_mod.copy()]
        self.health_history: Dict[str, List[float]] = {m: [] for m in self.modules}
        self.te_history: List[np.ndarray] = []

        self.t = 0

    def _compute_learning_rate(self) -> float:
        """η_arch = 1/√(t+1)"""
        return 1.0 / np.sqrt(self.t + 1)

    def _compute_ranks(self, values: np.ndarray) -> np.ndarray:
        """Convierte valores a ranks en [0, 1]."""
        n = len(values)
        ranks = np.zeros(n)
        for i in range(n):
            ranks[i] = np.mean([1 if values[i] > values[j] else 0
                               for j in range(n) if j != i])
        return ranks

    def update_transfer_entropy(self, TE: np.ndarray):
        """
        Actualiza matriz de influencia basada en Transfer Entropy.

        Args:
            TE: Matriz n×n de Transfer Entropy entre módulos
        """
        self.te_history.append(TE.copy())

        # Convertir a ranks por fila
        for i in range(self.n):
            self.W_mod[i, :] = self._compute_ranks(TE[i, :])

        # Normalizar filas
        for i in range(self.n):
            row_sum = self.W_mod[i, :].sum() + 1e-10
            self.W_mod[i, :] = self.W_mod[i, :] / row_sum

    def update_module_health(self, module: str, delta_sagi: float,
                            stability: float, collapses: int):
        """
        Actualiza la salud de un módulo.

        H_i = rank(ΔSAGI) + rank(stability) - rank(collapses)
        """
        if module not in self.module_health:
            return

        self.module_health[module].delta_sagi = delta_sagi
        self.module_health[module].stability = stability
        self.module_health[module].collapses = collapses

        # Calcular score (se hará en step con todos los módulos)

    def step(self) -> Dict:
        """
        Un paso de autopoiesis.

        Actualiza W_mod basándose en salud de módulos.
        """
        self.t += 1

        # Calcular scores de salud
        delta_sagis = np.array([self.module_health[m].delta_sagi for m in self.modules])
        stabilities = np.array([self.module_health[m].stability for m in self.modules])
        collapses = np.array([self.module_health[m].collapses for m in self.modules])

        # Ranks
        delta_sagi_ranks = self._compute_ranks(delta_sagis)
        stability_ranks = self._compute_ranks(stabilities)
        collapse_ranks = self._compute_ranks(collapses)

        # H_i = rank(ΔSAGI) + rank(stability) - rank(collapses)
        H = delta_sagi_ranks + stability_ranks - collapse_ranks

        # Normalizar H a [0, 1]
        H = (H - H.min()) / (H.max() - H.min() + 1e-10)

        # Guardar scores
        for i, m in enumerate(self.modules):
            self.module_health[m].score = H[i]
            self.health_history[m].append(H[i])

        # Regla autopoética
        eta = self._compute_learning_rate()

        for i in range(self.n):
            # ΔW_mod,ij = η * rank(H_i) * (W_mod,ij - mean_j W_mod,ij)
            row_mean = self.W_mod[i, :].mean()
            delta_W = eta * H[i] * (self.W_mod[i, :] - row_mean)
            self.W_mod[i, :] = self.W_mod[i, :] + delta_W

        # Normalizar filas
        for i in range(self.n):
            self.W_mod[i, :] = np.clip(self.W_mod[i, :], 0.01, None)
            row_sum = self.W_mod[i, :].sum() + 1e-10
            self.W_mod[i, :] = self.W_mod[i, :] / row_sum

        self.W_history.append(self.W_mod.copy())

        # Métricas
        out_degrees = self.W_mod.sum(axis=1)  # Cuánto influye cada módulo

        return {
            't': self.t,
            'W_mod': self.W_mod.copy(),
            'health_scores': H.copy(),
            'out_degrees': out_degrees,
            'top_module': self.modules[np.argmax(H)],
            'bottom_module': self.modules[np.argmin(H)]
        }

    def get_influence_ranking(self) -> List[Tuple[str, float]]:
        """Retorna módulos ordenados por influencia total."""
        out_degrees = self.W_mod.sum(axis=1)
        ranking = [(self.modules[i], out_degrees[i]) for i in range(self.n)]
        return sorted(ranking, key=lambda x: -x[1])

    def get_health_ranking(self) -> List[Tuple[str, float]]:
        """Retorna módulos ordenados por salud."""
        ranking = [(m, self.module_health[m].score) for m in self.modules]
        return sorted(ranking, key=lambda x: -x[1])


def test_R15_go_nogo(apl: AutopoiesisLight,
                     sagi_before: float, sagi_after: float,
                     n_nulls: int = 100) -> Dict:
    """
    Tests GO/NO-GO para R15.

    GO si:
    1. W_mod NO es constante: var(W_mod,ij(t)) > p95(null)
    2. Módulos con mayor H tienen mayor out-degree:
       corr(H_i, Σ_j W_mod,ij) > p95(null)
    3. SAGI mejora: SAGI_after > SAGI_before
    """
    results = {'passed': [], 'failed': []}

    # Test 1: W_mod no constante
    if len(apl.W_history) > 20:
        W_array = np.array(apl.W_history)
        var_real = np.var(W_array, axis=0).mean()

        # Nulos: W aleatorias
        null_vars = []
        for _ in range(n_nulls):
            W_random = np.random.dirichlet(np.ones(apl.n), size=(len(apl.W_history), apl.n))
            null_vars.append(np.var(W_random, axis=0).mean())

        p95 = np.percentile(null_vars, 95)

        if var_real > p95:
            results['passed'].append('W_mod_changes')
        else:
            results['failed'].append('W_mod_changes')
    else:
        results['failed'].append('insufficient_history')

    # Test 2: Correlación H vs out-degree
    H = np.array([apl.module_health[m].score for m in apl.modules])
    out_degrees = apl.W_mod.sum(axis=1)

    corr_real = np.corrcoef(H, out_degrees)[0, 1]
    if np.isnan(corr_real):
        corr_real = 0

    # Nulos
    null_corrs = []
    for _ in range(n_nulls):
        H_shuffled = H.copy()
        np.random.shuffle(H_shuffled)
        c = np.corrcoef(H_shuffled, out_degrees)[0, 1]
        null_corrs.append(c if not np.isnan(c) else 0)

    p95 = np.percentile(null_corrs, 95)

    if corr_real > p95:
        results['passed'].append('health_predicts_influence')
    else:
        results['failed'].append('health_predicts_influence')

    # Test 3: SAGI mejora
    if sagi_after > sagi_before:
        results['passed'].append('sagi_improved')
    else:
        results['failed'].append(f'sagi_not_improved_{sagi_after-sagi_before:.3f}')

    results['is_go'] = len(results['failed']) == 0
    results['summary'] = "GO" if results['is_go'] else f"NO-GO: {results['failed']}"

    return results


if __name__ == "__main__":
    print("R15 – Autopoiesis Light")
    print("=" * 50)

    apl = AutopoiesisLight()

    np.random.seed(42)

    for t in range(100):
        # Simular Transfer Entropy (aleatorio pero con estructura)
        TE = np.random.rand(apl.n, apl.n) * 0.5
        # Algunos módulos tienen más TE
        TE[0, :] += 0.2  # drives influye más
        TE[4, :] += 0.15  # irreversibilidad influye

        apl.update_transfer_entropy(TE)

        # Simular salud de módulos
        for i, m in enumerate(apl.modules):
            delta_sagi = np.random.randn() * 0.1 + (0.05 if i < 4 else -0.02)
            stability = 0.5 + np.random.randn() * 0.1 + (0.1 if i in [0, 3] else 0)
            collapses = max(0, int(np.random.randn() + (2 if i > 6 else 0)))

            apl.update_module_health(m, delta_sagi, stability, collapses)

        info = apl.step()

        if t % 25 == 0:
            print(f"\nt={t}")
            print(f"Top módulo: {info['top_module']}")
            print(f"Bottom módulo: {info['bottom_module']}")
            print(f"Ranking influencia: {apl.get_influence_ranking()[:3]}")

    # Test
    results = test_R15_go_nogo(apl, 0.5, 0.55)
    print(f"\n{results['summary']}")
    print(f"Passed: {results['passed']}")
    print(f"\nRanking final de salud:")
    for m, h in apl.get_health_ranking():
        print(f"  {m}: {h:.3f}")
