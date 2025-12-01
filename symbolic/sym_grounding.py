"""
Symbolic Grounding: Anclaje de símbolos a WORLD-1
=================================================

Mide el 'grounding' estructural de símbolos respecto a:
- Regímenes del mundo (estable, volátil, transicional)
- Contexto social (qué agentes están presentes)
- Impacto real en cambios de estado

Todo endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from collections import defaultdict

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import (
    L_t, max_history, compute_adaptive_percentile, normalized_entropy
)

from symbolic.sym_atoms import Symbol


@dataclass
class SymbolGroundingStats:
    """Estadísticas de grounding de un símbolo."""
    symbol_id: int
    sel_world: float          # Selectividad de mundo [0,1]
    sel_social: float         # Selectividad social [0,1]
    impact: float             # Impacto estructural
    grounded_score: float     # Score total de grounding
    dominant_regime: int      # Régimen dominante donde aparece
    dominant_agents: List[str]  # Agentes con los que más co-ocurre
    last_update_t: int


class SymbolGrounding:
    """
    Mide el 'grounding' estructural de símbolos respecto a WORLD-1
    y a las dinámicas internas.
    """

    def __init__(self, agent_id: str, n_regimes: int = 3):
        self.agent_id = agent_id
        self.n_regimes = n_regimes

        # Estadísticas por símbolo
        self.stats_by_symbol: Dict[int, SymbolGroundingStats] = {}

        # Históricos para percentiles
        self.sel_world_hist: List[float] = []
        self.sel_social_hist: List[float] = []
        self.impact_hist: List[float] = []

        # Contadores por símbolo
        self.regime_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.agent_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.value_changes: Dict[int, List[float]] = defaultdict(list)
        self.sagi_changes: Dict[int, List[float]] = defaultdict(list)
        self.total_counts: Dict[int, int] = defaultdict(int)

        # Entropías base del sistema
        self.regime_entropy_base: float = np.log(n_regimes)
        self.agent_entropy_base: float = 0.0

        self.t = 0

    def observe_symbol_in_context(
        self,
        t: int,
        symbol_id: int,
        regime: int,
        agents_present: List[str],
        delta_value: float,
        delta_sagi: float,
    ) -> None:
        """
        Registra la aparición de un símbolo en un contexto específico.
        """
        self.t = t

        # Actualizar contadores
        self.regime_counts[symbol_id][regime] += 1
        for agent in agents_present:
            self.agent_counts[symbol_id][agent] += 1
        self.total_counts[symbol_id] += 1

        # Registrar cambios
        self.value_changes[symbol_id].append(delta_value)
        self.sagi_changes[symbol_id].append(delta_sagi)

        # Limitar históricos
        max_h = max_history(t)
        if len(self.value_changes[symbol_id]) > max_h:
            self.value_changes[symbol_id] = self.value_changes[symbol_id][-max_h:]
        if len(self.sagi_changes[symbol_id]) > max_h:
            self.sagi_changes[symbol_id] = self.sagi_changes[symbol_id][-max_h:]

        # Actualizar entropía base de agentes
        all_agents = set()
        for sym_agents in self.agent_counts.values():
            all_agents.update(sym_agents.keys())
        if len(all_agents) > 1:
            self.agent_entropy_base = np.log(len(all_agents))

    def update_grounding(
        self,
        symbols: Dict[int, Symbol],
    ) -> None:
        """
        Calcula selectividades e impacto para todos los símbolos registrados.
        """
        for symbol_id in self.total_counts.keys():
            if self.total_counts[symbol_id] < L_t(self.t):
                continue

            # Selectividad de mundo
            sel_world = self._compute_world_selectivity(symbol_id)

            # Selectividad social
            sel_social = self._compute_social_selectivity(symbol_id)

            # Impacto
            impact = self._compute_impact(symbol_id, symbols)

            # Score de grounding
            grounded_score = self._compute_grounded_score(sel_world, sel_social, impact)

            # Régimen dominante
            regime_dist = self.regime_counts[symbol_id]
            dominant_regime = max(regime_dist, key=regime_dist.get) if regime_dist else 0

            # Agentes dominantes
            agent_dist = self.agent_counts[symbol_id]
            sorted_agents = sorted(agent_dist.items(), key=lambda x: x[1], reverse=True)
            dominant_agents = [a for a, _ in sorted_agents[:3]]

            # Crear/actualizar stats
            self.stats_by_symbol[symbol_id] = SymbolGroundingStats(
                symbol_id=symbol_id,
                sel_world=sel_world,
                sel_social=sel_social,
                impact=impact,
                grounded_score=grounded_score,
                dominant_regime=dominant_regime,
                dominant_agents=dominant_agents,
                last_update_t=self.t
            )

            # Registrar históricos
            self.sel_world_hist.append(sel_world)
            self.sel_social_hist.append(sel_social)
            self.impact_hist.append(impact)

        # Limitar históricos globales
        max_h = max_history(self.t)
        if len(self.sel_world_hist) > max_h:
            self.sel_world_hist = self.sel_world_hist[-max_h:]
            self.sel_social_hist = self.sel_social_hist[-max_h:]
            self.impact_hist = self.impact_hist[-max_h:]

    def _compute_world_selectivity(self, symbol_id: int) -> float:
        """
        Selectividad de mundo: 1 - H(regimes|S_k) / H(regimes)
        Alta selectividad = símbolo concentrado en ciertos regímenes
        """
        regime_dist = self.regime_counts[symbol_id]
        if not regime_dist:
            return 0.5

        counts = np.array(list(regime_dist.values()), dtype=float)
        if counts.sum() == 0:
            return 0.5

        probs = counts / counts.sum()
        entropy_given = -np.sum(probs * np.log(probs + 1e-10))

        if self.regime_entropy_base > 0:
            selectivity = 1.0 - entropy_given / self.regime_entropy_base
        else:
            selectivity = 0.5

        return float(np.clip(selectivity, 0, 1))

    def _compute_social_selectivity(self, symbol_id: int) -> float:
        """
        Selectividad social: 1 - H(agents|S_k) / H(agents)
        Alta selectividad = símbolo aparece con ciertos agentes
        """
        agent_dist = self.agent_counts[symbol_id]
        if not agent_dist or self.agent_entropy_base <= 0:
            return 0.5

        counts = np.array(list(agent_dist.values()), dtype=float)
        if counts.sum() == 0:
            return 0.5

        probs = counts / counts.sum()
        entropy_given = -np.sum(probs * np.log(probs + 1e-10))

        selectivity = 1.0 - entropy_given / self.agent_entropy_base

        return float(np.clip(selectivity, 0, 1))

    def _compute_impact(self, symbol_id: int, symbols: Dict[int, Symbol]) -> float:
        """
        Impacto = Consistencia * Robustez
        Consistencia: de consecuencias del símbolo
        Robustez: si el símbolo existe en el diccionario, usar su stats
        """
        # Consistencia basada en variabilidad de efectos
        value_changes = self.value_changes.get(symbol_id, [])
        sagi_changes = self.sagi_changes.get(symbol_id, [])

        if not value_changes:
            return 0.5

        # Variabilidad normalizada
        value_std = np.std(value_changes)
        sagi_std = np.std(sagi_changes) if sagi_changes else 0

        # Percentil 95 para normalización
        all_values = []
        all_sagis = []
        for changes in self.value_changes.values():
            all_values.extend(changes)
        for changes in self.sagi_changes.values():
            all_sagis.extend(changes)

        p95_value = np.percentile(np.abs(all_values), 95) if all_values else 1.0
        p95_sagi = np.percentile(np.abs(all_sagis), 95) if all_sagis else 1.0

        consistency = 1.0 - (value_std / (p95_value + 1e-8) + sagi_std / (p95_sagi + 1e-8)) / 2
        consistency = np.clip(consistency, 0, 1)

        # Robustez del símbolo (si existe)
        robustness = 0.5
        if symbol_id in symbols:
            sym = symbols[symbol_id]
            robustness = sym.stats.sym_score

        impact = float(consistency * robustness)
        return impact

    def _compute_grounded_score(
        self,
        sel_world: float,
        sel_social: float,
        impact: float
    ) -> float:
        """
        Score total de grounding.
        Combina selectividades e impacto de forma endógena.
        """
        # Ponderación endógena basada en varianzas históricas
        if self.sel_world_hist and self.sel_social_hist and self.impact_hist:
            var_world = np.var(self.sel_world_hist[-L_t(self.t):])
            var_social = np.var(self.sel_social_hist[-L_t(self.t):])
            var_impact = np.var(self.impact_hist[-L_t(self.t):])

            total_var = var_world + var_social + var_impact + 1e-8

            # Peso inversamente proporcional a la varianza
            w_world = (1.0 / (var_world + 1e-8)) / (3.0 / total_var)
            w_social = (1.0 / (var_social + 1e-8)) / (3.0 / total_var)
            w_impact = (1.0 / (var_impact + 1e-8)) / (3.0 / total_var)

            # Normalizar
            w_sum = w_world + w_social + w_impact
            w_world /= w_sum
            w_social /= w_sum
            w_impact /= w_sum
        else:
            w_world = w_social = w_impact = 1.0 / 3.0

        score = w_world * sel_world + w_social * sel_social + w_impact * impact
        return float(np.clip(score, 0, 1))

    def get_grounding_stats(self, symbol_id: int) -> Optional[SymbolGroundingStats]:
        """Obtiene estadísticas de grounding de un símbolo."""
        return self.stats_by_symbol.get(symbol_id)

    def get_grounded_symbols(self, t: int) -> List[SymbolGroundingStats]:
        """Devuelve símbolos bien 'grounded' según scores y percentiles."""
        if not self.stats_by_symbol:
            return []

        # Umbral endógeno
        all_scores = [s.grounded_score for s in self.stats_by_symbol.values()]
        threshold = np.percentile(all_scores, 75)

        grounded = [s for s in self.stats_by_symbol.values() if s.grounded_score >= threshold]
        return sorted(grounded, key=lambda x: x.grounded_score, reverse=True)

    def get_symbols_by_regime(self, regime: int) -> List[int]:
        """Obtiene símbolos que aparecen predominantemente en un régimen."""
        symbols = []
        for sym_id, stats in self.stats_by_symbol.items():
            if stats.dominant_regime == regime:
                symbols.append(sym_id)
        return symbols

    def get_symbols_by_agent(self, agent_name: str) -> List[int]:
        """Obtiene símbolos que co-ocurren frecuentemente con un agente."""
        symbols = []
        for sym_id, stats in self.stats_by_symbol.items():
            if agent_name in stats.dominant_agents:
                symbols.append(sym_id)
        return symbols

    def get_statistics(self) -> Dict[str, Any]:
        """Estadísticas del sistema de grounding."""
        grounded = self.get_grounded_symbols(self.t)

        return {
            'agent_id': self.agent_id,
            't': self.t,
            'total_symbols_tracked': len(self.stats_by_symbol),
            'well_grounded': len(grounded),
            'mean_sel_world': np.mean(self.sel_world_hist) if self.sel_world_hist else 0,
            'mean_sel_social': np.mean(self.sel_social_hist) if self.sel_social_hist else 0,
            'mean_impact': np.mean(self.impact_hist) if self.impact_hist else 0,
            'mean_grounded_score': np.mean([s.grounded_score for s in self.stats_by_symbol.values()]) if self.stats_by_symbol else 0
        }


def test_symbol_grounding():
    """Test del sistema de grounding."""
    print("=" * 60)
    print("TEST: SYMBOL GROUNDING")
    print("=" * 60)

    from symbolic.sym_atoms import Symbol, SymbolStats

    grounding = SymbolGrounding('NEO', n_regimes=3)

    np.random.seed(42)

    # Crear símbolos
    symbols = {}
    for i in range(10):
        stats = SymbolStats(
            mu=np.random.randn(6),
            Sigma=np.eye(6) * 0.01,
            gamma=np.random.randn(6) * 0.1,
            stab=0.5 + np.random.rand() * 0.5,
            consistency=0.5 + np.random.rand() * 0.5,
            sym_score=0.5 + np.random.rand() * 0.5,
            n_episodes=50,
            last_update_t=0
        )
        symbols[i] = Symbol(symbol_id=i, agent_id='NEO', stats=stats)

    # Simular observaciones con patrones
    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']

    for t in range(300):
        # Símbolo con patrón de régimen
        sym_id = t % 10
        regime = sym_id % 3  # Símbolos 0,3,6,9 en régimen 0; 1,4,7 en régimen 1; etc.

        # Agentes con patrón
        n_agents = 2 + (sym_id % 3)
        agents_present = agents[:n_agents]

        # Efectos
        delta_v = 0.1 * (regime - 1) + np.random.randn() * 0.05
        delta_sagi = 0.05 * sym_id / 10 + np.random.randn() * 0.02

        grounding.observe_symbol_in_context(t, sym_id, regime, agents_present, delta_v, delta_sagi)

        if (t + 1) % 50 == 0:
            grounding.update_grounding(symbols)
            stats = grounding.get_statistics()
            print(f"\n  t={t+1}:")
            print(f"    Símbolos rastreados: {stats['total_symbols_tracked']}")
            print(f"    Bien grounded: {stats['well_grounded']}")
            print(f"    Sel. mundo media: {stats['mean_sel_world']:.3f}")
            print(f"    Sel. social media: {stats['mean_sel_social']:.3f}")
            print(f"    Impacto medio: {stats['mean_impact']:.3f}")

    print("\n" + "=" * 60)
    print("SÍMBOLOS BIEN GROUNDED")
    print("=" * 60)

    grounded = grounding.get_grounded_symbols(grounding.t)
    for gs in grounded[:5]:
        print(f"\nSímbolo {gs.symbol_id}:")
        print(f"  Grounded Score: {gs.grounded_score:.3f}")
        print(f"  Sel. Mundo: {gs.sel_world:.3f}")
        print(f"  Sel. Social: {gs.sel_social:.3f}")
        print(f"  Impacto: {gs.impact:.3f}")
        print(f"  Régimen dominante: {gs.dominant_regime}")
        print(f"  Agentes dominantes: {gs.dominant_agents}")

    print("\n" + "=" * 60)
    print("TEST COMPLETADO")
    print("=" * 60)

    return grounding


if __name__ == "__main__":
    test_symbol_grounding()
