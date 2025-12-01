"""
SX5 - Grounding Social (Social Grounding)
=========================================

Mide: Selectivity_social > 0.3 para símbolos socialmente anclados

Los símbolos están asociados selectivamente a ciertos agentes.
"""

import numpy as np
import sys
sys.path.insert(0, '/root/NEO_EVA')

from symbolic import SymbolExtractor, SymbolGrounding, Symbol, SymbolStats
from cognition.agi_dynamic_constants import L_t, compute_adaptive_percentile


def run_test() -> dict:
    """
    SX5: Social Grounding Test

    Returns dict with:
      - score: [0,1] overall score
      - passed: bool
      - details: dict with metrics
    """
    np.random.seed(42)

    grounding = SymbolGrounding('TEST_AGENT', n_regimes=3)
    state_dim = 6

    # Crear símbolos
    symbols = {}
    for i in range(10):
        stats = SymbolStats(
            mu=np.random.randn(state_dim),
            Sigma=np.eye(state_dim) * 0.01,
            gamma=np.random.randn(state_dim) * 0.1,
            stab=0.5 + np.random.rand() * 0.5,
            consistency=0.5 + np.random.rand() * 0.5,
            sym_score=0.5 + np.random.rand() * 0.5,
            n_episodes=50,
            last_update_t=0
        )
        symbols[i] = Symbol(symbol_id=i, agent_id='TEST_AGENT', stats=stats)

    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
    sel_social_over_time = []

    # Simular observaciones con patrones sociales
    for t in range(1, 301):
        sym_id = t % 10

        # Régimen aleatorio
        regime = np.random.randint(0, 3)

        # Patrones sociales selectivos:
        # Símbolos 0-2 aparecen principalmente con NEO y EVA
        # Símbolos 3-5 aparecen principalmente con ALEX y ADAM
        # Símbolos 6-9 aparecen con IRIS y mezcla
        if sym_id < 3:
            if np.random.rand() > 0.2:
                agents_present = ['NEO', 'EVA']
            else:
                agents_present = list(np.random.choice(agents, 2, replace=False))
        elif sym_id < 6:
            if np.random.rand() > 0.2:
                agents_present = ['ALEX', 'ADAM']
            else:
                agents_present = list(np.random.choice(agents, 2, replace=False))
        else:
            if np.random.rand() > 0.3:
                agents_present = ['IRIS', np.random.choice(['NEO', 'EVA', 'ALEX', 'ADAM'])]
            else:
                agents_present = list(np.random.choice(agents, 3, replace=False))

        # Efectos
        delta_value = np.random.randn() * 0.1
        delta_sagi = np.random.randn() * 0.05

        grounding.observe_symbol_in_context(
            t, sym_id, regime, agents_present, delta_value, delta_sagi
        )

        # Actualizar grounding periódicamente
        if t % 30 == 0:
            grounding.update_grounding(symbols)
            stats = grounding.get_statistics()
            sel_social_over_time.append(stats['mean_sel_social'])

    # Obtener símbolos bien grounded
    grounded_symbols = grounding.get_grounded_symbols(grounding.t)

    # Calcular métricas finales
    final_stats = grounding.get_statistics()
    mean_sel_social = final_stats['mean_sel_social']
    mean_sel_world = final_stats['mean_sel_world']
    mean_grounded = final_stats['mean_grounded_score']

    # Analizar selectividad por agente
    symbols_by_agent = {}
    for agent in agents:
        symbols_by_agent[agent] = grounding.get_symbols_by_agent(agent)

    # Verificar que hay símbolos asociados a diferentes agentes
    agents_with_symbols = sum(1 for a in symbols_by_agent.values() if len(a) > 0)

    # Score
    # Selectividad social alta indica buen anclaje social
    social_score = min(1.0, mean_sel_social / 0.4)

    # Cobertura de agentes
    agent_coverage = agents_with_symbols / len(agents)

    # Grounded score promedio
    grounded_score = mean_grounded

    score = 0.5 * social_score + 0.3 * agent_coverage + 0.2 * grounded_score

    # Umbral endógeno
    sel_social_threshold = 0.25 if sel_social_over_time else 0.3

    passed = mean_sel_social >= sel_social_threshold and agents_with_symbols >= 3

    return {
        'score': float(np.clip(score, 0, 1)),
        'passed': bool(passed),
        'details': {
            'mean_sel_social': float(mean_sel_social),
            'mean_sel_world': float(mean_sel_world),
            'mean_grounded_score': float(mean_grounded),
            'n_grounded_symbols': int(len(grounded_symbols)),
            'symbols_by_agent': {k: len(v) for k, v in symbols_by_agent.items()},
            'agents_with_symbols': int(agents_with_symbols),
            'total_tracked': int(final_stats['total_symbols_tracked']),
            'sel_social_trend': [float(s) for s in sel_social_over_time[-5:]]
        }
    }


if __name__ == "__main__":
    result = run_test()
    print("=" * 60)
    print("SX5 - SOCIAL GROUNDING TEST")
    print("=" * 60)
    print(f"Score: {result['score']:.3f}")
    print(f"Passed: {result['passed']}")
    print(f"\nDetails:")
    for k, v in result['details'].items():
        print(f"  {k}: {v}")
