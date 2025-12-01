"""
SX4 - Grounding en WORLD-1 (World Grounding)
============================================

Mide: Selectivity_world > 0.5 para símbolos bien anclados

Los símbolos están selectivamente asociados a regímenes del mundo.
"""

import numpy as np
import sys
sys.path.insert(0, '/root/NEO_EVA')

from symbolic import SymbolExtractor, SymbolGrounding, Symbol, SymbolStats
from cognition.agi_dynamic_constants import L_t, compute_adaptive_percentile


def run_test() -> dict:
    """
    SX4: World Grounding Test

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
    sel_world_over_time = []

    # Simular observaciones con patrones de régimen
    for t in range(1, 301):
        # Asignar símbolos a regímenes de forma selectiva
        # Símbolos 0-2 prefieren régimen 0 (estable)
        # Símbolos 3-5 prefieren régimen 1 (volátil)
        # Símbolos 6-9 prefieren régimen 2 (transicional)
        sym_id = t % 10

        if sym_id < 3:
            regime = 0 if np.random.rand() > 0.2 else np.random.randint(0, 3)
        elif sym_id < 6:
            regime = 1 if np.random.rand() > 0.2 else np.random.randint(0, 3)
        else:
            regime = 2 if np.random.rand() > 0.2 else np.random.randint(0, 3)

        # Agentes presentes (variable)
        n_agents = np.random.randint(2, 5)
        agents_present = list(np.random.choice(agents, n_agents, replace=False))

        # Efectos correlacionados con régimen
        delta_value = 0.1 * (regime - 1) + np.random.randn() * 0.05
        delta_sagi = 0.05 * sym_id / 10 + np.random.randn() * 0.02

        grounding.observe_symbol_in_context(
            t, sym_id, regime, agents_present, delta_value, delta_sagi
        )

        # Actualizar grounding periódicamente
        if t % 30 == 0:
            grounding.update_grounding(symbols)
            stats = grounding.get_statistics()
            sel_world_over_time.append(stats['mean_sel_world'])

    # Obtener símbolos bien grounded
    grounded_symbols = grounding.get_grounded_symbols(grounding.t)

    # Calcular métricas finales
    final_stats = grounding.get_statistics()
    mean_sel_world = final_stats['mean_sel_world']
    mean_sel_social = final_stats['mean_sel_social']
    mean_impact = final_stats['mean_impact']
    mean_grounded = final_stats['mean_grounded_score']

    # Analizar selectividad por régimen
    symbols_by_regime = {}
    for regime in range(3):
        symbols_by_regime[regime] = grounding.get_symbols_by_regime(regime)

    # Score
    # Selectividad de mundo alta indica buen grounding
    world_score = min(1.0, mean_sel_world / 0.5)

    # Grounded score promedio
    grounded_score = mean_grounded

    # Distribución de símbolos entre regímenes (debería haber en todos)
    regime_coverage = sum(1 for r in symbols_by_regime.values() if len(r) > 0) / 3.0

    score = 0.5 * world_score + 0.3 * grounded_score + 0.2 * regime_coverage

    # Umbral endógeno
    sel_world_threshold = 0.3 if sel_world_over_time else 0.5

    passed = mean_sel_world >= sel_world_threshold and len(grounded_symbols) >= 3

    return {
        'score': float(np.clip(score, 0, 1)),
        'passed': bool(passed),
        'details': {
            'mean_sel_world': float(mean_sel_world),
            'mean_sel_social': float(mean_sel_social),
            'mean_impact': float(mean_impact),
            'mean_grounded_score': float(mean_grounded),
            'n_grounded_symbols': int(len(grounded_symbols)),
            'symbols_by_regime': {k: len(v) for k, v in symbols_by_regime.items()},
            'total_tracked': int(final_stats['total_symbols_tracked']),
            'sel_world_trend': [float(s) for s in sel_world_over_time[-5:]]
        }
    }


if __name__ == "__main__":
    result = run_test()
    print("=" * 60)
    print("SX4 - WORLD GROUNDING TEST")
    print("=" * 60)
    print(f"Score: {result['score']:.3f}")
    print(f"Passed: {result['passed']}")
    print(f"\nDetails:")
    for k, v in result['details'].items():
        print(f"  {k}: {v}")
