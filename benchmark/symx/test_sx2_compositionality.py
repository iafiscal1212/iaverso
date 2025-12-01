"""
SX2 - Composicionalidad (Compositionality)
==========================================

Mide: Lift(bigramas) > 1, ΔCons(útiles) > 0

Los símbolos se componen en frases con lift significativo
y consecuencia conjunta no reducible a individuales.
"""

import numpy as np
import sys
sys.path.insert(0, '/root/NEO_EVA')

from symbolic import SymbolExtractor, SymbolBindingManager
from cognition.agi_dynamic_constants import L_t, compute_adaptive_percentile


def run_test() -> dict:
    """
    SX2: Compositionality Test

    Returns dict with:
      - score: [0,1] overall score
      - passed: bool
      - details: dict with metrics
    """
    np.random.seed(42)

    # Crear extractor y manager de bindings
    extractor = SymbolExtractor('TEST_AGENT', state_dim=6)
    binding_manager = SymbolBindingManager('TEST_AGENT', max_order=3)

    state_dim = 6
    lift_over_time = []
    delta_cons_over_time = []

    # Simular episodios con patrones recurrentes
    for t in range(1, 301):
        # Crear secuencia de estados con patrón
        if t % 4 == 0:
            # Patrón A: símbolos 0, 1, 2 frecuentemente juntos
            symbol_ids = [0, 1, 2, np.random.randint(3, 6)]
        elif t % 4 == 1:
            # Patrón B: símbolos 1, 3, 4
            symbol_ids = [1, 3, 4, np.random.randint(0, 3)]
        elif t % 4 == 2:
            # Patrón C: símbolos 2, 4, 5
            symbol_ids = [2, 4, 5, np.random.randint(0, 2)]
        else:
            # Aleatorio
            symbol_ids = list(np.random.randint(0, 6, size=4))

        # Simular estados y deltas
        states = [np.random.randn(state_dim) * 0.3 for _ in symbol_ids]
        deltas = [np.random.randn(state_dim) * 0.1 for _ in symbol_ids]

        binding_manager.observe_sequence(t, symbol_ids, states, deltas)

        # Registrar métricas periódicamente
        if t % 20 == 0:
            stats = binding_manager.get_statistics()
            lift_over_time.append(stats['mean_lift'])
            delta_cons_over_time.append(stats['mean_delta_cons'])

    # Obtener bindings útiles
    useful_bindings = binding_manager.get_useful_bindings(binding_manager.t)
    bigrams = binding_manager.get_bindings_by_order(2)
    trigrams = binding_manager.get_bindings_by_order(3)

    # Calcular métricas finales
    if lift_over_time:
        final_lift = np.mean(lift_over_time[-5:])
        mean_delta_cons = np.mean(delta_cons_over_time[-5:]) if delta_cons_over_time else 0
    else:
        final_lift = 1.0
        mean_delta_cons = 0.0

    # Lift de bindings útiles
    useful_lifts = [b.lift for b in useful_bindings] if useful_bindings else [1.0]
    mean_useful_lift = np.mean(useful_lifts)

    # Delta consistencia de útiles
    useful_delta_cons = [b.delta_consistency for b in useful_bindings] if useful_bindings else [0.0]
    mean_useful_delta_cons = np.mean(useful_delta_cons)

    # Score
    # Lift > 1 es lo esperado para co-ocurrencias no aleatorias
    lift_score = min(1.0, (mean_useful_lift - 1.0) / 2.0) if mean_useful_lift > 1 else 0.0
    lift_score = max(0, lift_score)

    # Delta consistencia > 0 indica que el binding aporta
    delta_score = min(1.0, mean_useful_delta_cons + 0.5) if mean_useful_delta_cons > -0.5 else 0.0
    delta_score = max(0, delta_score)

    # Penalizar si hay muy pocos bindings útiles
    coverage_score = min(1.0, len(useful_bindings) / 10.0)

    score = 0.4 * lift_score + 0.3 * delta_score + 0.3 * coverage_score

    passed = mean_useful_lift > 1.0 and len(useful_bindings) >= 3

    return {
        'score': float(np.clip(score, 0, 1)),
        'passed': bool(passed),
        'details': {
            'mean_useful_lift': float(mean_useful_lift),
            'mean_useful_delta_cons': float(mean_useful_delta_cons),
            'n_useful_bindings': int(len(useful_bindings)),
            'n_bigrams': int(len(bigrams)),
            'n_trigrams': int(len(trigrams)),
            'total_bindings': int(len(binding_manager.bindings)),
            'lift_trend': [float(l) for l in lift_over_time[-5:]],
            'delta_cons_trend': [float(d) for d in delta_cons_over_time[-5:]]
        }
    }


if __name__ == "__main__":
    result = run_test()
    print("=" * 60)
    print("SX2 - COMPOSITIONALITY TEST")
    print("=" * 60)
    print(f"Score: {result['score']:.3f}")
    print(f"Passed: {result['passed']}")
    print(f"\nDetails:")
    for k, v in result['details'].items():
        print(f"  {k}: {v}")
