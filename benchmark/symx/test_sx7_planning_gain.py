"""
SX7 - Ganancia en Planificación (Planning Gain)
===============================================

Mide: V(symbolic_plan) > V(random_actions)

Los planes simbólicos generan mayor valor que acciones aleatorias.
"""

import numpy as np
import sys
sys.path.insert(0, '/root/NEO_EVA')

from symbolic import (
    SymbolExtractor, SymbolAlphabet, SymbolBindingManager,
    SymbolicCognitionUse, Symbol, SymbolStats
)
from cognition.agi_dynamic_constants import L_t, compute_adaptive_percentile


def run_test() -> dict:
    """
    SX7: Planning Gain Test

    Returns dict with:
      - score: [0,1] overall score
      - passed: bool
      - details: dict with metrics
    """
    np.random.seed(42)

    extractor = SymbolExtractor('TEST_AGENT', state_dim=6)
    alphabet = SymbolAlphabet('TEST_AGENT')
    binding_manager = SymbolBindingManager('TEST_AGENT', max_order=3)
    cognition = SymbolicCognitionUse('TEST_AGENT')

    state_dim = 6

    # Entrenar sistema simbólico
    for t in range(1, 201):
        n_states = 8
        states = []
        for _ in range(n_states):
            cluster = np.random.randint(0, 5)
            state = np.random.randn(state_dim) * 0.3 + cluster * 0.5
            states.append(state)

        consequences = [np.random.randn(state_dim) * 0.1 for _ in states]

        for i, s in enumerate(states):
            extractor.observe_state(t, s, consequences[i])

        # Observar secuencias en binding manager
        symbol_ids = list(np.random.randint(0, 5, size=4))
        deltas = [np.random.randn(state_dim) * 0.1 for _ in symbol_ids]
        binding_manager.observe_sequence(t, symbol_ids, states[:4], deltas)

        if t % 30 == 0:
            symbols = extractor.extract_symbols(t)
            activations = [(sid, sym.stats.sym_score) for sid, sym in symbols.items()]
            alphabet.update_alphabet(t, list(symbols.values()), dict(activations))

    # Obtener símbolos finales
    symbols = extractor.extract_symbols(200)
    alphabet.update_alphabet(200, list(symbols.values()), {sid: sym.stats.sym_score for sid, sym in symbols.items()})

    # Simular función de valor para símbolos
    # Algunos símbolos tienen valor alto, otros bajo
    symbol_values = {}
    for sym_id in symbols:
        if sym_id < 2:
            symbol_values[sym_id] = 0.8 + np.random.rand() * 0.2  # Alto valor
        elif sym_id < 4:
            symbol_values[sym_id] = 0.4 + np.random.rand() * 0.2  # Medio valor
        else:
            symbol_values[sym_id] = 0.1 + np.random.rand() * 0.2  # Bajo valor

    # Test de planificación
    symbolic_plan_values = []
    random_plan_values = []

    for trial in range(30):
        # Estado inicial aleatorio
        current_state = np.random.randn(state_dim) * 0.5

        # Generar plan simbólico
        plans = cognition.symbolic_plan_candidates(
            t=200 + trial,
            current_state=current_state,
            goal_state=np.zeros(state_dim),  # Meta: estado neutro
            symbols=symbols,
            max_depth=5,
            n_candidates=3
        )

        # Evaluar plan simbólico
        if plans:
            best_plan = plans[0]
            plan_value = sum(symbol_values.get(sym_id, 0.3) for sym_id in best_plan.symbol_sequence)
            plan_value /= len(best_plan.symbol_sequence) if best_plan.symbol_sequence else 1
            symbolic_plan_values.append(plan_value)
        else:
            symbolic_plan_values.append(0.3)

        # Comparar con plan aleatorio
        random_plan_length = 5
        random_symbols = list(np.random.choice(list(symbol_values.keys()), random_plan_length, replace=True))
        random_value = sum(symbol_values.get(sym_id, 0.3) for sym_id in random_symbols) / random_plan_length
        random_plan_values.append(random_value)

    # Calcular métricas
    mean_symbolic_value = np.mean(symbolic_plan_values)
    mean_random_value = np.mean(random_plan_values)
    gain = mean_symbolic_value - mean_random_value

    # Proporción de veces que plan simbólico es mejor
    n_better = sum(1 for s, r in zip(symbolic_plan_values, random_plan_values) if s > r)
    proportion_better = n_better / len(symbolic_plan_values)

    # Score
    # Ganancia positiva indica que planes simbólicos son mejores
    gain_score = min(1.0, (gain + 0.2) / 0.4) if gain > -0.2 else 0.0
    gain_score = max(0, gain_score)

    # Proporción de victorias
    proportion_score = proportion_better

    score = 0.6 * gain_score + 0.4 * proportion_score

    passed = gain > 0 and proportion_better > 0.5

    return {
        'score': float(np.clip(score, 0, 1)),
        'passed': bool(passed),
        'details': {
            'mean_symbolic_value': float(mean_symbolic_value),
            'mean_random_value': float(mean_random_value),
            'planning_gain': float(gain),
            'proportion_better': float(proportion_better),
            'n_trials': int(len(symbolic_plan_values)),
            'n_plans_generated': int(cognition.n_plans),
            'symbolic_values_sample': [float(v) for v in symbolic_plan_values[-5:]],
            'random_values_sample': [float(v) for v in random_plan_values[-5:]]
        }
    }


if __name__ == "__main__":
    result = run_test()
    print("=" * 60)
    print("SX7 - PLANNING GAIN TEST")
    print("=" * 60)
    print(f"Score: {result['score']:.3f}")
    print(f"Passed: {result['passed']}")
    print(f"\nDetails:")
    for k, v in result['details'].items():
        print(f"  {k}: {v}")
