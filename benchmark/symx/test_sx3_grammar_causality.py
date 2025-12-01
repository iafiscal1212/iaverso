"""
SX3 - Gramática con Efecto Causal (Grammar with Causal Effect)
==============================================================

Mide: Corr(role_seq_effect, actual_ΔV) > 0

Las reglas gramaticales predicen efectos causales reales.
"""

import numpy as np
import sys
sys.path.insert(0, '/root/NEO_EVA')

from symbolic import SymbolExtractor, SymbolGrammar, Symbol, SymbolStats
from cognition.agi_dynamic_constants import L_t, compute_adaptive_percentile


def run_test() -> dict:
    """
    SX3: Grammar Causality Test

    Returns dict with:
      - score: [0,1] overall score
      - passed: bool
      - details: dict with metrics
    """
    np.random.seed(42)

    grammar = SymbolGrammar('TEST_AGENT', n_roles=4)
    state_dim = 6

    # Crear símbolos con efectos característicos
    symbols = {}
    effects_by_symbol = {}

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

        # Efectos específicos por símbolo (para crear roles diferenciados)
        if i < 3:
            # Símbolos evaluativos (alto efecto en SAGI)
            effects_by_symbol[i] = np.array([0.8 + np.random.rand() * 0.2, 0.1, 0.1, 0.1])
        elif i < 6:
            # Símbolos operativos (alto efecto en V)
            effects_by_symbol[i] = np.array([0.1, 0.8 + np.random.rand() * 0.2, 0.1, 0.1])
        else:
            # Símbolos descriptivos (bajo efecto)
            effects_by_symbol[i] = np.array([0.1, 0.1, 0.1, 0.1]) + np.random.rand(4) * 0.1

    # Inferir roles
    grammar.infer_roles(symbols, effects_by_symbol)

    # Registrar secuencias con efectos correlacionados
    predicted_effects = []
    actual_effects_v = []
    actual_effects_sagi = []

    for t in range(1, 301):
        # Crear secuencia con patrón
        if t % 3 == 0:
            # Secuencia evaluativa -> operativa (debería tener alto efecto)
            sequence = [0, 3, 6]  # eval -> oper -> desc
            delta_v = 0.5 + np.random.randn() * 0.1
            delta_sagi = 0.6 + np.random.randn() * 0.1
        elif t % 3 == 1:
            # Secuencia descriptiva -> descriptiva (bajo efecto)
            sequence = [6, 7, 8]
            delta_v = 0.1 + np.random.randn() * 0.05
            delta_sagi = 0.1 + np.random.randn() * 0.05
        else:
            # Aleatorio
            sequence = list(np.random.randint(0, 10, size=3))
            delta_v = np.random.randn() * 0.3
            delta_sagi = np.random.randn() * 0.2

        grammar.observe_symbol_sequence(t, sequence, delta_v, delta_sagi)

        # Obtener predicción de la gramática para la secuencia
        if t > 50:
            role_seq = []
            for sym_id in sequence[:2]:  # bigrama
                if sym_id in grammar.roles:
                    role_seq.append(grammar.roles[sym_id].role_id)

            if len(role_seq) == 2:
                role_tuple = tuple(role_seq)
                pred_v, pred_sagi = grammar.predict_effect(role_tuple)
                predicted_effects.append(pred_v + pred_sagi)
                actual_effects_v.append(delta_v)
                actual_effects_sagi.append(delta_sagi)

    # Calcular correlación entre predicciones y efectos reales
    if len(predicted_effects) > 10:
        predicted = np.array(predicted_effects)
        actual_v = np.array(actual_effects_v)
        actual_sagi = np.array(actual_effects_sagi)
        actual_total = actual_v + actual_sagi

        # Correlación
        corr_v = np.corrcoef(predicted, actual_v)[0, 1] if np.std(predicted) > 0 and np.std(actual_v) > 0 else 0
        corr_sagi = np.corrcoef(predicted, actual_sagi)[0, 1] if np.std(predicted) > 0 and np.std(actual_sagi) > 0 else 0
        corr_total = np.corrcoef(predicted, actual_total)[0, 1] if np.std(predicted) > 0 and np.std(actual_total) > 0 else 0

        # Manejar NaN
        corr_v = 0 if np.isnan(corr_v) else corr_v
        corr_sagi = 0 if np.isnan(corr_sagi) else corr_sagi
        corr_total = 0 if np.isnan(corr_total) else corr_total
    else:
        corr_v = 0
        corr_sagi = 0
        corr_total = 0

    # Obtener estadísticas de gramática
    stats = grammar.get_statistics()
    strong_rules = grammar.get_strong_rules(grammar.t)

    # Score
    # Correlación positiva indica que la gramática predice efectos
    corr_score = (corr_total + 1) / 2  # Normalizar de [-1,1] a [0,1]

    # Bonus por tener reglas fuertes
    rules_score = min(1.0, len(strong_rules) / 5.0)

    # Bonus por diferenciación de roles
    role_dist = stats.get('role_distribution', {})
    n_roles_used = len([v for v in role_dist.values() if v > 0])
    role_diversity = n_roles_used / 4.0

    score = 0.5 * corr_score + 0.3 * rules_score + 0.2 * role_diversity

    passed = corr_total > 0 and len(strong_rules) >= 2

    return {
        'score': float(np.clip(score, 0, 1)),
        'passed': bool(passed),
        'details': {
            'corr_v': float(corr_v),
            'corr_sagi': float(corr_sagi),
            'corr_total': float(corr_total),
            'n_strong_rules': int(len(strong_rules)),
            'n_total_rules': int(stats['n_rules']),
            'role_distribution': role_dist,
            'mean_lift': float(stats['mean_lift']),
            'n_predictions': int(len(predicted_effects))
        }
    }


if __name__ == "__main__":
    result = run_test()
    print("=" * 60)
    print("SX3 - GRAMMAR CAUSALITY TEST")
    print("=" * 60)
    print(f"Score: {result['score']:.3f}")
    print(f"Passed: {result['passed']}")
    print(f"\nDetails:")
    for k, v in result['details'].items():
        print(f"  {k}: {v}")
