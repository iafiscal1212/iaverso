"""
SX10 - Madurez Simbólica (Symbolic Maturity)
============================================

Mide: SYM_X = ponderación de SYM-1 a SYM-5

Score global de madurez del sistema simbólico.
"""

import numpy as np
import sys
sys.path.insert(0, '/root/NEO_EVA')

from symbolic import (
    SymbolExtractor, SymbolAlphabet, SymbolBindingManager,
    SymbolGrammar, SymbolGrounding, SymbolicCognitionUse,
    SymbolicAuditor, Symbol, SymbolStats
)
from cognition.agi_dynamic_constants import L_t, compute_adaptive_percentile, normalized_entropy


def run_test() -> dict:
    """
    SX10: Symbolic Maturity Test

    Returns dict with:
      - score: [0,1] overall score
      - passed: bool
      - details: dict with metrics
    """
    np.random.seed(42)

    # Crear todos los componentes
    extractor = SymbolExtractor('TEST_AGENT', state_dim=6)
    alphabet = SymbolAlphabet('TEST_AGENT')
    binding_manager = SymbolBindingManager('TEST_AGENT', max_order=3)
    grammar = SymbolGrammar('TEST_AGENT', n_roles=4)
    grounding = SymbolGrounding('TEST_AGENT', n_regimes=3)
    cognition = SymbolicCognitionUse('TEST_AGENT')
    auditor = SymbolicAuditor()

    state_dim = 6
    agents = ['TEST_AGENT', 'EVA', 'ALEX']

    # Entrenar sistema completo
    for t in range(1, 301):
        # Generar estados
        n_states = 8
        states = []
        for _ in range(n_states):
            cluster = np.random.randint(0, 5)
            state = np.random.randn(state_dim) * 0.3 + cluster * 0.5
            states.append(state)

        consequences = [np.random.randn(state_dim) * 0.1 for _ in states]

        for i, s in enumerate(states):
            extractor.observe_state(t, s, consequences[i])

        # Secuencia de símbolos
        symbol_ids = list(np.random.randint(0, 5, size=4))
        deltas = [np.random.randn(state_dim) * 0.1 for _ in symbol_ids]
        binding_manager.observe_sequence(t, symbol_ids, states[:4], deltas)

        # Observar gramática
        delta_v = np.random.randn() * 0.2
        delta_sagi = np.random.randn() * 0.1
        grammar.observe_symbol_sequence(t, symbol_ids, delta_v, delta_sagi)

        # Grounding
        regime = np.random.randint(0, 3)
        agents_present = list(np.random.choice(agents, 2, replace=False))
        for sym_id in symbol_ids:
            grounding.observe_symbol_in_context(
                t, sym_id, regime, agents_present, delta_v, delta_sagi
            )

        # Actualizar periódicamente
        if t % 30 == 0:
            symbols = extractor.extract_symbols(t)
            activations = [(sid, sym.stats.sym_score) for sid, sym in symbols.items()]
            alphabet.update_alphabet(t, list(symbols.values()), dict(activations))

            # Inferir roles
            effects = {sid: np.random.randn(4) * 0.5 for sid in symbols.keys()}
            grammar.infer_roles(symbols, effects)

            # Actualizar grounding
            grounding.update_grounding(symbols)

    # Obtener símbolos finales
    symbols = extractor.extract_symbols(300)
    alphabet.update_alphabet(300, list(symbols.values()), {sid: sym.stats.sym_score for sid, sym in symbols.items()})

    # Ejecutar auditoría completa
    audit_results = auditor.run_full_audit(
        t=300,
        extractor=extractor,
        alphabet=alphabet,
        binding_manager=binding_manager,
        grammar=grammar,
        grounding=grounding,
        cognition=cognition,
        symbols=symbols,
        delta_t=50
    )

    # Calcular SYM_X score
    sym_x_score = auditor.compute_sym_x_score()

    # Extraer métricas individuales
    individual_scores = {}
    individual_passed = {}
    for test_name, result in audit_results.items():
        individual_scores[test_name] = result.value
        individual_passed[test_name] = result.passed

    # Contar tests pasados
    n_passed = sum(1 for r in audit_results.values() if r.passed)
    n_total = len(audit_results)
    pass_rate = n_passed / n_total if n_total > 0 else 0

    # Estadísticas adicionales
    extractor_stats = extractor.get_statistics()
    alphabet_stats = alphabet.get_statistics()
    binding_stats = binding_manager.get_statistics()
    grammar_stats = grammar.get_statistics()
    grounding_stats = grounding.get_statistics()

    # Score final
    # Combinar SYM_X con pass rate
    score = 0.7 * sym_x_score + 0.3 * pass_rate

    # Passed si SYM_X > 0.5 y al menos 3/5 tests pasan
    passed = sym_x_score > 0.5 and n_passed >= 3

    return {
        'score': float(np.clip(score, 0, 1)),
        'passed': bool(passed),
        'details': {
            'sym_x_score': float(sym_x_score),
            'pass_rate': float(pass_rate),
            'tests_passed': int(n_passed),
            'tests_total': int(n_total),
            'individual_scores': {k: float(v) for k, v in individual_scores.items()},
            'individual_passed': individual_passed,
            'n_symbols': int(len(symbols)),
            'n_active': int(len(alphabet.get_active_symbols(300))),
            'n_bindings': int(binding_stats['total_bindings']),
            'n_rules': int(grammar_stats['n_rules']),
            'n_grounded': int(grounding_stats['well_grounded'])
        }
    }


if __name__ == "__main__":
    result = run_test()
    print("=" * 60)
    print("SX10 - SYMBOLIC MATURITY TEST")
    print("=" * 60)
    print(f"Score: {result['score']:.3f}")
    print(f"Passed: {result['passed']}")
    print(f"\nDetails:")
    for k, v in result['details'].items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")
