"""
SX9 - Robustez de Símbolos (Symbol Robustness)
==============================================

Mide: SymScore se mantiene bajo perturbaciones

Los símbolos son robustos ante ruido y cambios de contexto.
"""

import numpy as np
import sys
sys.path.insert(0, '/root/NEO_EVA')

from symbolic import SymbolExtractor, SymbolAlphabet, Symbol, SymbolStats
from cognition.agi_dynamic_constants import L_t, compute_adaptive_percentile


def run_test() -> dict:
    """
    SX9: Symbol Robustness Test

    Returns dict with:
      - score: [0,1] overall score
      - passed: bool
      - details: dict with metrics
    """
    np.random.seed(42)

    extractor = SymbolExtractor('TEST_AGENT', state_dim=6)
    alphabet = SymbolAlphabet('TEST_AGENT')

    state_dim = 6

    # Fase 1: Entrenamiento normal
    for t in range(1, 151):
        n_states = 10
        states = []
        for _ in range(n_states):
            cluster = np.random.randint(0, 5)
            state = np.random.randn(state_dim) * 0.3 + cluster * 0.5
            states.append(state)

        consequences = [np.random.randn(state_dim) * 0.1 for _ in states]

        for i, s in enumerate(states):
            extractor.observe_state(t, s, consequences[i])

        if t % 30 == 0:
            symbols = extractor.extract_symbols(t)
            activations = [(sid, sym.stats.sym_score) for sid, sym in symbols.items()]
            alphabet.update_alphabet(t, list(symbols.values()), dict(activations))

    # Guardar scores pre-perturbación
    symbols_pre = extractor.extract_symbols(150)
    scores_pre = {sid: sym.stats.sym_score for sid, sym in symbols_pre.items()}
    alphabet.update_alphabet(150, list(symbols_pre.values()), {sid: sym.stats.sym_score for sid, sym in symbols_pre.items()})
    active_pre = set(alphabet.get_active_symbols(150))

    # Fase 2: Perturbación (ruido aumentado, distribución cambiada)
    for t in range(151, 251):
        n_states = 10
        states = []
        for _ in range(n_states):
            # Más ruido
            cluster = np.random.randint(0, 5)
            noise_scale = 0.8 if t < 200 else 0.5  # Ruido alto luego se reduce
            state = np.random.randn(state_dim) * noise_scale + cluster * 0.5
            states.append(state)

        # Consecuencias más ruidosas
        consequences = [np.random.randn(state_dim) * 0.3 for _ in states]

        for i, s in enumerate(states):
            extractor.observe_state(t, s, consequences[i])

        if t % 30 == 0:
            symbols = extractor.extract_symbols(t)
            activations = [(sid, sym.stats.sym_score) for sid, sym in symbols.items()]
            alphabet.update_alphabet(t, list(symbols.values()), dict(activations))

    # Guardar scores post-perturbación
    symbols_post = extractor.extract_symbols(250)
    scores_post = {sid: sym.stats.sym_score for sid, sym in symbols_post.items()}
    alphabet.update_alphabet(250, list(symbols_post.values()), {sid: sym.stats.sym_score for sid, sym in symbols_post.items()})
    active_post = set(alphabet.get_active_symbols(250))

    # Calcular métricas de robustez
    # 1. Retención de símbolos activos
    if active_pre:
        retention = len(active_pre.intersection(active_post)) / len(active_pre)
    else:
        retention = 0.0

    # 2. Estabilidad de SymScores
    score_changes = []
    for sid in scores_pre:
        if sid in scores_post:
            change = abs(scores_post[sid] - scores_pre[sid])
            score_changes.append(change)

    mean_score_change = np.mean(score_changes) if score_changes else 1.0

    # 3. Proporción de símbolos con score estable
    stable_count = sum(1 for c in score_changes if c < 0.2)
    stability_proportion = stable_count / len(score_changes) if score_changes else 0.0

    # 4. Correlación entre scores pre y post
    common_ids = set(scores_pre.keys()).intersection(scores_post.keys())
    if len(common_ids) > 3:
        pre_arr = np.array([scores_pre[sid] for sid in common_ids])
        post_arr = np.array([scores_post[sid] for sid in common_ids])
        if np.std(pre_arr) > 0 and np.std(post_arr) > 0:
            correlation = np.corrcoef(pre_arr, post_arr)[0, 1]
            correlation = 0 if np.isnan(correlation) else correlation
        else:
            correlation = 0
    else:
        correlation = 0

    # Score
    retention_score = retention
    stability_score = 1.0 - min(1.0, mean_score_change / 0.3)
    proportion_score = stability_proportion
    correlation_score = (correlation + 1) / 2

    score = 0.3 * retention_score + 0.3 * stability_score + 0.2 * proportion_score + 0.2 * correlation_score

    passed = retention >= 0.5 and mean_score_change < 0.25

    return {
        'score': float(np.clip(score, 0, 1)),
        'passed': bool(passed),
        'details': {
            'symbol_retention': float(retention),
            'mean_score_change': float(mean_score_change),
            'stability_proportion': float(stability_proportion),
            'score_correlation': float(correlation),
            'n_active_pre': int(len(active_pre)),
            'n_active_post': int(len(active_post)),
            'n_symbols_pre': int(len(scores_pre)),
            'n_symbols_post': int(len(scores_post)),
            'sample_changes': [float(c) for c in score_changes[:5]]
        }
    }


if __name__ == "__main__":
    result = run_test()
    print("=" * 60)
    print("SX9 - SYMBOL ROBUSTNESS TEST")
    print("=" * 60)
    print(f"Score: {result['score']:.3f}")
    print(f"Passed: {result['passed']}")
    print(f"\nDetails:")
    for k, v in result['details'].items():
        print(f"  {k}: {v}")
