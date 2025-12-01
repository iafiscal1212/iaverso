"""
SX1 - Riqueza Simbólica (Symbolic Richness)
============================================

Mide: |Σ_A(t)| / √t ≥ α_richness, Entropy(w) ≥ β_entropy

La riqueza simbólica crece con √t y mantiene entropía alta en pesos.
Todo endógeno.
"""

import numpy as np
import sys
sys.path.insert(0, '/root/NEO_EVA')

from symbolic import SymbolExtractor, SymbolAlphabet, Symbol, SymbolStats
from cognition.agi_dynamic_constants import L_t, compute_adaptive_percentile, normalized_entropy


def run_test() -> dict:
    """
    SX1: Symbolic Richness Test

    Returns dict with:
      - score: [0,1] overall score
      - passed: bool
      - details: dict with metrics
    """
    np.random.seed(42)

    # Crear extractor y alfabeto
    extractor = SymbolExtractor('TEST_AGENT', state_dim=6)
    alphabet = SymbolAlphabet('TEST_AGENT')

    # Simular episodios y extraer símbolos
    state_dim = 6
    richness_over_time = []
    entropy_over_time = []

    for t in range(1, 301):
        # Generar estados de episodio
        n_states = 10 + int(np.sqrt(t))
        states = []
        for _ in range(n_states):
            # Estados con estructura de clusters
            cluster = np.random.randint(0, 5)
            state = np.random.randn(state_dim) * 0.3 + cluster * 0.5
            states.append(state)

        # Consecuencias simuladas
        consequences = [np.random.randn(state_dim) * 0.1 for _ in states]

        # Observar en extractor
        for i, s in enumerate(states):
            extractor.observe_state(t, s, consequences[i])

        # Extraer símbolos periódicamente
        if t % 20 == 0:
            symbols = extractor.extract_symbols(t)

            # Actualizar alfabeto
            activations = []
            for sym_id, sym in symbols.items():
                activations.append((sym_id, sym.stats.sym_score))

            alphabet.update_alphabet(t, list(symbols.values()), dict(activations))

            # Medir riqueza
            active_symbols = alphabet.get_active_symbols(t)
            n_active = len(active_symbols)
            richness = n_active / np.sqrt(t + 1)
            richness_over_time.append(richness)

            # Medir entropía de pesos
            if active_symbols:
                weights = [alphabet.activations[s].weight for s in active_symbols if s in alphabet.activations]
                if weights and sum(weights) > 0:
                    weights = np.array(weights) / sum(weights)
                    entropy = normalized_entropy(weights)
                else:
                    entropy = 0.0
            else:
                entropy = 0.0

            entropy_over_time.append(entropy)

    # Calcular métricas finales
    if richness_over_time:
        final_richness = np.mean(richness_over_time[-5:])
        mean_entropy = np.mean(entropy_over_time[-5:]) if entropy_over_time else 0
    else:
        final_richness = 0
        mean_entropy = 0

    # Umbrales endógenos basados en percentiles
    if richness_over_time:
        alpha_richness = np.percentile(richness_over_time, 25)
        beta_entropy = np.percentile(entropy_over_time, 25) if entropy_over_time else 0.3
    else:
        alpha_richness = 0.5
        beta_entropy = 0.3

    # Score
    richness_score = min(1.0, final_richness / (alpha_richness + 0.5))
    entropy_score = min(1.0, mean_entropy / (beta_entropy + 0.3))
    score = 0.6 * richness_score + 0.4 * entropy_score

    passed = final_richness >= alpha_richness and mean_entropy >= beta_entropy

    return {
        'score': float(np.clip(score, 0, 1)),
        'passed': bool(passed),
        'details': {
            'final_richness': float(final_richness),
            'alpha_richness': float(alpha_richness),
            'mean_entropy': float(mean_entropy),
            'beta_entropy': float(beta_entropy),
            'n_symbols_final': int(len(alphabet.activations)),
            'richness_trend': [float(r) for r in richness_over_time[-5:]],
            'entropy_trend': [float(e) for e in entropy_over_time[-5:]]
        }
    }


if __name__ == "__main__":
    result = run_test()
    print("=" * 60)
    print("SX1 - SYMBOLIC RICHNESS TEST")
    print("=" * 60)
    print(f"Score: {result['score']:.3f}")
    print(f"Passed: {result['passed']}")
    print(f"\nDetails:")
    for k, v in result['details'].items():
        print(f"  {k}: {v}")
