"""
SX6 - Compresión Narrativa (Narrative Compression)
==================================================

Mide: Compression_ratio = |episode| / |symbolic_summary|
      Fidelity = correlation(original_effects, reconstructed_effects)

Los episodios se comprimen eficientemente en resúmenes simbólicos.
"""

import numpy as np
import sys
sys.path.insert(0, '/root/NEO_EVA')

from symbolic import (
    SymbolExtractor, SymbolAlphabet, SymbolicCognitionUse,
    Symbol, SymbolStats
)
from cognition.agi_dynamic_constants import L_t, compute_adaptive_percentile


def run_test() -> dict:
    """
    SX6: Narrative Compression Test

    Returns dict with:
      - score: [0,1] overall score
      - passed: bool
      - details: dict with metrics
    """
    np.random.seed(42)

    extractor = SymbolExtractor('TEST_AGENT', state_dim=6)
    alphabet = SymbolAlphabet('TEST_AGENT')
    cognition = SymbolicCognitionUse('TEST_AGENT')

    state_dim = 6

    # Entrenar extractor y alfabeto
    for t in range(1, 201):
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

    # Obtener símbolos finales
    symbols = extractor.extract_symbols(200)
    alphabet.update_alphabet(200, list(symbols.values()), {sid: sym.stats.sym_score for sid, sym in symbols.items()})

    # Test de compresión narrativa
    compression_ratios = []
    fidelities = []

    for episode_id in range(20):
        # Generar episodio
        episode_length = np.random.randint(15, 30)
        episode_states = []
        episode_values = []
        episode_sagis = []

        for step in range(episode_length):
            cluster = np.random.randint(0, 5)
            state = np.random.randn(state_dim) * 0.3 + cluster * 0.5
            value = np.random.rand()
            sagi = np.random.rand() * 0.5

            episode_states.append(state)
            episode_values.append(value)
            episode_sagis.append(sagi)

        # Comprimir a símbolos
        narrative = cognition.summarize_episode_to_symbols(
            t=200 + episode_id,
            episode_states=episode_states,
            episode_values=episode_values,
            episode_sagis=episode_sagis,
            symbols=symbols
        )

        # Calcular ratio de compresión
        if narrative and narrative.symbol_sequence:
            compression_ratio = episode_length / len(narrative.symbol_sequence)
            compression_ratios.append(compression_ratio)

            # Fidelidad: comparar estadísticas
            # Usar varianza como proxy de fidelidad
            original_v_var = np.var(episode_values)
            original_sagi_var = np.var(episode_sagis)

            # Reconstrucción aproximada desde símbolos
            reconstructed_values = []
            for sym_id in narrative.symbol_sequence:
                if sym_id in symbols:
                    # Usar mu del símbolo como proxy de valor
                    reconstructed_values.append(np.mean(symbols[sym_id].stats.mu))

            if reconstructed_values:
                reconstructed_var = np.var(reconstructed_values)
                # Fidelidad basada en preservación de varianza
                fidelity = 1.0 - abs(original_v_var - reconstructed_var) / (original_v_var + 0.01)
                fidelity = np.clip(fidelity, 0, 1)
            else:
                fidelity = 0.0

            fidelities.append(fidelity)

    # Calcular métricas finales
    if compression_ratios:
        mean_compression = np.mean(compression_ratios)
        mean_fidelity = np.mean(fidelities)
    else:
        mean_compression = 1.0
        mean_fidelity = 0.0

    # Score
    # Buen ratio de compresión (al menos 2:1)
    compression_score = min(1.0, (mean_compression - 1) / 3.0)
    compression_score = max(0, compression_score)

    # Alta fidelidad
    fidelity_score = mean_fidelity

    score = 0.5 * compression_score + 0.5 * fidelity_score

    passed = mean_compression >= 2.0 and mean_fidelity >= 0.3

    return {
        'score': float(np.clip(score, 0, 1)),
        'passed': bool(passed),
        'details': {
            'mean_compression_ratio': float(mean_compression),
            'mean_fidelity': float(mean_fidelity),
            'n_episodes_compressed': int(len(compression_ratios)),
            'compression_ratios': [float(c) for c in compression_ratios[-5:]],
            'fidelities': [float(f) for f in fidelities[-5:]],
            'n_symbols_available': int(len(symbols)),
            'n_narratives_created': int(cognition.n_narratives)
        }
    }


if __name__ == "__main__":
    result = run_test()
    print("=" * 60)
    print("SX6 - NARRATIVE COMPRESSION TEST")
    print("=" * 60)
    print(f"Score: {result['score']:.3f}")
    print(f"Passed: {result['passed']}")
    print(f"\nDetails:")
    for k, v in result['details'].items():
        print(f"  {k}: {v}")
