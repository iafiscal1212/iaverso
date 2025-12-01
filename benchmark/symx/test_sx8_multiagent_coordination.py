"""
SX8 - Coordinación Multi-Agente (Multi-Agent Coordination)
==========================================================

Mide: Overlap(Σ_A, Σ_B) > 0, Shared_symbols_used > 0

Los agentes comparten símbolos y los usan para coordinarse.
"""

import numpy as np
import sys
sys.path.insert(0, '/root/NEO_EVA')

from symbolic import SymbolExtractor, SymbolAlphabet, Symbol, SymbolStats
from cognition.agi_dynamic_constants import L_t, compute_adaptive_percentile


def run_test() -> dict:
    """
    SX8: Multi-Agent Coordination Test

    Returns dict with:
      - score: [0,1] overall score
      - passed: bool
      - details: dict with metrics
    """
    np.random.seed(42)

    # Crear sistemas simbólicos para múltiples agentes
    agents = ['NEO', 'EVA', 'ALEX']
    extractors = {a: SymbolExtractor(a, state_dim=6) for a in agents}
    alphabets = {a: SymbolAlphabet(a) for a in agents}

    state_dim = 6

    # Entrenar agentes con experiencias parcialmente compartidas
    for t in range(1, 201):
        # Evento compartido (todos los agentes lo observan)
        shared_event_prob = 0.4
        is_shared = np.random.rand() < shared_event_prob

        if is_shared:
            # Todos observan el mismo estado base
            base_state = np.random.randn(state_dim) * 0.5
            cluster = np.random.randint(0, 3)
            base_state += cluster * 0.5

            for agent in agents:
                # Variación perceptual por agente
                perceived = base_state + np.random.randn(state_dim) * 0.1
                consequence = np.random.randn(state_dim) * 0.1
                extractors[agent].observe_state(t, perceived, consequence)
        else:
            # Experiencias individuales
            for agent in agents:
                cluster = np.random.randint(0, 5)
                state = np.random.randn(state_dim) * 0.3 + cluster * 0.5
                consequence = np.random.randn(state_dim) * 0.1
                extractors[agent].observe_state(t, state, consequence)

        # Extraer símbolos periódicamente
        if t % 30 == 0:
            for agent in agents:
                symbols = extractors[agent].extract_symbols(t)
                activations = [(sid, sym.stats.sym_score) for sid, sym in symbols.items()]
                alphabets[agent].update_alphabet(t, list(symbols.values()), dict(activations))

    # Obtener símbolos finales
    final_symbols = {}
    final_active = {}
    for agent in agents:
        final_symbols[agent] = extractors[agent].extract_symbols(200)
        final_active[agent] = set(alphabets[agent].get_active_symbols(200))

    # Calcular overlap entre agentes
    overlaps = []
    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            # Overlap basado en similitud de centroides
            shared_count = 0
            total_comparisons = 0

            for sym1_id, sym1 in final_symbols[a1].items():
                for sym2_id, sym2 in final_symbols[a2].items():
                    # Comparar centroides
                    distance = np.linalg.norm(sym1.stats.mu - sym2.stats.mu)
                    if distance < 0.5:  # Umbral de similitud
                        shared_count += 1
                    total_comparisons += 1

            if total_comparisons > 0:
                overlap = shared_count / total_comparisons
                overlaps.append(overlap)

    # Calcular símbolos compartidos activos
    if len(agents) >= 2:
        shared_active = final_active[agents[0]]
        for agent in agents[1:]:
            # Buscar símbolos con centroides similares
            similar_ids = set()
            for sym1_id in final_active[agent]:
                if sym1_id in final_symbols[agent]:
                    sym1 = final_symbols[agent][sym1_id]
                    for sym0_id in final_active[agents[0]]:
                        if sym0_id in final_symbols[agents[0]]:
                            sym0 = final_symbols[agents[0]][sym0_id]
                            if np.linalg.norm(sym1.stats.mu - sym0.stats.mu) < 0.5:
                                similar_ids.add(sym1_id)
            # Actualizar shared
            shared_active = shared_active.intersection(similar_ids) if shared_active else set()
    else:
        shared_active = set()

    # Métricas
    mean_overlap = np.mean(overlaps) if overlaps else 0
    n_shared_active = len(shared_active)

    # Calcular uso de símbolos compartidos para coordinación
    coordination_score_base = 0.0
    if n_shared_active > 0:
        # Simular episodio de coordinación
        coordination_success = []
        for _ in range(20):
            # Cada agente elige un símbolo
            choices = {}
            for agent in agents:
                active = list(final_active[agent])
                if active:
                    choices[agent] = np.random.choice(active)
                else:
                    choices[agent] = -1

            # Éxito si eligen símbolos similares
            agent_list = list(choices.keys())
            if len(agent_list) >= 2:
                sym0 = choices[agent_list[0]]
                sym1 = choices[agent_list[1]]
                if sym0 >= 0 and sym1 >= 0:
                    if sym0 in final_symbols[agent_list[0]] and sym1 in final_symbols[agent_list[1]]:
                        d = np.linalg.norm(
                            final_symbols[agent_list[0]][sym0].stats.mu -
                            final_symbols[agent_list[1]][sym1].stats.mu
                        )
                        coordination_success.append(1 if d < 0.5 else 0)

        coordination_score_base = np.mean(coordination_success) if coordination_success else 0

    # Score
    overlap_score = min(1.0, mean_overlap / 0.1) if mean_overlap > 0 else 0
    shared_score = min(1.0, n_shared_active / 3.0)
    coord_score = coordination_score_base

    score = 0.4 * overlap_score + 0.3 * shared_score + 0.3 * coord_score

    passed = mean_overlap > 0.01 and n_shared_active >= 1

    return {
        'score': float(np.clip(score, 0, 1)),
        'passed': bool(passed),
        'details': {
            'mean_overlap': float(mean_overlap),
            'n_shared_active': int(n_shared_active),
            'coordination_score': float(coordination_score_base),
            'overlaps': [float(o) for o in overlaps],
            'symbols_per_agent': {a: len(final_symbols[a]) for a in agents},
            'active_per_agent': {a: len(final_active[a]) for a in agents}
        }
    }


if __name__ == "__main__":
    result = run_test()
    print("=" * 60)
    print("SX8 - MULTI-AGENT COORDINATION TEST")
    print("=" * 60)
    print(f"Score: {result['score']:.3f}")
    print(f"Passed: {result['passed']}")
    print(f"\nDetails:")
    for k, v in result['details'].items():
        print(f"  {k}: {v}")
