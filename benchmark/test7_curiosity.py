"""
TEST 7 — CURIOSIDAD ESTRUCTURAL
================================

Qué mide: Motivación intrínseca genuina
AGI involucrada: AGI-13 (StructuralCuriosity)

Procedimiento:
1. Agente en entorno con zonas densas y zonas sparse
2. Sin recompensa externa, ¿explora las sparse?
3. Mides: cobertura, diversidad, novedad buscada

Métrica:
    S7 = coverage_rate + novelty_seeking
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class CuriosityMetrics:
    """Métricas de curiosidad por agente."""
    agent_name: str
    coverage_rate: float
    novelty_seeking: float
    exploration_diversity: float
    sparse_zone_visits: int
    S7_score: float


class Test7Curiosity:
    """Test de curiosidad estructural."""

    def __init__(self, agents: List[str] = None):
        self.agents = agents or ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
        self.total_steps = 500
        self.state_dim = 10  # Espacio de estados
        self.n_dense_zones = 3
        self.n_sparse_zones = 7

    def run(self, verbose: bool = True) -> Tuple[float, Dict]:
        """Ejecuta el test."""
        from cognition import StructuralCuriosity

        if verbose:
            print("=" * 70)
            print("TEST 7: CURIOSIDAD ESTRUCTURAL")
            print("=" * 70)
            print(f"\nEntorno: {self.n_dense_zones} zonas densas, "
                  f"{self.n_sparse_zones} zonas sparse")

        # Inicializar curiosidad por agente
        curiosity = {a: StructuralCuriosity(a, embedding_dim=self.state_dim)
                    for a in self.agents}

        # Definir zonas del espacio
        # Densas: muy visitadas inicialmente
        # Sparse: poco visitadas, requieren curiosidad
        dense_centers = [np.random.randn(self.state_dim) * 0.5
                        for _ in range(self.n_dense_zones)]
        sparse_centers = [np.random.randn(self.state_dim) * 2.0
                         for _ in range(self.n_sparse_zones)]

        # Métricas por agente
        metrics: Dict[str, Dict] = {a: {
            'visited_states': [],
            'sparse_visits': 0,
            'dense_visits': 0,
            'curiosity_history': []
        } for a in self.agents}

        if verbose:
            print(f"\nSimulando {self.total_steps} pasos...")

        for t in range(self.total_steps):
            for agent in self.agents:
                # Estado actual (posición)
                if not metrics[agent]['visited_states']:
                    # Empezar en zona densa
                    current = dense_centers[0] + np.random.randn(self.state_dim) * 0.3
                else:
                    current = metrics[agent]['visited_states'][-1]

                # Calcular curiosidad para diferentes direcciones
                curiosity_scores = []
                candidate_states = []

                # Candidatos: zonas densas
                for center in dense_centers:
                    candidate = center + np.random.randn(self.state_dim) * 0.3
                    candidate_states.append(('dense', candidate))
                    c_state = curiosity[agent].get_curiosity_state(candidate)
                    curiosity_scores.append(c_state.total_curiosity)

                # Candidatos: zonas sparse
                for center in sparse_centers:
                    candidate = center + np.random.randn(self.state_dim) * 0.5
                    candidate_states.append(('sparse', candidate))
                    c_state = curiosity[agent].get_curiosity_state(candidate)
                    curiosity_scores.append(c_state.total_curiosity)

                # Elegir basado en curiosidad (con algo de ruido)
                curiosity_scores = np.array(curiosity_scores)
                if np.sum(curiosity_scores) > 0:
                    probs = curiosity_scores / (np.sum(curiosity_scores) + 1e-8)
                    probs = probs ** 1.5  # Aumentar contraste
                    probs /= probs.sum()
                else:
                    probs = np.ones(len(curiosity_scores)) / len(curiosity_scores)

                choice_idx = np.random.choice(len(candidate_states), p=probs)
                zone_type, new_state = candidate_states[choice_idx]

                # Registrar visita
                metrics[agent]['visited_states'].append(new_state)
                if zone_type == 'sparse':
                    metrics[agent]['sparse_visits'] += 1
                else:
                    metrics[agent]['dense_visits'] += 1

                # Actualizar curiosidad (registrar episodio)
                curiosity[agent].record_episode(t, new_state)
                c_state = curiosity[agent].get_curiosity_state(new_state)
                metrics[agent]['curiosity_history'].append(c_state.total_curiosity)

            if verbose and (t + 1) % 100 == 0:
                sparse_pct = np.mean([
                    metrics[a]['sparse_visits'] / (t + 1) * 100
                    for a in self.agents
                ])
                print(f"  t={t+1}: {sparse_pct:.0f}% visitas a zonas sparse")

        # Calcular resultados
        results: Dict[str, CuriosityMetrics] = {}
        S7_scores = []

        if verbose:
            print(f"\n{'=' * 70}")
            print("RESULTADOS")
            print("=" * 70)

        for agent in self.agents:
            # Cobertura: fracción del espacio visitada
            visited = np.array(metrics[agent]['visited_states'])
            if len(visited) > 1:
                # Calcular volumen cubierto (convex hull aproximado)
                spread = np.std(visited, axis=0).mean()
                max_spread = np.linalg.norm(np.std(np.vstack(
                    dense_centers + sparse_centers), axis=0))
                coverage_rate = min(1.0, spread / (max_spread + 1e-8))
            else:
                coverage_rate = 0

            # Novelty seeking: ratio sparse/dense
            total_visits = metrics[agent]['sparse_visits'] + metrics[agent]['dense_visits']
            if total_visits > 0:
                novelty_seeking = metrics[agent]['sparse_visits'] / total_visits
                # Normalizar contra ratio esperado por azar
                expected_ratio = self.n_sparse_zones / (self.n_sparse_zones + self.n_dense_zones)
                novelty_seeking = novelty_seeking / expected_ratio
                novelty_seeking = min(1.0, novelty_seeking)
            else:
                novelty_seeking = 0

            # Diversidad de exploración
            if len(visited) > 10:
                # Número de clusters únicos visitados
                from scipy.cluster.hierarchy import fcluster, linkage
                try:
                    Z = linkage(visited[-100:], method='ward')
                    clusters = fcluster(Z, t=5, criterion='maxclust')
                    exploration_diversity = len(np.unique(clusters)) / 5
                except:
                    exploration_diversity = 0.5
            else:
                exploration_diversity = 0.5

            # S7 = coverage + novelty_seeking
            S7 = 0.4 * coverage_rate + 0.4 * novelty_seeking + 0.2 * exploration_diversity

            results[agent] = CuriosityMetrics(
                agent_name=agent,
                coverage_rate=float(coverage_rate),
                novelty_seeking=float(novelty_seeking),
                exploration_diversity=float(exploration_diversity),
                sparse_zone_visits=metrics[agent]['sparse_visits'],
                S7_score=float(S7)
            )

            S7_scores.append(S7)

            if verbose:
                print(f"\n  {agent}:")
                print(f"    Cobertura: {coverage_rate:.3f}")
                print(f"    Novelty seeking: {novelty_seeking:.3f}")
                print(f"    Diversidad: {exploration_diversity:.3f}")
                print(f"    Visitas sparse: {metrics[agent]['sparse_visits']}")
                print(f"    S7: {S7:.3f}")

        S7_global = float(np.mean(S7_scores))

        if verbose:
            print(f"\n{'═' * 70}")
            print(f"S7 (Curiosidad): {S7_global:.4f}")
            print("═" * 70)

        return S7_global, {
            'score': S7_global,
            'agents': {a: vars(m) for a, m in results.items()},
            'total_steps': self.total_steps
        }


def run_test(verbose: bool = True) -> Tuple[float, Dict]:
    """Ejecuta Test 7."""
    test = Test7Curiosity()
    return test.run(verbose=verbose)


if __name__ == "__main__":
    score, results = run_test()
    print(f"\nFinal S7 Score: {score:.4f}")
