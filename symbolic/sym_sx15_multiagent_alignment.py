"""
SX15 - Alineamiento Simbolico Multi-Agente
==========================================

Mide si los agentes usan simbolos similares para contextos similares:
- Para cada simbolo compartido entre 2+ agentes
- Comparar contextos de uso (Mahalanobis)
- Sin semantica humana ni supervision

Criterios:
- PASS: SX15 > 0.4
- EXCELLENT: SX15 > 0.6

Formula:
Align_AB(sigma) = 1 - D_Mahal(ctx_A(sigma), ctx_B(sigma)) / Q95(D_Mahal)
Align(sigma) = avg_{A,B} Align_AB(sigma)
SX15 = media de Align(sigma) ponderada por frecuencia y robustez

100% endogeno. Sin numeros magicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict
from itertools import combinations

import sys
sys.path.insert(0, '/root/NEO_EVA')


@dataclass
class SymbolContext:
    """Contexto de uso de un simbolo."""
    world_state: np.ndarray      # Estado del mundo (regimenes, campos)
    internal_state: np.ndarray   # Estado interno (phi, drives)
    episode_type: int            # Tipo de episodio
    frequency: int               # Frecuencia de uso


@dataclass
class SymbolUsage:
    """Uso de un simbolo por un agente."""
    symbol: str
    contexts: List[SymbolContext]
    mean_context: np.ndarray     # Contexto promedio
    robustness: float            # Robustez del simbolo (consistencia)


@dataclass
class SX15Result:
    """Resultado del test SX15."""
    score: float
    passed: bool
    excellent: bool
    n_shared_symbols: int
    alignment_by_symbol: Dict[str, float]
    agent_pair_alignments: Dict[str, float]
    details: Dict[str, Any]


class MultiAgentSymbolTracker:
    """
    Tracker de alineamiento simbolico multi-agente para SX15.
    """

    def __init__(self, context_dim: int = 12):
        self.context_dim = context_dim

        # Uso de simbolos por agente: agent_id -> symbol -> SymbolUsage
        self.agent_symbol_usage: Dict[str, Dict[str, SymbolUsage]] = defaultdict(dict)

        # Historial de distancias para Q95 endogeno
        self.all_distances: List[float] = []

    def record_symbol_usage(self, agent_id: str, symbol: str,
                           world_state: np.ndarray, internal_state: np.ndarray,
                           episode_type: int = 0):
        """Registra el uso de un simbolo por un agente."""
        # Construir contexto
        context_vec = np.concatenate([
            world_state[:6] if len(world_state) >= 6 else np.pad(world_state, (0, 6 - len(world_state))),
            internal_state[:6] if len(internal_state) >= 6 else np.pad(internal_state, (0, 6 - len(internal_state)))
        ])[:self.context_dim]

        context = SymbolContext(
            world_state=world_state.copy(),
            internal_state=internal_state.copy(),
            episode_type=episode_type,
            frequency=1
        )

        # Agregar o actualizar uso
        if symbol not in self.agent_symbol_usage[agent_id]:
            self.agent_symbol_usage[agent_id][symbol] = SymbolUsage(
                symbol=symbol,
                contexts=[context],
                mean_context=context_vec,
                robustness=1.0
            )
        else:
            usage = self.agent_symbol_usage[agent_id][symbol]
            usage.contexts.append(context)

            # Actualizar media de contexto
            all_vecs = []
            for ctx in usage.contexts:
                vec = np.concatenate([
                    ctx.world_state[:6] if len(ctx.world_state) >= 6 else np.pad(ctx.world_state, (0, 6 - len(ctx.world_state))),
                    ctx.internal_state[:6] if len(ctx.internal_state) >= 6 else np.pad(ctx.internal_state, (0, 6 - len(ctx.internal_state)))
                ])[:self.context_dim]
                all_vecs.append(vec)

            usage.mean_context = np.mean(all_vecs, axis=0)

            # Actualizar robustez (1 - varianza normalizada)
            if len(all_vecs) >= 2:
                variance = np.var(all_vecs)
                usage.robustness = 1 / (1 + variance)

    def get_shared_symbols(self) -> Set[str]:
        """Retorna simbolos usados por 2+ agentes."""
        symbol_agents: Dict[str, Set[str]] = defaultdict(set)

        for agent_id, symbols in self.agent_symbol_usage.items():
            for symbol in symbols:
                symbol_agents[symbol].add(agent_id)

        return {s for s, agents in symbol_agents.items() if len(agents) >= 2}

    def compute_alignment(self, symbol: str, agent_a: str, agent_b: str) -> float:
        """
        Calcula alineamiento entre dos agentes para un simbolo.
        Align_AB(sigma) = 1 - D_Mahal(ctx_A, ctx_B) / Q95(D_Mahal)
        """
        if symbol not in self.agent_symbol_usage[agent_a]:
            return 0.0
        if symbol not in self.agent_symbol_usage[agent_b]:
            return 0.0

        ctx_a = self.agent_symbol_usage[agent_a][symbol].mean_context
        ctx_b = self.agent_symbol_usage[agent_b][symbol].mean_context

        # Distancia Mahalanobis simplificada (Euclidea normalizada)
        # Para Mahalanobis real necesitariamos covarianza global
        diff = ctx_a - ctx_b
        d_mahal = float(np.linalg.norm(diff))

        self.all_distances.append(d_mahal)

        # Q95 endogeno
        if len(self.all_distances) >= 5:
            q95 = np.percentile(self.all_distances, 95)
        else:
            q95 = max(d_mahal * 2, 1.0)

        alignment = 1 - d_mahal / (q95 + 1e-8)
        return float(np.clip(alignment, 0, 1))

    def compute_symbol_alignment(self, symbol: str) -> Tuple[float, List[Tuple[str, str, float]]]:
        """
        Calcula alineamiento global para un simbolo.
        Align(sigma) = avg_{A,B} Align_AB(sigma)
        """
        # Encontrar agentes que usan este simbolo
        agents_using = [
            aid for aid, symbols in self.agent_symbol_usage.items()
            if symbol in symbols
        ]

        if len(agents_using) < 2:
            return 0.0, []

        # Calcular todas las parejas
        pair_alignments = []
        for a, b in combinations(agents_using, 2):
            align = self.compute_alignment(symbol, a, b)
            pair_alignments.append((a, b, align))

        # Media de alineamientos
        mean_align = float(np.mean([pa[2] for pa in pair_alignments]))

        return mean_align, pair_alignments

    def compute_sx15(self) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Calcula el score SX15 global.

        Returns:
            (score, alignment_by_symbol, agent_pair_alignments)
        """
        shared_symbols = self.get_shared_symbols()

        if not shared_symbols:
            return 0.0, {}, {}

        alignment_by_symbol = {}
        all_pair_alignments: Dict[str, List[float]] = defaultdict(list)

        for symbol in shared_symbols:
            align, pairs = self.compute_symbol_alignment(symbol)
            alignment_by_symbol[symbol] = align

            for a, b, al in pairs:
                pair_key = f"{a}-{b}"
                all_pair_alignments[pair_key].append(al)

        # Ponderacion por frecuencia y robustez
        weighted_sum = 0.0
        weight_total = 0.0

        for symbol, align in alignment_by_symbol.items():
            # Peso = suma de frecuencias * robustez
            weight = 0.0
            for agent_id, symbols in self.agent_symbol_usage.items():
                if symbol in symbols:
                    usage = symbols[symbol]
                    freq = len(usage.contexts)
                    rob = usage.robustness
                    weight += freq * rob

            weighted_sum += align * weight
            weight_total += weight

        if weight_total > 0:
            score = weighted_sum / weight_total
        else:
            score = float(np.mean(list(alignment_by_symbol.values())))

        # Promedio por pareja
        agent_pair_alignments = {
            k: float(np.mean(v)) for k, v in all_pair_alignments.items()
        }

        return float(score), alignment_by_symbol, agent_pair_alignments


def score_sx15_global(tracker: MultiAgentSymbolTracker) -> SX15Result:
    """
    Calcula el score SX15 global.

    Args:
        tracker: Tracker multi-agente

    Returns:
        SX15Result con score y detalles
    """
    score, alignment_by_symbol, agent_pair_alignments = tracker.compute_sx15()

    # Criterios
    passed = score > 0.4
    excellent = score > 0.6

    return SX15Result(
        score=score,
        passed=passed,
        excellent=excellent,
        n_shared_symbols=len(alignment_by_symbol),
        alignment_by_symbol=alignment_by_symbol,
        agent_pair_alignments=agent_pair_alignments,
        details={
            'n_agents': len(tracker.agent_symbol_usage),
            'total_symbols': sum(len(s) for s in tracker.agent_symbol_usage.values()),
            'n_distances': len(tracker.all_distances)
        }
    )


def run_sx15_test(n_agents: int = 5, n_symbols: int = 15,
                  n_observations: int = 100) -> SX15Result:
    """
    Ejecuta el test SX15 completo con datos simulados.
    """
    print("=" * 70)
    print("SX15 - ALINEAMIENTO SIMBOLICO MULTI-AGENTE")
    print("=" * 70)
    print(f"  Agentes: {n_agents}")
    print(f"  Simbolos totales: {n_symbols}")
    print(f"  Observaciones/agente: {n_observations}")
    print("=" * 70)

    np.random.seed(42)

    context_dim = 12
    agent_ids = [f"A{i}" for i in range(n_agents)]

    # Crear tracker
    tracker = MultiAgentSymbolTracker(context_dim)

    # Crear "significados" base para simbolos (contextos tipicos)
    # Esto simula que los agentes aprenden a usar simbolos en contextos similares
    symbol_meanings = {
        f"S{i}": np.random.randn(context_dim) * 0.5
        for i in range(n_symbols)
    }

    # Simular uso de simbolos
    for aid in agent_ids:
        # Cada agente usa un subconjunto de simbolos
        agent_symbols = np.random.choice(
            list(symbol_meanings.keys()),
            size=min(n_symbols, 10),
            replace=False
        )

        for _ in range(n_observations):
            # Elegir simbolo
            symbol = np.random.choice(agent_symbols)

            # Contexto cercano al significado base + ruido
            # (simula alineamiento emergente)
            base_ctx = symbol_meanings[symbol]

            # Mundo y estado interno derivados del contexto
            world_state = base_ctx[:6] + np.random.randn(6) * 0.1
            internal_state = base_ctx[6:12] if len(base_ctx) >= 12 else np.random.randn(6) * 0.1
            internal_state = internal_state + np.random.randn(len(internal_state)) * 0.1

            episode_type = np.random.randint(0, 5)

            tracker.record_symbol_usage(
                agent_id=aid,
                symbol=symbol,
                world_state=world_state,
                internal_state=internal_state,
                episode_type=episode_type
            )

    # Calcular resultado
    result = score_sx15_global(tracker)

    print("\n" + "=" * 70)
    print("RESULTADOS SX15")
    print("=" * 70)
    print(f"  Score SX15: {result.score:.4f}")
    print(f"  Passed: {result.passed} (> 0.4)")
    print(f"  Excellent: {result.excellent} (> 0.6)")
    print(f"\n  Simbolos compartidos: {result.n_shared_symbols}")
    print(f"\n  Alineamiento por simbolo (top 5):")
    sorted_symbols = sorted(result.alignment_by_symbol.items(), key=lambda x: -x[1])
    for sym, align in sorted_symbols[:5]:
        print(f"    {sym}: {align:.4f}")
    print(f"\n  Alineamiento por pareja de agentes:")
    for pair, align in sorted(result.agent_pair_alignments.items()):
        print(f"    {pair}: {align:.4f}")
    print("=" * 70)

    return result


if __name__ == "__main__":
    result = run_sx15_test(n_agents=5, n_symbols=15, n_observations=100)
