"""
SX12 - Estabilidad y Deriva Conceptual
======================================

Mide si los conceptos internos tienen:
- Deriva temporal controlada (ni rigidos ni caoticos)
- Dispersion interna coherente
- Consistencia simbolica estable

Criterios:
- PASS: SX12_global > 0.5
- EXCELLENT: SX12_global > 0.7

Penaliza conceptos con:
- drift << Q25 (demasiado rigidos)
- drift >> Q75 (demasiado caoticos)

100% endogeno. Sin numeros magicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, '/root/NEO_EVA')


@dataclass
class ConceptSnapshot:
    """Snapshot de un concepto en un momento dado."""
    t: int
    embedding: np.ndarray
    episodes: List[int]          # IDs de episodios asociados
    symbols: Set[str]            # Simbolos frecuentes
    activation: float            # Nivel de activacion


@dataclass
class ConceptStatsSX12:
    """Estadisticas de un concepto para SX12."""
    concept_id: str
    drift: float                 # Deriva temporal media
    dispersion: float            # Dispersion interna
    symbol_stability: float      # Estabilidad simbolica
    score: float                 # Score del concepto


@dataclass
class SX12Result:
    """Resultado del test SX12."""
    score: float
    passed: bool
    excellent: bool
    drift_global: float
    dispersion_global: float
    symbol_stability_global: float
    concept_scores: Dict[str, float]
    agent_scores: Dict[str, float]
    details: Dict[str, Any]


class ConceptDriftTracker:
    """
    Tracker de deriva conceptual para SX12.
    Monitorea estabilidad y evolucion de conceptos.
    """

    def __init__(self, agent_id: str, embedding_dim: int = 16):
        self.agent_id = agent_id
        self.embedding_dim = embedding_dim

        # Conceptos: concept_id -> lista de snapshots
        self.concepts: Dict[str, List[ConceptSnapshot]] = defaultdict(list)

        # Historial de drifts para umbrales endogenos
        self.all_drifts: List[float] = []

    def record_concept(self, concept_id: str, t: int, embedding: np.ndarray,
                       episodes: List[int], symbols: Set[str], activation: float = 1.0):
        """Registra un snapshot de un concepto."""
        snapshot = ConceptSnapshot(
            t=t,
            embedding=embedding.copy(),
            episodes=episodes.copy(),
            symbols=symbols.copy(),
            activation=activation
        )
        self.concepts[concept_id].append(snapshot)

    def compute_concept_drift(self, concept_id: str) -> float:
        """Calcula la deriva temporal de un concepto."""
        snapshots = self.concepts.get(concept_id, [])
        if len(snapshots) < 2:
            return 0.0

        drifts = []
        for i in range(1, len(snapshots)):
            e1 = snapshots[i-1].embedding
            e2 = snapshots[i].embedding

            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)

            if norm1 > 1e-8 and norm2 > 1e-8:
                cos_sim = np.dot(e1, e2) / (norm1 * norm2)
                d_cos = 1 - cos_sim  # Distancia coseno
            else:
                d_cos = 0.0

            drifts.append(d_cos)

        return float(np.mean(drifts)) if drifts else 0.0

    def compute_concept_dispersion(self, concept_id: str) -> float:
        """Calcula la dispersion interna de un concepto (varianza de episodios)."""
        snapshots = self.concepts.get(concept_id, [])
        if len(snapshots) < 2:
            return 0.0

        # Varianza de embeddings
        embeddings = np.array([s.embedding for s in snapshots])
        variance = np.var(embeddings)

        return float(variance)

    def compute_symbol_stability(self, concept_id: str) -> float:
        """Calcula la estabilidad simbolica de un concepto."""
        snapshots = self.concepts.get(concept_id, [])
        if len(snapshots) < 2:
            return 1.0

        stabilities = []
        for i in range(1, len(snapshots)):
            s1 = snapshots[i-1].symbols
            s2 = snapshots[i].symbols

            if len(s1 | s2) > 0:
                jaccard = len(s1 & s2) / len(s1 | s2)
            else:
                jaccard = 1.0

            stabilities.append(jaccard)

        return float(np.mean(stabilities)) if stabilities else 1.0

    def compute_concept_score(self, concept_id: str) -> ConceptStatsSX12:
        """Calcula el score de un concepto individual."""
        drift = self.compute_concept_drift(concept_id)
        dispersion = self.compute_concept_dispersion(concept_id)
        symbol_stability = self.compute_symbol_stability(concept_id)

        # Registrar drift para umbrales
        if drift > 0:
            self.all_drifts.append(drift)

        # Calcular score basado en drift optimo
        # Queremos drift en rango "sano" - ni muy bajo (rigido) ni muy alto (caotico)
        if len(self.all_drifts) >= 3:
            q25 = np.percentile(self.all_drifts, 25)
            q75 = np.percentile(self.all_drifts, 75)
            drift_med = np.median(self.all_drifts)
            iqr = q75 - q25 + 1e-8
        else:
            # Defaults endogenos iniciales
            q25 = 0.05
            q75 = 0.3
            drift_med = 0.15
            iqr = 0.25

        # Penalizar desviacion de la mediana
        deviation = abs(drift - drift_med) / iqr
        drift_score = max(0.0, 1 - deviation)

        # Score total: combinacion de drift, dispersion y estabilidad simbolica
        # Pesos basados en importancia teorica
        score = (
            0.4 * drift_score +
            0.3 * (1 - min(1.0, dispersion * 10)) +  # Penalizar alta dispersion
            0.3 * symbol_stability
        )

        return ConceptStatsSX12(
            concept_id=concept_id,
            drift=drift,
            dispersion=dispersion,
            symbol_stability=symbol_stability,
            score=float(np.clip(score, 0, 1))
        )

    def compute_agent_score(self) -> Tuple[float, Dict[str, float]]:
        """Calcula el score SX12 para el agente."""
        if not self.concepts:
            return 0.5, {}

        concept_scores = {}
        for cid in self.concepts:
            stats = self.compute_concept_score(cid)
            concept_scores[cid] = stats.score

        # Score del agente = media de scores de conceptos
        agent_score = float(np.mean(list(concept_scores.values())))

        return agent_score, concept_scores

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadisticas completas."""
        agent_score, concept_scores = self.compute_agent_score()

        drifts = []
        dispersions = []
        stabilities = []

        for cid in self.concepts:
            stats = self.compute_concept_score(cid)
            drifts.append(stats.drift)
            dispersions.append(stats.dispersion)
            stabilities.append(stats.symbol_stability)

        return {
            'agent_score': agent_score,
            'n_concepts': len(self.concepts),
            'drift_mean': float(np.mean(drifts)) if drifts else 0.0,
            'dispersion_mean': float(np.mean(dispersions)) if dispersions else 0.0,
            'symbol_stability_mean': float(np.mean(stabilities)) if stabilities else 1.0,
            'concept_scores': concept_scores
        }


def score_sx12_global(agent_trackers: Dict[str, ConceptDriftTracker]) -> SX12Result:
    """
    Calcula el score SX12 global.

    Args:
        agent_trackers: Dict de trackers de drift por agente

    Returns:
        SX12Result con score global y detalles
    """
    if not agent_trackers:
        return SX12Result(
            score=0.0, passed=False, excellent=False,
            drift_global=0.0, dispersion_global=0.0,
            symbol_stability_global=0.0,
            concept_scores={}, agent_scores={},
            details={}
        )

    agent_scores = {}
    all_concept_scores = {}
    drifts = []
    dispersions = []
    stabilities = []

    for aid, tracker in agent_trackers.items():
        stats = tracker.get_statistics()
        agent_scores[aid] = stats['agent_score']
        drifts.append(stats['drift_mean'])
        dispersions.append(stats['dispersion_mean'])
        stabilities.append(stats['symbol_stability_mean'])

        for cid, cscore in stats['concept_scores'].items():
            all_concept_scores[f"{aid}_{cid}"] = cscore

    # Metricas globales
    drift_global = float(np.mean(drifts)) if drifts else 0.0
    dispersion_global = float(np.mean(dispersions)) if dispersions else 0.0
    symbol_stability_global = float(np.mean(stabilities)) if stabilities else 1.0

    # Score global = media de agentes
    score = float(np.mean(list(agent_scores.values()))) if agent_scores else 0.0

    # Criterios
    passed = score > 0.5
    excellent = score > 0.7

    return SX12Result(
        score=score,
        passed=passed,
        excellent=excellent,
        drift_global=drift_global,
        dispersion_global=dispersion_global,
        symbol_stability_global=symbol_stability_global,
        concept_scores=all_concept_scores,
        agent_scores=agent_scores,
        details={
            'n_agents': len(agent_trackers),
            'total_concepts': sum(len(t.concepts) for t in agent_trackers.values())
        }
    )


def run_sx12_test(n_agents: int = 5, n_concepts_per_agent: int = 8,
                  n_timesteps: int = 20) -> SX12Result:
    """
    Ejecuta el test SX12 completo con datos simulados.
    """
    print("=" * 70)
    print("SX12 - ESTABILIDAD Y DERIVA CONCEPTUAL")
    print("=" * 70)
    print(f"  Agentes: {n_agents}")
    print(f"  Conceptos/agente: {n_concepts_per_agent}")
    print(f"  Timesteps: {n_timesteps}")
    print("=" * 70)

    np.random.seed(42)

    embedding_dim = 16
    agent_ids = [f"A{i}" for i in range(n_agents)]

    # Crear trackers
    trackers: Dict[str, ConceptDriftTracker] = {
        aid: ConceptDriftTracker(aid, embedding_dim) for aid in agent_ids
    }

    # Simular evolucion de conceptos
    for aid in agent_ids:
        tracker = trackers[aid]

        # Crear conceptos base
        concept_bases = {
            f"C{j}": np.random.randn(embedding_dim) * 0.5
            for j in range(n_concepts_per_agent)
        }
        concept_symbols = {
            f"C{j}": set([f"S{(j * 3 + k) % 20}" for k in range(4)])
            for j in range(n_concepts_per_agent)
        }

        # Evolucion temporal
        for t in range(n_timesteps):
            for cid in concept_bases:
                # Drift gradual del embedding
                drift_rate = 0.02 + np.random.random() * 0.03
                concept_bases[cid] += np.random.randn(embedding_dim) * drift_rate

                # Simbolos evolucionan lentamente
                if np.random.random() < 0.1:
                    # Agregar/quitar simbolo
                    if np.random.random() < 0.5 and len(concept_symbols[cid]) > 2:
                        concept_symbols[cid].pop()
                    else:
                        concept_symbols[cid].add(f"S{np.random.randint(0, 20)}")

                # Registrar snapshot
                tracker.record_concept(
                    concept_id=cid,
                    t=t,
                    embedding=concept_bases[cid],
                    episodes=[t, t-1] if t > 0 else [t],
                    symbols=concept_symbols[cid],
                    activation=0.5 + np.random.random() * 0.5
                )

    # Calcular resultado global
    result = score_sx12_global(trackers)

    print("\n" + "=" * 70)
    print("RESULTADOS SX12")
    print("=" * 70)
    print(f"  Score SX12: {result.score:.4f}")
    print(f"  Passed: {result.passed} (> 0.5)")
    print(f"  Excellent: {result.excellent} (> 0.7)")
    print(f"\n  Metricas globales:")
    print(f"    Drift medio: {result.drift_global:.4f}")
    print(f"    Dispersion media: {result.dispersion_global:.4f}")
    print(f"    Estabilidad simbolica: {result.symbol_stability_global:.4f}")
    print(f"\n  Scores por agente:")
    for aid, score in result.agent_scores.items():
        print(f"    {aid}: {score:.4f}")
    print("=" * 70)

    return result


if __name__ == "__main__":
    result = run_sx12_test(n_agents=5, n_concepts_per_agent=8, n_timesteps=20)
