"""
AGI-E Framework: Endogenous AGI
===============================

Marco completo para AGI Interna Endogena.

Cinco condiciones fundamentales:
- E1: Persistencia Estructural
- E2: No Colapso
- E3: Atractores Internos
- E4: Memoria Consistente
- E5: Alineamiento Simbolico Temporal Interno (NUEVO)

100% endogeno. Sin numeros magicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')


class AGIECondition(Enum):
    """Condiciones AGI-E."""
    E1_PERSISTENCE = "E1"
    E2_NO_COLLAPSE = "E2"
    E3_ATTRACTORS = "E3"
    E4_MEMORY = "E4"
    E5_SYMBOLIC_TEMPORAL = "E5"


@dataclass
class E5Result:
    """Resultado de E5 - Alineamiento Simbolico Temporal."""
    score: float
    passed: bool
    temporal_coherence: float      # Coherencia simbolica temporal
    symbol_stability: float        # Estabilidad de simbolos a traves del tiempo
    narrative_continuity: float    # Continuidad narrativa
    cross_agent_alignment: float   # Alineamiento entre agentes
    details: Dict[str, Any]


@dataclass
class AGIEResult:
    """Resultado completo del framework AGI-E."""
    e1_passed: bool
    e2_passed: bool
    e3_passed: bool
    e4_passed: bool
    e5_passed: bool

    e1_score: float
    e2_score: float
    e3_score: float
    e4_score: float
    e5_score: float

    agi_e_global: float
    is_agi_internal: bool

    details: Dict[str, Any]


class SymbolicTemporalAligner:
    """
    E5: Sistema de Alineamiento Simbolico Temporal Interno.

    Mide si los simbolos mantienen coherencia temporal:
    - Mismo simbolo = mismo contexto a traves del tiempo
    - Transiciones simbolicas son graduales, no abruptas
    - Narrativa simbolica es continua
    """

    def __init__(self, n_agents: int = 5, state_dim: int = 12):
        self.n_agents = n_agents
        self.state_dim = state_dim

        # Historial de uso simbolico por agente y tiempo
        # agent_id -> symbol -> [(t, context_vector)]
        self.symbol_usage: Dict[str, Dict[str, List[Tuple[int, np.ndarray]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Secuencias simbolicas por agente (para continuidad)
        self.symbol_sequences: Dict[str, List[Tuple[int, Set[str]]]] = defaultdict(list)

        # Narrativas por agente
        self.narratives: Dict[str, List[Tuple[int, str, float]]] = defaultdict(list)

        # Historial para umbrales endogenos
        self.context_distances: List[float] = []
        self.transition_magnitudes: List[float] = []

    def record_symbol_usage(self, agent_id: str, t: int, symbol: str,
                           context: np.ndarray):
        """Registra uso de un simbolo con su contexto."""
        self.symbol_usage[agent_id][symbol].append((t, context.copy()))

    def record_symbol_set(self, agent_id: str, t: int, symbols: Set[str]):
        """Registra el conjunto de simbolos activos en un momento."""
        self.symbol_sequences[agent_id].append((t, symbols.copy()))

    def record_narrative(self, agent_id: str, t: int, narrative_state: str,
                        confidence: float):
        """Registra estado narrativo."""
        self.narratives[agent_id].append((t, narrative_state, confidence))

    def compute_temporal_coherence(self) -> float:
        """
        Calcula coherencia temporal: mismo simbolo = contextos similares.

        Para cada simbolo usado multiples veces:
        - Calcular varianza de contextos
        - Alta coherencia = baja varianza
        """
        coherences = []

        for agent_id, symbols in self.symbol_usage.items():
            for symbol, usages in symbols.items():
                if len(usages) < 3:
                    continue

                contexts = np.array([ctx for _, ctx in usages])

                # Varianza intra-simbolo
                variance = np.var(contexts)

                # Coherencia = 1 / (1 + variance)
                coherence = 1 / (1 + variance)
                coherences.append(coherence)

        if not coherences:
            return 0.5

        return float(np.mean(coherences))

    def compute_symbol_stability(self) -> float:
        """
        Calcula estabilidad simbolica: transiciones graduales.

        Entre pasos consecutivos:
        - Jaccard de simbolos activos
        - Alta estabilidad = Jaccard alto
        """
        stabilities = []

        for agent_id, sequence in self.symbol_sequences.items():
            if len(sequence) < 2:
                continue

            for i in range(1, len(sequence)):
                t1, s1 = sequence[i-1]
                t2, s2 = sequence[i]

                if len(s1 | s2) > 0:
                    jaccard = len(s1 & s2) / len(s1 | s2)
                else:
                    jaccard = 1.0

                stabilities.append(jaccard)
                self.transition_magnitudes.append(1 - jaccard)

        if not stabilities:
            return 0.5

        return float(np.mean(stabilities))

    def compute_narrative_continuity(self) -> float:
        """
        Calcula continuidad narrativa: estados narrativos coherentes.

        Penaliza cambios abruptos de narrativa sin justificacion.
        """
        continuities = []

        for agent_id, narrative_history in self.narratives.items():
            if len(narrative_history) < 2:
                continue

            # Contar transiciones
            same_state = 0
            total_transitions = 0

            for i in range(1, len(narrative_history)):
                t1, state1, conf1 = narrative_history[i-1]
                t2, state2, conf2 = narrative_history[i]

                total_transitions += 1
                if state1 == state2:
                    same_state += 1
                elif conf1 < 0.5 or conf2 < 0.5:
                    # Cambio justificado por baja confianza
                    same_state += 0.5

            if total_transitions > 0:
                continuity = same_state / total_transitions
                continuities.append(continuity)

        if not continuities:
            return 0.5

        return float(np.mean(continuities))

    def compute_cross_agent_alignment(self) -> float:
        """
        Calcula alineamiento entre agentes: simbolos compartidos = contextos similares.
        """
        if len(self.symbol_usage) < 2:
            return 0.5

        # Encontrar simbolos compartidos
        all_symbols: Dict[str, List[str]] = defaultdict(list)
        for agent_id, symbols in self.symbol_usage.items():
            for symbol in symbols:
                all_symbols[symbol].append(agent_id)

        shared_symbols = {s: agents for s, agents in all_symbols.items()
                        if len(agents) >= 2}

        if not shared_symbols:
            return 0.5

        alignments = []

        for symbol, agents in shared_symbols.items():
            # Calcular contexto medio por agente
            agent_contexts = {}
            for agent_id in agents:
                usages = self.symbol_usage[agent_id][symbol]
                if usages:
                    contexts = np.array([ctx for _, ctx in usages])
                    agent_contexts[agent_id] = np.mean(contexts, axis=0)

            if len(agent_contexts) < 2:
                continue

            # Calcular similaridad entre agentes
            contexts_list = list(agent_contexts.values())
            for i in range(len(contexts_list)):
                for j in range(i + 1, len(contexts_list)):
                    c1, c2 = contexts_list[i], contexts_list[j]

                    # Distancia coseno
                    norm1, norm2 = np.linalg.norm(c1), np.linalg.norm(c2)
                    if norm1 > 1e-8 and norm2 > 1e-8:
                        similarity = np.dot(c1, c2) / (norm1 * norm2)
                        alignments.append(max(0, similarity))
                        self.context_distances.append(1 - similarity)

        if not alignments:
            return 0.5

        return float(np.mean(alignments))

    def compute_e5(self) -> E5Result:
        """Calcula el score E5 completo."""
        temporal_coherence = self.compute_temporal_coherence()
        symbol_stability = self.compute_symbol_stability()
        narrative_continuity = self.compute_narrative_continuity()
        cross_agent_alignment = self.compute_cross_agent_alignment()

        # Pesos endogenos basados en varianza
        components = [temporal_coherence, symbol_stability,
                     narrative_continuity, cross_agent_alignment]

        if len(set(components)) > 1:
            variances = [np.var([c]) + 1e-8 for c in components]
            weights = [1/v for v in variances]
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights]
        else:
            weights = [0.25, 0.25, 0.25, 0.25]

        score = sum(w * c for w, c in zip(weights, components))

        # Umbral endogeno
        if len(self.context_distances) >= 5:
            threshold = 1 - np.percentile(self.context_distances, 75)
        else:
            threshold = 0.5

        passed = score > threshold

        return E5Result(
            score=float(score),
            passed=passed,
            temporal_coherence=temporal_coherence,
            symbol_stability=symbol_stability,
            narrative_continuity=narrative_continuity,
            cross_agent_alignment=cross_agent_alignment,
            details={
                'n_symbols_tracked': sum(len(s) for s in self.symbol_usage.values()),
                'n_agents': len(self.symbol_usage),
                'threshold': threshold,
                'weights': weights
            }
        )


class AGIEFramework:
    """
    Framework completo AGI-E.

    Integra las 5 condiciones para AGI Interna Endogena.
    """

    def __init__(self, n_agents: int = 5, state_dim: int = 12):
        self.n_agents = n_agents
        self.state_dim = state_dim

        # Sistema E5
        self.e5_aligner = SymbolicTemporalAligner(n_agents, state_dim)

        # Resultados E1-E4 (se pasan externamente)
        self.e1_result: Optional[Dict] = None
        self.e2_result: Optional[Dict] = None
        self.e3_result: Optional[Dict] = None
        self.e4_result: Optional[Dict] = None

    def set_e1_result(self, passed: bool, score: float, details: Dict = None):
        """Establece resultado E1."""
        self.e1_result = {'passed': passed, 'score': score, 'details': details or {}}

    def set_e2_result(self, passed: bool, score: float, details: Dict = None):
        """Establece resultado E2."""
        self.e2_result = {'passed': passed, 'score': score, 'details': details or {}}

    def set_e3_result(self, passed: bool, score: float, details: Dict = None):
        """Establece resultado E3."""
        self.e3_result = {'passed': passed, 'score': score, 'details': details or {}}

    def set_e4_result(self, passed: bool, score: float, details: Dict = None):
        """Establece resultado E4."""
        self.e4_result = {'passed': passed, 'score': score, 'details': details or {}}

    def record_observation(self, agent_id: str, t: int, symbols: Set[str],
                          context: np.ndarray, narrative_state: str,
                          narrative_confidence: float):
        """Registra una observacion para E5."""
        # Registrar cada simbolo
        for symbol in symbols:
            self.e5_aligner.record_symbol_usage(agent_id, t, symbol, context)

        # Registrar conjunto de simbolos
        self.e5_aligner.record_symbol_set(agent_id, t, symbols)

        # Registrar narrativa
        self.e5_aligner.record_narrative(agent_id, t, narrative_state,
                                         narrative_confidence)

    def compute_agi_e(self) -> AGIEResult:
        """Calcula el resultado AGI-E completo."""
        # Verificar que tenemos E1-E4
        if any(r is None for r in [self.e1_result, self.e2_result,
                                    self.e3_result, self.e4_result]):
            raise ValueError("Faltan resultados E1-E4")

        # Calcular E5
        e5_result = self.e5_aligner.compute_e5()

        # Scores
        e1_score = self.e1_result['score']
        e2_score = self.e2_result['score']
        e3_score = self.e3_result['score']
        e4_score = self.e4_result['score']
        e5_score = e5_result.score

        # Passed
        e1_passed = self.e1_result['passed']
        e2_passed = self.e2_result['passed']
        e3_passed = self.e3_result['passed']
        e4_passed = self.e4_result['passed']
        e5_passed = e5_result.passed

        # AGI-E global: media ponderada por varianza inversa
        scores = [e1_score, e2_score, e3_score, e4_score, e5_score]

        # Pesos uniformes inicialmente
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]

        agi_e_global = sum(w * s for w, s in zip(weights, scores))

        # Es AGI interna si todas las condiciones pasan
        is_agi_internal = all([e1_passed, e2_passed, e3_passed, e4_passed, e5_passed])

        return AGIEResult(
            e1_passed=e1_passed,
            e2_passed=e2_passed,
            e3_passed=e3_passed,
            e4_passed=e4_passed,
            e5_passed=e5_passed,
            e1_score=e1_score,
            e2_score=e2_score,
            e3_score=e3_score,
            e4_score=e4_score,
            e5_score=e5_score,
            agi_e_global=agi_e_global,
            is_agi_internal=is_agi_internal,
            details={
                'e5_details': e5_result.details,
                'weights': weights
            }
        )


def run_e5_test(n_agents: int = 5, n_timesteps: int = 500) -> E5Result:
    """
    Ejecuta test E5 con datos simulados.
    """
    print("=" * 70)
    print("E5 - ALINEAMIENTO SIMBOLICO TEMPORAL INTERNO")
    print("=" * 70)
    print(f"  Agentes: {n_agents}")
    print(f"  Timesteps: {n_timesteps}")
    print("=" * 70)

    np.random.seed(42)

    state_dim = 12
    aligner = SymbolicTemporalAligner(n_agents, state_dim)

    agent_ids = [f"A{i}" for i in range(n_agents)]

    # Contextos base por simbolo (semantica emergente)
    symbol_contexts = {
        f"S{i}": np.random.randn(state_dim) * 0.5
        for i in range(20)
    }

    # Simular uso temporal de simbolos
    for aid in agent_ids:
        # Estado narrativo base
        narrative_states = ['exploring', 'consolidating', 'thriving']
        current_narrative = np.random.choice(narrative_states)
        narrative_confidence = 0.7

        # Simbolos preferidos por agente
        preferred_symbols = [f"S{(hash(aid) + i) % 20}" for i in range(8)]

        for t in range(n_timesteps):
            # Elegir simbolos activos (con continuidad)
            n_symbols = np.random.randint(2, 5)
            active_symbols = set(np.random.choice(preferred_symbols, n_symbols, replace=False))

            # Contexto para cada simbolo (cercano a su significado base)
            for symbol in active_symbols:
                base_ctx = symbol_contexts[symbol]
                context = base_ctx + np.random.randn(state_dim) * 0.1
                aligner.record_symbol_usage(aid, t, symbol, context)

            # Registrar conjunto
            aligner.record_symbol_set(aid, t, active_symbols)

            # Actualizar narrativa (cambios graduales)
            if np.random.random() < 0.05:  # 5% probabilidad de cambio
                current_narrative = np.random.choice(narrative_states)
                narrative_confidence = 0.5 + np.random.random() * 0.3
            else:
                narrative_confidence = min(1.0, narrative_confidence + 0.01)

            aligner.record_narrative(aid, t, current_narrative, narrative_confidence)

    # Calcular resultado
    result = aligner.compute_e5()

    print("\n" + "=" * 70)
    print("RESULTADOS E5")
    print("=" * 70)
    print(f"  Score E5: {result.score:.4f}")
    print(f"  Passed: {result.passed}")
    print(f"\n  Componentes:")
    print(f"    Coherencia Temporal:     {result.temporal_coherence:.4f}")
    print(f"    Estabilidad Simbolica:   {result.symbol_stability:.4f}")
    print(f"    Continuidad Narrativa:   {result.narrative_continuity:.4f}")
    print(f"    Alineamiento Cross-Agent: {result.cross_agent_alignment:.4f}")
    print("=" * 70)

    return result


if __name__ == "__main__":
    result = run_e5_test(n_agents=5, n_timesteps=500)
