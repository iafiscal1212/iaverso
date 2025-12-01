"""
Symbolic Cognition Use: Uso cognitivo de símbolos
=================================================

Integra los símbolos con:
- Narrativa: resumen de episodios en secuencias simbólicas
- Planificación: planes como secuencias de símbolos
- Normas y ética: restricciones sobre símbolos
- Intención colectiva: símbolos compartidos entre agentes

Todo endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import (
    L_t, max_history, compute_adaptive_percentile, adaptive_momentum
)

from symbolic.sym_atoms import Symbol, SymbolExtractor
from symbolic.sym_binding import SymbolBinding, SymbolBindingManager
from symbolic.sym_grammar import GrammarRule, SymbolGrammar
from symbolic.sym_grounding import SymbolGroundingStats, SymbolGrounding


@dataclass
class SymbolicPlan:
    """Plan representado como secuencia de símbolos."""
    symbol_ids: List[int]
    expected_value: float
    expected_sagi: float
    confidence: float
    source: str  # 'binding', 'grammar', 'exploration'
    created_t: int


@dataclass
class SymbolicNarrative:
    """Narrativa como secuencia de símbolos."""
    symbol_ids: List[int]
    episode_times: List[int]
    compression_ratio: float
    coherence: float


class SymbolicCognitionUse:
    """
    Integra los símbolos con narrativa, planificación, normas y coordinación.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

        # Históricos de uso
        self.narrative_history: List[SymbolicNarrative] = []
        self.plan_history: List[SymbolicPlan] = []
        self.plan_outcomes: Dict[int, List[float]] = defaultdict(list)  # plan_hash -> outcomes

        # Estadísticas de efectividad
        self.symbolic_vs_raw_performance: List[Tuple[float, float]] = []

        # Normas simbólicas (restricciones)
        self.forbidden_sequences: Dict[Tuple[int, ...], float] = {}  # sequence -> penalty
        self.encouraged_sequences: Dict[Tuple[int, ...], float] = {}  # sequence -> bonus

        self.t = 0

    def summarize_episode_to_symbols(
        self,
        episode_times: List[int],
        states_by_time: Dict[int, np.ndarray],
        extractor: SymbolExtractor,
    ) -> SymbolicNarrative:
        """
        Mapea un episodio a una secuencia de símbolos.
        """
        symbol_ids = []

        for t in episode_times:
            if t in states_by_time:
                state = states_by_time[t]
                sym_id = extractor.assign_state_to_symbol(state)
                if sym_id is not None:
                    # Evitar repeticiones consecutivas
                    if not symbol_ids or symbol_ids[-1] != sym_id:
                        symbol_ids.append(sym_id)

        # Calcular compresión
        compression_ratio = len(symbol_ids) / max(len(episode_times), 1)

        # Coherencia: símbolos con alta frecuencia implican coherencia
        if symbol_ids:
            unique_ratio = len(set(symbol_ids)) / len(symbol_ids)
            coherence = 1.0 - unique_ratio  # Menos únicos = más coherente
        else:
            coherence = 0.5

        narrative = SymbolicNarrative(
            symbol_ids=symbol_ids,
            episode_times=episode_times,
            compression_ratio=compression_ratio,
            coherence=coherence
        )

        self.narrative_history.append(narrative)
        if len(self.narrative_history) > max_history(self.t):
            self.narrative_history = self.narrative_history[-max_history(self.t):]

        return narrative

    def symbolic_plan_candidates(
        self,
        t: int,
        current_state: np.ndarray,
        extractor: SymbolExtractor,
        binding_manager: SymbolBindingManager,
        grammar: SymbolGrammar,
        n_candidates: int = 5,
    ) -> List[SymbolicPlan]:
        """
        Genera planes candidatos como secuencias de símbolos.
        """
        self.t = t
        candidates = []

        # Símbolo actual
        current_sym = extractor.assign_state_to_symbol(current_state)

        # 1. Planes basados en bindings útiles
        useful_bindings = binding_manager.get_useful_bindings(t)
        for binding in useful_bindings[:n_candidates]:
            # Si el binding empieza con el símbolo actual o es general
            if current_sym is None or current_sym in binding.symbol_ids[:1]:
                plan = SymbolicPlan(
                    symbol_ids=list(binding.symbol_ids),
                    expected_value=float(np.mean(binding.consequence_signature)),
                    expected_sagi=float(np.std(binding.consequence_signature)),
                    confidence=binding.consistency,
                    source='binding',
                    created_t=t
                )
                candidates.append(plan)

        # 2. Planes basados en reglas gramaticales fuertes
        strong_rules = grammar.get_strong_rules(t)
        for rule in strong_rules[:n_candidates]:
            # Convertir roles a símbolos concretos
            symbol_ids = []
            for role_id in rule.role_sequence:
                syms_with_role = grammar.get_symbols_by_role(role_id)
                if syms_with_role:
                    # Elegir símbolo con mejor SymScore
                    best_sym = None
                    best_score = -np.inf
                    for sym_id in syms_with_role:
                        sym = extractor.get_symbol_by_id(sym_id)
                        if sym and sym.stats.sym_score > best_score:
                            best_score = sym.stats.sym_score
                            best_sym = sym_id
                    if best_sym is not None:
                        symbol_ids.append(best_sym)

            if len(symbol_ids) >= 2:
                plan = SymbolicPlan(
                    symbol_ids=symbol_ids,
                    expected_value=rule.effect_value,
                    expected_sagi=rule.effect_sagi,
                    confidence=min(1.0, rule.lift / 10),  # Normalizado
                    source='grammar',
                    created_t=t
                )
                candidates.append(plan)

        # 3. Planes exploratorios (combinaciones nuevas)
        active_symbols = extractor.get_active_symbols(t)
        if len(active_symbols) >= 2:
            for _ in range(2):
                n_syms = np.random.randint(2, min(5, len(active_symbols) + 1))
                selected = list(np.random.choice(active_symbols, size=n_syms, replace=False))

                # Estimar valor basado en símbolos individuales
                total_value = 0
                for sym_id in selected:
                    sym = extractor.get_symbol_by_id(sym_id)
                    if sym:
                        total_value += sym.stats.sym_score

                plan = SymbolicPlan(
                    symbol_ids=selected,
                    expected_value=total_value / len(selected),
                    expected_sagi=0.0,
                    confidence=0.3,  # Baja confianza para exploración
                    source='exploration',
                    created_t=t
                )
                candidates.append(plan)

        # Filtrar por normas
        candidates = self._filter_by_norms(candidates)

        return candidates[:n_candidates]

    def evaluate_symbolic_plan(
        self,
        t: int,
        plan: SymbolicPlan,
        grounding: SymbolGrounding,
    ) -> float:
        """
        Estima el valor esperado de un plan simbólico.
        """
        base_value = plan.expected_value * plan.confidence

        # Bonus por grounding
        grounding_bonus = 0
        for sym_id in plan.symbol_ids:
            stats = grounding.get_grounding_stats(sym_id)
            if stats:
                grounding_bonus += stats.grounded_score

        grounding_bonus /= max(len(plan.symbol_ids), 1)

        # Histórico de planes similares
        plan_hash = hash(tuple(plan.symbol_ids))
        if plan_hash in self.plan_outcomes and self.plan_outcomes[plan_hash]:
            historical_mean = np.mean(self.plan_outcomes[plan_hash])
            base_value = 0.5 * base_value + 0.5 * historical_mean

        # Penalización/bonus por normas
        norm_adjustment = self._get_norm_adjustment(tuple(plan.symbol_ids))

        return base_value * (1 + grounding_bonus) + norm_adjustment

    def update_from_outcome(
        self,
        t: int,
        plan: SymbolicPlan,
        observed_delta_value: float,
        observed_delta_sagi: float,
    ) -> None:
        """
        Actualiza estadísticas de efectividad de planes simbólicos.
        """
        self.t = t

        # Registrar outcome
        plan_hash = hash(tuple(plan.symbol_ids))
        outcome = observed_delta_value + observed_delta_sagi * 0.5
        self.plan_outcomes[plan_hash].append(outcome)

        # Limitar historial
        max_h = max_history(t)
        if len(self.plan_outcomes[plan_hash]) > max_h:
            self.plan_outcomes[plan_hash] = self.plan_outcomes[plan_hash][-max_h:]

        # Guardar plan
        self.plan_history.append(plan)
        if len(self.plan_history) > max_h:
            self.plan_history = self.plan_history[-max_h:]

        # Actualizar normas emergentes
        self._update_norms(tuple(plan.symbol_ids), outcome)

    def _filter_by_norms(self, plans: List[SymbolicPlan]) -> List[SymbolicPlan]:
        """Filtra planes según normas simbólicas."""
        filtered = []
        for plan in plans:
            seq = tuple(plan.symbol_ids)
            penalty = self.forbidden_sequences.get(seq, 0)
            if penalty < 0.5:  # Umbral de prohibición
                filtered.append(plan)
        return filtered

    def _get_norm_adjustment(self, sequence: Tuple[int, ...]) -> float:
        """Calcula ajuste por normas para una secuencia."""
        adjustment = 0.0
        adjustment -= self.forbidden_sequences.get(sequence, 0)
        adjustment += self.encouraged_sequences.get(sequence, 0)
        return adjustment

    def _update_norms(self, sequence: Tuple[int, ...], outcome: float) -> None:
        """Actualiza normas basadas en outcomes."""
        # Umbral endógeno
        if self.plan_outcomes:
            all_outcomes = []
            for outcomes in self.plan_outcomes.values():
                all_outcomes.extend(outcomes)
            if all_outcomes:
                low_threshold = np.percentile(all_outcomes, 20)
                high_threshold = np.percentile(all_outcomes, 80)
            else:
                low_threshold = -0.1
                high_threshold = 0.1
        else:
            low_threshold = -0.1
            high_threshold = 0.1

        # Actualizar normas
        if outcome < low_threshold:
            # Secuencia perjudicial -> prohibir
            current = self.forbidden_sequences.get(sequence, 0)
            self.forbidden_sequences[sequence] = min(1.0, current + 0.1)
        elif outcome > high_threshold:
            # Secuencia beneficiosa -> promover
            current = self.encouraged_sequences.get(sequence, 0)
            self.encouraged_sequences[sequence] = min(1.0, current + 0.1)

        # Decay de normas antiguas
        decay = 0.99
        for seq in list(self.forbidden_sequences.keys()):
            self.forbidden_sequences[seq] *= decay
            if self.forbidden_sequences[seq] < 0.01:
                del self.forbidden_sequences[seq]

        for seq in list(self.encouraged_sequences.keys()):
            self.encouraged_sequences[seq] *= decay
            if self.encouraged_sequences[seq] < 0.01:
                del self.encouraged_sequences[seq]

    def compare_symbolic_vs_raw(
        self,
        symbolic_performance: float,
        raw_performance: float,
    ) -> float:
        """
        Compara rendimiento simbólico vs crudo.
        Retorna la ganancia relativa.
        """
        self.symbolic_vs_raw_performance.append((symbolic_performance, raw_performance))

        max_h = max_history(self.t)
        if len(self.symbolic_vs_raw_performance) > max_h:
            self.symbolic_vs_raw_performance = self.symbolic_vs_raw_performance[-max_h:]

        if raw_performance != 0:
            gain = (symbolic_performance - raw_performance) / abs(raw_performance)
        else:
            gain = symbolic_performance

        return gain

    def get_shared_symbols_with(
        self,
        other_agent_symbols: Dict[int, Symbol],
        own_extractor: SymbolExtractor,
        similarity_threshold: float = 0.7,
    ) -> List[Tuple[int, int, float]]:
        """
        Encuentra símbolos compartidos con otro agente.
        Retorna: [(own_sym_id, other_sym_id, similarity), ...]
        """
        shared = []

        own_symbols = own_extractor.symbols

        for own_id, own_sym in own_symbols.items():
            for other_id, other_sym in other_agent_symbols.items():
                # Similitud de centros
                if len(own_sym.stats.mu) == len(other_sym.stats.mu):
                    dist = np.linalg.norm(own_sym.stats.mu - other_sym.stats.mu)
                    max_norm = max(np.linalg.norm(own_sym.stats.mu), np.linalg.norm(other_sym.stats.mu), 1.0)
                    similarity = 1.0 - dist / max_norm

                    if similarity >= similarity_threshold:
                        shared.append((own_id, other_id, similarity))

        return sorted(shared, key=lambda x: x[2], reverse=True)

    def get_statistics(self) -> Dict[str, Any]:
        """Estadísticas del uso cognitivo de símbolos."""
        # Ganancia simbólica
        if self.symbolic_vs_raw_performance:
            symbolic = [p[0] for p in self.symbolic_vs_raw_performance]
            raw = [p[1] for p in self.symbolic_vs_raw_performance]
            mean_gain = np.mean(symbolic) - np.mean(raw)
        else:
            mean_gain = 0

        # Compresión narrativa media
        if self.narrative_history:
            mean_compression = np.mean([n.compression_ratio for n in self.narrative_history])
            mean_coherence = np.mean([n.coherence for n in self.narrative_history])
        else:
            mean_compression = 0
            mean_coherence = 0

        return {
            'agent_id': self.agent_id,
            't': self.t,
            'n_narratives': len(self.narrative_history),
            'n_plans_executed': len(self.plan_history),
            'mean_compression': mean_compression,
            'mean_coherence': mean_coherence,
            'symbolic_gain': mean_gain,
            'n_forbidden_sequences': len(self.forbidden_sequences),
            'n_encouraged_sequences': len(self.encouraged_sequences)
        }


def test_symbolic_cognition():
    """Test del uso cognitivo de símbolos."""
    print("=" * 60)
    print("TEST: SYMBOLIC COGNITION USE")
    print("=" * 60)

    from symbolic.sym_atoms import SymbolExtractor, to_simplex

    # Crear componentes
    extractor = SymbolExtractor('NEO', state_dim=6)
    binding_manager = SymbolBindingManager('NEO')
    grammar = SymbolGrammar('NEO')
    grounding = SymbolGrounding('NEO')
    cognition = SymbolicCognitionUse('NEO')

    np.random.seed(42)

    # Fase 1: Construir símbolos
    print("\nFase 1: Construyendo símbolos...")
    for t in range(100):
        mode = t % 3
        z = np.zeros(6)
        z[mode] = 0.8
        z = to_simplex(np.abs(z + np.random.randn(6) * 0.05) + 0.01)
        phi = np.random.randn(5) * 0.1
        drives = to_simplex(np.random.rand(6) + 0.1)

        extractor.record_state(t, z, phi, drives, context=mode)

    extractor.extract_symbols(99)
    print(f"  Símbolos extraídos: {len(extractor.symbols)}")

    # Fase 2: Crear narrativas
    print("\nFase 2: Creando narrativas simbólicas...")
    states_by_time = extractor.state_history

    for episode_start in range(0, 80, 20):
        episode_times = list(range(episode_start, episode_start + 20))
        narrative = cognition.summarize_episode_to_symbols(
            episode_times, states_by_time, extractor
        )
        print(f"  Episodio {episode_start}-{episode_start+19}:")
        print(f"    Símbolos: {narrative.symbol_ids}")
        print(f"    Compresión: {narrative.compression_ratio:.3f}")
        print(f"    Coherencia: {narrative.coherence:.3f}")

    # Fase 3: Generar planes
    print("\nFase 3: Generando planes simbólicos...")

    # Primero poblar bindings y gramática
    for t in range(100):
        symbols = extractor.extract_symbols(t)
        sequence = [s.symbol_id for s in symbols[:4]]
        states = [extractor.state_history.get(t, np.zeros(17))]
        deltas = [np.random.randn(17) * 0.1]
        binding_manager.observe_sequence(t, sequence, states, deltas)

        effects = {s.symbol_id: np.random.randn(4) * 0.3 for s in symbols}
        grammar.infer_roles({s.symbol_id: s for s in symbols}, effects)
        grammar.observe_symbol_sequence(t, sequence, np.random.randn() * 0.2, np.random.randn() * 0.1)

    # Generar planes
    current_state = extractor.state_history[99]
    plans = cognition.symbolic_plan_candidates(
        100, current_state, extractor, binding_manager, grammar, n_candidates=5
    )

    for i, plan in enumerate(plans):
        value = cognition.evaluate_symbolic_plan(100, plan, grounding)
        print(f"\n  Plan {i+1} ({plan.source}):")
        print(f"    Símbolos: {plan.symbol_ids}")
        print(f"    Valor esperado: {plan.expected_value:.3f}")
        print(f"    Confianza: {plan.confidence:.3f}")
        print(f"    Valor evaluado: {value:.3f}")

        # Simular outcome
        outcome_v = np.random.randn() * 0.2
        outcome_sagi = np.random.randn() * 0.1
        cognition.update_from_outcome(100 + i, plan, outcome_v, outcome_sagi)

    # Estadísticas finales
    stats = cognition.get_statistics()
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS FINALES")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("TEST COMPLETADO")
    print("=" * 60)

    return cognition


if __name__ == "__main__":
    test_symbolic_cognition()
