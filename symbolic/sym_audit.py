"""
Symbolic Audit: Auditoría de capacidades simbólicas
===================================================

Tests SYM-1 a SYM-5 para validar que existe estructura simbólica real:

SYM-1: Riqueza y calidad de símbolos
SYM-2: Composicionalidad (bindings útiles)
SYM-3: Gramática emergente con efecto
SYM-4: Grounding estructural
SYM-5: Uso cognitivo efectivo

Todo endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import (
    L_t, max_history, compute_adaptive_percentile
)

from symbolic.sym_atoms import Symbol, SymbolExtractor
from symbolic.sym_alphabet import SymbolAlphabet
from symbolic.sym_binding import SymbolBinding, SymbolBindingManager
from symbolic.sym_grammar import GrammarRule, SymbolGrammar
from symbolic.sym_grounding import SymbolGroundingStats, SymbolGrounding
from symbolic.sym_use_cognition import SymbolicCognitionUse


@dataclass
class SymAuditResult:
    """Resultado de un test de auditoría simbólica."""
    test_name: str
    score: float     # [0,1]
    go: bool         # GO/NO-GO
    details: Dict[str, Any]


class SymbolicAuditor:
    """
    Auditoría simbólica completa:
    - SYM-1: Riqueza y calidad de símbolos
    - SYM-2: Composicionalidad (bindings)
    - SYM-3: Gramática emergente
    - SYM-4: Grounding estructural
    - SYM-5: Uso cognitivo eficaz
    """

    # Umbrales GO/NO-GO (mínimos para considerar exitoso)
    GO_THRESHOLD = 0.5

    def __init__(self):
        self.history: List[SymAuditResult] = []
        self.score_history: Dict[str, List[float]] = {
            'SYM-1': [],
            'SYM-2': [],
            'SYM-3': [],
            'SYM-4': [],
            'SYM-5': []
        }

    def test_sym1_richness(
        self,
        extractor: SymbolExtractor,
        t: int,
    ) -> SymAuditResult:
        """
        SYM-1: Riqueza y calidad de símbolos.

        Mide:
        - Número de símbolos válidos
        - SymScore medio
        - Diversidad de símbolos
        """
        symbols = extractor.get_active_symbols(t)
        n_total = len(extractor.symbols)

        if n_total == 0:
            result = SymAuditResult(
                test_name='SYM-1',
                score=0.0,
                go=False,
                details={'error': 'No symbols'}
            )
            self.history.append(result)
            return result

        n_valid = len(symbols)
        valid_ratio = n_valid / n_total

        # SymScore medio
        if symbols:
            sym_scores = [s.stats.sym_score for s in symbols]
            mean_score = np.mean(sym_scores)

            # Percentil del score medio respecto a historial
            if extractor.global_scores:
                p75 = np.percentile(extractor.global_scores, 75)
                score_quality = mean_score / (p75 + 1e-8)
            else:
                score_quality = mean_score
        else:
            mean_score = 0
            score_quality = 0

        # Diversidad (entropía de distribución de episodios)
        if symbols:
            episode_counts = [s.stats.n_episodes for s in symbols]
            total_eps = sum(episode_counts)
            if total_eps > 0:
                probs = np.array(episode_counts) / total_eps
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(len(symbols))
                diversity = entropy / (max_entropy + 1e-8)
            else:
                diversity = 0
        else:
            diversity = 0

        # Score final
        score = 0.4 * valid_ratio + 0.4 * score_quality + 0.2 * diversity
        score = float(np.clip(score, 0, 1))

        go = score >= self.GO_THRESHOLD and n_valid >= L_t(t)

        result = SymAuditResult(
            test_name='SYM-1',
            score=score,
            go=go,
            details={
                'n_total': n_total,
                'n_valid': n_valid,
                'valid_ratio': valid_ratio,
                'mean_sym_score': mean_score,
                'diversity': diversity
            }
        )

        self.history.append(result)
        self.score_history['SYM-1'].append(score)

        return result

    def test_sym2_compositionality(
        self,
        binding_manager: SymbolBindingManager,
        t: int,
    ) -> SymAuditResult:
        """
        SYM-2: Composicionalidad (bindings útiles).

        Mide:
        - Proporción de bindings útiles
        - ΔConsistencia media
        - Lift medio
        """
        useful = binding_manager.get_useful_bindings(t)
        total = len(binding_manager.bindings)

        if total == 0:
            result = SymAuditResult(
                test_name='SYM-2',
                score=0.0,
                go=False,
                details={'error': 'No bindings'}
            )
            self.history.append(result)
            return result

        useful_ratio = len(useful) / total

        # ΔConsistencia media de bindings útiles
        if useful:
            delta_cons = np.mean([b.delta_consistency for b in useful])
            lifts = np.mean([b.lift for b in useful])
        else:
            delta_cons = 0
            lifts = 0

        # Normalizar lift por histórico
        if binding_manager.lift_history:
            p75_lift = np.percentile(binding_manager.lift_history, 75)
            lift_quality = min(1.0, lifts / (p75_lift + 1e-8))
        else:
            lift_quality = 0.5

        # Score final
        score = 0.3 * useful_ratio + 0.35 * max(0, delta_cons + 0.5) + 0.35 * lift_quality
        score = float(np.clip(score, 0, 1))

        go = score >= self.GO_THRESHOLD and len(useful) >= L_t(t) // 2

        result = SymAuditResult(
            test_name='SYM-2',
            score=score,
            go=go,
            details={
                'total_bindings': total,
                'useful_bindings': len(useful),
                'useful_ratio': useful_ratio,
                'mean_delta_cons': delta_cons,
                'mean_lift': lifts
            }
        )

        self.history.append(result)
        self.score_history['SYM-2'].append(score)

        return result

    def test_sym3_grammar(
        self,
        grammar: SymbolGrammar,
        t: int,
    ) -> SymAuditResult:
        """
        SYM-3: Gramática emergente con efecto.

        Mide:
        - Número de reglas fuertes
        - Efecto medio en ΔV/ΔSAGI
        - Diversidad de roles
        """
        strong_rules = grammar.get_strong_rules(t)
        total_rules = len(grammar.rules)

        if total_rules == 0:
            result = SymAuditResult(
                test_name='SYM-3',
                score=0.0,
                go=False,
                details={'error': 'No grammar rules'}
            )
            self.history.append(result)
            return result

        strong_ratio = len(strong_rules) / total_rules

        # Efecto medio
        if strong_rules:
            effects = [abs(r.effect_value) + abs(r.effect_sagi) for r in strong_rules]
            mean_effect = np.mean(effects)

            # Normalizar por histórico
            if grammar.effect_history:
                p75_effect = np.percentile(grammar.effect_history, 75)
                effect_quality = min(1.0, mean_effect / (p75_effect + 1e-8))
            else:
                effect_quality = mean_effect
        else:
            mean_effect = 0
            effect_quality = 0

        # Diversidad de roles
        n_roles = len(set(r.role_id for r in grammar.roles.values()))
        role_diversity = min(1.0, n_roles / max(grammar.n_roles, 1))

        # Score final
        score = 0.3 * strong_ratio + 0.4 * effect_quality + 0.3 * role_diversity
        score = float(np.clip(score, 0, 1))

        go = score >= self.GO_THRESHOLD and len(strong_rules) >= 1

        result = SymAuditResult(
            test_name='SYM-3',
            score=score,
            go=go,
            details={
                'total_rules': total_rules,
                'strong_rules': len(strong_rules),
                'strong_ratio': strong_ratio,
                'mean_effect': mean_effect,
                'n_roles': n_roles
            }
        )

        self.history.append(result)
        self.score_history['SYM-3'].append(score)

        return result

    def test_sym4_grounding(
        self,
        grounding: SymbolGrounding,
        t: int,
    ) -> SymAuditResult:
        """
        SYM-4: Grounding estructural.

        Mide:
        - Selectividad de mundo
        - Selectividad social
        - Impacto
        """
        grounded = grounding.get_grounded_symbols(t)
        total = len(grounding.stats_by_symbol)

        if total == 0:
            result = SymAuditResult(
                test_name='SYM-4',
                score=0.0,
                go=False,
                details={'error': 'No grounding data'}
            )
            self.history.append(result)
            return result

        grounded_ratio = len(grounded) / total

        # Métricas medias
        if grounding.stats_by_symbol:
            mean_sel_world = np.mean([s.sel_world for s in grounding.stats_by_symbol.values()])
            mean_sel_social = np.mean([s.sel_social for s in grounding.stats_by_symbol.values()])
            mean_impact = np.mean([s.impact for s in grounding.stats_by_symbol.values()])
        else:
            mean_sel_world = 0
            mean_sel_social = 0
            mean_impact = 0

        # Score final
        score = 0.25 * grounded_ratio + 0.25 * mean_sel_world + 0.25 * mean_sel_social + 0.25 * mean_impact
        score = float(np.clip(score, 0, 1))

        go = score >= self.GO_THRESHOLD and len(grounded) >= L_t(t) // 2

        result = SymAuditResult(
            test_name='SYM-4',
            score=score,
            go=go,
            details={
                'total_symbols': total,
                'well_grounded': len(grounded),
                'grounded_ratio': grounded_ratio,
                'mean_sel_world': mean_sel_world,
                'mean_sel_social': mean_sel_social,
                'mean_impact': mean_impact
            }
        )

        self.history.append(result)
        self.score_history['SYM-4'].append(score)

        return result

    def test_sym5_cognitive_use(
        self,
        cognition: SymbolicCognitionUse,
        t: int,
        symbolic_performance: Optional[float] = None,
        baseline_performance: Optional[float] = None,
    ) -> SymAuditResult:
        """
        SYM-5: Uso cognitivo efectivo.

        Mide:
        - Compresión narrativa
        - Coherencia narrativa
        - Ganancia simbólica vs baseline
        """
        stats = cognition.get_statistics()

        # Compresión
        compression = stats['mean_compression']
        # Queremos compresión alta (menos símbolos que estados originales)
        compression_score = min(1.0, compression / 0.5) if compression < 1 else 1.0 / compression

        # Coherencia
        coherence = stats['mean_coherence']

        # Ganancia simbólica
        if symbolic_performance is not None and baseline_performance is not None:
            gain = cognition.compare_symbolic_vs_raw(symbolic_performance, baseline_performance)
            gain_score = min(1.0, max(0, 0.5 + gain))
        elif cognition.symbolic_vs_raw_performance:
            sym_perfs = [p[0] for p in cognition.symbolic_vs_raw_performance]
            raw_perfs = [p[1] for p in cognition.symbolic_vs_raw_performance]
            gain = np.mean(sym_perfs) - np.mean(raw_perfs)
            gain_score = min(1.0, max(0, 0.5 + gain))
        else:
            gain_score = 0.5
            gain = 0

        # Uso activo (tiene planes y narrativas)
        n_narratives = stats['n_narratives']
        n_plans = stats['n_plans_executed']
        usage_score = min(1.0, (n_narratives + n_plans) / (L_t(t) * 2))

        # Score final
        score = 0.2 * compression_score + 0.2 * coherence + 0.3 * gain_score + 0.3 * usage_score
        score = float(np.clip(score, 0, 1))

        go = score >= self.GO_THRESHOLD and n_narratives >= 1

        result = SymAuditResult(
            test_name='SYM-5',
            score=score,
            go=go,
            details={
                'compression': compression,
                'coherence': coherence,
                'symbolic_gain': gain,
                'n_narratives': n_narratives,
                'n_plans': n_plans
            }
        )

        self.history.append(result)
        self.score_history['SYM-5'].append(score)

        return result

    def run_full_audit(
        self,
        t: int,
        extractor: SymbolExtractor,
        alphabet: SymbolAlphabet,
        binding_manager: SymbolBindingManager,
        grammar: SymbolGrammar,
        grounding: SymbolGrounding,
        cognition: SymbolicCognitionUse,
        symbolic_performance: Optional[float] = None,
        baseline_performance: Optional[float] = None,
    ) -> Dict[str, SymAuditResult]:
        """Ejecuta todos los tests de auditoría."""
        results = {}

        results['SYM-1'] = self.test_sym1_richness(extractor, t)
        results['SYM-2'] = self.test_sym2_compositionality(binding_manager, t)
        results['SYM-3'] = self.test_sym3_grammar(grammar, t)
        results['SYM-4'] = self.test_sym4_grounding(grounding, t)
        results['SYM-5'] = self.test_sym5_cognitive_use(
            cognition, t, symbolic_performance, baseline_performance
        )

        return results

    def compute_sym_x_score(self) -> float:
        """
        SYM_X = score global de capacidad simbólica.
        Ponderación endógena basada en varianzas.
        """
        if not all(self.score_history[f'SYM-{i}'] for i in range(1, 6)):
            # Sin suficientes datos, promedio simple
            recent_scores = []
            for i in range(1, 6):
                if self.score_history[f'SYM-{i}']:
                    recent_scores.append(self.score_history[f'SYM-{i}'][-1])
            return float(np.mean(recent_scores)) if recent_scores else 0.0

        # Ponderación por inverso de varianza
        weights = []
        latest_scores = []

        for i in range(1, 6):
            scores = self.score_history[f'SYM-{i}']
            latest_scores.append(scores[-1])

            if len(scores) >= 3:
                var = np.var(scores[-10:])
                weight = 1.0 / (var + 0.01)
            else:
                weight = 1.0

            weights.append(weight)

        # Normalizar pesos
        weights = np.array(weights)
        weights /= weights.sum()

        sym_x = float(np.dot(weights, latest_scores))
        return sym_x

    def get_summary(self) -> Dict[str, Any]:
        """Resumen de la auditoría."""
        latest = {}
        for i in range(1, 6):
            key = f'SYM-{i}'
            if self.score_history[key]:
                latest[key] = self.score_history[key][-1]
            else:
                latest[key] = 0.0

        return {
            'sym_x_score': self.compute_sym_x_score(),
            'individual_scores': latest,
            'all_go': all(latest.get(f'SYM-{i}', 0) >= self.GO_THRESHOLD for i in range(1, 6)),
            'n_audits': len(self.history)
        }


def test_symbolic_auditor():
    """Test del auditor simbólico."""
    print("=" * 60)
    print("TEST: SYMBOLIC AUDITOR")
    print("=" * 60)

    from symbolic.sym_atoms import SymbolExtractor, to_simplex

    # Crear componentes
    extractor = SymbolExtractor('NEO', state_dim=6)
    alphabet = SymbolAlphabet('NEO')
    binding_manager = SymbolBindingManager('NEO')
    grammar = SymbolGrammar('NEO')
    grounding = SymbolGrounding('NEO')
    cognition = SymbolicCognitionUse('NEO')
    auditor = SymbolicAuditor()

    np.random.seed(42)

    # Poblar sistemas
    print("\nPoblando sistemas simbólicos...")

    for t in range(200):
        mode = t % 3
        z = np.zeros(6)
        z[mode] = 0.8
        z = to_simplex(np.abs(z + np.random.randn(6) * 0.05) + 0.01)
        phi = np.random.randn(5) * 0.1
        drives = to_simplex(np.random.rand(6) + 0.1)

        extractor.record_state(t, z, phi, drives, context=mode)

        if (t + 1) % 20 == 0:
            symbols = extractor.extract_symbols(t)

            # Alfabeto
            values = {s.symbol_id: s.stats.sym_score + np.random.randn() * 0.1 for s in symbols}
            alphabet.update_alphabet(t, symbols, values)

            # Bindings
            sequence = [s.symbol_id for s in symbols[:4]] if symbols else []
            states = [extractor.state_history.get(t, np.zeros(17))]
            deltas = [np.random.randn(17) * 0.1]
            binding_manager.observe_sequence(t, sequence, states, deltas)

            # Gramática
            effects = {s.symbol_id: np.random.randn(4) * 0.3 for s in symbols}
            grammar.infer_roles({s.symbol_id: s for s in symbols}, effects)
            grammar.observe_symbol_sequence(t, sequence, np.random.randn() * 0.2, np.random.randn() * 0.1)

            # Grounding
            for sym in symbols:
                grounding.observe_symbol_in_context(
                    t, sym.symbol_id, mode, ['NEO', 'EVA'],
                    np.random.randn() * 0.1, np.random.randn() * 0.05
                )
            grounding.update_grounding({s.symbol_id: s for s in symbols})

            # Cognición
            episode_times = list(range(max(0, t-10), t))
            if episode_times:
                cognition.summarize_episode_to_symbols(
                    episode_times, extractor.state_history, extractor
                )

    # Ejecutar auditoría
    print("\nEjecutando auditoría...")
    results = auditor.run_full_audit(
        199, extractor, alphabet, binding_manager, grammar, grounding, cognition
    )

    print("\n" + "=" * 60)
    print("RESULTADOS DE AUDITORÍA")
    print("=" * 60)

    for test_name, result in results.items():
        status = "GO" if result.go else "NO-GO"
        print(f"\n{test_name}: {result.score:.3f} [{status}]")
        for k, v in result.details.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.3f}")
            else:
                print(f"    {k}: {v}")

    summary = auditor.get_summary()
    print("\n" + "=" * 60)
    print(f"SYM-X SCORE: {summary['sym_x_score']:.3f}")
    print(f"ALL GO: {summary['all_go']}")
    print("=" * 60)

    return auditor


if __name__ == "__main__":
    test_symbolic_auditor()
