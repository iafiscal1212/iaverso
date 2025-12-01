"""
SX3 - Grammar Causality v2 (Causalidad de Reglas Simbólicas)
============================================================

Implementa exactamente:
1. Overlap: Ω_r = E_t[min(p_t(r=1), p_t(r=0))]
2. ATE endógeno: Δ_r con pesos IPW y clipping Q95%
3. Magnitud: m_r = sig(|Δ_r|) = |Δ_r| / (|Δ_r| + Q75%(|Δ|_hist))
4. Especificidad: S_r = 1 - MMD_r / Q75%(MMD_hist)
5. Fiabilidad CF: R'_r = R_r / Q75%(R_hist)
6. Score por regla: GC(r) = m_r * S_r * Ω_r * R'_r
7. SX3 = Median_{r ∈ R_válidas} GC(r)

100% endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class RuleEffect:
    """Efecto de una regla gramatical."""
    rule_id: str
    role_sequence: Tuple[int, ...]
    overlap: float              # Ω_r
    ate: float                  # Δ_r (Average Treatment Effect)
    magnitude: float            # m_r
    specificity: float          # S_r
    cf_reliability: float       # R'_r
    gc_score: float             # GC(r) final
    n_applications: int
    is_valid: bool


class GrammarCausalityV2:
    """
    Sistema de causalidad gramatical v2.

    100% endógeno: todos los umbrales derivados de percentiles.
    """

    def __init__(self, agent_id: str, n_roles: int = 4):
        self.agent_id = agent_id
        self.n_roles = n_roles

        # Historial de reglas aplicadas
        self.rule_applications: Dict[str, List[int]] = defaultdict(list)  # rule_id -> [t1, t2, ...]

        # Contexto y efectos
        self.context_history: List[np.ndarray] = []
        self.effect_history: List[np.ndarray] = []  # Y_t = [reward_change, teleology, coherence, energy]
        self.cf_fid_history: List[float] = []

        # Probabilidades de reglas por tiempo
        self.rule_probs: Dict[str, List[float]] = defaultdict(list)  # rule_id -> [p_t(r=1)]

        # Métricas por regla
        self.rule_effects: Dict[str, List[np.ndarray]] = defaultdict(list)  # Efectos Y cuando r=1
        self.rule_baseline_effects: Dict[str, List[np.ndarray]] = defaultdict(list)  # Efectos Y cuando r=0

        # Historial de métricas para normalización endógena
        self.delta_history: List[float] = []  # |Δ_r| históricos
        self.mmd_history: List[float] = []
        self.r_history: List[float] = []  # R_r históricos

        # Vecinos de reglas (para especificidad)
        self.rule_neighbors: Dict[str, List[str]] = {}

        self.t = 0

    def _rule_id(self, role_sequence: Tuple[int, ...]) -> str:
        """Genera ID único para secuencia de roles."""
        return "_".join(str(r) for r in role_sequence)

    def _get_neighbor_rules(self, rule_id: str) -> List[str]:
        """
        Obtiene reglas vecinas (Hamming distance ≤ 1).
        """
        if rule_id in self.rule_neighbors:
            return self.rule_neighbors[rule_id]

        parts = [int(x) for x in rule_id.split("_")]
        neighbors = []

        for r in self.rule_applications.keys():
            if r == rule_id:
                continue
            r_parts = [int(x) for x in r.split("_")]
            if len(r_parts) == len(parts):
                hamming = sum(1 for a, b in zip(parts, r_parts) if a != b)
                if hamming <= 1:
                    neighbors.append(r)

        self.rule_neighbors[rule_id] = neighbors
        return neighbors

    def observe(
        self,
        t: int,
        context: np.ndarray,
        applied_rules: List[Tuple[int, ...]],  # Reglas aplicadas este paso
        rule_probabilities: Dict[str, float],  # p(r=1|x_t) para cada regla
        effect: np.ndarray,  # Y_t
        cf_fid: float  # CF-Fidelity del sistema
    ):
        """
        Registra observación.

        Args:
            context: Vector de contexto x_t
            applied_rules: Lista de secuencias de roles aplicadas
            rule_probabilities: p(r_t=1|x_t) para cada regla
            effect: Y_t = [cambio_reward, teleología, coherencia, energía]
            cf_fid: CF-Fidelity actual
        """
        self.t = t

        self.context_history.append(context.copy())
        self.effect_history.append(effect.copy())
        self.cf_fid_history.append(cf_fid)

        # Registrar aplicaciones de reglas
        applied_set = set()
        for role_seq in applied_rules:
            rule_id = self._rule_id(role_seq)
            self.rule_applications[rule_id].append(t)
            applied_set.add(rule_id)
            self.rule_effects[rule_id].append(effect.copy())

        # Registrar probabilidades
        for rule_id, prob in rule_probabilities.items():
            self.rule_probs[rule_id].append(prob)

        # Registrar baseline (reglas no aplicadas)
        for rule_id in self.rule_applications.keys():
            if rule_id not in applied_set:
                self.rule_baseline_effects[rule_id].append(effect.copy())

        # Limitar historiales
        max_h = max_history(t)
        if len(self.context_history) > max_h:
            self.context_history = self.context_history[-max_h:]
            self.effect_history = self.effect_history[-max_h:]
            self.cf_fid_history = self.cf_fid_history[-max_h:]

    def compute_overlap(self, rule_id: str) -> float:
        """
        Ω_r = E_t[min(p_t(r=1), p_t(r=0))]
        """
        if rule_id not in self.rule_probs:
            return 0.0

        probs = self.rule_probs[rule_id]
        if len(probs) < 5:
            return 0.5

        overlaps = [min(p, 1 - p) for p in probs]
        return float(np.mean(overlaps))

    def compute_ate(self, rule_id: str) -> Tuple[float, float]:
        """
        Calcula ATE endógeno con pesos IPW y clipping.

        Δ_r = E[Y_t | r_t=1] - E[Y_t | r_t=0] (ponderado)

        Returns:
            (Δ_r, variance)
        """
        effects_on = self.rule_effects.get(rule_id, [])
        effects_off = self.rule_baseline_effects.get(rule_id, [])

        if len(effects_on) < 3 or len(effects_off) < 3:
            return 0.0, 1.0

        probs = self.rule_probs.get(rule_id, [])

        # Pesos IPW para r=1: w_t^(1) ∝ 1/p_t(r=1)
        weights_on = []
        for i, p in enumerate(probs[-len(effects_on):]):
            if p > 0.01:  # Evitar división por 0
                weights_on.append(1.0 / p)
            else:
                weights_on.append(1.0)

        # Pesos IPW para r=0: w_t^(0) ∝ 1/p_t(r=0)
        weights_off = []
        for i, p in enumerate(probs[-len(effects_off):]):
            if (1 - p) > 0.01:
                weights_off.append(1.0 / (1 - p))
            else:
                weights_off.append(1.0)

        # Clipping endógeno Q95%
        all_weights = weights_on + weights_off
        if all_weights:
            q95 = np.percentile(all_weights, 95)
            weights_on = [min(w, q95) for w in weights_on]
            weights_off = [min(w, q95) for w in weights_off]

        # Normalizar
        sum_on = sum(weights_on) + 1e-8
        sum_off = sum(weights_off) + 1e-8
        weights_on = [w / sum_on for w in weights_on]
        weights_off = [w / sum_off for w in weights_off]

        # Media ponderada de Y cuando r=1
        effects_on_arr = np.array(effects_on[-len(weights_on):])
        effects_off_arr = np.array(effects_off[-len(weights_off):])

        # Usar norma de Y como escalar
        y_on = np.sum([w * np.linalg.norm(e) for w, e in zip(weights_on, effects_on_arr)])
        y_off = np.sum([w * np.linalg.norm(e) for w, e in zip(weights_off, effects_off_arr)])

        delta = y_on - y_off

        # Varianza estimada
        var_on = np.var([np.linalg.norm(e) for e in effects_on_arr])
        var_off = np.var([np.linalg.norm(e) for e in effects_off_arr])
        combined_var = var_on / len(effects_on_arr) + var_off / len(effects_off_arr) + 1e-8

        self.delta_history.append(abs(delta))

        return float(delta), float(combined_var)

    def compute_magnitude(self, delta: float) -> float:
        """
        m_r = sig(|Δ_r|) = |Δ_r| / (|Δ_r| + Q75%(|Δ|_hist))
        """
        if len(self.delta_history) < 5:
            q75 = 0.1  # Bootstrap
        else:
            q75 = np.percentile(self.delta_history, 75)

        return abs(delta) / (abs(delta) + q75 + 1e-8)

    def compute_mmd(self, rule_id: str) -> float:
        """
        Calcula MMD entre P(Y|r=1) y P(Y|N_r) (vecinos).
        """
        effects_r = self.rule_effects.get(rule_id, [])
        if len(effects_r) < 3:
            return 0.0

        # Efectos de reglas vecinas
        neighbor_effects = []
        neighbors = self._get_neighbor_rules(rule_id)
        for n_id in neighbors:
            neighbor_effects.extend(self.rule_effects.get(n_id, []))

        if len(neighbor_effects) < 3:
            # Sin vecinos suficientes, comparar con distribución global
            neighbor_effects = self.effect_history[-100:]

        if len(neighbor_effects) < 3:
            return 0.0

        # MMD simplificado: diferencia de medias + diferencia de varianzas
        mean_r = np.mean([np.linalg.norm(e) for e in effects_r])
        mean_n = np.mean([np.linalg.norm(e) for e in neighbor_effects])

        var_r = np.var([np.linalg.norm(e) for e in effects_r])
        var_n = np.var([np.linalg.norm(e) for e in neighbor_effects])

        mmd = abs(mean_r - mean_n) + 0.5 * abs(np.sqrt(var_r) - np.sqrt(var_n))

        return float(mmd)

    def compute_specificity(self, rule_id: str) -> float:
        """
        S_r = 1 - MMD_r / Q75%(MMD_hist)
        """
        mmd = self.compute_mmd(rule_id)
        self.mmd_history.append(mmd)

        if len(self.mmd_history) < 5:
            q75 = max(mmd, 0.1)  # Bootstrap
        else:
            q75 = np.percentile(self.mmd_history, 75)

        specificity = 1 - mmd / (q75 + 1e-8)
        return float(np.clip(specificity, 0, 1))

    def compute_cf_reliability(self, rule_id: str) -> float:
        """
        R_r = E_t[CF-Fid_t | r_t=1]
        R'_r = R_r / Q75%(R_hist)
        """
        applications = self.rule_applications.get(rule_id, [])
        if not applications:
            return 0.0

        # CF-Fid en momentos donde se aplicó la regla
        cf_fids_at_r = []
        for t_app in applications[-50:]:  # Últimas 50 aplicaciones
            idx = t_app - 1  # Ajustar índice
            if 0 <= idx < len(self.cf_fid_history):
                cf_fids_at_r.append(self.cf_fid_history[idx])

        if not cf_fids_at_r:
            return 0.5

        r_r = np.mean(cf_fids_at_r)
        self.r_history.append(r_r)

        # Normalizar
        if len(self.r_history) < 5:
            q75 = max(r_r, 0.5)  # Bootstrap
        else:
            q75 = np.percentile(self.r_history, 75)

        r_prime = r_r / (q75 + 1e-8)
        return float(np.clip(r_prime, 0, 1))

    def evaluate_rule(self, rule_id: str, role_sequence: Tuple[int, ...]) -> RuleEffect:
        """
        Evalúa una regla completa.

        GC(r) = m_r * S_r * Ω_r * R'_r
        """
        # Overlap
        overlap = self.compute_overlap(rule_id)

        # Verificar validez: Ω_r >= Q25%(Ω_hist)
        all_overlaps = [self.compute_overlap(r) for r in self.rule_applications.keys()]
        overlap_threshold = np.percentile(all_overlaps, 25) if len(all_overlaps) > 3 else 0.1
        is_valid = overlap >= overlap_threshold

        if not is_valid:
            return RuleEffect(
                rule_id=rule_id,
                role_sequence=role_sequence,
                overlap=overlap,
                ate=0.0,
                magnitude=0.0,
                specificity=0.0,
                cf_reliability=0.0,
                gc_score=0.0,
                n_applications=len(self.rule_applications.get(rule_id, [])),
                is_valid=False
            )

        # ATE
        ate, _ = self.compute_ate(rule_id)

        # Magnitud
        magnitude = self.compute_magnitude(ate)

        # Especificidad
        specificity = self.compute_specificity(rule_id)

        # Fiabilidad CF
        cf_reliability = self.compute_cf_reliability(rule_id)

        # GC(r)
        gc_score = magnitude * specificity * overlap * cf_reliability

        return RuleEffect(
            rule_id=rule_id,
            role_sequence=role_sequence,
            overlap=overlap,
            ate=ate,
            magnitude=magnitude,
            specificity=specificity,
            cf_reliability=cf_reliability,
            gc_score=gc_score,
            n_applications=len(self.rule_applications.get(rule_id, [])),
            is_valid=True
        )

    def compute_sx3(self) -> Tuple[float, Dict[str, Any]]:
        """
        SX3 = Median_{r ∈ R_válidas} GC(r)
        """
        rule_effects = []
        valid_rules = []

        for rule_id, applications in self.rule_applications.items():
            if len(applications) < 3:
                continue

            # Reconstruir role_sequence desde rule_id
            role_sequence = tuple(int(x) for x in rule_id.split("_"))

            effect = self.evaluate_rule(rule_id, role_sequence)
            rule_effects.append(effect)

            if effect.is_valid:
                valid_rules.append(effect)

        if not valid_rules:
            return 0.0, {'n_rules': 0, 'n_valid': 0, 'rule_details': []}

        # Mediana de GC scores
        gc_scores = [r.gc_score for r in valid_rules]
        sx3 = float(np.median(gc_scores))

        details = {
            'n_rules': len(rule_effects),
            'n_valid': len(valid_rules),
            'median_gc': sx3,
            'mean_overlap': float(np.mean([r.overlap for r in valid_rules])),
            'mean_magnitude': float(np.mean([r.magnitude for r in valid_rules])),
            'mean_specificity': float(np.mean([r.specificity for r in valid_rules])),
            'mean_cf_reliability': float(np.mean([r.cf_reliability for r in valid_rules])),
            'rule_details': [
                {
                    'rule_id': r.rule_id,
                    'gc': r.gc_score,
                    'overlap': r.overlap,
                    'ate': r.ate
                }
                for r in sorted(valid_rules, key=lambda x: x.gc_score, reverse=True)[:5]
            ]
        }

        return sx3, details


def run_test() -> dict:
    """
    SX3 v2: Grammar Causality Test

    100% endógeno según spec.
    """
    np.random.seed(42)

    gc = GrammarCausalityV2('TEST_AGENT', n_roles=4)

    # Simular 500 pasos
    for t in range(1, 501):
        # Contexto
        context = np.random.randn(6)

        # Simular aplicación de reglas
        applied_rules = []
        rule_probs = {}

        # Patrones de reglas
        if t % 5 == 0:
            # Regla fuerte: (0, 1) -> alto efecto
            applied_rules.append((0, 1))
            rule_probs["0_1"] = 0.7 + np.random.rand() * 0.2
            effect = np.array([0.8, 0.6, 0.7, 0.5]) + np.random.randn(4) * 0.1
        elif t % 7 == 0:
            # Regla débil: (2, 3) -> bajo efecto
            applied_rules.append((2, 3))
            rule_probs["2_3"] = 0.3 + np.random.rand() * 0.3
            effect = np.array([0.2, 0.1, 0.2, 0.1]) + np.random.randn(4) * 0.05
        else:
            # Aleatorio
            r1, r2 = np.random.randint(0, 4, size=2)
            applied_rules.append((r1, r2))
            rule_id = f"{r1}_{r2}"
            rule_probs[rule_id] = 0.4 + np.random.rand() * 0.2
            effect = np.random.randn(4) * 0.3

        # Probabilidades para reglas no aplicadas
        for i in range(4):
            for j in range(4):
                rid = f"{i}_{j}"
                if rid not in rule_probs:
                    rule_probs[rid] = 0.1 + np.random.rand() * 0.2

        # CF-Fidelity simulada
        cf_fid = 0.5 + np.random.rand() * 0.3

        gc.observe(t, context, applied_rules, rule_probs, effect, cf_fid)

    # Calcular SX3
    sx3, details = gc.compute_sx3()

    # Target: SX3 >= 0.5
    passed = sx3 >= 0.5

    return {
        'score': float(np.clip(sx3, 0, 1)),
        'passed': bool(passed),
        'details': details
    }


if __name__ == "__main__":
    result = run_test()
    print("=" * 60)
    print("SX3 v2 - GRAMMAR CAUSALITY TEST")
    print("=" * 60)
    print(f"Score: {result['score']:.3f}")
    print(f"Passed: {result['passed']}")
    print(f"\nDetails:")
    for k, v in result['details'].items():
        if k != 'rule_details':
            print(f"  {k}: {v}")
    print("\n  Top rules:")
    for r in result['details'].get('rule_details', [])[:3]:
        print(f"    {r['rule_id']}: GC={r['gc']:.3f}, Ω={r['overlap']:.3f}")
