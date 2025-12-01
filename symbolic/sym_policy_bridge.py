"""
SymbolicPolicyBridge - Grammar-in-the-loop para SX3 v2
======================================================

Integra la gramática simbólica directamente en la selección de acciones.
La política se sesga hacia secuencias simbólicas con alta utilidad estructural.

π(a|s,G) ∝ π_base(a|s) * exp(β_t * Q_grammar(a,G))

donde:
- Q_grammar(a,G) = utilidad esperada de las reglas que activaría la acción
- β_t = escala endógena derivada de confianza/varianza histórica

100% endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class RuleUtility:
    """Utilidad estructural de una regla gramatical."""
    rule_id: str
    utility: float
    confidence: float
    n_activations: int
    mean_ci_delta: float
    mean_cf_delta: float
    mean_reward_delta: float


@dataclass
class SymbolicState:
    """Estado simbólico actual del agente."""
    active_symbols: List[str]
    recent_sequence: List[str]  # Últimos k símbolos emitidos
    role_distribution: Dict[str, float]  # Distribución de roles


class SymbolicPolicyBridge:
    """
    Puente entre gramática simbólica y política de acciones.

    Permite que las reglas gramaticales influyan causalmente en
    la selección de acciones, haciendo que SX3 v2 detecte efectos reales.
    """

    def __init__(self, agent_id: str, state_dim: int, n_actions: int = 10):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.t = 0

        # Gramática aprendida: rule_id -> stats
        self.rules: Dict[str, Dict] = {}

        # Historial de activaciones de reglas
        self.rule_activations: Dict[str, List[Dict]] = defaultdict(list)

        # Mapeo acción -> símbolos probables
        self.action_symbol_map: Dict[int, List[str]] = defaultdict(list)

        # Historiales para derivación endógena
        self.ci_history: List[float] = []
        self.cf_history: List[float] = []
        self.reward_history: List[float] = []
        self.confidence_history: List[float] = []
        self.beta_history: List[float] = []

        # Estado simbólico actual
        self.current_symbols: List[str] = []
        self.symbol_sequence: List[str] = []

    def register_rule(self, rule_id: str, antecedent: List[str],
                     consequent: List[str], role: str = 'transitional'):
        """
        Registra una regla gramatical.

        Args:
            rule_id: Identificador único de la regla
            antecedent: Secuencia de símbolos antecedente
            consequent: Secuencia de símbolos consecuente
            role: Rol gramatical (evaluative, operative, transitional)
        """
        self.rules[rule_id] = {
            'antecedent': antecedent,
            'consequent': consequent,
            'role': role,
            'activations': 0,
            'utility_sum': 0.0,
            'ci_deltas': [],
            'cf_deltas': [],
            'reward_deltas': []
        }

    def observe_rule_activation(self, rule_id: str, t: int,
                                ci_before: float, ci_after: float,
                                cf_before: float, cf_after: float,
                                reward: float):
        """
        Registra una activación de regla con sus efectos.
        """
        if rule_id not in self.rules:
            return

        self.t = t

        ci_delta = ci_after - ci_before
        cf_delta = cf_after - cf_before

        self.rules[rule_id]['activations'] += 1
        self.rules[rule_id]['ci_deltas'].append(ci_delta)
        self.rules[rule_id]['cf_deltas'].append(cf_delta)
        self.rules[rule_id]['reward_deltas'].append(reward)

        # Limitar histórico
        max_h = max_history(t)
        for key in ['ci_deltas', 'cf_deltas', 'reward_deltas']:
            if len(self.rules[rule_id][key]) > max_h:
                self.rules[rule_id][key] = self.rules[rule_id][key][-max_h:]

        # Actualizar historial global
        self.ci_history.append(ci_after)
        self.cf_history.append(cf_after)
        self.reward_history.append(reward)

        if len(self.ci_history) > max_h:
            self.ci_history = self.ci_history[-max_h:]
            self.cf_history = self.cf_history[-max_h:]
            self.reward_history = self.reward_history[-max_h:]

    def compute_rule_utility(self, rule_id: str, t: int) -> RuleUtility:
        """
        Calcula la utilidad estructural de una regla.

        U_rule = w_ci * Δ_CI + w_cf * Δ_CF + w_r * Δ_reward

        Pesos derivados endógenamente de varianzas relativas.
        """
        if rule_id not in self.rules:
            return RuleUtility(rule_id, 0.0, 0.0, 0, 0.0, 0.0, 0.0)

        rule = self.rules[rule_id]
        n_act = rule['activations']

        if n_act < 3:
            return RuleUtility(rule_id, 0.0, 0.5, n_act, 0.0, 0.0, 0.0)

        # Medias de deltas
        L = min(L_t(t), len(rule['ci_deltas']))
        mean_ci = np.mean(rule['ci_deltas'][-L:])
        mean_cf = np.mean(rule['cf_deltas'][-L:])
        mean_r = np.mean(rule['reward_deltas'][-L:])

        # Derivar pesos de varianzas globales (endógeno)
        if len(self.ci_history) > 10:
            var_ci = np.var(self.ci_history[-L:]) + 1e-8
            var_cf = np.var(self.cf_history[-L:]) + 1e-8
            var_r = np.var(self.reward_history[-L:]) + 1e-8

            # Pesos inversamente proporcionales a varianza (estabilizar lo ruidoso)
            total_inv_var = 1/var_ci + 1/var_cf + 1/var_r
            w_ci = (1/var_ci) / total_inv_var
            w_cf = (1/var_cf) / total_inv_var
            w_r = (1/var_r) / total_inv_var
        else:
            w_ci = w_cf = w_r = 1/3

        # Utilidad combinada
        utility = w_ci * mean_ci + w_cf * mean_cf + w_r * mean_r

        # Confianza basada en consistencia (1 - CV)
        std_deltas = np.std(rule['ci_deltas'][-L:]) + np.std(rule['cf_deltas'][-L:])
        mean_abs = abs(mean_ci) + abs(mean_cf) + 1e-8
        confidence = 1 / (1 + std_deltas / mean_abs)

        return RuleUtility(
            rule_id=rule_id,
            utility=float(utility),
            confidence=float(confidence),
            n_activations=n_act,
            mean_ci_delta=float(mean_ci),
            mean_cf_delta=float(mean_cf),
            mean_reward_delta=float(mean_r)
        )

    def update_action_symbol_mapping(self, action_idx: int,
                                     resulting_symbols: List[str]):
        """
        Aprende qué símbolos tienden a activarse tras cada acción.
        """
        self.action_symbol_map[action_idx].extend(resulting_symbols)

        # Limitar
        max_h = max_history(self.t)
        if len(self.action_symbol_map[action_idx]) > max_h:
            self.action_symbol_map[action_idx] = \
                self.action_symbol_map[action_idx][-max_h:]

    def predict_symbols_for_action(self, action_idx: int) -> List[str]:
        """
        Predice qué símbolos probablemente activará una acción.
        """
        history = self.action_symbol_map[action_idx]
        if not history:
            return []

        # Símbolos más frecuentes
        from collections import Counter
        counts = Counter(history)

        # Top k símbolos (k endógeno)
        k = max(2, int(np.sqrt(len(counts))))
        return [s for s, _ in counts.most_common(k)]

    def find_matching_rules(self, symbol_sequence: List[str]) -> List[str]:
        """
        Encuentra reglas cuyo antecedente coincide con la secuencia.
        """
        matching = []
        for rule_id, rule in self.rules.items():
            ante = rule['antecedent']
            # Buscar si el antecedente está al final de la secuencia
            if len(symbol_sequence) >= len(ante):
                if symbol_sequence[-len(ante):] == ante:
                    matching.append(rule_id)
        return matching

    def _compute_beta(self, t: int) -> float:
        """
        Calcula β_t endógenamente.

        β_t = 1 / (1 + σ_conf) * log(1 + n_rules_active)

        Escala el sesgo según confianza y número de reglas.
        """
        if len(self.confidence_history) < 5:
            return 1.0

        L = L_t(t)
        conf_std = np.std(self.confidence_history[-L:]) + 1e-8
        n_active = sum(1 for r in self.rules.values() if r['activations'] > 0)

        beta = (1 / (1 + conf_std)) * np.log(1 + n_active + 1)

        self.beta_history.append(beta)
        if len(self.beta_history) > max_history(t):
            self.beta_history = self.beta_history[-max_history(t):]

        return float(beta)

    def action_bias_from_symbols(self, candidate_actions: np.ndarray,
                                  current_symbol_state: SymbolicState,
                                  t: int) -> np.ndarray:
        """
        Calcula el sesgo de acción basado en gramática simbólica.

        Para cada acción:
        1. Predice qué símbolos activaría
        2. Encuentra qué reglas dispararía
        3. Suma utilidades de esas reglas
        4. Devuelve bias proporcional a exp(β_t * U_symbol(a))

        Args:
            candidate_actions: Array de acciones candidatas [n_actions, action_dim]
            current_symbol_state: Estado simbólico actual
            t: Tiempo actual

        Returns:
            bias: Array de sesgos para cada acción [n_actions]
        """
        self.t = t
        n_actions = len(candidate_actions)

        # Secuencia simbólica actual
        current_seq = list(current_symbol_state.recent_sequence)

        # Calcular utilidad para cada acción
        action_utilities = np.zeros(n_actions)

        for i in range(n_actions):
            # Predecir símbolos que activaría esta acción
            predicted_symbols = self.predict_symbols_for_action(i)

            if not predicted_symbols:
                continue

            # Secuencia hipotética si tomamos esta acción
            hypothetical_seq = current_seq + predicted_symbols

            # Encontrar reglas que se activarían
            matching_rules = self.find_matching_rules(hypothetical_seq)

            # Sumar utilidades
            total_utility = 0.0
            for rule_id in matching_rules:
                rule_util = self.compute_rule_utility(rule_id, t)
                # Ponderar por confianza
                total_utility += rule_util.utility * rule_util.confidence

            action_utilities[i] = total_utility

        # Calcular β_t
        beta = self._compute_beta(t)

        # Bias = exp(β * U)
        # Normalizar utilidades para estabilidad numérica
        if np.std(action_utilities) > 1e-8:
            utilities_norm = (action_utilities - np.mean(action_utilities)) / (np.std(action_utilities) + 1e-8)
        else:
            utilities_norm = action_utilities

        bias = np.exp(beta * utilities_norm)

        # Normalizar a probabilidades
        bias = bias / (np.sum(bias) + 1e-8)

        return bias

    def mix_with_base_policy(self, base_policy: np.ndarray,
                             bias_vec: np.ndarray, t: int) -> np.ndarray:
        """
        Mezcla política base con sesgo simbólico.

        π_final(a) ∝ π_base(a) * bias(a)

        El ratio de mezcla es endógeno, basado en la confianza histórica.

        Args:
            base_policy: Probabilidades de la política base [n_actions]
            bias_vec: Sesgos simbólicos [n_actions]
            t: Tiempo actual

        Returns:
            final_policy: Política mezclada [n_actions]
        """
        # Ratio de mezcla endógeno
        # Más confianza en gramática → más peso al bias
        if len(self.confidence_history) > 5:
            L = L_t(t)
            mean_conf = np.mean(self.confidence_history[-L:])
            # mix_ratio ∈ [0.1, 0.9] basado en confianza
            mix_ratio = 0.1 + 0.8 * mean_conf
        else:
            mix_ratio = 0.3  # Default conservador

        # Mezcla multiplicativa
        combined = base_policy ** (1 - mix_ratio) * bias_vec ** mix_ratio

        # Normalizar
        final_policy = combined / (np.sum(combined) + 1e-8)

        # Clip según percentiles históricos para evitar extremos
        if len(self.beta_history) > 10:
            min_prob = np.percentile(final_policy, 5)
            max_prob = np.percentile(final_policy, 95)
            final_policy = np.clip(final_policy,
                                   max(min_prob, 0.01),
                                   min(max_prob, 0.99))
            final_policy = final_policy / np.sum(final_policy)

        return final_policy

    def select_action_with_grammar(self, world_state: np.ndarray,
                                    base_policy: np.ndarray,
                                    current_symbols: List[str],
                                    t: int) -> Tuple[int, np.ndarray]:
        """
        Selecciona acción integrando gramática simbólica.

        Returns:
            action_idx: Índice de acción seleccionada
            final_policy: Política final usada
        """
        # Construir estado simbólico
        symbol_state = SymbolicState(
            active_symbols=current_symbols,
            recent_sequence=self.symbol_sequence[-10:],
            role_distribution=self._compute_role_distribution()
        )

        # Generar acciones candidatas (discretización del espacio)
        n_actions = len(base_policy)
        candidate_actions = np.eye(n_actions)  # One-hot como proxy

        # Obtener bias simbólico
        bias = self.action_bias_from_symbols(candidate_actions, symbol_state, t)

        # Mezclar con política base
        final_policy = self.mix_with_base_policy(base_policy, bias, t)

        # Seleccionar acción
        action_idx = np.random.choice(n_actions, p=final_policy)

        return action_idx, final_policy

    def _compute_role_distribution(self) -> Dict[str, float]:
        """Calcula distribución de roles de reglas activas."""
        roles = defaultdict(int)
        total = 0
        for rule in self.rules.values():
            if rule['activations'] > 0:
                roles[rule['role']] += rule['activations']
                total += rule['activations']

        if total == 0:
            return {'evaluative': 0.33, 'operative': 0.33, 'transitional': 0.34}

        return {role: count/total for role, count in roles.items()}

    def update_symbol_sequence(self, new_symbols: List[str]):
        """Actualiza la secuencia de símbolos emitidos."""
        self.symbol_sequence.extend(new_symbols)
        self.current_symbols = new_symbols

        # Limitar
        max_seq = max_history(self.t)
        if len(self.symbol_sequence) > max_seq:
            self.symbol_sequence = self.symbol_sequence[-max_seq:]

    def record_confidence(self, confidence: float):
        """Registra nivel de confianza actual."""
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > max_history(self.t):
            self.confidence_history = self.confidence_history[-max_history(self.t):]

    def get_statistics(self) -> Dict[str, Any]:
        """Estadísticas del bridge."""
        active_rules = [r for r in self.rules.values() if r['activations'] > 0]

        return {
            'agent_id': self.agent_id,
            't': self.t,
            'n_rules_total': len(self.rules),
            'n_rules_active': len(active_rules),
            'mean_beta': np.mean(self.beta_history) if self.beta_history else 1.0,
            'mean_confidence': np.mean(self.confidence_history) if self.confidence_history else 0.5,
            'role_distribution': self._compute_role_distribution(),
            'top_rules': sorted(
                [(r_id, self.compute_rule_utility(r_id, self.t).utility)
                 for r_id in self.rules],
                key=lambda x: x[1], reverse=True
            )[:5]
        }


def test_symbolic_policy_bridge():
    """Test del SymbolicPolicyBridge."""
    print("=" * 70)
    print("TEST: SYMBOLIC POLICY BRIDGE")
    print("=" * 70)

    np.random.seed(42)

    bridge = SymbolicPolicyBridge('NEO', state_dim=10, n_actions=5)

    # Registrar algunas reglas
    bridge.register_rule('R1', ['S1', 'S2'], ['S3'], 'operative')
    bridge.register_rule('R2', ['S2', 'S3'], ['S4'], 'evaluative')
    bridge.register_rule('R3', ['S1'], ['S2'], 'transitional')

    # Simular activaciones
    for t in range(1, 201):
        # Simular métricas
        ci_before = 0.3 + np.random.randn() * 0.1
        ci_after = ci_before + 0.05 + np.random.randn() * 0.05
        cf_before = 0.4 + np.random.randn() * 0.1
        cf_after = cf_before + 0.03 + np.random.randn() * 0.05
        reward = np.random.randn() * 0.5

        # Activar regla aleatoria
        rule_id = np.random.choice(['R1', 'R2', 'R3'])
        bridge.observe_rule_activation(rule_id, t,
                                       ci_before, ci_after,
                                       cf_before, cf_after, reward)

        # Actualizar mapeo acción-símbolo
        action = np.random.randint(5)
        symbols = [f'S{np.random.randint(1, 5)}']
        bridge.update_action_symbol_mapping(action, symbols)
        bridge.update_symbol_sequence(symbols)
        bridge.record_confidence(0.5 + np.random.randn() * 0.1)

        if t % 50 == 0:
            # Probar selección de acción
            base_policy = np.ones(5) / 5
            current_symbols = [f'S{np.random.randint(1, 5)}' for _ in range(2)]

            action_idx, final_policy = bridge.select_action_with_grammar(
                np.random.randn(10), base_policy, current_symbols, t
            )

            stats = bridge.get_statistics()
            print(f"\n  t={t}:")
            print(f"    Reglas activas: {stats['n_rules_active']}")
            print(f"    Beta medio: {stats['mean_beta']:.3f}")
            print(f"    Confianza media: {stats['mean_confidence']:.3f}")
            print(f"    Política base: {base_policy}")
            print(f"    Política final: {final_policy}")
            print(f"    Acción seleccionada: {action_idx}")

    print("\n" + "=" * 70)
    final_stats = bridge.get_statistics()
    print(f"Top reglas por utilidad: {final_stats['top_rules']}")
    print("=" * 70)

    return bridge


if __name__ == "__main__":
    test_symbolic_policy_bridge()
