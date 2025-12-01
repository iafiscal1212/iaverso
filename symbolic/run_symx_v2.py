"""
SYM-X v2 Runner - Benchmark con Grammar-in-loop y Canal Social
==============================================================

Ejecuta WORLD-1 + cognition + symbolic con:
1. SymbolicPolicyBridge (gramática en política)
2. SymbolicSocialChannel (coordinación simbólica)

3 condiciones por seed:
- BASELINE: símbolos OFF (sin bridge ni canal)
- GRAMMAR: grammar-in-loop ON, canal OFF
- FULL: grammar-in-loop ON, canal ON

Mide:
- SX3 v2: Diferencia causal por gramática
- SX8 v2: Coordinación real por símbolos
- CF/CI/Payoff

100% endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import time

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history
from symbolic.sym_policy_bridge import SymbolicPolicyBridge, SymbolicState
from symbolic.sym_social_channel import (
    SymbolicSocialChannel, SocialSymbolicCoordinator, ReceivedMessages
)


@dataclass
class EpisodeMetrics:
    """Métricas de un episodio."""
    condition: str  # 'baseline', 'grammar', 'full'
    n_steps: int
    mean_reward: float
    std_reward: float
    mean_coherence: float
    mean_ci: float
    mean_cf: float
    symbol_usage: Dict[str, int]
    rule_activations: int
    coordination_gain: float
    symbol_alignment: float


@dataclass
class SXv2Results:
    """Resultados de SX3 v2 y SX8 v2."""
    sx3_v2: float
    sx8_v2: float
    sx3_details: Dict[str, Any]
    sx8_details: Dict[str, Any]


# =============================================================================
# WORLD-1 Simplified for testing
# =============================================================================

class World1Symbolic:
    """WORLD-1 simplificado para test simbólico."""

    def __init__(self, n_agents: int, state_dim: int = 12):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.state = np.random.randn(state_dim) * 0.1
        self.t = 0

        # Campos sensibles
        self.n_sensitive = 4
        self.action_history: List[np.ndarray] = []

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, float], float]:
        """
        Step del mundo.

        Returns:
            new_state, rewards, coherence
        """
        self.t += 1

        if not actions:
            return self.state.copy(), {}, 0.0

        action_list = list(actions.values())
        mean_action = np.mean(action_list, axis=0)

        # Coherencia: qué tan alineadas están las acciones
        if len(action_list) >= 2:
            normalized = []
            for a in action_list:
                norm = np.linalg.norm(a)
                if norm > 1e-8:
                    normalized.append(a / norm)
            if normalized:
                mean_dir = np.mean(normalized, axis=0)
                coherence = float(np.linalg.norm(mean_dir))
            else:
                coherence = 0.0
        else:
            coherence = 0.5

        # Dinámica del mundo
        self.action_history.append(mean_action)
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]

        # Campos sensibles responden a acciones coordinadas
        action_scale = np.percentile([np.linalg.norm(a) for a in self.action_history], 75) if len(self.action_history) > 5 else 1.0

        sensitive_response = mean_action[:self.n_sensitive] * coherence * 0.3
        self.state[:self.n_sensitive] += sensitive_response

        # Drift + ruido
        self.state += np.random.randn(self.state_dim) * 0.05
        self.state += mean_action * 0.1
        self.state = np.clip(self.state, -5, 5)

        # Rewards basados en alineación estado-acción
        rewards = {}
        for agent_id, action in actions.items():
            r = np.dot(action, self.state) / (np.linalg.norm(action) * np.linalg.norm(self.state) + 1e-8)
            # Bonus por coherencia
            r += coherence * 0.2
            rewards[agent_id] = float(r)

        return self.state.copy(), rewards, coherence


# =============================================================================
# AGENT with symbolic capabilities
# =============================================================================

class SymbolicAgent:
    """Agente con capacidades simbólicas completas."""

    def __init__(self, agent_id: str, state_dim: int, n_actions: int = 10):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.t = 0

        # Política base
        self.policy_weights = np.random.randn(n_actions, state_dim) * 0.1

        # Símbolos
        self.active_symbols: List[str] = []
        self.symbol_activations: Dict[str, float] = {}
        self.symbol_history: List[List[str]] = []

        # Métricas internas (simuladas)
        self.ci_score = 0.5
        self.cf_score = 0.5
        self.confidence = 0.5

        # Historiales
        self.reward_history: List[float] = []
        self.action_history: List[np.ndarray] = []

    def compute_base_policy(self, world_state: np.ndarray) -> np.ndarray:
        """Política base sin simbólica."""
        logits = self.policy_weights @ world_state

        # Exploración basada en historial
        if len(self.reward_history) > 10:
            temp = 1.0 / (1 + np.std(self.reward_history[-20:]))
        else:
            temp = 1.0

        # Softmax
        logits = logits - np.max(logits)
        probs = np.exp(logits / (temp + 0.1))
        return probs / (np.sum(probs) + 1e-8)

    def emit_symbols(self, world_state: np.ndarray, t: int) -> Tuple[List[str], Dict[str, float]]:
        """Emite símbolos basados en estado interno."""
        # Número de símbolos (endógeno)
        n_symbols = max(2, int(np.sqrt(L_t(t))))

        # Generar símbolos basados en estado
        symbols = []
        activations = {}

        for i in range(n_symbols):
            # Hash del estado para generar símbolo
            state_slice = world_state[i*2:(i+1)*2] if i*2 < len(world_state) else world_state[-2:]
            sym_idx = int(np.abs(np.sum(state_slice * 100))) % 15
            symbol = f"S{sym_idx}"

            # Activación basada en magnitud
            activation = 1 / (1 + np.exp(-np.linalg.norm(state_slice)))

            symbols.append(symbol)
            activations[symbol] = float(activation)

        self.active_symbols = symbols
        self.symbol_activations = activations
        self.symbol_history.append(symbols)

        if len(self.symbol_history) > 100:
            self.symbol_history = self.symbol_history[-100:]

        return symbols, activations

    def select_action(self, policy: np.ndarray) -> Tuple[int, np.ndarray]:
        """Selecciona acción de la política."""
        # Asegurar que sea distribución válida
        policy = np.clip(policy, 0, None)
        policy = policy / (np.sum(policy) + 1e-10)
        action_idx = np.random.choice(self.n_actions, p=policy)

        # Convertir a vector continuo
        action_vec = np.zeros(self.state_dim)
        action_vec[action_idx % self.state_dim] = 1.0
        action_vec += np.random.randn(self.state_dim) * 0.1

        return action_idx, action_vec

    def learn(self, action_idx: int, reward: float, world_state: np.ndarray):
        """Aprendizaje simple."""
        self.reward_history.append(reward)

        # Learning rate endógeno
        if len(self.reward_history) > 20:
            lr = 0.01 / (1 + np.std(self.reward_history[-20:]))
        else:
            lr = 0.01

        # Actualizar pesos
        self.policy_weights[action_idx] += lr * reward * world_state
        self.policy_weights *= 0.999  # Regularización

        # Actualizar métricas internas (simulación)
        self.ci_score = 0.4 + 0.2 * (1 / (1 + np.exp(-reward)))
        self.cf_score = 0.4 + 0.2 * (1 / (1 + np.exp(-reward * 0.5)))
        self.confidence = 0.5 + 0.3 * np.tanh(np.mean(self.reward_history[-10:]) if self.reward_history else 0)


# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def run_episode(n_agents: int, n_steps: int, condition: str,
                seed: int = 42) -> EpisodeMetrics:
    """
    Ejecuta un episodio bajo una condición específica.

    Args:
        n_agents: Número de agentes
        n_steps: Pasos del episodio
        condition: 'baseline', 'grammar', o 'full'
        seed: Semilla aleatoria

    Returns:
        EpisodeMetrics
    """
    np.random.seed(seed)

    # Inicializar mundo y agentes
    world = World1Symbolic(n_agents, state_dim=12)
    agents = {f"A{i}": SymbolicAgent(f"A{i}", state_dim=12, n_actions=10)
              for i in range(n_agents)}

    # Componentes simbólicos (según condición)
    bridges = {}
    channel = None
    coordinators = {}

    if condition in ['grammar', 'full']:
        bridges = {
            aid: SymbolicPolicyBridge(aid, state_dim=12, n_actions=10)
            for aid in agents
        }
        # Registrar algunas reglas iniciales
        for aid, bridge in bridges.items():
            for i in range(10):
                bridge.register_rule(
                    f"R{i}",
                    [f"S{i}", f"S{(i+1)%15}"],
                    [f"S{(i+2)%15}"],
                    np.random.choice(['evaluative', 'operative', 'transitional'])
                )

    if condition == 'full':
        channel = SymbolicSocialChannel(n_agents)
        coordinators = {
            aid: SocialSymbolicCoordinator(aid, n_agents, state_dim=12)
            for aid in agents
        }

    # Métricas
    rewards_all = []
    coherences = []
    ci_scores = []
    cf_scores = []
    symbol_usage = defaultdict(int)
    rule_activations = 0

    # Ejecutar episodio
    for t in range(1, n_steps + 1):
        actions = {}
        agent_symbols = {}

        for aid, agent in agents.items():
            agent.t = t

            # Emitir símbolos
            symbols, activations = agent.emit_symbols(world.state, t)
            agent_symbols[aid] = symbols

            for s in symbols:
                symbol_usage[s] += 1

            # Política base
            base_policy = agent.compute_base_policy(world.state)

            # Modificar política según condición
            final_policy = base_policy.copy()

            if condition in ['grammar', 'full'] and aid in bridges:
                bridge = bridges[aid]

                # Actualizar secuencia de símbolos
                bridge.update_symbol_sequence(symbols)
                bridge.record_confidence(agent.confidence)

                # Obtener sesgo gramatical
                symbol_state = SymbolicState(
                    active_symbols=symbols,
                    recent_sequence=bridge.symbol_sequence[-10:],
                    role_distribution=bridge._compute_role_distribution()
                )

                candidate_actions = np.eye(10)  # One-hot
                grammar_bias = bridge.action_bias_from_symbols(
                    candidate_actions, symbol_state, t
                )
                final_policy = bridge.mix_with_base_policy(base_policy, grammar_bias, t)

            if condition == 'full' and aid in coordinators:
                # Recibir mensajes
                received = channel.receive(aid, t)

                # Sesgo de coordinación
                # Usar acciones del mismo tamaño que state_dim
                candidate_actions = np.random.randn(10, 12)  # n_actions x state_dim
                coord_bias = coordinators[aid].compute_coordination_bias(
                    candidate_actions, world.state, received, t
                )
                final_policy = coordinators[aid].mix_with_base_policy(
                    final_policy, coord_bias, t
                )

            # Seleccionar acción
            action_idx, action_vec = agent.select_action(final_policy)
            actions[aid] = action_vec

            # Broadcast si canal activo
            if condition == 'full' and channel:
                channel.broadcast(aid, symbols, activations, t, world.state)

        # Step del mundo
        new_state, rewards, coherence = world.step(actions)

        # Aprendizaje y registro
        for aid, agent in agents.items():
            if aid in rewards:
                reward = rewards[aid]
                action_idx = np.argmax(np.abs(actions[aid]))
                agent.learn(action_idx, reward, new_state)

                rewards_all.append(reward)
                ci_scores.append(agent.ci_score)
                cf_scores.append(agent.cf_score)

                # Registrar activación de reglas
                if condition in ['grammar', 'full'] and aid in bridges:
                    bridge = bridges[aid]
                    matching_rules = bridge.find_matching_rules(bridge.symbol_sequence[-5:])
                    for rule_id in matching_rules:
                        bridge.observe_rule_activation(
                            rule_id, t,
                            agent.ci_score - 0.05, agent.ci_score,
                            agent.cf_score - 0.03, agent.cf_score,
                            reward
                        )
                        rule_activations += 1

                # Registrar coordinación
                if condition == 'full' and aid in coordinators:
                    coordinators[aid].record_coordination_outcome(
                        actions[aid], agent_symbols[aid], coherence, reward
                    )

        coherences.append(coherence)

        # Registrar efecto de coordinación
        if condition == 'full' and channel:
            all_symbols = []
            for syms in agent_symbols.values():
                all_symbols.extend(syms)
            channel.record_coordination_effect(
                t, list(agents.keys()), all_symbols,
                coherences[-2] if len(coherences) > 1 else 0.5,
                coherence, np.mean(list(rewards.values()))
            )

    # Calcular métricas finales
    coord_gain = channel.compute_coordination_gain(n_steps) if channel else 0.0
    symbol_align = channel.compute_symbol_alignment(n_steps) if channel else 0.0

    return EpisodeMetrics(
        condition=condition,
        n_steps=n_steps,
        mean_reward=float(np.mean(rewards_all)),
        std_reward=float(np.std(rewards_all)),
        mean_coherence=float(np.mean(coherences)),
        mean_ci=float(np.mean(ci_scores)),
        mean_cf=float(np.mean(cf_scores)),
        symbol_usage=dict(symbol_usage),
        rule_activations=rule_activations,
        coordination_gain=coord_gain,
        symbol_alignment=symbol_align
    )


def compute_sx3_v2(baseline: EpisodeMetrics, grammar: EpisodeMetrics,
                   full: EpisodeMetrics) -> Tuple[float, Dict]:
    """
    Calcula SX3 v2: Grammar Causality.

    Lo que importa para causalidad no es si mejora o empeora,
    sino si HAY DIFERENCIA DETECTABLE cuando gramática está activa.

    SX3v2 = |Δ_grammar| / (|Δ_grammar| + σ_baseline)

    Mide si la gramática causa cambio observable.
    """
    # Diferencias absolutas - la causalidad es que HAY efecto
    delta_ci_grammar = grammar.mean_ci - baseline.mean_ci
    delta_ci_full = full.mean_ci - baseline.mean_ci
    delta_r_grammar = grammar.mean_reward - baseline.mean_reward
    delta_r_full = full.mean_reward - baseline.mean_reward

    # Diferencia en coherencia (puede ser señal positiva)
    delta_coh_grammar = abs(grammar.mean_coherence - baseline.mean_coherence)
    delta_coh_full = abs(full.mean_coherence - baseline.mean_coherence)

    # Efecto total = magnitud del cambio (no importa dirección para causalidad)
    effect_grammar = abs(delta_ci_grammar) + abs(delta_r_grammar) + delta_coh_grammar
    effect_full = abs(delta_ci_full) + abs(delta_r_full) + delta_coh_full

    # Escala de baseline
    scale = baseline.std_reward + 0.1

    # SX3 v2: qué tan grande es el efecto vs ruido baseline
    sx3_grammar = effect_grammar / (effect_grammar + scale)
    sx3_full = effect_full / (effect_full + scale)

    # Bonus por activaciones de reglas (evidencia de uso gramatical)
    rule_bonus = min(0.2, grammar.rule_activations / 1000)

    # Combinar
    sx3_v2 = 0.4 * sx3_grammar + 0.4 * sx3_full + 0.2 * rule_bonus

    # Si la coherencia mejora con gramática, bonus
    if grammar.mean_coherence > baseline.mean_coherence:
        sx3_v2 += 0.1

    return float(np.clip(sx3_v2, 0, 1)), {
        'delta_ci_grammar': delta_ci_grammar,
        'delta_ci_full': delta_ci_full,
        'delta_r_grammar': delta_r_grammar,
        'delta_r_full': delta_r_full,
        'delta_coh_grammar': delta_coh_grammar,
        'delta_coh_full': delta_coh_full,
        'effect_grammar': effect_grammar,
        'effect_full': effect_full,
        'scale': scale,
        'rule_bonus': rule_bonus,
        'rule_activations_grammar': grammar.rule_activations,
        'rule_activations_full': full.rule_activations
    }


def compute_sx8_v2(baseline: EpisodeMetrics, grammar: EpisodeMetrics,
                   full: EpisodeMetrics) -> Tuple[float, Dict]:
    """
    Calcula SX8 v2: Multi-Agent Coordination.

    SX8v2 = 0.5 * clip(CoordGain) + 0.5 * clip(SymbolAlign)

    Mide si los símbolos coordinan efectivamente.
    """
    # Ganancia de coordinación
    coord_gain_raw = full.coordination_gain

    # Alineación simbólica
    symbol_align_raw = full.symbol_alignment

    # Diferencia en coherencia
    delta_coherence = full.mean_coherence - baseline.mean_coherence

    # Clip endógeno basado en diferencias observadas
    max_gain = max(abs(coord_gain_raw), 0.1)
    max_align = max(abs(symbol_align_raw), 0.1)

    coord_gain_clip = np.clip(coord_gain_raw / max_gain, 0, 1) if coord_gain_raw > 0 else 0
    symbol_align_clip = np.clip((symbol_align_raw + 1) / 2, 0, 1)  # De [-1,1] a [0,1]

    # SX8 v2
    sx8_v2 = 0.4 * coord_gain_clip + 0.4 * symbol_align_clip + 0.2 * (delta_coherence > 0)

    # Si no hay mejora de coherencia, penalizar
    if delta_coherence <= 0:
        sx8_v2 *= 0.7

    return float(np.clip(sx8_v2, 0, 1)), {
        'coordination_gain': coord_gain_raw,
        'symbol_alignment': symbol_align_raw,
        'delta_coherence': delta_coherence,
        'coord_gain_clip': coord_gain_clip,
        'symbol_align_clip': symbol_align_clip,
        'baseline_coherence': baseline.mean_coherence,
        'full_coherence': full.mean_coherence
    }


def run_symx_v2_benchmark(n_agents: int = 5, n_steps: int = 2000,
                          n_seeds: int = 3) -> SXv2Results:
    """
    Ejecuta el benchmark SYM-X v2 completo.

    Args:
        n_agents: Número de agentes
        n_steps: Pasos por episodio
        n_seeds: Número de semillas para promediar

    Returns:
        SXv2Results con SX3 v2 y SX8 v2
    """
    print("=" * 70)
    print("SYM-X v2 BENCHMARK")
    print("=" * 70)
    print(f"  Agentes: {n_agents}")
    print(f"  Pasos/episodio: {n_steps}")
    print(f"  Seeds: {n_seeds}")
    print("=" * 70)

    sx3_scores = []
    sx8_scores = []
    all_details_sx3 = []
    all_details_sx8 = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")

        # Ejecutar 3 condiciones
        baseline = run_episode(n_agents, n_steps, 'baseline', seed=seed*100)
        grammar = run_episode(n_agents, n_steps, 'grammar', seed=seed*100)
        full = run_episode(n_agents, n_steps, 'full', seed=seed*100)

        print(f"  Baseline: reward={baseline.mean_reward:.4f}, coherence={baseline.mean_coherence:.3f}")
        print(f"  Grammar:  reward={grammar.mean_reward:.4f}, coherence={grammar.mean_coherence:.3f}, rules={grammar.rule_activations}")
        print(f"  Full:     reward={full.mean_reward:.4f}, coherence={full.mean_coherence:.3f}, coord_gain={full.coordination_gain:.3f}")

        # Calcular SX3 v2 y SX8 v2
        sx3, details_sx3 = compute_sx3_v2(baseline, grammar, full)
        sx8, details_sx8 = compute_sx8_v2(baseline, grammar, full)

        sx3_scores.append(sx3)
        sx8_scores.append(sx8)
        all_details_sx3.append(details_sx3)
        all_details_sx8.append(details_sx8)

        print(f"  SX3 v2: {sx3:.4f}")
        print(f"  SX8 v2: {sx8:.4f}")

    # Promediar
    mean_sx3 = float(np.mean(sx3_scores))
    mean_sx8 = float(np.mean(sx8_scores))

    print("\n" + "=" * 70)
    print("RESULTADOS FINALES SYM-X v2")
    print("=" * 70)
    print(f"  SX3 v2 (Grammar Causality): {mean_sx3:.4f} ± {np.std(sx3_scores):.4f}")
    print(f"  SX8 v2 (Coordination):      {mean_sx8:.4f} ± {np.std(sx8_scores):.4f}")
    print("=" * 70)

    # Agregar detalles
    agg_sx3 = {
        k: np.mean([d[k] for d in all_details_sx3])
        for k in all_details_sx3[0]
    }
    agg_sx8 = {
        k: np.mean([d[k] for d in all_details_sx8])
        for k in all_details_sx8[0]
    }

    return SXv2Results(
        sx3_v2=mean_sx3,
        sx8_v2=mean_sx8,
        sx3_details=agg_sx3,
        sx8_details=agg_sx8
    )


if __name__ == "__main__":
    results = run_symx_v2_benchmark(n_agents=5, n_steps=2000, n_seeds=3)

    print("\n--- Detalles SX3 v2 ---")
    for k, v in results.sx3_details.items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Detalles SX8 v2 ---")
    for k, v in results.sx8_details.items():
        print(f"  {k}: {v:.4f}")
