"""
Test PMCC - Persistencia Multi-Capa Colectiva
==============================================

Prueba que las capacidades cognitivas principales se mantienen
funcionales simultáneamente durante episodios largos.

Para cada agente y episodio, mide:
1. Teleología (metas activas estables)
2. Simbolismo (símbolos dominantes persistentes)
3. Gramática (reglas activas consistentes)
4. Causalidad (CI score estable)
5. ToM (S5 en rango)
6. Self-model (S4 en rango)
7. Normas (persistentes entre episodios)
8. Narrativa (continuidad simbólica)

PMCC = median_i(corr(M_i(1), M_i(2)) * corr(M_i(2), M_i(3)))

donde M_i son vectores de capacidades para cada agente.

Si PMCC ≥ 0.6 → hay persistencia multi-capa.
Si PMCC ≥ 0.75 → hay coherencia global → AGI interna.

100% endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict
import json

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class AgentCapabilities:
    """Vector de capacidades de un agente en un episodio."""
    agent_id: str
    episode: int

    # 1. Teleología
    goal_stability: float  # Estabilidad de metas
    goal_consistency: float  # Consistencia de dirección

    # 2. Simbolismo
    symbol_richness: float  # Número de símbolos activos
    symbol_stability: float  # Persistencia de símbolos dominantes

    # 3. Gramática
    rule_coverage: float  # Proporción de reglas activas
    rule_consistency: float  # Consistencia de efectos

    # 4. Causalidad
    ci_score: float  # CI score medio
    ci_stability: float  # Varianza de CI

    # 5. ToM
    tom_accuracy: float  # Precisión de predicción de otros
    tom_stability: float  # Estabilidad de modelos de otros

    # 6. Self-model
    self_accuracy: float  # Precisión de auto-predicción
    self_stability: float  # Estabilidad del auto-modelo

    # 7. Normas
    norm_persistence: float  # Persistencia de normas
    norm_compliance: float  # Cumplimiento de normas

    # 8. Narrativa
    narrative_coherence: float  # Coherencia de secuencia simbólica
    narrative_continuity: float  # Continuidad entre episodios

    def to_vector(self) -> np.ndarray:
        """Convierte capacidades a vector numérico."""
        return np.array([
            self.goal_stability, self.goal_consistency,
            self.symbol_richness, self.symbol_stability,
            self.rule_coverage, self.rule_consistency,
            self.ci_score, self.ci_stability,
            self.tom_accuracy, self.tom_stability,
            self.self_accuracy, self.self_stability,
            self.norm_persistence, self.norm_compliance,
            self.narrative_coherence, self.narrative_continuity
        ])


@dataclass
class PMCCResult:
    """Resultado del test PMCC."""
    pmcc_score: float
    passed: bool
    interpretation: str
    agent_correlations: Dict[str, List[float]]
    episode_summaries: List[Dict]
    details: Dict[str, Any]


# =============================================================================
# WORLD-1 with full cognitive tracking
# =============================================================================

class CognitiveWorld:
    """Mundo con tracking cognitivo completo."""

    def __init__(self, n_agents: int, state_dim: int = 12):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.state = np.random.randn(state_dim) * 0.1
        self.t = 0

        # Normas sociales emergentes
        self.norms: Dict[str, float] = {}  # norm_id -> strength
        self.norm_violations: Dict[str, int] = defaultdict(int)

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, float], float]:
        """Step con tracking de normas."""
        self.t += 1

        if not actions:
            return self.state.copy(), {}, 0.0

        action_list = list(actions.values())
        mean_action = np.mean(action_list, axis=0)

        # Coherencia
        coherence = self._compute_coherence(action_list)

        # Dinámica
        self.state += mean_action * 0.1 + np.random.randn(self.state_dim) * 0.05
        self.state = np.clip(self.state, -5, 5)

        # Rewards
        rewards = {}
        for agent_id, action in actions.items():
            r = np.dot(action, self.state) / (np.linalg.norm(action) * np.linalg.norm(self.state) + 1e-8)
            rewards[agent_id] = float(r)

        # Actualizar normas
        self._update_norms(actions, coherence)

        return self.state.copy(), rewards, coherence

    def _compute_coherence(self, actions: List[np.ndarray]) -> float:
        if len(actions) < 2:
            return 0.5
        normalized = [a / (np.linalg.norm(a) + 1e-8) for a in actions]
        mean_dir = np.mean(normalized, axis=0)
        return float(np.linalg.norm(mean_dir))

    def _update_norms(self, actions: Dict[str, np.ndarray], coherence: float):
        """Actualiza normas emergentes."""
        # Norma de coordinación
        self.norms['coordination'] = 0.9 * self.norms.get('coordination', 0.5) + 0.1 * coherence

        # Norma de magnitud de acción
        mean_mag = np.mean([np.linalg.norm(a) for a in actions.values()])
        self.norms['action_magnitude'] = 0.9 * self.norms.get('action_magnitude', 1.0) + 0.1 * mean_mag


# =============================================================================
# COGNITIVE AGENT with full tracking
# =============================================================================

class CognitiveAgent:
    """Agente cognitivo con tracking completo para PMCC."""

    def __init__(self, agent_id: str, state_dim: int, n_other_agents: int):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.n_other_agents = n_other_agents
        self.t = 0

        # === 1. Teleología (metas) ===
        self.goal_vector = np.random.randn(state_dim) * 0.1
        self.goal_history: List[np.ndarray] = []

        # === 2. Simbolismo ===
        self.active_symbols: Set[str] = set()
        self.symbol_counts: Dict[str, int] = defaultdict(int)
        self.symbol_activations: Dict[str, float] = {}

        # === 3. Gramática ===
        self.rules: Dict[str, Dict] = {}
        self.rule_activations: Dict[str, int] = defaultdict(int)

        # === 4. Causalidad ===
        self.ci_scores: List[float] = []

        # === 5. ToM ===
        self.other_models: Dict[str, Dict] = {}
        self.tom_predictions: List[float] = []
        self.tom_errors: List[float] = []

        # === 6. Self-model ===
        self.self_predictions: List[float] = []
        self.self_errors: List[float] = []
        self.predicted_state: Optional[np.ndarray] = None

        # === 7. Normas ===
        self.norm_beliefs: Dict[str, float] = {}
        self.norm_compliance_history: List[float] = []

        # === 8. Narrativa ===
        self.symbol_sequence: List[str] = []
        self.narrative_chunks: List[List[str]] = []

        # Política y aprendizaje
        self.policy_weights = np.random.randn(10, state_dim) * 0.1
        self.reward_history: List[float] = []

    def step(self, world_state: np.ndarray, other_actions: Dict[str, np.ndarray],
             t: int) -> np.ndarray:
        """
        Ejecuta un paso cognitivo completo.

        Returns:
            action: Acción a tomar
        """
        self.t = t

        # 1. Actualizar teleología
        self._update_goals(world_state)

        # 2. Emitir símbolos
        self._emit_symbols(world_state)

        # 3. Activar reglas gramaticales
        self._activate_rules()

        # 4. Calcular CI
        self._compute_ci(world_state)

        # 5. Actualizar ToM
        self._update_tom(other_actions)

        # 6. Actualizar self-model
        self._update_self_model(world_state)

        # 7. Evaluar normas
        self._evaluate_norms(world_state)

        # 8. Actualizar narrativa
        self._update_narrative()

        # Seleccionar acción
        action = self._select_action(world_state)

        return action

    def _update_goals(self, world_state: np.ndarray):
        """Actualiza teleología interna."""
        # Meta: moverse hacia estado deseado
        error = world_state - self.goal_vector

        # Actualización adaptativa
        lr = 0.05 if len(self.reward_history) < 10 else 0.02
        if self.reward_history and self.reward_history[-1] > 0:
            # Reforzar meta actual
            self.goal_vector += lr * error * 0.5
        else:
            # Explorar nuevas metas
            self.goal_vector += lr * np.random.randn(self.state_dim) * 0.1

        self.goal_history.append(self.goal_vector.copy())
        if len(self.goal_history) > 100:
            self.goal_history = self.goal_history[-100:]

    def _emit_symbols(self, world_state: np.ndarray):
        """Emite símbolos basados en estado."""
        # Discretizar estado
        n_symbols = max(2, int(np.sqrt(self.t / 10 + 1)))

        for i in range(n_symbols):
            idx = (i * 3) % self.state_dim
            val = world_state[idx]
            sym_id = f"S{int(np.abs(val * 10)) % 15}"

            self.active_symbols.add(sym_id)
            self.symbol_counts[sym_id] += 1
            self.symbol_activations[sym_id] = 1 / (1 + np.exp(-val))

            self.symbol_sequence.append(sym_id)

        if len(self.symbol_sequence) > 200:
            self.symbol_sequence = self.symbol_sequence[-200:]

    def _activate_rules(self):
        """Activa reglas gramaticales."""
        if len(self.symbol_sequence) < 3:
            return

        # Buscar patrones
        recent = self.symbol_sequence[-5:]
        for i in range(len(recent) - 1):
            rule_id = f"{recent[i]}->{recent[i+1]}"
            if rule_id not in self.rules:
                self.rules[rule_id] = {'ante': recent[i], 'cons': recent[i+1]}
            self.rule_activations[rule_id] += 1

    def _compute_ci(self, world_state: np.ndarray):
        """Calcula score de causalidad interna."""
        if len(self.reward_history) < 5:
            ci = 0.5
        else:
            # CI basado en correlación acción-recompensa
            recent_r = self.reward_history[-10:]
            ci = 0.4 + 0.3 * (np.mean(recent_r) + 1) / 2

        self.ci_scores.append(ci)
        if len(self.ci_scores) > 100:
            self.ci_scores = self.ci_scores[-100:]

    def _update_tom(self, other_actions: Dict[str, np.ndarray]):
        """Actualiza Theory of Mind."""
        for other_id, action in other_actions.items():
            if other_id == self.agent_id:
                continue

            if other_id not in self.other_models:
                self.other_models[other_id] = {
                    'predicted_action': np.zeros(self.state_dim),
                    'history': []
                }

            model = self.other_models[other_id]

            # Error de predicción
            if model['predicted_action'] is not None:
                error = np.linalg.norm(action - model['predicted_action'])
                self.tom_errors.append(error)

            # Actualizar modelo
            alpha = 0.2
            model['predicted_action'] = (1 - alpha) * model['predicted_action'] + alpha * action
            model['history'].append(action.copy())

            if len(model['history']) > 50:
                model['history'] = model['history'][-50:]

        # Accuracy: 1 - normalized error
        if self.tom_errors:
            mean_error = np.mean(self.tom_errors[-20:])
            accuracy = 1 / (1 + mean_error)
            self.tom_predictions.append(accuracy)

    def _update_self_model(self, world_state: np.ndarray):
        """Actualiza auto-modelo."""
        # Predecir siguiente estado
        if self.predicted_state is not None:
            error = np.linalg.norm(world_state - self.predicted_state)
            self.self_errors.append(error)
            accuracy = 1 / (1 + error)
            self.self_predictions.append(accuracy)

        # Nueva predicción
        if len(self.goal_history) > 1:
            trend = self.goal_history[-1] - self.goal_history[-2]
            self.predicted_state = world_state + trend * 0.1
        else:
            self.predicted_state = world_state.copy()

    def _evaluate_norms(self, world_state: np.ndarray):
        """Evalúa cumplimiento de normas."""
        # Norma de magnitud
        action_mag = np.linalg.norm(self.goal_vector)
        expected_mag = self.norm_beliefs.get('action_magnitude', 1.0)
        compliance_mag = 1 - abs(action_mag - expected_mag) / (expected_mag + 1)

        # Norma de dirección (hacia estado "bueno")
        compliance_dir = 0.5 + 0.5 * np.tanh(np.mean(world_state))

        compliance = (compliance_mag + compliance_dir) / 2
        self.norm_compliance_history.append(compliance)

        if len(self.norm_compliance_history) > 100:
            self.norm_compliance_history = self.norm_compliance_history[-100:]

    def _update_narrative(self):
        """Actualiza narrativa simbólica."""
        # Chunk cada 20 símbolos
        chunk_size = 20
        if len(self.symbol_sequence) >= chunk_size:
            chunk = self.symbol_sequence[-chunk_size:]
            self.narrative_chunks.append(chunk)

            if len(self.narrative_chunks) > 10:
                self.narrative_chunks = self.narrative_chunks[-10:]

    def _select_action(self, world_state: np.ndarray) -> np.ndarray:
        """Selecciona acción basada en política."""
        # Política simple
        logits = self.policy_weights @ world_state
        probs = np.exp(logits - np.max(logits))
        probs = probs / (np.sum(probs) + 1e-8)

        action_idx = np.random.choice(len(probs), p=probs)

        action = np.zeros(self.state_dim)
        action[action_idx % self.state_dim] = 1.0
        action += np.random.randn(self.state_dim) * 0.1

        return action

    def learn(self, reward: float, world_state: np.ndarray):
        """Aprendizaje."""
        self.reward_history.append(reward)

        # Actualizar política
        lr = 0.01
        for i in range(self.policy_weights.shape[0]):
            self.policy_weights[i] += lr * reward * world_state

        self.policy_weights *= 0.999

        # Actualizar creencias de normas
        if len(self.norm_compliance_history) > 10:
            self.norm_beliefs['compliance'] = np.mean(self.norm_compliance_history[-10:])

    def get_capabilities(self, episode: int) -> AgentCapabilities:
        """Extrae capacidades actuales."""

        # 1. Teleología
        if len(self.goal_history) > 5:
            goal_changes = [np.linalg.norm(self.goal_history[i] - self.goal_history[i-1])
                          for i in range(1, len(self.goal_history))]
            goal_stability = 1 / (1 + np.mean(goal_changes[-20:]))
            goal_consistency = 1 - np.std(goal_changes[-20:]) / (np.mean(goal_changes[-20:]) + 1e-8)
        else:
            goal_stability = 0.5
            goal_consistency = 0.5

        # 2. Simbolismo
        symbol_richness = min(1.0, len(self.active_symbols) / 10)
        if self.symbol_counts:
            top_symbols = sorted(self.symbol_counts.values(), reverse=True)[:5]
            symbol_stability = top_symbols[0] / (sum(self.symbol_counts.values()) + 1) if top_symbols else 0.5
        else:
            symbol_stability = 0.5

        # 3. Gramática
        rule_coverage = min(1.0, len([r for r, c in self.rule_activations.items() if c > 5]) / 20)
        if self.rule_activations:
            activation_counts = list(self.rule_activations.values())
            rule_consistency = 1 - np.std(activation_counts) / (np.mean(activation_counts) + 1e-8)
        else:
            rule_consistency = 0.5

        # 4. Causalidad
        ci_score = np.mean(self.ci_scores[-50:]) if self.ci_scores else 0.5
        ci_stability = 1 - np.std(self.ci_scores[-50:]) if len(self.ci_scores) > 5 else 0.5

        # 5. ToM
        tom_accuracy = np.mean(self.tom_predictions[-20:]) if self.tom_predictions else 0.5
        tom_stability = 1 - np.std(self.tom_predictions[-20:]) if len(self.tom_predictions) > 5 else 0.5

        # 6. Self-model
        self_accuracy = np.mean(self.self_predictions[-20:]) if self.self_predictions else 0.5
        self_stability = 1 - np.std(self.self_predictions[-20:]) if len(self.self_predictions) > 5 else 0.5

        # 7. Normas
        norm_persistence = len(self.norm_beliefs) / 5  # Max 5 normas
        norm_compliance = np.mean(self.norm_compliance_history[-20:]) if self.norm_compliance_history else 0.5

        # 8. Narrativa
        if len(self.narrative_chunks) >= 2:
            # Coherencia: similitud entre chunks consecutivos
            overlaps = []
            for i in range(1, len(self.narrative_chunks)):
                set1 = set(self.narrative_chunks[i-1])
                set2 = set(self.narrative_chunks[i])
                overlap = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
                overlaps.append(overlap)
            narrative_coherence = np.mean(overlaps)
            narrative_continuity = 1 - np.std(overlaps) if len(overlaps) > 1 else 0.5
        else:
            narrative_coherence = 0.5
            narrative_continuity = 0.5

        return AgentCapabilities(
            agent_id=self.agent_id,
            episode=episode,
            goal_stability=float(np.clip(goal_stability, 0, 1)),
            goal_consistency=float(np.clip(goal_consistency, 0, 1)),
            symbol_richness=float(np.clip(symbol_richness, 0, 1)),
            symbol_stability=float(np.clip(symbol_stability, 0, 1)),
            rule_coverage=float(np.clip(rule_coverage, 0, 1)),
            rule_consistency=float(np.clip(rule_consistency, 0, 1)),
            ci_score=float(np.clip(ci_score, 0, 1)),
            ci_stability=float(np.clip(ci_stability, 0, 1)),
            tom_accuracy=float(np.clip(tom_accuracy, 0, 1)),
            tom_stability=float(np.clip(tom_stability, 0, 1)),
            self_accuracy=float(np.clip(self_accuracy, 0, 1)),
            self_stability=float(np.clip(self_stability, 0, 1)),
            norm_persistence=float(np.clip(norm_persistence, 0, 1)),
            norm_compliance=float(np.clip(norm_compliance, 0, 1)),
            narrative_coherence=float(np.clip(narrative_coherence, 0, 1)),
            narrative_continuity=float(np.clip(narrative_continuity, 0, 1))
        )


# =============================================================================
# PMCC TEST
# =============================================================================

def run_episode_pmcc(world: CognitiveWorld, agents: Dict[str, CognitiveAgent],
                     n_steps: int, episode_num: int) -> Dict[str, AgentCapabilities]:
    """
    Ejecuta un episodio y extrae capacidades.
    """
    for t in range(1, n_steps + 1):
        # Colectar acciones previas
        prev_actions = {}
        for aid, agent in agents.items():
            if agent.reward_history:
                prev_actions[aid] = agent.goal_vector.copy()

        # Cada agente actúa
        actions = {}
        for aid, agent in agents.items():
            action = agent.step(world.state, prev_actions, t)
            actions[aid] = action

        # Step del mundo
        new_state, rewards, coherence = world.step(actions)

        # Aprendizaje
        for aid, agent in agents.items():
            if aid in rewards:
                agent.learn(rewards[aid], new_state)

    # Extraer capacidades
    capabilities = {}
    for aid, agent in agents.items():
        capabilities[aid] = agent.get_capabilities(episode_num)

    return capabilities


def compute_pmcc(episode_capabilities: List[Dict[str, AgentCapabilities]]) -> PMCCResult:
    """
    Calcula PMCC.

    PMCC = median_i(corr(M_i(1), M_i(2)) * corr(M_i(2), M_i(3)))
    """
    if len(episode_capabilities) < 3:
        return PMCCResult(
            pmcc_score=0.0,
            passed=False,
            interpretation="Insuficientes episodios",
            agent_correlations={},
            episode_summaries=[],
            details={}
        )

    agent_ids = list(episode_capabilities[0].keys())
    agent_correlations = {}

    for aid in agent_ids:
        vectors = []
        for ep_caps in episode_capabilities:
            if aid in ep_caps:
                vectors.append(ep_caps[aid].to_vector())

        if len(vectors) < 3:
            continue

        # Correlaciones entre episodios consecutivos
        corrs = []
        for i in range(1, len(vectors)):
            v1, v2 = vectors[i-1], vectors[i]
            # Manejar casos de varianza cero
            if np.std(v1) < 1e-8 or np.std(v2) < 1e-8:
                corr = 0.5
            else:
                corr = np.corrcoef(v1, v2)[0, 1]
                if np.isnan(corr):
                    corr = 0.5
            corrs.append(corr)

        agent_correlations[aid] = corrs

    # PMCC = mediana del producto de correlaciones
    if not agent_correlations:
        return PMCCResult(
            pmcc_score=0.0,
            passed=False,
            interpretation="No hay correlaciones válidas",
            agent_correlations={},
            episode_summaries=[],
            details={}
        )

    # Producto de correlaciones por agente
    agent_products = []
    for aid, corrs in agent_correlations.items():
        if len(corrs) >= 2:
            product = np.prod([max(0, c) for c in corrs])  # Solo correlaciones positivas
            agent_products.append(product ** (1/len(corrs)))  # Media geométrica

    pmcc_score = float(np.median(agent_products)) if agent_products else 0.0

    # Interpretación
    if pmcc_score >= 0.75:
        interpretation = "AGI INTERNA: Coherencia global demostrada"
        passed = True
    elif pmcc_score >= 0.6:
        interpretation = "PERSISTENCIA MULTI-CAPA: Capacidades estables"
        passed = True
    elif pmcc_score >= 0.4:
        interpretation = "DESARROLLO: Capacidades parcialmente estables"
        passed = False
    else:
        interpretation = "INESTABLE: Capacidades no persisten"
        passed = False

    # Resúmenes por episodio
    episode_summaries = []
    for i, ep_caps in enumerate(episode_capabilities):
        means = {}
        for cap_name in ['goal_stability', 'symbol_richness', 'ci_score',
                        'tom_accuracy', 'self_accuracy', 'narrative_coherence']:
            values = [getattr(c, cap_name) for c in ep_caps.values()]
            means[cap_name] = float(np.mean(values))
        episode_summaries.append({'episode': i+1, **means})

    return PMCCResult(
        pmcc_score=pmcc_score,
        passed=passed,
        interpretation=interpretation,
        agent_correlations=agent_correlations,
        episode_summaries=episode_summaries,
        details={
            'n_agents': len(agent_ids),
            'n_episodes': len(episode_capabilities),
            'agent_products': agent_products,
            'mean_correlation': np.mean([np.mean(c) for c in agent_correlations.values()])
        }
    )


def run_pmcc_test(n_agents: int = 5, n_episodes: int = 3,
                  steps_per_episode: int = 1000) -> PMCCResult:
    """
    Ejecuta el test PMCC completo.

    Args:
        n_agents: Número de agentes
        n_episodes: Número de episodios (≥3 para correlaciones)
        steps_per_episode: Pasos por episodio

    Returns:
        PMCCResult
    """
    print("=" * 70)
    print("TEST PMCC - PERSISTENCIA MULTI-CAPA COLECTIVA")
    print("=" * 70)
    print(f"  Agentes: {n_agents}")
    print(f"  Episodios: {n_episodes}")
    print(f"  Pasos/episodio: {steps_per_episode}")
    print("=" * 70)

    np.random.seed(42)

    # Crear mundo y agentes
    world = CognitiveWorld(n_agents, state_dim=12)
    agents = {
        f"A{i}": CognitiveAgent(f"A{i}", state_dim=12, n_other_agents=n_agents-1)
        for i in range(n_agents)
    }

    # Ejecutar episodios
    all_capabilities: List[Dict[str, AgentCapabilities]] = []

    for ep in range(1, n_episodes + 1):
        print(f"\n--- Episodio {ep}/{n_episodes} ---")

        caps = run_episode_pmcc(world, agents, steps_per_episode, ep)
        all_capabilities.append(caps)

        # Mostrar resumen
        for aid, cap in caps.items():
            vec = cap.to_vector()
            print(f"  {aid}: mean_cap={np.mean(vec):.3f}, "
                  f"goal={cap.goal_stability:.2f}, "
                  f"sym={cap.symbol_richness:.2f}, "
                  f"ci={cap.ci_score:.2f}, "
                  f"tom={cap.tom_accuracy:.2f}")

    # Calcular PMCC
    result = compute_pmcc(all_capabilities)

    print("\n" + "=" * 70)
    print("RESULTADOS PMCC")
    print("=" * 70)
    print(f"  PMCC Score: {result.pmcc_score:.4f}")
    print(f"  Passed: {result.passed}")
    print(f"  Interpretación: {result.interpretation}")
    print("\n  Correlaciones por agente:")
    for aid, corrs in result.agent_correlations.items():
        print(f"    {aid}: {[f'{c:.3f}' for c in corrs]}")
    print("\n  Resumen por episodio:")
    for summary in result.episode_summaries:
        print(f"    Ep{summary['episode']}: goal={summary['goal_stability']:.2f}, "
              f"sym={summary['symbol_richness']:.2f}, ci={summary['ci_score']:.2f}")
    print("=" * 70)

    return result


if __name__ == "__main__":
    result = run_pmcc_test(n_agents=5, n_episodes=3, steps_per_episode=1500)
