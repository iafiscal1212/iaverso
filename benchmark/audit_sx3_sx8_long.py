"""
Auditoría larga de SX3 v2 y SX8 v2 en WORLD-1 social.

Solo lectura: no modifica políticas, recompensas ni arquitectura.
Ejecuta el sistema multi-agente durante miles de pasos y mide
la emergencia de causalidad gramatical y coordinación simbólica.
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict

from cognition.agi_dynamic_constants import L_t, max_history


# =============================================================================
# WORLD-1 SOCIAL (W3) - Simplified multi-agent world
# =============================================================================

class World1Social:
    """
    WORLD-1 con dinámica social simplificada para auditoría.
    No modifica agentes, solo provee entorno reactivo.
    """

    def __init__(self, n_agents: int = 5, state_dim: int = 12):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.t = 0

        # World state
        self.state = np.random.randn(state_dim) * 0.1

        # Agent states (internal, not modified by us)
        self.agent_states = {f"A{i}": np.random.randn(state_dim) * 0.1
                           for i in range(n_agents)}

        # Sensitive fields (first 4 dimensions respond nonlinearly)
        self.n_sensitive = 4

        # History for deriving thresholds
        self.action_history: List[np.ndarray] = []
        self.state_history: List[np.ndarray] = []
        self.reward_history: List[float] = []

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Execute one step. Actions come from agents, we just observe.
        Returns new state and rewards (for observation only).
        """
        self.t += 1

        # Aggregate actions
        all_actions = list(actions.values())
        if not all_actions:
            return self.state.copy(), {}

        mean_action = np.mean(all_actions, axis=0)
        action_norm = np.linalg.norm(mean_action)

        # Store for threshold derivation
        self.action_history.append(mean_action.copy())
        if len(self.action_history) > max_history(self.t):
            self.action_history = self.action_history[-max_history(self.t):]

        # === World dynamics ===
        # Base drift
        drift = np.random.randn(self.state_dim) * 0.05

        # Sensitive field response (nonlinear kick)
        if len(self.action_history) > 10:
            action_scale = np.percentile([np.linalg.norm(a) for a in self.action_history], 75)
        else:
            action_scale = 1.0

        # Coordination detection
        if len(all_actions) >= 2:
            # How aligned are agents?
            normalized_actions = []
            for a in all_actions:
                norm = np.linalg.norm(a)
                if norm > 1e-8:
                    normalized_actions.append(a / norm)
            if normalized_actions:
                mean_dir = np.mean(normalized_actions, axis=0)
                coordination = np.linalg.norm(mean_dir)  # 0 to 1
            else:
                coordination = 0
        else:
            coordination = 0

        # Apply nonlinear kick to sensitive fields
        sensitive_action = mean_action[:self.n_sensitive]
        activation = 1 / (1 + np.exp(-action_norm / (action_scale + 1e-8)))
        kick = activation * coordination * np.sign(sensitive_action) * np.abs(sensitive_action) ** 0.5

        # Update state
        self.state[:self.n_sensitive] += kick * 0.3
        self.state += drift
        self.state += mean_action * 0.1  # Linear response

        # Clip state
        self.state = np.clip(self.state, -5, 5)
        self.state_history.append(self.state.copy())

        # Compute rewards (observation only)
        rewards = {}
        for agent_id, action in actions.items():
            # Reward based on state-action alignment
            reward = np.dot(action, self.state) / (np.linalg.norm(action) * np.linalg.norm(self.state) + 1e-8)
            rewards[agent_id] = float(reward)
            self.reward_history.append(reward)

        return self.state.copy(), rewards


# =============================================================================
# SIMPLE AGENT (Autonomous, we don't control it)
# =============================================================================

class SimpleAgent:
    """
    Agente autónomo simple. Aprende por sí mismo, no lo dirigimos.
    """

    def __init__(self, agent_id: str, state_dim: int):
        self.agent_id = agent_id
        self.state_dim = state_dim

        # Internal state
        self.internal_state = np.random.randn(state_dim) * 0.1

        # Simple policy: linear with noise
        self.policy_weights = np.random.randn(state_dim, state_dim) * 0.1

        # Learning rate derived from history
        self.reward_history: List[float] = []
        self.action_history: List[np.ndarray] = []

        # Symbols this agent uses
        self.symbols: Set[str] = set()
        self.symbol_usage: Dict[str, int] = defaultdict(int)

    def act(self, world_state: np.ndarray) -> np.ndarray:
        """Generate action based on world state."""
        # Policy: linear transform + exploration noise
        base_action = self.policy_weights @ world_state

        # Exploration noise (derived from reward variance)
        if len(self.reward_history) > 10:
            noise_scale = np.std(self.reward_history[-50:]) * 0.5
        else:
            noise_scale = 0.3

        noise = np.random.randn(self.state_dim) * noise_scale
        action = base_action + noise

        # Normalize
        norm = np.linalg.norm(action)
        if norm > 1:
            action = action / norm

        self.action_history.append(action.copy())
        return action

    def learn(self, reward: float, world_state: np.ndarray):
        """Simple learning update."""
        self.reward_history.append(reward)

        # Derive learning rate from reward statistics
        if len(self.reward_history) > 20:
            lr = 0.01 / (1 + np.std(self.reward_history[-20:]))
        else:
            lr = 0.01

        # Simple policy gradient-like update
        if self.action_history:
            last_action = self.action_history[-1]
            # Update weights in direction of reward * state * action
            update = lr * reward * np.outer(last_action, world_state)
            self.policy_weights += update

            # Regularization
            self.policy_weights *= 0.999

    def emit_symbol(self, t: int) -> str:
        """Emit a symbol based on internal state + temporal context."""
        # Symbol combines internal state and recent reward trend
        state_hash = tuple(np.sign(self.internal_state[:4]).astype(int))
        base_symbol = hash(state_hash) % 15

        # Add reward-based modulation for variety
        if len(self.reward_history) > 5:
            reward_trend = np.mean(self.reward_history[-5:]) - np.mean(self.reward_history[-10:-5]) if len(self.reward_history) > 10 else 0
            if reward_trend > 0.1:
                base_symbol = (base_symbol + 1) % 15
            elif reward_trend < -0.1:
                base_symbol = (base_symbol + 2) % 15

        symbol = f"S{base_symbol}"

        self.symbols.add(symbol)
        self.symbol_usage[symbol] += 1

        return symbol

    def update_internal(self, world_state: np.ndarray, other_symbols: List[str]):
        """Update internal state based on world and social signals."""
        # World influence
        self.internal_state = 0.9 * self.internal_state + 0.1 * world_state

        # Social influence (symbols from others)
        if other_symbols:
            # Symbols influence internal state dimensions
            for sym in other_symbols:
                idx = hash(sym) % self.state_dim
                self.internal_state[idx] += 0.05 * np.sign(self.internal_state[idx])


# =============================================================================
# SX3 v2 METER - Grammar Causality
# =============================================================================

class SX3v2Meter:
    """
    Medidor SX3 v2: Grammar Causality.

    Para cada regla r = (antecedente → consecuente):
    - Ω_r: overlap de soporte común
    - Δ_r: ATE con IPW
    - m_r: magnitud sigmoid
    - S_r: especificidad via MMD
    - R'_r: confiabilidad CF

    GC(r) = m_r * S_r * Ω_r * R'_r
    SX3 = Mediana de GC(r) sobre reglas válidas
    """

    def __init__(self):
        self.rules: Dict[str, Dict] = {}  # rule_id -> stats
        self.symbol_occurrences: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)
        self.outcome_history: List[float] = []
        self.t = 0

    def observe(self, t: int, symbols: List[str], world_state: np.ndarray, rewards: Dict[str, float]):
        """Record observation for rule analysis."""
        self.t = t

        # Outcome: mean reward
        outcome = np.mean(list(rewards.values())) if rewards else 0
        self.outcome_history.append(outcome)

        # Record symbol occurrences with context
        for sym in symbols:
            self.symbol_occurrences[sym].append((t, world_state.copy()))

        # Limit history
        max_h = max_history(t)
        for sym in self.symbol_occurrences:
            if len(self.symbol_occurrences[sym]) > max_h:
                self.symbol_occurrences[sym] = self.symbol_occurrences[sym][-max_h:]

        # Build rules from symbol pairs
        if len(symbols) >= 2:
            for i, ante in enumerate(symbols):
                for j, cons in enumerate(symbols):
                    if i != j:
                        rule_id = f"{ante}->{cons}"
                        if rule_id not in self.rules:
                            self.rules[rule_id] = {
                                'ante': ante,
                                'cons': cons,
                                'co_occurrences': [],
                                'ante_only': [],
                                'cons_only': [],
                                'neither': []
                            }
                        self.rules[rule_id]['co_occurrences'].append((t, outcome, world_state.copy()))

    def _compute_overlap(self, rule_id: str) -> float:
        """Compute Ω_r: overlap of support."""
        rule = self.rules[rule_id]
        n_co = len(rule['co_occurrences'])
        n_ante = len(self.symbol_occurrences.get(rule['ante'], []))
        n_cons = len(self.symbol_occurrences.get(rule['cons'], []))

        if n_ante == 0 or n_cons == 0:
            return 0.0

        # Overlap = co-occurrence / min(ante, cons)
        overlap = n_co / (min(n_ante, n_cons) + 1e-8)
        return float(np.clip(overlap, 0, 1))

    def _compute_ate(self, rule_id: str) -> float:
        """Compute Δ_r: Average Treatment Effect."""
        rule = self.rules[rule_id]

        if len(rule['co_occurrences']) < 5:
            return 0.0

        # Outcomes when rule fires vs baseline
        co_outcomes = [o for _, o, _ in rule['co_occurrences']]

        # Baseline: overall mean
        L = L_t(self.t)
        baseline = np.mean(self.outcome_history[-L:]) if self.outcome_history else 0

        ate = np.mean(co_outcomes) - baseline
        return float(ate)

    def _compute_magnitude(self, ate: float) -> float:
        """Compute m_r: sigmoid magnitude."""
        # Derive scale from outcome variance
        if len(self.outcome_history) > 10:
            scale = np.std(self.outcome_history) + 1e-8
        else:
            scale = 0.5

        # Sigmoid
        m = 1 / (1 + np.exp(-abs(ate) / scale))
        return float(m)

    def _compute_specificity(self, rule_id: str) -> float:
        """Compute S_r: specificity via context variance."""
        rule = self.rules[rule_id]

        if len(rule['co_occurrences']) < 3:
            return 0.5

        # Context vectors when rule fires
        contexts = [c for _, _, c in rule['co_occurrences']]

        # Specificity = 1 - normalized variance
        context_var = np.mean([np.var(c) for c in contexts])

        # Derive normalization from overall state variance
        if self.outcome_history:
            all_contexts = []
            for sym in self.symbol_occurrences:
                for _, c in self.symbol_occurrences[sym][-20:]:
                    all_contexts.append(c)
            if all_contexts:
                global_var = np.mean([np.var(c) for c in all_contexts])
                specificity = 1 - context_var / (global_var + 1e-8)
                return float(np.clip(specificity, 0, 1))

        return 0.5

    def _compute_cf_reliability(self, rule_id: str) -> float:
        """Compute R'_r: counterfactual reliability."""
        rule = self.rules[rule_id]

        if len(rule['co_occurrences']) < 5:
            return 0.5

        # Check consistency of effect
        outcomes = [o for _, o, _ in rule['co_occurrences']]

        # Reliability = 1 - CV (coefficient of variation)
        mean_o = np.mean(outcomes)
        std_o = np.std(outcomes)

        if abs(mean_o) < 1e-8:
            return 0.5

        cv = std_o / (abs(mean_o) + 1e-8)
        reliability = 1 / (1 + cv)

        return float(reliability)

    def compute_gc(self, rule_id: str) -> Tuple[float, Dict]:
        """Compute GC(r) for a rule."""
        omega = self._compute_overlap(rule_id)
        ate = self._compute_ate(rule_id)
        m = self._compute_magnitude(ate)
        s = self._compute_specificity(rule_id)
        r_prime = self._compute_cf_reliability(rule_id)

        gc = m * s * omega * r_prime

        return gc, {
            'omega': omega,
            'ate': ate,
            'm': m,
            's': s,
            'r_prime': r_prime
        }

    def get_results(self) -> Dict:
        """Get SX3 v2 results."""
        valid_rules = []
        gc_values = []

        # Derive validity threshold from overlap distribution
        all_overlaps = [self._compute_overlap(r) for r in self.rules]
        if all_overlaps:
            omega_threshold = np.percentile(all_overlaps, 25)  # Q25 as minimum
        else:
            omega_threshold = 0.1

        for rule_id in self.rules:
            omega = self._compute_overlap(rule_id)
            if omega >= omega_threshold and len(self.rules[rule_id]['co_occurrences']) >= 5:
                gc, details = self.compute_gc(rule_id)
                valid_rules.append({
                    'rule_id': rule_id,
                    'gc': gc,
                    **details
                })
                gc_values.append(gc)

        if gc_values:
            median_gc = float(np.median(gc_values))
        else:
            median_gc = 0.0

        return {
            'sx3_v2': median_gc,
            'n_rules_total': len(self.rules),
            'n_rules_valid': len(valid_rules),
            'omega_threshold': omega_threshold,
            'top_rules': sorted(valid_rules, key=lambda x: x['gc'], reverse=True)[:5]
        }


# =============================================================================
# SX8 v2 METER - Multi-Agent Coordination
# =============================================================================

class SX8v2Meter:
    """
    Medidor SX8 v2: Multi-Agent Coordination.

    Para cada símbolo σ compartido:
    - ΔΦ_σ: cambio en world state
    - ΔC_σ: cambio en coherencia de agentes
    - G_σ, D_σ: sigmoid normalizados
    - A_σ: asociación temporal (convención)

    Coord(σ) = G_σ * D_σ * A_σ
    SX8 = Mediana de Coord(σ)
    """

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.t = 0

        # Symbol tracking
        self.symbol_events: Dict[str, List[Dict]] = defaultdict(list)
        self.agent_symbols: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # State tracking
        self.world_state_history: List[np.ndarray] = []
        self.coherence_history: List[float] = []
        self.action_history: Dict[str, List[np.ndarray]] = defaultdict(list)

    def observe(self, t: int, agent_symbols: Dict[str, str],
                world_state: np.ndarray, agent_actions: Dict[str, np.ndarray]):
        """Record observation for coordination analysis."""
        self.t = t
        self.world_state_history.append(world_state.copy())

        # Compute coherence (action alignment)
        if len(agent_actions) >= 2:
            actions = list(agent_actions.values())
            normalized = []
            for a in actions:
                norm = np.linalg.norm(a)
                if norm > 1e-8:
                    normalized.append(a / norm)
            if normalized:
                mean_dir = np.mean(normalized, axis=0)
                coherence = float(np.linalg.norm(mean_dir))
            else:
                coherence = 0
        else:
            coherence = 0
        self.coherence_history.append(coherence)

        # Record actions
        for agent_id, action in agent_actions.items():
            self.action_history[agent_id].append(action.copy())

        # Record symbol emissions
        for agent_id, symbol in agent_symbols.items():
            self.agent_symbols[agent_id][symbol] += 1

            # Record event with context
            self.symbol_events[symbol].append({
                't': t,
                'agent': agent_id,
                'world_state': world_state.copy(),
                'coherence': coherence,
                'n_users': sum(1 for a, s in agent_symbols.items() if s == symbol)
            })

        # Limit history
        max_h = max_history(t)
        if len(self.world_state_history) > max_h:
            self.world_state_history = self.world_state_history[-max_h:]
            self.coherence_history = self.coherence_history[-max_h:]
        for sym in self.symbol_events:
            if len(self.symbol_events[sym]) > max_h:
                self.symbol_events[sym] = self.symbol_events[sym][-max_h:]

    def _compute_delta_phi(self, symbol: str) -> float:
        """Compute ΔΦ_σ: world state change when symbol used."""
        events = self.symbol_events[symbol]
        if len(events) < 3 or len(self.world_state_history) < 10:
            return 0.0

        # State change at symbol events
        deltas = []
        for event in events[-20:]:
            t = event['t']
            if t > 0 and t < len(self.world_state_history):
                delta = np.linalg.norm(
                    self.world_state_history[t] - self.world_state_history[t-1]
                )
                deltas.append(delta)

        if not deltas:
            return 0.0

        # Compare to baseline
        all_deltas = []
        for i in range(1, min(100, len(self.world_state_history))):
            all_deltas.append(np.linalg.norm(
                self.world_state_history[i] - self.world_state_history[i-1]
            ))

        if not all_deltas:
            return 0.0

        delta_phi = np.mean(deltas) - np.mean(all_deltas)
        return float(delta_phi)

    def _compute_delta_c(self, symbol: str) -> float:
        """Compute ΔC_σ: coherence change when symbol used."""
        events = self.symbol_events[symbol]
        if len(events) < 3 or len(self.coherence_history) < 10:
            return 0.0

        # Coherence at symbol events (immediate)
        event_coherence = [e['coherence'] for e in events[-20:]]

        # Baseline: coherence in windows WITHOUT this symbol
        # Get timesteps when symbol was NOT used
        event_times = set(e['t'] for e in events)
        non_event_coherence = [
            self.coherence_history[i]
            for i in range(max(0, len(self.coherence_history) - 100), len(self.coherence_history))
            if i not in event_times
        ]

        if not non_event_coherence:
            # Fallback to global baseline
            baseline = np.mean(self.coherence_history[-100:])
        else:
            baseline = np.mean(non_event_coherence)

        delta_c = np.mean(event_coherence) - baseline
        return float(delta_c)

    def _compute_g_d(self, delta_phi: float, delta_c: float) -> Tuple[float, float]:
        """Compute G_σ and D_σ: sigmoid normalized impacts."""
        # Derive scales from history
        if len(self.world_state_history) > 20:
            phi_scale = np.std([np.linalg.norm(s) for s in self.world_state_history[-50:]]) + 1e-8
        else:
            phi_scale = 1.0

        if len(self.coherence_history) > 20:
            c_scale = np.std(self.coherence_history[-50:]) + 1e-8
        else:
            c_scale = 0.5

        # Sigmoid
        g = 1 / (1 + np.exp(-delta_phi / phi_scale))
        d = 1 / (1 + np.exp(-delta_c / c_scale))

        return float(g), float(d)

    def _compute_association(self, symbol: str) -> float:
        """Compute A_σ: temporal association (convention strength)."""
        events = self.symbol_events[symbol]
        if len(events) < 5:
            return 0.5

        # How many agents use this symbol?
        agents_using = set(e['agent'] for e in events)
        agent_coverage = len(agents_using) / self.n_agents

        # Consistency: do they use it together?
        multi_user_events = [e for e in events if e['n_users'] > 1]
        if events:
            co_use_rate = len(multi_user_events) / len(events)
        else:
            co_use_rate = 0

        # Association = coverage * co-use rate
        # Transformed to (0,1) via sigmoid
        raw_a = agent_coverage * co_use_rate
        a = 1 / (1 + np.exp(-5 * (raw_a - 0.2)))  # Shifted sigmoid

        return float(a)

    def compute_coord(self, symbol: str) -> Tuple[float, Dict]:
        """Compute Coord(σ) for a symbol."""
        delta_phi = self._compute_delta_phi(symbol)
        delta_c = self._compute_delta_c(symbol)
        g, d = self._compute_g_d(delta_phi, delta_c)
        a = self._compute_association(symbol)

        coord = g * d * a

        return coord, {
            'delta_phi': delta_phi,
            'delta_c': delta_c,
            'g': g,
            'd': d,
            'a': a
        }

    def get_results(self) -> Dict:
        """Get SX8 v2 results."""
        coord_values = []
        symbol_details = []

        # Only consider symbols with enough events
        min_events = max(5, int(np.sqrt(self.t / 10)))

        for symbol in self.symbol_events:
            if len(self.symbol_events[symbol]) >= min_events:
                coord, details = self.compute_coord(symbol)
                coord_values.append(coord)
                symbol_details.append({
                    'symbol': symbol,
                    'coord': coord,
                    'n_events': len(self.symbol_events[symbol]),
                    **details
                })

        if coord_values:
            median_coord = float(np.median(coord_values))
            q75_coord = float(np.percentile(coord_values, 75))
            max_coord = float(np.max(coord_values))
            n_positive = sum(1 for c in coord_values if c > 0)
        else:
            median_coord = 0.0
            q75_coord = 0.0
            max_coord = 0.0
            n_positive = 0

        # Find examples with positive ΔΦ, ΔC, and high A
        good_examples = [
            s for s in symbol_details
            if s['delta_phi'] > 0 and s['delta_c'] > 0 and s['a'] > 0.5
        ]

        return {
            'sx8_v2': median_coord,
            'sx8_v2_q75': q75_coord,
            'sx8_v2_max': max_coord,
            'n_symbols_positive': n_positive,
            'n_symbols_total': len(self.symbol_events),
            'n_symbols_active': len(coord_values),
            'min_events_threshold': min_events,
            'mean_coherence': np.mean(self.coherence_history) if self.coherence_history else 0,
            'top_symbols': sorted(symbol_details, key=lambda x: x['coord'], reverse=True)[:5],
            'good_examples': good_examples[:3]
        }


# =============================================================================
# MAIN AUDIT
# =============================================================================

def run_audit(n_steps: int = 5000, n_agents: int = 5, state_dim: int = 12,
              report_every: int = 1000):
    """
    Run long audit of SX3 v2 and SX8 v2.

    Pure observation - no intervention in policies or rewards.
    """
    print("=" * 70)
    print("AUDITORÍA LARGA: SX3 v2 + SX8 v2 en WORLD-1 SOCIAL")
    print("=" * 70)
    print(f"  Pasos: {n_steps}")
    print(f"  Agentes: {n_agents}")
    print(f"  Dimensión estado: {state_dim}")
    print("  Modo: Solo lectura (no intervención)")
    print("=" * 70)

    np.random.seed(42)

    # Initialize world and agents
    world = World1Social(n_agents=n_agents, state_dim=state_dim)
    agents = {f"A{i}": SimpleAgent(f"A{i}", state_dim) for i in range(n_agents)}

    # Initialize meters
    sx3_meter = SX3v2Meter()
    sx8_meter = SX8v2Meter(n_agents=n_agents)

    # Run simulation
    for t in range(1, n_steps + 1):
        # === Agents act (we don't control this) ===
        actions = {}
        for agent_id, agent in agents.items():
            actions[agent_id] = agent.act(world.state)

        # === World step ===
        new_state, rewards = world.step(actions)

        # === Agents learn (autonomous) ===
        for agent_id, agent in agents.items():
            if agent_id in rewards:
                agent.learn(rewards[agent_id], new_state)

        # === Symbol emission (autonomous) ===
        agent_symbols = {}
        all_symbols = []
        for agent_id, agent in agents.items():
            symbol = agent.emit_symbol(t)
            agent_symbols[agent_id] = symbol
            all_symbols.append(symbol)

        # === Update internal states (social influence) ===
        for agent_id, agent in agents.items():
            other_syms = [s for a, s in agent_symbols.items() if a != agent_id]
            agent.update_internal(new_state, other_syms)

        # === OBSERVATION ONLY: Record for meters ===
        sx3_meter.observe(t, all_symbols, new_state, rewards)
        sx8_meter.observe(t, agent_symbols, new_state, actions)

        # === Progress report ===
        if t % report_every == 0:
            sx3_results = sx3_meter.get_results()
            sx8_results = sx8_meter.get_results()

            print(f"\n  t={t}:")
            print(f"    SX3 v2: {sx3_results['sx3_v2']:.4f} ({sx3_results['n_rules_valid']}/{sx3_results['n_rules_total']} reglas válidas)")
            print(f"    SX8 v2: {sx8_results['sx8_v2']:.4f} ({sx8_results['n_symbols_active']} símbolos activos)")
            print(f"    Coherencia media: {sx8_results['mean_coherence']:.3f}")

    # === FINAL RESULTS ===
    print("\n" + "=" * 70)
    print("RESULTADOS FINALES")
    print("=" * 70)

    sx3_final = sx3_meter.get_results()
    sx8_final = sx8_meter.get_results()

    print("\n--- SX3 v2 (Grammar Causality) ---")
    print(f"  Mediana GC(r): {sx3_final['sx3_v2']:.4f}")
    print(f"  Reglas válidas: {sx3_final['n_rules_valid']} / {sx3_final['n_rules_total']}")
    print(f"  Umbral Ω: {sx3_final['omega_threshold']:.3f}")
    print("\n  Top 5 reglas:")
    for rule in sx3_final['top_rules']:
        print(f"    {rule['rule_id']}: GC={rule['gc']:.4f}, Ω={rule['omega']:.3f}, m={rule['m']:.3f}, S={rule['s']:.3f}")

    print("\n--- SX8 v2 (Multi-Agent Coordination) ---")
    print(f"  Mediana Coord(σ): {sx8_final['sx8_v2']:.4f}")
    print(f"  Q75 Coord(σ): {sx8_final.get('sx8_v2_q75', 0):.4f}")
    print(f"  Max Coord(σ): {sx8_final.get('sx8_v2_max', 0):.4f}")
    print(f"  Símbolos con Coord>0: {sx8_final.get('n_symbols_positive', 0)} / {sx8_final['n_symbols_active']}")
    print(f"  Coherencia media global: {sx8_final['mean_coherence']:.3f}")
    print("\n  Top 5 símbolos:")
    for sym in sx8_final['top_symbols']:
        print(f"    {sym['symbol']}: Coord={sym['coord']:.4f}, G={sym['g']:.3f}, D={sym['d']:.3f}, A={sym['a']:.3f}")

    if sx8_final['good_examples']:
        print("\n  Ejemplos con ΔΦ>0, ΔC>0, A alto:")
        for ex in sx8_final['good_examples']:
            print(f"    {ex['symbol']}: ΔΦ={ex['delta_phi']:.4f}, ΔC={ex['delta_c']:.4f}, A={ex['a']:.3f}")
    else:
        print("\n  (Aún no hay ejemplos con todos los criterios positivos)")

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"  SX3 v2: {sx3_final['sx3_v2']:.4f}")
    print(f"  SX8 v2: {sx8_final['sx8_v2']:.4f}")
    print("=" * 70)

    return {
        'sx3_v2': sx3_final,
        'sx8_v2': sx8_final,
        'n_steps': n_steps
    }


if __name__ == "__main__":
    results = run_audit(n_steps=5000, n_agents=5, report_every=1000)
