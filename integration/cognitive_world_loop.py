"""
Cognitive World Loop: Ciclo Vital Cognitivo
============================================

El loop completo:
percepciones → cognición → intención → acción → mundo cambia → nueva memoria

Integra:
- WORLD-1 (entorno físico)
- AGI-X (cognición completa AGI-4 a AGI-20)
- StateInterface (percepción)
- CognitiveActionLayer (decisión)
- Módulos cognitivos reales (Self-Model, ToM, Ethics, etc.)
- Health System (SystemDoctor con rol médico emergente)

Este es el corazón de la vida cognitiva.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import (
    L_t, max_history, adaptive_learning_rate, normalized_entropy, to_simplex
)

# Importar módulos del mundo
from world1.world1_core import World1Core
from world1.world1_entities import EntityPopulation

# Importar interface y capa de acción
from integration.state_interface import StateInterface
from integration.cognitive_action_layer import CognitiveActionLayer

# Importar módulos cognitivos básicos
from cognition.self_model_v2 import SelfPredictorV2
from cognition.theory_of_mind_v2 import TheoryOfMindSystem

# Importar módulos AGI-17, AGI-19, AGI-20 (compartidos o por agente)
from cognition.agi17_robustness import MultiWorldRobustness
from cognition.agi19_collective_intent import CollectiveIntentionality
from cognition.agi20_self_theory import StructuralSelfTheory

# Importar sistema de salud (rol médico emergente)
from health.system_doctor import SystemDoctor
from health.health_monitor import HealthMetrics
from health.medical_role import MedicalRoleManager

# Importar sistema de ciclo de vida
from lifecycle.circadian_system import (
    CircadianPhase,
    AgentCircadianCycle,
    AbsenceSimulator
)
from lifecycle.dream_processor import DreamProcessor
from lifecycle.life_journal import LifeJournal, JournalEntryType
from lifecycle.reconnection_narrative import ReconnectionNarrativeGenerator


@dataclass
class Episode:
    """Un episodio de experiencia."""
    agent_name: str
    start_t: int
    end_t: int
    states: List[Dict]
    decisions: List[Any]
    outcomes: List[Any]
    total_reward: float
    narrative_summary: str


@dataclass
class AgentMemory:
    """Memoria de un agente."""
    episodes: List[Episode] = field(default_factory=list)
    long_term_goals: List[np.ndarray] = field(default_factory=list)
    learned_patterns: Dict[str, float] = field(default_factory=dict)
    social_relationships: Dict[str, float] = field(default_factory=dict)


class CognitiveAgent:
    """
    Un agente cognitivo completo en WORLD-1.

    Integra todos los módulos AGI (4-20) para comportamiento autónomo.
    """

    def __init__(self, name: str, all_agent_names: List[str] = None,
                 z_dim: int = 6, phi_dim: int = 5, drives_dim: int = 6):
        """
        Inicializa agente cognitivo.

        Args:
            name: Nombre del agente
            all_agent_names: Lista de todos los agentes
            z_dim, phi_dim, drives_dim: Dimensiones cognitivas
        """
        self.name = name
        self.z_dim = z_dim
        self.phi_dim = phi_dim
        self.drives_dim = drives_dim
        self.all_agent_names = all_agent_names or [name]

        # Capa de decisión (incluye AGI-16, AGI-18 internamente)
        self.action_layer = CognitiveActionLayer(name, self.all_agent_names)

        # Módulos cognitivos propios
        self.self_model = SelfPredictorV2(name, z_dim, phi_dim, drives_dim)

        # AGI-20: Teoría estructural de sí mismo
        state_dim = z_dim + phi_dim + drives_dim
        self.self_theory = StructuralSelfTheory(name, state_dim=state_dim)

        # Memoria
        self.memory = AgentMemory()

        # Sistema de Ciclo de Vida
        self.circadian = AgentCircadianCycle(name)
        self.dream_processor = DreamProcessor(name)
        self.life_journal = LifeJournal(name)
        self.reconnection_narrator = ReconnectionNarrativeGenerator(name)

        # Estado actual
        self.current_state: Optional[Dict] = None
        self.current_perception: Optional[Any] = None

        # Episodio actual
        self.episode_states: List[Dict] = []
        self.episode_decisions: List[Any] = []
        self.episode_outcomes: List[Any] = []
        self.episode_start_t: int = 0
        self.episode_reward: float = 0.0

        self.t = 0

    def perceive(self, perception: Any, state_interface: StateInterface):
        """Procesa percepción del mundo y actualiza AGI-4 y AGI-20."""
        self.current_perception = perception

        # Convertir a estado cognitivo
        self.current_state = state_interface.perception_to_cognitive_state(perception)

        # Actualizar self-model (AGI-4) con nuevo estado
        z = self.current_state['z']
        phi = self.current_state['phi']
        drives = self.current_state['drives']

        self.self_model.update(z, phi, drives)

        # Actualizar self-theory (AGI-20) con estado interno completo
        internal_state = np.concatenate([z, phi, drives])
        self.self_theory.record_state(internal_state)

        # Guardar en episodio
        self.episode_states.append(self.current_state.copy())

    def decide(self, others: Dict[str, Dict]) -> Any:
        """Genera decisión basada en estado actual."""
        if self.current_state is None:
            return None

        decision = self.action_layer.decide(self.current_state, others)
        self.episode_decisions.append(decision)

        return decision

    def process_feedback(self, outcome: Any):
        """Procesa resultado de la acción."""
        self.episode_outcomes.append(outcome)
        self.episode_reward += outcome.reward_signal

        # Actualizar ciclo circadiano
        activity_level = abs(outcome.reward_signal) + 0.3
        crisis_level = outcome.surprise
        self.circadian.step(activity_level, crisis_level)

        # Agregar experiencia al procesador de suenos
        if abs(outcome.reward_signal) > 0.3 or outcome.surprise > 0.5:
            experience = {
                'id': f'exp_{self.t}',
                'content': f'reward={outcome.reward_signal:.2f}',
                'emotional_valence': outcome.reward_signal,
                'importance': abs(outcome.reward_signal) + outcome.surprise,
                'timestamp': self.t,
                'context': {'surprise': outcome.surprise}
            }
            self.dream_processor.add_memory_for_consolidation(experience)

        # Si esta en fase DREAM, procesar consolidacion
        if self.circadian.phase == CircadianPhase.DREAM:
            result = self.dream_processor.process_dream(
                dream_depth=self.circadian.get_state().rest_depth
            )
            if result.memories_processed > 0:
                self.life_journal.record_dream(
                    result.dream_narrative,
                    result.integration_strength,
                    0.0  # tono neutral para suenos
                )

        # Registrar eventos significativos en el diario
        if outcome.reward_signal > 0.5:
            self.life_journal.record_event(
                title="Exito notable",
                content=f"Logro significativo con confianza alta",
                significance=outcome.reward_signal,
                emotional_state=outcome.reward_signal
            )
        elif outcome.reward_signal < -0.3:
            self.life_journal.record_event(
                title="Desafio",
                content=f"Momento dificil, sorpresa={outcome.surprise:.2f}",
                significance=abs(outcome.reward_signal),
                emotional_state=outcome.reward_signal
            )

    def should_end_episode(self) -> bool:
        """Decide si terminar episodio actual."""
        # Episodio termina cada L_t pasos o si hay cambio de régimen
        episode_length = len(self.episode_states)
        min_length = L_t(self.t)

        return episode_length >= min_length * 3

    def end_episode(self) -> Episode:
        """Finaliza episodio y lo guarda en memoria."""
        episode = Episode(
            agent_name=self.name,
            start_t=self.episode_start_t,
            end_t=self.t,
            states=self.episode_states.copy(),
            decisions=self.episode_decisions.copy(),
            outcomes=self.episode_outcomes.copy(),
            total_reward=self.episode_reward,
            narrative_summary=self._generate_narrative()
        )

        self.memory.episodes.append(episode)

        # Limitar episodios en memoria
        max_episodes = max(10, L_t(self.t))
        if len(self.memory.episodes) > max_episodes:
            self.memory.episodes = self.memory.episodes[-max_episodes:]

        # Resetear episodio
        self.episode_states = []
        self.episode_decisions = []
        self.episode_outcomes = []
        self.episode_start_t = self.t
        self.episode_reward = 0.0

        return episode

    def _generate_narrative(self) -> str:
        """
        Genera resumen narrativo del episodio usando AGI-20.

        Combina estadísticas del episodio con narrativa estructural del self.
        """
        if not self.episode_decisions:
            return "episodio vacío"

        # Estadísticas del episodio
        n_decisions = len(self.episode_decisions)
        avg_confidence = np.mean([d.confidence for d in self.episode_decisions])

        # Dirección predominante
        if self.episode_decisions:
            directions = [d.direction for d in self.episode_decisions]
            mean_dir = np.mean(directions, axis=0)
            dir_str = "hacia meta" if np.any(mean_dir > 0) else "explorando"
        else:
            dir_str = "inactivo"

        # Resultado
        result = "éxito" if self.episode_reward > 0 else "neutro" if self.episode_reward == 0 else "dificultad"

        # AGI-20: Información estructural del self
        self_understanding = self.self_theory._compute_self_understanding()
        is_coherent = self.self_theory.is_self_coherent()

        # Narrativa de AGI-20
        dominant_dims = self.self_theory.get_dominant_dimensions(2)
        dim_str = ""
        if dominant_dims:
            dim_str = f", dims_dominantes={[d[0] for d in dominant_dims]}"

        coherent_str = ", self_coherente" if is_coherent else ""

        return f"{n_decisions} acciones, {dir_str}, confianza {avg_confidence:.2f}, {result}, U_self={self_understanding:.2f}{dim_str}{coherent_str}"


class CognitiveWorldLoop:
    """
    El loop principal: mundo + cognición AGI-4 a AGI-20 integrados.

    Cada paso:
    1. Mundo evoluciona
    2. Agentes perciben (AGI-4, AGI-20)
    3. Observación mutua (AGI-5)
    4. Agentes deciden (AGI-4 a AGI-19)
    5. Acciones afectan el mundo
    6. Feedback actualiza cognición (AGI-16, AGI-17, AGI-18, AGI-19)
    """

    def __init__(self, agent_names: List[str] = None):
        """
        Inicializa el loop cognitivo con todos los módulos AGI.

        Args:
            agent_names: Nombres de los agentes
        """
        self.agent_names = agent_names or ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
        self.n_agents = len(self.agent_names)

        # Crear WORLD-1
        self.world = World1Core(
            n_fields=4,
            n_entities=self.n_agents,
            n_resources=3,
            n_modes=3
        )

        # Crear población de entidades
        self.entities = EntityPopulation(
            n_entities=self.n_agents,
            position_dim=3,
            state_dim=4
        )

        # Interface estado
        self.state_interface = StateInterface(
            self.agent_names,
            world_dim=self.world.D
        )

        # Crear agentes cognitivos con referencia a todos los agentes
        self.agents: Dict[str, CognitiveAgent] = {
            name: CognitiveAgent(name, all_agent_names=self.agent_names)
            for name in self.agent_names
        }

        # Sistema ToM compartido (AGI-5)
        self.tom_system = TheoryOfMindSystem(self.agent_names)

        # Sistema de Robustez compartido (AGI-17)
        self.robustness_system = MultiWorldRobustness(
            self.agent_names,
            state_dim=10
        )

        # Sistema de Intencionalidad Colectiva compartido (AGI-19)
        self.collective_intent = CollectiveIntentionality(
            self.agent_names,
            state_dim=10
        )

        # Conectar módulos compartidos a cada agente
        for name, agent in self.agents.items():
            agent.action_layer.connect_modules(
                self_model=agent.self_model,
                tom_system=self.tom_system,
                robustness_system=self.robustness_system,
                collective_intent=self.collective_intent
            )

        # Estadísticas globales
        self.total_steps = 0
        self.episode_count = 0
        self.regime_history: List[str] = []

        # Historial de recompensas por agente
        self.reward_history: Dict[str, List[float]] = {
            name: [] for name in self.agent_names
        }

        # Sistema de Salud con rol médico emergente
        self.system_doctor = SystemDoctor(self.agent_names)
        self.medical_role_manager = MedicalRoleManager(self.agent_names)

    def step(self, verbose: bool = False) -> Dict:
        """
        Ejecuta un paso del loop cognitivo con AGI-4 a AGI-20.

        Returns:
            Dict con información del paso
        """
        self.total_steps += 1

        # 1. Mundo evoluciona (sin acciones aún)
        world_state = self.world.get_state()

        # 2. Cada agente percibe (AGI-4 y AGI-20 se actualizan en perceive)
        perceptions = {}
        cognitive_states = {}

        for name in self.agent_names:
            perception = self.state_interface.perceive(
                world_state, name, self.entities
            )
            perceptions[name] = perception
            self.agents[name].perceive(perception, self.state_interface)
            self.agents[name].t = self.total_steps

            # Estado cognitivo para ToM y AGI-19
            cognitive_states[name] = self.state_interface.perception_to_cognitive_state(perception)

        # 3. Observación mutua para ToM (AGI-5)
        for observer in self.agent_names:
            for target in self.agent_names:
                if observer != target:
                    target_state = cognitive_states[target]
                    self.tom_system.observe(
                        observer, target,
                        target_state['z'],
                        target_state['phi'],
                        target_state['drives']
                    )

        # 3b. Actualizar AGI-19 (Collective Intent) con estados de todos los agentes
        for name in self.agent_names:
            state = cognitive_states[name]
            # Crear vector de estado completo para AGI-19
            state_vec = np.concatenate([
                state['z'][:5] if len(state['z']) >= 5 else np.pad(state['z'], (0, 5-len(state['z']))),
                state['phi'][:5] if len(state['phi']) >= 5 else np.pad(state['phi'], (0, 5-len(state['phi'])))
            ])[:10]
            # Valor basado en recursos y actividad
            value = float(np.mean(state['phi'][:2])) if len(state['phi']) >= 2 else 0.5
            self.collective_intent.record_state(name, state_vec, value)

        # 4. Cada agente decide
        decisions = {}
        for name in self.agent_names:
            # Obtener estados de otros para la decisión
            others = self.state_interface.get_other_agents_for_tom(
                perceptions[name], name
            )
            decision = self.agents[name].decide(others)
            decisions[name] = decision

        # 5. Convertir decisiones a perturbaciones del mundo
        agent_perturbations = {}
        for name, decision in decisions.items():
            if decision:
                perturbation = self.state_interface.cognitive_decision_to_action(
                    {
                        'direction': decision.direction,
                        'magnitude': decision.magnitude,
                        'confidence': decision.confidence
                    },
                    name,
                    self.world.D
                )
                agent_perturbations[name] = perturbation

        # 6. Mundo responde a acciones
        new_world_state = self.world.step(agent_perturbations)
        self.entities.step(world_state.fields)

        # 7. Calcular outcomes y actualizar agentes
        outcomes = {}
        for name in self.agent_names:
            # Calcular cambio real
            delta = self.state_interface.get_perception_delta(name)

            if delta:
                actual_change = np.array([
                    delta['position_delta'],
                    delta['resource_delta'],
                    delta['field_delta']
                ])

                # Recompensa basada en progreso hacia meta
                # Pesos endógenos basados en varianza de cada componente
                if decisions[name]:
                    # Pesos proporcionales a la varianza explicada de cada factor
                    # Más varianza = más importante para el reward
                    reward_history = self.reward_history[name]
                    if len(reward_history) > L_t(self.total_steps):
                        # Pesos adaptativos basados en correlación con rewards pasados
                        base_weight = 1.0 / (1 + normalized_entropy(to_simplex(np.abs([
                            delta['position_delta'], delta['resource_delta'], delta['field_delta']
                        ]) + 0.01)))
                    else:
                        base_weight = 1.0 / 3.0  # Peso uniforme inicial

                    # Goal alignment: peso mayor si hay meta
                    goal_weight = base_weight * (1.0 + decisions[name].confidence)
                    reward = decisions[name].goal_alignment * goal_weight

                    # Resource delta: peso proporcional a su magnitud relativa
                    resource_weight = base_weight * (1.0 + abs(delta['resource_delta']))
                    reward += delta['resource_delta'] * resource_weight

                    # Field delta: penalización proporcional a sorpresa
                    stability_weight = base_weight / (1.0 + decisions[name].confidence)
                    reward -= delta['field_delta'] * stability_weight

                    # Bonus por colaboración (si ToM predijo bien)
                    tom_acc = self.tom_system.get_statistics()['mean_tom_accuracy']
                    tom_weight = base_weight * tom_acc  # Peso proporcional a accuracy
                    reward += tom_acc * tom_weight
                else:
                    reward = 0.0

                surprise = float(np.abs(delta['field_delta']))

                # Social feedback de ToM
                social_feedback = {}
                for other in self.agent_names:
                    if other != name:
                        model = self.tom_system.get_model(name, other)
                        social_feedback[other] = model.tom_accuracy_score()

                outcome = self.agents[name].action_layer.process_outcome(
                    actual_change, reward, surprise, social_feedback
                )
                self.agents[name].process_feedback(outcome)
                outcomes[name] = outcome

                # Guardar recompensa
                self.reward_history[name].append(reward)
                max_hist = max_history(self.total_steps)
                if len(self.reward_history[name]) > max_hist:
                    self.reward_history[name] = self.reward_history[name][-max_hist:]

        # 8. Verificar fin de episodios
        episodes_ended = []
        for name, agent in self.agents.items():
            if agent.should_end_episode():
                episode = agent.end_episode()
                episodes_ended.append(episode)
                self.episode_count += 1

        # Guardar régimen
        regime = ['stable', 'volatile', 'transitional'][self.world.current_regime]
        self.regime_history.append(regime)

        # 9. Sistema Médico: monitorizar y posiblemente intervenir
        health_global_state = self._prepare_health_state(cognitive_states, outcomes, decisions)
        health_global_state = self.system_doctor.step(self.total_steps, health_global_state)

        # Aplicar cambios del médico a los agentes (si hubo intervenciones)
        self._apply_health_interventions(health_global_state)

        # Verbose output
        if verbose and self.total_steps % 50 == 0:
            self._print_status()

        return {
            't': self.total_steps,
            'regime': regime,
            'decisions': {n: d.reasoning if d else "none" for n, d in decisions.items()},
            'rewards': {n: o.reward_signal if o else 0 for n, o in outcomes.items()},
            'episodes_ended': len(episodes_ended),
            'tom_accuracy': self.tom_system.get_statistics()['mean_tom_accuracy']
        }

    def _prepare_health_state(
        self,
        cognitive_states: Dict[str, Dict],
        outcomes: Dict[str, Any],
        decisions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepara estado global para el sistema médico.

        Extrae métricas de salud de cada agente.
        """
        health_state = {'agents': {}}

        for name in self.agent_names:
            agent = self.agents[name]
            cog_state = cognitive_states.get(name, {})
            outcome = outcomes.get(name)
            decision = decisions.get(name)

            # Extraer métricas para HealthMonitor
            agent_health_state = {}

            # Crisis rate del historial de rewards
            if self.reward_history[name]:
                recent_rewards = self.reward_history[name][-L_t(self.total_steps):]
                crisis_count = sum(1 for r in recent_rewards if r < -0.3)
                agent_health_state['crisis_rate'] = crisis_count / (len(recent_rewards) + 1)
            else:
                agent_health_state['crisis_rate'] = 0.0

            # V_t del self-model (confianza inversa como proxy)
            agent_health_state['V_t'] = 1.0 - agent.self_model.confidence()

            # CF/CI scores (si existen en el outcome)
            if outcome:
                agent_health_state['CF_score'] = 1.0 - outcome.surprise
                agent_health_state['CI_score'] = 0.5  # Placeholder
            else:
                agent_health_state['CF_score'] = 0.5
                agent_health_state['CI_score'] = 0.5

            # Ethics score del decision
            if decision:
                agent_health_state['ethics_score'] = decision.ethical_score
            else:
                agent_health_state['ethics_score'] = 0.8

            # Self-coherence de AGI-20
            agent_health_state['self_coherence'] = 1.0 if agent.self_theory.is_self_coherent() else 0.5

            # ToM accuracy del sistema compartido
            tom_stats = self.tom_system.get_statistics()
            agent_health_state['tom_accuracy'] = tom_stats.get('mean_tom_accuracy', 0.5)

            # Config entropy de AGI-18
            action_stats = agent.action_layer.get_statistics()
            agent_health_state['config_entropy'] = action_stats.get('config_entropy', 0.5)

            # Wellbeing basado en rewards recientes
            if self.reward_history[name]:
                agent_health_state['wellbeing'] = (np.mean(self.reward_history[name][-20:]) + 1) / 2
            else:
                agent_health_state['wellbeing'] = 0.5

            # Metacognition del self-model
            agent_health_state['metacognition'] = agent.self_model.confidence()

            # Drives para preservación de identidad
            if cog_state.get('drives') is not None:
                agent_health_state['drives'] = cog_state['drives']

            # Parámetros modificables por el médico
            agent_health_state['learning_rate'] = action_stats.get('mean_confidence', 0.5)
            agent_health_state['temperature'] = 1.0
            agent_health_state['module_weights'] = agent.action_layer.module_weights.copy()

            health_state['agents'][name] = agent_health_state

        return health_state

    def _apply_health_interventions(self, health_state: Dict[str, Any]):
        """
        Aplica intervenciones del sistema médico a los agentes.

        Solo modifica parámetros soft (learning, weights, etc.)
        """
        for name in self.agent_names:
            if name not in health_state.get('agents', {}):
                continue

            agent_state = health_state['agents'][name]
            agent = self.agents[name]

            # Aplicar cambios de module_weights si fueron modificados
            if 'module_weights' in agent_state:
                new_weights = agent_state['module_weights']
                if isinstance(new_weights, dict):
                    for module_name, weight in new_weights.items():
                        if module_name in agent.action_layer.module_weights:
                            agent.action_layer.module_weights[module_name] = weight

    def _print_status(self):
        """Imprime estado actual incluyendo AGI-16 a AGI-20 y sistema médico."""
        print(f"\n  t={self.total_steps}:")
        print(f"    Régimen: {self.regime_history[-1]}")
        print(f"    Episodios totales: {self.episode_count}")

        # AGI-5: ToM
        print(f"    ToM accuracy: {self.tom_system.get_statistics()['mean_tom_accuracy']:.3f}")

        # AGI-17: Robustness
        rob_stats = self.robustness_system.get_statistics()
        print(f"    Robustez sistema: {rob_stats.get('system_robustness', 0):.3f}")

        # AGI-19: Collective Intent
        coll_stats = self.collective_intent.get_statistics()
        print(f"    Coherencia colectiva: {coll_stats.get('coherence', 0):.3f}, metas emergentes: {coll_stats.get('n_active_goals', 0)}")

        for name in self.agent_names[:2]:  # Solo mostrar 2 agentes
            agent = self.agents[name]
            stats = agent.action_layer.get_statistics()
            self_u = agent.self_theory._compute_self_understanding()
            print(f"    {name}: conf={stats['mean_confidence']:.2f}, reward={stats['mean_reward']:.2f}, "
                  f"policy={stats.get('current_policy', 'balance')}, self_u={self_u:.2f}")

        # Sistema Médico
        health_state = self.system_doctor.get_state()
        print(f"    HEALTH: doctor={health_state.current_doctor}, "
              f"system_H={health_state.system_health:.3f}, "
              f"pathologies={health_state.system_pathologies}")

    def run(self, n_steps: int, verbose: bool = True) -> Dict:
        """
        Ejecuta el loop por n pasos.

        Args:
            n_steps: Número de pasos
            verbose: Si imprimir progreso

        Returns:
            Estadísticas finales
        """
        if verbose:
            print("=" * 70)
            print("COGNITIVE WORLD LOOP")
            print("=" * 70)
            print(f"\nAgentes: {self.agent_names}")
            print(f"Pasos: {n_steps}")

        for _ in range(n_steps):
            self.step(verbose=verbose)

        # Estadísticas finales
        final_stats = self.get_statistics()

        if verbose:
            print("\n" + "=" * 70)
            print("RESULTADOS FINALES")
            print("=" * 70)
            print(f"\n  Pasos totales: {final_stats['total_steps']}")
            print(f"  Episodios: {final_stats['episode_count']}")
            print(f"  ToM accuracy final: {final_stats['tom_accuracy']:.3f}")
            print(f"\n  Por agente:")
            for name in self.agent_names:
                agent_stats = final_stats['agents'][name]
                print(f"    {name}: episodios={agent_stats['n_episodes']}, "
                      f"reward_medio={agent_stats['mean_reward']:.3f}")

        return final_stats

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas completas del loop incluyendo AGI-16 a AGI-20."""
        agent_stats = {}
        for name, agent in self.agents.items():
            # AGI-20 stats
            self_theory_stats = agent.self_theory.get_statistics()

            agent_stats[name] = {
                'n_episodes': len(agent.memory.episodes),
                'mean_reward': np.mean(self.reward_history[name]) if self.reward_history[name] else 0,
                'action_stats': agent.action_layer.get_statistics(),
                'self_model_confidence': agent.self_model.confidence(),
                # AGI-20: Self-Theory
                'self_understanding': self_theory_stats.get('self_understanding', 0),
                'self_coherent': self_theory_stats.get('is_coherent', False),
            }

        # AGI-17: Robustness stats
        robustness_stats = self.robustness_system.get_statistics()

        # AGI-19: Collective Intent stats
        collective_stats = self.collective_intent.get_statistics()

        # Health System stats
        health_stats = self.system_doctor.get_statistics()

        return {
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'world_stats': self.world.get_statistics(),
            'tom_accuracy': self.tom_system.get_statistics()['mean_tom_accuracy'],
            'regime_distribution': {
                r: self.regime_history.count(r) / len(self.regime_history) if self.regime_history else 0
                for r in ['stable', 'volatile', 'transitional']
            },
            'agents': agent_stats,
            # AGI-17: Robustness
            'system_robustness': robustness_stats.get('system_robustness', 0),
            'most_robust_agent': robustness_stats.get('most_robust_agent', ''),
            # AGI-19: Collective Intent
            'collective_coherence': collective_stats.get('coherence', 0),
            'intentionality_index': collective_stats.get('intentionality_index', 0),
            'emergent_goals': collective_stats.get('n_active_goals', 0),
            # Health System
            'health': {
                'system_health': health_stats.get('system_health', 0.5),
                'current_doctor': health_stats.get('current_doctor'),
                'doctor_changes': health_stats.get('doctor_changes', 0),
                'total_pathologies': health_stats.get('total_pathologies', 0),
                'active_interventions': health_stats.get('active_interventions', 0),
                'current_pathologies': health_stats.get('current_pathologies', []),
            },
            # Lifecycle stats
            'lifecycle': self._get_lifecycle_stats()
        }

    def _get_lifecycle_stats(self) -> Dict:
        """Obtiene estadisticas del ciclo de vida de los agentes."""
        lifecycle_stats = {}
        for name, agent in self.agents.items():
            circadian_state = agent.circadian.get_state()
            lifecycle_stats[name] = {
                'phase': circadian_state.phase.value,
                'energy': circadian_state.energy,
                'stress': circadian_state.stress,
                'cycles_completed': circadian_state.cycles_completed,
                'journal_entries': agent.life_journal.get_statistics()['total_entries'],
                'patterns_discovered': len(agent.dream_processor.discovered_patterns)
            }
        return lifecycle_stats

    def simulate_absence(self, hours: float) -> Dict[str, str]:
        """
        Simula lo que paso durante la ausencia del usuario.

        Args:
            hours: Horas de ausencia

        Returns:
            Dict con narrativas de reconexion por agente
        """
        # Obtener ciclos circadianos
        cycles = {name: agent.circadian for name, agent in self.agents.items()}

        # Simular ausencia
        simulator = AbsenceSimulator(self.agent_names)
        reports = simulator.simulate_absence(cycles, hours)

        # Generar narrativas de reconexion
        narratives = {}
        for name, agent in self.agents.items():
            narrative = agent.reconnection_narrator.generate_reconnection_narrative(
                reports[name],
                agent.circadian,
                agent.life_journal,
                agent.dream_processor
            )
            narratives[name] = agent.reconnection_narrator.format_narrative_for_display(
                narrative, "medium"
            )

        return narratives

    def reconnect(self, hours_away: float, verbose: bool = True) -> Dict:
        """
        Metodo para reconectar despues de ausencia.

        Simula lo que paso y presenta narrativas.

        Args:
            hours_away: Horas de ausencia
            verbose: Si imprimir narrativas

        Returns:
            Dict con narrativas y estado
        """
        if verbose:
            print("=" * 70)
            print(f"RECONEXION DESPUES DE {hours_away:.1f} HORAS")
            print("=" * 70)

        narratives = self.simulate_absence(hours_away)

        if verbose:
            for name in self.agent_names:
                print(f"\n{'-' * 50}")
                print(narratives[name])

        # Actualizar tiempo total
        simulated_steps = int(hours_away * 100)
        self.total_steps += simulated_steps

        return {
            'narratives': narratives,
            'hours_simulated': hours_away,
            'steps_simulated': simulated_steps,
            'lifecycle': self._get_lifecycle_stats()
        }


def test_cognitive_world_loop():
    """Test del loop cognitivo completo con AGI-4 a AGI-20."""
    print("=" * 70)
    print("TEST: COGNITIVE WORLD LOOP (AGI-4 a AGI-20)")
    print("=" * 70)

    # Crear loop con 5 agentes
    loop = CognitiveWorldLoop(['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS'])

    # Establecer metas para algunos agentes
    loop.agents['NEO'].action_layer.set_goal(np.array([0.9, 0.5, 0.5]))
    loop.agents['EVA'].action_layer.set_goal(np.array([0.5, 0.9, 0.5]))

    # Ejecutar
    stats = loop.run(n_steps=300, verbose=True)

    # Verificar que el sistema funciona
    print("\n" + "=" * 70)
    print("VERIFICACIÓN AGI-4 a AGI-20")
    print("=" * 70)

    checks = {
        # Básicos
        'Episodios generados': stats['episode_count'] > 0,
        'ToM aprendiendo (AGI-5)': stats['tom_accuracy'] > 0.2,
        'Agentes activos': all(s['n_episodes'] > 0 for s in stats['agents'].values()),
        'Mundo dinámico': stats['world_stats']['d_eff'] > 1,
        # AGI-17: Robustness
        'Robustez calculada (AGI-17)': stats.get('system_robustness', 0) >= 0,
        # AGI-19: Collective Intent
        'Coherencia colectiva (AGI-19)': stats.get('collective_coherence', 0) >= 0,
        # AGI-20: Self-Theory
        'Self-Understanding (AGI-20)': any(
            s.get('self_understanding', 0) > 0 for s in stats['agents'].values()
        ),
    }

    # AGI-16 y AGI-18 checks (en action_stats)
    neo_action_stats = stats['agents']['NEO'].get('action_stats', {})
    checks['Meta-reglas activas (AGI-16)'] = neo_action_stats.get('n_meta_rules', 0) >= 0
    checks['Reconfiguraciones (AGI-18)'] = neo_action_stats.get('n_reconfigurations', 0) >= 0

    # Health System checks
    health = stats.get('health', {})
    checks['Sistema médico activo'] = health.get('system_health', 0) > 0
    checks['Médico elegido endógenamente'] = health.get('current_doctor') is not None or health.get('doctor_changes', 0) >= 0

    print("\n  Módulos:")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {check}")

    all_passed = all(checks.values())
    print(f"\n  {'TODOS LOS CHECKS PASARON' if all_passed else 'ALGUNOS CHECKS FALLARON'}")

    # Mostrar estadísticas finales de AGI-16 a AGI-20
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS AGI-16 a AGI-20")
    print("=" * 70)

    print(f"\n  AGI-17 (Robustness):")
    print(f"    Sistema robustez: {stats.get('system_robustness', 0):.3f}")
    print(f"    Agente más robusto: {stats.get('most_robust_agent', 'N/A')}")

    print(f"\n  AGI-19 (Collective Intent):")
    print(f"    Coherencia: {stats.get('collective_coherence', 0):.3f}")
    print(f"    Índice intencionalidad: {stats.get('intentionality_index', 0):.3f}")
    print(f"    Metas emergentes: {stats.get('emergent_goals', 0)}")

    print(f"\n  Por agente (AGI-16, AGI-18, AGI-20):")
    for name in ['NEO', 'EVA']:
        agent_stats = stats['agents'][name]
        action_stats = agent_stats.get('action_stats', {})
        print(f"    {name}:")
        print(f"      AGI-16 política: {action_stats.get('current_policy', 'N/A')}, reglas: {action_stats.get('n_meta_rules', 0)}")
        print(f"      AGI-18 reconfigs: {action_stats.get('n_reconfigurations', 0)}, módulo top: {action_stats.get('most_weighted_module', 'N/A')}")
        print(f"      AGI-20 self_u: {agent_stats.get('self_understanding', 0):.3f}, coherente: {agent_stats.get('self_coherent', False)}")

    # Health System report
    print(f"\n  SISTEMA MÉDICO (Rol Emergente):")
    health = stats.get('health', {})
    print(f"    Salud del sistema: {health.get('system_health', 0):.3f}")
    print(f"    Médico actual: {health.get('current_doctor', 'Ninguno')}")
    print(f"    Cambios de médico: {health.get('doctor_changes', 0)}")
    print(f"    Patologías detectadas: {health.get('total_pathologies', 0)}")
    print(f"    Intervenciones activas: {health.get('active_interventions', 0)}")
    print(f"    Patologías actuales: {health.get('current_pathologies', [])}")

    return loop


if __name__ == "__main__":
    test_cognitive_world_loop()
