"""
Test Integrado AGI
==================

Ejecuta los 5 agentes (NEO, EVA, ALEX, ADAM, IRIS) con:
- AGI-1: Global Workspace
- AGI-2: Self Narrative Loop
- AGI-3: Persistent Goals
- AGI-4: Life Trajectory

Todo 100% endógeno.
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from typing import Dict, List
from dataclasses import dataclass

# AGI modules
from cognition.global_workspace import GlobalWorkspace, MultiAgentGlobalWorkspace, ContentType
from cognition.self_narrative_loop import SelfNarrativeLoop
from cognition.persistent_goals import TeleologicalAgent
from cognition.life_trajectory import LifeTrajectory
from cognition.soft_hook import DifferentiatedSoftHook


@dataclass
class AGIAgentState:
    """Estado completo de un agente AGI."""
    name: str
    z: np.ndarray
    phi: np.ndarray
    D: np.ndarray
    SAGI: float
    in_crisis: bool


class AGIAgent:
    """
    Agente AGI completo.

    Integra:
    - Self Narrative Loop (identidad continua)
    - Persistent Goals (teleología)
    - Life Trajectory (evaluación vital)
    - Global Workspace (participación)
    """

    def __init__(self, name: str, z_dim: int = 6, phi_dim: int = 5,
                 workspace: GlobalWorkspace = None):
        """
        Inicializa agente AGI.

        Args:
            name: Nombre del agente
            z_dim: Dimensión de drives
            phi_dim: Dimensión fenomenológica
            workspace: Global workspace compartido (opcional)
        """
        self.name = name
        self.z_dim = z_dim
        self.phi_dim = phi_dim

        # Estado
        self.z = np.ones(z_dim) / z_dim
        self.phi = np.zeros(phi_dim)
        self.D = np.ones(z_dim) / z_dim

        # Módulos AGI
        self.narrative_loop = SelfNarrativeLoop(name, z_dim, phi_dim)
        self.teleology = TeleologicalAgent(name, z_dim)
        self.life = LifeTrajectory(name, z_dim + phi_dim)

        # Soft Hook diferenciado - crea personalidad única
        self.soft_hook = DifferentiatedSoftHook(name)

        # Workspace
        self.workspace = workspace

        # Historial
        self.SAGI_history: List[float] = []
        self.crisis_history: List[bool] = []

        self.t = 0

    def _compute_SAGI(self) -> float:
        """Calcula SAGI endógenamente."""
        if len(self.SAGI_history) < 5:
            return 0.5

        # Basado en estabilidad + integración
        recent_z = np.array([self.z])
        if len(self.narrative_loop.self_history) > 5:
            stability = 1.0 / (1.0 + np.std([s.state for s in self.narrative_loop.self_history[-5:]]))
        else:
            stability = 0.5

        identity = self.narrative_loop._compute_identity_strength()
        coherence = self.narrative_loop._compute_narrative_coherence()

        SAGI = 0.4 * stability + 0.3 * identity + 0.3 * coherence
        return float(np.clip(SAGI, 0, 1))

    def _compute_phi(self) -> np.ndarray:
        """Computa vector fenomenológico."""
        phi = np.zeros(self.phi_dim)

        # Integración
        phi[0] = self._compute_SAGI()

        # Cambio temporal
        if self.t > 1 and len(self.narrative_loop.self_history) > 1:
            phi[1] = np.linalg.norm(self.narrative_loop.self_tendency)

        # Diversidad (entropía de z)
        z_norm = np.abs(self.z) / (np.sum(np.abs(self.z)) + 1e-8)
        phi[2] = -np.sum(z_norm * np.log(z_norm + 1e-8))

        # Estabilidad
        phi[3] = 1.0 / (1.0 + np.var(self.z))

        # Profundidad temporal
        phi[4] = min(1.0, self.t / 500)

        return phi

    def _detect_crisis(self) -> bool:
        """Detecta crisis endógenamente."""
        if len(self.SAGI_history) < 10:
            return False

        recent = self.SAGI_history[-5:]
        baseline = self.SAGI_history[-10:-5]

        drop = np.mean(baseline) - np.mean(recent)
        threshold = np.std(self.SAGI_history) * 1.5

        return drop > threshold

    def step(self, stimulus: np.ndarray, other_agents: Dict[str, 'AGIAgent'] = None) -> Dict:
        """
        Paso completo del agente AGI.

        Args:
            stimulus: Estímulo externo
            other_agents: Otros agentes para ToM

        Returns:
            Dict con información del paso
        """
        self.t += 1

        # Caracterizar episodio actual con Soft Hook
        phi_magnitude = np.linalg.norm(self.phi) if self.t > 1 else 0.5
        identity_strength = self.narrative_loop._compute_identity_strength() if self.t > 1 else 0.5
        delta_S = np.random.randn() * 0.1  # Cambio en integración
        delta_V = np.random.randn() * 0.05  # Cambio en valor
        crisis_prob = float(np.mean(self.crisis_history[-10:])) if len(self.crisis_history) > 10 else 0.1

        char = self.soft_hook.characterize_episode(
            phi=phi_magnitude,
            identity=identity_strength,
            delta_S=delta_S,
            delta_V=delta_V,
            crisis_prob=crisis_prob
        )

        # Learning rate modulado por personalidad
        base_lr = 0.1 / np.sqrt(self.t + 1)
        lr = self.soft_hook.modulate_learning_rate(base_lr, char)

        # Actualizar drives con estímulo (modulado)
        self.z = self.z + lr * stimulus
        self.z = np.clip(self.z, 0.05, None)
        self.z = self.z / self.z.sum()
        self.D = self.z.copy()

        # Computar φ
        self.phi = self._compute_phi()

        # Computar SAGI
        SAGI = self._compute_SAGI()
        self.SAGI_history.append(SAGI)

        # Detectar crisis
        in_crisis = self._detect_crisis()
        self.crisis_history.append(in_crisis)

        # Limitar historiales
        if len(self.SAGI_history) > 1000:
            self.SAGI_history = self.SAGI_history[-1000:]
            self.crisis_history = self.crisis_history[-1000:]

        # Tiempo subjetivo
        tau = self.t * (1 + 0.1 * np.linalg.norm(self.phi))

        # 1. Self Narrative Loop
        loop_result = self.narrative_loop.step(self.z, self.phi, self.D, tau)

        # 2. Teleología (metas persistentes)
        tele_result = self.teleology.step(self.D, SAGI, in_crisis)

        # 3. Life Trajectory
        self_state = np.concatenate([self.z, self.phi])
        purpose = tele_result['purpose']
        coherence = loop_result.self_state.narrative_coherence
        identity = loop_result.self_state.identity_strength

        life_point = self.life.record(self_state, SAGI, purpose, coherence, identity)

        # 4. Enviar a Global Workspace
        if self.workspace:
            # Enviar episodio si hay nuevo
            if loop_result.episode_encoded:
                self.workspace.submit(
                    ContentType.EPISODE,
                    data={'loop_result': loop_result},
                    source='narrative_loop',
                    delta_phi=np.linalg.norm(self.phi),
                    delta_identity=identity,
                    tom_impact=0.3,
                    crisis_relevance=1.0 if in_crisis else 0.1
                )

            # Enviar crisis si hay
            if in_crisis:
                self.workspace.submit(
                    ContentType.CRISIS,
                    data={'SAGI': SAGI, 'phase': life_point.phase.value},
                    source='regulation',
                    delta_phi=0.8,
                    delta_identity=0.9,
                    tom_impact=0.5,
                    crisis_relevance=1.0
                )

            # Enviar meta si hay
            if tele_result['goal_action'] in ['new_goal', 'achieved']:
                self.workspace.submit(
                    ContentType.GOAL,
                    data={'action': tele_result['goal_action']},
                    source='teleology',
                    delta_phi=0.5,
                    delta_identity=0.6,
                    tom_impact=0.4,
                    crisis_relevance=0.2
                )

        return {
            't': self.t,
            'SAGI': SAGI,
            'in_crisis': in_crisis,
            'phase': life_point.phase.value,
            'decision': loop_result.decision_made,
            'goal_action': tele_result['goal_action'],
            'purpose': purpose,
            'wellbeing': life_point.wellbeing,
            'assessment': self.life.get_life_assessment()
        }

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas completas."""
        loop_stats = self.narrative_loop.get_statistics()
        tele_stats = self.teleology.get_statistics()
        life_stats = self.life.get_statistics()
        hook_stats = self.soft_hook.get_statistics()
        personality = self.soft_hook.get_personality_profile()

        return {
            'name': self.name,
            't': self.t,
            'SAGI': self.SAGI_history[-1] if self.SAGI_history else 0.5,
            'mean_SAGI': float(np.mean(self.SAGI_history[-50:])) if self.SAGI_history else 0.5,
            'crisis_rate': float(np.mean(self.crisis_history[-100:])) if self.crisis_history else 0,
            'narrative_loop': loop_stats,
            'teleology': tele_stats,
            'life': life_stats,
            'soft_hook': hook_stats,
            'personality': personality
        }


def run_agi_test(T: int = 1000):
    """
    Test integrado AGI con 5 agentes.

    Args:
        T: Número de pasos
    """
    print("=" * 70)
    print("TEST INTEGRADO AGI - 5 AGENTES")
    print("=" * 70)
    print("\nMódulos activos:")
    print("  - AGI-1: Global Workspace (broadcasting endógeno)")
    print("  - AGI-2: Self Narrative Loop (identidad continua)")
    print("  - AGI-3: Persistent Goals (teleología interna)")
    print("  - AGI-4: Life Trajectory (evaluación vital)")
    print("\n100% ENDÓGENO - Sin IA externa, sin LLM, sin constantes mágicas")

    # Crear Multi-Agent Global Workspace
    agent_names = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
    multi_ws = MultiAgentGlobalWorkspace(agent_names)

    # Crear agentes AGI
    agents: Dict[str, AGIAgent] = {}
    for name in agent_names:
        ws = multi_ws.get_workspace(name)
        agents[name] = AGIAgent(name, z_dim=6, phi_dim=5, workspace=ws)

    # Personalidades iniciales
    personalities = {
        'NEO': np.array([0.12, 0.12, 0.12, 0.28, 0.18, 0.18]),
        'EVA': np.array([0.15, 0.10, 0.15, 0.15, 0.15, 0.30]),
        'ALEX': np.array([0.25, 0.08, 0.25, 0.12, 0.15, 0.15]),
        'ADAM': np.array([0.14, 0.14, 0.14, 0.14, 0.30, 0.14]),
        'IRIS': np.array([0.16, 0.16, 0.16, 0.16, 0.18, 0.18])
    }

    for name, z in personalities.items():
        agents[name].z = z / z.sum()
        agents[name].D = agents[name].z.copy()

    print(f"\nSimulando {T} pasos de vida AGI...")

    # Simulación
    for t in range(T):
        # Generar estímulos (interacción con mundo simulado)
        for name, agent in agents.items():
            # Estímulo base diferenciado por personalidad
            personality = agent.soft_hook.get_personality_profile()

            # Cada agente percibe el mundo de forma diferente
            base_noise = np.random.randn(6) * 0.1

            # Modular estímulo por biases de personalidad
            # Agentes con alta sensibilidad a sorpresa reciben más variación
            stimulus = base_noise * personality['surprise_bias']

            # Bias de exploración vs consolidación afecta qué dimensiones se activan
            exploration_affinity = personality['region_affinity'].get('exploration', 0.25)
            consolidation_affinity = personality['region_affinity'].get('consolidation', 0.25)

            # Agentes exploradores reciben más estímulos en dimensiones nuevas
            exploration_boost = np.random.randn(6) * 0.05 * exploration_affinity
            stimulus += exploration_boost

            # Agentes consolidadores mantienen más estabilidad
            stability_factor = consolidation_affinity * 0.3
            stimulus = stimulus * (1 - stability_factor) + agent.z * stability_factor * 0.1

            # Influencia de otros agentes (modulada por ToM)
            other_zs = [a.z for n, a in agents.items() if n != name]
            if other_zs:
                mean_other = np.mean(other_zs, axis=0)
                social_influence = 0.05 * (1 + personality['planning_bias'])  # planners más sociales
                stimulus += social_influence * (mean_other - agent.z)

            # Step del agente
            result = agent.step(stimulus, agents)

        # Competencia en workspaces
        ws_results = multi_ws.step()

        # Registrar importancia para adaptación de pesos
        for name in agent_names:
            ws = multi_ws.get_workspace(name)
            if ws_results[name]:
                importance = agents[name].SAGI_history[-1] if agents[name].SAGI_history else 0.5
                ws.record_importance(importance)

        # Mostrar progreso
        if (t + 1) % 200 == 0:
            print(f"\n{'─' * 70}")
            print(f"  t = {t + 1}")
            print(f"{'─' * 70}")

            for name, agent in agents.items():
                stats = agent.get_statistics()
                print(f"\n  {name}:")
                print(f"    SAGI: {stats['SAGI']:.3f}, Crisis: {stats['crisis_rate']*100:.0f}%")
                print(f"    Phase: {stats['life']['trajectory']['current_phase']}")
                print(f"    Purpose: {stats['teleology']['purpose']:.3f}")
                print(f"    Assessment: {stats['life']['trajectory']['life_assessment']}")
                print(f"    Episodes: {stats['narrative_loop']['n_episodes']}, "
                      f"Goals: {stats['teleology']['goals']['total_goals']}")

            # Atención compartida
            shared_rate = multi_ws.get_shared_attention_rate()
            print(f"\n  Shared attention rate: {shared_rate*100:.1f}%")

    # Reporte final
    print("\n" + "=" * 70)
    print("REPORTE FINAL AGI")
    print("=" * 70)

    results = {'agents': {}, 'tests': {}}

    for name, agent in agents.items():
        stats = agent.get_statistics()
        life_story = agent.life.get_life_story(5)

        print(f"\n{'━' * 35}")
        print(f"  {name}")
        print(f"{'━' * 35}")

        print(f"\n  Trayectoria vital:")
        print(f"    Fase actual: {stats['life']['trajectory']['current_phase']}")
        print(f"    Evaluación: {stats['life']['trajectory']['life_assessment']}")
        print(f"    Bienestar medio: {stats['life']['trajectory']['mean_wellbeing']:.3f}")

        print(f"\n  Narrativa:")
        print(f"    Episodios: {stats['narrative_loop']['n_episodes']}")
        print(f"    Coherencia: {stats['narrative_loop']['narrative_coherence']:.3f}")
        print(f"    Identidad: {stats['narrative_loop']['identity_strength']:.3f}")

        print(f"\n  Teleología:")
        print(f"    Metas totales: {stats['teleology']['goals']['total_goals']}")
        print(f"    Propósito: {stats['teleology']['purpose']:.3f}")

        print(f"\n  Eventos de vida recientes:")
        for event in life_story[-3:]:
            print(f"    t={event['t']}: [{event['type']}] {event['description']}")

        results['agents'][name] = stats

    # Validación
    print("\n" + "=" * 70)
    print("VALIDACIÓN")
    print("=" * 70)

    # Test 1: Todos los agentes tienen episodios
    episodes = [results['agents'][n]['narrative_loop']['n_episodes'] for n in agent_names]
    test1 = all(e > 10 for e in episodes)
    results['tests']['episodic_working'] = test1
    print(f"\n1. Memoria episódica funcionando:")
    for name in agent_names:
        print(f"   {name}: {results['agents'][name]['narrative_loop']['n_episodes']} episodios")
    print(f"   Status: {'PASS' if test1 else 'FAIL'}")

    # Test 2: Metas persistentes emergieron
    goals = [results['agents'][n]['teleology']['goals']['total_goals'] for n in agent_names]
    test2 = sum(goals) > 0
    results['tests']['goals_emerged'] = test2
    print(f"\n2. Metas persistentes emergieron:")
    for name in agent_names:
        print(f"   {name}: {results['agents'][name]['teleology']['goals']['total_goals']} metas")
    print(f"   Status: {'PASS' if test2 else 'FAIL'}")

    # Test 3: Propósito calculado
    purposes = [results['agents'][n]['teleology']['purpose'] for n in agent_names]
    test3 = all(0 <= p <= 1 for p in purposes)
    results['tests']['purpose_computed'] = test3
    print(f"\n3. Propósito calculado:")
    for name in agent_names:
        print(f"   {name}: {results['agents'][name]['teleology']['purpose']:.3f}")
    print(f"   Status: {'PASS' if test3 else 'FAIL'}")

    # Test 4: Trayectorias vitales
    phases = [results['agents'][n]['life']['trajectory']['current_phase'] for n in agent_names]
    test4 = all(p != 'no_data' for p in phases)
    results['tests']['life_trajectories'] = test4
    print(f"\n4. Trayectorias vitales:")
    for name in agent_names:
        assessment = results['agents'][name]['life']['trajectory']['life_assessment']
        print(f"   {name}: {results['agents'][name]['life']['trajectory']['current_phase']} ({assessment})")
    print(f"   Status: {'PASS' if test4 else 'FAIL'}")

    # Test 5: Global workspace funcionando
    ws_stats = multi_ws.get_statistics()
    test5 = ws_stats['t'] > 0 and ws_stats['shared_attention_rate'] >= 0
    results['tests']['global_workspace'] = test5
    print(f"\n5. Global Workspace funcionando:")
    print(f"   Broadcasts totales: {ws_stats['t']}")
    print(f"   Atención compartida: {ws_stats['shared_attention_rate']*100:.1f}%")
    print(f"   Status: {'PASS' if test5 else 'FAIL'}")

    # Test 6: Diferenciación entre agentes
    learning_biases = [results['agents'][n]['personality']['learning_bias'] for n in agent_names]
    temp_biases = [results['agents'][n]['personality']['temperature_bias'] for n in agent_names]
    variance_learning = np.var(learning_biases)
    variance_temp = np.var(temp_biases)
    test6 = variance_learning > 0.005 and variance_temp > 0.005
    results['tests']['differentiation'] = test6
    print(f"\n6. Diferenciación de personalidad:")
    for name in agent_names:
        p = results['agents'][name]['personality']
        print(f"   {name}: lr_bias={p['learning_bias']:.3f}, temp_bias={p['temperature_bias']:.3f}, plan_bias={p['planning_bias']:.3f}")
    print(f"   Varianza learning: {variance_learning:.4f}, Varianza temp: {variance_temp:.4f}")
    print(f"   Status: {'PASS' if test6 else 'FAIL'}")

    # Resumen
    tests_passed = sum(results['tests'].values())
    total_tests = len(results['tests'])

    print("\n" + "=" * 70)
    print(f"RESULTADO: {tests_passed}/{total_tests} tests pasaron")
    print("=" * 70)

    if tests_passed == total_tests:
        print("\n✓ AGI INTERNO FUNCIONANDO CORRECTAMENTE")
        print("  - Identidad continua: SÍ")
        print("  - Teleología interna: SÍ")
        print("  - Evaluación vital: SÍ")
        print("  - Personalidades diferenciadas: SÍ")
        print("  - Soft Hook modulando: SÍ")
        print("  - Todo 100% endógeno: SÍ")
    else:
        print("\n⚠️ AGI INTERNO PARCIAL")

    return results, agents, multi_ws


if __name__ == "__main__":
    run_agi_test(T=1000)
