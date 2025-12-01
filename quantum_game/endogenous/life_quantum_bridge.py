#!/usr/bin/env python3
"""
LIFE QUANTUM BRIDGE - Puente entre DualLife y Quantum Coalition Game
====================================================================

Este módulo conecta:
- NEO, EVA, ALEX (agentes de life.py)
- Quantum Coalition Game endógeno

Los agentes REALES juegan el juego cuántico:
- Sus drives reales son los estados cuánticos
- Sus interacciones reales son las rondas del juego
- Sus métricas reales son los payoffs

Esto permite que los agentes "ganen experiencia" jugando.
"""

import numpy as np
import sys
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousAgent, AutonomousDualLife
from quantum_game.endogenous.coalition_game_qg1 import CoalitionGameQG1, AgentGameState
from quantum_game.endogenous.payoff_endogenous import PayoffCalculator, CooperationMetric
from quantum_game.endogenous.state_encoding import QuantumStateEncoding


@dataclass
class LifeQuantumBridge:
    """
    Puente que conecta DualLife con el juego cuántico.

    Los agentes reales (NEO, EVA) juegan el juego y
    sus experiencias modifican su dinámica interna.
    """
    life: AutonomousDualLife = None
    game: CoalitionGameQG1 = None

    # Mapeo de agentes life → game
    agent_map: Dict[str, str] = field(default_factory=dict)

    # Calculadoras
    payoff_calc: PayoffCalculator = None
    coop_metric: CooperationMetric = None

    # Historias
    payoff_history: Dict[str, List[float]] = field(default_factory=dict)
    experience_history: List[Dict] = field(default_factory=list)

    # Configuración
    learning_rate: float = 0.1  # Cuánto afecta el juego a los drives

    def __post_init__(self):
        if self.life is None:
            self.life = AutonomousDualLife(dim=6)

        # Crear juego con los agentes
        self.agent_map = {'NEO': 'NEO', 'EVA': 'EVA'}
        agent_names = list(self.agent_map.values())

        self.game = CoalitionGameQG1(agent_names=agent_names, dim=6)
        self.payoff_calc = PayoffCalculator(agent_names=agent_names)
        self.coop_metric = CooperationMetric(agent_names=agent_names)

        for name in agent_names:
            self.payoff_history[name] = []

    def sync_life_to_game(self):
        """
        Sincroniza el estado de AutonomousDualLife al juego cuántico.

        Toma los drives reales y los inyecta en los agentes del juego.
        """
        for life_name, game_name in self.agent_map.items():
            life_agent = getattr(self.life, life_name.lower())
            game_agent = self.game.agents[game_name]

            # Sincronizar drives desde z (estado actual del agente)
            life_drives = life_agent.z.copy()

            # Normalizar para el juego
            life_drives = np.clip(life_drives, 1e-16, None)
            life_drives = life_drives / life_drives.sum()

            game_agent.drives = life_drives
            game_agent.drive_history.append(life_drives.copy())

            # Sincronizar historia z
            if hasattr(life_agent, 'z_history') and life_agent.z_history:
                game_agent.z_history = [z.copy() for z in life_agent.z_history[-100:]]

            # Actualizar estado cuántico
            game_agent.quantum_state = QuantumStateEncoding.from_drives(
                life_drives,
                game_agent.drive_history
            )

            # Sincronizar métricas
            game_agent.coherence = getattr(life_agent, 'identity_strength', 0.5)
            game_agent.in_crisis = life_agent.in_crisis

            # Calcular φ desde z_history
            if len(game_agent.z_history) > 10:
                game_agent.phi = game_agent.compute_phi_endogenous()

            # Calcular attachment
            other_name = 'EVA' if life_name == 'NEO' else 'NEO'
            other_agent = getattr(self.life, other_name.lower())
            other_drives = other_agent.z.copy()
            # Attachment como correlación de drives
            if len(game_agent.drive_history) > 20:
                corr = np.corrcoef(life_drives, other_drives)[0, 1]
                if not np.isnan(corr):
                    game_agent.attachments[other_name] = abs(corr)

    def apply_game_to_life(self, round_data, payoffs: Dict[str, float]):
        """
        Aplica los resultados del juego de vuelta a AutonomousDualLife.

        Los payoffs modifican ligeramente el estado z de los agentes.
        """
        for life_name, game_name in self.agent_map.items():
            life_agent = getattr(self.life, life_name.lower())

            payoff = payoffs.get(game_name, 0)
            operator = round_data.operators_selected.get(game_name, '')

            # Modificar z basándose en operador seleccionado
            # Esto simula "aprendizaje por experiencia"
            z = life_agent.z.copy()

            # El operador seleccionado indica la "preferencia" del agente
            # Reforzamos el drive correspondiente si el payoff fue positivo
            operator_drive_map = {
                'homeostasis': [1, 3],     # neg_surprise, stability
                'exploration': [0, 2],      # entropy, novelty
                'integration': [4],         # integration
                'attachment': [5],          # otherness
                'momentum': [],             # No refuerza directamente
                'crisis': []                # No refuerza directamente
            }

            drives_to_boost = operator_drive_map.get(operator, [])

            # Calcular boost basado en payoff normalizado
            if self.payoff_history[game_name]:
                p_mean = np.mean(self.payoff_history[game_name][-50:])
                p_std = np.std(self.payoff_history[game_name][-50:]) + 1e-16
                normalized_payoff = (payoff - p_mean) / p_std
            else:
                normalized_payoff = 0

            boost = self.learning_rate * np.tanh(normalized_payoff)

            for idx in drives_to_boost:
                z[idx] += boost * max(z[idx], 0.01)

            # Renormalizar
            z = np.clip(z, 1e-16, None)
            z = z / z.sum()

            life_agent.z = z

    def play_game_round(self) -> Dict:
        """
        Juega una ronda del juego cuántico con los agentes reales.
        """
        # 1. Sincronizar estado de life al juego
        self.sync_life_to_game()

        # 2. Jugar ronda
        round_data = self.game.play_round()

        # 3. Calcular payoffs
        payoffs = self.payoff_calc.compute_all_payoffs(round_data)
        self.coop_metric.update(payoffs)

        # 4. Registrar
        for name, payoff in payoffs.items():
            self.payoff_history[name].append(payoff)

        # 5. Aplicar experiencia de vuelta a life
        self.apply_game_to_life(round_data, payoffs)

        # 6. Registrar experiencia
        experience = {
            'round': self.game.current_round,
            'operators': round_data.operators_selected.copy(),
            'payoffs': payoffs.copy(),
            'entanglement': self.game.get_entanglement_matrix()[0, 1],
            'system_payoff': self.coop_metric.system_payoff_history[-1],
            'fairness': self.coop_metric.get_fairness_index()
        }
        self.experience_history.append(experience)

        return experience

    def run_life_with_game(self, total_steps: int, game_frequency: int = 10) -> Dict:
        """
        Ejecuta DualLife integrando rondas del juego cuántico.

        Args:
            total_steps: Pasos totales de simulación
            game_frequency: Cada cuántos pasos jugar una ronda
        """
        results = {
            'life_metrics': [],
            'game_metrics': [],
            'combined_metrics': []
        }

        print(f"Ejecutando Life + Quantum Game ({total_steps} pasos, juego cada {game_frequency})...")

        for t in range(total_steps):
            # Paso de vida normal con estímulo aleatorio
            stimulus = np.random.randn(6) * 0.1
            self.life.step(stimulus)

            # Jugar ronda del juego periódicamente
            if t > 0 and t % game_frequency == 0:
                experience = self.play_game_round()
                results['game_metrics'].append(experience)

                if t % 100 == 0:
                    print(f"  t={t}: Entanglement={experience['entanglement']:.3f}, "
                          f"Payoffs={experience['payoffs']}")

            # Registrar métricas combinadas
            if t % 50 == 0:
                combined = {
                    't': t,
                    'neo_crisis': self.life.neo.in_crisis,
                    'eva_crisis': self.life.eva.in_crisis,
                    'neo_drives': self.life.neo.meta_drive.weights.copy(),
                    'eva_drives': self.life.eva.meta_drive.weights.copy(),
                }

                if self.experience_history:
                    combined['last_entanglement'] = self.experience_history[-1]['entanglement']
                    combined['cumulative_payoff_neo'] = sum(self.payoff_history['NEO'])
                    combined['cumulative_payoff_eva'] = sum(self.payoff_history['EVA'])

                results['combined_metrics'].append(combined)

        # Resumen final
        results['summary'] = {
            'total_steps': total_steps,
            'game_rounds': len(self.experience_history),
            'final_entanglement': self.game.get_entanglement_matrix()[0, 1] if self.experience_history else 0,
            'mean_system_payoff': np.mean([e['system_payoff'] for e in self.experience_history]) if self.experience_history else 0,
            'mean_fairness': np.mean([e['fairness'] for e in self.experience_history]) if self.experience_history else 0,
            'neo_crisis_rate': sum(1 for m in results['combined_metrics'] if m['neo_crisis']) / len(results['combined_metrics']) if results['combined_metrics'] else 0,
            'eva_crisis_rate': sum(1 for m in results['combined_metrics'] if m['eva_crisis']) / len(results['combined_metrics']) if results['combined_metrics'] else 0,
        }

        return results


class TriadQuantumBridge(LifeQuantumBridge):
    """
    Extensión para incluir a ALEX en el juego.
    ALEX es un agente virtual que solo existe en el juego.
    """

    def __post_init__(self):
        if self.life is None:
            self.life = AutonomousDualLife(dim=6)

        # Crear ALEX como tercer agente virtual
        self.alex_drives = np.ones(6) / 6
        self.alex_z_history = []
        self.alex_in_crisis = False

        # Mapeo: NEO y EVA son reales, ALEX es virtual
        self.agent_map = {'NEO': 'NEO', 'EVA': 'EVA'}  # Solo reales en el mapeo
        agent_names = ['NEO', 'EVA', 'ALEX']  # Pero ALEX está en el juego

        self.game = CoalitionGameQG1(agent_names=agent_names, dim=6)
        self.payoff_calc = PayoffCalculator(agent_names=agent_names)
        self.coop_metric = CooperationMetric(agent_names=agent_names)

        for name in agent_names:
            self.payoff_history[name] = []

    def sync_life_to_game(self):
        """Sincroniza incluyendo a ALEX virtual."""
        # Sincronizar NEO y EVA como antes
        for life_name in ['NEO', 'EVA']:
            game_name = life_name
            life_agent = getattr(self.life, life_name.lower())
            game_agent = self.game.agents[game_name]

            life_drives = life_agent.z.copy()
            life_drives = np.clip(life_drives, 1e-16, None)
            life_drives = life_drives / life_drives.sum()

            game_agent.drives = life_drives
            game_agent.drive_history.append(life_drives.copy())

            if hasattr(life_agent, 'z_history') and life_agent.z_history:
                game_agent.z_history = [z.copy() for z in life_agent.z_history[-100:]]

            game_agent.quantum_state = QuantumStateEncoding.from_drives(
                life_drives,
                game_agent.drive_history
            )

            game_agent.coherence = getattr(life_agent, 'identity_strength', 0.5)
            game_agent.in_crisis = life_agent.in_crisis

        # Sincronizar ALEX virtual
        alex_agent = self.game.agents['ALEX']
        alex_agent.drives = self.alex_drives.copy()
        alex_agent.drive_history.append(self.alex_drives.copy())

        # ALEX evoluciona con su propia dinámica
        noise = np.random.normal(0, 0.05, 6)
        self.alex_drives = self.alex_drives + noise
        self.alex_drives = np.clip(self.alex_drives, 0.05, None)
        self.alex_drives = self.alex_drives / self.alex_drives.sum()

        alex_z = self.alex_drives - np.mean(self.alex_drives)
        self.alex_z_history.append(alex_z)
        if len(self.alex_z_history) > 100:
            self.alex_z_history = self.alex_z_history[-100:]

        alex_agent.z_history = self.alex_z_history.copy()
        alex_agent.quantum_state = QuantumStateEncoding.from_drives(
            self.alex_drives,
            alex_agent.drive_history
        )

        alex_agent.coherence = 0.4
        self.alex_in_crisis = np.random.random() < 0.1
        alex_agent.in_crisis = self.alex_in_crisis

    def apply_game_to_life(self, round_data, payoffs: Dict[str, float]):
        """Aplica feedback a NEO, EVA y ALEX."""
        # Aplicar a NEO y EVA (agentes reales)
        for life_name in ['NEO', 'EVA']:
            game_name = life_name
            life_agent = getattr(self.life, life_name.lower())

            payoff = payoffs.get(game_name, 0)
            operator = round_data.operators_selected.get(game_name, '')

            z = life_agent.z.copy()

            operator_drive_map = {
                'homeostasis': [1, 3],
                'exploration': [0, 2],
                'integration': [4],
                'attachment': [5],
                'momentum': [],
                'crisis': []
            }

            drives_to_boost = operator_drive_map.get(operator, [])

            if self.payoff_history[game_name]:
                p_mean = np.mean(self.payoff_history[game_name][-50:])
                p_std = np.std(self.payoff_history[game_name][-50:]) + 1e-16
                normalized_payoff = (payoff - p_mean) / p_std
            else:
                normalized_payoff = 0

            boost = self.learning_rate * np.tanh(normalized_payoff)

            for idx in drives_to_boost:
                z[idx] += boost * max(z[idx], 0.01)

            z = np.clip(z, 1e-16, None)
            z = z / z.sum()
            life_agent.z = z

        # Aplicar a ALEX virtual
        alex_payoff = payoffs.get('ALEX', 0)
        alex_operator = round_data.operators_selected.get('ALEX', '')
        alex_learning = self.learning_rate * 1.5

        if self.payoff_history['ALEX']:
            p_mean = np.mean(self.payoff_history['ALEX'][-50:])
            p_std = np.std(self.payoff_history['ALEX'][-50:]) + 1e-16
            normalized = (alex_payoff - p_mean) / p_std
        else:
            normalized = 0

        boost = alex_learning * np.tanh(normalized)

        for idx in operator_drive_map.get(alex_operator, []):
            self.alex_drives[idx] += boost * max(self.alex_drives[idx], 0.01)

        self.alex_drives = np.clip(self.alex_drives, 1e-16, None)
        self.alex_drives = self.alex_drives / self.alex_drives.sum()


def run_experience_experiment():
    """Experimento: NEO-EVA ganan experiencia jugando."""
    print("=" * 70)
    print("EXPERIMENTO: NEO-EVA GANAN EXPERIENCIA JUGANDO")
    print("=" * 70)

    bridge = LifeQuantumBridge()
    results = bridge.run_life_with_game(total_steps=1000, game_frequency=10)

    print("\n--- RESULTADOS ---")
    print(f"Pasos totales: {results['summary']['total_steps']}")
    print(f"Rondas jugadas: {results['summary']['game_rounds']}")
    print(f"Entanglement final: {results['summary']['final_entanglement']:.3f}")
    print(f"Payoff sistema medio: {results['summary']['mean_system_payoff']:.3f}")
    print(f"Fairness media: {results['summary']['mean_fairness']:.3f}")
    print(f"Tasa crisis NEO: {results['summary']['neo_crisis_rate']*100:.1f}%")
    print(f"Tasa crisis EVA: {results['summary']['eva_crisis_rate']*100:.1f}%")

    return results


def run_triad_experiment():
    """Experimento: NEO-EVA-ALEX juegan juntos."""
    print("\n" + "=" * 70)
    print("EXPERIMENTO: TRÍADA NEO-EVA-ALEX")
    print("=" * 70)

    bridge = TriadQuantumBridge()
    results = bridge.run_life_with_game(total_steps=1000, game_frequency=10)

    print("\n--- RESULTADOS TRÍADA ---")
    print(f"Rondas jugadas: {results['summary']['game_rounds']}")
    print(f"Entanglement final NEO-EVA: {bridge.game.get_entanglement_matrix()[0, 1]:.3f}")
    print(f"Entanglement final NEO-ALEX: {bridge.game.get_entanglement_matrix()[0, 2]:.3f}")
    print(f"Entanglement final EVA-ALEX: {bridge.game.get_entanglement_matrix()[1, 2]:.3f}")

    # Analizar coaliciones
    ent_matrix = bridge.game.get_entanglement_matrix()
    pairs = [('NEO-EVA', ent_matrix[0, 1]),
             ('NEO-ALEX', ent_matrix[0, 2]),
             ('EVA-ALEX', ent_matrix[1, 2])]
    strongest = max(pairs, key=lambda x: x[1])
    weakest = min(pairs, key=lambda x: x[1])

    print(f"\nVínculo más fuerte: {strongest[0]} ({strongest[1]:.3f})")
    print(f"Vínculo más débil: {weakest[0]} ({weakest[1]:.3f})")

    return results


if __name__ == "__main__":
    results_duo = run_experience_experiment()
    results_triad = run_triad_experiment()

    print("\n" + "=" * 70)
    print("✓ Experimentos completados")
    print("  NEO, EVA y ALEX han ganado experiencia jugando")
    print("=" * 70)
