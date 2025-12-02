"""
Simulación Larga de Omega Spaces
=================================

Ejecuta una simulación de 1000+ pasos observando métricas de:
- OmegaCompute: modos Ω_k activos por agente
- QField: coherencia C_Q(t) y energía de superposición E_Q(t)
- PhaseSpaceX: curvatura de trayectorias
- TensorMind: modos tensoriales fuertes

SOLO OBSERVA. No condiciona comportamiento.
"""

import numpy as np
import json
import csv
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Añadir path del proyecto
sys.path.insert(0, '/root/NEO_EVA')

from omega import OmegaCompute, QField, PhaseSpaceX, TensorMind
from lifecycle.circadian_system import AgentCircadianCycle, CircadianPhase


@dataclass
class AgentOmegaMetrics:
    """Métricas Omega de un agente en un instante."""
    t: int
    agent_id: str

    # OmegaCompute
    n_active_modes: int
    mode_activations: List[float]  # α_{i,k}(t)
    reconstruction_error: float

    # QField
    coherence_cq: float  # C_Q(t)
    superposition_energy_eq: float  # E_Q(t)
    collapse_degree: float

    # PhaseSpaceX
    speed: float
    acceleration: float
    curvature: float
    near_attractor: bool
    attractor_distance: float

    # Circadian (contexto)
    phase: str
    energy: float


@dataclass
class GlobalOmegaMetrics:
    """Métricas Omega globales del sistema."""
    t: int

    # OmegaCompute global
    total_modes: int
    total_variance_explained: float
    mean_reconstruction_error: float

    # QField global
    field_mean_coherence: float
    field_mean_energy: float
    field_entropy: float

    # PhaseSpaceX global
    n_attractors: int
    mean_speed: float
    mean_curvature: float

    # TensorMind global
    n_strong_modes: int  # Modos con varianza > umbral
    n_communities: int
    mean_interaction_strength: float


class OmegaLongSimulation:
    """
    Simulador largo de Omega Spaces.

    Solo observa métricas, no condiciona comportamiento.
    """

    def __init__(
        self,
        agents: List[str],
        state_dim: int = 10,
        prob_dim: int = 5,
        log_dir: str = '/root/NEO_EVA/logs/omega_simulation'
    ):
        self.agents = agents
        self.state_dim = state_dim
        self.prob_dim = prob_dim
        self.log_dir = log_dir

        # Crear directorio de logs
        os.makedirs(log_dir, exist_ok=True)

        # Inicializar Omega Spaces
        self.omega_compute = OmegaCompute()
        self.q_field = QField()
        self.phase_space = PhaseSpaceX()
        self.tensor_mind = TensorMind(max_order=3)

        # Sistemas circadianos por agente
        self.circadian_systems: Dict[str, AgentCircadianCycle] = {
            agent: AgentCircadianCycle(agent_id=agent) for agent in agents
        }

        # Estados internos simulados por agente
        self._agent_states: Dict[str, np.ndarray] = {}
        self._agent_identities: Dict[str, np.ndarray] = {}

        # Logs
        self.agent_metrics_log: List[AgentOmegaMetrics] = []
        self.global_metrics_log: List[GlobalOmegaMetrics] = []

        # Tiempo
        self.t = 0

        # Inicializar estados
        self._initialize_agents()

    def _initialize_agents(self):
        """Inicializa estados de agentes."""
        for agent in self.agents:
            # Estado inicial aleatorio
            self._agent_states[agent] = np.random.randn(self.state_dim)
            # Identidad inicial (ligeramente diferente del estado)
            self._agent_identities[agent] = (
                self._agent_states[agent] + 0.1 * np.random.randn(self.state_dim)
            )

    def _simulate_agent_dynamics(self, agent: str) -> Dict[str, Any]:
        """
        Simula dinámica interna de un agente.

        NO condiciona comportamiento, solo genera estados
        basados en dinámica interna endógena.
        """
        # Estado previo
        S_prev = self._agent_states[agent]
        I = self._agent_identities[agent]

        # Sistema circadiano
        circadian = self.circadian_systems[agent]
        phase = circadian.phase
        energy = circadian.energy

        # Dinámica endógena basada en fase circadiana
        # (NO es una regla, solo una tendencia natural simulada)
        if phase == CircadianPhase.WAKE:
            # En vigilia: más exploración
            noise_scale = 0.3
            drift_to_identity = 0.05
        elif phase == CircadianPhase.REST:
            # En descanso: menos movimiento
            noise_scale = 0.1
            drift_to_identity = 0.1
        elif phase == CircadianPhase.DREAM:
            # En sueño: movimiento caótico
            noise_scale = 0.5
            drift_to_identity = 0.02
        else:  # LIMINAL
            # En liminal: transición
            noise_scale = 0.2
            drift_to_identity = 0.08

        # Nuevo estado: tendencia hacia identidad + ruido
        noise = noise_scale * np.random.randn(self.state_dim)
        drift = drift_to_identity * (I - S_prev)
        S_new = S_prev + drift + noise

        # Actualizar estado
        self._agent_states[agent] = S_new

        # Identidad evoluciona lentamente hacia estado
        self._agent_identities[agent] = (
            I + 0.01 * (S_new - I) + 0.01 * np.random.randn(self.state_dim)
        )

        # Generar probabilidades para Q-Field (distribución interna)
        # Basadas en componentes del estado normalizado
        S_abs = np.abs(S_new[:self.prob_dim])
        probs = S_abs / (np.sum(S_abs) + np.finfo(float).eps)

        return {
            'state': S_new,
            'identity': self._agent_identities[agent],
            'probabilities': probs,
            'phase': phase,
            'energy': energy,
        }

    def step(self) -> Dict[str, Any]:
        """
        Ejecuta un paso de simulación.

        Solo observa, no condiciona.
        """
        self.t += 1

        agent_metrics = []

        # Procesar cada agente
        for agent in self.agents:
            # Simular dinámica
            dynamics = self._simulate_agent_dynamics(agent)
            state = dynamics['state']
            probs = dynamics['probabilities']

            # Actualizar sistema circadiano
            self.circadian_systems[agent].step()

            # === REGISTRAR EN OMEGA SPACES (solo observar) ===

            # OmegaCompute: registrar estado
            transition = self.omega_compute.register_state(agent, state)

            # QField: registrar probabilidades
            q_state = self.q_field.register_state(agent, probs)

            # PhaseSpaceX: registrar en espacio de fase
            phase_point = self.phase_space.register_state(agent, state)

            # TensorMind: registrar para interacciones
            self.tensor_mind.register_state(agent, state)

            # === RECOLECTAR MÉTRICAS (solo observar) ===

            # Métricas OmegaCompute
            activation = None
            if transition is not None:
                activation = self.omega_compute.project_transition(agent, transition)

            n_active_modes = 0
            mode_activations = []
            reconstruction_error = 0.0
            if activation is not None:
                mode_activations = activation.coefficients.tolist()
                n_active_modes = len([a for a in mode_activations if abs(a) > 0.1])
                reconstruction_error = activation.reconstruction_error

            # Métricas QField
            coherence_cq = q_state.coherence if q_state else 0.0
            superposition_energy = q_state.superposition_energy if q_state else 0.0

            collapse_info = self.q_field.measure_collapse(agent)
            collapse_degree = collapse_info['collapse_degree'] if collapse_info else 0.0

            # Métricas PhaseSpaceX
            speed = phase_point.speed if phase_point else 0.0
            acceleration = phase_point.acceleration if phase_point else 0.0

            trajectory = self.phase_space.get_trajectory(agent)
            curvature = trajectory.curvature if trajectory else 0.0

            attractor_info = self.phase_space.is_near_attractor(agent)
            near_attractor = attractor_info['within_radius'] if attractor_info else False
            attractor_distance = attractor_info['distance'] if attractor_info else float('inf')

            # Crear registro de métricas
            metrics = AgentOmegaMetrics(
                t=self.t,
                agent_id=agent,
                n_active_modes=n_active_modes,
                mode_activations=mode_activations[:5] if mode_activations else [],
                reconstruction_error=reconstruction_error,
                coherence_cq=coherence_cq,
                superposition_energy_eq=superposition_energy,
                collapse_degree=collapse_degree,
                speed=speed,
                acceleration=acceleration,
                curvature=curvature,
                near_attractor=near_attractor,
                attractor_distance=attractor_distance if attractor_distance != float('inf') else -1,
                phase=dynamics['phase'].name,
                energy=dynamics['energy'],
            )
            agent_metrics.append(metrics)
            self.agent_metrics_log.append(metrics)

        # Calcular interacciones tensoriales
        self.tensor_mind.compute_interactions()

        # Actualizar modos si hay suficientes datos
        if self.t % 10 == 0:
            self.omega_compute.update_modes()
            self.phase_space.detect_attractors()
            self.tensor_mind.extract_modes(order=2)

        # === MÉTRICAS GLOBALES ===

        # OmegaCompute global
        omega_stats = self.omega_compute.get_statistics()

        # QField global
        q_stats = self.q_field.get_statistics()

        # PhaseSpaceX global
        phase_portrait = self.phase_space.get_phase_portrait()

        # TensorMind global
        tensor_stats = self.tensor_mind.get_statistics()

        # Contar modos fuertes (varianza > media)
        n_strong_modes = 0
        if tensor_stats.get('modes'):
            variances = [m['variance_explained'] for m in tensor_stats['modes']]
            if variances:
                mean_var = np.mean(variances)
                n_strong_modes = len([v for v in variances if v > mean_var])

        global_metrics = GlobalOmegaMetrics(
            t=self.t,
            total_modes=omega_stats['n_modes'],
            total_variance_explained=omega_stats['total_variance_explained'],
            mean_reconstruction_error=omega_stats['mean_reconstruction_error'],
            field_mean_coherence=q_stats['mean_coherence'],
            field_mean_energy=q_stats['mean_energy'],
            field_entropy=q_stats.get('field_entropy', 0.0),
            n_attractors=phase_portrait.get('n_attractors', 0),
            mean_speed=phase_portrait.get('mean_speed', 0.0),
            mean_curvature=np.mean([m.curvature for m in agent_metrics]) if agent_metrics else 0.0,
            n_strong_modes=n_strong_modes,
            n_communities=len(tensor_stats.get('communities', {})),
            mean_interaction_strength=tensor_stats.get('order_2', {}).get('mean_strength', 0.0),
        )
        self.global_metrics_log.append(global_metrics)

        return {
            'agent_metrics': agent_metrics,
            'global_metrics': global_metrics,
        }

    def run(self, n_steps: int = 1000, log_interval: int = 100):
        """
        Ejecuta simulación completa.

        Args:
            n_steps: Número de pasos
            log_interval: Intervalo para mostrar progreso
        """
        print(f"=== Simulación Omega Spaces: {n_steps} pasos ===")
        print(f"Agentes: {self.agents}")
        print(f"Dimensión estado: {self.state_dim}")
        print(f"Dimensión probabilidades: {self.prob_dim}")
        print()

        start_time = datetime.now()

        for step in range(n_steps):
            result = self.step()

            if (step + 1) % log_interval == 0:
                gm = result['global_metrics']
                elapsed = (datetime.now() - start_time).total_seconds()

                print(f"Paso {step + 1}/{n_steps} ({elapsed:.1f}s)")
                print(f"  Ω-Compute: {gm.total_modes} modos, var={gm.total_variance_explained:.3f}")
                print(f"  Q-Field: C_Q={gm.field_mean_coherence:.3f}, E_Q={gm.field_mean_energy:.3f}")
                print(f"  PhaseSpace: {gm.n_attractors} atractores, curv={gm.mean_curvature:.3f}")
                print(f"  TensorMind: {gm.n_strong_modes} modos fuertes, {gm.n_communities} comunidades")
                print()

        total_time = (datetime.now() - start_time).total_seconds()
        print(f"=== Simulación completada en {total_time:.1f}s ===")

        return self.save_logs()

    def save_logs(self) -> Dict[str, str]:
        """Guarda logs a archivos."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # === JSON completo ===
        json_path = os.path.join(self.log_dir, f"omega_simulation_{timestamp}.json")

        log_data = {
            'metadata': {
                'timestamp': timestamp,
                'n_steps': self.t,
                'agents': self.agents,
                'state_dim': self.state_dim,
                'prob_dim': self.prob_dim,
            },
            'agent_metrics': [asdict(m) for m in self.agent_metrics_log],
            'global_metrics': [asdict(m) for m in self.global_metrics_log],
        }

        with open(json_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

        # === CSV de métricas por agente ===
        agent_csv_path = os.path.join(self.log_dir, f"omega_agent_metrics_{timestamp}.csv")

        with open(agent_csv_path, 'w', newline='') as f:
            if self.agent_metrics_log:
                writer = csv.DictWriter(f, fieldnames=asdict(self.agent_metrics_log[0]).keys())
                writer.writeheader()
                for m in self.agent_metrics_log:
                    row = asdict(m)
                    # Convertir lista a string
                    row['mode_activations'] = str(row['mode_activations'])
                    writer.writerow(row)

        # === CSV de métricas globales ===
        global_csv_path = os.path.join(self.log_dir, f"omega_global_metrics_{timestamp}.csv")

        with open(global_csv_path, 'w', newline='') as f:
            if self.global_metrics_log:
                writer = csv.DictWriter(f, fieldnames=asdict(self.global_metrics_log[0]).keys())
                writer.writeheader()
                for m in self.global_metrics_log:
                    writer.writerow(asdict(m))

        # === Resumen por agente ===
        summary_path = os.path.join(self.log_dir, f"omega_summary_{timestamp}.txt")

        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("RESUMEN SIMULACIÓN OMEGA SPACES\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Pasos totales: {self.t}\n")
            f.write(f"Agentes: {', '.join(self.agents)}\n\n")

            # Resumen por agente
            for agent in self.agents:
                agent_data = [m for m in self.agent_metrics_log if m.agent_id == agent]

                f.write(f"\n--- {agent} ---\n")

                # Modos Ω activos
                n_modes = [m.n_active_modes for m in agent_data]
                f.write(f"Modos Ω activos: media={np.mean(n_modes):.2f}, max={max(n_modes)}\n")

                # Coherencia Q
                cq = [m.coherence_cq for m in agent_data]
                f.write(f"C_Q(t): media={np.mean(cq):.3f}, std={np.std(cq):.3f}\n")

                # Energía superposición
                eq = [m.superposition_energy_eq for m in agent_data]
                f.write(f"E_Q(t): media={np.mean(eq):.3f}, std={np.std(eq):.3f}\n")

                # Curvatura
                curv = [m.curvature for m in agent_data]
                f.write(f"Curvatura: media={np.mean(curv):.3f}, max={max(curv):.3f}\n")

                # Cerca de atractor
                near = [m.near_attractor for m in agent_data]
                f.write(f"Cerca de atractor: {sum(near)}/{len(near)} pasos ({100*sum(near)/len(near):.1f}%)\n")

            # Resumen global
            f.write("\n" + "=" * 60 + "\n")
            f.write("MÉTRICAS GLOBALES\n")
            f.write("=" * 60 + "\n\n")

            gm = self.global_metrics_log

            f.write(f"Ω-Compute:\n")
            f.write(f"  Modos totales (final): {gm[-1].total_modes}\n")
            f.write(f"  Varianza explicada: {gm[-1].total_variance_explained:.3f}\n")

            f.write(f"\nQ-Field:\n")
            f.write(f"  Coherencia media: {np.mean([m.field_mean_coherence for m in gm]):.3f}\n")
            f.write(f"  Energía media: {np.mean([m.field_mean_energy for m in gm]):.3f}\n")

            f.write(f"\nPhaseSpace-X:\n")
            f.write(f"  Atractores (final): {gm[-1].n_attractors}\n")
            f.write(f"  Velocidad media: {np.mean([m.mean_speed for m in gm]):.3f}\n")

            f.write(f"\nTensorMind:\n")
            f.write(f"  Modos fuertes (final): {gm[-1].n_strong_modes}\n")
            f.write(f"  Comunidades: {gm[-1].n_communities}\n")
            f.write(f"  Fuerza interacción media: {np.mean([m.mean_interaction_strength for m in gm]):.3f}\n")

        print(f"\nLogs guardados en:")
        print(f"  JSON: {json_path}")
        print(f"  CSV agentes: {agent_csv_path}")
        print(f"  CSV global: {global_csv_path}")
        print(f"  Resumen: {summary_path}")

        return {
            'json': json_path,
            'agent_csv': agent_csv_path,
            'global_csv': global_csv_path,
            'summary': summary_path,
        }


def main():
    """Ejecuta simulación larga."""
    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']

    sim = OmegaLongSimulation(
        agents=agents,
        state_dim=10,
        prob_dim=5,
    )

    # Ejecutar 1000 pasos
    paths = sim.run(n_steps=1000, log_interval=100)

    return paths


if __name__ == "__main__":
    main()
