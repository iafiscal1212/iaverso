#!/usr/bin/env python3
"""
Test de Ausencia Prolongada (12h conceptuales) - Omega Spaces
=============================================================

Simula un "día largo" interno (~12h conceptuales) con:
1. Fase de calentamiento (warmup)
2. Fase de ausencia libre (free_run) - sin intervención

Solo observa. NO condiciona comportamiento.
NO impone reglas. NO usa números mágicos.

Loggea métricas de:
- OmegaCompute: modos Ω_k activos
- QField: coherencia C_Q(t), energía E_Q(t)
- PhaseSpaceX: curvatura de trayectorias
- TensorMind: modos tensoriales fuertes
- ComplexField: lambda_decoherence, collapse_pressure, psi_norm, phase_entropy

Ejecutar con:
    cd /root/NEO_EVA
    python -m tests.test_omega_last12h_002
    # o
    python tests/test_omega_last12h_002.py
"""

import numpy as np
import json
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field

# Añadir path del proyecto
sys.path.insert(0, '/root/NEO_EVA')

# Omega Spaces
from omega import OmegaCompute, QField, PhaseSpaceX, TensorMind

# ComplexField (si existe)
try:
    from cognition.complex_field import ComplexField, ComplexState
    COMPLEX_FIELD_AVAILABLE = True
except ImportError:
    COMPLEX_FIELD_AVAILABLE = False
    ComplexField = None
    ComplexState = None

# Coherencia existencial
try:
    from consciousness.coherence import CoherenciaExistencial, EstadoCoherencia
    from consciousness.identity import IdentidadComputacional
    COHERENCE_AVAILABLE = True
except ImportError:
    COHERENCE_AVAILABLE = False
    CoherenciaExistencial = None
    EstadoCoherencia = None
    IdentidadComputacional = None

# Circadian (opcional)
try:
    from lifecycle.circadian_system import AgentCircadianCycle, CircadianPhase
    CIRCADIAN_AVAILABLE = True
except ImportError:
    CIRCADIAN_AVAILABLE = False
    AgentCircadianCycle = None
    CircadianPhase = None


# =====================================================================
# Parámetros de simulación
# =====================================================================

WARMUP_STEPS = 500      # Fase de estabilización previa
FREE_RUN_STEPS = 2500   # Fase de "ausencia" prolongada

AGENTS = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
STATE_DIM = 10          # Dimensión del estado
PROB_DIM = 5            # Dimensión de probabilidades para Q-Field


# =====================================================================
# Dataclasses para métricas
# =====================================================================

@dataclass
class AgentStepMetrics:
    """Métricas de un agente en un paso."""
    t: int
    agent_id: str
    phase: str  # "warmup" o "free_run"

    # Estado
    S: List[float]

    # Coherencia Existencial
    CE: float
    internal_error: float  # Var[S - I]
    H_narr: float          # Entropía narrativa

    # OmegaCompute
    omega_modes_active: int
    reconstruction_error: float

    # QField
    qfield_coherence: float   # C_Q(t)
    qfield_energy: float      # E_Q(t)

    # PhaseSpaceX
    phase_curvature: float
    phase_speed: float
    near_attractor: bool

    # ComplexField (opcional)
    lambda_decoherence: float = 0.0
    collapse_pressure: float = 0.0
    psi_norm: float = 0.0
    phase_entropy: float = 0.0


@dataclass
class GlobalStepMetrics:
    """Métricas globales en un paso."""
    t: int
    phase: str

    # OmegaCompute global
    total_modes: int
    total_variance_explained: float

    # QField global
    field_mean_coherence: float
    field_mean_energy: float

    # PhaseSpaceX global
    n_attractors: int
    mean_speed: float

    # TensorMind global
    tensor_modes_strong: int
    n_communities: int
    mean_interaction_strength: float


# =====================================================================
# Clase principal de simulación
# =====================================================================

class OmegaLast12hSimulation:
    """
    Simulador de ausencia prolongada con Omega Spaces.

    Solo observa, no condiciona comportamiento.
    100% endógeno.
    """

    def __init__(
        self,
        agents: List[str],
        state_dim: int = STATE_DIM,
        prob_dim: int = PROB_DIM,
    ):
        self.agents = agents
        self.state_dim = state_dim
        self.prob_dim = prob_dim
        self.t = 0
        self.current_phase = "warmup"

        # Epsilon de máquina
        self.eps = np.finfo(float).eps

        # =====================================================
        # Inicializar Omega Spaces
        # =====================================================
        self.omega_compute = OmegaCompute()
        self.q_field = QField()
        self.phase_space = PhaseSpaceX()
        self.tensor_mind = TensorMind(max_order=3)

        # =====================================================
        # Inicializar ComplexField (si disponible)
        # =====================================================
        self.complex_field: Optional[ComplexField] = None
        self.complex_states: Dict[str, Any] = {}

        if COMPLEX_FIELD_AVAILABLE:
            self.complex_field = ComplexField(dim=state_dim)
            for agent in agents:
                self.complex_states[agent] = ComplexState()

        # =====================================================
        # Inicializar sistema de coherencia (si disponible)
        # =====================================================
        self.identidades: Dict[str, Any] = {}
        self.coherencias: Dict[str, Any] = {}

        if COHERENCE_AVAILABLE:
            for agent in agents:
                identidad = IdentidadComputacional(dimension=state_dim)
                self.identidades[agent] = identidad
                self.coherencias[agent] = CoherenciaExistencial(identidad)

        # =====================================================
        # Inicializar estados internos simulados
        # =====================================================
        self._agent_states: Dict[str, np.ndarray] = {}
        self._agent_identities: Dict[str, np.ndarray] = {}
        self._agent_prev_states: Dict[str, np.ndarray] = {}

        self._initialize_agents()

        # =====================================================
        # Logs
        # =====================================================
        self.agent_metrics_log: List[AgentStepMetrics] = []
        self.global_metrics_log: List[GlobalStepMetrics] = []

    def _initialize_agents(self):
        """Inicializa estados de agentes."""
        for agent in self.agents:
            # Estado inicial (distribución uniforme normalizada + ruido)
            state = np.random.randn(self.state_dim)
            state = state / (np.linalg.norm(state) + self.eps)
            self._agent_states[agent] = state
            self._agent_prev_states[agent] = state.copy()

            # Identidad inicial (ligeramente diferente del estado)
            identity = state + 0.1 * np.random.randn(self.state_dim)
            identity = identity / (np.linalg.norm(identity) + self.eps)
            self._agent_identities[agent] = identity

    def _simulate_agent_dynamics(self, agent: str) -> Dict[str, Any]:
        """
        Simula dinámica interna de un agente.

        NO condiciona comportamiento, solo genera estados
        basados en dinámica interna endógena.
        """
        S_prev = self._agent_states[agent]
        I = self._agent_identities[agent]

        # Escalas de ruido y drift endógenas
        # Basadas en fase pero SIN condicionar decisiones
        if self.current_phase == "warmup":
            # Fase de calentamiento: más exploración inicial
            noise_scale = 0.2
            drift_to_identity = 0.08
        else:
            # Fase de ausencia: dinámica libre, menos drift externo
            noise_scale = 0.15
            drift_to_identity = 0.05

        # Nuevo estado: tendencia hacia identidad + ruido
        noise = noise_scale * np.random.randn(self.state_dim)
        drift = drift_to_identity * (I - S_prev)
        S_new = S_prev + drift + noise

        # Normalizar suavemente (mantener escala similar)
        S_new = S_new / (np.linalg.norm(S_new) / np.linalg.norm(S_prev) + self.eps)

        # Guardar estado previo
        self._agent_prev_states[agent] = S_prev.copy()

        # Actualizar estado
        self._agent_states[agent] = S_new

        # Identidad evoluciona lentamente hacia estado
        I_new = I + 0.01 * (S_new - I) + 0.005 * np.random.randn(self.state_dim)
        I_new = I_new / (np.linalg.norm(I_new) + self.eps)
        self._agent_identities[agent] = I_new

        # Generar probabilidades para Q-Field
        S_abs = np.abs(S_new[:self.prob_dim])
        probs = S_abs / (np.sum(S_abs) + self.eps)

        return {
            'state': S_new,
            'prev_state': S_prev,
            'identity': I_new,
            'probabilities': probs,
        }

    def _compute_phase_entropy(self, psi: np.ndarray) -> float:
        """
        Calcula entropía de fases del vector complejo ψ.

        Discretiza fases en bins y calcula entropía de Shannon.
        Número de bins = sqrt(dim) (endógeno).
        """
        if psi is None:
            return 0.0

        phases = np.angle(psi)  # En [-π, π]

        # Normalizar a [0, 1]
        phases_norm = (phases + np.pi) / (2 * np.pi)

        # Bins endógenos: sqrt(dim)
        n_bins = max(2, int(np.sqrt(len(psi))))

        hist, _ = np.histogram(phases_norm, bins=n_bins, range=(0.0, 1.0))

        total = np.sum(hist)
        if total == 0:
            return 0.0

        p = hist / total

        # Entropía de Shannon
        entropy = 0.0
        for pi in p:
            if pi > self.eps:
                entropy -= pi * np.log(pi)

        return float(entropy)

    def step(self) -> Dict[str, Any]:
        """
        Ejecuta un paso de simulación.

        Solo observa, no condiciona.
        """
        self.t += 1

        agent_metrics_list = []

        # =====================================================
        # Procesar cada agente
        # =====================================================
        for agent in self.agents:
            # Simular dinámica
            dynamics = self._simulate_agent_dynamics(agent)
            state = dynamics['state']
            prev_state = dynamics['prev_state']
            identity = dynamics['identity']
            probs = dynamics['probabilities']

            # =================================================
            # Calcular CE y métricas de coherencia
            # =================================================
            CE = 0.5  # Default
            internal_error = 0.0
            H_narr = 0.0

            if COHERENCE_AVAILABLE:
                # Observar estado en sistema de coherencia
                self.coherencias[agent].observar_estado(state)

                # Calcular coherencia
                estado_coh = self.coherencias[agent].calcular(state, identity)
                CE = estado_coh.CE
                internal_error = estado_coh.varianza_desviacion
                H_narr = estado_coh.entropia_narrativa
            else:
                # Cálculo simplificado si no hay módulo de coherencia
                desviacion = state - identity
                internal_error = float(np.var(desviacion))
                H_narr = 0.0
                CE = 1.0 / (internal_error + H_narr + self.eps)
                CE = float(np.clip(CE / (1 + CE), 0, 1))  # Normalizar a [0, 1]

            # =================================================
            # Registrar en Omega Spaces (solo observar)
            # =================================================

            # OmegaCompute
            transition = self.omega_compute.register_state(agent, state)

            # QField
            q_state = self.q_field.register_state(agent, probs)

            # PhaseSpaceX
            phase_point = self.phase_space.register_state(agent, state)

            # TensorMind
            self.tensor_mind.register_state(agent, state)

            # =================================================
            # Recolectar métricas de Omega Spaces
            # =================================================

            # OmegaCompute métricas
            n_active_modes = 0
            reconstruction_error = 0.0

            if transition is not None:
                activation = self.omega_compute.project_transition(agent, transition)
                if activation is not None:
                    coeffs = activation.coefficients
                    # Modos activos: |α_k| > umbral endógeno (media de |α|)
                    if len(coeffs) > 0:
                        threshold = np.mean(np.abs(coeffs))
                        n_active_modes = int(np.sum(np.abs(coeffs) > threshold))
                    reconstruction_error = activation.reconstruction_error

            # QField métricas
            qfield_coherence = q_state.coherence if q_state else 0.0
            qfield_energy = q_state.superposition_energy if q_state else 0.0

            # PhaseSpaceX métricas
            phase_speed = phase_point.speed if phase_point else 0.0

            trajectory = self.phase_space.get_trajectory(agent)
            phase_curvature = trajectory.curvature if trajectory else 0.0

            attractor_info = self.phase_space.is_near_attractor(agent)
            near_attractor = attractor_info['within_radius'] if attractor_info else False

            # =================================================
            # ComplexField métricas (si disponible)
            # =================================================
            lambda_decoherence = 0.0
            collapse_pressure = 0.0
            psi_norm = 0.0
            phase_entropy = 0.0

            if COMPLEX_FIELD_AVAILABLE and self.complex_field is not None:
                cs = self.complex_states[agent]

                # Paso de ComplexField (solo observación)
                cf_metrics = self.complex_field.step(
                    cs,
                    state,
                    CE,
                    internal_error,
                    H_narr,
                )

                lambda_decoherence = cf_metrics['lambda_decoherence']
                collapse_pressure = cf_metrics['collapse_pressure']

                # Métricas de ψ
                if cs.psi is not None:
                    psi_norm = float(np.linalg.norm(cs.psi))
                    phase_entropy = self._compute_phase_entropy(cs.psi)

            # =================================================
            # Crear registro de métricas del agente
            # =================================================
            metrics = AgentStepMetrics(
                t=self.t,
                agent_id=agent,
                phase=self.current_phase,
                S=state.tolist(),
                CE=float(CE),
                internal_error=float(internal_error),
                H_narr=float(H_narr),
                omega_modes_active=n_active_modes,
                reconstruction_error=float(reconstruction_error),
                qfield_coherence=float(qfield_coherence),
                qfield_energy=float(qfield_energy),
                phase_curvature=float(phase_curvature),
                phase_speed=float(phase_speed),
                near_attractor=near_attractor,
                lambda_decoherence=float(lambda_decoherence),
                collapse_pressure=float(collapse_pressure),
                psi_norm=float(psi_norm),
                phase_entropy=float(phase_entropy),
            )
            agent_metrics_list.append(metrics)
            self.agent_metrics_log.append(metrics)

        # =====================================================
        # Calcular interacciones tensoriales
        # =====================================================
        self.tensor_mind.compute_interactions()

        # =====================================================
        # Actualizar modos periódicamente
        # =====================================================
        # Intervalo endógeno: sqrt(t)
        update_interval = max(5, int(np.sqrt(self.t)))
        if self.t % update_interval == 0:
            self.omega_compute.update_modes()
            self.phase_space.detect_attractors()
            self.tensor_mind.extract_modes(order=2)

        # =====================================================
        # Métricas globales
        # =====================================================
        omega_stats = self.omega_compute.get_statistics()
        q_stats = self.q_field.get_statistics()
        phase_portrait = self.phase_space.get_phase_portrait()
        tensor_stats = self.tensor_mind.get_statistics()

        # Contar modos TensorMind fuertes
        n_strong_modes = 0
        if tensor_stats.get('modes'):
            variances = [m['variance_explained'] for m in tensor_stats['modes']]
            if variances:
                mean_var = np.mean(variances)
                n_strong_modes = len([v for v in variances if v > mean_var])

        global_metrics = GlobalStepMetrics(
            t=self.t,
            phase=self.current_phase,
            total_modes=omega_stats['n_modes'],
            total_variance_explained=omega_stats['total_variance_explained'],
            field_mean_coherence=q_stats['mean_coherence'],
            field_mean_energy=q_stats['mean_energy'],
            n_attractors=phase_portrait.get('n_attractors', 0),
            mean_speed=phase_portrait.get('mean_speed', 0.0),
            tensor_modes_strong=n_strong_modes,
            n_communities=len(tensor_stats.get('communities', {})),
            mean_interaction_strength=tensor_stats.get('order_2', {}).get('mean_strength', 0.0),
        )
        self.global_metrics_log.append(global_metrics)

        return {
            'agent_metrics': agent_metrics_list,
            'global_metrics': global_metrics,
        }

    def run(
        self,
        warmup_steps: int = WARMUP_STEPS,
        free_run_steps: int = FREE_RUN_STEPS,
        log_interval: int = 100,
    ) -> Dict[str, str]:
        """
        Ejecuta simulación completa con fases warmup y free_run.
        """
        total_steps = warmup_steps + free_run_steps

        print("=" * 60)
        print("OMEGA LAST12H SIMULATION - Ausencia Prolongada")
        print("=" * 60)
        print(f"Agentes: {', '.join(self.agents)}")
        print(f"WARMUP_STEPS: {warmup_steps}")
        print(f"FREE_RUN_STEPS: {free_run_steps}")
        print(f"Total: {total_steps} pasos")
        print(f"ComplexField: {'ACTIVO' if COMPLEX_FIELD_AVAILABLE else 'NO DISPONIBLE'}")
        print(f"Coherence: {'ACTIVO' if COHERENCE_AVAILABLE else 'SIMPLIFICADO'}")
        print("=" * 60)
        print()

        start_time = datetime.now()

        # =====================================================
        # Fase WARMUP
        # =====================================================
        print("--- FASE WARMUP ---")
        self.current_phase = "warmup"

        for step in range(warmup_steps):
            result = self.step()

            if (step + 1) % log_interval == 0:
                gm = result['global_metrics']
                elapsed = (datetime.now() - start_time).total_seconds()

                print(f"[WARMUP] Paso {step + 1}/{warmup_steps} ({elapsed:.1f}s)")
                print(f"  Ω-Compute: {gm.total_modes} modos, var={gm.total_variance_explained:.3f}")
                print(f"  Q-Field: C_Q={gm.field_mean_coherence:.3f}, E_Q={gm.field_mean_energy:.3f}")
                print(f"  PhaseSpace: {gm.n_attractors} atractores")
                print(f"  TensorMind: {gm.tensor_modes_strong} modos fuertes")

        print()

        # =====================================================
        # Fase FREE_RUN
        # =====================================================
        print("--- FASE FREE_RUN (ausencia) ---")
        self.current_phase = "free_run"

        for step in range(free_run_steps):
            result = self.step()

            if (step + 1) % log_interval == 0:
                gm = result['global_metrics']
                elapsed = (datetime.now() - start_time).total_seconds()

                print(f"[FREE_RUN] Paso {step + 1}/{free_run_steps} ({elapsed:.1f}s)")
                print(f"  Ω-Compute: {gm.total_modes} modos, var={gm.total_variance_explained:.3f}")
                print(f"  Q-Field: C_Q={gm.field_mean_coherence:.3f}, E_Q={gm.field_mean_energy:.3f}")
                print(f"  PhaseSpace: {gm.n_attractors} atractores")
                print(f"  TensorMind: {gm.tensor_modes_strong} modos fuertes")

        print()

        total_time = (datetime.now() - start_time).total_seconds()
        print("=" * 60)
        print(f"Simulación completada en {total_time:.1f}s")
        print("=" * 60)

        # Guardar logs y generar resumen
        return self.save_logs()

    def save_logs(self) -> Dict[str, str]:
        """Guarda logs a archivos."""
        # Crear directorios
        root_logs = Path('/root/NEO_EVA/logs/omega_last12h')
        root_figs = Path('/root/NEO_EVA/figuras/omega_last12h')
        root_logs.mkdir(parents=True, exist_ok=True)
        root_figs.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_prefix = f"omega_last12h_{timestamp}"

        # =====================================================
        # JSON completo
        # =====================================================
        json_path = root_logs / f"{run_prefix}.json"

        log_data = {
            'metadata': {
                'timestamp': timestamp,
                'warmup_steps': WARMUP_STEPS,
                'free_run_steps': FREE_RUN_STEPS,
                'total_steps': self.t,
                'agents': self.agents,
                'state_dim': self.state_dim,
                'prob_dim': self.prob_dim,
                'complex_field_available': COMPLEX_FIELD_AVAILABLE,
                'coherence_available': COHERENCE_AVAILABLE,
            },
            'agent_metrics': [asdict(m) for m in self.agent_metrics_log],
            'global_metrics': [asdict(m) for m in self.global_metrics_log],
        }

        with open(json_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

        # =====================================================
        # CSV por agente
        # =====================================================
        agents_csv_path = root_logs / f"{run_prefix}_agents.csv"

        with open(agents_csv_path, 'w', newline='') as f:
            if self.agent_metrics_log:
                fieldnames = [
                    't', 'agent_id', 'phase', 'CE', 'internal_error', 'H_narr',
                    'omega_modes_active', 'reconstruction_error',
                    'qfield_coherence', 'qfield_energy',
                    'phase_curvature', 'phase_speed', 'near_attractor',
                    'lambda_decoherence', 'collapse_pressure', 'psi_norm', 'phase_entropy',
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for m in self.agent_metrics_log:
                    row = {
                        't': m.t,
                        'agent_id': m.agent_id,
                        'phase': m.phase,
                        'CE': m.CE,
                        'internal_error': m.internal_error,
                        'H_narr': m.H_narr,
                        'omega_modes_active': m.omega_modes_active,
                        'reconstruction_error': m.reconstruction_error,
                        'qfield_coherence': m.qfield_coherence,
                        'qfield_energy': m.qfield_energy,
                        'phase_curvature': m.phase_curvature,
                        'phase_speed': m.phase_speed,
                        'near_attractor': m.near_attractor,
                        'lambda_decoherence': m.lambda_decoherence,
                        'collapse_pressure': m.collapse_pressure,
                        'psi_norm': m.psi_norm,
                        'phase_entropy': m.phase_entropy,
                    }
                    writer.writerow(row)

        # =====================================================
        # CSV global
        # =====================================================
        global_csv_path = root_logs / f"{run_prefix}_global.csv"

        with open(global_csv_path, 'w', newline='') as f:
            if self.global_metrics_log:
                writer = csv.DictWriter(f, fieldnames=asdict(self.global_metrics_log[0]).keys())
                writer.writeheader()
                for m in self.global_metrics_log:
                    writer.writerow(asdict(m))

        # =====================================================
        # Resumen TXT
        # =====================================================
        summary_path = root_logs / f"{run_prefix}_summary.txt"
        self._write_summary(summary_path)

        # =====================================================
        # Gráficas
        # =====================================================
        self._generate_plots(root_figs, run_prefix)

        print(f"\nLogs guardados en:")
        print(f"  JSON: {json_path}")
        print(f"  CSV agentes: {agents_csv_path}")
        print(f"  CSV global: {global_csv_path}")
        print(f"  Resumen: {summary_path}")
        print(f"  Figuras: {root_figs}/")

        return {
            'json': str(json_path),
            'agents_csv': str(agents_csv_path),
            'global_csv': str(global_csv_path),
            'summary': str(summary_path),
            'figures_dir': str(root_figs),
        }

    def _write_summary(self, summary_path: Path):
        """Escribe resumen de la simulación."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(summary_path, 'w') as f:
            f.write("OMEGA LAST12H SIMULATION - " + timestamp + "\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Agentes: {', '.join(self.agents)}\n")
            f.write(f"Pasos: WARMUP={WARMUP_STEPS}, FREE_RUN={FREE_RUN_STEPS}\n")
            f.write(f"ComplexField: {'ACTIVO' if COMPLEX_FIELD_AVAILABLE else 'NO DISPONIBLE'}\n\n")

            # =====================================================
            # Por agente y fase
            # =====================================================
            f.write("[Por agente y fase]\n")
            f.write("-" * 40 + "\n\n")

            for agent in self.agents:
                f.write(f"- {agent}:\n")

                for phase in ["warmup", "free_run"]:
                    agent_phase_data = [
                        m for m in self.agent_metrics_log
                        if m.agent_id == agent and m.phase == phase
                    ]

                    if not agent_phase_data:
                        continue

                    f.write(f"  - {phase}:\n")

                    # CE
                    ce_vals = [m.CE for m in agent_phase_data]
                    f.write(f"      CE_mean={np.mean(ce_vals):.4f}, CE_std={np.std(ce_vals):.4f}\n")

                    # internal_error
                    ie_vals = [m.internal_error for m in agent_phase_data]
                    f.write(f"      internal_error_mean={np.mean(ie_vals):.4f}, std={np.std(ie_vals):.4f}\n")

                    # H_narr
                    hn_vals = [m.H_narr for m in agent_phase_data]
                    f.write(f"      H_narr_mean={np.mean(hn_vals):.4f}, std={np.std(hn_vals):.4f}\n")

                    # omega_modes_active
                    om_vals = [m.omega_modes_active for m in agent_phase_data]
                    f.write(f"      omega_modes_active_mean={np.mean(om_vals):.2f}\n")

                    # qfield_coherence
                    qc_vals = [m.qfield_coherence for m in agent_phase_data]
                    f.write(f"      qfield_coherence_mean={np.mean(qc_vals):.4f}, std={np.std(qc_vals):.4f}\n")

                    # qfield_energy
                    qe_vals = [m.qfield_energy for m in agent_phase_data]
                    f.write(f"      qfield_energy_mean={np.mean(qe_vals):.4f}, std={np.std(qe_vals):.4f}\n")

                    # phase_curvature
                    pc_vals = [m.phase_curvature for m in agent_phase_data]
                    f.write(f"      phase_curvature_mean={np.mean(pc_vals):.4f}, std={np.std(pc_vals):.4f}\n")

                    # ComplexField métricas
                    if COMPLEX_FIELD_AVAILABLE:
                        ld_vals = [m.lambda_decoherence for m in agent_phase_data]
                        cp_vals = [m.collapse_pressure for m in agent_phase_data]
                        pn_vals = [m.psi_norm for m in agent_phase_data]
                        pe_vals = [m.phase_entropy for m in agent_phase_data]

                        f.write(f"      lambda_decoherence_mean={np.mean(ld_vals):.4f}\n")
                        f.write(f"      collapse_pressure_mean={np.mean(cp_vals):.4f}\n")
                        f.write(f"      psi_norm_mean={np.mean(pn_vals):.4f}\n")
                        f.write(f"      phase_entropy_mean={np.mean(pe_vals):.4f}\n")

                    f.write("\n")

            # =====================================================
            # Global Omega
            # =====================================================
            f.write("\n[Global Omega]\n")
            f.write("-" * 40 + "\n")

            # TensorMind modes
            tm_vals = [m.tensor_modes_strong for m in self.global_metrics_log]
            f.write(f"- TensorMind modes (mean +/- std): {np.mean(tm_vals):.2f} +/- {np.std(tm_vals):.2f}\n")

            # Communities
            nc_vals = [m.n_communities for m in self.global_metrics_log]
            f.write(f"- Communities (mean +/- std): {np.mean(nc_vals):.2f} +/- {np.std(nc_vals):.2f}\n")

            # =====================================================
            # Observaciones automáticas (solo números)
            # =====================================================
            f.write("\n[Observaciones automaticas simples]\n")
            f.write("-" * 40 + "\n")

            for agent in self.agents:
                warmup_data = [m for m in self.agent_metrics_log if m.agent_id == agent and m.phase == "warmup"]
                freerun_data = [m for m in self.agent_metrics_log if m.agent_id == agent and m.phase == "free_run"]

                if warmup_data and freerun_data:
                    ce_warmup = np.mean([m.CE for m in warmup_data])
                    ce_freerun = np.mean([m.CE for m in freerun_data])
                    delta_ce = ce_freerun - ce_warmup
                    direction = "sube" if delta_ce > 0 else "baja"
                    f.write(f"- CE de {agent}: {direction} en FREE_RUN vs WARMUP (delta={delta_ce:+.4f})\n")

                    qe_warmup = np.mean([m.qfield_energy for m in warmup_data])
                    qe_freerun = np.mean([m.qfield_energy for m in freerun_data])
                    delta_qe = qe_freerun - qe_warmup
                    stability = "estable" if abs(delta_qe) < 0.05 else ("drift+" if delta_qe > 0 else "drift-")
                    f.write(f"- qfield_energy de {agent}: {stability} (delta={delta_qe:+.4f})\n")

            # Global drift check
            warmup_global = [m for m in self.global_metrics_log if m.phase == "warmup"]
            freerun_global = [m for m in self.global_metrics_log if m.phase == "free_run"]

            if warmup_global and freerun_global:
                coh_w = np.mean([m.field_mean_coherence for m in warmup_global])
                coh_f = np.mean([m.field_mean_coherence for m in freerun_global])
                f.write(f"\n- Global Q-Field coherence: WARMUP={coh_w:.4f}, FREE_RUN={coh_f:.4f}\n")

    def _generate_plots(self, root_figs: Path, run_prefix: str):
        """Genera todas las gráficas."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Backend sin GUI
        except ImportError:
            print("WARNING: matplotlib no disponible, saltando gráficas")
            return

        # Colores por agente
        colors = {
            'NEO': '#1f77b4',
            'EVA': '#ff7f0e',
            'ALEX': '#2ca02c',
            'ADAM': '#d62728',
            'IRIS': '#9467bd',
        }

        # Datos por agente y tiempo
        agent_data = {agent: [] for agent in self.agents}
        for m in self.agent_metrics_log:
            agent_data[m.agent_id].append(m)

        # =====================================================
        # 1. CE Timeline
        # =====================================================
        fig, ax = plt.subplots(figsize=(12, 6))

        for agent in self.agents:
            data = agent_data[agent]
            t_vals = [m.t for m in data]
            ce_vals = [m.CE for m in data]
            ax.plot(t_vals, ce_vals, label=agent, color=colors.get(agent, 'gray'), alpha=0.8)

        # Marcar transición WARMUP -> FREE_RUN
        ax.axvline(x=WARMUP_STEPS, color='red', linestyle='--', alpha=0.5, label='WARMUP->FREE_RUN')
        ax.axvspan(0, WARMUP_STEPS, alpha=0.1, color='yellow')
        ax.axvspan(WARMUP_STEPS, WARMUP_STEPS + FREE_RUN_STEPS, alpha=0.1, color='green')

        ax.set_xlabel('t (pasos)')
        ax.set_ylabel('CE (Coherencia Existencial)')
        ax.set_title('Omega Last12h: CE Timeline')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(root_figs / f"{run_prefix}_ce_timeline.png", dpi=150)
        plt.close()

        # =====================================================
        # 2. Q-Field Energy
        # =====================================================
        fig, ax = plt.subplots(figsize=(12, 6))

        for agent in self.agents:
            data = agent_data[agent]
            t_vals = [m.t for m in data]
            qe_vals = [m.qfield_energy for m in data]
            ax.plot(t_vals, qe_vals, label=agent, color=colors.get(agent, 'gray'), alpha=0.8)

        ax.axvline(x=WARMUP_STEPS, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('t (pasos)')
        ax.set_ylabel('E_Q(t) (Superposition Energy)')
        ax.set_title('Omega Last12h: Q-Field Energy')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(root_figs / f"{run_prefix}_qfield_energy.png", dpi=150)
        plt.close()

        # =====================================================
        # 3. Omega Modes Active
        # =====================================================
        fig, ax = plt.subplots(figsize=(12, 6))

        for agent in self.agents:
            data = agent_data[agent]
            t_vals = [m.t for m in data]
            om_vals = [m.omega_modes_active for m in data]
            ax.plot(t_vals, om_vals, label=agent, color=colors.get(agent, 'gray'), alpha=0.8)

        ax.axvline(x=WARMUP_STEPS, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('t (pasos)')
        ax.set_ylabel('Omega Modes Active')
        ax.set_title('Omega Last12h: Omega Modes Active')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(root_figs / f"{run_prefix}_omega_modes.png", dpi=150)
        plt.close()

        # =====================================================
        # 4. Phase Curvature
        # =====================================================
        fig, ax = plt.subplots(figsize=(12, 6))

        for agent in self.agents:
            data = agent_data[agent]
            t_vals = [m.t for m in data]
            pc_vals = [m.phase_curvature for m in data]
            ax.plot(t_vals, pc_vals, label=agent, color=colors.get(agent, 'gray'), alpha=0.8)

        ax.axvline(x=WARMUP_STEPS, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('t (pasos)')
        ax.set_ylabel('Phase Curvature')
        ax.set_title('Omega Last12h: PhaseSpace Curvature')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(root_figs / f"{run_prefix}_curvature.png", dpi=150)
        plt.close()

        # =====================================================
        # 5. TensorMind Modes (Global)
        # =====================================================
        fig, ax = plt.subplots(figsize=(12, 6))

        t_vals = [m.t for m in self.global_metrics_log]
        tm_vals = [m.tensor_modes_strong for m in self.global_metrics_log]
        ax.plot(t_vals, tm_vals, color='purple', alpha=0.8, linewidth=2)

        ax.axvline(x=WARMUP_STEPS, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('t (pasos)')
        ax.set_ylabel('TensorMind Strong Modes')
        ax.set_title('Omega Last12h: TensorMind Strong Modes (Global)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(root_figs / f"{run_prefix}_tensormind_modes.png", dpi=150)
        plt.close()

        # =====================================================
        # ComplexField plots (si disponible)
        # =====================================================
        if COMPLEX_FIELD_AVAILABLE:
            # 6. Lambda Decoherence
            fig, ax = plt.subplots(figsize=(12, 6))

            for agent in self.agents:
                data = agent_data[agent]
                t_vals = [m.t for m in data]
                ld_vals = [m.lambda_decoherence for m in data]
                ax.plot(t_vals, ld_vals, label=agent, color=colors.get(agent, 'gray'), alpha=0.8)

            ax.axvline(x=WARMUP_STEPS, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('t (pasos)')
            ax.set_ylabel('Lambda Decoherence')
            ax.set_title('ComplexField: Lambda Decoherence')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(root_figs / f"complex_last12h_lambda.png", dpi=150)
            plt.close()

            # 7. Collapse Pressure
            fig, ax = plt.subplots(figsize=(12, 6))

            for agent in self.agents:
                data = agent_data[agent]
                t_vals = [m.t for m in data]
                cp_vals = [m.collapse_pressure for m in data]
                ax.plot(t_vals, cp_vals, label=agent, color=colors.get(agent, 'gray'), alpha=0.8)

            ax.axvline(x=WARMUP_STEPS, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('t (pasos)')
            ax.set_ylabel('Collapse Pressure')
            ax.set_title('ComplexField: Collapse Pressure')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(root_figs / f"complex_last12h_collapse_pressure.png", dpi=150)
            plt.close()

            # 8. Psi Norm
            fig, ax = plt.subplots(figsize=(12, 6))

            for agent in self.agents:
                data = agent_data[agent]
                t_vals = [m.t for m in data]
                pn_vals = [m.psi_norm for m in data]
                ax.plot(t_vals, pn_vals, label=agent, color=colors.get(agent, 'gray'), alpha=0.8)

            ax.axvline(x=WARMUP_STEPS, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('t (pasos)')
            ax.set_ylabel('Psi Norm')
            ax.set_title('ComplexField: Psi Norm')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(root_figs / f"complex_last12h_psi_norm.png", dpi=150)
            plt.close()

            # 9. Phase Entropy
            fig, ax = plt.subplots(figsize=(12, 6))

            for agent in self.agents:
                data = agent_data[agent]
                t_vals = [m.t for m in data]
                pe_vals = [m.phase_entropy for m in data]
                ax.plot(t_vals, pe_vals, label=agent, color=colors.get(agent, 'gray'), alpha=0.8)

            ax.axvline(x=WARMUP_STEPS, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('t (pasos)')
            ax.set_ylabel('Phase Entropy')
            ax.set_title('ComplexField: Phase Entropy')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(root_figs / f"complex_last12h_phase_entropy.png", dpi=150)
            plt.close()

        print(f"Graficas generadas en: {root_figs}/")


# =====================================================================
# Main
# =====================================================================

def main():
    """Ejecuta simulacion de ausencia prolongada."""
    sim = OmegaLast12hSimulation(
        agents=AGENTS,
        state_dim=STATE_DIM,
        prob_dim=PROB_DIM,
    )

    paths = sim.run(
        warmup_steps=WARMUP_STEPS,
        free_run_steps=FREE_RUN_STEPS,
        log_interval=100,
    )

    return paths


if __name__ == "__main__":
    main()
