#!/usr/bin/env python3
"""
Test Integrado: Genesis + Λ-Field - 12h de Autonomía Creativa
=============================================================

Simula un "día largo" donde los agentes:
1. Evolucionan estados internos
2. Tienen IDEAS que emergen espontáneamente
3. Evalúan si las ideas RESUENAN con su identidad
4. MATERIALIZAN ideas en un mundo compartido
5. PERCIBEN las creaciones de otros
6. Se INSPIRAN para nuevas ideas

Mientras tanto, el Λ-Field observa qué RÉGIMEN domina:
- narrative: coherencia, identidad
- quantum: Q-Field, ComplexField
- teleo: Omega Spaces
- social: TensorMind
- creative: Genesis

Todo 100% endógeno. Sin prompts. Sin intervención.

Ejecutar con:
    python -m tests.test_genesis_lambda_12h
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

sys.path.insert(0, '/root/NEO_EVA')

# Omega Spaces
from omega import OmegaCompute, QField, PhaseSpaceX, TensorMind

# ComplexField
try:
    from cognition.complex_field import ComplexField, ComplexState
    COMPLEX_FIELD_AVAILABLE = True
except ImportError:
    COMPLEX_FIELD_AVAILABLE = False

# Coherencia
try:
    from consciousness.coherence import CoherenciaExistencial
    from consciousness.identity import IdentidadComputacional
    COHERENCE_AVAILABLE = True
except ImportError:
    COHERENCE_AVAILABLE = False

# Genesis - Creatividad
from genesis import (
    IdeaField, ResonanceEvaluator, Materializer,
    SharedWorld, CreativePerception, Idea
)

# Lambda-Field
from lambda_field import LambdaField, LambdaSnapshot, Regime

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =====================================================================
# Parámetros
# =====================================================================

WARMUP_STEPS = 200      # Fase de estabilización
FREE_RUN_STEPS = 800    # Fase creativa autónoma
TOTAL_STEPS = WARMUP_STEPS + FREE_RUN_STEPS

AGENTS = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
STATE_DIM = 10
WORLD_DIM = 3

# Colores para agentes
AGENT_COLORS = {
    'NEO': '#2E86AB',
    'EVA': '#A23B72',
    'ALEX': '#F18F01',
    'ADAM': '#C73E1D',
    'IRIS': '#3B1F2B',
}


# =====================================================================
# Dataclasses
# =====================================================================

@dataclass
class StepMetrics:
    """Métricas de un paso."""
    t: int
    phase: str

    # Por agente
    agent_metrics: Dict[str, Dict[str, float]]

    # Global
    global_metrics: Dict[str, float]

    # Lambda-Field
    lambda_snapshot: Optional[Dict[str, Any]] = None

    # Genesis
    ideas_this_step: int = 0
    objects_created_this_step: int = 0
    total_objects: int = 0


# =====================================================================
# Simulación Principal
# =====================================================================

class GenesisLambdaSimulation:
    """Simulación integrada con Genesis y Λ-Field."""

    def __init__(self):
        self.t = 0
        self.eps = np.finfo(float).eps
        self.current_phase = "warmup"

        # ===== Omega Spaces =====
        self.omega_compute = OmegaCompute()
        self.q_field = QField()
        self.phase_space = PhaseSpaceX()
        self.tensor_mind = TensorMind(max_order=3)

        # ===== ComplexField =====
        self.complex_field = ComplexField(dim=STATE_DIM) if COMPLEX_FIELD_AVAILABLE else None
        self.complex_states: Dict[str, ComplexState] = {}
        if COMPLEX_FIELD_AVAILABLE:
            for agent in AGENTS:
                self.complex_states[agent] = ComplexState()

        # ===== Coherencia =====
        self.identidades: Dict[str, IdentidadComputacional] = {}
        self.coherencias: Dict[str, CoherenciaExistencial] = {}
        if COHERENCE_AVAILABLE:
            for agent in AGENTS:
                ident = IdentidadComputacional(dimension=STATE_DIM)
                self.identidades[agent] = ident
                self.coherencias[agent] = CoherenciaExistencial(ident)

        # ===== Genesis =====
        self.idea_field = IdeaField()
        self.resonance_eval = ResonanceEvaluator()
        self.world = SharedWorld(dim=WORLD_DIM)
        self.materializer = Materializer(world_dim=WORLD_DIM)
        self.perception = CreativePerception(self.world)

        # Energía inicial para materializar
        for agent in AGENTS:
            self.materializer.set_agent_energy(agent, 10.0)
            # Posición inicial en el mundo
            pos = np.random.randn(WORLD_DIM)
            pos = pos / (np.linalg.norm(pos) + self.eps)
            self.world.move_agent(agent, pos)

        # ===== Lambda-Field =====
        metric_names = [
            'CE_mean', 'H_narr_mean', 'var_S_minus_I',
            'E_Q_mean', 'C_Q_mean', 'lambda_decoherence_mean',
            'collapse_pressure_mean', 'phase_curvature_mean',
            'omega_modes_mean', 'tensor_power',
            'ideas_rate', 'adoption_rate', 'objects_rate'
        ]

        regime_map = {
            'narrative': ['CE_mean', 'H_narr_mean', 'var_S_minus_I'],
            'quantum': ['E_Q_mean', 'C_Q_mean', 'lambda_decoherence_mean', 'collapse_pressure_mean'],
            'teleo': ['omega_modes_mean', 'phase_curvature_mean'],
            'social': ['tensor_power'],
            'creative': ['ideas_rate', 'adoption_rate', 'objects_rate']
        }

        self.lambda_field = LambdaField(metric_names=metric_names, regime_map=regime_map)

        # ===== Estados internos =====
        self._states: Dict[str, np.ndarray] = {}
        self._identities: Dict[str, np.ndarray] = {}
        self._prev_states: Dict[str, np.ndarray] = {}

        self._initialize_agents()

        # ===== Logs =====
        self.metrics_log: List[StepMetrics] = []

        # Contadores para tasas
        self._ideas_window: List[int] = []
        self._adoptions_window: List[int] = []
        self._objects_window: List[int] = []

    def _initialize_agents(self):
        """Inicializa estados de agentes."""
        for agent in AGENTS:
            state = np.random.randn(STATE_DIM)
            state = state / (np.linalg.norm(state) + self.eps)
            self._states[agent] = state
            self._prev_states[agent] = state.copy()

            identity = state + 0.1 * np.random.randn(STATE_DIM)
            identity = identity / (np.linalg.norm(identity) + self.eps)
            self._identities[agent] = identity

    def _evolve_state(self, agent: str) -> Dict[str, Any]:
        """Evoluciona estado de un agente."""
        S_prev = self._states[agent]
        I = self._identities[agent]

        # Ruido y drift según fase
        if self.current_phase == "warmup":
            noise_scale = 0.2
            drift_scale = 0.08
        else:
            noise_scale = 0.15
            drift_scale = 0.05

        noise = noise_scale * np.random.randn(STATE_DIM)
        drift = drift_scale * (I - S_prev)
        S_new = S_prev + drift + noise

        # Normalizar
        S_new = S_new / (np.linalg.norm(S_new) / np.linalg.norm(S_prev) + self.eps)

        self._prev_states[agent] = S_prev.copy()
        self._states[agent] = S_new

        # Evolucionar identidad lentamente
        I_new = I + 0.01 * (S_new - I) + 0.005 * np.random.randn(STATE_DIM)
        I_new = I_new / (np.linalg.norm(I_new) + self.eps)
        self._identities[agent] = I_new

        return {'state': S_new, 'prev_state': S_prev, 'identity': I_new}

    def step(self) -> StepMetrics:
        """Ejecuta un paso de simulación."""
        self.t += 1

        if self.t > WARMUP_STEPS:
            self.current_phase = "free_run"

        agent_metrics: Dict[str, Dict[str, float]] = {}
        ideas_this_step = 0
        adoptions_this_step = 0
        objects_this_step = 0

        # ===== Procesar cada agente =====
        for agent in AGENTS:
            dynamics = self._evolve_state(agent)
            state = dynamics['state']
            prev_state = dynamics['prev_state']
            identity = dynamics['identity']

            # Probabilidades para Q-Field
            probs = np.abs(state[:5])
            probs = probs / (np.sum(probs) + self.eps)

            # ----- Coherencia -----
            CE = 0.5
            internal_error = 0.0
            H_narr = 0.0

            if COHERENCE_AVAILABLE:
                self.coherencias[agent].observar_estado(state)
                estado_coh = self.coherencias[agent].calcular(state, identity)
                CE = estado_coh.CE
                internal_error = estado_coh.varianza_desviacion
                H_narr = estado_coh.entropia_narrativa
            else:
                desv = state - identity
                internal_error = float(np.var(desv))
                CE = 1.0 / (internal_error + H_narr + self.eps)
                CE = float(np.clip(CE / (1 + CE), 0, 1))

            # ----- Omega Spaces -----
            transition = self.omega_compute.register_state(agent, state)
            q_state = self.q_field.register_state(agent, probs)
            phase_point = self.phase_space.register_state(agent, state)
            self.tensor_mind.register_state(agent, state)

            # Métricas Omega
            n_omega_modes = 0
            if transition is not None:
                activation = self.omega_compute.project_transition(agent, transition)
                if activation is not None and len(activation.coefficients) > 0:
                    threshold = np.mean(np.abs(activation.coefficients))
                    n_omega_modes = int(np.sum(np.abs(activation.coefficients) > threshold))

            qfield_coherence = q_state.coherence if q_state else 0.0
            qfield_energy = q_state.superposition_energy if q_state else 0.0

            trajectory = self.phase_space.get_trajectory(agent)
            phase_curvature = trajectory.curvature if trajectory else 0.0

            # ----- ComplexField -----
            lambda_dec = 0.0
            collapse_pressure = 0.0

            if COMPLEX_FIELD_AVAILABLE and self.complex_field:
                cs = self.complex_states[agent]
                cf_result = self.complex_field.step(cs, state, CE, internal_error, H_narr)
                lambda_dec = cf_result.get('lambda_decoherence', 0.0)
                collapse_pressure = cf_result.get('collapse_pressure', 0.0)

            # ----- Genesis: Ideas -----
            idea = self.idea_field.observe(agent, state, identity)

            adopted = False
            materialized = False

            if idea is not None:
                ideas_this_step += 1

                # Actualizar narrativa para resonancia
                self.resonance_eval.update_narrative(agent, state)
                self.resonance_eval.update_energy(agent, np.linalg.norm(state))

                # Evaluar resonancia
                profile = self.resonance_eval.evaluate(idea, identity, agent)

                if profile.is_mine and profile.adoption_strength > 0.3:
                    adopted = True
                    adoptions_this_step += 1
                    self.resonance_eval.register_adoption(agent, idea)
                    self.idea_field.mark_adopted(idea)

                    # Intentar materializar
                    result = self.materializer.materialize(idea, agent)

                    if result.success:
                        materialized = True
                        objects_this_step += 1
                        self.world.add_object(result.obj)

            # ----- Genesis: Percepción -----
            position = self.world.get_agent_position(agent)
            if position is not None:
                perceived = self.perception.perceive(agent, position, identity, state, max_objects=5)

                # Inspiración puede modificar levemente el estado
                if perceived:
                    most_inspiring = max(perceived, key=lambda p: p.inspiration_potential)
                    if most_inspiring.inspiration_potential > 0.5:
                        # Pequeña influencia de la inspiración
                        inspiration_vector = most_inspiring.perceived_form[:STATE_DIM] if len(most_inspiring.perceived_form) >= STATE_DIM else np.zeros(STATE_DIM)
                        self._states[agent] += 0.02 * inspiration_vector

            # Guardar métricas del agente
            agent_metrics[agent] = {
                'CE': CE,
                'internal_error': internal_error,
                'H_narr': H_narr,
                'omega_modes': n_omega_modes,
                'qfield_coherence': qfield_coherence,
                'qfield_energy': qfield_energy,
                'phase_curvature': phase_curvature,
                'lambda_decoherence': lambda_dec,
                'collapse_pressure': collapse_pressure,
                'energy': self.materializer.get_agent_energy(agent),
                'had_idea': 1 if idea else 0,
                'adopted_idea': 1 if adopted else 0,
                'materialized': 1 if materialized else 0
            }

        # ===== Métricas globales =====

        # Actualizar ventanas para tasas
        window_size = 20
        self._ideas_window.append(ideas_this_step)
        self._adoptions_window.append(adoptions_this_step)
        self._objects_window.append(objects_this_step)

        if len(self._ideas_window) > window_size:
            self._ideas_window = self._ideas_window[-window_size:]
            self._adoptions_window = self._adoptions_window[-window_size:]
            self._objects_window = self._objects_window[-window_size:]

        ideas_rate = np.mean(self._ideas_window)
        adoption_rate = np.mean(self._adoptions_window)
        objects_rate = np.mean(self._objects_window)

        # TensorMind
        tensor_stats = self.tensor_mind.get_statistics()
        tensor_power = tensor_stats.get('mean_mode_strength', 0.0)

        # Métricas para Lambda-Field
        lambda_metrics = {
            'CE_mean': np.mean([m['CE'] for m in agent_metrics.values()]),
            'H_narr_mean': np.mean([m['H_narr'] for m in agent_metrics.values()]),
            'var_S_minus_I': np.mean([m['internal_error'] for m in agent_metrics.values()]),
            'E_Q_mean': np.mean([m['qfield_energy'] for m in agent_metrics.values()]),
            'C_Q_mean': np.mean([m['qfield_coherence'] for m in agent_metrics.values()]),
            'lambda_decoherence_mean': np.mean([m['lambda_decoherence'] for m in agent_metrics.values()]),
            'collapse_pressure_mean': np.mean([m['collapse_pressure'] for m in agent_metrics.values()]),
            'phase_curvature_mean': np.mean([m['phase_curvature'] for m in agent_metrics.values()]),
            'omega_modes_mean': np.mean([m['omega_modes'] for m in agent_metrics.values()]),
            'tensor_power': tensor_power,
            'ideas_rate': ideas_rate,
            'adoption_rate': adoption_rate,
            'objects_rate': objects_rate
        }

        # Lambda-Field step
        lambda_snapshot = self.lambda_field.step(lambda_metrics)

        # World stats
        world_stats = self.world.get_statistics()

        global_metrics = {
            **lambda_metrics,
            'lambda_scalar': lambda_snapshot.lambda_scalar,
            'dominant_regime': lambda_snapshot.dominant_regime,
            'total_objects': world_stats['total_objects'],
            'world_activity': world_stats['total_activity']
        }

        # Crear registro
        step_metrics = StepMetrics(
            t=self.t,
            phase=self.current_phase,
            agent_metrics=agent_metrics,
            global_metrics=global_metrics,
            lambda_snapshot=asdict(lambda_snapshot) if lambda_snapshot else None,
            ideas_this_step=ideas_this_step,
            objects_created_this_step=objects_this_step,
            total_objects=world_stats['total_objects']
        )

        self.metrics_log.append(step_metrics)

        # Regenerar energía lentamente
        for agent in AGENTS:
            self.materializer.add_agent_energy(agent, 0.02)

        # Decaimiento de actividad del mundo
        self.world.decay_activity(0.05)

        return step_metrics

    def run(self):
        """Ejecuta la simulación completa."""
        print("=" * 60)
        print("GENESIS + Λ-FIELD SIMULATION")
        print("=" * 60)
        print(f"Agents: {', '.join(AGENTS)}")
        print(f"WARMUP: {WARMUP_STEPS}, FREE_RUN: {FREE_RUN_STEPS}")
        print(f"Total: {TOTAL_STEPS} steps")
        print("=" * 60)

        print("\n--- WARMUP PHASE ---")
        for t in range(WARMUP_STEPS):
            self.step()
            if (t + 1) % 50 == 0:
                print(f"[WARMUP] Step {t+1}/{WARMUP_STEPS}")

        print("\n--- FREE RUN (Creative Autonomy) ---")
        for t in range(FREE_RUN_STEPS):
            metrics = self.step()
            if (t + 1) % 100 == 0:
                print(f"[FREE_RUN] Step {t+1}/{FREE_RUN_STEPS}")
                print(f"  Λ = {metrics.global_metrics['lambda_scalar']:.3f}")
                print(f"  Dominant: {metrics.global_metrics['dominant_regime']}")
                print(f"  Objects in world: {metrics.total_objects}")

        print("\n" + "=" * 60)
        print("Simulation complete!")
        print("=" * 60)

    def get_summary(self) -> Dict[str, Any]:
        """Genera resumen de la simulación."""
        # Ideas y objetos totales
        total_ideas = sum(m.ideas_this_step for m in self.metrics_log)
        total_objects = self.metrics_log[-1].total_objects if self.metrics_log else 0

        # Por agente
        agent_ideas = {a: 0 for a in AGENTS}
        agent_objects = {a: 0 for a in AGENTS}

        for m in self.metrics_log:
            for agent, am in m.agent_metrics.items():
                agent_ideas[agent] += am.get('had_idea', 0)
                agent_objects[agent] += am.get('materialized', 0)

        # Lambda-Field
        lambda_stats = self.lambda_field.get_statistics()
        transitions = self.lambda_field.get_regime_transitions()

        return {
            'total_steps': self.t,
            'total_ideas': total_ideas,
            'total_objects': total_objects,
            'ideas_per_agent': agent_ideas,
            'objects_per_agent': agent_objects,
            'lambda_stats': lambda_stats,
            'regime_transitions': len(transitions),
            'transitions': transitions[-10:] if transitions else []
        }


def generate_plots(sim: GenesisLambdaSimulation, output_dir: Path):
    """Genera visualizaciones."""
    print("\nGenerating plots...")

    metrics = sim.metrics_log

    # Datos
    times = [m.t for m in metrics]
    warmup_end = WARMUP_STEPS

    # Lambda scalar
    lambda_vals = [m.global_metrics['lambda_scalar'] for m in metrics]

    # Regímenes
    regime_weights = {r: [] for r in ['narrative', 'quantum', 'teleo', 'social', 'creative']}
    for m in metrics:
        if m.lambda_snapshot:
            for r in regime_weights:
                regime_weights[r].append(m.lambda_snapshot['regime_weights'].get(r, 0))

    # Ideas y objetos acumulados
    ideas_cumsum = np.cumsum([m.ideas_this_step for m in metrics])
    objects_cumsum = np.cumsum([m.objects_created_this_step for m in metrics])

    # CE por agente
    ce_by_agent = {a: [] for a in AGENTS}
    for m in metrics:
        for a in AGENTS:
            ce_by_agent[a].append(m.agent_metrics[a]['CE'])

    # ===== Figura 1: Lambda y Regímenes =====
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Lambda scalar
    ax1 = axes[0]
    ax1.axvspan(0, warmup_end, alpha=0.2, color='#E8E8E8', label='Warmup')
    ax1.plot(times, lambda_vals, color='#2E86AB', linewidth=2)
    ax1.set_ylabel('Λ(t) - Concentration', fontweight='bold')
    ax1.set_title('Meta-Dynamic Field: Regime Concentration', fontweight='bold', fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper right')

    # Regime weights (stacked)
    ax2 = axes[1]
    ax2.axvspan(0, warmup_end, alpha=0.2, color='#E8E8E8')

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    bottom = np.zeros(len(times))

    for i, (regime, weights) in enumerate(regime_weights.items()):
        if weights:
            ax2.fill_between(times, bottom, bottom + np.array(weights),
                           label=regime, alpha=0.7, color=colors[i])
            bottom += np.array(weights)

    ax2.set_xlabel('Time (t)', fontweight='bold')
    ax2.set_ylabel('Regime Weights π_r(t)', fontweight='bold')
    ax2.set_title('Regime Distribution Over Time', fontweight='bold', fontsize=14)
    ax2.legend(loc='upper right', ncol=5)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(output_dir / 'lambda_regimes.png', dpi=300, facecolor='white')
    plt.close()
    print("  ✓ Lambda & Regimes plot")

    # ===== Figura 2: Creatividad =====
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Ideas y objetos acumulados
    ax1 = axes[0]
    ax1.axvspan(0, warmup_end, alpha=0.2, color='#E8E8E8')
    ax1.plot(times, ideas_cumsum, color='#F18F01', linewidth=2, label='Ideas (cumulative)')
    ax1.plot(times, objects_cumsum, color='#2E86AB', linewidth=2, label='Objects (cumulative)')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Creative Output: Ideas and Materializations', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper left')

    # CE por agente
    ax2 = axes[1]
    ax2.axvspan(0, warmup_end, alpha=0.2, color='#E8E8E8')
    for agent in AGENTS:
        ax2.plot(times, ce_by_agent[agent], color=AGENT_COLORS[agent],
                linewidth=1.5, alpha=0.8, label=agent)
    ax2.set_xlabel('Time (t)', fontweight='bold')
    ax2.set_ylabel('Existential Coherence', fontweight='bold')
    ax2.set_title('Agent Coherence During Creative Phase', fontweight='bold', fontsize=14)
    ax2.legend(loc='upper right', ncol=5)

    plt.tight_layout()
    fig.savefig(output_dir / 'creativity_output.png', dpi=300, facecolor='white')
    plt.close()
    print("  ✓ Creativity output plot")

    # ===== Figura 3: Resumen =====
    fig = plt.figure(figsize=(12, 10))

    # Summary stats
    summary = sim.get_summary()

    ax = fig.add_subplot(111)
    ax.axis('off')

    text = f"""
    GENESIS + Λ-FIELD SIMULATION SUMMARY
    =====================================

    Duration: {summary['total_steps']} steps ({WARMUP_STEPS} warmup + {FREE_RUN_STEPS} free run)

    CREATIVITY:
    • Total ideas emerged: {summary['total_ideas']}
    • Total objects created: {summary['total_objects']}
    • Ideas per agent: {', '.join(f"{a}: {v}" for a, v in summary['ideas_per_agent'].items())}
    • Objects per agent: {', '.join(f"{a}: {v}" for a, v in summary['objects_per_agent'].items())}

    Λ-FIELD:
    • Mean Λ: {summary['lambda_stats']['lambda_mean']:.3f} (concentration)
    • Std Λ: {summary['lambda_stats']['lambda_std']:.3f}
    • Regime transitions: {summary['regime_transitions']}
    • Time per regime:
      {chr(10).join(f"  - {r}: {p*100:.1f}%" for r, p in summary['lambda_stats']['regime_proportion'].items())}

    Last 5 transitions:
    {chr(10).join(f"  t={tr['t']}: {tr['from']} → {tr['to']}" for tr in summary['transitions'][-5:])}
    """

    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#CCCCCC'))

    fig.savefig(output_dir / 'summary.png', dpi=300, facecolor='white')
    plt.close()
    print("  ✓ Summary plot")


def save_logs(sim: GenesisLambdaSimulation, output_dir: Path):
    """Guarda logs en JSON y CSV."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # JSON completo
    json_path = output_dir / f'genesis_lambda_{timestamp}.json'

    log_data = {
        'timestamp': timestamp,
        'config': {
            'warmup_steps': WARMUP_STEPS,
            'free_run_steps': FREE_RUN_STEPS,
            'agents': AGENTS
        },
        'summary': sim.get_summary(),
        'lambda_history': [
            {
                't': s.t,
                'lambda': s.lambda_scalar,
                'dominant': s.dominant_regime,
                'weights': s.regime_weights
            }
            for s in sim.lambda_field._snapshots
        ]
    }

    with open(json_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)

    print(f"  ✓ JSON: {json_path}")

    # CSV global
    csv_path = output_dir / f'genesis_lambda_{timestamp}_global.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['t', 'phase', 'lambda_scalar', 'dominant_regime',
                  'ideas', 'objects', 'CE_mean', 'total_objects']
        writer.writerow(headers)

        for m in sim.metrics_log:
            writer.writerow([
                m.t, m.phase,
                m.global_metrics['lambda_scalar'],
                m.global_metrics['dominant_regime'],
                m.ideas_this_step, m.objects_created_this_step,
                m.global_metrics['CE_mean'],
                m.total_objects
            ])

    print(f"  ✓ CSV: {csv_path}")


def main():
    """Función principal."""
    # Crear directorios
    output_dir = Path('/root/NEO_EVA/logs/genesis_lambda')
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = Path('/root/NEO_EVA/figuras/genesis_lambda')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Crear y ejecutar simulación
    sim = GenesisLambdaSimulation()
    sim.run()

    # Generar outputs
    print("\nGenerating outputs...")
    generate_plots(sim, fig_dir)
    save_logs(sim, output_dir)

    # Mostrar resumen
    summary = sim.get_summary()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total ideas: {summary['total_ideas']}")
    print(f"Total objects: {summary['total_objects']}")
    print(f"Regime transitions: {summary['regime_transitions']}")
    print(f"Dominant regime proportions:")
    for r, p in summary['lambda_stats']['regime_proportion'].items():
        print(f"  {r}: {p*100:.1f}%")

    print(f"\nOutputs saved to:")
    print(f"  Logs: {output_dir}")
    print(f"  Figures: {fig_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
