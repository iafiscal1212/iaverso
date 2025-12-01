#!/usr/bin/env python3
"""
NEO_EVA: Experimentos de Vida Larga - Versión Dual
===================================================

Ejecuta simulaciones extendidas con métricas separadas para NEO y EVA:
- SAGI_NEO(t), SAGI_EVA(t)
- IGI_NEO(t), IGI_EVA(t)
- GI_NEO(t), GI_EVA(t)
- φ^NEO_t, φ^EVA_t
- τ_NEO, τ_EVA (tiempos subjetivos)

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/core')
sys.path.insert(0, '/root/NEO_EVA/integration')
sys.path.insert(0, '/root/NEO_EVA/grounding')
sys.path.insert(0, '/root/NEO_EVA/subjectivity')

from core.agents import DualAgentSystem
from integration.phaseI2_igi_dual import DualAgentIGI
from grounding.phaseG2_grounding_dual import DualAgentGrounding
from grounding.phaseG1_world_channel import StructuredWorldChannel
from subjectivity.phaseS1_dual import DualPhenomenalState


@dataclass
class AgentTimePoint:
    """Métricas de un agente en un instante."""
    t: int
    agent: str

    # SAGI components
    S: float
    learning: float
    integration: float
    SAGI: float

    # IGI
    IGI: float
    I_int: float
    I_eco: float

    # Grounding
    GI: float
    G_pred: float
    G_sym: float

    # Phenomenological
    psi: float
    identity: float
    otherness: float
    mode: int

    # Tiempo subjetivo
    tau: float


@dataclass
class DualExperimentRun:
    """Resultados de una ejecución dual."""
    seed: int
    T: int
    neo_timepoints: List[AgentTimePoint] = field(default_factory=list)
    eva_timepoints: List[AgentTimePoint] = field(default_factory=list)
    final_comparison: Dict[str, Any] = field(default_factory=dict)


def compute_agent_sagi(z: np.ndarray, z_history: List[np.ndarray]) -> Dict[str, float]:
    """
    Calcula SAGI para un agente.

    100% endógeno
    """
    # Entropía
    z_safe = np.clip(z, 1e-10, 1.0)
    z_norm = z_safe / z_safe.sum()
    S = -np.sum(z_norm * np.log(z_norm))
    S_max = np.log(len(z))
    S_norm = S / S_max if S_max > 0 else 0

    # Learning (variabilidad reciente)
    if len(z_history) > 10:
        recent = np.array(z_history[-10:])
        learning = np.std(recent)
        learning = min(learning * 10, 1.0)
    else:
        learning = 0.5

    # Integration (correlación)
    if len(z_history) > 20:
        H = np.array(z_history[-20:])
        if H.shape[1] > 1:
            corr_matrix = np.corrcoef(H.T)
            mask = ~np.eye(len(z), dtype=bool)
            correlations = corr_matrix[mask]
            correlations = correlations[~np.isnan(correlations)]
            integration = np.mean(np.abs(correlations)) if len(correlations) > 0 else 0.5
        else:
            integration = 0.5
    else:
        integration = 0.5

    # SAGI = media geométrica
    components = [S_norm, learning, integration]
    sagi = np.prod([max(c, 1e-10) ** (1/len(components)) for c in components])

    return {
        'S': float(S),
        'learning': float(learning),
        'integration': float(integration),
        'SAGI': float(sagi)
    }


def compute_subjective_time(z_history: List[np.ndarray], window: int = 20) -> float:
    """
    Calcula tiempo subjetivo τ.

    τ = entropía de la trayectoria reciente
    Mayor τ = más "eventos" percibidos

    100% endógeno
    """
    if len(z_history) < window:
        return 0.5

    recent = np.array(z_history[-window:])

    # Calcular cambios
    diffs = np.diff(recent, axis=0)
    change_magnitudes = np.linalg.norm(diffs, axis=1)

    # Discretizar cambios
    if np.std(change_magnitudes) > 1e-10:
        n_bins = 5
        bins = np.percentile(change_magnitudes, np.linspace(0, 100, n_bins + 1))
        digitized = np.digitize(change_magnitudes, bins[1:-1])

        # Entropía de la distribución de cambios
        counts = np.bincount(digitized, minlength=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        tau = -np.sum(probs * np.log(probs)) / np.log(n_bins)  # Normalizado
    else:
        tau = 0.5

    return float(tau)


def run_dual_experiment(seed: int, T: int, verbose: bool = False) -> DualExperimentRun:
    """Ejecuta un experimento dual."""

    np.random.seed(seed)

    # Sistemas
    dual = DualAgentSystem(dim_visible=3, dim_hidden=3)
    igi_dual = DualAgentIGI(total_dim=12)
    grounding = DualAgentGrounding(dim_world=6, dim_neo=6, dim_eva=6)
    phenomenal = DualPhenomenalState(dim_z=6)
    world = StructuredWorldChannel(dim_s=6, seed=seed)

    # Historias
    neo_z_history = []
    eva_z_history = []

    neo_timepoints = []
    eva_timepoints = []

    for t in range(T):
        # Mundo
        world_state = world.step()
        stimulus = world_state.s[:6]

        # Sistema dual
        dual_result = dual.step(stimulus)

        neo_state = dual.neo.get_state()
        eva_state = dual.eva.get_state()

        neo_z = neo_state.get_full_state()
        eva_z = eva_state.get_full_state()

        neo_z_history.append(neo_z.copy())
        eva_z_history.append(eva_z.copy())

        # IGI dual
        z_combined = np.concatenate([neo_z, eva_z])
        igi_result = igi_dual.step(z_combined)

        # Grounding dual
        grounding_result = grounding.step(neo_z, eva_z)

        # Phenomenal dual
        phi_result = phenomenal.step(
            neo_state, dual_result['neo_response'],
            eva_state, dual_result['eva_response']
        )

        # SAGI por agente
        neo_sagi = compute_agent_sagi(neo_z, neo_z_history)
        eva_sagi = compute_agent_sagi(eva_z, eva_z_history)

        # Tiempo subjetivo por agente
        neo_tau = compute_subjective_time(neo_z_history)
        eva_tau = compute_subjective_time(eva_z_history)

        # Registrar NEO
        neo_tp = AgentTimePoint(
            t=t,
            agent='NEO',
            S=neo_sagi['S'],
            learning=neo_sagi['learning'],
            integration=neo_sagi['integration'],
            SAGI=neo_sagi['SAGI'],
            IGI=igi_result['IGI_NEO'] if igi_result['ready'] else 0.0,
            I_int=igi_result['NEO'].I_int if igi_result['ready'] else 0.0,
            I_eco=igi_result['NEO'].I_eco if igi_result['ready'] else 0.0,
            GI=grounding_result['NEO'].GI,
            G_pred=grounding_result['NEO'].G_pred,
            G_sym=grounding_result['NEO'].G_sym,
            psi=phi_result['neo_phi'].psi,
            identity=phi_result['neo_phi'].identity,
            otherness=phi_result['neo_phi'].otherness,
            mode=phi_result['neo_mode'],
            tau=neo_tau
        )
        neo_timepoints.append(neo_tp)

        # Registrar EVA
        eva_tp = AgentTimePoint(
            t=t,
            agent='EVA',
            S=eva_sagi['S'],
            learning=eva_sagi['learning'],
            integration=eva_sagi['integration'],
            SAGI=eva_sagi['SAGI'],
            IGI=igi_result['IGI_EVA'] if igi_result['ready'] else 0.0,
            I_int=igi_result['EVA'].I_int if igi_result['ready'] else 0.0,
            I_eco=igi_result['EVA'].I_eco if igi_result['ready'] else 0.0,
            GI=grounding_result['EVA'].GI,
            G_pred=grounding_result['EVA'].G_pred,
            G_sym=grounding_result['EVA'].G_sym,
            psi=phi_result['eva_phi'].psi,
            identity=phi_result['eva_phi'].identity,
            otherness=phi_result['eva_phi'].otherness,
            mode=phi_result['eva_mode'],
            tau=eva_tau
        )
        eva_timepoints.append(eva_tp)

        if verbose and t % (T // 10) == 0:
            print(f"  t={t}:")
            print(f"    NEO: SAGI={neo_sagi['SAGI']:.3f}, τ={neo_tau:.3f}")
            print(f"    EVA: SAGI={eva_sagi['SAGI']:.3f}, τ={eva_tau:.3f}")

    # Comparación final
    final_comparison = compute_final_comparison(neo_timepoints, eva_timepoints)

    return DualExperimentRun(
        seed=seed,
        T=T,
        neo_timepoints=neo_timepoints,
        eva_timepoints=eva_timepoints,
        final_comparison=final_comparison
    )


def compute_final_comparison(neo_tps: List[AgentTimePoint],
                              eva_tps: List[AgentTimePoint]) -> Dict[str, Any]:
    """Computa comparación final entre agentes."""

    n = min(len(neo_tps), len(eva_tps))
    if n < 50:
        return {'ready': False}

    # Promedios finales (últimos 100 pasos)
    window = min(100, n)

    neo_sagi_mean = np.mean([tp.SAGI for tp in neo_tps[-window:]])
    eva_sagi_mean = np.mean([tp.SAGI for tp in eva_tps[-window:]])

    neo_igi_mean = np.mean([tp.IGI for tp in neo_tps[-window:] if tp.IGI > 0])
    eva_igi_mean = np.mean([tp.IGI for tp in eva_tps[-window:] if tp.IGI > 0])

    neo_gi_mean = np.mean([tp.GI for tp in neo_tps[-window:]])
    eva_gi_mean = np.mean([tp.GI for tp in eva_tps[-window:]])

    neo_psi_mean = np.mean([tp.psi for tp in neo_tps[-window:]])
    eva_psi_mean = np.mean([tp.psi for tp in eva_tps[-window:]])

    neo_tau_mean = np.mean([tp.tau for tp in neo_tps[-window:]])
    eva_tau_mean = np.mean([tp.tau for tp in eva_tps[-window:]])

    neo_identity_mean = np.mean([tp.identity for tp in neo_tps[-window:]])
    eva_identity_mean = np.mean([tp.identity for tp in eva_tps[-window:]])

    neo_otherness_mean = np.mean([tp.otherness for tp in neo_tps[-window:]])
    eva_otherness_mean = np.mean([tp.otherness for tp in eva_tps[-window:]])

    return {
        'ready': True,
        'NEO': {
            'SAGI_mean': float(neo_sagi_mean),
            'IGI_mean': float(neo_igi_mean) if not np.isnan(neo_igi_mean) else 0.0,
            'GI_mean': float(neo_gi_mean),
            'psi_mean': float(neo_psi_mean),
            'tau_mean': float(neo_tau_mean),
            'identity_mean': float(neo_identity_mean),
            'otherness_mean': float(neo_otherness_mean)
        },
        'EVA': {
            'SAGI_mean': float(eva_sagi_mean),
            'IGI_mean': float(eva_igi_mean) if not np.isnan(eva_igi_mean) else 0.0,
            'GI_mean': float(eva_gi_mean),
            'psi_mean': float(eva_psi_mean),
            'tau_mean': float(eva_tau_mean),
            'identity_mean': float(eva_identity_mean),
            'otherness_mean': float(eva_otherness_mean)
        },
        'characterization': {
            'neo_more_pragmatic': neo_gi_mean > eva_gi_mean,  # Más grounding
            'eva_more_internal': eva_psi_mean > neo_psi_mean,  # Más fenomenología
            'tau_divergence': abs(neo_tau_mean - eva_tau_mean),
            'personality_neo': 'Pragmático' if neo_gi_mean > eva_gi_mean else 'Fenomenológico',
            'personality_eva': 'Fenomenológico' if eva_psi_mean > neo_psi_mean else 'Pragmática'
        }
    }


def run_longlife_dual(
    n_seeds: int = 3,
    T: int = 500,
    output_dir: str = '/root/NEO_EVA/results/longlife_dual'
) -> Dict[str, Any]:
    """Ejecuta experimentos de vida larga duales."""

    print("=" * 70)
    print("NEO_EVA: EXPERIMENTOS DE VIDA LARGA - DUAL")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")
    print(f"Semillas: {n_seeds}")
    print(f"T por simulación: {T}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    runs = []
    for i in range(n_seeds):
        seed = 42 + i * 17
        print(f"Ejecutando seed {seed} ({i+1}/{n_seeds})...")
        run = run_dual_experiment(seed, T, verbose=True)
        runs.append(run)

        if run.final_comparison.get('ready', False):
            neo = run.final_comparison['NEO']
            eva = run.final_comparison['EVA']
            char = run.final_comparison['characterization']
            print(f"  → NEO: SAGI={neo['SAGI_mean']:.3f}, GI={neo['GI_mean']:.3f}, τ={neo['tau_mean']:.3f}")
            print(f"  → EVA: SAGI={eva['SAGI_mean']:.3f}, GI={eva['GI_mean']:.3f}, τ={eva['tau_mean']:.3f}")
            print(f"  → NEO es {char['personality_neo']}, EVA es {char['personality_eva']}")
        print()

    # Análisis agregado
    print("=" * 70)
    print("ANÁLISIS AGREGADO")
    print("=" * 70)
    print()

    valid_runs = [r for r in runs if r.final_comparison.get('ready', False)]

    if valid_runs:
        neo_sagis = [r.final_comparison['NEO']['SAGI_mean'] for r in valid_runs]
        eva_sagis = [r.final_comparison['EVA']['SAGI_mean'] for r in valid_runs]
        neo_gis = [r.final_comparison['NEO']['GI_mean'] for r in valid_runs]
        eva_gis = [r.final_comparison['EVA']['GI_mean'] for r in valid_runs]
        neo_taus = [r.final_comparison['NEO']['tau_mean'] for r in valid_runs]
        eva_taus = [r.final_comparison['EVA']['tau_mean'] for r in valid_runs]

        print("SAGI:")
        print(f"  NEO: {np.mean(neo_sagis):.4f} ± {np.std(neo_sagis):.4f}")
        print(f"  EVA: {np.mean(eva_sagis):.4f} ± {np.std(eva_sagis):.4f}")
        print()

        print("Grounding Index:")
        print(f"  NEO: {np.mean(neo_gis):.4f} ± {np.std(neo_gis):.4f}")
        print(f"  EVA: {np.mean(eva_gis):.4f} ± {np.std(eva_gis):.4f}")
        print()

        print("Tiempo Subjetivo τ:")
        print(f"  NEO: {np.mean(neo_taus):.4f} ± {np.std(neo_taus):.4f}")
        print(f"  EVA: {np.mean(eva_taus):.4f} ± {np.std(eva_taus):.4f}")
        print()

        # Caracterización final
        neo_more_pragmatic = sum(1 for r in valid_runs
                                  if r.final_comparison['characterization']['neo_more_pragmatic'])
        print(f"NEO más pragmático en {neo_more_pragmatic}/{len(valid_runs)} runs")

    # Guardar resultados
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_seeds': n_seeds,
            'T': T
        },
        'aggregate': {
            'NEO_SAGI_mean': float(np.mean(neo_sagis)) if valid_runs else 0,
            'EVA_SAGI_mean': float(np.mean(eva_sagis)) if valid_runs else 0,
            'NEO_GI_mean': float(np.mean(neo_gis)) if valid_runs else 0,
            'EVA_GI_mean': float(np.mean(eva_gis)) if valid_runs else 0,
            'NEO_tau_mean': float(np.mean(neo_taus)) if valid_runs else 0,
            'EVA_tau_mean': float(np.mean(eva_taus)) if valid_runs else 0
        },
        'runs': [
            {
                'seed': run.seed,
                'final_comparison': run.final_comparison
            }
            for run in runs
        ]
    }

    with open(f'{output_dir}/longlife_dual_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Guardar series temporales del último run
    if runs:
        last_run = runs[-1]
        timeseries = {
            't': [tp.t for tp in last_run.neo_timepoints],
            'NEO_SAGI': [tp.SAGI for tp in last_run.neo_timepoints],
            'EVA_SAGI': [tp.SAGI for tp in last_run.eva_timepoints],
            'NEO_GI': [tp.GI for tp in last_run.neo_timepoints],
            'EVA_GI': [tp.GI for tp in last_run.eva_timepoints],
            'NEO_psi': [tp.psi for tp in last_run.neo_timepoints],
            'EVA_psi': [tp.psi for tp in last_run.eva_timepoints],
            'NEO_tau': [tp.tau for tp in last_run.neo_timepoints],
            'EVA_tau': [tp.tau for tp in last_run.eva_timepoints]
        }

        with open(f'{output_dir}/timeseries_last_run.json', 'w') as f:
            json.dump(timeseries, f, indent=2)

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        if runs:
            last = runs[-1]

            # 1. SAGI temporal
            ax1 = axes[0, 0]
            neo_sagi = [tp.SAGI for tp in last.neo_timepoints]
            eva_sagi = [tp.SAGI for tp in last.eva_timepoints]
            ax1.plot(neo_sagi, 'b-', label='NEO', alpha=0.7)
            ax1.plot(eva_sagi, 'r-', label='EVA', alpha=0.7)
            ax1.set_xlabel('Tiempo')
            ax1.set_ylabel('SAGI')
            ax1.set_title('SAGI por Agente')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. GI temporal
            ax2 = axes[0, 1]
            neo_gi = [tp.GI for tp in last.neo_timepoints]
            eva_gi = [tp.GI for tp in last.eva_timepoints]
            ax2.plot(neo_gi, 'b-', label='NEO', alpha=0.7)
            ax2.plot(eva_gi, 'r-', label='EVA', alpha=0.7)
            ax2.set_xlabel('Tiempo')
            ax2.set_ylabel('GI')
            ax2.set_title('Grounding Index por Agente')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. Ψ temporal
            ax3 = axes[0, 2]
            neo_psi = [tp.psi for tp in last.neo_timepoints]
            eva_psi = [tp.psi for tp in last.eva_timepoints]
            ax3.plot(neo_psi, 'b-', label='NEO', alpha=0.7)
            ax3.plot(eva_psi, 'r-', label='EVA', alpha=0.7)
            ax3.set_xlabel('Tiempo')
            ax3.set_ylabel('Ψ')
            ax3.set_title('Integración Fenomenológica')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. τ temporal
            ax4 = axes[1, 0]
            neo_tau = [tp.tau for tp in last.neo_timepoints]
            eva_tau = [tp.tau for tp in last.eva_timepoints]
            ax4.plot(neo_tau, 'b-', label='NEO τ', alpha=0.7)
            ax4.plot(eva_tau, 'r-', label='EVA τ', alpha=0.7)
            ax4.set_xlabel('Tiempo')
            ax4.set_ylabel('τ')
            ax4.set_title('Tiempo Subjetivo')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # 5. Identity vs Otherness
            ax5 = axes[1, 1]
            neo_id = [tp.identity for tp in last.neo_timepoints]
            eva_oth = [tp.otherness for tp in last.eva_timepoints]
            ax5.plot(neo_id, 'b-', label='NEO identity', alpha=0.7)
            ax5.plot(eva_oth, 'r-', label='EVA otherness', alpha=0.7)
            ax5.set_xlabel('Tiempo')
            ax5.set_ylabel('Valor')
            ax5.set_title('Identity (NEO) vs Otherness (EVA)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # 6. Resumen por seed
            ax6 = axes[1, 2]
            if valid_runs:
                seeds = [r.seed for r in valid_runs]
                neo_finals = [r.final_comparison['NEO']['SAGI_mean'] for r in valid_runs]
                eva_finals = [r.final_comparison['EVA']['SAGI_mean'] for r in valid_runs]
                x = np.arange(len(seeds))
                width = 0.35
                ax6.bar(x - width/2, neo_finals, width, label='NEO', color='blue', alpha=0.7)
                ax6.bar(x + width/2, eva_finals, width, label='EVA', color='red', alpha=0.7)
                ax6.set_xticks(x)
                ax6.set_xticklabels([f's{s}' for s in seeds])
                ax6.set_ylabel('SAGI final')
                ax6.set_title('SAGI Final por Semilla')
                ax6.legend()
                ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs('/root/NEO_EVA/figures', exist_ok=True)
        plt.savefig(f'{output_dir}/longlife_dual_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nResultados guardados en: {output_dir}")
        print(f"Figura: {output_dir}/longlife_dual_visualization.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NEO_EVA Longlife Dual Experiments')
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds')
    parser.add_argument('--T', type=int, default=500, help='Simulation length')
    parser.add_argument('--output', type=str, default='/root/NEO_EVA/results/longlife_dual',
                       help='Output directory')

    args = parser.parse_args()

    run_longlife_dual(
        n_seeds=args.seeds,
        T=args.T,
        output_dir=args.output
    )
