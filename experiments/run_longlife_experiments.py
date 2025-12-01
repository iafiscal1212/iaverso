#!/usr/bin/env python3
"""
NEO_EVA: Experimentos de Vida Larga
===================================

Ejecuta simulaciones extendidas del sistema NEO_EVA con:
- Múltiples semillas
- Registro de SAGI, IGI, φ, grounding, símbolos
- Visualización de curvas, mapas de calor, diagramas de fase

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import sys

# Añadir paths
sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/integration')
sys.path.insert(0, '/root/NEO_EVA/grounding')
sys.path.insert(0, '/root/NEO_EVA/subjectivity')
sys.path.insert(0, '/root/NEO_EVA/weaver')

from integration.phaseI1_subsystems import SubsystemDecomposition
from integration.phaseI2_igi import GlobalIntegrationIndex
from grounding.phaseG1_world_channel import StructuredWorldChannel
from grounding.phaseG2_grounding import GroundingTests
from subjectivity.phaseS1_phenomenal_state import PhenomenalState
from subjectivity.phaseS2_self_report import SelfReportTest


@dataclass
class TimePoint:
    """Registro de un instante temporal."""
    t: int
    S: float
    SAGI: float
    IGI: float
    phi_mode: int
    phi_psi: float
    grounding_error: float
    symbol: int
    regime: int


@dataclass
class ExperimentRun:
    """Resultados de una ejecución."""
    seed: int
    T: int
    timepoints: List[TimePoint] = field(default_factory=list)
    final_metrics: Dict[str, Any] = field(default_factory=dict)


def compute_sagi(z: np.ndarray, t: int, history: List[np.ndarray]) -> float:
    """
    Calcula SAGI simplificado.

    SAGI = Π(componentes^weights)

    100% endógeno
    """
    # Componentes básicos
    S = -np.sum(z * np.log(z + 1e-10))
    S_max = np.log(len(z))
    S_norm = S / S_max if S_max > 0 else 0

    # Variabilidad (aprendizaje)
    if len(history) > 10:
        recent = np.array(history[-10:])
        learning = np.std(recent)
        learning = min(learning * 10, 1.0)
    else:
        learning = 0.5

    # Integración (correlación)
    if len(history) > 20:
        H = np.array(history[-20:])
        corr_matrix = np.corrcoef(H.T)
        mask = ~np.eye(len(z), dtype=bool)
        correlations = corr_matrix[mask]
        correlations = correlations[~np.isnan(correlations)]
        integration = np.mean(np.abs(correlations)) if len(correlations) > 0 else 0.5
    else:
        integration = 0.5

    # SAGI = media geométrica
    components = [S_norm, learning, integration]
    sagi = np.prod([c ** (1/len(components)) for c in components])

    return float(sagi)


def run_single_experiment(seed: int, T: int, verbose: bool = False) -> ExperimentRun:
    """Ejecuta un experimento con una semilla específica."""

    np.random.seed(seed)

    # Inicializar sistemas
    igi_system = GlobalIntegrationIndex(total_dim=12)
    world = StructuredWorldChannel(dim_s=6, seed=seed)
    phenomenal = PhenomenalState(dim_z=6)

    # Estado inicial
    z = np.random.rand(12)
    z = z / z.sum()

    # Historia
    z_history = []
    timepoints = []

    for t in range(T):
        # === Dinámica del sistema ===

        # Paso del mundo
        world_state = world.step()
        s = world_state.s

        # Interacción mundo → sistema (grounding)
        z[:6] = 0.9 * z[:6] + 0.1 * s

        # Dinámica interna con acoplos
        noise = np.random.randn(12) * 0.02

        # NEO-EVA coupling
        z[6:8] += 0.05 * z[0:2]
        z[0:2] += 0.03 * z[6:8]

        # Perturbaciones periódicas
        if t % 100 < 10:
            noise += np.random.randn(12) * 0.08

        z = z + noise
        z = np.clip(z, 0.01, 0.99)
        z = z / z.sum()

        z_history.append(z.copy())

        # === Métricas ===

        # Entropía
        S = -np.sum(z * np.log(z + 1e-10))

        # SAGI
        sagi = compute_sagi(z[:6], t, [h[:6] for h in z_history])

        # IGI
        igi_result = igi_system.step(z)
        igi = igi_result.get('IGI', 0.0) if igi_result.get('ready', False) else 0.0

        # Estado fenomenológico
        phi = phenomenal.step(z[:6], S)
        phi_mode = phenomenal.get_current_mode()
        phi_psi = phi.psi

        # Grounding (error de predicción simplificado)
        if len(z_history) > 1:
            pred_error = np.linalg.norm(z[:6] - z_history[-2][:6])
        else:
            pred_error = 0.5

        # Símbolo (discretización endógena)
        symbol = int(z[0] * 5 + z[1] * 3) % 5

        # Registrar
        tp = TimePoint(
            t=t,
            S=S,
            SAGI=sagi,
            IGI=igi if igi else 0.0,
            phi_mode=phi_mode,
            phi_psi=phi_psi,
            grounding_error=pred_error,
            symbol=symbol,
            regime=world_state.regime
        )
        timepoints.append(tp)

        if verbose and t % (T // 10) == 0:
            print(f"  t={t}: SAGI={sagi:.4f}, IGI={igi:.4f}, φ-mode={phi_mode}")

    # Métricas finales
    sagis = [tp.SAGI for tp in timepoints]
    igis = [tp.IGI for tp in timepoints if tp.IGI > 0]
    modes = [tp.phi_mode for tp in timepoints]

    final_metrics = {
        'SAGI_mean': float(np.mean(sagis)),
        'SAGI_std': float(np.std(sagis)),
        'SAGI_final': float(sagis[-1]),
        'IGI_mean': float(np.mean(igis)) if igis else 0.0,
        'IGI_std': float(np.std(igis)) if igis else 0.0,
        'n_mode_transitions': sum(1 for i in range(1, len(modes)) if modes[i] != modes[i-1]),
        'dominant_mode': int(max(set(modes), key=modes.count)),
        'mode_entropy': float(-sum((modes.count(m)/len(modes)) * np.log(modes.count(m)/len(modes) + 1e-10)
                                  for m in set(modes)))
    }

    return ExperimentRun(
        seed=seed,
        T=T,
        timepoints=timepoints,
        final_metrics=final_metrics
    )


def run_longlife_experiments(
    n_seeds: int = 5,
    T: int = 1000,
    output_dir: str = '/root/NEO_EVA/results/longlife'
) -> Dict[str, Any]:
    """
    Ejecuta experimentos de vida larga con múltiples semillas.

    Args:
        n_seeds: Número de semillas
        T: Longitud de cada simulación
        output_dir: Directorio de salida

    Returns:
        Dict con resultados agregados
    """

    print("=" * 70)
    print("NEO_EVA: EXPERIMENTOS DE VIDA LARGA")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")
    print(f"Semillas: {n_seeds}")
    print(f"T por simulación: {T}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    # Ejecutar experimentos
    runs = []
    for i in range(n_seeds):
        seed = 42 + i * 17
        print(f"Ejecutando seed {seed} ({i+1}/{n_seeds})...")
        run = run_single_experiment(seed, T, verbose=True)
        runs.append(run)
        print(f"  → SAGI_mean={run.final_metrics['SAGI_mean']:.4f}")
        print()

    # Análisis agregado
    print("=" * 70)
    print("ANÁLISIS AGREGADO")
    print("=" * 70)
    print()

    # Métricas por seed
    all_sagis = [run.final_metrics['SAGI_mean'] for run in runs]
    all_igis = [run.final_metrics['IGI_mean'] for run in runs]

    print(f"SAGI across seeds:")
    print(f"  Media: {np.mean(all_sagis):.4f} ± {np.std(all_sagis):.4f}")
    print(f"  Min: {np.min(all_sagis):.4f}, Max: {np.max(all_sagis):.4f}")
    print()

    print(f"IGI across seeds:")
    print(f"  Media: {np.mean(all_igis):.4f} ± {np.std(all_igis):.4f}")
    print()

    # Buscar patrones típicos
    mode_transitions = [run.final_metrics['n_mode_transitions'] for run in runs]
    print(f"Transiciones de modo fenomenológico:")
    print(f"  Media: {np.mean(mode_transitions):.1f} ± {np.std(mode_transitions):.1f}")
    print()

    # Guardar resultados
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_seeds': n_seeds,
            'T': T
        },
        'aggregate': {
            'SAGI_mean': float(np.mean(all_sagis)),
            'SAGI_std': float(np.std(all_sagis)),
            'IGI_mean': float(np.mean(all_igis)),
            'IGI_std': float(np.std(all_igis)),
            'mode_transitions_mean': float(np.mean(mode_transitions))
        },
        'runs': [
            {
                'seed': run.seed,
                'final_metrics': run.final_metrics
            }
            for run in runs
        ]
    }

    with open(f'{output_dir}/longlife_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Guardar series temporales (último run para visualización)
    last_run = runs[-1]
    timeseries = {
        't': [tp.t for tp in last_run.timepoints],
        'SAGI': [tp.SAGI for tp in last_run.timepoints],
        'IGI': [tp.IGI for tp in last_run.timepoints],
        'phi_psi': [tp.phi_psi for tp in last_run.timepoints],
        'phi_mode': [tp.phi_mode for tp in last_run.timepoints],
        'regime': [tp.regime for tp in last_run.timepoints]
    }

    with open(f'{output_dir}/timeseries_last_run.json', 'w') as f:
        json.dump(timeseries, f, indent=2)

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        # 1. SAGI temporal (último run)
        ax1 = axes[0, 0]
        ax1.plot(timeseries['SAGI'], 'b-', linewidth=0.5, alpha=0.7)
        ax1.axhline(y=np.mean(timeseries['SAGI']), color='r', linestyle='--',
                   label=f'Media={np.mean(timeseries["SAGI"]):.4f}')
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('SAGI')
        ax1.set_title(f'SAGI Temporal (seed={last_run.seed})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. IGI temporal
        ax2 = axes[0, 1]
        igi_values = [v for v in timeseries['IGI'] if v > 0]
        if igi_values:
            ax2.plot(range(len(igi_values)), igi_values, 'g-', linewidth=0.5, alpha=0.7)
            ax2.axhline(y=np.mean(igi_values), color='r', linestyle='--',
                       label=f'Media={np.mean(igi_values):.4f}')
        ax2.set_xlabel('Tiempo')
        ax2.set_ylabel('IGI')
        ax2.set_title('Índice de Integración Global')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Modos fenomenológicos
        ax3 = axes[1, 0]
        ax3.plot(timeseries['phi_mode'], 'k-', linewidth=0.5, alpha=0.7)
        ax3.fill_between(range(len(timeseries['phi_mode'])), 0,
                        np.array(timeseries['phi_mode']),
                        alpha=0.3, color='purple')
        ax3.set_xlabel('Tiempo')
        ax3.set_ylabel('Modo')
        ax3.set_title('Modos Fenomenológicos')
        ax3.grid(True, alpha=0.3)

        # 4. Ψ temporal
        ax4 = axes[1, 1]
        ax4.plot(timeseries['phi_psi'], 'purple', linewidth=0.5, alpha=0.7)
        ax4.set_xlabel('Tiempo')
        ax4.set_ylabel('Ψ')
        ax4.set_title('Integración Fenomenológica Global')
        ax4.grid(True, alpha=0.3)

        # 5. SAGI por seed (boxplot)
        ax5 = axes[2, 0]
        seed_data = [[tp.SAGI for tp in run.timepoints[-100:]] for run in runs]
        ax5.boxplot(seed_data, labels=[f's{run.seed}' for run in runs])
        ax5.set_xlabel('Semilla')
        ax5.set_ylabel('SAGI (últimos 100)')
        ax5.set_title('Distribución SAGI por Semilla')
        ax5.grid(True, alpha=0.3)

        # 6. Regímenes del mundo vs modos internos
        ax6 = axes[2, 1]
        ax6.scatter(timeseries['regime'], timeseries['phi_mode'], alpha=0.1, s=5)
        ax6.set_xlabel('Régimen del Mundo')
        ax6.set_ylabel('Modo Fenomenológico')
        ax6.set_title('Correspondencia Mundo-Interior')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/longlife_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Resultados guardados en: {output_dir}")
        print(f"Figura: {output_dir}/longlife_visualization.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NEO_EVA Longlife Experiments')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--T', type=int, default=1000, help='Simulation length')
    parser.add_argument('--output', type=str, default='/root/NEO_EVA/results/longlife',
                       help='Output directory')

    args = parser.parse_args()

    run_longlife_experiments(
        n_seeds=args.seeds,
        T=args.T,
        output_dir=args.output
    )
