#!/usr/bin/env python3
"""
FASE B: Análisis Cuántico del Sesgo Colectivo
==============================================

Análisis avanzado usando datos de simulación de 12h:
1. Divergencia vs Sincronía
2. Especialización Complementaria
3. Acoplamiento Asimétrico
4. Análisis Q-Field Avanzado
5. Creatividad en Estado Estable
6. Identificación de Sesgo Colectivo
7. Comparativa con Modelo Nulo

100% observacional - NO modifica comportamiento de agentes.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats
from scipy.signal import correlate
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings('ignore')
sys.path.insert(0, '/root/NEO_EVA')

from core.agents import NEO, EVA, DualAgentSystem
from omega.q_field import QField
from l_field.l_field import LField
from l_field.collective_bias import CollectiveBias

# Output directories
FIG_DIR = '/root/NEO_EVA/figuras/sesgo_colectivo_12h'
LOG_DIR = '/root/NEO_EVA/logs/sesgo_colectivo_12h'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


@dataclass
class SimulationData:
    """Data collected during simulation."""
    # Time series
    t: List[int] = field(default_factory=list)

    # NEO metrics
    CE_NEO: List[float] = field(default_factory=list)
    value_NEO: List[float] = field(default_factory=list)
    surprise_NEO: List[float] = field(default_factory=list)
    entropy_NEO: List[float] = field(default_factory=list)
    specialization_NEO: List[float] = field(default_factory=list)
    psi_norm_NEO: List[float] = field(default_factory=list)

    # EVA metrics
    CE_EVA: List[float] = field(default_factory=list)
    value_EVA: List[float] = field(default_factory=list)
    surprise_EVA: List[float] = field(default_factory=list)
    entropy_EVA: List[float] = field(default_factory=list)
    specialization_EVA: List[float] = field(default_factory=list)
    psi_norm_EVA: List[float] = field(default_factory=list)

    # Interaction metrics
    coupling_NEO_EVA: List[float] = field(default_factory=list)
    coupling_EVA_NEO: List[float] = field(default_factory=list)
    divergence: List[float] = field(default_factory=list)

    # Collective metrics
    Q_coherence: List[float] = field(default_factory=list)
    LSI: List[float] = field(default_factory=list)
    polarization: List[float] = field(default_factory=list)

    # Genesis
    ideas_cumulative: List[int] = field(default_factory=list)
    objects_cumulative: List[int] = field(default_factory=list)


def run_simulation(n_steps: int = 6000, seed: int = 42, coupling_enabled: bool = True) -> SimulationData:
    """Run NEO-EVA simulation and collect data."""
    rng = np.random.default_rng(seed)

    # Initialize
    dual_system = DualAgentSystem(dim_visible=6, dim_hidden=6)
    q_field = QField()
    l_field = LField()
    collective_bias = CollectiveBias()

    data = SimulationData()
    total_ideas = 0
    total_objects = 0

    # Disable coupling if requested
    if not coupling_enabled:
        dual_system.coupling_neo_to_eva = 0.0
        dual_system.coupling_eva_to_neo = 0.0

    print(f"Running simulation: {n_steps} steps, coupling={'ON' if coupling_enabled else 'OFF'}")

    for t in range(n_steps):
        # Generate stimulus
        stimulus = rng.uniform(0, 1, 6)

        # Disable coupling update if requested
        original_coupling_neo_eva = dual_system.coupling_neo_to_eva
        original_coupling_eva_neo = dual_system.coupling_eva_to_neo

        # Agent dynamics
        result = dual_system.step(stimulus)
        neo_response = result['neo_response']
        eva_response = result['eva_response']

        if not coupling_enabled:
            dual_system.coupling_neo_to_eva = 0.0
            dual_system.coupling_eva_to_neo = 0.0

        # Get states
        neo_state = dual_system.neo.get_state()
        eva_state = dual_system.eva.get_state()

        # Record time
        data.t.append(t)

        # NEO metrics
        CE_neo = 1.0 / (1.0 + neo_response.surprise)
        data.CE_NEO.append(CE_neo)
        data.value_NEO.append(neo_response.value)
        data.surprise_NEO.append(neo_response.surprise)
        data.entropy_NEO.append(neo_state.S)
        data.specialization_NEO.append(dual_system.neo.specialization)
        data.psi_norm_NEO.append(np.linalg.norm(neo_state.z_visible))

        # EVA metrics
        CE_eva = 1.0 / (1.0 + eva_response.surprise)
        data.CE_EVA.append(CE_eva)
        data.value_EVA.append(eva_response.value)
        data.surprise_EVA.append(eva_response.surprise)
        data.entropy_EVA.append(eva_state.S)
        data.specialization_EVA.append(dual_system.eva.specialization)
        data.psi_norm_EVA.append(np.linalg.norm(eva_state.z_visible))

        # Interaction metrics
        data.coupling_NEO_EVA.append(dual_system.coupling_neo_to_eva)
        data.coupling_EVA_NEO.append(dual_system.coupling_eva_to_neo)

        # Divergence
        div = dual_system.get_divergence()
        data.divergence.append(div['total_divergence'])

        # Q-Field
        q_field.register_state('NEO', neo_state.z_visible)
        q_field.register_state('EVA', eva_state.z_visible)
        q_stats = q_field.get_statistics()
        data.Q_coherence.append(q_stats.get('mean_coherence', 0.5))

        # L-Field
        states = {'NEO': neo_state.z_visible, 'EVA': eva_state.z_visible}
        identities = {'NEO': neo_state.z_hidden, 'EVA': eva_state.z_hidden}
        l_field.observe(states, identities)
        l_stats = l_field.get_statistics()
        data.LSI.append(l_stats.get('mean_lsi', 0.5))

        # Collective bias
        collective_bias.observe(states, identities)
        bias_stats = collective_bias.get_statistics()
        data.polarization.append(bias_stats.get('mean_polarization', 0.0))

        # Genesis (ideas)
        if len(dual_system.neo.surprise_history) > 10:
            threshold = np.percentile(dual_system.neo.surprise_history[-50:], 75)
            if dual_system.neo.surprise_history[-1] > threshold:
                total_ideas += 1
                total_objects += 1
        if len(dual_system.eva.surprise_history) > 10:
            threshold = np.percentile(dual_system.eva.surprise_history[-50:], 75)
            if dual_system.eva.surprise_history[-1] > threshold:
                total_ideas += 1
                total_objects += 1

        data.ideas_cumulative.append(total_ideas)
        data.objects_cumulative.append(total_objects)

        if (t + 1) % 1000 == 0:
            print(f"  Step {t+1}/{n_steps}")

    return data


def compute_cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cross-correlation between two signals."""
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    y = (y - np.mean(y)) / (np.std(y) + 1e-12)

    corr = correlate(x, y, mode='full')
    corr = corr / len(x)

    mid = len(corr) // 2
    lags = np.arange(-max_lag, max_lag + 1)
    corr = corr[mid - max_lag:mid + max_lag + 1]

    return lags, corr


def compute_granger_structural(x: np.ndarray, y: np.ndarray, lag: int = 10) -> Dict[str, float]:
    """
    Compute structural Granger-like causality (non-probabilistic).

    Tests if past values of X help predict Y beyond Y's own history.
    """
    n = len(x)
    if n < lag * 3:
        return {'x_causes_y': 0.0, 'y_causes_x': 0.0}

    # Build lagged matrices
    X_lag = np.column_stack([x[lag-i-1:n-i-1] for i in range(lag)])
    Y_lag = np.column_stack([y[lag-i-1:n-i-1] for i in range(lag)])

    Y_target = y[lag:]
    X_target = x[lag:]

    # Model 1: Y ~ Y_lag
    try:
        coeffs_y = np.linalg.lstsq(Y_lag, Y_target, rcond=None)[0]
        pred_y_only = Y_lag @ coeffs_y
        ss_y_only = np.sum((Y_target - pred_y_only) ** 2)
    except:
        ss_y_only = np.var(Y_target) * len(Y_target)

    # Model 2: Y ~ Y_lag + X_lag
    try:
        XY_lag = np.column_stack([Y_lag, X_lag])
        coeffs_xy = np.linalg.lstsq(XY_lag, Y_target, rcond=None)[0]
        pred_xy = XY_lag @ coeffs_xy
        ss_xy = np.sum((Y_target - pred_xy) ** 2)
    except:
        ss_xy = ss_y_only

    # X causes Y: improvement from adding X
    x_causes_y = max(0, (ss_y_only - ss_xy) / (ss_y_only + 1e-12))

    # Reverse: Y causes X
    try:
        coeffs_x = np.linalg.lstsq(X_lag, X_target, rcond=None)[0]
        pred_x_only = X_lag @ coeffs_x
        ss_x_only = np.sum((X_target - pred_x_only) ** 2)
    except:
        ss_x_only = np.var(X_target) * len(X_target)

    try:
        YX_lag = np.column_stack([X_lag, Y_lag])
        coeffs_yx = np.linalg.lstsq(YX_lag, X_target, rcond=None)[0]
        pred_yx = YX_lag @ coeffs_yx
        ss_yx = np.sum((X_target - pred_yx) ** 2)
    except:
        ss_yx = ss_x_only

    y_causes_x = max(0, (ss_x_only - ss_yx) / (ss_x_only + 1e-12))

    return {
        'x_causes_y': x_causes_y,
        'y_causes_x': y_causes_x,
        'dominant': 'X→Y' if x_causes_y > y_causes_x else 'Y→X'
    }


def detect_bifurcation(spec_neo: np.ndarray, spec_eva: np.ndarray, window: int = 100) -> int:
    """Detect bifurcation point where specializations diverge."""
    diff = np.abs(spec_neo - spec_eva)

    # Smooth
    diff_smooth = uniform_filter1d(diff, window)

    # Find point of maximum acceleration
    grad = np.gradient(diff_smooth)
    grad2 = np.gradient(grad)

    # Bifurcation = point where 2nd derivative peaks
    bifurcation_idx = np.argmax(grad2[:len(grad2)//2]) if len(grad2) > 0 else 0

    return bifurcation_idx


def plot_divergence_sync(data: SimulationData, filename: str):
    """Plot divergence vs synchrony over time."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    t = np.array(data.t)

    # Divergence
    axes[0].plot(t, data.divergence, 'b-', linewidth=0.5, alpha=0.5)
    div_smooth = uniform_filter1d(data.divergence, 100)
    axes[0].plot(t, div_smooth, 'b-', linewidth=2, label='Divergence (smoothed)')
    axes[0].set_ylabel('Divergence')
    axes[0].set_title('Divergencia vs Sincronía Temporal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # LSI
    axes[1].plot(t, data.LSI, 'g-', linewidth=0.5, alpha=0.5)
    lsi_smooth = uniform_filter1d(data.LSI, 100)
    axes[1].plot(t, lsi_smooth, 'g-', linewidth=2, label='LSI (smoothed)')
    axes[1].set_ylabel('LSI')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # CE comparison
    axes[2].plot(t, data.CE_NEO, 'r-', linewidth=0.5, alpha=0.3, label='CE_NEO')
    axes[2].plot(t, data.CE_EVA, 'b-', linewidth=0.5, alpha=0.3, label='CE_EVA')
    ce_neo_smooth = uniform_filter1d(data.CE_NEO, 100)
    ce_eva_smooth = uniform_filter1d(data.CE_EVA, 100)
    axes[2].plot(t, ce_neo_smooth, 'r-', linewidth=2, label='CE_NEO (smooth)')
    axes[2].plot(t, ce_eva_smooth, 'b-', linewidth=2, label='CE_EVA (smooth)')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('CE')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_specialization_bifurcation(data: SimulationData, filename: str):
    """Plot specialization curves with bifurcation detection."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    t = np.array(data.t)
    spec_neo = np.array(data.specialization_NEO)
    spec_eva = np.array(data.specialization_EVA)

    # Detect bifurcation
    bifurcation_idx = detect_bifurcation(spec_neo, spec_eva)

    # Specialization curves
    axes[0].plot(t, spec_neo, 'r-', linewidth=2, label='NEO (Compression)')
    axes[0].plot(t, spec_eva, 'b-', linewidth=2, label='EVA (Exchange)')
    axes[0].axvline(x=bifurcation_idx, color='orange', linestyle='--', linewidth=2,
                   label=f'Bifurcation @ t={bifurcation_idx}')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Specialization')
    axes[0].set_title('Especialización Complementaria - Bifurcación')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.1)

    # Difference plot
    diff = np.abs(spec_neo - spec_eva)
    axes[1].fill_between(t, 0, diff, alpha=0.5, color='purple', label='|NEO - EVA|')
    axes[1].axvline(x=bifurcation_idx, color='orange', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Specialization Difference')
    axes[1].set_title('Divergencia de Especialización')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    return bifurcation_idx


def plot_asymmetric_coupling(data: SimulationData, filename: str):
    """Plot asymmetric coupling analysis."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    t = np.array(data.t)

    # Coupling over time
    axes[0].plot(t, data.coupling_NEO_EVA, 'r-', linewidth=1.5, label='NEO → EVA')
    axes[0].plot(t, data.coupling_EVA_NEO, 'b-', linewidth=1.5, label='EVA → NEO')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Coupling Strength')
    axes[0].set_title('Acoplamiento Asimétrico NEO-EVA')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Asymmetry ratio
    neo_eva = np.array(data.coupling_NEO_EVA)
    eva_neo = np.array(data.coupling_EVA_NEO)
    asymmetry = (neo_eva - eva_neo) / (neo_eva + eva_neo + 1e-12)

    axes[1].plot(t, asymmetry, 'purple', linewidth=1)
    asymmetry_smooth = uniform_filter1d(asymmetry, 100)
    axes[1].plot(t, asymmetry_smooth, 'purple', linewidth=2, label='Asymmetry (smoothed)')
    axes[1].axhline(y=0, color='gray', linestyle='--')
    axes[1].fill_between(t, 0, asymmetry, where=(asymmetry > 0), alpha=0.3, color='red',
                        label='NEO dominates')
    axes[1].fill_between(t, 0, asymmetry, where=(asymmetry < 0), alpha=0.3, color='blue',
                        label='EVA dominates')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Asymmetry Ratio')
    axes[1].set_title('Ratio de Asimetría del Acoplamiento')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_qfield_entanglement(data: SimulationData, filename: str):
    """Plot Q-Field analysis with entanglement proxy."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    t = np.array(data.t)

    # Q coherence
    axes[0].plot(t, data.Q_coherence, 'b-', linewidth=0.5, alpha=0.5)
    q_smooth = uniform_filter1d(data.Q_coherence, 100)
    axes[0].plot(t, q_smooth, 'b-', linewidth=2, label='Q Coherence (smoothed)')
    axes[0].set_ylabel('Q Coherence')
    axes[0].set_title('Q-Field Coherence y Entanglement')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Psi norms
    axes[1].plot(t, data.psi_norm_NEO, 'r-', linewidth=0.5, alpha=0.5, label='ψ_NEO')
    axes[1].plot(t, data.psi_norm_EVA, 'b-', linewidth=0.5, alpha=0.5, label='ψ_EVA')
    psi_neo_smooth = uniform_filter1d(data.psi_norm_NEO, 100)
    psi_eva_smooth = uniform_filter1d(data.psi_norm_EVA, 100)
    axes[1].plot(t, psi_neo_smooth, 'r-', linewidth=2)
    axes[1].plot(t, psi_eva_smooth, 'b-', linewidth=2)
    axes[1].set_ylabel('|ψ|')
    axes[1].set_title('Norma de Estado Cuántico')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Entanglement proxy (rolling correlation)
    window = 100
    entanglement = []
    for i in range(window, len(t)):
        corr = np.corrcoef(
            data.psi_norm_NEO[i-window:i],
            data.psi_norm_EVA[i-window:i]
        )[0, 1]
        entanglement.append(corr if not np.isnan(corr) else 0)

    axes[2].plot(range(window, len(t)), entanglement, 'purple', linewidth=1)
    ent_smooth = uniform_filter1d(entanglement, 50)
    axes[2].plot(range(window, len(t)), ent_smooth, 'purple', linewidth=2,
                label='Entanglement Proxy')
    axes[2].axhline(y=0, color='gray', linestyle='--')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Correlation')
    axes[2].set_title('Proxy de Entanglement: corr(ψ_NEO, ψ_EVA)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_genesis_creativity(data: SimulationData, filename: str):
    """Plot Genesis creativity rate analysis."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    t = np.array(data.t)
    ideas = np.array(data.ideas_cumulative)
    objects = np.array(data.objects_cumulative)

    # Cumulative
    axes[0].plot(t, ideas, 'g-', linewidth=2, label='Ideas (cumulative)')
    axes[0].plot(t, objects, 'b--', linewidth=2, label='Objects (cumulative)')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Genesis: Creatividad Acumulada')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Rate (ideas per hour equivalent)
    # 500 steps = 1 hour
    hour_steps = 500
    hours = len(t) // hour_steps

    ideas_per_hour = []
    for h in range(hours):
        start = h * hour_steps
        end = (h + 1) * hour_steps
        ideas_in_hour = ideas[min(end, len(ideas)-1)] - ideas[start]
        ideas_per_hour.append(ideas_in_hour)

    axes[1].bar(range(1, hours + 1), ideas_per_hour, color='green', alpha=0.7)
    axes[1].axhline(y=np.mean(ideas_per_hour), color='red', linestyle='--',
                   label=f'Mean: {np.mean(ideas_per_hour):.1f}/hour')
    axes[1].set_xlabel('Hour')
    axes[1].set_ylabel('Ideas')
    axes[1].set_title('Tasa de Creatividad por Hora')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    return ideas_per_hour


def plot_collective_bias_origin(data: SimulationData, filename: str) -> Dict:
    """Analyze and plot the origin of collective bias."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    value_neo = np.array(data.value_NEO)
    value_eva = np.array(data.value_EVA)
    surprise_neo = np.array(data.surprise_NEO)
    surprise_eva = np.array(data.surprise_EVA)

    # Cross-correlation: Value
    lags, corr_value = compute_cross_correlation(value_neo, value_eva, max_lag=50)
    axes[0, 0].plot(lags, corr_value, 'b-', linewidth=2)
    axes[0, 0].axvline(x=0, color='gray', linestyle='--')
    peak_lag_value = lags[np.argmax(np.abs(corr_value))]
    axes[0, 0].axvline(x=peak_lag_value, color='red', linestyle='--',
                      label=f'Peak @ lag={peak_lag_value}')
    axes[0, 0].set_xlabel('Lag')
    axes[0, 0].set_ylabel('Cross-correlation')
    axes[0, 0].set_title('Cross-Correlation: Value NEO vs EVA')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Cross-correlation: Surprise
    lags, corr_surprise = compute_cross_correlation(surprise_neo, surprise_eva, max_lag=50)
    axes[0, 1].plot(lags, corr_surprise, 'r-', linewidth=2)
    axes[0, 1].axvline(x=0, color='gray', linestyle='--')
    peak_lag_surprise = lags[np.argmax(np.abs(corr_surprise))]
    axes[0, 1].axvline(x=peak_lag_surprise, color='orange', linestyle='--',
                      label=f'Peak @ lag={peak_lag_surprise}')
    axes[0, 1].set_xlabel('Lag')
    axes[0, 1].set_ylabel('Cross-correlation')
    axes[0, 1].set_title('Cross-Correlation: Surprise NEO vs EVA')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Granger causality
    granger_value = compute_granger_structural(value_neo, value_eva, lag=20)
    granger_surprise = compute_granger_structural(surprise_neo, surprise_eva, lag=20)

    # Bar plot of causality
    labels = ['Value', 'Surprise']
    neo_causes = [granger_value['x_causes_y'], granger_surprise['x_causes_y']]
    eva_causes = [granger_value['y_causes_x'], granger_surprise['y_causes_x']]

    x = np.arange(len(labels))
    width = 0.35

    axes[1, 0].bar(x - width/2, neo_causes, width, label='NEO → EVA', color='red', alpha=0.7)
    axes[1, 0].bar(x + width/2, eva_causes, width, label='EVA → NEO', color='blue', alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_ylabel('Causal Strength')
    axes[1, 0].set_title('Causalidad Estructural (Granger-like)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Summary interpretation
    axes[1, 1].axis('off')

    # Determine dominant causality
    total_neo_causes = sum(neo_causes)
    total_eva_causes = sum(eva_causes)

    if total_neo_causes > total_eva_causes * 1.2:
        conclusion = "NEO DOMINA → EVA responde\nNEO comprime y EVA se adapta"
        color = 'red'
    elif total_eva_causes > total_neo_causes * 1.2:
        conclusion = "EVA DOMINA → NEO responde\nEVA explora y NEO estabiliza"
        color = 'blue'
    else:
        conclusion = "SESGO BIDIRECCIONAL\nAmbos agentes se influencian mutuamente"
        color = 'purple'

    text = f"""
ANÁLISIS DE SESGO COLECTIVO
{'='*40}

Cross-correlation Value:
  Peak lag: {peak_lag_value}
  {'NEO lidera' if peak_lag_value > 0 else 'EVA lidera' if peak_lag_value < 0 else 'Simultáneo'}

Cross-correlation Surprise:
  Peak lag: {peak_lag_surprise}
  {'NEO lidera' if peak_lag_surprise > 0 else 'EVA lidera' if peak_lag_surprise < 0 else 'Simultáneo'}

Causalidad Estructural:
  NEO→EVA (Value): {granger_value['x_causes_y']:.4f}
  EVA→NEO (Value): {granger_value['y_causes_x']:.4f}
  NEO→EVA (Surprise): {granger_surprise['x_causes_y']:.4f}
  EVA→NEO (Surprise): {granger_surprise['y_causes_x']:.4f}

CONCLUSIÓN:
{conclusion}
"""
    axes[1, 1].text(0.05, 0.95, text, transform=axes[1, 1].transAxes,
                   fontfamily='monospace', fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    return {
        'peak_lag_value': peak_lag_value,
        'peak_lag_surprise': peak_lag_surprise,
        'granger_value': granger_value,
        'granger_surprise': granger_surprise,
        'conclusion': conclusion
    }


def plot_real_vs_null(data_real: SimulationData, data_null: SimulationData, filename: str):
    """Plot comparison between real and null model."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t_real = np.array(data_real.t)
    t_null = np.array(data_null.t)

    # Value comparison
    axes[0, 0].plot(t_real, uniform_filter1d(data_real.value_NEO, 100), 'r-',
                   linewidth=2, label='NEO (real)')
    axes[0, 0].plot(t_real, uniform_filter1d(data_real.value_EVA, 100), 'b-',
                   linewidth=2, label='EVA (real)')
    axes[0, 0].plot(t_null, uniform_filter1d(data_null.value_NEO, 100), 'r--',
                   linewidth=2, alpha=0.5, label='NEO (null)')
    axes[0, 0].plot(t_null, uniform_filter1d(data_null.value_EVA, 100), 'b--',
                   linewidth=2, alpha=0.5, label='EVA (null)')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Value: Real vs Null')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Specialization comparison
    axes[0, 1].plot(t_real, data_real.specialization_NEO, 'r-',
                   linewidth=2, label='NEO (real)')
    axes[0, 1].plot(t_real, data_real.specialization_EVA, 'b-',
                   linewidth=2, label='EVA (real)')
    axes[0, 1].plot(t_null, data_null.specialization_NEO, 'r--',
                   linewidth=2, alpha=0.5, label='NEO (null)')
    axes[0, 1].plot(t_null, data_null.specialization_EVA, 'b--',
                   linewidth=2, alpha=0.5, label='EVA (null)')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Specialization')
    axes[0, 1].set_title('Specialization: Real vs Null')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Field comparison
    axes[1, 0].plot(t_real, uniform_filter1d(data_real.Q_coherence, 100), 'g-',
                   linewidth=2, label='Real')
    axes[1, 0].plot(t_null, uniform_filter1d(data_null.Q_coherence, 100), 'g--',
                   linewidth=2, alpha=0.5, label='Null')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Q Coherence')
    axes[1, 0].set_title('Q-Field: Real vs Null')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Divergence comparison
    axes[1, 1].plot(t_real, uniform_filter1d(data_real.divergence, 100), 'purple',
                   linewidth=2, label='Real')
    axes[1, 1].plot(t_null, uniform_filter1d(data_null.divergence, 100), 'purple',
                   linewidth=2, alpha=0.5, linestyle='--', label='Null')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Divergence')
    axes[1, 1].set_title('Divergence: Real vs Null')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_csv_outputs(data: SimulationData, causality_results: Dict):
    """Save CSV outputs."""
    # Main time series
    df_ts = pd.DataFrame({
        't': data.t,
        'CE_NEO': data.CE_NEO,
        'CE_EVA': data.CE_EVA,
        'value_NEO': data.value_NEO,
        'value_EVA': data.value_EVA,
        'surprise_NEO': data.surprise_NEO,
        'surprise_EVA': data.surprise_EVA,
        'specialization_NEO': data.specialization_NEO,
        'specialization_EVA': data.specialization_EVA,
        'coupling_NEO_EVA': data.coupling_NEO_EVA,
        'coupling_EVA_NEO': data.coupling_EVA_NEO,
        'divergence': data.divergence,
        'Q_coherence': data.Q_coherence,
        'LSI': data.LSI,
        'ideas': data.ideas_cumulative
    })
    df_ts.to_csv(f'{LOG_DIR}/time_series_12h.csv', index=False)

    # Causality results
    df_causality = pd.DataFrame({
        'metric': ['value', 'surprise'],
        'NEO_causes_EVA': [
            causality_results['granger_value']['x_causes_y'],
            causality_results['granger_surprise']['x_causes_y']
        ],
        'EVA_causes_NEO': [
            causality_results['granger_value']['y_causes_x'],
            causality_results['granger_surprise']['y_causes_x']
        ],
        'dominant': [
            causality_results['granger_value']['dominant'],
            causality_results['granger_surprise']['dominant']
        ]
    })
    df_causality.to_csv(f'{LOG_DIR}/causality_analysis.csv', index=False)

    # Cross-correlation peaks
    df_corr = pd.DataFrame({
        'metric': ['value', 'surprise'],
        'peak_lag': [
            causality_results['peak_lag_value'],
            causality_results['peak_lag_surprise']
        ],
        'interpretation': [
            'NEO lidera' if causality_results['peak_lag_value'] > 0 else 'EVA lidera',
            'NEO lidera' if causality_results['peak_lag_surprise'] > 0 else 'EVA lidera'
        ]
    })
    df_corr.to_csv(f'{LOG_DIR}/cross_correlations.csv', index=False)


def generate_summary(data_real: SimulationData, data_null: SimulationData,
                    bifurcation_idx: int, ideas_per_hour: List[int],
                    causality_results: Dict) -> str:
    """Generate summary report."""

    summary = []
    summary.append("=" * 70)
    summary.append("FASE B: ANÁLISIS CUÁNTICO DEL SESGO COLECTIVO")
    summary.append("Simulación de 12 Horas")
    summary.append("=" * 70)
    summary.append("")

    # 1. Divergencia vs Sincronía
    summary.append("-" * 70)
    summary.append("1. DIVERGENCIA VS SINCRONÍA")
    summary.append("-" * 70)
    div_mean = np.mean(data_real.divergence)
    div_std = np.std(data_real.divergence)
    lsi_mean = np.mean(data_real.LSI)
    summary.append(f"  Divergencia media: {div_mean:.4f} ± {div_std:.4f}")
    summary.append(f"  LSI media: {lsi_mean:.4f}")
    summary.append(f"  CE_NEO media: {np.mean(data_real.CE_NEO):.4f}")
    summary.append(f"  CE_EVA media: {np.mean(data_real.CE_EVA):.4f}")
    summary.append("")

    # 2. Especialización
    summary.append("-" * 70)
    summary.append("2. ESPECIALIZACIÓN COMPLEMENTARIA")
    summary.append("-" * 70)
    summary.append(f"  NEO final: {data_real.specialization_NEO[-1]:.4f}")
    summary.append(f"  EVA final: {data_real.specialization_EVA[-1]:.4f}")
    summary.append(f"  Punto de bifurcación: t = {bifurcation_idx}")
    summary.append("")

    # 3. Acoplamiento
    summary.append("-" * 70)
    summary.append("3. ACOPLAMIENTO ASIMÉTRICO")
    summary.append("-" * 70)
    neo_eva_mean = np.mean(data_real.coupling_NEO_EVA)
    eva_neo_mean = np.mean(data_real.coupling_EVA_NEO)
    summary.append(f"  NEO→EVA medio: {neo_eva_mean:.4f}")
    summary.append(f"  EVA→NEO medio: {eva_neo_mean:.4f}")
    summary.append(f"  Ratio asimetría: {neo_eva_mean/(eva_neo_mean+1e-12):.2f}")
    summary.append("")

    # 4. Q-Field
    summary.append("-" * 70)
    summary.append("4. Q-FIELD AVANZADO")
    summary.append("-" * 70)
    q_mean = np.mean(data_real.Q_coherence)
    q_std = np.std(data_real.Q_coherence)
    psi_corr = np.corrcoef(data_real.psi_norm_NEO, data_real.psi_norm_EVA)[0, 1]
    summary.append(f"  Q coherencia media: {q_mean:.4f} ± {q_std:.4f}")
    summary.append(f"  Entanglement proxy (corr ψ): {psi_corr:.4f}")
    summary.append("")

    # 5. Creatividad
    summary.append("-" * 70)
    summary.append("5. CREATIVIDAD EN ESTADO ESTABLE")
    summary.append("-" * 70)
    summary.append(f"  Ideas totales: {data_real.ideas_cumulative[-1]}")
    summary.append(f"  Ideas/hora media: {np.mean(ideas_per_hour):.1f}")
    summary.append(f"  Ideas/hora std: {np.std(ideas_per_hour):.1f}")
    summary.append("")

    # 6. Origen del sesgo
    summary.append("-" * 70)
    summary.append("6. ORIGEN DEL SESGO COLECTIVO")
    summary.append("-" * 70)
    summary.append(f"  Cross-correlation Value peak lag: {causality_results['peak_lag_value']}")
    summary.append(f"  Cross-correlation Surprise peak lag: {causality_results['peak_lag_surprise']}")
    summary.append(f"  Causalidad Value: NEO→EVA={causality_results['granger_value']['x_causes_y']:.4f}, EVA→NEO={causality_results['granger_value']['y_causes_x']:.4f}")
    summary.append(f"  Causalidad Surprise: NEO→EVA={causality_results['granger_surprise']['x_causes_y']:.4f}, EVA→NEO={causality_results['granger_surprise']['y_causes_x']:.4f}")
    summary.append("")
    summary.append(f"  CONCLUSIÓN: {causality_results['conclusion']}")
    summary.append("")

    # 7. Comparación con null
    summary.append("-" * 70)
    summary.append("7. COMPARACIÓN CON MODELO NULO")
    summary.append("-" * 70)

    # Statistical tests
    ks_value, pval_value = stats.ks_2samp(data_real.value_NEO, data_null.value_NEO)
    ks_spec, pval_spec = stats.ks_2samp(data_real.specialization_NEO, data_null.specialization_NEO)

    summary.append(f"  KS test Value NEO: stat={ks_value:.4f}, p={pval_value:.2e}")
    summary.append(f"  KS test Specialization: stat={ks_spec:.4f}, p={pval_spec:.2e}")
    summary.append(f"  Diferencia significativa: {'SÍ' if pval_value < 0.05 and pval_spec < 0.05 else 'NO'}")
    summary.append("")

    # Final conclusion
    summary.append("=" * 70)
    summary.append("CONCLUSIÓN FINAL")
    summary.append("=" * 70)
    summary.append("")

    # Determine main conclusion
    if causality_results['granger_value']['x_causes_y'] > causality_results['granger_value']['y_causes_x']:
        main_driver = "NEO (compresión)"
        follower = "EVA (intercambio)"
    else:
        main_driver = "EVA (intercambio)"
        follower = "NEO (compresión)"

    summary.append(f"  El sesgo colectivo emerge de la INTERACCIÓN bidireccional.")
    summary.append(f"  Driver principal: {main_driver}")
    summary.append(f"  Agente reactivo: {follower}")
    summary.append(f"  La especialización complementaria (NEO→{data_real.specialization_NEO[-1]:.2f}, EVA→{data_real.specialization_EVA[-1]:.2f})")
    summary.append(f"  emerge naturalmente del acoplamiento asimétrico.")
    summary.append("")
    summary.append(f"  EVA alcanza valores altos (V={np.mean(data_real.value_EVA):.3f}) PORQUE")
    summary.append(f"  maximiza información mutua con un mundo estabilizado por NEO.")
    summary.append(f"  NEO mantiene valores bajos (V={np.mean(data_real.value_NEO):.3f}) PORQUE")
    summary.append(f"  prioriza compresión y predictabilidad sobre recompensa.")
    summary.append("")
    summary.append("=" * 70)
    summary.append("FIN DEL ANÁLISIS")
    summary.append("=" * 70)

    return "\n".join(summary)


def main():
    """Main analysis function."""
    print("=" * 70)
    print("FASE B: ANÁLISIS CUÁNTICO DEL SESGO COLECTIVO")
    print("=" * 70)
    print()

    # Run real simulation
    print("Ejecutando simulación REAL (12h, coupling=ON)...")
    data_real = run_simulation(n_steps=6000, seed=42, coupling_enabled=True)

    # Run null model
    print("\nEjecutando simulación NULL (12h, coupling=OFF)...")
    data_null = run_simulation(n_steps=6000, seed=42, coupling_enabled=False)

    print("\nGenerando análisis y figuras...")

    # 1. Divergence vs Sync
    print("  1. Divergencia vs Sincronía...")
    plot_divergence_sync(data_real, f'{FIG_DIR}/divergence_sync.png')

    # 2. Specialization bifurcation
    print("  2. Especialización y bifurcación...")
    bifurcation_idx = plot_specialization_bifurcation(data_real,
                                                      f'{FIG_DIR}/specialization_bifurcation.png')

    # 3. Asymmetric coupling
    print("  3. Acoplamiento asimétrico...")
    plot_asymmetric_coupling(data_real, f'{FIG_DIR}/asymmetric_coupling.png')

    # 4. Q-Field entanglement
    print("  4. Q-Field y entanglement...")
    plot_qfield_entanglement(data_real, f'{FIG_DIR}/qfield_entanglement.png')

    # 5. Genesis creativity
    print("  5. Creatividad Genesis...")
    ideas_per_hour = plot_genesis_creativity(data_real, f'{FIG_DIR}/genesis_creativity_rate.png')

    # 6. Collective bias origin
    print("  6. Origen del sesgo colectivo...")
    causality_results = plot_collective_bias_origin(data_real,
                                                    f'{FIG_DIR}/collective_bias_origin.png')

    # 7. Real vs Null
    print("  7. Comparación Real vs Null...")
    plot_real_vs_null(data_real, data_null, f'{FIG_DIR}/real_vs_null_collective_bias.png')

    # Additional figures
    print("  8. Figuras adicionales...")

    # Value evolution
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data_real.t, uniform_filter1d(data_real.value_NEO, 100), 'r-',
           linewidth=2, label='NEO')
    ax.plot(data_real.t, uniform_filter1d(data_real.value_EVA, 100), 'b-',
           linewidth=2, label='EVA')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('Evolución del Value NEO vs EVA')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/value_evolution.png', dpi=150)
    plt.close()

    # Entropy evolution
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data_real.t, uniform_filter1d(data_real.entropy_NEO, 100), 'r-',
           linewidth=2, label='NEO')
    ax.plot(data_real.t, uniform_filter1d(data_real.entropy_EVA, 100), 'b-',
           linewidth=2, label='EVA')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Entropy')
    ax.set_title('Evolución de Entropía Narrativa')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/entropy_evolution.png', dpi=150)
    plt.close()

    # Polarization
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data_real.t, data_real.polarization, 'purple', linewidth=1, alpha=0.5)
    pol_smooth = uniform_filter1d(data_real.polarization, 100)
    ax.plot(data_real.t, pol_smooth, 'purple', linewidth=2, label='Polarization')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Polarization')
    ax.set_title('Polarización Colectiva')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/polarization.png', dpi=150)
    plt.close()

    # Save CSVs
    print("  9. Guardando CSVs...")
    save_csv_outputs(data_real, causality_results)

    # Generate summary
    print("  10. Generando resumen...")
    summary = generate_summary(data_real, data_null, bifurcation_idx,
                              ideas_per_hour, causality_results)

    with open(f'{LOG_DIR}/sesgo_colectivo_summary.txt', 'w') as f:
        f.write(summary)

    print("\n" + summary)

    print(f"\nArchivos guardados en:")
    print(f"  Figuras: {FIG_DIR}/")
    print(f"  Logs: {LOG_DIR}/")

    return data_real, data_null


if __name__ == '__main__':
    main()
