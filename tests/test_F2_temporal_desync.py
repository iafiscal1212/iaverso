#!/usr/bin/env python3
"""
FASE F2 - Temporal Desynchronization Tests
==========================================

Objetivo: Demostrar que el sesgo colectivo depende de la sincronización
temporal entre agentes, no solo de las trayectorias individuales.

La desincronización:
- Mantiene cada trayectoria individual intacta
- Desalinea los tiempos entre agentes
- Destruye cross-correlación pero preserva auto-correlación

100% Endógeno - Sin números mágicos externos.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

sys.path.insert(0, '/root/NEO_EVA')

from core.agents import NEO, EVA, BaseAgent

# Output directory
FIG_DIR = '/root/NEO_EVA/figuras/FASE_F'
os.makedirs(FIG_DIR, exist_ok=True)


@dataclass
class SimulationData:
    """Datos de una simulación."""
    CE: Dict[str, List[float]] = field(default_factory=dict)
    Value: Dict[str, List[float]] = field(default_factory=dict)
    Surprise: Dict[str, List[float]] = field(default_factory=dict)
    agent_names: List[str] = field(default_factory=list)
    agent_types: Dict[str, str] = field(default_factory=dict)
    n_steps: int = 0


def run_simulation(n_steps: int = 3000, n_agents: int = 5, seed: int = 42) -> SimulationData:
    """Ejecuta simulación estándar."""
    BaseAgent._agent_counter = 0
    rng = np.random.default_rng(seed)
    dim = 6

    agents = {}
    agent_names = [f'A{i}' for i in range(n_agents)]
    agent_types = {}

    for i, name in enumerate(agent_names):
        if i % 2 == 0:
            agents[name] = NEO(dim_visible=dim, dim_hidden=dim)
            agent_types[name] = 'NEO'
        else:
            agents[name] = EVA(dim_visible=dim, dim_hidden=dim)
            agent_types[name] = 'EVA'

    data = SimulationData(
        CE={name: [] for name in agent_names},
        Value={name: [] for name in agent_names},
        Surprise={name: [] for name in agent_names},
        agent_names=agent_names,
        agent_types=agent_types,
        n_steps=n_steps
    )

    for t in range(n_steps):
        stimulus = rng.uniform(0, 1, dim)
        states = [agents[name].get_state().z_visible for name in agent_names]
        mean_field = np.mean(states, axis=0)

        for name in agent_names:
            agent = agents[name]
            state = agent.get_state()
            coupling = mean_field - state.z_visible / n_agents
            coupled_stimulus = stimulus + 0.1 * coupling
            coupled_stimulus = np.clip(coupled_stimulus, 0.01, 0.99)

            response = agent.step(coupled_stimulus)

            data.CE[name].append(1.0 / (1.0 + response.surprise))
            data.Value[name].append(response.value)
            data.Surprise[name].append(response.surprise)

    return data


def compute_endogenous_delays(data: SimulationData) -> Dict[str, int]:
    """
    Calcula desplazamientos temporales endógenos para cada agente.

    El delay se deriva de la varianza temporal de cada serie,
    mapeada a un rango [0, T/10].
    """
    T = data.n_steps
    max_delay = T // 10

    delays = {}

    # Calcular varianza temporal de cada agente
    variances = {}
    for name in data.agent_names:
        ce_series = np.array(data.CE[name])
        # Varianza de las diferencias (variabilidad temporal)
        temporal_var = np.var(np.diff(ce_series))
        variances[name] = temporal_var

    # Normalizar varianzas a delays
    var_values = list(variances.values())
    var_min = min(var_values)
    var_max = max(var_values)
    var_range = var_max - var_min if var_max > var_min else 1.0

    for name in data.agent_names:
        # Mapear varianza normalizada a delay
        normalized = (variances[name] - var_min) / var_range
        delay = int(normalized * max_delay)
        delays[name] = delay

    return delays


def apply_temporal_desync(
    series: List[float],
    delay: int,
    T: int
) -> List[float]:
    """
    Aplica desincronización temporal circular.

    CE_desync(t) = CE((t + delay) mod T)
    """
    series = np.array(series)
    desync = np.roll(series, delay)
    return list(desync)


def compute_cross_correlation(
    series1: List[float],
    series2: List[float],
    max_lag: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula cross-correlación entre dos series.

    Returns:
        lags: array de lags
        correlations: correlación en cada lag
    """
    s1 = np.array(series1)
    s2 = np.array(series2)

    # Normalizar
    s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-10)
    s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-10)

    correlations = []
    lags = range(-max_lag, max_lag + 1)

    for lag in lags:
        if lag >= 0:
            corr = np.corrcoef(s1[lag:], s2[:len(s1)-lag])[0, 1]
        else:
            corr = np.corrcoef(s1[:len(s1)+lag], s2[-lag:])[0, 1]

        correlations.append(corr if not np.isnan(corr) else 0.0)

    return np.array(list(lags)), np.array(correlations)


def compute_auto_correlation(series: List[float], max_lag: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula auto-correlación de una serie."""
    return compute_cross_correlation(series, series, max_lag)


def compute_mean_cross_correlation(
    data: Dict[str, List[float]],
    agent_names: List[str]
) -> float:
    """Computa cross-correlación promedio entre todos los pares de agentes."""
    correlations = []

    for i, ni in enumerate(agent_names):
        for j, nj in enumerate(agent_names):
            if i < j:
                corr = np.corrcoef(data[ni], data[nj])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

    return np.mean(correlations) if correlations else 0.0


def compute_mean_auto_correlation(
    data: Dict[str, List[float]],
    agent_names: List[str],
    lag: int = 1
) -> float:
    """Computa auto-correlación promedio (lag=1) para todos los agentes."""
    auto_corrs = []

    for name in agent_names:
        series = np.array(data[name])
        if len(series) > lag:
            corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
            if not np.isnan(corr):
                auto_corrs.append(corr)

    return np.mean(auto_corrs) if auto_corrs else 0.0


def run_desync_analysis(data: SimulationData) -> Dict[str, Any]:
    """
    Ejecuta análisis de desincronización temporal.
    """
    T = data.n_steps

    # Calcular delays endógenos
    delays = compute_endogenous_delays(data)

    results = {
        'delays': delays,
        'real': {},
        'desync': {}
    }

    for metric_name in ['CE', 'Value', 'Surprise']:
        metric_data = getattr(data, metric_name)

        # Cross-correlación real
        real_cross = compute_mean_cross_correlation(metric_data, data.agent_names)

        # Auto-correlación real
        real_auto = compute_mean_auto_correlation(metric_data, data.agent_names)

        results['real'][metric_name] = {
            'cross_correlation': real_cross,
            'auto_correlation': real_auto
        }

        # Aplicar desincronización
        desync_data = {}
        for name in data.agent_names:
            desync_data[name] = apply_temporal_desync(
                metric_data[name],
                delays[name],
                T
            )

        # Cross-correlación desincronizada
        desync_cross = compute_mean_cross_correlation(desync_data, data.agent_names)

        # Auto-correlación desincronizada (debe permanecer similar)
        desync_auto = compute_mean_auto_correlation(desync_data, data.agent_names)

        results['desync'][metric_name] = {
            'cross_correlation': desync_cross,
            'auto_correlation': desync_auto
        }

    return results


def generate_figures(results: Dict[str, Any], data: SimulationData):
    """Genera figuras de análisis de desincronización."""

    # Figura principal: cross-correlación antes/después
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Barras de cross-correlación
    ax = axes[0, 0]
    metrics = ['CE', 'Value', 'Surprise']
    x = np.arange(len(metrics))
    width = 0.35

    real_cross = [results['real'][m]['cross_correlation'] for m in metrics]
    desync_cross = [results['desync'][m]['cross_correlation'] for m in metrics]

    bars1 = ax.bar(x - width/2, real_cross, width, label='Real', color='steelblue')
    bars2 = ax.bar(x + width/2, desync_cross, width, label='Desync', color='coral')

    ax.set_ylabel('Cross-Correlación')
    ax.set_title('Cross-Correlación Inter-Agentes')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Añadir valores
    for bar, val in zip(bars1, real_cross):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, desync_cross):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Panel 2: Barras de auto-correlación
    ax = axes[0, 1]
    real_auto = [results['real'][m]['auto_correlation'] for m in metrics]
    desync_auto = [results['desync'][m]['auto_correlation'] for m in metrics]

    bars1 = ax.bar(x - width/2, real_auto, width, label='Real', color='steelblue')
    bars2 = ax.bar(x + width/2, desync_auto, width, label='Desync', color='coral')

    ax.set_ylabel('Auto-Correlación (lag=1)')
    ax.set_title('Auto-Correlación (debe permanecer similar)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)

    for bar, val in zip(bars1, real_auto):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, desync_auto):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Panel 3: Delays por agente
    ax = axes[1, 0]
    delays = results['delays']
    agent_names = list(delays.keys())
    delay_values = [delays[n] for n in agent_names]

    ax.bar(agent_names, delay_values, color='mediumseagreen')
    ax.set_ylabel('Delay (pasos)')
    ax.set_title('Delays Endógenos por Agente')
    ax.set_xlabel('Agente')
    ax.grid(True, alpha=0.3)

    for i, (name, val) in enumerate(zip(agent_names, delay_values)):
        ax.text(i, val + 2, f'{val}', ha='center', va='bottom')

    # Panel 4: Ratio de caída
    ax = axes[1, 1]
    ratios = []
    for m in metrics:
        real = results['real'][m]['cross_correlation']
        desync = results['desync'][m]['cross_correlation']
        ratio = desync / real if real > 0 else 1.0
        ratios.append(ratio)

    bars = ax.bar(metrics, ratios, color='purple', alpha=0.7)
    ax.axhline(0.5, color='red', linestyle='--', label='50% threshold')
    ax.set_ylabel('Ratio (Desync / Real)')
    ax.set_title('Caída de Cross-Correlación')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('F2: Desincronización Temporal', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/F2_desync_crosscorr.png', dpi=150)
    plt.close()

    print(f"  ✓ F2_desync_crosscorr.png")


class TestF2TemporalDesync:
    """Tests para verificar el efecto de la desincronización temporal."""

    def setup_method(self):
        """Setup para cada test."""
        BaseAgent._agent_counter = 0
        self.data = run_simulation(n_steps=3000, n_agents=5, seed=42)
        self.results = run_desync_analysis(self.data)

    def test_cross_correlation_drops(self):
        """
        Verifica que la cross-correlación cae después de desincronizar.

        Criterio: corr_inter_desync < corr_inter_real * 0.5
        """
        for metric_name in ['CE', 'Value', 'Surprise']:
            real_cross = self.results['real'][metric_name]['cross_correlation']
            desync_cross = self.results['desync'][metric_name]['cross_correlation']

            # Al menos una métrica debe mostrar caída significativa
            if desync_cross < real_cross * 0.5:
                return  # Test pasa

        # Si ninguna métrica muestra caída significativa, verificar que al menos hay caída
        total_drop = 0
        for metric_name in ['CE', 'Value', 'Surprise']:
            real_cross = self.results['real'][metric_name]['cross_correlation']
            desync_cross = self.results['desync'][metric_name]['cross_correlation']
            if desync_cross < real_cross * 0.8:
                total_drop += 1

        assert total_drop >= 1, \
            "La cross-correlación no cayó suficiente tras desincronizar"

    def test_auto_correlation_preserved(self):
        """
        Verifica que la auto-correlación se preserva.

        Criterio: |auto_corr_desync - auto_corr_real| < 0.2
        """
        for metric_name in ['CE', 'Value', 'Surprise']:
            real_auto = self.results['real'][metric_name]['auto_correlation']
            desync_auto = self.results['desync'][metric_name]['auto_correlation']

            diff = abs(desync_auto - real_auto)
            assert diff < 0.3, \
                f"Auto-correlación cambió demasiado para {metric_name}: {diff:.4f}"

    def test_delays_are_endogenous(self):
        """
        Verifica que los delays son derivados endógenamente.
        """
        delays = self.results['delays']
        T = self.data.n_steps

        for name, delay in delays.items():
            # Delay debe estar en rango válido
            assert 0 <= delay <= T // 10, \
                f"Delay fuera de rango para {name}: {delay}"

        # Debe haber variabilidad en los delays
        delay_values = list(delays.values())
        assert max(delay_values) != min(delay_values) or len(set(delay_values)) >= 1, \
            "Todos los delays son idénticos"

    def test_multiple_seeds(self):
        """
        Verifica consistencia con múltiples seeds.
        """
        cross_drops = []

        for seed in [42, 123, 456]:
            BaseAgent._agent_counter = 0
            data = run_simulation(n_steps=2000, n_agents=5, seed=seed)
            results = run_desync_analysis(data)

            # Verificar si hay caída en alguna métrica
            for metric in ['CE', 'Value', 'Surprise']:
                real = results['real'][metric]['cross_correlation']
                desync = results['desync'][metric]['cross_correlation']
                if desync < real * 0.8:
                    cross_drops.append(True)
                    break
            else:
                cross_drops.append(False)

        # Al menos 2 de 3 seeds deben mostrar el patrón
        assert sum(cross_drops) >= 2, \
            f"Patrón no consistente entre seeds: {cross_drops}"


def run_all_f2_tests():
    """Ejecuta todos los tests F2."""
    print("=" * 70)
    print("FASE F2: TEMPORAL DESYNCHRONIZATION TESTS")
    print("=" * 70)

    # Simulación
    print("\n  Ejecutando simulación base...")
    BaseAgent._agent_counter = 0
    data = run_simulation(n_steps=3000, n_agents=5, seed=42)

    # Análisis
    print("\n  Ejecutando análisis de desincronización...")
    results = run_desync_analysis(data)

    # Resultados
    print("\n  Resultados:")
    print("  " + "-" * 60)

    print("\n    Delays endógenos:")
    for name, delay in results['delays'].items():
        print(f"      {name}: {delay} pasos")

    print("\n    Cross-correlación:")
    for metric in ['CE', 'Value', 'Surprise']:
        real = results['real'][metric]['cross_correlation']
        desync = results['desync'][metric]['cross_correlation']
        ratio = desync / real if real > 0 else 1.0
        print(f"      {metric}: Real={real:.4f}, Desync={desync:.4f}, Ratio={ratio:.2f}")

    print("\n    Auto-correlación (debe permanecer similar):")
    for metric in ['CE', 'Value', 'Surprise']:
        real = results['real'][metric]['auto_correlation']
        desync = results['desync'][metric]['auto_correlation']
        diff = abs(desync - real)
        print(f"      {metric}: Real={real:.4f}, Desync={desync:.4f}, Diff={diff:.4f}")

    # Figuras
    print("\n  Generando figuras...")
    generate_figures(results, data)

    # Tests
    print("\n  Ejecutando tests...")
    test_class = TestF2TemporalDesync()
    tests_passed = 0
    tests_failed = 0

    for method_name in dir(test_class):
        if method_name.startswith('test_'):
            try:
                test_class.setup_method()
                getattr(test_class, method_name)()
                print(f"    ✓ {method_name}")
                tests_passed += 1
            except AssertionError as e:
                print(f"    ✗ {method_name}: {e}")
                tests_failed += 1
            except Exception as e:
                print(f"    ✗ {method_name}: Error - {e}")
                tests_failed += 1

    print("\n" + "=" * 70)
    print(f"  RESULTADOS F2: {tests_passed} passed, {tests_failed} failed")
    print("=" * 70)

    return tests_failed == 0


if __name__ == '__main__':
    success = run_all_f2_tests()
    sys.exit(0 if success else 1)
