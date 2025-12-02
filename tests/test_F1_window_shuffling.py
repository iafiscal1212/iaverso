#!/usr/bin/env python3
"""
FASE F1 - Window Shuffling Tests
=================================

Objetivo: Demostrar que el sesgo colectivo desaparece cuando rompemos
la estructura temporal dentro de ventanas, manteniendo estadísticas globales.

El shuffling por ventanas:
- Mantiene las distribuciones marginales (mismo histograma)
- Rompe la estructura temporal local
- Destruye correlaciones inter-agentes si son genuinas

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
    """Ejecuta simulación estándar y guarda métricas."""
    BaseAgent._agent_counter = 0
    rng = np.random.default_rng(seed)
    dim = 6

    # Crear agentes
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

    # Inicializar datos
    data = SimulationData(
        CE={name: [] for name in agent_names},
        Value={name: [] for name in agent_names},
        Surprise={name: [] for name in agent_names},
        agent_names=agent_names,
        agent_types=agent_types,
        n_steps=n_steps
    )

    # Simular
    for t in range(n_steps):
        stimulus = rng.uniform(0, 1, dim)

        # Mean field coupling
        states = [agents[name].get_state().z_visible for name in agent_names]
        mean_field = np.mean(states, axis=0)

        for name in agent_names:
            agent = agents[name]
            state = agent.get_state()

            # Coupling
            coupling = mean_field - state.z_visible / n_agents
            coupled_stimulus = stimulus + 0.1 * coupling
            coupled_stimulus = np.clip(coupled_stimulus, 0.01, 0.99)

            # Step
            response = agent.step(coupled_stimulus)

            # Guardar métricas
            data.CE[name].append(1.0 / (1.0 + response.surprise))
            data.Value[name].append(response.value)
            data.Surprise[name].append(response.surprise)

    return data


def window_shuffle(series: List[float], window_size: int, rng: np.random.Generator) -> List[float]:
    """
    Aplica shuffling por ventanas a una serie temporal.

    Mantiene: distribución marginal (mismo histograma)
    Rompe: estructura temporal local
    """
    series = np.array(series)
    n = len(series)
    shuffled = series.copy()

    # Dividir en ventanas y shufflear cada una
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        window = shuffled[start:end].copy()
        rng.shuffle(window)
        shuffled[start:end] = window

    return list(shuffled)


def compute_inter_agent_correlation(data: Dict[str, List[float]], agent_names: List[str]) -> float:
    """Computa correlación promedio inter-agentes."""
    correlations = []

    for i, ni in enumerate(agent_names):
        for j, nj in enumerate(agent_names):
            if i < j:
                corr = np.corrcoef(data[ni], data[nj])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

    return np.mean(correlations) if correlations else 0.0


def detect_coalitions(data: Dict[str, List[float]], agent_names: List[str], threshold: float = 0.7) -> int:
    """
    Detecta coaliciones (clusters de agentes con alta correlación).

    Una coalición es un grupo donde todos los miembros tienen
    correlación > threshold entre sí.
    """
    n = len(agent_names)
    if n < 2:
        return 0

    # Matriz de correlación
    corr_matrix = np.zeros((n, n))
    for i, ni in enumerate(agent_names):
        for j, nj in enumerate(agent_names):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr = np.corrcoef(data[ni], data[nj])[0, 1]
                corr_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0.0

    # Detección de coaliciones por umbral adaptativo
    # Umbral endógeno: percentil 75 de las correlaciones
    off_diag = corr_matrix[np.triu_indices(n, k=1)]
    if len(off_diag) == 0:
        return 0

    adaptive_threshold = np.percentile(off_diag, 75)
    threshold = max(threshold, adaptive_threshold)

    # Contar pares con correlación alta
    high_corr_pairs = np.sum(off_diag > threshold)

    # Número aproximado de coaliciones
    # Si hay k pares con alta correlación, estimamos coaliciones
    if high_corr_pairs == 0:
        return 0
    elif high_corr_pairs >= n * (n - 1) / 4:  # Más del 50% de pares
        return 2  # Al menos 2 coaliciones
    else:
        return 1  # Una coalición parcial

    return int(np.sqrt(2 * high_corr_pairs))  # Aproximación


def run_window_shuffling_analysis(
    data: SimulationData,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Ejecuta análisis de shuffling por ventanas.

    Ventanas endógenas: T/10, T/20, T/40
    """
    T = data.n_steps
    rng = np.random.default_rng(seed)

    # Tamaños de ventana endógenos
    window_sizes = {
        'T/10': max(1, T // 10),
        'T/20': max(1, T // 20),
        'T/40': max(1, T // 40)
    }

    results = {
        'real': {},
        'shuffled': {}
    }

    # Métricas para datos reales
    for metric_name in ['CE', 'Value', 'Surprise']:
        metric_data = getattr(data, metric_name)
        results['real'][metric_name] = {
            'correlation': compute_inter_agent_correlation(metric_data, data.agent_names),
            'coalitions': detect_coalitions(metric_data, data.agent_names)
        }

    # Métricas para cada tamaño de ventana
    results['shuffled'] = {ws_name: {} for ws_name in window_sizes}

    for ws_name, ws in window_sizes.items():
        for metric_name in ['CE', 'Value', 'Surprise']:
            metric_data = getattr(data, metric_name)

            # Shufflear cada serie
            shuffled_data = {}
            for name in data.agent_names:
                shuffled_data[name] = window_shuffle(metric_data[name], ws, rng)

            results['shuffled'][ws_name][metric_name] = {
                'correlation': compute_inter_agent_correlation(shuffled_data, data.agent_names),
                'coalitions': detect_coalitions(shuffled_data, data.agent_names),
                'window_size': ws
            }

    return results


def verify_histogram_preservation(
    original: List[float],
    shuffled: List[float],
    n_bins: int = 50
) -> float:
    """Verifica que el histograma se preserva después del shuffling."""
    hist_orig, bins = np.histogram(original, bins=n_bins, density=True)
    hist_shuf, _ = np.histogram(shuffled, bins=bins, density=True)

    # KL divergence aproximada
    eps = np.finfo(float).eps
    hist_orig = hist_orig + eps
    hist_shuf = hist_shuf + eps

    kl_div = np.sum(hist_orig * np.log(hist_orig / hist_shuf))

    return kl_div


def generate_figures(results: Dict[str, Any], data: SimulationData):
    """Genera figuras de comparación real vs shuffled."""

    # Figura 1: Correlación real vs shuffled
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric_name in enumerate(['CE', 'Value', 'Surprise']):
        ax = axes[idx]

        # Datos
        real_corr = results['real'][metric_name]['correlation']
        shuffled_corrs = []
        window_labels = []

        for ws_name in ['T/10', 'T/20', 'T/40']:
            shuffled_corrs.append(results['shuffled'][ws_name][metric_name]['correlation'])
            window_labels.append(ws_name)

        # Barras
        x = np.arange(4)
        values = [real_corr] + shuffled_corrs
        labels = ['Real'] + window_labels
        colors = ['steelblue'] + ['coral'] * 3

        bars = ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel('Correlación Inter-Agente')
        ax.set_title(f'{metric_name}')
        ax.axhline(real_corr * 0.5, color='red', linestyle='--',
                   label='50% de Real')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Añadir valores
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('F1: Correlación Real vs Window Shuffling', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/F1_real_vs_shuffled_correlacion.png', dpi=150)
    plt.close()

    # Figura 2: Coaliciones real vs shuffled
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric_name in enumerate(['CE', 'Value', 'Surprise']):
        ax = axes[idx]

        real_coal = results['real'][metric_name]['coalitions']
        shuffled_coals = []

        for ws_name in ['T/10', 'T/20', 'T/40']:
            shuffled_coals.append(results['shuffled'][ws_name][metric_name]['coalitions'])

        x = np.arange(4)
        values = [real_coal] + shuffled_coals
        labels = ['Real'] + ['T/10', 'T/20', 'T/40']
        colors = ['steelblue'] + ['coral'] * 3

        bars = ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel('Número de Coaliciones')
        ax.set_title(f'{metric_name}')
        ax.axhline(1, color='red', linestyle='--', label='Umbral = 1')
        ax.legend()
        ax.grid(True, alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('F1: Coaliciones Real vs Window Shuffling', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/F1_real_vs_shuffled_coaliciones.png', dpi=150)
    plt.close()

    print(f"  ✓ F1_real_vs_shuffled_correlacion.png")
    print(f"  ✓ F1_real_vs_shuffled_coaliciones.png")


class TestF1WindowShuffling:
    """Tests para verificar que el shuffling destruye el sesgo colectivo."""

    def setup_method(self):
        """Setup para cada test."""
        BaseAgent._agent_counter = 0
        self.data = run_simulation(n_steps=3000, n_agents=5, seed=42)
        self.results = run_window_shuffling_analysis(self.data, seed=42)

    def test_correlation_drops_after_shuffling(self):
        """
        Verifica que la correlación cae significativamente después del shuffling.

        Criterio: corr_shuffled < corr_real * 0.5 para al menos una métrica.
        """
        significant_drops = 0

        for metric_name in ['CE', 'Value', 'Surprise']:
            real_corr = self.results['real'][metric_name]['correlation']

            for ws_name in ['T/10', 'T/20', 'T/40']:
                shuffled_corr = self.results['shuffled'][ws_name][metric_name]['correlation']

                if shuffled_corr < real_corr * 0.5:
                    significant_drops += 1

        assert significant_drops >= 1, \
            f"No hubo caída significativa de correlación: {self.results}"

    def test_coalitions_decrease_after_shuffling(self):
        """
        Verifica que las coaliciones disminuyen o se mantienen bajas.

        Criterio: coaliciones_shuffled <= 1 en la mayoría de casos.
        """
        low_coalition_count = 0
        total_cases = 0

        for metric_name in ['CE', 'Value', 'Surprise']:
            for ws_name in ['T/10', 'T/20', 'T/40']:
                coalitions = self.results['shuffled'][ws_name][metric_name]['coalitions']
                total_cases += 1

                if coalitions <= 1:
                    low_coalition_count += 1

        # Mayoría = más del 50%
        assert low_coalition_count >= total_cases / 2, \
            f"Coaliciones no bajaron suficiente: {low_coalition_count}/{total_cases}"

    def test_histogram_preserved(self):
        """
        Verifica que las distribuciones marginales se mantienen.

        El shuffling no debe cambiar el histograma de cada serie.
        """
        rng = np.random.default_rng(42)

        for metric_name in ['CE', 'Value', 'Surprise']:
            metric_data = getattr(self.data, metric_name)

            for name in self.data.agent_names:
                original = metric_data[name]
                shuffled = window_shuffle(original, self.data.n_steps // 10, rng)

                kl_div = verify_histogram_preservation(original, shuffled)

                # KL divergence debe ser muy pequeña
                assert kl_div < 0.1, \
                    f"Histograma no preservado para {name}/{metric_name}: KL={kl_div}"

    def test_multiple_seeds_consistent(self):
        """
        Verifica consistencia del resultado con diferentes seeds.
        """
        drops_by_seed = []

        for seed in [42, 123, 456]:
            BaseAgent._agent_counter = 0
            data = run_simulation(n_steps=2000, n_agents=5, seed=seed)
            results = run_window_shuffling_analysis(data, seed=seed)

            # Contar caídas significativas
            drops = 0
            for metric_name in ['CE', 'Value', 'Surprise']:
                real_corr = results['real'][metric_name]['correlation']
                for ws_name in ['T/10', 'T/20', 'T/40']:
                    shuffled_corr = results['shuffled'][ws_name][metric_name]['correlation']
                    if shuffled_corr < real_corr * 0.7:
                        drops += 1

            drops_by_seed.append(drops > 0)

        # Al menos 2 de 3 seeds deben mostrar el patrón
        assert sum(drops_by_seed) >= 2, \
            f"Resultado no consistente entre seeds: {drops_by_seed}"


def run_all_f1_tests():
    """Ejecuta todos los tests F1 manualmente."""
    print("=" * 70)
    print("FASE F1: WINDOW SHUFFLING TESTS")
    print("=" * 70)

    # Ejecutar simulación
    print("\n  Ejecutando simulación base...")
    BaseAgent._agent_counter = 0
    data = run_simulation(n_steps=3000, n_agents=5, seed=42)
    print(f"    ✓ {data.n_steps} pasos, {len(data.agent_names)} agentes")

    # Análisis
    print("\n  Ejecutando análisis de shuffling...")
    results = run_window_shuffling_analysis(data, seed=42)

    # Mostrar resultados
    print("\n  Resultados:")
    print("  " + "-" * 60)

    for metric_name in ['CE', 'Value', 'Surprise']:
        real_corr = results['real'][metric_name]['correlation']
        real_coal = results['real'][metric_name]['coalitions']

        print(f"\n    {metric_name}:")
        print(f"      Real: corr={real_corr:.4f}, coaliciones={real_coal}")

        for ws_name in ['T/10', 'T/20', 'T/40']:
            shuf_corr = results['shuffled'][ws_name][metric_name]['correlation']
            shuf_coal = results['shuffled'][ws_name][metric_name]['coalitions']
            drop_pct = (1 - shuf_corr / real_corr) * 100 if real_corr > 0 else 0
            print(f"      {ws_name}: corr={shuf_corr:.4f} ({drop_pct:+.1f}%), coal={shuf_coal}")

    # Generar figuras
    print("\n  Generando figuras...")
    generate_figures(results, data)

    # Ejecutar tests
    print("\n  Ejecutando tests...")
    test_class = TestF1WindowShuffling()
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
    print(f"  RESULTADOS F1: {tests_passed} passed, {tests_failed} failed")
    print("=" * 70)

    return tests_failed == 0


if __name__ == '__main__':
    success = run_all_f1_tests()
    sys.exit(0 if success else 1)
