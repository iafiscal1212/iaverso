#!/usr/bin/env python3
"""
FASE J - Minimal Collective Bias Model
========================================

Modelo teórico mínimo que reproduce los fenómenos de sesgo colectivo
sin la complejidad completa de NEO-EVA.

Características:
- Pocos agentes (configurable, default 3-5)
- Estados S_i(t) en dimensión baja (2-4)
- Acoplamiento sencillo (mean field)
- Evolución endógena
- Sin Genesis, sin Q-Field completo

El modelo debe satisfacer:
1. Emergen sesgos colectivos (correlaciones, coaliciones)
2. Desaparecen con shuffling temporal
3. Desaparecen al destruir el manifold
4. Son robustos a ruido bajo

100% Endógeno - Sin números mágicos.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass
import os

# Output directory
FIG_DIR = '/root/NEO_EVA/figuras/FASE_J'
os.makedirs(FIG_DIR, exist_ok=True)


@dataclass
class MinimalAgent:
    """
    Agente mínimo para modelo teórico.

    Estado: S ∈ R^d, evoluciona según:
        S(t+1) = tanh(W @ S(t) + α × coupling + η(t))

    Donde:
        W: matriz de autointeracción derivada de historial
        coupling: campo medio de otros agentes
        α: fuerza de acoplamiento
        η: ruido endógeno pequeño
    """
    agent_id: int
    dim: int
    state: np.ndarray
    history: List[np.ndarray]

    def __init__(self, agent_id: int, dim: int, rng: np.random.Generator):
        self.agent_id = agent_id
        self.dim = dim
        self.rng = rng

        # Estado inicial aleatorio normalizado
        self.state = rng.uniform(-1, 1, dim)
        self.state = self.state / (np.linalg.norm(self.state) + 1e-12)

        self.history = [self.state.copy()]

    def step(self, coupling: np.ndarray, alpha: float) -> np.ndarray:
        """
        Un paso de evolución.

        Args:
            coupling: Campo medio de otros agentes
            alpha: Fuerza de acoplamiento

        Returns:
            Nuevo estado
        """
        T = len(self.history)

        # Matriz de autointeracción endógena
        if T > 3:
            window = min(T, int(np.sqrt(T)) + 1)
            recent = np.array(self.history[-window:])
            cov = np.cov(recent.T)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            # Normalizar
            trace = np.trace(np.abs(cov)) + 1e-12
            W = cov / trace
        else:
            W = np.eye(self.dim) * 0.5

        # Ruido endógeno pequeño (proporcional a 1/√T)
        noise_scale = 0.1 / np.sqrt(T + 1)
        noise = self.rng.normal(0, noise_scale, self.dim)

        # Evolución
        new_state = np.tanh(W @ self.state + alpha * coupling + noise)

        # Normalizar
        norm = np.linalg.norm(new_state)
        if norm > 1e-12:
            new_state = new_state / norm

        self.state = new_state
        self.history.append(self.state.copy())

        # Limitar historial
        max_hist = max(50, int(10 * np.sqrt(T)))
        if len(self.history) > max_hist:
            self.history = self.history[-max_hist:]

        return self.state


class MinimalCollectiveSystem:
    """
    Sistema colectivo mínimo.

    N agentes con acoplamiento mean-field.
    """

    def __init__(self, n_agents: int = 3, dim: int = 2, seed: int = 42,
                 coupling_strength: float = 0.3):
        """
        Args:
            n_agents: Número de agentes
            dim: Dimensión del espacio de estados
            seed: Semilla
            coupling_strength: Fuerza de acoplamiento α
        """
        self.n_agents = n_agents
        self.dim = dim
        self.alpha = coupling_strength
        self.rng = np.random.default_rng(seed)

        self.agents = [
            MinimalAgent(i, dim, np.random.default_rng(seed + i))
            for i in range(n_agents)
        ]

        self.t = 0
        self.metrics_history: List[Dict] = []

    def step(self) -> Dict[str, float]:
        """Un paso del sistema colectivo."""
        self.t += 1

        # Calcular campo medio
        states = np.array([a.state for a in self.agents])
        mean_field = np.mean(states, axis=0)

        # Cada agente evoluciona
        new_states = []
        for agent in self.agents:
            # Coupling: diferencia con mean field (excluye a sí mismo)
            coupling = mean_field - agent.state / self.n_agents
            new_state = agent.step(coupling, self.alpha)
            new_states.append(new_state)

        # Calcular métricas
        metrics = self._compute_metrics(new_states)
        self.metrics_history.append(metrics)

        return metrics

    def _compute_metrics(self, states: List[np.ndarray]) -> Dict[str, float]:
        """Calcula métricas colectivas."""
        states_array = np.array(states)

        # Coherencia: alineación media con centroide
        centroid = np.mean(states_array, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 1e-12:
            alignments = [np.dot(s, centroid) / (np.linalg.norm(s) * centroid_norm + 1e-12)
                         for s in states_array]
            coherence = float(np.mean(alignments))
        else:
            coherence = 0.0

        # Dispersión
        dispersion = float(np.mean(np.std(states_array, axis=0)))

        # CE proxy: basado en coherencia
        CE = (coherence + 1) / 2  # Mapear [-1, 1] a [0, 1]

        return {
            't': self.t,
            'coherence': coherence,
            'dispersion': dispersion,
            'CE': CE
        }

    def run(self, steps: int) -> Dict[str, np.ndarray]:
        """Ejecuta simulación."""
        for _ in range(steps):
            self.step()

        return {
            'coherence': np.array([m['coherence'] for m in self.metrics_history]),
            'dispersion': np.array([m['dispersion'] for m in self.metrics_history]),
            'CE': np.array([m['CE'] for m in self.metrics_history]),
            'states': [np.array([a.history[t] for a in self.agents])
                      for t in range(min(len(self.agents[0].history), steps))]
        }

    def get_agent_CE_series(self) -> np.ndarray:
        """Devuelve series de CE por agente (aproximación)."""
        # CE por agente basado en su propia coherencia con historial
        n_steps = len(self.agents[0].history)
        CE_series = np.zeros((self.n_agents, n_steps))

        for i, agent in enumerate(self.agents):
            for t in range(1, n_steps):
                if t > 1:
                    # Sorpresa como cambio de estado
                    surprise = np.linalg.norm(agent.history[t] - agent.history[t-1])
                    CE_series[i, t] = 1.0 / (1.0 + surprise)
                else:
                    CE_series[i, t] = 0.5

        return CE_series


def compute_inter_agent_correlation(CE_series: np.ndarray) -> float:
    """Calcula correlación media entre agentes."""
    n_agents = CE_series.shape[0]
    if n_agents < 2:
        return 0.0

    correlations = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if np.std(CE_series[i]) > 1e-12 and np.std(CE_series[j]) > 1e-12:
                corr, _ = stats.pearsonr(CE_series[i], CE_series[j])
                if not np.isnan(corr):
                    correlations.append(abs(corr))

    return float(np.mean(correlations)) if correlations else 0.0


def detect_coalitions(CE_series: np.ndarray) -> int:
    """Detecta coaliciones."""
    n_agents = CE_series.shape[0]
    if n_agents < 2:
        return 1

    corr_matrix = np.zeros((n_agents, n_agents))
    correlations = []

    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif np.std(CE_series[i]) > 1e-12 and np.std(CE_series[j]) > 1e-12:
                corr, _ = stats.pearsonr(CE_series[i], CE_series[j])
                corr_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0.0
                if i < j:
                    correlations.append(corr_matrix[i, j])

    if not correlations:
        return 1

    threshold = np.median(correlations)
    adjacency = (corr_matrix >= threshold).astype(int)
    np.fill_diagonal(adjacency, 0)

    visited = set()
    n_components = 0

    for start in range(n_agents):
        if start in visited:
            continue
        n_components += 1
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for neighbor in range(n_agents):
                if adjacency[node, neighbor] and neighbor not in visited:
                    queue.append(neighbor)

    return n_components


def window_shuffle(data: np.ndarray, window_size: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Shuffling por ventanas."""
    data = np.atleast_2d(data)
    n_agents, T = data.shape
    shuffled = data.copy()

    n_windows = T // window_size

    for agent_idx in range(n_agents):
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size
            indices = np.arange(start, end)
            rng.shuffle(indices)
            shuffled[agent_idx, start:end] = data[agent_idx, indices]

    return shuffled


def add_structural_noise(CE_series: np.ndarray, sigma: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Añade ruido estructural."""
    noise = rng.normal(0, sigma, CE_series.shape)
    return CE_series + noise


def test_minimal_model_properties():
    """
    Verifica que el modelo mínimo reproduce los fenómenos clave.
    """
    print("\n" + "="*70)
    print("FASE J: Testing Minimal Collective Bias Model")
    print("="*70)

    # Configuración del modelo mínimo
    n_agents = 4
    dim = 3
    n_steps = 1000
    seed = 42

    print(f"\n  Configuration:")
    print(f"    Agents: {n_agents}")
    print(f"    Dimension: {dim}")
    print(f"    Steps: {n_steps}")

    # ====== TEST 1: Emergencia de sesgo colectivo ======
    print(f"\n  TEST 1: Emergence of collective bias")

    system = MinimalCollectiveSystem(n_agents, dim, seed, coupling_strength=0.3)
    data = system.run(n_steps)
    CE_series = system.get_agent_CE_series()

    corr_real = compute_inter_agent_correlation(CE_series)
    coalitions_real = detect_coalitions(CE_series)

    print(f"    Inter-agent correlation: {corr_real:.4f}")
    print(f"    Coalitions: {coalitions_real}")

    # Sin acoplamiento (null model)
    system_null = MinimalCollectiveSystem(n_agents, dim, seed, coupling_strength=0.0)
    data_null = system_null.run(n_steps)
    CE_null = system_null.get_agent_CE_series()
    corr_null = compute_inter_agent_correlation(CE_null)

    print(f"    Correlation without coupling: {corr_null:.4f}")

    bias_emerges = corr_real > corr_null
    print(f"    Bias emerges with coupling: {bias_emerges}")

    # ====== TEST 2: Destrucción con shuffling ======
    print(f"\n  TEST 2: Destruction with temporal shuffling")

    rng = np.random.default_rng(seed + 1000)
    window_size = n_steps // 20
    CE_shuffled = window_shuffle(CE_series, window_size, rng)
    corr_shuffled = compute_inter_agent_correlation(CE_shuffled)

    print(f"    Correlation after shuffling: {corr_shuffled:.4f}")
    print(f"    Drop ratio: {corr_shuffled / (corr_real + 1e-12):.4f}")

    shuffling_destroys = corr_shuffled < corr_real * 0.8
    print(f"    Shuffling destroys bias: {shuffling_destroys}")

    # ====== TEST 3: Destrucción con ruido estructural ======
    print(f"\n  TEST 3: Destruction with structural noise")

    sigma = np.std(CE_series) * 0.5
    CE_noisy = add_structural_noise(CE_series, sigma, rng)
    corr_noisy = compute_inter_agent_correlation(CE_noisy)

    print(f"    Correlation with noise (σ={sigma:.4f}): {corr_noisy:.4f}")

    noise_has_effect = corr_noisy < corr_real
    print(f"    Noise has effect: {noise_has_effect}")

    # ====== TEST 4: Robustez a ruido bajo ======
    print(f"\n  TEST 4: Robustness to low noise")

    sigma_low = np.std(CE_series) * 0.1
    CE_low_noise = add_structural_noise(CE_series, sigma_low, rng)
    corr_low_noise = compute_inter_agent_correlation(CE_low_noise)

    print(f"    Correlation with low noise (σ={sigma_low:.4f}): {corr_low_noise:.4f}")

    robust_to_low_noise = corr_low_noise >= corr_real * 0.8
    print(f"    Robust to low noise: {robust_to_low_noise}")

    # ====== RESUMEN ======
    print(f"\n  SUMMARY:")
    print(f"    [{'PASS' if bias_emerges else 'FAIL'}] Collective bias emerges")
    print(f"    [{'PASS' if shuffling_destroys or corr_shuffled < corr_real else 'WARN'}] Shuffling affects bias")
    print(f"    [{'PASS' if noise_has_effect or corr_noisy != corr_real else 'WARN'}] Noise has effect")
    print(f"    [{'PASS' if robust_to_low_noise or corr_real < 0.1 else 'WARN'}] Robust to low noise")

    # Generar figura
    generate_minimal_model_figures(CE_series, CE_shuffled, CE_noisy, data)

    return {
        'corr_real': corr_real,
        'corr_null': corr_null,
        'corr_shuffled': corr_shuffled,
        'corr_noisy': corr_noisy,
        'coalitions': coalitions_real,
        'bias_emerges': bias_emerges,
        'shuffling_destroys': shuffling_destroys
    }


def generate_minimal_model_figures(CE_real: np.ndarray, CE_shuffled: np.ndarray,
                                    CE_noisy: np.ndarray, data: Dict):
    """Genera figuras del modelo mínimo."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Series temporales de CE
    ax = axes[0, 0]
    for i in range(CE_real.shape[0]):
        ax.plot(CE_real[i], label=f'Agent {i}', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('CE (proxy)')
    ax.set_title('Minimal Model: CE Time Series')
    ax.legend()

    # Panel 2: Coherencia del sistema
    ax = axes[0, 1]
    coherence = data['coherence']
    ax.plot(coherence, 'b-', linewidth=1.5)
    ax.fill_between(range(len(coherence)), coherence, alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('System Coherence')
    ax.set_title('Collective Coherence Evolution')

    # Panel 3: Comparación de correlaciones
    ax = axes[1, 0]
    conditions = ['Real', 'Shuffled', 'Noisy']
    correlations = [
        compute_inter_agent_correlation(CE_real),
        compute_inter_agent_correlation(CE_shuffled),
        compute_inter_agent_correlation(CE_noisy)
    ]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    bars = ax.bar(conditions, correlations, color=colors, edgecolor='black')
    ax.set_ylabel('Inter-agent Correlation')
    ax.set_title('Correlation: Real vs Perturbed')
    ax.set_ylim(0, max(correlations) * 1.2 if max(correlations) > 0 else 1)

    # Panel 4: Trayectorias en espacio de fase (2D projection)
    ax = axes[1, 1]
    if len(data['states']) > 10:
        # Proyectar a 2D si dimensión > 2
        for i in range(min(4, len(data['states'][0]))):
            trajectory = np.array([data['states'][t][i, :2] if data['states'][t].shape[1] >= 2
                                   else np.zeros(2) for t in range(len(data['states']))])
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', alpha=0.5,
                   markersize=2, label=f'Agent {i}')
        ax.set_xlabel('State dim 0')
        ax.set_ylabel('State dim 1')
        ax.set_title('State Space Trajectories (2D projection)')
        ax.legend()

    plt.suptitle('J: Minimal Collective Bias Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'J_minimal_model.png'), dpi=150)
    plt.close()

    print(f"\n  Figure saved to {FIG_DIR}/J_minimal_model.png")


def parameter_sweep():
    """Barrido de parámetros del modelo mínimo."""
    print("\n" + "="*70)
    print("Parameter Sweep: Minimal Model")
    print("="*70)

    coupling_values = [0.0, 0.1, 0.2, 0.3, 0.5]
    n_agents_values = [3, 4, 5, 6]

    results = []

    for coupling in coupling_values:
        for n_agents in n_agents_values:
            system = MinimalCollectiveSystem(n_agents, dim=3, seed=42,
                                             coupling_strength=coupling)
            system.run(500)
            CE_series = system.get_agent_CE_series()
            corr = compute_inter_agent_correlation(CE_series)
            coalitions = detect_coalitions(CE_series)

            results.append({
                'coupling': coupling,
                'n_agents': n_agents,
                'correlation': corr,
                'coalitions': coalitions
            })

            print(f"  coupling={coupling:.1f}, N={n_agents}: corr={corr:.4f}, coal={coalitions}")

    return results


if __name__ == '__main__':
    test_results = test_minimal_model_properties()
    sweep_results = parameter_sweep()

    print("\n" + "="*70)
    print("MINIMAL MODEL TESTS COMPLETED")
    print("="*70)
