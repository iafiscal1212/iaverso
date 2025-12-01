#!/usr/bin/env python3
"""
Cartografía de Atractores de Personalidad
==========================================

OBJETIVO 1: Mapa de atractores
- Rejilla de condiciones iniciales de drives
- Simulación hasta estabilización
- Clustering endógeno de personalidades finales
- Visualización de cuencas de atracción

OBJETIVO 2: Análisis de bifurcaciones
- Variar parámetros de control
- Detectar cambios bruscos de régimen

100% ENDÓGENO
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousAgent, AutonomousDualLife


@dataclass
class SimulationResult:
    """Resultado de una simulación."""
    initial_neo_drives: np.ndarray
    initial_eva_drives: np.ndarray
    final_neo_drives: np.ndarray
    final_eva_drives: np.ndarray
    neo_crises: int
    eva_crises: int
    neo_dominant: str
    eva_dominant: str
    stabilized: bool
    stabilization_time: int
    final_correlation: float


class AttractorCartographer:
    """
    Cartógrafo de atractores de personalidad.

    Explora el espacio de condiciones iniciales y mapea
    las cuencas de atracción de diferentes "personalidades".
    """

    # Componentes de drive a explorar
    DRIVE_COMPONENTS = ['entropy', 'neg_surprise', 'novelty', 'stability', 'integration', 'otherness']

    def __init__(self, explore_dims: List[str] = None):
        """
        Args:
            explore_dims: Dimensiones a explorar (default: identity, stability, novelty, otherness)
        """
        if explore_dims is None:
            self.explore_dims = ['stability', 'novelty', 'integration', 'otherness']
        else:
            self.explore_dims = explore_dims

        self.explore_indices = [self.DRIVE_COMPONENTS.index(d) for d in self.explore_dims]
        self.results: List[SimulationResult] = []

    def _create_drive_vector(self, explore_values: np.ndarray, base_value: float = 0.1) -> np.ndarray:
        """
        Crea vector de drives con valores explorados en las dimensiones seleccionadas.

        El resto de dimensiones tienen valor base.
        """
        drives = np.ones(6) * base_value

        for i, idx in enumerate(self.explore_indices):
            drives[idx] = explore_values[i]

        # Normalizar
        drives = np.clip(drives, 0.01, None)
        drives = drives / drives.sum()

        return drives

    def _compute_stabilization_time(self, t: int) -> int:
        """Tiempo máximo endógeno: basado en sqrt(t)."""
        return max(500, int(50 * np.sqrt(t + 1)))

    def _check_stabilization(self, drive_history: List[np.ndarray], window: int = None) -> Tuple[bool, int]:
        """
        Verifica si los drives se han estabilizado.

        Criterio endógeno: varianza < percentil 10 de varianzas históricas.
        """
        if len(drive_history) < 100:
            return False, 0

        if window is None:
            window = max(20, int(np.sqrt(len(drive_history))))

        recent = np.array(drive_history[-window:])
        var_recent = np.var(recent, axis=0).mean()

        # Percentil de varianzas históricas
        all_vars = []
        for i in range(0, len(drive_history) - window, window // 2):
            chunk = np.array(drive_history[i:i+window])
            all_vars.append(np.var(chunk, axis=0).mean())

        if not all_vars:
            return False, 0

        threshold = np.percentile(all_vars, 10)

        if var_recent < threshold:
            # Encontrar cuándo se estabilizó
            for t in range(len(drive_history) - window, 0, -window // 2):
                chunk = np.array(drive_history[t:t+window])
                if np.var(chunk, axis=0).mean() >= threshold:
                    return True, t + window
            return True, window

        return False, 0

    def run_single_simulation(self, neo_initial: np.ndarray, eva_initial: np.ndarray,
                              T_max: int = 1500, seed: int = 42) -> SimulationResult:
        """
        Corre una simulación con condiciones iniciales específicas.
        """
        np.random.seed(seed)

        life = AutonomousDualLife(dim=6)

        # Establecer drives iniciales
        life.neo.meta_drive.weights = neo_initial.copy()
        life.eva.meta_drive.weights = eva_initial.copy()

        neo_drive_history = [neo_initial.copy()]
        eva_drive_history = [eva_initial.copy()]

        stabilized = False
        stabilization_time = T_max

        for t in range(T_max):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            if np.random.rand() < 0.02:
                stimulus += np.random.randn(6) * 0.2
                stimulus = np.clip(stimulus, 0.01, 0.99)
                stimulus = stimulus / stimulus.sum()

            life.step(stimulus)

            neo_drive_history.append(life.neo.meta_drive.weights.copy())
            eva_drive_history.append(life.eva.meta_drive.weights.copy())

            # Verificar estabilización cada sqrt(t) pasos
            check_interval = max(50, int(np.sqrt(t + 1) * 10))
            if t > 200 and t % check_interval == 0:
                neo_stable, neo_time = self._check_stabilization(neo_drive_history)
                eva_stable, eva_time = self._check_stabilization(eva_drive_history)

                if neo_stable and eva_stable:
                    stabilized = True
                    stabilization_time = max(neo_time, eva_time)
                    break

        # Calcular correlación final
        if len(life.neo.identity_history) > 50 and len(life.eva.identity_history) > 50:
            corr = np.corrcoef(
                life.neo.identity_history[-50:],
                life.eva.identity_history[-50:]
            )[0, 1]
            corr = corr if not np.isnan(corr) else 0
        else:
            corr = 0

        # Determinar dominante
        neo_final = life.neo.meta_drive.weights
        eva_final = life.eva.meta_drive.weights
        neo_dominant = self.DRIVE_COMPONENTS[np.argmax(neo_final)]
        eva_dominant = self.DRIVE_COMPONENTS[np.argmax(eva_final)]

        return SimulationResult(
            initial_neo_drives=neo_initial,
            initial_eva_drives=eva_initial,
            final_neo_drives=neo_final,
            final_eva_drives=eva_final,
            neo_crises=len(life.neo.crises),
            eva_crises=len(life.eva.crises),
            neo_dominant=neo_dominant,
            eva_dominant=eva_dominant,
            stabilized=stabilized,
            stabilization_time=stabilization_time,
            final_correlation=float(corr)
        )

    def generate_grid(self, n_points: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Genera rejilla de condiciones iniciales.

        Para 2D: rejilla regular
        Para >2D: muestreo de Sobol o Latin Hypercube
        """
        n_dims = len(self.explore_dims)

        if n_dims == 2:
            # Rejilla regular 2D
            x = np.linspace(0.1, 0.5, n_points)
            y = np.linspace(0.1, 0.5, n_points)
            grid = []
            for xi in x:
                for yi in y:
                    grid.append(np.array([xi, yi]))
        else:
            # Latin Hypercube para más dimensiones
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=n_dims)
            samples = sampler.random(n=n_points * n_points)
            # Escalar a [0.1, 0.5]
            grid = [0.1 + 0.4 * s for s in samples]

        # Crear pares (neo_initial, eva_initial)
        # NEO y EVA empiezan igual para ver divergencia
        pairs = [(self._create_drive_vector(g), self._create_drive_vector(g)) for g in grid]

        return pairs

    def run_cartography(self, n_points: int = 10, T_max: int = 1000,
                        base_seed: int = 42) -> Dict:
        """
        Ejecuta la cartografía completa.
        """
        print("=" * 70)
        print("CARTOGRAFÍA DE ATRACTORES")
        print("=" * 70)
        print(f"Dimensiones exploradas: {self.explore_dims}")
        print(f"Puntos en rejilla: {n_points}x{n_points} = {n_points**2}")
        print(f"T_max por simulación: {T_max}")
        print()

        grid = self.generate_grid(n_points)
        total = len(grid)

        self.results = []

        for i, (neo_init, eva_init) in enumerate(grid):
            if i % max(1, total // 10) == 0:
                print(f"Progreso: {i}/{total} ({100*i/total:.0f}%)")

            result = self.run_single_simulation(neo_init, eva_init, T_max, base_seed + i)
            self.results.append(result)

        print(f"Completado: {len(self.results)} simulaciones")

        return self._analyze_results()

    def _analyze_results(self) -> Dict:
        """
        Analiza resultados y realiza clustering endógeno.
        """
        print("\n" + "=" * 50)
        print("ANÁLISIS DE ATRACTORES")
        print("=" * 50)

        # Extraer vectores finales
        neo_finals = np.array([r.final_neo_drives for r in self.results])
        eva_finals = np.array([r.final_eva_drives for r in self.results])

        # Determinar k endógenamente usando eigenvalues
        def determine_k_endogenous(data: np.ndarray) -> int:
            """Determina k basado en varianza explicada."""
            cov = np.cov(data.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]

            # k = número de eigenvalues que explican >90% de varianza
            total_var = eigenvalues.sum()
            cumsum = np.cumsum(eigenvalues)
            k = np.searchsorted(cumsum, 0.9 * total_var) + 1

            return min(max(2, k), 6)  # Entre 2 y 6 clusters

        k_neo = determine_k_endogenous(neo_finals)
        k_eva = determine_k_endogenous(eva_finals)

        print(f"k endógeno NEO: {k_neo}")
        print(f"k endógeno EVA: {k_eva}")

        # Clustering
        kmeans_neo = KMeans(n_clusters=k_neo, random_state=42, n_init=10)
        kmeans_eva = KMeans(n_clusters=k_eva, random_state=42, n_init=10)

        neo_labels = kmeans_neo.fit_predict(neo_finals)
        eva_labels = kmeans_eva.fit_predict(eva_finals)

        # Asignar nombres a clusters basados en drive dominante del centroide
        def name_clusters(centroids: np.ndarray) -> Dict[int, str]:
            names = {}
            for i, c in enumerate(centroids):
                dom_idx = np.argmax(c)
                dom_name = self.DRIVE_COMPONENTS[dom_idx]
                weight = c[dom_idx]
                names[i] = f"{dom_name}_{weight:.2f}"
            return names

        neo_cluster_names = name_clusters(kmeans_neo.cluster_centers_)
        eva_cluster_names = name_clusters(kmeans_eva.cluster_centers_)

        print(f"\nClusters NEO: {neo_cluster_names}")
        print(f"Clusters EVA: {eva_cluster_names}")

        # Estadísticas por cluster
        print("\n--- Estadísticas por cluster ---")
        for label in range(k_neo):
            mask = neo_labels == label
            n = mask.sum()
            avg_crises = np.mean([self.results[i].neo_crises for i in range(len(self.results)) if mask[i]])
            avg_corr = np.mean([self.results[i].final_correlation for i in range(len(self.results)) if mask[i]])
            print(f"NEO {neo_cluster_names[label]}: n={n}, crisis={avg_crises:.1f}, corr={avg_corr:.2f}")

        # Construir mapa de atractores
        analysis = {
            'n_simulations': len(self.results),
            'k_neo': k_neo,
            'k_eva': k_eva,
            'neo_cluster_names': neo_cluster_names,
            'eva_cluster_names': eva_cluster_names,
            'neo_labels': neo_labels.tolist(),
            'eva_labels': eva_labels.tolist(),
            'neo_centroids': kmeans_neo.cluster_centers_.tolist(),
            'eva_centroids': kmeans_eva.cluster_centers_.tolist(),
            'stabilization_rate': np.mean([r.stabilized for r in self.results]),
            'avg_stabilization_time': np.mean([r.stabilization_time for r in self.results]),
            'initial_conditions': [
                {
                    'neo_init': r.initial_neo_drives.tolist(),
                    'eva_init': r.initial_eva_drives.tolist(),
                    'neo_final': r.final_neo_drives.tolist(),
                    'eva_final': r.final_eva_drives.tolist(),
                    'neo_label': int(neo_labels[i]),
                    'eva_label': int(eva_labels[i]),
                    'neo_dominant': r.neo_dominant,
                    'eva_dominant': r.eva_dominant,
                    'correlation': r.final_correlation
                }
                for i, r in enumerate(self.results)
            ]
        }

        return analysis

    def visualize(self, analysis: Dict, save_path: str = None):
        """
        Genera visualizaciones de los atractores.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
        except ImportError:
            print("Matplotlib no disponible para visualización")
            return

        n = int(np.sqrt(len(self.results)))

        # Extraer condiciones iniciales (primeras 2 dimensiones exploradas)
        init_coords = np.array([
            [r.initial_neo_drives[self.explore_indices[0]],
             r.initial_neo_drives[self.explore_indices[1]]]
            for r in self.results
        ])

        neo_labels = np.array(analysis['neo_labels'])
        eva_labels = np.array(analysis['eva_labels'])

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Colores para clusters
        colors = plt.cm.tab10(np.linspace(0, 1, max(analysis['k_neo'], analysis['k_eva'])))

        # 1. Mapa de atractores NEO
        ax = axes[0, 0]
        scatter = ax.scatter(init_coords[:, 0], init_coords[:, 1],
                            c=neo_labels, cmap='tab10', s=50, alpha=0.7)
        ax.set_xlabel(self.explore_dims[0])
        ax.set_ylabel(self.explore_dims[1])
        ax.set_title('NEO: Cuencas de Atracción')
        plt.colorbar(scatter, ax=ax, label='Cluster')

        # 2. Mapa de atractores EVA
        ax = axes[0, 1]
        scatter = ax.scatter(init_coords[:, 0], init_coords[:, 1],
                            c=eva_labels, cmap='tab10', s=50, alpha=0.7)
        ax.set_xlabel(self.explore_dims[0])
        ax.set_ylabel(self.explore_dims[1])
        ax.set_title('EVA: Cuencas de Atracción')
        plt.colorbar(scatter, ax=ax, label='Cluster')

        # 3. Correlación final
        ax = axes[0, 2]
        correlations = [r.final_correlation for r in self.results]
        scatter = ax.scatter(init_coords[:, 0], init_coords[:, 1],
                            c=correlations, cmap='RdYlBu', s=50, alpha=0.7,
                            vmin=-1, vmax=1)
        ax.set_xlabel(self.explore_dims[0])
        ax.set_ylabel(self.explore_dims[1])
        ax.set_title('Correlación NEO-EVA Final')
        plt.colorbar(scatter, ax=ax, label='Correlación')

        # 4. Centroides NEO en espacio de drives
        ax = axes[1, 0]
        centroids = np.array(analysis['neo_centroids'])
        x = np.arange(6)
        width = 0.8 / len(centroids)
        for i, c in enumerate(centroids):
            ax.bar(x + i * width, c, width, label=analysis['neo_cluster_names'][i], alpha=0.7)
        ax.set_xticks(x + width * len(centroids) / 2)
        ax.set_xticklabels(self.DRIVE_COMPONENTS, rotation=45)
        ax.set_ylabel('Peso')
        ax.set_title('NEO: Centroides de Clusters')
        ax.legend(fontsize=8)

        # 5. Centroides EVA
        ax = axes[1, 1]
        centroids = np.array(analysis['eva_centroids'])
        for i, c in enumerate(centroids):
            ax.bar(x + i * width, c, width, label=analysis['eva_cluster_names'][i], alpha=0.7)
        ax.set_xticks(x + width * len(centroids) / 2)
        ax.set_xticklabels(self.DRIVE_COMPONENTS, rotation=45)
        ax.set_ylabel('Peso')
        ax.set_title('EVA: Centroides de Clusters')
        ax.legend(fontsize=8)

        # 6. Distribución de crisis por cluster
        ax = axes[1, 2]
        neo_crises_by_cluster = {}
        for i, r in enumerate(self.results):
            label = neo_labels[i]
            if label not in neo_crises_by_cluster:
                neo_crises_by_cluster[label] = []
            neo_crises_by_cluster[label].append(r.neo_crises)

        positions = list(neo_crises_by_cluster.keys())
        data = [neo_crises_by_cluster[p] for p in positions]
        ax.boxplot(data, positions=positions)
        ax.set_xlabel('Cluster NEO')
        ax.set_ylabel('Número de Crisis')
        ax.set_title('Crisis por Tipo de Personalidad')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura guardada: {save_path}")

        plt.close()


class BifurcationAnalyzer:
    """
    Analiza bifurcaciones variando parámetros de control.
    """

    def __init__(self):
        self.results = []

    def analyze_coupling_strength(self, coupling_values: List[float],
                                   T: int = 1000, n_seeds: int = 3) -> Dict:
        """
        Analiza cómo cambia el sistema con la fuerza de acople.
        """
        print("\n" + "=" * 70)
        print("ANÁLISIS DE BIFURCACIÓN: Fuerza de Acople")
        print("=" * 70)

        results = []

        for coupling in coupling_values:
            print(f"\nCoupling = {coupling}")

            seed_results = []
            for seed in range(n_seeds):
                np.random.seed(42 + seed)

                life = AutonomousDualLife(dim=6)

                for t in range(T):
                    stimulus = np.random.dirichlet(np.ones(6) * 2)

                    # Modificar acople antes de step
                    original_attachment_neo = life.neo.attachment
                    original_attachment_eva = life.eva.attachment

                    life.neo.attachment = coupling
                    life.eva.attachment = coupling

                    life.step(stimulus)

                    # Restaurar para que evolucione naturalmente
                    life.neo.attachment = min(1.0, original_attachment_neo + 0.001 * (coupling - original_attachment_neo))
                    life.eva.attachment = min(1.0, original_attachment_eva + 0.001 * (coupling - original_attachment_eva))

                neo_dominant = life.neo.meta_drive.component_names[np.argmax(life.neo.meta_drive.weights)]
                eva_dominant = life.eva.meta_drive.component_names[np.argmax(life.eva.meta_drive.weights)]

                # Correlación
                if len(life.neo.identity_history) > 50:
                    corr = np.corrcoef(
                        life.neo.identity_history[-50:],
                        life.eva.identity_history[-50:]
                    )[0, 1]
                else:
                    corr = 0

                seed_results.append({
                    'neo_dominant': neo_dominant,
                    'eva_dominant': eva_dominant,
                    'neo_crises': len(life.neo.crises),
                    'eva_crises': len(life.eva.crises),
                    'correlation': float(corr) if not np.isnan(corr) else 0,
                    'neo_weights': life.neo.meta_drive.weights.tolist(),
                    'eva_weights': life.eva.meta_drive.weights.tolist()
                })

            # Agregar
            avg_neo_crises = np.mean([r['neo_crises'] for r in seed_results])
            avg_eva_crises = np.mean([r['eva_crises'] for r in seed_results])
            avg_corr = np.mean([r['correlation'] for r in seed_results])

            # Contar dominantes
            neo_dominants = [r['neo_dominant'] for r in seed_results]
            eva_dominants = [r['eva_dominant'] for r in seed_results]

            results.append({
                'coupling': coupling,
                'avg_neo_crises': avg_neo_crises,
                'avg_eva_crises': avg_eva_crises,
                'avg_correlation': avg_corr,
                'neo_dominants': neo_dominants,
                'eva_dominants': eva_dominants,
                'seed_results': seed_results
            })

            print(f"  Crisis: NEO={avg_neo_crises:.1f}, EVA={avg_eva_crises:.1f}")
            print(f"  Correlación: {avg_corr:.3f}")
            print(f"  NEO dominantes: {set(neo_dominants)}")

        return self._detect_bifurcations(results, 'coupling')

    def analyze_shock_intensity(self, shock_values: List[float],
                                 T: int = 1000, n_seeds: int = 3) -> Dict:
        """
        Analiza cómo cambia el sistema con la intensidad de shocks.
        """
        print("\n" + "=" * 70)
        print("ANÁLISIS DE BIFURCACIÓN: Intensidad de Shocks")
        print("=" * 70)

        results = []

        for shock in shock_values:
            print(f"\nShock intensity = {shock}")

            seed_results = []
            for seed in range(n_seeds):
                np.random.seed(42 + seed)

                life = AutonomousDualLife(dim=6)

                for t in range(T):
                    stimulus = np.random.dirichlet(np.ones(6) * 2)

                    # Shocks con intensidad variable
                    if np.random.rand() < 0.02:
                        stimulus += np.random.randn(6) * shock
                        stimulus = np.clip(stimulus, 0.01, 0.99)
                        stimulus = stimulus / stimulus.sum()

                    life.step(stimulus)

                neo_dominant = life.neo.meta_drive.component_names[np.argmax(life.neo.meta_drive.weights)]
                eva_dominant = life.eva.meta_drive.component_names[np.argmax(life.eva.meta_drive.weights)]

                if len(life.neo.identity_history) > 50:
                    corr = np.corrcoef(
                        life.neo.identity_history[-50:],
                        life.eva.identity_history[-50:]
                    )[0, 1]
                else:
                    corr = 0

                seed_results.append({
                    'neo_dominant': neo_dominant,
                    'eva_dominant': eva_dominant,
                    'neo_crises': len(life.neo.crises),
                    'eva_crises': len(life.eva.crises),
                    'correlation': float(corr) if not np.isnan(corr) else 0
                })

            avg_neo_crises = np.mean([r['neo_crises'] for r in seed_results])
            avg_eva_crises = np.mean([r['eva_crises'] for r in seed_results])
            avg_corr = np.mean([r['correlation'] for r in seed_results])

            results.append({
                'shock': shock,
                'avg_neo_crises': avg_neo_crises,
                'avg_eva_crises': avg_eva_crises,
                'avg_correlation': avg_corr,
                'neo_dominants': [r['neo_dominant'] for r in seed_results],
                'eva_dominants': [r['eva_dominant'] for r in seed_results]
            })

            print(f"  Crisis: NEO={avg_neo_crises:.1f}, EVA={avg_eva_crises:.1f}")
            print(f"  Correlación: {avg_corr:.3f}")

        return self._detect_bifurcations(results, 'shock')

    def _detect_bifurcations(self, results: List[Dict], param_name: str) -> Dict:
        """
        Detecta puntos de bifurcación en los resultados.
        """
        print("\n" + "-" * 50)
        print("DETECCIÓN DE BIFURCACIONES")
        print("-" * 50)

        param_values = [r[param_name] for r in results]
        crises = [r['avg_neo_crises'] + r['avg_eva_crises'] for r in results]
        correlations = [r['avg_correlation'] for r in results]

        # Detectar cambios bruscos (derivada alta)
        bifurcation_points = []

        for i in range(1, len(results) - 1):
            # Derivada numérica
            d_crisis = abs(crises[i+1] - crises[i-1]) / (param_values[i+1] - param_values[i-1] + 1e-10)
            d_corr = abs(correlations[i+1] - correlations[i-1]) / (param_values[i+1] - param_values[i-1] + 1e-10)

            # Umbral endógeno: percentil 90 de todas las derivadas
            all_d_crisis = [abs(crises[j+1] - crises[j-1]) for j in range(1, len(results)-1)]
            threshold = np.percentile(all_d_crisis, 90) if all_d_crisis else 10

            if d_crisis > threshold or d_corr > 0.5:
                bifurcation_points.append({
                    'param_value': param_values[i],
                    'd_crisis': d_crisis,
                    'd_correlation': d_corr
                })
                print(f"  Bifurcación en {param_name}={param_values[i]:.2f}: Δcrisis={d_crisis:.1f}")

        # Cambio de régimen (dominantes diferentes)
        regime_changes = []
        for i in range(1, len(results)):
            prev_neo = set(results[i-1]['neo_dominants'])
            curr_neo = set(results[i]['neo_dominants'])
            if prev_neo != curr_neo:
                regime_changes.append({
                    'param_value': param_values[i],
                    'from': list(prev_neo),
                    'to': list(curr_neo)
                })
                print(f"  Cambio de régimen en {param_name}={param_values[i]:.2f}: {prev_neo} → {curr_neo}")

        return {
            'param_name': param_name,
            'param_values': param_values,
            'crises': crises,
            'correlations': correlations,
            'bifurcation_points': bifurcation_points,
            'regime_changes': regime_changes,
            'raw_results': results
        }

    def visualize_bifurcations(self, analysis: Dict, save_path: str = None):
        """Visualiza diagrama de bifurcación."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        param = analysis['param_name']
        x = analysis['param_values']

        # Crisis totales
        ax = axes[0]
        ax.plot(x, analysis['crises'], 'b-o', linewidth=2)
        for bp in analysis['bifurcation_points']:
            ax.axvline(bp['param_value'], color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel(param)
        ax.set_ylabel('Crisis Totales')
        ax.set_title('Diagrama de Bifurcación: Crisis')
        ax.grid(True, alpha=0.3)

        # Correlación
        ax = axes[1]
        ax.plot(x, analysis['correlations'], 'g-o', linewidth=2)
        for bp in analysis['bifurcation_points']:
            ax.axvline(bp['param_value'], color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel(param)
        ax.set_ylabel('Correlación NEO-EVA')
        ax.set_title('Diagrama de Bifurcación: Sincronía')
        ax.grid(True, alpha=0.3)

        # Régimen (drives dominantes)
        ax = axes[2]
        neo_crisis = [r['avg_neo_crises'] for r in analysis['raw_results']]
        eva_crisis = [r['avg_eva_crises'] for r in analysis['raw_results']]
        ax.plot(x, neo_crisis, 'b-o', label='NEO', linewidth=2)
        ax.plot(x, eva_crisis, 'r-o', label='EVA', linewidth=2)
        for rc in analysis['regime_changes']:
            ax.axvline(rc['param_value'], color='purple', linestyle=':', alpha=0.7)
        ax.set_xlabel(param)
        ax.set_ylabel('Crisis por Agente')
        ax.set_title('Crisis NEO vs EVA')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figura guardada: {save_path}")

        plt.close()


def run_full_analysis():
    """Ejecuta análisis completo."""
    print("=" * 70)
    print("ANÁLISIS COMPLETO DE ATRACTORES Y BIFURCACIONES")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")

    os.makedirs('/root/NEO_EVA/results/attractors', exist_ok=True)
    os.makedirs('/root/NEO_EVA/figures', exist_ok=True)

    # OBJETIVO 1: Cartografía de atractores
    print("\n" + "#" * 70)
    print("OBJETIVO 1: CARTOGRAFÍA DE ATRACTORES")
    print("#" * 70)

    cartographer = AttractorCartographer(explore_dims=['stability', 'novelty'])
    attractor_analysis = cartographer.run_cartography(n_points=8, T_max=800)

    cartographer.visualize(attractor_analysis, '/root/NEO_EVA/figures/attractor_map.png')

    with open('/root/NEO_EVA/results/attractors/cartography.json', 'w') as f:
        json.dump(attractor_analysis, f, indent=2, default=str)

    # OBJETIVO 2: Bifurcaciones
    print("\n" + "#" * 70)
    print("OBJETIVO 2: ANÁLISIS DE BIFURCACIONES")
    print("#" * 70)

    bifurcation = BifurcationAnalyzer()

    # Bifurcación por acople
    coupling_analysis = bifurcation.analyze_coupling_strength(
        coupling_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        T=800, n_seeds=3
    )
    bifurcation.visualize_bifurcations(coupling_analysis, '/root/NEO_EVA/figures/bifurcation_coupling.png')

    # Bifurcación por shocks
    shock_analysis = bifurcation.analyze_shock_intensity(
        shock_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        T=800, n_seeds=3
    )
    bifurcation.visualize_bifurcations(shock_analysis, '/root/NEO_EVA/figures/bifurcation_shock.png')

    # Guardar todo
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'attractor_cartography': attractor_analysis,
        'bifurcation_coupling': coupling_analysis,
        'bifurcation_shock': shock_analysis
    }

    with open('/root/NEO_EVA/results/attractors/full_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)
    print(f"Resultados en: /root/NEO_EVA/results/attractors/")
    print(f"Figuras en: /root/NEO_EVA/figures/")

    return all_results


if __name__ == "__main__":
    run_full_analysis()
