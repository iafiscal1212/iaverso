#!/usr/bin/env python3
"""
3.1 MAPA DE ATRACTORES FORMAL - DIAGRAMA DE FASES DE PERSONALIDAD
=================================================================

Barrido sistemático de parámetros:
- coupling (fuerza de acople)
- shock_intensity (perturbaciones externas)
- irreversibility (peso de cambios irreversibles)

Mediciones:
- Drives dominantes finales
- Tamaño de cuencas de atracción
- Transiciones de fase (bifurcaciones)

Objetivo: Diagrama de fases estructural de personalidad
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousAgent, AutonomousDualLife

DRIVE_NAMES = ['entropy', 'neg_surprise', 'novelty', 'stability', 'integration', 'otherness']


@dataclass
class PhasePoint:
    """Un punto en el diagrama de fases."""
    coupling: float
    shock_intensity: float
    irreversibility: float

    neo_dominant: str
    eva_dominant: str
    neo_drives: np.ndarray
    eva_drives: np.ndarray

    neo_crises: int
    eva_crises: int
    correlation: float

    # Métricas adicionales
    neo_entropy: float
    eva_entropy: float
    stability_index: float


def compute_drive_entropy(drives: np.ndarray) -> float:
    """Entropía de Shannon de la distribución de drives."""
    drives = np.clip(drives, 1e-10, None)
    drives = drives / drives.sum()
    return -np.sum(drives * np.log(drives))


def compute_stability_index(history: List[np.ndarray], window: int = 50) -> float:
    """Índice de estabilidad: inverso de varianza reciente."""
    if len(history) < window:
        return 0.0
    recent = np.array(history[-window:])
    variance = np.mean(np.var(recent, axis=0))
    return 1.0 / (1.0 + 10 * variance)


class ModifiedDualLife(AutonomousDualLife):
    """Sistema dual con parámetros modificables."""

    def __init__(self, dim: int = 6, coupling: float = 0.5,
                 shock_intensity: float = 0.0, irreversibility: float = 0.1):
        super().__init__(dim)
        self.dim = dim
        self.coupling_strength = coupling
        self.shock_intensity = shock_intensity
        self.irreversibility = irreversibility

        # Historias de drives
        self.neo_drive_history = []
        self.eva_drive_history = []

    def step(self, world_stimulus: np.ndarray) -> Dict:
        # Aplicar shock con cierta probabilidad
        if self.shock_intensity > 0 and np.random.random() < 0.02:
            shock = np.random.randn(self.dim) * self.shock_intensity
            world_stimulus = world_stimulus + shock
            world_stimulus = np.clip(world_stimulus, 0.01, None)
            world_stimulus = world_stimulus / world_stimulus.sum()

        # Modificar attachment según coupling
        self.neo.attachment = self.coupling_strength
        self.eva.attachment = self.coupling_strength

        # Step normal
        result = super().step(world_stimulus)

        # Guardar drives
        self.neo_drive_history.append(self.neo.meta_drive.weights.copy())
        self.eva_drive_history.append(self.eva.meta_drive.weights.copy())

        # Aplicar irreversibilidad (los cambios de drives son parcialmente permanentes)
        if len(self.neo_drive_history) > 1:
            prev_neo = self.neo_drive_history[-2]
            prev_eva = self.eva_drive_history[-2]

            # Mezcla: nuevo = (1-irrev)*nuevo + irrev*anterior
            # Esto hace que los cambios sean más "costosos"
            self.neo.meta_drive.weights = (1 - self.irreversibility) * self.neo.meta_drive.weights + \
                                          self.irreversibility * prev_neo
            self.eva.meta_drive.weights = (1 - self.irreversibility) * self.eva.meta_drive.weights + \
                                          self.irreversibility * prev_eva

        return result


def run_phase_diagram(
    coupling_values: List[float],
    shock_values: List[float],
    T: int = 500,
    n_seeds: int = 3
) -> List[PhasePoint]:
    """
    Genera el diagrama de fases barriendo coupling y shock_intensity.
    """
    print("=" * 70)
    print("DIAGRAMA DE FASES DE PERSONALIDAD")
    print("=" * 70)
    print(f"Coupling: {coupling_values}")
    print(f"Shock: {shock_values}")
    print(f"T={T}, seeds={n_seeds}")

    results = []
    total = len(coupling_values) * len(shock_values) * n_seeds
    count = 0

    for coupling in coupling_values:
        for shock in shock_values:
            for seed in range(n_seeds):
                count += 1
                if count % 10 == 0:
                    print(f"  Progreso: {count}/{total}")

                np.random.seed(42 + seed)

                life = ModifiedDualLife(
                    dim=6,
                    coupling=coupling,
                    shock_intensity=shock,
                    irreversibility=0.1
                )

                for t in range(T):
                    stimulus = np.random.dirichlet(np.ones(6) * 2)
                    life.step(stimulus)

                # Extraer resultados
                neo_w = life.neo.meta_drive.weights
                eva_w = life.eva.meta_drive.weights

                neo_dominant = DRIVE_NAMES[np.argmax(neo_w)]
                eva_dominant = DRIVE_NAMES[np.argmax(eva_w)]

                # Correlación
                if len(life.neo.identity_history) > 50:
                    corr = np.corrcoef(
                        life.neo.identity_history[-50:],
                        life.eva.identity_history[-50:]
                    )[0, 1]
                    corr = corr if not np.isnan(corr) else 0
                else:
                    corr = 0

                # Entropía de drives
                neo_entropy = compute_drive_entropy(neo_w)
                eva_entropy = compute_drive_entropy(eva_w)

                # Estabilidad
                stability = compute_stability_index(life.neo_drive_history)

                point = PhasePoint(
                    coupling=coupling,
                    shock_intensity=shock,
                    irreversibility=0.1,
                    neo_dominant=neo_dominant,
                    eva_dominant=eva_dominant,
                    neo_drives=neo_w,
                    eva_drives=eva_w,
                    neo_crises=len(life.neo.crises),
                    eva_crises=len(life.eva.crises),
                    correlation=corr,
                    neo_entropy=neo_entropy,
                    eva_entropy=eva_entropy,
                    stability_index=stability
                )

                results.append(point)

    return results


def analyze_phase_diagram(results: List[PhasePoint]) -> Dict:
    """Analiza el diagrama de fases."""
    print("\n" + "=" * 70)
    print("ANÁLISIS DEL DIAGRAMA DE FASES")
    print("=" * 70)

    # 1. Distribución de atractores
    neo_attractors = Counter([p.neo_dominant for p in results])
    eva_attractors = Counter([p.eva_dominant for p in results])

    print("\n1. DISTRIBUCIÓN DE ATRACTORES")
    print("-" * 40)
    print("NEO:")
    for attr, count in neo_attractors.most_common():
        print(f"  {attr}: {count} ({100*count/len(results):.1f}%)")
    print("EVA:")
    for attr, count in eva_attractors.most_common():
        print(f"  {attr}: {count} ({100*count/len(results):.1f}%)")

    # 2. Transiciones de fase
    print("\n2. TRANSICIONES DE FASE")
    print("-" * 40)

    # Agrupar por coupling
    coupling_values = sorted(set(p.coupling for p in results))

    for i in range(len(coupling_values) - 1):
        c1, c2 = coupling_values[i], coupling_values[i+1]

        points_c1 = [p for p in results if p.coupling == c1]
        points_c2 = [p for p in results if p.coupling == c2]

        neo_dom_c1 = Counter([p.neo_dominant for p in points_c1]).most_common(1)[0][0]
        neo_dom_c2 = Counter([p.neo_dominant for p in points_c2]).most_common(1)[0][0]

        if neo_dom_c1 != neo_dom_c2:
            print(f"  TRANSICIÓN NEO en coupling [{c1} → {c2}]: {neo_dom_c1} → {neo_dom_c2}")

        eva_dom_c1 = Counter([p.eva_dominant for p in points_c1]).most_common(1)[0][0]
        eva_dom_c2 = Counter([p.eva_dominant for p in points_c2]).most_common(1)[0][0]

        if eva_dom_c1 != eva_dom_c2:
            print(f"  TRANSICIÓN EVA en coupling [{c1} → {c2}]: {eva_dom_c1} → {eva_dom_c2}")

    # 3. Cuencas de atracción
    print("\n3. CUENCAS DE ATRACCIÓN (tamaño)")
    print("-" * 40)

    # Para cada atractor, contar en qué rango de parámetros aparece
    for attractor in set(neo_attractors.keys()):
        points = [p for p in results if p.neo_dominant == attractor]
        if points:
            coupling_range = (min(p.coupling for p in points), max(p.coupling for p in points))
            shock_range = (min(p.shock_intensity for p in points), max(p.shock_intensity for p in points))
            print(f"  {attractor}:")
            print(f"    Coupling: [{coupling_range[0]:.2f}, {coupling_range[1]:.2f}]")
            print(f"    Shock: [{shock_range[0]:.2f}, {shock_range[1]:.2f}]")
            print(f"    Tamaño: {len(points)} puntos ({100*len(points)/len(results):.1f}%)")

    # 4. Regímenes especiales
    print("\n4. REGÍMENES ESPECIALES")
    print("-" * 40)

    # ¿Hay puntos donde integration domina?
    integration_points = [p for p in results if p.neo_dominant == 'integration' or p.eva_dominant == 'integration']
    if integration_points:
        print(f"  Régimen INTEGRATION: {len(integration_points)} puntos")
        avg_coupling = np.mean([p.coupling for p in integration_points])
        avg_shock = np.mean([p.shock_intensity for p in integration_points])
        print(f"    Coupling promedio: {avg_coupling:.3f}")
        print(f"    Shock promedio: {avg_shock:.3f}")
    else:
        print("  No se encontró régimen dominado por INTEGRATION")

    # ¿Alta correlación implica cierto atractor?
    high_corr_points = [p for p in results if p.correlation > 0.5]
    if high_corr_points:
        neo_in_high_corr = Counter([p.neo_dominant for p in high_corr_points])
        print(f"\n  En alta correlación (>0.5):")
        for attr, count in neo_in_high_corr.most_common(3):
            print(f"    {attr}: {count}")

    # 5. Matriz de transición
    print("\n5. MATRIZ DE COEXISTENCIA NEO-EVA")
    print("-" * 40)

    coex = {}
    for p in results:
        key = (p.neo_dominant, p.eva_dominant)
        coex[key] = coex.get(key, 0) + 1

    print(f"  {'':12}", end="")
    for d in DRIVE_NAMES[:4]:
        print(f"{d[:6]:>8}", end="")
    print()

    for neo_d in DRIVE_NAMES[:4]:
        print(f"  {neo_d[:10]:12}", end="")
        for eva_d in DRIVE_NAMES[:4]:
            count = coex.get((neo_d, eva_d), 0)
            print(f"{count:>8}", end="")
        print()

    return {
        'neo_attractors': dict(neo_attractors),
        'eva_attractors': dict(eva_attractors),
        'n_points': len(results)
    }


def visualize_phase_diagram(results: List[PhasePoint], save_path: str):
    """Genera visualización del diagrama de fases."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Extraer datos
        couplings = np.array([p.coupling for p in results])
        shocks = np.array([p.shock_intensity for p in results])

        # Mapear atractores a números
        attractor_map = {name: i for i, name in enumerate(DRIVE_NAMES)}
        neo_attrs = np.array([attractor_map[p.neo_dominant] for p in results])
        eva_attrs = np.array([attractor_map[p.eva_dominant] for p in results])
        correlations = np.array([p.correlation for p in results])
        neo_crises = np.array([p.neo_crises for p in results])
        stability = np.array([p.stability_index for p in results])
        neo_entropy = np.array([p.neo_entropy for p in results])

        # 1. Atractor NEO
        ax = axes[0, 0]
        scatter = ax.scatter(couplings, shocks, c=neo_attrs, cmap='tab10',
                            s=50, alpha=0.7)
        ax.set_xlabel('Coupling')
        ax.set_ylabel('Shock Intensity')
        ax.set_title('NEO: Atractor Dominante')
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(6))
        cbar.set_ticklabels([d[:3] for d in DRIVE_NAMES])

        # 2. Atractor EVA
        ax = axes[0, 1]
        scatter = ax.scatter(couplings, shocks, c=eva_attrs, cmap='tab10',
                            s=50, alpha=0.7)
        ax.set_xlabel('Coupling')
        ax.set_ylabel('Shock Intensity')
        ax.set_title('EVA: Atractor Dominante')
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(6))
        cbar.set_ticklabels([d[:3] for d in DRIVE_NAMES])

        # 3. Correlación
        ax = axes[0, 2]
        scatter = ax.scatter(couplings, shocks, c=correlations, cmap='RdYlBu',
                            s=50, alpha=0.7, vmin=-1, vmax=1)
        ax.set_xlabel('Coupling')
        ax.set_ylabel('Shock Intensity')
        ax.set_title('Correlación NEO-EVA')
        plt.colorbar(scatter, ax=ax)

        # 4. Crisis NEO
        ax = axes[1, 0]
        scatter = ax.scatter(couplings, shocks, c=neo_crises, cmap='YlOrRd',
                            s=50, alpha=0.7)
        ax.set_xlabel('Coupling')
        ax.set_ylabel('Shock Intensity')
        ax.set_title('Crisis NEO')
        plt.colorbar(scatter, ax=ax)

        # 5. Estabilidad
        ax = axes[1, 1]
        scatter = ax.scatter(couplings, shocks, c=stability, cmap='viridis',
                            s=50, alpha=0.7)
        ax.set_xlabel('Coupling')
        ax.set_ylabel('Shock Intensity')
        ax.set_title('Índice de Estabilidad')
        plt.colorbar(scatter, ax=ax)

        # 6. Entropía de drives
        ax = axes[1, 2]
        scatter = ax.scatter(couplings, shocks, c=neo_entropy, cmap='plasma',
                            s=50, alpha=0.7)
        ax.set_xlabel('Coupling')
        ax.set_ylabel('Shock Intensity')
        ax.set_title('Entropía de Drives (NEO)')
        plt.colorbar(scatter, ax=ax)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigura guardada: {save_path}")
        plt.close()

    except Exception as e:
        print(f"Error en visualización: {e}")


def run_full_phase_analysis():
    """Ejecuta análisis completo del diagrama de fases."""
    print("=" * 70)
    print("3.1 MAPA DE ATRACTORES FORMAL")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")

    os.makedirs('/root/NEO_EVA/results/phase_diagram', exist_ok=True)

    # Parámetros a barrer
    coupling_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    shock_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Generar diagrama
    results = run_phase_diagram(
        coupling_values=coupling_values,
        shock_values=shock_values,
        T=400,
        n_seeds=2
    )

    # Analizar
    analysis = analyze_phase_diagram(results)

    # Visualizar
    visualize_phase_diagram(
        results,
        '/root/NEO_EVA/results/phase_diagram/phase_diagram.png'
    )

    # Guardar resultados
    results_json = []
    for p in results:
        results_json.append({
            'coupling': p.coupling,
            'shock_intensity': p.shock_intensity,
            'neo_dominant': p.neo_dominant,
            'eva_dominant': p.eva_dominant,
            'neo_drives': p.neo_drives.tolist(),
            'eva_drives': p.eva_drives.tolist(),
            'correlation': p.correlation,
            'neo_crises': p.neo_crises,
            'eva_crises': p.eva_crises,
            'stability_index': p.stability_index
        })

    with open('/root/NEO_EVA/results/phase_diagram/results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'coupling_values': coupling_values,
                'shock_values': shock_values,
                'T': 400,
                'n_seeds': 2
            },
            'analysis': analysis,
            'points': results_json
        }, f, indent=2)

    print(f"\nResultados guardados en /root/NEO_EVA/results/phase_diagram/")

    return results, analysis


if __name__ == "__main__":
    run_full_phase_analysis()
