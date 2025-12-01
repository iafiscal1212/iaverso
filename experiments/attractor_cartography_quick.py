#!/usr/bin/env python3
"""
Cartografía de Atractores - Versión Rápida
==========================================

Versión reducida para análisis inicial.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousAgent, AutonomousDualLife

# Componentes de drive
DRIVE_COMPONENTS = ['entropy', 'neg_surprise', 'novelty', 'stability', 'integration', 'otherness']


def run_quick_cartography(n_points: int = 5, T: int = 500, seed: int = 42) -> Dict:
    """
    Cartografía rápida: rejilla n×n con T pasos.

    Explora: stability vs novelty (dimensiones más interesantes)
    """
    print("=" * 70)
    print("CARTOGRAFÍA RÁPIDA DE ATRACTORES")
    print("=" * 70)
    print(f"Rejilla: {n_points}×{n_points} = {n_points**2} puntos")
    print(f"T = {T} pasos por punto")

    # Rejilla de exploración
    values = np.linspace(0.1, 0.6, n_points)

    results = []

    for i, stability in enumerate(values):
        for j, novelty in enumerate(values):
            np.random.seed(seed)

            # Crear drives iniciales
            neo_drives = np.array([0.15, 0.15, novelty, stability, 0.15, 0.15])
            neo_drives = neo_drives / neo_drives.sum()

            eva_drives = np.array([0.15, 0.15, 0.5-novelty, 0.5-stability, 0.15, 0.15])
            eva_drives = np.clip(eva_drives, 0.05, None)
            eva_drives = eva_drives / eva_drives.sum()

            # Crear sistema
            life = AutonomousDualLife(dim=6)
            life.neo.meta_drive.weights = neo_drives.copy()
            life.eva.meta_drive.weights = eva_drives.copy()

            # Simular
            for t in range(T):
                stimulus = np.random.dirichlet(np.ones(6) * 2)
                life.step(stimulus)

            # Extraer resultados
            neo_w = life.neo.meta_drive.weights
            eva_w = life.eva.meta_drive.weights
            result = {
                'init_stability': stability,
                'init_novelty': novelty,
                'neo_final_drives': neo_w.tolist(),
                'eva_final_drives': eva_w.tolist(),
                'neo_dominant': DRIVE_COMPONENTS[np.argmax(neo_w)],
                'eva_dominant': DRIVE_COMPONENTS[np.argmax(eva_w)],
                'neo_crises': len(life.neo.crises),
                'eva_crises': len(life.eva.crises),
                'correlation': float(np.corrcoef(
                    life.neo.identity_history[-100:],
                    life.eva.identity_history[-100:]
                )[0, 1]) if len(life.neo.identity_history) > 100 else 0
            }

            results.append(result)

            if (i * n_points + j + 1) % 5 == 0:
                print(f"  Progreso: {i * n_points + j + 1}/{n_points**2}")

    # Análisis
    print("\n" + "=" * 70)
    print("ANÁLISIS DE ATRACTORES")
    print("=" * 70)

    # Contar personalidades dominantes
    neo_dominants = {}
    eva_dominants = {}

    for r in results:
        neo_dom = r['neo_dominant']
        eva_dom = r['eva_dominant']
        neo_dominants[neo_dom] = neo_dominants.get(neo_dom, 0) + 1
        eva_dominants[eva_dom] = eva_dominants.get(eva_dom, 0) + 1

    print("\nNEO - Distribución de personalidades finales:")
    for dom, count in sorted(neo_dominants.items(), key=lambda x: -x[1]):
        print(f"  {dom}: {count} ({100*count/len(results):.1f}%)")

    print("\nEVA - Distribución de personalidades finales:")
    for dom, count in sorted(eva_dominants.items(), key=lambda x: -x[1]):
        print(f"  {dom}: {count} ({100*count/len(results):.1f}%)")

    # Clustering simple por drives finales
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    neo_final = np.array([r['neo_final_drives'] for r in results])
    eva_final = np.array([r['eva_final_drives'] for r in results])

    # Determinar k endógenamente
    pca = PCA()
    pca.fit(neo_final)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k_neo = max(2, min(5, np.searchsorted(cumvar, 0.90) + 1))

    pca.fit(eva_final)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k_eva = max(2, min(5, np.searchsorted(cumvar, 0.90) + 1))

    print(f"\nClusters endógenos: NEO={k_neo}, EVA={k_eva}")

    # Clustering
    kmeans_neo = KMeans(n_clusters=k_neo, random_state=42, n_init=10)
    neo_labels = kmeans_neo.fit_predict(neo_final)

    kmeans_eva = KMeans(n_clusters=k_eva, random_state=42, n_init=10)
    eva_labels = kmeans_eva.fit_predict(eva_final)

    # Nombrar clusters
    neo_cluster_names = []
    for i, centroid in enumerate(kmeans_neo.cluster_centers_):
        top_drive = DRIVE_COMPONENTS[np.argmax(centroid)]
        second_drive = DRIVE_COMPONENTS[np.argsort(centroid)[-2]]
        neo_cluster_names.append(f"{top_drive[:3]}-{second_drive[:3]}")

    eva_cluster_names = []
    for i, centroid in enumerate(kmeans_eva.cluster_centers_):
        top_drive = DRIVE_COMPONENTS[np.argmax(centroid)]
        second_drive = DRIVE_COMPONENTS[np.argsort(centroid)[-2]]
        eva_cluster_names.append(f"{top_drive[:3]}-{second_drive[:3]}")

    print(f"\nNEO clusters: {neo_cluster_names}")
    print(f"EVA clusters: {eva_cluster_names}")

    # Añadir labels a resultados
    for i, r in enumerate(results):
        r['neo_cluster'] = int(neo_labels[i])
        r['eva_cluster'] = int(eva_labels[i])
        r['neo_cluster_name'] = neo_cluster_names[neo_labels[i]]
        r['eva_cluster_name'] = eva_cluster_names[eva_labels[i]]

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        stability_vals = [r['init_stability'] for r in results]
        novelty_vals = [r['init_novelty'] for r in results]

        # 1. Mapa NEO
        ax = axes[0, 0]
        scatter = ax.scatter(stability_vals, novelty_vals, c=neo_labels,
                            cmap='tab10', s=100, alpha=0.8)
        ax.set_xlabel('Initial Stability')
        ax.set_ylabel('Initial Novelty')
        ax.set_title('NEO: Cuencas de Atracción')
        plt.colorbar(scatter, ax=ax)

        # 2. Mapa EVA
        ax = axes[0, 1]
        scatter = ax.scatter(stability_vals, novelty_vals, c=eva_labels,
                            cmap='tab10', s=100, alpha=0.8)
        ax.set_xlabel('Initial Stability')
        ax.set_ylabel('Initial Novelty')
        ax.set_title('EVA: Cuencas de Atracción')
        plt.colorbar(scatter, ax=ax)

        # 3. Correlación
        ax = axes[0, 2]
        corrs = [r['correlation'] for r in results]
        scatter = ax.scatter(stability_vals, novelty_vals, c=corrs,
                            cmap='RdYlBu', s=100, alpha=0.8, vmin=-1, vmax=1)
        ax.set_xlabel('Initial Stability')
        ax.set_ylabel('Initial Novelty')
        ax.set_title('Correlación Final NEO-EVA')
        plt.colorbar(scatter, ax=ax)

        # 4. Centroides NEO
        ax = axes[1, 0]
        x = np.arange(6)
        width = 0.8 / k_neo
        for i, centroid in enumerate(kmeans_neo.cluster_centers_):
            ax.bar(x + i*width, centroid, width, label=neo_cluster_names[i], alpha=0.7)
        ax.set_xticks(x + width * k_neo / 2)
        ax.set_xticklabels(['ent', 'neg', 'nov', 'sta', 'int', 'oth'], rotation=45)
        ax.set_title('NEO: Centroides')
        ax.legend(fontsize=8)

        # 5. Centroides EVA
        ax = axes[1, 1]
        for i, centroid in enumerate(kmeans_eva.cluster_centers_):
            ax.bar(x + i*width, centroid, width, label=eva_cluster_names[i], alpha=0.7)
        ax.set_xticks(x + width * k_eva / 2)
        ax.set_xticklabels(['ent', 'neg', 'nov', 'sta', 'int', 'oth'], rotation=45)
        ax.set_title('EVA: Centroides')
        ax.legend(fontsize=8)

        # 6. Crisis por cluster
        ax = axes[1, 2]
        neo_crises_by_cluster = {}
        for i, r in enumerate(results):
            c = neo_labels[i]
            if c not in neo_crises_by_cluster:
                neo_crises_by_cluster[c] = []
            neo_crises_by_cluster[c].append(r['neo_crises'])

        data = [neo_crises_by_cluster[i] for i in range(k_neo)]
        bp = ax.boxplot(data, labels=[neo_cluster_names[i] for i in range(k_neo)])
        ax.set_ylabel('Crisis')
        ax.set_title('NEO: Crisis por Personalidad')
        plt.xticks(rotation=45)

        plt.tight_layout()

        os.makedirs('/root/NEO_EVA/results/attractor_map', exist_ok=True)
        plt.savefig('/root/NEO_EVA/results/attractor_map/quick_cartography.png', dpi=150)
        print("\nFigura guardada: /root/NEO_EVA/results/attractor_map/quick_cartography.png")
        plt.close()

    except Exception as e:
        print(f"Error en visualización: {e}")

    # Guardar resultados
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'n_points': n_points,
        'T': T,
        'k_neo': k_neo,
        'k_eva': k_eva,
        'neo_cluster_names': neo_cluster_names,
        'eva_cluster_names': eva_cluster_names,
        'neo_centroids': kmeans_neo.cluster_centers_.tolist(),
        'eva_centroids': kmeans_eva.cluster_centers_.tolist(),
        'results': results
    }

    with open('/root/NEO_EVA/results/attractor_map/quick_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    return final_results


def run_quick_bifurcation(T: int = 500) -> Dict:
    """
    Análisis rápido de bifurcaciones variando acople.
    """
    print("\n" + "=" * 70)
    print("ANÁLISIS RÁPIDO DE BIFURCACIONES")
    print("=" * 70)

    coupling_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    results = []

    for coupling in coupling_values:
        print(f"\nCoupling = {coupling}")

        np.random.seed(42)
        life = AutonomousDualLife(dim=6)

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)

            # Forzar nivel de acople
            life.neo.attachment = coupling
            life.eva.attachment = coupling

            life.step(stimulus)

        # Métricas
        corr = float(np.corrcoef(
            life.neo.identity_history[-100:],
            life.eva.identity_history[-100:]
        )[0, 1]) if len(life.neo.identity_history) > 100 else 0

        neo_w = life.neo.meta_drive.weights
        eva_w = life.eva.meta_drive.weights
        result = {
            'coupling': coupling,
            'neo_crises': len(life.neo.crises),
            'eva_crises': len(life.eva.crises),
            'correlation': corr,
            'neo_identity_mean': float(np.mean(life.neo.identity_history)),
            'eva_identity_mean': float(np.mean(life.eva.identity_history)),
            'neo_dominant': DRIVE_COMPONENTS[np.argmax(neo_w)],
            'eva_dominant': DRIVE_COMPONENTS[np.argmax(eva_w)]
        }

        results.append(result)
        print(f"  Crises: NEO={result['neo_crises']}, EVA={result['eva_crises']}")
        print(f"  Correlación: {result['correlation']:.3f}")

    # Detectar bifurcaciones
    print("\n--- Detección de Bifurcaciones ---")

    crisis_total = [r['neo_crises'] + r['eva_crises'] for r in results]
    correlations = [r['correlation'] for r in results]

    # Derivada numérica
    crisis_deriv = np.diff(crisis_total)
    corr_deriv = np.diff(correlations)

    # Umbral endógeno: cambio > 2 * desviación estándar
    crisis_threshold = 2 * np.std(crisis_deriv) if np.std(crisis_deriv) > 0 else np.mean(np.abs(crisis_deriv))
    corr_threshold = 2 * np.std(corr_deriv) if np.std(corr_deriv) > 0 else np.mean(np.abs(corr_deriv))

    bifurcations = []
    for i in range(len(crisis_deriv)):
        if abs(crisis_deriv[i]) > crisis_threshold or abs(corr_deriv[i]) > corr_threshold:
            bifurcations.append({
                'coupling_from': coupling_values[i],
                'coupling_to': coupling_values[i+1],
                'crisis_change': int(crisis_deriv[i]),
                'corr_change': float(corr_deriv[i])
            })
            print(f"  BIFURCACIÓN en coupling [{coupling_values[i]} → {coupling_values[i+1]}]")
            print(f"    Δcrisis = {crisis_deriv[i]:+.0f}, Δcorr = {corr_deriv[i]:+.3f}")

    if not bifurcations:
        print("  No se detectaron bifurcaciones abruptas")

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Crisis vs Coupling
        ax = axes[0]
        ax.plot(coupling_values, [r['neo_crises'] for r in results], 'b-o', label='NEO')
        ax.plot(coupling_values, [r['eva_crises'] for r in results], 'r-o', label='EVA')
        ax.set_xlabel('Coupling Strength')
        ax.set_ylabel('Número de Crisis')
        ax.set_title('Crisis vs Acople')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Correlación vs Coupling
        ax = axes[1]
        ax.plot(coupling_values, correlations, 'g-o', linewidth=2)
        ax.set_xlabel('Coupling Strength')
        ax.set_ylabel('Correlación')
        ax.set_title('Sincronización vs Acople')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Marcar bifurcaciones
        for bif in bifurcations:
            mid = (bif['coupling_from'] + bif['coupling_to']) / 2
            ax.axvline(x=mid, color='red', linestyle='--', alpha=0.7)

        # 3. Identidad media vs Coupling
        ax = axes[2]
        ax.plot(coupling_values, [r['neo_identity_mean'] for r in results], 'b-o', label='NEO')
        ax.plot(coupling_values, [r['eva_identity_mean'] for r in results], 'r-o', label='EVA')
        ax.set_xlabel('Coupling Strength')
        ax.set_ylabel('Identidad Media')
        ax.set_title('Identidad vs Acople')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/root/NEO_EVA/results/attractor_map/quick_bifurcation.png', dpi=150)
        print("\nFigura guardada: /root/NEO_EVA/results/attractor_map/quick_bifurcation.png")
        plt.close()

    except Exception as e:
        print(f"Error en visualización: {e}")

    return {
        'results': results,
        'bifurcations': bifurcations
    }


if __name__ == "__main__":
    print("=" * 70)
    print("OBJETIVO 1: MAPA DE ATRACTORES")
    print("=" * 70)

    cart_results = run_quick_cartography(n_points=6, T=500)

    print("\n" * 2)
    print("=" * 70)
    print("OBJETIVO 2: ANÁLISIS DE BIFURCACIONES")
    print("=" * 70)

    bif_results = run_quick_bifurcation(T=500)

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    print(f"\nAtractores encontrados:")
    print(f"  NEO: {cart_results['k_neo']} tipos - {cart_results['neo_cluster_names']}")
    print(f"  EVA: {cart_results['k_eva']} tipos - {cart_results['eva_cluster_names']}")

    if bif_results['bifurcations']:
        print(f"\nBifurcaciones detectadas: {len(bif_results['bifurcations'])}")
        for bif in bif_results['bifurcations']:
            print(f"  - Coupling {bif['coupling_from']} → {bif['coupling_to']}")
    else:
        print("\nNo se detectaron bifurcaciones abruptas")

    print("\nResultados en: /root/NEO_EVA/results/attractor_map/")
