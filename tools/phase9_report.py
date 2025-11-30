#!/usr/bin/env python3
"""
Phase 9: Reporte de Plasticidad
===============================
Genera figuras y reporte markdown con:
- Radar intramundo (5 componentes)
- Barras alpha_inter vs nulos
- Trayectorias (V,A) e histeresis
- Serie temporal alpha_global

Sin interpretacion, solo descripciones operativas.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from datetime import datetime

sys.path.insert(0, '/root/NEO_EVA/tools')

# Configuracion matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight'
})


def load_results(output_dir: str) -> dict:
    """Carga todos los resultados de Phase 9."""
    results = {}

    files = [
        'alpha_intraworld_neo.json',
        'alpha_intraworld_eva.json',
        'alpha_interworld.json',
        'alpha_structural.json',
        'alpha_global.json',
        'nulls_bootstrap.json',
        'pad_signals.json'
    ]

    for f in files:
        path = f"{output_dir}/{f}"
        if os.path.exists(path):
            with open(path) as fp:
                key = f.replace('.json', '')
                results[key] = json.load(fp)

    return results


def fig_radar_intraworld(results: dict, world: str, output_dir: str):
    """
    Radar plot de 5 componentes intramundo.
    """
    key = f'alpha_intraworld_{world.lower()}'
    if key not in results:
        print(f"    [SKIP] No data for {key}")
        return

    data = results[key]

    # Componentes
    components = ['alpha_affect', 'alpha_hyst', 'alpha_switch', 'alpha_recov', 'alpha_sus']
    labels = ['Affect\n(volumen)', 'Hyst\n(histeresis)', 'Switch\n(cambio)', 'Recov\n(recuperacion)', 'Sus\n(susceptibilidad)']

    values = [data[c] for c in components]

    # Normalizar a [0,1] para visualizacion
    max_val = max(values) if max(values) > 0 else 1
    values_norm = [v / max_val for v in values]

    # Cerrar el poligono
    values_norm += values_norm[:1]
    angles = np.linspace(0, 2*np.pi, len(components), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, values_norm, 'o-', linewidth=2, color='steelblue')
    ax.fill(angles, values_norm, alpha=0.25, color='steelblue')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_title(f'{world}: Componentes Intramundo\n(alpha_intra = {data["alpha_intra"]:.4f})',
                 fontweight='bold', pad=20)

    # Anadir valores originales como texto
    for angle, val, val_orig in zip(angles[:-1], values_norm, values):
        ax.annotate(f'{val_orig:.4f}',
                    xy=(angle, val),
                    xytext=(angle, val + 0.15),
                    ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/radar_intraworld_{world.lower()}.png")
    plt.close()
    print(f"    [OK] radar_intraworld_{world.lower()}.png")


def fig_bars_inter_vs_null(results: dict, output_dir: str):
    """
    Barras de indices inter vs distribucion nula.
    """
    if 'alpha_interworld' not in results or 'nulls_bootstrap' not in results:
        print("    [SKIP] No data for inter vs null comparison")
        return

    inter = results['alpha_interworld']
    nulls = results['nulls_bootstrap']

    # Indices inter
    indices = ['alpha_consent_elast', 'alpha_cross_sus', 'alpha_coord', 'alpha_homeo']
    labels = ['Consent\nElast', 'Cross\nSus', 'Coord', 'Homeo']

    observed = [inter[idx] for idx in indices]

    # Obtener nulos si existen
    null_means = []
    null_errs = []
    p_values = []

    for idx in indices:
        null_key = f'inter_{idx}'
        if null_key in nulls:
            null_means.append(nulls[null_key]['null_mean'])
            null_errs.append(nulls[null_key]['null_std'])
            p_values.append(nulls[null_key]['p_value'])
        else:
            null_means.append(0)
            null_errs.append(0)
            p_values.append(1.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(indices))
    width = 0.35

    bars1 = ax.bar(x - width/2, observed, width, label='Observado', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, null_means, width, yerr=null_errs, label='Nulo (media +/- std)',
                   color='gray', alpha=0.7, edgecolor='black', capsize=3)

    ax.set_ylabel('Valor del indice')
    ax.set_title('Indices Intermundos: Observado vs Nulo')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Anadir p-values
    for i, (obs, null, p) in enumerate(zip(observed, null_means, p_values)):
        y_max = max(obs, null + null_errs[i] if i < len(null_errs) else null)
        ax.annotate(f'p={p:.3f}',
                    xy=(i, y_max),
                    xytext=(i, y_max * 1.1),
                    ha='center', fontsize=9,
                    color='red' if p < 0.05 else 'black')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/bars_inter_vs_null.png")
    plt.close()
    print("    [OK] bars_inter_vs_null.png")


def fig_hysteresis_VA(results: dict, world: str, output_dir: str):
    """
    Trayectoria (V,A) mostrando histeresis.
    """
    if 'pad_signals' not in results:
        print(f"    [SKIP] No PAD signals for {world}")
        return

    pad = results['pad_signals'][world.lower()]
    V = np.array(pad['V'])
    A = np.array(pad['A'])

    # Subsamplear para claridad
    step = max(1, len(V) // 500)
    V_sub = V[::step]
    A_sub = A[::step]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Crear colormap por tiempo
    t = np.arange(len(V_sub))
    scatter = ax.scatter(V_sub, A_sub, c=t, cmap='viridis', alpha=0.6, s=10)

    # Conectar puntos con lineas
    ax.plot(V_sub, A_sub, 'k-', alpha=0.2, linewidth=0.5)

    # Calcular area de histeresis (convex hull como proxy)
    from scipy.spatial import ConvexHull
    try:
        points = np.column_stack([V_sub, A_sub])
        hull = ConvexHull(points)
        area = hull.volume

        # Dibujar hull
        hull_points = points[hull.vertices]
        hull_polygon = Polygon(hull_points, fill=False, edgecolor='red',
                               linewidth=2, linestyle='--', label=f'Hull area={area:.4f}')
        ax.add_patch(hull_polygon)
    except:
        area = 0

    ax.set_xlabel('Valencia (V)', fontweight='bold')
    ax.set_ylabel('Activacion (A)', fontweight='bold')
    ax.set_title(f'{world}: Trayectoria Afectiva (V,A)\nHisteresis proxy (hull area) = {area:.4f}',
                 fontweight='bold')

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)

    plt.colorbar(scatter, label='Tiempo (ciclos)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/hysteresis_VA.{world.lower()}.png")
    plt.close()
    print(f"    [OK] hysteresis_VA.{world.lower()}.png")


def fig_alpha_global_timeseries(results: dict, output_dir: str):
    """
    Serie temporal de alpha_global.
    """
    if 'alpha_global' not in results:
        print("    [SKIP] No alpha_global data")
        return

    global_data = results['alpha_global']
    series = global_data.get('alpha_global_series', [])

    if not series:
        print("    [SKIP] No alpha_global series")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(series))
    ax.plot(x, series, 'b-', linewidth=1.5, alpha=0.7)

    # Anadir media movil
    window = max(5, len(series) // 20)
    if len(series) > window:
        rolling_mean = np.convolve(series, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(series)), rolling_mean, 'r-',
                linewidth=2, label=f'Media movil (w={window})')

    ax.axhline(global_data['alpha_global'], color='green', linestyle='--',
               linewidth=2, label=f'alpha_global = {global_data["alpha_global"]:.4f}')

    ax.set_xlabel('Ventana', fontweight='bold')
    ax.set_ylabel('alpha_global (por ventana)', fontweight='bold')
    ax.set_title('Serie Temporal: Indice Global de Plasticidad', fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/alpha_global_timeseries.png")
    plt.close()
    print("    [OK] alpha_global_timeseries.png")


def generate_markdown_report(results: dict, output_dir: str) -> str:
    """
    Genera reporte markdown con descripciones operativas (sin interpretacion).
    """
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

    report = f"""# Phase 9: Plasticidad Afectiva - Reporte

**Generado:** {timestamp}

---

## 1. Metodologia

### 1.1 Ventanas y Estandarizacion

- Ventana deslizante: w = max(10, sqrt(T))
- Estandarizacion robusta: x_tilde = (x - med_w) / (IQR_w + eps)
- eps = minimo positivo representable del dtype

### 1.2 PAD Latente

Coordenadas afectivas calculadas por ranks en ventana w:
- V (Valencia) = rank_w(R_soc + h - r)
- A (Activacion) = rank_w(m + e)
- D (Dominancia) = rank_w(c + s)

---

## 2. Indices Intramundo

"""

    # NEO
    if 'alpha_intraworld_neo' in results:
        neo = results['alpha_intraworld_neo']
        report += f"""### 2.1 NEO

| Componente | Valor | Descripcion |
|------------|-------|-------------|
| alpha_affect | {neo['alpha_affect']:.6f} | sqrt(det(Cov_w([V,A,D]))) - volumen afectivo |
| alpha_hyst | {neo['alpha_hyst']:.6f} | Area poligonal (shoelace) de (V,A) en ventana |
| alpha_switch | {neo['alpha_switch']:.6f} | Tasa de cambio de estado en ventana |
| alpha_recov | {neo['alpha_recov']:.6f} | 1/med(tau) - inverso del tiempo de recuperacion |
| alpha_sus | {neo['alpha_sus']:.6f} | med(chi) - susceptibilidad OU endogena |
| **alpha_intra** | **{neo['alpha_intra']:.4f}** | Suma de ranks normalizados |

![Radar NEO](figures/radar_intraworld_neo.png)

"""

    # EVA
    if 'alpha_intraworld_eva' in results:
        eva = results['alpha_intraworld_eva']
        report += f"""### 2.2 EVA

| Componente | Valor | Descripcion |
|------------|-------|-------------|
| alpha_affect | {eva['alpha_affect']:.6f} | sqrt(det(Cov_w([V,A,D]))) - volumen afectivo |
| alpha_hyst | {eva['alpha_hyst']:.6f} | Area poligonal (shoelace) de (V,A) en ventana |
| alpha_switch | {eva['alpha_switch']:.6f} | Tasa de cambio de estado en ventana |
| alpha_recov | {eva['alpha_recov']:.6f} | 1/med(tau) - inverso del tiempo de recuperacion |
| alpha_sus | {eva['alpha_sus']:.6f} | med(chi) - susceptibilidad OU endogena |
| **alpha_intra** | **{eva['alpha_intra']:.4f}** | Suma de ranks normalizados |

![Radar EVA](figures/radar_intraworld_eva.png)

"""

    report += """---

## 3. Indices Intermundos

"""

    if 'alpha_interworld' in results:
        inter = results['alpha_interworld']
        report += f"""| Componente | Valor | Descripcion |
|------------|-------|-------------|
| alpha_consent_elast | {inter['alpha_consent_elast']:.6f} | IQR_w(Delta_p / Delta_pi) - elasticidad del consentimiento |
| alpha_cross_sus | {inter['alpha_cross_sus']:.6f} | Theil-Sen de Delta_PAD entre mundos durante consent |
| alpha_coord | {inter['alpha_coord']:.6f} | IQR(r_PAD(+1) - r_PAD(-1)) - coordinacion por modos |
| alpha_homeo | {inter['alpha_homeo']:.6f} | IQR(Delta_R_soc_ema) - homeostasis de reciprocidad |
| **alpha_inter** | **{inter['alpha_inter']:.4f}** | Suma de ranks normalizados |

![Inter vs Null](figures/bars_inter_vs_null.png)

"""

    report += """---

## 4. Indices Estructurales

"""

    if 'alpha_structural' in results:
        struct = results['alpha_structural']
        report += f"""| Componente | Valor | Descripcion |
|------------|-------|-------------|
| alpha_weights | {struct['alpha_weights']:.6f} | sum_i IQR_w(Delta_w_i) - movilidad de pesos adaptativos |
| alpha_manifold | {struct['alpha_manifold']:.6f} | IQR_w(Delta_coords) - deriva en variedad latente PCA |
| **alpha_struct** | **{struct['alpha_struct']:.4f}** | Suma de ranks normalizados |

"""

    report += """---

## 5. Indice Global

"""

    if 'alpha_global' in results:
        global_data = results['alpha_global']
        report += f"""**alpha_global = {global_data['alpha_global']:.4f}**

Composicion:
- rank(alpha_intra_NEO) + rank(alpha_intra_EVA) + rank(alpha_inter) + rank(alpha_struct)

Componentes:
"""
        for k, v in global_data['components'].items():
            report += f"- {k}: {v:.4f}\n"

        report += """
![Alpha Global Timeseries](figures/alpha_global_timeseries.png)

"""

    report += """---

## 6. Trayectorias Afectivas (Histeresis)

### 6.1 NEO
![Hysteresis NEO](figures/hysteresis_VA.neo.png)

### 6.2 EVA
![Hysteresis EVA](figures/hysteresis_VA.eva.png)

---

## 7. Nulos Bootstrap

"""

    if 'nulls_bootstrap' in results:
        nulls = results['nulls_bootstrap']
        report += """| Indice | Observado | Nulo (media) | Nulo (std) | p-value |
|--------|-----------|--------------|------------|---------|
"""
        for name, data in nulls.items():
            sig = "**" if data['p_value'] < 0.05 else ""
            report += f"| {name} | {data['observed']:.6f} | {data['null_mean']:.6f} | {data['null_std']:.6f} | {sig}{data['p_value']:.3f}{sig} |\n"

    report += f"""
---

## 8. Notas Metodologicas

1. Todos los indices calculados sin constantes fijas
2. Parametros derivados de: cuantiles, IQR, sqrt(T), ACF, PCA, ranks
3. Nulos generados por: block shuffle, phase surrogates, coupling shuffle
4. Bootstrap: {results.get('nulls_bootstrap', {}).get('neo_alpha_affect', {}).get('n_bootstrap', 'N/A')} iteraciones

---

*Reporte generado automaticamente. Sin interpretacion de resultados.*
"""

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 9: Reporte de Plasticidad')
    parser.add_argument('--output-dir', default='/root/NEO_EVA/results/phase9',
                        help='Directorio de salida')
    args = parser.parse_args()

    os.makedirs(f"{args.output_dir}/figures", exist_ok=True)

    print("=" * 70)
    print("PHASE 9: REPORTE DE PLASTICIDAD")
    print("=" * 70)

    # Cargar resultados
    print("\n[1] Cargando resultados...")
    results = load_results(args.output_dir)
    print(f"    Archivos cargados: {list(results.keys())}")

    # Generar figuras
    print("\n[2] Generando figuras...")

    fig_radar_intraworld(results, 'NEO', args.output_dir)
    fig_radar_intraworld(results, 'EVA', args.output_dir)
    fig_bars_inter_vs_null(results, args.output_dir)
    fig_hysteresis_VA(results, 'NEO', args.output_dir)
    fig_hysteresis_VA(results, 'EVA', args.output_dir)
    fig_alpha_global_timeseries(results, args.output_dir)

    # Generar reporte markdown
    print("\n[3] Generando reporte markdown...")
    report = generate_markdown_report(results, args.output_dir)

    report_path = f"{args.output_dir}/Phase9_Plasticity_Report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"    [OK] {report_path}")

    print(f"\n[OK] Reporte completo en {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
