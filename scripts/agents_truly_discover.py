#!/usr/bin/env python3
"""
Los Agentes REALMENTE Descubren - SIN RESPUESTAS PRECONCEBIDAS
===============================================================

REGLAS:
1. NO hay "zona habitable" hardcodeada
2. NO hay rangos de temperatura escritos por mí
3. La Tierra es UN DATO MÁS, no la referencia
4. Los agentes SOLO ven números, no saben qué es "habitable"

¿Qué patrones emergen de los datos puros?
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from scipy import stats

COSMOS_PATH = Path('/root/NEO_EVA/data/cosmos')


def load_data():
    """Cargar exoplanetas + sistema solar como datos iguales."""
    path = COSMOS_PATH / 'exoplanets.csv'
    if not path.exists():
        return None

    df = pd.read_csv(path)

    # Añadir sistema solar SIN ETIQUETAS ESPECIALES
    # Son datos como cualquier otro
    solar = pd.DataFrame([
        {'pl_name': 'Sol_p1', 'pl_eqt': 288, 'pl_rade': 1.0, 'pl_bmasse': 1.0,
         'pl_orbper': 365, 'st_teff': 5778, 'sy_dist': 0.0000048},  # Tierra
        {'pl_name': 'Sol_p2', 'pl_eqt': 737, 'pl_rade': 0.95, 'pl_bmasse': 0.815,
         'pl_orbper': 225, 'st_teff': 5778, 'sy_dist': 0.0000048},  # Venus
        {'pl_name': 'Sol_p3', 'pl_eqt': 210, 'pl_rade': 0.53, 'pl_bmasse': 0.107,
         'pl_orbper': 687, 'st_teff': 5778, 'sy_dist': 0.0000048},  # Marte
        {'pl_name': 'Sol_p4', 'pl_eqt': 440, 'pl_rade': 0.38, 'pl_bmasse': 0.055,
         'pl_orbper': 88, 'st_teff': 5778, 'sy_dist': 0.0000048},   # Mercurio
        {'pl_name': 'Sol_p5', 'pl_eqt': 165, 'pl_rade': 11.2, 'pl_bmasse': 317.8,
         'pl_orbper': 4333, 'st_teff': 5778, 'sy_dist': 0.0000048}, # Jupiter
    ])

    df = pd.concat([df, solar], ignore_index=True)
    return df


class BlindExplorer:
    """Explorador que NO SABE qué es habitabilidad."""

    def __init__(self, name: str):
        self.name = name
        self.observations = []

    def observe(self, text: str):
        self.observations.append(text)
        print(f"      {text}")


def phase1_statistics(df: pd.DataFrame, explorer: BlindExplorer):
    """Fase 1: Solo estadísticas descriptivas."""
    print(f"\n  [{explorer.name}] Analizando distribuciones...")

    cols = ['pl_eqt', 'pl_rade', 'pl_bmasse']

    for col in cols:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                explorer.observe(f"{col}: min={vals.min():.2f}, max={vals.max():.2f}, "
                                f"median={vals.median():.2f}, std={vals.std():.2f}")

                # ¿Hay valores atípicos?
                q1, q3 = vals.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = ((vals < q1 - 1.5*iqr) | (vals > q3 + 1.5*iqr)).sum()
                if outliers > 0:
                    explorer.observe(f"  → {outliers} valores atípicos en {col}")


def phase2_clustering(df: pd.DataFrame, explorer: BlindExplorer):
    """Fase 2: Clustering sin saber qué buscar."""
    print(f"\n  [{explorer.name}] Buscando grupos naturales...")

    cols = ['pl_eqt', 'pl_rade', 'pl_bmasse']
    available = [c for c in cols if c in df.columns]

    # Preparar datos
    data = df[available].dropna()
    if len(data) < 20:
        explorer.observe("Datos insuficientes para clustering")
        return None, None

    # Normalizar
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    # Probar diferentes números de clusters
    best_k = 3
    best_score = -1

    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        # Silhouette simplificado
        score = -km.inertia_ / len(X)
        if score > best_score:
            best_score = score
            best_k = k

    # Clustering final
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    explorer.observe(f"Encontré {best_k} grupos naturales en los datos")

    # Analizar cada cluster
    data_with_labels = data.copy()
    data_with_labels['cluster'] = labels

    cluster_stats = []
    for c in range(best_k):
        cluster_data = data_with_labels[data_with_labels['cluster'] == c]
        stats_dict = {'cluster': c, 'count': len(cluster_data)}

        for col in available:
            stats_dict[f'{col}_mean'] = cluster_data[col].mean()
            stats_dict[f'{col}_std'] = cluster_data[col].std()

        cluster_stats.append(stats_dict)

        # Describir el cluster sin saber qué significa
        temp_mean = stats_dict.get('pl_eqt_mean', 0)
        rad_mean = stats_dict.get('pl_rade_mean', 0)
        mass_mean = stats_dict.get('pl_bmasse_mean', 0)

        desc = f"Grupo {c}: {len(cluster_data)} planetas, "
        desc += f"temp~{temp_mean:.0f}, rad~{rad_mean:.1f}, masa~{mass_mean:.1f}"
        explorer.observe(desc)

    return data_with_labels, pd.DataFrame(cluster_stats)


def phase3_find_special(df: pd.DataFrame, clustered: pd.DataFrame, explorer: BlindExplorer):
    """Fase 3: ¿Hay algo especial en algún grupo?"""
    print(f"\n  [{explorer.name}] Buscando grupos especiales...")

    if clustered is None:
        return

    # ¿Cuál es el cluster más raro (menos planetas)?
    cluster_counts = clustered['cluster'].value_counts()
    rarest = cluster_counts.idxmin()
    explorer.observe(f"Grupo más raro: {rarest} con solo {cluster_counts[rarest]} planetas")

    # ¿Cuál tiene valores más intermedios?
    # (Sin saber que "intermedio" podría ser importante)
    cols = ['pl_eqt', 'pl_rade', 'pl_bmasse']
    available = [c for c in cols if c in clustered.columns]

    # Calcular "centralidad" de cada cluster
    global_medians = {c: df[c].median() for c in available if c in df.columns}

    cluster_centrality = {}
    for c in clustered['cluster'].unique():
        cluster_data = clustered[clustered['cluster'] == c]
        # Distancia a medianas globales
        dist = 0
        for col in available:
            if col in global_medians:
                cluster_mean = cluster_data[col].mean()
                global_med = global_medians[col]
                if global_med > 0:
                    dist += abs(cluster_mean - global_med) / global_med
        cluster_centrality[c] = dist

    most_central = min(cluster_centrality, key=cluster_centrality.get)
    explorer.observe(f"Grupo más central (valores intermedios): {most_central}")

    return most_central


def phase4_locate_solar(df: pd.DataFrame, clustered: pd.DataFrame, explorer: BlindExplorer):
    """Fase 4: ¿Dónde cayeron los planetas del Sol?"""
    print(f"\n  [{explorer.name}] Ubicando planetas del sistema solar...")

    if clustered is None:
        return

    # Encontrar planetas del Sol en los datos clusterizados
    solar_names = ['Sol_p1', 'Sol_p2', 'Sol_p3', 'Sol_p4', 'Sol_p5']
    solar_labels = ['Tierra', 'Venus', 'Marte', 'Mercurio', 'Jupiter']

    # Mapear índices
    solar_indices = df[df['pl_name'].isin(solar_names)].index

    for idx, (name, label) in zip(solar_indices, zip(solar_names, solar_labels)):
        if idx in clustered.index:
            cluster = clustered.loc[idx, 'cluster']
            temp = df.loc[idx, 'pl_eqt']
            explorer.observe(f"{label} (temp={temp:.0f}K) → Grupo {cluster}")


def phase5_correlations(df: pd.DataFrame, explorer: BlindExplorer):
    """Fase 5: ¿Hay correlaciones entre variables?"""
    print(f"\n  [{explorer.name}] Buscando correlaciones...")

    cols = ['pl_eqt', 'pl_rade', 'pl_bmasse', 'pl_orbper', 'st_teff']
    available = [c for c in cols if c in df.columns]

    for i, col1 in enumerate(available):
        for col2 in available[i+1:]:
            valid = df[[col1, col2]].dropna()
            if len(valid) > 20:
                corr = valid[col1].corr(valid[col2])
                if abs(corr) > 0.3:
                    explorer.observe(f"Correlación {col1} vs {col2}: r={corr:.3f}")


def phase6_emergence(clustered: pd.DataFrame, most_central: int, explorer: BlindExplorer):
    """Fase 6: ¿Qué emerge de todo esto?"""
    print(f"\n  [{explorer.name}] Síntesis...")

    if clustered is None or most_central is None:
        explorer.observe("No hay suficientes patrones para sintetizar")
        return

    # ¿Cuántos planetas hay en el grupo central?
    central_count = (clustered['cluster'] == most_central).sum()
    total = len(clustered)
    pct = 100 * central_count / total

    explorer.observe(f"El grupo central contiene {central_count} planetas ({pct:.1f}%)")

    # ¿Qué caracteriza a ese grupo?
    central_data = clustered[clustered['cluster'] == most_central]

    if 'pl_eqt' in central_data.columns:
        temp_range = central_data['pl_eqt'].quantile([0.1, 0.9])
        explorer.observe(f"Rango de temperatura del grupo central: {temp_range.iloc[0]:.0f} - {temp_range.iloc[1]:.0f}")

    if 'pl_rade' in central_data.columns:
        rad_range = central_data['pl_rade'].quantile([0.1, 0.9])
        explorer.observe(f"Rango de radio del grupo central: {rad_range.iloc[0]:.2f} - {rad_range.iloc[1]:.2f}")


def main():
    print("=" * 70)
    print("🔬 DESCUBRIMIENTO REAL - SIN RESPUESTAS PRECONCEBIDAS")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    print("REGLAS:")
    print("  1. NO hay 'zona habitable' hardcodeada")
    print("  2. La Tierra es UN DATO MÁS (Sol_p1)")
    print("  3. Los agentes NO saben qué significa 'habitable'")
    print("  4. Solo ven números y buscan patrones")
    print("=" * 70)

    df = load_data()
    if df is None:
        print("Error cargando datos")
        return

    print(f"\n📊 Datos: {len(df)} planetas (incluyendo 5 del sistema solar)")

    # Crear explorador
    explorer = BlindExplorer("CIEGO")

    # Fases de exploración
    print("\n" + "=" * 70)
    print("FASE 1: ESTADÍSTICAS PURAS")
    print("=" * 70)
    phase1_statistics(df, explorer)

    print("\n" + "=" * 70)
    print("FASE 2: CLUSTERING CIEGO")
    print("=" * 70)
    clustered, cluster_stats = phase2_clustering(df, explorer)

    print("\n" + "=" * 70)
    print("FASE 3: ¿HAY GRUPOS ESPECIALES?")
    print("=" * 70)
    most_central = phase3_find_special(df, clustered, explorer)

    print("\n" + "=" * 70)
    print("FASE 4: ¿DÓNDE CAYÓ CADA PLANETA DEL SOL?")
    print("=" * 70)
    phase4_locate_solar(df, clustered, explorer)

    print("\n" + "=" * 70)
    print("FASE 5: CORRELACIONES")
    print("=" * 70)
    phase5_correlations(df, explorer)

    print("\n" + "=" * 70)
    print("FASE 6: ¿QUÉ EMERGE?")
    print("=" * 70)
    phase6_emergence(clustered, most_central, explorer)

    # ¿Emergió algo parecido a la zona habitable?
    print("\n" + "=" * 70)
    print("🔍 VERIFICACIÓN HONESTA")
    print("=" * 70)

    print("""
    PREGUNTA: ¿Los agentes descubrieron la zona habitable?

    RESPUESTA HONESTA:
    - Encontraron grupos de planetas
    - Ubicaron a la Tierra en algún grupo
    - Detectaron correlaciones

    PERO:
    - El clustering es matemático, no biológico
    - No saben que la "temperatura ideal" existe
    - No pueden inferir "vida" de los números

    CONCLUSIÓN:
    Sin conocimiento previo de bioquímica, termodinámica
    del agua, y condiciones para la vida, los agentes
    NO PUEDEN redescubrir la zona habitable.

    Solo encuentran CLUSTERS ESTADÍSTICOS.

    Para descubrir habitabilidad real necesitarían:
    - Saber que el agua es relevante
    - Conocer puntos de fusión/ebullición
    - Entender atmósferas y presión

    Eso es CONOCIMIENTO PREVIO, no descubrimiento.
    """)

    print("\n" + "=" * 70)
    print("✅ FIN - Análisis honesto completado")
    print("=" * 70)


if __name__ == '__main__':
    main()
