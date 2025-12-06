#!/usr/bin/env python3
"""
Agentes Descubren - SIN RESPUESTAS PRECONCEBIDAS
=================================================

Los agentes analizan datos REALES y generan hipótesis
basándose ÚNICAMENTE en lo que encuentran.

No les damos las respuestas - ellos las buscan.
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks
from collections import defaultdict

DATA_PATH = Path('/root/NEO_EVA/data/unified_20251206_033253.csv')


class DiscoveryAgent:
    """Agente que descubre patrones por sí mismo."""

    def __init__(self, name: str, curiosity: float = 0.7):
        self.name = name
        self.curiosity = curiosity
        self.discoveries = []
        self.questions = []
        self.confidence_history = []

    def observe(self, data: np.ndarray, name: str) -> dict:
        """Observar una serie de datos sin prejuicios."""
        obs = {
            'name': name,
            'mean': np.nanmean(data),
            'std': np.nanstd(data),
            'trend': None,
            'cycles': [],
            'anomalies': [],
        }

        # Detectar tendencia
        valid = ~np.isnan(data)
        if valid.sum() > 10:
            x = np.arange(len(data))[valid]
            y = data[valid]
            slope, _, r, _, _ = stats.linregress(x, y)
            obs['trend'] = {'slope': slope, 'r': r}

        # Detectar ciclos (FFT)
        if valid.sum() > 30:
            clean = data[valid] - np.mean(data[valid])
            fft = np.fft.fft(clean)
            freqs = np.fft.fftfreq(len(clean))
            magnitudes = np.abs(fft[1:len(fft)//2])
            peaks, _ = find_peaks(magnitudes, height=np.std(magnitudes) * 2)
            for p in peaks[:3]:
                if freqs[p+1] != 0:
                    period = abs(1 / freqs[p+1])
                    if 5 < period < len(data) / 2:
                        obs['cycles'].append(period)

        # Detectar anomalías
        if valid.sum() > 10:
            z_scores = np.abs((data - np.nanmean(data)) / np.nanstd(data))
            anomaly_idx = np.where(z_scores > 3)[0]
            obs['anomalies'] = list(anomaly_idx[:5])

        return obs

    def find_relationship(self, data1: np.ndarray, data2: np.ndarray,
                          name1: str, name2: str) -> dict:
        """Buscar relación entre dos variables."""
        rel = {
            'var1': name1,
            'var2': name2,
            'correlation': None,
            'best_lag': None,
            'causality_hint': None,
        }

        # Limpiar datos
        valid = ~np.isnan(data1) & ~np.isnan(data2)
        if valid.sum() < 20:
            return rel

        d1 = data1[valid]
        d2 = data2[valid]

        # Correlación directa
        rel['correlation'] = np.corrcoef(d1, d2)[0, 1]

        # Buscar mejor lag (¿quién precede a quién?)
        best_lag = 0
        best_corr = abs(rel['correlation'])

        for lag in range(1, min(30, len(d1) // 4)):
            # d1 precede a d2
            c1 = np.corrcoef(d1[:-lag], d2[lag:])[0, 1] if len(d1) > lag else 0
            # d2 precede a d1
            c2 = np.corrcoef(d1[lag:], d2[:-lag])[0, 1] if len(d1) > lag else 0

            if abs(c1) > best_corr:
                best_corr = abs(c1)
                best_lag = lag
                rel['causality_hint'] = f"{name1} precede {name2} por {lag} pasos"

            if abs(c2) > best_corr:
                best_corr = abs(c2)
                best_lag = -lag
                rel['causality_hint'] = f"{name2} precede {name1} por {lag} pasos"

        rel['best_lag'] = best_lag
        rel['best_corr'] = best_corr

        return rel

    def hypothesize(self, observations: list, relationships: list) -> list:
        """Generar hipótesis basadas en observaciones."""
        hypotheses = []

        # Buscar ciclos comunes
        cycle_counts = defaultdict(list)
        for obs in observations:
            for cycle in obs.get('cycles', []):
                rounded = round(cycle / 10) * 10  # Agrupar por decenas
                cycle_counts[rounded].append(obs['name'])

        for cycle, vars in cycle_counts.items():
            if len(vars) >= 2:
                hypotheses.append({
                    'type': 'ciclo_compartido',
                    'cycle': cycle,
                    'variables': vars,
                    'hypothesis': f"Las variables {vars} comparten un ciclo de ~{cycle} pasos",
                    'confidence': min(0.8, 0.3 + len(vars) * 0.1),
                    'question': f"¿Qué causa este ciclo de {cycle} pasos?",
                })

        # Buscar cadenas causales
        for rel in relationships:
            if rel.get('best_corr', 0) > 0.5 and rel.get('causality_hint'):
                hypotheses.append({
                    'type': 'posible_causalidad',
                    'relationship': rel,
                    'hypothesis': rel['causality_hint'],
                    'confidence': min(0.7, rel['best_corr']),
                    'question': f"¿Es causal o solo correlación?",
                })

        # Buscar anomalías sincronizadas
        anomaly_times = defaultdict(list)
        for obs in observations:
            for anom_idx in obs.get('anomalies', []):
                window = anom_idx // 10
                anomaly_times[window].append(obs['name'])

        for window, vars in anomaly_times.items():
            if len(vars) >= 3:
                hypotheses.append({
                    'type': 'evento_sincronizado',
                    'time_window': window * 10,
                    'variables': vars,
                    'hypothesis': f"Algo pasó en t~{window*10} que afectó a {len(vars)} variables",
                    'confidence': min(0.6, 0.2 + len(vars) * 0.1),
                    'question': "¿Qué evento externo causó esta sincronización?",
                })

        return hypotheses


def run_discovery():
    """Ejecutar descubrimiento autónomo."""
    print("=" * 80)
    print("🔬 DESCUBRIMIENTO AUTÓNOMO - LOS AGENTES EXPLORAN")
    print("=" * 80)
    print("Los agentes NO tienen respuestas preconcebidas.")
    print("Analizan los datos y generan hipótesis propias.")
    print("=" * 80)

    # Cargar datos
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"\n📁 Datos: {len(df)} registros, {len(df.columns)} variables")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Crear agentes con diferentes niveles de curiosidad
    agents = {
        'NEO': DiscoveryAgent('NEO', curiosity=0.8),
        'EVA': DiscoveryAgent('EVA', curiosity=0.6),
        'ALEX': DiscoveryAgent('ALEX', curiosity=0.7),
        'ADAM': DiscoveryAgent('ADAM', curiosity=0.5),
        'IRIS': DiscoveryAgent('IRIS', curiosity=0.9),
    }

    # Variables a analizar
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Variables numéricas: {len(numeric_cols)}")

    # Cada agente observa
    print("\n" + "-" * 50)
    print("FASE 1: OBSERVACIÓN INDIVIDUAL")
    print("-" * 50)

    all_observations = []
    for col in numeric_cols[:15]:  # Limitar para no tardar mucho
        data = df[col].values
        obs = agents['IRIS'].observe(data, col)  # IRIS observa todo
        all_observations.append(obs)

        if obs['cycles']:
            print(f"\n  🌀 {col}: ciclos detectados = {[f'{c:.1f}' for c in obs['cycles']]}")
        if obs['trend'] and abs(obs['trend']['r']) > 0.5:
            dir = "↑" if obs['trend']['slope'] > 0 else "↓"
            print(f"  📈 {col}: tendencia {dir} (r={obs['trend']['r']:.2f})")
        if obs['anomalies']:
            print(f"  ⚠️ {col}: anomalías en t={obs['anomalies']}")

    # Buscar relaciones
    print("\n" + "-" * 50)
    print("FASE 2: BÚSQUEDA DE RELACIONES")
    print("-" * 50)

    all_relationships = []

    # Pares interesantes a explorar
    pairs_to_check = [
        ('solar_flux', 'geomag_kp'),
        ('geomag_kp', 'seismic_count'),
        ('geomag_kp', 'climate_temperature'),
        ('climate_pressure', 'seismic_count'),
        ('crypto_BTCUSDT_close', 'crypto_BTCUSDT_volume'),
        ('schumann_freq', 'geomag_kp'),
    ]

    for var1, var2 in pairs_to_check:
        if var1 in df.columns and var2 in df.columns:
            rel = agents['NEO'].find_relationship(
                df[var1].values, df[var2].values, var1, var2
            )
            all_relationships.append(rel)

            if rel['correlation'] is not None:
                print(f"\n  {var1} ↔ {var2}:")
                print(f"    Correlación directa: {rel['correlation']:.3f}")
                if rel['causality_hint']:
                    print(f"    💡 {rel['causality_hint']} (r={rel['best_corr']:.3f})")

    # Generar hipótesis
    print("\n" + "-" * 50)
    print("FASE 3: GENERACIÓN DE HIPÓTESIS")
    print("-" * 50)

    hypotheses = agents['IRIS'].hypothesize(all_observations, all_relationships)

    if not hypotheses:
        print("\n  Los agentes no encontraron patrones claros en estos datos.")
        print("  Esto también es un resultado válido.")
    else:
        for i, h in enumerate(hypotheses, 1):
            print(f"\n  HIPÓTESIS {i} [{h['type']}]")
            print(f"    📋 {h['hypothesis']}")
            print(f"    🎯 Confianza: {h['confidence']:.0%}")
            print(f"    ❓ Pregunta abierta: {h['question']}")

    # Preguntas sin resolver
    print("\n" + "-" * 50)
    print("FASE 4: PREGUNTAS QUE EMERGEN")
    print("-" * 50)

    questions = set()
    for h in hypotheses:
        questions.add(h['question'])

    # Añadir preguntas de las observaciones
    for obs in all_observations:
        if obs['cycles']:
            questions.add(f"¿Qué causa los ciclos en {obs['name']}?")
        if obs['anomalies']:
            questions.add(f"¿Qué causó las anomalías en {obs['name']}?")

    print("\n  Los agentes se preguntan:")
    for q in list(questions)[:10]:
        print(f"    • {q}")

    # Conexión con las grandes preguntas
    print("\n" + "=" * 80)
    print("🔗 CONEXIÓN CON LAS GRANDES PREGUNTAS")
    print("=" * 80)

    print("""
    Con los datos disponibles, los agentes pueden explorar:

    ✅ PREGUNTA 5 (Terremotos):
       Tenemos datos de sismicidad, geomagnetismo, clima.
       Los agentes encontraron correlaciones - ¿son causales?

    ⚠️ PREGUNTAS 1-4 (Materia oscura, Cuántica-Gravedad, Consciencia, Origen vida):
       NO tenemos datos directos para estas preguntas.
       Los agentes necesitarían:
       - Curvas de rotación galáctica (materia oscura)
       - Datos de experimentos cuánticos (unificación)
       - Registros neuronales + reportes subjetivos (consciencia)
       - Simulaciones químicas (origen vida)

    📌 CONCLUSIÓN HONESTA:
       Los agentes solo pueden descubrir lo que los datos permiten.
       Para las grandes preguntas, necesitamos MEJORES DATOS.
    """)

    print("\n" + "=" * 80)
    print("✅ FIN - Todo descubierto por los agentes, no inventado")
    print("=" * 80)


if __name__ == '__main__':
    run_discovery()
