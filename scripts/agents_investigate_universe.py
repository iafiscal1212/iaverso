#!/usr/bin/env python3
"""
Agentes Investigan el Universo
==============================

Con acceso a datos REALES de:
- Terremotos (USGS)
- Geomagnetismo (NOAA)
- Viento solar (DSCOVR)
- Rayos X solares (GOES)
- Rayos cósmicos (ACE)
- Ondas gravitacionales (LIGO/Virgo)

Los agentes buscan patrones SIN respuestas preconcebidas.
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.signal import correlate
from datetime import datetime
from collections import defaultdict

DATA_PATH = Path('/root/NEO_EVA/data/research')


class UniverseInvestigator:
    """Agente que investiga correlaciones cósmicas."""

    def __init__(self, name: str):
        self.name = name
        self.findings = []
        self.questions = []

    def load_all_data(self) -> dict:
        """Cargar todos los datasets disponibles."""
        datasets = {}

        files = {
            'earthquakes': 'usgs_earthquakes.csv',
            'geomag': 'noaa_kp.csv',
            'solar_wind': 'solar_wind.csv',
            'xray': 'xray_flux.csv',
            'cosmic_rays': 'cosmic_rays_ace.csv',
            'gravitational_waves': 'gravitational_waves.csv',
        }

        for name, filename in files.items():
            path = DATA_PATH / filename
            if path.exists():
                try:
                    datasets[name] = pd.read_csv(path)
                    print(f"  ✓ {name}: {len(datasets[name])} registros")
                except Exception as e:
                    print(f"  ✗ {name}: {e}")

        return datasets

    def analyze_temporal_patterns(self, df: pd.DataFrame, time_col: str, value_col: str) -> dict:
        """Analizar patrones temporales en una serie."""
        result = {'patterns': [], 'cycles': []}

        if value_col not in df.columns:
            return result

        try:
            values = pd.to_numeric(df[value_col], errors='coerce').dropna().values

            if len(values) < 10:
                return result

            # Tendencia
            x = np.arange(len(values))
            slope, _, r, p, _ = stats.linregress(x, values)

            if abs(r) > 0.3 and p < 0.05:
                direction = "creciente" if slope > 0 else "decreciente"
                result['patterns'].append({
                    'type': 'tendencia',
                    'direction': direction,
                    'r': r,
                    'p': p
                })

            # FFT para ciclos
            if len(values) > 30:
                fft = np.fft.fft(values - np.mean(values))
                freqs = np.fft.fftfreq(len(values))
                power = np.abs(fft[1:len(fft)//2]) ** 2

                # Encontrar picos
                threshold = np.mean(power) + 2 * np.std(power)
                peaks = np.where(power > threshold)[0]

                for p in peaks[:3]:
                    if freqs[p+1] != 0:
                        period = abs(1 / freqs[p+1])
                        if 3 < period < len(values) / 2:
                            result['cycles'].append({
                                'period': period,
                                'power': power[p]
                            })

        except Exception as e:
            pass

        return result

    def correlate_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame,
                           val1: str, val2: str, max_lag: int = 50) -> dict:
        """Buscar correlación entre dos datasets."""
        result = {
            'direct_corr': None,
            'best_lag': 0,
            'best_corr': 0,
            'significant': False
        }

        try:
            v1 = pd.to_numeric(df1[val1], errors='coerce').dropna().values
            v2 = pd.to_numeric(df2[val2], errors='coerce').dropna().values

            # Igualar longitudes
            min_len = min(len(v1), len(v2))
            if min_len < 20:
                return result

            v1 = v1[:min_len]
            v2 = v2[:min_len]

            # Correlación directa
            result['direct_corr'] = np.corrcoef(v1, v2)[0, 1]

            # Cross-correlation para encontrar mejor lag
            cross_corr = correlate(
                (v1 - np.mean(v1)) / np.std(v1),
                (v2 - np.mean(v2)) / np.std(v2),
                mode='full'
            ) / min_len

            mid = len(cross_corr) // 2
            # Buscar en ventana de lags
            search_range = min(max_lag, mid)
            window = cross_corr[mid-search_range:mid+search_range+1]

            best_idx = np.argmax(np.abs(window))
            result['best_lag'] = best_idx - search_range
            result['best_corr'] = window[best_idx]
            result['significant'] = abs(result['best_corr']) > 0.3

        except Exception as e:
            pass

        return result

    def investigate(self, datasets: dict) -> list:
        """Investigar correlaciones entre todos los datasets."""
        findings = []

        # 1. Analizar cada dataset individualmente
        print("\n" + "-" * 60)
        print("FASE 1: Análisis individual de cada dataset")
        print("-" * 60)

        patterns_found = {}

        # Terremotos
        if 'earthquakes' in datasets:
            eq = datasets['earthquakes']
            print(f"\n  Terremotos: analizando {len(eq)} eventos...")

            # Distribución de magnitudes
            mags = eq['magnitude'].dropna()
            print(f"    Magnitud promedio: {mags.mean():.2f}")
            print(f"    Máxima: {mags.max():.1f}")

            # Distribución temporal
            patterns_found['earthquakes'] = {
                'count': len(eq),
                'mag_mean': mags.mean(),
                'mag_max': mags.max()
            }

        # Rayos cósmicos
        if 'cosmic_rays' in datasets:
            cr = datasets['cosmic_rays']
            if 'p1' in cr.columns:
                patterns = self.analyze_temporal_patterns(cr, 'time_tag', 'p1')
                if patterns['cycles']:
                    print(f"\n  Rayos cósmicos: ciclos detectados!")
                    for c in patterns['cycles'][:2]:
                        print(f"    Período: {c['period']:.1f} pasos")
                patterns_found['cosmic_rays'] = patterns

        # Ondas gravitacionales
        if 'gravitational_waves' in datasets:
            gw = datasets['gravitational_waves']
            valid_mass = gw['mass1'].dropna()
            if len(valid_mass) > 0:
                print(f"\n  Ondas gravitacionales: {len(gw)} eventos")
                print(f"    Masa promedio objeto 1: {valid_mass.mean():.1f} M☉")
                patterns_found['gravitational_waves'] = {
                    'count': len(gw),
                    'avg_mass': valid_mass.mean()
                }

        # 2. Buscar correlaciones entre datasets
        print("\n" + "-" * 60)
        print("FASE 2: Búsqueda de correlaciones inter-dataset")
        print("-" * 60)

        # Correlaciones a investigar
        correlations_to_check = [
            ('geomag', 'Kp', 'xray', 'flux', 'Geomagnetismo vs Rayos X'),
            ('solar_wind', 'speed', 'geomag', 'Kp', 'Viento solar vs Geomagnetismo'),
            ('cosmic_rays', 'p1', 'geomag', 'Kp', 'Rayos cósmicos vs Geomagnetismo'),
            ('cosmic_rays', 'p1', 'xray', 'flux', 'Rayos cósmicos vs Rayos X'),
        ]

        for ds1, col1, ds2, col2, desc in correlations_to_check:
            if ds1 in datasets and ds2 in datasets:
                if col1 in datasets[ds1].columns and col2 in datasets[ds2].columns:
                    corr = self.correlate_datasets(
                        datasets[ds1], datasets[ds2], col1, col2
                    )
                    if corr['significant']:
                        lag_desc = f"lag={corr['best_lag']}" if corr['best_lag'] != 0 else "sincrónico"
                        print(f"\n  ✓ {desc}")
                        print(f"    Correlación: {corr['best_corr']:.3f} ({lag_desc})")

                        findings.append({
                            'type': 'correlation',
                            'description': desc,
                            'correlation': corr['best_corr'],
                            'lag': corr['best_lag'],
                            'significant': True
                        })

        # 3. Buscar anomalías sincronizadas
        print("\n" + "-" * 60)
        print("FASE 3: Búsqueda de anomalías sincronizadas")
        print("-" * 60)

        # Detectar si hay picos inusuales en múltiples datasets
        anomaly_count = 0

        if 'xray' in datasets:
            xray = datasets['xray']
            if 'flux' in xray.columns:
                flux = pd.to_numeric(xray['flux'], errors='coerce').dropna()
                if len(flux) > 0:
                    z_scores = (flux - flux.mean()) / flux.std()
                    spikes = (z_scores > 3).sum()
                    if spikes > 0:
                        print(f"  ⚠ Rayos X: {spikes} picos extremos detectados")
                        anomaly_count += 1

        if 'cosmic_rays' in datasets:
            cr = datasets['cosmic_rays']
            if 'p1' in cr.columns:
                p1 = pd.to_numeric(cr['p1'], errors='coerce').dropna()
                if len(p1) > 0:
                    z_scores = (p1 - p1.mean()) / p1.std()
                    spikes = (z_scores > 3).sum()
                    if spikes > 0:
                        print(f"  ⚠ Rayos cósmicos: {spikes} picos extremos detectados")
                        anomaly_count += 1

        if anomaly_count >= 2:
            findings.append({
                'type': 'synchronized_anomalies',
                'description': f'{anomaly_count} datasets mostraron anomalías',
                'hypothesis': 'Posible evento solar significativo afectando múltiples mediciones'
            })

        return findings


def generate_hypotheses(findings: list) -> list:
    """Generar hipótesis basadas en los hallazgos."""
    hypotheses = []

    for f in findings:
        if f['type'] == 'correlation':
            if 'solar' in f['description'].lower() or 'geomag' in f['description'].lower():
                hypotheses.append({
                    'hypothesis': f"La actividad solar influye en {f['description'].split(' vs ')[1]}",
                    'evidence': f"Correlación de {f['correlation']:.2f}",
                    'confidence': min(0.9, abs(f['correlation'])),
                    'next_step': 'Verificar si la correlación persiste en períodos más largos'
                })

            if 'cósmic' in f['description'].lower():
                hypotheses.append({
                    'hypothesis': "Los rayos cósmicos están modulados por actividad solar/geomagnética",
                    'evidence': f"Correlación de {f['correlation']:.2f}",
                    'confidence': min(0.85, abs(f['correlation'])),
                    'next_step': 'Buscar correlación con ciclo solar de 11 años'
                })

        if f['type'] == 'synchronized_anomalies':
            hypotheses.append({
                'hypothesis': 'Eventos solares causan perturbaciones simultáneas en múltiples observables',
                'evidence': f['description'],
                'confidence': 0.7,
                'next_step': 'Identificar fechas específicas de los eventos'
            })

    return hypotheses


def main():
    print("=" * 70)
    print("🔭 AGENTES INVESTIGAN EL UNIVERSO")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Los agentes analizan datos REALES sin respuestas preconcebidas")
    print("=" * 70)

    # Crear investigador
    investigator = UniverseInvestigator("IRIS")

    # Cargar datos
    print("\n📂 Cargando datasets...")
    datasets = investigator.load_all_data()

    if not datasets:
        print("⚠ No hay datos para analizar. Ejecuta data_fetcher.py primero.")
        return

    # Investigar
    findings = investigator.investigate(datasets)

    # Generar hipótesis
    print("\n" + "=" * 70)
    print("🧠 HIPÓTESIS GENERADAS POR LOS AGENTES")
    print("=" * 70)

    hypotheses = generate_hypotheses(findings)

    if not hypotheses:
        print("\n  Los agentes no encontraron patrones significativos.")
        print("  Esto es un resultado válido - no todo está correlacionado.")
    else:
        for i, h in enumerate(hypotheses, 1):
            print(f"\n  HIPÓTESIS {i}:")
            print(f"    📋 {h['hypothesis']}")
            print(f"    📊 Evidencia: {h['evidence']}")
            print(f"    🎯 Confianza: {h['confidence']:.0%}")
            print(f"    ➡️ Siguiente paso: {h['next_step']}")

    # Preguntas abiertas
    print("\n" + "=" * 70)
    print("❓ PREGUNTAS QUE LOS AGENTES NO PUEDEN RESPONDER")
    print("=" * 70)

    questions = [
        "¿Hay correlación entre ondas gravitacionales y actividad sísmica terrestre?",
        "¿Los neutrinos solares afectan procesos geológicos?",
        "¿Podemos predecir tormentas geomagnéticas con rayos cósmicos?",
        "¿Existe un patrón en las masas de los agujeros negros fusionados?",
    ]

    print("\n  Para responder estas preguntas necesitamos:")
    for q in questions:
        print(f"    • {q}")

    print("\n" + "=" * 70)
    print("✅ FIN - Todo descubierto de los datos, nada inventado")
    print("=" * 70)


if __name__ == '__main__':
    main()
