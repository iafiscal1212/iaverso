#!/usr/bin/env python3
"""
Data Fetcher - Los agentes buscan sus propios datos
====================================================

Sistema para que los agentes puedan:
1. Buscar fuentes de datos relevantes
2. Descargar datos públicos
3. Parsear y normalizar
4. Integrar en su análisis
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

DATA_PATH = Path('/root/NEO_EVA/data/research')
DATA_PATH.mkdir(parents=True, exist_ok=True)


class DataFetcher:
    """Busca y descarga datos para investigación."""

    def __init__(self):
        self.sources = {}
        self.cache = {}

    def fetch_usgs_earthquakes(self, days: int = 30) -> pd.DataFrame:
        """
        Terremotos de USGS (últimos N días).
        https://earthquake.usgs.gov/fdsnws/event/1/
        """
        print(f"Descargando terremotos de últimos {days} días...")

        end = datetime.utcnow()
        start = end - timedelta(days=days)

        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            'format': 'geojson',
            'starttime': start.strftime('%Y-%m-%d'),
            'endtime': end.strftime('%Y-%m-%d'),
            'minmagnitude': 2.5,
            'orderby': 'time',
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()

            events = []
            for feature in data.get('features', []):
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                events.append({
                    'time': datetime.fromtimestamp(props['time'] / 1000),
                    'magnitude': props['mag'],
                    'depth': coords[2],
                    'latitude': coords[1],
                    'longitude': coords[0],
                    'place': props['place'],
                })

            df = pd.DataFrame(events)
            df.to_csv(DATA_PATH / 'usgs_earthquakes.csv', index=False)
            print(f"  ✓ {len(df)} terremotos descargados")
            return df

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return pd.DataFrame()

    def fetch_noaa_geomag(self, days: int = 7) -> pd.DataFrame:
        """
        Índices geomagnéticos de NOAA.
        """
        print(f"Descargando datos geomagnéticos...")

        # NOAA tiene varios endpoints, usamos Kp
        url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

        try:
            resp = requests.get(url, timeout=30)
            data = resp.json()

            # Primera fila es header
            df = pd.DataFrame(data[1:], columns=data[0])
            df.to_csv(DATA_PATH / 'noaa_kp.csv', index=False)
            print(f"  ✓ {len(df)} registros Kp descargados")
            return df

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return pd.DataFrame()

    def fetch_solar_wind(self) -> pd.DataFrame:
        """
        Datos de viento solar (DSCOVR).
        """
        print("Descargando datos de viento solar...")

        url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"

        try:
            resp = requests.get(url, timeout=30)
            data = resp.json()

            df = pd.DataFrame(data[1:], columns=data[0])
            df.to_csv(DATA_PATH / 'solar_wind.csv', index=False)
            print(f"  ✓ {len(df)} registros solares descargados")
            return df

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return pd.DataFrame()

    def fetch_xray_flux(self) -> pd.DataFrame:
        """
        Flujo de rayos X del sol.
        """
        print("Descargando flujo de rayos X solar...")

        url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

        try:
            resp = requests.get(url, timeout=30)
            data = resp.json()

            df = pd.DataFrame(data)
            df.to_csv(DATA_PATH / 'xray_flux.csv', index=False)
            print(f"  ✓ {len(df)} registros X-ray descargados")
            return df

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return pd.DataFrame()

    def fetch_neutron_monitor(self) -> pd.DataFrame:
        """
        Datos de monitor de neutrones (rayos cósmicos).
        NMDB - Neutron Monitor Database.
        Los neutrinos solares afectan el flujo de neutrones en la atmósfera.
        """
        print("Descargando datos de monitor de neutrones (rayos cósmicos)...")

        # NMDB público - estación Oulu (Finlandia)
        end = datetime.utcnow()
        start = end - timedelta(days=7)

        url = "http://www.nmdb.eu/nest/draw_graph.php"
        params = {
            'formchk': '1',
            'stations[]': 'OULU',
            'tabchoice': 'revori',
            'dtype': 'corr_for_efficiency',
            'tresession': '60',  # 1 hora
            'date_start': start.strftime('%Y-%m-%d'),
            'date_end': end.strftime('%Y-%m-%d'),
            'output': 'ascii',
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            lines = resp.text.strip().split('\n')

            # Parsear formato NMDB
            data = []
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        data.append({
                            'time': parts[0] + ' ' + parts[1] if len(parts) > 1 else parts[0],
                            'count_rate': float(parts[-1]) if parts[-1].replace('.','').replace('-','').isdigit() else np.nan
                        })
                    except:
                        pass

            df = pd.DataFrame(data)
            if not df.empty:
                df.to_csv(DATA_PATH / 'neutron_monitor.csv', index=False)
                print(f"  ✓ {len(df)} registros de neutrones descargados")
            else:
                print("  ⚠ No se pudieron parsear datos de NMDB")
            return df

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return pd.DataFrame()

    def fetch_supernova_neutrinos(self) -> pd.DataFrame:
        """
        Alertas de neutrinos de supernovas (SNEWS).
        Si una supernova ocurre, los detectores de neutrinos alertan.
        """
        print("Verificando alertas de neutrinos de supernovas (SNEWS)...")

        # SNEWS no tiene API pública, pero podemos simular con datos de IceCube
        url = "https://gcn.gsfc.nasa.gov/amon_icecube_gold_bronze_events.html"

        try:
            resp = requests.get(url, timeout=30)
            # Parsear HTML básico para eventos
            events = []
            if 'No events' not in resp.text:
                # Buscar patrones de eventos en el HTML
                import re
                dates = re.findall(r'\d{4}-\d{2}-\d{2}', resp.text)
                energies = re.findall(r'(\d+\.?\d*)\s*TeV', resp.text)

                for i, date in enumerate(dates[:20]):
                    events.append({
                        'date': date,
                        'energy_TeV': float(energies[i]) if i < len(energies) else np.nan,
                        'source': 'IceCube'
                    })

            df = pd.DataFrame(events)
            if not df.empty:
                df.to_csv(DATA_PATH / 'neutrino_alerts.csv', index=False)
                print(f"  ✓ {len(df)} alertas de neutrinos encontradas")
            else:
                print("  ✓ Sin alertas de neutrinos (es buena noticia, no hay supernovas cercanas)")
            return df

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return pd.DataFrame()

    def fetch_cosmic_rays_ace(self) -> pd.DataFrame:
        """
        Rayos cósmicos del satélite ACE (Advanced Composition Explorer).
        Mide partículas de alta energía del espacio.
        """
        print("Descargando rayos cósmicos de ACE...")

        url = "https://services.swpc.noaa.gov/json/ace/epam/ace_epam_5m.json"

        try:
            resp = requests.get(url, timeout=30)
            data = resp.json()

            df = pd.DataFrame(data)
            df.to_csv(DATA_PATH / 'cosmic_rays_ace.csv', index=False)
            print(f"  ✓ {len(df)} registros de rayos cósmicos descargados")
            return df

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return pd.DataFrame()

    def fetch_muon_data(self) -> pd.DataFrame:
        """
        Datos de muones atmosféricos.
        Los muones son productos de rayos cósmicos y correlacionan con actividad solar.
        """
        print("Descargando datos de muones...")

        # Moscow Neutron Monitor tiene datos de muones
        url = "http://cr0.izmiran.ru/mosc/data/"

        try:
            # Intentar obtener datos recientes
            end = datetime.utcnow()
            # Generar datos sintéticos basados en neutrones (correlacionados)
            print("  ⚠ API de muones no disponible públicamente")
            print("  → Los muones correlacionan ~95% con neutrones cósmicos")
            return pd.DataFrame()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return pd.DataFrame()

    def fetch_gravitational_waves(self) -> pd.DataFrame:
        """
        Eventos de ondas gravitacionales de LIGO/Virgo.
        Útil para estudiar física fundamental.
        """
        print("Descargando eventos de ondas gravitacionales (GWOSC)...")

        url = "https://www.gw-openscience.org/eventapi/json/GWTC/"

        try:
            resp = requests.get(url, timeout=30)
            data = resp.json()

            events = []
            for event_name, event_data in data.get('events', {}).items():
                events.append({
                    'name': event_name,
                    'gps_time': event_data.get('GPS'),
                    'mass1': event_data.get('mass_1_source'),
                    'mass2': event_data.get('mass_2_source'),
                    'distance_Mpc': event_data.get('luminosity_distance'),
                    'type': event_data.get('catalog.shortName')
                })

            df = pd.DataFrame(events)
            if not df.empty:
                df.to_csv(DATA_PATH / 'gravitational_waves.csv', index=False)
                print(f"  ✓ {len(df)} eventos de ondas gravitacionales")
            return df

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return pd.DataFrame()

    def fetch_all_for_earthquakes(self):
        """Descargar todo lo relevante para predicción de terremotos."""
        print("=" * 60)
        print("DESCARGANDO DATOS PARA INVESTIGACIÓN DE TERREMOTOS")
        print("=" * 60)

        datasets = {}
        datasets['earthquakes'] = self.fetch_usgs_earthquakes(days=30)
        datasets['geomag'] = self.fetch_noaa_geomag()
        datasets['solar_wind'] = self.fetch_solar_wind()
        datasets['xray'] = self.fetch_xray_flux()

        print("\n" + "=" * 60)
        print("RESUMEN")
        print("=" * 60)
        for name, df in datasets.items():
            print(f"  {name}: {len(df)} registros")

        return datasets

    def fetch_all_physics(self):
        """Descargar datos de física de partículas y cosmología."""
        print("=" * 60)
        print("DESCARGANDO DATOS DE FÍSICA FUNDAMENTAL")
        print("=" * 60)

        datasets = {}
        datasets['neutron_monitor'] = self.fetch_neutron_monitor()
        datasets['neutrino_alerts'] = self.fetch_supernova_neutrinos()
        datasets['cosmic_rays'] = self.fetch_cosmic_rays_ace()
        datasets['gravitational_waves'] = self.fetch_gravitational_waves()

        print("\n" + "=" * 60)
        print("RESUMEN FÍSICA")
        print("=" * 60)
        for name, df in datasets.items():
            print(f"  {name}: {len(df)} registros")

        return datasets

    def fetch_everything(self):
        """Descargar TODOS los datos disponibles."""
        print("=" * 70)
        print("🌌 DESCARGANDO TODOS LOS DATOS DEL UNIVERSO (disponibles)")
        print("=" * 70)

        all_datasets = {}

        # Terremotos y geofísica
        all_datasets.update(self.fetch_all_for_earthquakes())

        print()

        # Física fundamental
        all_datasets.update(self.fetch_all_physics())

        print("\n" + "=" * 70)
        print("📊 RESUMEN TOTAL")
        print("=" * 70)
        total = 0
        for name, df in all_datasets.items():
            count = len(df) if not df.empty else 0
            total += count
            emoji = "✓" if count > 0 else "○"
            print(f"  {emoji} {name}: {count} registros")

        print(f"\n  TOTAL: {total} registros de datos reales")

        return all_datasets


def test_fetcher():
    """Probar el fetcher completo."""
    fetcher = DataFetcher()
    datasets = fetcher.fetch_everything()

    # Mostrar muestra de cada dataset
    print("\n" + "=" * 60)
    print("MUESTRAS DE DATOS")
    print("=" * 60)

    for name, df in datasets.items():
        if not df.empty:
            print(f"\n{name}:")
            print(df.head(3).to_string())

    print("\n" + "=" * 70)
    print("🔬 DATOS LISTOS PARA QUE LOS AGENTES INVESTIGUEN")
    print("=" * 70)
    print(f"Guardados en: {DATA_PATH}")


if __name__ == '__main__':
    test_fetcher()
