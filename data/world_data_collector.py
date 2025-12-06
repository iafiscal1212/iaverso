#!/usr/bin/env python3
"""
World Data Collector - Datos del Mundo Real
============================================
Recolecta datos de múltiples dominios para que el agente descubra estructura.

Dominios:
- Cripto (Binance)
- Solar/Geomagnético (NOAA)
- Sismos (USGS)
- Clima (Open-Meteo)

Todo guardado en formato unificado para análisis cruzado.
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Directorio de datos
DATA_DIR = Path('/root/NEO_EVA/data')

# =============================================================================
# BINANCE - Cripto
# =============================================================================

class BinanceCollector:
    """Recolector de datos de Binance."""

    BASE_URL = "https://api.binance.com/api/v3"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({'X-MBX-APIKEY': api_key})

    def get_klines(self, symbol: str, interval: str = '1h', limit: int = 500) -> pd.DataFrame:
        """
        Obtiene velas (OHLCV) de un par.

        interval: 1m, 5m, 15m, 1h, 4h, 1d
        """
        url = f"{self.BASE_URL}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        try:
            resp = self.session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df['symbol'] = symbol

            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                df[col] = df[col].astype(float)

            df['trades'] = df['trades'].astype(int)

            return df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'trades']]

        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return pd.DataFrame()

    def get_multi_symbols(self, symbols: List[str], interval: str = '1h', limit: int = 500) -> pd.DataFrame:
        """Obtiene datos de múltiples símbolos."""
        dfs = []
        for symbol in symbols:
            df = self.get_klines(symbol, interval, limit)
            if not df.empty:
                dfs.append(df)
            time.sleep(0.1)  # Rate limiting

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()


# =============================================================================
# NOAA - Solar y Geomagnético
# =============================================================================

class NOAACollector:
    """Recolector de datos solares y geomagnéticos de NOAA."""

    def get_solar_wind(self) -> pd.DataFrame:
        """
        Datos de viento solar en tiempo real (últimas 24h).
        Velocidad, densidad, temperatura del plasma solar.
        """
        url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"

        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()

            # Primera fila es header
            df = pd.DataFrame(data[1:], columns=data[0])
            df['timestamp'] = pd.to_datetime(df['time_tag'])

            for col in ['density', 'speed', 'temperature']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df[['timestamp', 'density', 'speed', 'temperature']].dropna()

        except Exception as e:
            logger.error(f"Error getting solar wind: {e}")
            return pd.DataFrame()

    def get_geomagnetic_index(self) -> pd.DataFrame:
        """
        Índice Kp (actividad geomagnética global).
        Kp 0-3: quieto, 4: activo, 5+: tormenta
        """
        url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()

            df = pd.DataFrame(data[1:], columns=data[0])
            df['timestamp'] = pd.to_datetime(df['time_tag'])
            df['kp'] = pd.to_numeric(df['Kp'], errors='coerce')

            return df[['timestamp', 'kp']].dropna()

        except Exception as e:
            logger.error(f"Error getting Kp index: {e}")
            return pd.DataFrame()

    def get_xray_flux(self) -> pd.DataFrame:
        """
        Flujo de rayos X solares (indica llamaradas).
        """
        url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()

            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['time_tag'])
            df['xray_flux'] = pd.to_numeric(df['flux'], errors='coerce')

            return df[['timestamp', 'xray_flux']].dropna()

        except Exception as e:
            logger.error(f"Error getting X-ray flux: {e}")
            return pd.DataFrame()


# =============================================================================
# USGS - Sismos
# =============================================================================

class USGSCollector:
    """Recolector de datos sísmicos de USGS."""

    def get_earthquakes(self, days: int = 7, min_magnitude: float = 2.5) -> pd.DataFrame:
        """
        Sismos de los últimos N días.
        """
        end = datetime.utcnow()
        start = end - timedelta(days=days)

        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            'format': 'geojson',
            'starttime': start.isoformat(),
            'endtime': end.isoformat(),
            'minmagnitude': min_magnitude,
            'orderby': 'time'
        }

        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            events = []
            for feature in data['features']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                events.append({
                    'timestamp': pd.to_datetime(props['time'], unit='ms'),
                    'magnitude': props['mag'],
                    'depth': coords[2],
                    'latitude': coords[1],
                    'longitude': coords[0],
                    'place': props.get('place', ''),
                })

            return pd.DataFrame(events)

        except Exception as e:
            logger.error(f"Error getting earthquakes: {e}")
            return pd.DataFrame()


# =============================================================================
# Open-Meteo - Clima
# =============================================================================

class ClimateCollector:
    """Recolector de datos climáticos de Open-Meteo."""

    def get_weather(self, lat: float = 40.4168, lon: float = -3.7038,
                    days_back: int = 7) -> pd.DataFrame:
        """
        Datos meteorológicos históricos.
        Default: Madrid
        """
        end = datetime.utcnow().date()
        start = end - timedelta(days=days_back)

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start.isoformat(),
            'end_date': end.isoformat(),
            'hourly': 'temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,precipitation',
            'timezone': 'UTC'
        }

        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            hourly = data['hourly']
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(hourly['time']),
                'temperature': hourly['temperature_2m'],
                'humidity': hourly['relative_humidity_2m'],
                'pressure': hourly['pressure_msl'],
                'wind_speed': hourly['wind_speed_10m'],
                'precipitation': hourly['precipitation'],
            })

            return df.dropna()

        except Exception as e:
            logger.error(f"Error getting weather: {e}")
            return pd.DataFrame()


# =============================================================================
# Collector Unificado
# =============================================================================

class WorldDataCollector:
    """
    Recolector unificado de todos los dominios.
    """

    def __init__(self, binance_key: str, binance_secret: str):
        self.binance = BinanceCollector(binance_key, binance_secret)
        self.noaa = NOAACollector()
        self.usgs = USGSCollector()
        self.climate = ClimateCollector()

        self.data_dir = DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def collect_all(self, crypto_symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Recolecta datos de todos los dominios.
        """
        if crypto_symbols is None:
            crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']

        results = {}

        # Cripto
        logger.info("Collecting crypto data...")
        results['crypto'] = self.binance.get_multi_symbols(crypto_symbols, interval='1h', limit=168)  # 7 días

        # Solar
        logger.info("Collecting solar data...")
        results['solar_wind'] = self.noaa.get_solar_wind()
        results['geomagnetic'] = self.noaa.get_geomagnetic_index()
        results['xray'] = self.noaa.get_xray_flux()

        # Sismos
        logger.info("Collecting seismic data...")
        results['earthquakes'] = self.usgs.get_earthquakes(days=7)

        # Clima
        logger.info("Collecting climate data...")
        results['climate'] = self.climate.get_weather(days_back=7)

        return results

    def save_all(self, data: Dict[str, pd.DataFrame], prefix: str = None):
        """
        Guarda todos los datos.
        """
        if prefix is None:
            prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

        for name, df in data.items():
            if df is not None and not df.empty:
                path = self.data_dir / f"{name}_{prefix}.csv"
                df.to_csv(path, index=False)
                logger.info(f"Saved {name}: {len(df)} rows -> {path}")

    def create_unified_timeseries(self, data: Dict[str, pd.DataFrame],
                                   freq: str = '1H') -> pd.DataFrame:
        """
        Crea una serie temporal unificada con todas las variables.
        Resamplea todo a la misma frecuencia.
        """
        # Determinar rango temporal común
        all_timestamps = []
        for name, df in data.items():
            if df is not None and not df.empty and 'timestamp' in df.columns:
                # Normalizar a UTC naive
                ts = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                all_timestamps.extend(ts.tolist())

        if not all_timestamps:
            return pd.DataFrame()

        min_ts = min(all_timestamps)
        max_ts = max(all_timestamps)

        # Crear índice temporal
        idx = pd.date_range(start=min_ts, end=max_ts, freq=freq)
        unified = pd.DataFrame(index=idx)
        unified.index.name = 'timestamp'

        # Cripto - pivotear por símbolo
        if 'crypto' in data and not data['crypto'].empty:
            crypto = data['crypto'].copy()
            crypto['timestamp'] = pd.to_datetime(crypto['timestamp']).dt.tz_localize(None)
            for symbol in crypto['symbol'].unique():
                sym_data = crypto[crypto['symbol'] == symbol].set_index('timestamp')
                for col in ['close', 'volume']:
                    col_name = f"crypto_{symbol}_{col}"
                    unified[col_name] = sym_data[col].reindex(idx, method='ffill')

        # Solar wind
        if 'solar_wind' in data and not data['solar_wind'].empty:
            sw = data['solar_wind'].copy()
            sw['timestamp'] = pd.to_datetime(sw['timestamp']).dt.tz_localize(None)
            sw = sw.set_index('timestamp')
            for col in ['density', 'speed', 'temperature']:
                unified[f"solar_{col}"] = sw[col].reindex(idx, method='ffill')

        # Geomagnético
        if 'geomagnetic' in data and not data['geomagnetic'].empty:
            geo = data['geomagnetic'].copy()
            geo['timestamp'] = pd.to_datetime(geo['timestamp']).dt.tz_localize(None)
            geo = geo.set_index('timestamp')
            unified['geomag_kp'] = geo['kp'].reindex(idx, method='ffill')

        # X-ray
        if 'xray' in data and not data['xray'].empty:
            xr = data['xray'].copy()
            xr['timestamp'] = pd.to_datetime(xr['timestamp']).dt.tz_localize(None)
            xr = xr.drop_duplicates('timestamp').set_index('timestamp')
            unified['solar_xray'] = xr['xray_flux'].reindex(idx, method='ffill')

        # Sismos - contar por hora
        if 'earthquakes' in data and not data['earthquakes'].empty:
            eq = data['earthquakes'].copy()
            eq['timestamp'] = pd.to_datetime(eq['timestamp']).dt.tz_localize(None)
            eq['hour'] = eq['timestamp'].dt.floor('H')
            eq_count = eq.groupby('hour').size()
            eq_mag = eq.groupby('hour')['magnitude'].max()
            unified['seismic_count'] = eq_count.reindex(idx).fillna(0)
            unified['seismic_max_mag'] = eq_mag.reindex(idx).fillna(0)

        # Clima
        if 'climate' in data and not data['climate'].empty:
            clim = data['climate'].copy()
            clim['timestamp'] = pd.to_datetime(clim['timestamp']).dt.tz_localize(None)
            clim = clim.set_index('timestamp')
            for col in ['temperature', 'humidity', 'pressure', 'wind_speed']:
                unified[f"climate_{col}"] = clim[col].reindex(idx, method='ffill')

        return unified.dropna(how='all')


# =============================================================================
# Main
# =============================================================================

def main():
    """Test de recolección."""

    # Credenciales Binance
    API_KEY = "xSJwtN2YqTa8ONYblN1iLrpU72bthf0bYg0hGWOPD1THmgPSgiX2Epqy0qaKJ1fJ"
    API_SECRET = "owWXUN3Zc8rr4Z1cVc4SOA2sGwjws0ouqCdGWp2XhNjwKKiCNo3wQqgvu3h3uAe3"

    collector = WorldDataCollector(API_KEY, API_SECRET)

    print("=" * 70)
    print("WORLD DATA COLLECTOR")
    print("=" * 70)

    # Recolectar
    data = collector.collect_all()

    # Resumen
    print()
    print("DATOS RECOLECTADOS:")
    print("-" * 40)
    for name, df in data.items():
        if df is not None and not df.empty:
            print(f"  {name}: {len(df)} filas, {len(df.columns)} columnas")
        else:
            print(f"  {name}: SIN DATOS")

    # Guardar
    collector.save_all(data)

    # Crear serie unificada
    print()
    print("CREANDO SERIE TEMPORAL UNIFICADA...")
    unified = collector.create_unified_timeseries(data)

    if not unified.empty:
        print(f"Serie unificada: {len(unified)} timestamps, {len(unified.columns)} variables")
        print()
        print("Variables disponibles:")
        for col in unified.columns:
            non_null = unified[col].notna().sum()
            print(f"  {col}: {non_null} valores")

        # Guardar
        unified_path = DATA_DIR / f"unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        unified.to_csv(unified_path)
        print()
        print(f"Guardado: {unified_path}")

    return unified


if __name__ == '__main__':
    main()
