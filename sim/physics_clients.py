"""
Physics Data Clients - Datos físicos reales

NORMA DURA:
- Solo OBTIENE datos, no interpreta
- Transforma datos a dimensiones estructurales abstractas
- NO genera predicciones físicas
- Los datos se usan para simulación in silico

Fuentes:
- CERN Open Data: Física de partículas (LHC)
- NASA: Física solar, espacial, magnetosfera
- USGS: Sismos, geofísica
- NOAA: Clima, océanos, atmósfera
- ESA: Datos astronómicos europeos

Dimensiones abstractas:
- energy_density: Densidad energética normalizada
- field_intensity: Intensidad de campo
- temporal_stability: Estabilidad temporal
- spatial_coherence: Coherencia espacial
- event_frequency: Frecuencia de eventos
- magnitude_distribution: Distribución de magnitudes
- correlation_strength: Fuerza de correlación
- anomaly_index: Índice de anomalía
"""

import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import math


@dataclass
class PhysicsState:
    """Estado físico abstracto derivado de datos reales."""
    energy_density: float        # 0-1: Densidad energética
    field_intensity: float       # 0-1: Intensidad de campo
    temporal_stability: float    # 0-1: Estabilidad temporal
    spatial_coherence: float     # 0-1: Coherencia espacial
    event_frequency: float       # 0-1: Frecuencia de eventos
    magnitude_distribution: float # 0-1: Distribución de magnitudes
    correlation_strength: float  # 0-1: Fuerza de correlación
    anomaly_index: float         # 0-1: Índice de anomalía

    source: str = ""
    dataset: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'energy_density': self.energy_density,
            'field_intensity': self.field_intensity,
            'temporal_stability': self.temporal_stability,
            'spatial_coherence': self.spatial_coherence,
            'event_frequency': self.event_frequency,
            'magnitude_distribution': self.magnitude_distribution,
            'correlation_strength': self.correlation_strength,
            'anomaly_index': self.anomaly_index,
            'source': self.source,
            'dataset': self.dataset,
            'timestamp': self.timestamp.isoformat()
        }


# =============================================================================
# CERN OPEN DATA CLIENT
# =============================================================================

class CERNClient:
    """
    Cliente para CERN Open Data Portal.

    Acceso: Abierto
    Datos: Física de partículas (LHC), colisiones, eventos
    Uso: Inferencia estructural, patrones no triviales

    ⚠️ Solo lectura. NO genera predicciones físicas.
    """

    BASE_URL = "https://opendata.cern.ch/api"

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def search_records(self, query: str, max_results: int = 10) -> Dict:
        """Busca registros en CERN Open Data."""
        session = await self._get_session()

        params = {
            'q': query,
            'size': max_results,
            'type': 'Dataset'
        }

        try:
            async with session.get(f"{self.BASE_URL}/records/", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    hits = data.get('hits', {}).get('hits', [])
                    return {
                        'query': query,
                        'total': data.get('hits', {}).get('total', 0),
                        'records': [
                            {
                                'id': h.get('id'),
                                'title': h.get('metadata', {}).get('title', ''),
                                'experiment': h.get('metadata', {}).get('experiment', ''),
                                'type': h.get('metadata', {}).get('type', {}).get('primary', '')
                            }
                            for h in hits[:10]
                        ]
                    }
        except Exception as e:
            pass

        return {'query': query, 'total': 0, 'records': []}

    async def get_experiments(self) -> List[Dict]:
        """Obtiene lista de experimentos disponibles."""
        session = await self._get_session()

        try:
            async with session.get(f"{self.BASE_URL}/records/?type=Dataset&size=50") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    hits = data.get('hits', {}).get('hits', [])

                    experiments = {}
                    for h in hits:
                        exp = h.get('metadata', {}).get('experiment', 'Unknown')
                        if exp not in experiments:
                            experiments[exp] = 0
                        experiments[exp] += 1

                    return [
                        {'name': k, 'datasets': v}
                        for k, v in sorted(experiments.items(), key=lambda x: -x[1])
                    ]
        except:
            pass

        return []

    async def get_abstract_state(self, experiment: str = None) -> PhysicsState:
        """
        Genera estado abstracto basado en datos CERN.
        """
        query = experiment or "CMS"
        results = await self.search_records(query)

        # Transformar a dimensiones abstractas
        n_datasets = results['total']

        return PhysicsState(
            energy_density=0.8,  # LHC = alta energía
            field_intensity=0.7,
            temporal_stability=0.6,  # Colisiones = eventos discretos
            spatial_coherence=0.5,
            event_frequency=min(1.0, n_datasets / 100),
            magnitude_distribution=0.7,  # Distribución conocida
            correlation_strength=0.6,
            anomaly_index=0.3,  # Física bien entendida
            source='CERN',
            dataset=query
        )

    async def ping(self) -> bool:
        """Verifica conexión con CERN."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.BASE_URL}/records/?size=1") as resp:
                return resp.status == 200
        except:
            return False


# =============================================================================
# NASA OPEN DATA CLIENT
# =============================================================================

class NASAClient:
    """
    Cliente para NASA Open Data.

    Acceso: Abierto
    Datos: Viento solar, magnetosfera, radiación, clima espacial
    Uso: Sistemas dinámicos, inferencia temporal

    ⚠️ Solo lectura. NO genera predicciones.
    """

    # NASA APIs
    DONKI_URL = "https://api.nasa.gov/DONKI"  # Space Weather
    NEO_URL = "https://api.nasa.gov/neo/rest/v1"  # Near Earth Objects
    API_KEY = "DEMO_KEY"  # Clave demo pública

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_solar_flares(self, days: int = 30) -> Dict:
        """Obtiene erupciones solares recientes."""
        session = await self._get_session()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        params = {
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d'),
            'api_key': self.API_KEY
        }

        try:
            async with session.get(f"{self.DONKI_URL}/FLR", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        'type': 'solar_flares',
                        'count': len(data) if isinstance(data, list) else 0,
                        'events': data[:10] if isinstance(data, list) else []
                    }
        except:
            pass

        return {'type': 'solar_flares', 'count': 0, 'events': []}

    async def get_geomagnetic_storms(self, days: int = 30) -> Dict:
        """Obtiene tormentas geomagnéticas recientes."""
        session = await self._get_session()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        params = {
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d'),
            'api_key': self.API_KEY
        }

        try:
            async with session.get(f"{self.DONKI_URL}/GST", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        'type': 'geomagnetic_storms',
                        'count': len(data) if isinstance(data, list) else 0,
                        'events': data[:10] if isinstance(data, list) else []
                    }
        except:
            pass

        return {'type': 'geomagnetic_storms', 'count': 0, 'events': []}

    async def get_coronal_mass_ejections(self, days: int = 30) -> Dict:
        """Obtiene eyecciones de masa coronal."""
        session = await self._get_session()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        params = {
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d'),
            'api_key': self.API_KEY
        }

        try:
            async with session.get(f"{self.DONKI_URL}/CME", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        'type': 'coronal_mass_ejections',
                        'count': len(data) if isinstance(data, list) else 0,
                        'events': data[:10] if isinstance(data, list) else []
                    }
        except:
            pass

        return {'type': 'coronal_mass_ejections', 'count': 0, 'events': []}

    async def get_abstract_state(self) -> PhysicsState:
        """
        Genera estado abstracto basado en datos NASA.
        """
        flares = await self.get_solar_flares(30)
        storms = await self.get_geomagnetic_storms(30)
        cmes = await self.get_coronal_mass_ejections(30)

        # Calcular dimensiones
        total_events = flares['count'] + storms['count'] + cmes['count']
        event_frequency = min(1.0, total_events / 50)

        # Más eventos = más actividad solar = menos estabilidad
        temporal_stability = max(0.2, 1.0 - event_frequency)

        return PhysicsState(
            energy_density=0.6 + (0.3 if flares['count'] > 5 else 0),
            field_intensity=0.5 + (0.3 if storms['count'] > 0 else 0),
            temporal_stability=round(temporal_stability, 3),
            spatial_coherence=0.6,
            event_frequency=round(event_frequency, 3),
            magnitude_distribution=0.5,
            correlation_strength=0.7,  # Eventos correlacionados
            anomaly_index=0.4 if total_events > 10 else 0.2,
            source='NASA',
            dataset='DONKI_SpaceWeather'
        )

    async def ping(self) -> bool:
        """Verifica conexión con NASA."""
        try:
            session = await self._get_session()
            params = {'api_key': self.API_KEY}
            async with session.get(f"{self.DONKI_URL}/notifications?type=all", params=params) as resp:
                return resp.status == 200
        except:
            return False


# =============================================================================
# USGS EARTHQUAKE CLIENT
# =============================================================================

class USGSClient:
    """
    Cliente para USGS Earthquake Catalog.

    Acceso: Abierto
    Datos: Sismos globales (tiempo, magnitud, localización)
    Uso: Dinámica no lineal, eventos extremos

    ⚠️ Solo lectura. NO genera predicciones sísmicas.
    """

    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1"

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_recent_earthquakes(
        self,
        min_magnitude: float = 4.0,
        days: int = 7,
        limit: int = 100
    ) -> Dict:
        """Obtiene sismos recientes."""
        session = await self._get_session()

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        params = {
            'format': 'geojson',
            'starttime': start_time.strftime('%Y-%m-%d'),
            'endtime': end_time.strftime('%Y-%m-%d'),
            'minmagnitude': min_magnitude,
            'limit': limit,
            'orderby': 'time'
        }

        try:
            async with session.get(f"{self.BASE_URL}/query", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    features = data.get('features', [])

                    return {
                        'count': len(features),
                        'min_magnitude': min_magnitude,
                        'days': days,
                        'earthquakes': [
                            {
                                'magnitude': f['properties']['mag'],
                                'place': f['properties']['place'],
                                'time': f['properties']['time'],
                                'depth': f['geometry']['coordinates'][2] if len(f['geometry']['coordinates']) > 2 else None,
                                'coordinates': f['geometry']['coordinates'][:2]
                            }
                            for f in features[:20]
                        ]
                    }
        except:
            pass

        return {'count': 0, 'earthquakes': []}

    async def get_significant_earthquakes(self, days: int = 30) -> Dict:
        """Obtiene sismos significativos."""
        session = await self._get_session()

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        params = {
            'format': 'geojson',
            'starttime': start_time.strftime('%Y-%m-%d'),
            'endtime': end_time.strftime('%Y-%m-%d'),
            'minmagnitude': 6.0
        }

        try:
            async with session.get(f"{self.BASE_URL}/query", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    features = data.get('features', [])

                    magnitudes = [f['properties']['mag'] for f in features if f['properties']['mag']]

                    return {
                        'count': len(features),
                        'max_magnitude': max(magnitudes) if magnitudes else 0,
                        'avg_magnitude': sum(magnitudes) / len(magnitudes) if magnitudes else 0,
                        'earthquakes': [
                            {
                                'magnitude': f['properties']['mag'],
                                'place': f['properties']['place'],
                                'time': f['properties']['time']
                            }
                            for f in features[:10]
                        ]
                    }
        except:
            pass

        return {'count': 0, 'max_magnitude': 0, 'earthquakes': []}

    async def get_abstract_state(self, days: int = 7) -> PhysicsState:
        """
        Genera estado abstracto basado en datos sísmicos.
        """
        recent = await self.get_recent_earthquakes(min_magnitude=4.0, days=days)
        significant = await self.get_significant_earthquakes(days=30)

        # Calcular dimensiones
        event_count = recent['count']
        event_frequency = min(1.0, event_count / 50)

        # Más sismos = menos estabilidad
        temporal_stability = max(0.2, 1.0 - event_frequency * 0.5)

        # Magnitud máxima afecta anomalía
        max_mag = significant.get('max_magnitude', 0)
        anomaly_index = min(1.0, max_mag / 9.0) if max_mag > 0 else 0.2

        return PhysicsState(
            energy_density=0.5 + (0.3 if max_mag > 6 else 0),
            field_intensity=0.4,
            temporal_stability=round(temporal_stability, 3),
            spatial_coherence=0.5,  # Sismos distribuidos globalmente
            event_frequency=round(event_frequency, 3),
            magnitude_distribution=0.6,  # Ley de Gutenberg-Richter
            correlation_strength=0.4,  # Correlación limitada
            anomaly_index=round(anomaly_index, 3),
            source='USGS',
            dataset='Earthquake_Catalog'
        )

    async def ping(self) -> bool:
        """Verifica conexión con USGS."""
        try:
            session = await self._get_session()
            params = {'format': 'geojson', 'limit': 1}
            async with session.get(f"{self.BASE_URL}/query", params=params) as resp:
                return resp.status == 200
        except:
            return False


# =============================================================================
# NOAA CLIENT
# =============================================================================

class NOAAClient:
    """
    Cliente para NOAA (National Oceanic and Atmospheric Administration).

    Acceso: Abierto
    Datos: Océanos, atmósfera, clima
    Uso: Estabilidad/inestabilidad, análisis multiescala

    ⚠️ Solo lectura. NO genera predicciones climáticas.
    """

    # NOAA Climate Data Online
    BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
    # Token público de demo (limitado)
    TOKEN = "demo"

    # También usamos datos de Space Weather
    SPACE_WEATHER_URL = "https://services.swpc.noaa.gov/products"

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_solar_wind(self) -> Dict:
        """Obtiene datos de viento solar (NOAA Space Weather)."""
        session = await self._get_session()

        try:
            async with session.get(f"{self.SPACE_WEATHER_URL}/solar-wind/plasma-7-day.json") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # El primer elemento son headers
                    if len(data) > 1:
                        recent = data[-10:]  # Últimos 10 registros
                        return {
                            'type': 'solar_wind',
                            'count': len(data) - 1,
                            'recent': recent
                        }
        except:
            pass

        return {'type': 'solar_wind', 'count': 0, 'recent': []}

    async def get_geomagnetic_indices(self) -> Dict:
        """Obtiene índices geomagnéticos (Kp)."""
        session = await self._get_session()

        try:
            async with session.get(f"{self.SPACE_WEATHER_URL}/noaa-planetary-k-index.json") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if len(data) > 1:
                        # Extraer valores Kp
                        kp_values = []
                        for row in data[1:]:
                            try:
                                kp = float(row[1])
                                kp_values.append(kp)
                            except:
                                pass

                        return {
                            'type': 'kp_index',
                            'count': len(kp_values),
                            'max_kp': max(kp_values) if kp_values else 0,
                            'avg_kp': sum(kp_values) / len(kp_values) if kp_values else 0,
                            'recent': data[-10:] if len(data) > 10 else data[1:]
                        }
        except:
            pass

        return {'type': 'kp_index', 'count': 0, 'max_kp': 0}

    async def get_abstract_state(self) -> PhysicsState:
        """
        Genera estado abstracto basado en datos NOAA.
        """
        solar_wind = await self.get_solar_wind()
        kp_data = await self.get_geomagnetic_indices()

        # Kp alto = perturbación geomagnética
        max_kp = kp_data.get('max_kp', 0)
        avg_kp = kp_data.get('avg_kp', 0)

        # Normalizar Kp (escala 0-9)
        field_intensity = min(1.0, max_kp / 9.0)
        temporal_stability = max(0.2, 1.0 - avg_kp / 9.0)

        return PhysicsState(
            energy_density=0.5,
            field_intensity=round(field_intensity, 3),
            temporal_stability=round(temporal_stability, 3),
            spatial_coherence=0.6,
            event_frequency=0.5,
            magnitude_distribution=0.5,
            correlation_strength=0.7,  # Viento solar correlacionado con Kp
            anomaly_index=0.3 if max_kp > 5 else 0.1,
            source='NOAA',
            dataset='Space_Weather'
        )

    async def ping(self) -> bool:
        """Verifica conexión con NOAA."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.SPACE_WEATHER_URL}/noaa-planetary-k-index.json") as resp:
                return resp.status == 200
        except:
            return False


# =============================================================================
# UNIFIED PHYSICS CLIENT
# =============================================================================

class PhysicsDataClient:
    """
    Cliente unificado para todas las fuentes físicas.

    Combina datos de:
    - CERN: Física de partículas
    - NASA: Física solar/espacial
    - USGS: Geofísica/sismos
    - NOAA: Clima espacial

    ⚠️ Solo lectura. NO genera predicciones.
    """

    def __init__(self):
        self.cern = CERNClient()
        self.nasa = NASAClient()
        self.usgs = USGSClient()
        self.noaa = NOAAClient()

        self._sources = {
            'CERN': self.cern,
            'NASA': self.nasa,
            'USGS': self.usgs,
            'NOAA': self.noaa
        }

    async def close(self):
        """Cierra todas las sesiones."""
        for client in self._sources.values():
            await client.close()

    async def ping_all(self) -> Dict[str, bool]:
        """Verifica conexión con todas las fuentes."""
        results = {}
        for name, client in self._sources.items():
            try:
                results[name] = await client.ping()
            except:
                results[name] = False
        return results

    async def get_unified_state(self, sources: List[str] = None) -> PhysicsState:
        """
        Obtiene estado unificado combinando múltiples fuentes.
        """
        sources = sources or list(self._sources.keys())

        states = []

        for source_name in sources:
            if source_name in self._sources:
                try:
                    client = self._sources[source_name]
                    state = await client.get_abstract_state()
                    states.append(state)
                except Exception as e:
                    continue

        if not states:
            return PhysicsState(
                energy_density=0.5,
                field_intensity=0.5,
                temporal_stability=0.5,
                spatial_coherence=0.5,
                event_frequency=0.5,
                magnitude_distribution=0.5,
                correlation_strength=0.5,
                anomaly_index=0.3,
                source='default',
                dataset='none'
            )

        # Promediar estados
        n = len(states)
        return PhysicsState(
            energy_density=round(sum(s.energy_density for s in states) / n, 3),
            field_intensity=round(sum(s.field_intensity for s in states) / n, 3),
            temporal_stability=round(sum(s.temporal_stability for s in states) / n, 3),
            spatial_coherence=round(sum(s.spatial_coherence for s in states) / n, 3),
            event_frequency=round(sum(s.event_frequency for s in states) / n, 3),
            magnitude_distribution=round(sum(s.magnitude_distribution for s in states) / n, 3),
            correlation_strength=round(sum(s.correlation_strength for s in states) / n, 3),
            anomaly_index=round(sum(s.anomaly_index for s in states) / n, 3),
            source=','.join(s.source for s in states),
            dataset=','.join(s.dataset for s in states)
        )

    def list_sources(self) -> List[Dict]:
        """Lista todas las fuentes disponibles."""
        return [
            {
                'id': 'CERN',
                'name': 'CERN Open Data',
                'type': 'particle_physics',
                'access': 'open',
                'description': 'Física de partículas (LHC), colisiones, eventos'
            },
            {
                'id': 'NASA',
                'name': 'NASA DONKI',
                'type': 'space_physics',
                'access': 'open',
                'description': 'Física solar, viento solar, tormentas geomagnéticas'
            },
            {
                'id': 'USGS',
                'name': 'USGS Earthquake Catalog',
                'type': 'geophysics',
                'access': 'open',
                'description': 'Sismos globales, magnitudes, localizaciones'
            },
            {
                'id': 'NOAA',
                'name': 'NOAA Space Weather',
                'type': 'space_weather',
                'access': 'open',
                'description': 'Clima espacial, índices geomagnéticos'
            }
        ]


# =============================================================================
# SINGLETON
# =============================================================================

_physics_client: Optional[PhysicsDataClient] = None

def get_physics_client() -> PhysicsDataClient:
    global _physics_client
    if _physics_client is None:
        _physics_client = PhysicsDataClient()
    return _physics_client
