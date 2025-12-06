#!/usr/bin/env python3
"""
Fenómenos Cósmicos para el Mundo Vivo
=====================================

Conecta los descubrimientos de NEO (datos reales) con los seres del mundo.

Los seres con tendencia a "mirar el cielo" pueden percibir:
- Actividad solar
- Tormentas geomagnéticas
- Patrones climáticos
- Correlaciones entre fenómenos

NO todos los seres perciben lo mismo:
- El ADN afecta QUÉ pueden percibir (sky_gazing, pattern_seeking)
- La experiencia desarrolla la capacidad de VER patrones
- Los astrólogos naturales pueden "sentir" las conexiones

Es endógeno: los datos vienen del mundo real, pero la PERCEPCIÓN
y el SIGNIFICADO que le dan emerge de cada ser.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class CosmicState:
    """Estado actual del cosmos (desde datos reales)."""
    # Sol
    solar_flux: float = 0.0  # Actividad solar
    solar_change: float = 0.0  # Cambio reciente

    # Geomagnetismo
    geomag_kp: float = 0.0  # Índice Kp
    geomag_storm: bool = False  # Tormenta activa

    # Clima
    temperature: float = 0.0
    pressure: float = 0.0
    humidity: float = 0.0

    # Seísmos
    seismic_activity: float = 0.0

    # Patrones detectados (descubrimientos de NEO)
    active_patterns: List[Dict] = None

    def __post_init__(self):
        if self.active_patterns is None:
            self.active_patterns = []


class CosmicPhenomena:
    """
    El cosmos que los seres pueden observar.

    Carga datos reales y los traduce a fenómenos
    que los seres pueden percibir según su ADN.
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or '/root/NEO_EVA/data/unified_20251206_033253.csv'
        self.data = None
        self.current_t = 0
        self.state = CosmicState()

        # Patrones conocidos (descubrimientos de NEO)
        self.known_patterns = [
            {
                'name': 'sol_afecta_tierra',
                'source': 'solar_flux',
                'target': 'geomag_kp',
                'lag': 8,
                'description': 'El sol agita el campo de la tierra',
            },
            {
                'name': 'geomagnetismo_afecta_clima',
                'source': 'geomag_kp',
                'target': 'climate_temperature',
                'lag': 0,
                'description': 'El campo de la tierra cambia el aire',
            },
            {
                'name': 'presion_predice_geomagnetismo',
                'source': 'climate_pressure',
                'target': 'geomag_kp',
                'lag': 8,
                'description': 'El aire presiona y el campo responde',
            },
            {
                'name': 'geomagnetismo_precede_sismos',
                'source': 'geomag_kp',
                'target': 'seismic_max_mag',
                'lag': 11,
                'description': 'El campo tiembla antes que la tierra',
            },
        ]

        self._load_data()

    def _load_data(self):
        """Cargar datos reales."""
        path = Path(self.data_path)
        if path.exists():
            try:
                self.data = pd.read_csv(path)
                # Normalizar columnas a 0-1
                for col in self.data.columns:
                    if col != 'timestamp' and self.data[col].dtype in ['float64', 'int64']:
                        min_val = self.data[col].min()
                        max_val = self.data[col].max()
                        if max_val > min_val:
                            self.data[col] = (self.data[col] - min_val) / (max_val - min_val)
            except Exception as e:
                print(f"Error cargando datos: {e}")
                self.data = None

    def tick(self, world_tick: int):
        """Avanzar el cosmos un paso."""
        if self.data is None:
            return

        # Mapear tick del mundo a índice de datos
        # (el mundo va más rápido que los datos reales)
        data_idx = (world_tick // 10) % len(self.data)
        self.current_t = data_idx

        row = self.data.iloc[data_idx]

        # Actualizar estado
        prev_solar = self.state.solar_flux

        self.state.solar_flux = row.get('solar_flux', 0.5)
        self.state.solar_change = self.state.solar_flux - prev_solar
        self.state.geomag_kp = row.get('geomag_kp', 0.3)
        self.state.geomag_storm = self.state.geomag_kp > 0.7
        self.state.temperature = row.get('climate_temperature', 0.5)
        self.state.pressure = row.get('climate_pressure', 0.5)
        self.state.humidity = row.get('climate_humidity', 0.5)
        self.state.seismic_activity = row.get('seismic_count', 0.0)

        # Detectar patrones activos
        self.state.active_patterns = []
        for pattern in self.known_patterns:
            source_col = pattern['source']
            if source_col in self.data.columns:
                source_val = row.get(source_col, 0.5)
                # Patrón "activo" si la fuente está alta
                if source_val > 0.6:
                    self.state.active_patterns.append({
                        'name': pattern['name'],
                        'description': pattern['description'],
                        'intensity': source_val,
                    })

    def what_being_perceives(self, dna: Dict[str, float],
                             likes: Dict[str, Tuple[float, int]]) -> Dict:
        """
        Qué percibe un ser del cosmos, según su ADN y experiencia.

        No todos ven lo mismo. Los astrólogos naturales ven más.
        """
        perception = {
            'sees_sky': False,
            'feels_patterns': False,
            'senses': [],
            'insights': [],
            'pleasure': 0.0,
        }

        # Tendencia a mirar el cielo
        sky_gazing = dna.get('sky_gazing', 0.5)
        pattern_seeking = dna.get('pattern_seeking', 0.5)
        mystical_sense = dna.get('mystical_sense', 0.5)
        analytical_mind = dna.get('analytical_mind', 0.5)

        # ¿Mira el cielo?
        if np.random.random() < sky_gazing:
            perception['sees_sky'] = True

            # Qué ve en el cielo
            if self.state.solar_change > 0.1:
                perception['senses'].append('el_sol_cambia')
            if self.state.geomag_storm:
                perception['senses'].append('el_cielo_vibra')
            if self.state.solar_flux > 0.7:
                perception['senses'].append('sol_brillante')

        # ¿Busca patrones?
        if np.random.random() < pattern_seeking:
            perception['feels_patterns'] = True

            # Patrones que puede percibir
            for pattern in self.state.active_patterns:
                # Probabilidad de "ver" el patrón depende de su intensidad
                # y la capacidad del ser
                perceive_prob = pattern['intensity'] * pattern_seeking
                if np.random.random() < perceive_prob:
                    perception['senses'].append(pattern['name'])

        # Insights (solo para seres muy desarrollados)
        # Necesita: mirar cielo + buscar patrones + experiencia
        likes_sky = likes.get('observar_cielo', (0, 0))
        likes_patterns = likes.get('buscar_patrones', (0, 0))

        is_experienced = (likes_sky[1] >= 5 and likes_sky[0] > 0.3 and
                         likes_patterns[1] >= 5 and likes_patterns[0] > 0.3)

        if is_experienced and perception['sees_sky'] and perception['feels_patterns']:
            # Puede tener insights
            for pattern in self.state.active_patterns:
                if np.random.random() < mystical_sense:
                    perception['insights'].append(pattern['description'])

        # Placer de la observación
        if perception['sees_sky']:
            base_pleasure = 0.2
            if self.state.solar_flux > 0.6:
                base_pleasure += 0.1
            if self.state.geomag_storm:
                base_pleasure += 0.15  # Las tormentas son fascinantes
            if perception['insights']:
                base_pleasure += 0.3  # Insights dan mucho placer

            # Modulado por estado emocional implícito
            perception['pleasure'] = min(1.0, base_pleasure)

        return perception

    def describe_sky(self) -> str:
        """Describir el cielo actual."""
        desc = []

        if self.state.solar_flux > 0.7:
            desc.append("El sol brilla intensamente")
        elif self.state.solar_flux < 0.3:
            desc.append("El sol está tranquilo")

        if self.state.geomag_storm:
            desc.append("El cielo vibra con energía invisible")

        if self.state.active_patterns:
            desc.append(f"Hay {len(self.state.active_patterns)} patrones activos")

        return ". ".join(desc) if desc else "El cielo está en calma"


# Singleton global del cosmos
_cosmos = None

def get_cosmos() -> CosmicPhenomena:
    """Obtener el cosmos global."""
    global _cosmos
    if _cosmos is None:
        _cosmos = CosmicPhenomena()
    return _cosmos


def perceive_cosmos(dna: Dict[str, float],
                   likes: Dict[str, Tuple[float, int]],
                   tick: int) -> Dict:
    """
    Función helper para que un ser perciba el cosmos.

    Uso desde complete_being.py:
        from worlds.cosmic_phenomena import perceive_cosmos
        perception = perceive_cosmos(self.dna, self.mind.likes, tick)
    """
    cosmos = get_cosmos()
    cosmos.tick(tick)
    return cosmos.what_being_perceives(dna, likes)
