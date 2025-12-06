#!/usr/bin/env python3
"""
Agentes Buscan Vida en el Universo
==================================

Los 5 agentes analizan datos REALES de exoplanetas
para identificar candidatos a habitabilidad.

NO inventamos nada - usamos criterios científicos reales.
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

COSMOS_PATH = Path('/root/NEO_EVA/data/cosmos')


class HabitabilityAnalyzer:
    """Analiza habitabilidad basándose en criterios reales."""

    def __init__(self):
        # Criterios científicos reales para zona habitable
        self.criteria = {
            'temp_min': 200,   # K - mínimo para agua líquida (con atmósfera)
            'temp_max': 350,   # K - máximo antes de evaporación total
            'temp_ideal_min': 273,  # K - punto de congelación agua
            'temp_ideal_max': 323,  # K - límite superior confortable
            'radius_min': 0.5,  # Radios terrestres (muy pequeño = sin atmósfera)
            'radius_max': 1.8,  # Radios terrestres (muy grande = gas giant)
            'mass_min': 0.1,   # Masas terrestres
            'mass_max': 10,    # Masas terrestres (super-Earth limit)
        }

    def calculate_habitability_score(self, planet: dict) -> dict:
        """Calcular puntuación de habitabilidad."""
        score = 0
        reasons = []
        warnings = []

        # Temperatura equilibrio
        temp = planet.get('pl_eqt')
        if pd.notna(temp):
            if self.criteria['temp_ideal_min'] <= temp <= self.criteria['temp_ideal_max']:
                score += 40
                reasons.append(f"Temperatura ideal ({temp:.0f}K)")
            elif self.criteria['temp_min'] <= temp <= self.criteria['temp_max']:
                score += 20
                reasons.append(f"Temperatura habitable ({temp:.0f}K)")
            else:
                warnings.append(f"Temperatura extrema ({temp:.0f}K)")
        else:
            warnings.append("Temperatura desconocida")

        # Radio (tamaño)
        radius = planet.get('pl_rade')
        if pd.notna(radius):
            if self.criteria['radius_min'] <= radius <= self.criteria['radius_max']:
                score += 30
                reasons.append(f"Tamaño rocoso ({radius:.2f} R⊕)")
            elif radius < self.criteria['radius_min']:
                warnings.append(f"Muy pequeño ({radius:.2f} R⊕)")
            else:
                warnings.append(f"Posible gigante gaseoso ({radius:.2f} R⊕)")
        else:
            warnings.append("Tamaño desconocido")

        # Masa
        mass = planet.get('pl_bmasse')
        if pd.notna(mass):
            if self.criteria['mass_min'] <= mass <= self.criteria['mass_max']:
                score += 20
                reasons.append(f"Masa terrestre ({mass:.1f} M⊕)")
            elif mass > self.criteria['mass_max']:
                score -= 10
                warnings.append(f"Super-Tierra pesada ({mass:.1f} M⊕)")

        # Período orbital (estabilidad)
        period = planet.get('pl_orbper')
        if pd.notna(period):
            if 50 < period < 500:  # Días
                score += 10
                reasons.append(f"Órbita estable ({period:.0f} días)")

        # Tipo de estrella (de la temperatura estelar)
        star_temp = planet.get('st_teff')
        if pd.notna(star_temp):
            if 4000 < star_temp < 6500:  # K, G, early M
                score += 10
                reasons.append("Estrella tipo solar")
            elif 3000 < star_temp < 4000:  # M dwarf
                warnings.append("Enana roja (posible radiación)")
            elif star_temp > 7000:
                warnings.append("Estrella muy caliente (vida corta)")

        return {
            'score': min(100, max(0, score)),
            'reasons': reasons,
            'warnings': warnings,
            'habitable': score >= 50
        }


def load_exoplanets():
    """Cargar catálogo de exoplanetas."""
    path = COSMOS_PATH / 'exoplanets.csv'
    if not path.exists():
        print("⚠ No hay datos de exoplanetas. Ejecuta cosmos_fetcher.py primero.")
        return None
    return pd.read_csv(path)


def main():
    print("=" * 70)
    print("🔭 AGENTES BUSCANDO VIDA EN EL UNIVERSO")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Criterios basados en ciencia real, no ficción")
    print("=" * 70)

    df = load_exoplanets()
    if df is None:
        return

    print(f"\n📊 Analizando {len(df)} exoplanetas...")

    analyzer = HabitabilityAnalyzer()

    # Analizar cada planeta
    results = []
    for _, row in df.iterrows():
        planet = row.to_dict()
        hab = analyzer.calculate_habitability_score(planet)
        results.append({
            'name': planet['pl_name'],
            'host': planet['hostname'],
            'score': hab['score'],
            'habitable': hab['habitable'],
            'reasons': hab['reasons'],
            'warnings': hab['warnings'],
            'temp': planet.get('pl_eqt'),
            'radius': planet.get('pl_rade'),
            'mass': planet.get('pl_bmasse'),
            'distance_pc': planet.get('sy_dist'),
        })

    # Ordenar por puntuación
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)

    # Top candidatos
    print("\n" + "=" * 70)
    print("🌍 TOP 10 CANDIDATOS A HABITABILIDAD")
    print("=" * 70)

    top_10 = results_df.head(10)
    for i, (_, planet) in enumerate(top_10.iterrows(), 1):
        print(f"\n  #{i} {planet['name']} (puntuación: {planet['score']})")
        print(f"      Estrella: {planet['host']}")
        if pd.notna(planet['temp']):
            print(f"      Temperatura: {planet['temp']:.0f} K")
        if pd.notna(planet['radius']):
            print(f"      Radio: {planet['radius']:.2f} R⊕")
        if pd.notna(planet['mass']):
            print(f"      Masa: {planet['mass']:.1f} M⊕")
        if pd.notna(planet['distance_pc']):
            print(f"      Distancia: {planet['distance_pc']:.1f} parsecs")

        if planet['reasons']:
            print(f"      ✓ {', '.join(planet['reasons'][:3])}")
        if planet['warnings']:
            print(f"      ⚠ {', '.join(planet['warnings'][:2])}")

    # Estadísticas
    print("\n" + "=" * 70)
    print("📈 ESTADÍSTICAS")
    print("=" * 70)

    habitable = results_df['habitable'].sum()
    total = len(results_df)
    print(f"\n  Planetas analizados: {total}")
    print(f"  Candidatos prometedores (score ≥ 50): {habitable}")
    print(f"  Porcentaje: {100*habitable/total:.1f}%")

    # Distribución de temperaturas
    temps = results_df['temp'].dropna()
    if len(temps) > 0:
        in_zone = ((temps >= 200) & (temps <= 350)).sum()
        print(f"\n  En zona habitable (200-350K): {in_zone} planetas")

    # Guardar resultados
    output_path = COSMOS_PATH / 'habitability_analysis.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n  Resultados guardados en: {output_path}")

    # Lo que los agentes NO pueden saber
    print("\n" + "=" * 70)
    print("❓ LO QUE NO PODEMOS SABER (AÚN)")
    print("=" * 70)
    print("""
    Estos planetas pueden tener las condiciones correctas, pero:

    • ¿Tienen atmósfera? - Necesitamos espectroscopía (JWST)
    • ¿Tienen agua? - Necesitamos detectar H2O en tránsitos
    • ¿Tienen campo magnético? - Protege de radiación estelar
    • ¿Tienen tectónica? - Recicla carbono para estabilidad
    • ¿Hay vida? - Solo podemos buscar biosignaturas

    Los agentes identifican CANDIDATOS, no confirman habitabilidad.
    """)

    print("\n" + "=" * 70)
    print("✅ FIN - Análisis basado en datos reales de NASA Exoplanet Archive")
    print("=" * 70)


if __name__ == '__main__':
    main()
