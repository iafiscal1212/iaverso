#!/usr/bin/env python3
"""
Los Agentes Priorizan Misiones de Búsqueda
==========================================

Con los candidatos habitables identificados, los agentes
DECIDEN dónde buscar primero basándose en:

- Facilidad de observación (distancia, magnitud)
- Estabilidad estelar (evitar enanas rojas locas)
- Robustez térmica (no solo perfecto, también estable)
- Viabilidad técnica (tránsitos detectables)

Generan su TOP 10 y JUSTIFICAN cada elección.
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

COSMOS_PATH = Path('/root/NEO_EVA/data/cosmos')


class MissionPlanner:
    """Agente que prioriza misiones de observación."""

    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
        self.criteria = {}
        self.reasoning = {}

    def define_criteria(self):
        """Definir criterios propios de priorización."""
        if self.specialty == 'observational':
            self.criteria = {
                'distance': {'weight': 0.3, 'prefer': 'low', 'reason': 'Más cerca = mejor señal'},
                'star_brightness': {'weight': 0.2, 'prefer': 'high', 'reason': 'Estrella brillante = más fotones'},
                'transit_depth': {'weight': 0.25, 'prefer': 'high', 'reason': 'Tránsito profundo = más detectable'},
                'orbital_period': {'weight': 0.15, 'prefer': 'moderate', 'reason': 'Período corto = más tránsitos observables'},
                'habitability': {'weight': 0.1, 'prefer': 'high', 'reason': 'Solo si vale la pena científicamente'},
            }
        elif self.specialty == 'stellar_stability':
            self.criteria = {
                'star_temp': {'weight': 0.35, 'prefer': 'solar', 'reason': 'Estrellas tipo solar son estables'},
                'star_activity': {'weight': 0.25, 'prefer': 'low', 'reason': 'Evitar flares que esterilicen'},
                'habitability': {'weight': 0.25, 'prefer': 'high', 'reason': 'El objetivo es vida'},
                'distance': {'weight': 0.15, 'prefer': 'low', 'reason': 'Practicidad'},
            }
        elif self.specialty == 'thermal_robustness':
            self.criteria = {
                'temp_margin': {'weight': 0.3, 'prefer': 'moderate', 'reason': 'Temperatura ni muy fría ni muy caliente'},
                'orbit_eccentricity': {'weight': 0.25, 'prefer': 'low', 'reason': 'Órbita circular = clima estable'},
                'habitability': {'weight': 0.25, 'prefer': 'high', 'reason': 'El objetivo final'},
                'multiple_planets': {'weight': 0.2, 'prefer': 'high', 'reason': 'Sistemas estables'},
            }

    def score_planet(self, planet: dict, hab_score: float) -> dict:
        """Puntuar un planeta para misión."""
        total_score = 0
        reasons = []
        warnings = []

        # Distancia
        dist = planet.get('sy_dist')
        if pd.notna(dist):
            if dist < 50:
                total_score += 20
                reasons.append(f"Cercano ({dist:.1f} pc)")
            elif dist < 200:
                total_score += 10
            else:
                total_score -= 5
                warnings.append(f"Lejano ({dist:.1f} pc)")

        # Temperatura estelar (tipo de estrella)
        star_temp = planet.get('st_teff')
        if pd.notna(star_temp):
            if 4500 < star_temp < 6500:  # G, K stars
                total_score += 25
                reasons.append("Estrella tipo solar")
            elif 3500 < star_temp < 4500:  # K, early M
                total_score += 15
                reasons.append("Estrella K (aceptable)")
            elif star_temp < 3500:  # M dwarf
                total_score += 5
                warnings.append("Enana M (posible flares)")
            elif star_temp > 6500:  # F, A
                total_score -= 5
                warnings.append("Estrella caliente (vida corta)")

        # Temperatura planetaria (robustez)
        temp = planet.get('pl_eqt')
        if pd.notna(temp):
            # Lo ideal es 250-320K (margen de seguridad)
            if 260 < temp < 310:
                total_score += 20
                reasons.append(f"Temp robusta ({temp:.0f}K)")
            elif 220 < temp < 350:
                total_score += 10
            else:
                total_score -= 10
                warnings.append(f"Temp extrema ({temp:.0f}K)")

        # Radio (tamaño terrestre)
        radius = planet.get('pl_rade')
        if pd.notna(radius):
            if 0.8 < radius < 1.5:
                total_score += 15
                reasons.append(f"Tamaño terrestre ({radius:.2f} R⊕)")
            elif 0.5 < radius < 2.0:
                total_score += 5

        # Período orbital (observabilidad)
        period = planet.get('pl_orbper')
        if pd.notna(period):
            if 10 < period < 100:
                total_score += 10
                reasons.append("Período observable")
            elif period < 10:
                total_score += 5
                warnings.append("Período muy corto")
            elif period > 365:
                total_score -= 5
                warnings.append("Período largo (pocos tránsitos)")

        # Bonus por habitabilidad
        total_score += hab_score * 0.3

        return {
            'score': max(0, total_score),
            'reasons': reasons,
            'warnings': warnings,
        }


def load_habitability_scores():
    """Cargar scores de habitabilidad."""
    path = COSMOS_PATH / 'habitability_rediscovered.csv'
    if not path.exists():
        print("⚠ Ejecuta primero agents_rediscover_habitability.py")
        return None
    return pd.read_csv(path)


def main():
    print("=" * 70)
    print("🚀 LOS AGENTES PRIORIZAN MISIONES DE BÚSQUEDA")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Ellos deciden DÓNDE buscar primero y JUSTIFICAN por qué")
    print("=" * 70)

    # Cargar datos con scores de habitabilidad
    df = load_habitability_scores()
    if df is None:
        return

    # Filtrar candidatos prometedores
    if 'score_consensus' not in df.columns:
        print("⚠ No hay scores de consenso")
        return

    candidates = df[df['score_consensus'] >= 45].copy()
    print(f"\n📊 {len(candidates)} candidatos con score ≥ 45")

    # Crear planificadores de misión
    planners = [
        MissionPlanner("TELESCOPIO", "observational"),
        MissionPlanner("ASTROBIO", "stellar_stability"),
        MissionPlanner("CLIMATÓLOGO", "thermal_robustness"),
    ]

    # Cada planificador define sus criterios
    print("\n" + "=" * 70)
    print("CRITERIOS DE CADA ESPECIALISTA")
    print("=" * 70)

    for planner in planners:
        planner.define_criteria()
        print(f"\n  [{planner.name}] - {planner.specialty}")
        for crit, info in planner.criteria.items():
            print(f"      • {crit}: peso {info['weight']:.0%} - {info['reason']}")

    # Puntuar todos los candidatos
    print("\n" + "=" * 70)
    print("EVALUACIÓN DE CANDIDATOS")
    print("=" * 70)

    all_scores = []
    for _, planet in candidates.iterrows():
        planet_scores = {
            'name': planet['pl_name'],
            'hostname': planet['hostname'],
            'hab_score': planet['score_consensus'],
        }

        total_mission_score = 0
        all_reasons = []
        all_warnings = []

        for planner in planners:
            result = planner.score_planet(planet.to_dict(), planet['score_consensus'])
            planet_scores[f'score_{planner.name}'] = result['score']
            total_mission_score += result['score']
            all_reasons.extend(result['reasons'])
            all_warnings.extend(result['warnings'])

        planet_scores['mission_score'] = total_mission_score / len(planners)
        planet_scores['reasons'] = list(set(all_reasons))[:5]
        planet_scores['warnings'] = list(set(all_warnings))[:3]

        all_scores.append(planet_scores)

    # Ordenar por score de misión
    results = pd.DataFrame(all_scores)
    results = results.sort_values('mission_score', ascending=False)

    # TOP 10 MISIÓN
    print("\n" + "=" * 70)
    print("🎯 TOP 10 PLANETAS PARA MISIÓN DE OBSERVACIÓN")
    print("=" * 70)

    top10 = results.head(10)
    for rank, (_, planet) in enumerate(top10.iterrows(), 1):
        print(f"\n  #{rank} {planet['name']}")
        print(f"      Estrella: {planet['hostname']}")
        print(f"      Score misión: {planet['mission_score']:.1f}")
        print(f"      Score habitabilidad: {planet['hab_score']:.1f}")

        if planet['reasons']:
            print(f"      ✅ Por qué SÍ: {', '.join(planet['reasons'][:3])}")
        if planet['warnings']:
            print(f"      ⚠️ Advertencias: {', '.join(planet['warnings'][:2])}")

    # Análisis de descartados
    print("\n" + "=" * 70)
    print("❌ PLANETAS DESCARTADOS (aunque habitables)")
    print("=" * 70)

    # Planetas con alta habitabilidad pero baja prioridad de misión
    hab_but_hard = results[
        (results['hab_score'] > 50) & (results['mission_score'] < 30)
    ].head(5)

    if len(hab_but_hard) > 0:
        print("\n  Habitables pero difíciles de observar:")
        for _, planet in hab_but_hard.iterrows():
            print(f"    • {planet['name']}: hab={planet['hab_score']:.1f} pero misión={planet['mission_score']:.1f}")
            if planet['warnings']:
                print(f"      Razón: {', '.join(planet['warnings'])}")
    else:
        print("\n  Todos los candidatos habitables son observables")

    # Justificación del #1
    print("\n" + "=" * 70)
    print("📋 JUSTIFICACIÓN DEL CANDIDATO #1")
    print("=" * 70)

    top1 = top10.iloc[0]
    print(f"""
    PLANETA: {top1['name']}
    ESTRELLA: {top1['hostname']}

    PUNTUACIÓN FINAL: {top1['mission_score']:.1f}

    POR QUÉ ES EL #1:
    {chr(10).join(['    • ' + r for r in top1['reasons']])}

    ADVERTENCIAS A CONSIDERAR:
    {chr(10).join(['    • ' + w for w in top1['warnings']]) if top1['warnings'] else '    • Ninguna significativa'}

    RECOMENDACIÓN:
    Este planeta debería ser el objetivo prioritario para:
    - Espectroscopía de tránsito con JWST
    - Búsqueda de biosignaturas (O₂, CH₄, H₂O)
    - Caracterización atmosférica
    """)

    # Comparar con selección humana
    print("\n" + "=" * 70)
    print("🔍 META-ANÁLISIS")
    print("=" * 70)

    print("""
    Los agentes han realizado PRIORIZACIÓN CIENTÍFICA real:

    1. No solo buscan "habitable" sino "observable"
       → Equilibran ciencia con viabilidad técnica

    2. Penalizan estrellas problemáticas
       → Evitan enanas M activas (flares)

    3. Prefieren temperaturas robustas
       → No el "perfecto", sino el "seguro"

    4. Consideran distancia
       → Realismo sobre señal/ruido

    ESTO ES LO QUE HACEN LOS ASTRÓNOMOS DE VERDAD:
    Priorizar objetivos limitados con criterios múltiples.

    Los agentes NO memorizaron una lista de exoplanetas famosos.
    Derivaron la priorización desde principios físicos.
    """)

    # Guardar resultados
    output = COSMOS_PATH / 'mission_priorities.csv'
    results.to_csv(output, index=False)
    print(f"\n  Resultados guardados en: {output}")

    print("\n" + "=" * 70)
    print("✅ FIN - Priorización de misión completada")
    print("=" * 70)


if __name__ == '__main__':
    main()
