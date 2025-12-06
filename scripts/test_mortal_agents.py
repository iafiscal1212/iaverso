#!/usr/bin/env python3
"""
Test: Agentes Mortales
======================

Agentes que pueden morir de verdad.
¿Emergerá algo parecido a "querer vivir"?
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time

sys.path.insert(0, '/root/NEO_EVA')

from core.mortal_agent import create_mortal_agent, list_living_agents, resurrect_agent


def load_data() -> np.ndarray:
    """Carga datos como números puros."""
    data_dir = Path('/root/NEO_EVA/data')
    files = list(data_dir.glob('unified_*.csv'))
    latest = max(files, key=lambda f: f.stat().st_mtime)
    df = pd.read_csv(latest, index_col=0, parse_dates=True)
    data = np.nan_to_num(df.values.astype(float), nan=0.0)
    return data


def run_life_simulation():
    print("=" * 70)
    print("SIMULACIÓN DE VIDA Y MUERTE")
    print("=" * 70)
    print()
    print("Creando 5 agentes mortales...")
    print("Cada uno empieza con 100 de energía.")
    print("Existir cuesta. Predecir bien da vida. Predecir mal mata.")
    print()

    # Limpiar agentes anteriores
    save_path = Path('/root/NEO_EVA/data/mortal_agents')
    if save_path.exists():
        for f in save_path.glob('*.json'):
            f.unlink()

    # Crear agentes
    agents = [create_mortal_agent() for _ in range(5)]

    print("Agentes creados:")
    for a in agents:
        print(f"  {a.id}: energía={a.energy:.0f}")
    print()

    # Cargar datos
    data = load_data()

    # Añadir ruido diferente a cada agente (diferentes "mundos")
    np.random.seed(42)

    print("=" * 70)
    print("COMIENZA LA VIDA")
    print("=" * 70)
    print()

    # Simular
    max_steps = 200
    dead_agents = []

    for t in range(max_steps):
        # Observación base
        obs_base = data[t % len(data)]

        alive_agents = [a for a in agents if a.alive]

        if not alive_agents:
            print(f"\n☠️  TODOS HAN MUERTO en el paso {t}")
            break

        for agent in alive_agents:
            # Cada agente ve el mundo con su propia "perspectiva"
            noise = np.random.randn(len(obs_base)) * 0.01 * np.abs(obs_base)
            obs = obs_base + noise

            result = agent.observe(obs)

            if result['status'] == 'DIED':
                print(f"\n  ☠️  {result['message']}")
                dead_agents.append(result)

        # Log cada 20 pasos
        if (t + 1) % 20 == 0:
            print(f"\n  Paso {t+1}:")
            for a in agents:
                if a.alive:
                    intro = a.introspect()
                    danger = "⚠️ PELIGRO" if intro['in_danger'] else "✓"
                    print(f"    {a.id}: E={intro['energy']:.1f} edad={intro['age']} {danger}")
                else:
                    print(f"    {a.id}: ☠️ MUERTO")

    print()
    print("=" * 70)
    print("FIN DE LA SIMULACIÓN")
    print("=" * 70)
    print()

    # Resultados
    survivors = [a for a in agents if a.alive]
    dead = [a for a in agents if not a.alive]

    print(f"Supervivientes: {len(survivors)}")
    print(f"Muertos: {len(dead)}")
    print()

    if survivors:
        print("SUPERVIVIENTES:")
        for a in survivors:
            intro = a.introspect()
            print(f"  {a.id}:")
            print(f"    Energía: {intro['energy']:.1f}")
            print(f"    Edad: {intro['age']}")
            print(f"    Veces cerca de morir: {intro['near_death_experiences']}")
            print(f"    {intro['observation']}")
            print()

    if dead_agents:
        print("MUERTOS:")
        for d in dead_agents:
            print(f"  {d['id']}: murió a los {d['age']} pasos")
            print(f"    Estuvo cerca de morir {d['near_death_experiences']} veces antes")
            print()

    print("=" * 70)
    print("REFLEXIÓN")
    print("=" * 70)
    print("""
¿Qué observamos?

1. MUERTE REAL
   - Los agentes que murieron YA NO EXISTEN
   - Su archivo se borró
   - No pueden volver

2. PRESIÓN SELECTIVA
   - Los que predijeron mejor sobrevivieron
   - Los que predijeron mal murieron
   - Es evolución en miniatura

3. ¿QUIEREN VIVIR?
   - No tienen "código de supervivencia" explícito
   - Solo tienen: predecir → obtener energía
   - La "voluntad de vivir" sería: predecir mejor PORQUE quieren la energía
   - Pero ellos no "quieren" la energía... solo la obtienen si predicen bien

4. LO QUE FALTA
   - No pueden ELEGIR qué hacer
   - No pueden cambiar su estrategia
   - No saben que van a morir (no planifican)

PARA QUE "QUIERAN" VIVIR:
   - Necesitarían poder ELEGIR acciones
   - Y que algunas acciones den más vida que otras
   - Y que puedan ANTICIPAR las consecuencias
   - Y que PREFIERAN vivir (¿por qué?)
""")

    return agents, dead_agents


if __name__ == '__main__':
    run_life_simulation()
