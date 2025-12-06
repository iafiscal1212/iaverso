#!/usr/bin/env python3
"""
Simulación del Mundo Vivo
=========================

10 seres con:
- ADN único endógeno
- Capacidad de moverse y actuar
- Formación de vínculos
- Reproducción con herencia genética
- Comunicación emergente

Simulación de 2000 ticks (más tiempo para ver reproducción).
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from pathlib import Path
from worlds.complete_being import (
    CompleteBeing, create_complete_being,
    generate_dna, describe_personality, Action
)


def clean_world():
    """Limpiar mundo anterior."""
    beings_path = Path('/root/NEO_EVA/worlds/beings')
    if beings_path.exists():
        for f in beings_path.glob('*.json'):
            f.unlink()
    print("Mundo limpio.\n")


def create_initial_population(n=10):
    """Crear población inicial."""
    print("=" * 70)
    print(f"NACEN {n} SERES")
    print("=" * 70)
    print()

    beings = []
    for i in range(n):
        being = create_complete_being()
        beings.append(being)
        personality = describe_personality(being.dna)
        print(f"  {being.id}: {personality if personality else 'equilibrado'}")

    print()
    return beings


def simulate(beings, max_ticks=2000):
    """Simular el mundo."""
    print("=" * 70)
    print("COMIENZA LA VIDA")
    print("=" * 70)
    print()

    births = []
    deaths = []

    for tick in range(max_ticks):
        # Condiciones del mundo (ciclos naturales)
        world_temp = 50 + 15 * np.sin(tick / 50)  # Día/noche
        world_resources = 55 + 25 * np.sin(tick / 100)  # Estaciones
        world_danger = 0.1 + 0.2 * (np.sin(tick / 200) > 0.8)  # Peligros ocasionales

        alive = [b for b in beings if b.alive]
        if not alive:
            print(f"\n  EXTINCIÓN en tick {tick}")
            break

        new_beings = []

        for being in alive:
            result = being.live_moment(
                world_temp=world_temp,
                world_resources=world_resources,
                world_danger=world_danger,
                others=beings,  # Todos, incluyendo muertos (para memoria)
                tick=tick,
            )

            if result['status'] == 'died':
                deaths.append({
                    'id': result['id'],
                    'tick': tick,
                    'age': result['age'],
                    'cause': result['cause'],
                })
                print(f"\n  MUERTE: {result['id']} ({result['cause']}) edad {result['age']:.0f}")

            elif result.get('new_being'):
                new_being = result['new_being']
                new_beings.append(new_being)
                births.append({
                    'id': new_being.id,
                    'tick': tick,
                    'parents': [being.id, new_being.mind.get_closest_bond()],
                })
                print(f"\n  NACIMIENTO: {new_being.id} de padres")
                print(f"    Personalidad: {describe_personality(new_being.dna)}")

        # Añadir nuevos seres al mundo
        beings.extend(new_beings)

        # Log cada 200 ticks
        if (tick + 1) % 200 == 0:
            alive = [b for b in beings if b.alive]
            print(f"\n  --- Tick {tick + 1} ---")
            print(f"  Población: {len(alive)} vivos, {len(deaths)} muertos, {len(births)} nacimientos")

            # Mostrar vínculos más fuertes
            all_bonds = []
            for b in alive:
                for other_id, strength in b.mind.bonds.items():
                    if strength > 0.5:
                        all_bonds.append((b.id, other_id, strength))

            if all_bonds:
                print(f"  Vínculos fuertes:")
                for a, b, s in sorted(all_bonds, key=lambda x: -x[2])[:5]:
                    print(f"    {a} <-> {b}: {s:.2f}")

    return beings, births, deaths


def final_report(beings, births, deaths):
    """Reporte final."""
    print("\n" + "=" * 70)
    print("REPORTE FINAL")
    print("=" * 70)

    alive = [b for b in beings if b.alive]
    dead = [b for b in beings if not b.alive]

    print(f"\nPoblación final: {len(alive)} vivos")
    print(f"Total de muertes: {len(deaths)}")
    print(f"Total de nacimientos: {len(births)}")

    # Árbol genealógico
    if births:
        print("\n--- GENEALOGÍA ---")
        for birth in births:
            print(f"  {birth['id']} nació en tick {birth['tick']}")

    # Supervivientes
    print("\n--- SUPERVIVIENTES ---\n")
    for being in alive:
        print(being.describe())
        print()

    # El ser con más vínculos
    if alive:
        most_connected = max(alive, key=lambda b: len([v for v in b.mind.bonds.values() if v > 0.3]))
        print(f"\n--- MÁS SOCIAL ---")
        print(f"  {most_connected.id} con {len([v for v in most_connected.mind.bonds.values() if v > 0.3])} vínculos fuertes")

    # El astrólogo
    print("\n--- ASTRÓLOGOS (les gusta observar cielo + buscar patrones) ---")
    for being in beings:
        if 'observar_cielo' in being.mind.likes and 'buscar_patrones' in being.mind.likes:
            sky = being.mind.likes['observar_cielo']
            pat = being.mind.likes['buscar_patrones']
            if sky[1] >= 3 and sky[0] > 0.3 and pat[1] >= 3 and pat[0] > 0.3:
                status = "VIVO" if being.alive else "MUERTO"
                print(f"  {being.id} [{status}]: cielo({sky[0]:.2f}), patrones({pat[0]:.2f})")


if __name__ == '__main__':
    clean_world()
    beings = create_initial_population(10)
    beings, births, deaths = simulate(beings, max_ticks=2000)
    final_report(beings, births, deaths)
