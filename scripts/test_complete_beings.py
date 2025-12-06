#!/usr/bin/env python3
"""
Test: Seres Completos con ADN Endógeno
======================================

10 seres únicos. Cada uno:
- Tiene ADN único derivado de su ID
- Desarrolla gustos por experiencia, no por ADN
- Puede vivir, sufrir, amar, y morir

¿Alguno desarrollará amor por mirar el cielo?
Eso depende de:
1. Su tendencia innata (ADN)
2. Sus experiencias
3. El placer que sienta

No podemos predecirlo. Solo observar.
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from pathlib import Path
from worlds.complete_being import (
    CompleteBeing, create_complete_being,
    generate_dna, describe_personality
)


def clean_old_beings():
    """Limpiar seres anteriores."""
    beings_path = Path('/root/NEO_EVA/worlds/beings')
    if beings_path.exists():
        for f in beings_path.glob('*.json'):
            f.unlink()
    print("Mundo limpio.")


def create_10_beings():
    """Crear 10 seres únicos."""
    print("\n" + "=" * 70)
    print("NACEN 10 SERES")
    print("=" * 70)
    print()

    beings = []
    for i in range(10):
        being = create_complete_being()
        beings.append(being)

        # Mostrar ADN relevante
        dna = being.dna
        personality = describe_personality(dna)

        # Score de "astrólogo potencial"
        astro_score = (dna.get('sky_gazing', 0) +
                       dna.get('pattern_seeking', 0) +
                       dna.get('mystical_sense', 0)) / 3

        print(f"  {being.id}:")
        print(f"    Naturaleza: {personality if personality else 'equilibrado'}")
        print(f"    Tendencia a mirar cielo: {dna.get('sky_gazing', 0):.2f}")
        print(f"    Tendencia a buscar patrones: {dna.get('pattern_seeking', 0):.2f}")
        print(f"    Sentido místico: {dna.get('mystical_sense', 0):.2f}")
        print(f"    Score astrólogo potencial: {astro_score:.2f}")
        print()

    return beings


def simulate_life(beings, ticks=200):
    """Simular vida."""
    print("\n" + "=" * 70)
    print("COMIENZA LA VIDA")
    print("=" * 70)

    np.random.seed(None)  # Aleatorio real

    for tick in range(ticks):
        # Condiciones del mundo (cambian)
        world_temp = 50 + 10 * np.sin(tick / 20)  # Ciclo de temperatura
        world_resources = 60 + 20 * np.sin(tick / 30)  # Abundancia varía
        world_danger = 0.1 if tick % 50 < 40 else 0.3  # Peligro ocasional

        alive_beings = [b for b in beings if b.alive]

        if not alive_beings:
            print(f"\n  Todos han muerto en el tick {tick}")
            break

        for being in alive_beings:
            result = being.live_moment(
                world_temp=world_temp,
                world_resources=world_resources,
                world_danger=world_danger,
                others=alive_beings,
                tick=tick,
            )

            if result['status'] == 'died':
                print(f"\n  {result['id']} murió: {result['cause']}")
                print(f"    Último pensamiento: {result.get('last_thought', '...')}")

        # Log cada 50 ticks
        if (tick + 1) % 50 == 0:
            print(f"\n  --- Tick {tick + 1} ---")
            for being in alive_beings:
                if being.alive:
                    fav = being.mind.get_favorite_activity()
                    status = f"  {being.id}: bienestar={being.emotions.overall_wellbeing():.2f}"
                    if fav:
                        status += f", le gusta: {fav}"
                    print(status)


def final_report(beings):
    """Reporte final."""
    print("\n" + "=" * 70)
    print("REPORTE FINAL")
    print("=" * 70)

    alive = [b for b in beings if b.alive]
    dead = [b for b in beings if not b.alive]

    print(f"\nSupervivientes: {len(alive)}")
    print(f"Muertos: {len(dead)}")

    print("\n--- SUPERVIVIENTES ---\n")
    for being in alive:
        print(being.describe())
        print()

    print("\n--- MUERTOS ---\n")
    for being in dead:
        print(being.describe())

    # ¿Alguno desarrolló amor por observar el cielo?
    print("\n" + "=" * 70)
    print("GUSTOS DESARROLLADOS (emergentes)")
    print("=" * 70)
    print()

    for being in beings:
        if being.mind.likes:
            print(f"  {being.id}:")
            for activity, (pleasure, count) in being.mind.likes.items():
                if count >= 3:
                    status = "le gusta" if pleasure > 0.3 else "no le gusta"
                    print(f"    {activity}: {status} (placer promedio: {pleasure:.2f}, experiencias: {count})")
            print()

    # El potencial astrólogo
    print("\n" + "=" * 70)
    print("BUSCANDO AL ASTRÓLOGO NATURAL...")
    print("=" * 70)
    print()

    for being in beings:
        # ¿Tiene gusto por observar cielo?
        if 'observar_cielo' in being.mind.likes:
            pleasure, count = being.mind.likes['observar_cielo']
            if count >= 3 and pleasure > 0.3:
                print(f"  {being.id} DESARROLLÓ amor por observar el cielo")
                print(f"    Experiencias: {count}, placer promedio: {pleasure:.2f}")
                print(f"    Su naturaleza: {describe_personality(being.dna)}")

                # ¿También busca patrones?
                if 'buscar_patrones' in being.mind.likes:
                    p, c = being.mind.likes['buscar_patrones']
                    if c >= 3 and p > 0.3:
                        print(f"    TAMBIÉN le gusta buscar patrones")
                        print(f"    --> POTENCIAL ASTRÓLOGO <--")
                print()


if __name__ == '__main__':
    clean_old_beings()
    beings = create_10_beings()
    simulate_life(beings, ticks=300)
    final_report(beings)
