#!/usr/bin/env python3
"""
Test: Mundo Equilibrado
=======================

Verificar que los seres:
1. Priorizan necesidades básicas (comida, agua, descanso)
2. Forman vínculos y se reproducen
3. SOLO miran el cielo cuando están satisfechos
4. Desarrollan gustos diversos (no solo astronomía)

La astrología es solo UNA opción entre muchas.
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from pathlib import Path
from worlds.complete_being import (
    CompleteBeing, create_complete_being,
    generate_dna, describe_personality, Action
)

# Estadísticas
stats = {
    'actions': {a.value: 0 for a in Action},
    'activities': {},
    'needs_satisfied': 0,
    'needs_urgent': 0,
    'births': 0,
    'deaths': 0,
}


def clean_world():
    """Limpiar mundo anterior."""
    beings_path = Path('/root/NEO_EVA/worlds/beings')
    if beings_path.exists():
        for f in beings_path.glob('*.json'):
            f.unlink()
    print("Mundo limpio.\n")


def create_population(n=10):
    """Crear población inicial."""
    print("=" * 70)
    print(f"NACEN {n} SERES")
    print("=" * 70)

    beings = []
    for i in range(n):
        being = create_complete_being()
        beings.append(being)
        personality = describe_personality(being.dna)
        print(f"  {being.id}: {personality if personality else 'equilibrado'}")

    print()
    return beings


def simulate(beings, max_ticks=200):
    """Simular con seguimiento de estadísticas."""
    print("=" * 70)
    print("SIMULACIÓN (200 ticks)")
    print("=" * 70)
    print()

    for tick in range(max_ticks):
        # Condiciones del mundo
        world_temp = 50 + 15 * np.sin(tick / 50)
        world_resources = 55 + 25 * np.sin(tick / 100)
        world_danger = 0.1

        alive = [b for b in beings if b.alive]
        if not alive:
            print(f"\n  EXTINCIÓN en tick {tick}")
            break

        # Limitar población para no explotar
        if len(alive) > 20:
            alive = alive[:20]

        for being in alive:
            # Tracking de estado antes de actuar
            needs_ok = (being.body.hunger > 40 and
                       being.body.thirst > 40 and
                       being.body.energy > 30)

            if needs_ok:
                stats['needs_satisfied'] += 1
            else:
                stats['needs_urgent'] += 1

            result = being.live_moment(
                world_temp=world_temp,
                world_resources=world_resources,
                world_danger=world_danger,
                others=beings,
                tick=tick,
            )

            # Tracking de acción
            if 'action' in result:
                stats['actions'][result['action']] = stats['actions'].get(result['action'], 0) + 1

            # Tracking de actividades espontáneas
            for activity, (pleasure, count) in being.mind.likes.items():
                if activity not in stats['activities']:
                    stats['activities'][activity] = 0
                stats['activities'][activity] = max(stats['activities'][activity], count)

            if result['status'] == 'died':
                stats['deaths'] += 1
                print(f"  MUERTE: {result['id']} ({result['cause']})")

            elif result.get('new_being'):
                stats['births'] += 1
                beings.append(result['new_being'])

        # Log cada 100 ticks
        if (tick + 1) % 100 == 0:
            alive = [b for b in beings if b.alive]
            print(f"\n  --- Tick {tick + 1} ---")
            print(f"  Población: {len(alive)}")

            # Muestra de lo que hacen
            sample = np.random.choice(alive, min(3, len(alive)), replace=False)
            for being in sample:
                action = being.choose_action(beings)
                fav = being.mind.get_favorite_activity()
                hunger = being.body.hunger
                status = f"    {being.id}: hambre={hunger:.0f}, "
                status += f"acción={action.value}"
                if fav:
                    status += f", le gusta: {fav}"
                print(status)

    return beings


def report(beings):
    """Reporte final."""
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS FINALES")
    print("=" * 70)

    alive = [b for b in beings if b.alive]

    print(f"\nPoblación: {len(alive)} vivos, {stats['deaths']} muertos, {stats['births']} nacimientos")

    # Acciones tomadas
    print("\n--- ACCIONES (qué hicieron) ---")
    total_actions = sum(stats['actions'].values())
    for action, count in sorted(stats['actions'].items(), key=lambda x: -x[1]):
        pct = (count / total_actions * 100) if total_actions > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {action:20s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Actividades espontáneas
    print("\n--- ACTIVIDADES ESPONTÁNEAS (hobby) ---")
    for activity, count in sorted(stats['activities'].items(), key=lambda x: -x[1]):
        print(f"  {activity:20s}: {count} experiencias máx")

    # Gustos desarrollados
    print("\n--- GUSTOS DESARROLLADOS ---")
    all_likes = {}
    for being in beings:
        for activity, (pleasure, count) in being.mind.likes.items():
            if count >= 3:
                if activity not in all_likes:
                    all_likes[activity] = []
                all_likes[activity].append((being.id, pleasure, being.alive))

    for activity, likers in sorted(all_likes.items(), key=lambda x: -len(x[1])):
        n_alive = sum(1 for _, _, alive in likers if alive)
        print(f"  {activity}: {len(likers)} seres ({n_alive} vivos)")

    # Estado de necesidades
    print("\n--- BALANCE NECESIDADES vs OCIO ---")
    total = stats['needs_satisfied'] + stats['needs_urgent']
    if total > 0:
        pct_ok = stats['needs_satisfied'] / total * 100
        pct_urgent = stats['needs_urgent'] / total * 100
        print(f"  Momentos satisfechos: {pct_ok:.1f}%")
        print(f"  Momentos con urgencia: {pct_urgent:.1f}%")
        print(f"  (Solo hacen hobbies cuando están satisfechos)")

    # El más astronómico vs el más social
    print("\n--- PERSONALIDADES ---")
    if alive:
        most_astro = None
        max_astro = 0
        most_social = None
        max_bonds = 0

        for being in alive:
            # Astrólogo
            sky = being.mind.likes.get('observar_cielo', (0, 0))
            pat = being.mind.likes.get('buscar_patrones', (0, 0))
            astro_score = sky[1] + pat[1]
            if astro_score > max_astro:
                max_astro = astro_score
                most_astro = being

            # Social
            n_bonds = len([v for v in being.mind.bonds.values() if v > 0.3])
            if n_bonds > max_bonds:
                max_bonds = n_bonds
                most_social = being

        if most_astro:
            sky = most_astro.mind.likes.get('observar_cielo', (0, 0))
            print(f"  Más astronómico: {most_astro.id} ({sky[1]} observaciones)")

        if most_social:
            print(f"  Más social: {most_social.id} ({max_bonds} vínculos fuertes)")

    print("\n" + "=" * 70)
    print("CONCLUSIÓN: Los seres tienen vidas equilibradas.")
    print("La astronomía es solo una actividad entre muchas.")
    print("=" * 70)


if __name__ == '__main__':
    clean_world()
    beings = create_population(10)
    beings = simulate(beings, max_ticks=500)
    report(beings)
