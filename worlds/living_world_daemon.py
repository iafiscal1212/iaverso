#!/usr/bin/env python3
"""
Mundo Vivo - Daemon 24/7
========================

Los seres viven continuamente con:
- Ciclos circadianos (día/noche)
- Estaciones
- Eventos naturales

El daemon corre indefinidamente, guardando estado cada minuto.
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
import time
import json
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from worlds.complete_being import (
    CompleteBeing, create_complete_being,
    describe_personality, Action
)

# Configuración
CONFIG = {
    'tick_interval': 1.0,      # Segundos entre ticks
    'ticks_per_day': 100,      # Ticks por día virtual
    'max_population': 50,      # Límite de población
    'initial_population': 10,  # Población inicial
    'save_interval': 60,       # Guardar estado cada N segundos
    'log_interval': 10,        # Log cada N ticks
}

# Estado global
world_state = {
    'tick': 0,
    'day': 0,
    'hour': 0.0,  # 0-24
    'season': 'primavera',
    'is_night': False,
    'births_total': 0,
    'deaths_total': 0,
    'started_at': None,
    'last_save': None,
}

# Ruta de datos
DATA_PATH = Path('/root/NEO_EVA/worlds/living_world_state.json')
LOG_PATH = Path('/root/NEO_EVA/logs/living_world.log')

# Flag para shutdown graceful
running = True


def signal_handler(sig, frame):
    """Manejar Ctrl+C."""
    global running
    print("\n\nRecibida señal de parada. Guardando estado...")
    running = False


def get_circadian_state(tick: int) -> Dict:
    """
    Calcular estado circadiano.

    Un día = 100 ticks
    Noche: 20:00 - 06:00 (ticks 83-100, 0-25)
    """
    ticks_per_day = CONFIG['ticks_per_day']

    # Hora del día (0-24)
    day_progress = (tick % ticks_per_day) / ticks_per_day
    hour = day_progress * 24

    # Día número
    day = tick // ticks_per_day

    # ¿Es de noche? (20:00 - 06:00)
    is_night = hour >= 20 or hour < 6

    # Estación (cada 30 días virtuales)
    season_idx = (day // 30) % 4
    seasons = ['primavera', 'verano', 'otoño', 'invierno']
    season = seasons[season_idx]

    # Luz solar (afecta energía y ánimo)
    if is_night:
        sunlight = 0.1
    elif 6 <= hour < 8:  # Amanecer
        sunlight = 0.3 + (hour - 6) * 0.35
    elif 18 <= hour < 20:  # Atardecer
        sunlight = 1.0 - (hour - 18) * 0.45
    else:  # Día pleno
        sunlight = 1.0

    # Temperatura base por estación
    season_temps = {
        'primavera': 50,
        'verano': 70,
        'otoño': 45,
        'invierno': 25,
    }
    base_temp = season_temps[season]

    # Variación día/noche
    if is_night:
        temp = base_temp - 15
    else:
        temp = base_temp + 10 * np.sin((hour - 6) / 12 * np.pi)

    # Recursos varían por estación
    season_resources = {
        'primavera': 70,
        'verano': 80,
        'otoño': 60,
        'invierno': 40,
    }
    resources = season_resources[season] + np.random.uniform(-10, 10)

    return {
        'hour': hour,
        'day': day,
        'is_night': is_night,
        'season': season,
        'sunlight': sunlight,
        'temperature': temp,
        'resources': resources,
    }


def apply_circadian_effects(being: CompleteBeing, circadian: Dict):
    """Aplicar efectos circadianos a un ser."""

    # De noche: más cansancio, menos actividad
    if circadian['is_night']:
        # Necesitan descansar más
        being.body.energy -= 0.3  # Extra fatiga nocturna

        # Menos ansiedad de noche (si están seguros)
        if being.body.health > 50:
            being.emotions.anxiety = max(0, being.emotions.anxiety - 0.02)
            being.emotions.peace = min(1, being.emotions.peace + 0.02)

        # Los que miran el cielo, de noche ven más
        if being.dna.get('sky_gazing', 0) > 0.6:
            # Noche despejada = mejor para observar
            being.mind.think("Las estrellas brillan...")

    # De día: más energía de la luz
    else:
        # La luz da energía (fotosíntesis emocional)
        sunlight = circadian['sunlight']
        being.emotions.joy = min(1, being.emotions.joy + sunlight * 0.01)

        # Amanecer/atardecer son momentos especiales
        hour = circadian['hour']
        if 6 <= hour < 7:  # Amanecer
            being.mind.think("Nuevo día...")
            being.emotions.peace += 0.05
        elif 19 <= hour < 20:  # Atardecer
            being.mind.think("El día termina...")

    # Efectos estacionales
    season = circadian['season']
    if season == 'invierno':
        # Invierno duro
        if being.body.warmth < 40:
            being.emotions.sadness = min(1, being.emotions.sadness + 0.02)
    elif season == 'primavera':
        # Primavera alegre
        being.emotions.joy = min(1, being.emotions.joy + 0.01)
        # Más ganas de reproducirse
        if being.body.age > 100:
            being.emotions.love = min(1, being.emotions.love + 0.01)


def save_world_state(beings: List[CompleteBeing]):
    """Guardar estado del mundo."""
    alive = [b for b in beings if b.alive]

    state = {
        'world': world_state,
        'population': len(alive),
        'beings': [b.id for b in alive],
        'timestamp': datetime.now().isoformat(),
    }

    with open(DATA_PATH, 'w') as f:
        json.dump(state, f, indent=2)

    world_state['last_save'] = datetime.now().isoformat()


def load_world_state() -> List[CompleteBeing]:
    """Cargar estado del mundo si existe."""
    global world_state

    if not DATA_PATH.exists():
        return None

    try:
        with open(DATA_PATH, 'r') as f:
            state = json.load(f)

        world_state.update(state['world'])

        # Cargar seres
        beings = []
        for being_id in state['beings']:
            try:
                being = CompleteBeing(being_id)
                if being.alive:
                    beings.append(being)
            except:
                pass

        if beings:
            print(f"Cargados {len(beings)} seres del estado anterior")
            return beings
    except Exception as e:
        print(f"Error cargando estado: {e}")

    return None


def log_world(beings: List[CompleteBeing], circadian: Dict):
    """Log del estado actual."""
    alive = [b for b in beings if b.alive]

    hour = circadian['hour']
    hour_str = f"{int(hour):02d}:{int((hour % 1) * 60):02d}"

    day_night = "🌙" if circadian['is_night'] else "☀️"
    season_emoji = {
        'primavera': '🌸',
        'verano': '🌻',
        'otoño': '🍂',
        'invierno': '❄️',
    }

    msg = f"[Día {circadian['day']} {hour_str}] {day_night} {season_emoji[circadian['season']]} "
    msg += f"Población: {len(alive)} | "
    msg += f"Nacimientos: {world_state['births_total']} | "
    msg += f"Muertes: {world_state['deaths_total']}"

    print(msg)

    # Log a archivo
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'a') as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


def run_world():
    """Ejecutar el mundo."""
    global running, world_state

    print("=" * 70)
    print("MUNDO VIVO - DAEMON 24/7")
    print("=" * 70)
    print(f"Tick interval: {CONFIG['tick_interval']}s")
    print(f"Ticks por día: {CONFIG['ticks_per_day']}")
    print(f"Max población: {CONFIG['max_population']}")
    print("Presiona Ctrl+C para detener")
    print("=" * 70)
    print()

    # Cargar o crear mundo
    beings = load_world_state()

    if beings is None:
        print("Creando nuevo mundo...")
        beings = [create_complete_being() for _ in range(CONFIG['initial_population'])]
        world_state['started_at'] = datetime.now().isoformat()

        print(f"\nNacen {len(beings)} seres:")
        for b in beings:
            print(f"  {b.id}: {describe_personality(b.dna)}")
        print()

    last_save_time = time.time()

    # Loop principal
    while running:
        tick_start = time.time()

        # Estado circadiano
        circadian = get_circadian_state(world_state['tick'])
        world_state['hour'] = circadian['hour']
        world_state['day'] = circadian['day']
        world_state['season'] = circadian['season']
        world_state['is_night'] = circadian['is_night']

        # Simular
        alive = [b for b in beings if b.alive]

        if not alive:
            print("\n¡EXTINCIÓN! Creando nueva generación...")
            beings = [create_complete_being() for _ in range(CONFIG['initial_population'])]
            world_state['tick'] = 0
            continue

        # Limitar procesamiento si hay muchos
        to_process = alive[:CONFIG['max_population']]

        new_beings = []
        for being in to_process:
            # Aplicar efectos circadianos
            apply_circadian_effects(being, circadian)

            # Vivir momento
            result = being.live_moment(
                world_temp=circadian['temperature'],
                world_resources=circadian['resources'],
                world_danger=0.1,
                others=alive,
                tick=world_state['tick'],
            )

            if result['status'] == 'died':
                world_state['deaths_total'] += 1
                print(f"  💀 {being.id} murió: {result['cause']}")

            elif result.get('new_being'):
                new_being = result['new_being']
                new_beings.append(new_being)
                world_state['births_total'] += 1
                print(f"  👶 Nace {new_being.id}")

        beings.extend(new_beings)

        # Log periódico
        if world_state['tick'] % CONFIG['log_interval'] == 0:
            log_world(beings, circadian)

        # Guardar periódicamente
        if time.time() - last_save_time > CONFIG['save_interval']:
            save_world_state(beings)
            last_save_time = time.time()

        world_state['tick'] += 1

        # Esperar hasta el próximo tick
        elapsed = time.time() - tick_start
        sleep_time = max(0, CONFIG['tick_interval'] - elapsed)
        time.sleep(sleep_time)

    # Guardar al salir
    print("\nGuardando estado final...")
    save_world_state(beings)
    print(f"Estado guardado en {DATA_PATH}")
    print(f"Log en {LOG_PATH}")
    print("\n¡Hasta pronto!")


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    run_world()
