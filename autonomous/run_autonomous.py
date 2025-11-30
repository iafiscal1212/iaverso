#!/usr/bin/env python3
"""
NEO_EVA Autonomous Runner
=========================

Script principal para ejecutar el sistema autónomo.

Uso:
    python run_autonomous.py                    # Run indefinidamente
    python run_autonomous.py --steps 1000       # Run 1000 pasos
    python run_autonomous.py --test             # Test rápido (100 pasos)
    python run_autonomous.py --status           # Ver estado actual
    python run_autonomous.py --reset            # Resetear estado

El sistema:
1. Maximiza S (proto-subjetividad)
2. Se auto-optimiza
3. Puede evolucionar su código
4. Interactúa con archivos en world/
5. Es monitoreado por watchdog

Para detener: Ctrl+C (guardará estado)
"""

import sys
import signal
import argparse
import json
from pathlib import Path
from datetime import datetime

# Añadir path al proyecto
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autonomous.autonomous_core import AutonomousCore
from autonomous.watchdog import Watchdog, WatchdogDaemon


def signal_handler(signum, frame):
    """Maneja Ctrl+C gracefully."""
    print("\n\nDeteniendo sistema autónomo...")
    raise KeyboardInterrupt


def run_autonomous(max_steps: int = None, verbose: bool = True):
    """
    Ejecuta el sistema autónomo.

    Args:
        max_steps: Número máximo de pasos (None = infinito)
        verbose: Mostrar progreso
    """
    print("=" * 60)
    print("NEO_EVA AUTONOMOUS SYSTEM")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Max steps: {max_steps if max_steps else 'unlimited'}")
    print("-" * 60)

    # Configurar signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Iniciar watchdog daemon
    watchdog_daemon = WatchdogDaemon(check_interval=1.0)
    watchdog_daemon.start()
    print("Watchdog daemon started")

    # Crear core autónomo
    try:
        core = AutonomousCore()
        print(f"Core initialized. Starting S: {core.state.S:.4f}")
    except Exception as e:
        print(f"ERROR initializing core: {e}")
        watchdog_daemon.stop()
        return

    print("-" * 60)
    print("Running... (Ctrl+C to stop)")
    print("-" * 60)

    try:
        # Ejecutar loop principal
        final_state = core.run(max_steps=max_steps)

        print("\n" + "=" * 60)
        print("AUTONOMOUS RUN COMPLETED")
        print("=" * 60)

        if final_state:
            print(f"Final S: {final_state.get('S', 'N/A'):.6f}")
            print(f"Final stability: {final_state.get('stability', 'N/A'):.6f}")
            print(f"Total steps: {final_state.get('step', 0)}")

            # Estadísticas
            print("\nComponent Statistics:")
            for component, stats in final_state.get('statistics', {}).items():
                print(f"  {component}:")
                for key, val in stats.items():
                    if isinstance(val, float):
                        print(f"    {key}: {val:.4f}")
                    else:
                        print(f"    {key}: {val}")

    except KeyboardInterrupt:
        print("\n" + "-" * 60)
        print("Interrupted by user")
        print(f"Final S: {core.state.S:.6f}")
        print(f"Steps completed: {core.state.step}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Detener watchdog
        watchdog_daemon.stop()
        print("\nWatchdog daemon stopped")
        print(f"Ended: {datetime.now().isoformat()}")


def show_status():
    """Muestra estado actual del sistema."""
    print("=" * 60)
    print("NEO_EVA AUTONOMOUS STATUS")
    print("=" * 60)

    state_dir = Path(__file__).parent / "state"

    # Core state
    core_file = state_dir / "core_state.json"
    if core_file.exists():
        with open(core_file, 'r') as f:
            core = json.load(f)
        print("\nCore State:")
        print(f"  S: {core.get('S', 'N/A')}")
        print(f"  Stability: {core.get('stability', 'N/A')}")
        print(f"  Step: {core.get('step', 0)}")
        print(f"  Last action: {core.get('last_action', 'N/A')}")
    else:
        print("\nCore State: Not initialized")

    # Optimizer state
    opt_file = state_dir / "optimizer_state.json"
    if opt_file.exists():
        with open(opt_file, 'r') as f:
            opt = json.load(f)
        print("\nOptimizer State:")
        print(f"  Optimizations: {opt.get('n_optimizations', 0)}")
        print(f"  Improvements: {opt.get('n_improvements', 0)}")
    else:
        print("\nOptimizer State: Not initialized")

    # Evolver state
    evo_file = state_dir / "evolver_state.json"
    if evo_file.exists():
        with open(evo_file, 'r') as f:
            evo = json.load(f)
        print("\nEvolver State:")
        print(f"  Evolutions: {evo.get('n_evolutions', 0)}")
        print(f"  Successful: {evo.get('n_successful', 0)}")
        print(f"  Modules: {evo.get('modules', [])}")
    else:
        print("\nEvolver State: Not initialized")

    # World state
    world_file = state_dir / "world_state.json"
    if world_file.exists():
        with open(world_file, 'r') as f:
            world = json.load(f)
        print("\nWorld State:")
        print(f"  Perceptions: {world.get('n_perceptions', 0)}")
        print(f"  Actions: {world.get('n_actions', 0)}")
        print(f"  Files: {world.get('n_files', 0)}")
    else:
        print("\nWorld State: Not initialized")

    # Watchdog state
    watch_file = state_dir / "watchdog_state.json"
    if watch_file.exists():
        with open(watch_file, 'r') as f:
            watch = json.load(f)
        print("\nWatchdog State:")
        print(f"  Checks: {watch.get('n_checks', 0)}")
        print(f"  Warnings: {watch.get('n_warnings', 0)}")
        print(f"  Stops: {watch.get('n_stops', 0)}")
        print(f"  Running: {watch.get('is_running', True)}")
    else:
        print("\nWatchdog State: Not initialized")


def reset_state():
    """Resetea todo el estado del sistema."""
    print("Resetting autonomous system state...")

    state_dir = Path(__file__).parent / "state"
    world_dir = Path(__file__).parent / "world"
    code_dir = Path(__file__).parent / "code"
    logs_dir = Path(__file__).parent / "logs"

    # Limpiar state
    for f in state_dir.glob("*.json"):
        f.unlink()
        print(f"  Deleted: {f.name}")

    # Limpiar world
    for f in world_dir.glob("*"):
        if f.is_file():
            f.unlink()
            print(f"  Deleted: world/{f.name}")

    # Limpiar code (excepto __init__.py)
    for f in code_dir.glob("*.py"):
        if f.name != "__init__.py":
            f.unlink()
            print(f"  Deleted: code/{f.name}")

    # Limpiar logs antiguos
    for f in logs_dir.glob("*.log"):
        f.unlink()
        print(f"  Deleted: logs/{f.name}")

    print("\nReset complete.")


def main():
    parser = argparse.ArgumentParser(
        description="NEO_EVA Autonomous System Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_autonomous.py              # Run indefinitely
    python run_autonomous.py --steps 100  # Run 100 steps
    python run_autonomous.py --test       # Quick test (100 steps)
    python run_autonomous.py --status     # Show current state
    python run_autonomous.py --reset      # Reset all state
        """
    )

    parser.add_argument('--steps', type=int, default=None,
                        help="Maximum number of steps to run")
    parser.add_argument('--test', action='store_true',
                        help="Quick test run (100 steps)")
    parser.add_argument('--status', action='store_true',
                        help="Show current system status")
    parser.add_argument('--reset', action='store_true',
                        help="Reset all system state")
    parser.add_argument('--quiet', action='store_true',
                        help="Reduce output verbosity")

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.reset:
        confirm = input("This will delete all state. Continue? [y/N] ")
        if confirm.lower() == 'y':
            reset_state()
        else:
            print("Aborted.")
    elif args.test:
        run_autonomous(max_steps=100, verbose=not args.quiet)
    else:
        run_autonomous(max_steps=args.steps, verbose=not args.quiet)


if __name__ == "__main__":
    main()
