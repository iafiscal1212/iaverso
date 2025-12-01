"""
Test Rápido de Memoria Episódica
================================

Verifica que la segmentación sea razonable:
- 30-80 episodios en 2000 pasos
- Cortes donde hay cambios reales (crisis, régimen, drives)
- No mecánico (variabilidad en longitudes)
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from cognition.episodic_memory import EpisodicMemory
from cognition.narrative_memory import NarrativeMemory
from world1.world1_core import World1Core
from world1.world1_regimes import RegimeDetector
from experiments.autonomous_life import AutonomousAgent


@dataclass
class EpisodeAnalysis:
    """Análisis de un episodio."""
    idx: int
    length: int
    delta_drives: float  # Cambio en drives al cortar
    delta_phi: float     # Cambio en φ al cortar
    regime_at_cut: int   # Régimen cuando se cortó
    had_crisis: bool     # Si hubo crisis durante el episodio


def run_episodic_test(T: int = 2000):
    """
    Test de memoria episódica con WORLD-1 + agentes.

    Qué esperamos:
    - 30-80 episodios (longitud media 25-70)
    - Cortes concentrados en cambios de drives/crisis/régimen
    - Variabilidad en longitudes (no mecánico)
    """
    print("=" * 70)
    print("TEST RÁPIDO DE MEMORIA EPISÓDICA")
    print("=" * 70)
    print(f"\nSimulando T={T} pasos...")

    # Crear mundo
    world = World1Core(n_fields=4, n_entities=5, n_resources=3, n_modes=3)
    regime_detector = RegimeDetector(world.D)

    # Crear agentes con memoria episódica
    agent_names = ['NEO', 'EVA']
    agents: Dict[str, AutonomousAgent] = {}
    memories: Dict[str, EpisodicMemory] = {}

    for name in agent_names:
        agents[name] = AutonomousAgent(name, dim=6)
        memories[name] = EpisodicMemory(z_dim=6, phi_dim=5, D_dim=6)

    # Tracking
    regime_history = []
    crisis_times = {name: [] for name in agent_names}
    drive_changes = {name: [] for name in agent_names}

    # Simular
    for t in range(T):
        w = world.w

        # Detectar régimen
        regime = regime_detector.detect_regime(w)
        regime_history.append(regime)

        for name, agent in agents.items():
            # Stimulus del mundo
            stimulus = np.zeros(6)
            for i in range(min(len(w), 6)):
                stimulus[i] = w[i % len(w)] * 0.1

            # Otro agente
            other_name = 'EVA' if name == 'NEO' else 'NEO'
            other_z = agents[other_name].z

            # Guardar drives antes
            z_before = agent.z.copy()

            # Step del agente
            agent.step(stimulus, other_z)

            # Calcular cambio en drives
            delta_z = np.linalg.norm(agent.z - z_before)
            drive_changes[name].append(delta_z)

            # Registrar crisis
            if agent.in_crisis:
                crisis_times[name].append(t)

            # Computar φ simple
            phi = np.zeros(5)
            phi[0] = agent.integration
            if len(agent.z_history) > 1:
                phi[1] = np.linalg.norm(agent.z - agent.z_history[-1])
            z_norm = np.abs(agent.z) / (np.sum(np.abs(agent.z)) + 1e-8)
            phi[2] = -np.sum(z_norm * np.log(z_norm + 1e-8))
            phi[3] = agent.identity_strength
            phi[4] = min(1.0, t / 500)

            # Grabar en memoria episódica
            tau = t * (1 + 0.1 * np.linalg.norm(phi))
            memories[name].record(agent.z, phi, agent.z, tau)

        # Evolucionar mundo
        world.step({})

        if (t + 1) % 500 == 0:
            print(f"  t={t+1}: ", end="")
            for name in agent_names:
                n_ep = len(memories[name].episodes)
                print(f"{name}={n_ep} episodios  ", end="")
            print()

    # Análisis
    print("\n" + "=" * 70)
    print("ANÁLISIS POR AGENTE")
    print("=" * 70)

    results = {}

    for name in agent_names:
        mem = memories[name]
        episodes = mem.episodes

        print(f"\n{'─' * 35}")
        print(f"  {name}")
        print(f"{'─' * 35}")

        n_episodes = len(episodes)
        print(f"  Nº episodios: {n_episodes}")

        if n_episodes < 2:
            print("  ⚠️  Muy pocos episodios para analizar")
            continue

        # Distribución de longitudes
        lengths = [e.length for e in episodes]
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)

        print(f"\n  Longitudes de episodio:")
        print(f"    Media: {mean_len:.1f}")
        print(f"    Std: {std_len:.1f}")
        print(f"    Min: {min(lengths)}, Max: {max(lengths)}")
        print(f"    Percentiles: p25={np.percentile(lengths, 25):.0f}, "
              f"p50={np.percentile(lengths, 50):.0f}, "
              f"p75={np.percentile(lengths, 75):.0f}")

        # Histograma simple
        bins = [0, 10, 20, 30, 50, 100, 200, 500]
        hist, _ = np.histogram(lengths, bins=bins)
        print(f"\n  Histograma de longitudes:")
        for i in range(len(hist)):
            bar = "█" * int(hist[i] * 40 / max(hist) if max(hist) > 0 else 0)
            print(f"    {bins[i]:3d}-{bins[i+1]:3d}: {bar} ({hist[i]})")

        # Análisis de cortes
        print(f"\n  Análisis de cortes:")

        cuts_at_crisis = 0
        cuts_at_regime_change = 0
        cuts_at_high_drive_change = 0

        # Percentil 75 de cambios de drive para umbral
        drive_threshold = np.percentile(drive_changes[name], 75)

        for i, ep in enumerate(episodes[:-1]):  # Excepto el último
            t_cut = ep.t_end

            # ¿Hubo crisis cerca del corte?
            crisis_near = any(abs(ct - t_cut) < 5 for ct in crisis_times[name])
            if crisis_near:
                cuts_at_crisis += 1

            # ¿Cambió el régimen cerca del corte?
            if t_cut < len(regime_history) - 1:
                regime_changed = regime_history[t_cut] != regime_history[min(t_cut + 1, len(regime_history) - 1)]
                if regime_changed:
                    cuts_at_regime_change += 1

            # ¿Alto cambio de drives cerca del corte?
            if t_cut < len(drive_changes[name]):
                if drive_changes[name][t_cut] > drive_threshold:
                    cuts_at_high_drive_change += 1

        n_cuts = n_episodes - 1
        print(f"    Cortes en crisis: {cuts_at_crisis}/{n_cuts} ({100*cuts_at_crisis/max(1,n_cuts):.0f}%)")
        print(f"    Cortes en cambio régimen: {cuts_at_regime_change}/{n_cuts} ({100*cuts_at_regime_change/max(1,n_cuts):.0f}%)")
        print(f"    Cortes en alto Δdrives: {cuts_at_high_drive_change}/{n_cuts} ({100*cuts_at_high_drive_change/max(1,n_cuts):.0f}%)")

        # Variabilidad (coeficiente de variación)
        cv = std_len / mean_len if mean_len > 0 else 0
        print(f"\n  Coef. variación longitudes: {cv:.2f}")

        if cv < 0.3:
            print("    ⚠️  Longitudes muy uniformes (posible umbral mecánico)")
        elif cv > 2.0:
            print("    ⚠️  Variabilidad extrema (posible inestabilidad)")
        else:
            print("    ✓  Variabilidad saludable")

        # Evaluación
        print(f"\n  EVALUACIÓN:")

        # Criterio 1: Número de episodios razonable
        if 30 <= n_episodes <= 80:
            print(f"    ✓ Nº episodios en rango [30, 80]: {n_episodes}")
        elif n_episodes < 30:
            print(f"    ⚠️  Pocos episodios ({n_episodes}): umbral muy alto")
        else:
            print(f"    ⚠️  Muchos episodios ({n_episodes}): umbral muy bajo")

        # Criterio 2: Longitud media razonable
        expected_mean = T / n_episodes
        if 25 <= mean_len <= 70:
            print(f"    ✓ Longitud media en rango [25, 70]: {mean_len:.1f}")
        else:
            print(f"    ⚠️  Longitud media fuera de rango: {mean_len:.1f}")

        # Criterio 3: Cortes no aleatorios
        meaningful_cuts = cuts_at_crisis + cuts_at_regime_change + cuts_at_high_drive_change
        meaningful_pct = 100 * meaningful_cuts / max(1, n_cuts)
        if meaningful_pct > 40:
            print(f"    ✓ Cortes significativos: {meaningful_pct:.0f}%")
        else:
            print(f"    ⚠️  Pocos cortes significativos: {meaningful_pct:.0f}%")

        results[name] = {
            'n_episodes': n_episodes,
            'mean_length': mean_len,
            'std_length': std_len,
            'cv': cv,
            'cuts_at_crisis_pct': 100 * cuts_at_crisis / max(1, n_cuts),
            'cuts_at_regime_pct': 100 * cuts_at_regime_change / max(1, n_cuts),
            'meaningful_cuts_pct': meaningful_pct
        }

    # Test de coherencia narrativa
    print("\n" + "=" * 70)
    print("TEST DE COHERENCIA NARRATIVA")
    print("=" * 70)

    for name in agent_names:
        mem = memories[name]
        narrative = NarrativeMemory(mem)

        # Actualizar narrativa
        for _ in range(5):
            narrative.update()

        print(f"\n{'─' * 35}")
        print(f"  {name}")
        print(f"{'─' * 35}")

        # Calcular coherencias entre episodios consecutivos
        coherences = []
        for i in range(len(mem.episodes) - 1):
            e1 = mem.episodes[i]
            e2 = mem.episodes[i + 1]
            coh = mem.similarity(e1, e2)
            coherences.append((i, i+1, coh))

        if len(coherences) == 0:
            print("  No hay suficientes episodios")
            continue

        coh_values = [c[2] for c in coherences]

        print(f"  Coherencia entre episodios consecutivos:")
        print(f"    Media: {np.mean(coh_values):.3f}")
        print(f"    p25: {np.percentile(coh_values, 25):.3f}")
        print(f"    p50: {np.percentile(coh_values, 50):.3f}")
        print(f"    p75: {np.percentile(coh_values, 75):.3f}")

        # Top 10 saltos raros (coherencia más baja)
        sorted_coh = sorted(coherences, key=lambda x: x[2])
        print(f"\n  Top 10 'giros de trama' (menor coherencia):")
        for i, (e1_idx, e2_idx, coh) in enumerate(sorted_coh[:10]):
            e1 = mem.episodes[e1_idx]
            print(f"    {i+1}. Ep {e1_idx}→{e2_idx}: coh={coh:.3f} (t={e1.t_end})")

        # Cadena narrativa dominante
        summary = narrative.get_narrative_summary()
        if 'dominant_chain' in summary:
            print(f"\n  Cadena narrativa dominante:")
            print(f"    Episodios: {summary['dominant_chain'][:10]}...")
            print(f"    Probabilidad: {summary['dominant_chain_prob']:.6f}")

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    all_pass = True
    for name, r in results.items():
        print(f"\n  {name}:")
        checks = [
            30 <= r['n_episodes'] <= 80,
            25 <= r['mean_length'] <= 70,
            0.3 <= r['cv'] <= 2.0,
            r['meaningful_cuts_pct'] > 40
        ]
        status = "✓ PASS" if all(checks) else "⚠️  PARCIAL"
        if not all(checks):
            all_pass = False
        print(f"    {status}: {r['n_episodes']} episodios, "
              f"long={r['mean_length']:.1f}±{r['std_length']:.1f}, "
              f"cortes_signif={r['meaningful_cuts_pct']:.0f}%")

    print("\n" + "=" * 70)
    if all_pass:
        print("  MEMORIA EPISÓDICA: FUNCIONANDO CORRECTAMENTE")
    else:
        print("  MEMORIA EPISÓDICA: NECESITA AJUSTES")
    print("=" * 70)

    return results, memories


if __name__ == "__main__":
    run_episodic_test(T=2000)
