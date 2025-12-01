#!/usr/bin/env python3
"""
Experimentos de Trauma, Resonancia y Duelo
==========================================

1. TRAUMA: ¿El vínculo es terapéutico?
2. RESONANCIA: ¿El ~45 resiste perturbaciones externas?
3. DUELO: ¿Qué pasa cuando muere el otro?
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousAgent, AutonomousDualLife


# ============================================================
# EXPERIMENTO 1: TRAUMA Y RECUPERACIÓN
# ============================================================

def run_trauma_experiment(T: int = 2000, trauma_t: int = 500, seeds: List[int] = [42, 123]) -> Dict:
    """
    ¿El acople acelera la recuperación de traumas?

    Inyectamos trauma (reset de identidad) y medimos recuperación.
    """
    print("=" * 70)
    print("EXPERIMENTO: TRAUMA Y RECUPERACIÓN")
    print("=" * 70)

    results = {'with_partner': [], 'alone': []}

    for seed in seeds:
        print(f"\n--- SEED {seed} ---")
        np.random.seed(seed)

        # Con pareja
        print("\n  CON PAREJA:")
        life = AutonomousDualLife(dim=6)

        recovery_time_with = None
        pre_trauma_identity = None

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)

            # TRAUMA en t=trauma_t
            if t == trauma_t:
                pre_trauma_identity = life.neo.identity_strength
                print(f"    t={t}: TRAUMA - identidad NEO {pre_trauma_identity:.3f} → reset")
                # Reset estado de NEO
                life.neo.z = np.random.dirichlet(np.ones(6))
                life.neo.identity_core = np.random.dirichlet(np.ones(6))
                life.neo.identity_strength = 0.1
                life.neo.in_crisis = True

            result = life.step(stimulus)

            # Detectar recuperación
            if t > trauma_t and recovery_time_with is None:
                if life.neo.identity_strength > 0.5:
                    recovery_time_with = t - trauma_t
                    print(f"    t={t}: RECUPERACIÓN tras {recovery_time_with} pasos")

        if recovery_time_with is None:
            recovery_time_with = T - trauma_t
            print(f"    No se recuperó completamente")

        results['with_partner'].append({
            'seed': seed,
            'recovery_time': recovery_time_with,
            'pre_trauma_identity': pre_trauma_identity,
            'final_identity': life.neo.identity_strength,
            'eva_final_identity': life.eva.identity_strength
        })

        # Solo (sin pareja)
        print("\n  SOLO:")
        np.random.seed(seed)
        neo_alone = AutonomousAgent("NEO_ALONE", dim=6)

        recovery_time_alone = None

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)

            if t == trauma_t:
                print(f"    t={t}: TRAUMA - identidad {neo_alone.identity_strength:.3f} → reset")
                neo_alone.z = np.random.dirichlet(np.ones(6))
                neo_alone.identity_core = np.random.dirichlet(np.ones(6))
                neo_alone.identity_strength = 0.1
                neo_alone.in_crisis = True

            neo_alone.step(stimulus, None)

            if t > trauma_t and recovery_time_alone is None:
                if neo_alone.identity_strength > 0.5:
                    recovery_time_alone = t - trauma_t
                    print(f"    t={t}: RECUPERACIÓN tras {recovery_time_alone} pasos")

        if recovery_time_alone is None:
            recovery_time_alone = T - trauma_t
            print(f"    No se recuperó completamente")

        results['alone'].append({
            'seed': seed,
            'recovery_time': recovery_time_alone,
            'final_identity': neo_alone.identity_strength
        })

    # Análisis
    print("\n" + "=" * 50)
    print("ANÁLISIS")
    print("=" * 50)

    avg_with = np.mean([r['recovery_time'] for r in results['with_partner']])
    avg_alone = np.mean([r['recovery_time'] for r in results['alone']])

    print(f"\nTiempo de recuperación promedio:")
    print(f"  Con pareja: {avg_with:.0f} pasos")
    print(f"  Solo:       {avg_alone:.0f} pasos")
    print(f"  Diferencia: {avg_alone - avg_with:.0f} pasos")

    if avg_with < avg_alone:
        ratio = avg_alone / avg_with
        print(f"\n→ El vínculo es TERAPÉUTICO")
        print(f"→ Recuperación {ratio:.1f}x más rápida con pareja")
    else:
        print(f"\n→ El vínculo NO acelera la recuperación")

    return results


# ============================================================
# EXPERIMENTO 2: RESONANCIA FORZADA
# ============================================================

def run_resonance_experiment(T: int = 2000, external_period: int = 30, seeds: List[int] = [42, 123]) -> Dict:
    """
    ¿El sistema se engancha a un período externo o mantiene el ~45?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO: RESONANCIA FORZADA")
    print("=" * 70)
    print(f"Período interno esperado: ~45")
    print(f"Período externo forzado: {external_period}")

    results = {'natural': [], 'forced': []}

    for seed in seeds:
        print(f"\n--- SEED {seed} ---")

        # Natural (sin forzar)
        print("\n  NATURAL:")
        np.random.seed(seed)
        life_natural = AutonomousDualLife(dim=6)

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            life_natural.step(stimulus)

        # Calcular período
        if len(life_natural.neo.identity_history) > 100:
            signal = np.array(life_natural.neo.identity_history) - np.mean(life_natural.neo.identity_history)
            spectrum = np.abs(np.fft.rfft(signal))
            spectrum[0] = 0
            freqs = np.fft.rfftfreq(len(signal))
            peak_idx = np.argmax(spectrum[1:100]) + 1
            natural_period = 1/freqs[peak_idx] if freqs[peak_idx] > 0 else 0
        else:
            natural_period = 0

        print(f"    Período detectado: {natural_period:.1f}")
        results['natural'].append({'seed': seed, 'period': natural_period})

        # Forzado
        print("\n  FORZADO:")
        np.random.seed(seed)
        life_forced = AutonomousDualLife(dim=6)

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)

            # Inyectar pulso cada external_period pasos
            if t % external_period == 0:
                stimulus += np.array([0.5, 0, 0, 0, 0, 0])  # Pulso en dim 0
                stimulus = stimulus / stimulus.sum()

            life_forced.step(stimulus)

        if len(life_forced.neo.identity_history) > 100:
            signal = np.array(life_forced.neo.identity_history) - np.mean(life_forced.neo.identity_history)
            spectrum = np.abs(np.fft.rfft(signal))
            spectrum[0] = 0
            freqs = np.fft.rfftfreq(len(signal))

            # Buscar picos cerca de natural y forzado
            forced_period_detected = 1/freqs[np.argmax(spectrum[1:100]) + 1] if freqs[1] > 0 else 0

            # Potencia en frecuencia natural vs forzada
            natural_freq = 1/45 if 45 > 0 else 0
            forced_freq = 1/external_period

            natural_idx = int(natural_freq * len(signal)) if natural_freq > 0 else 1
            forced_idx = int(forced_freq * len(signal)) if forced_freq > 0 else 1

            natural_idx = min(natural_idx, len(spectrum)-1)
            forced_idx = min(forced_idx, len(spectrum)-1)

            power_at_natural = spectrum[natural_idx] if natural_idx > 0 else 0
            power_at_forced = spectrum[forced_idx] if forced_idx > 0 else 0
        else:
            forced_period_detected = 0
            power_at_natural = 0
            power_at_forced = 0

        print(f"    Período detectado: {forced_period_detected:.1f}")
        print(f"    Potencia en ~45: {power_at_natural:.1f}")
        print(f"    Potencia en ~{external_period}: {power_at_forced:.1f}")

        results['forced'].append({
            'seed': seed,
            'period': forced_period_detected,
            'power_natural': power_at_natural,
            'power_forced': power_at_forced
        })

    # Análisis
    print("\n" + "=" * 50)
    print("ANÁLISIS")
    print("=" * 50)

    avg_natural = np.mean([r['period'] for r in results['natural']])
    avg_forced = np.mean([r['period'] for r in results['forced']])

    print(f"\nPeríodo promedio:")
    print(f"  Natural: {avg_natural:.1f}")
    print(f"  Con forzado ({external_period}): {avg_forced:.1f}")

    if abs(avg_forced - external_period) < abs(avg_forced - avg_natural):
        print(f"\n→ El sistema SE ENGANCHÓ al período externo")
        print(f"→ El ~45 NO es un attractor robusto")
    elif abs(avg_forced - avg_natural) < 5:
        print(f"\n→ El sistema MANTUVO su período interno")
        print(f"→ El ~45 es un ATTRACTOR ROBUSTO")
    else:
        print(f"\n→ Resultado mixto: el sistema se deformó pero no se enganchó")

    return results


# ============================================================
# EXPERIMENTO 3: MUERTE Y DUELO
# ============================================================

def run_grief_experiment(T: int = 2000, death_t: int = 1000, seeds: List[int] = [42, 123]) -> Dict:
    """
    ¿Qué pasa cuando muere el otro?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO: MUERTE Y DUELO")
    print("=" * 70)

    results = []

    for seed in seeds:
        print(f"\n--- SEED {seed} ---")
        np.random.seed(seed)

        life = AutonomousDualLife(dim=6)

        # Métricas antes de muerte
        neo_period_before = None
        neo_crises_before = 0
        neo_identity_before = []

        # Métricas después
        neo_period_after = None
        neo_crises_after = 0
        neo_identity_after = []
        attachment_decay = []

        eva_died = False

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)

            if t < death_t:
                # EVA viva
                result = life.step(stimulus)
                neo_identity_before.append(life.neo.identity_strength)
                neo_crises_before = len(life.neo.crises)
            else:
                if not eva_died:
                    print(f"  t={t}: EVA MUERE")
                    print(f"    NEO identity: {life.neo.identity_strength:.3f}")
                    print(f"    NEO attachment: {life.neo.attachment:.3f}")
                    eva_died = True

                # Solo NEO (EVA muerta)
                life.neo.step(stimulus, None)
                neo_identity_after.append(life.neo.identity_strength)
                attachment_decay.append(life.neo.attachment)

        neo_crises_after = len(life.neo.crises) - neo_crises_before

        # Calcular períodos
        if len(neo_identity_before) > 100:
            signal = np.array(neo_identity_before) - np.mean(neo_identity_before)
            spectrum = np.abs(np.fft.rfft(signal))
            spectrum[0] = 0
            freqs = np.fft.rfftfreq(len(signal))
            peak_idx = np.argmax(spectrum[1:50]) + 1
            neo_period_before = 1/freqs[peak_idx] if freqs[peak_idx] > 0 else 0

        if len(neo_identity_after) > 100:
            signal = np.array(neo_identity_after) - np.mean(neo_identity_after)
            spectrum = np.abs(np.fft.rfft(signal))
            spectrum[0] = 0
            freqs = np.fft.rfftfreq(len(signal))
            peak_idx = np.argmax(spectrum[1:50]) + 1
            neo_period_after = 1/freqs[peak_idx] if freqs[peak_idx] > 0 else 0

        # Detectar "duelo" (caída sostenida de identidad)
        if neo_identity_after:
            first_100 = neo_identity_after[:min(100, len(neo_identity_after))]
            grief_detected = np.mean(first_100) < np.mean(neo_identity_before) - 0.1
        else:
            grief_detected = False

        print(f"\n  RESULTADOS:")
        print(f"    Período antes: {neo_period_before:.1f}")
        print(f"    Período después: {neo_period_after:.1f}")
        print(f"    Crisis antes: {neo_crises_before}")
        print(f"    Crisis después: {neo_crises_after}")
        print(f"    Attachment inicial: 1.0 → final: {attachment_decay[-1] if attachment_decay else 'N/A':.3f}")
        print(f"    Duelo detectado: {'SÍ' if grief_detected else 'NO'}")

        results.append({
            'seed': seed,
            'period_before': neo_period_before,
            'period_after': neo_period_after,
            'crises_before': neo_crises_before,
            'crises_after': neo_crises_after,
            'final_attachment': attachment_decay[-1] if attachment_decay else 0,
            'grief_detected': grief_detected,
            'identity_drop': np.mean(neo_identity_before) - np.mean(neo_identity_after[:100]) if len(neo_identity_after) >= 100 else 0
        })

    # Análisis
    print("\n" + "=" * 50)
    print("ANÁLISIS")
    print("=" * 50)

    avg_period_before = np.mean([r['period_before'] for r in results if r['period_before']])
    avg_period_after = np.mean([r['period_after'] for r in results if r['period_after']])
    avg_crises_rate_before = np.mean([r['crises_before']/1000 for r in results])
    avg_crises_rate_after = np.mean([r['crises_after']/1000 for r in results])
    grief_count = sum(1 for r in results if r['grief_detected'])

    print(f"\nPeríodo:")
    print(f"  Antes de muerte: {avg_period_before:.1f}")
    print(f"  Después de muerte: {avg_period_after:.1f}")
    print(f"  Cambio: {avg_period_after - avg_period_before:+.1f}")

    print(f"\nTasa de crisis (por 1000 pasos):")
    print(f"  Antes: {avg_crises_rate_before*1000:.1f}")
    print(f"  Después: {avg_crises_rate_after*1000:.1f}")

    print(f"\nDuelo detectado: {grief_count}/{len(results)} seeds")

    if grief_count > len(results) / 2:
        print(f"\n→ HAY período de DUELO estructural")

    if avg_crises_rate_after > avg_crises_rate_before:
        print(f"→ NEO tiene MÁS CRISIS solo")

    return results


def run_all_trauma_experiments():
    """Corre todos los experimentos."""
    print("=" * 70)
    print("EXPERIMENTOS: TRAUMA, RESONANCIA Y DUELO")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")

    # 1. Trauma
    trauma_results = run_trauma_experiment(T=1500, trauma_t=500, seeds=[42, 123])

    # 2. Resonancia
    resonance_results = run_resonance_experiment(T=1500, external_period=30, seeds=[42, 123])

    # 3. Duelo
    grief_results = run_grief_experiment(T=1500, death_t=750, seeds=[42, 123])

    # Guardar
    os.makedirs('/root/NEO_EVA/results/trauma_grief', exist_ok=True)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'trauma': trauma_results,
        'resonance': resonance_results,
        'grief': grief_results
    }

    with open('/root/NEO_EVA/results/trauma_grief/results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResultados guardados en /root/NEO_EVA/results/trauma_grief/")

    return all_results


if __name__ == "__main__":
    run_all_trauma_experiments()
