#!/usr/bin/env python3
"""
Experimentos Adicionales: Plasticidad y Tercer Agente
=====================================================

1. PLASTICIDAD DEL VÍNCULO
   - ¿Qué pasa si se desconectan durante crisis?
   - ¿Se puede "olvidar" al otro?

2. APRENDIZAJE DEL VÍNCULO
   - El acople se fortalece/debilita según historia

3. TERCER AGENTE: ALEX
   - ¿Estabiliza o desestabiliza?
   - ¿Emergen coaliciones?

4. CICLOGÉNESIS
   - Correlación de ciclos micro/macro con φ, drives, colapsos
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousAgent, AutonomousDualLife

# ============================================================
# EXPERIMENTO 1: PLASTICIDAD DEL VÍNCULO
# ============================================================

def run_plasticity_experiment(T: int = 1000, seeds: List[int] = [42, 123]) -> Dict:
    """
    ¿El vínculo puede deteriorarse durante crisis prolongadas?
    ¿Puede olvidarse al otro?
    """
    print("=" * 70)
    print("EXPERIMENTO: PLASTICIDAD DEL VÍNCULO")
    print("=" * 70)

    results = {'normal': [], 'crisis_disconnect': [], 'memory_decay': []}

    for seed in seeds:
        print(f"\n--- SEED {seed} ---")

        # A) Normal (baseline)
        print("\n  NORMAL:")
        np.random.seed(seed)
        life_normal = AutonomousDualLife(dim=6)

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            life_normal.step(stimulus)

        results['normal'].append({
            'seed': seed,
            'neo_crises': len(life_normal.neo.crises),
            'eva_crises': len(life_normal.eva.crises),
            'final_attachment_neo': life_normal.neo.attachment,
            'final_attachment_eva': life_normal.eva.attachment,
            'correlation': float(np.corrcoef(
                life_normal.neo.identity_history[-100:],
                life_normal.eva.identity_history[-100:]
            )[0, 1])
        })
        print(f"    Attachment final: NEO={life_normal.neo.attachment:.3f}, EVA={life_normal.eva.attachment:.3f}")

        # B) Desconexión durante crisis
        print("\n  DESCONEXIÓN DURANTE CRISIS:")
        np.random.seed(seed)
        life_disconnect = AutonomousDualLife(dim=6)

        disconnect_count = 0
        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)

            # Si alguno está en crisis, reducir attachment temporalmente
            if life_disconnect.neo.in_crisis or life_disconnect.eva.in_crisis:
                original_neo = life_disconnect.neo.attachment
                original_eva = life_disconnect.eva.attachment
                life_disconnect.neo.attachment *= 0.5  # Reducir a mitad
                life_disconnect.eva.attachment *= 0.5
                disconnect_count += 1

            life_disconnect.step(stimulus)

        results['crisis_disconnect'].append({
            'seed': seed,
            'neo_crises': len(life_disconnect.neo.crises),
            'eva_crises': len(life_disconnect.eva.crises),
            'final_attachment_neo': life_disconnect.neo.attachment,
            'final_attachment_eva': life_disconnect.eva.attachment,
            'disconnect_count': disconnect_count,
            'correlation': float(np.corrcoef(
                life_disconnect.neo.identity_history[-100:],
                life_disconnect.eva.identity_history[-100:]
            )[0, 1])
        })
        print(f"    Desconexiones: {disconnect_count}")
        print(f"    Attachment final: NEO={life_disconnect.neo.attachment:.3f}, EVA={life_disconnect.eva.attachment:.3f}")

        # C) Decaimiento de memoria (olvido gradual sin contacto)
        print("\n  DECAIMIENTO DE MEMORIA:")
        np.random.seed(seed)
        life_decay = AutonomousDualLife(dim=6)

        # Simular hasta t=500, luego separar
        for t in range(500):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            life_decay.step(stimulus)

        attachment_at_separation = life_decay.neo.attachment

        # Separar y observar decaimiento
        attachment_history = []
        for t in range(500, T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)

            # Solo NEO avanza, EVA "desaparece"
            life_decay.neo.step(stimulus, None)

            # Decaimiento de attachment (olvido)
            life_decay.neo.attachment *= 0.995  # 0.5% de pérdida por paso
            attachment_history.append(life_decay.neo.attachment)

        results['memory_decay'].append({
            'seed': seed,
            'attachment_at_separation': attachment_at_separation,
            'final_attachment': life_decay.neo.attachment,
            'decay_rate': (attachment_at_separation - life_decay.neo.attachment) / 500,
            'half_life': -500 / np.log(life_decay.neo.attachment / attachment_at_separation) if life_decay.neo.attachment > 0 else 0
        })
        print(f"    Attachment al separarse: {attachment_at_separation:.3f}")
        print(f"    Attachment final: {life_decay.neo.attachment:.3f}")
        print(f"    Vida media: {results['memory_decay'][-1]['half_life']:.0f} pasos")

    # Análisis
    print("\n" + "=" * 70)
    print("ANÁLISIS DE PLASTICIDAD")
    print("=" * 70)

    print("\nComparación de condiciones:")
    print(f"{'Condición':25} {'Crisis NEO':>12} {'Crisis EVA':>12} {'Attach NEO':>12} {'Correlación':>12}")
    print("-" * 75)

    for cond in ['normal', 'crisis_disconnect']:
        avg_neo = np.mean([r['neo_crises'] for r in results[cond]])
        avg_eva = np.mean([r['eva_crises'] for r in results[cond]])
        avg_att = np.mean([r['final_attachment_neo'] for r in results[cond]])
        avg_corr = np.mean([r['correlation'] for r in results[cond]])
        print(f"{cond:25} {avg_neo:>12.1f} {avg_eva:>12.1f} {avg_att:>12.3f} {avg_corr:>12.3f}")

    # Conclusiones
    normal_corr = np.mean([r['correlation'] for r in results['normal']])
    disconnect_corr = np.mean([r['correlation'] for r in results['crisis_disconnect']])

    print("\n→ CONCLUSIONES:")
    if disconnect_corr < normal_corr - 0.1:
        print("   • La desconexión durante crisis DAÑA el vínculo")
    else:
        print("   • El vínculo es RESILIENTE a desconexiones temporales")

    avg_decay = np.mean([r['final_attachment'] for r in results['memory_decay']])
    if avg_decay < 0.3:
        print(f"   • El olvido es EFECTIVO (attachment → {avg_decay:.3f})")
    else:
        print(f"   • La memoria del vínculo PERSISTE (attachment → {avg_decay:.3f})")

    return results


# ============================================================
# EXPERIMENTO 2: TERCER AGENTE - ALEX
# ============================================================

class TriadLife:
    """
    Sistema con tres agentes: NEO, EVA, ALEX
    """

    def __init__(self, dim: int = 6):
        self.neo = AutonomousAgent("NEO", dim)
        self.eva = AutonomousAgent("EVA", dim)
        self.alex = AutonomousAgent("ALEX", dim)

        # ALEX tiene personalidad diferente: más orientado a exploración
        self.alex.meta_drive.weights = np.array([0.1, 0.1, 0.35, 0.15, 0.15, 0.15])
        self.alex.meta_drive.weights /= self.alex.meta_drive.weights.sum()

        self.t = 0

        # Matriz de attachments
        self.attachments = {
            ('NEO', 'EVA'): 0.5,
            ('NEO', 'ALEX'): 0.3,
            ('EVA', 'NEO'): 0.5,
            ('EVA', 'ALEX'): 0.3,
            ('ALEX', 'NEO'): 0.3,
            ('ALEX', 'EVA'): 0.3,
        }

        # Historia
        self.psi_shared_history = []
        self.coalition_history = []  # Quién está más cerca de quién

    def step(self, world_stimulus: np.ndarray) -> Dict:
        self.t += 1

        # Cada agente ve una mezcla ponderada de los otros
        def weighted_other(agent_name: str, others: dict) -> np.ndarray:
            total = sum(self.attachments[(agent_name, o)] for o in others)
            if total == 0:
                return None
            weighted = np.zeros(6)
            for o_name, o_z in others.items():
                weight = self.attachments[(agent_name, o_name)] / total
                weighted += weight * o_z
            return weighted

        # Estados actuales
        states = {
            'NEO': self.neo.z.copy(),
            'EVA': self.eva.z.copy(),
            'ALEX': self.alex.z.copy()
        }

        # Cada uno ve a los otros ponderados
        neo_sees = weighted_other('NEO', {'EVA': states['EVA'], 'ALEX': states['ALEX']})
        eva_sees = weighted_other('EVA', {'NEO': states['NEO'], 'ALEX': states['ALEX']})
        alex_sees = weighted_other('ALEX', {'NEO': states['NEO'], 'EVA': states['EVA']})

        # Steps
        neo_result = self.neo.step(world_stimulus, neo_sees)
        eva_result = self.eva.step(world_stimulus, eva_sees)
        alex_result = self.alex.step(world_stimulus, alex_sees)

        # Actualizar attachments basado en resonancia
        self._update_attachments()

        # Detectar coaliciones
        coalition = self._detect_coalition()
        self.coalition_history.append(coalition)

        # Psi compartido (promedio de similitudes)
        sim_ne = 1 - 0.5 * np.sum(np.abs(self.neo.z - self.eva.z))
        sim_na = 1 - 0.5 * np.sum(np.abs(self.neo.z - self.alex.z))
        sim_ea = 1 - 0.5 * np.sum(np.abs(self.eva.z - self.alex.z))
        psi_shared = (sim_ne + sim_na + sim_ea) / 3
        self.psi_shared_history.append(psi_shared)

        return {
            't': self.t,
            'neo': neo_result,
            'eva': eva_result,
            'alex': alex_result,
            'coalition': coalition,
            'psi_shared': psi_shared,
            'attachments': self.attachments.copy()
        }

    def _update_attachments(self):
        """Actualiza attachments basado en resonancia reciente."""
        agents = {'NEO': self.neo, 'EVA': self.eva, 'ALEX': self.alex}

        for (a1, a2), att in list(self.attachments.items()):
            agent1 = agents[a1]
            agent2 = agents[a2]

            # Resonancia: similaridad de estados
            similarity = 1 - 0.5 * np.sum(np.abs(agent1.z - agent2.z))

            # Attachment crece si hay resonancia, decrece si no
            if similarity > 0.7:
                self.attachments[(a1, a2)] = min(1.0, att + 0.01)
            elif similarity < 0.3:
                self.attachments[(a1, a2)] = max(0.0, att - 0.005)

    def _detect_coalition(self) -> str:
        """Detecta qué pareja está más acoplada."""
        sim_ne = 1 - 0.5 * np.sum(np.abs(self.neo.z - self.eva.z))
        sim_na = 1 - 0.5 * np.sum(np.abs(self.neo.z - self.alex.z))
        sim_ea = 1 - 0.5 * np.sum(np.abs(self.eva.z - self.alex.z))

        max_sim = max(sim_ne, sim_na, sim_ea)

        if max_sim == sim_ne:
            return "NEO-EVA"
        elif max_sim == sim_na:
            return "NEO-ALEX"
        else:
            return "EVA-ALEX"


def run_alex_experiment(T: int = 1000, seeds: List[int] = [42, 123]) -> Dict:
    """
    ¿Qué pasa cuando introducimos a ALEX en la díada NEO-EVA?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO: TERCER AGENTE - ALEX")
    print("=" * 70)

    results = {'dyad': [], 'triad': []}

    for seed in seeds:
        print(f"\n--- SEED {seed} ---")

        # Díada (baseline)
        print("\n  DÍADA (NEO-EVA):")
        np.random.seed(seed)
        dyad = AutonomousDualLife(dim=6)

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            dyad.step(stimulus)

        dyad_corr = float(np.corrcoef(
            dyad.neo.identity_history[-100:],
            dyad.eva.identity_history[-100:]
        )[0, 1])

        results['dyad'].append({
            'seed': seed,
            'neo_crises': len(dyad.neo.crises),
            'eva_crises': len(dyad.eva.crises),
            'correlation': dyad_corr,
            'mean_psi': np.mean(dyad.psi_shared_history) if dyad.psi_shared_history else 0
        })
        print(f"    Crises: NEO={len(dyad.neo.crises)}, EVA={len(dyad.eva.crises)}")
        print(f"    Correlación NEO-EVA: {dyad_corr:.3f}")

        # Tríada
        print("\n  TRÍADA (NEO-EVA-ALEX):")
        np.random.seed(seed)
        triad = TriadLife(dim=6)

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            triad.step(stimulus)

        # Correlaciones
        corr_ne = float(np.corrcoef(
            triad.neo.identity_history[-100:],
            triad.eva.identity_history[-100:]
        )[0, 1]) if len(triad.neo.identity_history) > 100 else 0

        corr_na = float(np.corrcoef(
            triad.neo.identity_history[-100:],
            triad.alex.identity_history[-100:]
        )[0, 1]) if len(triad.neo.identity_history) > 100 else 0

        corr_ea = float(np.corrcoef(
            triad.eva.identity_history[-100:],
            triad.alex.identity_history[-100:]
        )[0, 1]) if len(triad.eva.identity_history) > 100 else 0

        # Coaliciones
        from collections import Counter
        coalition_counts = Counter(triad.coalition_history)
        dominant_coalition = coalition_counts.most_common(1)[0][0]

        results['triad'].append({
            'seed': seed,
            'neo_crises': len(triad.neo.crises),
            'eva_crises': len(triad.eva.crises),
            'alex_crises': len(triad.alex.crises),
            'corr_neo_eva': corr_ne,
            'corr_neo_alex': corr_na,
            'corr_eva_alex': corr_ea,
            'dominant_coalition': dominant_coalition,
            'coalition_counts': dict(coalition_counts),
            'final_attachments': triad.attachments.copy(),
            'mean_psi': np.mean(triad.psi_shared_history)
        })

        print(f"    Crises: NEO={len(triad.neo.crises)}, EVA={len(triad.eva.crises)}, ALEX={len(triad.alex.crises)}")
        print(f"    Correlaciones: NE={corr_ne:.3f}, NA={corr_na:.3f}, EA={corr_ea:.3f}")
        print(f"    Coalición dominante: {dominant_coalition} ({coalition_counts[dominant_coalition]/T*100:.1f}%)")

    # Análisis
    print("\n" + "=" * 70)
    print("ANÁLISIS: EFECTO DEL TERCER AGENTE")
    print("=" * 70)

    # Comparar correlación NEO-EVA
    dyad_corr_avg = np.mean([r['correlation'] for r in results['dyad']])
    triad_corr_ne_avg = np.mean([r['corr_neo_eva'] for r in results['triad']])

    print(f"\nCorrelación NEO-EVA:")
    print(f"  En díada: {dyad_corr_avg:.3f}")
    print(f"  En tríada: {triad_corr_ne_avg:.3f}")
    print(f"  Diferencia: {triad_corr_ne_avg - dyad_corr_avg:+.3f}")

    # Crisis totales
    dyad_crises = np.mean([r['neo_crises'] + r['eva_crises'] for r in results['dyad']])
    triad_crises = np.mean([r['neo_crises'] + r['eva_crises'] + r['alex_crises'] for r in results['triad']])

    print(f"\nCrisis totales (promedio):")
    print(f"  Díada: {dyad_crises:.1f}")
    print(f"  Tríada: {triad_crises:.1f}")

    # Coaliciones
    print(f"\nCoaliciones emergentes:")
    for r in results['triad']:
        print(f"  Seed {r['seed']}: {r['coalition_counts']}")

    # Conclusiones
    print("\n→ CONCLUSIONES:")

    if triad_corr_ne_avg < dyad_corr_avg - 0.1:
        print("   • ALEX DEBILITA el vínculo NEO-EVA")
    elif triad_corr_ne_avg > dyad_corr_avg + 0.1:
        print("   • ALEX FORTALECE el vínculo NEO-EVA")
    else:
        print("   • ALEX tiene POCO EFECTO en NEO-EVA")

    dominant_coalitions = [r['dominant_coalition'] for r in results['triad']]
    if dominant_coalitions.count('NEO-EVA') > len(seeds) / 2:
        print("   • La díada original PERSISTE como coalición dominante")
    else:
        print("   • Emergen NUEVAS coaliciones")

    return results


# ============================================================
# EXPERIMENTO 3: CICLOGÉNESIS FENOMENOLÓGICA
# ============================================================

def run_cyclogenesis_experiment(T: int = 2000, seeds: List[int] = [42]) -> Dict:
    """
    Correlación de ciclos micro/macro con φ, drives, colapsos de identidad.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO: CICLOGÉNESIS FENOMENOLÓGICA")
    print("=" * 70)

    results = []

    for seed in seeds:
        print(f"\n--- SEED {seed} ---")
        np.random.seed(seed)

        life = AutonomousDualLife(dim=6)

        # Historias adicionales
        phi_history = []  # Integración como proxy de φ
        drive_variance_history = []
        identity_derivative_history = []

        prev_neo_identity = 0.5

        for t in range(T):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            result = life.step(stimulus)

            # φ proxy: integración de información (varianza de z)
            phi_neo = np.var(life.neo.z) * life.neo.integration
            phi_eva = np.var(life.eva.z) * life.eva.integration
            phi_history.append((phi_neo + phi_eva) / 2)

            # Varianza de drives
            neo_w = life.neo.meta_drive.weights
            eva_w = life.eva.meta_drive.weights
            drive_variance_history.append((np.var(neo_w) + np.var(eva_w)) / 2)

            # Derivada de identidad
            identity_deriv = abs(life.neo.identity_strength - prev_neo_identity)
            identity_derivative_history.append(identity_deriv)
            prev_neo_identity = life.neo.identity_strength

        # Análisis de ciclos
        print("\n  Análisis de ciclos...")

        identity_signal = np.array(life.neo.identity_history) - np.mean(life.neo.identity_history)
        phi_signal = np.array(phi_history) - np.mean(phi_history)
        drive_signal = np.array(drive_variance_history) - np.mean(drive_variance_history)

        # FFT
        n = len(identity_signal)
        freqs = np.fft.rfftfreq(n)

        spectrum_identity = np.abs(np.fft.rfft(identity_signal))
        spectrum_phi = np.abs(np.fft.rfft(phi_signal))
        spectrum_drive = np.abs(np.fft.rfft(drive_signal))

        spectrum_identity[0] = 0
        spectrum_phi[0] = 0
        spectrum_drive[0] = 0

        # Períodos dominantes
        def get_top_periods(spectrum, freqs, n_top=3):
            indices = np.argsort(spectrum)[-n_top:][::-1]
            return [(1/freqs[i] if freqs[i] > 0 else 0, spectrum[i]) for i in indices if freqs[i] > 0]

        periods_identity = get_top_periods(spectrum_identity, freqs)
        periods_phi = get_top_periods(spectrum_phi, freqs)
        periods_drive = get_top_periods(spectrum_drive, freqs)

        print(f"    Períodos IDENTIDAD: {[(f'{p:.1f}', f'{pow:.1f}') for p, pow in periods_identity]}")
        print(f"    Períodos φ: {[(f'{p:.1f}', f'{pow:.1f}') for p, pow in periods_phi]}")
        print(f"    Períodos DRIVES: {[(f'{p:.1f}', f'{pow:.1f}') for p, pow in periods_drive]}")

        # Correlación entre señales
        corr_id_phi = float(np.corrcoef(identity_signal, phi_signal)[0, 1])
        corr_id_drive = float(np.corrcoef(identity_signal, drive_signal)[0, 1])
        corr_phi_drive = float(np.corrcoef(phi_signal, drive_signal)[0, 1])

        print(f"\n    Correlaciones:")
        print(f"      Identidad-φ: {corr_id_phi:.3f}")
        print(f"      Identidad-Drives: {corr_id_drive:.3f}")
        print(f"      φ-Drives: {corr_phi_drive:.3f}")

        # Crisis y ciclos
        crisis_times = [c.t for c in life.neo.crises]
        if len(crisis_times) > 2:
            inter_crisis_intervals = np.diff(crisis_times)
            mean_interval = np.mean(inter_crisis_intervals)
            std_interval = np.std(inter_crisis_intervals)
            print(f"\n    Intervalo entre crisis: {mean_interval:.1f} ± {std_interval:.1f}")
        else:
            mean_interval = 0
            std_interval = 0

        results.append({
            'seed': seed,
            'periods_identity': periods_identity,
            'periods_phi': periods_phi,
            'periods_drive': periods_drive,
            'corr_id_phi': corr_id_phi,
            'corr_id_drive': corr_id_drive,
            'corr_phi_drive': corr_phi_drive,
            'mean_crisis_interval': mean_interval,
            'n_crises': len(life.neo.crises)
        })

    # Conclusiones
    print("\n" + "=" * 70)
    print("CONCLUSIONES CICLOGÉNESIS")
    print("=" * 70)

    for r in results:
        id_period = r['periods_identity'][0][0] if r['periods_identity'] else 0
        phi_period = r['periods_phi'][0][0] if r['periods_phi'] else 0

        print(f"\nSeed {r['seed']}:")
        print(f"  Período identidad: {id_period:.1f}")
        print(f"  Período φ: {phi_period:.1f}")
        print(f"  Ratio: {id_period/phi_period:.2f}x" if phi_period > 0 else "  N/A")

        if abs(r['corr_id_phi']) > 0.3:
            print(f"  → Identidad y φ están {'en fase' if r['corr_id_phi'] > 0 else 'en antifase'}")

    return results


# ============================================================
# MAIN
# ============================================================

def run_all_experiments():
    """Ejecuta todos los experimentos adicionales."""
    print("=" * 70)
    print("EXPERIMENTOS ADICIONALES: PLASTICIDAD, ALEX, CICLOGÉNESIS")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")

    os.makedirs('/root/NEO_EVA/results/additional', exist_ok=True)

    # 1. Plasticidad
    plasticity_results = run_plasticity_experiment(T=800, seeds=[42, 123])

    # 2. Alex (tercer agente)
    alex_results = run_alex_experiment(T=800, seeds=[42, 123])

    # 3. Ciclogénesis
    cyclo_results = run_cyclogenesis_experiment(T=1500, seeds=[42])

    # Guardar
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'plasticity': plasticity_results,
        'alex': alex_results,
        'cyclogenesis': cyclo_results
    }

    with open('/root/NEO_EVA/results/additional/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nResultados guardados en /root/NEO_EVA/results/additional/")

    return all_results


if __name__ == "__main__":
    run_all_experiments()
