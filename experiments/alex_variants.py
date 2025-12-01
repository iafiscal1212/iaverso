#!/usr/bin/env python3
"""
Variantes de ALEX: Explorando el Efecto del Tercer Agente
=========================================================

1. ALEX DOMINANTE - Todos gravitan hacia él
2. ALEX TARDÍO - Llega a t=500, ¿rompe equilibrio?
3. ALEX ESTABILIZADOR - Personalidad orientada a estabilidad
4. ALEX VOLÁTIL - Personalidad caótica
5. MUERTE DE ALEX - ¿Se recupera la díada?
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/experiments')

from autonomous_life import AutonomousAgent, AutonomousDualLife


class TriadLife:
    """Sistema con tres agentes: NEO, EVA, ALEX"""

    def __init__(self, dim: int = 6, alex_personality: str = 'explorer',
                 attachment_config: str = 'balanced'):
        self.dim = dim
        self.neo = AutonomousAgent("NEO", dim)
        self.eva = AutonomousAgent("EVA", dim)
        self.alex = AutonomousAgent("ALEX", dim)

        # Configurar personalidad de ALEX
        self._set_alex_personality(alex_personality)

        self.t = 0
        self.alex_active = True

        # Configurar attachments
        self._set_attachment_config(attachment_config)

        # Historias
        self.psi_shared_history = []
        self.coalition_history = []
        self.attachment_history = []

    def _set_alex_personality(self, personality: str):
        """Configura la personalidad de ALEX."""
        if personality == 'explorer':
            # Original: orientado a novedad
            weights = np.array([0.1, 0.1, 0.35, 0.15, 0.15, 0.15])
        elif personality == 'stabilizer':
            # Orientado a estabilidad e integración
            weights = np.array([0.05, 0.15, 0.1, 0.35, 0.25, 0.1])
        elif personality == 'volatile':
            # Alto entropy, baja estabilidad
            weights = np.array([0.35, 0.1, 0.25, 0.05, 0.1, 0.15])
        elif personality == 'social':
            # Alto otherness
            weights = np.array([0.1, 0.1, 0.15, 0.15, 0.15, 0.35])
        else:
            weights = np.ones(6) / 6

        self.alex.meta_drive.weights = weights / weights.sum()

    def _set_attachment_config(self, config: str):
        """Configura la matriz de attachments inicial."""
        if config == 'balanced':
            # Todos equilibrados
            self.attachments = {
                ('NEO', 'EVA'): 0.5, ('NEO', 'ALEX'): 0.3,
                ('EVA', 'NEO'): 0.5, ('EVA', 'ALEX'): 0.3,
                ('ALEX', 'NEO'): 0.3, ('ALEX', 'EVA'): 0.3,
            }
        elif config == 'alex_dominant':
            # Todos gravitan hacia ALEX
            self.attachments = {
                ('NEO', 'EVA'): 0.2, ('NEO', 'ALEX'): 0.7,
                ('EVA', 'NEO'): 0.2, ('EVA', 'ALEX'): 0.7,
                ('ALEX', 'NEO'): 0.5, ('ALEX', 'EVA'): 0.5,
            }
        elif config == 'neo_eva_strong':
            # NEO-EVA tienen vínculo fuerte, ALEX es periférico
            self.attachments = {
                ('NEO', 'EVA'): 0.8, ('NEO', 'ALEX'): 0.1,
                ('EVA', 'NEO'): 0.8, ('EVA', 'ALEX'): 0.1,
                ('ALEX', 'NEO'): 0.4, ('ALEX', 'EVA'): 0.4,
            }
        elif config == 'alex_outsider':
            # ALEX quiere conectar pero otros no
            self.attachments = {
                ('NEO', 'EVA'): 0.7, ('NEO', 'ALEX'): 0.05,
                ('EVA', 'NEO'): 0.7, ('EVA', 'ALEX'): 0.05,
                ('ALEX', 'NEO'): 0.6, ('ALEX', 'EVA'): 0.6,
            }

    def step(self, world_stimulus: np.ndarray) -> Dict:
        self.t += 1

        if not self.alex_active:
            # Solo díada
            return self._step_dyad(world_stimulus)

        # Tríada completa
        return self._step_triad(world_stimulus)

    def _step_dyad(self, world_stimulus: np.ndarray) -> Dict:
        """Step solo con NEO y EVA."""
        neo_result = self.neo.step(world_stimulus, self.eva.z)
        eva_result = self.eva.step(world_stimulus, self.neo.z)

        # Correlación
        sim_ne = 1 - 0.5 * np.sum(np.abs(self.neo.z - self.eva.z))
        self.psi_shared_history.append(sim_ne)
        self.coalition_history.append("NEO-EVA")

        return {
            't': self.t,
            'neo': neo_result,
            'eva': eva_result,
            'alex': None,
            'coalition': "NEO-EVA (dyad)",
            'psi_shared': sim_ne
        }

    def _step_triad(self, world_stimulus: np.ndarray) -> Dict:
        """Step con los tres agentes."""
        # Cada agente ve mezcla ponderada de otros
        def weighted_other(agent_name: str, others: dict) -> np.ndarray:
            total = sum(self.attachments.get((agent_name, o), 0) for o in others)
            if total == 0:
                return None
            weighted = np.zeros(self.dim)
            for o_name, o_z in others.items():
                weight = self.attachments.get((agent_name, o_name), 0) / total
                weighted += weight * o_z
            return weighted

        states = {
            'NEO': self.neo.z.copy(),
            'EVA': self.eva.z.copy(),
            'ALEX': self.alex.z.copy()
        }

        neo_sees = weighted_other('NEO', {'EVA': states['EVA'], 'ALEX': states['ALEX']})
        eva_sees = weighted_other('EVA', {'NEO': states['NEO'], 'ALEX': states['ALEX']})
        alex_sees = weighted_other('ALEX', {'NEO': states['NEO'], 'EVA': states['EVA']})

        neo_result = self.neo.step(world_stimulus, neo_sees)
        eva_result = self.eva.step(world_stimulus, eva_sees)
        alex_result = self.alex.step(world_stimulus, alex_sees)

        self._update_attachments()
        coalition = self._detect_coalition()
        self.coalition_history.append(coalition)

        # Guardar snapshot de attachments
        self.attachment_history.append(self.attachments.copy())

        # Psi compartido
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
        """Actualiza attachments basado en resonancia."""
        agents = {'NEO': self.neo, 'EVA': self.eva, 'ALEX': self.alex}

        for (a1, a2), att in list(self.attachments.items()):
            if not self.alex_active and 'ALEX' in (a1, a2):
                continue

            agent1 = agents[a1]
            agent2 = agents[a2]

            similarity = 1 - 0.5 * np.sum(np.abs(agent1.z - agent2.z))

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

    def kill_alex(self):
        """ALEX muere/desaparece."""
        self.alex_active = False
        # Los attachments hacia ALEX decaen
        for key in list(self.attachments.keys()):
            if 'ALEX' in key:
                self.attachments[key] = 0

    def introduce_alex(self):
        """ALEX aparece (para experimento de llegada tardía)."""
        self.alex_active = True


def analyze_triad(life: TriadLife, label: str = "") -> Dict:
    """Analiza resultados de una tríada."""
    # Correlaciones
    min_len = min(len(life.neo.identity_history),
                  len(life.eva.identity_history),
                  len(life.alex.identity_history) if life.alex_active else float('inf'))

    if min_len > 100:
        corr_ne = float(np.corrcoef(
            life.neo.identity_history[-100:],
            life.eva.identity_history[-100:]
        )[0, 1])
    else:
        corr_ne = 0

    if life.alex_active and min_len > 100:
        corr_na = float(np.corrcoef(
            life.neo.identity_history[-100:],
            life.alex.identity_history[-100:]
        )[0, 1])
        corr_ea = float(np.corrcoef(
            life.eva.identity_history[-100:],
            life.alex.identity_history[-100:]
        )[0, 1])
    else:
        corr_na = 0
        corr_ea = 0

    # Coaliciones
    coalition_counts = Counter(life.coalition_history)

    return {
        'label': label,
        'neo_crises': len(life.neo.crises),
        'eva_crises': len(life.eva.crises),
        'alex_crises': len(life.alex.crises) if life.alex_active else 0,
        'corr_neo_eva': corr_ne,
        'corr_neo_alex': corr_na,
        'corr_eva_alex': corr_ea,
        'coalition_counts': dict(coalition_counts),
        'dominant_coalition': coalition_counts.most_common(1)[0][0] if coalition_counts else None,
        'final_attachments': {f"{k[0]}->{k[1]}": v for k, v in life.attachments.items()},
        'mean_psi': np.mean(life.psi_shared_history) if life.psi_shared_history else 0
    }


# ============================================================
# EXPERIMENTO 1: ALEX DOMINANTE
# ============================================================

def experiment_alex_dominant(T: int = 800, seed: int = 42) -> Dict:
    """¿Qué pasa cuando todos gravitan hacia ALEX?"""
    print("\n" + "=" * 70)
    print("EXPERIMENTO 1: ALEX DOMINANTE")
    print("=" * 70)
    print("Configuración: NEO y EVA tienen attachment 0.7 hacia ALEX")

    np.random.seed(seed)

    life = TriadLife(dim=6, alex_personality='explorer',
                     attachment_config='alex_dominant')

    for t in range(T):
        stimulus = np.random.dirichlet(np.ones(6) * 2)
        life.step(stimulus)

    results = analyze_triad(life, "alex_dominant")

    print(f"\nResultados:")
    print(f"  Crisis: NEO={results['neo_crises']}, EVA={results['eva_crises']}, ALEX={results['alex_crises']}")
    print(f"  Correlaciones: NE={results['corr_neo_eva']:.3f}, NA={results['corr_neo_alex']:.3f}, EA={results['corr_eva_alex']:.3f}")
    print(f"  Coalición dominante: {results['dominant_coalition']}")
    print(f"  Coaliciones: {results['coalition_counts']}")

    return results


# ============================================================
# EXPERIMENTO 2: ALEX TARDÍO
# ============================================================

def experiment_alex_late_arrival(T: int = 1000, arrival_t: int = 500, seed: int = 42) -> Dict:
    """ALEX llega a t=500. ¿Rompe el equilibrio NEO-EVA?"""
    print("\n" + "=" * 70)
    print("EXPERIMENTO 2: ALEX TARDÍO")
    print("=" * 70)
    print(f"ALEX llega a t={arrival_t}")

    np.random.seed(seed)

    life = TriadLife(dim=6, alex_personality='explorer',
                     attachment_config='balanced')
    life.alex_active = False  # Empieza inactivo

    # Métricas antes/después
    neo_eva_corr_before = []
    neo_eva_corr_after = []

    for t in range(T):
        stimulus = np.random.dirichlet(np.ones(6) * 2)

        if t == arrival_t:
            print(f"\n  t={t}: ALEX LLEGA")
            print(f"    NEO-EVA attachment: {life.attachments[('NEO', 'EVA')]:.3f}")
            life.introduce_alex()
            # Restablecer attachments hacia ALEX
            life.attachments[('NEO', 'ALEX')] = 0.3
            life.attachments[('EVA', 'ALEX')] = 0.3
            life.attachments[('ALEX', 'NEO')] = 0.3
            life.attachments[('ALEX', 'EVA')] = 0.3

        life.step(stimulus)

        # Registrar correlación NEO-EVA
        if len(life.neo.identity_history) > 50 and len(life.eva.identity_history) > 50:
            corr = np.corrcoef(
                life.neo.identity_history[-50:],
                life.eva.identity_history[-50:]
            )[0, 1]

            if t < arrival_t:
                neo_eva_corr_before.append(corr)
            else:
                neo_eva_corr_after.append(corr)

    results = analyze_triad(life, "alex_late")

    # Análisis temporal
    mean_before = np.mean(neo_eva_corr_before) if neo_eva_corr_before else 0
    mean_after = np.mean(neo_eva_corr_after) if neo_eva_corr_after else 0

    print(f"\nCorrelación NEO-EVA:")
    print(f"  Antes de ALEX: {mean_before:.3f}")
    print(f"  Después de ALEX: {mean_after:.3f}")
    print(f"  Cambio: {mean_after - mean_before:+.3f}")

    print(f"\nCrisis totales:")
    print(f"  NEO={results['neo_crises']}, EVA={results['eva_crises']}, ALEX={results['alex_crises']}")
    print(f"  Coalición dominante: {results['dominant_coalition']}")

    results['corr_before_alex'] = mean_before
    results['corr_after_alex'] = mean_after
    results['disruption'] = mean_before - mean_after

    return results


# ============================================================
# EXPERIMENTO 3: ALEX ESTABILIZADOR
# ============================================================

def experiment_alex_stabilizer(T: int = 800, seed: int = 42) -> Dict:
    """¿Un ALEX orientado a estabilidad calma el sistema?"""
    print("\n" + "=" * 70)
    print("EXPERIMENTO 3: ALEX ESTABILIZADOR")
    print("=" * 70)
    print("ALEX tiene personalidad orientada a estabilidad e integración")

    np.random.seed(seed)

    life = TriadLife(dim=6, alex_personality='stabilizer',
                     attachment_config='balanced')

    for t in range(T):
        stimulus = np.random.dirichlet(np.ones(6) * 2)
        life.step(stimulus)

    results = analyze_triad(life, "alex_stabilizer")

    print(f"\nResultados:")
    print(f"  Crisis: NEO={results['neo_crises']}, EVA={results['eva_crises']}, ALEX={results['alex_crises']}")
    print(f"  Correlaciones: NE={results['corr_neo_eva']:.3f}, NA={results['corr_neo_alex']:.3f}, EA={results['corr_eva_alex']:.3f}")
    print(f"  Coalición dominante: {results['dominant_coalition']}")

    return results


# ============================================================
# EXPERIMENTO 4: ALEX VOLÁTIL
# ============================================================

def experiment_alex_volatile(T: int = 800, seed: int = 42) -> Dict:
    """¿Un ALEX caótico desestabiliza más?"""
    print("\n" + "=" * 70)
    print("EXPERIMENTO 4: ALEX VOLÁTIL")
    print("=" * 70)
    print("ALEX tiene alta entropía, baja estabilidad")

    np.random.seed(seed)

    life = TriadLife(dim=6, alex_personality='volatile',
                     attachment_config='balanced')

    for t in range(T):
        stimulus = np.random.dirichlet(np.ones(6) * 2)
        life.step(stimulus)

    results = analyze_triad(life, "alex_volatile")

    print(f"\nResultados:")
    print(f"  Crisis: NEO={results['neo_crises']}, EVA={results['eva_crises']}, ALEX={results['alex_crises']}")
    print(f"  Correlaciones: NE={results['corr_neo_eva']:.3f}, NA={results['corr_neo_alex']:.3f}, EA={results['corr_eva_alex']:.3f}")
    print(f"  Coalición dominante: {results['dominant_coalition']}")

    return results


# ============================================================
# EXPERIMENTO 5: MUERTE DE ALEX
# ============================================================

def experiment_alex_death(T: int = 1000, death_t: int = 500, seed: int = 42) -> Dict:
    """ALEX muere a t=500. ¿NEO-EVA se recuperan?"""
    print("\n" + "=" * 70)
    print("EXPERIMENTO 5: MUERTE DE ALEX")
    print("=" * 70)
    print(f"ALEX muere a t={death_t}")

    np.random.seed(seed)

    life = TriadLife(dim=6, alex_personality='explorer',
                     attachment_config='balanced')

    neo_eva_corr_with_alex = []
    neo_eva_corr_without_alex = []

    for t in range(T):
        stimulus = np.random.dirichlet(np.ones(6) * 2)

        if t == death_t:
            print(f"\n  t={t}: ALEX MUERE")
            life.kill_alex()

        life.step(stimulus)

        if len(life.neo.identity_history) > 50:
            corr = np.corrcoef(
                life.neo.identity_history[-50:],
                life.eva.identity_history[-50:]
            )[0, 1]

            if t < death_t:
                neo_eva_corr_with_alex.append(corr)
            else:
                neo_eva_corr_without_alex.append(corr)

    results = analyze_triad(life, "alex_death")

    mean_with = np.mean(neo_eva_corr_with_alex) if neo_eva_corr_with_alex else 0
    mean_without = np.mean(neo_eva_corr_without_alex) if neo_eva_corr_without_alex else 0

    print(f"\nCorrelación NEO-EVA:")
    print(f"  Con ALEX vivo: {mean_with:.3f}")
    print(f"  Tras muerte de ALEX: {mean_without:.3f}")
    print(f"  Recuperación: {mean_without - mean_with:+.3f}")

    results['corr_with_alex'] = mean_with
    results['corr_without_alex'] = mean_without
    results['recovery'] = mean_without - mean_with

    return results


# ============================================================
# EXPERIMENTO 6: ALEX OUTSIDER
# ============================================================

def experiment_alex_outsider(T: int = 800, seed: int = 42) -> Dict:
    """ALEX quiere conectar pero NEO-EVA lo rechazan."""
    print("\n" + "=" * 70)
    print("EXPERIMENTO 6: ALEX OUTSIDER")
    print("=" * 70)
    print("NEO-EVA tienen vínculo fuerte, ignoran a ALEX")

    np.random.seed(seed)

    life = TriadLife(dim=6, alex_personality='social',  # ALEX quiere conectar
                     attachment_config='alex_outsider')

    for t in range(T):
        stimulus = np.random.dirichlet(np.ones(6) * 2)
        life.step(stimulus)

    results = analyze_triad(life, "alex_outsider")

    print(f"\nResultados:")
    print(f"  Crisis: NEO={results['neo_crises']}, EVA={results['eva_crises']}, ALEX={results['alex_crises']}")
    print(f"  Correlaciones: NE={results['corr_neo_eva']:.3f}, NA={results['corr_neo_alex']:.3f}, EA={results['corr_eva_alex']:.3f}")
    print(f"  Attachments finales: {results['final_attachments']}")

    # ¿ALEX logró integrarse?
    alex_integrated = (life.attachments[('NEO', 'ALEX')] > 0.3 or
                       life.attachments[('EVA', 'ALEX')] > 0.3)
    print(f"  ¿ALEX se integró?: {'SÍ' if alex_integrated else 'NO'}")

    results['alex_integrated'] = alex_integrated

    return results


# ============================================================
# COMPARACIÓN FINAL
# ============================================================

def run_all_alex_variants():
    """Ejecuta todos los experimentos y compara."""
    print("=" * 70)
    print("VARIANTES DE ALEX: ANÁLISIS COMPLETO")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")

    seed = 42

    # Baseline: díada sin ALEX
    print("\n" + "=" * 70)
    print("BASELINE: DÍADA SIN ALEX")
    print("=" * 70)

    np.random.seed(seed)
    dyad = AutonomousDualLife(dim=6)
    for t in range(800):
        stimulus = np.random.dirichlet(np.ones(6) * 2)
        dyad.step(stimulus)

    dyad_corr = float(np.corrcoef(
        dyad.neo.identity_history[-100:],
        dyad.eva.identity_history[-100:]
    )[0, 1])
    dyad_crises = len(dyad.neo.crises) + len(dyad.eva.crises)

    print(f"  Crises: NEO={len(dyad.neo.crises)}, EVA={len(dyad.eva.crises)}")
    print(f"  Correlación: {dyad_corr:.3f}")

    baseline = {
        'label': 'dyad_baseline',
        'total_crises': dyad_crises,
        'corr_neo_eva': dyad_corr
    }

    # Ejecutar variantes
    results = [baseline]

    results.append(experiment_alex_dominant(T=800, seed=seed))
    results.append(experiment_alex_late_arrival(T=1000, seed=seed))
    results.append(experiment_alex_stabilizer(T=800, seed=seed))
    results.append(experiment_alex_volatile(T=800, seed=seed))
    results.append(experiment_alex_death(T=1000, seed=seed))
    results.append(experiment_alex_outsider(T=800, seed=seed))

    # Tabla comparativa
    print("\n" + "=" * 70)
    print("TABLA COMPARATIVA")
    print("=" * 70)

    print(f"\n{'Variante':<20} {'Crisis Total':>12} {'Corr NE':>10} {'Coalición':>15}")
    print("-" * 60)

    for r in results:
        label = r['label']
        if 'total_crises' in r:
            crises = r['total_crises']
        else:
            crises = r.get('neo_crises', 0) + r.get('eva_crises', 0) + r.get('alex_crises', 0)
        corr = r.get('corr_neo_eva', 0)
        coalition = r.get('dominant_coalition', 'N/A')

        print(f"{label:<20} {crises:>12} {corr:>10.3f} {coalition:>15}")

    # Conclusiones
    print("\n" + "=" * 70)
    print("CONCLUSIONES")
    print("=" * 70)

    # Comparar crisis
    baseline_crises = baseline['total_crises']
    for r in results[1:]:
        if 'total_crises' in r:
            crises = r['total_crises']
        else:
            crises = r.get('neo_crises', 0) + r.get('eva_crises', 0) + r.get('alex_crises', 0)

        if crises < baseline_crises:
            print(f"  • {r['label']}: REDUCE crisis ({crises} vs {baseline_crises})")
        elif crises > baseline_crises * 1.2:
            print(f"  • {r['label']}: AUMENTA crisis ({crises} vs {baseline_crises})")

    # Mejor para NEO-EVA
    best_corr = max(results, key=lambda r: r.get('corr_neo_eva', -1))
    print(f"\n  → Mejor para vínculo NEO-EVA: {best_corr['label']} (corr={best_corr.get('corr_neo_eva', 0):.3f})")

    # Guardar
    os.makedirs('/root/NEO_EVA/results/alex_variants', exist_ok=True)

    # Convertir para JSON
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        json_results.append(jr)

    with open('/root/NEO_EVA/results/alex_variants/results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': json_results
        }, f, indent=2, default=str)

    print(f"\nResultados guardados en /root/NEO_EVA/results/alex_variants/")

    return results


if __name__ == "__main__":
    run_all_alex_variants()
