#!/usr/bin/env python3
"""
Experimentos de Lóbulos Frontales: R11-R15 + Experimentos B y C
================================================================

Integra todas las fases frontales en un sistema coherente
y corre los experimentos de:
- Inversión de drives (B)
- Interrupción a mitad de crisis (C)

100% ENDÓGENO
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import os
import sys
import copy

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/frontal')

from phaseR11_self_drive_redefinition import SelfDriveRedefinition
from phaseR12_drive_coherence import DriveCoherenceCompetition
from phaseR13_meta_preferences import MetaPreferences
from phaseR14_structural_intentionality import StructuralIntentionality
from phaseR15_autopoiesis_light import AutopoiesisLight


@dataclass
class FrontalAgentState:
    """Estado completo de un agente con lóbulos frontales."""
    z: np.ndarray  # Estado interno
    w: np.ndarray  # Pesos de drive
    phi: np.ndarray  # Features fenomenológicos
    D: float  # Drive actual
    V: float  # Valor interno
    identity: float
    in_crisis: bool
    crisis_start: Optional[int]


class FrontalDualSystem:
    """
    Sistema dual con lóbulos frontales completos.

    Integra:
    - R11: Self-Drive Redefinition
    - R12: Drive-Coherence Competition
    - R13: Meta-Preferences
    - R14: Structural Intentionality
    - R15: Autopoiesis Light
    """

    FEATURE_NAMES = [
        'integration', 'neg_surprise', 'entropy',
        'stability', 'novelty', 'otherness', 'identity'
    ]

    def __init__(self, dim: int = 6, swap_drives: bool = False):
        self.dim = dim
        self.d_features = len(self.FEATURE_NAMES)
        self.swap_drives = swap_drives

        # Inicializar estados
        self.neo = FrontalAgentState(
            z=np.ones(dim) / dim,
            w=np.array([0.25, 0.20, 0.10, 0.10, 0.15, 0.10, 0.10]),  # NEO: exploration
            phi=np.zeros(self.d_features),
            D=0.5,
            V=0.5,
            identity=0.5,
            in_crisis=False,
            crisis_start=None
        )

        self.eva = FrontalAgentState(
            z=np.ones(dim) / dim,
            w=np.array([0.10, 0.10, 0.20, 0.25, 0.10, 0.15, 0.10]),  # EVA: stability
            phi=np.zeros(self.d_features),
            D=0.5,
            V=0.5,
            identity=0.5,
            in_crisis=False,
            crisis_start=None
        )

        # Si swap_drives, intercambiar pesos iniciales
        if swap_drives:
            self.neo.w, self.eva.w = self.eva.w.copy(), self.neo.w.copy()

        # Módulos R11-R15
        self.sdr = SelfDriveRedefinition(self.FEATURE_NAMES)
        self.sdr.register_agent("NEO")
        self.sdr.register_agent("EVA")

        self.dcc = DriveCoherenceCompetition()

        self.mpf = MetaPreferences(self.d_features, self.FEATURE_NAMES)
        self.mpf.register_agent("NEO")
        self.mpf.register_agent("EVA")

        self.si = StructuralIntentionality(dim)
        self.si.register_agent("NEO")
        self.si.register_agent("EVA")

        self.apl = AutopoiesisLight()

        # Historias
        self.z_history = {'NEO': [], 'EVA': []}
        self.w_history = {'NEO': [], 'EVA': []}
        self.crisis_history = {'NEO': [], 'EVA': []}
        self.identity_history = {'NEO': [], 'EVA': []}
        self.sagi_history = []
        self.divergence_history = []

        self.t = 0

    def _compute_features(self, agent: FrontalAgentState, other: FrontalAgentState) -> np.ndarray:
        """Calcula features fenomenológicos."""
        # Integration: correlación interna
        integration = 0.5  # Placeholder, usar historia real si disponible

        # Neg_surprise: opuesto de cambio
        neg_surprise = 0.5

        # Entropy
        p = np.clip(agent.z, 1e-10, 1)
        entropy = -np.sum(p * np.log(p)) / np.log(self.dim)

        # Stability: opuesto de varianza
        stability = 1 - np.var(agent.z) * 10

        # Novelty: distancia al estado típico
        novelty = np.linalg.norm(agent.z - np.ones(self.dim)/self.dim)

        # Otherness: distancia al otro
        otherness = np.linalg.norm(agent.z - other.z)

        # Identity
        identity = agent.identity

        return np.array([integration, neg_surprise, entropy, stability, novelty, otherness, identity])

    def _compute_identity(self, agent: FrontalAgentState, z_history: List[np.ndarray]) -> float:
        """Calcula fuerza de identidad."""
        if len(z_history) < 10:
            return 0.5

        # Centroide reciente
        recent = np.array(z_history[-20:])
        centroid = recent.mean(axis=0)

        # Distancia al centroide
        dist = np.linalg.norm(agent.z - centroid)
        typical_dist = np.mean([np.linalg.norm(z - centroid) for z in recent])

        return float(1.0 / (1.0 + dist / (typical_dist + 1e-10)))

    def _detect_crisis(self, agent: FrontalAgentState, name: str) -> bool:
        """Detecta si el agente entra en crisis."""
        if len(self.identity_history[name]) < 20:
            return False

        recent = np.mean(self.identity_history[name][-5:])
        baseline = np.mean(self.identity_history[name][-20:-5])

        # Crisis si caída > percentil 90 de caídas históricas
        drop = baseline - recent

        if len(self.identity_history[name]) > 50:
            drops = [self.identity_history[name][i] - self.identity_history[name][i+5]
                    for i in range(len(self.identity_history[name])-5)]
            threshold = np.percentile(drops, 90)
        else:
            threshold = 0.15

        return drop > threshold

    def _compute_sagi(self) -> float:
        """Calcula SAGI del sistema."""
        # Simplificado: promedio de identidades + correlación
        id_neo = self.neo.identity
        id_eva = self.eva.identity

        if len(self.z_history['NEO']) > 10:
            recent_neo = np.array(self.z_history['NEO'][-10:])
            recent_eva = np.array(self.z_history['EVA'][-10:])
            corr = np.mean([np.corrcoef(recent_neo[:, d], recent_eva[:, d])[0, 1]
                           for d in range(self.dim)
                           if not np.isnan(np.corrcoef(recent_neo[:, d], recent_eva[:, d])[0, 1])])
            if np.isnan(corr):
                corr = 0
        else:
            corr = 0

        return (id_neo + id_eva) / 2 + 0.2 * corr

    def step(self, stimulus: np.ndarray) -> Dict:
        """Un paso del sistema frontal completo."""
        self.t += 1

        # === Dinámica básica ===

        # Respuesta al estímulo
        response_neo = self.neo.D * stimulus[:self.dim]
        response_eva = self.eva.D * stimulus[:self.dim]

        # Interacción
        cross_neo = 0.1 * (self.eva.z - self.neo.z)
        cross_eva = 0.1 * (self.neo.z - self.eva.z)

        # Ruido
        noise_scale = 0.05 if not self.neo.in_crisis else 0.1
        noise_neo = np.random.randn(self.dim) * noise_scale
        noise_eva = np.random.randn(self.dim) * noise_scale

        # Update estados
        self.neo.z = self.neo.z + 0.1 * response_neo + cross_neo + noise_neo
        self.eva.z = self.eva.z + 0.1 * response_eva + cross_eva + noise_eva

        self.neo.z = np.clip(self.neo.z, 0.01, 0.99)
        self.eva.z = np.clip(self.eva.z, 0.01, 0.99)
        self.neo.z = self.neo.z / self.neo.z.sum()
        self.eva.z = self.eva.z / self.eva.z.sum()

        # Guardar
        self.z_history['NEO'].append(self.neo.z.copy())
        self.z_history['EVA'].append(self.eva.z.copy())

        # === Features ===
        self.neo.phi = self._compute_features(self.neo, self.eva)
        self.eva.phi = self._compute_features(self.eva, self.neo)

        # === Identidad ===
        self.neo.identity = self._compute_identity(self.neo, self.z_history['NEO'])
        self.eva.identity = self._compute_identity(self.eva, self.z_history['EVA'])
        self.identity_history['NEO'].append(self.neo.identity)
        self.identity_history['EVA'].append(self.eva.identity)

        # === Crisis ===
        was_in_crisis_neo = self.neo.in_crisis
        was_in_crisis_eva = self.eva.in_crisis

        if self._detect_crisis(self.neo, 'NEO') and not self.neo.in_crisis:
            self.neo.in_crisis = True
            self.neo.crisis_start = self.t
            self.crisis_history['NEO'].append(self.t)

        if self._detect_crisis(self.eva, 'EVA') and not self.eva.in_crisis:
            self.eva.in_crisis = True
            self.eva.crisis_start = self.t
            self.crisis_history['EVA'].append(self.t)

        # Salir de crisis
        if self.neo.in_crisis and len(self.identity_history['NEO']) > 10:
            if np.mean(self.identity_history['NEO'][-5:]) > np.mean(self.identity_history['NEO'][-10:-5]):
                self.neo.in_crisis = False

        if self.eva.in_crisis and len(self.identity_history['EVA']) > 10:
            if np.mean(self.identity_history['EVA'][-5:]) > np.mean(self.identity_history['EVA'][-10:-5]):
                self.eva.in_crisis = False

        # === R11: Self-Drive Redefinition ===
        metrics_neo = {'score': self.neo.identity, 'igi': 0.5, 'gi': 0.5}
        metrics_eva = {'score': self.eva.identity, 'igi': 0.5, 'gi': 0.5}

        sdr_neo = self.sdr.step('NEO', self.neo.phi, metrics_neo)
        sdr_eva = self.sdr.step('EVA', self.eva.phi, metrics_eva)

        # Propuesta de nuevos pesos
        delta_w_neo = self.sdr.agents['NEO'].weights - self.neo.w
        delta_w_eva = self.sdr.agents['EVA'].weights - self.eva.w

        # === R13: Meta-Preferences (filtrar cambios) ===
        self.mpf.update_meta_preferences('NEO', self.neo.w, sdr_neo['value'])
        self.mpf.update_meta_preferences('EVA', self.eva.w, sdr_eva['value'])

        delta_w_neo_filtered = self.mpf.filter_weight_change('NEO', delta_w_neo)
        delta_w_eva_filtered = self.mpf.filter_weight_change('EVA', delta_w_eva)

        # Aplicar cambios filtrados
        self.neo.w = self.neo.w + delta_w_neo_filtered
        self.eva.w = self.eva.w + delta_w_eva_filtered

        self.neo.w = np.clip(self.neo.w, 0.01, None)
        self.eva.w = np.clip(self.eva.w, 0.01, None)
        self.neo.w = self.neo.w / self.neo.w.sum()
        self.eva.w = self.eva.w / self.eva.w.sum()

        # === Drives ===
        self.neo.D = float(np.dot(self.neo.w, self.neo.phi))
        self.eva.D = float(np.dot(self.eva.w, self.eva.phi))

        # === R12: Drive-Coherence Competition ===
        self.neo.V = sdr_neo['value']
        self.eva.V = sdr_eva['value']

        w_neo_coh, w_eva_coh, dcc_info = self.dcc.step(
            self.neo.D, self.eva.D,
            self.neo.V, self.eva.V,
            self.neo.w, self.eva.w
        )

        # Blend con coherencia
        blend = 0.3  # Cuánto influye DCC
        self.neo.w = (1 - blend) * self.neo.w + blend * w_neo_coh
        self.eva.w = (1 - blend) * self.eva.w + blend * w_eva_coh

        self.neo.w = self.neo.w / self.neo.w.sum()
        self.eva.w = self.eva.w / self.eva.w.sum()

        self.w_history['NEO'].append(self.neo.w.copy())
        self.w_history['EVA'].append(self.eva.w.copy())

        # === R14: Structural Intentionality ===
        # Usar solo primeras dim dimensiones de w para SI
        si_neo = self.si.step('NEO', self.neo.z, self.neo.w[:self.dim])
        si_eva = self.si.step('EVA', self.eva.z, self.eva.w[:self.dim])

        # === R15: Autopoiesis Light ===
        # Simular TE entre módulos (simplificado)
        TE = np.random.rand(self.apl.n, self.apl.n) * 0.3
        # Módulos que funcionan bien tienen más TE
        if si_neo['sii'] > 0.5:
            TE[3, :] += 0.2  # intentionality
        if dcc_info['coherence'] > 0.5:
            TE[1, :] += 0.2  # coherence

        self.apl.update_transfer_entropy(TE)

        # Actualizar salud de módulos
        sagi = self._compute_sagi()
        for i, m in enumerate(self.apl.modules):
            delta_sagi = sagi - (self.sagi_history[-1] if self.sagi_history else 0.5)
            stability = 0.5 + 0.1 * (1 - int(self.neo.in_crisis or self.eva.in_crisis))
            collapses = len(self.crisis_history['NEO']) + len(self.crisis_history['EVA'])
            self.apl.update_module_health(m, delta_sagi, stability, collapses // (self.t + 1))

        apl_info = self.apl.step()

        # === Métricas globales ===
        self.sagi_history.append(sagi)
        self.divergence_history.append(np.linalg.norm(self.neo.w - self.eva.w))

        return {
            't': self.t,
            'neo': {
                'z': self.neo.z.copy(),
                'w': self.neo.w.copy(),
                'D': self.neo.D,
                'V': self.neo.V,
                'identity': self.neo.identity,
                'in_crisis': self.neo.in_crisis,
                'sii': si_neo['sii']
            },
            'eva': {
                'z': self.eva.z.copy(),
                'w': self.eva.w.copy(),
                'D': self.eva.D,
                'V': self.eva.V,
                'identity': self.eva.identity,
                'in_crisis': self.eva.in_crisis,
                'sii': si_eva['sii']
            },
            'dcc': dcc_info,
            'apl': apl_info,
            'sagi': sagi,
            'divergence': self.divergence_history[-1]
        }

    def get_state_snapshot(self) -> Dict:
        """Retorna snapshot del estado actual para restaurar."""
        return {
            'neo': copy.deepcopy(self.neo),
            'eva': copy.deepcopy(self.eva),
            'z_history': copy.deepcopy(self.z_history),
            'w_history': copy.deepcopy(self.w_history),
            'identity_history': copy.deepcopy(self.identity_history),
            'crisis_history': copy.deepcopy(self.crisis_history),
            't': self.t
        }

    def restore_from_snapshot(self, snapshot: Dict):
        """Restaura estado desde snapshot."""
        self.neo = copy.deepcopy(snapshot['neo'])
        self.eva = copy.deepcopy(snapshot['eva'])
        self.z_history = copy.deepcopy(snapshot['z_history'])
        self.w_history = copy.deepcopy(snapshot['w_history'])
        self.identity_history = copy.deepcopy(snapshot['identity_history'])
        self.crisis_history = copy.deepcopy(snapshot['crisis_history'])
        self.t = snapshot['t']


def run_experiment_A(T: int = 1000, seed: int = 42) -> Dict:
    """
    Experimento A: Sistema frontal normal.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENTO A: Sistema Frontal Normal")
    print("=" * 60)

    np.random.seed(seed)
    system = FrontalDualSystem(swap_drives=False)

    for t in range(T):
        stimulus = np.random.dirichlet(np.ones(6) * 2)
        result = system.step(stimulus)

        if t % (T // 5) == 0:
            print(f"t={t}: NEO identity={result['neo']['identity']:.3f}, "
                  f"EVA identity={result['eva']['identity']:.3f}, "
                  f"SAGI={result['sagi']:.3f}")

    # Análisis
    neo_dom, neo_w = system.sdr.get_dominant_feature('NEO')
    eva_dom, eva_w = system.sdr.get_dominant_feature('EVA')

    print(f"\nNEO dominante: {neo_dom} ({neo_w:.3f})")
    print(f"EVA dominante: {eva_dom} ({eva_w:.3f})")
    print(f"Crisis NEO: {len(system.crisis_history['NEO'])}")
    print(f"Crisis EVA: {len(system.crisis_history['EVA'])}")

    return {
        'condition': 'normal',
        'neo_dominant': neo_dom,
        'eva_dominant': eva_dom,
        'neo_crises': len(system.crisis_history['NEO']),
        'eva_crises': len(system.crisis_history['EVA']),
        'final_divergence': system.divergence_history[-1],
        'final_sagi': system.sagi_history[-1],
        'system': system
    }


def run_experiment_B(T: int = 1000, seed: int = 42) -> Dict:
    """
    Experimento B: Inversión de drives.

    NEO empieza con drives de EVA, EVA con drives de NEO.
    ¿Intercambian personalidades o las personalidades son emergentes?
    """
    print("\n" + "=" * 60)
    print("EXPERIMENTO B: Inversión de Drives")
    print("=" * 60)

    np.random.seed(seed)
    system = FrontalDualSystem(swap_drives=True)

    print("Drives iniciales INVERTIDOS:")
    print(f"  NEO: {system.neo.w.round(2)} (originalmente EVA)")
    print(f"  EVA: {system.eva.w.round(2)} (originalmente NEO)")

    for t in range(T):
        stimulus = np.random.dirichlet(np.ones(6) * 2)
        result = system.step(stimulus)

        if t % (T // 5) == 0:
            print(f"t={t}: NEO identity={result['neo']['identity']:.3f}, "
                  f"EVA identity={result['eva']['identity']:.3f}")

    # Análisis
    neo_dom, neo_w = system.sdr.get_dominant_feature('NEO')
    eva_dom, eva_w = system.sdr.get_dominant_feature('EVA')

    print(f"\nDrives finales:")
    print(f"  NEO: {system.neo.w.round(2)} → dominante: {neo_dom}")
    print(f"  EVA: {system.eva.w.round(2)} → dominante: {eva_dom}")

    return {
        'condition': 'swapped',
        'neo_dominant': neo_dom,
        'eva_dominant': eva_dom,
        'neo_crises': len(system.crisis_history['NEO']),
        'eva_crises': len(system.crisis_history['EVA']),
        'final_divergence': system.divergence_history[-1],
        'final_sagi': system.sagi_history[-1],
        'system': system
    }


def run_experiment_C(T: int = 1000, seed: int = 42) -> Dict:
    """
    Experimento C: Interrupción a mitad de crisis.

    Correr hasta que uno entre en crisis, guardar estado,
    luego continuar desde ahí múltiples veces.

    Si vuelven al ciclo → attractor en espacio de estados
    Si divergen → attractor depende de historia
    """
    print("\n" + "=" * 60)
    print("EXPERIMENTO C: Interrupción a Mitad de Crisis")
    print("=" * 60)

    np.random.seed(seed)
    system = FrontalDualSystem(swap_drives=False)

    # Correr hasta primera crisis
    crisis_snapshot = None
    crisis_t = None

    for t in range(T // 2):
        stimulus = np.random.dirichlet(np.ones(6) * 2)
        result = system.step(stimulus)

        if (result['neo']['in_crisis'] or result['eva']['in_crisis']) and crisis_snapshot is None:
            print(f"t={t}: CRISIS detectada - guardando snapshot")
            crisis_snapshot = system.get_state_snapshot()
            crisis_t = t
            break

    if crisis_snapshot is None:
        print("No se detectó crisis. Usando estado a T/4.")
        for t in range(T // 4):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            system.step(stimulus)
        crisis_snapshot = system.get_state_snapshot()
        crisis_t = T // 4

    print(f"Snapshot guardado en t={crisis_t}")

    # Correr múltiples continuaciones desde el snapshot
    n_continuations = 5
    continuation_results = []

    for i in range(n_continuations):
        print(f"\nContinuación {i+1}/{n_continuations}")

        # Restaurar
        system.restore_from_snapshot(crisis_snapshot)

        # Nueva semilla para ruido
        np.random.seed(seed * 100 + i)

        # Continuar
        for t in range(T // 2):
            stimulus = np.random.dirichlet(np.ones(6) * 2)
            result = system.step(stimulus)

        continuation_results.append({
            'final_neo_w': system.neo.w.copy(),
            'final_eva_w': system.eva.w.copy(),
            'final_sagi': system.sagi_history[-1] if system.sagi_history else 0,
            'neo_crises': len(system.crisis_history['NEO']),
            'eva_crises': len(system.crisis_history['EVA'])
        })

        print(f"  Final NEO w: {system.neo.w.round(2)}")
        print(f"  Final SAGI: {system.sagi_history[-1]:.3f}")

    # Análisis de divergencia entre continuaciones
    w_neos = np.array([r['final_neo_w'] for r in continuation_results])
    w_evas = np.array([r['final_eva_w'] for r in continuation_results])

    var_neo = np.mean(np.var(w_neos, axis=0))
    var_eva = np.mean(np.var(w_evas, axis=0))

    print(f"\nVarianza entre continuaciones:")
    print(f"  NEO: {var_neo:.4f}")
    print(f"  EVA: {var_eva:.4f}")

    if var_neo < 0.01 and var_eva < 0.01:
        print("→ Baja varianza: ATTRACTOR está en espacio de estados")
    else:
        print("→ Alta varianza: ATTRACTOR depende de historia")

    return {
        'condition': 'interrupted',
        'crisis_t': crisis_t,
        'n_continuations': n_continuations,
        'var_neo': var_neo,
        'var_eva': var_eva,
        'continuations': continuation_results
    }


def run_all_experiments(T: int = 1000, seeds: List[int] = [42, 123, 456]) -> Dict:
    """Corre todos los experimentos."""
    print("=" * 70)
    print("LÓBULOS FRONTALES: R11-R15 + EXPERIMENTOS")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")
    print(f"T = {T}, Seeds = {seeds}")

    all_results = {
        'A': [],  # Normal
        'B': [],  # Swapped
        'C': []   # Interrupted
    }

    for seed in seeds:
        print(f"\n{'#' * 60}")
        print(f"SEED = {seed}")
        print('#' * 60)

        result_A = run_experiment_A(T, seed)
        del result_A['system']  # No serializable
        all_results['A'].append(result_A)

        result_B = run_experiment_B(T, seed)
        del result_B['system']
        all_results['B'].append(result_B)

        result_C = run_experiment_C(T, seed)
        all_results['C'].append(result_C)

    # Análisis agregado
    print("\n" + "=" * 70)
    print("ANÁLISIS AGREGADO")
    print("=" * 70)

    # Experimento A vs B
    print("\n--- A vs B: ¿Las personalidades son emergentes? ---")

    A_neo_doms = [r['neo_dominant'] for r in all_results['A']]
    A_eva_doms = [r['eva_dominant'] for r in all_results['A']]
    B_neo_doms = [r['neo_dominant'] for r in all_results['B']]
    B_eva_doms = [r['eva_dominant'] for r in all_results['B']]

    print(f"Normal - NEO dominantes: {A_neo_doms}")
    print(f"Normal - EVA dominantes: {A_eva_doms}")
    print(f"Swapped - NEO dominantes: {B_neo_doms}")
    print(f"Swapped - EVA dominantes: {B_eva_doms}")

    # ¿Se intercambiaron?
    if A_neo_doms == B_eva_doms and A_eva_doms == B_neo_doms:
        print("→ Las personalidades SE INTERCAMBIARON → vienen de los drives")
    elif A_neo_doms == B_neo_doms and A_eva_doms == B_eva_doms:
        print("→ Las personalidades NO cambiaron → son EMERGENTES de la dinámica")
    else:
        print("→ Resultado mixto → ambos factores influyen")

    # Experimento C
    print("\n--- C: ¿Memoria estructural? ---")
    avg_var_neo = np.mean([r['var_neo'] for r in all_results['C']])
    avg_var_eva = np.mean([r['var_eva'] for r in all_results['C']])

    print(f"Varianza promedio NEO: {avg_var_neo:.4f}")
    print(f"Varianza promedio EVA: {avg_var_eva:.4f}")

    if avg_var_neo < 0.02 and avg_var_eva < 0.02:
        print("→ ATTRACTOR en espacio de estados (no depende de historia)")
    else:
        print("→ MEMORIA ESTRUCTURAL no trivial (depende de historia)")

    # Guardar
    os.makedirs('/root/NEO_EVA/results/frontal', exist_ok=True)

    final_results = {
        'timestamp': datetime.now().isoformat(),
        'T': T,
        'seeds': seeds,
        'experiments': all_results,
        'conclusions': {
            'personality_source': 'unknown',  # Se llenará abajo
            'memory_type': 'unknown'
        }
    }

    # Conclusiones
    if A_neo_doms == B_eva_doms and A_eva_doms == B_neo_doms:
        final_results['conclusions']['personality_source'] = 'drives'
    elif A_neo_doms == B_neo_doms and A_eva_doms == B_eva_doms:
        final_results['conclusions']['personality_source'] = 'emergent'
    else:
        final_results['conclusions']['personality_source'] = 'mixed'

    if avg_var_neo < 0.02 and avg_var_eva < 0.02:
        final_results['conclusions']['memory_type'] = 'state_space_attractor'
    else:
        final_results['conclusions']['memory_type'] = 'history_dependent'

    with open('/root/NEO_EVA/results/frontal/experiments.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\nResultados guardados en /root/NEO_EVA/results/frontal/experiments.json")

    return final_results


if __name__ == "__main__":
    run_all_experiments(T=1000, seeds=[42, 123, 456])
