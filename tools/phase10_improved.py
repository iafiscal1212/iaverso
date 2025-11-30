#!/usr/bin/env python3
"""
Phase 10: Mejoras Endógenas
===========================
8 mejoras manteniendo principio 100% endógeno:

1. Gate continuo: gate_strength = rank(ρ) * (1 - rank(var_I))
2. Alpha adaptativo para R_soc_ema: α = 1/sqrt(n_couplings + 1)
3. Inicialización sin sesgo: ambos [1/3, 1/3, 1/3]
4. PAD por PCA (no combinaciones arbitrarias)
5. Entropía de estados como métrica de salud
6. Coupling graduado: intensity = min(π_NEO, π_EVA) * corr(signals)
7. Train/test split para validación out-of-sample
8. Nulos agresivos en phase10_nulls.py
"""

import sys
import os
import json
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
from scipy import stats
from sklearn.decomposition import PCA

sys.path.insert(0, '/root/NEO_EVA/tools')

# =============================================================================
# ENUMS Y CONSTANTES
# =============================================================================

class LifeState(Enum):
    SLEEP = "SLEEP"
    WAKE = "WAKE"
    WORK = "WORK"
    LEARN = "LEARN"
    SOCIAL = "SOCIAL"


# =============================================================================
# UTILIDADES ENDÓGENAS
# =============================================================================

def get_epsilon(dtype=np.float64) -> float:
    """Epsilon mínimo positivo del dtype."""
    return np.finfo(dtype).eps


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """Normaliza por ranks a [0, 1]."""
    if len(x) < 2:
        return np.array([0.5] * len(x))
    ranks = stats.rankdata(x)
    return (ranks - 1) / (len(ranks) - 1)


def rolling_rank(value: float, history: deque) -> float:
    """Rank de value en history, normalizado a [0,1]."""
    if len(history) < 2:
        return 0.5
    arr = np.array(list(history) + [value])
    rank = stats.rankdata(arr)[-1]
    return (rank - 1) / (len(arr) - 1)


def compute_entropy(probs: Dict[str, float]) -> float:
    """Entropía de Shannon normalizada."""
    values = np.array(list(probs.values()))
    values = values[values > 0]
    if len(values) <= 1:
        return 0.0
    entropy = -np.sum(values * np.log(values))
    max_entropy = np.log(len(values))
    return entropy / max_entropy if max_entropy > 0 else 0.0


# =============================================================================
# SISTEMA AFECTIVO CON PCA ENDÓGENA
# =============================================================================

class EndogenousPAD:
    """
    PAD derivado por PCA de las 8 señales internas.
    Sin combinaciones arbitrarias.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.signal_history = deque(maxlen=window_size)
        self.pca = None
        self.pca_fitted = False

    def update(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Actualiza con nuevas señales y retorna PAD."""
        # Vector de 8 señales
        signal_vec = np.array([
            signals.get('r', 0.5),
            signals.get('s', 0.5),
            signals.get('m', 0.5),
            signals.get('c', 0.5),
            signals.get('R_soc', 0.5),
            signals.get('e', 0.5),
            signals.get('q', 0.5),
            signals.get('h', 0.5)
        ])

        self.signal_history.append(signal_vec)

        # Necesitamos suficientes datos para PCA
        if len(self.signal_history) < 10:
            return {'P': 0.5, 'A': 0.5, 'D': 0.5}

        # Fit PCA cada window_size/10 pasos o si no está fitted
        if not self.pca_fitted or len(self.signal_history) % (self.window_size // 10) == 0:
            self._fit_pca()

        # Transformar señal actual
        if self.pca is not None:
            # Centrar con mediana (robusto)
            data = np.array(self.signal_history)
            centered = signal_vec - np.median(data, axis=0)

            # Proyectar
            coords = self.pca.transform(centered.reshape(1, -1))[0]

            # Normalizar a [0,1] por ranks históricos
            P = self._rank_in_history(coords[0], 0) if len(coords) > 0 else 0.5
            A = self._rank_in_history(coords[1], 1) if len(coords) > 1 else 0.5
            D = self._rank_in_history(coords[2], 2) if len(coords) > 2 else 0.5

            return {'P': P, 'A': A, 'D': D}

        return {'P': 0.5, 'A': 0.5, 'D': 0.5}

    def _fit_pca(self):
        """Ajusta PCA a la historia de señales."""
        data = np.array(self.signal_history)

        # Centrar con mediana (robusto)
        centered = data - np.median(data, axis=0)

        # PCA con 3 componentes
        n_components = min(3, data.shape[1], data.shape[0])
        self.pca = PCA(n_components=n_components)
        self.pca.fit(centered)
        self.pca_fitted = True

        # Guardar coordenadas históricas para ranking
        self.coord_history = self.pca.transform(centered)

    def _rank_in_history(self, value: float, component: int) -> float:
        """Rank de value en historia del componente."""
        if not hasattr(self, 'coord_history') or self.coord_history.shape[1] <= component:
            return 0.5

        hist = self.coord_history[:, component]
        rank = np.sum(hist <= value) / len(hist)
        return rank


# =============================================================================
# GATE CONTINUO
# =============================================================================

class ContinuousGate:
    """
    Gate continuo en lugar de binario.
    gate_strength = rank(ρ) * (1 - rank(var_I))
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.rho_history = deque(maxlen=window_size)
        self.var_history = deque(maxlen=window_size)

    def update(self, rho: float, var_I: float) -> float:
        """Retorna gate_strength en [0, 1]."""
        self.rho_history.append(rho)
        self.var_history.append(var_I)

        if len(self.rho_history) < 5:
            return 0.5

        # Ranks endógenos
        rho_rank = rolling_rank(rho, self.rho_history)
        var_rank = rolling_rank(var_I, self.var_history)

        # Gate continuo: alto ρ Y baja varianza
        gate_strength = rho_rank * (1 - var_rank)

        return gate_strength


# =============================================================================
# R_SOC CON ALPHA ADAPTATIVO
# =============================================================================

class AdaptiveRSoc:
    """
    R_soc_ema con alpha adaptativo: α = 1/sqrt(n_couplings + 1)
    """

    def __init__(self):
        self.R_soc_ema = 0.5
        self.n_couplings = 0
        self.history = deque(maxlen=1000)

        # Para métricas durante coupling
        self.in_coupling = False
        self.current_metrics = []

    def get_alpha(self) -> float:
        """Alpha adaptativo."""
        return 1.0 / np.sqrt(self.n_couplings + 1)

    def start_coupling(self):
        """Inicia ventana de coupling."""
        self.in_coupling = True
        self.current_metrics = []

    def add_metric(self, rmse: float, mdl: float, mi: float):
        """Añade métricas durante coupling."""
        if self.in_coupling:
            self.current_metrics.append({'rmse': rmse, 'mdl': mdl, 'mi': mi})

    def end_coupling(self) -> float:
        """Cierra ventana y actualiza R_soc_ema."""
        self.in_coupling = False

        if not self.current_metrics:
            return self.R_soc_ema

        # Calcular reward por BordaRank
        n = len(self.current_metrics)
        if n < 1:
            return self.R_soc_ema

        rmses = [m['rmse'] for m in self.current_metrics]
        mdls = [m['mdl'] for m in self.current_metrics]
        mis = [m['mi'] for m in self.current_metrics]

        # Ranks (RMSE y MDL: menor es mejor, MI: mayor es mejor)
        rank_rmse = 1 - np.mean(rank_normalize(np.array(rmses))) if len(rmses) > 1 else 0.5
        rank_mdl = 1 - np.mean(rank_normalize(np.array(mdls))) if len(mdls) > 1 else 0.5
        rank_mi = np.mean(rank_normalize(np.array(mis))) if len(mis) > 1 else 0.5

        # BordaRank
        reward = (rank_rmse + rank_mdl + rank_mi) / 3

        # Actualizar con alpha adaptativo
        alpha = self.get_alpha()
        self.R_soc_ema = (1 - alpha) * self.R_soc_ema + alpha * reward

        self.n_couplings += 1
        self.history.append(self.R_soc_ema)

        return self.R_soc_ema

    def get_rank(self) -> float:
        """Rank de R_soc_ema actual en historia."""
        if len(self.history) < 2:
            return 0.5
        return rolling_rank(self.R_soc_ema, self.history)


# =============================================================================
# COUPLING GRADUADO
# =============================================================================

class GraduatedCoupling:
    """
    Coupling con intensidad graduada en lugar de binario.
    intensity = min(π_NEO, π_EVA) * correlation(signals)
    """

    def __init__(self):
        self.intensity_history = deque(maxlen=500)

    def compute_intensity(self, pi_neo: float, pi_eva: float,
                          signals_neo: Dict[str, float],
                          signals_eva: Dict[str, float]) -> float:
        """Calcula intensidad de coupling."""

        # Base: mínimo de voluntades
        base = min(pi_neo, pi_eva)

        # Correlación de señales
        keys = ['r', 's', 'm', 'c', 'R_soc', 'e', 'q', 'h']
        vec_neo = np.array([signals_neo.get(k, 0.5) for k in keys])
        vec_eva = np.array([signals_eva.get(k, 0.5) for k in keys])

        # Correlación (si hay varianza)
        if np.std(vec_neo) > 0 and np.std(vec_eva) > 0:
            corr = np.corrcoef(vec_neo, vec_eva)[0, 1]
            corr = (corr + 1) / 2  # Normalizar a [0, 1]
        else:
            corr = 0.5

        intensity = base * corr
        self.intensity_history.append(intensity)

        return intensity

    def get_threshold(self) -> float:
        """Threshold endógeno: mediana histórica."""
        if len(self.intensity_history) < 10:
            return 0.1
        return np.median(list(self.intensity_history))

    def should_couple(self, intensity: float) -> Tuple[bool, float]:
        """Decide si acoplar y con qué intensidad."""
        threshold = self.get_threshold()
        couples = intensity > threshold
        return couples, intensity


# =============================================================================
# MÉTRICA DE SALUD (ENTROPÍA)
# =============================================================================

class SystemHealth:
    """
    Métrica de salud basada en entropía de estados.
    Sistema sano: distribución uniforme (H alta).
    """

    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self.state_history = deque(maxlen=window_size)
        self.entropy_history = deque(maxlen=window_size)

    def update(self, state: LifeState) -> Dict[str, float]:
        """Actualiza con nuevo estado y retorna métricas de salud."""
        self.state_history.append(state)

        if len(self.state_history) < 10:
            return {'entropy': 1.0, 'health': 1.0, 'alarm': False}

        # Contar estados
        counts = {}
        for s in self.state_history:
            counts[s] = counts.get(s, 0) + 1

        # Probabilidades
        total = sum(counts.values())
        probs = {s: c/total for s, c in counts.items()}

        # Entropía normalizada
        entropy = compute_entropy(probs)
        self.entropy_history.append(entropy)

        # Health: rank de entropía actual (más alta = más sano)
        if len(self.entropy_history) < 5:
            health = entropy
        else:
            health = rolling_rank(entropy, self.entropy_history)

        # Alarma si entropía cae por debajo del q10 histórico
        alarm = False
        if len(self.entropy_history) >= 20:
            q10 = np.percentile(list(self.entropy_history), 10)
            alarm = entropy < q10

        return {
            'entropy': entropy,
            'health': health,
            'alarm': alarm,
            'state_probs': probs
        }


# =============================================================================
# MUNDO MEJORADO (PHASE 10)
# =============================================================================

class ImprovedWorld:
    """
    Mundo con todas las mejoras endógenas.
    """

    def __init__(self, name: str):
        self.name = name
        self.t = 0

        # Estado inicial SIN SESGO - ambos [1/3, 1/3, 1/3]
        self.I = np.array([1/3, 1/3, 1/3])
        self.current_state = LifeState.WAKE

        # Componentes mejorados
        self.pad = EndogenousPAD(window_size=100)
        self.gate = ContinuousGate(window_size=100)
        self.r_soc = AdaptiveRSoc()
        self.health = SystemHealth(window_size=500)

        # Historial para métricas
        self.I_history = deque(maxlen=1000)
        self.pi_history = deque(maxlen=1000)
        self.state_counts = {s: 0 for s in LifeState}

        # Métricas acumuladas
        self.coupling_count = 0
        self.total_intensity = 0.0

        # Logs
        self.log = []

    def compute_signals(self) -> Dict[str, float]:
        """Calcula las 8 señales internas."""
        # r: recurso (inversamente proporcional a |I|)
        r = 1 - np.linalg.norm(self.I)

        # s: estabilidad (inverso de varianza reciente de I)
        if len(self.I_history) >= 5:
            recent = np.array(list(self.I_history)[-10:])
            var = np.var(recent)
            s = 1 / (1 + var)
        else:
            s = 0.5

        # m: motivación (entropía de I)
        I_probs = np.clip(self.I, 0.01, 0.99)
        I_probs = I_probs / I_probs.sum()
        m = compute_entropy({f'c{i}': p for i, p in enumerate(I_probs)})

        # c: control (max componente de I)
        c = np.max(self.I)

        # R_soc: del componente AdaptiveRSoc
        R_soc = self.r_soc.R_soc_ema

        # e: energía (1 - fatiga, proxy por tiempo en WORK)
        work_ratio = self.state_counts[LifeState.WORK] / max(1, self.t)
        e = 1 - work_ratio

        # q: calidad (consistencia de transiciones)
        q = 0.5  # Se actualizará con métricas reales

        # h: armonía (proporción de SOCIAL)
        social_ratio = self.state_counts[LifeState.SOCIAL] / max(1, self.t)
        h = social_ratio

        return {'r': r, 's': s, 'm': m, 'c': c, 'R_soc': R_soc, 'e': e, 'q': q, 'h': h}

    def compute_pi(self, signals: Dict[str, float], other_signals: Dict[str, float]) -> Tuple[float, float, Dict]:
        """
        Calcula índice volitivo π con gate continuo.
        Retorna (π, gate_strength, debug_info)
        """
        # Benefit basado en señales
        benefit = (signals['R_soc'] + signals['h'] + signals['m']) / 3

        # Cost basado en recursos
        cost = signals['r']

        # π base
        pi_base = benefit / (cost + get_epsilon())
        pi_base = np.clip(pi_base, 0, 1)

        # Gate continuo
        # ρ: correlación con otro mundo
        keys = list(signals.keys())
        vec_self = np.array([signals[k] for k in keys])
        vec_other = np.array([other_signals.get(k, 0.5) for k in keys])

        if np.std(vec_self) > 0 and np.std(vec_other) > 0:
            rho = np.corrcoef(vec_self, vec_other)[0, 1]
        else:
            rho = 0

        # var_I: varianza de estado interno
        if len(self.I_history) >= 5:
            var_I = np.var(np.array(list(self.I_history)[-20:]))
        else:
            var_I = 0.1

        gate_strength = self.gate.update(rho, var_I)

        # π final modulado por gate
        pi = pi_base * gate_strength

        self.pi_history.append(pi)

        return pi, gate_strength, {'benefit': benefit, 'cost': cost, 'rho': rho, 'var_I': var_I}

    def select_state(self) -> LifeState:
        """Selecciona estado por softmax de utilidades."""
        # Drives
        D_rest = 1 - np.mean([self.I_history[-1][0] if self.I_history else 0.5])
        D_nov = np.std(list(self.I_history)[-10:]) if len(self.I_history) >= 10 else 0.5
        D_learn = self.I[1] if len(self.I) > 1 else 0.5
        D_soc = self.r_soc.R_soc_ema

        # Utilidades
        utilities = {
            LifeState.SLEEP: D_rest * 0.8,
            LifeState.WAKE: (1 - D_rest) * 0.5,
            LifeState.WORK: self.I[0] * 0.7,
            LifeState.LEARN: D_learn * 0.6 + D_nov * 0.3,
            LifeState.SOCIAL: D_soc * 0.9
        }

        # Softmax con temperatura endógena (IQR de utilidades)
        vals = np.array(list(utilities.values()))
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        gamma = 1 / (iqr + get_epsilon())
        gamma = np.clip(gamma, 0.5, 10)

        exp_u = np.exp(gamma * vals)
        probs = exp_u / exp_u.sum()

        states = list(utilities.keys())
        idx = np.random.choice(len(states), p=probs)

        return states[idx]

    def step(self, other_signals: Dict[str, float], coupling_intensity: float = 0.0):
        """Ejecuta un paso de simulación."""
        self.t += 1

        # Guardar I
        self.I_history.append(self.I.copy())

        # Calcular señales
        signals = self.compute_signals()

        # Actualizar PAD por PCA
        pad = self.pad.update(signals)

        # Seleccionar estado
        new_state = self.select_state()
        self.current_state = new_state
        self.state_counts[new_state] += 1

        # Actualizar salud
        health = self.health.update(new_state)

        # Dinámica de I (simplificada)
        # Hacia el atractor del estado actual
        attractors = {
            LifeState.SLEEP: np.array([0.2, 0.2, 0.6]),
            LifeState.WAKE: np.array([0.33, 0.33, 0.34]),
            LifeState.WORK: np.array([0.6, 0.2, 0.2]),
            LifeState.LEARN: np.array([0.2, 0.6, 0.2]),
            LifeState.SOCIAL: np.array([0.25, 0.25, 0.5])
        }

        target = attractors[new_state]

        # Tasa de cambio endógena
        rate = 0.1 / np.sqrt(self.t + 1)

        # Influencia del coupling
        if coupling_intensity > 0:
            # Moverse hacia señales del otro
            other_vec = np.array([other_signals.get('r', 0.5),
                                  other_signals.get('m', 0.5),
                                  other_signals.get('h', 0.5)])
            other_vec = other_vec / (other_vec.sum() + get_epsilon())
            target = (1 - coupling_intensity) * target + coupling_intensity * other_vec

        self.I = (1 - rate) * self.I + rate * target
        self.I = np.clip(self.I, 0.01, 0.99)
        self.I = self.I / self.I.sum()

        # Métricas para R_soc si en coupling
        if coupling_intensity > 0 and self.r_soc.in_coupling:
            rmse = np.sqrt(np.mean((self.I - target)**2))
            mdl = -np.sum(self.I * np.log(self.I + get_epsilon()))
            mi = coupling_intensity  # Proxy
            self.r_soc.add_metric(rmse, mdl, mi)

        # Log
        self.log.append({
            't': self.t,
            'state': new_state.value,
            'I': self.I.tolist(),
            'signals': signals,
            'PAD': pad,
            'health': health,
            'coupling_intensity': coupling_intensity
        })

        return signals, pad, health


# =============================================================================
# EXPERIMENTO PHASE 10
# =============================================================================

def run_phase10_experiment(n_cycles: int = 25000,
                           output_dir: str = '/root/NEO_EVA/results/phase10') -> Dict:
    """
    Ejecuta experimento Phase 10 con todas las mejoras.
    Incluye train/test split para validación out-of-sample.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    print("=" * 70)
    print("PHASE 10: MEJORAS ENDÓGENAS")
    print("=" * 70)

    # Crear mundos (ambos sin sesgo de inicialización)
    neo = ImprovedWorld("NEO")
    eva = ImprovedWorld("EVA")

    # Sistema de coupling graduado
    coupling_system = GraduatedCoupling()

    # Registros
    bilateral_events = []
    pi_neo_log = []
    pi_eva_log = []
    intensity_log = []
    health_log = []

    # Train/test split
    train_end = int(n_cycles * 0.6)  # 60% train

    print(f"\nCiclos totales: {n_cycles}")
    print(f"Train: 0-{train_end}, Test: {train_end}-{n_cycles}")
    print(f"Inicialización: NEO={neo.I}, EVA={eva.I}")

    print("\n[Simulando...]")

    for t in range(1, n_cycles + 1):
        # Señales
        neo_signals = neo.compute_signals()
        eva_signals = eva.compute_signals()

        # π con gate continuo
        pi_neo, gate_neo, debug_neo = neo.compute_pi(neo_signals, eva_signals)
        pi_eva, gate_eva, debug_eva = eva.compute_pi(eva_signals, neo_signals)

        # Coupling graduado
        intensity = coupling_system.compute_intensity(pi_neo, pi_eva, neo_signals, eva_signals)
        should_couple, actual_intensity = coupling_system.should_couple(intensity)

        # Si coupling, iniciar ventana de métricas
        if should_couple:
            neo.r_soc.start_coupling()
            eva.r_soc.start_coupling()

        # Step de cada mundo
        neo.step(eva_signals, actual_intensity if should_couple else 0)
        eva.step(neo_signals, actual_intensity if should_couple else 0)

        # Si hubo coupling, cerrar ventana
        if should_couple:
            neo.r_soc.end_coupling()
            eva.r_soc.end_coupling()
            neo.coupling_count += 1
            eva.coupling_count += 1
            neo.total_intensity += actual_intensity
            eva.total_intensity += actual_intensity

            bilateral_events.append({
                't': t,
                'intensity': actual_intensity,
                'pi_neo': pi_neo,
                'pi_eva': pi_eva,
                'phase': 'train' if t <= train_end else 'test'
            })

        # Logs
        pi_neo_log.append({'t': t, 'pi': pi_neo, 'gate': gate_neo})
        pi_eva_log.append({'t': t, 'pi': pi_eva, 'gate': gate_eva})
        intensity_log.append({'t': t, 'intensity': intensity, 'coupled': should_couple})

        # Health cada 100 ciclos
        if t % 100 == 0:
            health_log.append({
                't': t,
                'neo': neo.health.entropy_history[-1] if neo.health.entropy_history else 0,
                'eva': eva.health.entropy_history[-1] if eva.health.entropy_history else 0
            })

        if t % 5000 == 0:
            print(f"  t={t}: bilateral={len(bilateral_events)}, "
                  f"H_neo={neo.health.entropy_history[-1]:.3f}, "
                  f"H_eva={eva.health.entropy_history[-1]:.3f}")

    # Análisis de resultados
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    n_bilateral = len(bilateral_events)
    train_events = [e for e in bilateral_events if e['phase'] == 'train']
    test_events = [e for e in bilateral_events if e['phase'] == 'test']

    print(f"\nEventos bilaterales: {n_bilateral} ({n_bilateral/n_cycles*100:.2f}%)")
    print(f"  Train: {len(train_events)}")
    print(f"  Test: {len(test_events)}")

    # Calcular AUC train y test
    from sklearn.metrics import roc_auc_score

    bilateral_ts = set(e['t'] for e in bilateral_events)
    train_bilateral_ts = set(e['t'] for e in train_events)
    test_bilateral_ts = set(e['t'] for e in test_events)

    # AUC Train
    train_pis = [p['pi'] for p in pi_neo_log if p['t'] <= train_end]
    train_labels = [1 if t <= train_end and t in train_bilateral_ts else 0
                    for t in range(1, train_end + 1)]

    if sum(train_labels) >= 5:
        auc_train = roc_auc_score(train_labels, train_pis)
    else:
        auc_train = None

    # AUC Test
    test_pis = [p['pi'] for p in pi_neo_log if p['t'] > train_end]
    test_labels = [1 if t in test_bilateral_ts else 0
                   for t in range(train_end + 1, n_cycles + 1)]

    if sum(test_labels) >= 5:
        auc_test = roc_auc_score(test_labels, test_pis)
    else:
        auc_test = None

    print(f"\nAUC (π predice bilateral):")
    print(f"  Train: {auc_train:.4f}" if auc_train else "  Train: N/A")
    print(f"  Test:  {auc_test:.4f}" if auc_test else "  Test: N/A")

    if auc_train and auc_test:
        auc_drop = (auc_train - auc_test) / auc_train * 100
        print(f"  Drop:  {auc_drop:.1f}%")

    # Entropía final
    print(f"\nEntropía de estados (salud):")
    print(f"  NEO: {neo.health.entropy_history[-1]:.4f}")
    print(f"  EVA: {eva.health.entropy_history[-1]:.4f}")

    # Especialización (si emergió)
    print(f"\nEstado interno final:")
    print(f"  NEO I: {neo.I}")
    print(f"  EVA I: {eva.I}")

    # Guardar resultados
    results = {
        'n_cycles': n_cycles,
        'train_end': train_end,
        'n_bilateral': n_bilateral,
        'n_train_events': len(train_events),
        'n_test_events': len(test_events),
        'auc_train': auc_train,
        'auc_test': auc_test,
        'entropy_neo': float(neo.health.entropy_history[-1]) if neo.health.entropy_history else 0,
        'entropy_eva': float(eva.health.entropy_history[-1]) if eva.health.entropy_history else 0,
        'I_neo_final': neo.I.tolist(),
        'I_eva_final': eva.I.tolist(),
        'mean_intensity': np.mean([e['intensity'] for e in bilateral_events]) if bilateral_events else 0
    }

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    with open(f"{output_dir}/bilateral_events.json", 'w') as f:
        json.dump(bilateral_events, f)

    with open(f"{output_dir}/pi_log_neo.json", 'w') as f:
        json.dump(pi_neo_log, f)

    with open(f"{output_dir}/pi_log_eva.json", 'w') as f:
        json.dump(pi_eva_log, f)

    with open(f"{output_dir}/health_log.json", 'w') as f:
        json.dump(health_log, f)

    # Serializar logs con conversion de tipos numpy
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {(k.value if isinstance(k, LifeState) else k): convert_for_json(v)
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, LifeState):
            return obj.value
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    with open(f"{output_dir}/neo_log.json", 'w') as f:
        json.dump(convert_for_json(neo.log[-1000:]), f)

    with open(f"{output_dir}/eva_log.json", 'w') as f:
        json.dump(convert_for_json(eva.log[-1000:]), f)

    print(f"\n[OK] Resultados guardados en {output_dir}/")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Phase 10: Mejoras Endógenas')
    parser.add_argument('--cycles', type=int, default=25000)
    parser.add_argument('--output-dir', default='/root/NEO_EVA/results/phase10')
    args = parser.parse_args()

    run_phase10_experiment(args.cycles, args.output_dir)
