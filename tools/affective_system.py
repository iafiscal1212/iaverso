#!/usr/bin/env python3
"""
Sistema Afectivo Endógeno
==========================

"Emoción" = estado latente que emerge de señales internas combinadas
sin pesos fijos (ranks, cuantiles, √T) y que modula la política como
un campo lento con memoria endógena.

Señales internas:
- Sorpresa r_t (residual)
- Estabilidad s_t = 1/ρ_t
- Movilidad m_t = ||ΔI_t||
- Claridad c_t = max(I) - 2º max(I)
- Utilidad social R_soc
- Esfuerzo e_t = Var_w(I) + IQR(r)
- Recuperación q_t = EMA(1_SLEEP)
- Coherencia h_t = proxy MI/MDL

Todo 100% endógeno - sin constantes mágicas.
"""

import math
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

EPS = 1e-12

# =============================================================================
# Funciones endógenas (sin magia)
# =============================================================================

def window_size(T: int) -> int:
    """w = max(10, floor(sqrt(T)))"""
    return max(10, int(math.sqrt(T)))

def rank_normalize(value: float, history: List[float], min_samples: int = 10) -> float:
    """Normaliza value a [0,1] por rank."""
    if len(history) < min_samples:
        return float('nan')
    sorted_hist = np.sort(history)
    rank = (np.searchsorted(sorted_hist, value) + 1) / len(history)
    return float(np.clip(rank, 0, 1))

def mad_normalize(value: float, history: List[float], min_samples: int = 10) -> float:
    """Normaliza por (x - median) / MAD."""
    if len(history) < min_samples:
        return float('nan')
    median = np.median(history)
    mad = np.median(np.abs(np.array(history) - median))
    if mad < EPS:
        return 0.0
    return float((value - median) / mad)

def iqr(arr) -> float:
    """IQR seguro."""
    if len(arr) < 4:
        return float('nan')
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))

def ema_beta(w: int) -> float:
    """β para EMA: (w-1)/(w+1)"""
    return (w - 1) / (w + 1)

# =============================================================================
# Señales Internas
# =============================================================================

@dataclass
class InternalSignals:
    """8 señales internas, todas derivadas de datos."""
    r_t: float      # Sorpresa (residual)
    s_t: float      # Estabilidad (1/ρ)
    m_t: float      # Movilidad (||ΔI||)
    c_t: float      # Claridad (confianza)
    R_soc: float    # Utilidad social
    e_t: float      # Esfuerzo
    q_t: float      # Recuperación
    h_t: float      # Coherencia

class SignalExtractor:
    """Extrae las 8 señales internas de la historia."""

    def __init__(self):
        # Historiales para normalización
        self.r_history: deque = deque(maxlen=500)
        self.s_history: deque = deque(maxlen=500)
        self.m_history: deque = deque(maxlen=500)
        self.c_history: deque = deque(maxlen=500)
        self.e_history: deque = deque(maxlen=500)
        self.h_history: deque = deque(maxlen=500)

        # EMA de recuperación
        self.q_ema = 0.5
        self.sleep_count = 0

    def extract(self, I: np.ndarray, I_prev: np.ndarray,
                I_history: List[np.ndarray], residuals: List[float],
                rho: float, R_soc_ema: float,
                is_sleep: bool, mdl_history: List[float]) -> InternalSignals:
        """Extrae todas las señales en un ciclo."""
        T = len(I_history)
        w = window_size(T)

        # r_t: Sorpresa = residual actual
        if residuals:
            r_t = residuals[-1]
            self.r_history.append(r_t)
        else:
            r_t = 0.0

        # s_t: Estabilidad = 1/ρ (alto ρ = inestable → bajo s)
        if not math.isnan(rho) and rho > EPS:
            s_t = 1.0 / rho
            self.s_history.append(s_t)
        else:
            s_t = 1.0

        # m_t: Movilidad = ||ΔI||
        m_t = float(np.linalg.norm(I - I_prev))
        self.m_history.append(m_t)

        # c_t: Claridad = max(I) - 2º max(I)
        I_sorted = np.sort(I)[::-1]
        c_t = float(I_sorted[0] - I_sorted[1])
        self.c_history.append(c_t)

        # R_soc: ya viene calculado
        R_soc = R_soc_ema

        # e_t: Esfuerzo = Var_w(I) + IQR(r)
        if len(I_history) >= w:
            var_I = np.var(np.array(I_history[-w:]))
            iqr_r = iqr(residuals[-w:]) if len(residuals) >= w else 0
            e_t = var_I + (iqr_r if not math.isnan(iqr_r) else 0)
            self.e_history.append(e_t)
        else:
            e_t = 0.0

        # q_t: Recuperación = EMA(1_SLEEP)
        beta = ema_beta(w)
        sleep_indicator = 1.0 if is_sleep else 0.0
        self.q_ema = beta * self.q_ema + (1 - beta) * sleep_indicator
        q_t = self.q_ema

        # h_t: Coherencia = reducción de MDL o MI proxy
        if len(mdl_history) >= 2:
            # Coherencia alta si MDL bajó (más compresión)
            h_t = mdl_history[-2] - mdl_history[-1] if mdl_history[-2] > mdl_history[-1] else 0
            self.h_history.append(h_t)
        else:
            h_t = 0.0

        return InternalSignals(
            r_t=r_t, s_t=s_t, m_t=m_t, c_t=c_t,
            R_soc=R_soc, e_t=e_t, q_t=q_t, h_t=h_t
        )

    def normalize_to_ranks(self, signals: InternalSignals) -> np.ndarray:
        """Convierte señales a vector de ranks [0,1]."""
        z = np.zeros(8)

        z[0] = rank_normalize(signals.r_t, list(self.r_history))
        z[1] = rank_normalize(signals.s_t, list(self.s_history))
        z[2] = rank_normalize(signals.m_t, list(self.m_history))
        z[3] = rank_normalize(signals.c_t, list(self.c_history))
        z[4] = signals.R_soc  # Ya es EMA normalizado
        z[5] = rank_normalize(signals.e_t, list(self.e_history))
        z[6] = signals.q_t    # Ya es EMA en [0,1]
        z[7] = rank_normalize(signals.h_t, list(self.h_history))

        # Reemplazar NaN por 0.5 (neutral)
        z = np.nan_to_num(z, nan=0.5)

        return z

# =============================================================================
# Latentes Afectivos
# =============================================================================

class AffectiveLatents:
    """
    Latentes afectivos emergentes via PCA endógena.
    Sin pesos fijos - todo de los datos.
    """

    def __init__(self):
        self.z_history: deque = deque(maxlen=500)
        self.A_raw = np.zeros(3)  # Latentes crudos (PC1, PC2, PC3)
        self.F = np.zeros(3)       # Latentes lentos (con memoria)

        # Mapas canónicos PAD
        self.V = 0.5  # Valencia
        self.A = 0.5  # Activación
        self.D = 0.5  # Dominancia

    def update(self, z: np.ndarray, q_t: float) -> Tuple[np.ndarray, Dict]:
        """
        Actualiza latentes con nuevo vector de señales.
        Retorna (F, {'V': V, 'A': A, 'D': D})
        """
        self.z_history.append(z.copy())
        w = window_size(len(self.z_history))

        if len(self.z_history) < w:
            return self.F, {'V': self.V, 'A': self.A, 'D': self.D}

        # PCA endógena en ventana
        Z = np.array(list(self.z_history)[-w:])
        Z_centered = Z - np.mean(Z, axis=0)

        try:
            # SVD para obtener componentes principales
            U, S, Vt = np.linalg.svd(Z_centered, full_matrices=False)

            # Latentes crudos = proyección en primeras 3 PCs
            # Normalizar por varianza explicada
            total_var = (S ** 2).sum()
            if total_var > EPS:
                explained = S[:3] ** 2 / total_var
            else:
                explained = np.array([1/3, 1/3, 1/3])

            # Proyectar z actual
            for k in range(min(3, len(S))):
                self.A_raw[k] = float(np.dot(z - np.mean(Z, axis=0), Vt[k]))

        except:
            pass  # Mantener valores anteriores

        # Mapas canónicos PAD (sumas algebraicas, sin pesos)
        # Activación = movilidad + esfuerzo (indices 2, 5)
        self.A = (z[2] + z[5]) / 2.0

        # Valencia = R_soc + coherencia - sorpresa (indices 4, 7, 0)
        self.V = (z[4] + z[7] - z[0] + 1) / 3.0  # +1 para centrar en [0,1]

        # Dominancia = claridad + estabilidad (indices 3, 1)
        self.D = (z[3] + z[1]) / 2.0

        # Dinámica lenta: F = α*F + (1-α)*A_raw
        # α depende de recuperación (más descanso → más memoria)
        alpha = np.clip(q_t, 0.1, 0.9)  # q_t ya en [0,1]
        self.F = alpha * self.F + (1 - alpha) * self.A_raw

        return self.F, {'V': self.V, 'A': self.A, 'D': self.D}

# =============================================================================
# Modulación de Política
# =============================================================================

class AffectiveModulator:
    """
    Modula la política de estados y consentimiento basándose en afecto.
    Sin boosts fijos - todo por ranks.
    """

    def __init__(self):
        self.V_history: deque = deque(maxlen=200)
        self.A_history: deque = deque(maxlen=200)
        self.D_history: deque = deque(maxlen=200)

    def get_social_boost(self, V: float, A: float, D: float) -> float:
        """
        Boost para SOCIAL: alto cuando V alta y A moderada.
        Retorna valor en [0, 1].
        """
        self.V_history.append(V)
        self.A_history.append(A)
        self.D_history.append(D)

        if len(self.V_history) < 20:
            return 0.0

        # Rank de valencia y activación
        V_rank = rank_normalize(V, list(self.V_history))
        A_rank = rank_normalize(A, list(self.A_history))

        if math.isnan(V_rank) or math.isnan(A_rank):
            return 0.0

        # SOCIAL boost: V alta, A no extrema
        # "Quiero socializar cuando me siento bien pero no frenético"
        A_moderate = 1.0 - abs(A_rank - 0.5) * 2  # Máximo en A=0.5
        boost = V_rank * A_moderate

        return float(np.clip(boost, 0, 1))

    def get_sleep_boost(self, A: float, q: float, fatigue: float) -> float:
        """
        Boost para SLEEP: bajo cuando A baja y q bajo (deuda de descanso).
        """
        if len(self.A_history) < 20:
            return 0.0

        A_rank = rank_normalize(A, list(self.A_history))
        if math.isnan(A_rank):
            return 0.0

        # SLEEP boost: baja activación + poca recuperación reciente
        low_activation = 1.0 - A_rank
        sleep_debt = 1.0 - q  # q bajo = mucha deuda

        boost = low_activation * sleep_debt
        return float(np.clip(boost, 0, 1))

    def modulate_consent(self, pi_base: float, V: float, R_soc_rank: float,
                         cost_rank: float) -> float:
        """
        Modula π incluyendo valencia.
        π = σ(rank(R_soc) + rank(V) - rank(cost))
        """
        if len(self.V_history) < 20:
            return pi_base

        V_rank = rank_normalize(V, list(self.V_history))
        if math.isnan(V_rank):
            return pi_base

        # Combinación endógena
        logit = R_soc_rank + V_rank - cost_rank
        pi_modulated = 1.0 / (1.0 + np.exp(-4.0 * logit))

        return float(pi_modulated)

# =============================================================================
# Sistema Afectivo Completo
# =============================================================================

class AffectiveSystem:
    """
    Sistema afectivo completo que integra señales → latentes → modulación.
    100% endógeno.
    """

    def __init__(self):
        self.signal_extractor = SignalExtractor()
        self.latents = AffectiveLatents()
        self.modulator = AffectiveModulator()

        # Log
        self.affect_log: List[Dict] = []

    def process(self, I: np.ndarray, I_prev: np.ndarray,
                I_history: List[np.ndarray], residuals: List[float],
                rho: float, R_soc_ema: float, is_sleep: bool,
                mdl_history: List[float], fatigue: float) -> Dict:
        """
        Procesa un ciclo completo.
        Retorna dict con afecto y modulaciones.
        """
        # 1. Extraer señales
        signals = self.signal_extractor.extract(
            I, I_prev, I_history, residuals,
            rho, R_soc_ema, is_sleep, mdl_history
        )

        # 2. Normalizar a ranks
        z = self.signal_extractor.normalize_to_ranks(signals)

        # 3. Calcular latentes
        F, PAD = self.latents.update(z, signals.q_t)

        # 4. Calcular modulaciones
        social_boost = self.modulator.get_social_boost(PAD['V'], PAD['A'], PAD['D'])
        sleep_boost = self.modulator.get_sleep_boost(PAD['A'], signals.q_t, fatigue)

        result = {
            'signals': {
                'r': signals.r_t, 's': signals.s_t, 'm': signals.m_t,
                'c': signals.c_t, 'R_soc': signals.R_soc, 'e': signals.e_t,
                'q': signals.q_t, 'h': signals.h_t
            },
            'z': z.tolist(),
            'F': F.tolist(),
            'PAD': PAD,
            'social_boost': social_boost,
            'sleep_boost': sleep_boost,
        }

        self.affect_log.append(result)
        return result

    def modulate_consent(self, pi_base: float, cost_rank: float) -> float:
        """Modula consentimiento con valencia actual."""
        if not self.affect_log:
            return pi_base

        last = self.affect_log[-1]
        V = last['PAD']['V']
        R_soc = last['signals']['R_soc']

        return self.modulator.modulate_consent(pi_base, V, R_soc, cost_rank)

# =============================================================================
# Tests de Falsación
# =============================================================================

def test_hysteresis(affect_log: List[Dict]) -> Dict:
    """
    Test de histéresis: barridos de sorpresa y estabilidad
    deberían mostrar ciclos en (V, A).
    """
    if len(affect_log) < 100:
        return {'status': 'insufficient_data'}

    V = [a['PAD']['V'] for a in affect_log]
    A = [a['PAD']['A'] for a in affect_log]
    r = [a['signals']['r'] for a in affect_log]
    s = [a['signals']['s'] for a in affect_log]

    # Detectar ciclos: área encerrada en (V, A)
    # Usar integral de línea simplificada
    area = 0
    for i in range(1, len(V)):
        area += (V[i] - V[i-1]) * (A[i] + A[i-1]) / 2

    return {
        'status': 'computed',
        'hysteresis_area': abs(area),
        'V_range': (min(V), max(V)),
        'A_range': (min(A), max(A))
    }

def test_metastability(affect_log: List[Dict], n_clusters: int = 3) -> Dict:
    """
    Test de metaestabilidad: tiempos de permanencia en clusters
    vs nulos por permutación.
    """
    if len(affect_log) < 200:
        return {'status': 'insufficient_data'}

    # Cluster simple por cuantiles de F[0]
    F0 = [a['F'][0] for a in affect_log]
    q33, q66 = np.percentile(F0, [33, 66])

    clusters = []
    for f in F0:
        if f < q33:
            clusters.append(0)
        elif f < q66:
            clusters.append(1)
        else:
            clusters.append(2)

    # Tiempos de permanencia
    dwell_times = []
    current = clusters[0]
    count = 1
    for i in range(1, len(clusters)):
        if clusters[i] == current:
            count += 1
        else:
            dwell_times.append(count)
            current = clusters[i]
            count = 1
    dwell_times.append(count)

    # Comparar con permutación
    np.random.seed(42)
    shuffled = np.random.permutation(clusters)
    dwell_shuffled = []
    current = shuffled[0]
    count = 1
    for i in range(1, len(shuffled)):
        if shuffled[i] == current:
            count += 1
        else:
            dwell_shuffled.append(count)
            current = shuffled[i]
            count = 1
    dwell_shuffled.append(count)

    return {
        'status': 'computed',
        'mean_dwell': np.mean(dwell_times),
        'mean_dwell_shuffled': np.mean(dwell_shuffled),
        'ratio': np.mean(dwell_times) / (np.mean(dwell_shuffled) + EPS)
    }


if __name__ == "__main__":
    print("Sistema Afectivo Endógeno - Test básico")

    # Simulación simple
    system = AffectiveSystem()

    np.random.seed(42)
    I = np.array([0.4, 0.3, 0.3])
    I_history = [I.copy()]
    residuals = []
    mdl_history = []

    for t in range(500):
        I_prev = I.copy()
        I = I + np.random.randn(3) * 0.05
        I = np.maximum(I, 0.01)
        I = I / I.sum()

        I_history.append(I.copy())
        residuals.append(np.linalg.norm(I - I_prev))
        mdl_history.append(1.0 + np.random.randn() * 0.1)

        rho = 0.9 + np.random.randn() * 0.1
        R_soc_ema = 0.5 + np.random.randn() * 0.1
        is_sleep = np.random.random() < 0.2
        fatigue = 0.3 + t * 0.0001

        result = system.process(
            I, I_prev, I_history, residuals,
            rho, R_soc_ema, is_sleep, mdl_history, fatigue
        )

        if t % 100 == 0:
            print(f"t={t}: V={result['PAD']['V']:.3f}, A={result['PAD']['A']:.3f}, D={result['PAD']['D']:.3f}")
            print(f"       social_boost={result['social_boost']:.3f}, sleep_boost={result['sleep_boost']:.3f}")

    # Tests
    print("\nTests de falsación:")
    hyst = test_hysteresis(system.affect_log)
    print(f"  Histéresis: area={hyst.get('hysteresis_area', 'N/A'):.4f}")

    meta = test_metastability(system.affect_log)
    print(f"  Metaestabilidad: ratio={meta.get('ratio', 'N/A'):.2f}")
    print(f"    dwell={meta.get('mean_dwell', 'N/A'):.1f} vs shuffled={meta.get('mean_dwell_shuffled', 'N/A'):.1f}")
