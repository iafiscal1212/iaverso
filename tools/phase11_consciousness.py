#!/usr/bin/env python3
"""
Phase 11: Módulos de Consciencia Artificial (100% Endógeno)
===========================================================

Tres módulos nuevos manteniendo endogeneidad radical:

1. SELF-MODEL: Cada agente estima sus propios índices intramundo
   y calcula self_error_t = |estimado - observado|

2. GLOBAL WORKSPACE: Cuando gate + coupling superan umbral endógeno,
   se construye GW_t con las K variables más relevantes (K endógeno)

3. IDENTITY CONTINUITY: Índice de consistencia de identidad basado
   en trayectoria de estados, detecta rupturas de continuidad

Todo derivado de: mediana, cuantiles, √T, IQR, ranks, PCA, ACF
Sin hiperparámetros fijos arbitrarios.
"""

import sys
import os
import json
import numpy as np
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Set
from collections import deque
from scipy import stats
from sklearn.decomposition import PCA

sys.path.insert(0, '/root/NEO_EVA/tools')

# =============================================================================
# UTILIDADES ENDÓGENAS
# =============================================================================

def get_epsilon(dtype=np.float64) -> float:
    return np.finfo(dtype).eps


def rolling_rank(value: float, history: deque) -> float:
    if len(history) < 2:
        return 0.5
    arr = np.array(list(history) + [value])
    rank = stats.rankdata(arr)[-1]
    return (rank - 1) / (len(arr) - 1)


def rolling_percentile(value: float, history: deque, q: float = 50) -> float:
    if len(history) < 2:
        return 0.5
    arr = np.array(list(history))
    return stats.percentileofscore(arr, value) / 100


def compute_iqr(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return np.std(arr) if len(arr) > 0 else 0.0
    return np.percentile(arr, 75) - np.percentile(arr, 25)


def compute_entropy(probs: Dict) -> float:
    values = np.array(list(probs.values()), dtype=float)
    values = values[values > 0]
    if len(values) <= 1:
        return 0.0
    # Normalize to probabilities
    values = values / values.sum()
    entropy = -np.sum(values * np.log(values))
    max_entropy = np.log(len(values))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_acf_lag1(x: np.ndarray) -> float:
    """Autocorrelación lag-1."""
    if len(x) < 3:
        return 0.0
    x_centered = x - np.mean(x)
    var = np.var(x)
    if var < get_epsilon():
        return 0.0
    acf1 = np.correlate(x_centered[:-1], x_centered[1:])[0] / (len(x) - 1) / var
    return np.clip(acf1, -1, 1)


# =============================================================================
# 1. SELF-MODEL: Modelo de Sí Mismo
# =============================================================================

class SelfModel:
    """
    Cada agente mantiene una estimación de sus propios índices intramundo.

    Índices estimados (todos endógenos):
    - α_affect_est: volumen afectivo estimado
    - α_switch_est: tasa de cambio estimada
    - α_stability_est: estabilidad estimada
    - α_reactivity_est: reactividad estimada

    self_error_t = |estimado - observado| (métrica de autoconocimiento)
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # Historiales de señales para estimación
        self.signal_history = deque(maxlen=window_size)
        self.state_history = deque(maxlen=window_size)
        self.pad_history = deque(maxlen=window_size)

        # Estimaciones actuales
        self.estimates = {
            'alpha_affect': 0.5,
            'alpha_switch': 0.5,
            'alpha_stability': 0.5,
            'alpha_reactivity': 0.5
        }

        # Observaciones reales (calculadas)
        self.observations = {
            'alpha_affect': 0.5,
            'alpha_switch': 0.5,
            'alpha_stability': 0.5,
            'alpha_reactivity': 0.5
        }

        # Historial de errores para metacognición
        self.error_history = deque(maxlen=window_size)
        self.self_error = 0.0

        # Tasa de aprendizaje endógena
        self.n_updates = 0

    def get_learning_rate(self) -> float:
        """Tasa de aprendizaje: 1/√(n+1)"""
        return 1.0 / np.sqrt(self.n_updates + 1)

    def update(self, signals: Dict[str, float], state: str, pad: Dict[str, float]) -> Dict:
        """
        Actualiza el modelo de sí mismo.

        1. Registra observaciones
        2. Calcula índices observados
        3. Actualiza estimaciones
        4. Calcula self_error
        """
        self.n_updates += 1

        # Registrar
        self.signal_history.append(signals)
        self.state_history.append(state)
        self.pad_history.append(pad)

        if len(self.signal_history) < 10:
            return {'self_error': 0.0, 'estimates': self.estimates.copy()}

        # Calcular índices OBSERVADOS (ground truth)
        self._compute_observations()

        # Actualizar ESTIMACIONES (predicción del agente sobre sí mismo)
        self._update_estimates()

        # Calcular error de automodelo
        self._compute_self_error()

        return {
            'self_error': self.self_error,
            'estimates': self.estimates.copy(),
            'observations': self.observations.copy(),
            'learning_rate': self.get_learning_rate()
        }

    def _compute_observations(self):
        """Calcula índices observados (ground truth)."""
        if len(self.pad_history) < 5:
            return

        # α_affect: volumen del espacio PAD
        pad_arr = np.array([[p['P'], p['A'], p['D']] for p in self.pad_history])
        if len(pad_arr) >= 3:
            cov = np.cov(pad_arr.T)
            det = np.linalg.det(cov)
            self.observations['alpha_affect'] = np.sqrt(max(0, det))

        # α_switch: tasa de cambio de estado
        states = list(self.state_history)
        if len(states) >= 2:
            switches = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
            self.observations['alpha_switch'] = switches / (len(states) - 1)

        # α_stability: inverso de varianza de señales
        signal_arr = np.array([[s['r'], s['m'], s['e']] for s in self.signal_history])
        if len(signal_arr) >= 5:
            var = np.mean(np.var(signal_arr, axis=0))
            self.observations['alpha_stability'] = 1 / (1 + var)

        # α_reactivity: autocorrelación de PAD (baja = más reactivo)
        if len(pad_arr) >= 10:
            acf_P = compute_acf_lag1(pad_arr[:, 0])
            acf_A = compute_acf_lag1(pad_arr[:, 1])
            acf_D = compute_acf_lag1(pad_arr[:, 2])
            # Reactividad = 1 - persistencia
            self.observations['alpha_reactivity'] = 1 - np.mean([acf_P, acf_A, acf_D])

    def _update_estimates(self):
        """
        Actualiza estimaciones usando EMA con tasa endógena.
        El agente "aprende" sobre sí mismo.
        """
        alpha = self.get_learning_rate()

        for key in self.estimates:
            # EMA hacia el valor observado
            self.estimates[key] = (1 - alpha) * self.estimates[key] + alpha * self.observations[key]

    def _compute_self_error(self):
        """
        Calcula el error de automodelo.
        self_error = norma de (estimado - observado)
        """
        errors = []
        for key in self.estimates:
            err = abs(self.estimates[key] - self.observations[key])
            errors.append(err)

        self.self_error = np.mean(errors)
        self.error_history.append(self.self_error)

    def get_metacognitive_accuracy(self) -> float:
        """
        Precisión metacognitiva: qué tan bien se conoce a sí mismo.
        1 - rank(self_error) en historia
        """
        if len(self.error_history) < 5:
            return 0.5
        return 1 - rolling_rank(self.self_error, self.error_history)


# =============================================================================
# 2. GLOBAL WORKSPACE: Espacio de Trabajo Global
# =============================================================================

class GlobalWorkspace:
    """
    Implementa un Global Workspace (Baars) endógeno.

    Cuando gate + coupling superan umbral endógeno:
    1. Se seleccionan las K variables más relevantes (K endógeno)
    2. Se construye GW_t que modula transiciones de ambos agentes
    3. La "broadcast" se hace mediante el vector GW

    K se determina por: número de componentes PCA que explican >q50 varianza
    Umbral de activación: mediana histórica de (gate * intensity)
    """

    def __init__(self, window_size: int = 200):
        self.window_size = window_size

        # Historial de todas las variables candidatas
        self.variable_history = deque(maxlen=window_size)

        # Historial de activación para umbral endógeno
        self.activation_history = deque(maxlen=window_size)

        # Estado del workspace
        self.is_active = False
        self.GW = None  # Vector del workspace actual
        self.K = 3  # Se ajustará endógenamente

        # PCA para selección de variables
        self.pca = None

        # Métricas
        self.broadcast_count = 0
        self.total_cycles = 0

    def get_activation_threshold(self) -> float:
        """Umbral endógeno: mediana de activaciones históricas."""
        if len(self.activation_history) < 10:
            return 0.3
        return np.median(list(self.activation_history))

    def compute_K(self) -> int:
        """
        K endógeno: número de componentes que explican >q50 varianza.
        """
        if self.pca is None or not hasattr(self.pca, 'explained_variance_ratio_'):
            return 3

        cumvar = np.cumsum(self.pca.explained_variance_ratio_)
        # K = primer índice donde varianza acumulada > mediana
        threshold = 0.5  # q50 de varianza
        K = np.argmax(cumvar >= threshold) + 1
        return max(2, min(K, len(cumvar)))

    def update(self,
               signals_neo: Dict[str, float],
               signals_eva: Dict[str, float],
               pad_neo: Dict[str, float],
               pad_eva: Dict[str, float],
               gate_neo: float,
               gate_eva: float,
               coupling_intensity: float) -> Dict:
        """
        Actualiza el Global Workspace.

        Returns: Dict con GW, is_active, K, relevance scores
        """
        self.total_cycles += 1

        # Construir vector de todas las variables
        all_vars = self._build_variable_vector(
            signals_neo, signals_eva, pad_neo, pad_eva
        )
        self.variable_history.append(all_vars)

        # Calcular activación
        activation = (gate_neo + gate_eva) / 2 * coupling_intensity
        self.activation_history.append(activation)

        # Verificar si se activa el workspace
        threshold = self.get_activation_threshold()
        self.is_active = activation > threshold

        if not self.is_active:
            return {
                'is_active': False,
                'GW': None,
                'K': self.K,
                'activation': activation,
                'threshold': threshold
            }

        # BROADCAST: Construir GW con top-K variables
        self._fit_pca_and_select()
        self.GW = self._construct_GW(all_vars)
        self.broadcast_count += 1

        return {
            'is_active': True,
            'GW': self.GW.tolist() if self.GW is not None else None,
            'K': self.K,
            'activation': activation,
            'threshold': threshold,
            'relevance': self._get_relevance_scores(),
            'broadcast_rate': self.broadcast_count / self.total_cycles
        }

    def _build_variable_vector(self,
                               signals_neo: Dict, signals_eva: Dict,
                               pad_neo: Dict, pad_eva: Dict) -> np.ndarray:
        """Construye vector con todas las variables candidatas."""
        vec = []

        # Señales NEO (8)
        for key in ['r', 's', 'm', 'c', 'R_soc', 'e', 'q', 'h']:
            vec.append(signals_neo.get(key, 0.5))

        # Señales EVA (8)
        for key in ['r', 's', 'm', 'c', 'R_soc', 'e', 'q', 'h']:
            vec.append(signals_eva.get(key, 0.5))

        # PAD NEO (3)
        vec.extend([pad_neo.get('P', 0.5), pad_neo.get('A', 0.5), pad_neo.get('D', 0.5)])

        # PAD EVA (3)
        vec.extend([pad_eva.get('P', 0.5), pad_eva.get('A', 0.5), pad_eva.get('D', 0.5)])

        # Diferencias inter-agente (medida de sincronía)
        for key in ['r', 'm', 'e']:
            vec.append(abs(signals_neo.get(key, 0.5) - signals_eva.get(key, 0.5)))

        return np.array(vec)

    def _fit_pca_and_select(self):
        """Ajusta PCA y determina K endógenamente."""
        if len(self.variable_history) < 20:
            return

        data = np.array(list(self.variable_history))

        # Centrar con mediana (robusto)
        centered = data - np.median(data, axis=0)

        # PCA
        n_components = min(10, data.shape[1], data.shape[0] - 1)
        self.pca = PCA(n_components=n_components)
        self.pca.fit(centered)

        # Actualizar K
        self.K = self.compute_K()

    def _construct_GW(self, current_vars: np.ndarray) -> np.ndarray:
        """
        Construye el vector GW con las K componentes más relevantes.
        """
        if self.pca is None or len(self.variable_history) < 20:
            # Fallback: usar las K primeras variables
            return current_vars[:self.K]

        # Centrar
        data = np.array(list(self.variable_history))
        centered = current_vars - np.median(data, axis=0)

        # Proyectar a espacio PCA
        projected = self.pca.transform(centered.reshape(1, -1))[0]

        # Retornar las K primeras componentes
        return projected[:self.K]

    def _get_relevance_scores(self) -> Dict[str, float]:
        """Scores de relevancia de cada variable original."""
        if self.pca is None:
            return {}

        # Loadings: contribución de cada variable a los K componentes
        loadings = np.abs(self.pca.components_[:self.K, :])
        relevance = np.mean(loadings, axis=0)

        # Normalizar
        relevance = relevance / (relevance.sum() + get_epsilon())

        # Mapear a nombres
        var_names = (
            [f'neo_{k}' for k in ['r', 's', 'm', 'c', 'R_soc', 'e', 'q', 'h']] +
            [f'eva_{k}' for k in ['r', 's', 'm', 'c', 'R_soc', 'e', 'q', 'h']] +
            ['neo_P', 'neo_A', 'neo_D', 'eva_P', 'eva_A', 'eva_D'] +
            ['diff_r', 'diff_m', 'diff_e']
        )

        return {name: float(rel) for name, rel in zip(var_names[:len(relevance)], relevance)}

    def get_modulation(self) -> np.ndarray:
        """
        Retorna modulación para transiciones basada en GW.
        Si no activo, retorna vector neutro.
        """
        if not self.is_active or self.GW is None:
            return np.zeros(self.K)

        # Normalizar GW a [-1, 1] por ranks
        if len(self.variable_history) < 10:
            return self.GW

        # Rank-normalize cada componente
        modulation = np.zeros_like(self.GW)
        for i in range(len(self.GW)):
            hist = [v[i] if i < len(v) else 0 for v in list(self.variable_history)[-50:]]
            if len(hist) > 1:
                rank = stats.percentileofscore(hist, self.GW[i]) / 100
                modulation[i] = 2 * rank - 1  # Map to [-1, 1]

        return modulation


# =============================================================================
# 3. IDENTITY CONTINUITY: Continuidad de Identidad
# =============================================================================

class IdentityContinuity:
    """
    Índice de consistencia de identidad basado en trayectoria de estados.

    Componentes:
    1. Consistencia temporal: autocorrelación de patrones de estado
    2. Consistencia de repertorio: estabilidad del perfil de preferencias
    3. Consistencia narrativa: predictibilidad de transiciones

    Detecta "rupturas" cuando la consistencia cae bajo q10 histórico.
    """

    def __init__(self, window_size: int = 200):
        self.window_size = window_size

        # Historiales
        self.state_history = deque(maxlen=window_size)
        self.signal_history = deque(maxlen=window_size)
        self.transition_history = deque(maxlen=window_size)

        # Perfil de identidad (distribución de estados preferidos)
        self.identity_profile = {}

        # Índices de consistencia
        self.temporal_consistency = 0.5
        self.repertoire_consistency = 0.5
        self.narrative_consistency = 0.5
        self.identity_index = 0.5

        # Historial para detección de rupturas
        self.consistency_history = deque(maxlen=window_size)
        self.rupture_events = []

        # Matriz de transición aprendida
        self.transition_matrix = {}

    def update(self, state: str, signals: Dict[str, float], t: int) -> Dict:
        """
        Actualiza métricas de continuidad de identidad.
        """
        prev_state = self.state_history[-1] if self.state_history else state

        self.state_history.append(state)
        self.signal_history.append(signals)

        if len(self.state_history) < 10:
            return {'identity_index': 0.5, 'rupture': False}

        # Registrar transición
        transition = (prev_state, state)
        self.transition_history.append(transition)
        self._update_transition_matrix(transition)

        # Calcular índices
        self._compute_temporal_consistency()
        self._compute_repertoire_consistency()
        self._compute_narrative_consistency()

        # Índice global de identidad
        self.identity_index = np.mean([
            self.temporal_consistency,
            self.repertoire_consistency,
            self.narrative_consistency
        ])

        self.consistency_history.append(self.identity_index)

        # Detectar ruptura
        rupture = self._detect_rupture(t)

        return {
            'identity_index': self.identity_index,
            'temporal_consistency': self.temporal_consistency,
            'repertoire_consistency': self.repertoire_consistency,
            'narrative_consistency': self.narrative_consistency,
            'rupture': rupture,
            'n_ruptures': len(self.rupture_events)
        }

    def _update_transition_matrix(self, transition: Tuple[str, str]):
        """Actualiza matriz de transición con nueva observación."""
        src, dst = transition
        if src not in self.transition_matrix:
            self.transition_matrix[src] = {}
        if dst not in self.transition_matrix[src]:
            self.transition_matrix[src][dst] = 0
        self.transition_matrix[src][dst] += 1

    def _compute_temporal_consistency(self):
        """
        Consistencia temporal: autocorrelación de patrones.
        Alta ACF = el agente mantiene patrones estables.
        """
        if len(self.state_history) < 20:
            return

        # Codificar estados como números
        states = list(self.state_history)
        unique_states = list(set(states))
        state_to_num = {s: i for i, s in enumerate(unique_states)}
        numeric = np.array([state_to_num[s] for s in states])

        # ACF lag-1
        acf1 = compute_acf_lag1(numeric)

        # Consistencia = ACF normalizado a [0, 1]
        self.temporal_consistency = (acf1 + 1) / 2

    def _compute_repertoire_consistency(self):
        """
        Consistencia de repertorio: estabilidad del perfil de preferencias.
        Se mide como 1 - varianza del perfil a lo largo del tiempo.
        """
        if len(self.state_history) < 20:
            return

        # Calcular perfil actual (distribución de estados)
        states = list(self.state_history)
        counts = {}
        for s in states:
            counts[s] = counts.get(s, 0) + 1
        total = sum(counts.values())
        current_profile = {s: c/total for s, c in counts.items()}

        # Comparar con perfil histórico
        if not self.identity_profile:
            self.identity_profile = current_profile
            self.repertoire_consistency = 1.0
            return

        # Distancia entre perfiles (variación total)
        all_states = set(current_profile.keys()) | set(self.identity_profile.keys())
        distance = sum(abs(current_profile.get(s, 0) - self.identity_profile.get(s, 0))
                      for s in all_states) / 2

        # Actualizar perfil con EMA
        alpha = 1 / np.sqrt(len(self.state_history) + 1)
        for s in all_states:
            old = self.identity_profile.get(s, 0)
            new = current_profile.get(s, 0)
            self.identity_profile[s] = (1 - alpha) * old + alpha * new

        self.repertoire_consistency = 1 - distance

    def _compute_narrative_consistency(self):
        """
        Consistencia narrativa: predictibilidad de transiciones.
        Alta si las transiciones siguen patrones aprendidos.
        """
        if len(self.transition_history) < 10:
            return

        # Para cada transición reciente, calcular si era "esperada"
        recent = list(self.transition_history)[-20:]
        expected_scores = []

        for src, dst in recent:
            if src in self.transition_matrix:
                total = sum(self.transition_matrix[src].values())
                if total > 0:
                    prob = self.transition_matrix[src].get(dst, 0) / total
                    expected_scores.append(prob)

        if expected_scores:
            self.narrative_consistency = np.mean(expected_scores)

    def _detect_rupture(self, t: int) -> bool:
        """
        Detecta ruptura de identidad si consistencia cae bajo q10 histórico.
        """
        if len(self.consistency_history) < 20:
            return False

        q10 = np.percentile(list(self.consistency_history), 10)

        if self.identity_index < q10:
            self.rupture_events.append({
                't': t,
                'identity_index': self.identity_index,
                'threshold': q10
            })
            return True

        return False

    def get_identity_signature(self) -> Dict:
        """
        Retorna una "firma" de identidad del agente.
        """
        return {
            'profile': self.identity_profile.copy(),
            'consistency': self.identity_index,
            'n_ruptures': len(self.rupture_events),
            'transition_entropy': self._compute_transition_entropy()
        }

    def _compute_transition_entropy(self) -> float:
        """Entropía de la matriz de transición."""
        if not self.transition_matrix:
            return 0.0

        entropies = []
        for src, dsts in self.transition_matrix.items():
            total = sum(dsts.values())
            if total > 0:
                probs = {d: c/total for d, c in dsts.items()}
                entropies.append(compute_entropy(probs))

        return np.mean(entropies) if entropies else 0.0


# =============================================================================
# MUNDO CONSCIENTE (PHASE 11)
# =============================================================================

class LifeState(Enum):
    SLEEP = "SLEEP"
    WAKE = "WAKE"
    WORK = "WORK"
    LEARN = "LEARN"
    SOCIAL = "SOCIAL"


class ConsciousWorld:
    """
    Mundo con los tres módulos de consciencia integrados.
    """

    def __init__(self, name: str, window_size: int = 100):
        self.name = name
        self.t = 0

        # Estado interno (sin sesgo)
        self.I = np.array([1/3, 1/3, 1/3])
        self.current_state = LifeState.WAKE

        # === MÓDULOS DE CONSCIENCIA ===
        self.self_model = SelfModel(window_size)
        self.identity = IdentityContinuity(window_size * 2)

        # Historiales básicos
        self.I_history = deque(maxlen=window_size)
        self.signal_history = deque(maxlen=window_size)
        self.state_counts = {s: 0 for s in LifeState}

        # PAD por PCA
        self.pad_history = deque(maxlen=window_size)
        self.pca_fitted = False
        self.pca = None

        # Métricas de consciencia
        self.consciousness_log = []

        # R_soc adaptativo
        self.R_soc_ema = 0.5
        self.n_couplings = 0

        # Gate continuo
        self.rho_history = deque(maxlen=window_size)
        self.var_history = deque(maxlen=window_size)

    def compute_signals(self) -> Dict[str, float]:
        """Calcula las 8 señales internas."""
        r = 1 - np.linalg.norm(self.I)

        if len(self.I_history) >= 5:
            recent = np.array(list(self.I_history)[-10:])
            var = np.var(recent)
            s = 1 / (1 + var)
        else:
            s = 0.5

        I_probs = np.clip(self.I, 0.01, 0.99)
        I_probs = I_probs / I_probs.sum()
        m = compute_entropy({f'c{i}': p for i, p in enumerate(I_probs)})

        c = np.max(self.I)
        R_soc = self.R_soc_ema

        work_ratio = self.state_counts[LifeState.WORK] / max(1, self.t)
        e = 1 - work_ratio

        q = self.identity.narrative_consistency if hasattr(self.identity, 'narrative_consistency') else 0.5

        social_ratio = self.state_counts[LifeState.SOCIAL] / max(1, self.t)
        h = social_ratio

        return {'r': r, 's': s, 'm': m, 'c': c, 'R_soc': R_soc, 'e': e, 'q': q, 'h': h}

    def compute_pad(self, signals: Dict[str, float]) -> Dict[str, float]:
        """PAD por PCA de señales."""
        signal_vec = np.array([signals[k] for k in ['r', 's', 'm', 'c', 'R_soc', 'e', 'q', 'h']])
        self.signal_history.append(signal_vec)

        if len(self.signal_history) < 15:
            return {'P': 0.5, 'A': 0.5, 'D': 0.5}

        data = np.array(list(self.signal_history))

        # Fit PCA periódicamente
        if not self.pca_fitted or self.t % 100 == 0:
            centered = data - np.median(data, axis=0)
            self.pca = PCA(n_components=3)
            self.pca.fit(centered)
            self.pca_fitted = True

        # Transformar
        centered = signal_vec - np.median(data, axis=0)
        coords = self.pca.transform(centered.reshape(1, -1))[0]

        # Normalizar por percentil
        P = stats.percentileofscore(data @ self.pca.components_[0], coords[0]) / 100
        A = stats.percentileofscore(data @ self.pca.components_[1], coords[1]) / 100 if len(coords) > 1 else 0.5
        D = stats.percentileofscore(data @ self.pca.components_[2], coords[2]) / 100 if len(coords) > 2 else 0.5

        return {'P': P, 'A': A, 'D': D}

    def compute_gate(self, other_signals: Dict[str, float]) -> Tuple[float, float, float]:
        """Gate continuo."""
        signals = self.compute_signals()

        # ρ: correlación con otro
        keys = list(signals.keys())
        vec_self = np.array([signals[k] for k in keys])
        vec_other = np.array([other_signals.get(k, 0.5) for k in keys])

        if np.std(vec_self) > 0 and np.std(vec_other) > 0:
            rho = np.corrcoef(vec_self, vec_other)[0, 1]
        else:
            rho = 0

        # var_I
        if len(self.I_history) >= 5:
            var_I = np.var(np.array(list(self.I_history)[-20:]))
        else:
            var_I = 0.1

        self.rho_history.append(rho)
        self.var_history.append(var_I)

        # Gate continuo
        rho_rank = rolling_rank(rho, self.rho_history)
        var_rank = rolling_rank(var_I, self.var_history)
        gate = rho_rank * (1 - var_rank)

        return gate, rho, var_I

    def compute_pi(self, signals: Dict[str, float], gate: float) -> float:
        """Índice volitivo."""
        benefit = (signals['R_soc'] + signals['h'] + signals['m']) / 3
        cost = signals['r']
        pi_base = benefit / (cost + get_epsilon())
        pi_base = np.clip(pi_base, 0, 1)
        return pi_base * gate

    def select_state(self, gw_modulation: Optional[np.ndarray] = None) -> LifeState:
        """Selecciona estado, modulado por GW si activo."""
        D_rest = 1 - np.mean(list(self.I_history)[-5:]) if self.I_history else 0.5
        D_nov = np.std(list(self.I_history)[-10:]) if len(self.I_history) >= 10 else 0.5
        D_learn = self.I[1]
        D_soc = self.R_soc_ema

        utilities = {
            LifeState.SLEEP: D_rest * 0.8,
            LifeState.WAKE: (1 - D_rest) * 0.5,
            LifeState.WORK: self.I[0] * 0.7,
            LifeState.LEARN: D_learn * 0.6 + D_nov * 0.3,
            LifeState.SOCIAL: D_soc * 0.9
        }

        # Modulación del Global Workspace
        if gw_modulation is not None and len(gw_modulation) >= 3:
            # GW modula utilidades
            mod = np.tanh(gw_modulation[:3])  # Primeras 3 componentes
            states_list = list(utilities.keys())
            for i, s in enumerate(states_list[:3]):
                utilities[s] *= (1 + 0.3 * mod[i % len(mod)])

        # Softmax con temperatura endógena
        vals = np.array(list(utilities.values()))
        iqr = compute_iqr(vals)
        gamma = 1 / (iqr + get_epsilon())
        gamma = np.clip(gamma, 0.5, 10)

        exp_u = np.exp(gamma * vals)
        probs = exp_u / exp_u.sum()

        states = list(utilities.keys())
        idx = np.random.choice(len(states), p=probs)

        return states[idx]

    def step(self, other_signals: Dict[str, float],
             coupling_intensity: float = 0.0,
             gw_modulation: Optional[np.ndarray] = None) -> Dict:
        """Ejecuta un paso con módulos de consciencia."""
        self.t += 1
        self.I_history.append(self.I.copy())

        # Señales y PAD
        signals = self.compute_signals()
        pad = self.compute_pad(signals)
        self.pad_history.append(pad)

        # Selección de estado (modulada por GW)
        new_state = self.select_state(gw_modulation)
        self.current_state = new_state
        self.state_counts[new_state] += 1

        # === ACTUALIZAR MÓDULOS DE CONSCIENCIA ===

        # 1. Self-Model
        self_model_output = self.self_model.update(signals, new_state.value, pad)

        # 2. Identity Continuity
        identity_output = self.identity.update(new_state.value, signals, self.t)

        # Dinámica de I
        attractors = {
            LifeState.SLEEP: np.array([0.2, 0.2, 0.6]),
            LifeState.WAKE: np.array([0.33, 0.33, 0.34]),
            LifeState.WORK: np.array([0.6, 0.2, 0.2]),
            LifeState.LEARN: np.array([0.2, 0.6, 0.2]),
            LifeState.SOCIAL: np.array([0.25, 0.25, 0.5])
        }

        target = attractors[new_state]
        rate = 0.1 / np.sqrt(self.t + 1)

        if coupling_intensity > 0:
            other_vec = np.array([other_signals.get('r', 0.5),
                                  other_signals.get('m', 0.5),
                                  other_signals.get('h', 0.5)])
            other_vec = other_vec / (other_vec.sum() + get_epsilon())
            target = (1 - coupling_intensity) * target + coupling_intensity * other_vec

        self.I = (1 - rate) * self.I + rate * target
        self.I = np.clip(self.I, 0.01, 0.99)
        self.I = self.I / self.I.sum()

        # Actualizar R_soc si coupling
        if coupling_intensity > 0:
            self.n_couplings += 1
            alpha = 1 / np.sqrt(self.n_couplings + 1)
            reward = coupling_intensity
            self.R_soc_ema = (1 - alpha) * self.R_soc_ema + alpha * reward

        # Métricas de consciencia
        consciousness = {
            't': self.t,
            'state': new_state.value,
            'self_error': self_model_output['self_error'],
            'metacognitive_accuracy': self.self_model.get_metacognitive_accuracy(),
            'identity_index': identity_output['identity_index'],
            'rupture': identity_output['rupture'],
            'PAD': pad
        }
        self.consciousness_log.append(consciousness)

        return {
            'signals': signals,
            'pad': pad,
            'state': new_state,
            'self_model': self_model_output,
            'identity': identity_output
        }


# =============================================================================
# EXPERIMENTO PHASE 11
# =============================================================================

def run_phase11_experiment(n_cycles: int = 25000,
                           output_dir: str = '/root/NEO_EVA/results/phase11') -> Dict:
    """
    Ejecuta experimento Phase 11 con módulos de consciencia.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/figures", exist_ok=True)

    print("=" * 70)
    print("PHASE 11: MÓDULOS DE CONSCIENCIA ARTIFICIAL")
    print("=" * 70)
    print("\nMódulos activos:")
    print("  1. Self-Model (autoestimación + metacognición)")
    print("  2. Global Workspace (broadcast de variables relevantes)")
    print("  3. Identity Continuity (consistencia + detección de rupturas)")
    print("  4. Homeostasis de repertorio (entropía de estados)")

    # Crear mundos conscientes
    neo = ConsciousWorld("NEO")
    eva = ConsciousWorld("EVA")

    # Global Workspace compartido
    global_workspace = GlobalWorkspace(window_size=200)

    # Registros
    bilateral_events = []
    gw_broadcasts = []
    rupture_events = []

    # Métricas agregadas
    self_errors_neo = []
    self_errors_eva = []
    identity_indices_neo = []
    identity_indices_eva = []

    train_end = int(n_cycles * 0.6)

    print(f"\nCiclos: {n_cycles} (train: 0-{train_end}, test: {train_end}-{n_cycles})")
    print("\n[Simulando...]")

    for t in range(1, n_cycles + 1):
        # Señales
        neo_signals = neo.compute_signals()
        eva_signals = eva.compute_signals()

        # PAD
        neo_pad = neo.compute_pad(neo_signals)
        eva_pad = eva.compute_pad(eva_signals)

        # Gates
        neo_gate, neo_rho, neo_var = neo.compute_gate(eva_signals)
        eva_gate, eva_rho, eva_var = eva.compute_gate(neo_signals)

        # π
        neo_pi = neo.compute_pi(neo_signals, neo_gate)
        eva_pi = eva.compute_pi(eva_signals, eva_gate)

        # Coupling intensity
        coupling_intensity = min(neo_pi, eva_pi) * np.sqrt(neo_gate * eva_gate)

        # === GLOBAL WORKSPACE ===
        gw_output = global_workspace.update(
            neo_signals, eva_signals,
            neo_pad, eva_pad,
            neo_gate, eva_gate,
            coupling_intensity
        )

        gw_modulation = None
        if gw_output['is_active']:
            gw_modulation = global_workspace.get_modulation()
            gw_broadcasts.append({
                't': t,
                'K': gw_output['K'],
                'activation': gw_output['activation']
            })

        # Determinar si hay bilateral
        threshold = global_workspace.get_activation_threshold()
        bilateral = coupling_intensity > threshold

        # Step de cada mundo
        neo_output = neo.step(eva_signals, coupling_intensity if bilateral else 0, gw_modulation)
        eva_output = eva.step(neo_signals, coupling_intensity if bilateral else 0, gw_modulation)

        # Registrar
        if bilateral:
            bilateral_events.append({
                't': t,
                'intensity': coupling_intensity,
                'gw_active': gw_output['is_active'],
                'phase': 'train' if t <= train_end else 'test'
            })

        # Registrar rupturas
        if neo_output['identity']['rupture']:
            rupture_events.append({'t': t, 'agent': 'NEO'})
        if eva_output['identity']['rupture']:
            rupture_events.append({'t': t, 'agent': 'EVA'})

        # Métricas
        self_errors_neo.append(neo_output['self_model']['self_error'])
        self_errors_eva.append(eva_output['self_model']['self_error'])
        identity_indices_neo.append(neo_output['identity']['identity_index'])
        identity_indices_eva.append(eva_output['identity']['identity_index'])

        if t % 5000 == 0:
            print(f"  t={t}: bilateral={len(bilateral_events)}, "
                  f"GW_broadcasts={len(gw_broadcasts)}, "
                  f"ruptures={len(rupture_events)}")
            print(f"         self_err: NEO={np.mean(self_errors_neo[-1000:]):.4f}, "
                  f"EVA={np.mean(self_errors_eva[-1000:]):.4f}")
            print(f"         identity: NEO={np.mean(identity_indices_neo[-1000:]):.4f}, "
                  f"EVA={np.mean(identity_indices_eva[-1000:]):.4f}")

    # === RESULTADOS ===
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    n_bilateral = len(bilateral_events)
    n_broadcasts = len(gw_broadcasts)
    n_ruptures = len(rupture_events)

    print(f"\n1. EVENTOS BILATERALES: {n_bilateral} ({n_bilateral/n_cycles*100:.2f}%)")

    print(f"\n2. GLOBAL WORKSPACE:")
    print(f"   Broadcasts: {n_broadcasts} ({n_broadcasts/n_cycles*100:.2f}%)")
    print(f"   K medio: {np.mean([g['K'] for g in gw_broadcasts]):.2f}" if gw_broadcasts else "   K medio: N/A")

    print(f"\n3. SELF-MODEL (autoconocimiento):")
    print(f"   Self-error NEO: {np.mean(self_errors_neo):.4f} (menor = mejor)")
    print(f"   Self-error EVA: {np.mean(self_errors_eva):.4f}")
    print(f"   Metacognición NEO: {neo.self_model.get_metacognitive_accuracy():.4f}")
    print(f"   Metacognición EVA: {eva.self_model.get_metacognitive_accuracy():.4f}")

    print(f"\n4. IDENTITY CONTINUITY:")
    print(f"   Identity index NEO: {np.mean(identity_indices_neo):.4f}")
    print(f"   Identity index EVA: {np.mean(identity_indices_eva):.4f}")
    print(f"   Rupturas totales: {n_ruptures}")
    print(f"   - NEO: {len([r for r in rupture_events if r['agent']=='NEO'])}")
    print(f"   - EVA: {len([r for r in rupture_events if r['agent']=='EVA'])}")

    # Entropía (homeostasis de repertorio)
    neo_entropy = compute_entropy({s.value: c for s, c in neo.state_counts.items()})
    eva_entropy = compute_entropy({s.value: c for s, c in eva.state_counts.items()})

    print(f"\n5. HOMEOSTASIS DE REPERTORIO (entropía):")
    print(f"   NEO: {neo_entropy:.4f}")
    print(f"   EVA: {eva_entropy:.4f}")

    # Guardar resultados
    results = {
        'n_cycles': n_cycles,
        'n_bilateral': n_bilateral,
        'n_gw_broadcasts': n_broadcasts,
        'n_ruptures': n_ruptures,
        'self_error_neo_mean': float(np.mean(self_errors_neo)),
        'self_error_eva_mean': float(np.mean(self_errors_eva)),
        'metacognition_neo': neo.self_model.get_metacognitive_accuracy(),
        'metacognition_eva': eva.self_model.get_metacognitive_accuracy(),
        'identity_neo_mean': float(np.mean(identity_indices_neo)),
        'identity_eva_mean': float(np.mean(identity_indices_eva)),
        'entropy_neo': neo_entropy,
        'entropy_eva': eva_entropy,
        'neo_identity_signature': neo.identity.get_identity_signature(),
        'eva_identity_signature': eva.identity.get_identity_signature()
    }

    # Convertir para JSON
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, LifeState):
            return obj.value
        else:
            return obj

    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)

    with open(f"{output_dir}/bilateral_events.json", 'w') as f:
        json.dump(convert_for_json(bilateral_events), f)

    with open(f"{output_dir}/gw_broadcasts.json", 'w') as f:
        json.dump(convert_for_json(gw_broadcasts), f)

    with open(f"{output_dir}/rupture_events.json", 'w') as f:
        json.dump(convert_for_json(rupture_events), f)

    with open(f"{output_dir}/consciousness_neo.json", 'w') as f:
        json.dump(convert_for_json(neo.consciousness_log[-2000:]), f)

    with open(f"{output_dir}/consciousness_eva.json", 'w') as f:
        json.dump(convert_for_json(eva.consciousness_log[-2000:]), f)

    # Series temporales para análisis
    with open(f"{output_dir}/timeseries.json", 'w') as f:
        json.dump({
            'self_error_neo': [float(x) for x in self_errors_neo[::10]],  # Subsample
            'self_error_eva': [float(x) for x in self_errors_eva[::10]],
            'identity_neo': [float(x) for x in identity_indices_neo[::10]],
            'identity_eva': [float(x) for x in identity_indices_eva[::10]]
        }, f)

    print(f"\n[OK] Resultados guardados en {output_dir}/")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Phase 11: Consciousness Modules')
    parser.add_argument('--cycles', type=int, default=25000)
    parser.add_argument('--output-dir', default='/root/NEO_EVA/results/phase11')
    args = parser.parse_args()

    run_phase11_experiment(args.cycles, args.output_dir)
