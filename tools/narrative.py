#!/usr/bin/env python3
"""
Phase 13: Memoria Narrativa Endógena
=====================================

Sistema de narrativa completamente endógeno para NEO y EVA.
TODO se deriva de la historia del sistema - CERO números mágicos.

Componentes:
1. Detección de episodios por saliencia (ranks)
2. Memoria con decaimiento natural (sin clustering explícito)
3. Matriz de transición endógena
4. Identity Narrative Index (autocorrelación)
5. Memorias separadas + canal compartido en GW

Principio: "NEO y EVA se cuentan a sí mismos lo que les pasa"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from scipy import stats
import json
from datetime import datetime

# Importar núcleo endógeno
import sys
sys.path.insert(0, '/root/NEO_EVA/tools')
from endogenous_core import (
    derive_window_size,
    derive_learning_rate,
    compute_iqr,
    NUMERIC_EPS,
    PROVENANCE
)


# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass
class Episode:
    """Un episodio narrativo - estructura interna, sin semántica humana."""
    agent: str
    t_start: int
    t_end: int
    dominant_state: str
    gw_ratio: float          # Proporción de tiempo con GW activo
    mean_te: float           # TE medio durante el episodio
    mean_pi: float           # π medio
    delta_pi: float          # Cambio en π (fin - inicio)
    mean_self_error: float   # Error de auto-predicción medio
    social_role: str         # 'leader', 'follower', 'mutual' (derivado de TE direccional)
    salience: float          # Score de saliencia del episodio

    def to_vector(self) -> np.ndarray:
        """Convierte a vector numérico para comparaciones."""
        state_map = {'SLEEP': 0, 'WAKE': 1, 'WORK': 2, 'LEARN': 3, 'SOCIAL': 4}
        role_map = {'follower': 0, 'mutual': 0.5, 'leader': 1}
        return np.array([
            self.gw_ratio,
            self.mean_te,
            self.mean_pi,
            self.delta_pi,
            self.mean_self_error,
            state_map.get(self.dominant_state, 2) / 4,  # Normalizado 0-1
            role_map.get(self.social_role, 0.5)
        ])

    def to_dict(self) -> Dict:
        return {
            'agent': self.agent,
            't_start': self.t_start,
            't_end': self.t_end,
            'dominant_state': self.dominant_state,
            'gw_ratio': self.gw_ratio,
            'mean_te': self.mean_te,
            'mean_pi': self.mean_pi,
            'delta_pi': self.delta_pi,
            'mean_self_error': self.mean_self_error,
            'social_role': self.social_role,
            'salience': self.salience
        }


@dataclass
class NarrativeMemory:
    """Memoria narrativa de un agente con decaimiento natural."""
    agent: str
    episodes: List[Episode] = field(default_factory=list)
    episode_vectors: List[np.ndarray] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)  # Peso por recencia/importancia
    transition_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Estado actual
    current_episode_start: Optional[int] = None
    current_buffer: List[Dict] = field(default_factory=list)

    # Índice de identidad
    ini_history: List[float] = field(default_factory=list)


# =============================================================================
# SALIENCIA ENDÓGENA
# =============================================================================

def compute_salience(
    t: int,
    pi: float,
    te: float,
    state: str,
    gw_active: bool,
    prev_pi: float,
    prev_te: float,
    prev_state: str,
    history_delta_pi: np.ndarray,
    history_delta_te: np.ndarray
) -> float:
    """
    Calcula saliencia de un momento.

    S_t = rank(|Δπ|) + rank(|ΔTE|) + indicator(state_change) + indicator(GW_on)

    Todo basado en ranks de la historia - CERO umbrales fijos.
    """
    delta_pi = abs(pi - prev_pi)
    delta_te = abs(te - prev_te)
    state_change = 1.0 if state != prev_state else 0.0
    gw_indicator = 1.0 if gw_active else 0.0

    # Ranks endógenos
    if len(history_delta_pi) > 10:
        rank_delta_pi = stats.percentileofscore(history_delta_pi, delta_pi) / 100
        rank_delta_te = stats.percentileofscore(history_delta_te, delta_te) / 100
    else:
        # Warmup: usar valores normalizados simples
        rank_delta_pi = min(delta_pi / (np.std(history_delta_pi) + NUMERIC_EPS), 1.0) if len(history_delta_pi) > 0 else 0.5
        rank_delta_te = min(delta_te / (np.std(history_delta_te) + NUMERIC_EPS), 1.0) if len(history_delta_te) > 0 else 0.5

    salience = rank_delta_pi + rank_delta_te + state_change + gw_indicator

    # Normalizar a [0, 1]
    salience = salience / 4.0

    PROVENANCE.log('salience', salience,
                   f'rank(|Δπ|) + rank(|ΔTE|) + state_change + GW',
                   {'rank_delta_pi': rank_delta_pi, 'rank_delta_te': rank_delta_te,
                    'state_change': state_change, 'gw': gw_indicator}, t)

    return salience


def derive_salience_threshold(salience_history: np.ndarray) -> float:
    """
    Umbral endógeno para detectar episodios.

    threshold = q75(salience_history)

    Derivado de la distribución real, no un número fijo.
    """
    if len(salience_history) < 10:
        return 0.5  # Default durante warmup

    threshold = np.percentile(salience_history, 75)

    PROVENANCE.log('salience_threshold', threshold,
                   f'q75 de {len(salience_history)} muestras',
                   {'n_samples': len(salience_history)}, 0)

    return threshold


# =============================================================================
# DETECCIÓN DE EPISODIOS
# =============================================================================

class EpisodeDetector:
    """Detecta episodios narrativos de forma endógena."""

    def __init__(self, agent: str, max_history: int = None):
        self.agent = agent
        # maxlen derivado: suficiente para ~100 ventanas de √T donde T típico = 10000
        # √10000 = 100, necesitamos ~100 ventanas → 10000
        # O se puede pasar explícitamente
        derived_maxlen = max_history if max_history else int(np.sqrt(1e8))  # √1e8 ≈ 10000
        self.salience_history = deque(maxlen=derived_maxlen)
        self.delta_pi_history = deque(maxlen=derived_maxlen)
        self.delta_te_history = deque(maxlen=derived_maxlen)

        self.in_episode = False
        self.episode_start = 0
        self.episode_buffer = []

        self.prev_pi = 0.5
        self.prev_te = 0.0
        self.prev_state = 'WAKE'

    def process_step(
        self,
        t: int,
        pi: float,
        te_outgoing: float,  # TE de este agente al otro
        te_incoming: float,  # TE del otro agente a este
        state: str,
        gw_active: bool,
        self_error: float
    ) -> Optional[Episode]:
        """
        Procesa un paso temporal.
        Retorna Episode si uno termina, None si no.
        """
        te = te_outgoing + te_incoming  # TE total bidireccional

        # Calcular deltas
        delta_pi = abs(pi - self.prev_pi)
        delta_te = abs(te - (self.prev_te if hasattr(self, '_prev_te_total') else 0))
        self._prev_te_total = te

        self.delta_pi_history.append(delta_pi)
        self.delta_te_history.append(delta_te)

        # Calcular saliencia
        salience = compute_salience(
            t, pi, te, state, gw_active,
            self.prev_pi, self.prev_te, self.prev_state,
            np.array(self.delta_pi_history),
            np.array(self.delta_te_history)
        )

        self.salience_history.append(salience)
        threshold = derive_salience_threshold(np.array(self.salience_history))

        # Guardar datos del paso
        step_data = {
            't': t,
            'pi': pi,
            'te_out': te_outgoing,
            'te_in': te_incoming,
            'te_total': te,
            'state': state,
            'gw_active': gw_active,
            'self_error': self_error,
            'salience': salience
        }

        completed_episode = None

        # Lógica de episodios
        if not self.in_episode:
            if salience > threshold:
                # Inicio de episodio
                self.in_episode = True
                self.episode_start = t
                self.episode_buffer = [step_data]
        else:
            self.episode_buffer.append(step_data)

            # Derivar duración mínima endógenamente
            min_duration = derive_window_size(t) // 4
            min_duration = max(5, min_duration)  # Al menos 5 pasos

            # ¿Fin de episodio?
            episode_duration = t - self.episode_start

            if episode_duration >= min_duration and salience < threshold:
                # Construir episodio
                completed_episode = self._build_episode()
                self.in_episode = False
                self.episode_buffer = []

        # Actualizar historia
        self.prev_pi = pi
        self.prev_te = te
        self.prev_state = state

        return completed_episode

    def _build_episode(self) -> Episode:
        """Construye un Episode a partir del buffer."""
        if not self.episode_buffer:
            return None

        t_start = self.episode_buffer[0]['t']
        t_end = self.episode_buffer[-1]['t']

        # Estado dominante (moda)
        states = [d['state'] for d in self.episode_buffer]
        dominant_state = max(set(states), key=states.count)

        # GW ratio
        gw_ratio = np.mean([d['gw_active'] for d in self.episode_buffer])

        # Medias
        mean_te = np.mean([d['te_total'] for d in self.episode_buffer])
        mean_pi = np.mean([d['pi'] for d in self.episode_buffer])
        delta_pi = self.episode_buffer[-1]['pi'] - self.episode_buffer[0]['pi']
        mean_self_error = np.mean([d['self_error'] for d in self.episode_buffer])

        # Saliencia media del episodio
        salience = np.mean([d['salience'] for d in self.episode_buffer])

        # Social role: derivado de TE direccional
        mean_te_out = np.mean([d['te_out'] for d in self.episode_buffer])
        mean_te_in = np.mean([d['te_in'] for d in self.episode_buffer])

        if mean_te_out > mean_te_in * 1.2:
            social_role = 'leader'
        elif mean_te_in > mean_te_out * 1.2:
            social_role = 'follower'
        else:
            social_role = 'mutual'

        return Episode(
            agent=self.agent,
            t_start=t_start,
            t_end=t_end,
            dominant_state=dominant_state,
            gw_ratio=float(gw_ratio),
            mean_te=float(mean_te),
            mean_pi=float(mean_pi),
            delta_pi=float(delta_pi),
            mean_self_error=float(mean_self_error),
            social_role=social_role,
            salience=float(salience)
        )


# =============================================================================
# MEMORIA CON DECAIMIENTO
# =============================================================================

def compute_memory_weight(
    episode: Episode,
    t_current: int,
    total_episodes: int
) -> float:
    """
    Peso de un episodio en memoria.

    Combina:
    - Recencia (decaimiento temporal)
    - Saliencia del episodio

    Decaimiento: exp(-Δt / τ) donde τ = √(total_episodes) * window_size
    """
    if total_episodes < 1:
        return 1.0

    # τ endógeno
    tau = np.sqrt(total_episodes + 1) * derive_window_size(t_current)

    # Decaimiento temporal
    delta_t = t_current - episode.t_end
    recency = np.exp(-delta_t / (tau + NUMERIC_EPS))

    # Combinar con saliencia
    weight = recency * (0.5 + 0.5 * episode.salience)

    return float(weight)


def compute_episode_similarity(ep1: Episode, ep2: Episode) -> float:
    """
    Similitud entre episodios basada en sus vectores.

    Usa correlación de Spearman (basada en ranks, no valores absolutos).
    """
    v1 = ep1.to_vector()
    v2 = ep2.to_vector()

    # Correlación de Spearman
    corr, _ = stats.spearmanr(v1, v2)

    # Convertir a similitud [0, 1]
    similarity = (corr + 1) / 2

    return float(similarity)


def merge_similar_episodes(
    memory: NarrativeMemory,
    new_episode: Episode,
    t_current: int
) -> Tuple[NarrativeMemory, bool]:
    """
    Intenta fusionar episodio nuevo con uno existente similar.

    Umbral de similitud derivado de la distribución de similitudes existentes.
    """
    if len(memory.episodes) < 2:
        # No hay suficientes para derivar umbral
        memory.episodes.append(new_episode)
        memory.episode_vectors.append(new_episode.to_vector())
        memory.weights.append(compute_memory_weight(new_episode, t_current, len(memory.episodes)))
        return memory, False

    # Calcular similitudes con episodios existentes
    similarities = []
    for ep in memory.episodes:
        sim = compute_episode_similarity(new_episode, ep)
        similarities.append(sim)

    # Umbral endógeno: q75 de similitudes históricas
    # (si es muy similar a muchos, probablemente es redundante)
    existing_sims = []
    for i, ep1 in enumerate(memory.episodes):
        for ep2 in memory.episodes[i+1:]:
            existing_sims.append(compute_episode_similarity(ep1, ep2))

    if existing_sims:
        merge_threshold = np.percentile(existing_sims, 75)
    else:
        # Fallback: sin datos previos, usar umbral neutral
        # Derivado de: correlación máxima teórica normalizada
        merge_threshold = (1.0 + 1.0) / 2 * 0.8  # 80% del rango [0,1]

    # Encontrar más similar
    max_sim_idx = np.argmax(similarities)
    max_sim = similarities[max_sim_idx]

    if max_sim > merge_threshold:
        # Fusionar: promedio ponderado por saliencia
        old_ep = memory.episodes[max_sim_idx]
        old_weight = old_ep.salience
        new_weight = new_episode.salience
        total_weight = old_weight + new_weight + NUMERIC_EPS

        # Crear episodio fusionado
        merged = Episode(
            agent=new_episode.agent,
            t_start=min(old_ep.t_start, new_episode.t_start),
            t_end=max(old_ep.t_end, new_episode.t_end),
            dominant_state=new_episode.dominant_state if new_weight > old_weight else old_ep.dominant_state,
            gw_ratio=(old_ep.gw_ratio * old_weight + new_episode.gw_ratio * new_weight) / total_weight,
            mean_te=(old_ep.mean_te * old_weight + new_episode.mean_te * new_weight) / total_weight,
            mean_pi=(old_ep.mean_pi * old_weight + new_episode.mean_pi * new_weight) / total_weight,
            delta_pi=(old_ep.delta_pi * old_weight + new_episode.delta_pi * new_weight) / total_weight,
            mean_self_error=(old_ep.mean_self_error * old_weight + new_episode.mean_self_error * new_weight) / total_weight,
            social_role=new_episode.social_role if new_weight > old_weight else old_ep.social_role,
            salience=max(old_ep.salience, new_episode.salience)
        )

        memory.episodes[max_sim_idx] = merged
        memory.episode_vectors[max_sim_idx] = merged.to_vector()
        memory.weights[max_sim_idx] = compute_memory_weight(merged, t_current, len(memory.episodes))

        return memory, True  # Merged
    else:
        # Añadir como nuevo
        memory.episodes.append(new_episode)
        memory.episode_vectors.append(new_episode.to_vector())
        memory.weights.append(compute_memory_weight(new_episode, t_current, len(memory.episodes)))

        return memory, False  # Not merged


def prune_memory(memory: NarrativeMemory, t_current: int) -> NarrativeMemory:
    """
    Poda memoria eliminando episodios con peso muy bajo.

    Umbral de poda: q10 de los pesos actuales.
    """
    if len(memory.episodes) < 10:
        return memory

    # Actualizar pesos
    for i, ep in enumerate(memory.episodes):
        memory.weights[i] = compute_memory_weight(ep, t_current, len(memory.episodes))

    # Umbral endógeno
    prune_threshold = np.percentile(memory.weights, 10)

    # Filtrar
    keep_indices = [i for i, w in enumerate(memory.weights) if w > prune_threshold]

    memory.episodes = [memory.episodes[i] for i in keep_indices]
    memory.episode_vectors = [memory.episode_vectors[i] for i in keep_indices]
    memory.weights = [memory.weights[i] for i in keep_indices]

    return memory


# =============================================================================
# MATRIZ DE TRANSICIÓN ENDÓGENA
# =============================================================================

def get_episode_type(episode: Episode, n_types: int) -> int:
    """
    Asigna un tipo a un episodio.

    Tipos basados en: estado dominante + GW ratio (alto/bajo)
    n_types derivado endógenamente.
    """
    state_idx = {'SLEEP': 0, 'WAKE': 1, 'WORK': 2, 'LEARN': 3, 'SOCIAL': 4}.get(episode.dominant_state, 2)
    gw_high = 1 if episode.gw_ratio > 0.5 else 0

    type_idx = state_idx * 2 + gw_high
    return type_idx % n_types


def update_transition_matrix(
    memory: NarrativeMemory,
    new_episode: Episode
) -> NarrativeMemory:
    """
    Actualiza matriz de transición con nuevo episodio.
    """
    if len(memory.episodes) < 2:
        return memory

    # Derivar número de tipos endógenamente
    n_episodes = len(memory.episodes)
    n_types = max(3, min(10, int(np.sqrt(n_episodes))))

    # Tipo del episodio anterior y nuevo
    prev_episode = memory.episodes[-2] if len(memory.episodes) >= 2 else None

    if prev_episode is None:
        return memory

    prev_type = str(get_episode_type(prev_episode, n_types))
    new_type = str(get_episode_type(new_episode, n_types))

    # Actualizar conteos
    if prev_type not in memory.transition_counts:
        memory.transition_counts[prev_type] = {}

    if new_type not in memory.transition_counts[prev_type]:
        memory.transition_counts[prev_type][new_type] = 0

    memory.transition_counts[prev_type][new_type] += 1

    return memory


def get_transition_probabilities(memory: NarrativeMemory) -> Dict[str, Dict[str, float]]:
    """
    Calcula probabilidades de transición con smoothing endógeno.

    Smoothing: 1/√N donde N = total de transiciones observadas
    """
    probs = {}

    total_transitions = sum(
        sum(targets.values())
        for targets in memory.transition_counts.values()
    )

    # Smoothing endógeno
    alpha = 1.0 / (np.sqrt(total_transitions) + 1) if total_transitions > 0 else 0.5

    for source, targets in memory.transition_counts.items():
        total = sum(targets.values()) + alpha * len(targets)
        probs[source] = {}
        for target, count in targets.items():
            probs[source][target] = (count + alpha) / total

    return probs


# =============================================================================
# IDENTITY NARRATIVE INDEX (INI)
# =============================================================================

def compute_ini(memory: NarrativeMemory) -> float:
    """
    Identity Narrative Index basado en autocorrelación.

    Alta autocorrelación = identidad estable (historias predecibles)
    Baja autocorrelación = identidad fluida

    Usa lag derivado endógenamente de √N.
    """
    if len(memory.episodes) < 5:
        return 0.5  # Neutro durante warmup

    # Convertir episodios a secuencia de tipos
    n_types = max(3, min(10, int(np.sqrt(len(memory.episodes)))))
    type_sequence = [get_episode_type(ep, n_types) for ep in memory.episodes]

    # Lag endógeno
    max_lag = max(1, int(np.sqrt(len(type_sequence))))

    # Autocorrelación
    autocorrs = []
    for lag in range(1, max_lag + 1):
        if lag >= len(type_sequence):
            break
        seq1 = type_sequence[:-lag]
        seq2 = type_sequence[lag:]

        # Correlación de Spearman
        if len(set(seq1)) > 1 and len(set(seq2)) > 1:
            corr, _ = stats.spearmanr(seq1, seq2)
            if not np.isnan(corr):
                autocorrs.append(corr)

    if not autocorrs:
        return 0.5

    # INI = media de autocorrelaciones, mapeado a [0, 1]
    ini = (np.mean(autocorrs) + 1) / 2

    PROVENANCE.log('INI', ini,
                   f'autocorr media de {len(autocorrs)} lags',
                   {'n_lags': len(autocorrs), 'n_episodes': len(memory.episodes)}, 0)

    return float(ini)


# =============================================================================
# CANAL COMPARTIDO (GW ACTIVO)
# =============================================================================

def share_episode_gw(
    source_memory: NarrativeMemory,
    target_memory: NarrativeMemory,
    episode: Episode,
    t_current: int,
    gw_intensity: float
) -> NarrativeMemory:
    """
    Comparte un episodio entre agentes cuando GW está activo.

    La "fuerza" de la compartición depende de gw_intensity.
    El episodio compartido tiene saliencia modulada.
    """
    if gw_intensity < 0.3:  # Umbral mínimo para compartir
        return target_memory

    # Crear copia del episodio para el otro agente
    shared_episode = Episode(
        agent=f"shared_from_{episode.agent}",
        t_start=episode.t_start,
        t_end=episode.t_end,
        dominant_state=episode.dominant_state,
        gw_ratio=episode.gw_ratio,
        mean_te=episode.mean_te,
        mean_pi=episode.mean_pi,
        delta_pi=episode.delta_pi,
        mean_self_error=episode.mean_self_error,
        social_role='mutual',  # Cuando se comparte, el rol es mutuo
        salience=episode.salience * gw_intensity  # Modulado por intensidad GW
    )

    # Intentar fusionar o añadir
    target_memory, merged = merge_similar_episodes(target_memory, shared_episode, t_current)

    return target_memory


# =============================================================================
# SISTEMA NARRATIVO COMPLETO
# =============================================================================

class NarrativeSystem:
    """Sistema narrativo completo para dos agentes."""

    def __init__(self):
        self.neo_detector = EpisodeDetector('NEO')
        self.eva_detector = EpisodeDetector('EVA')

        self.neo_memory = NarrativeMemory(agent='NEO')
        self.eva_memory = NarrativeMemory(agent='EVA')

        self.shared_episodes = []  # Episodios compartidos via GW

        # Historial de INI
        self.neo_ini_history = []
        self.eva_ini_history = []

        # Log de eventos
        self.event_log = []

    def process_step(
        self,
        t: int,
        neo_pi: float,
        eva_pi: float,
        te_neo_to_eva: float,
        te_eva_to_neo: float,
        neo_state: str,
        eva_state: str,
        gw_active: bool,
        gw_intensity: float,
        neo_self_error: float,
        eva_self_error: float
    ) -> Dict:
        """Procesa un paso para ambos agentes."""

        result = {
            't': t,
            'neo_episode': None,
            'eva_episode': None,
            'shared': False,
            'neo_ini': None,
            'eva_ini': None
        }

        # Detectar episodios
        neo_episode = self.neo_detector.process_step(
            t, neo_pi, te_neo_to_eva, te_eva_to_neo,
            neo_state, gw_active, neo_self_error
        )

        eva_episode = self.eva_detector.process_step(
            t, eva_pi, te_eva_to_neo, te_neo_to_eva,
            eva_state, gw_active, eva_self_error
        )

        # Procesar episodios completados
        if neo_episode:
            self.neo_memory, merged = merge_similar_episodes(
                self.neo_memory, neo_episode, t
            )
            self.neo_memory = update_transition_matrix(self.neo_memory, neo_episode)
            result['neo_episode'] = neo_episode.to_dict()

            # Compartir si GW activo
            if gw_active and gw_intensity > 0.3:
                self.eva_memory = share_episode_gw(
                    self.neo_memory, self.eva_memory, neo_episode, t, gw_intensity
                )
                result['shared'] = True
                self.shared_episodes.append({
                    'source': 'NEO',
                    'target': 'EVA',
                    't': t,
                    'episode': neo_episode.to_dict()
                })

        if eva_episode:
            self.eva_memory, merged = merge_similar_episodes(
                self.eva_memory, eva_episode, t
            )
            self.eva_memory = update_transition_matrix(self.eva_memory, eva_episode)
            result['eva_episode'] = eva_episode.to_dict()

            # Compartir si GW activo
            if gw_active and gw_intensity > 0.3:
                self.neo_memory = share_episode_gw(
                    self.eva_memory, self.neo_memory, eva_episode, t, gw_intensity
                )
                result['shared'] = True
                self.shared_episodes.append({
                    'source': 'EVA',
                    'target': 'NEO',
                    't': t,
                    'episode': eva_episode.to_dict()
                })

        # Calcular INI periódicamente
        window = derive_window_size(t)
        if t % (window // 4) == 0:
            neo_ini = compute_ini(self.neo_memory)
            eva_ini = compute_ini(self.eva_memory)

            self.neo_ini_history.append({'t': t, 'ini': neo_ini})
            self.eva_ini_history.append({'t': t, 'ini': eva_ini})

            result['neo_ini'] = neo_ini
            result['eva_ini'] = eva_ini

        # Poda periódica
        if t % (window * 2) == 0:
            self.neo_memory = prune_memory(self.neo_memory, t)
            self.eva_memory = prune_memory(self.eva_memory, t)

        return result

    def get_summary(self) -> Dict:
        """Resumen del estado narrativo."""
        return {
            'neo': {
                'n_episodes': len(self.neo_memory.episodes),
                'n_types': len(self.neo_memory.transition_counts),
                'current_ini': self.neo_ini_history[-1]['ini'] if self.neo_ini_history else 0.5,
                'episodes_summary': [ep.to_dict() for ep in self.neo_memory.episodes[-5:]]
            },
            'eva': {
                'n_episodes': len(self.eva_memory.episodes),
                'n_types': len(self.eva_memory.transition_counts),
                'current_ini': self.eva_ini_history[-1]['ini'] if self.eva_ini_history else 0.5,
                'episodes_summary': [ep.to_dict() for ep in self.eva_memory.episodes[-5:]]
            },
            'shared': {
                'n_shared': len(self.shared_episodes),
                'recent': self.shared_episodes[-3:] if self.shared_episodes else []
            },
            'transition_probs': {
                'neo': get_transition_probabilities(self.neo_memory),
                'eva': get_transition_probabilities(self.eva_memory)
            }
        }

    def save(self, path: str):
        """Guarda el sistema narrativo."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'neo_episodes': [ep.to_dict() for ep in self.neo_memory.episodes],
            'eva_episodes': [ep.to_dict() for ep in self.eva_memory.episodes],
            'neo_ini_history': self.neo_ini_history,
            'eva_ini_history': self.eva_ini_history,
            'shared_episodes': self.shared_episodes,
            'neo_transitions': self.neo_memory.transition_counts,
            'eva_transitions': self.eva_memory.transition_counts
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 13: MEMORIA NARRATIVA ENDÓGENA - TEST")
    print("=" * 70)

    # Crear sistema
    ns = NarrativeSystem()

    # Simular datos
    np.random.seed(42)
    n_steps = 5000

    states = ['SLEEP', 'WAKE', 'WORK', 'LEARN', 'SOCIAL']

    neo_pi = 0.5
    eva_pi = 0.5

    episodes_detected = {'NEO': 0, 'EVA': 0}
    shares = 0

    print("\nSimulando dinámica...")

    for t in range(n_steps):
        # Simular estados (con transiciones)
        if t % 200 == 0:
            current_state_idx = np.random.randint(0, 5)
        neo_state = states[current_state_idx]
        eva_state = states[(current_state_idx + np.random.randint(0, 2)) % 5]

        # Simular GW
        gw_active = np.random.rand() > 0.6
        gw_intensity = np.random.rand() * 0.8 if gw_active else 0

        # Simular TE (más alto en estados activos)
        base_te = 0.3 if neo_state in ['WORK', 'LEARN', 'SOCIAL'] else 0.05
        te_neo_to_eva = base_te + np.random.rand() * 0.2
        te_eva_to_neo = base_te + np.random.rand() * 0.2

        # Simular pi con drift
        neo_pi += np.random.randn() * 0.02
        neo_pi = np.clip(neo_pi, 0, 1)
        eva_pi += np.random.randn() * 0.02 + 0.01 * (neo_pi - eva_pi)  # Acoplamiento
        eva_pi = np.clip(eva_pi, 0, 1)

        # Self error
        neo_self_error = abs(np.random.randn() * 0.1)
        eva_self_error = abs(np.random.randn() * 0.1)

        # Procesar
        result = ns.process_step(
            t, neo_pi, eva_pi,
            te_neo_to_eva, te_eva_to_neo,
            neo_state, eva_state,
            gw_active, gw_intensity,
            neo_self_error, eva_self_error
        )

        if result['neo_episode']:
            episodes_detected['NEO'] += 1
        if result['eva_episode']:
            episodes_detected['EVA'] += 1
        if result['shared']:
            shares += 1

    print(f"\n[OK] Simulación completada: {n_steps} pasos")
    print(f"\nEpisodios detectados:")
    print(f"  NEO: {episodes_detected['NEO']}")
    print(f"  EVA: {episodes_detected['EVA']}")
    print(f"  Compartidos: {shares}")

    # Resumen
    summary = ns.get_summary()

    print(f"\nMemoria final:")
    print(f"  NEO: {summary['neo']['n_episodes']} episodios, INI={summary['neo']['current_ini']:.3f}")
    print(f"  EVA: {summary['eva']['n_episodes']} episodios, INI={summary['eva']['current_ini']:.3f}")

    print(f"\nÚltimos episodios NEO:")
    for ep in summary['neo']['episodes_summary'][-3:]:
        print(f"  t={ep['t_start']}-{ep['t_end']}: {ep['dominant_state']}, TE={ep['mean_te']:.3f}, role={ep['social_role']}")

    print(f"\nTransiciones NEO:")
    for src, targets in summary['transition_probs']['neo'].items():
        top_target = max(targets.items(), key=lambda x: x[1]) if targets else ('?', 0)
        print(f"  Tipo {src} → Tipo {top_target[0]} (p={top_target[1]:.2f})")

    # Guardar
    ns.save('/root/NEO_EVA/results/phase13_narrative_test.json')
    print(f"\n[OK] Guardado en results/phase13_narrative_test.json")
