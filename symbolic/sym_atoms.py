"""
Symbolic Atoms: Definición de símbolo estructural endógeno
==========================================================

Un símbolo es una clase de equivalencia sobre episodios/estados/acciones que:
- Se activa consistentemente en ciertos contextos
- Tiene consecuencias internas predecibles
- Se usa como átomo de narración, planificación y coordinación

Todo endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from sklearn.cluster import KMeans
from collections import defaultdict

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import (
    L_t, max_history, compute_adaptive_percentile, to_simplex, normalized_entropy
)


@dataclass
class SymbolStats:
    """Estadísticos estructurales de un símbolo."""
    mu: np.ndarray                    # Centro en espacio s_t
    Sigma: np.ndarray                 # Covarianza interna
    gamma: np.ndarray                 # Firma de consecuencia media
    stab: float                       # Estabilidad interna [0,1]
    consistency: float                # Consistencia de consecuencias [0,1]
    sym_score: float                  # SymScore = stab * consistency
    n_episodes: int                   # Tamaño del cluster
    last_update_t: int                # Último tiempo de actualización


@dataclass
class Symbol:
    """Símbolo estructural endógeno."""
    symbol_id: int
    agent_id: str
    stats: SymbolStats
    context_hist: List[int] = field(default_factory=list)     # Regímenes, modos, etc.
    world_hist: List[Any] = field(default_factory=list)       # Índices WORLD-1 relevantes
    value_hist: List[float] = field(default_factory=list)     # ΔV, ΔSAGI, etc.
    episode_ids: List[int] = field(default_factory=list)      # IDs de episodios asignados
    created_t: int = 0

    def is_valid(self, t: int, scores_history: np.ndarray) -> bool:
        """
        Decide si el símbolo sigue siendo válido usando:
        - stats.sym_score frente a percentiles históricos
        - n_episodes >= L_t(t)
        """
        if self.stats.n_episodes < L_t(t):
            return False

        if len(scores_history) < 3:
            return self.stats.sym_score > 0.25

        # Umbral endógeno: percentil 25 de scores históricos
        threshold = np.percentile(scores_history, 25)
        return self.stats.sym_score >= threshold

    def distance(self, s: np.ndarray, global_p95_dist: float) -> float:
        """Distancia estructural d_s(s, mu) normalizada por historial global."""
        raw_dist = np.linalg.norm(s - self.stats.mu)
        return raw_dist / (global_p95_dist + 1e-8)

    def expected_consequence(self) -> np.ndarray:
        """Devuelve gamma (firma media de consecuencia)."""
        return self.stats.gamma.copy()

    def update_from_episodes(
        self,
        states: List[np.ndarray],
        deltas: List[np.ndarray],
        global_p95_dist: float,
        global_p95_delta: float,
        t: int,
    ) -> None:
        """
        Recalcula mu, Sigma, gamma, stab, consistency, sym_score de forma online.
        """
        if not states:
            return

        states_arr = np.array(states)
        deltas_arr = np.array(deltas) if deltas else np.zeros_like(states_arr)

        # Centro
        self.stats.mu = np.mean(states_arr, axis=0)

        # Covarianza
        if len(states) >= 2:
            self.stats.Sigma = np.cov(states_arr.T) if states_arr.shape[0] > 1 else np.eye(states_arr.shape[1]) * 0.01
        else:
            self.stats.Sigma = np.eye(len(self.stats.mu)) * 0.01

        # Firma de consecuencia
        self.stats.gamma = np.mean(deltas_arr, axis=0) if len(deltas) > 0 else np.zeros_like(self.stats.mu)

        # Estabilidad: 1 - E[d(s_i, mu)] / p95(d)
        distances = [np.linalg.norm(s - self.stats.mu) for s in states]
        mean_dist = np.mean(distances)
        self.stats.stab = float(1.0 - mean_dist / (global_p95_dist + 1e-8))
        self.stats.stab = np.clip(self.stats.stab, 0, 1)

        # Consistencia: 1 - E[||Δs_i - gamma||] / p95(||Δs||)
        if deltas:
            delta_diffs = [np.linalg.norm(d - self.stats.gamma) for d in deltas]
            mean_delta_diff = np.mean(delta_diffs)
            self.stats.consistency = float(1.0 - mean_delta_diff / (global_p95_delta + 1e-8))
            self.stats.consistency = np.clip(self.stats.consistency, 0, 1)
        else:
            self.stats.consistency = 0.5

        # SymScore
        self.stats.sym_score = self.stats.stab * self.stats.consistency
        self.stats.n_episodes = len(states)
        self.stats.last_update_t = t


class SymbolExtractor:
    """
    Construye y actualiza símbolos a partir de episodios y estados internos.
    Se usa por agente.
    """

    def __init__(self, agent_id: str, state_dim: int):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.symbols: Dict[int, Symbol] = {}
        self.next_id: int = 0

        # Históricos globales para percentiles y normalización
        self.global_dists: List[float] = []
        self.global_deltas: List[float] = []
        self.global_scores: List[float] = []

        # Estados y deltas por tiempo
        self.state_history: Dict[int, np.ndarray] = {}  # t -> s_t
        self.delta_history: Dict[int, np.ndarray] = {}  # t -> Δs_t
        self.context_history: Dict[int, int] = {}       # t -> context

        self.t = 0

    def build_state_vector(
        self,
        z_t: np.ndarray,
        phi_t: np.ndarray,
        drives_t: np.ndarray
    ) -> np.ndarray:
        """Concatena z_t, φ_t, drives_t → s_t."""
        return np.concatenate([z_t.flatten(), phi_t.flatten(), drives_t.flatten()])

    def record_state(
        self,
        t: int,
        z_t: np.ndarray,
        phi_t: np.ndarray,
        drives_t: np.ndarray,
        context: int = 0
    ) -> np.ndarray:
        """Registra un estado y calcula delta si hay estado previo."""
        s_t = self.build_state_vector(z_t, phi_t, drives_t)

        self.state_history[t] = s_t.copy()
        self.context_history[t] = context

        # Calcular delta
        if t - 1 in self.state_history:
            delta = s_t - self.state_history[t - 1]
            self.delta_history[t] = delta

            # Actualizar históricos globales
            self.global_dists.append(float(np.linalg.norm(s_t)))
            self.global_deltas.append(float(np.linalg.norm(delta)))

        # Limitar históricos
        max_hist = max_history(t)
        if len(self.global_dists) > max_hist:
            self.global_dists = self.global_dists[-max_hist:]
            self.global_deltas = self.global_deltas[-max_hist:]

        self.t = t
        return s_t

    def observe_state(
        self,
        t: int,
        state: np.ndarray,
        consequence: Optional[np.ndarray] = None,
        context: int = 0
    ) -> np.ndarray:
        """
        Simplified API for observing a state directly.
        Splits state into approximate z_t, phi_t, drives_t.
        """
        state = np.asarray(state).flatten()
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]

        # Split state into three parts (roughly equal)
        third = self.state_dim // 3
        z_t = state[:third]
        phi_t = state[third:2*third]
        drives_t = state[2*third:]

        s_t = self.record_state(t, z_t, phi_t, drives_t, context)

        # Store consequence if provided
        if consequence is not None and t in self.state_history:
            self.delta_history[t] = np.asarray(consequence).flatten()[:self.state_dim]

        return s_t

    def extract_symbols(
        self,
        t: int,
        min_episodes: Optional[int] = None
    ) -> Dict[int, 'Symbol']:
        """
        Extrae símbolos mediante clustering de estados recientes.

        Returns:
            Lista de símbolos extraídos/actualizados
        """
        if min_episodes is None:
            min_episodes = L_t(t)

        # Obtener estados recientes
        window = max_history(t)
        recent_times = sorted([tt for tt in self.state_history.keys() if tt >= t - window])

        if len(recent_times) < min_episodes:
            return list(self.symbols.values())

        # Preparar datos para clustering
        states = [self.state_history[tt] for tt in recent_times]
        deltas = [self.delta_history.get(tt, np.zeros(self.state_dim * 3)) for tt in recent_times]
        contexts = [self.context_history.get(tt, 0) for tt in recent_times]

        states_arr = np.array(states)

        # Número de clusters endógeno
        n_clusters = max(2, min(len(states) // min_episodes, int(np.sqrt(len(states)))))

        # Clustering
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(states_arr)
        except:
            return list(self.symbols.values())

        # Percentiles globales
        p95_dist = np.percentile(self.global_dists, 95) if self.global_dists else 1.0
        p95_delta = np.percentile(self.global_deltas, 95) if self.global_deltas else 1.0

        # Crear/actualizar símbolos por cluster
        new_symbols = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = [i for i, m in enumerate(cluster_mask) if m]

            if len(cluster_indices) < min_episodes:
                continue

            cluster_states = [states[i] for i in cluster_indices]
            cluster_deltas = [deltas[i] for i in cluster_indices]
            cluster_contexts = [contexts[i] for i in cluster_indices]
            cluster_times = [recent_times[i] for i in cluster_indices]

            # Buscar símbolo existente cercano
            cluster_center = np.mean(cluster_states, axis=0)
            existing_symbol = self._find_nearest_symbol(cluster_center, threshold=p95_dist * 0.5)

            if existing_symbol is not None:
                # Actualizar símbolo existente
                existing_symbol.update_from_episodes(
                    cluster_states, cluster_deltas, p95_dist, p95_delta, t
                )
                existing_symbol.context_hist.extend(cluster_contexts)
                existing_symbol.episode_ids.extend(cluster_times)

                # Limitar históricos del símbolo
                max_h = max_history(t)
                if len(existing_symbol.context_hist) > max_h:
                    existing_symbol.context_hist = existing_symbol.context_hist[-max_h:]
                if len(existing_symbol.episode_ids) > max_h:
                    existing_symbol.episode_ids = existing_symbol.episode_ids[-max_h:]

                new_symbols.append(existing_symbol)
            else:
                # Crear nuevo símbolo
                stats = SymbolStats(
                    mu=cluster_center,
                    Sigma=np.eye(len(cluster_center)) * 0.01,
                    gamma=np.zeros(len(cluster_center)),
                    stab=0.5,
                    consistency=0.5,
                    sym_score=0.25,
                    n_episodes=len(cluster_states),
                    last_update_t=t
                )

                new_symbol = Symbol(
                    symbol_id=self.next_id,
                    agent_id=self.agent_id,
                    stats=stats,
                    context_hist=cluster_contexts,
                    episode_ids=cluster_times,
                    created_t=t
                )

                new_symbol.update_from_episodes(
                    cluster_states, cluster_deltas, p95_dist, p95_delta, t
                )

                self.symbols[self.next_id] = new_symbol
                self.next_id += 1
                new_symbols.append(new_symbol)

        # Actualizar scores globales
        for sym in self.symbols.values():
            self.global_scores.append(sym.stats.sym_score)

        if len(self.global_scores) > max_history(t):
            self.global_scores = self.global_scores[-max_history(t):]

        return self.symbols

    def get_symbols(self) -> Dict[int, 'Symbol']:
        """Returns the current symbol dictionary."""
        return self.symbols

    def _find_nearest_symbol(
        self,
        state: np.ndarray,
        threshold: float
    ) -> Optional[Symbol]:
        """Encuentra el símbolo más cercano dentro del umbral."""
        best_symbol = None
        best_dist = threshold

        for sym in self.symbols.values():
            dist = np.linalg.norm(state - sym.stats.mu)
            if dist < best_dist:
                best_dist = dist
                best_symbol = sym

        return best_symbol

    def assign_state_to_symbol(self, state: np.ndarray) -> Optional[int]:
        """Asigna un estado al símbolo más cercano válido."""
        if not self.symbols:
            return None

        p95_dist = np.percentile(self.global_dists, 95) if self.global_dists else 1.0
        scores_arr = np.array(self.global_scores) if self.global_scores else np.array([0.5])

        best_symbol_id = None
        best_score = -np.inf

        for sym_id, sym in self.symbols.items():
            if not sym.is_valid(self.t, scores_arr):
                continue

            dist = sym.distance(state, p95_dist)
            # Score: combina cercanía y calidad del símbolo
            score = sym.stats.sym_score / (1 + dist)

            if score > best_score:
                best_score = score
                best_symbol_id = sym_id

        return best_symbol_id

    def prune_symbols(self, t: int) -> int:
        """Elimina símbolos obsoletos. Retorna número eliminado."""
        scores_arr = np.array(self.global_scores) if self.global_scores else np.array([0.5])

        to_remove = []
        for sym_id, sym in self.symbols.items():
            # Eliminar si no válido o muy viejo sin uso
            if not sym.is_valid(t, scores_arr):
                to_remove.append(sym_id)
            elif t - sym.stats.last_update_t > max_history(t):
                to_remove.append(sym_id)

        for sym_id in to_remove:
            del self.symbols[sym_id]

        return len(to_remove)

    def get_active_symbols(self, t: int) -> List[Symbol]:
        """Devuelve símbolos activos en t."""
        scores_arr = np.array(self.global_scores) if self.global_scores else np.array([0.5])
        return [sym for sym in self.symbols.values() if sym.is_valid(t, scores_arr)]

    def get_symbol_by_id(self, symbol_id: int) -> Optional[Symbol]:
        """Obtiene símbolo por ID."""
        return self.symbols.get(symbol_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Estadísticas del extractor."""
        active = self.get_active_symbols(self.t)
        return {
            'agent_id': self.agent_id,
            't': self.t,
            'total_symbols': len(self.symbols),
            'active_symbols': len(active),
            'mean_sym_score': np.mean([s.stats.sym_score for s in active]) if active else 0,
            'mean_stability': np.mean([s.stats.stab for s in active]) if active else 0,
            'mean_consistency': np.mean([s.stats.consistency for s in active]) if active else 0,
            'total_states_recorded': len(self.state_history)
        }


def test_symbol_extractor():
    """Test del extractor de símbolos."""
    print("=" * 60)
    print("TEST: SYMBOL EXTRACTOR")
    print("=" * 60)

    extractor = SymbolExtractor('NEO', state_dim=6)

    # Simular estados con 3 clusters naturales
    np.random.seed(42)

    for t in range(200):
        # 3 modos diferentes
        mode = t % 3

        if mode == 0:
            z = np.array([0.8, 0.1, 0.1, 0.0, 0.0, 0.0]) + np.random.randn(6) * 0.05
        elif mode == 1:
            z = np.array([0.1, 0.8, 0.1, 0.0, 0.0, 0.0]) + np.random.randn(6) * 0.05
        else:
            z = np.array([0.1, 0.1, 0.8, 0.0, 0.0, 0.0]) + np.random.randn(6) * 0.05

        z = to_simplex(np.abs(z) + 0.01)
        phi = np.random.randn(5) * 0.1
        drives = to_simplex(np.random.rand(6) + 0.1)

        extractor.record_state(t, z, phi, drives, context=mode)

        # Extraer símbolos periódicamente
        if (t + 1) % 50 == 0:
            symbols = extractor.extract_symbols(t)
            stats = extractor.get_statistics()
            print(f"\n  t={t+1}:")
            print(f"    Símbolos activos: {stats['active_symbols']}")
            print(f"    SymScore medio: {stats['mean_sym_score']:.3f}")
            print(f"    Estabilidad media: {stats['mean_stability']:.3f}")
            print(f"    Consistencia media: {stats['mean_consistency']:.3f}")

    # Mostrar símbolos finales
    print("\n" + "=" * 60)
    print("SÍMBOLOS EXTRAÍDOS")
    print("=" * 60)

    for sym_id, sym in extractor.symbols.items():
        print(f"\nSímbolo {sym_id}:")
        print(f"  SymScore: {sym.stats.sym_score:.3f}")
        print(f"  Estabilidad: {sym.stats.stab:.3f}")
        print(f"  Consistencia: {sym.stats.consistency:.3f}")
        print(f"  Episodios: {sym.stats.n_episodes}")
        print(f"  Centro (primeros 3): {sym.stats.mu[:3]}")

    print("\n" + "=" * 60)
    print("TEST COMPLETADO")
    print("=" * 60)

    return extractor


if __name__ == "__main__":
    test_symbol_extractor()
