"""
Correlations - Deep Identity Coupling (DIC) & Narrative Resonance
=================================================================

Mide correlaciones profundas entre agentes:

1. DIC (Deep Identity Coupling):
   DIC(t) = (1 / N(N-1)) * Σ_{i≠j} corr(I_i(t), I_j(t))

   Cuánto se parecen las identidades de los agentes.
   Alto DIC = homogeneización del grupo.

2. Narrative Resonance:
   NR(t) = (1 / N(N-1)) * Σ_{i≠j} corr(H_narr,i(t), H_narr,j(t))

   Cuánto comparten narrativas los agentes.
   Alto NR = narrativa colectiva emergente.

100% endógeno. Solo observa. No influye.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class CorrelationSnapshot:
    """Estado de correlaciones en un instante."""
    t: int
    DIC: float                      # Deep Identity Coupling
    narrative_resonance: float      # Resonancia narrativa
    identity_clusters: int          # Clusters de identidad detectados
    narrative_clusters: int         # Clusters narrativos detectados
    max_identity_corr: float        # Máxima correlación de identidad
    max_narrative_corr: float       # Máxima correlación narrativa


class DeepCorrelations:
    """
    Calcula correlaciones profundas entre agentes.

    Observa:
    - Acoplamiento de identidades (DIC)
    - Resonancia de narrativas
    - Formación de clusters

    Sin influir en los agentes.
    """

    def __init__(self):
        """Inicializa el observador de correlaciones."""
        self.eps = np.finfo(float).eps
        self.t = 0

        # Historiales
        self._dic_history: List[float] = []
        self._nr_history: List[float] = []
        self._snapshots: List[CorrelationSnapshot] = []

    def _compute_dic(
        self,
        identities: Dict[str, np.ndarray]
    ) -> float:
        """
        Calcula Deep Identity Coupling.

        DIC(t) = (1 / N(N-1)) * Σ_{i≠j} corr(I_i, I_j)

        Rango: [-1, 1]
        - 1 = identidades idénticas
        - 0 = sin correlación
        - -1 = identidades opuestas
        """
        agents = list(identities.keys())
        N = len(agents)

        if N < 2:
            return 0.0

        corr_sum = 0.0
        n_pairs = 0

        for i in range(N):
            for j in range(i + 1, N):
                I_i = identities[agents[i]]
                I_j = identities[agents[j]]

                # Correlación de Pearson
                corr = np.corrcoef(I_i.flatten(), I_j.flatten())[0, 1]

                if not np.isnan(corr):
                    corr_sum += corr
                    n_pairs += 1

        if n_pairs == 0:
            return 0.0

        # Normalizar por N(N-1)/2 pares únicos, escalar a N(N-1)
        DIC = (2 * corr_sum) / (N * (N - 1) + self.eps)

        return float(DIC)

    def _compute_narrative_resonance(
        self,
        narratives: Dict[str, np.ndarray]
    ) -> float:
        """
        Calcula Narrative Resonance.

        NR(t) = (1 / N(N-1)) * Σ_{i≠j} corr(H_narr,i, H_narr,j)

        Mide cuánto comparten los agentes en sus narrativas.
        """
        agents = list(narratives.keys())
        N = len(agents)

        if N < 2:
            return 0.0

        corr_sum = 0.0
        n_pairs = 0

        for i in range(N):
            for j in range(i + 1, N):
                H_i = narratives[agents[i]]
                H_j = narratives[agents[j]]

                # Correlación de Pearson
                corr = np.corrcoef(H_i.flatten(), H_j.flatten())[0, 1]

                if not np.isnan(corr):
                    corr_sum += corr
                    n_pairs += 1

        if n_pairs == 0:
            return 0.0

        NR = (2 * corr_sum) / (N * (N - 1) + self.eps)

        return float(NR)

    def _detect_clusters(
        self,
        vectors: Dict[str, np.ndarray],
        threshold: float = None
    ) -> int:
        """
        Detecta clusters basándose en correlación.

        Umbral endógeno: 1/2 (correlación > 0.5 = mismo cluster)

        Usa algoritmo de componentes conectados simple.
        """
        if threshold is None:
            threshold = 1 / 2  # Endógeno

        agents = list(vectors.keys())
        N = len(agents)

        if N < 2:
            return N

        # Matriz de adyacencia
        adj = np.zeros((N, N), dtype=bool)

        for i in range(N):
            for j in range(i + 1, N):
                v_i = vectors[agents[i]]
                v_j = vectors[agents[j]]

                corr = np.corrcoef(v_i.flatten(), v_j.flatten())[0, 1]

                if not np.isnan(corr) and corr > threshold:
                    adj[i, j] = True
                    adj[j, i] = True

        # Encontrar componentes conectados (BFS simple)
        visited = [False] * N
        n_clusters = 0

        for start in range(N):
            if visited[start]:
                continue

            # BFS desde start
            queue = [start]
            visited[start] = True

            while queue:
                node = queue.pop(0)
                for neighbor in range(N):
                    if adj[node, neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            n_clusters += 1

        return n_clusters

    def _max_correlation(self, vectors: Dict[str, np.ndarray]) -> float:
        """Encuentra la máxima correlación entre pares."""
        agents = list(vectors.keys())
        N = len(agents)

        if N < 2:
            return 0.0

        max_corr = -1.0

        for i in range(N):
            for j in range(i + 1, N):
                v_i = vectors[agents[i]]
                v_j = vectors[agents[j]]

                corr = np.corrcoef(v_i.flatten(), v_j.flatten())[0, 1]

                if not np.isnan(corr) and corr > max_corr:
                    max_corr = corr

        return float(max_corr)

    def observe(
        self,
        identities: Dict[str, np.ndarray],
        narratives: Dict[str, np.ndarray]
    ) -> CorrelationSnapshot:
        """
        Observa correlaciones entre agentes.

        Args:
            identities: {agent_id: I(t)} - Vectores de identidad
            narratives: {agent_id: H_narr(t)} - Narrativas

        Returns:
            CorrelationSnapshot con métricas de correlación
        """
        self.t += 1

        # Calcular métricas
        DIC = self._compute_dic(identities)
        NR = self._compute_narrative_resonance(narratives)

        identity_clusters = self._detect_clusters(identities)
        narrative_clusters = self._detect_clusters(narratives)

        max_id_corr = self._max_correlation(identities)
        max_narr_corr = self._max_correlation(narratives)

        # Crear snapshot
        snapshot = CorrelationSnapshot(
            t=self.t,
            DIC=DIC,
            narrative_resonance=NR,
            identity_clusters=identity_clusters,
            narrative_clusters=narrative_clusters,
            max_identity_corr=max_id_corr,
            max_narrative_corr=max_narr_corr
        )

        self._dic_history.append(DIC)
        self._nr_history.append(NR)
        self._snapshots.append(snapshot)

        # Limitar historial
        max_history = 500
        if len(self._snapshots) > max_history:
            self._snapshots = self._snapshots[-max_history:]
        if len(self._dic_history) > max_history:
            self._dic_history = self._dic_history[-max_history:]
        if len(self._nr_history) > max_history:
            self._nr_history = self._nr_history[-max_history:]

        return snapshot

    def get_dic_history(self) -> List[float]:
        """Retorna historial de DIC."""
        return self._dic_history.copy()

    def get_nr_history(self) -> List[float]:
        """Retorna historial de Narrative Resonance."""
        return self._nr_history.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de correlaciones."""
        if not self._dic_history:
            return {'t': 0, 'n_observations': 0}

        dic_arr = np.array(self._dic_history)
        nr_arr = np.array(self._nr_history)

        return {
            't': self.t,
            'n_observations': len(self._dic_history),
            'DIC_mean': float(np.mean(dic_arr)),
            'DIC_std': float(np.std(dic_arr)),
            'DIC_current': self._dic_history[-1],
            'DIC_max': float(np.max(dic_arr)),
            'NR_mean': float(np.mean(nr_arr)),
            'NR_std': float(np.std(nr_arr)),
            'NR_current': self._nr_history[-1],
            'NR_max': float(np.max(nr_arr)),
            'high_dic_ratio': float(np.mean(dic_arr > 1/2)),  # % tiempo en alto DIC
            'high_nr_ratio': float(np.mean(nr_arr > 1/2))    # % tiempo en alta resonancia
        }
