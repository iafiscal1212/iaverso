#!/usr/bin/env python3
"""
WEAVER Phase Graph
==================

Grafo de dependencias entre fases con Transfer Entropy.
100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict


class PhaseGraph:
    """
    Grafo de dependencias entre fases usando Transfer Entropy.

    100% Endógeno:
    - Pesos = Transfer Entropy entre series temporales de fases
    - Umbral de conexión = mediana de TE (endógeno)
    - Detección de ciclos por DFS
    """

    def __init__(self):
        # Nodos (fases)
        self.nodes: Set[str] = set()

        # Historia de métricas por fase
        self.phase_metrics: Dict[str, List[float]] = defaultdict(list)

        # Matriz de Transfer Entropy
        self.te_matrix: Dict[Tuple[str, str], float] = {}

        # Grafo de dependencias (aristas significativas)
        self.edges: Dict[str, Set[str]] = defaultdict(set)

        # Historia de TE para umbral adaptativo
        self.te_history: List[float] = []

    def add_phase(self, name: str) -> None:
        """Registra una nueva fase en el grafo."""
        self.nodes.add(name)

    def update_metric(self, phase: str, value: float) -> None:
        """Actualiza métrica de una fase."""
        if phase not in self.nodes:
            self.add_phase(phase)
        self.phase_metrics[phase].append(value)

    def _estimate_transfer_entropy(self, source: str, target: str,
                                    lag: int = 1) -> float:
        """
        Estima Transfer Entropy de source → target.

        TE(X→Y) ≈ I(Y_t; X_{t-lag} | Y_{t-1})

        Aproximación por correlación parcial (simplificado).
        100% endógeno.
        """
        X = self.phase_metrics.get(source, [])
        Y = self.phase_metrics.get(target, [])

        min_len = min(len(X), len(Y))
        if min_len < lag + 2:
            return 0.0

        # Alinear series
        X = np.array(X[-min_len:])
        Y = np.array(Y[-min_len:])

        # Y_t, X_{t-lag}, Y_{t-1}
        Y_t = Y[lag:]
        X_lag = X[:-lag] if lag > 0 else X
        Y_prev = Y[lag-1:-1] if lag > 1 else Y[:-1]

        # Ajustar longitudes
        n = min(len(Y_t), len(X_lag), len(Y_prev))
        if n < 3:
            return 0.0

        Y_t = Y_t[:n]
        X_lag = X_lag[:n]
        Y_prev = Y_prev[:n]

        # Correlación parcial aproximada
        try:
            # Correlación Y_t con X_lag
            r_yx = np.corrcoef(Y_t, X_lag)[0, 1]

            # Correlación Y_t con Y_prev
            r_yy = np.corrcoef(Y_t, Y_prev)[0, 1]

            # Correlación X_lag con Y_prev
            r_xy = np.corrcoef(X_lag, Y_prev)[0, 1]

            # Correlación parcial (aproximación de TE)
            if np.isnan(r_yx) or np.isnan(r_yy) or np.isnan(r_xy):
                return 0.0

            denom = np.sqrt((1 - r_yy**2) * (1 - r_xy**2))
            if denom < 1e-10:
                return 0.0

            r_partial = (r_yx - r_yy * r_xy) / denom

            # TE ≈ -0.5 * log(1 - r_partial^2)
            te = -0.5 * np.log(1 - r_partial**2 + 1e-10)
            return max(0.0, te)

        except Exception:
            return 0.0

    def compute_all_te(self) -> None:
        """Computa Transfer Entropy entre todas las fases."""
        nodes_list = list(self.nodes)

        for source in nodes_list:
            for target in nodes_list:
                if source != target:
                    te = self._estimate_transfer_entropy(source, target)
                    self.te_matrix[(source, target)] = te
                    if te > 0:
                        self.te_history.append(te)

        # Actualizar grafo con aristas significativas
        self._update_edges()

    def _update_edges(self) -> None:
        """
        Actualiza aristas del grafo basado en TE significativo.

        Umbral = mediana de TE histórico (100% endógeno)
        """
        if not self.te_history:
            return

        # Umbral endógeno: mediana
        threshold = np.median(self.te_history)

        self.edges.clear()
        for (source, target), te in self.te_matrix.items():
            if te > threshold:
                self.edges[source].add(target)

    def get_dependencies(self, phase: str) -> Set[str]:
        """Retorna fases de las que depende la fase dada."""
        deps = set()
        for source, targets in self.edges.items():
            if phase in targets:
                deps.add(source)
        return deps

    def get_dependents(self, phase: str) -> Set[str]:
        """Retorna fases que dependen de la fase dada."""
        return self.edges.get(phase, set())

    def detect_cycles(self) -> List[List[str]]:
        """
        Detecta ciclos en el grafo de dependencias.

        Returns:
            Lista de ciclos encontrados
        """
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.edges.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Encontrado ciclo
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            rec_stack.remove(node)

        for node in self.nodes:
            if node not in visited:
                dfs(node, [])

        return cycles

    def get_topological_order(self) -> List[str]:
        """
        Retorna orden topológico de fases (si no hay ciclos).

        Returns:
            Lista ordenada de fases por dependencias
        """
        # Calcular in-degree
        in_degree = {node: 0 for node in self.nodes}
        for source, targets in self.edges.items():
            for target in targets:
                if target in in_degree:
                    in_degree[target] += 1

        # Kahn's algorithm
        queue = [n for n, d in in_degree.items() if d == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in self.edges.get(node, set()):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        return result

    def get_centrality(self) -> Dict[str, float]:
        """
        Calcula centralidad de cada fase (PageRank simplificado).

        100% endógeno: basado en estructura del grafo
        """
        n = len(self.nodes)
        if n == 0:
            return {}

        nodes_list = list(self.nodes)
        centrality = {node: 1.0 / n for node in nodes_list}

        # Iteraciones de PageRank simplificado
        damping = 0.85
        iterations = int(np.sqrt(len(self.te_history) + 1))  # Endógeno
        iterations = max(5, min(iterations, 50))

        for _ in range(iterations):
            new_centrality = {}
            for node in nodes_list:
                rank = (1 - damping) / n
                for source in self.get_dependencies(node):
                    out_degree = len(self.edges.get(source, set()))
                    if out_degree > 0:
                        rank += damping * centrality[source] / out_degree
                new_centrality[node] = rank
            centrality = new_centrality

        return centrality

    def to_dict(self) -> Dict[str, Any]:
        """Serializa grafo a diccionario."""
        return {
            'nodes': list(self.nodes),
            'n_edges': sum(len(targets) for targets in self.edges.values()),
            'te_threshold': np.median(self.te_history) if self.te_history else 0.0,
            'edges': {
                source: list(targets)
                for source, targets in self.edges.items()
            },
            'centrality': self.get_centrality(),
            'cycles': self.detect_cycles()
        }
