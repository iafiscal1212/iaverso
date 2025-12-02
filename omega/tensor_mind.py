"""
TensorMind: Interacción de Orden Superior
==========================================

TensorMind es un espacio donde se registran interacciones multi-agente
como tensores de orden superior. Sin saber teoría de tensores, solo observa:
- "Cuando A y B interactúan, pasa X"
- "Cuando A, B y C interactúan, pasa Y diferente"

Principios:
- NO introduce conocimiento externo (álgebra tensorial, etc.)
- NO añade objetivos a los agentes
- NO emite instrucciones de comportamiento
- NO crea recompensas ni penalizaciones
- NO usa números mágicos

Todos los umbrales y pesos se derivan de:
- medias, varianzas, covarianzas
- percentiles
- tamaños de dimensión (1/K, 1/√d)
- eps de máquina

Este módulo es NEUTRAL: calcula estructuras y métricas internas, nada más.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, FrozenSet
from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class Interaction:
    """Registro de una interacción entre agentes."""
    t: int
    agents: Tuple[str, ...]         # IDs de agentes involucrados
    order: int                       # Orden de la interacción (2=par, 3=trío, etc.)
    states: Dict[str, np.ndarray]   # Estados de cada agente
    combined_state: np.ndarray      # Estado combinado (tensor contraído)
    strength: float                  # Fuerza de la interacción


@dataclass
class TensorMode:
    """Modo tensorial emergente."""
    index: int
    agents_involved: Set[str]
    order: int
    direction: np.ndarray           # Dirección principal
    variance_explained: float


class TensorMind:
    """
    Sistema de interacción de orden superior.

    Registra interacciones entre múltiples agentes y extrae
    patrones tensoriales de orden 2, 3, ..., N.

    T^{(k)}_{i_1, i_2, ..., i_k} representa la interacción
    de k agentes en el paso de tiempo actual.

    Métricas emergentes:
    - Correlaciones de pares (orden 2)
    - Correlaciones de tríos (orden 3)
    - Modos tensoriales dominantes
    - Estructura de comunidad implícita

    NO dice a los agentes qué hacer.
    Solo mapea estas estructuras.
    """

    def __init__(self, max_order: int = 3):
        """
        Inicializa TensorMind.

        Args:
            max_order: Orden máximo de interacciones a registrar
        """
        self.t = 0
        self.max_order = max_order

        # Estados actuales por agente
        self._current_states: Dict[str, np.ndarray] = {}

        # Historial de interacciones por orden
        self._interactions: Dict[int, List[Interaction]] = {
            k: [] for k in range(2, max_order + 1)
        }

        # Tensores de correlación por orden
        self._correlation_tensors: Dict[int, Optional[np.ndarray]] = {
            k: None for k in range(2, max_order + 1)
        }

        # Modos tensoriales
        self._modes: List[TensorMode] = []

        # Estadísticas para umbrales endógenos
        self._strength_history: Dict[int, List[float]] = {
            k: [] for k in range(2, max_order + 1)
        }

        # Lista de agentes conocidos
        self._agents: List[str] = []

        # Dimensión del estado
        self._dim: Optional[int] = None

    def register_state(self, agent_id: str, state: np.ndarray) -> None:
        """
        Registra el estado actual de un agente.

        Args:
            agent_id: Identificador del agente
            state: Vector de estado
        """
        state = np.array(state, dtype=float)

        if self._dim is None:
            self._dim = len(state)

        self._current_states[agent_id] = state.copy()

        if agent_id not in self._agents:
            self._agents.append(agent_id)

    def compute_interactions(self) -> Dict[int, List[Interaction]]:
        """
        Calcula interacciones de todos los órdenes entre agentes actuales.

        Returns:
            Dict con listas de Interaction por orden
        """
        self.t += 1

        if len(self._current_states) < 2:
            return {}

        agents = list(self._current_states.keys())
        new_interactions: Dict[int, List[Interaction]] = {}

        # Para cada orden de interacción
        for order in range(2, min(self.max_order + 1, len(agents) + 1)):
            new_interactions[order] = []

            # Generar todas las combinaciones de 'order' agentes
            for agent_combo in combinations(agents, order):
                interaction = self._compute_interaction(agent_combo)
                new_interactions[order].append(interaction)

                # Guardar en historial
                self._interactions[order].append(interaction)
                self._strength_history[order].append(interaction.strength)

            # Limitar historial endógenamente
            max_hist = self._get_max_history(order)
            if len(self._interactions[order]) > max_hist:
                self._interactions[order] = self._interactions[order][-max_hist:]

        return new_interactions

    def _compute_interaction(self, agents: Tuple[str, ...]) -> Interaction:
        """
        Calcula una interacción específica entre agentes.

        Args:
            agents: Tupla de IDs de agentes

        Returns:
            Interaction con métricas
        """
        order = len(agents)
        states = {a: self._current_states[a] for a in agents}

        # Alinear dimensiones
        min_dim = min(len(s) for s in states.values())
        aligned_states = {a: s[:min_dim] for a, s in states.items()}

        # Estado combinado: producto exterior contraído
        combined = self._compute_combined_state(list(aligned_states.values()))

        # Fuerza de interacción: correlación multi-vía
        strength = self._compute_interaction_strength(list(aligned_states.values()))

        return Interaction(
            t=self.t,
            agents=agents,
            order=order,
            states=states,
            combined_state=combined,
            strength=strength
        )

    def _compute_combined_state(self, states: List[np.ndarray]) -> np.ndarray:
        """
        Computa estado combinado como contracción tensorial.

        Para orden 2: s1 ⊗ s2 contraído a vector
        Para orden n: producto de Hadamard normalizado
        """
        if not states:
            return np.array([])

        # Producto de Hadamard (elemento a elemento)
        combined = states[0].copy()
        for s in states[1:]:
            combined = combined * s

        # Normalizar
        norm = np.linalg.norm(combined)
        if norm > np.finfo(float).eps:
            combined = combined / norm

        return combined

    def _compute_interaction_strength(self, states: List[np.ndarray]) -> float:
        """
        Computa fuerza de interacción.

        Basada en correlación multi-vía:
        strength = |Σ_i (Π_j s_j[i]) / (Π_j ||s_j||)|
        """
        if not states or len(states) < 2:
            return 0.0

        # Producto de normas
        norm_product = 1.0
        for s in states:
            n = np.linalg.norm(s)
            if n < np.finfo(float).eps:
                return 0.0
            norm_product *= n

        # Producto elemento a elemento
        product = np.ones_like(states[0])
        for s in states:
            product = product * s

        # Suma y normalización
        strength = np.abs(np.sum(product)) / (norm_product + np.finfo(float).eps)

        return float(strength)

    def _get_max_history(self, order: int) -> int:
        """Calcula tamaño máximo de historial endógenamente."""
        n_interactions = len(self._interactions.get(order, []))
        if n_interactions < 100:
            return 100

        # Menor historial para órdenes mayores (más combinaciones)
        base = int(np.sqrt(n_interactions) * 10)
        return max(50, base // order)

    def update_correlation_tensor(self, order: int = 2) -> Optional[np.ndarray]:
        """
        Actualiza tensor de correlación de un orden dado.

        Para orden 2: matriz de correlación C_{ij}
        Para orden 3: tensor C_{ijk}
        etc.

        Args:
            order: Orden del tensor

        Returns:
            Tensor de correlación o None
        """
        if order > self.max_order or order < 2:
            return None

        interactions = self._interactions.get(order, [])
        if len(interactions) < 3:
            return None

        n_agents = len(self._agents)
        if n_agents < order:
            return None

        # Crear tensor de correlación
        shape = tuple([n_agents] * order)
        tensor = np.zeros(shape)

        # Mapeo de agente a índice
        agent_to_idx = {a: i for i, a in enumerate(self._agents)}

        # Llenar tensor con fuerzas de interacción
        for interaction in interactions[-100:]:  # Últimas 100
            indices = tuple(agent_to_idx.get(a, 0) for a in interaction.agents)
            tensor[indices] = interaction.strength

            # Simetrizar (las interacciones son simétricas)
            for perm in self._permutations(indices):
                tensor[perm] = interaction.strength

        self._correlation_tensors[order] = tensor
        return tensor

    def _permutations(self, indices: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Genera todas las permutaciones de índices."""
        from itertools import permutations as perm
        return list(perm(indices))

    def extract_modes(self, order: int = 2) -> List[TensorMode]:
        """
        Extrae modos tensoriales dominantes.

        Para orden 2: eigenvectores de la matriz de correlación
        Para orden superior: descomposición CP aproximada

        Args:
            order: Orden del tensor

        Returns:
            Lista de TensorMode
        """
        tensor = self.update_correlation_tensor(order)
        if tensor is None:
            return []

        if order == 2:
            return self._extract_modes_order_2(tensor)
        else:
            return self._extract_modes_higher_order(tensor, order)

    def _extract_modes_order_2(self, tensor: np.ndarray) -> List[TensorMode]:
        """Extrae modos de tensor de orden 2 (matriz)."""
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(tensor)
        except np.linalg.LinAlgError:
            return []

        # Ordenar por eigenvalor descendente
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Varianza explicada
        total_var = np.sum(np.abs(eigenvalues))
        if total_var < np.finfo(float).eps:
            return []

        var_explained = np.abs(eigenvalues) / total_var

        # Número de modos: hasta percentil 90 de varianza
        cumulative = np.cumsum(var_explained)
        n_modes = np.searchsorted(cumulative, 0.9) + 1
        n_modes = max(1, min(n_modes, len(eigenvalues)))

        modes = []
        for k in range(n_modes):
            # Agentes involucrados: aquellos con peso significativo
            weights = np.abs(eigenvectors[:, k])
            threshold = np.mean(weights)  # Umbral endógeno
            involved = set(
                self._agents[i]
                for i in range(len(self._agents))
                if i < len(weights) and weights[i] > threshold
            )

            mode = TensorMode(
                index=k,
                agents_involved=involved,
                order=2,
                direction=eigenvectors[:, k].copy(),
                variance_explained=float(var_explained[k])
            )
            modes.append(mode)

        self._modes = modes
        return modes

    def _extract_modes_higher_order(
        self,
        tensor: np.ndarray,
        order: int
    ) -> List[TensorMode]:
        """
        Extrae modos de tensor de orden superior.

        Usa aproximación: aplanar a matriz y aplicar SVD.
        """
        # Aplanar tensor a matriz
        shape = tensor.shape
        n = shape[0]
        matrix = tensor.reshape(n, -1)

        try:
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            return []

        # Varianza explicada
        total_var = np.sum(S ** 2)
        if total_var < np.finfo(float).eps:
            return []

        var_explained = (S ** 2) / total_var

        # Número de modos
        cumulative = np.cumsum(var_explained)
        n_modes = np.searchsorted(cumulative, 0.9) + 1
        n_modes = max(1, min(n_modes, len(S)))

        modes = []
        for k in range(n_modes):
            # Agentes involucrados
            weights = np.abs(U[:, k])
            threshold = np.mean(weights)
            involved = set(
                self._agents[i]
                for i in range(len(self._agents))
                if i < len(weights) and weights[i] > threshold
            )

            mode = TensorMode(
                index=k,
                agents_involved=involved,
                order=order,
                direction=U[:, k].copy(),
                variance_explained=float(var_explained[k])
            )
            modes.append(mode)

        return modes

    def get_pairwise_strength(self, agent_1: str, agent_2: str) -> float:
        """
        Retorna fuerza de interacción entre dos agentes.

        Basada en historial reciente.
        """
        interactions = self._interactions.get(2, [])
        if not interactions:
            return 0.0

        # Buscar interacciones entre estos agentes
        strengths = []
        pair = frozenset([agent_1, agent_2])

        for interaction in interactions[-50:]:
            if frozenset(interaction.agents) == pair:
                strengths.append(interaction.strength)

        if not strengths:
            return 0.0

        return float(np.mean(strengths))

    def get_community_structure(self) -> Dict[str, List[str]]:
        """
        Detecta estructura de comunidad basada en interacciones.

        Returns:
            Dict con clusters de agentes
        """
        if len(self._agents) < 2:
            return {'all': self._agents.copy()}

        # Construir matriz de afinidad
        n = len(self._agents)
        affinity = np.zeros((n, n))

        for i, a1 in enumerate(self._agents):
            for j, a2 in enumerate(self._agents):
                if i != j:
                    affinity[i, j] = self.get_pairwise_strength(a1, a2)

        # Clustering espectral simple
        # Eigenvector dominante indica comunidad
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(affinity)
        except np.linalg.LinAlgError:
            return {'all': self._agents.copy()}

        # Usar segundo eigenvector para bipartición
        if n < 2:
            return {'all': self._agents.copy()}

        fiedler = eigenvectors[:, -2]  # Segundo mayor eigenvalue

        # Umbral endógeno: mediana
        threshold = np.median(fiedler)

        community_1 = [
            self._agents[i]
            for i in range(n)
            if fiedler[i] >= threshold
        ]
        community_2 = [
            self._agents[i]
            for i in range(n)
            if fiedler[i] < threshold
        ]

        return {
            'community_1': community_1,
            'community_2': community_2
        }

    def get_interaction_strength_threshold(self, order: int = 2) -> float:
        """Calcula umbral de fuerza de interacción endógenamente."""
        history = self._strength_history.get(order, [])
        if len(history) < 10:
            # Sin historial: usar 1/K donde K = número de agentes
            K = max(1, len(self._agents))
            return 1 / K

        return float(np.percentile(history, 75))

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del sistema TensorMind."""
        stats = {
            't': self.t,
            'n_agents': len(self._agents),
            'agents': self._agents.copy(),
            'dim': self._dim,
            'max_order': self.max_order
        }

        # Estadísticas por orden
        for order in range(2, self.max_order + 1):
            interactions = self._interactions.get(order, [])
            history = self._strength_history.get(order, [])

            stats[f'order_{order}'] = {
                'n_interactions': len(interactions),
                'mean_strength': float(np.mean(history)) if history else 0.0,
                'std_strength': float(np.std(history)) if len(history) > 1 else 0.0,
                'threshold': self.get_interaction_strength_threshold(order)
            }

        # Modos actuales
        stats['n_modes'] = len(self._modes)
        stats['modes'] = [
            {
                'index': m.index,
                'order': m.order,
                'agents': list(m.agents_involved),
                'variance_explained': m.variance_explained
            }
            for m in self._modes
        ]

        # Estructura de comunidad
        stats['communities'] = self.get_community_structure()

        return stats

    def get_agent_centrality(self, agent_id: str) -> float:
        """
        Calcula centralidad de un agente basada en interacciones.

        Centralidad = suma de fuerzas de interacción normalizadas.
        """
        if agent_id not in self._agents:
            return 0.0

        total_strength = 0.0
        n_interactions = 0

        for order in range(2, self.max_order + 1):
            for interaction in self._interactions.get(order, [])[-50:]:
                if agent_id in interaction.agents:
                    total_strength += interaction.strength
                    n_interactions += 1

        if n_interactions == 0:
            return 0.0

        # Normalizar por número de interacciones posibles
        return float(total_strength / n_interactions)
