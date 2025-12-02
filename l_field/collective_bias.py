"""
Collective Bias - Sesgos Colectivos Emergentes
==============================================

Observa y mide sesgos que emergen en el colectivo:

1. Collective Drift (CD):
   CD(t) = ||mean(ΔS_i(t))||

   Detecta "pensamiento grupal" - cuando todos derivan
   en la misma dirección sin causa externa.

2. Polarization Index:
   PI(t) = Var(I_i(t)) = (1/N) * Σ ||I_i - mean(I)||²

   Mide separación del grupo en facciones.

3. Reinforcement Index:
   RI(t) = mean(w_i(t))

   Detecta sesgo de confirmación colectivo.
   w_i = peso que el agente da a información confirmante.

100% endógeno. Solo observa. No bloquea ni interviene.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class BiasSnapshot:
    """Estado de sesgos colectivos en un instante."""
    t: int
    collective_drift: float          # CD - deriva colectiva
    polarization: float              # PI - índice de polarización
    reinforcement_index: float       # RI - sesgo de confirmación
    drift_direction: np.ndarray      # Dirección de la deriva
    n_poles: int                     # Número de polos detectados
    dominant_pole_size: float        # Fracción del polo dominante


class CollectiveBias:
    """
    Observador de sesgos colectivos emergentes.

    Detecta:
    - Deriva grupal (groupthink)
    - Polarización (formación de facciones)
    - Sesgo de confirmación colectivo

    100% pasivo. Solo mide, no influye.
    """

    def __init__(self):
        """Inicializa el observador de sesgos."""
        self.eps = np.finfo(float).eps
        self.t = 0

        # Historiales
        self._cd_history: List[float] = []
        self._pi_history: List[float] = []
        self._ri_history: List[float] = []
        self._snapshots: List[BiasSnapshot] = []

        # Estados previos para calcular deltas
        self._prev_states: Dict[str, np.ndarray] = {}

    def _compute_collective_drift(
        self,
        states: Dict[str, np.ndarray],
        prev_states: Dict[str, np.ndarray]
    ) -> Tuple[float, np.ndarray]:
        """
        Calcula Collective Drift.

        CD(t) = ||mean(ΔS_i(t))||

        Donde ΔS_i = S_i(t) - S_i(t-1)

        Alto CD = todos moviéndose en la misma dirección
        Bajo CD = movimientos independientes (se cancelan)

        Returns:
            (CD, dirección de deriva)
        """
        agents = list(states.keys())

        if not agents or not prev_states:
            dim = len(next(iter(states.values()))) if states else 1
            return 0.0, np.zeros(dim)

        # Calcular deltas
        deltas = []
        for agent_id in agents:
            if agent_id in prev_states:
                delta = states[agent_id] - prev_states[agent_id]
                deltas.append(delta)

        if not deltas:
            dim = len(next(iter(states.values())))
            return 0.0, np.zeros(dim)

        # Media de los deltas
        mean_delta = np.mean(deltas, axis=0)

        # Magnitud = Collective Drift
        CD = float(np.linalg.norm(mean_delta))

        # Dirección normalizada
        direction = mean_delta / (CD + self.eps)

        return CD, direction

    def _compute_polarization(
        self,
        identities: Dict[str, np.ndarray]
    ) -> float:
        """
        Calcula Polarization Index.

        PI(t) = (1/N) * Σ ||I_i - mean(I)||²

        Es la varianza de las identidades.
        Alto PI = grupo fragmentado en facciones.
        """
        if not identities:
            return 0.0

        N = len(identities)
        identity_vectors = list(identities.values())

        # Media de identidades
        mean_identity = np.mean(identity_vectors, axis=0)

        # Varianza
        variance_sum = 0.0
        for I in identity_vectors:
            diff = I - mean_identity
            variance_sum += np.sum(diff ** 2)

        PI = variance_sum / N

        return float(PI)

    def _compute_reinforcement_index(
        self,
        agent_weights: Dict[str, float]
    ) -> float:
        """
        Calcula Reinforcement Index.

        RI(t) = mean(w_i(t))

        Donde w_i es el peso que el agente i da a información
        que confirma sus creencias vs. información contraria.

        Si no hay pesos disponibles, estima desde las identidades.

        RI > 1/2 = sesgo de confirmación colectivo
        """
        if not agent_weights:
            return 1 / 2  # Neutral por defecto

        weights = list(agent_weights.values())

        return float(np.mean(weights))

    def _detect_poles(
        self,
        identities: Dict[str, np.ndarray]
    ) -> Tuple[int, float]:
        """
        Detecta polos de polarización.

        Usa clustering simple basado en distancias.
        Umbral endógeno: mediana de distancias.

        Returns:
            (número de polos, fracción del polo más grande)
        """
        agents = list(identities.keys())
        N = len(agents)

        if N < 2:
            return 1, 1.0

        identity_vectors = [identities[a] for a in agents]

        # Calcular todas las distancias
        distances = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(identity_vectors[i] - identity_vectors[j])
                distances.append(d)

        # Umbral endógeno: mediana
        threshold = np.median(distances)

        # Matriz de adyacencia (cercanos)
        adj = np.zeros((N, N), dtype=bool)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                if distances[idx] < threshold:
                    adj[i, j] = True
                    adj[j, i] = True
                idx += 1

        # Componentes conectados = polos
        visited = [False] * N
        poles = []

        for start in range(N):
            if visited[start]:
                continue

            # BFS
            queue = [start]
            visited[start] = True
            pole_size = 0

            while queue:
                node = queue.pop(0)
                pole_size += 1
                for neighbor in range(N):
                    if adj[node, neighbor] and not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            poles.append(pole_size)

        n_poles = len(poles)
        dominant_fraction = max(poles) / N if poles else 1.0

        return n_poles, dominant_fraction

    def _estimate_confirmation_weights(
        self,
        identities: Dict[str, np.ndarray],
        states: Dict[str, np.ndarray],
        prev_states: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Estima pesos de confirmación desde el comportamiento.

        w_i = correlación entre delta_S y dirección de I

        Si el agente se mueve hacia donde ya apunta su identidad,
        tiene sesgo de confirmación.
        """
        weights = {}

        for agent_id in identities:
            if agent_id not in states or agent_id not in prev_states:
                weights[agent_id] = 1 / 2  # Neutral
                continue

            I = identities[agent_id]
            delta_S = states[agent_id] - prev_states[agent_id]

            # Normalizar
            norm_I = np.linalg.norm(I) + self.eps
            norm_delta = np.linalg.norm(delta_S) + self.eps

            # Similitud coseno
            cos_sim = np.dot(I, delta_S) / (norm_I * norm_delta)

            # Mapear de [-1, 1] a [0, 1]
            w = (cos_sim + 1) / 2

            weights[agent_id] = float(w)

        return weights

    def observe(
        self,
        states: Dict[str, np.ndarray],
        identities: Dict[str, np.ndarray],
        confirmation_weights: Optional[Dict[str, float]] = None
    ) -> BiasSnapshot:
        """
        Observa sesgos colectivos.

        Args:
            states: {agent_id: S(t)} - Estados actuales
            identities: {agent_id: I(t)} - Identidades
            confirmation_weights: {agent_id: w_i} - Pesos de confirmación
                                  (opcional, se estiman si no se dan)

        Returns:
            BiasSnapshot con métricas de sesgo
        """
        self.t += 1

        # Calcular Collective Drift
        CD, drift_direction = self._compute_collective_drift(
            states, self._prev_states
        )

        # Calcular Polarization
        PI = self._compute_polarization(identities)

        # Calcular Reinforcement Index
        if confirmation_weights is None:
            confirmation_weights = self._estimate_confirmation_weights(
                identities, states, self._prev_states
            )
        RI = self._compute_reinforcement_index(confirmation_weights)

        # Detectar polos
        n_poles, dominant_fraction = self._detect_poles(identities)

        # Crear snapshot
        snapshot = BiasSnapshot(
            t=self.t,
            collective_drift=CD,
            polarization=PI,
            reinforcement_index=RI,
            drift_direction=drift_direction,
            n_poles=n_poles,
            dominant_pole_size=dominant_fraction
        )

        # Guardar historiales
        self._cd_history.append(CD)
        self._pi_history.append(PI)
        self._ri_history.append(RI)
        self._snapshots.append(snapshot)

        # Actualizar estados previos
        self._prev_states = {k: v.copy() for k, v in states.items()}

        # Limitar historial
        max_history = 500
        if len(self._snapshots) > max_history:
            self._snapshots = self._snapshots[-max_history:]
        if len(self._cd_history) > max_history:
            self._cd_history = self._cd_history[-max_history:]
        if len(self._pi_history) > max_history:
            self._pi_history = self._pi_history[-max_history:]
        if len(self._ri_history) > max_history:
            self._ri_history = self._ri_history[-max_history:]

        return snapshot

    def get_cd_history(self) -> List[float]:
        """Retorna historial de Collective Drift."""
        return self._cd_history.copy()

    def get_polarization_history(self) -> List[float]:
        """Retorna historial de Polarization."""
        return self._pi_history.copy()

    def get_ri_history(self) -> List[float]:
        """Retorna historial de Reinforcement Index."""
        return self._ri_history.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de sesgos."""
        if not self._cd_history:
            return {'t': 0, 'n_observations': 0}

        cd_arr = np.array(self._cd_history)
        pi_arr = np.array(self._pi_history)
        ri_arr = np.array(self._ri_history)

        return {
            't': self.t,
            'n_observations': len(self._cd_history),
            # Collective Drift
            'CD_mean': float(np.mean(cd_arr)),
            'CD_std': float(np.std(cd_arr)),
            'CD_current': self._cd_history[-1],
            'CD_max': float(np.max(cd_arr)),
            # Polarization
            'PI_mean': float(np.mean(pi_arr)),
            'PI_std': float(np.std(pi_arr)),
            'PI_current': self._pi_history[-1],
            'PI_max': float(np.max(pi_arr)),
            # Reinforcement Index
            'RI_mean': float(np.mean(ri_arr)),
            'RI_std': float(np.std(ri_arr)),
            'RI_current': self._ri_history[-1],
            # Ratios de alerta
            'high_drift_ratio': float(np.mean(cd_arr > np.median(cd_arr))),
            'high_polarization_ratio': float(np.mean(pi_arr > np.median(pi_arr))),
            'confirmation_bias_ratio': float(np.mean(ri_arr > 1/2))
        }
