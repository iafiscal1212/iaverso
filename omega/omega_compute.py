"""
Ω-Compute: Computación Interna Emergente
=========================================

Ω-Compute es un espacio donde se registran patrones de transformación interna:
cómo se deforman los estados S(t), identidades I(t), narrativas H_narr(t).

NO sabe qué es "computación" en sentido humano.
Solo observa: "Así es como tiendo a transformarme cuando actúo."

Principios:
- NO introduce conocimiento externo
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
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class OmegaMode:
    """Un modo de transformación interna Ω_k."""
    index: int
    vector: np.ndarray              # Vector base del modo
    variance_explained: float       # Varianza explicada por este modo
    cumulative_variance: float      # Varianza acumulada hasta este modo


@dataclass
class ModeActivation:
    """Activación de modos para una transición."""
    t: int
    agent_id: str
    coefficients: np.ndarray        # α_{i,k}(t) para cada modo
    reconstruction_error: float     # Error de reconstrucción


class OmegaCompute:
    """
    Sistema de computación interna emergente.

    Registra transiciones T_i(t) = S(t+1) - S(t) y extrae
    modos Ω_k por descomposición SVD/PCA endógena.

    NO dice a los agentes qué hacer.
    Solo deja disponible este campo estructural.
    """

    def __init__(self):
        """Inicializa Ω-Compute."""
        self.t = 0

        # Historial de transiciones por agente: T_i(t) = S(t+1) - S(t)
        self._transitions: Dict[str, List[np.ndarray]] = {}

        # Estados previos para calcular transiciones
        self._prev_states: Dict[str, np.ndarray] = {}

        # Modos Ω_k (vectores base de transformación)
        self._modes: Optional[List[OmegaMode]] = None

        # Matriz de transiciones global para SVD
        self._transition_matrix: Optional[np.ndarray] = None

        # Activaciones α_{i,k}(t) por agente
        self._activations: Dict[str, List[ModeActivation]] = {}

        # Estadísticas para umbrales endógenos
        self._variance_history: List[float] = []

    def register_transition(
        self,
        agent_id: str,
        S_t: np.ndarray,
        S_t1: np.ndarray
    ) -> np.ndarray:
        """
        Registra una transición T_i(t) = S(t+1) - S(t).

        Args:
            agent_id: Identificador del agente
            S_t: Estado en tiempo t
            S_t1: Estado en tiempo t+1

        Returns:
            Vector de transición T_i(t)
        """
        self.t += 1

        # Calcular transición
        T_t = S_t1 - S_t

        # Inicializar historial si es necesario
        if agent_id not in self._transitions:
            self._transitions[agent_id] = []
            self._activations[agent_id] = []

        # Guardar transición
        self._transitions[agent_id].append(T_t.copy())

        # Guardar estado actual para próxima transición
        self._prev_states[agent_id] = S_t1.copy()

        # Limitar historial endógenamente
        max_hist = self._get_max_history()
        if len(self._transitions[agent_id]) > max_hist:
            self._transitions[agent_id] = self._transitions[agent_id][-max_hist:]

        return T_t

    def register_state(self, agent_id: str, S_t: np.ndarray) -> Optional[np.ndarray]:
        """
        Registra un estado y calcula transición si hay estado previo.

        Args:
            agent_id: Identificador del agente
            S_t: Estado actual

        Returns:
            Transición T_i(t-1) si existe estado previo, None si no
        """
        if agent_id in self._prev_states:
            S_prev = self._prev_states[agent_id]
            return self.register_transition(agent_id, S_prev, S_t)
        else:
            self._prev_states[agent_id] = S_t.copy()
            if agent_id not in self._transitions:
                self._transitions[agent_id] = []
                self._activations[agent_id] = []
            return None

    def _get_max_history(self) -> int:
        """
        Calcula tamaño máximo de historial endógenamente.

        Basado en el número total de transiciones observadas.
        """
        total_transitions = sum(len(t) for t in self._transitions.values())
        if total_transitions < 100:
            return 100

        # Percentil 75 del historial visto
        return max(100, int(np.sqrt(total_transitions) * 10))

    def _build_transition_matrix(self) -> np.ndarray:
        """
        Construye matriz de transiciones [T_i(t_1), T_i(t_2), ...].

        Returns:
            Matriz d × M donde d es dimensión y M es número de transiciones
        """
        all_transitions = []

        for agent_id, transitions in self._transitions.items():
            all_transitions.extend(transitions)

        if not all_transitions:
            return np.array([])

        # Asegurar misma dimensión (usar mínima)
        min_dim = min(len(t) for t in all_transitions)
        aligned = [t[:min_dim] for t in all_transitions]

        # Matriz d × M
        matrix = np.array(aligned).T

        return matrix

    def update_modes(self) -> List[OmegaMode]:
        """
        Extrae modos Ω_k por SVD de la matriz de transiciones.

        El número de modos se elige endógenamente por varianza explicada:
        - Se seleccionan modos hasta alcanzar el percentil 90 de varianza.

        Returns:
            Lista de OmegaMode
        """
        # Construir matriz de transiciones
        T_matrix = self._build_transition_matrix()

        if T_matrix.size == 0:
            self._modes = []
            return []

        # SVD
        try:
            U, S, Vt = np.linalg.svd(T_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            self._modes = []
            return []

        # Varianza explicada por cada modo
        total_var = np.sum(S ** 2)
        if total_var < np.finfo(float).eps:
            self._modes = []
            return []

        var_explained = (S ** 2) / total_var
        cumulative_var = np.cumsum(var_explained)

        # Seleccionar número de modos endógenamente
        # Usar varianza acumulada >= percentil basado en historial
        if len(self._variance_history) > 10:
            threshold = np.percentile(self._variance_history, 75)
        else:
            # Sin historial suficiente: usar 1 - 1/K donde K = número de componentes
            K = len(S)
            threshold = 1 - 1 / K if K > 1 else 1/2

        # Encontrar número de modos
        n_modes = np.searchsorted(cumulative_var, threshold) + 1
        n_modes = max(1, min(n_modes, len(S)))

        # Crear modos
        self._modes = []
        for k in range(n_modes):
            mode = OmegaMode(
                index=k,
                vector=U[:, k].copy(),
                variance_explained=float(var_explained[k]),
                cumulative_variance=float(cumulative_var[k])
            )
            self._modes.append(mode)

        # Actualizar historial de varianza
        self._variance_history.append(float(cumulative_var[n_modes - 1]))

        # Guardar matriz de transiciones
        self._transition_matrix = T_matrix

        return self._modes

    def project_transition(
        self,
        agent_id: str,
        T_t: np.ndarray
    ) -> Optional[ModeActivation]:
        """
        Proyecta una transición en la base de modos Ω_k.

        Args:
            agent_id: Identificador del agente
            T_t: Vector de transición

        Returns:
            ModeActivation con coeficientes α_{i,k}(t)
        """
        if not self._modes:
            return None

        # Alinear dimensión
        mode_dim = len(self._modes[0].vector)
        if len(T_t) != mode_dim:
            # Ajustar dimensión
            if len(T_t) > mode_dim:
                T_t = T_t[:mode_dim]
            else:
                T_t_padded = np.zeros(mode_dim)
                T_t_padded[:len(T_t)] = T_t
                T_t = T_t_padded

        # Proyectar: α_k = Ω_k · T_t
        coefficients = np.array([
            np.dot(mode.vector, T_t)
            for mode in self._modes
        ])

        # Reconstrucción
        reconstruction = sum(
            coefficients[k] * self._modes[k].vector
            for k in range(len(self._modes))
        )

        # Error de reconstrucción
        error = np.linalg.norm(T_t - reconstruction)
        error_normalized = error / (np.linalg.norm(T_t) + np.finfo(float).eps)

        activation = ModeActivation(
            t=self.t,
            agent_id=agent_id,
            coefficients=coefficients,
            reconstruction_error=float(error_normalized)
        )

        # Guardar activación
        if agent_id not in self._activations:
            self._activations[agent_id] = []
        self._activations[agent_id].append(activation)

        return activation

    def get_modes(self) -> List[OmegaMode]:
        """Retorna modos Ω_k actuales."""
        return self._modes or []

    def get_agent_activations(self, agent_id: str) -> List[ModeActivation]:
        """Retorna historial de activaciones de un agente."""
        return self._activations.get(agent_id, [])

    def get_mode_correlation(self) -> np.ndarray:
        """
        Calcula correlación entre modos basada en activaciones.

        Returns:
            Matriz de correlación K × K
        """
        if not self._modes:
            return np.array([])

        # Recolectar todas las activaciones
        all_coeffs = []
        for agent_activations in self._activations.values():
            for act in agent_activations:
                all_coeffs.append(act.coefficients)

        if len(all_coeffs) < 2:
            return np.eye(len(self._modes))

        coeffs_matrix = np.array(all_coeffs)

        # Correlación
        corr = np.corrcoef(coeffs_matrix.T)

        return corr

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del sistema."""
        n_modes = len(self._modes) if self._modes else 0

        total_var_explained = 0
        if self._modes:
            total_var_explained = self._modes[-1].cumulative_variance

        return {
            't': self.t,
            'n_agents': len(self._transitions),
            'total_transitions': sum(len(t) for t in self._transitions.values()),
            'n_modes': n_modes,
            'total_variance_explained': total_var_explained,
            'mode_variances': [m.variance_explained for m in self._modes] if self._modes else [],
            'mean_reconstruction_error': self._compute_mean_error()
        }

    def _compute_mean_error(self) -> float:
        """Calcula error de reconstrucción medio."""
        all_errors = []
        for agent_activations in self._activations.values():
            for act in agent_activations:
                all_errors.append(act.reconstruction_error)

        if not all_errors:
            return 0

        return float(np.mean(all_errors))
