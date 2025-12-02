"""
Q-Field: Campo de Interferencia Interna
========================================

Q-Field es un espacio donde los agentes registran "amplitudes" internas
sin saber qué es mecánica cuántica. Solo observan:
- "Tengo varios estados posibles con diferentes pesos"
- "Cuando actúo, estos pesos interfieren de alguna forma"

Principios:
- NO introduce conocimiento externo (física cuántica, etc.)
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
class QState:
    """Estado Q con amplitudes internas."""
    t: int
    agent_id: str
    amplitudes: np.ndarray          # ψ(t) = [√p_1, √p_2, ..., √p_K]
    probabilities: np.ndarray       # p_j(t) = |ψ_j|²
    coherence: float                # C_Q(t)
    superposition_energy: float     # E_Q(t)


@dataclass
class QInterference:
    """Registro de interferencia entre estados."""
    t: int
    agent_id: str
    interference_matrix: np.ndarray  # Matriz de interferencia
    constructive: float              # Interferencia constructiva total
    destructive: float               # Interferencia destructiva total
    net_interference: float          # Neto


class QField:
    """
    Campo de interferencia interna.

    Los agentes mantienen "amplitudes" ψ_j(t) sobre K estados internos.
    Las probabilidades p_j(t) = |ψ_j(t)|² suman 1.

    Métricas emergentes:
    - Coherencia C_Q(t): qué tan correlacionadas están las amplitudes
    - Superposición E_Q(t): qué tan "mezclados" están los estados
    - Interferencia: cómo cambian las amplitudes al interactuar

    NO dice a los agentes qué hacer.
    Solo calcula estas estructuras.
    """

    def __init__(self):
        """Inicializa Q-Field."""
        self.t = 0

        # Historial de estados Q por agente
        self._q_states: Dict[str, List[QState]] = {}

        # Historial de interferencias
        self._interferences: Dict[str, List[QInterference]] = {}

        # Dimensión K (se determina del primer registro)
        self._K: Optional[int] = None

        # Estadísticas para umbrales endógenos
        self._coherence_history: List[float] = []
        self._energy_history: List[float] = []

    def register_state(
        self,
        agent_id: str,
        probabilities: np.ndarray
    ) -> QState:
        """
        Registra un estado Q a partir de probabilidades.

        Args:
            agent_id: Identificador del agente
            probabilities: Vector de probabilidades p_j(t)

        Returns:
            QState con amplitudes y métricas
        """
        self.t += 1

        # Normalizar probabilidades
        p = np.array(probabilities, dtype=float)
        p = np.clip(p, 0, None)  # No negativos
        p_sum = np.sum(p)
        if p_sum > np.finfo(float).eps:
            p = p / p_sum
        else:
            # Distribución uniforme si suma es cero
            p = np.ones_like(p) / len(p)

        # Establecer dimensión K
        if self._K is None:
            self._K = len(p)

        # Calcular amplitudes: ψ_j = √p_j
        amplitudes = np.sqrt(p)

        # Calcular coherencia C_Q(t)
        coherence = self._compute_coherence(agent_id, amplitudes)

        # Calcular energía de superposición E_Q(t) = Σ p_j(1-p_j)
        superposition_energy = float(np.sum(p * (1 - p)))

        # Crear estado Q
        q_state = QState(
            t=self.t,
            agent_id=agent_id,
            amplitudes=amplitudes.copy(),
            probabilities=p.copy(),
            coherence=coherence,
            superposition_energy=superposition_energy
        )

        # Inicializar historial si necesario
        if agent_id not in self._q_states:
            self._q_states[agent_id] = []
            self._interferences[agent_id] = []

        # Guardar estado
        self._q_states[agent_id].append(q_state)

        # Actualizar historial de estadísticas
        self._coherence_history.append(coherence)
        self._energy_history.append(superposition_energy)

        # Limitar historial endógenamente
        max_hist = self._get_max_history()
        if len(self._q_states[agent_id]) > max_hist:
            self._q_states[agent_id] = self._q_states[agent_id][-max_hist:]

        return q_state

    def _compute_coherence(
        self,
        agent_id: str,
        amplitudes: np.ndarray
    ) -> float:
        """
        Calcula coherencia C_Q(t) basada en covarianza de amplitudes históricas.

        C_Q(t) = Σ|cov(ψ_j, ψ_k)| / Σvar(ψ_j)

        Sin historial suficiente, usa entropía normalizada.
        """
        if agent_id not in self._q_states or len(self._q_states[agent_id]) < 2:
            # Sin historial: usar 1 - entropía normalizada
            p = amplitudes ** 2
            p = p[p > np.finfo(float).eps]
            if len(p) == 0:
                return 0.0
            entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(len(amplitudes)) if len(amplitudes) > 1 else 1.0
            return float(1 - entropy / (max_entropy + np.finfo(float).eps))

        # Con historial: calcular covarianzas
        history = [s.amplitudes for s in self._q_states[agent_id][-20:]]
        history.append(amplitudes)

        # Asegurar misma dimensión
        min_dim = min(len(h) for h in history)
        aligned = np.array([h[:min_dim] for h in history])

        if aligned.shape[0] < 2:
            return 0.0

        # Matriz de covarianza
        cov_matrix = np.cov(aligned.T)
        if cov_matrix.ndim == 0:
            return 0.0

        # C_Q = Σ|cov| / Σvar
        total_cov = np.sum(np.abs(cov_matrix)) - np.trace(np.abs(cov_matrix))
        total_var = np.trace(np.abs(cov_matrix))

        if total_var < np.finfo(float).eps:
            return 0.0

        coherence = total_cov / (total_var + np.finfo(float).eps)

        # Normalizar a [0, 1]
        K = cov_matrix.shape[0]
        max_coherence = K - 1 if K > 1 else 1
        coherence = coherence / max_coherence

        return float(np.clip(coherence, 0, 1))

    def compute_interference(
        self,
        agent_id: str,
        other_agent_id: str
    ) -> Optional[QInterference]:
        """
        Calcula interferencia entre dos agentes.

        Interferencia = ψ_A ⊗ ψ_B - (ψ_A ⊗ ψ_A + ψ_B ⊗ ψ_B) / 2

        Args:
            agent_id: Primer agente
            other_agent_id: Segundo agente

        Returns:
            QInterference o None si no hay estados
        """
        if agent_id not in self._q_states or other_agent_id not in self._q_states:
            return None

        if not self._q_states[agent_id] or not self._q_states[other_agent_id]:
            return None

        # Estados más recientes
        psi_A = self._q_states[agent_id][-1].amplitudes
        psi_B = self._q_states[other_agent_id][-1].amplitudes

        # Alinear dimensiones
        min_dim = min(len(psi_A), len(psi_B))
        psi_A = psi_A[:min_dim]
        psi_B = psi_B[:min_dim]

        # Productos externos
        AB = np.outer(psi_A, psi_B)
        AA = np.outer(psi_A, psi_A)
        BB = np.outer(psi_B, psi_B)

        # Matriz de interferencia
        interference = AB - (AA + BB) / 2

        # Componentes constructiva y destructiva
        constructive = float(np.sum(interference[interference > 0]))
        destructive = float(np.sum(np.abs(interference[interference < 0])))
        net = constructive - destructive

        q_interference = QInterference(
            t=self.t,
            agent_id=f"{agent_id}-{other_agent_id}",
            interference_matrix=interference,
            constructive=constructive,
            destructive=destructive,
            net_interference=net
        )

        # Guardar
        if agent_id not in self._interferences:
            self._interferences[agent_id] = []
        self._interferences[agent_id].append(q_interference)

        return q_interference

    def _get_max_history(self) -> int:
        """Calcula tamaño máximo de historial endógenamente."""
        total_states = sum(len(s) for s in self._q_states.values())
        if total_states < 100:
            return 100
        return max(100, int(np.sqrt(total_states) * 10))

    def get_field_state(self) -> Dict[str, Any]:
        """
        Retorna estado actual del campo Q.

        Incluye estadísticas agregadas de todos los agentes.
        """
        if not self._q_states:
            return {
                't': self.t,
                'n_agents': 0,
                'mean_coherence': 0.0,
                'mean_energy': 0.0,
                'field_entropy': 0.0
            }

        # Recolectar estados más recientes
        latest_states = []
        for agent_states in self._q_states.values():
            if agent_states:
                latest_states.append(agent_states[-1])

        if not latest_states:
            return {
                't': self.t,
                'n_agents': 0,
                'mean_coherence': 0.0,
                'mean_energy': 0.0,
                'field_entropy': 0.0
            }

        # Estadísticas
        coherences = [s.coherence for s in latest_states]
        energies = [s.superposition_energy for s in latest_states]

        # Entropía del campo (basada en distribución de coherencias)
        if len(coherences) > 1:
            c_array = np.array(coherences)
            c_array = c_array[c_array > np.finfo(float).eps]
            if len(c_array) > 0:
                c_norm = c_array / np.sum(c_array)
                field_entropy = float(-np.sum(c_norm * np.log(c_norm + np.finfo(float).eps)))
            else:
                field_entropy = 0.0
        else:
            field_entropy = 0.0

        return {
            't': self.t,
            'n_agents': len(latest_states),
            'mean_coherence': float(np.mean(coherences)),
            'std_coherence': float(np.std(coherences)) if len(coherences) > 1 else 0.0,
            'mean_energy': float(np.mean(energies)),
            'std_energy': float(np.std(energies)) if len(energies) > 1 else 0.0,
            'field_entropy': field_entropy
        }

    def get_agent_state(self, agent_id: str) -> Optional[QState]:
        """Retorna estado Q más reciente de un agente."""
        if agent_id not in self._q_states or not self._q_states[agent_id]:
            return None
        return self._q_states[agent_id][-1]

    def get_agent_history(self, agent_id: str) -> List[QState]:
        """Retorna historial de estados Q de un agente."""
        return self._q_states.get(agent_id, [])

    def get_coherence_threshold(self) -> float:
        """
        Calcula umbral de coherencia endógenamente.

        Basado en percentil del historial.
        """
        if len(self._coherence_history) < 10:
            # Sin historial: usar 1/2
            return 1/2

        return float(np.percentile(self._coherence_history, 75))

    def get_energy_threshold(self) -> float:
        """
        Calcula umbral de energía de superposición endógenamente.

        Basado en percentil del historial.
        """
        if len(self._energy_history) < 10:
            # Sin historial: usar 1/K si K conocido
            if self._K is not None:
                return 1 / self._K
            return 1/2

        return float(np.percentile(self._energy_history, 75))

    def measure_collapse(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Mide el "colapso" del estado Q (concentración de probabilidad).

        NO fuerza colapso, solo observa cuándo las probabilidades
        se concentran naturalmente.

        Returns:
            Dict con índice dominante y grado de colapso, o None
        """
        state = self.get_agent_state(agent_id)
        if state is None:
            return None

        p = state.probabilities

        # Índice dominante
        dominant_idx = int(np.argmax(p))
        dominant_prob = float(p[dominant_idx])

        # Grado de colapso: qué tan lejos está de uniforme
        # colapso = 0 si uniforme, 1 si delta
        K = len(p)
        uniform_prob = 1 / K

        # Distancia a uniforme (normalizada)
        collapse_degree = (dominant_prob - uniform_prob) / (1 - uniform_prob + np.finfo(float).eps)
        collapse_degree = float(np.clip(collapse_degree, 0, 1))

        return {
            'dominant_index': dominant_idx,
            'dominant_probability': dominant_prob,
            'collapse_degree': collapse_degree,
            'is_collapsed': collapse_degree > self.get_energy_threshold()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del sistema Q-Field."""
        field_state = self.get_field_state()

        # Historial de interferencias
        n_interferences = sum(len(i) for i in self._interferences.values())

        return {
            **field_state,
            'K': self._K,
            'total_states_recorded': sum(len(s) for s in self._q_states.values()),
            'n_interferences': n_interferences,
            'coherence_threshold': self.get_coherence_threshold(),
            'energy_threshold': self.get_energy_threshold()
        }
