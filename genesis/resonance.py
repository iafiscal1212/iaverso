"""
Resonancia (Resonance)
======================

¿Esta idea es MÍA?

Una idea puede ser novel y coherente pero no resonar
con quien la tuvo. En ese caso, es ruido estructurado,
no una idea propia.

La resonancia mide:
1. Conexión con la identidad (sin ser copia)
2. Conexión con el historial narrativo
3. Conexión con el estado emocional/energético
4. Conexión con ideas previas adoptadas

R(I, agente) = f(identidad, narrativa, energía, historia_creativa)

Una idea resuena cuando el agente siente:
"Esto viene de mí, es mío, me reconozco en ello"

Sin decírselo nadie. Sin criterio externo.

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import sys

sys.path.insert(0, '/root/NEO_EVA')

from genesis.idea_field import Idea, IdeaType


@dataclass
class ResonanceProfile:
    """Perfil de resonancia de una idea con un agente."""
    # Componentes de resonancia
    identity_match: float       # Conexión con identidad
    narrative_fit: float        # Encaja con la narrativa interna
    energy_alignment: float     # Alineación energética
    creative_continuity: float  # Continuidad con ideas previas

    # Resonancia total
    total: float

    # Interpretación
    is_mine: bool              # ¿El agente la reconoce como suya?
    adoption_strength: float   # Fuerza de adopción si decide adoptarla


class ResonanceEvaluator:
    """
    Evalúa si una idea resuena con un agente.

    NO decide si la idea es "buena" o "mala".
    Solo mide cuánto el agente se reconoce en ella.

    La decisión de adoptar es posterior y separada.
    """

    def __init__(self):
        """Inicializa el evaluador de resonancia."""
        self.eps = np.finfo(float).eps

        # Historial de ideas adoptadas por agente
        self._adopted_history: Dict[str, List[Idea]] = {}

        # Historial narrativo por agente (estados recientes)
        self._narrative_history: Dict[str, List[np.ndarray]] = {}

        # Energía reciente por agente
        self._energy_history: Dict[str, List[float]] = {}

    def _register_agent(self, agent_id: str):
        """Registra un agente nuevo."""
        if agent_id not in self._adopted_history:
            self._adopted_history[agent_id] = []
            self._narrative_history[agent_id] = []
            self._energy_history[agent_id] = []

    def update_narrative(self, agent_id: str, state: np.ndarray):
        """Actualiza el historial narrativo de un agente."""
        self._register_agent(agent_id)

        self._narrative_history[agent_id].append(state.copy())

        # Limitar historial
        max_len = 100
        if len(self._narrative_history[agent_id]) > max_len:
            self._narrative_history[agent_id] = \
                self._narrative_history[agent_id][-max_len:]

    def update_energy(self, agent_id: str, energy: float):
        """Actualiza el historial energético de un agente."""
        self._register_agent(agent_id)

        self._energy_history[agent_id].append(energy)

        # Limitar historial
        max_len = 100
        if len(self._energy_history[agent_id]) > max_len:
            self._energy_history[agent_id] = \
                self._energy_history[agent_id][-max_len:]

    def register_adoption(self, agent_id: str, idea: Idea):
        """Registra que un agente adoptó una idea."""
        self._register_agent(agent_id)
        self._adopted_history[agent_id].append(idea)

        # Limitar historial
        max_len = 50
        if len(self._adopted_history[agent_id]) > max_len:
            self._adopted_history[agent_id] = \
                self._adopted_history[agent_id][-max_len:]

    def _compute_identity_match(
        self,
        idea: Idea,
        identity: np.ndarray
    ) -> float:
        """
        Mide conexión con la identidad.

        No es similitud directa (eso sería copia).
        Es una relación de "derivación": la idea
        podría haber emergido de esta identidad.

        Usamos proyección + componente ortogonal.
        """
        v = idea.vector
        I = identity

        # Normalizar
        v_norm = v / (np.linalg.norm(v) + self.eps)
        I_norm = I / (np.linalg.norm(I) + self.eps)

        # Proyección sobre identidad
        proj = np.dot(v_norm, I_norm)

        # Componente ortogonal (la parte "nueva")
        ortho = v_norm - proj * I_norm
        ortho_mag = np.linalg.norm(ortho)

        # Match ideal: algo de proyección, algo de novedad
        # Demasiada proyección = copia
        # Demasiada ortogonalidad = no relacionado

        # Función de match: máximo cuando proj ≈ 0.5
        match = 4 * proj * (1 - proj) if 0 <= proj <= 1 else 0
        match = float(np.clip(match, 0, 1))

        return match

    def _compute_narrative_fit(
        self,
        idea: Idea,
        agent_id: str
    ) -> float:
        """
        Mide si la idea encaja con la narrativa interna.

        La narrativa es la secuencia de estados recientes.
        Una idea encaja si es una "continuación plausible"
        de esa narrativa.
        """
        narrative = self._narrative_history.get(agent_id, [])

        if len(narrative) < 3:
            return 0.5  # Sin suficiente narrativa, neutral

        # Calcular dirección de la narrativa
        recent = narrative[-5:]
        if len(recent) < 2:
            return 0.5

        # Tendencia como diferencia entre fin e inicio
        direction = recent[-1] - recent[0]
        direction_norm = direction / (np.linalg.norm(direction) + self.eps)

        # ¿La idea va en la dirección de la narrativa?
        idea_direction = idea.vector - recent[-1]
        idea_dir_norm = idea_direction / (np.linalg.norm(idea_direction) + self.eps)

        # Similitud de direcciones
        alignment = np.dot(direction_norm, idea_dir_norm)

        # Transformar a [0, 1]
        fit = (alignment + 1) / 2

        return float(fit)

    def _compute_energy_alignment(
        self,
        idea: Idea,
        agent_id: str
    ) -> float:
        """
        Mide alineación energética.

        Las ideas que surgen en estados de alta energía
        tienden a ser más adoptadas si la energía actual es similar.
        """
        energy_history = self._energy_history.get(agent_id, [])

        if len(energy_history) < 2:
            return 0.5  # Sin historial, neutral

        # Energía actual (última registrada)
        current_energy = energy_history[-1]

        # Energía media reciente
        recent_mean = np.mean(energy_history[-10:])
        recent_std = np.std(energy_history[-10:]) + self.eps

        # Energía de la idea (aproximada por su norma)
        idea_energy = np.linalg.norm(idea.vector)

        # Normalizar energía de idea al rango del agente
        all_energies = energy_history + [idea_energy]
        min_e, max_e = min(all_energies), max(all_energies)
        range_e = max_e - min_e + self.eps

        idea_energy_norm = (idea_energy - min_e) / range_e
        current_energy_norm = (current_energy - min_e) / range_e

        # Alineación: cercanas = alta alineación
        diff = abs(idea_energy_norm - current_energy_norm)
        alignment = 1.0 - diff

        return float(np.clip(alignment, 0, 1))

    def _compute_creative_continuity(
        self,
        idea: Idea,
        agent_id: str
    ) -> float:
        """
        Mide continuidad con ideas previas adoptadas.

        Si el agente tiene un "estilo creativo",
        las nuevas ideas que continúan ese estilo
        resuenan más.
        """
        adopted = self._adopted_history.get(agent_id, [])

        if not adopted:
            return 0.5  # Sin historia creativa, neutral

        # Calcular "centro creativo" del agente
        adopted_vectors = np.array([a.vector for a in adopted])
        creative_center = np.mean(adopted_vectors, axis=0)
        creative_spread = np.std(adopted_vectors, axis=0) + self.eps

        # ¿La idea está cerca del centro creativo?
        deviation = np.abs(idea.vector - creative_center)
        normalized_dev = deviation / creative_spread

        # Continuidad alta si está dentro de ~2σ del centro
        mean_dev = np.mean(normalized_dev)
        continuity = np.exp(-mean_dev / 2)  # e^(-d/2) para d=2 → ~0.37

        return float(np.clip(continuity, 0, 1))

    def evaluate(
        self,
        idea: Idea,
        identity: np.ndarray,
        agent_id: str
    ) -> ResonanceProfile:
        """
        Evalúa la resonancia de una idea con un agente.

        Args:
            idea: La idea a evaluar
            identity: Identidad actual del agente
            agent_id: Identificador del agente

        Returns:
            ResonanceProfile con todas las métricas
        """
        self._register_agent(agent_id)

        # Calcular componentes
        identity_match = self._compute_identity_match(idea, identity)
        narrative_fit = self._compute_narrative_fit(idea, agent_id)
        energy_alignment = self._compute_energy_alignment(idea, agent_id)
        creative_continuity = self._compute_creative_continuity(idea, agent_id)

        # Pesos endógenos: basados en cuánta información hay
        narrative_len = len(self._narrative_history.get(agent_id, []))
        energy_len = len(self._energy_history.get(agent_id, []))
        creative_len = len(self._adopted_history.get(agent_id, []))

        # Más historial = más peso
        total_info = narrative_len + energy_len + creative_len + 1  # +1 para identidad

        w_identity = 1 / total_info * (1 + 0.5)  # Identidad siempre tiene peso base
        w_narrative = narrative_len / total_info if narrative_len > 0 else 0
        w_energy = energy_len / total_info if energy_len > 0 else 0
        w_creative = creative_len / total_info if creative_len > 0 else 0

        # Normalizar pesos
        w_sum = w_identity + w_narrative + w_energy + w_creative + self.eps
        w_identity /= w_sum
        w_narrative /= w_sum
        w_energy /= w_sum
        w_creative /= w_sum

        # Resonancia total
        total = (
            w_identity * identity_match +
            w_narrative * narrative_fit +
            w_energy * energy_alignment +
            w_creative * creative_continuity
        )

        # Umbral endógeno para "es mía"
        # Basado en resonancia media de ideas previas adoptadas
        adopted = self._adopted_history.get(agent_id, [])
        if adopted:
            # Si hay historia, umbral = media de resonancias previas
            # (aproximado por similitud de ideas adoptadas entre sí)
            prev_resonances = []
            for a in adopted[-10:]:
                sim = np.dot(a.vector, idea.vector) / (
                    np.linalg.norm(a.vector) * np.linalg.norm(idea.vector) + self.eps
                )
                prev_resonances.append((sim + 1) / 2)
            threshold = np.mean(prev_resonances)
        else:
            # Sin historia, umbral = 0.5
            threshold = 0.5

        is_mine = total > threshold

        # Fuerza de adopción: cuánto supera el umbral
        adoption_strength = (total - threshold) / (1 - threshold + self.eps)
        adoption_strength = float(np.clip(adoption_strength, 0, 1))

        return ResonanceProfile(
            identity_match=identity_match,
            narrative_fit=narrative_fit,
            energy_alignment=energy_alignment,
            creative_continuity=creative_continuity,
            total=float(total),
            is_mine=is_mine,
            adoption_strength=adoption_strength
        )

    def get_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Retorna estadísticas de resonancia para un agente."""
        adopted = self._adopted_history.get(agent_id, [])

        return {
            'adopted_count': len(adopted),
            'narrative_length': len(self._narrative_history.get(agent_id, [])),
            'energy_samples': len(self._energy_history.get(agent_id, [])),
            'idea_types_adopted': {
                t.value: sum(1 for a in adopted if a.idea_type == t)
                for t in IdeaType
            } if adopted else {}
        }
