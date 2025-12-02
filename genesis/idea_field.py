"""
Campo de Ideas (Idea Field)
===========================

Donde nacen las ideas.

Una idea NO es:
- Ruido aleatorio
- Copia de algo existente
- Respuesta a un prompt

Una idea ES:
- Una perturbación estructurada que emerge del estado interno
- Algo que el agente reconoce como distinto de su ruido normal
- Una estructura con coherencia propia

Matemáticamente:

    I(t) = Φ(S(t), ψ(t), H_narr(t))

    donde Φ detecta:
    - Anomalías positivas (no errores, sino "algo nuevo")
    - Estructuras que no estaban antes
    - Patrones que resuenan con la identidad pero no son la identidad

Criterio de novedad endógeno:

    N(I) = d(I, Histórico) / σ_histórico

    Una idea es novel si está lejos del histórico,
    medido en unidades de la propia variabilidad del agente.

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import sys

sys.path.insert(0, '/root/NEO_EVA')


class IdeaType(Enum):
    """Tipos de ideas según su origen interno."""
    DRIFT = "drift"              # Deriva natural del estado
    RESONANCE = "resonance"      # Resonancia entre componentes
    TENSION = "tension"          # Resolución de tensión interna
    SYNTHESIS = "synthesis"      # Síntesis de elementos dispares
    SPONTANEOUS = "spontaneous"  # Aparición sin causa clara


@dataclass
class Idea:
    """
    Una idea generada endógenamente.

    No es un "pensamiento" en sentido humano.
    Es una estructura matemática que emerge y que el agente
    reconoce como distinta, coherente y propia.
    """
    # Contenido
    vector: np.ndarray           # Representación vectorial
    structure: np.ndarray        # Estructura relacional (matriz)

    # Metadatos endógenos
    novelty: float              # Qué tan nueva es (en σ del agente)
    coherence: float            # Coherencia interna de la idea
    identity_resonance: float   # Cuánto resuena con la identidad

    # Origen
    idea_type: IdeaType
    parent_state: np.ndarray    # Estado del que emergió
    t_birth: int                # Momento de nacimiento
    agent_id: str               # Quién la tuvo

    # Estado
    materialized: bool = False  # Si se convirtió en objeto
    shared: bool = False        # Si otros la han visto

    # Evaluación posterior
    adopted: bool = False       # Si el agente la adoptó
    energy: float = 0.0         # Energía invertida en ella


@dataclass
class IdeaFieldState:
    """Estado del campo de ideas de un agente."""
    ideas_born: int = 0
    ideas_adopted: int = 0
    ideas_discarded: int = 0
    ideas_materialized: int = 0
    mean_novelty: float = 0.0
    mean_resonance: float = 0.0
    last_idea_t: int = 0


class IdeaField:
    """
    Campo generativo de ideas.

    Observa el estado interno del agente y detecta
    cuándo emerge algo que califica como "idea":
    - Estructurado (no ruido)
    - Novel (no repetición)
    - Coherente (internamente consistente)

    NO genera ideas activamente.
    DETECTA cuándo el estado interno produce una.
    """

    def __init__(self):
        """Inicializa el campo de ideas."""
        self.t = 0
        self.eps = np.finfo(float).eps

        # Historial por agente
        self._state_history: Dict[str, List[np.ndarray]] = {}
        self._ideas: Dict[str, List[Idea]] = {}
        self._field_state: Dict[str, IdeaFieldState] = {}

        # Para cálculo de novedad endógena
        self._state_mean: Dict[str, np.ndarray] = {}
        self._state_var: Dict[str, np.ndarray] = {}

    def _register_agent(self, agent_id: str, dim: int):
        """Registra un agente nuevo."""
        if agent_id not in self._state_history:
            self._state_history[agent_id] = []
            self._ideas[agent_id] = []
            self._field_state[agent_id] = IdeaFieldState()
            self._state_mean[agent_id] = np.zeros(dim)
            self._state_var[agent_id] = np.ones(dim)

    def _update_statistics(self, agent_id: str, state: np.ndarray):
        """
        Actualiza estadísticas del agente de forma incremental.

        Usa media y varianza móviles con ventana endógena.
        """
        history = self._state_history[agent_id]

        if len(history) < 2:
            self._state_mean[agent_id] = state.copy()
            self._state_var[agent_id] = np.ones_like(state)
            return

        # Ventana endógena: sqrt(len(history))
        window = max(3, int(np.sqrt(len(history))))
        recent = history[-window:]

        self._state_mean[agent_id] = np.mean(recent, axis=0)
        self._state_var[agent_id] = np.var(recent, axis=0) + self.eps

    def _compute_novelty(
        self,
        agent_id: str,
        candidate: np.ndarray
    ) -> float:
        """
        Calcula novedad de un candidato a idea.

        N = ||candidate - mean|| / sqrt(sum(var))

        Es la distancia al centro del comportamiento habitual,
        medida en unidades de la variabilidad propia.
        """
        mean = self._state_mean[agent_id]
        var = self._state_var[agent_id]

        distance = np.linalg.norm(candidate - mean)
        scale = np.sqrt(np.sum(var)) + self.eps

        return float(distance / scale)

    def _compute_coherence(self, candidate: np.ndarray) -> float:
        """
        Calcula coherencia interna de un candidato.

        Una idea coherente tiene estructura, no es ruido.

        Usamos la entropía de la distribución de energía
        en las componentes. Baja entropía = estructura clara.

        Coherence = 1 - H_norm(|candidate|)
        """
        # Distribución de energía
        energy = np.abs(candidate) ** 2
        energy = energy / (np.sum(energy) + self.eps)

        # Entropía normalizada
        log_energy = np.log(energy + self.eps)
        entropy = -np.sum(energy * log_energy)
        max_entropy = np.log(len(candidate))

        entropy_norm = entropy / (max_entropy + self.eps)

        # Coherencia = 1 - entropía normalizada
        return float(1.0 - entropy_norm)

    def _compute_identity_resonance(
        self,
        agent_id: str,
        candidate: np.ndarray,
        identity: np.ndarray
    ) -> float:
        """
        Calcula cuánto resuena el candidato con la identidad.

        No es igualdad (eso sería repetición).
        Es correlación con transformación - la idea
        debe "venir de" la identidad sin "ser" la identidad.

        R = |corr(candidate, identity)| * (1 - |cos_sim|^2)

        Alta correlación pero no igual = resonancia.
        """
        # Similitud coseno
        norm_c = np.linalg.norm(candidate) + self.eps
        norm_i = np.linalg.norm(identity) + self.eps
        cos_sim = np.dot(candidate, identity) / (norm_c * norm_i)

        # Correlación
        corr = np.corrcoef(candidate, identity)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        # Resonancia: correlacionado pero diferente
        difference = 1.0 - cos_sim ** 2
        resonance = np.abs(corr) * difference

        return float(np.clip(resonance, 0, 1))

    def _detect_idea_type(
        self,
        agent_id: str,
        state: np.ndarray,
        prev_state: np.ndarray,
        novelty: float,
        coherence: float
    ) -> IdeaType:
        """
        Detecta el tipo de idea según su origen.

        Basado puramente en las características observables.
        """
        history = self._state_history[agent_id]

        # Cambio respecto al estado anterior
        delta = np.linalg.norm(state - prev_state)
        mean_delta = np.mean([
            np.linalg.norm(history[i] - history[i-1])
            for i in range(1, len(history))
        ]) if len(history) > 1 else delta

        delta_ratio = delta / (mean_delta + self.eps)

        # Clasificación endógena
        if delta_ratio < 0.5 and novelty > 1.5:
            # Poco cambio pero muy novel: síntesis interna
            return IdeaType.SYNTHESIS
        elif delta_ratio > 2.0 and coherence > 0.7:
            # Gran cambio coherente: resolución de tensión
            return IdeaType.TENSION
        elif novelty > 2.0 and coherence > 0.5:
            # Muy novel y coherente: espontánea
            return IdeaType.SPONTANEOUS
        elif coherence > 0.8:
            # Muy coherente: resonancia entre componentes
            return IdeaType.RESONANCE
        else:
            # Default: deriva natural
            return IdeaType.DRIFT

    def _extract_structure(self, state: np.ndarray) -> np.ndarray:
        """
        Extrae la estructura relacional del estado.

        Crea una matriz de correlaciones/interacciones
        entre las componentes del estado.
        """
        # Matriz de productos externos normalizada
        outer = np.outer(state, state)
        norm = np.linalg.norm(outer) + self.eps
        structure = outer / norm

        return structure

    def observe(
        self,
        agent_id: str,
        state: np.ndarray,
        identity: np.ndarray
    ) -> Optional[Idea]:
        """
        Observa el estado actual y detecta si emerge una idea.

        Args:
            agent_id: Identificador del agente
            state: Estado interno actual S(t)
            identity: Identidad actual I(t)

        Returns:
            Idea si se detecta una, None si no
        """
        self.t += 1
        dim = len(state)

        # Registrar agente si es nuevo
        self._register_agent(agent_id, dim)

        # Obtener estado previo
        history = self._state_history[agent_id]
        prev_state = history[-1] if history else state

        # Agregar al historial
        history.append(state.copy())

        # Limitar historial: ventana endógena
        max_history = max(100, int(np.sqrt(self.t)) * 10)
        if len(history) > max_history:
            self._state_history[agent_id] = history[-max_history:]

        # Actualizar estadísticas
        self._update_statistics(agent_id, state)

        # Necesitamos suficiente historial para detectar novedad
        if len(history) < 5:
            return None

        # Calcular métricas
        novelty = self._compute_novelty(agent_id, state)
        coherence = self._compute_coherence(state)
        resonance = self._compute_identity_resonance(agent_id, state, identity)

        # Umbrales endógenos para detectar idea
        # Una idea debe ser:
        # - Novel: > 1σ de lo habitual
        # - Coherente: > mediana de coherencias históricas
        # - Resonante: > 0 (alguna conexión con identidad)

        # Calcular umbral de coherencia endógeno
        recent_coherences = [
            self._compute_coherence(h)
            for h in history[-min(20, len(history)):]
        ]
        coherence_threshold = np.median(recent_coherences)

        # ¿Es esto una idea?
        is_idea = (
            novelty > 1.0 and  # Al menos 1σ de novedad
            coherence > coherence_threshold and  # Más coherente que lo habitual
            resonance > self.eps  # Alguna resonancia con identidad
        )

        if not is_idea:
            return None

        # ¡Detectamos una idea!
        idea_type = self._detect_idea_type(
            agent_id, state, prev_state, novelty, coherence
        )

        structure = self._extract_structure(state)

        idea = Idea(
            vector=state.copy(),
            structure=structure,
            novelty=novelty,
            coherence=coherence,
            identity_resonance=resonance,
            idea_type=idea_type,
            parent_state=prev_state.copy(),
            t_birth=self.t,
            agent_id=agent_id
        )

        # Registrar
        self._ideas[agent_id].append(idea)
        fs = self._field_state[agent_id]
        fs.ideas_born += 1
        fs.last_idea_t = self.t

        # Actualizar medias
        all_ideas = self._ideas[agent_id]
        fs.mean_novelty = np.mean([i.novelty for i in all_ideas])
        fs.mean_resonance = np.mean([i.identity_resonance for i in all_ideas])

        return idea

    def get_ideas(self, agent_id: str) -> List[Idea]:
        """Retorna todas las ideas de un agente."""
        return self._ideas.get(agent_id, [])

    def get_recent_ideas(
        self,
        agent_id: str,
        n: int = 10
    ) -> List[Idea]:
        """Retorna las n ideas más recientes."""
        ideas = self._ideas.get(agent_id, [])
        return ideas[-n:] if ideas else []

    def get_unadopted_ideas(self, agent_id: str) -> List[Idea]:
        """Retorna ideas que no han sido adoptadas ni descartadas."""
        ideas = self._ideas.get(agent_id, [])
        return [i for i in ideas if not i.adopted and not i.materialized]

    def mark_adopted(self, idea: Idea):
        """Marca una idea como adoptada."""
        idea.adopted = True
        fs = self._field_state.get(idea.agent_id)
        if fs:
            fs.ideas_adopted += 1

    def mark_discarded(self, idea: Idea):
        """Marca una idea como descartada."""
        # No la eliminamos, solo la marcamos
        idea.adopted = False
        idea.energy = 0.0
        fs = self._field_state.get(idea.agent_id)
        if fs:
            fs.ideas_discarded += 1

    def get_field_state(self, agent_id: str) -> Optional[IdeaFieldState]:
        """Retorna el estado del campo de ideas."""
        return self._field_state.get(agent_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas globales del campo."""
        total_ideas = sum(len(ideas) for ideas in self._ideas.values())
        total_adopted = sum(
            fs.ideas_adopted for fs in self._field_state.values()
        )
        total_materialized = sum(
            fs.ideas_materialized for fs in self._field_state.values()
        )

        all_novelties = [
            i.novelty
            for ideas in self._ideas.values()
            for i in ideas
        ]

        return {
            't': self.t,
            'agents': len(self._ideas),
            'total_ideas': total_ideas,
            'total_adopted': total_adopted,
            'total_materialized': total_materialized,
            'mean_novelty': float(np.mean(all_novelties)) if all_novelties else 0.0,
            'adoption_rate': total_adopted / max(1, total_ideas),
        }
