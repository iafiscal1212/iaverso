"""
Narrative Waveform: Complemento para L3
========================================

Analiza la "onda" narrativa del agente:
    - Arcos narrativos
    - Coherencia temporal
    - Resonancia entre episodios

100% endogeno.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class WaveformState:
    """Estado de la onda narrativa."""
    coherence: float            # [0, 1] coherencia de onda
    arc_completeness: float     # Completitud de arcos
    temporal_flow: float        # Flujo temporal
    episode_resonance: float    # Resonancia entre episodios
    narrative_momentum: float   # Momentum narrativo
    t: int


class NarrativeWaveform:
    """
    Analiza la estructura de onda de la narrativa del agente.

    Una narrativa "sana" tiene:
        - Arcos que se completan
        - Flujo temporal coherente
        - Resonancia entre episodios (temas recurrentes)
        - Momentum (la historia avanza)
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.t = 0

        # Historial de episodios
        self._episode_history: List[Dict] = []
        self._arc_history: List[Dict] = []
        self._coherence_history: List[float] = []

    def add_episode(
        self,
        content: str,
        themes: List[str],
        emotional_valence: float,
        significance: float
    ):
        """Agrega un episodio a la narrativa."""
        episode = {
            't': self.t,
            'content': content,
            'themes': themes,
            'valence': emotional_valence,
            'significance': significance
        }
        self._episode_history.append(episode)

        max_len = max_history(self.t)
        if len(self._episode_history) > max_len:
            self._episode_history = self._episode_history[-max_len:]

    def _compute_arc_completeness(self) -> float:
        """
        Calcula completitud de arcos narrativos.

        Un arco tiene: inicio -> desarrollo -> resolucion.
        """
        if len(self._episode_history) < 3:
            return 0.5

        # Analizar valencias: un arco es cambio-retorno
        valences = [ep['valence'] for ep in self._episode_history]

        # Detectar patrones de arco (subida-bajada o bajada-subida)
        arcs_found = 0
        arcs_complete = 0

        window = min(L_t(self.t), len(valences))
        for i in range(window - 2):
            v1, v2, v3 = valences[i], valences[i+1], valences[i+2]

            # Arco positivo: sube y baja
            if v2 > v1 and v3 < v2:
                arcs_found += 1
                if abs(v3 - v1) < 0.3:  # Vuelve cerca del inicio
                    arcs_complete += 1

            # Arco negativo: baja y sube
            if v2 < v1 and v3 > v2:
                arcs_found += 1
                if abs(v3 - v1) < 0.3:
                    arcs_complete += 1

        if arcs_found == 0:
            return 0.5

        return arcs_complete / arcs_found

    def _compute_temporal_flow(self) -> float:
        """
        Calcula fluidez del flujo temporal.

        Buen flujo: significancias distribuidas, no todo junto.
        """
        if len(self._episode_history) < 3:
            return 0.5

        # Calcular intervalos entre episodios significativos
        sig_times = [ep['t'] for ep in self._episode_history if ep['significance'] > 0.5]

        if len(sig_times) < 2:
            return 0.5

        intervals = [sig_times[i+1] - sig_times[i] for i in range(len(sig_times)-1)]

        # Buen flujo = intervalos regulares (baja varianza relativa)
        if not intervals:
            return 0.5

        mean_int = np.mean(intervals)
        std_int = np.std(intervals)

        cv = std_int / (mean_int + 1e-8)
        flow = 1 / (1 + cv)

        return float(np.clip(flow, 0, 1))

    def _compute_episode_resonance(self) -> float:
        """
        Calcula resonancia tematica entre episodios.

        Alta resonancia = temas recurrentes (coherencia).
        """
        if len(self._episode_history) < 3:
            return 0.5

        # Contar frecuencia de temas
        from collections import Counter
        all_themes = []
        for ep in self._episode_history:
            all_themes.extend(ep.get('themes', []))

        if not all_themes:
            return 0.5

        theme_counts = Counter(all_themes)
        total = len(all_themes)

        # Temas que aparecen multiples veces = resonancia
        recurring = sum(1 for count in theme_counts.values() if count > 1)
        total_themes = len(theme_counts)

        if total_themes == 0:
            return 0.5

        resonance = recurring / total_themes

        return float(np.clip(resonance, 0, 1))

    def _compute_momentum(self) -> float:
        """
        Calcula momentum narrativo.

        Momentum = la historia avanza (no se estanca).
        """
        if len(self._episode_history) < 3:
            return 0.5

        # Momentum = variedad de contenido reciente
        window = min(L_t(self.t), len(self._episode_history))
        recent = self._episode_history[-window:]

        # Diversidad de temas recientes
        recent_themes = set()
        for ep in recent:
            recent_themes.update(ep.get('themes', []))

        # Diversidad de valencias
        valences = [ep['valence'] for ep in recent]
        valence_range = max(valences) - min(valences) if valences else 0

        # Momentum = diversidad * rango
        theme_diversity = len(recent_themes) / (window + 1)
        momentum = (theme_diversity + valence_range) / 2

        return float(np.clip(momentum, 0, 1))

    def compute(self, observations: Dict[str, Any] = None) -> WaveformState:
        """
        Calcula estado de la onda narrativa.

        observations opcionales:
            - new_episode: Dict con episodio a agregar
        """
        self.t += 1

        # Agregar episodio si hay
        if observations and 'new_episode' in observations:
            ep = observations['new_episode']
            self.add_episode(
                content=ep.get('content', ''),
                themes=ep.get('themes', []),
                emotional_valence=ep.get('valence', 0.0),
                significance=ep.get('significance', 0.5)
            )

        # Calcular componentes
        arc = self._compute_arc_completeness()
        flow = self._compute_temporal_flow()
        resonance = self._compute_episode_resonance()
        momentum = self._compute_momentum()

        # Coherencia general
        coherence = np.mean([arc, flow, resonance, momentum])
        self._coherence_history.append(coherence)

        max_len = max_history(self.t)
        if len(self._coherence_history) > max_len:
            self._coherence_history = self._coherence_history[-max_len:]

        return WaveformState(
            coherence=float(coherence),
            arc_completeness=float(arc),
            temporal_flow=float(flow),
            episode_resonance=float(resonance),
            narrative_momentum=float(momentum),
            t=self.t
        )
