"""
Coherence Surface: Capas L1-L4, L6 de Coherencia
=================================================

L1: Coherencia Cognitiva Interna (AGI-X)
L2: Coherencia Simbolica (SYM-X + STX)
L3: Coherencia Narrativa (episodic + narrative)
L4: Coherencia Vital (LX1-LX10)
L6: Coherencia Social (SX5 + AGI-19)

Cada coherencia mide que tan "integrado" esta un aspecto del agente.
Valores altos = estabilidad, continuidad, armonia.
Valores bajos = fragmentacion, conflicto, drift.

100% endogeno.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from abc import ABC

import sys
sys.path.insert(0, '/root/NEO_EVA')

from ellex_map.layer_emergence import ExistentialLayer, LayerType, LayerState
from cognition.agi_dynamic_constants import L_t, max_history


class CoherenceSurface(ExistentialLayer, ABC):
    """
    Clase base para superficies de coherencia.

    Una superficie de coherencia mide integracion
    en un dominio especifico del agente.
    """

    def __init__(self, agent_id: str, layer_type: LayerType):
        super().__init__(agent_id, layer_type)

        # Sub-componentes de coherencia
        self._component_histories: Dict[str, List[float]] = {}

    def _update_component(self, name: str, value: float):
        """Actualiza un sub-componente."""
        if name not in self._component_histories:
            self._component_histories[name] = []

        self._component_histories[name].append(value)

        # Recortar
        max_len = max_history(self.t)
        if len(self._component_histories[name]) > max_len:
            self._component_histories[name] = \
                self._component_histories[name][-max_len:]

        self._current_components[name] = value

    def _get_component_weight(self, name: str) -> float:
        """Obtiene peso endogeno de un componente."""
        if name not in self._component_histories:
            return 1.0

        history = self._component_histories[name]
        if len(history) < 3:
            return 1.0

        window = min(L_t(self.t), len(history))
        variance = np.var(history[-window:])

        return 1.0 / (variance + 0.01)

    def _aggregate_components(self) -> float:
        """Agrega componentes con pesos endogenos."""
        if not self._current_components:
            return 0.5

        weights = []
        values = []

        for name, value in self._current_components.items():
            weights.append(self._get_component_weight(name))
            values.append(value)

        total_weight = sum(weights)
        if total_weight < 1e-8:
            return np.mean(values)

        weights = [w / total_weight for w in weights]
        aggregated = sum(v * w for v, w in zip(values, weights))

        return float(np.clip(aggregated, 0, 1))


class CognitiveCoherence(CoherenceSurface):
    """
    L1: Coherencia Cognitiva Interna (AGI-X)

    Mide integracion de:
        - Continuidad del self (AGI-4)
        - Auto-modelo (AGI-4, AGI-20)
        - Teoria de la mente (AGI-5)
        - Regulacion teleologica (AGI-8)
        - Estabilidad de proyectos (AGI-6)
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id, LayerType.COGNITIVE)

    def compute(self, observations: Dict[str, Any]) -> float:
        """
        Calcula coherencia cognitiva.

        observations esperadas:
            - self_continuity: [0,1]
            - self_model_accuracy: [0,1]
            - tom_accuracy: [0,1]
            - drive_stability: [0,1]
            - prospection_coherence: [0,1]
        """
        # Extraer metricas
        self_continuity = observations.get('self_continuity', 0.5)
        self_model = observations.get('self_model_accuracy', 0.5)
        tom = observations.get('tom_accuracy', 0.5)
        drives = observations.get('drive_stability', 0.5)
        prospection = observations.get('prospection_coherence', 0.5)

        # Actualizar componentes
        self._update_component('self_continuity', self_continuity)
        self._update_component('self_model', self_model)
        self._update_component('tom', tom)
        self._update_component('drives', drives)
        self._update_component('prospection', prospection)

        # Agregar con pesos endogenos
        return self._aggregate_components()


class SymbolicCoherence(CoherenceSurface):
    """
    L2: Coherencia Simbolica (SYM-X + STX)

    Mide integracion de:
        - Drift conceptual (bajo = bueno)
        - Gramatica emergente
        - Madurez simbolica
        - Coordinacion multi-agente simbolica
        - Arquetipos dependientes de fase
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id, LayerType.SYMBOLIC)

    def compute(self, observations: Dict[str, Any]) -> float:
        """
        Calcula coherencia simbolica.

        observations esperadas:
            - conceptual_drift: [0,1] (invertido: bajo drift = alta coherencia)
            - grammar_coherence: [0,1]
            - symbolic_maturity: [0,1]
            - multiagent_symbolic_sync: [0,1]
            - phase_archetype_alignment: [0,1]
        """
        # Extraer metricas
        drift = observations.get('conceptual_drift', 0.5)
        grammar = observations.get('grammar_coherence', 0.5)
        maturity = observations.get('symbolic_maturity', 0.5)
        sync = observations.get('multiagent_symbolic_sync', 0.5)
        archetype = observations.get('phase_archetype_alignment', 0.5)

        # Invertir drift (bajo drift = alta coherencia)
        drift_coherence = 1 - drift

        # Actualizar componentes
        self._update_component('drift_coherence', drift_coherence)
        self._update_component('grammar', grammar)
        self._update_component('maturity', maturity)
        self._update_component('sync', sync)
        self._update_component('archetype', archetype)

        return self._aggregate_components()


class NarrativeCoherence(CoherenceSurface):
    """
    L3: Coherencia Narrativa

    Mide integracion de:
        - Continuidad episodica
        - Consistencia narrativa
        - Auto-historia estable
        - Puente narrativo entre fases
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id, LayerType.NARRATIVE)

    def compute(self, observations: Dict[str, Any]) -> float:
        """
        Calcula coherencia narrativa.

        observations esperadas:
            - episodic_continuity: [0,1]
            - narrative_consistency: [0,1]
            - self_history_stability: [0,1]
            - phase_narrative_bridge: [0,1]
        """
        episodic = observations.get('episodic_continuity', 0.5)
        narrative = observations.get('narrative_consistency', 0.5)
        history = observations.get('self_history_stability', 0.5)
        bridge = observations.get('phase_narrative_bridge', 0.5)

        self._update_component('episodic', episodic)
        self._update_component('narrative', narrative)
        self._update_component('history', history)
        self._update_component('bridge', bridge)

        return self._aggregate_components()


class LifeCoherence(CoherenceSurface):
    """
    L4: Coherencia Vital (LX1-LX10)

    Mide integracion del ciclo de vida:
        - Estabilidad del ciclo circadiano
        - Sueno eficaz
        - Integracion de vida
        - Maduracion ciclica
        - Resonancia inter-agente
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id, LayerType.LIFE)

    def compute(self, observations: Dict[str, Any]) -> float:
        """
        Calcula coherencia vital.

        observations esperadas:
            - circadian_stability: [0,1]
            - dream_efficacy: [0,1]
            - life_integration: [0,1]
            - cyclic_maturation: [0,1]
            - interagent_resonance: [0,1]
            - lx10_score: [0,1] (opcional, agregado LX)
        """
        circadian = observations.get('circadian_stability', 0.5)
        dream = observations.get('dream_efficacy', 0.5)
        integration = observations.get('life_integration', 0.5)
        maturation = observations.get('cyclic_maturation', 0.5)
        resonance = observations.get('interagent_resonance', 0.5)

        # Si hay score LX10, usarlo directamente
        lx10 = observations.get('lx10_score')
        if lx10 is not None:
            self._update_component('lx10', lx10)
            # Combinar LX10 con componentes individuales
            self._update_component('circadian', circadian)
            self._update_component('dream', dream)
            # Dar mas peso a LX10 agregado
            components_value = self._aggregate_components()
            return 0.6 * lx10 + 0.4 * components_value

        self._update_component('circadian', circadian)
        self._update_component('dream', dream)
        self._update_component('integration', integration)
        self._update_component('maturation', maturation)
        self._update_component('resonance', resonance)

        return self._aggregate_components()


class SocialCoherence(CoherenceSurface):
    """
    L6: Coherencia Social (SX5 + AGI-19)

    Mide integracion social:
        - Intencionalidad colectiva
        - Resonancia de metas
        - Cooperacion emergente
        - Simetria social
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id, LayerType.SOCIAL)

    def compute(self, observations: Dict[str, Any]) -> float:
        """
        Calcula coherencia social.

        observations esperadas:
            - collective_intentionality: [0,1]
            - goal_resonance: [0,1]
            - emergent_cooperation: [0,1]
            - social_symmetry: [0,1]
        """
        intentionality = observations.get('collective_intentionality', 0.5)
        resonance = observations.get('goal_resonance', 0.5)
        cooperation = observations.get('emergent_cooperation', 0.5)
        symmetry = observations.get('social_symmetry', 0.5)

        self._update_component('intentionality', intentionality)
        self._update_component('resonance', resonance)
        self._update_component('cooperation', cooperation)
        self._update_component('symmetry', symmetry)

        return self._aggregate_components()
