"""
ELLEX-MAP: Existential Life Layer Explorer
==========================================

Orquestador principal del mapa existencial.

Combina todas las capas L1-L10 en una vision integrada
del estado existencial del agente.

100% endogeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import sys
sys.path.insert(0, '/root/NEO_EVA')

from ellex_map.layer_emergence import ExistentialLayer, LayerType, LayerState
from ellex_map.coherence_surface import (
    CognitiveCoherence, SymbolicCoherence, NarrativeCoherence,
    LifeCoherence, SocialCoherence
)
from ellex_map.existential_tension import ExistentialTension
from ellex_map.health_equilibrium import HealthEquilibrium
from ellex_map.circadian_phase_space import CircadianPhaseSpace
from ellex_map.ellex_index import ELLEXIndex, ELLEXState
from ellex_map.symbolic_cohesion import SymbolicCohesion
from ellex_map.narrative_waveform import NarrativeWaveform

from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class IdentityState:
    """Estado de identidad persistente (L8)."""
    identity: float             # [0, 1] coherencia de identidad
    core_stability: float       # Estabilidad de valores core
    narrative_continuity: float # Continuidad narrativa
    behavioral_consistency: float  # Consistencia conductual
    t: int


@dataclass
class ELLEXMapState:
    """Estado completo del mapa ELLEX."""
    # Indice total
    ellex: float
    ellex_state: ELLEXState

    # Capas individuales
    L1_cognitive: float
    L2_symbolic: float
    L3_narrative: float
    L4_life: float
    L5_health: float
    L6_social: float
    L7_tension: float
    L8_identity: float
    L9_phase: float

    # Estados detallados
    tension_zone: str           # 'stagnant', 'healthy', 'crisis'
    health_status: str          # 'unhealthy', 'recovering', 'healthy'
    existential_zone: str       # 'struggling', 'balanced', 'flourishing'

    # Metricas agregadas
    coherence_mean: float       # Promedio de coherencias
    stability: float            # Estabilidad general
    trend: float               # Tendencia general

    t: int


class IdentityLayer(ExistentialLayer):
    """
    L8: Identidad Persistente

    Mide la estabilidad de la identidad del agente:
        - Valores core estables
        - Narrativa continua
        - Comportamiento consistente
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id, LayerType.IDENTIDAD)

        self._core_values_history: List[Dict[str, float]] = []
        self._narrative_history: List[str] = []
        self._behavior_history: List[str] = []

    def _compute_core_stability(self) -> float:
        """Calcula estabilidad de valores core."""
        if len(self._core_values_history) < 3:
            return 0.5

        window = L_t(self.t)
        recent = self._core_values_history[-window:]

        # Calcular varianza de cada valor core
        all_keys = set()
        for values in recent:
            all_keys.update(values.keys())

        if not all_keys:
            return 0.5

        stabilities = []
        for key in all_keys:
            key_values = [v.get(key, 0.5) for v in recent]
            std = np.std(key_values)
            stability = 1 / (1 + std * 2)
            stabilities.append(stability)

        return float(np.mean(stabilities))

    def _compute_narrative_continuity(self) -> float:
        """Calcula continuidad narrativa."""
        if len(self._narrative_history) < 3:
            return 0.5

        # Medir cuantos elementos narrativos se repiten
        window = L_t(self.t)
        recent = self._narrative_history[-window:]

        from collections import Counter
        counts = Counter(recent)

        if not counts:
            return 0.5

        # Continuidad = proporcion de elementos recurrentes
        recurring = sum(1 for c in counts.values() if c > 1)
        total = len(counts)

        return recurring / total if total > 0 else 0.5

    def _compute_behavioral_consistency(self) -> float:
        """Calcula consistencia conductual."""
        if len(self._behavior_history) < 3:
            return 0.5

        window = L_t(self.t)
        recent = self._behavior_history[-window:]

        from collections import Counter
        counts = Counter(recent)

        if not counts:
            return 0.5

        # Consistencia = entropia inversa normalizada
        total = len(recent)
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
        max_entropy = np.log2(len(counts) + 1)

        if max_entropy > 0:
            norm_entropy = entropy / max_entropy
            consistency = 1 - norm_entropy
        else:
            consistency = 1.0

        return float(np.clip(consistency, 0, 1))

    def calcular(self, observaciones: Dict[str, Any]) -> float:
        """Alias para compute."""
        return self.compute(observaciones)

    def compute(self, observations: Dict[str, Any]) -> float:
        """
        Calcula identidad persistente.

        observations esperadas:
            - core_values: Dict[str, float] - valores core actuales
            - narrative_element: str - elemento narrativo actual
            - behavior: str - comportamiento actual
        """
        # Extraer observaciones
        core_values = observations.get('core_values', {})
        narrative = observations.get('narrative_element', '')
        behavior = observations.get('behavior', '')

        # Actualizar historiales
        if core_values:
            self._core_values_history.append(core_values)
            max_len = max_history(self.t)
            if len(self._core_values_history) > max_len:
                self._core_values_history = self._core_values_history[-max_len:]

        if narrative:
            self._narrative_history.append(narrative)
            max_len = max_history(self.t)
            if len(self._narrative_history) > max_len:
                self._narrative_history = self._narrative_history[-max_len:]

        if behavior:
            self._behavior_history.append(behavior)
            max_len = max_history(self.t)
            if len(self._behavior_history) > max_len:
                self._behavior_history = self._behavior_history[-max_len:]

        # Calcular componentes
        core_stability = self._compute_core_stability()
        narrative_cont = self._compute_narrative_continuity()
        behavioral_cons = self._compute_behavioral_consistency()

        # Agregar con pesos endogenos (varianza inversa simplificada)
        components = [core_stability, narrative_cont, behavioral_cons]
        identity = np.mean(components)  # Simplificado: igual peso

        identity = float(np.clip(identity, 0, 1))

        self._current_components = {
            'core_stability': core_stability,
            'narrative_continuity': narrative_cont,
            'behavioral_consistency': behavioral_cons
        }

        return identity

    def get_identity_state(self) -> IdentityState:
        """Obtiene estado de identidad."""
        return IdentityState(
            identity=self._current_value,
            core_stability=self._current_components.get('core_stability', 0.5),
            narrative_continuity=self._current_components.get('narrative_continuity', 0.5),
            behavioral_consistency=self._current_components.get('behavioral_consistency', 0.5),
            t=self.t
        )


class ELLEXMap:
    """
    ELLEX-MAP: Mapa Existencial Completo

    Orquesta todas las capas existenciales para dar una
    vision integrada del estado del agente.

    Capas:
        L1: Coherencia Cognitiva
        L2: Coherencia Simbolica
        L3: Coherencia Narrativa
        L4: Coherencia de Vida
        L5: Salud Interior
        L6: Coherencia Social
        L7: Tension Existencial
        L8: Identidad Persistente
        L9: Equilibrio de Fases
        L10: ELLEX Index (integracion)
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.t = 0

        # Inicializar todas las capas
        self.L1 = CognitiveCoherence(agent_id)
        self.L2 = SymbolicCoherence(agent_id)
        self.L3 = NarrativeCoherence(agent_id)
        self.L4 = LifeCoherence(agent_id)
        self.L5 = HealthEquilibrium(agent_id)
        self.L6 = SocialCoherence(agent_id)
        self.L7 = ExistentialTension(agent_id)
        self.L8 = IdentityLayer(agent_id)
        self.L9 = CircadianPhaseSpace(agent_id)
        self.L10 = ELLEXIndex(agent_id)

        # Complementos
        self._symbolic_cohesion = SymbolicCohesion(agent_id)
        self._narrative_waveform = NarrativeWaveform(agent_id)

        # Historial de estados
        self._state_history: List[ELLEXMapState] = []

    def update(self, observations: Dict[str, Any]) -> ELLEXMapState:
        """
        Actualiza todo el mapa ELLEX.

        observations puede incluir:
            # Para L1 (Cognitiva)
            - cognitive_load: float
            - attention_focus: float
            - memory_coherence: float

            # Para L2 (Simbolica)
            - active_concepts: Dict[str, float]
            - connections: List[Tuple[str, str]]

            # Para L3 (Narrativa)
            - new_episode: Dict con {content, themes, valence, significance}

            # Para L4 (Vida)
            - drives: Dict[str, float]
            - goals: Dict[str, float]
            - environment_fit: float

            # Para L5 (Salud)
            - diagnosis_quality, treatment_efficacy, iatrogenesis_rate, etc.

            # Para L6 (Social)
            - social_connections: List[str]
            - interaction_quality: float

            # Para L7 (Tension)
            - drives, goals, stress, transitions

            # Para L8 (Identidad)
            - core_values: Dict[str, float]
            - narrative_element: str
            - behavior: str

            # Para L9 (Fases)
            - current_phase, phase_efficacy, multiagent_sync
        """
        self.t += 1

        # Actualizar complementos
        symbolic_obs = {
            'active_concepts': observations.get('active_concepts', {}),
            'connections': observations.get('connections', [])
        }
        self._symbolic_cohesion.compute(symbolic_obs)

        narrative_obs = {}
        if 'new_episode' in observations:
            narrative_obs['new_episode'] = observations['new_episode']
        self._narrative_waveform.compute(narrative_obs)

        # Calcular cada capa
        # L1: Cognitiva
        l1_obs = {
            'cognitive_load': observations.get('cognitive_load', 0.5),
            'attention_focus': observations.get('attention_focus', 0.5),
            'memory_coherence': observations.get('memory_coherence', 0.5)
        }
        l1_value = self.L1.update(l1_obs).value

        # L2: Simbolica (usa complemento)
        l2_obs = {
            'active_concepts': observations.get('active_concepts', {}),
            'concept_stability': self._symbolic_cohesion._compute_concept_stability()
            if hasattr(self._symbolic_cohesion, '_compute_concept_stability') else 0.5,
            'connection_density': 0.5
        }
        l2_value = self.L2.update(l2_obs).value

        # L3: Narrativa (usa complemento)
        l3_obs = {
            'arc_completeness': 0.5,
            'temporal_flow': 0.5,
            'episode_resonance': 0.5
        }
        if hasattr(self._narrative_waveform, '_coherence_history'):
            if self._narrative_waveform._coherence_history:
                l3_obs['arc_completeness'] = self._narrative_waveform._compute_arc_completeness()
                l3_obs['temporal_flow'] = self._narrative_waveform._compute_temporal_flow()
                l3_obs['episode_resonance'] = self._narrative_waveform._compute_episode_resonance()
        l3_value = self.L3.update(l3_obs).value

        # L4: Vida
        l4_obs = {
            'drives': observations.get('drives', {}),
            'goals': observations.get('goals', {}),
            'environment_fit': observations.get('environment_fit', 0.5)
        }
        l4_value = self.L4.update(l4_obs).value

        # L5: Salud
        l5_obs = {
            'diagnosis_quality': observations.get('diagnosis_quality', 0.5),
            'treatment_efficacy': observations.get('treatment_efficacy', 0.5),
            'iatrogenesis_rate': observations.get('iatrogenesis_rate', 0.0),
            'rotation_health': observations.get('rotation_health', 0.5),
            'health_history': observations.get('health_history', []),
            'stress_events': observations.get('stress_events', [])
        }
        l5_value = self.L5.update(l5_obs).value

        # L6: Social
        l6_obs = {
            'social_connections': observations.get('social_connections', []),
            'interaction_quality': observations.get('interaction_quality', 0.5),
            'trust_levels': observations.get('trust_levels', {})
        }
        l6_value = self.L6.update(l6_obs).value

        # L7: Tension
        l7_obs = {
            'drives': observations.get('drives', []),
            'goals': observations.get('goals', []),
            'stress': observations.get('stress', 0.0),
            'transitions': observations.get('transitions', [])
        }
        l7_value = self.L7.update(l7_obs).value

        # L8: Identidad
        l8_obs = {
            'core_values': observations.get('core_values', {}),
            'narrative_element': observations.get('narrative_element', ''),
            'behavior': observations.get('behavior', '')
        }
        l8_value = self.L8.update(l8_obs).value

        # L9: Fases
        l9_obs = {
            'current_phase': observations.get('current_phase', 'wake'),
            'phase_efficacy': observations.get('phase_efficacy', 0.5),
            'multiagent_sync': observations.get('multiagent_sync', 0.5)
        }
        l9_value = self.L9.update(l9_obs).value

        # L10: ELLEX Index
        l10_obs = {
            'L1_cognitive': l1_value,
            'L2_symbolic': l2_value,
            'L3_narrative': l3_value,
            'L4_life': l4_value,
            'L5_health': l5_value,
            'L6_social': l6_value,
            'L7_tension': l7_value,
            'L8_identity': l8_value,
            'L9_phase': l9_value
        }
        ellex_value = self.L10.update(l10_obs).value

        # Obtener estados detallados
        tension_state = self.L7.get_tension_state()
        ellex_state = self.L10.get_ellex_state()

        # Determinar status de salud
        if l5_value < 0.4:
            health_status = 'unhealthy'
        elif l5_value < 0.6:
            health_status = 'recovering'
        else:
            health_status = 'healthy'

        # Calcular metricas agregadas
        coherences = [l1_value, l2_value, l3_value, l4_value, l6_value]
        coherence_mean = np.mean(coherences)

        # Crear estado completo
        state = ELLEXMapState(
            ellex=ellex_value,
            ellex_state=ellex_state,
            L1_cognitive=l1_value,
            L2_symbolic=l2_value,
            L3_narrative=l3_value,
            L4_life=l4_value,
            L5_health=l5_value,
            L6_social=l6_value,
            L7_tension=l7_value,
            L8_identity=l8_value,
            L9_phase=l9_value,
            tension_zone=tension_state.zone,
            health_status=health_status,
            existential_zone=ellex_state.zone,
            coherence_mean=float(coherence_mean),
            stability=ellex_state.stability,
            trend=ellex_state.trend,
            t=self.t
        )

        # Guardar en historial
        self._state_history.append(state)
        max_len = max_history(self.t)
        if len(self._state_history) > max_len:
            self._state_history = self._state_history[-max_len:]

        return state

    def get_current_state(self) -> Optional[ELLEXMapState]:
        """Obtiene el estado actual."""
        if self._state_history:
            return self._state_history[-1]
        return None

    def get_layer_summary(self) -> Dict[str, float]:
        """Obtiene resumen de todas las capas."""
        return {
            'L1_cognitive': self.L1._current_value,
            'L2_symbolic': self.L2._current_value,
            'L3_narrative': self.L3._current_value,
            'L4_life': self.L4._current_value,
            'L5_health': self.L5._current_value,
            'L6_social': self.L6._current_value,
            'L7_tension': self.L7._current_value,
            'L8_identity': self.L8._current_value,
            'L9_phase': self.L9._current_value,
            'L10_ellex': self.L10._current_value
        }

    def get_weakest_areas(self, n: int = 3) -> List[Tuple[str, float]]:
        """Obtiene las n areas mas debiles."""
        summary = self.get_layer_summary()

        # Excluir L7 (tension) y L10 (agregado) de "debilidades"
        filtered = {k: v for k, v in summary.items()
                   if k not in ['L7_tension', 'L10_ellex']}

        items = sorted(filtered.items(), key=lambda x: x[1])
        return items[:n]

    def get_strongest_areas(self, n: int = 3) -> List[Tuple[str, float]]:
        """Obtiene las n areas mas fuertes."""
        summary = self.get_layer_summary()

        # Excluir L7 y L10
        filtered = {k: v for k, v in summary.items()
                   if k not in ['L7_tension', 'L10_ellex']}

        items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        return items[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadisticas del mapa."""
        if not self._state_history:
            return {'t': 0, 'no_data': True}

        recent_states = self._state_history[-L_t(self.t):]

        ellex_values = [s.ellex for s in recent_states]
        tension_values = [s.L7_tension for s in recent_states]

        # Contar zonas
        zone_counts = {}
        for state in recent_states:
            zone = state.existential_zone
            zone_counts[zone] = zone_counts.get(zone, 0) + 1

        return {
            't': self.t,
            'current_ellex': self.L10._current_value,
            'ellex_mean': np.mean(ellex_values),
            'ellex_std': np.std(ellex_values),
            'tension_mean': np.mean(tension_values),
            'tension_std': np.std(tension_values),
            'zone_distribution': zone_counts,
            'weakest_areas': self.get_weakest_areas(3),
            'strongest_areas': self.get_strongest_areas(3)
        }
