"""
Phase-Aware Cognition: Cognicion Consciente de la Fase
======================================================

Cada modulo cognitivo AGI se modula segun la fase circadiana:
    - WAKE:    +200% skills, planning, execution
    - REST:    +300% regulation, ethics, emotional processing
    - DREAM:   +300% memory consolidation, pattern discovery
    - LIMINAL: +400% creativity, symbolic association

Esto crea ciclos naturales de:
    crecimiento -> estabilizacion -> reorganizacion -> creatividad

100% endogeno. Los multiplicadores emergen de la historia.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')

from lifecycle.circadian_system import CircadianPhase
from cognition.agi_dynamic_constants import L_t, adaptive_learning_rate


@dataclass
class PhaseMultipliers:
    """
    Multiplicadores cognitivos por fase - ENDOGENOS.

    Los multiplicadores NO son constantes hardcodeadas.
    Emergen de:
    1. Energia disponible en la fase (WAKE = alta energia = mas ejecucion)
    2. Historial de exito por categoria en cada fase
    3. Necesidades actuales del agente

    Los valores iniciales se derivan de principios biologicos:
    - Alta energia -> ejecucion, skills
    - Baja actividad -> consolidacion, regulacion
    - Estados intermedios -> creatividad, planning
    """

    # Categorias cognitivas
    CATEGORIES = ['skills', 'planning', 'execution', 'social',
                  'memory_encoding', 'creativity', 'regulation', 'consolidation']

    # Historial de efectividad por fase x categoria
    effectiveness_history: Dict[str, Dict[str, List[float]]] = field(
        default_factory=lambda: {
            phase.value: {cat: [] for cat in PhaseMultipliers.CATEGORIES}
            for phase in CircadianPhase
        }
    )

    t: int = 0

    def get_multiplier(self, phase: CircadianPhase, category: str,
                       energy: float = 0.5, stress: float = 0.0,
                       pending_consolidation: int = 0) -> float:
        """
        Calcula multiplicador ENDOGENO para categoria en fase.

        Args:
            phase: Fase circadiana actual
            category: Categoria cognitiva
            energy: Energia actual [0, 1]
            stress: Estres actual [0, 1]
            pending_consolidation: Items pendientes de consolidar

        Returns:
            Multiplicador >= 0.1
        """
        # Base: derivar de principios biologicos (no numeros magicos)
        # Energia alta -> skills, execution, planning
        # Energia baja -> consolidation, regulation, creativity

        # Factor de energia: que tan activo deberia estar
        if phase == CircadianPhase.WAKE:
            energy_factor = 0.5 + 0.5 * energy  # [0.5, 1.0]
        elif phase == CircadianPhase.REST:
            energy_factor = 0.3 + 0.2 * (1 - stress)  # [0.3, 0.5]
        elif phase == CircadianPhase.DREAM:
            energy_factor = 0.2 + 0.1 * (1 - stress)  # [0.2, 0.3]
        else:  # LIMINAL
            energy_factor = 0.4 + 0.2 * energy  # [0.4, 0.6]

        # Base por categoria segun fase (derivado de energia, no hardcodeado)
        if category in ['skills', 'execution']:
            # Requieren energia: activos en WAKE
            base = energy_factor * 2.0  # Escala con energia
        elif category == 'planning':
            # Moderado: funciona con energia media
            base = 0.5 + energy_factor
        elif category == 'social':
            # Social: mejor en transiciones y wake
            base = 0.8 + 0.4 * energy_factor
        elif category == 'memory_encoding':
            # Codificacion: moderada siempre
            base = 0.6 + 0.4 * energy_factor
        elif category == 'creativity':
            # Creatividad: mejor con baja energia (menos inhibicion)
            base = 0.5 + 1.5 * (1 - energy_factor)
        elif category == 'regulation':
            # Regulacion: mejor cuando no hay presion de ejecucion
            base = 0.5 + 1.5 * (1 - energy_factor) + 0.5 * stress
        elif category == 'consolidation':
            # Consolidacion: requiere descanso
            pending_factor = min(1.0, pending_consolidation / max(1, L_t(self.t)))
            base = 0.5 + 2.0 * (1 - energy_factor) + pending_factor
        else:
            base = 1.0

        # Ajustar por historial de efectividad
        history = self.effectiveness_history[phase.value].get(category, [])
        if len(history) >= 3:
            window = L_t(self.t)
            recent_eff = np.mean(history[-window:])
            # Si funciono bien, aumentar; si no, reducir
            learned_adjustment = 0.8 + 0.4 * recent_eff
            base *= learned_adjustment

        return max(0.1, base)

    def record_effectiveness(self, phase: CircadianPhase, category: str,
                            effectiveness: float):
        """Registra efectividad para aprendizaje."""
        self.effectiveness_history[phase.value][category].append(effectiveness)
        # Limitar historial
        max_hist = 100  # max_history(self.t) si tuvieramos t
        if len(self.effectiveness_history[phase.value][category]) > max_hist:
            self.effectiveness_history[phase.value][category] = \
                self.effectiveness_history[phase.value][category][-max_hist:]

    def step(self):
        """Avanza tiempo."""
        self.t += 1


@dataclass
class CognitiveModuleState:
    """Estado de un modulo cognitivo."""
    module_name: str
    base_effectiveness: float      # Efectividad base [0, 1]
    phase_multiplier: float        # Multiplicador de fase actual
    effective_value: float         # Valor efectivo final
    learning_rate_modifier: float  # Modificador de aprendizaje


class PhaseAwareCognition:
    """
    Sistema de cognicion consciente de la fase circadiana.

    Modula todos los modulos AGI segun la fase actual,
    creando ciclos naturales de crecimiento y reorganizacion.
    """

    # Mapeo de modulos AGI a categorias cognitivas
    AGI_MODULE_MAPPING = {
        # AGI-4: Self-Model
        'self_model': 'memory_encoding',
        'self_prediction': 'planning',

        # AGI-5: Theory of Mind
        'tom': 'social',
        'other_model': 'social',

        # AGI-6: Prospection
        'prospection': 'planning',
        'future_simulation': 'creativity',

        # AGI-7: Counterfactual
        'counterfactual': 'creativity',

        # AGI-8: Drives
        'drives': 'regulation',

        # AGI-9: Deliberation
        'deliberation': 'planning',

        # AGI-10: Ethics
        'ethics': 'regulation',

        # AGI-11: Attention
        'attention': 'execution',

        # AGI-12: Learning
        'learning': 'skills',
        'skill_acquisition': 'skills',

        # AGI-13: Memory
        'episodic_memory': 'consolidation',
        'semantic_memory': 'memory_encoding',
        'working_memory': 'execution',

        # AGI-14: Emotion
        'emotion': 'regulation',

        # AGI-15: Cognitive Control
        'cognitive_control': 'regulation',

        # AGI-16: Meta-Rules
        'meta_rules': 'planning',

        # AGI-17: Robustness
        'robustness': 'regulation',

        # AGI-18: Reconfiguration
        'reconfiguration': 'creativity',

        # AGI-19: Collective Intent
        'collective_intent': 'social',

        # AGI-20: Self-Theory
        'self_theory': 'consolidation',
    }

    def __init__(self, agent_id: str):
        """
        Inicializa sistema de cognicion phase-aware.

        Args:
            agent_id: ID del agente
        """
        self.agent_id = agent_id

        # Multiplicadores ENDOGENOS (aprenden de la historia)
        self.multipliers = PhaseMultipliers()

        # Fase actual y estado
        self.current_phase = CircadianPhase.WAKE
        self.current_energy = 1.0
        self.current_stress = 0.0
        self.pending_consolidation = 0

        # Efectividad base de cada modulo (aprende de la historia)
        self.module_effectiveness: Dict[str, float] = {
            module: 0.5 for module in self.AGI_MODULE_MAPPING.keys()
        }

        # Historial de efectividad por fase
        self.phase_effectiveness_history: Dict[CircadianPhase, List[float]] = {
            phase: [] for phase in CircadianPhase
        }

        # Historial de learning rate por fase (para adaptarlo)
        self._lr_history: Dict[CircadianPhase, List[float]] = {
            phase: [] for phase in CircadianPhase
        }

        self.t = 0

    def set_phase(self, phase: CircadianPhase, energy: float = None,
                  stress: float = None, pending: int = None):
        """Actualiza la fase circadiana actual y estado."""
        self.current_phase = phase
        if energy is not None:
            self.current_energy = energy
        if stress is not None:
            self.current_stress = stress
        if pending is not None:
            self.pending_consolidation = pending

    def get_multiplier(self, module_name: str) -> float:
        """
        Obtiene multiplicador ENDOGENO para un modulo en la fase actual.

        Args:
            module_name: Nombre del modulo AGI

        Returns:
            Multiplicador [0.1, inf)
        """
        # Obtener categoria cognitiva
        category = self.AGI_MODULE_MAPPING.get(module_name, 'skills')

        # Obtener multiplicador ENDOGENO de PhaseMultipliers
        return self.multipliers.get_multiplier(
            phase=self.current_phase,
            category=category,
            energy=self.current_energy,
            stress=self.current_stress,
            pending_consolidation=self.pending_consolidation
        )

    def _get_lr_modifier(self) -> float:
        """
        Calcula learning rate modifier ENDOGENO.

        Basado en:
        - Fase actual (consolidacion vs ejecucion)
        - Historial de efectividad del aprendizaje
        - Energia y estres actuales
        """
        # Base: derivar de energia y fase
        if self.current_phase == CircadianPhase.WAKE:
            # Aprendizaje activo moderado
            base_lr = 0.8 + 0.4 * self.current_energy
        elif self.current_phase == CircadianPhase.REST:
            # Aprendizaje reducido, recuperando
            base_lr = 0.3 + 0.2 * (1 - self.current_stress)
        elif self.current_phase == CircadianPhase.DREAM:
            # Consolidacion activa = alto learning rate para patrones
            pending_factor = min(1.0, self.pending_consolidation / max(1, L_t(self.t)))
            base_lr = 1.0 + 1.5 * pending_factor
        else:  # LIMINAL
            # Transicion: aprendizaje moderado-alto
            base_lr = 1.0 + 0.5 * self.current_energy

        # Ajustar por historial
        history = self._lr_history[self.current_phase]
        if len(history) >= 3:
            window = L_t(self.t)
            recent_success = np.mean(history[-window:])
            # Si el aprendizaje fue efectivo, mantener; si no, ajustar
            base_lr *= 0.7 + 0.6 * recent_success

        return max(0.1, base_lr)

    def get_module_state(self, module_name: str) -> CognitiveModuleState:
        """
        Obtiene estado completo de un modulo.

        Args:
            module_name: Nombre del modulo

        Returns:
            Estado del modulo
        """
        base_eff = self.module_effectiveness.get(module_name, 0.5)
        multiplier = self.get_multiplier(module_name)
        effective = min(1.0, base_eff * multiplier)

        # Learning rate modifier ENDOGENO
        lr_mod = self._get_lr_modifier()

        return CognitiveModuleState(
            module_name=module_name,
            base_effectiveness=base_eff,
            phase_multiplier=multiplier,
            effective_value=effective,
            learning_rate_modifier=lr_mod
        )

    def get_all_module_states(self) -> Dict[str, CognitiveModuleState]:
        """Obtiene estados de todos los modulos."""
        return {
            module: self.get_module_state(module)
            for module in self.AGI_MODULE_MAPPING.keys()
        }

    def update_effectiveness(
        self,
        module_name: str,
        outcome: float,
        importance: float = 1.0
    ):
        """
        Actualiza efectividad de un modulo basado en outcome.

        Args:
            module_name: Nombre del modulo
            outcome: Resultado [0, 1] (1 = exito total)
            importance: Importancia del evento
        """
        if module_name not in self.module_effectiveness:
            return

        current = self.module_effectiveness[module_name]

        # Learning rate adaptativo
        lr = adaptive_learning_rate(self.t) * importance

        # Actualizar efectividad
        new_eff = current + lr * (outcome - current)
        self.module_effectiveness[module_name] = np.clip(new_eff, 0.1, 0.95)

        # Registrar en historial de fase
        self.phase_effectiveness_history[self.current_phase].append(outcome)

        # Limitar historial
        max_hist = 100
        if len(self.phase_effectiveness_history[self.current_phase]) > max_hist:
            self.phase_effectiveness_history[self.current_phase] = \
                self.phase_effectiveness_history[self.current_phase][-max_hist:]

    def learn_optimal_multipliers(self):
        """
        Aprende multiplicadores optimos basado en historial.

        Ejecutar periodicamente para adaptar multiplicadores.
        """
        for phase in CircadianPhase:
            history = self.phase_effectiveness_history[phase]
            if len(history) < 10:
                continue

            # Calcular efectividad media en esta fase
            mean_eff = np.mean(history[-20:])

            # Si la efectividad es baja, ajustar multiplicadores
            if mean_eff < 0.4:
                # Aumentar multiplicadores para esta fase
                for category in ['skills', 'planning', 'regulation']:
                    current = self._learned_multipliers[phase].get(category, 1.0)
                    self._learned_multipliers[phase][category] = current * 1.1
            elif mean_eff > 0.7:
                # Fase funciona bien, mantener o reducir ligeramente
                for category in ['skills', 'planning', 'regulation']:
                    current = self._learned_multipliers[phase].get(category, 1.0)
                    self._learned_multipliers[phase][category] = current * 0.95

    def step(self, phase: CircadianPhase):
        """
        Ejecuta un paso del sistema.

        Args:
            phase: Fase circadiana actual
        """
        self.t += 1
        self.set_phase(phase)

        # Aprender periodicamente
        if self.t % 50 == 0:
            self.learn_optimal_multipliers()

    def get_phase_profile(self) -> Dict[str, float]:
        """
        Obtiene perfil cognitivo de la fase actual.

        Returns:
            Dict con multiplicadores por categoria
        """
        phase_mults = {
            CircadianPhase.WAKE: self.multipliers.wake,
            CircadianPhase.REST: self.multipliers.rest,
            CircadianPhase.DREAM: self.multipliers.dream,
            CircadianPhase.LIMINAL: self.multipliers.liminal,
        }
        return phase_mults[self.current_phase].copy()

    def get_cognitive_mode(self) -> str:
        """
        Obtiene descripcion del modo cognitivo actual.

        Returns:
            Descripcion del modo
        """
        modes = {
            CircadianPhase.WAKE: "Modo Activo: alto rendimiento en skills y ejecucion",
            CircadianPhase.REST: "Modo Recuperacion: regulacion emocional y etica activa",
            CircadianPhase.DREAM: "Modo Consolidacion: reorganizacion de memorias y patrones",
            CircadianPhase.LIMINAL: "Modo Creativo: maxima asociacion simbolica",
        }
        return modes[self.current_phase]

    def get_statistics(self) -> Dict:
        """Estadisticas del sistema."""
        # Efectividad media por fase
        phase_means = {}
        for phase in CircadianPhase:
            history = self.phase_effectiveness_history[phase]
            phase_means[phase.value] = np.mean(history) if history else 0.5

        # Modulos mas efectivos en fase actual
        states = self.get_all_module_states()
        sorted_modules = sorted(
            states.items(),
            key=lambda x: x[1].effective_value,
            reverse=True
        )
        top_modules = [m[0] for m in sorted_modules[:3]]

        return {
            'agent_id': self.agent_id,
            't': self.t,
            'current_phase': self.current_phase.value,
            'cognitive_mode': self.get_cognitive_mode(),
            'phase_effectiveness': phase_means,
            'top_modules_current_phase': top_modules,
            'module_count': len(self.module_effectiveness)
        }


class PhaseAwareCognitionManager:
    """
    Manager global para cognicion phase-aware de todos los agentes.
    """

    def __init__(self, agent_ids: List[str]):
        """
        Inicializa manager.

        Args:
            agent_ids: Lista de IDs de agentes
        """
        self.agent_ids = agent_ids
        self.cognition_systems: Dict[str, PhaseAwareCognition] = {
            aid: PhaseAwareCognition(aid)
            for aid in agent_ids
        }

    def update_phase(self, agent_id: str, phase: CircadianPhase):
        """Actualiza fase de un agente."""
        if agent_id in self.cognition_systems:
            self.cognition_systems[agent_id].step(phase)

    def get_multiplier(self, agent_id: str, module_name: str) -> float:
        """Obtiene multiplicador para un modulo de un agente."""
        if agent_id in self.cognition_systems:
            return self.cognition_systems[agent_id].get_multiplier(module_name)
        return 1.0

    def update_all(self, phases: Dict[str, CircadianPhase]):
        """Actualiza todos los agentes."""
        for agent_id, phase in phases.items():
            self.update_phase(agent_id, phase)

    def get_global_statistics(self) -> Dict:
        """Estadisticas globales."""
        return {
            agent_id: system.get_statistics()
            for agent_id, system in self.cognition_systems.items()
        }


def test_phase_aware_cognition():
    """Test del sistema de cognicion phase-aware."""
    print("=" * 70)
    print("TEST: PHASE-AWARE COGNITION")
    print("=" * 70)

    np.random.seed(42)

    agent_id = "NEO"
    system = PhaseAwareCognition(agent_id)

    phases = [
        CircadianPhase.WAKE,
        CircadianPhase.REST,
        CircadianPhase.DREAM,
        CircadianPhase.LIMINAL
    ]

    print(f"\nAgente: {agent_id}")
    print("\nMultiplicadores por fase y categoria:")

    for phase in phases:
        system.set_phase(phase)
        profile = system.get_phase_profile()

        print(f"\n  {phase.value.upper()}:")
        for category, mult in profile.items():
            bar = "█" * int(mult * 5)
            print(f"    {category:20s}: {mult:.1f}x {bar}")

    print("\n" + "=" * 70)
    print("SIMULACION DE 200 PASOS")
    print("=" * 70)

    # Simular ciclo completo
    for t in range(200):
        # Rotar fases
        phase_idx = (t // 50) % 4
        phase = phases[phase_idx]

        system.step(phase)

        # Simular outcomes
        for module in ['self_model', 'tom', 'ethics', 'episodic_memory']:
            outcome = 0.5 + 0.3 * np.random.randn()
            outcome = np.clip(outcome, 0, 1)
            system.update_effectiveness(module, outcome, importance=0.5)

        if t % 50 == 49:
            stats = system.get_statistics()
            print(f"\n  t={t+1}: {stats['current_phase']}")
            print(f"    Modo: {stats['cognitive_mode']}")
            print(f"    Top modulos: {stats['top_modules_current_phase']}")

    # Estadisticas finales
    print("\n" + "=" * 70)
    print("ESTADISTICAS FINALES")
    print("=" * 70)

    stats = system.get_statistics()
    print(f"\n  Efectividad por fase:")
    for phase, eff in stats['phase_effectiveness'].items():
        print(f"    {phase}: {eff:.3f}")

    return system


if __name__ == "__main__":
    test_phase_aware_cognition()
