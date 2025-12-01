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
    """Multiplicadores cognitivos por fase."""
    # Modulos AGI y sus multiplicadores por fase
    # Formato: {modulo: {fase: multiplicador}}

    # Base multipliers (pueden evolucionar)
    wake: Dict[str, float] = field(default_factory=lambda: {
        'skills': 2.0,           # +200% habilidades
        'planning': 2.0,         # +200% planeamiento
        'execution': 1.8,        # +180% ejecucion
        'social': 1.5,           # +150% interaccion social
        'memory_encoding': 1.5,  # +150% codificacion
        'creativity': 0.8,       # -20% creatividad (muy enfocado)
        'regulation': 0.7,       # -30% regulacion (activo)
        'consolidation': 0.5,    # -50% consolidacion
    })

    rest: Dict[str, float] = field(default_factory=lambda: {
        'skills': 0.5,           # -50% habilidades
        'planning': 0.7,         # -30% planeamiento
        'execution': 0.3,        # -70% ejecucion
        'social': 0.8,           # -20% social
        'memory_encoding': 0.6,  # -40% codificacion
        'creativity': 1.2,       # +20% creatividad
        'regulation': 3.0,       # +300% regulacion emocional/etica
        'consolidation': 1.5,    # +50% consolidacion
    })

    dream: Dict[str, float] = field(default_factory=lambda: {
        'skills': 0.3,           # -70% habilidades
        'planning': 0.4,         # -60% planeamiento
        'execution': 0.1,        # -90% ejecucion
        'social': 0.5,           # -50% social
        'memory_encoding': 0.3,  # -70% codificacion nueva
        'creativity': 2.5,       # +250% creatividad
        'regulation': 1.5,       # +50% regulacion
        'consolidation': 3.0,    # +300% consolidacion
    })

    liminal: Dict[str, float] = field(default_factory=lambda: {
        'skills': 0.8,           # -20% habilidades
        'planning': 1.0,         # normal
        'execution': 0.5,        # -50% ejecucion
        'social': 1.0,           # normal
        'memory_encoding': 1.0,  # normal
        'creativity': 4.0,       # +400% creatividad
        'regulation': 1.2,       # +20% regulacion
        'consolidation': 1.5,    # +50% consolidacion
    })


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

        # Multiplicadores base (pueden evolucionar)
        self.multipliers = PhaseMultipliers()

        # Fase actual
        self.current_phase = CircadianPhase.WAKE

        # Efectividad base de cada modulo (aprende de la historia)
        self.module_effectiveness: Dict[str, float] = {
            module: 0.5 for module in self.AGI_MODULE_MAPPING.keys()
        }

        # Historial de efectividad por fase
        self.phase_effectiveness_history: Dict[CircadianPhase, List[float]] = {
            phase: [] for phase in CircadianPhase
        }

        # Aprendizaje de multiplicadores optimos
        self._learned_multipliers: Dict[CircadianPhase, Dict[str, float]] = {
            phase: {} for phase in CircadianPhase
        }

        self.t = 0

    def set_phase(self, phase: CircadianPhase):
        """Actualiza la fase circadiana actual."""
        self.current_phase = phase

    def get_multiplier(self, module_name: str) -> float:
        """
        Obtiene multiplicador para un modulo en la fase actual.

        Args:
            module_name: Nombre del modulo AGI

        Returns:
            Multiplicador [0, inf)
        """
        # Obtener categoria cognitiva
        category = self.AGI_MODULE_MAPPING.get(module_name, 'skills')

        # Obtener multiplicador base de la fase
        phase_mults = {
            CircadianPhase.WAKE: self.multipliers.wake,
            CircadianPhase.REST: self.multipliers.rest,
            CircadianPhase.DREAM: self.multipliers.dream,
            CircadianPhase.LIMINAL: self.multipliers.liminal,
        }

        base_mult = phase_mults[self.current_phase].get(category, 1.0)

        # Ajustar por aprendizaje historico
        learned = self._learned_multipliers[self.current_phase].get(category, 1.0)
        adjustment = 0.9 * base_mult + 0.1 * learned

        return max(0.1, adjustment)

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

        # Learning rate modifier basado en fase
        lr_modifiers = {
            CircadianPhase.WAKE: 1.0,
            CircadianPhase.REST: 0.5,
            CircadianPhase.DREAM: 2.0,  # Consolidacion rapida
            CircadianPhase.LIMINAL: 1.5,
        }
        lr_mod = lr_modifiers[self.current_phase]

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
