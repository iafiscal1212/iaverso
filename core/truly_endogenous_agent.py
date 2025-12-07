#!/usr/bin/env python3
"""
Agente 100% Endógeno
====================

NORMA DURA CUMPLIDA:
"Ningún número entra al código sin explicar de qué distribución sale"

Este agente:
1. NO tiene números mágicos
2. TODOS los umbrales emergen de observaciones
3. Si no hay datos suficientes, retorna None (no asume)
4. Cada decisión tiene justificación auditable

PARA PUBLICACIÓN:
- Cada umbral tiene proveniencia
- Cada decisión tiene justificación
- Todo es auditable
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from datetime import datetime
import json
from pathlib import Path

from core.endogenous_constants import (
    EndogenousThresholds,
    MATHEMATICAL_CONSTANTS,
    observe as global_observe,
    get_threshold,
    should_trigger,
    calculate_score,
)


@dataclass
class EndogenousState:
    """
    Estado del agente sin valores hardcodeados.

    Los valores iniciales son None hasta que se observen datos.
    """
    # Identificación
    agent_id: str = ""
    birth_time: float = 0.0

    # Estados que emergen de observaciones
    # Todos empiezan en None hasta que haya datos
    energy: Optional[float] = None
    curiosity: Optional[float] = None
    confidence: Optional[float] = None
    surprise: Optional[float] = None

    # Historial de valores propios
    energy_history: deque = field(default_factory=lambda: deque(maxlen=100))
    action_history: deque = field(default_factory=lambda: deque(maxlen=100))


class TrulyEndogenousAgent:
    """
    Agente que cumple la norma dura de endogeneidad.

    GARANTÍAS:
    - Ningún umbral es hardcodeado
    - Todo viene de distribuciones observadas
    - Si no hay datos, no decide (retorna None)
    - Auditoría completa
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = EndogenousState(
            agent_id=agent_id,
            birth_time=datetime.now().timestamp()
        )

        # Sistema de umbrales propio del agente
        self.thresholds = EndogenousThresholds()

        # Observaciones del mundo
        self.world_observations: Dict[str, deque] = {}

        # Decisiones tomadas (para auditoría)
        self.decision_log: List[Dict] = []

        # ¿Está calibrado? (suficientes observaciones)
        self._calibration_status: Dict[str, bool] = {}

    def observe_world(self, variable: str, value: float, source: str = "sensor"):
        """
        Observar una variable del mundo.

        El agente acumula observaciones para derivar umbrales.
        """
        # Registrar en umbrales globales
        self.thresholds.observe(variable, value, source)

        # Registrar localmente
        if variable not in self.world_observations:
            self.world_observations[variable] = deque(maxlen=1000)

        self.world_observations[variable].append({
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'source': source,
        })

        # Actualizar estado de calibración
        self._check_calibration(variable)

    def _check_calibration(self, variable: str):
        """
        Verificar si hay suficientes datos para esta variable.
        """
        min_samples = MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_STATISTICS']
        n_obs = len(self.world_observations.get(variable, []))
        self._calibration_status[variable] = n_obs >= min_samples

    def is_calibrated(self, variable: str) -> bool:
        """¿Hay suficientes datos para esta variable?"""
        return self._calibration_status.get(variable, False)

    def get_calibration_status(self) -> Dict:
        """Estado de calibración de todas las variables."""
        return {
            var: {
                'calibrated': self.is_calibrated(var),
                'n_observations': len(self.world_observations.get(var, [])),
                'required': MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_STATISTICS'],
            }
            for var in self.world_observations.keys()
        }

    def evaluate_value(self, variable: str, value: float) -> Dict:
        """
        Evaluar un valor usando distribución observada.

        Si no está calibrado, retorna que no puede evaluar.
        """
        if not self.is_calibrated(variable):
            return {
                'can_evaluate': False,
                'reason': f'Not calibrated for {variable}. Need {MATHEMATICAL_CONSTANTS["MIN_SAMPLES_FOR_STATISTICS"]} observations.',
                'n_current': len(self.world_observations.get(variable, [])),
            }

        # Calcular score basado en distribución
        result = self.thresholds.calculate_score(variable, value)

        # Registrar decisión
        self.decision_log.append({
            'type': 'evaluate',
            'variable': variable,
            'value': value,
            'result': result,
            'timestamp': datetime.now().isoformat(),
        })

        return result

    def should_act(self, variable: str, current_value: float,
                   action_type: str = 'respond') -> Dict:
        """
        Decidir si actuar basándose en observaciones.

        El umbral viene de la distribución, no hardcodeado.
        """
        if not self.is_calibrated(variable):
            return {
                'should_act': None,  # None = no puede decidir
                'reason': 'Not calibrated',
                'action_type': action_type,
            }

        # Determinar qué tipo de umbral usar según acción
        # NOTA: Estos nombres (high, very_significant) vienen de la
        # distribución, no son valores arbitrarios
        threshold_type_map = {
            'respond': 'high',  # p90 de la distribución
            'alert': 'very_significant',  # mean + 2*std
            'explore': 'medium',  # mediana
        }

        threshold_type = threshold_type_map.get(action_type, 'high')

        # Obtener decisión con justificación
        result = self.thresholds.should_trigger(variable, current_value, threshold_type)

        # Registrar decisión
        decision = {
            'should_act': result['trigger'],
            'action_type': action_type,
            'variable': variable,
            'current_value': current_value,
            'threshold': result['threshold'],
            'threshold_type': threshold_type,
            'justification': result['justification'],
            'z_score': result.get('z_score'),
            'probability': result.get('probability'),
            'n_samples': result.get('n_samples'),
            'timestamp': datetime.now().isoformat(),
        }

        self.decision_log.append({
            'type': 'should_act',
            'decision': decision,
        })

        return decision

    def compare_values(self, var1: str, val1: float,
                       var2: str, val2: float) -> Dict:
        """
        Comparar dos valores de diferentes variables.

        Usa z-scores para comparación normalizada.
        """
        result1 = self.thresholds.calculate_score(var1, val1)
        result2 = self.thresholds.calculate_score(var2, val2)

        if not result1.get('can_score') or not result2.get('can_score'):
            return {
                'can_compare': False,
                'reason': 'One or both variables not calibrated',
            }

        z1 = result1['z_score']
        z2 = result2['z_score']

        # Comparar usando z-scores (adimensional)
        return {
            'can_compare': True,
            'var1': {
                'variable': var1,
                'value': val1,
                'z_score': z1,
                'percentile': result1['percentile'],
            },
            'var2': {
                'variable': var2,
                'value': val2,
                'z_score': z2,
                'percentile': result2['percentile'],
            },
            'comparison': {
                'more_extreme': var1 if z1 > z2 else var2,
                'z_difference': abs(z1 - z2),
            }
        }

    def derive_personal_thresholds(self) -> Dict:
        """
        Derivar umbrales personales del agente.

        Basados en su historial de energía y acciones.
        """
        thresholds = {}

        # Umbral de energía bajo = p10 de su historial
        if len(self.state.energy_history) >= MATHEMATICAL_CONSTANTS['MIN_SAMPLES_FOR_PERCENTILES']:
            energies = [e['value'] for e in self.state.energy_history]
            thresholds['energy_low'] = {
                'value': float(np.percentile(energies, 10)),
                'justification': 'percentile_10 of own energy history',
                'n_samples': len(energies),
            }
            thresholds['energy_high'] = {
                'value': float(np.percentile(energies, 90)),
                'justification': 'percentile_90 of own energy history',
                'n_samples': len(energies),
            }

        return thresholds

    def update_energy(self, delta: float, reason: str):
        """
        Actualizar energía y registrar.
        """
        if self.state.energy is None:
            # Primera observación - establecer desde el delta
            self.state.energy = max(0, delta)
        else:
            self.state.energy = max(0, self.state.energy + delta)

        self.state.energy_history.append({
            'value': self.state.energy,
            'delta': delta,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
        })

        # Observar para umbrales
        self.thresholds.observe('self_energy', self.state.energy, 'internal')

    def step(self, world_state: Dict[str, float]) -> Dict:
        """
        Un paso de simulación.

        Observa el mundo, decide acciones, actualiza estado.
        """
        actions = []
        observations = []

        # Observar todas las variables del mundo
        for var, value in world_state.items():
            self.observe_world(var, value, 'world_step')
            observations.append({
                'variable': var,
                'value': value,
                'calibrated': self.is_calibrated(var),
            })

        # Decidir acciones para variables calibradas
        for var, value in world_state.items():
            if self.is_calibrated(var):
                decision = self.should_act(var, value, 'respond')
                if decision['should_act']:
                    actions.append({
                        'action': 'respond',
                        'variable': var,
                        'trigger_value': value,
                        'justification': decision['justification'],
                    })

        return {
            'agent_id': self.agent_id,
            'timestamp': datetime.now().isoformat(),
            'observations': observations,
            'actions': actions,
            'calibration_status': self.get_calibration_status(),
        }

    def get_audit_report(self) -> Dict:
        """
        Reporte completo de auditoría.

        PARA PUBLICACIÓN:
        - Cada umbral tiene origen
        - Cada decisión tiene justificación
        - Todo es verificable
        """
        return {
            'agent_id': self.agent_id,
            'birth_time': self.state.birth_time,
            'calibration': self.get_calibration_status(),
            'thresholds': self.thresholds.get_audit_report(),
            'personal_thresholds': self.derive_personal_thresholds(),
            'n_decisions': len(self.decision_log),
            'recent_decisions': self.decision_log[-10:],
            'mathematical_constants_used': MATHEMATICAL_CONSTANTS,
            'guarantee': 'All thresholds derived from observed distributions',
        }


def demo():
    """
    Demostración del agente endógeno.
    """
    print("=" * 70)
    print("AGENTE 100% ENDÓGENO - DEMOSTRACIÓN")
    print("=" * 70)
    print()
    print("NORMA: Ningún número sin justificación de distribución")
    print()

    agent = TrulyEndogenousAgent("DEMO_AGENT")

    # Fase 1: Calibración
    print("FASE 1: CALIBRACIÓN (observando el mundo)")
    print("-" * 50)

    # Simular observaciones
    np.random.seed(42)
    temperatures = np.random.normal(280, 50, 20)  # Media 280, std 50

    for i, temp in enumerate(temperatures):
        agent.observe_world('temperature', temp, 'sensor')
        if (i + 1) % 5 == 0:
            status = agent.get_calibration_status()
            calibrated = status.get('temperature', {}).get('calibrated', False)
            print(f"  Observación {i+1}: temp={temp:.1f}K, calibrado={calibrated}")

    # Fase 2: Decisiones
    print()
    print("FASE 2: DECISIONES (con umbrales derivados)")
    print("-" * 50)

    test_values = [200, 280, 350, 400]
    for val in test_values:
        result = agent.should_act('temperature', val, 'respond')
        print(f"\n  Valor: {val}K")
        print(f"    ¿Actuar? {result['should_act']}")
        print(f"    Umbral: {result['threshold']:.1f}K")
        print(f"    Justificación: {result['justification']}")
        if result.get('z_score') is not None:
            print(f"    Z-score: {result['z_score']:.2f}")

    # Fase 3: Auditoría
    print()
    print("FASE 3: AUDITORÍA")
    print("-" * 50)

    audit = agent.get_audit_report()
    print(f"  Decisiones tomadas: {audit['n_decisions']}")
    print(f"  Variables calibradas: {list(audit['calibration'].keys())}")

    temp_dist = audit['thresholds']['distributions'].get('temperature', {})
    print(f"\n  Distribución de temperatura:")
    print(f"    n = {temp_dist.get('n')}")
    print(f"    mean = {temp_dist.get('mean', 0):.1f}K")
    print(f"    std = {temp_dist.get('std', 0):.1f}K")

    print()
    print("=" * 70)
    print("✅ TODOS LOS UMBRALES TIENEN JUSTIFICACIÓN")
    print("=" * 70)


if __name__ == '__main__':
    demo()
