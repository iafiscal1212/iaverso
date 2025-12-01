#!/usr/bin/env python3
"""
WEAVER Global State
===================

Estado compartido entre todas las fases del sistema.
100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PhaseState:
    """Estado de una fase individual."""
    name: str
    go_status: bool = False
    criteria_passed: int = 0
    criteria_total: int = 0
    last_update: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class GlobalState:
    """
    Estado global compartido entre todas las fases.

    100% Endógeno:
    - Ventanas temporales: W_k = √t, 2√t, 4√t (de especificación)
    - Buffers dimensionados por historia
    """

    def __init__(self):
        # Estado de cada fase
        self.phases: Dict[str, PhaseState] = {}

        # Historia global
        self.t: int = 0
        self.S_global: List[float] = []
        self.z_global: List[np.ndarray] = []

        # Métricas agregadas
        self.metrics_history: List[Dict[str, float]] = []

        # Banderas de sistema
        self.system_ready: bool = False
        self.all_phases_go: bool = False

        # Registro de eventos
        self.events: List[Dict[str, Any]] = []

    def register_phase(self, name: str, criteria_total: int = 5) -> None:
        """Registra una nueva fase en el estado global."""
        self.phases[name] = PhaseState(
            name=name,
            criteria_total=criteria_total
        )

    def update_phase(self, name: str, go_status: bool,
                     criteria_passed: int, metrics: Dict[str, float]) -> None:
        """Actualiza el estado de una fase."""
        if name not in self.phases:
            self.register_phase(name)

        phase = self.phases[name]
        phase.go_status = go_status
        phase.criteria_passed = criteria_passed
        phase.metrics = metrics
        phase.last_update = datetime.now()

        # Verificar si todas las fases son GO
        self.all_phases_go = all(p.go_status for p in self.phases.values())

        # Registrar evento
        self.events.append({
            'time': self.t,
            'type': 'phase_update',
            'phase': name,
            'go_status': go_status,
            'timestamp': datetime.now().isoformat()
        })

    def step(self, z: np.ndarray, S: float) -> None:
        """Avanza el tiempo global y registra estado."""
        self.t += 1
        self.S_global.append(S)
        self.z_global.append(z.copy())
        self.system_ready = len(self.S_global) > 10

    def get_window_sizes(self) -> List[int]:
        """
        Retorna tamaños de ventana multi-escala.
        W_k = √t, 2√t, 4√t (100% endógeno)
        """
        sqrt_t = max(1, int(np.sqrt(self.t + 1)))
        return [sqrt_t, 2 * sqrt_t, 4 * sqrt_t]

    def get_recent_history(self, window: Optional[int] = None) -> Dict[str, Any]:
        """Retorna historia reciente dentro de ventana."""
        if window is None:
            window = self.get_window_sizes()[0]

        window = min(window, len(self.S_global))

        return {
            'S': self.S_global[-window:] if window > 0 else [],
            'z': self.z_global[-window:] if window > 0 else [],
            't_start': max(0, self.t - window),
            't_end': self.t
        }

    def get_system_summary(self) -> Dict[str, Any]:
        """Retorna resumen del estado del sistema."""
        go_count = sum(1 for p in self.phases.values() if p.go_status)

        return {
            't': self.t,
            'phases_registered': len(self.phases),
            'phases_go': go_count,
            'all_go': self.all_phases_go,
            'system_ready': self.system_ready,
            'S_mean': np.mean(self.S_global) if self.S_global else 0.0,
            'S_std': np.std(self.S_global) if len(self.S_global) > 1 else 0.0
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializa estado a diccionario."""
        return {
            't': self.t,
            'phases': {
                name: {
                    'go_status': p.go_status,
                    'criteria_passed': p.criteria_passed,
                    'criteria_total': p.criteria_total,
                    'metrics': p.metrics
                }
                for name, p in self.phases.items()
            },
            'system': self.get_system_summary(),
            'n_events': len(self.events)
        }
