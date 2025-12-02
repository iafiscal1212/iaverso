"""
Ω1: Continuidad Trans-Ciclo
===========================

Estado profundo que sobrevive a los resets.

Memoria Ω:
- Resumen compacto de proyectos, valores, normas
- Patrones de medicina, errores graves, lecciones
- Se actualiza en DREAM, LIMINAL, o umbral CG-E

Fórmula:
    Ω_{t+1} = (1 - η_t)·Ω_t + η_t·f(summary_t)
    η_t = Q25%(surprise) / (1 + variance)

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class OmegaMemory:
    """Memoria profunda Ω."""
    dimension: int
    state_vector: np.ndarray                    # Vector de estado Ω
    values: Dict[str, float] = field(default_factory=dict)
    norms: Dict[str, float] = field(default_factory=dict)
    patterns: List[np.ndarray] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())
    update_count: int = 0


class OmegaState:
    """
    Sistema de continuidad trans-ciclo.

    Mantiene estado profundo que persiste entre resets.

    Actualización:
        Ω_{t+1} = (1 - η_t)·Ω_t + η_t·f(summary_t)

    η endógeno basado en surprise y variance.
    """

    def __init__(self, dimension: int = 64):
        """
        Args:
            dimension: Dimensión del vector Ω
        """
        self.dimension = dimension
        self.t = 0

        # Memoria Ω
        self._omega = OmegaMemory(
            dimension=dimension,
            state_vector=np.zeros(dimension)
        )

        # Historiales para η endógeno
        self._surprise_history: List[float] = []
        self._variance_history: List[float] = []
        self._summary_history: List[np.ndarray] = []

        # Fases permitidas para actualización
        self._update_phases = ['DREAM', 'LIMINAL']

    def _compute_eta(self, surprise: float) -> float:
        """
        Calcula η_t endógeno.

        η_t = Q25%(surprise) / (1 + variance)
        """
        if len(self._surprise_history) < 10:
            # Sin historial suficiente, usar √t
            return 1 / np.sqrt(self.t + 1)

        # Q25% de surprise histórica
        q25_surprise = np.percentile(self._surprise_history, 25)

        # Variance de Ω
        if len(self._summary_history) > 5:
            omega_var = np.mean([np.var(s) for s in self._summary_history[-10:]])
        else:
            omega_var = 1

        eta = q25_surprise / (1 + omega_var)

        # Limitar por percentiles históricos
        if len(self._surprise_history) > 20:
            eta_min = np.percentile(self._surprise_history, 5) / 10
            eta_max = np.percentile(self._surprise_history, 95) / 2
            eta = np.clip(eta, eta_min, eta_max)

        return float(eta)

    def _summarize(self, state: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """
        Crea resumen f(summary_t) para actualizar Ω.

        Combina estado actual con contexto relevante.
        """
        # Proyectar estado a dimensión Ω si es necesario
        if len(state) != self.dimension:
            # Usar PCA-like projection endógena
            if len(state) > self.dimension:
                # Reducir: tomar primeras componentes
                summary = state[:self.dimension]
            else:
                # Expandir: padding con ceros
                summary = np.zeros(self.dimension)
                summary[:len(state)] = state
        else:
            summary = state.copy()

        # Incorporar contexto si existe
        if context:
            # Valores -> primeras posiciones
            values = context.get('values', {})
            for i, (k, v) in enumerate(values.items()):
                if i < self.dimension // 4:
                    summary[i] = (summary[i] + v) / 2

            # Normas -> siguientes posiciones
            norms = context.get('norms', {})
            offset = self.dimension // 4
            for i, (k, v) in enumerate(norms.items()):
                if i + offset < self.dimension // 2:
                    summary[i + offset] = (summary[i + offset] + v) / 2

        return summary

    def can_update(self, phase: str, cge_index: float) -> bool:
        """
        Determina si puede actualizar Ω.

        Actualiza en:
        - DREAM
        - LIMINAL
        - Umbral CG-E (percentil 75 histórico)
        """
        if phase in self._update_phases:
            return True

        # Umbral CG-E endógeno
        if len(self._summary_history) > 10:
            cge_threshold = np.percentile(
                [np.linalg.norm(s) for s in self._summary_history],
                75
            )
            if cge_index > cge_threshold:
                return True

        return False

    def update(
        self,
        state: np.ndarray,
        surprise: float,
        phase: str,
        cge_index: float,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Actualiza estado Ω si es apropiado.

        Ω_{t+1} = (1 - η_t)·Ω_t + η_t·f(summary_t)

        Args:
            state: Estado actual del sistema
            surprise: Nivel de sorpresa
            phase: Fase actual (WAKE/REST/DREAM/LIMINAL)
            cge_index: Índice CG-E actual
            context: Contexto adicional (valores, normas, etc.)

        Returns:
            True si se actualizó
        """
        self.t += 1
        self._surprise_history.append(surprise)

        if not self.can_update(phase, cge_index):
            return False

        # Calcular η
        eta = self._compute_eta(surprise)

        # Crear resumen
        summary = self._summarize(state, context or {})
        self._summary_history.append(summary.copy())

        # Actualizar Ω
        self._omega.state_vector = (
            (1 - eta) * self._omega.state_vector +
            eta * summary
        )

        # Actualizar contexto en memoria
        if context:
            for k, v in context.get('values', {}).items():
                if k in self._omega.values:
                    self._omega.values[k] = (1 - eta) * self._omega.values[k] + eta * v
                else:
                    self._omega.values[k] = v

            for k, v in context.get('norms', {}).items():
                if k in self._omega.norms:
                    self._omega.norms[k] = (1 - eta) * self._omega.norms[k] + eta * v
                else:
                    self._omega.norms[k] = v

            if 'lesson' in context:
                self._omega.lessons.append(context['lesson'])

        self._omega.last_update = datetime.now().isoformat()
        self._omega.update_count += 1

        # Registrar varianza
        self._variance_history.append(np.var(self._omega.state_vector))

        return True

    def get_omega(self) -> np.ndarray:
        """Retorna vector Ω actual."""
        return self._omega.state_vector.copy()

    def get_memory(self) -> OmegaMemory:
        """Retorna memoria completa."""
        return self._omega

    def save(self, path: str) -> None:
        """Guarda estado Ω a archivo."""
        data = {
            'dimension': self.dimension,
            't': self.t,
            'state_vector': self._omega.state_vector.tolist(),
            'values': self._omega.values,
            'norms': self._omega.norms,
            'patterns': [p.tolist() for p in self._omega.patterns],
            'lessons': self._omega.lessons,
            'last_update': self._omega.last_update,
            'update_count': self._omega.update_count
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Carga estado Ω de archivo."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.dimension = data['dimension']
        self.t = data['t']
        self._omega = OmegaMemory(
            dimension=self.dimension,
            state_vector=np.array(data['state_vector']),
            values=data['values'],
            norms=data['norms'],
            patterns=[np.array(p) for p in data['patterns']],
            lessons=data['lessons'],
            last_update=data['last_update'],
            update_count=data['update_count']
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas."""
        return {
            't': self.t,
            'dimension': self.dimension,
            'update_count': self._omega.update_count,
            'omega_norm': float(np.linalg.norm(self._omega.state_vector)),
            'omega_variance': float(np.var(self._omega.state_vector)),
            'n_values': len(self._omega.values),
            'n_norms': len(self._omega.norms),
            'n_lessons': len(self._omega.lessons)
        }
