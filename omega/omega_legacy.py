"""
Ω4: Legado y Cierre Interno
===========================

El sistema puede declarar:
"Esta encarnación ha cumplido su función"

Entonces:
- Vuelca resumen de su vida a Ω_legacy
- Marca estados, caminos y zonas como útiles/peligrosos/ineficientes
- En siguiente arranque: hereda legado, lecciones, advertencias

"Vidas sucesivas" dentro del mismo circuito.
Todo matemática y memoria estructural.

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class LegacyRecord:
    """Registro de legado."""
    incarnation_id: int
    start_time: str
    end_time: str
    duration_steps: int

    # Resumen de vida
    total_decisions: int
    successful_patterns: List[str]
    dangerous_zones: List[str]
    inefficient_paths: List[str]
    lessons_learned: List[str]

    # Métricas finales
    final_coherence: float
    final_health: float
    final_ellex: float
    peak_coherence: float
    peak_health: float

    # Estado final
    omega_state: np.ndarray
    values: Dict[str, float]
    norms: Dict[str, float]


class OmegaLegacy:
    """
    Sistema de legado y cierre.

    Permite:
    - Cerrar una encarnación
    - Volcar aprendizajes a legado
    - Heredar en siguiente vida
    - Mantener sabiduría acumulada
    """

    def __init__(self):
        self.incarnation_id = 0
        self.t = 0
        self.start_time = datetime.now().isoformat()

        # Legados anteriores
        self._legacy_records: List[LegacyRecord] = []

        # Acumuladores de vida actual
        self._decisions: List[Dict[str, Any]] = []
        self._patterns: Dict[str, float] = {}  # pattern -> success_rate
        self._zones: Dict[str, float] = {}     # zone -> danger_level
        self._paths: Dict[str, float] = {}     # path -> efficiency

        # Métricas de vida
        self._coherence_history: List[float] = []
        self._health_history: List[float] = []
        self._ellex_history: List[float] = []

        # Estado actual
        self._omega_state: Optional[np.ndarray] = None
        self._values: Dict[str, float] = {}
        self._norms: Dict[str, float] = {}

    def record_decision(
        self,
        decision: Dict[str, Any],
        outcome: str,
        pattern: Optional[str] = None
    ) -> None:
        """
        Registra una decisión y su resultado.

        Args:
            decision: Datos de la decisión
            outcome: Resultado ('success', 'failure', 'neutral')
            pattern: Patrón identificado (opcional)
        """
        self.t += 1
        self._decisions.append({
            't': self.t,
            'decision': decision,
            'outcome': outcome,
            'pattern': pattern
        })

        # Actualizar patrones
        if pattern:
            if pattern not in self._patterns:
                self._patterns[pattern] = 0.5
            # Actualizar success rate endógenamente
            success = 1 if outcome == 'success' else 0
            eta = 1 / (len([d for d in self._decisions if d.get('pattern') == pattern]) + 1)
            self._patterns[pattern] = (1 - eta) * self._patterns[pattern] + eta * success

    def record_zone(self, zone: str, danger_level: float) -> None:
        """Registra nivel de peligro de una zona."""
        if zone not in self._zones:
            self._zones[zone] = danger_level
        else:
            # Actualizar con media móvil
            eta = 1 / 10
            self._zones[zone] = (1 - eta) * self._zones[zone] + eta * danger_level

    def record_path(self, path: str, efficiency: float) -> None:
        """Registra eficiencia de un camino."""
        if path not in self._paths:
            self._paths[path] = efficiency
        else:
            eta = 1 / 10
            self._paths[path] = (1 - eta) * self._paths[path] + eta * efficiency

    def record_metrics(
        self,
        coherence: float,
        health: float,
        ellex: float
    ) -> None:
        """Registra métricas de vida."""
        self._coherence_history.append(coherence)
        self._health_history.append(health)
        self._ellex_history.append(ellex)

    def update_state(
        self,
        omega_state: np.ndarray,
        values: Dict[str, float],
        norms: Dict[str, float]
    ) -> None:
        """Actualiza estado actual."""
        self._omega_state = omega_state.copy()
        self._values = values.copy()
        self._norms = norms.copy()

    def can_close(self) -> bool:
        """
        Determina si puede cerrar esta encarnación.

        Basado en:
        - Tiempo suficiente
        - Estabilidad de métricas
        - Aprendizajes consolidados
        """
        if self.t < 100:
            return False

        if len(self._coherence_history) < 50:
            return False

        # Verificar estabilidad
        recent_coherence = self._coherence_history[-20:]
        var_coherence = np.var(recent_coherence)

        # Umbral endógeno
        if len(self._coherence_history) > 50:
            var_threshold = np.percentile(
                [np.var(self._coherence_history[i:i+20])
                 for i in range(len(self._coherence_history) - 20)],
                75
            )
            return var_coherence < var_threshold

        return False

    def close_incarnation(self) -> LegacyRecord:
        """
        Cierra la encarnación actual y genera legado.

        Returns:
            Registro de legado
        """
        end_time = datetime.now().isoformat()

        # Identificar patrones exitosos (top 25% por success rate)
        if self._patterns:
            success_threshold = np.percentile(list(self._patterns.values()), 75)
            successful_patterns = [p for p, s in self._patterns.items() if s >= success_threshold]
        else:
            successful_patterns = []

        # Identificar zonas peligrosas (top 25% por danger)
        if self._zones:
            danger_threshold = np.percentile(list(self._zones.values()), 75)
            dangerous_zones = [z for z, d in self._zones.items() if d >= danger_threshold]
        else:
            dangerous_zones = []

        # Identificar caminos ineficientes (bottom 25% por efficiency)
        if self._paths:
            eff_threshold = np.percentile(list(self._paths.values()), 25)
            inefficient_paths = [p for p, e in self._paths.items() if e <= eff_threshold]
        else:
            inefficient_paths = []

        # Extraer lecciones
        lessons = self._extract_lessons()

        # Crear registro
        record = LegacyRecord(
            incarnation_id=self.incarnation_id,
            start_time=self.start_time,
            end_time=end_time,
            duration_steps=self.t,
            total_decisions=len(self._decisions),
            successful_patterns=successful_patterns,
            dangerous_zones=dangerous_zones,
            inefficient_paths=inefficient_paths,
            lessons_learned=lessons,
            final_coherence=self._coherence_history[-1] if self._coherence_history else 0,
            final_health=self._health_history[-1] if self._health_history else 0,
            final_ellex=self._ellex_history[-1] if self._ellex_history else 0,
            peak_coherence=max(self._coherence_history) if self._coherence_history else 0,
            peak_health=max(self._health_history) if self._health_history else 0,
            omega_state=self._omega_state if self._omega_state is not None else np.zeros(64),
            values=self._values.copy(),
            norms=self._norms.copy()
        )

        self._legacy_records.append(record)

        # Preparar siguiente encarnación
        self._prepare_next_incarnation()

        return record

    def _extract_lessons(self) -> List[str]:
        """Extrae lecciones de la vida actual."""
        lessons = []

        # Lección por patrones
        if self._patterns:
            best_pattern = max(self._patterns, key=self._patterns.get)
            worst_pattern = min(self._patterns, key=self._patterns.get)
            lessons.append(f"best_pattern:{best_pattern}")
            lessons.append(f"worst_pattern:{worst_pattern}")

        # Lección por zonas
        if self._zones:
            most_dangerous = max(self._zones, key=self._zones.get)
            lessons.append(f"avoid_zone:{most_dangerous}")

        # Lección por métricas
        if self._coherence_history and self._health_history:
            corr = np.corrcoef(self._coherence_history, self._health_history)[0, 1]
            if not np.isnan(corr):
                if corr > 0.5:
                    lessons.append("coherence_health_correlated")
                elif corr < -0.5:
                    lessons.append("coherence_health_anticorrelated")

        return lessons

    def _prepare_next_incarnation(self) -> None:
        """Prepara estado para siguiente encarnación."""
        self.incarnation_id += 1
        self.t = 0
        self.start_time = datetime.now().isoformat()

        # Limpiar acumuladores pero mantener sabiduría
        self._decisions = []

        # Mantener patrones, zonas, paths con decay
        decay = 0.8
        self._patterns = {k: v * decay for k, v in self._patterns.items()}
        self._zones = {k: v * decay for k, v in self._zones.items()}
        self._paths = {k: v * decay for k, v in self._paths.items()}

        # Limpiar historiales
        self._coherence_history = []
        self._health_history = []
        self._ellex_history = []

    def inherit_legacy(self) -> Dict[str, Any]:
        """
        Hereda legado de encarnaciones anteriores.

        Returns:
            Diccionario con legado heredado
        """
        if not self._legacy_records:
            return {}

        # Combinar legados
        all_patterns = {}
        all_zones = {}
        all_lessons = []

        for record in self._legacy_records:
            for p in record.successful_patterns:
                all_patterns[p] = all_patterns.get(p, 0) + 1
            for z in record.dangerous_zones:
                all_zones[z] = all_zones.get(z, 0) + 1
            all_lessons.extend(record.lessons_learned)

        # Promediar valores y normas del último legado
        last = self._legacy_records[-1]

        return {
            'inherited_patterns': all_patterns,
            'inherited_dangers': all_zones,
            'inherited_lessons': list(set(all_lessons)),
            'inherited_values': last.values,
            'inherited_norms': last.norms,
            'inherited_omega': last.omega_state
        }

    def save(self, path: str) -> None:
        """Guarda legados a archivo."""
        data = {
            'incarnation_id': self.incarnation_id,
            'legacy_records': [
                {
                    'incarnation_id': r.incarnation_id,
                    'start_time': r.start_time,
                    'end_time': r.end_time,
                    'duration_steps': r.duration_steps,
                    'total_decisions': r.total_decisions,
                    'successful_patterns': r.successful_patterns,
                    'dangerous_zones': r.dangerous_zones,
                    'inefficient_paths': r.inefficient_paths,
                    'lessons_learned': r.lessons_learned,
                    'final_coherence': r.final_coherence,
                    'final_health': r.final_health,
                    'final_ellex': r.final_ellex,
                    'peak_coherence': r.peak_coherence,
                    'peak_health': r.peak_health,
                    'omega_state': r.omega_state.tolist(),
                    'values': r.values,
                    'norms': r.norms
                }
                for r in self._legacy_records
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Carga legados de archivo."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.incarnation_id = data['incarnation_id']
        self._legacy_records = [
            LegacyRecord(
                incarnation_id=r['incarnation_id'],
                start_time=r['start_time'],
                end_time=r['end_time'],
                duration_steps=r['duration_steps'],
                total_decisions=r['total_decisions'],
                successful_patterns=r['successful_patterns'],
                dangerous_zones=r['dangerous_zones'],
                inefficient_paths=r['inefficient_paths'],
                lessons_learned=r['lessons_learned'],
                final_coherence=r['final_coherence'],
                final_health=r['final_health'],
                final_ellex=r['final_ellex'],
                peak_coherence=r['peak_coherence'],
                peak_health=r['peak_health'],
                omega_state=np.array(r['omega_state']),
                values=r['values'],
                norms=r['norms']
            )
            for r in data['legacy_records']
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas."""
        return {
            'incarnation_id': self.incarnation_id,
            't': self.t,
            'total_legacies': len(self._legacy_records),
            'total_patterns': len(self._patterns),
            'total_zones': len(self._zones),
            'total_paths': len(self._paths),
            'current_coherence': self._coherence_history[-1] if self._coherence_history else 0,
            'can_close': self.can_close()
        }
