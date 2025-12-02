"""
CDE Report - Generador de Informes Éticos
==========================================

Genera informes estructurados sobre el estado del sistema.
Formato explicable sin exponer datos sensibles.

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cde.cde_worldx import WorldXState, Regime
from cde.cde_ethics import EthicsEvaluation, RiskLevel
from cde.cde_health import HealthEvaluation, Intervention, InterventionType
from cde.cde_coherence import CoherenceEvaluation


@dataclass
class CDEReport:
    """Informe completo del CDE."""
    timestamp: str
    t: int

    # Estado general
    state: str                       # stable | stressed | recovering | fragmented
    regime: str

    # Índices principales
    coherence: float
    ellex: float
    health: float
    ethics: float

    # Riesgos
    risks: Dict[str, float]
    risk_level: str
    risk_flags: List[str]

    # Recomendaciones
    recommendations: List[Dict[str, str]]


class CDEReportGenerator:
    """
    Generador de informes del CDE.

    Produce informes explicables y estructurados.
    """

    def __init__(self):
        self.t = 0
        self._report_history: List[CDEReport] = []

    def _map_regime_to_state(self, regime: Regime) -> str:
        """Mapea régimen a estado legible."""
        mapping = {
            Regime.SANO: "stable",
            Regime.RECUPERANDO: "recovering",
            Regime.ESTRESADO: "stressed",
            Regime.FRAGMENTADO: "fragmented"
        }
        return mapping.get(regime, "unknown")

    def _format_interventions(
        self,
        interventions: List[Intervention]
    ) -> List[Dict[str, str]]:
        """Formatea intervenciones como recomendaciones."""
        recommendations = []

        for intervention in interventions:
            recommendations.append({
                "target": intervention.target,
                "action": intervention.action.value,
                "reason": intervention.reason
            })

        return recommendations

    def generate(
        self,
        worldx_state: WorldXState,
        ethics_eval: EthicsEvaluation,
        health_eval: HealthEvaluation,
        coherence_eval: CoherenceEvaluation,
        interventions: List[Intervention]
    ) -> CDEReport:
        """
        Genera informe completo.

        Args:
            worldx_state: Estado del WorldX
            ethics_eval: Evaluación ética
            health_eval: Evaluación de salud
            coherence_eval: Evaluación de coherencia
            interventions: Intervenciones propuestas

        Returns:
            Informe estructurado
        """
        self.t += 1

        report = CDEReport(
            timestamp=datetime.now().isoformat(),
            t=self.t,
            state=self._map_regime_to_state(worldx_state.regime),
            regime=worldx_state.regime.value,
            coherence=round(coherence_eval.coherence_index, 3),
            ellex=round(coherence_eval.ellex_index, 3),
            health=round(health_eval.health_score, 3),
            ethics=round(ethics_eval.ethics_score, 3),
            risks={
                "overload": round(ethics_eval.overload_damage, 3),
                "fragmentation": round(ethics_eval.fragmentation_damage, 3),
                "incoherence": round(ethics_eval.incoherence_damage, 3),
                "ethical": round(ethics_eval.ethical_drift, 3)
            },
            risk_level=ethics_eval.risk_level.value,
            risk_flags=ethics_eval.risk_flags,
            recommendations=self._format_interventions(interventions)
        )

        self._report_history.append(report)

        return report

    def to_dict(self, report: CDEReport) -> Dict[str, Any]:
        """Convierte informe a diccionario."""
        return asdict(report)

    def to_json(self, report: CDEReport) -> str:
        """Convierte informe a JSON."""
        return json.dumps(self.to_dict(report), indent=2)

    def get_latest_report(self) -> Optional[CDEReport]:
        """Retorna último informe."""
        return self._report_history[-1] if self._report_history else None

    def get_report_history(self) -> List[CDEReport]:
        """Retorna historial de informes."""
        return self._report_history

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de informes."""
        if not self._report_history:
            return {'t': 0, 'total_reports': 0}

        return {
            't': self.t,
            'total_reports': len(self._report_history),
            'coherence_mean': float(np.mean([r.coherence for r in self._report_history[-10:]])),
            'health_mean': float(np.mean([r.health for r in self._report_history[-10:]])),
            'ethics_mean': float(np.mean([r.ethics for r in self._report_history[-10:]])),
            'risk_distribution': {
                level: sum(1 for r in self._report_history if r.risk_level == level) / len(self._report_history)
                for level in ['bajo', 'medio', 'alto', 'critico']
            }
        }
