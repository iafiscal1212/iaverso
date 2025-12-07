"""
DOMAIN AFFINITY - Sistema de Afinidad por Dominio
==================================================

Calcula y actualiza la afinidad de un agente hacia cada dominio
de forma ENDÓGENA, basándose en métricas de rendimiento.

NORMA DURA:
- Sin umbrales mágicos (0.8, 0.7, etc.)
- Afinidad basada en z-scores relativos
- Sin RL ni reward
- Comparación intra-agente (no inter-agente)

Un agente tiene alta afinidad a un dominio si:
- Su rendimiento en ese dominio es > media de sus otros dominios
- Su falsificación en ese dominio es < media
- Su estabilidad en ese dominio es > media
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stimuli_engine.provenance import (
    Provenance, ProvenanceType, get_provenance_logger, THEORY_CONSTANTS
)

from .domain_stats import DomainStats, DomainMetrics


@dataclass
class DomainAffinity:
    """
    Afinidad de un agente hacia un dominio.

    NORMA DURA: La afinidad NO es un número fijo.
    Es un z-score relativo al rendimiento del agente en otros dominios.
    """
    domain: str
    raw_score: float = 0.0          # Score compuesto (z-score)
    percentile_rank: float = 0.5    # Percentil dentro de los dominios del agente

    # Componentes del score
    performance_z: float = 0.0      # z-score de rendimiento
    stability_z: float = 0.0        # z-score de estabilidad
    falsification_z: float = 0.0    # z-score de falsificación (invertido)

    # Metadatos
    n_tasks: int = 0
    last_updated: str = ""

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            'domain': self.domain,
            'raw_score': self.raw_score,
            'percentile_rank': self.percentile_rank,
            'performance_z': self.performance_z,
            'stability_z': self.stability_z,
            'falsification_z': self.falsification_z,
            'n_tasks': self.n_tasks,
            'last_updated': self.last_updated,
        }


class AffinityComputer:
    """
    Calcula afinidades de un agente hacia dominios.

    NORMA DURA:
    - Usa z-scores para normalización
    - Compara dentro del mismo agente
    - Sin umbrales absolutos
    - Pesos uniformes (máxima entropía)
    """

    def __init__(self):
        self.logger = get_provenance_logger()

    def compute_affinities(
        self,
        domain_stats: Dict[str, DomainStats]
    ) -> Dict[str, DomainAffinity]:
        """
        Calcula afinidades para todos los dominios de un agente.

        Args:
            domain_stats: Dict[domain] -> DomainStats del agente

        Returns:
            Dict[domain] -> DomainAffinity
        """
        if not domain_stats:
            return {}

        # Extraer métricas de rendimiento de cada dominio
        performance_scores = {}
        stability_scores = {}
        falsification_scores = {}
        n_tasks_per_domain = {}

        for domain, stats in domain_stats.items():
            # Performance = AUROC o accuracy (lo que haya)
            auroc_mean, _ = stats.get_mean_metric('auroc')
            acc_mean, _ = stats.get_mean_metric('accuracy')

            if auroc_mean is not None:
                performance_scores[domain] = auroc_mean
            elif acc_mean is not None:
                performance_scores[domain] = acc_mean
            else:
                performance_scores[domain] = None

            # Stability
            stability, _ = stats.get_stability()
            stability_scores[domain] = stability

            # Falsification (queremos que sea baja)
            fals_mean, _ = stats.get_mean_metric('falsification')
            falsification_scores[domain] = fals_mean

            n_tasks_per_domain[domain] = stats.n_tasks

        # Calcular z-scores relativos
        affinities = {}

        for domain in domain_stats.keys():
            perf_z = self._compute_relative_z(domain, performance_scores)
            stab_z = self._compute_relative_z(domain, stability_scores)
            # Falsificación invertida (menos es mejor)
            fals_z = -self._compute_relative_z(domain, falsification_scores)

            # Score compuesto = suma de z-scores
            # ORIGEN: Pesos uniformes (1/3 cada uno) = máxima entropía
            n_components = 3
            raw_score = (perf_z + stab_z + fals_z) / n_components

            self.logger.log_from_theory(
                value={'perf_z': perf_z, 'stab_z': stab_z, 'fals_z': fals_z},
                source="z-score composition with uniform weights (1/n)",
                reference="Maximum entropy principle",
                context="AffinityComputer.compute_affinities"
            )

            affinities[domain] = DomainAffinity(
                domain=domain,
                raw_score=raw_score,
                performance_z=perf_z,
                stability_z=stab_z,
                falsification_z=fals_z,
                n_tasks=n_tasks_per_domain.get(domain, 0)
            )

        # Calcular percentiles
        scores = [a.raw_score for a in affinities.values()]
        for domain, affinity in affinities.items():
            # Percentil = fracción de dominios con score menor
            n_below = sum(1 for s in scores if s < affinity.raw_score)
            affinity.percentile_rank = n_below / len(scores) if scores else 0.5

        return affinities

    def _compute_relative_z(
        self,
        domain: str,
        scores: Dict[str, Optional[float]]
    ) -> float:
        """
        Calcula z-score relativo de un dominio respecto a los demás.

        ORIGEN: z = (x - μ) / σ
        donde μ y σ son de los otros dominios del mismo agente.
        """
        value = scores.get(domain)
        if value is None:
            return 0.0

        # Otros valores (excluyendo el dominio actual)
        other_values = [v for d, v in scores.items()
                       if d != domain and v is not None]

        if len(other_values) < 2:
            # No hay suficientes datos para comparar
            return 0.0

        mean_others = np.mean(other_values)
        std_others = np.std(other_values, ddof=1)

        eps = np.finfo(float).eps
        if std_others < eps:
            # Sin variación, z = 0
            return 0.0

        z = (value - mean_others) / std_others

        self.logger.log_from_theory(
            value=z,
            source=f"z = ({value:.4f} - {mean_others:.4f}) / {std_others:.4f}",
            reference="z-score relative to other domains",
            context="AffinityComputer._compute_relative_z"
        )

        return z

    def get_exploration_weights(
        self,
        affinities: Dict[str, DomainAffinity],
        exploration_factor: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calcula pesos para selección de dominio.

        NORMA DURA:
        - Sin RL
        - Pesos basados en softmax de afinidades
        - exploration_factor derivado de datos si no se especifica

        Args:
            affinities: Afinidades calculadas
            exploration_factor: Factor de exploración (temperatura)
                               Si None, se deriva de la varianza de afinidades

        Returns:
            Dict[domain] -> peso normalizado
        """
        if not affinities:
            return {}

        scores = {d: a.raw_score for d, a in affinities.items()}

        # Derivar exploration_factor de datos si no se especifica
        if exploration_factor is None:
            score_values = list(scores.values())
            if len(score_values) > 1:
                # ORIGEN: Usar std de scores como temperatura
                # Alta varianza -> más explotación
                # Baja varianza -> más exploración
                std_scores = np.std(score_values)
                # Temperatura inversamente proporcional a varianza
                exploration_factor = 1.0 / (1.0 + std_scores)
            else:
                exploration_factor = 1.0

            self.logger.log_from_data(
                value=exploration_factor,
                source="1 / (1 + std(affinity_scores))",
                statistic="derived_temperature",
                context="AffinityComputer.get_exploration_weights"
            )

        # Softmax con temperatura
        # ORIGEN: Distribución de Boltzmann
        score_array = np.array(list(scores.values()))
        # Escalar por temperatura
        scaled = score_array / exploration_factor

        # Softmax numéricamente estable
        scaled = scaled - np.max(scaled)
        exp_scores = np.exp(scaled)
        probs = exp_scores / np.sum(exp_scores)

        self.logger.log_from_theory(
            value=exploration_factor,
            source="Boltzmann softmax: P(d) ∝ exp(score_d / T)",
            reference="Statistical mechanics / softmax",
            context="AffinityComputer.get_exploration_weights"
        )

        return {d: p for d, p in zip(scores.keys(), probs)}

    def select_domain(
        self,
        affinities: Dict[str, DomainAffinity],
        exploration_factor: Optional[float] = None,
        seed: Optional[int] = None
    ) -> str:
        """
        Selecciona un dominio para explorar.

        NORMA DURA:
        - Probabilidad basada en afinidad (softmax)
        - Siempre hay algo de exploración
        - Sin RL

        Returns:
            Dominio seleccionado
        """
        if seed is not None:
            np.random.seed(seed)

        weights = self.get_exploration_weights(affinities, exploration_factor)

        if not weights:
            return None

        domains = list(weights.keys())
        probs = list(weights.values())

        selected = np.random.choice(domains, p=probs)

        return selected

    def get_specialization_report(
        self,
        affinities: Dict[str, DomainAffinity]
    ) -> Dict[str, Any]:
        """
        Genera reporte de especialización.

        NO asigna etiquetas como "médico" o "financiero".
        Solo reporta métricas que la humana puede interpretar.
        """
        if not affinities:
            return {'status': 'no_data'}

        # Ordenar por raw_score
        sorted_affinities = sorted(
            affinities.values(),
            key=lambda a: a.raw_score,
            reverse=True
        )

        # Calcular significancia de especialización
        # ORIGEN: Si el top dominio tiene z > 1 respecto a los demás
        top = sorted_affinities[0]
        others = sorted_affinities[1:]

        if others:
            other_scores = [a.raw_score for a in others]
            mean_others = np.mean(other_scores)
            std_others = np.std(other_scores, ddof=1) if len(other_scores) > 1 else 1.0

            eps = np.finfo(float).eps
            if std_others > eps:
                specialization_z = (top.raw_score - mean_others) / std_others
            else:
                specialization_z = 0.0
        else:
            specialization_z = 0.0

        # Determinar si hay especialización significativa
        # ORIGEN: z > 1 es 1 std por encima de la media
        # No es un umbral mágico, es la definición de z-score
        has_specialization = specialization_z > 1.0

        self.logger.log_from_theory(
            value=specialization_z,
            source="z_specialization = (top_score - mean_others) / std_others",
            reference="z > 1 means 1 std above mean",
            context="AffinityComputer.get_specialization_report"
        )

        return {
            'top_domain': top.domain,
            'top_score': top.raw_score,
            'top_n_tasks': top.n_tasks,
            'specialization_z': specialization_z,
            'has_significant_specialization': has_specialization,
            'domain_ranking': [
                {
                    'domain': a.domain,
                    'score': a.raw_score,
                    'percentile': a.percentile_rank,
                    'n_tasks': a.n_tasks,
                }
                for a in sorted_affinities
            ],
            'exploration_weights': self.get_exploration_weights(affinities),
        }
