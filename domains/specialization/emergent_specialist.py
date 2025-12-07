"""
EMERGENT SPECIALIST - Agente con Especialización Emergente
============================================================

Un agente que explora múltiples dominios y se especializa
de forma ENDÓGENA basándose en su rendimiento.

NORMA DURA:
- Sin roles asignados (no es MedicalAgent ni FinanceAgent)
- Especialización emerge de métricas
- Sin RL ni reward
- Comparación intra-agente

El agente es "médico" o "financiero" solo porque sus métricas
en ese dominio son sistemáticamente superiores a otros dominios.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stimuli_engine.provenance import get_provenance_logger, THEORY_CONSTANTS

from .domain_stats import DomainStats, DomainMetrics
from .domain_affinity import DomainAffinity, AffinityComputer
from .task_sampler import DomainTaskSampler, Task, TaskResult, TaskType


class EmergentSpecialist:
    """
    Agente que desarrolla especialización de forma emergente.

    NO tiene dominio asignado. Explora todos los dominios
    y naturalmente se especializa donde rinde mejor.

    CICLO:
    1. Selecciona dominio (basado en afinidades)
    2. Obtiene tarea del dominio
    3. Intenta resolver la tarea
    4. Recibe métricas de evaluación
    5. Actualiza estadísticas del dominio
    6. Recalcula afinidades
    7. Repeat

    La especialización EMERGE de este ciclo, no se impone.
    """

    def __init__(
        self,
        agent_id: str,
        domains: Optional[List[str]] = None
    ):
        """
        Args:
            agent_id: Identificador del agente
            domains: Lista de dominios disponibles
        """
        self.agent_id = agent_id
        self.logger = get_provenance_logger()

        # Dominios disponibles
        self.domains = domains or ['medicine', 'finance', 'cosmology', 'engineering']

        # Estadísticas por dominio
        self.domain_stats: Dict[str, DomainStats] = {
            d: DomainStats(domain=d) for d in self.domains
        }

        # Afinidades (se calculan después de suficientes tareas)
        self.affinities: Dict[str, DomainAffinity] = {}

        # Componentes
        self.task_sampler = DomainTaskSampler(domains=self.domains)
        self.affinity_computer = AffinityComputer()

        # Estado
        self.t = 0
        self.total_tasks = 0
        self.task_history: List[Dict] = []

        # Personalidad (afecta cómo resuelve tareas, no qué dominio explora)
        self.personality = self._discover_personality()

    def _discover_personality(self) -> Dict[str, float]:
        """
        El agente descubre su personalidad.

        NOTA: La personalidad afecta CÓMO resuelve tareas,
        NO qué dominio explora. La exploración depende de rendimiento.
        """
        name_hash = hashlib.sha256(self.agent_id.encode()).hexdigest()

        def trait(offset: int) -> float:
            return int(name_hash[offset:offset+4], 16) / 0xFFFF

        return {
            'analytical': trait(0) * 0.6 + 0.4,      # Prefiere análisis riguroso
            'intuitive': trait(4) * 0.6 + 0.4,       # Usa heurísticas
            'conservative': trait(8) * 0.6 + 0.4,   # Predicciones conservadoras
            'exploratory': trait(12) * 0.6 + 0.4,   # Explora hipótesis
            '_self_discovered': True,
        }

    def select_domain(self) -> str:
        """
        Selecciona dominio para la siguiente tarea.

        NORMA DURA:
        - Usa afinidades calculadas (no asignación fija)
        - Siempre hay exploración (softmax)
        - Fase inicial: uniforme

        Returns:
            Dominio seleccionado
        """
        # Fase inicial: exploración uniforme
        min_samples = THEORY_CONSTANTS['min_samples_corr'].value

        # Si no hay suficientes tareas en cada dominio, explorar uniformemente
        min_tasks_per_domain = min(
            self.domain_stats[d].n_tasks for d in self.domains
        )

        if min_tasks_per_domain < min_samples:
            # Exploración uniforme
            selected = np.random.choice(self.domains)
            self.logger.log_from_theory(
                value=selected,
                source=f"Uniform exploration (min_tasks={min_tasks_per_domain} < {min_samples})",
                context="EmergentSpecialist.select_domain"
            )
            return selected

        # Fase de especialización: usar afinidades
        if not self.affinities:
            self._update_affinities()

        selected = self.affinity_computer.select_domain(self.affinities)

        return selected

    def _update_affinities(self):
        """Recalcula afinidades basándose en estadísticas."""
        self.affinities = self.affinity_computer.compute_affinities(
            self.domain_stats
        )

    def get_task(self, domain: str) -> Task:
        """Obtiene una tarea del dominio."""
        return self.task_sampler.sample_task(domain)

    def attempt_task(self, task: Task) -> TaskResult:
        """
        Intenta resolver una tarea.

        El agente usa su propia estrategia (basada en personalidad)
        para generar predicciones.

        NOTA: Esta es una implementación simplificada.
        En producción, el agente usaría sus herramientas de
        causalidad, sincronicidad, etc.
        """
        self.t += 1
        self.total_tasks += 1

        result = TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            started_at=datetime.now().isoformat()
        )

        # Estrategia de resolución basada en personalidad
        X_train, X_test, y_train, y_test = task.get_train_test_split(
            test_fraction=0.2
        )

        if task.task_type == TaskType.CLASSIFICATION:
            predictions, probabilities = self._solve_classification(
                X_train, y_train, X_test
            )
            result.predictions = predictions
            result.probabilities = probabilities

        elif task.task_type == TaskType.REGRESSION:
            predictions = self._solve_regression(X_train, y_train, X_test)
            result.predictions = predictions

        elif task.task_type == TaskType.ANOMALY:
            predictions, probabilities = self._solve_anomaly(X_train, y_train, X_test)
            result.predictions = predictions
            result.probabilities = probabilities

        result.completed_at = datetime.now().isoformat()

        # Generar hipótesis (simulado)
        n_hypotheses = int(3 + self.personality['exploratory'] * 5)
        result.hypotheses_generated = [
            {'id': f"h_{i}", 'type': 'correlation'}
            for i in range(n_hypotheses)
        ]

        # Simular confirmación/falsificación basada en rendimiento
        # (en producción, esto vendría de tests reales de hipótesis)

        return result

    def _solve_classification(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resuelve tarea de clasificación.

        Estrategia basada en personalidad del agente.
        """
        if y_train is None:
            return np.zeros(len(X_test)), np.ones(len(X_test)) * 0.5

        # Estrategia simple: regresión logística manual
        # (en producción, el agente usaría métodos más sofisticados)

        # Añadir bias
        X_train_b = np.c_[np.ones(len(X_train)), X_train]
        X_test_b = np.c_[np.ones(len(X_test)), X_test]

        # Resolver por mínimos cuadrados (aproximación)
        try:
            # ORIGEN: Pseudo-inversa de Moore-Penrose
            weights = np.linalg.lstsq(X_train_b, y_train, rcond=None)[0]
        except:
            weights = np.zeros(X_train_b.shape[1])

        # Predicciones
        logits = X_test_b @ weights

        # Ajustar por personalidad
        if self.personality['conservative'] > 0.6:
            # Predicciones más cercanas a 0.5
            logits = logits * 0.8

        # Sigmoid
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        predictions = (probabilities > 0.5).astype(int)

        return predictions, probabilities

    def _solve_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray
    ) -> np.ndarray:
        """Resuelve tarea de regresión."""
        if y_train is None:
            return np.zeros(len(X_test))

        X_train_b = np.c_[np.ones(len(X_train)), X_train]
        X_test_b = np.c_[np.ones(len(X_test)), X_test]

        try:
            weights = np.linalg.lstsq(X_train_b, y_train, rcond=None)[0]
        except:
            weights = np.zeros(X_train_b.shape[1])

        predictions = X_test_b @ weights

        # Ajustar por personalidad
        if self.personality['conservative'] > 0.6:
            # Regresión hacia la media
            mean_y = np.mean(y_train)
            predictions = predictions * 0.8 + mean_y * 0.2

        return predictions

    def _solve_anomaly(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resuelve tarea de detección de anomalías."""
        # Calcular distancia al centroide de X_train
        centroid = np.mean(X_train, axis=0)
        distances = np.linalg.norm(X_test - centroid, axis=1)

        # Umbral basado en percentil de distancias de entrenamiento
        train_distances = np.linalg.norm(X_train - centroid, axis=1)

        # ORIGEN: Percentil 95 para umbral de anomalía
        # Basado en la distribución de los datos de entrenamiento
        threshold = np.percentile(train_distances, 95)

        self.logger.log_from_data(
            value=threshold,
            source="percentile(train_distances, 95)",
            statistic="anomaly_threshold",
            context="EmergentSpecialist._solve_anomaly"
        )

        predictions = (distances > threshold).astype(int)
        probabilities = distances / (distances.max() + 1e-10)

        return predictions, probabilities

    def evaluate_and_update(self, task: Task, result: TaskResult) -> DomainMetrics:
        """
        Evalúa resultado y actualiza estadísticas.

        Este es el paso clave donde el agente aprende
        en qué dominios rinde mejor.
        """
        # Evaluar con oracle
        metrics_dict = self.task_sampler.evaluate_result(task, result)

        # Simular confirmación/falsificación de hipótesis
        n_gen = len(result.hypotheses_generated)
        if metrics_dict.get('accuracy', 0.5) > 0.6:
            n_confirmed = int(n_gen * 0.6)
        else:
            n_confirmed = int(n_gen * 0.3)
        n_falsified = n_gen - n_confirmed

        result.hypotheses_confirmed = result.hypotheses_generated[:n_confirmed]
        result.hypotheses_falsified = result.hypotheses_generated[n_confirmed:]

        # Crear métricas de dominio
        domain_metrics = DomainMetrics(
            task_id=task.task_id,
            domain=task.domain,
            task_type=task.task_type.value,
            n_samples=task.n_samples,
            auroc=metrics_dict.get('auroc'),
            accuracy=metrics_dict.get('accuracy'),
            brier_score=metrics_dict.get('brier_score'),
            fpr=metrics_dict.get('fpr'),
            fnr=metrics_dict.get('fnr'),
            mse=metrics_dict.get('mse'),
            mae=metrics_dict.get('mae'),
            r_squared=metrics_dict.get('r_squared'),
            n_hypotheses_generated=n_gen,
            n_hypotheses_confirmed=n_confirmed,
            n_hypotheses_falsified=n_falsified,
        )

        # Actualizar estadísticas del dominio
        self.domain_stats[task.domain].add_metrics(domain_metrics)

        # Registrar en historial
        self.task_history.append({
            't': self.t,
            'domain': task.domain,
            'task_type': task.task_type.value,
            'accuracy': metrics_dict.get('accuracy'),
            'auroc': metrics_dict.get('auroc'),
        })

        # Recalcular afinidades periódicamente
        # ORIGEN: Cada √n tareas
        update_interval = int(np.sqrt(self.total_tasks)) + 1
        if self.total_tasks % update_interval == 0:
            self._update_affinities()

        return domain_metrics

    def run_exploration_cycle(self, n_tasks: int = 100) -> Dict[str, Any]:
        """
        Ejecuta ciclo de exploración.

        El agente explora dominios y naturalmente se especializa.
        """
        for i in range(n_tasks):
            # 1. Seleccionar dominio
            domain = self.select_domain()

            # 2. Obtener tarea
            task = self.get_task(domain)

            # 3. Intentar resolver
            result = self.attempt_task(task)

            # 4. Evaluar y actualizar
            self.evaluate_and_update(task, result)

            # Progress
            if (i + 1) % 20 == 0:
                self._update_affinities()

        # Reporte final
        return self.get_specialization_report()

    def get_specialization_report(self) -> Dict[str, Any]:
        """
        Genera reporte de especialización.

        NO asigna etiquetas como "médico" o "financiero".
        Solo reporta métricas objetivas.
        """
        self._update_affinities()

        report = self.affinity_computer.get_specialization_report(self.affinities)

        # Añadir estadísticas por dominio
        report['agent_id'] = self.agent_id
        report['total_tasks'] = self.total_tasks
        report['domain_stats'] = {
            d: stats.get_summary()
            for d, stats in self.domain_stats.items()
        }
        report['personality'] = {
            k: v for k, v in self.personality.items()
            if not k.startswith('_')
        }

        return report

    def get_domain_profile(self) -> Dict[str, DomainStats]:
        """Retorna perfil de dominio (para inspección externa)."""
        return self.domain_stats


# =============================================================================
# TEST
# =============================================================================

def test_emergent_specialization():
    """Test del sistema de especialización emergente."""
    print("=" * 70)
    print("TEST: EMERGENT SPECIALIZATION")
    print("Los agentes se especializan por rendimiento, no por asignación")
    print("=" * 70)

    # Crear agentes
    agents = [
        EmergentSpecialist("NEO"),
        EmergentSpecialist("EVA"),
        EmergentSpecialist("ALEX"),
    ]

    print("\n=== PERSONALIDADES ===")
    for agent in agents:
        print(f"{agent.agent_id}:")
        for k, v in agent.personality.items():
            if not k.startswith('_'):
                print(f"  {k}: {v:.3f}")

    print("\n=== CICLO DE EXPLORACIÓN (100 tareas cada uno) ===")
    for agent in agents:
        print(f"\n{agent.agent_id} explorando...")
        report = agent.run_exploration_cycle(n_tasks=100)

        print(f"\n  Resultado de {agent.agent_id}:")
        print(f"    Total tareas: {report['total_tasks']}")
        print(f"    Top dominio: {report['top_domain']} (z={report['specialization_z']:.3f})")
        print(f"    ¿Especialización significativa? {report['has_significant_specialization']}")

        print(f"\n    Ranking de dominios:")
        for i, dr in enumerate(report['domain_ranking']):
            print(f"      {i+1}. {dr['domain']}: score={dr['score']:.3f}, "
                  f"n_tasks={dr['n_tasks']}, percentile={dr['percentile']:.2f}")

        print(f"\n    Pesos de exploración:")
        for d, w in report['exploration_weights'].items():
            print(f"      {d}: {w:.3f}")

    print("\n=== RESUMEN FINAL ===")
    print("\nNOTA: Los agentes NO tienen roles asignados.")
    print("Su 'especialización' es simplemente el dominio donde rinden mejor.")
    print("La humana interpreta las métricas.")

    return agents


if __name__ == "__main__":
    test_emergent_specialization()
