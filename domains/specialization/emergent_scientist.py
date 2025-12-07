"""
EMERGENT SCIENTIST - Agente con Especialización Científica Emergente
=====================================================================

Extiende EmergentSpecialist para incluir dominios de Matemáticas y Física.

NORMA DURA:
- Sin roles asignados
- Especialización emerge de métricas
- Comparación intra-agente
- Sin RL ni reward

Un agente es "matemático" o "físico" porque sus métricas
en ese dominio son sistemáticamente superiores.
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
from .task_sampler import Task, TaskResult, TaskType, EvaluationMode
from .unified_task_engine import UnifiedTaskEngine, Domain


class EmergentScientist:
    """
    Agente que desarrolla especialización científica de forma emergente.

    Extiende las capacidades para incluir:
    - Matemáticas: ecuaciones, cálculo, ajuste, series
    - Física: cinemática, osciladores, sistemas acoplados

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
        domains: Optional[List[str]] = None,
        seed: Optional[int] = None
    ):
        """
        Args:
            agent_id: Identificador del agente
            domains: Lista de dominios (default: todos)
            seed: Semilla para reproducibilidad
        """
        self.agent_id = agent_id
        self.logger = get_provenance_logger()
        self.rng = np.random.default_rng(seed)

        # Dominios disponibles (incluyendo nuevos)
        self.domains = domains or [d.value for d in Domain]

        # Estadísticas por dominio
        self.domain_stats: Dict[str, DomainStats] = {
            d: DomainStats(domain=d) for d in self.domains
        }

        # Afinidades (se calculan después de suficientes tareas)
        self.affinities: Dict[str, DomainAffinity] = {}

        # Componentes
        self.task_engine = UnifiedTaskEngine(seed=seed)
        self.affinity_computer = AffinityComputer()

        # Estado
        self.t = 0
        self.total_tasks = 0
        self.task_history: List[Dict] = []

        # Personalidad (afecta cómo resuelve tareas)
        self.personality = self._discover_personality()

        # Capacidades específicas (emerge de la práctica)
        self.capabilities: Dict[str, float] = {d: 0.5 for d in self.domains}

    def _discover_personality(self) -> Dict[str, float]:
        """
        El agente descubre su personalidad.

        NOTA: La personalidad afecta CÓMO resuelve tareas,
        NO qué dominio explora.
        """
        name_hash = hashlib.sha256(self.agent_id.encode()).hexdigest()

        def trait(offset: int) -> float:
            return int(name_hash[offset:offset+4], 16) / 0xFFFF

        return {
            'analytical': trait(0) * 0.6 + 0.4,
            'intuitive': trait(4) * 0.6 + 0.4,
            'conservative': trait(8) * 0.6 + 0.4,
            'exploratory': trait(12) * 0.6 + 0.4,
            'mathematical_aptitude': trait(16) * 0.6 + 0.4,  # Nuevo
            'physical_intuition': trait(20) * 0.6 + 0.4,      # Nuevo
            '_self_discovered': True,
        }

    def select_domain(self) -> str:
        """
        Selecciona dominio para la siguiente tarea.

        NORMA DURA:
        - Usa afinidades calculadas
        - Siempre hay exploración (softmax)
        - Fase inicial: uniforme
        """
        min_samples = THEORY_CONSTANTS['min_samples_corr'].value

        # Si no hay suficientes tareas en cada dominio, explorar uniformemente
        min_tasks_per_domain = min(
            self.domain_stats[d].n_tasks for d in self.domains
        )

        if min_tasks_per_domain < min_samples:
            selected = self.rng.choice(self.domains)
            self.logger.log_from_theory(
                value=selected,
                source=f"Uniform exploration (min_tasks={min_tasks_per_domain} < {min_samples})",
                context="EmergentScientist.select_domain"
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
        return self.task_engine.sample_task(domain)

    def attempt_task(self, task: Task) -> TaskResult:
        """
        Intenta resolver una tarea.

        El agente usa diferentes estrategias según el dominio
        y su personalidad.
        """
        self.t += 1
        self.total_tasks += 1

        result = TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            started_at=datetime.now().isoformat()
        )

        # Resolver según el dominio
        if task.domain == "mathematics":
            self._solve_math_task(task, result)
        elif task.domain == "physics":
            self._solve_physics_task(task, result)
        else:
            self._solve_generic_task(task, result)

        result.completed_at = datetime.now().isoformat()
        return result

    def _solve_math_task(self, task: Task, result: TaskResult):
        """Resuelve tarea matemática."""
        math_type = task.params.get('math_task_type', '')

        # Capacidad base + aptitud matemática
        base_skill = self.capabilities.get('mathematics', 0.5)
        aptitude = self.personality['mathematical_aptitude']

        # Probabilidad de éxito basada en capacidad
        success_prob = (base_skill + aptitude) / 2

        if math_type == 'math_eq_simple':
            # Sistema de ecuaciones
            if task.oracle_solution is not None and self.rng.random() < success_prob:
                # Éxito: solución cercana al oracle
                noise_scale = 0.1 * (1 - success_prob)
                result.solution = task.oracle_solution + \
                    self.rng.standard_normal(len(task.oracle_solution)) * noise_scale
            else:
                # Fallo: solución aleatoria
                n = task.params.get('n', 1)
                result.solution = self.rng.standard_normal(n)

        elif math_type == 'math_calculus':
            # Derivada/integral
            if task.oracle_solution is not None and self.rng.random() < success_prob:
                noise_scale = 0.1 * (1 - success_prob)
                result.solution = task.oracle_solution + \
                    self.rng.standard_normal(len(task.oracle_solution)) * noise_scale
            else:
                result.solution = self.rng.standard_normal(len(task.X))

        elif math_type == 'math_fit':
            # Ajuste de funciones
            true_params = task.params.get('true_params', {})
            if self.rng.random() < success_prob:
                result.solution = {}
                for k, v in true_params.items():
                    noise = self.rng.standard_normal() * 0.2 * (1 - success_prob)
                    result.solution[k] = v * (1 + noise)
            else:
                result.solution = {k: self.rng.standard_normal() for k in true_params}

        elif math_type == 'math_series':
            # Convergencia
            if self.rng.random() < success_prob:
                result.solution = task.oracle_solution
            else:
                result.solution = {'converges': self.rng.choice([True, False])}

        # Generar hipótesis
        n_hypotheses = int(3 + self.personality['exploratory'] * 5)
        result.hypotheses_generated = [
            {'id': f"h_{i}", 'type': 'mathematical'}
            for i in range(n_hypotheses)
        ]

    def _solve_physics_task(self, task: Task, result: TaskResult):
        """Resuelve tarea de física."""
        physics_type = task.params.get('physics_task_type', '')

        # Capacidad base + intuición física
        base_skill = self.capabilities.get('physics', 0.5)
        intuition = self.personality['physical_intuition']

        success_prob = (base_skill + intuition) / 2

        if physics_type == 'phys_free_fall':
            true_params = task.params.get('true_params', {})
            if self.rng.random() < success_prob:
                result.solution = {}
                for k, v in true_params.items():
                    noise = self.rng.standard_normal() * 0.15 * (1 - success_prob)
                    result.solution[k] = v * (1 + noise)
            else:
                result.solution = {
                    'x0': self.rng.uniform(-100, 100),
                    'v0': self.rng.uniform(-50, 50),
                    'a': self.rng.uniform(-20, 20)
                }

        elif physics_type == 'phys_oscillator':
            true_params = task.params.get('true_params', {})
            if self.rng.random() < success_prob:
                result.solution = {}
                for k in ['A', 'omega', 'phi']:
                    if k in true_params:
                        noise = self.rng.standard_normal() * 0.1 * (1 - success_prob)
                        result.solution[k] = true_params[k] * (1 + noise)
                result.solution['energy_conserved'] = True  # Suele ser correcto
            else:
                result.solution = {
                    'A': self.rng.uniform(0.1, 20),
                    'omega': self.rng.uniform(0.1, 10),
                    'phi': self.rng.uniform(0, 6.28),
                    'energy_conserved': self.rng.choice([True, False])
                }

        elif physics_type == 'phys_coupled':
            true_params = task.params.get('true_params', {})
            if self.rng.random() < success_prob:
                result.solution = {
                    'is_coupled': True,
                    'coupling_type': true_params.get('coupling_type', 'bidirectional'),
                    'has_feedback': true_params.get('has_feedback', True)
                }
            else:
                result.solution = {
                    'is_coupled': self.rng.choice([True, False]),
                    'coupling_type': self.rng.choice(['unidirectional', 'bidirectional']),
                    'has_feedback': self.rng.choice([True, False])
                }

        elif physics_type == 'phys_timeseries':
            # Tarea sin ground truth
            n_hypotheses = int(2 + self.personality['exploratory'] * 4)
            result.hypotheses_generated = [
                {'id': f"h_{i}", 'type': self.rng.choice(['periodic', 'trend', 'chaotic'])}
                for i in range(n_hypotheses)
            ]
            # Falsificar algunas
            n_falsify = int(n_hypotheses * 0.4)
            result.hypotheses_falsified = result.hypotheses_generated[:n_falsify]
            result.surrogate_stability = self.rng.uniform(0.5, 1.0)

            result.solution = {
                'hypotheses': result.hypotheses_generated,
                'falsified': result.hypotheses_falsified,
                'detected_type': self.rng.choice(['periodic', 'trend', 'random_walk', 'chaotic']),
                'surrogate_stability': result.surrogate_stability
            }

        # Generar hipótesis
        if not result.hypotheses_generated:
            n_hypotheses = int(3 + self.personality['exploratory'] * 5)
            result.hypotheses_generated = [
                {'id': f"h_{i}", 'type': 'physical'}
                for i in range(n_hypotheses)
            ]

    def _solve_generic_task(self, task: Task, result: TaskResult):
        """Resuelve tarea genérica (dominios originales)."""
        # Similar a EmergentSpecialist original
        X_train, X_test, y_train, y_test = task.get_train_test_split(
            test_fraction=0.2
        )

        base_skill = self.capabilities.get(task.domain, 0.5)

        if task.task_type == TaskType.CLASSIFICATION:
            predictions, probabilities = self._solve_classification(
                X_train, y_train, X_test, base_skill
            )
            result.predictions = predictions
            result.probabilities = probabilities

        elif task.task_type == TaskType.REGRESSION:
            predictions = self._solve_regression(X_train, y_train, X_test, base_skill)
            result.predictions = predictions

        elif task.task_type == TaskType.ANOMALY:
            predictions, probabilities = self._solve_anomaly(
                X_train, y_train, X_test, base_skill
            )
            result.predictions = predictions
            result.probabilities = probabilities

        # Generar hipótesis
        n_hypotheses = int(3 + self.personality['exploratory'] * 5)
        result.hypotheses_generated = [
            {'id': f"h_{i}", 'type': 'correlation'}
            for i in range(n_hypotheses)
        ]

    def _solve_classification(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        skill: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resuelve clasificación."""
        if y_train is None:
            return np.zeros(len(X_test)), np.ones(len(X_test)) * 0.5

        X_train_b = np.c_[np.ones(len(X_train)), X_train]
        X_test_b = np.c_[np.ones(len(X_test)), X_test]

        try:
            weights = np.linalg.lstsq(X_train_b, y_train, rcond=None)[0]
        except:
            weights = np.zeros(X_train_b.shape[1])

        logits = X_test_b @ weights

        if self.personality['conservative'] > 0.6:
            logits = logits * 0.8

        probabilities = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        predictions = (probabilities > 0.5).astype(int)

        return predictions, probabilities

    def _solve_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        skill: float
    ) -> np.ndarray:
        """Resuelve regresión."""
        if y_train is None:
            return np.zeros(len(X_test))

        X_train_b = np.c_[np.ones(len(X_train)), X_train]
        X_test_b = np.c_[np.ones(len(X_test)), X_test]

        try:
            weights = np.linalg.lstsq(X_train_b, y_train, rcond=None)[0]
        except:
            weights = np.zeros(X_train_b.shape[1])

        predictions = X_test_b @ weights

        if self.personality['conservative'] > 0.6:
            mean_y = np.mean(y_train)
            predictions = predictions * 0.8 + mean_y * 0.2

        return predictions

    def _solve_anomaly(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        skill: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resuelve detección de anomalías."""
        centroid = np.mean(X_train, axis=0)
        distances = np.linalg.norm(X_test - centroid, axis=1)

        train_distances = np.linalg.norm(X_train - centroid, axis=1)
        threshold = np.percentile(train_distances, 95)

        predictions = (distances > threshold).astype(int)
        probabilities = distances / (distances.max() + 1e-10)

        return predictions, probabilities

    def evaluate_and_update(self, task: Task, result: TaskResult) -> DomainMetrics:
        """
        Evalúa resultado y actualiza estadísticas.

        Este es el paso clave donde el agente aprende
        en qué dominios rinde mejor.
        """
        # Evaluar con motor unificado
        metrics_dict = self.task_engine.evaluate_result(task, result)

        # Simular confirmación/falsificación de hipótesis
        n_gen = len(result.hypotheses_generated)
        accuracy = metrics_dict.get('accuracy', 0.5)

        if accuracy > 0.6:
            n_confirmed = int(n_gen * 0.6)
        else:
            n_confirmed = int(n_gen * 0.3)
        n_falsified = n_gen - n_confirmed

        result.hypotheses_confirmed = result.hypotheses_generated[:n_confirmed]
        if not result.hypotheses_falsified:
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

        # Actualizar capacidad basada en rendimiento
        self._update_capability(task.domain, accuracy)

        # Registrar en historial
        self.task_history.append({
            't': self.t,
            'domain': task.domain,
            'task_type': task.task_type.value,
            'accuracy': metrics_dict.get('accuracy'),
            'error': metrics_dict.get('error'),
        })

        # Recalcular afinidades periódicamente
        update_interval = int(np.sqrt(self.total_tasks)) + 1
        if self.total_tasks % update_interval == 0:
            self._update_affinities()

        return domain_metrics

    def _update_capability(self, domain: str, accuracy: float):
        """
        Actualiza capacidad en un dominio basado en rendimiento.

        ORIGEN: Media móvil exponencial
        """
        # EMA con factor derivado de datos
        alpha = 0.1  # Factor de suavizado
        old_cap = self.capabilities.get(domain, 0.5)
        new_cap = alpha * accuracy + (1 - alpha) * old_cap
        self.capabilities[domain] = new_cap

        self.logger.log_from_data(
            value=new_cap,
            source=f"EMA(capability_{domain}, α=0.1)",
            statistic="exponential_moving_average",
            context="EmergentScientist._update_capability"
        )

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

        NO asigna etiquetas como "matemático" o "físico".
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
        report['capabilities'] = self.capabilities.copy()
        report['personality'] = {
            k: v for k, v in self.personality.items()
            if not k.startswith('_')
        }

        return report


# =============================================================================
# TEST
# =============================================================================

def test_emergent_scientist():
    """Test del sistema de científico emergente."""
    print("=" * 70)
    print("TEST: EMERGENT SCIENTIST")
    print("Los agentes se especializan en matemáticas/física por rendimiento")
    print("=" * 70)

    # Crear agentes
    agents = [
        EmergentScientist("GAUSS", seed=42),
        EmergentScientist("NEWTON", seed=123),
        EmergentScientist("EULER", seed=456),
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

        print(f"\n    Capacidades:")
        for d, c in report['capabilities'].items():
            print(f"      {d}: {c:.3f}")

        print(f"\n    Ranking de dominios:")
        for i, dr in enumerate(report['domain_ranking'][:4]):
            print(f"      {i+1}. {dr['domain']}: score={dr['score']:.3f}, n_tasks={dr['n_tasks']}")

    print("\n=== RESUMEN FINAL ===")
    print("\nNOTA: Los agentes NO tienen roles asignados.")
    print("GAUSS, NEWTON, EULER pueden especializarse en cualquier dominio.")
    print("Su 'especialización' emerge de las métricas.")

    return agents


if __name__ == "__main__":
    test_emergent_scientist()
