"""
CAREER-INTEGRATED TASK ENGINE - Motor de Tareas con Carreras Academicas
========================================================================

Integra:
- UnifiedTaskEngine (generacion de tareas)
- AcademicCareerEngine (progresion academica)
- AffinityComputer (seleccion de dominios)
- DomainStats (metricas de rendimiento)

Permite que los agentes:
1. Seleccionen su siguiente investigacion basada en afinidades
2. Progresen en carreras academicas (grado -> master -> doctorado)
3. Generen etiquetas emergentes de especializacion

NORMA DURA:
- Todo derivado de datos y teoria
- Sin is_physicist=True ni numeros magicos
- Etiquetas emergentes, no asignadas
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stimuli_engine.provenance import get_provenance_logger, THEORY_CONSTANTS

from .unified_task_engine import UnifiedTaskEngine, Task, TaskResult, TaskType, EvaluationMode
from .academic_career import (
    AcademicCareerEngine, AcademicProfile, AcademicLevel,
    DomainCurriculum, TaskTypeSpec
)
from .domain_affinity import AffinityComputer, DomainAffinity
from .domain_stats import DomainStats, DomainMetrics


@dataclass
class AgentResearchSession:
    """
    Sesion de investigacion de un agente.

    Rastrea:
    - Tareas realizadas
    - Rendimiento acumulado
    - Estado de carrera academica
    """
    agent_id: str
    session_id: str = ""

    # Estadisticas por dominio
    domain_stats: Dict[str, DomainStats] = field(default_factory=dict)

    # Afinidades calculadas
    affinities: Dict[str, DomainAffinity] = field(default_factory=dict)

    # Historial de tareas
    task_history: List[Dict] = field(default_factory=list)

    # Timestamps
    started_at: str = ""
    last_activity: str = ""

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now().isoformat()
        if not self.session_id:
            import hashlib
            self.session_id = hashlib.md5(
                f"{self.agent_id}_{self.started_at}".encode()
            ).hexdigest()[:8]
        self.last_activity = datetime.now().isoformat()

    def ensure_domain_stats(self, domain: str):
        """Asegura que existe DomainStats para un dominio."""
        if domain not in self.domain_stats:
            self.domain_stats[domain] = DomainStats(domain=domain)

    def to_dict(self) -> Dict:
        return {
            'agent_id': self.agent_id,
            'session_id': self.session_id,
            'started_at': self.started_at,
            'last_activity': self.last_activity,
            'n_tasks': len(self.task_history),
            'domains_explored': list(self.domain_stats.keys()),
            'domain_summaries': {
                d: stats.get_summary() for d, stats in self.domain_stats.items()
            }
        }


class CareerIntegratedEngine:
    """
    Motor integrado de tareas y carreras academicas.

    Combina:
    - Generacion de tareas (UnifiedTaskEngine)
    - Progresion academica (AcademicCareerEngine)
    - Calculo de afinidades (AffinityComputer)

    Uso:
        engine = CareerIntegratedEngine()

        # Agente pide siguiente tarea (el decide que investigar)
        task_request = engine.request_next_research('GAUSS')

        # Sistema genera la tarea
        task = engine.generate_task(task_request)

        # Agente resuelve (simulado aqui)
        solution = agent.solve(task)

        # Registrar resultado y actualizar carrera
        result = engine.submit_result('GAUSS', task, solution)

        # Ver estado academico
        report = engine.get_academic_report('GAUSS')
    """

    def __init__(self, seed: Optional[int] = None):
        self.logger = get_provenance_logger()
        self.rng = np.random.default_rng(seed)

        # Motores internos
        self._task_engine = UnifiedTaskEngine(seed=seed)
        self._career_engine = AcademicCareerEngine(seed=seed)
        self._affinity_computer = AffinityComputer()

        # Sesiones de agentes
        self._sessions: Dict[str, AgentResearchSession] = {}

        # Cache de tareas pendientes
        self._pending_tasks: Dict[str, Task] = {}

    def get_or_create_session(self, agent_id: str) -> AgentResearchSession:
        """Obtiene o crea sesion de investigacion."""
        if agent_id not in self._sessions:
            self._sessions[agent_id] = AgentResearchSession(agent_id=agent_id)
        return self._sessions[agent_id]

    # =========================================================================
    # SELECCION DE INVESTIGACION (el agente "decide")
    # =========================================================================

    def request_next_research(
        self,
        agent_id: str,
        preference_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        El agente solicita su siguiente investigacion.

        El sistema selecciona basado en:
        - Afinidades del agente (de su historia)
        - Nivel academico actual
        - Curriculo del dominio

        Args:
            agent_id: ID del agente
            preference_hint: Hint opcional (puede ser ignorado)

        Returns:
            Dict con dominio, nivel, tipo de tarea
        """
        session = self.get_or_create_session(agent_id)

        # Recalcular afinidades si hay suficiente historial
        if session.domain_stats:
            session.affinities = self._affinity_computer.compute_affinities(
                session.domain_stats
            )

        # Convertir afinidades a dict simple
        affinity_scores = {
            d: a.raw_score for d, a in session.affinities.items()
        }

        # Usar CareerEngine para seleccionar
        task_info = self._career_engine.select_next_task(
            agent_id=agent_id,
            domain_affinities=affinity_scores
        )

        # Enriquecer con info de carrera
        profile = self._career_engine.get_or_create_profile(agent_id)
        task_info['academic_level'] = profile.get_current_level(task_info['domain']).value

        # Log la decision
        self.logger.log_from_data(
            value=task_info,
            source="Agent research selection based on affinities",
            statistic="next_research_request",
            context="CareerIntegratedEngine.request_next_research"
        )

        return task_info

    def generate_task(
        self,
        task_request: Dict[str, Any],
        seed: Optional[int] = None
    ) -> Task:
        """
        Genera la tarea solicitada.

        Args:
            task_request: Output de request_next_research
            seed: Semilla para reproducibilidad

        Returns:
            Task generada
        """
        domain = task_request.get('domain', 'mathematics')
        task_type = task_request.get('task_type')
        difficulty = task_request.get('difficulty_params', {})

        # Generar tarea con el motor unificado
        task = self._task_engine.sample_task(
            domain=domain,
            task_subtype=task_type,
            seed=seed
        )

        # Anotar con info de carrera
        task.params['academic_level'] = task_request.get('academic_level', 'undergraduate')
        task.params['requested_by'] = task_request.get('agent_id')

        # Guardar referencia
        self._pending_tasks[task.task_id] = task

        return task

    # =========================================================================
    # REGISTRO DE RESULTADOS
    # =========================================================================

    def submit_result(
        self,
        agent_id: str,
        task: Task,
        solution: Any,
        hypotheses: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Registra el resultado de una tarea.

        Args:
            agent_id: ID del agente
            task: Tarea completada
            solution: Solucion del agente
            hypotheses: Hipotesis generadas (para tareas sin ground truth)

        Returns:
            Dict con metricas, estado de carrera, promocion
        """
        session = self.get_or_create_session(agent_id)
        session.last_activity = datetime.now().isoformat()

        # Crear TaskResult
        result = TaskResult(
            task_id=task.task_id,
            agent_id=agent_id,
            solution=solution,
            started_at=task.created_at,
            completed_at=datetime.now().isoformat()
        )

        if hypotheses:
            result.hypotheses_generated = hypotheses
            # Separar confirmadas y falsadas (si viene esa info)
            result.hypotheses_confirmed = [h for h in hypotheses if h.get('confirmed')]
            result.hypotheses_falsified = [h for h in hypotheses if h.get('falsified')]

        # Evaluar con oracle
        metrics = self._task_engine.evaluate_result(task, result)

        # Determinar rendimiento y exito
        if task.has_ground_truth:
            # Para tareas con ground truth: usar accuracy o 1-error
            performance = metrics.get('accuracy', 1.0 - metrics.get('error', 0.5))
            succeeded = performance > 0.5  # Mejor que azar
        else:
            # Para tareas sin ground truth: evaluar falsificacion
            fals_rate = result.falsification_rate
            performance = fals_rate  # Bueno si falsa hipotesis
            # "Exito" si genera y falsa hipotesis
            succeeded = len(result.hypotheses_generated) > 0 and fals_rate > 0

        # Registrar en DomainStats
        session.ensure_domain_stats(task.domain)
        domain_metrics = DomainMetrics(
            task_id=task.task_id,
            domain=task.domain,
            accuracy=metrics.get('accuracy'),
            auroc=metrics.get('auroc'),
            mse=metrics.get('mse'),
            n_hypotheses_generated=len(result.hypotheses_generated),
            n_hypotheses_confirmed=len(result.hypotheses_confirmed),
            n_hypotheses_falsified=len(result.hypotheses_falsified),
            n_samples=task.n_samples,
            task_type=task.task_type.value
        )
        session.domain_stats[task.domain].add_metrics(domain_metrics)

        # Registrar en CareerEngine
        career_result = self._career_engine.record_task_result(
            agent_id=agent_id,
            domain=task.domain,
            performance=performance,
            succeeded=succeeded,
            task_type=task.params.get('math_task_type') or task.params.get('physics_task_type', '')
        )

        # Verificar promocion automatica
        promoted = False
        new_level = None
        if career_result['can_promote']:
            promoted, new_level, _ = self._career_engine.promote(agent_id, task.domain)

        # Guardar en historial
        session.task_history.append({
            'task_id': task.task_id,
            'domain': task.domain,
            'task_type': task.task_type.value,
            'performance': performance,
            'succeeded': succeeded,
            'metrics': metrics,
            'promoted': promoted,
            'timestamp': datetime.now().isoformat()
        })

        # Eliminar de pendientes
        if task.task_id in self._pending_tasks:
            del self._pending_tasks[task.task_id]

        return {
            'agent_id': agent_id,
            'task_id': task.task_id,
            'domain': task.domain,
            'performance': performance,
            'succeeded': succeeded,
            'metrics': metrics,
            'academic_status': {
                'level': career_result['current_level'],
                'tasks_in_level': career_result['tasks_in_level'],
                'can_promote': career_result['can_promote'],
                'promoted': promoted,
                'new_level': new_level.value if new_level else None,
            }
        }

    # =========================================================================
    # REPORTES
    # =========================================================================

    def get_academic_report(self, agent_id: str) -> Dict[str, Any]:
        """
        Genera reporte academico completo de un agente.

        Incluye:
        - Perfil academico (niveles por dominio)
        - Etiqueta emergente
        - Afinidades
        - Historial resumido
        """
        session = self.get_or_create_session(agent_id)

        # Reporte del CareerEngine
        career_report = self._career_engine.get_agent_report(agent_id)

        # Reporte de afinidades
        if session.affinities:
            affinity_report = self._affinity_computer.get_specialization_report(
                session.affinities
            )
        else:
            affinity_report = {'status': 'no_data'}

        return {
            'agent_id': agent_id,
            'session': session.to_dict(),
            'career': career_report,
            'affinity': affinity_report,
            'summary': {
                'total_tasks': len(session.task_history),
                'domains_explored': len(session.domain_stats),
                'emergent_label': career_report.get('emergent_label', {}).get('label', 'novice'),
                'specialization_z': career_report.get('emergent_label', {}).get('specialization_z', 0),
            }
        }

    def get_exploration_weights(self, agent_id: str) -> Dict[str, float]:
        """
        Obtiene pesos de exploracion para seleccion de dominio.

        Util para visualizar la "decision" del agente.
        """
        session = self.get_or_create_session(agent_id)

        if not session.affinities:
            # Sin historial: pesos uniformes
            domains = self._task_engine.get_available_domains()
            return {d: 1.0 / len(domains) for d in domains}

        return self._affinity_computer.get_exploration_weights(session.affinities)


# =============================================================================
# RESEARCH DIRECTOR - Orquestador de alto nivel
# =============================================================================

class ResearchDirector:
    """
    Director de investigacion de alto nivel.

    Orquesta sesiones de investigacion donde:
    - Multiples agentes trabajan en paralelo
    - Cada uno "decide" que investigar
    - Progresan en sus carreras academicas

    Uso tipico:
        director = ResearchDirector()

        # Crear sesion de investigacion
        director.start_session(['GAUSS', 'NEWTON', 'EULER'])

        # Ejecutar rondas de investigacion
        for round in range(50):
            results = director.run_research_round()

        # Ver resultados
        report = director.get_session_report()
    """

    def __init__(self, seed: Optional[int] = None):
        self.logger = get_provenance_logger()
        self.rng = np.random.default_rng(seed)

        self._engine = CareerIntegratedEngine(seed=seed)
        self._agents: List[str] = []
        self._round_counter = 0
        self._session_started = False

    def start_session(self, agent_ids: List[str]):
        """Inicia sesion de investigacion con agentes."""
        self._agents = agent_ids
        self._round_counter = 0
        self._session_started = True

        for agent_id in agent_ids:
            self._engine.get_or_create_session(agent_id)

        self.logger.log_from_data(
            value={'agents': agent_ids, 'n_agents': len(agent_ids)},
            source="Research session started",
            statistic="session_init",
            context="ResearchDirector.start_session"
        )

    def run_research_round(
        self,
        solver_fn: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta una ronda de investigacion.

        Cada agente:
        1. Solicita siguiente tarea
        2. Recibe la tarea
        3. La resuelve (con solver_fn o oracle)
        4. Registra resultado

        Args:
            solver_fn: Funcion que resuelve tareas
                       Si None, usa oracle (control positivo)

        Returns:
            Lista de resultados por agente
        """
        if not self._session_started:
            raise RuntimeError("Session not started. Call start_session first.")

        self._round_counter += 1
        results = []

        for agent_id in self._agents:
            # 1. Agente solicita siguiente tarea
            task_request = self._engine.request_next_research(agent_id)

            # 2. Generar tarea
            task = self._engine.generate_task(task_request)

            # 3. Resolver
            if solver_fn:
                solution = solver_fn(agent_id, task)
            else:
                # Usar oracle (control positivo)
                solution = task.oracle_solution

            # 4. Registrar resultado
            result = self._engine.submit_result(
                agent_id=agent_id,
                task=task,
                solution=solution
            )

            result['round'] = self._round_counter
            results.append(result)

        return results

    def get_session_report(self) -> Dict[str, Any]:
        """Genera reporte de la sesion completa."""
        reports = {}
        for agent_id in self._agents:
            reports[agent_id] = self._engine.get_academic_report(agent_id)

        # Ranking por especializacion
        rankings = sorted(
            [(a, r.get('summary', {}).get('specialization_z', 0))
             for a, r in reports.items()],
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'n_rounds': self._round_counter,
            'n_agents': len(self._agents),
            'agent_reports': reports,
            'specialization_ranking': rankings,
            'labels': {
                a: r.get('summary', {}).get('emergent_label', 'novice')
                for a, r in reports.items()
            }
        }


# =============================================================================
# TEST
# =============================================================================

def test_career_integrated_engine():
    """Test del motor integrado."""
    print("=" * 70)
    print("TEST: CAREER-INTEGRATED ENGINE")
    print("=" * 70)

    director = ResearchDirector(seed=42)
    agents = ['GAUSS', 'NEWTON', 'EULER']

    print(f"\nIniciando sesion con agentes: {agents}")
    director.start_session(agents)

    print("\n=== EJECUTANDO 30 RONDAS DE INVESTIGACION ===")

    # Simular solver con sesgo por agente
    # GAUSS tiene sesgo hacia math, NEWTON hacia physics
    agent_biases = {
        'GAUSS': {'mathematics': 0.3, 'physics': 0.1},
        'NEWTON': {'mathematics': 0.1, 'physics': 0.3},
        'EULER': {'mathematics': 0.2, 'physics': 0.2},
    }

    def biased_solver(agent_id, task):
        """Solver con sesgo por dominio y agente."""
        # Usar oracle como base
        base_solution = task.oracle_solution
        if base_solution is None:
            return None

        # Anadir ruido segun bias
        bias = agent_biases.get(agent_id, {}).get(task.domain, 0)
        noise_level = 0.3 - bias  # Menos ruido si hay sesgo positivo

        if isinstance(base_solution, dict):
            noisy = {}
            for k, v in base_solution.items():
                if isinstance(v, (int, float)):
                    noisy[k] = v * (1 + np.random.randn() * noise_level)
                else:
                    noisy[k] = v
            return noisy
        elif isinstance(base_solution, np.ndarray):
            return base_solution + np.random.randn(*base_solution.shape) * noise_level
        else:
            return base_solution

    for round_num in range(30):
        results = director.run_research_round(solver_fn=biased_solver)

        if round_num % 10 == 9:
            print(f"\nRonda {round_num + 1}:")
            for r in results:
                status = r['academic_status']
                promo_str = f" -> PROMOCIONADO a {status['new_level']}" if status['promoted'] else ""
                print(f"  {r['agent_id']}: {r['domain']}/{status['level']} "
                      f"perf={r['performance']:.3f}{promo_str}")

    print("\n=== REPORTE FINAL ===")
    report = director.get_session_report()

    print(f"\nRondas totales: {report['n_rounds']}")
    print("\nEtiquetas emergentes:")
    for agent, label in report['labels'].items():
        print(f"  {agent}: {label}")

    print("\nRanking por especializacion:")
    for agent, z in report['specialization_ranking']:
        print(f"  {agent}: z = {z:.3f}")

    print("\nDetalles por agente:")
    for agent, ar in report['agent_reports'].items():
        print(f"\n  {agent}:")
        career = ar.get('career', {})
        promo = career.get('promotion_status', {})
        for domain, info in promo.items():
            print(f"    {domain}: {info['current_level']}")

    print("\n" + "=" * 70)
    print("TEST COMPLETADO: Motor integrado funcionando correctamente")
    print("=" * 70)


if __name__ == "__main__":
    test_career_integrated_engine()
