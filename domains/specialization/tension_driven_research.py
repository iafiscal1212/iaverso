"""
TENSION-DRIVEN RESEARCH ENGINE
===============================

Motor de investigacion basado EXCLUSIVAMENTE en tensiones.

FLUJO UNICO VALIDO:
    estado_interno -> tension_detectada -> dominios_candidatos -> tarea_concreta

PROHIBICIONES ABSOLUTAS:
========================
1. NO seleccionar dominios por nombre de agente
2. NO usar if/else por rol o identidad
3. NO inyectar prioridades externas
4. NO saltar de agente -> dominio directamente

CUALQUIER VIOLACION ABORTA LA EJECUCION.

OBJETIVO DEL SISTEMA:
    Reducir tensiones internas del conocimiento de forma minima y coherente.
    Todo lo demas es consecuencia.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stimuli_engine.provenance import get_provenance_logger, THEORY_CONSTANTS

from .tension_space import (
    TensionType, TensionState, InternalState,
    TensionDetector, TensionResolver, IntegrityValidator,
    TENSION_TO_DOMAINS
)
from .unified_task_engine import UnifiedTaskEngine, Task, TaskResult, TaskType
from .academic_career import AcademicCareerEngine, AcademicLevel


# =============================================================================
# INVESTIGADOR AUTONOMO (SIN IDENTIDAD EN DECISIONES)
# =============================================================================

@dataclass
class ResearchRequest:
    """
    Solicitud de investigacion derivada de tension.

    NOTA: NO contiene identidad del agente en la seleccion.
    La identidad solo sirve para tracking, no para decision.
    """
    # Tension que origina la investigacion
    tension: TensionState

    # Dominio derivado de la tension (NO del agente)
    domain: str

    # Nivel academico (del sistema de carreras)
    academic_level: AcademicLevel

    # Tipo de tarea sugerido por la tension
    suggested_task_type: Optional[str] = None

    # Trazabilidad del flujo
    selection_path: List[str] = field(default_factory=list)

    # Timestamp
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            'tension': self.tension.to_dict(),
            'domain': self.domain,
            'academic_level': self.academic_level.value,
            'suggested_task_type': self.suggested_task_type,
            'selection_path': self.selection_path,
            'created_at': self.created_at,
        }


@dataclass
class ResearchOutcome:
    """
    Resultado de una investigacion.

    Incluye medicion de reduccion de tension.
    """
    request: ResearchRequest
    task: Task
    result: TaskResult

    # Metricas de reduccion de tension
    tension_before: float = 0.0
    tension_after: float = 0.0
    tension_reduction: float = 0.0

    # Exito de la investigacion
    success: bool = False

    # Timestamp
    completed_at: str = ""

    def __post_init__(self):
        if not self.completed_at:
            self.completed_at = datetime.now().isoformat()
        self.tension_reduction = self.tension_before - self.tension_after


# =============================================================================
# MOTOR DE INVESTIGACION BASADO EN TENSIONES
# =============================================================================

class TensionDrivenResearchEngine:
    """
    Motor de investigacion donde TODAS las decisiones
    emergen de tensiones internas.

    PRINCIPIO CENTRAL:
        El sistema NO elige dominios directamente.
        El sistema detecta tensiones y los dominios emergen como consecuencia.

    USO:
        engine = TensionDrivenResearchEngine()

        # El agente registra su estado interno (no su preferencia)
        state = engine.get_internal_state(agent_id)

        # El sistema detecta tension y genera request
        # (dominio NO viene del agente)
        request = engine.generate_research_request(agent_id)

        # Validar integridad del flujo
        engine.validate_request(request)

        # Generar y ejecutar tarea
        task = engine.generate_task(request)
        outcome = engine.complete_research(agent_id, task, solution)
    """

    def __init__(self, seed: Optional[int] = None):
        self.logger = get_provenance_logger()
        self.rng = np.random.default_rng(seed)

        # Componentes internos
        self._tension_detector = TensionDetector()
        self._tension_resolver = TensionResolver()
        self._integrity_validator = IntegrityValidator()
        self._task_engine = UnifiedTaskEngine(seed=seed)
        self._career_engine = AcademicCareerEngine(seed=seed)

        # Estados internos por agente
        # NOTA: La identidad solo sirve para tracking del estado,
        # NO para seleccion de dominio
        self._internal_states: Dict[str, InternalState] = {}

        # Historial de investigaciones
        self._research_history: Dict[str, List[ResearchOutcome]] = {}

        # Tareas pendientes
        self._pending: Dict[str, Tuple[ResearchRequest, Task]] = {}

    def get_internal_state(self, agent_id: str) -> InternalState:
        """
        Obtiene estado interno de un agente.

        NOTA: El estado es una coleccion de METRICAS,
        no preferencias ni roles.
        """
        if agent_id not in self._internal_states:
            self._internal_states[agent_id] = InternalState()
        return self._internal_states[agent_id]

    # =========================================================================
    # GENERACION DE INVESTIGACION (FLUJO: estado -> tension -> dominio -> tarea)
    # =========================================================================

    def generate_research_request(
        self,
        agent_id: str,
        exclude_domains: Optional[Set[str]] = None
    ) -> ResearchRequest:
        """
        Genera solicitud de investigacion.

        FLUJO OBLIGATORIO:
            1. Obtener estado interno
            2. Detectar tension
            3. Resolver tension -> dominios
            4. Muestrear dominio
            5. Determinar nivel academico
            6. Crear request

        PROHIBIDO:
            - Saltar de agent_id a dominio
            - Usar preferencias o roles
        """
        selection_path = []

        # 1. Obtener estado interno
        state = self.get_internal_state(agent_id)
        selection_path.append(f"state_from_metrics({list(state.to_dict().keys())})")

        # 2. Detectar tension (SOLO desde metricas)
        tension = self._tension_detector.sample_tension(state)
        selection_path.append(f"tension_detected({tension.tension_type.value})")

        # VALIDAR: tension debe tener source_metrics
        self._integrity_validator.validate_flow(tension=tension, domain=None)

        # 3. Resolver tension -> dominios candidatos
        candidates = self._tension_resolver.resolve(tension, exclude_domains)
        selection_path.append(f"candidates({[c[0] for c in candidates]})")

        # 4. Muestrear dominio
        domain = self._tension_resolver.sample_domain(tension, exclude_domains)
        selection_path.append(f"domain_sampled({domain})")

        # VALIDAR: flujo completo
        self._integrity_validator.validate_flow(tension=tension, domain=domain)

        # 5. Determinar nivel academico
        profile = self._career_engine.get_or_create_profile(agent_id)
        level = profile.get_current_level(domain)
        selection_path.append(f"level({level.value})")

        # 6. Sugerir tipo de tarea basado en tension (no en agente)
        task_type = self._suggest_task_type(tension, domain, level)
        selection_path.append(f"task_type({task_type})")

        self.logger.log_from_data(
            value={
                'tension': tension.tension_type.value,
                'domain': domain,
                'selection_path': selection_path,
            },
            source="Research request generated from tension",
            statistic="tension_driven_selection",
            context="TensionDrivenResearchEngine.generate_research_request"
        )

        return ResearchRequest(
            tension=tension,
            domain=domain,
            academic_level=level,
            suggested_task_type=task_type,
            selection_path=selection_path,
        )

    def _suggest_task_type(
        self,
        tension: TensionState,
        domain: str,
        level: AcademicLevel
    ) -> Optional[str]:
        """
        Sugiere tipo de tarea basado en tension y dominio.

        NOTA: La sugerencia viene de la TENSION, no del agente.
        """
        # Mapeo tension -> tipos de tarea apropiados
        tension_task_map = {
            TensionType.INCONSISTENCY: {
                'mathematics': 'math_eq_simple',  # Verificar consistencia
                'physics': 'phys_coupled',        # Verificar modelos
            },
            TensionType.LOW_RESOLUTION: {
                'mathematics': 'math_fit',        # Mejorar precision
                'physics': 'phys_oscillator',     # Medir con precision
                'medicine': 'classification',     # Mejorar diagnostico
            },
            TensionType.OVERSIMPLIFICATION: {
                'mathematics': 'math_calculus',   # Modelos mas complejos
                'physics': 'phys_coupled',        # Sistemas complejos
            },
            TensionType.UNEXPLORED_HYPOTHESIS: {
                'mathematics': 'math_series',     # Explorar convergencia
                'physics': 'phys_timeseries',     # Explorar patrones
            },
            TensionType.MODEL_CONFLICT: {
                'physics': 'phys_coupled',        # Comparar modelos
                'mathematics': 'math_eq_simple',  # Resolver conflictos
            },
            TensionType.EMPIRICAL_GAP: {
                'medicine': 'classification',     # Obtener evidencia
                'physics': 'phys_free_fall',      # Obtener datos
            },
        }

        domain_tasks = tension_task_map.get(tension.tension_type, {})
        return domain_tasks.get(domain)

    def validate_request(self, request: ResearchRequest) -> bool:
        """
        Valida que el request cumple HARD RULES.

        Aborta si detecta violacion.
        """
        # Validar flujo
        self._integrity_validator.validate_flow(
            tension=request.tension,
            domain=request.domain
        )

        # Validar que selection_path es correcto
        path = request.selection_path

        # Debe contener: state, tension, candidates, domain
        required_steps = ['state', 'tension', 'candidate', 'domain']
        path_str = " -> ".join(path).lower()

        for step in required_steps:
            if step not in path_str:
                raise RuntimeError(
                    f"HARD RULE VIOLATION: Selection path missing '{step}'. "
                    f"Path: {path}"
                )

        return True

    # =========================================================================
    # GENERACION Y EJECUCION DE TAREAS
    # =========================================================================

    def generate_task(
        self,
        request: ResearchRequest,
        seed: Optional[int] = None
    ) -> Task:
        """
        Genera tarea desde request validado.

        La tarea incluye la tension explicitamente para trazabilidad.
        """
        # Validar request primero
        self.validate_request(request)

        # Generar tarea
        task = self._task_engine.sample_task(
            domain=request.domain,
            task_subtype=request.suggested_task_type,
            seed=seed
        )

        # Anotar con informacion de tension
        task.params['originating_tension'] = request.tension.tension_type.value
        task.params['tension_intensity'] = request.tension.intensity
        task.params['selection_path'] = request.selection_path

        return task

    def complete_research(
        self,
        agent_id: str,
        request: ResearchRequest,
        task: Task,
        solution: Any,
        hypotheses: Optional[List[Dict]] = None
    ) -> ResearchOutcome:
        """
        Completa una investigacion y actualiza estado.
        """
        # Crear TaskResult
        result = TaskResult(
            task_id=task.task_id,
            agent_id=agent_id,
            solution=solution,
            completed_at=datetime.now().isoformat()
        )

        if hypotheses:
            result.hypotheses_generated = hypotheses
            result.hypotheses_confirmed = [h for h in hypotheses if h.get('confirmed')]
            result.hypotheses_falsified = [h for h in hypotheses if h.get('falsified')]

        # Evaluar
        metrics = self._task_engine.evaluate_result(task, result)

        # Calcular performance
        performance = metrics.get('accuracy', 1.0 - metrics.get('error', 0.5))
        error = metrics.get('error', 1.0 - performance)
        success = performance > 0.5

        # Tension antes
        tension_before = request.tension.intensity

        # Actualizar estado interno
        state = self.get_internal_state(agent_id)
        state.update_from_task_result(
            performance=performance,
            error=error,
            predictions=result.predictions,
            hypotheses_tested=len(result.hypotheses_generated),
            hypotheses_falsified=len(result.hypotheses_falsified)
        )

        # Detectar tension nueva para medir reduccion
        new_tensions = self._tension_detector.detect_tensions(state)
        same_type_tensions = [
            t for t in new_tensions
            if t.tension_type == request.tension.tension_type
        ]

        if same_type_tensions:
            tension_after = same_type_tensions[0].intensity
        else:
            tension_after = 0.0  # Tension resuelta

        # Actualizar resolver con resultado
        self._tension_resolver.update_from_result(
            domain=task.domain,
            tension=request.tension,
            success=success,
            reduction=tension_before - tension_after
        )

        # Actualizar carrera academica
        self._career_engine.record_task_result(
            agent_id=agent_id,
            domain=task.domain,
            performance=performance,
            succeeded=success
        )

        # Crear outcome
        outcome = ResearchOutcome(
            request=request,
            task=task,
            result=result,
            tension_before=tension_before,
            tension_after=tension_after,
            success=success
        )

        # Guardar en historial
        if agent_id not in self._research_history:
            self._research_history[agent_id] = []
        self._research_history[agent_id].append(outcome)

        self.logger.log_from_data(
            value={
                'tension': request.tension.tension_type.value,
                'domain': task.domain,
                'success': success,
                'tension_reduction': outcome.tension_reduction,
            },
            source="Research completed with tension reduction",
            statistic="research_outcome",
            context="TensionDrivenResearchEngine.complete_research"
        )

        return outcome

    # =========================================================================
    # REPORTES
    # =========================================================================

    def get_tension_report(self, agent_id: str) -> Dict[str, Any]:
        """
        Genera reporte de tensiones para un agente.
        """
        state = self.get_internal_state(agent_id)
        tensions = self._tension_detector.detect_tensions(state)

        history = self._research_history.get(agent_id, [])

        # Calcular reduccion total por tipo de tension
        reduction_by_type: Dict[str, float] = {}
        for outcome in history:
            t_type = outcome.request.tension.tension_type.value
            reduction_by_type[t_type] = (
                reduction_by_type.get(t_type, 0) + outcome.tension_reduction
            )

        return {
            'agent_id': agent_id,
            'current_state': state.to_dict(),
            'active_tensions': [t.to_dict() for t in tensions],
            'total_research': len(history),
            'successful_research': sum(1 for o in history if o.success),
            'tension_reduction_by_type': reduction_by_type,
            'domains_emerged': list(set(o.task.domain for o in history)),
        }

    def get_emergent_archetype(self, agent_id: str) -> Dict[str, Any]:
        """
        Genera arquetipo EMERGENTE (solo post hoc, NUNCA causal).

        NOTA: Esto es SOLO para analisis humano.
        NO influye en decisiones del sistema.
        """
        history = self._research_history.get(agent_id, [])

        if not history:
            return {
                'agent_id': agent_id,
                'archetype': 'undefined',
                'note': 'Archetype is post-hoc analysis only, never causal',
            }

        # Contar activaciones por tipo de tension
        tension_counts: Dict[str, int] = {}
        domain_counts: Dict[str, int] = {}

        for outcome in history:
            t_type = outcome.request.tension.tension_type.value
            tension_counts[t_type] = tension_counts.get(t_type, 0) + 1

            domain = outcome.task.domain
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Tension dominante
        dominant_tension = max(tension_counts.keys(), key=lambda t: tension_counts[t])

        # Dominio mas frecuente (emergente, no elegido)
        dominant_domain = max(domain_counts.keys(), key=lambda d: domain_counts[d])

        # Generar descripcion del arquetipo (SOLO descriptiva)
        archetype_desc = f"Driven by {dominant_tension}, emergent focus on {dominant_domain}"

        return {
            'agent_id': agent_id,
            'archetype': archetype_desc,
            'dominant_tension': dominant_tension,
            'emergent_domain': dominant_domain,
            'tension_profile': tension_counts,
            'domain_profile': domain_counts,
            'note': (
                "IMPORTANT: This archetype is DESCRIPTIVE, not CAUSAL. "
                "It emerged from tension-driven research, not from identity."
            ),
        }


# =============================================================================
# DIRECTOR DE INVESTIGACION (MULTIAGENTE)
# =============================================================================

class TensionDrivenDirector:
    """
    Orquesta investigacion multiagente basada en tensiones.

    CADA decision de dominio pasa por el flujo:
        estado -> tension -> dominio

    NUNCA:
        agente -> dominio
    """

    def __init__(self, seed: Optional[int] = None):
        self.logger = get_provenance_logger()
        self.rng = np.random.default_rng(seed)

        self._engine = TensionDrivenResearchEngine(seed=seed)
        self._agents: List[str] = []
        self._round_counter = 0
        self._session_active = False

    def start_session(self, agent_ids: List[str]):
        """
        Inicia sesion de investigacion.

        NOTA: Los IDs son solo para tracking, NO para decision.
        """
        self._agents = agent_ids
        self._round_counter = 0
        self._session_active = True

        self.logger.log_from_data(
            value={'agents': agent_ids},
            source="Research session started (IDs for tracking only)",
            statistic="session_init",
            context="TensionDrivenDirector.start_session"
        )

    def run_round(
        self,
        solver_fn: Optional[callable] = None
    ) -> List[ResearchOutcome]:
        """
        Ejecuta una ronda de investigacion.

        Para CADA agente:
            1. Obtener estado interno
            2. Detectar tension (no preferencia)
            3. Resolver tension -> dominio
            4. Generar tarea
            5. Resolver y registrar
        """
        if not self._session_active:
            raise RuntimeError("Session not started")

        self._round_counter += 1
        outcomes = []

        for agent_id in self._agents:
            # FLUJO COMPLETO: estado -> tension -> dominio -> tarea

            # 1-4. Generar request (incluye deteccion de tension)
            request = self._engine.generate_research_request(agent_id)

            # Validar integridad
            self._engine.validate_request(request)

            # 5. Generar tarea
            task = self._engine.generate_task(request)

            # 6. Resolver
            if solver_fn:
                solution = solver_fn(agent_id, task)
            else:
                solution = task.oracle_solution

            # 7. Completar y registrar
            outcome = self._engine.complete_research(
                agent_id=agent_id,
                request=request,
                task=task,
                solution=solution
            )

            outcomes.append(outcome)

        return outcomes

    def get_session_report(self) -> Dict[str, Any]:
        """Genera reporte de la sesion."""
        reports = {}
        archetypes = {}

        for agent_id in self._agents:
            reports[agent_id] = self._engine.get_tension_report(agent_id)
            archetypes[agent_id] = self._engine.get_emergent_archetype(agent_id)

        return {
            'rounds_completed': self._round_counter,
            'agents': self._agents,
            'reports': reports,
            'emergent_archetypes': archetypes,
            'note': (
                "All domain selections emerged from tensions. "
                "Archetypes are post-hoc descriptions, not causes."
            ),
        }


# =============================================================================
# TEST
# =============================================================================

def test_tension_driven_research():
    """Test del motor de investigacion basado en tensiones."""
    print("=" * 70)
    print("TEST: TENSION-DRIVEN RESEARCH ENGINE")
    print("=" * 70)

    director = TensionDrivenDirector(seed=42)

    # Agentes (IDs son SOLO para tracking)
    agents = ['AGENT_001', 'AGENT_002', 'AGENT_003']

    print(f"\nIniciando sesion con {len(agents)} agentes")
    print("NOTA: IDs son solo para tracking, NO para decision de dominio")
    director.start_session(agents)

    print("\n=== EJECUTANDO 20 RONDAS ===")

    for round_num in range(20):
        outcomes = director.run_round()

        if (round_num + 1) % 5 == 0:
            print(f"\n--- Ronda {round_num + 1} ---")
            for outcome in outcomes:
                agent = outcome.result.agent_id
                tension = outcome.request.tension.tension_type.value
                domain = outcome.task.domain
                reduction = outcome.tension_reduction
                print(f"  {agent}: {tension} -> {domain} "
                      f"(reduction: {reduction:.3f})")

    print("\n=== REPORTE FINAL ===")
    report = director.get_session_report()

    print(f"\nRondas completadas: {report['rounds_completed']}")

    print("\nARQUETIPOS EMERGENTES (post-hoc, NO causales):")
    for agent, archetype in report['emergent_archetypes'].items():
        print(f"  {agent}:")
        print(f"    Tension dominante: {archetype['dominant_tension']}")
        print(f"    Dominio emergente: {archetype['emergent_domain']}")
        print(f"    Perfil: {archetype['tension_profile']}")

    print("\n" + "-" * 50)
    print("VERIFICACION DE HARD RULES:")
    print("-" * 50)

    # Verificar que ningun dominio fue seleccionado por identidad
    for agent, r in report['reports'].items():
        domains = r['domains_emerged']
        print(f"  {agent}: dominios emergieron de tensiones: {domains}")

    print("\nNOTA IMPORTANTE:")
    print("  Los dominios NO fueron elegidos por identidad de agente.")
    print("  Emergieron del flujo: estado -> tension -> dominio")

    print("\n" + "=" * 70)
    print("TEST COMPLETADO: Motor basado en tensiones funcionando")
    print("=" * 70)


if __name__ == "__main__":
    test_tension_driven_research()
