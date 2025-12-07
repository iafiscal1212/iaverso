"""
MODO OBSERVACIÓN ENDÓGENA
=========================

SOLO registro · NO intervención · NO interpretación

PRINCIPIOS NO NEGOCIABLES:
--------------------------
❌ NO intervenir en decisiones del sistema
❌ NO ajustar pesos, métricas ni mapeos
❌ NO introducir interpretaciones humanas
❌ NO publicar ni resumir con narrativa
✅ SOLO observar, medir y registrar resultados

Si alguna de estas condiciones se viola → abort_execution()

ARQUITECTURA:
-------------
core_research_nucleus (RUN)
     ↓
observer_layer (READ-ONLY)
     ↓
structured_logs (WRITE)
"""

import yaml
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from pathlib import Path
from enum import Enum
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# VALIDADOR DE NO-INTERVENCIÓN
# =============================================================================

class ObserverIntegrityError(Exception):
    """Error de integridad del observador."""
    pass


class ObserverValidator:
    """
    Valida que el observador NO interviene.

    Si detecta intervención → abort_execution()
    """

    _FORBIDDEN_ACTIONS = {
        'modify_weights',
        'adjust_metrics',
        'change_mappings',
        'set_domain',
        'force_tension',
        'override_level',
        'inject_preference',
    }

    _FORBIDDEN_LANGUAGE = {
        'parece preferir',
        'seems to prefer',
        'is interested in',
        'le interesa',
        'quiere',
        'wants to',
        'likes',
        'enjoys',
        'feels',
        'thinks',
        'believes',
        'curiosity',
        'motivation',
    }

    def __init__(self):
        self.violations: List[str] = []

    def validate_no_intervention(self, action: str):
        """Valida que una acción no es intervención."""
        if action in self._FORBIDDEN_ACTIONS:
            self._abort(f"Intervention detected: {action}")

    def validate_no_interpretation(self, text: str):
        """Valida que un texto no contiene interpretación."""
        text_lower = text.lower()
        for phrase in self._FORBIDDEN_LANGUAGE:
            if phrase in text_lower:
                self._abort(f"Interpretation detected: '{phrase}' in text")

    def _abort(self, reason: str):
        """Aborta ejecución."""
        self.violations.append(reason)
        raise ObserverIntegrityError(f"ABORT: OBSERVER VIOLATION - {reason}")


# =============================================================================
# REGISTRO DE OBSERVACIÓN (ESTRUCTURADO)
# =============================================================================

@dataclass
class TensionObservation:
    """Observación de tensión (solo datos)."""
    tension_type: str
    intensity_L2: float
    persistence: float
    delta: float
    direction: str  # increasing, decreasing, stable


@dataclass
class DomainObservation:
    """Observación de resolución de dominio (solo datos)."""
    candidates: List[str]
    selected: str


@dataclass
class TaskObservation:
    """Observación de tarea (solo datos)."""
    level: str
    name: str
    percentile_reason: float


@dataclass
class OutcomeObservation:
    """Observación de resultado (solo datos)."""
    performance: float
    relative_performance: float
    tension_effect: str  # decreasing, stable, increasing


@dataclass
class PromotionObservation:
    """Observación de promoción (solo datos)."""
    evaluated: bool
    promoted: bool
    new_level: Optional[str] = None


@dataclass
class AgentObservation:
    """Observación completa de un agente en una ronda."""
    agent_id: str
    tension: TensionObservation
    domain_resolution: DomainObservation
    task: TaskObservation
    outcome: OutcomeObservation
    promotion: PromotionObservation


@dataclass
class RoundObservation:
    """Observación completa de una ronda."""
    round: int
    timestamp: str
    agents: Dict[str, AgentObservation]

    # Estado global (derivado, no interpretado)
    dominant_tensions: List[str] = field(default_factory=list)
    average_task_level: str = "UNDERGRADUATE"
    resolved_tensions: List[str] = field(default_factory=list)


@dataclass
class SessionObservation:
    """Observación completa de una sesión."""
    session_id: str
    start_time: str
    rounds: List[RoundObservation] = field(default_factory=list)

    # Métricas longitudinales (acumuladas, no interpretadas)
    tension_frequencies: Dict[str, int] = field(default_factory=dict)
    tension_appearances: Dict[str, List[int]] = field(default_factory=dict)
    tension_disappearances: Dict[str, List[int]] = field(default_factory=dict)
    level_evolution: List[float] = field(default_factory=list)
    promotions_per_agent: Dict[str, int] = field(default_factory=dict)

    # Tests de endogeneidad
    endogeneity_tests: Dict[str, Any] = field(default_factory=dict)

    # Eventos objetivos (sin interpretación)
    objective_events: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# CAPA OBSERVADORA (READ-ONLY)
# =============================================================================

class EndogenousObserver:
    """
    Capa de observación pasiva.

    NO devuelve output al core.
    NO influye en decisiones.
    NO altera estados.
    SOLO observa, mide y registra.
    """

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._validator = ObserverValidator()
        self._session: Optional[SessionObservation] = None
        self._previous_tensions: Dict[str, Set[str]] = {}

        # Archivo de log principal
        self._log_file = self.log_dir / "research_observation_log.yaml"

    def start_session(self, session_id: Optional[str] = None):
        """Inicia sesión de observación."""
        self._session = SessionObservation(
            session_id=session_id or datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            start_time=datetime.now().isoformat(),
        )
        self._previous_tensions = {}

    def observe_round(
        self,
        round_num: int,
        results: List[Any],  # List[ResearchResult] from TERA
    ) -> RoundObservation:
        """
        Observa una ronda completa.

        SOLO lectura de resultados.
        NO modifica nada.
        """
        if self._session is None:
            raise ObserverIntegrityError("Session not started")

        timestamp = datetime.now().isoformat()
        agents_obs = {}

        current_tensions: Dict[str, Set[str]] = {}
        levels = []

        for result in results:
            agent_id = result.task.task_id.split('_')[0] if hasattr(result.task, 'task_id') else "UNKNOWN"

            # Extraer datos del resultado (SOLO LECTURA)
            tension = result.task.tension

            # Determinar dirección de delta
            if tension.delta_intensity > 0.1:
                direction = "increasing"
            elif tension.delta_intensity < -0.1:
                direction = "decreasing"
            else:
                direction = "stable"

            # Determinar efecto en tensión
            if result.performance > 0.7:
                tension_effect = "decreasing"
            elif result.performance < 0.4:
                tension_effect = "increasing"
            else:
                tension_effect = "stable"

            # Calcular performance relativo (vs promedio propio)
            # Esto es una métrica, no una interpretación
            relative_perf = result.performance - 0.5  # baseline neutro

            # Registrar observación del agente
            agents_obs[result.task.task_id] = AgentObservation(
                agent_id=result.task.task_id,
                tension=TensionObservation(
                    tension_type=tension.tension_type.value,
                    intensity_L2=float(tension.intensity),
                    persistence=float(tension.persistence),
                    delta=float(tension.delta_intensity),
                    direction=direction,
                ),
                domain_resolution=DomainObservation(
                    candidates=list(result.task.report.domain_candidates) if result.task.report else [],
                    selected=str(result.task.domain),
                ),
                task=TaskObservation(
                    level=result.task.level.value,
                    name=result.task.task_type,
                    percentile_reason=float(tension.percentile_rank),
                ),
                outcome=OutcomeObservation(
                    performance=float(result.performance),
                    relative_performance=float(relative_perf),
                    tension_effect=tension_effect,
                ),
                promotion=PromotionObservation(
                    evaluated=True,
                    promoted=result.promoted,
                    new_level=result.new_level.value if result.new_level else None,
                ),
            )

            # Acumular para estado global
            if result.task.task_id not in current_tensions:
                current_tensions[result.task.task_id] = set()
            current_tensions[result.task.task_id].add(tension.tension_type.value)

            levels.append({"undergraduate": 1, "graduate": 2, "doctoral": 3}.get(
                result.task.level.value, 1
            ))

            # Actualizar métricas longitudinales
            t_type = tension.tension_type.value
            self._session.tension_frequencies[t_type] = (
                self._session.tension_frequencies.get(t_type, 0) + 1
            )

            if result.promoted:
                self._session.promotions_per_agent[result.task.task_id] = (
                    self._session.promotions_per_agent.get(result.task.task_id, 0) + 1
                )

        # Calcular estado global (derivado, no interpretado)
        all_tensions = set()
        for tensions in current_tensions.values():
            all_tensions.update(tensions)

        dominant = sorted(all_tensions, key=lambda t: self._session.tension_frequencies.get(t, 0), reverse=True)

        avg_level = sum(levels) / len(levels) if levels else 1
        avg_level_str = "UNDERGRADUATE" if avg_level < 1.5 else "GRADUATE" if avg_level < 2.5 else "DOCTORAL"

        # Detectar tensiones resueltas (desaparecieron)
        resolved = []
        for agent_id, prev_tensions in self._previous_tensions.items():
            curr = current_tensions.get(agent_id, set())
            for t in prev_tensions:
                if t not in curr:
                    resolved.append(t)
                    if t not in self._session.tension_disappearances:
                        self._session.tension_disappearances[t] = []
                    self._session.tension_disappearances[t].append(round_num)

        # Detectar tensiones nuevas
        for agent_id, curr_tensions in current_tensions.items():
            prev = self._previous_tensions.get(agent_id, set())
            for t in curr_tensions:
                if t not in prev:
                    if t not in self._session.tension_appearances:
                        self._session.tension_appearances[t] = []
                    self._session.tension_appearances[t].append(round_num)

        self._previous_tensions = current_tensions

        # Registrar evolución de nivel
        self._session.level_evolution.append(avg_level)

        # Crear observación de ronda
        round_obs = RoundObservation(
            round=round_num,
            timestamp=timestamp,
            agents=agents_obs,
            dominant_tensions=dominant[:3],
            average_task_level=avg_level_str,
            resolved_tensions=list(set(resolved)),
        )

        self._session.rounds.append(round_obs)

        # Detectar eventos objetivos (sin interpretación)
        self._check_objective_events(round_num, round_obs)

        return round_obs

    def _check_objective_events(self, round_num: int, round_obs: RoundObservation):
        """
        Detecta eventos objetivos.

        SOLO notifica si ocurre:
        - Una tensión desaparece persistentemente
        - Una tensión nueva aparece de forma estable
        - Dos agentes convergen estructuralmente
        """
        # Tensión desaparece persistentemente (3+ rondas sin aparecer)
        for t_type, disappearances in self._session.tension_disappearances.items():
            if len(disappearances) >= 3:
                last_three = disappearances[-3:]
                if last_three == list(range(last_three[0], last_three[0] + 3)):
                    self._session.objective_events.append({
                        'round': round_num,
                        'type': 'tension_persistent_disappearance',
                        'tension': t_type,
                        'data': {'consecutive_rounds': 3},
                    })

        # Tensión nueva estable (aparece 3+ rondas consecutivas)
        for t_type, appearances in self._session.tension_appearances.items():
            if len(appearances) >= 3:
                last_three = appearances[-3:]
                if last_three == list(range(last_three[0], last_three[0] + 3)):
                    self._session.objective_events.append({
                        'round': round_num,
                        'type': 'tension_stable_appearance',
                        'tension': t_type,
                        'data': {'consecutive_rounds': 3},
                    })

    def record_endogeneity_tests(self) -> Dict[str, Any]:
        """
        Ejecuta y registra tests de endogeneidad.

        NO corrige.
        NO parchea.
        SOLO registra.
        """
        timestamp = datetime.now().isoformat()

        # Ejecutar tests
        test_files = [
            'tests/test_tera_hard_fail.py',
            'tests/test_endogenous_hard_fail.py',
            'tests/test_tension_hard_rules.py',
        ]

        results = {
            'timestamp': timestamp,
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'failures': [],
        }

        for test_file in test_files:
            try:
                proc = subprocess.run(
                    ['python3', '-m', 'pytest', test_file, '-v', '--tb=no', '-q'],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(Path(__file__).parent.parent.parent),
                )

                # Parsear output
                output = proc.stdout
                for line in output.split('\n'):
                    if 'passed' in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if 'passed' in p and i > 0:
                                try:
                                    results['passed'] += int(parts[i-1])
                                    results['total_tests'] += int(parts[i-1])
                                except:
                                    pass
                    if 'failed' in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if 'failed' in p and i > 0:
                                try:
                                    results['failed'] += int(parts[i-1])
                                    results['total_tests'] += int(parts[i-1])
                                except:
                                    pass

                # Registrar fallos si los hay
                if proc.returncode != 0:
                    results['failures'].append({
                        'file': test_file,
                        'returncode': proc.returncode,
                    })

            except Exception as e:
                results['failures'].append({
                    'file': test_file,
                    'error': str(e),
                })

        if self._session:
            self._session.endogeneity_tests = results

            # Si hay fallos, registrar como evento objetivo
            if results['failed'] > 0:
                self._session.objective_events.append({
                    'round': len(self._session.rounds),
                    'type': 'endogeneity_test_failure',
                    'data': {
                        'failed': results['failed'],
                        'total': results['total_tests'],
                    },
                })

        return results

    def save_log(self):
        """Guarda el log estructurado."""
        if self._session is None:
            return

        # Convertir a dict para YAML
        data = self._session_to_dict()

        # Guardar
        with open(self._log_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _session_to_dict(self) -> Dict[str, Any]:
        """Convierte sesión a diccionario."""
        if self._session is None:
            return {}

        rounds_data = []
        for r in self._session.rounds:
            agents_data = {}
            for agent_id, obs in r.agents.items():
                agents_data[agent_id] = {
                    'tension': {
                        'type': obs.tension.tension_type,
                        'intensity_L2': round(obs.tension.intensity_L2, 4),
                        'persistence': round(obs.tension.persistence, 4),
                        'trend': {
                            'delta': round(obs.tension.delta, 4),
                            'direction': obs.tension.direction,
                        },
                    },
                    'domain_resolution': {
                        'candidates': obs.domain_resolution.candidates,
                        'selected': obs.domain_resolution.selected,
                    },
                    'task': {
                        'level': obs.task.level,
                        'name': obs.task.name,
                        'percentile_reason': round(obs.task.percentile_reason, 2),
                    },
                    'outcome': {
                        'performance': round(obs.outcome.performance, 4),
                        'relative_performance': round(obs.outcome.relative_performance, 4),
                        'tension_effect': obs.outcome.tension_effect,
                    },
                    'promotion': {
                        'evaluated': obs.promotion.evaluated,
                        'promoted': obs.promotion.promoted,
                        'new_level': obs.promotion.new_level,
                    },
                }

            rounds_data.append({
                'round': r.round,
                'timestamp': r.timestamp,
                'global_state': {
                    'dominant_tensions': r.dominant_tensions,
                    'average_task_level': r.average_task_level,
                    'resolved_tensions': r.resolved_tensions,
                },
                'agents': agents_data,
            })

        return {
            'session_id': self._session.session_id,
            'start_time': self._session.start_time,
            'rounds': rounds_data,
            'longitudinal_metrics': {
                'tension_frequencies': self._session.tension_frequencies,
                'tension_appearances': self._session.tension_appearances,
                'tension_disappearances': self._session.tension_disappearances,
                'level_evolution': [round(l, 2) for l in self._session.level_evolution],
                'promotions_per_agent': self._session.promotions_per_agent,
            },
            'endogeneity_tests': self._session.endogeneity_tests,
            'objective_events': self._session.objective_events,
        }

    def get_session(self) -> Optional[SessionObservation]:
        """Retorna sesión actual (solo lectura)."""
        return self._session


# =============================================================================
# RUNNER DE OBSERVACIÓN
# =============================================================================

def run_observation_session(
    n_rounds: int = 50,
    agents: List[str] = None,
    seed: int = 42,
    log_dir: str = "logs/observation",
) -> SessionObservation:
    """
    Ejecuta una sesión de observación completa.

    El núcleo se ejecuta sin cambios.
    El observador solo registra.
    """
    from domains.specialization.tera_nucleus import TeraDirector

    if agents is None:
        agents = ['AGENT_001', 'AGENT_002', 'AGENT_003']

    # Crear director (núcleo sin modificar)
    director = TeraDirector(seed=seed)
    director.start_session(agents)

    # Crear observador (capa pasiva)
    observer = EndogenousObserver(log_dir=Path(log_dir))
    observer.start_session()

    # Ejecutar rondas
    for round_num in range(1, n_rounds + 1):
        # Núcleo ejecuta (sin intervención)
        results = director.run_round()

        # Observador registra (solo lectura)
        observer.observe_round(round_num, results)

    # Registrar tests de endogeneidad
    observer.record_endogeneity_tests()

    # Guardar log
    observer.save_log()

    return observer.get_session()


# =============================================================================
# FUNCIONES DE ANÁLISIS (READ-ONLY, SIN INTERPRETACIÓN)
# =============================================================================

def _load_sessions(logs_path: str, limit: Optional[int] = None, metadata_only: bool = False) -> List[Dict[str, Any]]:
    """
    Carga sesiones desde el directorio de logs.

    SOLO lectura. NO modificación.

    Args:
        logs_path: Ruta al directorio de sesiones
        limit: Número máximo de sesiones a cargar (None = todas)
        metadata_only: Si True, solo carga metadata (más rápido)
    """
    logs_dir = Path(logs_path)
    if not logs_dir.exists():
        return []

    sessions = []
    session_dirs = sorted(logs_dir.glob("session_*"), key=lambda p: p.name, reverse=True)

    # Aplicar límite por defecto para evitar OOM
    if limit is None:
        limit = 500  # Default: últimas 500 sesiones

    session_dirs = session_dirs[:limit]

    for session_dir in session_dirs:
        metadata_file = session_dir / "session_metadata.yaml"
        log_file = session_dir / "research_observation_log.yaml"

        data = {'session_dir': str(session_dir)}

        # Siempre cargar metadata si existe (es pequeño)
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data['metadata'] = yaml.safe_load(f)
            except Exception:
                pass

        # Solo cargar log completo si no es metadata_only
        if not metadata_only and log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    full_data = yaml.safe_load(f)
                data.update(full_data)
            except Exception:
                pass

        if data.get('metadata') or data.get('rounds'):
            sessions.append(data)

    return sessions


def summarize_past_investigation(logs_path: str, limit: int = 200) -> Dict[str, Any]:
    """
    Resume la investigación pasada.

    SOLO DATOS, NO INTERPRETACIÓN.

    Args:
        logs_path: Ruta a las sesiones
        limit: Número máximo de sesiones a analizar (default 200)

    Retorna:
    - tension_frequencies: frecuencias de cada tipo de tensión
    - domains_selected: dominios seleccionados y sus conteos
    - levels_reached: niveles académicos alcanzados
    - promotions: conteo de promociones
    - last_session_snapshot: datos de la última sesión
    - performance_by_domain: rendimiento por dominio
    - endogeneity_status: estado de tests de endogeneidad
    """
    # Primero obtener conteo total (solo metadata)
    all_metadata = _load_sessions(logs_path, limit=5000, metadata_only=True)
    total_sessions_available = len(all_metadata)

    # Agregar datos de metadata (rápido)
    tests_passed_total = sum(s.get('metadata', {}).get('tests_passed', 0) for s in all_metadata)
    tests_failed_total = sum(s.get('metadata', {}).get('tests_failed', 0) for s in all_metadata)
    total_tasks_from_meta = sum(s.get('metadata', {}).get('tasks_total', 0) for s in all_metadata)
    total_rounds_from_meta = sum(s.get('metadata', {}).get('rounds', 0) for s in all_metadata)

    # Cargar muestra para análisis detallado
    sessions = _load_sessions(logs_path, limit=limit)

    if not sessions:
        return {'error': 'no_sessions_found', 'path': logs_path}

    # Agregaciones de la muestra
    tension_freq: Dict[str, int] = {}
    domain_counts: Dict[str, int] = {}
    level_counts: Dict[str, int] = {}
    promotion_count = 0
    performance_by_domain: Dict[str, List[float]] = {}

    for session in sessions:
        # Métricas longitudinales (si existen)
        if 'longitudinal_metrics' in session:
            lm = session['longitudinal_metrics']
            for t, count in lm.get('tension_frequencies', {}).items():
                tension_freq[t] = tension_freq.get(t, 0) + count

        # Rondas
        for rnd in session.get('rounds', []):
            for task_id, agent_data in rnd.get('agents', {}).items():
                # Dominio
                domain = agent_data.get('domain_resolution', {}).get('selected', 'unknown')
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

                # Nivel
                level = agent_data.get('task', {}).get('level', 'unknown')
                level_counts[level] = level_counts.get(level, 0) + 1

                # Promoción
                if agent_data.get('promotion', {}).get('promoted', False):
                    promotion_count += 1

                # Performance por dominio (sample)
                perf = agent_data.get('outcome', {}).get('performance')
                if perf is not None:
                    if domain not in performance_by_domain:
                        performance_by_domain[domain] = []
                    if len(performance_by_domain[domain]) < 1000:  # Limitar para memoria
                        performance_by_domain[domain].append(perf)

    # Calcular medias de performance
    perf_means = {}
    for domain, perfs in performance_by_domain.items():
        if perfs:
            perf_means[domain] = {
                'mean': round(sum(perfs) / len(perfs), 4),
                'n': len(perfs),
                'min': round(min(perfs), 4),
                'max': round(max(perfs), 4),
            }

    # Snapshot de última sesión
    last_snapshot = None
    if sessions:
        last = sessions[0]
        last_rounds = last.get('rounds', [])
        if last_rounds:
            last_round = last_rounds[-1]
            last_snapshot = {
                'session_id': last.get('session_id'),
                'round': last_round.get('round'),
                'dominant_tensions': last_round.get('global_state', {}).get('dominant_tensions', []),
                'average_level': last_round.get('global_state', {}).get('average_task_level'),
                'resolved_tensions': last_round.get('global_state', {}).get('resolved_tensions', []),
            }

    return {
        'total_sessions_available': total_sessions_available,
        'sessions_analyzed': len(sessions),
        'total_rounds_estimated': total_rounds_from_meta,
        'total_tasks_estimated': total_tasks_from_meta,
        'tension_frequencies': dict(sorted(tension_freq.items(), key=lambda x: x[1], reverse=True)),
        'domains_selected': dict(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)),
        'levels_reached': level_counts,
        'promotions_in_sample': promotion_count,
        'last_session_snapshot': last_snapshot,
        'performance_by_domain': perf_means,
        'endogeneity_status': {
            'tests_passed': tests_passed_total,
            'tests_failed': tests_failed_total,
        },
    }


def get_latest_snapshot(logs_path: str) -> Dict[str, Any]:
    """
    Obtiene snapshot de la última ronda registrada.

    SOLO DATOS, NO INTERPRETACIÓN.
    """
    sessions = _load_sessions(logs_path, limit=1)

    if not sessions:
        return {'error': 'no_sessions_found'}

    last = sessions[0]
    rounds = last.get('rounds', [])

    if not rounds:
        return {'error': 'no_rounds_in_session', 'session_id': last.get('session_id')}

    last_round = rounds[-1]

    tasks = []
    for task_id, agent_data in last_round.get('agents', {}).items():
        tasks.append({
            'task_id': task_id,
            'tension': agent_data.get('tension', {}).get('type'),
            'domain': agent_data.get('domain_resolution', {}).get('selected'),
            'level': agent_data.get('task', {}).get('level'),
            'task_type': agent_data.get('task', {}).get('name'),
            'performance': agent_data.get('outcome', {}).get('performance'),
        })

    return {
        'session_id': last.get('session_id'),
        'round': last_round.get('round'),
        'timestamp': last_round.get('timestamp'),
        'tasks': tasks,
        'global_state': {
            'dominant_tensions': last_round.get('global_state', {}).get('dominant_tensions', []),
            'mean_level': last_round.get('global_state', {}).get('average_task_level'),
            'resolved_tensions': last_round.get('global_state', {}).get('resolved_tensions', []),
        },
    }


def daily_brief(logs_path: str, limit: int = 100) -> Dict[str, Any]:
    """
    Parte diario de la investigación.

    SOLO DATOS de las últimas sesiones.
    NO INTERPRETACIÓN.

    Args:
        logs_path: Ruta a las sesiones
        limit: Número máximo de sesiones recientes a analizar (default 100)
    """
    # Cargar metadata de todas las sesiones disponibles
    all_meta = _load_sessions(logs_path, limit=5000, metadata_only=True)

    # Calcular estadísticas globales desde metadata
    total_sessions = len(all_meta)
    perfect_tests = sum(1 for s in all_meta if s.get('metadata', {}).get('tests_failed', 0) == 0)

    # Cargar muestra reciente para análisis detallado
    sessions = _load_sessions(logs_path, limit=limit)

    if not sessions:
        return {'error': 'no_sessions_found'}

    # Agregaciones de la muestra
    tension_counts: Dict[str, int] = {}
    task_counts: Dict[str, int] = {}
    level_sum = 0
    level_n = 0
    anomalies = []

    for session in sessions:
        for rnd in session.get('rounds', []):
            for task_id, agent_data in rnd.get('agents', {}).items():
                # Tensiones
                t = agent_data.get('tension', {}).get('type')
                if t:
                    tension_counts[t] = tension_counts.get(t, 0) + 1

                # Tasks
                task_name = agent_data.get('task', {}).get('name')
                if task_name:
                    task_counts[task_name] = task_counts.get(task_name, 0) + 1

                # Nivel
                level_str = agent_data.get('task', {}).get('level', '')
                level_val = {'undergraduate': 1, 'graduate': 2, 'doctoral': 3}.get(level_str.lower(), 0)
                if level_val:
                    level_sum += level_val
                    level_n += 1

        # Eventos objetivos (anomalías estructurales)
        for event in session.get('objective_events', []):
            if event.get('type') in ['endogeneity_test_failure', 'tension_persistent_disappearance']:
                anomalies.append({
                    'session': session.get('session_id'),
                    'event': event,
                })

    mean_level = level_sum / level_n if level_n > 0 else 0
    mean_level_str = "UNDERGRADUATE" if mean_level < 1.5 else "GRADUATE" if mean_level < 2.5 else "DOCTORAL"

    return {
        'period': f'last_{limit}_sessions',
        'total_sessions_available': total_sessions,
        'sessions_analyzed': len(sessions),
        'tension_frequencies': dict(sorted(tension_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        'task_frequencies': dict(sorted(task_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        'mean_level': mean_level_str,
        'mean_level_numeric': round(mean_level, 2),
        'perfect_test_sessions_pct': round(perfect_tests / total_sessions * 100, 1) if total_sessions > 0 else 0,
        'structural_anomalies': anomalies[:10],
    }


def full_investigation_report(logs_path: str, limit: int = 100) -> Dict[str, Any]:
    """
    Informe completo de investigación (pasado + presente).

    SOLO DATOS, NO INTERPRETACIÓN.

    Args:
        logs_path: Ruta a las sesiones
        limit: Número de sesiones para análisis detallado (default 100)
    """
    past = summarize_past_investigation(logs_path, limit=limit)
    current = get_latest_snapshot(logs_path)
    brief = daily_brief(logs_path, limit=limit)

    # Tabla longitudinal de tensiones (reutilizar sesiones ya cargadas)
    sessions = _load_sessions(logs_path, limit=limit)
    tension_timeline: Dict[str, List[int]] = {}

    for i, session in enumerate(reversed(sessions)):
        session_tensions = set()
        for rnd in session.get('rounds', []):
            for task_id, agent_data in rnd.get('agents', {}).items():
                t = agent_data.get('tension', {}).get('type')
                if t:
                    session_tensions.add(t)

        for t in session_tensions:
            if t not in tension_timeline:
                tension_timeline[t] = []
            tension_timeline[t].append(i)

    # Promociones observadas
    promotions_timeline = []
    for session in sessions:
        session_promos = 0
        for rnd in session.get('rounds', []):
            for task_id, agent_data in rnd.get('agents', {}).items():
                if agent_data.get('promotion', {}).get('promoted', False):
                    session_promos += 1
        promotions_timeline.append(session_promos)

    return {
        'past_summary': past,
        'current_snapshot': current,
        'daily_brief': brief,
        'tension_timeline': {t: {'first_seen': min(v), 'last_seen': max(v), 'occurrences': len(v)}
                            for t, v in tension_timeline.items() if v},
        'promotions_timeline': promotions_timeline[-50:],  # Últimos 50 para no saturar
        'consistency': {
            'tests_perfect_rate': brief.get('perfect_test_sessions_pct', 0),
        },
    }


def export_report_as_markdown(logs_path: str, output_path: str) -> str:
    """
    Exporta informe como Markdown para publicación.

    SOLO DATOS ESTRUCTURADOS, NO INTERPRETACIÓN.
    """
    report = full_investigation_report(logs_path)

    lines = [
        "# ENDOGENOUS INVESTIGATION REPORT",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
        f"- Total sessions available: {report['past_summary'].get('total_sessions_available', 0)}",
        f"- Sessions analyzed: {report['past_summary'].get('sessions_analyzed', 0)}",
        f"- Total rounds (estimated): {report['past_summary'].get('total_rounds_estimated', 0)}",
        f"- Total tasks (estimated): {report['past_summary'].get('total_tasks_estimated', 0)}",
        f"- Promotions in sample: {report['past_summary'].get('promotions_in_sample', 0)}",
        "",
        "## Tension Frequencies",
        "",
        "| Tension Type | Count |",
        "|-------------|-------|",
    ]

    for t, count in report['past_summary'].get('tension_frequencies', {}).items():
        lines.append(f"| {t} | {count} |")

    lines.extend([
        "",
        "## Domain Distribution",
        "",
        "| Domain | Tasks |",
        "|--------|-------|",
    ])

    for d, count in report['past_summary'].get('domains_selected', {}).items():
        lines.append(f"| {d} | {count} |")

    lines.extend([
        "",
        "## Level Distribution",
        "",
        "| Level | Count |",
        "|-------|-------|",
    ])

    for level, count in report['past_summary'].get('levels_reached', {}).items():
        lines.append(f"| {level} | {count} |")

    lines.extend([
        "",
        "## Performance by Domain",
        "",
        "| Domain | Mean | N | Min | Max |",
        "|--------|------|---|-----|-----|",
    ])

    for d, stats in report['past_summary'].get('performance_by_domain', {}).items():
        lines.append(f"| {d} | {stats['mean']} | {stats['n']} | {stats['min']} | {stats['max']} |")

    lines.extend([
        "",
        "## Endogeneity Tests",
        "",
        f"- Passed: {report['past_summary'].get('endogeneity_status', {}).get('tests_passed', 0)}",
        f"- Failed: {report['past_summary'].get('endogeneity_status', {}).get('tests_failed', 0)}",
        "",
        "## Current Snapshot",
        "",
        f"- Session: {report['current_snapshot'].get('session_id', 'N/A')}",
        f"- Round: {report['current_snapshot'].get('round', 'N/A')}",
        f"- Dominant tensions: {report['current_snapshot'].get('global_state', {}).get('dominant_tensions', [])}",
        f"- Mean level: {report['current_snapshot'].get('global_state', {}).get('mean_level', 'N/A')}",
        "",
        "---",
        "",
        "*Report generated automatically. Data only, no interpretation.*",
    ])

    content = "\n".join(lines)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(content)

    return str(output_file)


def structural_analysis(logs_path: str) -> Dict[str, Any]:
    """
    Análisis estructural objetivo (sin interpretación).

    Retorna:
    - persistent_patterns: patrones que aparecen consistentemente
    - variability_by_tension: variabilidad por tipo de tensión
    - change_rates: velocidades de cambio (slopes)
    - level_stability: estabilidad del nivel académico
    - endogenous_flow_consistency: consistencia del flujo endógeno
    """
    sessions = _load_sessions(logs_path)

    if not sessions:
        return {'error': 'no_sessions_found'}

    # Calcular patrones persistentes (tensiones que aparecen en >50% de sesiones)
    tension_presence: Dict[str, int] = {}
    total_sessions = len(sessions)

    for session in sessions:
        session_tensions = set()
        for rnd in session.get('rounds', []):
            for task_id, agent_data in rnd.get('agents', {}).items():
                t = agent_data.get('tension', {}).get('type')
                if t:
                    session_tensions.add(t)
        for t in session_tensions:
            tension_presence[t] = tension_presence.get(t, 0) + 1

    persistent = {t: round(count / total_sessions, 3)
                  for t, count in tension_presence.items()
                  if count / total_sessions > 0.5}

    # Variabilidad por tensión (stddev de performance)
    import math
    tension_perfs: Dict[str, List[float]] = {}

    for session in sessions:
        for rnd in session.get('rounds', []):
            for task_id, agent_data in rnd.get('agents', {}).items():
                t = agent_data.get('tension', {}).get('type')
                perf = agent_data.get('outcome', {}).get('performance')
                if t and perf is not None:
                    if t not in tension_perfs:
                        tension_perfs[t] = []
                    tension_perfs[t].append(perf)

    variability = {}
    for t, perfs in tension_perfs.items():
        if len(perfs) > 1:
            mean = sum(perfs) / len(perfs)
            variance = sum((p - mean) ** 2 for p in perfs) / len(perfs)
            std = math.sqrt(variance)
            variability[t] = round(std, 4)

    # Cambio de niveles (slope simple)
    level_values = []
    for session in sessions[-100:]:  # Últimas 100
        session_levels = []
        for rnd in session.get('rounds', []):
            for task_id, agent_data in rnd.get('agents', {}).items():
                level_str = agent_data.get('task', {}).get('level', '')
                level_val = {'undergraduate': 1, 'graduate': 2, 'doctoral': 3}.get(level_str.lower(), 0)
                if level_val:
                    session_levels.append(level_val)
        if session_levels:
            level_values.append(sum(session_levels) / len(session_levels))

    level_slope = 0
    if len(level_values) > 10:
        # Slope simple: (último - primero) / n
        level_slope = round((level_values[-1] - level_values[0]) / len(level_values), 4)

    # Estabilidad de nivel (varianza)
    level_stability = 0
    if level_values:
        mean_level = sum(level_values) / len(level_values)
        variance = sum((l - mean_level) ** 2 for l in level_values) / len(level_values)
        level_stability = round(1 - min(variance, 1), 4)  # 1 = muy estable, 0 = muy variable

    # Consistencia del flujo endógeno (% tests perfectos)
    perfect_sessions = 0
    tested_sessions = 0
    for session in sessions:
        if 'metadata' in session:
            tested_sessions += 1
            if session['metadata'].get('tests_failed', 0) == 0:
                perfect_sessions += 1

    flow_consistency = round(perfect_sessions / tested_sessions, 4) if tested_sessions > 0 else 0

    return {
        'persistent_patterns': persistent,
        'variability_by_tension': dict(sorted(variability.items(), key=lambda x: x[1], reverse=True)),
        'change_rates': {
            'level_slope': level_slope,
            'direction': 'increasing' if level_slope > 0.001 else 'decreasing' if level_slope < -0.001 else 'stable',
        },
        'level_stability': level_stability,
        'endogenous_flow_consistency': flow_consistency,
        'sessions_analyzed': total_sessions,
    }


# =============================================================================
# EXPORTACIONES
# =============================================================================

__all__ = [
    'ObserverIntegrityError',
    'ObserverValidator',
    'TensionObservation',
    'DomainObservation',
    'TaskObservation',
    'OutcomeObservation',
    'PromotionObservation',
    'AgentObservation',
    'RoundObservation',
    'SessionObservation',
    'EndogenousObserver',
    'run_observation_session',
    # Funciones de análisis
    'summarize_past_investigation',
    'get_latest_snapshot',
    'daily_brief',
    'full_investigation_report',
    'export_report_as_markdown',
    'structural_analysis',
]
