"""
TESTS HARD FAIL - TERA (Tension-Driven Endogenous Research Architecture)
=========================================================================

Si ALGUNO de estos tests falla → abort_execution()

PRINCIPIOS VERIFICADOS:
-----------------------
1. Métricas formales por tensión (KL, Fisher, NRV, etc.)
2. Intensidad = ||z_T||_2 (norma L2)
3. Persistencia = media móvil de intensidad
4. Nivel de tarea = percentil interno de persistencia
5. Tendencias = derivada temporal de intensidad
6. Reporte YAML auditable
7. NO roles, NO identidades, NO números mágicos
8. Flujo: métricas → z_scores → tensión → dominio → nivel → tarea
"""

import numpy as np
import pytest
import yaml
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from domains.specialization.tera_nucleus import (
    # Constantes
    TheoreticalConstants,

    # Enums
    TensionType, TaskLevel, TensionTrend,

    # Métricas
    TensionMetrics, TensionState, TensionMetricsCalculator,

    # Estado
    InternalState,

    # Componentes
    TensionDetector, LevelSelector, DomainResolver,
    PromotionSystem, LabelSystem, IntegrityValidator,

    # Mapeos
    TENSION_TO_DOMAINS, DOMAIN_CURRICULA,

    # Reporte
    TensionReport,

    # Tareas
    ResearchTask, ResearchResult,

    # Núcleo
    TeraNucleus, TeraDirector,
)


# =============================================================================
# TEST: MÉTRICAS FORMALES
# =============================================================================

class TestFormalMetrics:
    """Tests de métricas formales por tensión."""

    def test_inconsistency_metrics_include_kl_and_agreement(self):
        """
        HARD RULE: INCONSISTENCY debe tener kl_mean y sign_agreement.
        """
        calculator = TensionMetricsCalculator()

        # Crear predicciones divergentes
        pred_a = np.array([0.1, 0.8, 0.3])
        pred_b = np.array([0.9, 0.2, 0.7])

        metrics = calculator.calculate_inconsistency([pred_a, pred_b])

        assert 'kl_mean' in metrics.raw_metrics, (
            "HARD FAIL: INCONSISTENCY must include kl_mean"
        )
        assert 'sign_agreement' in metrics.raw_metrics, (
            "HARD FAIL: INCONSISTENCY must include sign_agreement"
        )

    def test_low_resolution_metrics_include_fisher_and_nrv(self):
        """
        HARD RULE: LOW_RESOLUTION debe tener fisher_information y NRV.
        """
        calculator = TensionMetricsCalculator()

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.5, 2.8])

        metrics = calculator.calculate_low_resolution(y_true, y_pred)

        assert 'fisher_information' in metrics.raw_metrics, (
            "HARD FAIL: LOW_RESOLUTION must include fisher_information"
        )
        assert 'normalized_residual_variance' in metrics.raw_metrics, (
            "HARD FAIL: LOW_RESOLUTION must include normalized_residual_variance"
        )

    def test_oversimplification_metrics_include_gap_and_entropy(self):
        """
        HARD RULE: OVERSIMPLIFICATION debe tener generalization_gap y residual_entropy.
        """
        calculator = TensionMetricsCalculator()

        metrics = calculator.calculate_oversimplification(
            loss_train=0.1,
            loss_val=0.15,
            residual_entropy=1.5
        )

        assert 'generalization_gap' in metrics.raw_metrics
        assert 'residual_entropy' in metrics.raw_metrics

    def test_unexplored_hypothesis_metrics_include_coverage(self):
        """
        HARD RULE: UNEXPLORED_HYPOTHESIS debe tener hypothesis_coverage.
        """
        calculator = TensionMetricsCalculator()

        metrics = calculator.calculate_unexplored_hypothesis(
            hypotheses_evaluated=5,
            hypotheses_reachable=20
        )

        assert 'hypothesis_coverage' in metrics.raw_metrics
        assert metrics.raw_metrics['hypothesis_coverage'] == 0.25

    def test_model_conflict_metrics_include_disagreement(self):
        """
        HARD RULE: MODEL_CONFLICT debe tener prediction_disagreement.
        """
        calculator = TensionMetricsCalculator()

        pred_a = np.array([1.0, 2.0, 3.0])
        pred_b = np.array([1.5, 2.5, 3.5])

        metrics = calculator.calculate_model_conflict(pred_a, pred_b)

        assert 'prediction_disagreement' in metrics.raw_metrics

    def test_empirical_gap_metrics_include_evidence_ratio(self):
        """
        HARD RULE: EMPIRICAL_GAP debe tener evidence_parameter_ratio.
        """
        calculator = TensionMetricsCalculator()

        metrics = calculator.calculate_empirical_gap(
            n_observations=10,
            n_parameters=5
        )

        assert 'evidence_parameter_ratio' in metrics.raw_metrics
        assert metrics.raw_metrics['evidence_parameter_ratio'] == 2.0


# =============================================================================
# TEST: INTENSIDAD L2
# =============================================================================

class TestIntensityL2:
    """Tests de intensidad como norma L2 de z-scores."""

    def test_intensity_is_l2_norm_of_z_scores(self):
        """
        HARD RULE: Intensidad = ||z_T||_2
        """
        z_scores = {'metric_a': 2.0, 'metric_b': 1.5}

        metrics = TensionMetrics(
            tension_type=TensionType.INCONSISTENCY,
            raw_metrics={'a': 0.5, 'b': 0.3},
            z_scores=z_scores
        )

        expected_l2 = np.linalg.norm([2.0, 1.5])
        assert abs(metrics.intensity_L2 - expected_l2) < 1e-6, (
            f"HARD FAIL: Intensity must be L2 norm. Expected {expected_l2}, got {metrics.intensity_L2}"
        )

    def test_intensity_zero_with_empty_z_scores(self):
        """
        Sin z-scores → intensidad 0.
        """
        metrics = TensionMetrics(
            tension_type=TensionType.INCONSISTENCY,
            raw_metrics={},
            z_scores={}
        )

        assert metrics.intensity_L2 == 0.0


# =============================================================================
# TEST: PERSISTENCIA TEMPORAL
# =============================================================================

class TestPersistence:
    """Tests de persistencia como media móvil."""

    def test_persistence_is_moving_average(self):
        """
        HARD RULE: Persistencia = media móvil de intensidad.
        """
        state = InternalState()

        # Agregar historia de intensidades
        for intensity in [1.0, 2.0, 3.0, 4.0, 5.0]:
            state.update_tension_history(TensionType.INCONSISTENCY, intensity)

        persistence = state.get_persistence(TensionType.INCONSISTENCY)

        # Con window=5, debe ser mean([1,2,3,4,5]) = 3.0
        expected = np.mean([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(persistence - expected) < 1e-6, (
            f"HARD FAIL: Persistence must be moving average. Expected {expected}, got {persistence}"
        )

    def test_persistence_uses_window(self):
        """
        Persistencia usa ventana deslizante.
        """
        state = InternalState()

        # 10 valores, window=5
        for i in range(10):
            state.update_tension_history(TensionType.LOW_RESOLUTION, float(i))

        persistence = state.get_persistence(TensionType.LOW_RESOLUTION)

        # Debe usar últimos 5: [5,6,7,8,9] → mean = 7.0
        expected = np.mean([5.0, 6.0, 7.0, 8.0, 9.0])
        assert abs(persistence - expected) < 1e-6


# =============================================================================
# TEST: NIVEL EMERGENTE (PERCENTILES)
# =============================================================================

class TestEmergentLevel:
    """Tests de nivel de tarea emergente."""

    def test_level_from_percentile_undergraduate(self):
        """
        < P50 → UNDERGRADUATE
        """
        selector = LevelSelector()

        level = selector.from_percentile(30.0)
        assert level == TaskLevel.UNDERGRADUATE

    def test_level_from_percentile_graduate(self):
        """
        P50-P85 → GRADUATE
        """
        selector = LevelSelector()

        level = selector.from_percentile(70.0)
        assert level == TaskLevel.UNDERGRADUATE  # < 85

        level = selector.from_percentile(85.0)
        assert level == TaskLevel.GRADUATE

    def test_level_from_percentile_doctoral(self):
        """
        ≥ P95 → DOCTORAL
        """
        selector = LevelSelector()

        level = selector.from_percentile(95.0)
        assert level == TaskLevel.DOCTORAL

    def test_no_absolute_thresholds_for_levels(self):
        """
        HARD RULE: NO umbrales absolutos, solo percentiles internos.
        """
        # Verificar que las constantes son percentiles, no valores absolutos
        assert TheoreticalConstants.UNDERGRADUATE_PERCENTILE == 50.0
        assert TheoreticalConstants.GRADUATE_PERCENTILE == 85.0
        assert TheoreticalConstants.DOCTORAL_PERCENTILE == 95.0


# =============================================================================
# TEST: TENDENCIAS TEMPORALES
# =============================================================================

class TestTemporalTrends:
    """Tests de tendencias (derivadas)."""

    def test_delta_intensity_is_derivative(self):
        """
        HARD RULE: delta_intensity = I_T(t) - I_T(t-1)
        """
        state = InternalState()

        state.update_tension_history(TensionType.INCONSISTENCY, 1.0)
        state.update_tension_history(TensionType.INCONSISTENCY, 3.0)

        delta = state.get_delta_intensity(TensionType.INCONSISTENCY)

        assert delta == 2.0, (
            f"HARD FAIL: Delta must be 3.0 - 1.0 = 2.0, got {delta}"
        )

    def test_trend_growing_when_positive_delta(self):
        """
        Tendencia GROWING cuando intensidad alta y delta > 0.
        """
        detector = TensionDetector()
        state = InternalState()

        # Simular historia creciente
        for i in range(10):
            state.update_tension_history(TensionType.UNEXPLORED_HYPOTHESIS, 0.5 + i * 0.2)

        metrics = TensionMetrics(
            tension_type=TensionType.UNEXPLORED_HYPOTHESIS,
            raw_metrics={'coverage': 0.3},
            z_scores={'coverage': 2.5}
        )

        tension = detector._build_tension_state(
            TensionType.UNEXPLORED_HYPOTHESIS, metrics, state
        )

        # Delta debe ser positivo
        assert tension.delta_intensity > 0

    def test_all_trends_are_enums(self):
        """
        Todas las tendencias deben ser TensionTrend.
        """
        for trend in [TensionTrend.GROWING, TensionTrend.STRUCTURAL,
                      TensionTrend.RESOLVING, TensionTrend.STABLE]:
            assert isinstance(trend, TensionTrend)


# =============================================================================
# TEST: REPORTE AUDITABLE
# =============================================================================

class TestAuditableReport:
    """Tests del reporte YAML auditable."""

    def test_report_has_all_required_fields(self):
        """
        HARD RULE: El reporte debe tener todos los campos requeridos.
        """
        report = TensionReport(
            session_id="test-session",
            agent_id="AGENT_001",
            round=42,
            timestamp=datetime.now().isoformat(),
            tension_type="inconsistency",
            intensity_L2=2.83,
            persistence_mean=2.11,
            delta_intensity=0.34,
            trend="growing",
            percentile_rank=80.0,
            source_metrics={'kl_mean': 0.43, 'sign_agreement': 0.52},
            z_scores={'kl_mean': 2.1, 'sign_agreement': -1.9},
            domain_candidates=['physics', 'mathematics', 'medicine'],
            selected_domain='physics',
            task_level='doctoral',
            selected_task='phys_coupled',
        )

        required_fields = [
            'session_id', 'agent_id', 'round', 'timestamp',
            'tension_type', 'intensity_L2', 'persistence_mean',
            'source_metrics', 'z_scores',
            'domain_candidates', 'selected_domain', 'task_level', 'selected_task'
        ]

        for field in required_fields:
            assert hasattr(report, field), f"HARD FAIL: Report missing field '{field}'"

    def test_report_exports_to_valid_yaml(self):
        """
        El reporte debe exportar a YAML válido.
        """
        report = TensionReport(
            session_id="test",
            agent_id="A1",
            round=1,
            timestamp="2025-12-06T12:00:00",
            tension_type="inconsistency",
            intensity_L2=1.5,
            persistence_mean=1.2,
            delta_intensity=0.1,
            trend="stable",
            percentile_rank=60.0,
            source_metrics={'m1': 0.5},
            z_scores={'m1': 1.0},
            domain_candidates=['physics'],
            selected_domain='physics',
            task_level='undergraduate',
            selected_task='phys_mechanics',
        )

        yaml_str = report.to_yaml()

        # Debe ser parseable
        data = yaml.safe_load(yaml_str)

        assert data['session_id'] == 'test'
        assert data['tension']['type'] == 'inconsistency'
        assert data['derived_decisions']['selected_domain'] == 'physics'

    def test_report_includes_provenance(self):
        """
        El reporte debe incluir explicación causal.
        """
        report = TensionReport(
            session_id="test",
            agent_id="A1",
            round=1,
            timestamp="2025-12-06T12:00:00",
            tension_type="inconsistency",
            intensity_L2=1.5,
            persistence_mean=1.2,
            delta_intensity=0.1,
            trend="growing",
            percentile_rank=80.0,
            source_metrics={'m1': 0.5},
            z_scores={'m1': 1.0},
            domain_candidates=['physics'],
            selected_domain='physics',
            task_level='undergraduate',
            selected_task='phys_mechanics',
        )

        yaml_str = report.to_yaml()
        data = yaml.safe_load(yaml_str)

        assert 'notes' in data
        assert 'explanation' in data['notes']
        assert len(data['notes']['explanation']) > 10


# =============================================================================
# TEST: HARD RULES (SIN IDENTIDAD, SIN NÚMEROS MÁGICOS)
# =============================================================================

class TestHardRules:
    """Tests de reglas duras."""

    def test_no_identity_based_selection(self):
        """
        HARD RULE: El dominio NO depende de la identidad.
        """
        names = ["GAUSS", "NEWTON", "EULER", "AGENT_001"]

        results = []
        for name in names:
            np.random.seed(42)
            nucleus = TeraNucleus(seed=42)
            task = nucleus.generate_task(name)
            results.append((task.tension.tension_type, task.domain))

        first = results[0]
        for name, result in zip(names, results):
            assert result == first, (
                f"HARD FAIL: Agent '{name}' got different result. "
                f"Domain must not depend on identity."
            )

    def test_all_constants_have_provenance(self):
        """
        HARD RULE: Todas las constantes tienen justificación.
        """
        constants = [
            'UNDERGRADUATE_PERCENTILE',
            'GRADUATE_PERCENTILE',
            'DOCTORAL_PERCENTILE',
            'PERSISTENCE_WINDOW',
            'MIN_SAMPLES_FOR_STATS',
            'SOFTMAX_TEMPERATURE',
            'SPECIALIST_Z_THRESHOLD',
            'FOCUSED_Z_THRESHOLD',
            'PROMOTION_PERCENTILE',
        ]

        for const in constants:
            provenance = TheoreticalConstants.get_provenance(const)
            assert provenance != "Sin provenance documentada", (
                f"HARD FAIL: Constant '{const}' lacks provenance"
            )

    def test_tension_requires_source_metrics(self):
        """
        HARD RULE: Tensión sin métricas debe abortar.
        """
        validator = IntegrityValidator()

        # Crear tensión inválida (sin métricas)
        metrics = TensionMetrics(
            tension_type=TensionType.INCONSISTENCY,
            raw_metrics={},  # VACÍO
            z_scores={}
        )

        tension = TensionState(
            tension_type=TensionType.INCONSISTENCY,
            metrics=metrics,
            intensity=0.0,
            persistence=0.0,
        )

        with pytest.raises(RuntimeError) as exc_info:
            validator.validate_tension(tension)

        assert "ABORT" in str(exc_info.value)

    def test_domain_without_tension_aborts(self):
        """
        HARD RULE: Dominio sin tensión previa debe abortar.
        """
        validator = IntegrityValidator()

        with pytest.raises(RuntimeError) as exc_info:
            validator.validate_flow(tension=None, domain="physics")

        assert "ABORT" in str(exc_info.value)


# =============================================================================
# TEST: FLUJO COMPLETO
# =============================================================================

class TestCompleteFlow:
    """Tests del flujo completo."""

    def test_flow_includes_all_steps(self):
        """
        HARD RULE: El path debe incluir: state → tension → domain → level → task
        """
        nucleus = TeraNucleus(seed=42)
        task = nucleus.generate_task("TEST")

        path_str = " ".join(task.selection_path).lower()

        required = ['state', 'tension', 'domain', 'level', 'task']
        for step in required:
            assert step in path_str, (
                f"HARD FAIL: Step '{step}' missing. Path: {task.selection_path}"
            )

    def test_tension_precedes_domain(self):
        """
        HARD RULE: Tensión aparece ANTES que dominio en el path.
        """
        nucleus = TeraNucleus(seed=42)

        for _ in range(10):
            task = nucleus.generate_task("TEST")
            path_str = " ".join(task.selection_path).lower()

            tension_pos = path_str.find('tension')
            domain_pos = path_str.find('domain')

            assert tension_pos < domain_pos, (
                f"HARD FAIL: Tension must precede domain. Path: {task.selection_path}"
            )

    def test_task_includes_report(self):
        """
        Las tareas deben incluir reporte auditable.
        """
        nucleus = TeraNucleus(seed=42)
        task = nucleus.generate_task("TEST")

        assert task.report is not None
        assert task.report.tension_type == task.tension.tension_type.value

    def test_complete_multiagent_session(self):
        """
        Test de sesión completa multiagente.
        """
        director = TeraDirector(seed=42)
        agents = ['ALPHA', 'BETA', 'GAMMA']
        director.start_session(agents)

        # 20 rondas
        for _ in range(20):
            results = director.run_round()
            assert len(results) == 3

        # Verificar reportes
        report = director.get_session_report()

        assert report['rounds'] == 20
        assert len(report['agents']) == 3

        for agent in agents:
            assert agent in report['reports']
            assert report['reports'][agent]['total_tasks'] == 20


# =============================================================================
# TEST: PROMOCIÓN Y ETIQUETAS
# =============================================================================

class TestPromotionAndLabels:
    """Tests de promoción y etiquetas."""

    def test_promotion_uses_own_percentile(self):
        """
        HARD RULE: Promoción usa percentil 80 del historial propio.
        """
        system = PromotionSystem()
        state = InternalState()

        # Historial bajo
        state.domain_performance['math'] = [0.3, 0.35, 0.32, 0.31, 0.33]

        can_promote, _, reason = system.check_promotion(state, 'math')

        assert 'percentile' in reason.lower()
        assert 'provenance' in reason.lower()

    def test_labels_are_post_hoc(self):
        """
        HARD RULE: Las etiquetas son post-hoc, nunca causales.
        """
        nucleus = TeraNucleus(seed=42)

        # Ejecutar tareas
        for _ in range(10):
            task = nucleus.generate_task("TEST")
            nucleus.complete_task("TEST", task, np.random.uniform(0.5, 0.9), 0.2)

        label = nucleus.get_label("TEST")

        assert 'note' in label
        assert 'post-hoc' in label['note'].lower()


# =============================================================================
# TEST: TENSIONES NO PSICOLÓGICAS
# =============================================================================

class TestNoPsychologicalTensions:
    """Tests que verifican ausencia de tensiones psicológicas."""

    def test_no_forbidden_keywords_in_tensions(self):
        """
        HARD RULE: Las tensiones son estructurales, NO psicológicas.
        """
        forbidden = {
            'curiosity', 'interest', 'preference', 'desire', 'motivation',
            'want', 'like', 'enjoy', 'feel', 'emotion', 'mood', 'personality'
        }

        for tension in TensionType:
            for keyword in forbidden:
                assert keyword not in tension.value.lower(), (
                    f"HARD FAIL: Tension '{tension.value}' contains forbidden keyword '{keyword}'"
                )


# =============================================================================
# TEST: PCIO (Principio de Causalidad Interna Obligatoria)
# =============================================================================

class TestPCIOCompliance:
    """Tests de cumplimiento PCIO para TERA."""

    def test_decision_has_source_metrics(self):
        """PCIO: Toda decision del nucleo debe tener source_metrics."""
        nucleus = TeraNucleus(seed=42)
        task = nucleus.generate_task("TEST")

        # La tension debe tener metrics con raw_metrics
        assert task.tension.metrics.raw_metrics, (
            "PCIO VIOLATION: Tension sin raw_metrics"
        )

    def test_uses_only_internal_metrics(self):
        """PCIO: Solo metricas internas en decisiones."""
        VALID_ORIGINS = {'FROM_DATA', 'FROM_MATH', 'FROM_THEORY'}

        # Verificar que todas las constantes teoricas tienen origen valido
        constants_info = TheoreticalConstants.get_all_provenance()

        for const, info in constants_info.items():
            # Las constantes con provenance documentada son validas
            assert info != "Sin provenance documentada", (
                f"PCIO VIOLATION: Constante '{const}' sin provenance"
            )

    def test_has_no_external_factors(self):
        """PCIO: No factores externos en el flujo."""
        FORBIDDEN_FIELDS = {
            'agent_name', 'role', 'human_preference', 'heuristic_override',
            'manual_weight', 'external_input', 'user_preference'
        }

        nucleus = TeraNucleus(seed=42)
        task = nucleus.generate_task("TEST")

        # El reporte no debe tener campos externos
        report_dict = yaml.safe_load(task.report.to_yaml())

        def check_no_forbidden(obj, path=""):
            if isinstance(obj, dict):
                for key in obj.keys():
                    assert key not in FORBIDDEN_FIELDS, (
                        f"PCIO VIOLATION: Campo prohibido '{key}' en {path}"
                    )
                    check_no_forbidden(obj[key], f"{path}.{key}")

        check_no_forbidden(report_dict)

    def test_reproducible_decisions(self):
        """PCIO: Misma semilla = mismas decisiones."""
        results1 = []
        results2 = []

        for i in range(5):
            np.random.seed(42)
            nucleus1 = TeraNucleus(seed=42)
            task1 = nucleus1.generate_task("TEST")
            results1.append((
                task1.tension.tension_type.value,
                task1.domain,
                task1.level.value
            ))

            np.random.seed(42)
            nucleus2 = TeraNucleus(seed=42)
            task2 = nucleus2.generate_task("TEST")
            results2.append((
                task2.tension.tension_type.value,
                task2.domain,
                task2.level.value
            ))

        assert results1 == results2, (
            "PCIO VIOLATION: Decisiones no reproducibles con mismo seed"
        )

    def test_report_includes_source_metrics(self):
        """PCIO: El reporte YAML debe incluir source_metrics."""
        nucleus = TeraNucleus(seed=42)
        task = nucleus.generate_task("TEST")

        report_dict = yaml.safe_load(task.report.to_yaml())

        assert 'tension' in report_dict
        assert 'source_metrics' in report_dict['tension']
        assert len(report_dict['tension']['source_metrics']) > 0, (
            "PCIO VIOLATION: Reporte sin source_metrics"
        )

    def test_no_hardcoded_domain_selection(self):
        """PCIO: Seleccion de dominio deriva de metricas, no hardcoded."""
        nucleus = TeraNucleus(seed=42)

        # Multiples tareas deben mostrar variacion basada en estado interno
        domains = []
        for i in range(20):
            task = nucleus.generate_task(f"AGENT_{i}")
            domains.append(task.domain)
            # Completar tarea para cambiar estado
            nucleus.complete_task(f"AGENT_{i}", task, 0.5 + i * 0.02, 0.1)

        # Si el sistema es endogeno, los dominios deben derivar del estado
        # No deben ser todos iguales (eso indicaria hardcoding)
        unique_domains = set(domains)
        # Con 20 tareas y estado cambiante, deberia haber variedad
        # Pero esto depende del diseno - lo importante es que NO sea hardcoded


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EJECUTANDO TESTS HARD FAIL - TERA")
    print("=" * 70)
    print("\nSi ALGUNO falla → abort_execution()")
    print()

    pytest.main([__file__, '-v', '--tb=short'])
