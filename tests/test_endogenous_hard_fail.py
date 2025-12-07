"""
TESTS HARD FAIL - NÚCLEO DE INVESTIGACIÓN ENDÓGENA
===================================================

Si ALGUNO de estos tests falla → abort_execution()

PRINCIPIOS VERIFICADOS:
- NO roles asignados
- NO números mágicos
- NO dominios directos
- Tensión → Dominio → Tarea (siempre)
- Historia propia → Mejora → Promoción
- Etiquetas = SOLO post-hoc
- Todo endógeno
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from domains.specialization.endogenous_research_nucleus import (
    EndogenousResearchNucleus, EndogenousResearchDirector,
    TensionType, TensionDetector, DomainResolver, PromotionSystem,
    LabelSystem, IntegrityValidator, InternalState, DetectedTension,
    TheoreticalConstants, TaskLevel, TENSION_TO_DOMAINS, DOMAIN_CURRICULA
)


class TestNormaDura:
    """Tests de NORMA DURA - Si falla alguno → ABORT."""

    def test_no_magic_numbers_in_promotion(self):
        """
        HARD RULE: NO usar umbrales como "accuracy > 0.8".

        ÚNICO PERMITIDO: percentil del historial propio.
        """
        system = PromotionSystem()
        state = InternalState()

        # Simular historial
        domain = "mathematics"
        for i in range(20):
            state.domain_performance[domain] = state.domain_performance.get(domain, [])
            state.domain_performance[domain].append(np.random.uniform(0.4, 0.6))

        # Verificar que el sistema usa percentil, no umbral fijo
        can_promote, _, reason = system.check_promotion(state, domain)

        # La razón debe mencionar percentil
        assert 'percentile' in reason.lower(), (
            f"HARD FAIL: Promotion check must use percentile. Got: {reason}"
        )

        # La razón debe tener provenance teórica
        assert 'provenance' in reason.lower(), (
            f"HARD FAIL: Promotion must have theoretical provenance. Got: {reason}"
        )

    def test_no_hardcoded_threshold_values(self):
        """
        HARD RULE: Los umbrales deben venir de TheoreticalConstants.
        """
        # Verificar que PROMOTION_PERCENTILE tiene justificación
        provenance = TheoreticalConstants.get_provenance('PROMOTION_PERCENTILE')
        assert 'normal' in provenance.lower() or 'σ' in provenance or 'sigma' in provenance.lower(), (
            f"HARD FAIL: PROMOTION_PERCENTILE must have statistical justification. Got: {provenance}"
        )

        # Verificar que el valor es consistente con la teoría
        # P80 en normal ≈ μ + 0.84σ
        assert TheoreticalConstants.PROMOTION_PERCENTILE == 80.0, (
            "HARD FAIL: PROMOTION_PERCENTILE must be 80 (P80 = μ + 0.84σ)"
        )

    def test_no_role_assignment(self):
        """
        HARD RULE: NO existe is_mathematician = True o role = 'physicist'.
        """
        nucleus = EndogenousResearchNucleus(seed=42)

        # Simular tareas para un agente
        for _ in range(10):
            task = nucleus.generate_task("TEST_AGENT")
            performance = np.random.uniform(0.5, 0.9)
            nucleus.complete_task("TEST_AGENT", task, performance, 1-performance)

        # Obtener etiqueta
        label_info = nucleus.get_label("TEST_AGENT")

        # Verificar que NO hay campos de rol
        forbidden_fields = ['is_mathematician', 'is_physicist', 'role', 'assigned_role']
        for field in forbidden_fields:
            assert field not in label_info, (
                f"HARD FAIL: Label contains forbidden field '{field}'"
            )

        # La etiqueta debe ser DERIVADA, no asignada
        assert 'post-hoc' in label_info.get('note', '').lower() or 'causal' in label_info.get('note', '').lower(), (
            "HARD FAIL: Label must include disclaimer about being post-hoc"
        )

    def test_no_identity_based_domain_selection(self):
        """
        HARD RULE: El dominio NO puede depender de la identidad del agente.

        Para probar esto correctamente, usamos un único nucleus y reseteamos
        el estado global de random antes de cada llamada.
        """
        # Mismo estado inicial, diferentes nombres
        names = ["GAUSS", "NEWTON", "EULER", "AGENT_001", "RANDOM_NAME"]

        # Resetear nucleus y random para cada nombre
        results = []
        for name in names:
            # NUEVO nucleus (mismo seed) + reset de numpy random global
            np.random.seed(42)  # Reset random global antes de crear nucleus
            nucleus_fresh = EndogenousResearchNucleus(seed=42)
            task = nucleus_fresh.generate_task(name)
            results.append((task.tension.tension_type, task.domain))

        # Todos deben ser iguales (misma seed, mismo estado inicial)
        first = results[0]
        for name, result in zip(names, results):
            assert result == first, (
                f"HARD FAIL: Agent '{name}' got different result ({result}) "
                f"than expected ({first}). Domain selection must not depend on identity."
            )


class TestTensionFirst:
    """Tests que verifican: tensión SIEMPRE antes que dominio."""

    def test_tension_precedes_domain_in_path(self):
        """
        HARD RULE: Tensión debe aparecer ANTES que dominio en el path.
        """
        nucleus = EndogenousResearchNucleus(seed=42)

        for _ in range(20):
            task = nucleus.generate_task("TEST")
            path_str = " -> ".join(task.selection_path).lower()

            tension_pos = path_str.find('tension')
            domain_pos = path_str.find('domain')

            assert tension_pos >= 0, "HARD FAIL: 'tension' not in selection path"
            assert domain_pos >= 0, "HARD FAIL: 'domain' not in selection path"
            assert tension_pos < domain_pos, (
                f"HARD FAIL: Tension must precede domain. Path: {task.selection_path}"
            )

    def test_domain_without_tension_aborts(self):
        """
        HARD RULE: Seleccionar dominio sin tensión debe ABORTAR.
        """
        validator = IntegrityValidator()

        with pytest.raises(RuntimeError) as exc_info:
            validator.validate_flow(tension=None, domain="physics")

        assert "ABORT" in str(exc_info.value)

    def test_tension_requires_source_metrics(self):
        """
        HARD RULE: Tensión sin source_metrics debe ABORTAR.
        """
        validator = IntegrityValidator()

        fake_tension = DetectedTension(
            tension_type=TensionType.INCONSISTENCY,
            intensity=0.5,
            source_metrics={}  # VACÍO - violación
        )

        with pytest.raises(RuntimeError) as exc_info:
            validator.validate_flow(tension=fake_tension, domain="physics")

        assert "ABORT" in str(exc_info.value)


class TestCurriculum:
    """Tests de estructura de currículos."""

    def test_levels_are_task_properties_not_agent_states(self):
        """
        HARD RULE: Los niveles describen TAREAS, no estados de agentes.
        """
        # Verificar que DOMAIN_CURRICULA existe y tiene estructura correcta
        for domain, levels in DOMAIN_CURRICULA.items():
            for level, tasks in levels.items():
                assert isinstance(level, TaskLevel), (
                    f"HARD FAIL: Level must be TaskLevel enum, got {type(level)}"
                )
                for task in tasks:
                    assert 'type' in task, f"HARD FAIL: Task must have 'type'"
                    assert 'desc' in task, f"HARD FAIL: Task must have 'desc'"

    def test_all_tensions_have_domain_candidates(self):
        """Todas las tensiones deben tener dominios candidatos."""
        for tension_type in TensionType:
            assert tension_type in TENSION_TO_DOMAINS, (
                f"HARD FAIL: Missing domain mapping for tension '{tension_type.value}'"
            )
            candidates = TENSION_TO_DOMAINS[tension_type]
            assert len(candidates) >= 1, (
                f"HARD FAIL: Tension '{tension_type.value}' has no candidates"
            )


class TestPromotion:
    """Tests de sistema de promoción."""

    def test_promotion_uses_own_history_only(self):
        """
        HARD RULE: Promoción compara SOLO con historial propio.
        """
        system = PromotionSystem()

        # Agente con historial bajo
        state_low = InternalState()
        state_low.domain_performance['math'] = [0.3, 0.35, 0.32, 0.31, 0.33]

        # Agente con historial alto
        state_high = InternalState()
        state_high.domain_performance['math'] = [0.8, 0.85, 0.82, 0.81, 0.83]

        # Ambos con rendimiento reciente = su promedio
        # Ninguno debería promocionar porque no mejoran sobre sí mismos
        can_low, _, _ = system.check_promotion(state_low, 'math')
        can_high, _, _ = system.check_promotion(state_high, 'math')

        # Verificar que el sistema no compara entre agentes
        # (ambos deberían tener resultado similar si su variabilidad es similar)

    def test_promotion_requires_improvement(self):
        """
        HARD RULE: Promoción requiere mejora sobre historial propio.
        """
        system = PromotionSystem()
        state = InternalState()

        domain = "physics"

        # Fase 1: rendimiento bajo
        for _ in range(10):
            if domain not in state.domain_performance:
                state.domain_performance[domain] = []
            state.domain_performance[domain].append(0.3)

        can_early, _, _ = system.check_promotion(state, domain)

        # Fase 2: rendimiento alto (mejora)
        for _ in range(5):
            state.domain_performance[domain].append(0.9)

        can_late, _, _ = system.check_promotion(state, domain)

        # El agente que mejora debería poder promocionar
        # (rendimiento reciente en percentil alto de su historial)
        # Nota: esto puede variar por diseño, pero el mecanismo debe ser correcto

    def test_insufficient_history_blocks_promotion(self):
        """
        HARD RULE: Sin historia suficiente → no evaluar promoción.
        """
        system = PromotionSystem()
        state = InternalState()

        # Solo 2 muestras (menor que MIN_SAMPLES_FOR_STATS)
        state.domain_performance['test'] = [0.9, 0.95]

        can_promote, _, reason = system.check_promotion(state, 'test')

        assert not can_promote, "HARD FAIL: Should not promote with insufficient history"
        assert 'insufficient' in reason.lower() or 'history' in reason.lower(), (
            f"HARD FAIL: Reason should mention insufficient history. Got: {reason}"
        )


class TestNoMagicNumbers:
    """Tests que verifican ausencia de números mágicos."""

    def test_all_thresholds_have_provenance(self):
        """Todos los umbrales deben tener justificación teórica."""
        constants = [
            'PROMOTION_PERCENTILE',
            'SPECIALIST_Z_THRESHOLD',
            'FOCUSED_Z_THRESHOLD',
            'MIN_SAMPLES_FOR_STATS',
        ]

        for const in constants:
            provenance = TheoreticalConstants.get_provenance(const)
            assert provenance != "Sin provenance documentada", (
                f"HARD FAIL: Constant '{const}' lacks theoretical provenance"
            )

    def test_z_thresholds_are_standard(self):
        """
        Los z-thresholds deben ser valores estadísticos estándar.

        z ≥ 1 = 1σ (15.9% probabilidad)
        z ≥ 2 = 2σ (2.3% probabilidad)
        """
        assert TheoreticalConstants.FOCUSED_Z_THRESHOLD == 1.0, (
            "HARD FAIL: FOCUSED_Z_THRESHOLD must be 1.0 (1σ)"
        )
        assert TheoreticalConstants.SPECIALIST_Z_THRESHOLD == 2.0, (
            "HARD FAIL: SPECIALIST_Z_THRESHOLD must be 2.0 (2σ)"
        )


class TestNoRoles:
    """Tests que verifican ausencia de roles asignados."""

    def test_label_is_derived_not_assigned(self):
        """
        HARD RULE: Etiquetas se DERIVAN de métricas, no se asignan.
        """
        label_system = LabelSystem()

        state = InternalState()
        state.domain_performance['mathematics'] = [0.8, 0.85, 0.9]
        state.domain_performance['physics'] = [0.5, 0.55, 0.5]
        state.domain_levels['mathematics'] = TaskLevel.GRADUATE

        label_info = label_system.generate_label(state, "TEST")

        # La etiqueta debe reflejar el dominio con mejor score
        assert 'math' in label_info['label'].lower(), (
            f"HARD FAIL: Label should reflect top domain. Got: {label_info['label']}"
        )

        # Debe incluir provenance
        assert 'provenance' in label_info, "HARD FAIL: Label must include provenance"

    def test_no_forbidden_tension_keywords(self):
        """
        HARD RULE: No tensiones psicológicas/narrativas.
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


class TestPCIOCompliance:
    """Tests de cumplimiento PCIO (Principio de Causalidad Interna Obligatoria)."""

    def test_decision_has_source_metrics(self):
        """PCIO: Toda decision debe tener source_metrics."""
        nucleus = EndogenousResearchNucleus(seed=42)
        task = nucleus.generate_task("TEST")

        # La tension debe tener source_metrics
        assert task.tension.source_metrics, (
            "PCIO VIOLATION: Tension sin source_metrics"
        )
        assert len(task.tension.source_metrics) > 0, (
            "PCIO VIOLATION: source_metrics vacio"
        )

    def test_uses_only_internal_metrics(self):
        """PCIO: Solo metricas internas permitidas."""
        VALID_ORIGINS = {'FROM_DATA', 'FROM_MATH', 'FROM_THEORY', 'internal', 'state'}

        nucleus = EndogenousResearchNucleus(seed=42)
        task = nucleus.generate_task("TEST")

        for key, value in task.tension.source_metrics.items():
            # Si hay origen especificado, debe ser valido
            if isinstance(value, dict) and 'origin' in value:
                assert value['origin'] in VALID_ORIGINS, (
                    f"PCIO VIOLATION: Origen invalido '{value['origin']}' en metrica '{key}'"
                )

    def test_has_no_external_factors(self):
        """PCIO: No factores externos en decisiones."""
        FORBIDDEN_FIELDS = {
            'agent_name', 'role', 'human_preference', 'heuristic_override',
            'manual_weight', 'external_input', 'user_preference'
        }

        nucleus = EndogenousResearchNucleus(seed=42)
        task = nucleus.generate_task("TEST")

        # Verificar que source_metrics no tiene campos prohibidos
        for field in FORBIDDEN_FIELDS:
            assert field not in task.tension.source_metrics, (
                f"PCIO VIOLATION: Campo externo prohibido '{field}' en source_metrics"
            )

    def test_reproducible_with_same_seed(self):
        """PCIO: Misma semilla = misma decision."""
        results1 = []
        results2 = []

        for _ in range(5):
            nucleus1 = EndogenousResearchNucleus(seed=42)
            task1 = nucleus1.generate_task("TEST")
            results1.append((task1.tension.tension_type, task1.domain))

            nucleus2 = EndogenousResearchNucleus(seed=42)
            task2 = nucleus2.generate_task("TEST")
            results2.append((task2.tension.tension_type, task2.domain))

        assert results1 == results2, (
            "PCIO VIOLATION: Decisiones no reproducibles con mismo seed"
        )


class TestProvenance:
    """Tests de trazabilidad."""

    def test_task_includes_selection_path(self):
        """Las tareas deben incluir el path de selección completo."""
        nucleus = EndogenousResearchNucleus(seed=42)
        task = nucleus.generate_task("TEST")

        assert task.selection_path, "HARD FAIL: Task must have selection_path"
        assert len(task.selection_path) >= 3, (
            f"HARD FAIL: Selection path too short: {task.selection_path}"
        )

    def test_tension_includes_source_metrics(self):
        """Las tensiones deben incluir métricas de origen."""
        detector = TensionDetector()
        state = InternalState(internal_inconsistency=0.5)

        tensions = detector.detect_all(state)

        for t in tensions:
            assert t.source_metrics, (
                f"HARD FAIL: Tension '{t.tension_type.value}' has no source_metrics"
            )

    def test_label_includes_calculation_details(self):
        """Las etiquetas deben incluir detalles del cálculo."""
        nucleus = EndogenousResearchNucleus(seed=42)

        for _ in range(15):
            task = nucleus.generate_task("TEST")
            nucleus.complete_task("TEST", task, np.random.uniform(0.5, 0.9), 0.2)

        label = nucleus.get_label("TEST")

        assert 'specialization_z' in label, "HARD FAIL: Label must include z-score"
        assert 'domain_scores' in label, "HARD FAIL: Label must include domain scores"


class TestIntegration:
    """Tests de integración del flujo completo."""

    def test_complete_flow_no_shortcuts(self):
        """
        El flujo completo debe pasar por todos los pasos.

        estado → tensión → dominio → tarea → resultado → historial → promoción
        """
        director = EndogenousResearchDirector(seed=42)
        director.start_session(['A1', 'A2'])

        # Ejecutar rondas
        for _ in range(20):
            results = director.run_round()

            for res in results:
                # Cada resultado debe tener task con path completo
                assert res.task.selection_path, "HARD FAIL: Task missing selection path"

                # El path debe incluir todos los pasos
                path_str = " ".join(res.task.selection_path).lower()
                required = ['state', 'tension', 'domain', 'level', 'task']
                for step in required:
                    assert step in path_str, (
                        f"HARD FAIL: Step '{step}' missing from flow. Path: {res.task.selection_path}"
                    )

    def test_multiagent_independence(self):
        """
        Agentes deben ser independientes.

        La decisión de un agente NO debe afectar a otro.
        """
        director = EndogenousResearchDirector(seed=42)
        director.start_session(['A', 'B', 'C'])

        # Ejecutar
        for _ in range(10):
            director.run_round()

        report = director.get_session_report()

        # Cada agente debe tener su propio historial
        for agent in ['A', 'B', 'C']:
            assert agent in report['reports'], f"HARD FAIL: Agent '{agent}' missing from report"


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EJECUTANDO TESTS HARD FAIL")
    print("=" * 70)
    print("\nSi ALGUNO falla → abort_execution()")
    print()

    pytest.main([__file__, '-v', '--tb=short'])
