"""
TEST: HARD RULES DEL SISTEMA DE TENSIONES
==========================================

Tests que DEBEN pasar para garantizar cumplimiento de NORMA DURA.

CUALQUIER VIOLACION DEBE ABORTAR LA EJECUCION.

PROHIBICIONES ABSOLUTAS:
1. NO seleccionar dominios por nombre de agente
2. NO usar if/else por rol o identidad
3. NO inyectar prioridades externas
4. NO saltar de agente -> dominio directamente

FLUJO UNICO VALIDO:
    estado_interno -> tension_detectada -> dominios_candidatos -> tarea_concreta
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from domains.specialization.tension_space import (
    TensionType, TensionState, InternalState,
    TensionDetector, TensionResolver, IntegrityValidator,
    TENSION_TO_DOMAINS, _FORBIDDEN_TENSION_KEYWORDS
)
from domains.specialization.tension_driven_research import (
    TensionDrivenResearchEngine, TensionDrivenDirector,
    ResearchRequest
)


class TestHardRuleViolations:
    """
    Tests que verifican que el sistema ABORTA ante violaciones.
    """

    def test_domain_without_tension_aborts(self):
        """
        HARD RULE: Dominio no puede existir sin tension previa.

        Debe ABORTAR si se intenta seleccionar dominio directamente.
        """
        validator = IntegrityValidator()

        # Intentar flujo invalido: dominio sin tension
        with pytest.raises(RuntimeError) as exc_info:
            validator.validate_flow(tension=None, domain="physics")

        assert "HARD RULE VIOLATION" in str(exc_info.value)
        assert "tension" in str(exc_info.value).lower()

    def test_tension_without_source_metrics_aborts(self):
        """
        HARD RULE: Tensiones deben derivar de metricas internas.

        Debe ABORTAR si tension no tiene source_metrics.
        """
        validator = IntegrityValidator()

        # Crear tension sin source_metrics
        fake_tension = TensionState(
            tension_type=TensionType.INCONSISTENCY,
            intensity=0.5,
            source_metrics={}  # VACIO - violacion
        )

        with pytest.raises(RuntimeError) as exc_info:
            validator.validate_flow(tension=fake_tension, domain="physics")

        assert "HARD RULE VIOLATION" in str(exc_info.value)
        assert "source_metrics" in str(exc_info.value).lower()

    def test_request_without_selection_path_aborts(self):
        """
        HARD RULE: El path de seleccion debe incluir tension.

        Debe ABORTAR si el path no muestra flujo correcto.
        """
        engine = TensionDrivenResearchEngine(seed=42)

        # Crear request con path incorrecto (sin tension)
        fake_request = ResearchRequest(
            tension=TensionState(
                tension_type=TensionType.INCONSISTENCY,
                intensity=0.5,
                source_metrics={'fake': True}
            ),
            domain="physics",
            academic_level=engine._career_engine.get_or_create_profile('test').get_current_level('physics'),
            selection_path=["agent -> domain"]  # INVALIDO: salto directo
        )

        with pytest.raises(RuntimeError) as exc_info:
            engine.validate_request(fake_request)

        assert "HARD RULE VIOLATION" in str(exc_info.value)

    def test_no_psychological_tensions_allowed(self):
        """
        HARD RULE: No pueden existir tensiones psicologicas/narrativas.

        Las tensiones deben ser estructurales.
        """
        # Verificar que keywords prohibidas no estan en ninguna tension
        for tension in TensionType:
            tension_name = tension.value.lower()
            for keyword in _FORBIDDEN_TENSION_KEYWORDS:
                assert keyword not in tension_name, (
                    f"HARD RULE VIOLATION: Tension '{tension.value}' "
                    f"contains forbidden keyword '{keyword}'"
                )

    def test_tension_detector_uses_only_metrics(self):
        """
        HARD RULE: TensionDetector.sample_tension() usa SOLO metricas.

        Nunca debe recibir dominio o identidad como input.
        """
        detector = TensionDetector()

        # Crear estado con metricas
        state = InternalState(
            accumulated_error=0.4,
            internal_inconsistency=0.6
        )

        # sample_tension NO debe aceptar parametros de dominio o agente
        # (verificar signature)
        import inspect
        sig = inspect.signature(detector.sample_tension)
        params = list(sig.parameters.keys())

        forbidden_params = ['domain', 'agent', 'agent_id', 'role', 'identity']
        for param in forbidden_params:
            assert param not in params, (
                f"HARD RULE VIOLATION: sample_tension accepts '{param}' parameter. "
                f"Only internal state metrics allowed."
            )

    def test_tension_resolver_no_agent_input(self):
        """
        HARD RULE: TensionResolver no acepta identidad de agente.
        """
        resolver = TensionResolver()

        # Verificar signature de resolve()
        import inspect
        sig = inspect.signature(resolver.resolve)
        params = list(sig.parameters.keys())

        forbidden_params = ['agent', 'agent_id', 'role', 'identity', 'name']
        for param in forbidden_params:
            assert param not in params, (
                f"HARD RULE VIOLATION: TensionResolver.resolve accepts '{param}'. "
                f"Domain selection must not depend on agent identity."
            )


class TestFlowIntegrity:
    """
    Tests que verifican el flujo correcto: estado -> tension -> dominio -> tarea
    """

    def test_complete_flow_has_all_steps(self):
        """
        Verifica que el flujo completo incluye todos los pasos requeridos.
        """
        engine = TensionDrivenResearchEngine(seed=42)

        # Generar request
        request = engine.generate_research_request("test_agent")

        # Verificar que selection_path tiene los pasos correctos
        path_str = " -> ".join(request.selection_path).lower()

        required_steps = ['state', 'tension', 'candidate', 'domain']
        for step in required_steps:
            assert step in path_str, (
                f"FLOW VIOLATION: Step '{step}' missing from selection path. "
                f"Path: {request.selection_path}"
            )

    def test_tension_precedes_domain_in_path(self):
        """
        Verifica que tension aparece ANTES que domain en el path.
        """
        engine = TensionDrivenResearchEngine(seed=42)

        for _ in range(10):  # Repetir para robustez
            request = engine.generate_research_request("test_agent")

            path_str = " -> ".join(request.selection_path).lower()

            # Encontrar posiciones
            tension_pos = path_str.find('tension')
            domain_pos = path_str.find('domain')

            assert tension_pos >= 0, "Tension not found in path"
            assert domain_pos >= 0, "Domain not found in path"
            assert tension_pos < domain_pos, (
                f"FLOW VIOLATION: Tension must precede domain. "
                f"Path: {request.selection_path}"
            )

    def test_task_includes_originating_tension(self):
        """
        Verifica que la tarea incluye la tension que la origino.
        """
        engine = TensionDrivenResearchEngine(seed=42)

        request = engine.generate_research_request("test_agent")
        task = engine.generate_task(request)

        # La tarea debe tener referencia a la tension
        assert 'originating_tension' in task.params
        assert task.params['originating_tension'] == request.tension.tension_type.value


class TestNoIdentityBasedSelection:
    """
    Tests que verifican que la identidad NO influye en la seleccion.
    """

    def test_same_state_same_distribution(self):
        """
        Agentes con el MISMO estado interno deben tener la MISMA
        distribucion de tensiones y dominios.
        """
        engine1 = TensionDrivenResearchEngine(seed=42)
        engine2 = TensionDrivenResearchEngine(seed=42)

        # Crear estados identicos para diferentes "agentes"
        state = InternalState(
            accumulated_error=0.3,
            internal_inconsistency=0.5,
            hypothesis_coverage=0.4
        )

        # Inyectar estado identico
        engine1._internal_states["AGENT_A"] = state
        engine2._internal_states["AGENT_B"] = state

        # Generar requests (con misma seed)
        np.random.seed(123)
        request_a = engine1.generate_research_request("AGENT_A")

        np.random.seed(123)
        request_b = engine2.generate_research_request("AGENT_B")

        # Deben ser iguales (misma tension y dominio)
        assert request_a.tension.tension_type == request_b.tension.tension_type
        assert request_a.domain == request_b.domain

    def test_different_names_same_result_given_same_state(self):
        """
        Nombres diferentes NO deben producir resultados diferentes
        si el estado interno es el mismo Y el estado del sistema es el mismo.

        NOTA: El dominio depende del estado del TensionResolver (historial),
        no del nombre del agente. Esto es correcto.
        """
        # Diferentes nombres, mismo estado, NUEVO engine para cada uno
        names = ["GAUSS", "NEWTON", "EULER", "AGENT_001", "TEST"]

        base_state = InternalState(
            accumulated_error=0.2,
            internal_inconsistency=0.4,
            hypothesis_coverage=0.6
        )

        results = []
        for name in names:
            # NUEVO engine para cada agente (mismo estado inicial del sistema)
            engine = TensionDrivenResearchEngine(seed=42)

            # Copiar estado
            import copy
            engine._internal_states[name] = copy.deepcopy(base_state)

            np.random.seed(456)
            request = engine.generate_research_request(name)
            results.append((request.tension.tension_type, request.domain))

        # Todos los resultados deben ser iguales
        # (misma seed, mismo estado interno, mismo estado del sistema)
        first = results[0]
        for name, result in zip(names, results):
            assert result == first, (
                f"IDENTITY VIOLATION: Agent '{name}' got different result "
                f"({result}) than expected ({first}) with same state."
            )


class TestArchetypeEmergence:
    """
    Tests que verifican que arquetipos son SOLO post-hoc.
    """

    def test_archetype_includes_disclaimer(self):
        """
        Los arquetipos deben incluir disclaimer de que son post-hoc.
        """
        engine = TensionDrivenResearchEngine(seed=42)

        # Simular algunas investigaciones
        for _ in range(5):
            request = engine.generate_research_request("test_agent")
            task = engine.generate_task(request)
            engine.complete_research(
                "test_agent", request, task, task.oracle_solution
            )

        archetype = engine.get_emergent_archetype("test_agent")

        # Debe incluir nota explicativa
        assert 'note' in archetype
        assert 'post-hoc' in archetype['note'].lower() or 'descriptive' in archetype['note'].lower()

    def test_archetype_not_used_for_selection(self):
        """
        Los arquetipos NO deben usarse para seleccion de dominio.
        """
        engine = TensionDrivenResearchEngine(seed=42)

        # Generar archetype
        archetype = engine.get_emergent_archetype("test_agent")

        # El archetype no debe ser input de generate_research_request
        import inspect
        sig = inspect.signature(engine.generate_research_request)
        params = list(sig.parameters.keys())

        forbidden = ['archetype', 'role', 'type', 'category', 'profile']
        for param in forbidden:
            assert param not in params, (
                f"HARD RULE VIOLATION: generate_research_request accepts '{param}'. "
                f"Archetypes must not influence selection."
            )


class TestTensionToDomainsMapping:
    """
    Tests del mapeo tension -> dominios.
    """

    def test_mapping_is_not_deterministic(self):
        """
        El mapeo debe ser probabilistico, no deterministico.
        """
        resolver = TensionResolver()

        tension = TensionState(
            tension_type=TensionType.INCONSISTENCY,
            intensity=0.5,
            source_metrics={'test': True}
        )

        # Muestrear multiples veces
        domains = []
        for i in range(100):
            domain = resolver.sample_domain(tension, seed=i)
            domains.append(domain)

        # Debe haber variacion (no siempre el mismo)
        unique = set(domains)
        assert len(unique) > 1, (
            "DETERMINISM VIOLATION: Domain selection appears deterministic. "
            "Must be probabilistic."
        )

    def test_all_tensions_have_candidates(self):
        """
        Todas las tensiones deben tener dominios candidatos.
        """
        for tension_type in TensionType:
            assert tension_type in TENSION_TO_DOMAINS, (
                f"Missing mapping for tension '{tension_type.value}'"
            )

            candidates = TENSION_TO_DOMAINS[tension_type]
            assert len(candidates) >= 1, (
                f"Tension '{tension_type.value}' has no candidate domains"
            )


# =============================================================================
# EJECUCION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EJECUTANDO TESTS DE HARD RULES")
    print("=" * 70)
    print("\nEstos tests DEBEN pasar para garantizar NORMA DURA")
    print()

    pytest.main([__file__, '-v', '--tb=short'])
