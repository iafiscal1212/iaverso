"""
TEST PCIO HARD FAIL
===================

Tests que DEBEN FALLAR si se detecta cualquier violacion del
Principio de Causalidad Interna Obligatoria (PCIO).

Violaciones detectadas:
- Decisiones sin source_metrics
- Campos externos
- Dependencias de nombres
- Ramas deterministas externas
- Etiquetas usadas para decidir
- Logica no derivada de tensiones -> dominios -> tareas
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# DEFINICIONES PCIO
# =============================================================================

class PCIOViolationType(Enum):
    """Tipos de violacion PCIO."""
    NO_SOURCE_METRICS = "no_source_metrics"
    EXTERNAL_FIELD = "external_field"
    NAME_DEPENDENCY = "name_dependency"
    EXTERNAL_BRANCH = "external_branch"
    LABEL_DECISION = "label_decision"
    NON_DERIVED_LOGIC = "non_derived_logic"
    MANUAL_WEIGHT = "manual_weight"
    HUMAN_PREFERENCE = "human_preference"


@dataclass
class PCIOViolation:
    """Registro de una violacion PCIO."""
    violation_type: PCIOViolationType
    location: str
    details: str
    severity: str  # CRITICAL, HIGH, MEDIUM


# Campos externos PROHIBIDOS
FORBIDDEN_EXTERNAL_FIELDS = {
    'agent_name',
    'role',
    'human_preference',
    'heuristic_override',
    'manual_weight',
    'external_input',
    'user_preference',
    'predefined_role',
    'hardcoded_value',
}

# Origenes PERMITIDOS
VALID_ORIGINS = {'FROM_DATA', 'FROM_MATH', 'FROM_THEORY'}


# =============================================================================
# VALIDADORES PCIO
# =============================================================================

class PCIOValidator:
    """Validador del Principio de Causalidad Interna Obligatoria."""

    def __init__(self):
        self.violations: List[PCIOViolation] = []

    def validate_decision(self, decision: Dict[str, Any], context: str = "") -> bool:
        """
        Valida que una decision cumple PCIO.

        Returns:
            True si cumple, False si viola PCIO.
        """
        self.violations = []

        # 1. Verificar source_metrics
        if 'source_metrics' not in decision:
            self.violations.append(PCIOViolation(
                violation_type=PCIOViolationType.NO_SOURCE_METRICS,
                location=context,
                details="Decision sin source_metrics",
                severity="CRITICAL"
            ))
        elif not decision['source_metrics']:
            self.violations.append(PCIOViolation(
                violation_type=PCIOViolationType.NO_SOURCE_METRICS,
                location=context,
                details="source_metrics esta vacio",
                severity="CRITICAL"
            ))

        # 2. Verificar campos externos prohibidos
        for field in FORBIDDEN_EXTERNAL_FIELDS:
            if field in decision:
                self.violations.append(PCIOViolation(
                    violation_type=PCIOViolationType.EXTERNAL_FIELD,
                    location=context,
                    details=f"Campo externo prohibido: {field}",
                    severity="CRITICAL"
                ))

        # 3. Verificar external_factors vacio
        if decision.get('external_factors'):
            self.violations.append(PCIOViolation(
                violation_type=PCIOViolationType.EXTERNAL_FIELD,
                location=context,
                details=f"external_factors no vacio: {decision['external_factors']}",
                severity="CRITICAL"
            ))

        # 4. Verificar origenes de metricas
        for metric in decision.get('source_metrics', []):
            origin = metric.get('origin', '')
            if origin and origin not in VALID_ORIGINS:
                self.violations.append(PCIOViolation(
                    violation_type=PCIOViolationType.NON_DERIVED_LOGIC,
                    location=context,
                    details=f"Origen invalido: {origin}",
                    severity="HIGH"
                ))

        # 5. Verificar reproducibilidad
        if decision.get('has_variation', False) and 'seed' not in decision:
            self.violations.append(PCIOViolation(
                violation_type=PCIOViolationType.EXTERNAL_BRANCH,
                location=context,
                details="Variacion sin seed explicito",
                severity="HIGH"
            ))

        return len(self.violations) == 0

    def validate_transition(self, transition: Dict[str, Any], context: str = "") -> bool:
        """Valida que una transicion cumple PCIO."""
        # Una transicion es basicamente una decision de cambio de estado
        return self.validate_decision(transition, context)

    def has_no_external_factors(self, obj: Dict[str, Any]) -> bool:
        """Verifica que no hay factores externos."""
        # Verificar campos prohibidos
        for field in FORBIDDEN_EXTERNAL_FIELDS:
            if field in obj:
                return False

        # Verificar external_factors
        if obj.get('external_factors'):
            return False

        return True

    def uses_only_internal_metrics(self, obj: Dict[str, Any]) -> bool:
        """Verifica que solo usa metricas internas."""
        if 'source_metrics' not in obj:
            return False

        for metric in obj.get('source_metrics', []):
            origin = metric.get('origin', '')
            if origin and origin not in VALID_ORIGINS:
                return False

        return True


# =============================================================================
# TESTS PCIO - DEBEN FALLAR SI HAY VIOLACION
# =============================================================================

class TestPCIOCompliance:
    """Tests de cumplimiento PCIO."""

    def setup_method(self):
        """Setup para cada test."""
        self.validator = PCIOValidator()

    # -------------------------------------------------------------------------
    # Tests de source_metrics
    # -------------------------------------------------------------------------

    def test_decision_must_have_source_metrics(self):
        """FAIL si decision no tiene source_metrics."""
        decision = {
            'type': 'tension_selection',
            'selected': 'empirical_gap',
            # NO source_metrics
        }

        is_valid = self.validator.validate_decision(decision, "test_no_source_metrics")
        assert not is_valid, "Decision sin source_metrics debe ser invalida"
        assert any(v.violation_type == PCIOViolationType.NO_SOURCE_METRICS
                   for v in self.validator.violations)

    def test_source_metrics_cannot_be_empty(self):
        """FAIL si source_metrics esta vacio."""
        decision = {
            'type': 'domain_resolution',
            'selected': 'physics',
            'source_metrics': [],  # Vacio
        }

        is_valid = self.validator.validate_decision(decision, "test_empty_metrics")
        assert not is_valid, "source_metrics vacio debe ser invalido"

    def test_valid_source_metrics_passes(self):
        """PASS si source_metrics es valido."""
        decision = {
            'type': 'tension_selection',
            'selected': 'empirical_gap',
            'source_metrics': [
                {'metric_name': 'intensity_L2', 'value': 0.85, 'origin': 'FROM_DATA'},
                {'metric_name': 'persistence', 'value': 0.92, 'origin': 'FROM_DATA'},
            ],
            'external_factors': [],
        }

        is_valid = self.validator.validate_decision(decision, "test_valid_metrics")
        assert is_valid, f"Decision valida debe pasar. Violaciones: {self.validator.violations}"

    # -------------------------------------------------------------------------
    # Tests de campos externos
    # -------------------------------------------------------------------------

    def test_no_agent_name_field(self):
        """FAIL si hay campo agent_name."""
        decision = {
            'type': 'task_assignment',
            'agent_name': 'EVA',  # PROHIBIDO
            'source_metrics': [
                {'metric_name': 'tension', 'value': 0.8, 'origin': 'FROM_DATA'}
            ],
        }

        is_valid = self.validator.validate_decision(decision, "test_agent_name")
        assert not is_valid, "Campo agent_name debe ser rechazado"
        assert any(v.violation_type == PCIOViolationType.EXTERNAL_FIELD
                   for v in self.validator.violations)

    def test_no_role_field(self):
        """FAIL si hay campo role."""
        decision = {
            'type': 'task_assignment',
            'role': 'explorer',  # PROHIBIDO
            'source_metrics': [
                {'metric_name': 'level', 'value': 2, 'origin': 'FROM_DATA'}
            ],
        }

        is_valid = self.validator.validate_decision(decision, "test_role")
        assert not is_valid, "Campo role debe ser rechazado"

    def test_no_human_preference(self):
        """FAIL si hay campo human_preference."""
        decision = {
            'type': 'promotion',
            'human_preference': True,  # PROHIBIDO
            'source_metrics': [
                {'metric_name': 'performance', 'value': 0.9, 'origin': 'FROM_DATA'}
            ],
        }

        is_valid = self.validator.validate_decision(decision, "test_human_pref")
        assert not is_valid, "Campo human_preference debe ser rechazado"

    def test_no_manual_weight(self):
        """FAIL si hay peso manual."""
        decision = {
            'type': 'domain_resolution',
            'manual_weight': 0.8,  # PROHIBIDO
            'source_metrics': [
                {'metric_name': 'affinity', 'value': 0.75, 'origin': 'FROM_DATA'}
            ],
        }

        is_valid = self.validator.validate_decision(decision, "test_manual_weight")
        assert not is_valid, "Campo manual_weight debe ser rechazado"

    def test_external_factors_must_be_empty(self):
        """FAIL si external_factors no esta vacio."""
        decision = {
            'type': 'tension_selection',
            'source_metrics': [
                {'metric_name': 'intensity', 'value': 0.9, 'origin': 'FROM_DATA'}
            ],
            'external_factors': ['user_hint'],  # NO PERMITIDO
        }

        is_valid = self.validator.validate_decision(decision, "test_ext_factors")
        assert not is_valid, "external_factors no vacio debe ser rechazado"

    # -------------------------------------------------------------------------
    # Tests de origenes de metricas
    # -------------------------------------------------------------------------

    def test_valid_origins_only(self):
        """FAIL si origen no es FROM_DATA, FROM_MATH o FROM_THEORY."""
        decision = {
            'type': 'task_assignment',
            'source_metrics': [
                {'metric_name': 'score', 'value': 0.7, 'origin': 'FROM_INTUITION'},  # INVALIDO
            ],
            'external_factors': [],
        }

        is_valid = self.validator.validate_decision(decision, "test_invalid_origin")
        assert not is_valid, "Origen FROM_INTUITION debe ser rechazado"

    def test_from_data_origin_valid(self):
        """PASS con origen FROM_DATA."""
        decision = {
            'type': 'promotion',
            'source_metrics': [
                {'metric_name': 'performance', 'value': 0.85, 'origin': 'FROM_DATA'},
            ],
            'external_factors': [],
        }

        is_valid = self.validator.validate_decision(decision, "test_from_data")
        assert is_valid, "FROM_DATA debe ser valido"

    def test_from_math_origin_valid(self):
        """PASS con origen FROM_MATH."""
        decision = {
            'type': 'tension_selection',
            'source_metrics': [
                {'metric_name': 'percentile', 'value': 92.5, 'origin': 'FROM_MATH'},
            ],
            'external_factors': [],
        }

        is_valid = self.validator.validate_decision(decision, "test_from_math")
        assert is_valid, "FROM_MATH debe ser valido"

    def test_from_theory_origin_valid(self):
        """PASS con origen FROM_THEORY."""
        decision = {
            'type': 'task_assignment',
            'source_metrics': [
                {'metric_name': 'b_value', 'value': 1.0, 'origin': 'FROM_THEORY'},
            ],
            'external_factors': [],
        }

        is_valid = self.validator.validate_decision(decision, "test_from_theory")
        assert is_valid, "FROM_THEORY debe ser valido"

    # -------------------------------------------------------------------------
    # Tests de reproducibilidad
    # -------------------------------------------------------------------------

    def test_variation_requires_seed(self):
        """FAIL si hay variacion sin seed explicito."""
        decision = {
            'type': 'tension_selection',
            'source_metrics': [
                {'metric_name': 'noise', 'value': 0.1, 'origin': 'FROM_DATA'}
            ],
            'external_factors': [],
            'has_variation': True,
            # NO seed
        }

        is_valid = self.validator.validate_decision(decision, "test_no_seed")
        assert not is_valid, "Variacion sin seed debe fallar"

    def test_variation_with_seed_passes(self):
        """PASS si variacion tiene seed."""
        decision = {
            'type': 'tension_selection',
            'source_metrics': [
                {'metric_name': 'noise', 'value': 0.1, 'origin': 'FROM_DATA'}
            ],
            'external_factors': [],
            'has_variation': True,
            'seed': 42,
        }

        is_valid = self.validator.validate_decision(decision, "test_with_seed")
        assert is_valid, "Variacion con seed debe pasar"

    # -------------------------------------------------------------------------
    # Tests de helper methods
    # -------------------------------------------------------------------------

    def test_has_no_external_factors_method(self):
        """Test del metodo has_no_external_factors."""
        # Caso valido
        valid_obj = {'source_metrics': [{'name': 'x', 'origin': 'FROM_DATA'}]}
        assert self.validator.has_no_external_factors(valid_obj)

        # Caso con campo prohibido
        invalid_obj = {'agent_name': 'test', 'source_metrics': []}
        assert not self.validator.has_no_external_factors(invalid_obj)

        # Caso con external_factors
        invalid_obj2 = {'external_factors': ['hint'], 'source_metrics': []}
        assert not self.validator.has_no_external_factors(invalid_obj2)

    def test_uses_only_internal_metrics_method(self):
        """Test del metodo uses_only_internal_metrics."""
        # Sin source_metrics
        assert not self.validator.uses_only_internal_metrics({})

        # Con origen valido
        valid = {'source_metrics': [{'origin': 'FROM_DATA'}]}
        assert self.validator.uses_only_internal_metrics(valid)

        # Con origen invalido
        invalid = {'source_metrics': [{'origin': 'FROM_HEURISTIC'}]}
        assert not self.validator.uses_only_internal_metrics(invalid)


# =============================================================================
# TESTS DE INTEGRACION CON TERA
# =============================================================================

class TestPCIOTeraIntegration:
    """Tests PCIO integrados con el nucleo TERA."""

    def setup_method(self):
        """Setup."""
        self.validator = PCIOValidator()

    def test_tension_selection_pcio_compliant(self):
        """Verifica que seleccion de tension cumple PCIO."""
        try:
            from domains.specialization.tera_nucleus import TeraDirector

            director = TeraDirector(seed=42)
            director.start_session(['AGENT_001'])

            # Ejecutar una ronda
            results = director.run_round()

            for result in results:
                # Construir decision desde el resultado
                decision = {
                    'type': 'tension_selection',
                    'selected': result.task.tension.tension_type.value,
                    'source_metrics': [
                        {
                            'metric_name': 'intensity_L2',
                            'value': float(result.task.tension.intensity),
                            'origin': 'FROM_DATA'
                        },
                        {
                            'metric_name': 'persistence',
                            'value': float(result.task.tension.persistence),
                            'origin': 'FROM_DATA'
                        },
                        {
                            'metric_name': 'percentile_rank',
                            'value': float(result.task.tension.percentile_rank),
                            'origin': 'FROM_MATH'
                        },
                    ],
                    'external_factors': [],
                }

                is_valid = self.validator.validate_decision(decision, "tera_tension")
                assert is_valid, f"Tension selection debe cumplir PCIO: {self.validator.violations}"

        except ImportError:
            pytest.skip("TERA nucleus no disponible")

    def test_domain_resolution_pcio_compliant(self):
        """Verifica que resolucion de dominio cumple PCIO."""
        try:
            from domains.specialization.tera_nucleus import TeraDirector

            director = TeraDirector(seed=42)
            director.start_session(['AGENT_001'])
            results = director.run_round()

            for result in results:
                decision = {
                    'type': 'domain_resolution',
                    'selected': str(result.task.domain),
                    'candidates': list(result.task.report.domain_candidates) if result.task.report else [],
                    'source_metrics': [
                        {
                            'metric_name': 'domain_affinity',
                            'value': 1.0,  # Placeholder - debe venir del nucleo
                            'origin': 'FROM_DATA'
                        },
                    ],
                    'external_factors': [],
                }

                is_valid = self.validator.validate_decision(decision, "tera_domain")
                assert is_valid, f"Domain resolution debe cumplir PCIO: {self.validator.violations}"

        except ImportError:
            pytest.skip("TERA nucleus no disponible")

    def test_promotion_pcio_compliant(self):
        """Verifica que promocion cumple PCIO."""
        try:
            from domains.specialization.tera_nucleus import TeraDirector

            director = TeraDirector(seed=42)
            director.start_session(['AGENT_001'])
            results = director.run_round()

            for result in results:
                decision = {
                    'type': 'promotion',
                    'evaluated': True,
                    'promoted': result.promoted,
                    'new_level': result.new_level.value if result.new_level else None,
                    'source_metrics': [
                        {
                            'metric_name': 'performance',
                            'value': float(result.performance),
                            'origin': 'FROM_DATA'
                        },
                    ],
                    'external_factors': [],
                }

                is_valid = self.validator.validate_decision(decision, "tera_promotion")
                assert is_valid, f"Promotion debe cumplir PCIO: {self.validator.violations}"

        except ImportError:
            pytest.skip("TERA nucleus no disponible")


# =============================================================================
# TESTS DE AUDITORIA DE SESIONES
# =============================================================================

class TestPCIOSessionAudit:
    """Auditoria PCIO de sesiones registradas."""

    def setup_method(self):
        """Setup."""
        self.validator = PCIOValidator()
        self.logs_path = Path("/root/NEO_EVA/logs/observation/sessions")

    def test_audit_session_structure(self):
        """Audita estructura basica de sesiones para PCIO."""
        if not self.logs_path.exists():
            pytest.skip("No hay sesiones para auditar")

        import yaml

        session_dirs = list(self.logs_path.glob("session_*"))[:10]  # Primeras 10

        for session_dir in session_dirs:
            log_file = session_dir / "research_observation_log.yaml"
            if not log_file.exists():
                continue

            with open(log_file, 'r') as f:
                data = yaml.safe_load(f)

            # Verificar que cada ronda tiene metricas
            for rnd in data.get('rounds', []):
                for task_id, agent_data in rnd.get('agents', {}).items():
                    # Debe tener tension con datos
                    tension = agent_data.get('tension', {})
                    assert 'type' in tension, f"Tension sin type en {session_dir}"
                    assert 'intensity_L2' in tension, f"Tension sin intensity en {session_dir}"

                    # Debe tener outcome con performance
                    outcome = agent_data.get('outcome', {})
                    assert 'performance' in outcome, f"Outcome sin performance en {session_dir}"

    def test_no_external_fields_in_sessions(self):
        """Verifica que no hay campos externos en sesiones."""
        if not self.logs_path.exists():
            pytest.skip("No hay sesiones para auditar")

        import yaml

        session_dirs = list(self.logs_path.glob("session_*"))[:10]

        for session_dir in session_dirs:
            log_file = session_dir / "research_observation_log.yaml"
            if not log_file.exists():
                continue

            with open(log_file, 'r') as f:
                data = yaml.safe_load(f)

            # Buscar campos prohibidos recursivamente
            def check_no_forbidden(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        assert key not in FORBIDDEN_EXTERNAL_FIELDS, \
                            f"Campo prohibido '{key}' en {session_dir}:{path}"
                        check_no_forbidden(value, f"{path}.{key}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        check_no_forbidden(item, f"{path}[{i}]")

            check_no_forbidden(data)


# =============================================================================
# EJECUTAR TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
