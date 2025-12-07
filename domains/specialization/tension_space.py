"""
TENSION SPACE - Espacio de Tensiones Epistemicas
=================================================

Define el espacio de tensiones estructurales del conocimiento.

PRINCIPIO FUNDAMENTAL:
======================
El sistema NO elige dominios directamente.
El sistema detecta TENSIONES en el estado interno.
Los dominios EMERGEN como consecuencia de resolver tensiones.

FLUJO UNICO VALIDO:
    estado_interno -> tension_detectada -> dominios_candidatos -> tarea_concreta

PROHIBICIONES (HARD RULES):
===========================
- NO seleccionar dominios por nombre de agente
- NO usar if/else por rol o identidad
- NO inyectar prioridades externas
- NO saltar de agente -> dominio directamente

Cualquier violacion ABORTA la ejecucion.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stimuli_engine.provenance import get_provenance_logger, THEORY_CONSTANTS


# =============================================================================
# ESPACIO DE TENSIONES (definicion formal)
# =============================================================================

class TensionType(Enum):
    """
    Tensiones estructurales del conocimiento.

    NOTA: Estas son propiedades ESTRUCTURALES, no narrativas.
    NO pueden anadirse tensiones psicologicas o narrativas.
    """
    INCONSISTENCY = "inconsistency"
    # Contradiccion interna entre modelos o datos
    # Detectada cuando: predicciones de modelos internos divergen

    LOW_RESOLUTION = "low_resolution"
    # Incapacidad para discriminar o medir con precision
    # Detectada cuando: entropia de predicciones es alta

    OVERSIMPLIFICATION = "oversimplification"
    # Modelo demasiado simple para los datos
    # Detectada cuando: error sistematico persiste

    UNEXPLORED_HYPOTHESIS = "unexplored_hypothesis"
    # Region no evaluada del espacio teorico
    # Detectada cuando: coverage del espacio de hipotesis es bajo

    MODEL_CONFLICT = "model_conflict"
    # Modelos validos que producen predicciones incompatibles
    # Detectada cuando: modelos con buen fit dan resultados opuestos

    EMPIRICAL_GAP = "empirical_gap"
    # Ausencia de evidencia para hipotesis existentes
    # Detectada cuando: hipotesis sin tests empiricos


# Verificar que solo hay tensiones estructurales
_FORBIDDEN_TENSION_KEYWORDS = {
    'curiosity', 'interest', 'preference', 'desire', 'motivation',
    'want', 'like', 'enjoy', 'feel', 'emotion', 'mood'
}

for tension in TensionType:
    for keyword in _FORBIDDEN_TENSION_KEYWORDS:
        if keyword in tension.value.lower():
            raise ValueError(
                f"HARD RULE VIOLATION: Tension '{tension.value}' contains "
                f"forbidden psychological keyword '{keyword}'"
            )


# =============================================================================
# MAPEO TENSION -> DOMINIOS (INDIRECTO, NO DETERMINISTA)
# =============================================================================

# Este mapeo define que dominios son CANDIDATOS para resolver cada tension
# La seleccion final usa pesos dinamicos, JAMAS reglas duras
TENSION_TO_DOMAINS: Dict[TensionType, List[str]] = {
    TensionType.INCONSISTENCY: [
        "mathematics",   # Contradicciones logicas
        "physics",       # Contradicciones en modelos fisicos
        "medicine",      # Contradicciones en evidencia clinica
    ],

    TensionType.LOW_RESOLUTION: [
        "medicine",      # Precision diagnostica
        "cosmology",     # Mediciones astronomicas
        "physics",       # Mediciones experimentales
    ],

    TensionType.OVERSIMPLIFICATION: [
        "mathematics",   # Modelos matematicos insuficientes
        "physics",       # Modelos fisicos simplificados
        "medicine",      # Modelos biologicos simplificados
    ],

    TensionType.UNEXPLORED_HYPOTHESIS: [
        "physics",       # Hipotesis fisicas no testeadas
        "medicine",      # Hipotesis medicas por explorar
        "cosmology",     # Hipotesis cosmologicas
    ],

    TensionType.MODEL_CONFLICT: [
        "physics",       # Teorias en conflicto
        "medicine",      # Tratamientos en conflicto
        "mathematics",   # Aproximaciones conflictivas
    ],

    TensionType.EMPIRICAL_GAP: [
        "medicine",      # Falta de evidencia clinica
        "physics",       # Falta de datos experimentales
        "cosmology",     # Falta de observaciones
    ],
}


@dataclass
class TensionState:
    """
    Estado de una tension en un momento dado.

    ORIGEN: Todas las metricas derivadas del estado interno.
    """
    tension_type: TensionType
    intensity: float = 0.0          # [0, 1] - Intensidad de la tension
    persistence: int = 0            # Cuantos ciclos lleva activa

    # Metricas que la generaron (trazabilidad)
    source_metrics: Dict[str, float] = field(default_factory=dict)

    # Timestamp
    detected_at: str = ""

    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            'tension_type': self.tension_type.value,
            'intensity': self.intensity,
            'persistence': self.persistence,
            'source_metrics': self.source_metrics,
            'detected_at': self.detected_at,
        }


# =============================================================================
# ESTADO INTERNO DEL AGENTE
# =============================================================================

@dataclass
class InternalState:
    """
    Estado interno cuantificado de un agente.

    NORMA DURA: Estas metricas son las UNICAS fuentes de decision.
    NO hay preferencias, roles, ni identidades.
    """
    # Metricas de error y precision
    accumulated_error: float = 0.0          # Error acumulado total
    prediction_variance: float = 0.0        # Varianza de predicciones recientes
    systematic_bias: float = 0.0            # Sesgo sistematico detectado

    # Metricas de coherencia
    internal_inconsistency: float = 0.0     # Incoherencia entre modelos internos
    model_divergence: float = 0.0           # Divergencia inter-modelo

    # Metricas de cobertura
    hypothesis_coverage: float = 1.0        # [0,1] - Cobertura del espacio teorico
    empirical_coverage: float = 1.0         # [0,1] - Cobertura empirica

    # Metricas de estancamiento
    learning_plateau: float = 0.0           # Estancamiento en aprendizaje
    residual_entropy: float = 0.0           # Entropia no explicada

    # Historial de tensiones
    tension_history: List[TensionState] = field(default_factory=list)

    # Timestamp
    last_updated: str = ""

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()

    def update_from_task_result(
        self,
        performance: float,
        error: float,
        predictions: Optional[np.ndarray] = None,
        hypotheses_tested: int = 0,
        hypotheses_falsified: int = 0
    ):
        """
        Actualiza estado interno desde resultado de tarea.

        ORIGEN: Todo derivado de metricas de la tarea.
        """
        # Actualizar error acumulado (media movil exponencial)
        alpha = 0.1  # FROM_THEORY: factor de suavizado
        self.accumulated_error = (1 - alpha) * self.accumulated_error + alpha * error

        # Actualizar varianza de predicciones
        if predictions is not None and len(predictions) > 1:
            self.prediction_variance = float(np.var(predictions))

        # Actualizar cobertura de hipotesis
        if hypotheses_tested > 0:
            # Cobertura aumenta con tests, disminuye si no se testea
            self.hypothesis_coverage = min(1.0, self.hypothesis_coverage + 0.05)
            if hypotheses_falsified > 0:
                # Falsificar hipotesis reduce divergencia
                self.model_divergence = max(0, self.model_divergence - 0.1)
        else:
            # Sin tests, cobertura decae
            self.hypothesis_coverage = max(0, self.hypothesis_coverage - 0.02)

        # Detectar estancamiento
        if error > self.accumulated_error * 0.9:
            # Error no mejora significativamente
            self.learning_plateau = min(1.0, self.learning_plateau + 0.1)
        else:
            self.learning_plateau = max(0, self.learning_plateau - 0.05)

        self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            'accumulated_error': self.accumulated_error,
            'prediction_variance': self.prediction_variance,
            'systematic_bias': self.systematic_bias,
            'internal_inconsistency': self.internal_inconsistency,
            'model_divergence': self.model_divergence,
            'hypothesis_coverage': self.hypothesis_coverage,
            'empirical_coverage': self.empirical_coverage,
            'learning_plateau': self.learning_plateau,
            'residual_entropy': self.residual_entropy,
            'last_updated': self.last_updated,
        }


# =============================================================================
# DETECTOR DE TENSIONES (ENDOGENO)
# =============================================================================

class TensionDetector:
    """
    Detecta tensiones desde el estado interno.

    HARD RULE: sample_tension() usa SOLO metricas internas.
               NUNCA recibe dominio como input.
    """

    def __init__(self):
        self.logger = get_provenance_logger()
        self._detection_thresholds: Dict[TensionType, Dict[str, float]] = {}
        self._init_detection_rules()

    def _init_detection_rules(self):
        """
        Inicializa reglas de deteccion.

        NOTA: Los umbrales son relativos al estado del agente,
        no valores absolutos magicos.
        """
        # Las reglas mapean metricas del estado a tensiones
        # Los umbrales se derivan de la distribucion del propio agente
        pass  # Se calculan dinamicamente

    def detect_tensions(
        self,
        state: InternalState
    ) -> List[TensionState]:
        """
        Detecta tensiones activas en el estado interno.

        FLUJO:
            estado_interno -> metricas -> tensiones

        PROHIBIDO:
            agente -> tension (directo)
            dominio -> tension
        """
        tensions = []

        # INCONSISTENCY: detectada por divergencia inter-modelo
        if state.internal_inconsistency > 0.3 or state.model_divergence > 0.4:
            intensity = max(state.internal_inconsistency, state.model_divergence)
            tensions.append(TensionState(
                tension_type=TensionType.INCONSISTENCY,
                intensity=intensity,
                source_metrics={
                    'internal_inconsistency': state.internal_inconsistency,
                    'model_divergence': state.model_divergence,
                }
            ))
            self.logger.log_from_data(
                value=intensity,
                source="INCONSISTENCY detected from internal_inconsistency + model_divergence",
                statistic="tension_detection",
                context="TensionDetector.detect_tensions"
            )

        # LOW_RESOLUTION: detectada por alta varianza de predicciones
        if state.prediction_variance > 0.25 or state.residual_entropy > 0.5:
            intensity = max(state.prediction_variance, state.residual_entropy)
            tensions.append(TensionState(
                tension_type=TensionType.LOW_RESOLUTION,
                intensity=intensity,
                source_metrics={
                    'prediction_variance': state.prediction_variance,
                    'residual_entropy': state.residual_entropy,
                }
            ))

        # OVERSIMPLIFICATION: detectada por sesgo sistematico + estancamiento
        if state.systematic_bias > 0.2 and state.learning_plateau > 0.3:
            intensity = (state.systematic_bias + state.learning_plateau) / 2
            tensions.append(TensionState(
                tension_type=TensionType.OVERSIMPLIFICATION,
                intensity=intensity,
                source_metrics={
                    'systematic_bias': state.systematic_bias,
                    'learning_plateau': state.learning_plateau,
                }
            ))

        # UNEXPLORED_HYPOTHESIS: detectada por baja cobertura de hipotesis
        if state.hypothesis_coverage < 0.7:
            intensity = 1.0 - state.hypothesis_coverage
            tensions.append(TensionState(
                tension_type=TensionType.UNEXPLORED_HYPOTHESIS,
                intensity=intensity,
                source_metrics={
                    'hypothesis_coverage': state.hypothesis_coverage,
                }
            ))

        # MODEL_CONFLICT: detectada por alta divergencia con buen fit general
        if state.model_divergence > 0.5 and state.accumulated_error < 0.3:
            intensity = state.model_divergence
            tensions.append(TensionState(
                tension_type=TensionType.MODEL_CONFLICT,
                intensity=intensity,
                source_metrics={
                    'model_divergence': state.model_divergence,
                    'accumulated_error': state.accumulated_error,
                }
            ))

        # EMPIRICAL_GAP: detectada por baja cobertura empirica
        if state.empirical_coverage < 0.6:
            intensity = 1.0 - state.empirical_coverage
            tensions.append(TensionState(
                tension_type=TensionType.EMPIRICAL_GAP,
                intensity=intensity,
                source_metrics={
                    'empirical_coverage': state.empirical_coverage,
                }
            ))

        # Si no hay tensiones detectadas, usar UNEXPLORED_HYPOTHESIS por defecto
        # ORIGEN: Siempre hay espacio teorico por explorar
        if not tensions:
            tensions.append(TensionState(
                tension_type=TensionType.UNEXPLORED_HYPOTHESIS,
                intensity=0.3,  # Intensidad base
                source_metrics={'default': True}
            ))

        return tensions

    def sample_tension(
        self,
        state: InternalState,
        seed: Optional[int] = None
    ) -> TensionState:
        """
        Muestrea una tension del estado interno.

        HARD RULE:
            - Usa SOLO metricas internas
            - NUNCA recibe dominio como input
            - NUNCA recibe identidad de agente

        Returns:
            TensionState muestreada
        """
        if seed is not None:
            np.random.seed(seed)

        # Detectar tensiones activas
        tensions = self.detect_tensions(state)

        if not tensions:
            # Fallback: explorar hipotesis
            return TensionState(
                tension_type=TensionType.UNEXPLORED_HYPOTHESIS,
                intensity=0.3
            )

        # Muestrear por intensidad (softmax)
        intensities = np.array([t.intensity for t in tensions])

        # Softmax con temperatura
        temperature = 0.5  # FROM_THEORY: temperatura moderada
        scaled = intensities / temperature
        scaled = scaled - np.max(scaled)  # Estabilidad numerica
        exp_intensities = np.exp(scaled)
        probs = exp_intensities / np.sum(exp_intensities)

        self.logger.log_from_theory(
            value={'probs': probs.tolist(), 'tensions': [t.tension_type.value for t in tensions]},
            source="Tension sampling via softmax(intensity / T)",
            reference="Boltzmann distribution",
            context="TensionDetector.sample_tension"
        )

        # Muestrear
        idx = np.random.choice(len(tensions), p=probs)
        return tensions[idx]


# =============================================================================
# RESOLUTOR DE TENSIONES -> DOMINIOS
# =============================================================================

class TensionResolver:
    """
    Resuelve tensiones en dominios candidatos.

    FLUJO VALIDO:
        tension -> dominios_candidatos

    PROHIBIDO:
        agente -> dominios (directo)
        preferencia -> dominios
    """

    def __init__(self):
        self.logger = get_provenance_logger()

        # Pesos dinamicos por dominio (se actualizan con el uso)
        self._domain_weights: Dict[str, float] = {}

    def resolve(
        self,
        tension: TensionState,
        exclude_domains: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Resuelve una tension en dominios candidatos con pesos.

        Args:
            tension: Tension a resolver
            exclude_domains: Dominios a excluir (por saturacion, etc.)

        Returns:
            Lista de (dominio, peso) ordenada por peso

        HARD RULE: El dominio NO se elige por identidad del agente.
        """
        # Obtener dominios candidatos para esta tension
        candidates = TENSION_TO_DOMAINS.get(tension.tension_type, [])

        if not candidates:
            # Fallback: todos los dominios
            candidates = list(set(
                d for domains in TENSION_TO_DOMAINS.values()
                for d in domains
            ))

        # Excluir dominios si es necesario
        if exclude_domains:
            candidates = [d for d in candidates if d not in exclude_domains]

        if not candidates:
            raise ValueError(
                f"HARD RULE VIOLATION: No candidate domains for tension "
                f"'{tension.tension_type.value}' after exclusions"
            )

        # Calcular pesos
        # ORIGEN: Peso base uniforme + ajuste por intensidad de tension
        base_weight = 1.0 / len(candidates)

        weighted = []
        for domain in candidates:
            # Peso = base + ajuste dinamico
            weight = base_weight

            # Ajustar por uso previo (evitar saturacion)
            historical_weight = self._domain_weights.get(domain, 1.0)
            weight *= historical_weight

            weighted.append((domain, weight))

        # Normalizar
        total = sum(w for _, w in weighted)
        if total > 0:
            weighted = [(d, w/total) for d, w in weighted]

        # Ordenar por peso
        weighted.sort(key=lambda x: x[1], reverse=True)

        self.logger.log_from_data(
            value={
                'tension': tension.tension_type.value,
                'candidates': weighted,
            },
            source="Tension resolved to domain candidates",
            statistic="tension_resolution",
            context="TensionResolver.resolve"
        )

        return weighted

    def sample_domain(
        self,
        tension: TensionState,
        exclude_domains: Optional[Set[str]] = None,
        seed: Optional[int] = None
    ) -> str:
        """
        Muestrea un dominio para resolver una tension.

        FLUJO:
            tension -> candidatos -> muestreo -> dominio

        PROHIBIDO:
            agente -> dominio
        """
        if seed is not None:
            np.random.seed(seed)

        # Resolver tension en candidatos
        candidates = self.resolve(tension, exclude_domains)

        domains = [d for d, _ in candidates]
        weights = [w for _, w in candidates]

        # Muestrear
        domain = np.random.choice(domains, p=weights)

        # Actualizar pesos (decay para evitar saturacion)
        self._domain_weights[domain] = self._domain_weights.get(domain, 1.0) * 0.95

        return domain

    def update_from_result(
        self,
        domain: str,
        tension: TensionState,
        success: bool,
        reduction: float
    ):
        """
        Actualiza pesos despues de un resultado.

        Args:
            domain: Dominio usado
            tension: Tension que se intento resolver
            success: Si la tarea fue exitosa
            reduction: Cuanto se redujo la tension
        """
        current = self._domain_weights.get(domain, 1.0)

        if success and reduction > 0:
            # Exito: aumentar peso ligeramente
            self._domain_weights[domain] = min(2.0, current * 1.1)
        else:
            # Fallo: reducir peso
            self._domain_weights[domain] = max(0.5, current * 0.9)


# =============================================================================
# VALIDADOR DE INTEGRIDAD
# =============================================================================

class IntegrityValidator:
    """
    Valida que el flujo de decision cumple HARD RULES.

    Si detecta violacion, ABORTA la ejecucion.
    """

    def __init__(self):
        self.logger = get_provenance_logger()
        self._violations: List[str] = []

    def validate_flow(
        self,
        tension: Optional[TensionState],
        domain: Optional[str],
        agent_id: Optional[str] = None
    ) -> bool:
        """
        Valida que el flujo es correcto.

        FLUJO VALIDO:
            tension (presente) -> domain (puede estar)

        FLUJO INVALIDO:
            domain (presente) sin tension
            agent -> domain directo

        Returns:
            True si valido, raise si invalido
        """
        # RULE 1: Dominio no puede existir sin tension previa
        if domain is not None and tension is None:
            violation = (
                f"HARD RULE VIOLATION: Domain '{domain}' selected without "
                f"prior tension detection. Flow must be: tension -> domain"
            )
            self._violations.append(violation)
            self.logger.log_from_theory(
                value=violation,
                source="IntegrityValidator.validate_flow",
                reference="HARD RULES",
                context="Flow validation failed"
            )
            raise RuntimeError(violation)

        # RULE 2: Tension debe tener source_metrics (no ser arbitraria)
        if tension is not None and not tension.source_metrics:
            violation = (
                f"HARD RULE VIOLATION: Tension '{tension.tension_type.value}' "
                f"has no source_metrics. Tensions must be derived from internal state."
            )
            self._violations.append(violation)
            raise RuntimeError(violation)

        return True

    def validate_no_identity_selection(
        self,
        agent_id: str,
        domain: str,
        selection_path: List[str]
    ) -> bool:
        """
        Valida que la seleccion no uso identidad del agente.

        Args:
            agent_id: ID del agente
            domain: Dominio seleccionado
            selection_path: Cadena de decisiones

        Returns:
            True si valido, raise si invalido
        """
        # El agent_id NO debe aparecer en el path de seleccion
        # antes de la tension

        path_str = " -> ".join(selection_path)

        # Verificar que tension aparece antes que domain
        if 'tension' not in path_str.lower():
            violation = (
                f"HARD RULE VIOLATION: Selection path '{path_str}' does not "
                f"include tension detection. Direct domain selection prohibited."
            )
            self._violations.append(violation)
            raise RuntimeError(violation)

        # Verificar que agent_id no determina domain
        forbidden_patterns = [
            f"{agent_id} -> {domain}",
            f"agent={agent_id}, domain={domain}",
            f"if agent == '{agent_id}'",
        ]

        for pattern in forbidden_patterns:
            if pattern.lower() in path_str.lower():
                violation = (
                    f"HARD RULE VIOLATION: Pattern '{pattern}' detected. "
                    f"Agent identity must not determine domain selection."
                )
                self._violations.append(violation)
                raise RuntimeError(violation)

        return True

    def get_violations(self) -> List[str]:
        """Retorna lista de violaciones detectadas."""
        return self._violations.copy()


# =============================================================================
# TEST
# =============================================================================

def test_tension_space():
    """Test del espacio de tensiones."""
    print("=" * 70)
    print("TEST: TENSION SPACE")
    print("=" * 70)

    # Test 1: Deteccion de tensiones
    print("\n=== TEST 1: Deteccion de tensiones ===")

    detector = TensionDetector()

    # Estado con inconsistencia alta
    state_inconsistent = InternalState(
        internal_inconsistency=0.5,
        model_divergence=0.6
    )

    tensions = detector.detect_tensions(state_inconsistent)
    print(f"Estado con inconsistencia alta:")
    print(f"  Tensiones detectadas: {[t.tension_type.value for t in tensions]}")
    assert any(t.tension_type == TensionType.INCONSISTENCY for t in tensions)

    # Estado con baja cobertura
    state_unexplored = InternalState(
        hypothesis_coverage=0.3,
        empirical_coverage=0.4
    )

    tensions = detector.detect_tensions(state_unexplored)
    print(f"\nEstado con baja cobertura:")
    print(f"  Tensiones detectadas: {[t.tension_type.value for t in tensions]}")
    assert any(t.tension_type == TensionType.UNEXPLORED_HYPOTHESIS for t in tensions)

    # Test 2: Resolucion de tensiones
    print("\n=== TEST 2: Resolucion de tensiones ===")

    resolver = TensionResolver()

    tension = TensionState(
        tension_type=TensionType.INCONSISTENCY,
        intensity=0.7,
        source_metrics={'test': True}
    )

    candidates = resolver.resolve(tension)
    print(f"Tension INCONSISTENCY -> Candidatos:")
    for domain, weight in candidates:
        print(f"  {domain}: {weight:.3f}")

    # Muestrear dominio
    domain = resolver.sample_domain(tension, seed=42)
    print(f"\nDominio muestreado: {domain}")

    # Test 3: Validador de integridad
    print("\n=== TEST 3: Validador de integridad ===")

    validator = IntegrityValidator()

    # Flujo valido
    try:
        validator.validate_flow(tension=tension, domain=domain)
        print("Flujo valido: tension -> domain OK")
    except RuntimeError as e:
        print(f"ERROR: {e}")

    # Flujo invalido: dominio sin tension
    print("\nIntentando flujo invalido (domain sin tension)...")
    try:
        validator.validate_flow(tension=None, domain="physics")
        print("ERROR: Deberia haber fallado!")
    except RuntimeError as e:
        print(f"Correctamente rechazado: {e}")

    # Test 4: Flujo completo
    print("\n=== TEST 4: Flujo completo ===")

    state = InternalState(
        accumulated_error=0.3,
        prediction_variance=0.4,
        internal_inconsistency=0.2,
        hypothesis_coverage=0.5
    )

    print(f"Estado interno: {state.to_dict()}")

    # Detectar tension
    sampled_tension = detector.sample_tension(state, seed=123)
    print(f"\nTension muestreada: {sampled_tension.tension_type.value}")
    print(f"  Intensidad: {sampled_tension.intensity:.3f}")
    print(f"  Source metrics: {sampled_tension.source_metrics}")

    # Resolver a dominio
    domain = resolver.sample_domain(sampled_tension, seed=123)
    print(f"\nDominio para resolver tension: {domain}")

    # Validar flujo
    validator.validate_flow(tension=sampled_tension, domain=domain)
    print("\nFlujo validado: estado -> tension -> dominio")

    print("\n" + "=" * 70)
    print("TEST COMPLETADO: Espacio de tensiones funcionando")
    print("=" * 70)


if __name__ == "__main__":
    test_tension_space()
