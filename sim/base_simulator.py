"""
Base Simulator - Lógica común para simuladores estructurales

NORMA DURA:
- Opera sobre ESTRUCTURA, no semántica
- Poblaciones abstractas
- Operadores estructurales
- Escenarios in silico
- Métricas de estabilidad (E-series)
- Sorpresa estructural

Este módulo define la interfaz común que heredan:
- GeneticSimulator (biología teórica)
- CryptoSimulator (dinámicas cripto)
"""

import random
import math
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import sys
sys.path.insert(0, '/opt/iaverso')

from core.endolens import get_endolens, StructuralState, ESeries, Signature
from core.neosynt import get_neosynt, Prediction


# ==============================================================================
# DATACLASSES COMUNES
# ==============================================================================

@dataclass
class Element:
    """Un elemento en la población (estructura individual)."""
    id: str
    data: Dict                  # Datos abstractos del elemento
    state: StructuralState      # Estado estructural calculado
    generation: int             # Generación de origen
    parent_id: Optional[str] = None
    operators_applied: List[str] = field(default_factory=list)
    fitness: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class Population:
    """Población abstracta de elementos."""
    id: str
    domain: str                 # 'genetic', 'crypto', etc.
    elements: List[Element]
    generation: int
    parameters: Dict
    created_at: datetime = field(default_factory=datetime.now)
    history: List[Dict] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.elements)

    @property
    def avg_fitness(self) -> float:
        if not self.elements:
            return 0.0
        return sum(e.fitness for e in self.elements) / len(self.elements)

    @property
    def best_element(self) -> Optional[Element]:
        if not self.elements:
            return None
        return max(self.elements, key=lambda e: e.fitness)

    @property
    def signature(self) -> str:
        """Firma agregada de la población."""
        if not self.elements:
            return "⟨EMPTY⟩"
        best = self.best_element
        return str(best.state.signature) if best else "⟨NULL⟩"


@dataclass
class Scenario:
    """Un escenario in silico."""
    id: str
    domain: str
    operator: str               # Qué operador aplicar
    prediction: Prediction      # Qué espera el modelo
    population_id: str          # Sobre qué población
    parameters: Dict            # Parámetros del escenario
    uncertainty: float          # Incertidumbre inicial


@dataclass
class ScenarioResult:
    """Resultado de un escenario in silico."""
    scenario_id: str
    observed: StructuralState   # Lo que realmente pasó
    predicted: ESeries          # Lo que esperaba
    surprise: float             # Divergencia modelo-resultado
    model_update: Dict          # Cómo actualizar el modelo
    success: bool               # ¿Redujo sorpresa?
    trace: List[str]            # Trazabilidad


@dataclass
class OperatorSpec:
    """Especificación de un operador estructural."""
    id: str
    name: str
    description: str
    parameters: List[str]       # Parámetros que acepta
    domain: str                 # 'genetic', 'crypto', 'all'


# ==============================================================================
# CLASE BASE ABSTRACTA
# ==============================================================================

class BaseSimulator(ABC):
    """
    Simulador base abstracto.

    Define la interfaz común para todos los simuladores de dominio.
    Los simuladores concretos heredan e implementan los métodos abstractos.
    """

    def __init__(self, domain: str):
        self.domain = domain
        self.endolens = get_endolens()
        self.neosynt = get_neosynt()
        self.populations: Dict[str, Population] = {}
        self._operators: Dict[str, Callable] = {}
        self._operator_specs: Dict[str, OperatorSpec] = {}

        # Registrar operadores del dominio
        self._register_operators()

    @abstractmethod
    def _register_operators(self):
        """Registra los operadores específicos del dominio."""
        pass

    @abstractmethod
    def _generate_element(self, seed: Any, params: Dict) -> Element:
        """Genera un elemento del dominio."""
        pass

    @abstractmethod
    def _calculate_fitness(self, element: Element) -> float:
        """Calcula el fitness de un elemento según el dominio."""
        pass

    @abstractmethod
    def _serialize_element(self, element: Element) -> str:
        """Serializa un elemento para calcular su estado estructural."""
        pass

    # ==========================================================================
    # MÉTODOS COMUNES (no abstractos)
    # ==========================================================================

    def register_operator(self, spec: OperatorSpec, func: Callable):
        """Registra un operador con su especificación."""
        self._operators[spec.id] = func
        self._operator_specs[spec.id] = spec

    def list_operators(self) -> List[OperatorSpec]:
        """Lista los operadores disponibles."""
        return list(self._operator_specs.values())

    def init_population(
        self,
        seed: Any,
        size: int = 20,
        parameters: Optional[Dict] = None
    ) -> Population:
        """
        Inicializa una población abstracta.

        Args:
            seed: Semilla para generación (texto, datos, etc.)
            size: Tamaño de la población
            parameters: Parámetros específicos del dominio

        Returns:
            Population inicializada
        """
        params = parameters or {}
        pop_id = str(uuid.uuid4())[:8]

        elements = []
        for i in range(size):
            element = self._generate_element(seed, params)
            element.generation = 0

            # Calcular estado estructural
            serialized = self._serialize_element(element)
            element.state = self.endolens.process(serialized)
            element.fitness = self._calculate_fitness(element)

            elements.append(element)

        population = Population(
            id=pop_id,
            domain=self.domain,
            elements=elements,
            generation=0,
            parameters=params
        )

        population.history.append({
            'event': 'init',
            'generation': 0,
            'size': size,
            'avg_fitness': population.avg_fitness,
            'timestamp': datetime.now().isoformat()
        })

        self.populations[pop_id] = population
        return population

    def apply_operator(
        self,
        population_id: str,
        operator_id: str,
        iterations: int = 1,
        parameters: Optional[Dict] = None
    ) -> Population:
        """
        Aplica un operador estructural a la población.

        Args:
            population_id: ID de la población
            operator_id: ID del operador
            iterations: Número de iteraciones
            parameters: Parámetros adicionales

        Returns:
            Población modificada
        """
        if population_id not in self.populations:
            raise ValueError(f"Population {population_id} not found")

        if operator_id not in self._operators:
            raise ValueError(f"Operator {operator_id} not found")

        population = self.populations[population_id]
        operator_func = self._operators[operator_id]
        params = parameters or {}

        for _ in range(iterations):
            # Aplicar operador a elementos
            new_elements = []
            for element in population.elements:
                modified = operator_func(element, params)
                modified.operators_applied.append(operator_id)
                modified.generation = population.generation + 1

                # Recalcular estado
                serialized = self._serialize_element(modified)
                modified.state = self.endolens.process(serialized)
                modified.fitness = self._calculate_fitness(modified)

                new_elements.append(modified)

            population.elements = new_elements
            population.generation += 1

        population.history.append({
            'event': 'operator_applied',
            'operator': operator_id,
            'iterations': iterations,
            'generation': population.generation,
            'avg_fitness': population.avg_fitness,
            'timestamp': datetime.now().isoformat()
        })

        return population

    def propose_scenario(
        self,
        population: Population,
        operator_id: Optional[str] = None
    ) -> Scenario:
        """
        Propone un escenario in silico.

        Args:
            population: Población sobre la que simular
            operator_id: Operador a usar (o elegir automáticamente)

        Returns:
            Scenario propuesto
        """
        if operator_id is None:
            # Elegir operador por entropía (más informativo primero)
            operator_id = random.choice(list(self._operators.keys()))

        # Obtener estado actual
        best = population.best_element
        current_state = best.state if best else None

        # Predecir resultado
        prediction = self.neosynt.predict_operator_effect(
            current_state,
            operator_id
        ) if current_state else Prediction(
            expected_stability=0.5,
            confidence=0.3,
            expected_eseries=None
        )

        scenario = Scenario(
            id=str(uuid.uuid4())[:8],
            domain=self.domain,
            operator=operator_id,
            prediction=prediction,
            population_id=population.id,
            parameters={},
            uncertainty=1.0 - prediction.confidence
        )

        return scenario

    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        """
        Ejecuta un escenario in silico.

        Args:
            scenario: Escenario a ejecutar

        Returns:
            Resultado con sorpresa y actualización de modelo
        """
        population = self.populations.get(scenario.population_id)
        if not population:
            raise ValueError(f"Population {scenario.population_id} not found")

        trace = [f"Ejecutando escenario {scenario.id}"]
        trace.append(f"Operador: {scenario.operator}")
        trace.append(f"Predicción: estabilidad={scenario.prediction.expected_stability:.3f}")

        # Aplicar operador
        self.apply_operator(
            scenario.population_id,
            scenario.operator,
            iterations=1
        )

        # Obtener resultado observado
        best = population.best_element
        observed = best.state if best else None

        # Calcular sorpresa
        if observed and scenario.prediction.expected_eseries:
            surprise = self._calculate_surprise(
                observed.eseries,
                scenario.prediction.expected_eseries
            )
        else:
            surprise = 0.5  # Incertidumbre máxima

        trace.append(f"Observado: estabilidad={observed.stability:.3f}" if observed else "Observado: NULL")
        trace.append(f"Sorpresa: {surprise:.3f}")

        # Determinar si fue exitoso (redujo sorpresa)
        success = surprise < scenario.uncertainty

        # Generar actualización de modelo
        model_update = {
            'operator': scenario.operator,
            'observed_effect': observed.stability if observed else None,
            'predicted_effect': scenario.prediction.expected_stability,
            'surprise': surprise,
            'adjust_prediction': surprise > 0.3
        }

        result = ScenarioResult(
            scenario_id=scenario.id,
            observed=observed,
            predicted=scenario.prediction.expected_eseries,
            surprise=surprise,
            model_update=model_update,
            success=success,
            trace=trace
        )

        return result

    def _calculate_surprise(self, observed: ESeries, predicted: ESeries) -> float:
        """Calcula la sorpresa (divergencia) entre observado y predicho."""
        if not observed or not predicted:
            return 0.5

        diff_E0 = abs(observed.E0 - predicted.E0)
        diff_E1 = abs(observed.E1 - predicted.E1)
        diff_E2 = abs(observed.E2 - predicted.E2)
        diff_E3 = abs(observed.E3 - predicted.E3)

        # Sorpresa como promedio de divergencias
        surprise = (diff_E0 + diff_E1 + diff_E2 + diff_E3) / 4.0
        return min(1.0, surprise)

    def get_population(self, population_id: str) -> Optional[Population]:
        """Obtiene una población por ID."""
        return self.populations.get(population_id)

    def list_populations(self) -> List[Dict]:
        """Lista todas las poblaciones activas."""
        return [
            {
                'id': p.id,
                'domain': p.domain,
                'size': p.size,
                'generation': p.generation,
                'avg_fitness': p.avg_fitness,
                'signature': p.signature,
                'created_at': p.created_at.isoformat()
            }
            for p in self.populations.values()
        ]

    def to_dict(self, population: Population) -> Dict:
        """Serializa una población a diccionario."""
        return {
            'id': population.id,
            'domain': population.domain,
            'size': population.size,
            'generation': population.generation,
            'avg_fitness': population.avg_fitness,
            'signature': population.signature,
            'parameters': population.parameters,
            'history': population.history[-10:],  # últimos 10 eventos
            'elements': [
                {
                    'id': e.id,
                    'data': e.data,
                    'fitness': e.fitness,
                    'signature': str(e.state.signature),
                    'stability': e.state.stability,
                    'operators': e.operators_applied
                }
                for e in population.elements[:10]  # primeros 10
            ],
            'best': {
                'id': population.best_element.id,
                'fitness': population.best_element.fitness,
                'signature': str(population.best_element.state.signature)
            } if population.best_element else None
        }
