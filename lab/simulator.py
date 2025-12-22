"""
Simulador Estructural - Motor de Inferencia Activa

NORMA DURA:
- Opera sobre ESTRUCTURA, no semántica
- Poblaciones abstractas de estructuras
- Operadores estructurales
- Escenarios in silico

Vocabulario abstracto:
- Población (antes: cultivo)
- Operador estructural (antes: vector)
- Aplicación (antes: transformación)
- Regla de simulación (antes: protocolo)
- Escenario in silico (antes: experimento)

Rol en Inferencia Activa: ACCIÓN
"¿Qué hago para confirmar mi predicción?"
"""

import random
import math
import uuid
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import sys
sys.path.insert(0, '/opt/iaverso')

from core.endolens import get_endolens, StructuralState, ESeries, Signature
from core.neosynt import get_neosynt, Prediction


@dataclass
class Element:
    """Un elemento en la población (estructura individual)."""
    id: str
    sequence: str           # Secuencia/representación
    state: StructuralState  # Estado estructural calculado
    generation: int         # Generación de origen
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


@dataclass
class Scenario:
    """Un escenario in silico."""
    id: str
    operator: str            # Qué operador aplicar
    prediction: Prediction   # Qué espera el modelo
    population_id: str       # Sobre qué población
    uncertainty: float       # Cuánta sorpresa reduciría
    computational_cost: float


@dataclass
class ScenarioResult:
    """Resultado de un escenario in silico."""
    scenario_id: str
    observed: StructuralState      # Lo que realmente pasó
    predicted: ESeries             # Lo que esperaba
    surprise: float                # Diferencia (free energy)
    model_update: Dict             # Cómo actualizar el modelo
    success: bool                  # ¿Redujo sorpresa?


class StructuralSimulator:
    """
    Simulador Estructural.
    
    Implementa el brazo de ACCIÓN en inferencia activa:
    - Inicializa poblaciones abstractas
    - Aplica operadores estructurales
    - Ejecuta reglas de simulación
    - Propone y ejecuta escenarios in silico
    
    NORMA DURA: Solo opera matemáticamente. No interpreta.
    """
    
    # Operadores estructurales disponibles
    OPERATORS = {
        'O_point': 'Modificación puntual',
        'O_region': 'Modificación regional',
        'O_combine': 'Combinación',
        'O_reduce': 'Reducción',
        'O_expand': 'Expansión',
        'O_invert': 'Inversión',
        'O_duplicate': 'Duplicación',
        'O_permute': 'Permutación',
    }
    
    def __init__(self):
        self.endolens = get_endolens()
        self.neosynt = get_neosynt()
        self.populations: Dict[str, Population] = {}
        self.scenarios: Dict[str, ScenarioResult] = {}
        
    # === POBLACIONES ===
    
    def init_population(
        self,
        seed: str,
        size: int = 20,
        parameters: Dict = None
    ) -> Population:
        """
        Inicializa una población abstracta.
        
        parameters = {
            'variation_rate': 0.1,    # Tasa de variación
            'selection': 'fitness',   # Criterio de selección
            'target': 'stability',    # Objetivo estructural
            'exploration': 1.0,       # Factor de exploración
        }
        """
        parameters = parameters or {
            'variation_rate': 0.1,
            'selection': 'fitness',
            'target': 'stability',
            'exploration': 1.0,
        }
        
        pop_id = str(uuid.uuid4())[:8]
        elements = []
        
        # Crear población inicial con variantes del seed
        for i in range(size):
            if i == 0:
                seq = seed
            else:
                seq = self._apply_variation(seed, parameters['variation_rate'])
            
            state = self.endolens.process(seq)
            element = Element(
                id=f"{pop_id}_{i}",
                sequence=seq,
                state=state,
                generation=0,
                fitness=self._evaluate_fitness(state, parameters['target'])
            )
            elements.append(element)
        
        population = Population(
            id=pop_id,
            elements=elements,
            generation=0,
            parameters=parameters
        )
        
        self.populations[pop_id] = population
        return population
    
    def apply_operator(
        self,
        population_id: str,
        operator: str,
        iterations: int = 10
    ) -> Population:
        """
        Aplica operador estructural a la población.
        """
        population = self.populations.get(population_id)
        if not population:
            raise ValueError(f"Population {population_id} not found")
        
        for gen in range(iterations):
            # 1. Evaluar fitness
            for element in population.elements:
                element.fitness = self._evaluate_fitness(
                    element.state, 
                    population.parameters['target']
                )
            
            # 2. Selección
            selected = self._select(population.elements, population.parameters)
            
            # 3. Aplicar operador con variación
            new_elements = []
            while len(new_elements) < len(population.elements):
                parent = random.choice(selected)
                
                # Decidir operación
                if random.random() < population.parameters['variation_rate']:
                    child_seq = self._apply_operator_to_sequence(parent.sequence, operator)
                    applied_op = operator
                elif len(selected) > 1 and random.random() < 0.3:
                    parent2 = random.choice([e for e in selected if e.id != parent.id])
                    child_seq = self._combine(parent.sequence, parent2.sequence)
                    applied_op = 'O_combine'
                else:
                    child_seq = parent.sequence
                    applied_op = 'copy'
                
                child_state = self.endolens.process(child_seq)
                child = Element(
                    id=f"{population_id}_{population.generation + gen + 1}_{len(new_elements)}",
                    sequence=child_seq,
                    state=child_state,
                    generation=population.generation + gen + 1,
                    parent_id=parent.id,
                    operators_applied=[applied_op],
                    fitness=self._evaluate_fitness(child_state, population.parameters['target'])
                )
                new_elements.append(child)
            
            # Registrar historia
            population.history.append({
                'generation': population.generation + gen + 1,
                'avg_fitness': sum(e.fitness for e in new_elements) / len(new_elements),
                'best_fitness': max(e.fitness for e in new_elements),
                'diversity': len(set(e.sequence for e in new_elements)) / len(new_elements)
            })
            
            population.elements = new_elements
            population.generation += 1
        
        return population
    
    # === ESCENARIOS IN SILICO ===
    
    def propose_scenario(
        self, 
        population: Population,
        target_state: Dict = None
    ) -> Scenario:
        """
        Propone un escenario in silico.
        
        "Si aplico X, espero Y. Ejecuto para confirmar."
        """
        best = population.best_element
        if not best:
            raise ValueError("Empty population")
        
        operators_to_try = ['O_point', 'O_combine', 'O_reduce']
        
        best_scenario = None
        max_uncertainty = 0
        
        for operator in operators_to_try:
            prediction = self.neosynt.predict(best.state, operator.replace('O_', 'T_'))
            uncertainty = 1 - prediction.confidence
            
            if uncertainty > max_uncertainty:
                max_uncertainty = uncertainty
                best_scenario = Scenario(
                    id=str(uuid.uuid4())[:8],
                    operator=operator,
                    prediction=prediction,
                    population_id=population.id,
                    uncertainty=uncertainty,
                    computational_cost=len(population.elements) * 0.1
                )
        
        return best_scenario
    
    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        """
        Ejecuta un escenario in silico.
        """
        population = self.populations.get(scenario.population_id)
        if not population:
            raise ValueError(f"Population {scenario.population_id} not found")
        
        best = population.best_element
        transformed = self._apply_operator_to_element(best, scenario.operator)
        
        surprise = self.neosynt.calculate_surprise(
            scenario.prediction,
            transformed.state
        )
        
        model_update = {}
        if surprise < 0.1:
            model_update['confidence_adjustment'] = +0.05
            model_update['action'] = 'reinforce'
        elif surprise > 0.3:
            model_update['confidence_adjustment'] = -0.1
            model_update['action'] = 'update'
            model_update['observed_delta'] = {
                'E0': transformed.state.eseries.E0 - scenario.prediction.expected_state.E0,
                'E1': transformed.state.eseries.E1 - scenario.prediction.expected_state.E1,
                'E2': transformed.state.eseries.E2 - scenario.prediction.expected_state.E2,
                'E3': transformed.state.eseries.E3 - scenario.prediction.expected_state.E3,
            }
        else:
            model_update['action'] = 'maintain'
        
        result = ScenarioResult(
            scenario_id=scenario.id,
            observed=transformed.state,
            predicted=scenario.prediction.expected_state,
            surprise=surprise,
            model_update=model_update,
            success=surprise < 0.2
        )
        
        self.scenarios[scenario.id] = result
        return result
    
    # === REGLAS DE SIMULACIÓN ===
    
    def run_simulation_rule(
        self,
        population: Population,
        rule: Dict
    ) -> Dict:
        """
        Ejecuta una regla de simulación sobre una población.
        
        rule = {
            'iterations': 10,         # Iteraciones
            'metric': 'stability',    # Qué medir
            'threshold': 0.7,         # Umbral de aceptación
            'replicates': 3,          # Repeticiones
        }
        """
        results = []
        
        for rep in range(rule.get('replicates', 1)):
            clone_pop = self.init_population(
                population.best_element.sequence if population.best_element else '',
                size=population.size,
                parameters=population.parameters.copy()
            )
            
            initial_state = clone_pop.best_element.state if clone_pop.best_element else None
            
            self.apply_operator(
                clone_pop.id,
                'O_point',
                iterations=rule.get('iterations', 10)
            )
            
            final_state = clone_pop.best_element.state if clone_pop.best_element else None
            
            if initial_state and final_state:
                results.append({
                    'replicate': rep + 1,
                    'initial_stability': initial_state.stability,
                    'final_stability': final_state.stability,
                    'stability_delta': final_state.stability - initial_state.stability,
                    'initial_E3': initial_state.eseries.E3,
                    'final_E3': final_state.eseries.E3,
                    'E3_delta': final_state.eseries.E3 - initial_state.eseries.E3,
                    'passed': final_state.stability >= rule.get('threshold', 0.7)
                })
        
        passed_count = sum(1 for r in results if r['passed'])
        avg_delta = sum(r['stability_delta'] for r in results) / len(results) if results else 0
        
        return {
            'rule': rule,
            'replicates': results,
            'summary': {
                'passed': passed_count,
                'total': len(results),
                'pass_rate': passed_count / len(results) if results else 0,
                'avg_stability_delta': avg_delta,
                'outcome': 'threshold_met' if passed_count > len(results) / 2 else 'threshold_not_met'
            }
        }
    
    # === MÉTODOS PRIVADOS ===
    
    def _evaluate_fitness(self, state: StructuralState, target: str) -> float:
        if target == 'stability':
            return state.stability
        elif target == 'complexity':
            return state.eseries.E3
        elif target == 'simplicity':
            return 1 - state.eseries.E3
        elif target == 'variability':
            return state.eseries.E1
        else:
            return state.stability
    
    def _select(self, elements: List[Element], parameters: Dict) -> List[Element]:
        sorted_elements = sorted(elements, key=lambda e: e.fitness, reverse=True)
        cutoff = max(2, len(sorted_elements) // 2)
        return sorted_elements[:cutoff]
    
    def _apply_variation(self, seq: str, rate: float) -> str:
        if not seq:
            return seq
        chars = list(seq)
        for i in range(len(chars)):
            if random.random() < rate:
                if chars[i].isalpha():
                    chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
                elif chars[i].isdigit():
                    chars[i] = random.choice('0123456789')
        return ''.join(chars)
    
    def _apply_operator_to_sequence(self, seq: str, operator: str) -> str:
        if not seq:
            return seq
            
        if operator == 'O_point':
            chars = list(seq)
            i = random.randint(0, len(chars) - 1)
            if chars[i].isalpha():
                chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
            return ''.join(chars)
        elif operator == 'O_region':
            if len(seq) < 3:
                return self._apply_operator_to_sequence(seq, 'O_point')
            start = random.randint(0, len(seq) - 3)
            region = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(3))
            return seq[:start] + region + seq[start + 3:]
        elif operator == 'O_reduce':
            if len(seq) <= 2:
                return seq
            start = random.randint(0, len(seq) - 2)
            return seq[:start] + seq[start + 2:]
        elif operator == 'O_expand':
            pos = random.randint(0, len(seq))
            insert = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(2))
            return seq[:pos] + insert + seq[pos:]
        elif operator == 'O_invert':
            if len(seq) < 4:
                return seq[::-1]
            start = random.randint(0, len(seq) - 4)
            end = start + random.randint(2, 4)
            return seq[:start] + seq[start:end][::-1] + seq[end:]
        elif operator == 'O_duplicate':
            if len(seq) < 3:
                return seq + seq
            start = random.randint(0, len(seq) - 2)
            size = random.randint(1, min(3, len(seq) - start))
            region = seq[start:start + size]
            return seq[:start + size] + region + seq[start + size:]
        elif operator == 'O_permute':
            chars = list(seq)
            random.shuffle(chars)
            return ''.join(chars)
        else:
            return seq
    
    def _apply_operator_to_element(self, element: Element, operator: str) -> Element:
        new_seq = self._apply_operator_to_sequence(element.sequence, operator)
        new_state = self.endolens.process(new_seq)
        
        return Element(
            id=str(uuid.uuid4())[:8],
            sequence=new_seq,
            state=new_state,
            generation=element.generation + 1,
            parent_id=element.id,
            operators_applied=element.operators_applied + [operator],
            fitness=0.0
        )
    
    def _combine(self, seq1: str, seq2: str) -> str:
        if not seq1 or not seq2:
            return seq1 or seq2
        point = random.randint(1, min(len(seq1), len(seq2)) - 1)
        return seq1[:point] + seq2[point:]


# Singleton
_instance = None

def get_simulator() -> StructuralSimulator:
    global _instance
    if _instance is None:
        _instance = StructuralSimulator()
    return _instance
