"""
Laboratorio Genético - Motor de Inferencia Activa

NORMA DURA:
- Opera sobre ESTRUCTURA, no semántica
- No dice "esto cura X", dice "esto reduce E3 de 0.8 a 0.5"
- Cultiva poblaciones de estructuras
- Evoluciona hacia objetivos estructurales
- El humano interpreta resultados

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
class Specimen:
    """Un espécimen en el laboratorio (estructura individual)."""
    id: str
    sequence: str           # Secuencia/texto original
    state: StructuralState  # Estado estructural calculado
    generation: int         # Generación de origen
    parent_id: Optional[str] = None
    mutations: List[str] = field(default_factory=list)
    fitness: float = 0.0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class Culture:
    """Cultivo de especímenes (población)."""
    id: str
    specimens: List[Specimen]
    generation: int
    conditions: Dict
    created_at: datetime = field(default_factory=datetime.now)
    history: List[Dict] = field(default_factory=list)
    
    @property
    def size(self) -> int:
        return len(self.specimens)
    
    @property
    def avg_fitness(self) -> float:
        if not self.specimens:
            return 0.0
        return sum(s.fitness for s in self.specimens) / len(self.specimens)
    
    @property
    def best_specimen(self) -> Optional[Specimen]:
        if not self.specimens:
            return None
        return max(self.specimens, key=lambda s: s.fitness)


@dataclass
class Experiment:
    """Un experimento de inferencia activa."""
    id: str
    action: str              # Qué transformación aplicar
    prediction: Prediction   # Qué espera el modelo
    culture_id: str          # Sobre qué cultivo
    uncertainty: float       # Cuánta sorpresa reduciría
    cost: float              # Recursos computacionales estimados


@dataclass
class ExperimentResult:
    """Resultado de un experimento."""
    experiment_id: str
    observed: StructuralState      # Lo que realmente pasó
    predicted: ESeries             # Lo que esperaba
    surprise: float                # Diferencia (free energy)
    model_update: Dict             # Cómo actualizar el modelo
    success: bool                  # ¿Redujo sorpresa?


class GeneticLab:
    """
    Laboratorio Genético Computacional.
    
    Implementa el brazo de ACCIÓN en inferencia activa:
    - Crea cultivos de estructuras
    - Aplica vectores (mutaciones, recombinaciones)
    - Ejecuta protocolos de validación
    - Propone y ejecuta experimentos
    
    NORMA DURA: Solo opera matemáticamente. No interpreta.
    """
    
    # Vectores de transformación disponibles
    VECTORS = {
        'V_mut_point': 'Mutación puntual',
        'V_mut_region': 'Mutación regional',
        'V_cross': 'Recombinación',
        'V_del': 'Deleción',
        'V_ins': 'Inserción',
        'V_inv': 'Inversión',
        'V_dup': 'Duplicación',
        'V_shuffle': 'Barajado',
    }
    
    def __init__(self):
        self.endolens = get_endolens()
        self.neosynt = get_neosynt()
        self.cultures: Dict[str, Culture] = {}
        self.experiments: Dict[str, ExperimentResult] = {}
        
    # === CULTIVOS ===
    
    def create_culture(
        self,
        seed: str,
        population_size: int = 20,
        conditions: Dict = None
    ) -> Culture:
        """
        Crea un cultivo a partir de una semilla.
        
        conditions = {
            'mutation_rate': 0.1,     # Probabilidad de mutación
            'selection': 'fitness',   # Criterio de selección
            'target': 'stability',    # Objetivo estructural
            'temperature': 1.0,       # Factor de exploración
        }
        """
        conditions = conditions or {
            'mutation_rate': 0.1,
            'selection': 'fitness',
            'target': 'stability',
            'temperature': 1.0,
        }
        
        culture_id = str(uuid.uuid4())[:8]
        specimens = []
        
        # Crear población inicial con variantes del seed
        for i in range(population_size):
            if i == 0:
                # Primer espécimen es el original
                seq = seed
            else:
                # Los demás son variantes
                seq = self._mutate_sequence(seed, conditions['mutation_rate'])
            
            state = self.endolens.process(seq)
            specimen = Specimen(
                id=f"{culture_id}_{i}",
                sequence=seq,
                state=state,
                generation=0,
                fitness=self._calculate_fitness(state, conditions['target'])
            )
            specimens.append(specimen)
        
        culture = Culture(
            id=culture_id,
            specimens=specimens,
            generation=0,
            conditions=conditions
        )
        
        self.cultures[culture_id] = culture
        return culture
    
    def evolve_culture(self, culture_id: str, generations: int = 10) -> Culture:
        """
        Evoluciona un cultivo por N generaciones.
        
        NORMA DURA: Evoluciona hacia objetivo ESTRUCTURAL, no semántico.
        """
        culture = self.cultures.get(culture_id)
        if not culture:
            raise ValueError(f"Culture {culture_id} not found")
        
        for gen in range(generations):
            # 1. Evaluar fitness
            for specimen in culture.specimens:
                specimen.fitness = self._calculate_fitness(
                    specimen.state, 
                    culture.conditions['target']
                )
            
            # 2. Selección
            selected = self._select(culture.specimens, culture.conditions)
            
            # 3. Reproducción con variación
            new_specimens = []
            while len(new_specimens) < len(culture.specimens):
                parent = random.choice(selected)
                
                # Decidir operación
                if random.random() < culture.conditions['mutation_rate']:
                    # Mutación
                    child_seq = self._mutate_sequence(
                        parent.sequence, 
                        culture.conditions['mutation_rate']
                    )
                    mutation = 'V_mut_point'
                elif len(selected) > 1 and random.random() < 0.3:
                    # Recombinación
                    parent2 = random.choice([s for s in selected if s.id != parent.id])
                    child_seq = self._crossover(parent.sequence, parent2.sequence)
                    mutation = 'V_cross'
                else:
                    # Copia
                    child_seq = parent.sequence
                    mutation = 'copy'
                
                child_state = self.endolens.process(child_seq)
                child = Specimen(
                    id=f"{culture_id}_{culture.generation + gen + 1}_{len(new_specimens)}",
                    sequence=child_seq,
                    state=child_state,
                    generation=culture.generation + gen + 1,
                    parent_id=parent.id,
                    mutations=[mutation],
                    fitness=self._calculate_fitness(child_state, culture.conditions['target'])
                )
                new_specimens.append(child)
            
            # Registrar historia
            culture.history.append({
                'generation': culture.generation + gen + 1,
                'avg_fitness': sum(s.fitness for s in new_specimens) / len(new_specimens),
                'best_fitness': max(s.fitness for s in new_specimens),
                'diversity': len(set(s.sequence for s in new_specimens)) / len(new_specimens)
            })
            
            culture.specimens = new_specimens
            culture.generation += 1
        
        return culture
    
    # === VECTORES DE TRANSFORMACIÓN ===
    
    def apply_vector(self, specimen: Specimen, vector: str) -> Specimen:
        """
        Aplica un vector de transformación a un espécimen.
        
        Retorna nuevo espécimen (no modifica el original).
        """
        seq = specimen.sequence
        
        if vector == 'V_mut_point':
            new_seq = self._mutate_point(seq)
        elif vector == 'V_mut_region':
            new_seq = self._mutate_region(seq)
        elif vector == 'V_del':
            new_seq = self._delete_region(seq)
        elif vector == 'V_ins':
            new_seq = self._insert_region(seq)
        elif vector == 'V_inv':
            new_seq = self._invert_region(seq)
        elif vector == 'V_dup':
            new_seq = self._duplicate_region(seq)
        elif vector == 'V_shuffle':
            new_seq = self._shuffle_sequence(seq)
        else:
            new_seq = seq
        
        new_state = self.endolens.process(new_seq)
        
        return Specimen(
            id=str(uuid.uuid4())[:8],
            sequence=new_seq,
            state=new_state,
            generation=specimen.generation + 1,
            parent_id=specimen.id,
            mutations=specimen.mutations + [vector],
            fitness=0.0  # Se calculará después
        )
    
    # === INFERENCIA ACTIVA ===
    
    def propose_experiment(
        self, 
        culture: Culture,
        target_state: Dict = None
    ) -> Experiment:
        """
        Propone un experimento que minimice sorpresa.
        
        "Si aplico X, espero Y. Ejecuto para confirmar."
        
        NORMA DURA: El experimento busca CERTEZA ESTRUCTURAL, no verdad semántica.
        """
        best = culture.best_specimen
        if not best:
            raise ValueError("Empty culture")
        
        # Determinar qué transformación probar
        # Elegir la que tenga mayor incertidumbre (más información potencial)
        vectors_to_try = ['V_mut_point', 'V_cross', 'V_del']
        
        best_experiment = None
        max_uncertainty = 0
        
        for vector in vectors_to_try:
            # Predecir resultado
            prediction = self.neosynt.predict(best.state, vector.replace('V_', 'T_'))
            
            # Estimar incertidumbre (qué tanto no sabemos)
            uncertainty = 1 - prediction.confidence
            
            if uncertainty > max_uncertainty:
                max_uncertainty = uncertainty
                best_experiment = Experiment(
                    id=str(uuid.uuid4())[:8],
                    action=vector,
                    prediction=prediction,
                    culture_id=culture.id,
                    uncertainty=uncertainty,
                    cost=len(culture.specimens) * 0.1  # Costo proporcional al tamaño
                )
        
        return best_experiment
    
    def execute_experiment(self, experiment: Experiment) -> ExperimentResult:
        """
        Ejecuta un experimento y compara con predicción.
        
        NORMA DURA: Solo reporta números. El humano interpreta.
        """
        culture = self.cultures.get(experiment.culture_id)
        if not culture:
            raise ValueError(f"Culture {experiment.culture_id} not found")
        
        # Aplicar la acción
        best = culture.best_specimen
        transformed = self.apply_vector(best, experiment.action)
        
        # Calcular sorpresa
        surprise = self.neosynt.calculate_surprise(
            experiment.prediction,
            transformed.state
        )
        
        # Determinar actualización del modelo
        model_update = {}
        if surprise < 0.1:
            # Predicción acertada
            model_update['confidence_adjustment'] = +0.05
            model_update['action'] = 'reinforce'
        elif surprise > 0.3:
            # Predicción fallida - aprender
            model_update['confidence_adjustment'] = -0.1
            model_update['action'] = 'update'
            model_update['observed_delta'] = {
                'E0': transformed.state.eseries.E0 - experiment.prediction.expected_state.E0,
                'E1': transformed.state.eseries.E1 - experiment.prediction.expected_state.E1,
                'E2': transformed.state.eseries.E2 - experiment.prediction.expected_state.E2,
                'E3': transformed.state.eseries.E3 - experiment.prediction.expected_state.E3,
            }
        else:
            model_update['action'] = 'maintain'
        
        result = ExperimentResult(
            experiment_id=experiment.id,
            observed=transformed.state,
            predicted=experiment.prediction.expected_state,
            surprise=surprise,
            model_update=model_update,
            success=surprise < 0.2
        )
        
        self.experiments[experiment.id] = result
        return result
    
    # === PROTOCOLOS DE VALIDACIÓN ===
    
    def validate_protocol(
        self,
        culture: Culture,
        protocol: Dict
    ) -> Dict:
        """
        Ejecuta un protocolo de validación sobre un cultivo.
        
        protocol = {
            'incubation': 10,         # Generaciones de evolución
            'measurement': 'eseries', # Qué medir
            'threshold': 0.7,         # Umbral de aceptación
            'replicates': 3,          # Repeticiones
        }
        
        NORMA DURA: El protocolo mide, no juzga.
        """
        results = []
        
        for rep in range(protocol.get('replicates', 1)):
            # Clonar cultivo para este replicado
            clone_culture = self.create_culture(
                culture.best_specimen.sequence if culture.best_specimen else '',
                population_size=culture.size,
                conditions=culture.conditions.copy()
            )
            
            # Estado inicial
            initial_state = clone_culture.best_specimen.state if clone_culture.best_specimen else None
            
            # Incubar (evolucionar)
            self.evolve_culture(
                clone_culture.id,
                generations=protocol.get('incubation', 10)
            )
            
            # Estado final
            final_state = clone_culture.best_specimen.state if clone_culture.best_specimen else None
            
            if initial_state and final_state:
                results.append({
                    'replicate': rep + 1,
                    'initial_stability': initial_state.stability,
                    'final_stability': final_state.stability,
                    'stability_delta': final_state.stability - initial_state.stability,
                    'initial_E3': initial_state.eseries.E3,
                    'final_E3': final_state.eseries.E3,
                    'E3_delta': final_state.eseries.E3 - initial_state.eseries.E3,
                    'passed': final_state.stability >= protocol.get('threshold', 0.7)
                })
        
        # Resumen
        passed_count = sum(1 for r in results if r['passed'])
        avg_delta = sum(r['stability_delta'] for r in results) / len(results) if results else 0
        
        return {
            'protocol': protocol,
            'replicates': results,
            'summary': {
                'passed': passed_count,
                'total': len(results),
                'pass_rate': passed_count / len(results) if results else 0,
                'avg_stability_delta': avg_delta,
                'conclusion': 'threshold_met' if passed_count > len(results) / 2 else 'threshold_not_met'
            }
        }
    
    # === MÉTODOS PRIVADOS ===
    
    def _calculate_fitness(self, state: StructuralState, target: str) -> float:
        """Calcula fitness según objetivo."""
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
    
    def _select(self, specimens: List[Specimen], conditions: Dict) -> List[Specimen]:
        """Selecciona especímenes para reproducción."""
        # Ordenar por fitness
        sorted_specimens = sorted(specimens, key=lambda s: s.fitness, reverse=True)
        
        # Seleccionar top 50%
        cutoff = max(2, len(sorted_specimens) // 2)
        return sorted_specimens[:cutoff]
    
    def _mutate_sequence(self, seq: str, rate: float) -> str:
        """Muta una secuencia con cierta probabilidad."""
        if not seq:
            return seq
        chars = list(seq)
        for i in range(len(chars)):
            if random.random() < rate:
                # Cambiar por caracter aleatorio
                if chars[i].isalpha():
                    chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
                elif chars[i].isdigit():
                    chars[i] = random.choice('0123456789')
        return ''.join(chars)
    
    def _mutate_point(self, seq: str) -> str:
        """Mutación en un punto aleatorio."""
        if not seq:
            return seq
        chars = list(seq)
        i = random.randint(0, len(chars) - 1)
        if chars[i].isalpha():
            chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
        return ''.join(chars)
    
    def _mutate_region(self, seq: str, region_size: int = 3) -> str:
        """Mutación en una región."""
        if len(seq) < region_size:
            return self._mutate_point(seq)
        start = random.randint(0, len(seq) - region_size)
        mutated = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(region_size))
        return seq[:start] + mutated + seq[start + region_size:]
    
    def _crossover(self, seq1: str, seq2: str) -> str:
        """Recombina dos secuencias."""
        if not seq1 or not seq2:
            return seq1 or seq2
        point = random.randint(1, min(len(seq1), len(seq2)) - 1)
        return seq1[:point] + seq2[point:]
    
    def _delete_region(self, seq: str, size: int = 2) -> str:
        """Elimina una región."""
        if len(seq) <= size:
            return seq
        start = random.randint(0, len(seq) - size)
        return seq[:start] + seq[start + size:]
    
    def _insert_region(self, seq: str, size: int = 2) -> str:
        """Inserta una región aleatoria."""
        pos = random.randint(0, len(seq))
        insert = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(size))
        return seq[:pos] + insert + seq[pos:]
    
    def _invert_region(self, seq: str) -> str:
        """Invierte una región."""
        if len(seq) < 4:
            return seq[::-1]
        start = random.randint(0, len(seq) - 4)
        end = start + random.randint(2, 4)
        return seq[:start] + seq[start:end][::-1] + seq[end:]
    
    def _duplicate_region(self, seq: str) -> str:
        """Duplica una región."""
        if len(seq) < 3:
            return seq + seq
        start = random.randint(0, len(seq) - 2)
        size = random.randint(1, min(3, len(seq) - start))
        region = seq[start:start + size]
        return seq[:start + size] + region + seq[start + size:]
    
    def _shuffle_sequence(self, seq: str) -> str:
        """Baraja la secuencia."""
        chars = list(seq)
        random.shuffle(chars)
        return ''.join(chars)


# Singleton
_instance = None

def get_genetic_lab() -> GeneticLab:
    global _instance
    if _instance is None:
        _instance = GeneticLab()
    return _instance
