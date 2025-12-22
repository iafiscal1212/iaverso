"""
NeoSynt Core - Resolución Simbólica

NORMA DURA:
- Genera alternativas estructurales
- Resuelve hacia estados objetivo
- No interpreta significado
- No juzga qué alternativa es "mejor" en sentido humano

Rol en Inferencia Activa: MODELO GENERATIVO
"¿Qué estados son posibles? ¿Qué predigo si aplico X?"
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random
import math

# Import local
from core.endolens import StructuralState, ESeries, Signature, get_endolens


@dataclass
class Resolution:
    """Resultado de resolución simbólica."""
    status: str           # 'resolved', 'open', 'blocked'
    stability_score: float
    operators_applied: List[str]
    trace: List[str]      # Historial de transformaciones
    final_state: Optional[StructuralState]
    alternatives: List['Alternative']


@dataclass
class Alternative:
    """Una alternativa estructural generada."""
    id: str
    state: StructuralState
    operator: str         # Qué operador la generó
    distance: float       # Distancia al estado original
    predicted_stability: float


@dataclass
class Prediction:
    """Predicción de lo que pasará si aplico una transformación."""
    operator: str
    expected_state: ESeries
    expected_stability: float
    confidence: float     # Qué tan seguro está el modelo


class NeoSyntCore:
    """
    Motor de resolución simbólica.
    
    Genera alternativas, predice resultados, explora espacio de estados.
    No sabe qué significa ningún estado.
    Solo transforma y compara.
    """
    
    # Operadores disponibles
    OPERATORS = {
        'T_equiv': 'Equivalencia gematrica',
        'T_norm': 'Normalización',
        'T_perturb': 'Perturbación',
        'T_reduce': 'Reducción de complejidad',
        'T_expand': 'Expansión de complejidad',
        'T_stabilize': 'Estabilización',
        'T_destabilize': 'Desestabilización',
    }
    
    def __init__(self):
        self.endolens = get_endolens()
        self.history: List[Resolution] = []
        
    def resolve(self, state: StructuralState, target: str = 'stable') -> Resolution:
        """
        Intenta resolver el estado hacia un objetivo.
        
        Targets:
        - 'stable': maximizar estabilidad
        - 'explore': maximizar diversidad
        - 'minimize_E3': reducir complejidad
        - 'maximize_E3': aumentar complejidad
        
        NORMA DURA: El target es estructural, no semántico.
        """
        trace = []
        operators_applied = []
        current = state
        alternatives = []
        
        # Generar alternativas
        for op_name in ['T_stabilize', 'T_perturb', 'T_reduce']:
            alt_state = self._apply_operator(current, op_name)
            if alt_state:
                distance = self._calculate_distance(state, alt_state)
                alternatives.append(Alternative(
                    id=f"alt_{op_name}_{len(alternatives)}",
                    state=alt_state,
                    operator=op_name,
                    distance=distance,
                    predicted_stability=alt_state.stability
                ))
                trace.append(f"Generated alternative via {op_name}")
        
        # Seleccionar mejor según target (sin juicio semántico)
        if target == 'stable':
            alternatives.sort(key=lambda a: a.predicted_stability, reverse=True)
        elif target == 'explore':
            alternatives.sort(key=lambda a: a.distance, reverse=True)
        elif target == 'minimize_E3':
            alternatives.sort(key=lambda a: a.state.eseries.E3)
        elif target == 'maximize_E3':
            alternatives.sort(key=lambda a: a.state.eseries.E3, reverse=True)
        
        # Determinar status
        if alternatives and alternatives[0].predicted_stability >= 0.7:
            status = 'resolved'
            final_state = alternatives[0].state
            operators_applied.append(alternatives[0].operator)
        elif alternatives:
            status = 'open'
            final_state = alternatives[0].state
        else:
            status = 'blocked'
            final_state = None
        
        resolution = Resolution(
            status=status,
            stability_score=final_state.stability if final_state else state.stability,
            operators_applied=operators_applied,
            trace=trace,
            final_state=final_state,
            alternatives=alternatives
        )
        
        self.history.append(resolution)
        return resolution
    
    def predict(self, state: StructuralState, operator: str) -> Prediction:
        """
        Predice qué pasará si aplico un operador.
        
        NORMA DURA: Solo predice números, no consecuencias.
        """
        # Aplicar operador para ver resultado
        result = self._apply_operator(state, operator)
        
        if result:
            return Prediction(
                operator=operator,
                expected_state=result.eseries,
                expected_stability=result.stability,
                confidence=0.8  # Modelo tiene 80% confianza en sus predicciones
            )
        else:
            return Prediction(
                operator=operator,
                expected_state=state.eseries,
                expected_stability=state.stability,
                confidence=0.2  # Baja confianza si no pudo aplicar
            )
    
    def generate_alternatives(self, state: StructuralState, n: int = 5) -> List[Alternative]:
        """
        Genera N alternativas estructurales.
        
        NORMA DURA: No juzga cuál es "mejor", solo genera diversidad.
        """
        alternatives = []
        operators = list(self.OPERATORS.keys())
        
        for i in range(n):
            op = random.choice(operators)
            alt_state = self._apply_operator(state, op)
            if alt_state:
                alternatives.append(Alternative(
                    id=f"gen_{i}",
                    state=alt_state,
                    operator=op,
                    distance=self._calculate_distance(state, alt_state),
                    predicted_stability=alt_state.stability
                ))
        
        return alternatives
    
    def _apply_operator(self, state: StructuralState, operator: str) -> Optional[StructuralState]:
        """Aplica un operador de transformación."""
        e = state.eseries
        
        if operator == 'T_stabilize':
            # Reduce variabilidad
            new_E1 = max(0, e.E1 * 0.7)
            new_E2 = max(0, e.E2 * 0.8)
        elif operator == 'T_destabilize':
            # Aumenta variabilidad
            new_E1 = min(1, e.E1 * 1.3)
            new_E2 = min(1, e.E2 * 1.2)
        elif operator == 'T_reduce':
            # Reduce complejidad
            new_E3 = max(0, e.E3 * 0.6)
            new_E1 = e.E1
            new_E2 = e.E2
        elif operator == 'T_expand':
            # Aumenta complejidad
            new_E3 = min(1, e.E3 * 1.4)
            new_E1 = e.E1
            new_E2 = e.E2
        elif operator == 'T_perturb':
            # Perturbación aleatoria pequeña
            new_E1 = max(0, min(1, e.E1 + random.uniform(-0.1, 0.1)))
            new_E2 = max(0, min(1, e.E2 + random.uniform(-0.1, 0.1)))
            new_E3 = e.E3
        elif operator == 'T_equiv':
            # Mantiene equivalencia (no cambia E-series)
            return state
        elif operator == 'T_norm':
            # Normaliza hacia valores medios
            new_E1 = (e.E1 + 0.5) / 2
            new_E2 = (e.E2 + 0.5) / 2
            new_E3 = e.E3
        else:
            return None
        
        # Crear nuevo ESeries
        new_eseries = ESeries(
            E0=e.E0,  # E0 generalmente se preserva
            E1=round(new_E1 if 'new_E1' in dir() else e.E1, 3),
            E2=round(new_E2 if 'new_E2' in dir() else e.E2, 3),
            E3=round(new_E3 if 'new_E3' in dir() else e.E3, 3)
        )
        
        # Generar nueva firma
        new_stability = new_eseries.stability_score()
        new_signature = Signature(
            attractor=state.signature.attractor,  # Attractor se preserva
            energy=int(new_eseries.E1 * 9),
            drift=int(new_eseries.E3 * 100),
            raw=f"⟨A{state.signature.attractor}·E{int(new_eseries.E1 * 9)}·Δ{int(new_eseries.E3 * 100)}⟩"
        )
        
        # Determinar status
        if new_stability >= 0.7:
            new_status = 'stable'
        elif new_stability >= 0.4:
            new_status = 'dynamic'
        else:
            new_status = 'unstable'
        
        return StructuralState(
            signature=new_signature,
            eseries=new_eseries,
            stability=new_stability,
            status=new_status,
            invariants=state.invariants,  # Invariantes se preservan
            tensions=[]  # Recalcular tensiones sería redundante aquí
        )
    
    def _calculate_distance(self, s1: StructuralState, s2: StructuralState) -> float:
        """Distancia euclidiana en espacio E-series."""
        e1, e2 = s1.eseries, s2.eseries
        return math.sqrt(
            (e1.E0 - e2.E0)**2 +
            (e1.E1 - e2.E1)**2 +
            (e1.E2 - e2.E2)**2 +
            (e1.E3 - e2.E3)**2
        )
    
    def calculate_surprise(self, predicted: Prediction, observed: StructuralState) -> float:
        """
        Calcula sorpresa (Free Energy) entre predicción y observación.
        
        NORMA DURA: Solo un número. Menor = modelo acertó.
        """
        pred_e = predicted.expected_state
        obs_e = observed.eseries
        
        # Distancia entre predicción y observación
        distance = math.sqrt(
            (pred_e.E0 - obs_e.E0)**2 +
            (pred_e.E1 - obs_e.E1)**2 +
            (pred_e.E2 - obs_e.E2)**2 +
            (pred_e.E3 - obs_e.E3)**2
        )
        
        # Sorpresa ponderada por confianza
        surprise = distance * (1 - predicted.confidence)
        return round(surprise, 4)


# Singleton
_instance = None

def get_neosynt() -> NeoSyntCore:
    global _instance
    if _instance is None:
        _instance = NeoSyntCore()
    return _instance
