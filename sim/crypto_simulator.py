"""
Crypto Simulator - Simulador de dinámicas cripto in silico

⚠️ IMPORTANTE:
- NO es trading
- NO es inversión
- NO ejecuta órdenes reales
- NO interactúa con exchanges, wallets ni dinero
- Es SIMULACIÓN ESTRUCTURAL in silico

NORMA DURA:
- Opera sobre ESTRUCTURA, no semántica
- Poblaciones abstractas de estados de mercado
- Operadores como perturbaciones estructurales
- Sin señales, recomendaciones ni predicciones de precio

Ontología abstracta:
- Población = conjunto de estados de mercado abstractos
- Estado = vector (liquidez, volatilidad, concentración, latencia, etc.)
- Operador = evento estructural / perturbación
- Escenario = simulación in silico
- Estabilidad (E0-E3) = coherencia / robustez estructural
- Sorpresa = divergencia modelo-resultado

Los operadores NO representan acciones humanas.
Representan PERTURBACIONES del sistema.
"""

import random
import math
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import sys
sys.path.insert(0, '/opt/iaverso')

from sim.base_simulator import (
    BaseSimulator, Element, Population, Scenario,
    ScenarioResult, OperatorSpec
)
from core.endolens import StructuralState


@dataclass
class MarketState:
    """
    Estado abstracto de mercado.

    NO representa precio ni valor monetario.
    Es un vector de propiedades estructurales.
    """
    liquidity: float        # 0-1: Disponibilidad de flujo
    volatility: float       # 0-1: Variabilidad estructural
    concentration: float    # 0-1: Distribución de nodos
    latency: float          # 0-1: Tiempo de propagación
    sentiment: float        # 0-1: Momentum agregado (no opinión)
    leverage: float         # 0-1: Apalancamiento sistémico
    network_load: float     # 0-1: Carga de red
    regulatory_pressure: float  # 0-1: Presión regulatoria

    def to_vector(self) -> List[float]:
        return [
            self.liquidity,
            self.volatility,
            self.concentration,
            self.latency,
            self.sentiment,
            self.leverage,
            self.network_load,
            self.regulatory_pressure
        ]

    def to_dict(self) -> Dict[str, float]:
        return {
            'liquidity': self.liquidity,
            'volatility': self.volatility,
            'concentration': self.concentration,
            'latency': self.latency,
            'sentiment': self.sentiment,
            'leverage': self.leverage,
            'network_load': self.network_load,
            'regulatory_pressure': self.regulatory_pressure
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'MarketState':
        # Convertir valores a float, ignorando campos no numéricos
        def safe_float(val, default=0.5):
            if isinstance(val, (int, float)):
                return float(val)
            return default

        return cls(
            liquidity=safe_float(d.get('liquidity'), 0.5),
            volatility=safe_float(d.get('volatility'), 0.5),
            concentration=safe_float(d.get('concentration'), 0.5),
            latency=safe_float(d.get('latency'), 0.5),
            sentiment=safe_float(d.get('sentiment'), 0.5),
            leverage=safe_float(d.get('leverage'), 0.5),
            network_load=safe_float(d.get('network_load'), 0.5),
            regulatory_pressure=safe_float(d.get('regulatory_pressure'), 0.5)
        )

    @classmethod
    def random(cls, base: Optional['MarketState'] = None, variance: float = 0.2) -> 'MarketState':
        """Genera un estado aleatorio, opcionalmente basado en otro."""
        if base:
            return cls(
                liquidity=max(0, min(1, base.liquidity + random.gauss(0, variance))),
                volatility=max(0, min(1, base.volatility + random.gauss(0, variance))),
                concentration=max(0, min(1, base.concentration + random.gauss(0, variance))),
                latency=max(0, min(1, base.latency + random.gauss(0, variance))),
                sentiment=max(0, min(1, base.sentiment + random.gauss(0, variance))),
                leverage=max(0, min(1, base.leverage + random.gauss(0, variance))),
                network_load=max(0, min(1, base.network_load + random.gauss(0, variance))),
                regulatory_pressure=max(0, min(1, base.regulatory_pressure + random.gauss(0, variance)))
            )
        else:
            return cls(
                liquidity=random.random(),
                volatility=random.random(),
                concentration=random.random(),
                latency=random.random(),
                sentiment=random.random(),
                leverage=random.random(),
                network_load=random.random(),
                regulatory_pressure=random.random()
            )


class CryptoSimulator(BaseSimulator):
    """
    Simulador de dinámicas cripto in silico.

    ⚠️ Solo simulación estructural. Sin trading real.

    Explora:
    - Fragilidad estructural
    - Robustez ante perturbaciones
    - Propagación de shocks
    - Coherencia sistémica

    NO hace:
    - Predicción de precios
    - Señales de compra/venta
    - Recomendaciones de inversión
    - Conexión a exchanges
    """

    def __init__(self):
        super().__init__(domain='crypto')

    def _register_operators(self):
        """Registra operadores estructurales cripto."""

        # O_volatility_spike: Pico de volatilidad
        self.register_operator(
            OperatorSpec(
                id='O_volatility_spike',
                name='Pico de volatilidad',
                description='Perturbación que incrementa variabilidad estructural',
                parameters=['intensity'],
                domain='crypto'
            ),
            self._op_volatility_spike
        )

        # O_liquidity_drain: Drenaje de liquidez
        self.register_operator(
            OperatorSpec(
                id='O_liquidity_drain',
                name='Drenaje de liquidez',
                description='Reducción de disponibilidad de flujo',
                parameters=['severity'],
                domain='crypto'
            ),
            self._op_liquidity_drain
        )

        # O_regulatory_shock: Shock regulatorio
        self.register_operator(
            OperatorSpec(
                id='O_regulatory_shock',
                name='Shock regulatorio',
                description='Incremento súbito de presión regulatoria',
                parameters=['magnitude'],
                domain='crypto'
            ),
            self._op_regulatory_shock
        )

        # O_whale_pressure: Presión de grandes actores
        self.register_operator(
            OperatorSpec(
                id='O_whale_pressure',
                name='Presión de concentración',
                description='Aumento de concentración en pocos nodos',
                parameters=['concentration_delta'],
                domain='crypto'
            ),
            self._op_whale_pressure
        )

        # O_network_congestion: Congestión de red
        self.register_operator(
            OperatorSpec(
                id='O_network_congestion',
                name='Congestión de red',
                description='Incremento de latencia y carga',
                parameters=['load_factor'],
                domain='crypto'
            ),
            self._op_network_congestion
        )

        # O_sentiment_shift: Cambio de momentum
        self.register_operator(
            OperatorSpec(
                id='O_sentiment_shift',
                name='Cambio de momentum',
                description='Perturbación del momentum agregado',
                parameters=['direction', 'magnitude'],
                domain='crypto'
            ),
            self._op_sentiment_shift
        )

        # O_leverage_unwind: Desapalancamiento
        self.register_operator(
            OperatorSpec(
                id='O_leverage_unwind',
                name='Desapalancamiento',
                description='Reducción forzada de apalancamiento sistémico',
                parameters=['speed'],
                domain='crypto'
            ),
            self._op_leverage_unwind
        )

        # O_cascade: Efecto cascada
        self.register_operator(
            OperatorSpec(
                id='O_cascade',
                name='Efecto cascada',
                description='Propagación de perturbación a múltiples variables',
                parameters=['trigger', 'propagation'],
                domain='crypto'
            ),
            self._op_cascade
        )

    # ==========================================================================
    # IMPLEMENTACIÓN DE MÉTODOS ABSTRACTOS
    # ==========================================================================

    def _generate_element(self, seed: Any, params: Dict) -> Element:
        """Genera un elemento (estado de mercado abstracto)."""
        # Parsear seed si es texto
        base_state = None

        # Verificar si hay un base_state de datos reales (Binance)
        if 'base_state' in params and params['base_state']:
            # Priorizar datos reales
            base_data = params['base_state']
            if isinstance(base_data, dict):
                base_state = MarketState.from_dict(base_data)

        elif isinstance(seed, str):
            # Extraer características del texto
            seed_lower = seed.lower()

            # Detectar condiciones mencionadas
            base_values = {
                'liquidity': 0.5,
                'volatility': 0.5,
                'concentration': 0.5,
                'latency': 0.5,
                'sentiment': 0.5,
                'leverage': 0.5,
                'network_load': 0.5,
                'regulatory_pressure': 0.5
            }

            # Ajustar según palabras clave
            if 'volatil' in seed_lower or 'inestab' in seed_lower:
                base_values['volatility'] = 0.8
            if 'líquid' in seed_lower or 'liquid' in seed_lower:
                base_values['liquidity'] = 0.7
            if 'regulat' in seed_lower:
                base_values['regulatory_pressure'] = 0.7
            if 'concentra' in seed_lower or 'whale' in seed_lower:
                base_values['concentration'] = 0.7
            if 'apalanca' in seed_lower or 'leverag' in seed_lower:
                base_values['leverage'] = 0.7
            if 'congest' in seed_lower or 'red' in seed_lower:
                base_values['network_load'] = 0.7

            base_state = MarketState.from_dict(base_values)

        elif isinstance(seed, dict):
            base_state = MarketState.from_dict(seed)

        # Generar estado con varianza
        variance = params.get('variance', 0.15)
        market_state = MarketState.random(base_state, variance)

        # Marcar si viene de datos reales
        source = params.get('source', 'simulated')

        element = Element(
            id=str(uuid.uuid4())[:8],
            data=market_state.to_dict(),
            state=None,  # Se calcula después
            generation=0
        )

        # Añadir metadatos de origen
        element.data['_source'] = source

        return element

    def _calculate_fitness(self, element: Element) -> float:
        """
        Calcula fitness estructural.

        NO es "mejor para invertir".
        Es: coherencia + robustez + estabilidad estructural.
        """
        state = element.state
        if not state:
            return 0.0

        # Fitness basado en estabilidad estructural
        stability_score = state.stability

        # Penalizar extremos (fragilidad)
        data = element.data
        extreme_penalty = 0.0
        for key, val in data.items():
            # Ignorar campos de metadatos
            if key.startswith('_') or not isinstance(val, (int, float)):
                continue
            if val > 0.9 or val < 0.1:
                extreme_penalty += 0.05

        # Penalizar combinaciones frágiles
        fragility_penalty = 0.0
        if data.get('volatility', 0) > 0.7 and data.get('leverage', 0) > 0.7:
            fragility_penalty += 0.2  # Alta volatilidad + alto apalancamiento
        if data.get('liquidity', 0) < 0.3 and data.get('concentration', 0) > 0.7:
            fragility_penalty += 0.15  # Baja liquidez + alta concentración

        fitness = max(0, stability_score - extreme_penalty - fragility_penalty)
        return fitness

    def _serialize_element(self, element: Element) -> str:
        """Serializa estado de mercado para EndoLens."""
        data = element.data
        # Crear representación textual estructurada
        parts = []
        for key, val in sorted(data.items()):
            # Ignorar campos de metadatos
            if key.startswith('_') or not isinstance(val, (int, float)):
                continue
            level = 'low' if val < 0.33 else ('mid' if val < 0.66 else 'high')
            parts.append(f"{key}:{level}:{val:.2f}")
        return '|'.join(parts)

    # ==========================================================================
    # OPERADORES ESTRUCTURALES
    # ==========================================================================

    def _op_volatility_spike(self, element: Element, params: Dict) -> Element:
        """Pico de volatilidad."""
        intensity = params.get('intensity', 0.3)
        data = element.data.copy()

        data['volatility'] = min(1.0, data['volatility'] + intensity)
        # Efectos secundarios
        data['sentiment'] = max(0, data['sentiment'] - intensity * 0.3)
        data['liquidity'] = max(0, data['liquidity'] - intensity * 0.2)

        return Element(
            id=element.id,
            data=data,
            state=element.state,
            generation=element.generation,
            parent_id=element.id,
            operators_applied=element.operators_applied.copy()
        )

    def _op_liquidity_drain(self, element: Element, params: Dict) -> Element:
        """Drenaje de liquidez."""
        severity = params.get('severity', 0.4)
        data = element.data.copy()

        data['liquidity'] = max(0, data['liquidity'] - severity)
        # Efectos secundarios
        data['volatility'] = min(1.0, data['volatility'] + severity * 0.4)
        data['latency'] = min(1.0, data['latency'] + severity * 0.2)

        return Element(
            id=element.id,
            data=data,
            state=element.state,
            generation=element.generation,
            parent_id=element.id,
            operators_applied=element.operators_applied.copy()
        )

    def _op_regulatory_shock(self, element: Element, params: Dict) -> Element:
        """Shock regulatorio."""
        magnitude = params.get('magnitude', 0.5)
        data = element.data.copy()

        data['regulatory_pressure'] = min(1.0, data['regulatory_pressure'] + magnitude)
        # Efectos secundarios
        data['sentiment'] = max(0, data['sentiment'] - magnitude * 0.4)
        data['leverage'] = max(0, data['leverage'] - magnitude * 0.3)
        data['volatility'] = min(1.0, data['volatility'] + magnitude * 0.3)

        return Element(
            id=element.id,
            data=data,
            state=element.state,
            generation=element.generation,
            parent_id=element.id,
            operators_applied=element.operators_applied.copy()
        )

    def _op_whale_pressure(self, element: Element, params: Dict) -> Element:
        """Presión de concentración."""
        delta = params.get('concentration_delta', 0.3)
        data = element.data.copy()

        data['concentration'] = min(1.0, data['concentration'] + delta)
        # Efectos secundarios
        data['volatility'] = min(1.0, data['volatility'] + delta * 0.2)

        return Element(
            id=element.id,
            data=data,
            state=element.state,
            generation=element.generation,
            parent_id=element.id,
            operators_applied=element.operators_applied.copy()
        )

    def _op_network_congestion(self, element: Element, params: Dict) -> Element:
        """Congestión de red."""
        load_factor = params.get('load_factor', 0.4)
        data = element.data.copy()

        data['network_load'] = min(1.0, data['network_load'] + load_factor)
        data['latency'] = min(1.0, data['latency'] + load_factor * 0.5)
        # Efectos secundarios
        data['liquidity'] = max(0, data['liquidity'] - load_factor * 0.2)

        return Element(
            id=element.id,
            data=data,
            state=element.state,
            generation=element.generation,
            parent_id=element.id,
            operators_applied=element.operators_applied.copy()
        )

    def _op_sentiment_shift(self, element: Element, params: Dict) -> Element:
        """Cambio de momentum."""
        direction = params.get('direction', 1)  # 1 o -1
        magnitude = params.get('magnitude', 0.3)
        data = element.data.copy()

        shift = direction * magnitude
        data['sentiment'] = max(0, min(1.0, data['sentiment'] + shift))
        # Efectos secundarios
        if shift < 0:
            data['volatility'] = min(1.0, data['volatility'] + abs(shift) * 0.3)
            data['leverage'] = max(0, data['leverage'] - abs(shift) * 0.2)

        return Element(
            id=element.id,
            data=data,
            state=element.state,
            generation=element.generation,
            parent_id=element.id,
            operators_applied=element.operators_applied.copy()
        )

    def _op_leverage_unwind(self, element: Element, params: Dict) -> Element:
        """Desapalancamiento."""
        speed = params.get('speed', 0.5)
        data = element.data.copy()

        data['leverage'] = max(0, data['leverage'] - speed)
        # Efectos secundarios (desapalancamiento rápido causa caos)
        if speed > 0.3:
            data['volatility'] = min(1.0, data['volatility'] + speed * 0.5)
            data['liquidity'] = max(0, data['liquidity'] - speed * 0.3)
            data['sentiment'] = max(0, data['sentiment'] - speed * 0.2)

        return Element(
            id=element.id,
            data=data,
            state=element.state,
            generation=element.generation,
            parent_id=element.id,
            operators_applied=element.operators_applied.copy()
        )

    def _op_cascade(self, element: Element, params: Dict) -> Element:
        """Efecto cascada - propagación de perturbación."""
        trigger = params.get('trigger', 'volatility')
        propagation = params.get('propagation', 0.5)
        data = element.data.copy()

        # Incrementar trigger
        if trigger in data:
            data[trigger] = min(1.0, data[trigger] + propagation)

        # Propagar a variables relacionadas
        cascade_map = {
            'volatility': ['sentiment', 'liquidity', 'leverage'],
            'liquidity': ['volatility', 'latency', 'concentration'],
            'regulatory_pressure': ['sentiment', 'leverage', 'volatility'],
            'leverage': ['volatility', 'liquidity', 'sentiment'],
            'sentiment': ['volatility', 'leverage'],
            'concentration': ['liquidity', 'volatility'],
            'network_load': ['latency', 'liquidity'],
            'latency': ['network_load', 'liquidity']
        }

        affected = cascade_map.get(trigger, [])
        for var in affected:
            if var in data:
                # Propagar con atenuación
                effect = propagation * 0.4 * random.uniform(0.5, 1.0)
                if trigger in ['regulatory_pressure', 'volatility', 'leverage']:
                    # Estos tienden a empeorar otras variables
                    data[var] = max(0, min(1.0, data[var] + effect * (-1 if var in ['sentiment', 'liquidity'] else 1)))
                else:
                    data[var] = max(0, min(1.0, data[var] + effect))

        return Element(
            id=element.id,
            data=data,
            state=element.state,
            generation=element.generation,
            parent_id=element.id,
            operators_applied=element.operators_applied.copy()
        )

    # ==========================================================================
    # MÉTODOS ESPECÍFICOS CRYPTO
    # ==========================================================================

    def analyze_fragility(self, population: Population) -> Dict:
        """
        Analiza fragilidad estructural de la población.

        NO es predicción. Es análisis de robustez.
        """
        if not population.elements:
            return {'fragility': 0.0, 'factors': []}

        fragility_factors = []
        total_fragility = 0.0

        for element in population.elements:
            data = element.data
            element_fragility = 0.0

            # Detectar combinaciones frágiles
            if data.get('volatility', 0) > 0.7 and data.get('leverage', 0) > 0.6:
                fragility_factors.append('alta_volatilidad_con_apalancamiento')
                element_fragility += 0.3

            if data.get('liquidity', 0) < 0.3:
                fragility_factors.append('baja_liquidez')
                element_fragility += 0.2

            if data.get('concentration', 0) > 0.7:
                fragility_factors.append('alta_concentracion')
                element_fragility += 0.15

            if data.get('regulatory_pressure', 0) > 0.7:
                fragility_factors.append('alta_presion_regulatoria')
                element_fragility += 0.2

            if data.get('network_load', 0) > 0.8:
                fragility_factors.append('congestion_red')
                element_fragility += 0.1

            total_fragility += min(1.0, element_fragility)

        avg_fragility = total_fragility / len(population.elements)

        return {
            'fragility': avg_fragility,
            'factors': list(set(fragility_factors)),
            'population_size': len(population.elements),
            'note': 'Análisis estructural, no predicción de comportamiento'
        }

    def stress_test(
        self,
        population: Population,
        operators: List[str],
        iterations: int = 5
    ) -> Dict:
        """
        Ejecuta test de estrés estructural.

        Aplica secuencia de operadores y mide degradación.
        """
        initial_fitness = population.avg_fitness
        initial_signature = population.signature

        results = []

        for op in operators:
            if op in self._operators:
                for _ in range(iterations):
                    self.apply_operator(population.id, op, iterations=1)

                results.append({
                    'operator': op,
                    'iterations': iterations,
                    'fitness_after': population.avg_fitness,
                    'signature_after': population.signature
                })

        final_fitness = population.avg_fitness
        degradation = initial_fitness - final_fitness

        return {
            'initial_fitness': initial_fitness,
            'final_fitness': final_fitness,
            'degradation': degradation,
            'degradation_percent': (degradation / initial_fitness * 100) if initial_fitness > 0 else 0,
            'initial_signature': initial_signature,
            'final_signature': population.signature,
            'steps': results,
            'note': 'Test de robustez estructural, no predicción'
        }


# Singleton
_crypto_sim_instance = None

def get_crypto_simulator() -> CryptoSimulator:
    global _crypto_sim_instance
    if _crypto_sim_instance is None:
        _crypto_sim_instance = CryptoSimulator()
    return _crypto_sim_instance
