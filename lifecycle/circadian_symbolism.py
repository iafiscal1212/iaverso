"""
Circadian Symbolism: Simbolismo Circadiano
==========================================

Los simbolos cambian de rol segun la fase circadiana:
    - WAKE:    Simbolos operativos (accion, meta, recurso)
    - REST:    Simbolos evaluativos (valor, juicio, balance)
    - DREAM:   Simbolos oniricos (asociacion, metafora, arquetipo)
    - LIMINAL: Simbolos transitorios (puente, umbral, transformacion)

Esto permite medir la vida simbolica de cada agente
y como sus simbolos evolucionan a lo largo del ciclo.

100% endogeno. Los simbolos emergen de la experiencia.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum

import sys
sys.path.insert(0, '/root/NEO_EVA')

from lifecycle.circadian_system import CircadianPhase
from cognition.agi_dynamic_constants import L_t, max_history


class SymbolType(Enum):
    """Tipos de simbolos segun fase."""
    # WAKE - Operativos
    ACTION = "action"           # Accion a realizar
    GOAL = "goal"               # Meta a alcanzar
    RESOURCE = "resource"       # Recurso disponible
    OBSTACLE = "obstacle"       # Obstaculo a superar

    # REST - Evaluativos
    VALUE = "value"             # Valor/importancia
    JUDGMENT = "judgment"       # Juicio/evaluacion
    BALANCE = "balance"         # Equilibrio/proporcion
    CLOSURE = "closure"         # Cierre/conclusion

    # DREAM - Oniricos
    ASSOCIATION = "association"  # Asociacion libre
    METAPHOR = "metaphor"       # Metafora
    ARCHETYPE = "archetype"     # Arquetipo profundo
    FRAGMENT = "fragment"       # Fragmento inconexo

    # LIMINAL - Transitorios
    BRIDGE = "bridge"           # Puente entre estados
    THRESHOLD = "threshold"     # Umbral/limite
    TRANSFORMATION = "transformation"  # Transformacion
    EMERGENCE = "emergence"     # Emergencia de algo nuevo


# Mapeo fase -> tipos de simbolos activos
PHASE_SYMBOL_TYPES = {
    CircadianPhase.WAKE: [
        SymbolType.ACTION, SymbolType.GOAL,
        SymbolType.RESOURCE, SymbolType.OBSTACLE
    ],
    CircadianPhase.REST: [
        SymbolType.VALUE, SymbolType.JUDGMENT,
        SymbolType.BALANCE, SymbolType.CLOSURE
    ],
    CircadianPhase.DREAM: [
        SymbolType.ASSOCIATION, SymbolType.METAPHOR,
        SymbolType.ARCHETYPE, SymbolType.FRAGMENT
    ],
    CircadianPhase.LIMINAL: [
        SymbolType.BRIDGE, SymbolType.THRESHOLD,
        SymbolType.TRANSFORMATION, SymbolType.EMERGENCE
    ],
}


@dataclass
class CircadianSymbol:
    """Un simbolo con contexto circadiano."""
    id: str
    content: str                # Contenido simbolico
    symbol_type: SymbolType
    phase_origin: CircadianPhase  # Fase donde se creo
    strength: float             # Fuerza/activacion [0, 1]
    valence: float              # Valencia emocional [-1, 1]
    connections: List[str]      # IDs de simbolos conectados
    created_t: int
    last_activated_t: int
    activation_count: int


@dataclass
class SymbolicDream:
    """Sueno simbolico generado durante DREAM."""
    symbols: List[CircadianSymbol]
    narrative: str
    emotional_tone: float
    coherence: float            # Que tan coherente es el sueno
    insights: List[str]         # Insights emergentes


@dataclass
class SymbolicTransition:
    """Transicion simbolica durante LIMINAL."""
    from_symbols: List[str]     # Simbolos de origen
    to_symbols: List[str]       # Simbolos emergentes
    transformation_type: str
    significance: float


class CircadianSymbolism:
    """
    Sistema de simbolismo circadiano.

    Gestiona la vida simbolica de un agente a lo largo
    del ciclo circadiano, con diferentes tipos de
    simbolos activos en cada fase.
    """

    def __init__(self, agent_id: str):
        """
        Inicializa sistema de simbolismo circadiano.

        Args:
            agent_id: ID del agente
        """
        self.agent_id = agent_id

        # Repositorio de simbolos
        self.symbols: Dict[str, CircadianSymbol] = {}

        # Simbolos activos por fase
        self.active_symbols: Dict[CircadianPhase, Set[str]] = {
            phase: set() for phase in CircadianPhase
        }

        # Fase actual
        self.current_phase = CircadianPhase.WAKE

        # Historiales
        self.dream_history: List[SymbolicDream] = []
        self.transition_history: List[SymbolicTransition] = []

        # Arquetipos emergentes (patrones simbolicos recurrentes)
        self.archetypes: Dict[str, float] = {}

        # Contadores
        self._symbol_counter = 0
        self.t = 0

    def _generate_symbol_id(self) -> str:
        """Genera ID unico para simbolo."""
        self._symbol_counter += 1
        return f"sym_{self.agent_id}_{self._symbol_counter}"

    def create_symbol(
        self,
        content: str,
        symbol_type: SymbolType = None,
        valence: float = 0.0,
        strength: float = 0.5
    ) -> CircadianSymbol:
        """
        Crea un nuevo simbolo.

        El tipo se infiere de la fase actual si no se especifica.

        Args:
            content: Contenido del simbolo
            symbol_type: Tipo (o None para inferir)
            valence: Valencia emocional
            strength: Fuerza inicial

        Returns:
            Simbolo creado
        """
        # Inferir tipo de la fase si no se especifica
        if symbol_type is None:
            valid_types = PHASE_SYMBOL_TYPES[self.current_phase]
            symbol_type = np.random.choice(valid_types)

        symbol = CircadianSymbol(
            id=self._generate_symbol_id(),
            content=content,
            symbol_type=symbol_type,
            phase_origin=self.current_phase,
            strength=strength,
            valence=valence,
            connections=[],
            created_t=self.t,
            last_activated_t=self.t,
            activation_count=1
        )

        self.symbols[symbol.id] = symbol
        self.active_symbols[self.current_phase].add(symbol.id)

        return symbol

    def activate_symbol(self, symbol_id: str, strength_boost: float = 0.1):
        """
        Activa un simbolo existente.

        Args:
            symbol_id: ID del simbolo
            strength_boost: Incremento de fuerza
        """
        if symbol_id not in self.symbols:
            return

        symbol = self.symbols[symbol_id]
        symbol.strength = min(1.0, symbol.strength + strength_boost)
        symbol.last_activated_t = self.t
        symbol.activation_count += 1

        # Agregar a activos de fase actual
        self.active_symbols[self.current_phase].add(symbol_id)

    def connect_symbols(self, id1: str, id2: str):
        """
        Conecta dos simbolos.

        Args:
            id1, id2: IDs de simbolos a conectar
        """
        if id1 not in self.symbols or id2 not in self.symbols:
            return

        if id2 not in self.symbols[id1].connections:
            self.symbols[id1].connections.append(id2)
        if id1 not in self.symbols[id2].connections:
            self.symbols[id2].connections.append(id1)

    def get_phase_symbols(
        self,
        phase: CircadianPhase = None
    ) -> List[CircadianSymbol]:
        """
        Obtiene simbolos activos en una fase.

        Args:
            phase: Fase (o actual si None)

        Returns:
            Lista de simbolos
        """
        phase = phase or self.current_phase
        return [
            self.symbols[sid]
            for sid in self.active_symbols[phase]
            if sid in self.symbols
        ]

    def get_symbols_by_type(
        self,
        symbol_type: SymbolType
    ) -> List[CircadianSymbol]:
        """Obtiene simbolos de un tipo especifico."""
        return [
            sym for sym in self.symbols.values()
            if sym.symbol_type == symbol_type
        ]

    def _decay_symbols(self):
        """Aplica decay a simbolos no activados."""
        decay_rate = 0.01

        for symbol in self.symbols.values():
            # Decay basado en tiempo desde ultima activacion
            time_since = self.t - symbol.last_activated_t
            if time_since > 10:
                symbol.strength *= (1 - decay_rate)

        # Eliminar simbolos muy debiles
        to_remove = [
            sid for sid, sym in self.symbols.items()
            if sym.strength < 0.05
        ]
        for sid in to_remove:
            self._remove_symbol(sid)

    def _remove_symbol(self, symbol_id: str):
        """Elimina un simbolo."""
        if symbol_id in self.symbols:
            # Quitar de activos
            for phase in CircadianPhase:
                self.active_symbols[phase].discard(symbol_id)

            # Quitar conexiones
            sym = self.symbols[symbol_id]
            for connected_id in sym.connections:
                if connected_id in self.symbols:
                    self.symbols[connected_id].connections.remove(symbol_id)

            del self.symbols[symbol_id]

    def generate_dream_symbols(self) -> SymbolicDream:
        """
        Genera simbolos oniricos durante fase DREAM.

        Los suenos crean asociaciones entre simbolos
        de diferentes fases y generan metaforas.

        Returns:
            Sueno simbolico
        """
        dream_symbols = []

        # Recolectar simbolos de todas las fases para asociar
        all_symbols = list(self.symbols.values())
        if not all_symbols:
            return SymbolicDream(
                symbols=[],
                narrative="sueno vacio",
                emotional_tone=0.0,
                coherence=0.0,
                insights=[]
            )

        # Seleccionar simbolos base para el sueno
        n_base = min(5, len(all_symbols))
        base_symbols = np.random.choice(
            all_symbols, n_base, replace=False
        ).tolist()

        # Crear asociaciones oniricas
        for i, sym1 in enumerate(base_symbols):
            # Asociacion
            assoc = self.create_symbol(
                content=f"asociacion:{sym1.content}",
                symbol_type=SymbolType.ASSOCIATION,
                valence=sym1.valence,
                strength=sym1.strength * 0.7
            )
            dream_symbols.append(assoc)
            self.connect_symbols(sym1.id, assoc.id)

            # Metafora (combinar dos simbolos)
            if i < len(base_symbols) - 1:
                sym2 = base_symbols[i + 1]
                metaphor = self.create_symbol(
                    content=f"metafora:{sym1.content}+{sym2.content}",
                    symbol_type=SymbolType.METAPHOR,
                    valence=(sym1.valence + sym2.valence) / 2,
                    strength=(sym1.strength + sym2.strength) / 2
                )
                dream_symbols.append(metaphor)
                self.connect_symbols(sym1.id, metaphor.id)
                self.connect_symbols(sym2.id, metaphor.id)

        # Detectar arquetipos
        self._detect_archetypes(base_symbols)

        # Crear simbolos arquetipicos si hay patrones fuertes
        for archetype, strength in self.archetypes.items():
            if strength > 0.5:
                arch_sym = self.create_symbol(
                    content=f"arquetipo:{archetype}",
                    symbol_type=SymbolType.ARCHETYPE,
                    valence=0.0,
                    strength=strength
                )
                dream_symbols.append(arch_sym)

        # Generar narrativa
        narrative = self._generate_dream_narrative(base_symbols, dream_symbols)

        # Calcular coherencia
        if dream_symbols:
            coherence = np.mean([s.strength for s in dream_symbols])
        else:
            coherence = 0.0

        # Calcular tono emocional
        all_dream_syms = base_symbols + dream_symbols
        if all_dream_syms:
            emotional_tone = np.mean([s.valence for s in all_dream_syms])
        else:
            emotional_tone = 0.0

        # Generar insights
        insights = self._extract_insights(dream_symbols)

        dream = SymbolicDream(
            symbols=dream_symbols,
            narrative=narrative,
            emotional_tone=emotional_tone,
            coherence=coherence,
            insights=insights
        )

        self.dream_history.append(dream)

        # Limitar historial
        max_dreams = max_history(self.t) // 10
        if len(self.dream_history) > max_dreams:
            self.dream_history = self.dream_history[-max_dreams:]

        return dream

    def _detect_archetypes(self, symbols: List[CircadianSymbol]):
        """Detecta patrones arquetipicos en simbolos."""
        # Arquetipos basicos
        archetype_patterns = {
            'hero': ['action', 'goal', 'obstacle'],
            'shadow': ['judgment', 'fragment'],
            'anima': ['association', 'metaphor', 'balance'],
            'self': ['transformation', 'emergence', 'archetype'],
            'trickster': ['bridge', 'threshold'],
        }

        for arch_name, patterns in archetype_patterns.items():
            # Contar coincidencias
            matches = 0
            for sym in symbols:
                if sym.symbol_type.value in patterns:
                    matches += 1

            if matches > 0:
                strength = matches / len(symbols)
                current = self.archetypes.get(arch_name, 0)
                self.archetypes[arch_name] = 0.9 * current + 0.1 * strength

    def _generate_dream_narrative(
        self,
        base: List[CircadianSymbol],
        dream: List[CircadianSymbol]
    ) -> str:
        """Genera narrativa del sueno."""
        if not base and not dream:
            return "vacio, silencio"

        # Elementos base
        base_contents = [s.content.split(':')[-1] for s in base[:3]]

        # Elementos oniricos
        dream_types = [s.symbol_type.value for s in dream[:3]]

        narrative = f"Sueno con elementos: {', '.join(base_contents)}. "
        narrative += f"Emerge: {', '.join(dream_types)}."

        # Tono
        if dream:
            tone = np.mean([s.valence for s in dream])
            if tone > 0.3:
                narrative += " Tono luminoso."
            elif tone < -0.3:
                narrative += " Tono oscuro."
            else:
                narrative += " Tono neutro."

        return narrative

    def _extract_insights(self, symbols: List[CircadianSymbol]) -> List[str]:
        """Extrae insights de los simbolos."""
        insights = []

        # Buscar metaforas significativas
        metaphors = [s for s in symbols if s.symbol_type == SymbolType.METAPHOR]
        for m in metaphors[:2]:
            if m.strength > 0.5:
                insights.append(f"Conexion descubierta: {m.content}")

        # Buscar arquetipos activos
        archetypes = [s for s in symbols if s.symbol_type == SymbolType.ARCHETYPE]
        for a in archetypes[:1]:
            insights.append(f"Patron profundo: {a.content}")

        return insights

    def process_liminal_transition(self) -> SymbolicTransition:
        """
        Procesa transicion simbolica durante LIMINAL.

        Transforma simbolos de una fase en otra.

        Returns:
            Transicion simbolica
        """
        # Obtener simbolos activos recientes
        recent_symbols = [
            sym for sym in self.symbols.values()
            if self.t - sym.last_activated_t < 20
        ]

        if not recent_symbols:
            return SymbolicTransition(
                from_symbols=[],
                to_symbols=[],
                transformation_type="none",
                significance=0.0
            )

        from_ids = [s.id for s in recent_symbols[:3]]
        to_symbols = []

        # Crear simbolos de transicion
        for sym in recent_symbols[:3]:
            # Puente
            bridge = self.create_symbol(
                content=f"puente:{sym.content}",
                symbol_type=SymbolType.BRIDGE,
                valence=sym.valence,
                strength=sym.strength * 0.8
            )
            to_symbols.append(bridge.id)
            self.connect_symbols(sym.id, bridge.id)

        # Crear simbolo de transformacion
        if len(recent_symbols) >= 2:
            combined_content = "+".join([s.content for s in recent_symbols[:2]])
            transform = self.create_symbol(
                content=f"transformacion:{combined_content}",
                symbol_type=SymbolType.TRANSFORMATION,
                valence=np.mean([s.valence for s in recent_symbols[:2]]),
                strength=np.mean([s.strength for s in recent_symbols[:2]])
            )
            to_symbols.append(transform.id)

        # Emergencia potencial
        if np.random.random() < 0.3:
            emergence = self.create_symbol(
                content=f"emergencia:nuevo_{self.t}",
                symbol_type=SymbolType.EMERGENCE,
                valence=0.3,
                strength=0.6
            )
            to_symbols.append(emergence.id)

        transition = SymbolicTransition(
            from_symbols=from_ids,
            to_symbols=to_symbols,
            transformation_type="bridge_transform",
            significance=np.mean([self.symbols[sid].strength for sid in to_symbols]) if to_symbols else 0
        )

        self.transition_history.append(transition)

        return transition

    def set_phase(self, phase: CircadianPhase):
        """Cambia la fase actual."""
        self.current_phase = phase

    def step(self, phase: CircadianPhase):
        """
        Ejecuta un paso del sistema.

        Args:
            phase: Fase circadiana actual
        """
        self.t += 1
        old_phase = self.current_phase
        self.set_phase(phase)

        # Decay de simbolos
        self._decay_symbols()

        # Acciones especificas por fase
        if phase == CircadianPhase.DREAM:
            # Generar simbolos oniricos
            if self.t % 5 == 0:  # No cada paso para evitar explosion
                self.generate_dream_symbols()

        elif phase == CircadianPhase.LIMINAL:
            # Procesar transiciones
            if old_phase != CircadianPhase.LIMINAL:
                self.process_liminal_transition()

    def get_symbolic_life_index(self) -> float:
        """
        Calcula indice de vida simbolica.

        Mide que tan activa es la vida simbolica del agente.

        Returns:
            Indice [0, 1]
        """
        if not self.symbols:
            return 0.0

        # Factores
        n_symbols = len(self.symbols)
        avg_strength = np.mean([s.strength for s in self.symbols.values()])
        n_connections = sum(len(s.connections) for s in self.symbols.values()) / 2
        n_archetypes = len([a for a, s in self.archetypes.items() if s > 0.3])

        # Normalizar
        symbol_factor = min(1, n_symbols / 50)
        strength_factor = avg_strength
        connection_factor = min(1, n_connections / 20)
        archetype_factor = min(1, n_archetypes / 3)

        # Combinar
        index = (
            0.3 * symbol_factor +
            0.3 * strength_factor +
            0.2 * connection_factor +
            0.2 * archetype_factor
        )

        return float(index)

    def get_statistics(self) -> Dict:
        """Estadisticas del sistema simbolico."""
        # Contar por tipo
        type_counts = {}
        for sym_type in SymbolType:
            count = len([s for s in self.symbols.values() if s.symbol_type == sym_type])
            type_counts[sym_type.value] = count

        # Simbolos por fase de origen
        phase_counts = {phase.value: 0 for phase in CircadianPhase}
        for sym in self.symbols.values():
            phase_counts[sym.phase_origin.value] += 1

        return {
            'agent_id': self.agent_id,
            't': self.t,
            'total_symbols': len(self.symbols),
            'symbol_types': type_counts,
            'symbols_by_origin_phase': phase_counts,
            'n_dreams': len(self.dream_history),
            'n_transitions': len(self.transition_history),
            'archetypes': self.archetypes.copy(),
            'symbolic_life_index': self.get_symbolic_life_index()
        }


def test_circadian_symbolism():
    """Test del sistema de simbolismo circadiano."""
    print("=" * 70)
    print("TEST: CIRCADIAN SYMBOLISM")
    print("=" * 70)

    np.random.seed(42)

    system = CircadianSymbolism("NEO")

    phases = [
        CircadianPhase.WAKE,
        CircadianPhase.REST,
        CircadianPhase.DREAM,
        CircadianPhase.LIMINAL
    ]

    print(f"\nAgente: NEO")
    print("\nSimulando 100 pasos a traves de fases...")

    for t in range(100):
        phase = phases[(t // 25) % 4]
        system.step(phase)

        # Crear simbolos aleatorios en WAKE
        if phase == CircadianPhase.WAKE and np.random.random() < 0.3:
            system.create_symbol(
                content=f"meta_{t}",
                symbol_type=SymbolType.GOAL,
                valence=0.5
            )

        # Crear simbolos de evaluacion en REST
        if phase == CircadianPhase.REST and np.random.random() < 0.2:
            system.create_symbol(
                content=f"valor_{t}",
                symbol_type=SymbolType.VALUE,
                valence=-0.2
            )

        if t % 25 == 24:
            print(f"\n  t={t+1}, Fase: {phase.value}")
            stats = system.get_statistics()
            print(f"    Total simbolos: {stats['total_symbols']}")
            print(f"    Indice vida simbolica: {stats['symbolic_life_index']:.3f}")
            print(f"    Suenos: {stats['n_dreams']}, Transiciones: {stats['n_transitions']}")

    # Estadisticas finales
    print("\n" + "=" * 70)
    print("ESTADISTICAS FINALES")
    print("=" * 70)

    stats = system.get_statistics()

    print(f"\n  Total simbolos: {stats['total_symbols']}")
    print(f"\n  Por tipo:")
    for sym_type, count in stats['symbol_types'].items():
        if count > 0:
            print(f"    {sym_type}: {count}")

    print(f"\n  Arquetipos emergentes:")
    for arch, strength in stats['archetypes'].items():
        if strength > 0.1:
            print(f"    {arch}: {strength:.3f}")

    print(f"\n  Indice de vida simbolica: {stats['symbolic_life_index']:.3f}")

    # Mostrar ultimo sueno
    if system.dream_history:
        last_dream = system.dream_history[-1]
        print(f"\n  Ultimo sueno:")
        print(f"    Narrativa: {last_dream.narrative}")
        print(f"    Tono: {last_dream.emotional_tone:.2f}")
        print(f"    Coherencia: {last_dream.coherence:.2f}")
        if last_dream.insights:
            print(f"    Insights: {last_dream.insights}")

    return system


if __name__ == "__main__":
    test_circadian_symbolism()
