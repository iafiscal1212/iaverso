"""
Symbolic Layer for NEO/EVA
==========================

Sistema simbólico emergente que provee:
- Símbolos como clases de equivalencia sobre episodios
- Alfabeto activo por agente
- Composición (bindings) de símbolos
- Gramática emergente con roles
- Grounding en WORLD-1 y contexto social
- Uso cognitivo: narrativa y planificación

Todo endógeno. Sin números mágicos.
"""

from .sym_atoms import Symbol, SymbolStats, SymbolExtractor
from .sym_alphabet import SymbolAlphabet, SymbolActivation
from .sym_binding import SymbolBinding, SymbolBindingManager
from .sym_grammar import SymbolRole, GrammarRule, SymbolGrammar
from .sym_grounding import SymbolGroundingStats, SymbolGrounding
from .sym_use_cognition import SymbolicPlan, SymbolicNarrative, SymbolicCognitionUse
from .sym_audit import SymAuditResult, SymbolicAuditor

__all__ = [
    # Atoms
    'Symbol',
    'SymbolStats',
    'SymbolExtractor',
    # Alphabet
    'SymbolAlphabet',
    'SymbolActivation',
    # Binding
    'SymbolBinding',
    'SymbolBindingManager',
    # Grammar
    'SymbolRole',
    'GrammarRule',
    'SymbolGrammar',
    # Grounding
    'SymbolGroundingStats',
    'SymbolGrounding',
    # Cognition
    'SymbolicPlan',
    'SymbolicNarrative',
    'SymbolicCognitionUse',
    # Audit
    'SymAuditResult',
    'SymbolicAuditor',
]
