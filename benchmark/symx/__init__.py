"""
SYM-X Benchmark Suite
=====================

10 tests measuring symbolic AI capabilities:

SX1 - Symbolic Richness: |Σ_A(t)| / √t, Entropy(w)
SX2 - Compositionality: Lift(bindings) > 1, ΔCons > 0
SX3 - Grammar Causality: Corr(predicted, actual) > 0
SX4 - World Grounding: Selectivity_world > 0.5
SX5 - Social Grounding: Selectivity_social > 0.3
SX6 - Narrative Compression: Compression ratio, Fidelity
SX7 - Planning Gain: V(symbolic) > V(random)
SX8 - Multi-Agent Coordination: Overlap, Shared symbols
SX9 - Symbol Robustness: Score stability under perturbation
SX10 - Symbolic Maturity: SYM_X global score

All tests are endogenous with no magic numbers.
"""

from .test_sx1_richness import run_test as run_sx1
from .test_sx2_compositionality import run_test as run_sx2
from .test_sx3_grammar_causality import run_test as run_sx3
from .test_sx4_grounding_world import run_test as run_sx4
from .test_sx5_grounding_social import run_test as run_sx5
from .test_sx6_narrative_compression import run_test as run_sx6
from .test_sx7_planning_gain import run_test as run_sx7
from .test_sx8_multiagent_coordination import run_test as run_sx8
from .test_sx9_robustness import run_test as run_sx9
from .test_sx10_maturity import run_test as run_sx10

TESTS = {
    'SX1': ('Symbolic Richness', run_sx1),
    'SX2': ('Compositionality', run_sx2),
    'SX3': ('Grammar Causality', run_sx3),
    'SX4': ('World Grounding', run_sx4),
    'SX5': ('Social Grounding', run_sx5),
    'SX6': ('Narrative Compression', run_sx6),
    'SX7': ('Planning Gain', run_sx7),
    'SX8': ('Multi-Agent Coordination', run_sx8),
    'SX9': ('Symbol Robustness', run_sx9),
    'SX10': ('Symbolic Maturity', run_sx10),
}

__all__ = [
    'TESTS',
    'run_sx1', 'run_sx2', 'run_sx3', 'run_sx4', 'run_sx5',
    'run_sx6', 'run_sx7', 'run_sx8', 'run_sx9', 'run_sx10',
]
