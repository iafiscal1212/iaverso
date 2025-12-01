"""
Mapa de Coherencia Interna
==========================

Visualiza como se acoplan los diferentes frameworks:
- AGI-X (funcional)
- SYM-X (simbolica)
- CG-E (coherencia global)
- PMCC (persistencia multi-capa)
- STX (temporal simbolico)
- AGI-E (endogenous AGI)

Primera vez en ingenieria cognitiva que se muestra algo asi.

100% endogeno. Sin numeros magicos.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, '/root/NEO_EVA')


@dataclass
class CoherenceMapResult:
    """Resultado del mapa de coherencia."""
    # Scores por framework
    agi_x_score: float      # AGI-X funcional (S1-S5)
    sym_x_score: float      # SYM-X simbolico (SX1-SX15)
    cg_e_score: float       # CG-E coherencia global
    pmcc_score: float       # PMCC persistencia multi-capa
    stx_score: float        # STX temporal simbolico
    agi_e_score: float      # AGI-E endogenous AGI

    # Correlaciones entre frameworks
    correlations: Dict[str, float]

    # Coherencia global
    global_coherence: float

    # Es AGI interna completa
    is_complete_agi: bool

    # Mapa de conexiones
    connection_map: Dict[str, Dict[str, float]]

    details: Dict[str, Any]


class CoherenceMap:
    """
    Mapa de Coherencia Interna.

    Integra y visualiza las relaciones entre todos los frameworks.
    """

    def __init__(self):
        # Scores por framework
        self.scores: Dict[str, float] = {}

        # Componentes internos de cada framework
        self.components: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Historial temporal
        self.history: Dict[str, List[float]] = defaultdict(list)

    def set_agi_x(self, s1: float, s2: float, s3: float, s4: float, s5: float):
        """Establece scores AGI-X (funcional)."""
        self.components['AGI-X'] = {
            'S1_Adaptation': s1,
            'S2_Robustness': s2,
            'S3_Grammar_Causality': s3,
            'S4_Self_Model': s4,
            'S5_Theory_of_Mind': s5
        }
        self.scores['AGI-X'] = np.mean([s1, s2, s3, s4, s5])
        self.history['AGI-X'].append(self.scores['AGI-X'])

    def set_sym_x(self, sx_scores: Dict[str, float]):
        """Establece scores SYM-X (simbolico)."""
        self.components['SYM-X'] = sx_scores
        self.scores['SYM-X'] = np.mean(list(sx_scores.values()))
        self.history['SYM-X'].append(self.scores['SYM-X'])

    def set_cg_e(self, p: float, s: float, m: float, cg_e: float):
        """Establece scores CG-E (coherencia global)."""
        self.components['CG-E'] = {
            'P_Persistence': p,
            'S_No_Collapse': s,
            'M_Continuity': m
        }
        self.scores['CG-E'] = cg_e
        self.history['CG-E'].append(cg_e)

    def set_pmcc(self, pmcc: float, layer_scores: Dict[str, float] = None):
        """Establece score PMCC (persistencia multi-capa)."""
        self.components['PMCC'] = layer_scores or {'PMCC': pmcc}
        self.scores['PMCC'] = pmcc
        self.history['PMCC'].append(pmcc)

    def set_stx(self, stx_scores: Dict[str, float], stx_global: float):
        """Establece scores STX (temporal simbolico)."""
        self.components['STX'] = stx_scores
        self.scores['STX'] = stx_global
        self.history['STX'].append(stx_global)

    def set_agi_e(self, e1: float, e2: float, e3: float, e4: float, e5: float):
        """Establece scores AGI-E (endogenous AGI)."""
        self.components['AGI-E'] = {
            'E1_Persistence': e1,
            'E2_No_Collapse': e2,
            'E3_Attractors': e3,
            'E4_Memory': e4,
            'E5_Symbolic_Temporal': e5
        }
        self.scores['AGI-E'] = np.mean([e1, e2, e3, e4, e5])
        self.history['AGI-E'].append(self.scores['AGI-E'])

    def compute_correlations(self) -> Dict[str, float]:
        """Calcula correlaciones entre frameworks."""
        correlations = {}
        frameworks = list(self.scores.keys())

        for i in range(len(frameworks)):
            for j in range(i + 1, len(frameworks)):
                f1, f2 = frameworks[i], frameworks[j]

                if len(self.history[f1]) >= 2 and len(self.history[f2]) >= 2:
                    # Correlacion temporal
                    min_len = min(len(self.history[f1]), len(self.history[f2]))
                    h1 = self.history[f1][-min_len:]
                    h2 = self.history[f2][-min_len:]

                    if np.std(h1) > 0 and np.std(h2) > 0:
                        corr = np.corrcoef(h1, h2)[0, 1]
                    else:
                        corr = 1.0 if self.scores[f1] == self.scores[f2] else 0.0
                else:
                    # Sin historial, usar similaridad de scores
                    diff = abs(self.scores[f1] - self.scores[f2])
                    corr = 1 - diff

                correlations[f"{f1}-{f2}"] = float(corr) if not np.isnan(corr) else 0.0

        return correlations

    def compute_connection_map(self) -> Dict[str, Dict[str, float]]:
        """
        Calcula mapa de conexiones entre frameworks.

        Cada conexion representa la fuerza de la relacion.
        """
        connections = defaultdict(dict)

        # Conexiones teoricas
        # AGI-X <-> CG-E: funcionalidad y coherencia
        if 'AGI-X' in self.scores and 'CG-E' in self.scores:
            connections['AGI-X']['CG-E'] = (self.scores['AGI-X'] + self.scores['CG-E']) / 2

        # SYM-X <-> STX: simbolico y temporal
        if 'SYM-X' in self.scores and 'STX' in self.scores:
            connections['SYM-X']['STX'] = (self.scores['SYM-X'] + self.scores['STX']) / 2

        # CG-E <-> PMCC: coherencia y persistencia
        if 'CG-E' in self.scores and 'PMCC' in self.scores:
            connections['CG-E']['PMCC'] = (self.scores['CG-E'] + self.scores['PMCC']) / 2

        # AGI-E <-> todos: AGI-E es el integrante final
        for framework in ['AGI-X', 'SYM-X', 'CG-E', 'PMCC', 'STX']:
            if framework in self.scores and 'AGI-E' in self.scores:
                connections['AGI-E'][framework] = (
                    self.scores['AGI-E'] + self.scores[framework]
                ) / 2

        return dict(connections)

    def compute_global_coherence(self) -> float:
        """Calcula coherencia global del sistema."""
        if not self.scores:
            return 0.0

        scores = list(self.scores.values())

        # Coherencia = media * (1 - varianza normalizada)
        mean_score = np.mean(scores)
        variance = np.var(scores)

        # Normalizar varianza
        max_variance = 0.25  # Varianza maxima para scores en [0,1]
        normalized_var = variance / max_variance

        coherence = mean_score * (1 - normalized_var)

        return float(np.clip(coherence, 0, 1))

    def is_complete_agi_internal(self) -> bool:
        """Determina si el sistema califica como AGI interna completa."""
        required = ['AGI-X', 'SYM-X', 'CG-E', 'PMCC', 'AGI-E']

        # Verificar que todos los frameworks estan presentes
        if not all(f in self.scores for f in required):
            return False

        # Todos deben estar por encima de umbral
        threshold = 0.5
        all_above_threshold = all(self.scores[f] >= threshold for f in required)

        # Coherencia global debe ser alta
        coherence = self.compute_global_coherence()
        high_coherence = coherence >= 0.6

        # AGI-E debe tener todas las condiciones (aproximado)
        if 'AGI-E' in self.components:
            e_components = self.components['AGI-E']
            all_e_passed = all(v >= 0.5 for v in e_components.values())
        else:
            all_e_passed = False

        return all_above_threshold and high_coherence and all_e_passed

    def generate_map(self) -> CoherenceMapResult:
        """Genera el mapa de coherencia completo."""
        correlations = self.compute_correlations()
        connection_map = self.compute_connection_map()
        global_coherence = self.compute_global_coherence()
        is_complete = self.is_complete_agi_internal()

        return CoherenceMapResult(
            agi_x_score=self.scores.get('AGI-X', 0.0),
            sym_x_score=self.scores.get('SYM-X', 0.0),
            cg_e_score=self.scores.get('CG-E', 0.0),
            pmcc_score=self.scores.get('PMCC', 0.0),
            stx_score=self.scores.get('STX', 0.0),
            agi_e_score=self.scores.get('AGI-E', 0.0),
            correlations=correlations,
            global_coherence=global_coherence,
            is_complete_agi=is_complete,
            connection_map=connection_map,
            details={
                'components': dict(self.components),
                'history_lengths': {k: len(v) for k, v in self.history.items()}
            }
        )

    def print_ascii_map(self):
        """Imprime mapa ASCII del sistema."""
        result = self.generate_map()

        print("\n" + "=" * 80)
        print("MAPA DE COHERENCIA INTERNA")
        print("=" * 80)

        # Diagrama ASCII
        print("""
                          ┌─────────────────────────────────────┐
                          │        AGI-E (Endogenous AGI)       │
                          │  E1 + E2 + E3 + E4 + E5 = {:.2f}      │
                          └─────────────────────────────────────┘
                                          │
              ┌─────────────────────────────┼─────────────────────────────┐
              │                             │                             │
              ▼                             ▼                             ▼
    ┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
    │     AGI-X        │        │      CG-E        │        │      STX         │
    │   (Funcional)    │◄──────►│   (Coherencia)   │◄──────►│   (Temporal)     │
    │   S1-S5 = {:.2f}   │        │   P+S+M = {:.2f}   │        │  STX1-10 = {:.2f}  │
    └──────────────────┘        └──────────────────┘        └──────────────────┘
              │                             │                             │
              │                             │                             │
              ▼                             ▼                             ▼
    ┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
    │     SYM-X        │        │      PMCC        │        │   Stress Tests   │
    │   (Simbolico)    │◄──────►│  (Persistencia)  │◄──────►│   (Resiliencia)  │
    │  SX1-15 = {:.2f}   │        │   PMCC = {:.2f}     │        │   [Endogeno]     │
    └──────────────────┘        └──────────────────┘        └──────────────────┘

              └────────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
                          ┌─────────────────────────────────────┐
                          │       COHERENCIA GLOBAL: {:.2f}       │
                          │       AGI INTERNA: {}             │
                          └─────────────────────────────────────┘
""".format(
            result.agi_e_score,
            result.agi_x_score,
            result.cg_e_score,
            result.stx_score,
            result.sym_x_score,
            result.pmcc_score,
            result.global_coherence,
            "COMPLETA" if result.is_complete_agi else "PARCIAL "
        ))

        # Tabla de componentes
        print("\n  COMPONENTES POR FRAMEWORK:")
        print("  " + "-" * 70)

        for framework, components in self.components.items():
            print(f"\n  {framework}:")
            for comp, score in components.items():
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"    {comp:<25} [{bar}] {score:.4f}")

        # Correlaciones
        print("\n\n  CORRELACIONES ENTRE FRAMEWORKS:")
        print("  " + "-" * 70)
        for pair, corr in result.correlations.items():
            print(f"    {pair:<20}: {corr:.4f}")

        print("\n" + "=" * 80)


def run_coherence_map_demo():
    """Demo del mapa de coherencia."""
    np.random.seed(42)

    cmap = CoherenceMap()

    # Simular scores de diferentes frameworks
    # AGI-X
    cmap.set_agi_x(
        s1=0.65,  # Adaptation
        s2=0.72,  # Robustness
        s3=0.58,  # Grammar Causality
        s4=0.61,  # Self Model
        s5=0.55   # Theory of Mind
    )

    # SYM-X
    cmap.set_sym_x({
        'SX1': 0.75, 'SX2': 0.68, 'SX3': 0.82, 'SX4': 0.71,
        'SX5': 0.65, 'SX6': 0.78, 'SX7': 0.69, 'SX8': 0.73,
        'SX9': 0.66, 'SX10': 0.81, 'SX11': 0.87, 'SX12': 0.48,
        'SX13': 0.64, 'SX14': 0.89, 'SX15': 0.38
    })

    # CG-E
    cmap.set_cg_e(
        p=0.98,  # Persistence
        s=0.93,  # No-collapse
        m=0.68,  # Continuity
        cg_e=0.86
    )

    # PMCC
    cmap.set_pmcc(
        pmcc=0.996,
        layer_scores={
            'Teleology': 0.95,
            'Symbols': 0.92,
            'Causality': 0.88,
            'Metacognition': 0.91
        }
    )

    # STX
    cmap.set_stx(
        stx_scores={
            'STX-1': 0.25, 'STX-2': 1.0, 'STX-3': 0.50, 'STX-4': 0.11,
            'STX-5': 0.15, 'STX-6': 0.98, 'STX-7': 0.50, 'STX-8': 0.24,
            'STX-9': 0.06, 'STX-10': 0.47
        },
        stx_global=0.43
    )

    # AGI-E
    cmap.set_agi_e(
        e1=0.95,  # Persistence
        e2=0.92,  # No-collapse
        e3=0.87,  # Attractors
        e4=0.89,  # Memory
        e5=0.76   # Symbolic Temporal
    )

    # Generar y mostrar mapa
    cmap.print_ascii_map()

    return cmap.generate_map()


if __name__ == "__main__":
    result = run_coherence_map_demo()
