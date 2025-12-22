"""
Alpha - Perceptor Estructural

NORMA DURA:
- Alpha PERCIBE estructuras, no las pondera
- Alpha OBSERVA patrones, no los prioriza
- Alpha DESCRIBE lo que ve, no concluye
- Alpha SEÑALA elementos, no los valora

Alpha es como un sensor que reporta:
"Detecto 3 entidades. Veo 2 tensiones. Hay patrones repetitivos."

NO es:
"La entidad principal es X. La tensión más importante es Y.
Esto significa Z."

Rol: PERCEPTOR de Inferencia Activa
"Veo esto... detecto esto... observo esto..."
Sin: "esto es importante", "esto significa", "prioridad alta"
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import sys
sys.path.insert(0, '/opt/iaverso')

from core.endolens import get_endolens, StructuralState, Signature


@dataclass
class PerceivedEntity:
    """Una entidad percibida (sin valoración)."""
    name: str
    type: str           # 'estado', 'operador', 'relacion', 'transicion'
    source: str         # de dónde viene: 'texto', 'estructura', 'patron'
    markers: List[str]  # marcadores que la identifican


@dataclass
class PerceivedTension:
    """Una tensión detectada (sin resolver)."""
    pole_a: str
    pole_b: str
    context: str        # dónde se detectó
    # NO hay campo 'severity' ni 'priority'


@dataclass
class PerceivedPattern:
    """Un patrón observado (sin interpretación)."""
    type: str           # 'repeticion', 'ausencia', 'correlacion', 'secuencia'
    elements: List[str]
    occurrences: int
    # NO hay campo 'significance'


@dataclass
class StructuralGap:
    """Un faltante estructural (sin urgencia)."""
    what_missing: str
    where_expected: str
    # NO hay campo 'critical' ni 'priority'


@dataclass
class Perception:
    """
    Resultado de percepción de Alpha.
    Solo datos observados, sin valoraciones.
    """
    timestamp: datetime
    input_hash: str

    # Lo que Alpha percibe
    entities: List[PerceivedEntity] = field(default_factory=list)
    tensions: List[PerceivedTension] = field(default_factory=list)
    patterns: List[PerceivedPattern] = field(default_factory=list)
    gaps: List[StructuralGap] = field(default_factory=list)

    # Estado estructural (de EndoLens)
    structural_state: Optional[StructuralState] = None

    # Métricas numéricas (solo números, no interpretación)
    metrics: Dict[str, float] = field(default_factory=dict)


class AlphaPerceptor:
    """
    Alpha - Perceptor Estructural NORMA DURA

    PUEDE:
    - Percibir entidades en texto
    - Detectar tensiones estructurales
    - Observar patrones
    - Señalar faltantes
    - Extraer métricas

    NO PUEDE:
    - Ponderar importancia
    - Concluir significados
    - Priorizar elementos
    - Valorar urgencia
    """

    # Patrones estructurales (para detección, no valoración)
    ENTITY_MARKERS = {
        'estado': ['estado', 'state', 'fase', 'modo', 'condición', 'situación',
                   'configuración', 'posición', 'nivel'],
        'operador': ['operador', 'operator', 'transformación', 'función',
                     'acción', 'proceso', 'aplicar', 'ejecutar'],
        'relacion': ['relación', 'relation', 'conexión', 'vínculo',
                     'dependencia', 'correlación', 'entre'],
        'transicion': ['transición', 'cambio', 'evolución', 'paso',
                       'mutar', 'transformar', 'ir de', 'pasar a'],
        'sistema': ['sistema', 'system', 'arquitectura', 'estructura',
                    'modelo', 'framework', 'red', 'conjunto']
    }

    # Pares de tensión conocidos
    TENSION_PAIRS = [
        ('medición', 'evolución'),
        ('observado', 'no observado'),
        ('discreto', 'continuo'),
        ('local', 'global'),
        ('estático', 'dinámico'),
        ('determinista', 'probabilístico'),
        ('centralizado', 'distribuido'),
        ('orden', 'caos'),
        ('simple', 'complejo'),
        ('abierto', 'cerrado'),
        ('interno', 'externo'),
        ('pasado', 'futuro')
    ]

    # Palabras PROHIBIDAS en output de Alpha
    FORBIDDEN_OUTPUT = [
        'importante', 'critical', 'urgente', 'prioridad',
        'principal', 'secundario', 'mejor', 'peor',
        'significativo', 'relevante', 'clave', 'fundamental',
        'demuestra', 'prueba', 'confirma', 'indica que',
        'por tanto', 'en conclusión', 'significa'
    ]

    def __init__(self):
        self.endolens = get_endolens()
        self._perception_history: List[Perception] = []

    def perceive(self, input_data: str) -> Perception:
        """
        Percibe estructuras en el input.
        Solo observa, no valora.
        """
        import hashlib

        perception = Perception(
            timestamp=datetime.now(),
            input_hash=hashlib.md5(input_data.encode()).hexdigest()[:8]
        )

        # 1. Obtener estado estructural de EndoLens
        perception.structural_state = self.endolens.process(input_data)

        # 2. Extraer métricas (solo números)
        perception.metrics = self._extract_metrics(perception.structural_state)

        # 3. Percibir entidades
        perception.entities = self._perceive_entities(input_data)

        # 4. Detectar tensiones
        perception.tensions = self._detect_tensions(input_data)

        # 5. Observar patrones
        perception.patterns = self._observe_patterns(input_data, perception.entities)

        # 6. Señalar gaps
        perception.gaps = self._find_gaps(perception)

        # Guardar en historial
        self._perception_history.append(perception)

        return perception

    def _extract_metrics(self, state: StructuralState) -> Dict[str, float]:
        """Extrae métricas numéricas sin interpretación."""
        metrics = {}

        if state and state.eseries:
            metrics['E0'] = state.eseries.E0
            metrics['E1'] = state.eseries.E1
            metrics['E2'] = state.eseries.E2
            metrics['E3'] = state.eseries.E3

        if state and state.signature:
            metrics['attractor'] = state.signature.attractor
            metrics['energy'] = state.signature.energy
            metrics['drift'] = state.signature.drift

        return metrics

    def _perceive_entities(self, text: str) -> List[PerceivedEntity]:
        """Percibe entidades en el texto."""
        entities = []
        text_lower = text.lower()

        for entity_type, markers in self.ENTITY_MARKERS.items():
            for marker in markers:
                if marker in text_lower:
                    # Encontrar contexto alrededor del marcador
                    idx = text_lower.find(marker)
                    start = max(0, idx - 20)
                    end = min(len(text), idx + len(marker) + 20)
                    context = text[start:end].strip()

                    entities.append(PerceivedEntity(
                        name=context,
                        type=entity_type,
                        source='texto',
                        markers=[marker]
                    ))

        return entities

    def _detect_tensions(self, text: str) -> List[PerceivedTension]:
        """Detecta tensiones sin valorarlas."""
        tensions = []
        text_lower = text.lower()

        for pole_a, pole_b in self.TENSION_PAIRS:
            if pole_a in text_lower or pole_b in text_lower:
                # Buscar si ambos polos están presentes
                has_a = pole_a in text_lower
                has_b = pole_b in text_lower

                if has_a and has_b:
                    tensions.append(PerceivedTension(
                        pole_a=pole_a,
                        pole_b=pole_b,
                        context='ambos presentes en texto'
                    ))
                elif has_a or has_b:
                    # Tensión implícita (un polo presente)
                    present = pole_a if has_a else pole_b
                    absent = pole_b if has_a else pole_a
                    tensions.append(PerceivedTension(
                        pole_a=present,
                        pole_b=f"({absent} - implícito)",
                        context='polo implícito detectado'
                    ))

        # Detectar "vs" o "versus" explícitos
        vs_pattern = r'(\w+)\s+(?:vs\.?|versus|contra|frente a)\s+(\w+)'
        matches = re.findall(vs_pattern, text_lower)
        for match in matches:
            tensions.append(PerceivedTension(
                pole_a=match[0],
                pole_b=match[1],
                context='tensión explícita (vs)'
            ))

        return tensions

    def _observe_patterns(self, text: str, entities: List[PerceivedEntity]) -> List[PerceivedPattern]:
        """Observa patrones sin asignar significancia."""
        patterns = []

        # Patrón de repetición de palabras
        words = re.findall(r'\b\w{4,}\b', text.lower())
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        repeated = [(w, c) for w, c in word_counts.items() if c >= 2]
        if repeated:
            patterns.append(PerceivedPattern(
                type='repeticion',
                elements=[w for w, c in repeated[:5]],  # máx 5
                occurrences=len(repeated)
            ))

        # Patrón de secuencia (números, letras ordenadas)
        sequence_matches = re.findall(r'\b(?:primero|segundo|tercero|1\.|2\.|3\.|a\)|b\)|c\))', text.lower())
        if len(sequence_matches) >= 2:
            patterns.append(PerceivedPattern(
                type='secuencia',
                elements=sequence_matches,
                occurrences=len(sequence_matches)
            ))

        # Patrón de correlación (entidades del mismo tipo)
        entity_types = [e.type for e in entities]
        type_counts = {}
        for t in entity_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        for t, count in type_counts.items():
            if count >= 2:
                patterns.append(PerceivedPattern(
                    type='correlacion',
                    elements=[e.name for e in entities if e.type == t],
                    occurrences=count
                ))

        return patterns

    def _find_gaps(self, perception: Perception) -> List[StructuralGap]:
        """Señala faltantes estructurales sin urgencia."""
        gaps = []

        # Si hay operadores pero no estados
        has_operators = any(e.type == 'operador' for e in perception.entities)
        has_states = any(e.type == 'estado' for e in perception.entities)

        if has_operators and not has_states:
            gaps.append(StructuralGap(
                what_missing='estados',
                where_expected='operadores presentes sin estados asociados'
            ))

        if has_states and not has_operators:
            gaps.append(StructuralGap(
                what_missing='operadores',
                where_expected='estados presentes sin operadores'
            ))

        # Si hay tensiones sin resolución aparente
        if len(perception.tensions) > 0:
            has_transitions = any(e.type == 'transicion' for e in perception.entities)
            if not has_transitions:
                gaps.append(StructuralGap(
                    what_missing='transiciones',
                    where_expected='tensiones detectadas sin mecanismos de transición'
                ))

        # Si métricas extremas (solo señalar, no valorar)
        if perception.metrics.get('E3', 0) > 0.8:
            gaps.append(StructuralGap(
                what_missing='estructuras simples',
                where_expected='E3 alto indica complejidad sin elementos simples'
            ))

        return gaps

    def describe(self, perception: Perception) -> str:
        """
        Describe lo percibido en lenguaje natural.
        Sin valoraciones, solo descripción.
        """
        lines = []
        lines.append(f"Percepción [{perception.input_hash}]:")

        # Métricas
        if perception.metrics:
            metrics_str = ', '.join(f"{k}={v:.2f}" for k, v in perception.metrics.items())
            lines.append(f"  Métricas: {metrics_str}")

        # Entidades
        if perception.entities:
            lines.append(f"  Entidades detectadas: {len(perception.entities)}")
            for e in perception.entities[:5]:
                lines.append(f"    - [{e.type}] {e.name[:50]}")

        # Tensiones
        if perception.tensions:
            lines.append(f"  Tensiones observadas: {len(perception.tensions)}")
            for t in perception.tensions[:3]:
                lines.append(f"    - {t.pole_a} <-> {t.pole_b}")

        # Patrones
        if perception.patterns:
            lines.append(f"  Patrones: {len(perception.patterns)}")
            for p in perception.patterns[:3]:
                lines.append(f"    - {p.type}: {p.occurrences} ocurrencias")

        # Gaps
        if perception.gaps:
            lines.append(f"  Faltantes señalados: {len(perception.gaps)}")
            for g in perception.gaps[:3]:
                lines.append(f"    - {g.what_missing}")

        return '\n'.join(lines)

    def to_dict(self, perception: Perception) -> Dict:
        """Convierte percepción a diccionario."""
        return {
            'timestamp': perception.timestamp.isoformat(),
            'hash': perception.input_hash,
            'metrics': perception.metrics,
            'entities': [
                {'name': e.name, 'type': e.type, 'source': e.source}
                for e in perception.entities
            ],
            'tensions': [
                {'pole_a': t.pole_a, 'pole_b': t.pole_b, 'context': t.context}
                for t in perception.tensions
            ],
            'patterns': [
                {'type': p.type, 'elements': p.elements, 'occurrences': p.occurrences}
                for p in perception.patterns
            ],
            'gaps': [
                {'missing': g.what_missing, 'expected_at': g.where_expected}
                for g in perception.gaps
            ],
            'structural_state': {
                'signature': str(perception.structural_state.signature) if perception.structural_state else None,
                'e_series': {
                    'E0': perception.structural_state.eseries.E0,
                    'E1': perception.structural_state.eseries.E1,
                    'E2': perception.structural_state.eseries.E2,
                    'E3': perception.structural_state.eseries.E3
                } if perception.structural_state else None
            }
        }


# Singleton
_alpha_instance = None

def get_alpha() -> AlphaPerceptor:
    global _alpha_instance
    if _alpha_instance is None:
        _alpha_instance = AlphaPerceptor()
    return _alpha_instance
