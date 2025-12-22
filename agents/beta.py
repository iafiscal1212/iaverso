"""
Beta - Explorador Relacional

NORMA DURA:
- Beta CONECTA elementos, no sintetiza conclusiones
- Beta PROPONE relaciones posibles, no decide cuál es correcta
- Beta ABRE hipótesis, no las cierra
- Beta EXPLORA caminos, no elige el mejor

Beta es como un cartógrafo que mapea:
"Veo conexión entre A y B. Posible relación con C.
Hay un camino por aquí, y otro por allá."

NO es:
"La conexión más importante es A-B. Esto demuestra X.
La relación correcta es Y. Síntesis: Z."

Rol: EXPLORADOR de Inferencia Activa
"Conecto esto con esto... posible relación... camino abierto..."
Sin: "síntesis", "conclusión", "lo más relevante"
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import sys
sys.path.insert(0, '/opt/iaverso')

from core.endolens import get_endolens, StructuralState
from core.neosynt import get_neosynt, Resolution


@dataclass
class Connection:
    """Una conexión explorada (sin valoración de importancia)."""
    source: str
    target: str
    type: str           # 'causal', 'temporal', 'espacial', 'logica', 'analogica'
    strength: float     # 0-1, solo métrica, no "importancia"
    evidence: str       # qué indica esta conexión


@dataclass
class Hypothesis:
    """Una hipótesis abierta (sin valorar probabilidad de verdad)."""
    statement: str
    based_on: List[str]     # elementos que la sostienen
    questions: List[str]    # preguntas que quedan abiertas
    # NO hay campo 'likelihood' ni 'confidence'


@dataclass
class Path:
    """Un camino explorado (sin decir cuál es mejor)."""
    nodes: List[str]
    connections: List[Connection]
    length: int
    # NO hay campo 'optimal' ni 'recommended'


@dataclass
class Exploration:
    """
    Resultado de exploración de Beta.
    Mapea conexiones y caminos, sin jerarquizar.
    """
    timestamp: datetime
    input_hash: str

    # Lo que Beta explora
    connections: List[Connection] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    paths: List[Path] = field(default_factory=list)

    # Dominios tocados
    domains_touched: Set[str] = field(default_factory=set)

    # Preguntas abiertas (siempre termina con preguntas)
    open_questions: List[str] = field(default_factory=list)


class BetaExplorer:
    """
    Beta - Explorador Relacional NORMA DURA

    PUEDE:
    - Conectar elementos detectados
    - Proponer relaciones posibles
    - Abrir hipótesis
    - Mapear caminos
    - Explorar dominios

    NO PUEDE:
    - Sintetizar conclusiones
    - Decidir relevancia final
    - Elegir "la mejor" opción
    - Cerrar preguntas
    """

    # Tipos de conexión que Beta puede explorar
    CONNECTION_TYPES = {
        'causal': ['causa', 'produce', 'genera', 'provoca', 'lleva a',
                   'results in', 'causes', 'leads to'],
        'temporal': ['antes', 'después', 'durante', 'mientras', 'cuando',
                     'before', 'after', 'during', 'while'],
        'espacial': ['dentro', 'fuera', 'entre', 'junto', 'separado',
                     'inside', 'outside', 'between', 'near'],
        'logica': ['si', 'entonces', 'porque', 'por tanto', 'implica',
                   'if', 'then', 'because', 'implies'],
        'analogica': ['como', 'similar', 'parecido', 'equivalente', 'análogo',
                      'like', 'similar', 'equivalent', 'analogous']
    }

    # Dominios conocidos para explorar
    KNOWN_DOMAINS = [
        'fisica', 'matematicas', 'biologia', 'quimica', 'computacion',
        'economia', 'sociologia', 'psicologia', 'filosofia', 'linguistica',
        'arte', 'musica', 'historia', 'geografia', 'medicina'
    ]

    # Palabras PROHIBIDAS en output de Beta
    FORBIDDEN_OUTPUT = [
        'en síntesis', 'en conclusión', 'por tanto',
        'lo más importante', 'lo principal', 'la clave',
        'definitivamente', 'claramente', 'obviamente',
        'la mejor opción', 'recomiendo', 'deberías',
        'esto demuestra', 'esto prueba', 'esto confirma'
    ]

    def __init__(self):
        self.endolens = get_endolens()
        self.neosynt = get_neosynt()
        self._exploration_history: List[Exploration] = []

    def explore(self, perception_data: Dict, context: str = "") -> Exploration:
        """
        Explora relaciones a partir de datos de percepción.
        Mapea conexiones, no sintetiza.
        """
        import hashlib

        exploration = Exploration(
            timestamp=datetime.now(),
            input_hash=hashlib.md5(str(perception_data).encode()).hexdigest()[:8],
            domains_touched=set()
        )

        # 1. Extraer elementos de la percepción
        entities = perception_data.get('entities', [])
        tensions = perception_data.get('tensions', [])
        patterns = perception_data.get('patterns', [])

        # 2. Buscar conexiones entre entidades
        exploration.connections = self._find_connections(entities, context)

        # 3. Proponer hipótesis basadas en tensiones
        exploration.hypotheses = self._propose_hypotheses(tensions, entities)

        # 4. Mapear caminos posibles
        exploration.paths = self._map_paths(exploration.connections, entities)

        # 5. Identificar dominios tocados
        exploration.domains_touched = self._identify_domains(context, entities)

        # 6. Generar preguntas abiertas (SIEMPRE)
        exploration.open_questions = self._generate_questions(exploration)

        # Guardar en historial
        self._exploration_history.append(exploration)

        return exploration

    def _find_connections(self, entities: List[Dict], context: str) -> List[Connection]:
        """Busca conexiones entre entidades."""
        connections = []
        context_lower = context.lower()

        # Buscar conexiones por tipo
        for conn_type, markers in self.CONNECTION_TYPES.items():
            for marker in markers:
                if marker in context_lower:
                    # Encontrar qué entidades están cerca del marcador
                    idx = context_lower.find(marker)
                    nearby_text = context_lower[max(0, idx-50):min(len(context), idx+50)]

                    # Buscar entidades mencionadas cerca
                    connected_entities = []
                    for entity in entities:
                        entity_name = entity.get('name', '').lower()
                        if entity_name[:20] in nearby_text:
                            connected_entities.append(entity)

                    # Crear conexiones entre pares
                    for i, e1 in enumerate(connected_entities):
                        for e2 in connected_entities[i+1:]:
                            connections.append(Connection(
                                source=e1.get('name', 'unknown')[:30],
                                target=e2.get('name', 'unknown')[:30],
                                type=conn_type,
                                strength=0.5,  # métrica neutral
                                evidence=f"marcador '{marker}' encontrado cerca"
                            ))

        # Conexiones por co-ocurrencia de tipos
        entity_types = {}
        for e in entities:
            t = e.get('type', 'unknown')
            if t not in entity_types:
                entity_types[t] = []
            entity_types[t].append(e)

        # Estados conectan con operadores
        states = entity_types.get('estado', [])
        operators = entity_types.get('operador', [])

        for state in states[:3]:
            for op in operators[:3]:
                connections.append(Connection(
                    source=state.get('name', '')[:30],
                    target=op.get('name', '')[:30],
                    type='logica',
                    strength=0.3,
                    evidence='estado-operador típicamente relacionados'
                ))

        return connections

    def _propose_hypotheses(self, tensions: List[Dict], entities: List[Dict]) -> List[Hypothesis]:
        """Propone hipótesis abiertas basadas en tensiones."""
        hypotheses = []

        for tension in tensions[:3]:
            pole_a = tension.get('pole_a', '')
            pole_b = tension.get('pole_b', '')

            # Hipótesis de que hay un mecanismo que conecta los polos
            hypotheses.append(Hypothesis(
                statement=f"Posible mecanismo que conecta {pole_a} con {pole_b}",
                based_on=[f"tensión detectada: {pole_a} <-> {pole_b}"],
                questions=[
                    f"¿Cómo se relacionan {pole_a} y {pole_b}?",
                    f"¿Hay transiciones entre {pole_a} y {pole_b}?",
                    "¿Qué condiciones favorecen cada polo?"
                ]
            ))

        # Hipótesis basadas en entidades sin conexiones claras
        entity_names = [e.get('name', '') for e in entities]
        if len(entity_names) >= 2:
            hypotheses.append(Hypothesis(
                statement=f"Posible estructura subyacente que une las entidades",
                based_on=entity_names[:3],
                questions=[
                    "¿Comparten un dominio común?",
                    "¿Hay relaciones no detectadas?",
                    "¿Son manifestaciones del mismo fenómeno?"
                ]
            ))

        return hypotheses

    def _map_paths(self, connections: List[Connection], entities: List[Dict]) -> List[Path]:
        """Mapea caminos posibles entre entidades."""
        paths = []

        if len(connections) < 2:
            return paths

        # Construir grafo simple
        graph = {}
        for conn in connections:
            if conn.source not in graph:
                graph[conn.source] = []
            graph[conn.source].append((conn.target, conn))

        # Encontrar algunos caminos (sin elegir el "mejor")
        for start_node in list(graph.keys())[:3]:
            visited = {start_node}
            current_path = [start_node]
            path_connections = []

            # Caminar hasta 4 pasos
            current = start_node
            for _ in range(4):
                if current not in graph:
                    break
                neighbors = [n for n, c in graph[current] if n not in visited]
                if not neighbors:
                    break
                next_node = neighbors[0]  # tomar cualquiera, no "el mejor"
                conn = [c for n, c in graph[current] if n == next_node][0]

                visited.add(next_node)
                current_path.append(next_node)
                path_connections.append(conn)
                current = next_node

            if len(current_path) > 1:
                paths.append(Path(
                    nodes=current_path,
                    connections=path_connections,
                    length=len(current_path) - 1
                ))

        return paths

    def _identify_domains(self, context: str, entities: List[Dict]) -> Set[str]:
        """Identifica dominios tocados."""
        domains = set()
        context_lower = context.lower()

        for domain in self.KNOWN_DOMAINS:
            if domain in context_lower:
                domains.add(domain)

        # También por entidades
        for entity in entities:
            entity_text = entity.get('name', '').lower()
            for domain in self.KNOWN_DOMAINS:
                if domain in entity_text:
                    domains.add(domain)

        return domains

    def _generate_questions(self, exploration: Exploration) -> List[str]:
        """
        Genera preguntas abiertas.
        Beta SIEMPRE termina con preguntas.
        """
        questions = []

        # Preguntas sobre conexiones
        if exploration.connections:
            questions.append("¿Hay conexiones no detectadas entre estos elementos?")
            questions.append("¿Las conexiones observadas son directas o hay intermediarios?")

        # Preguntas sobre hipótesis
        if exploration.hypotheses:
            questions.append("¿Qué evidencia adicional apoyaría o refutaría estas hipótesis?")

        # Preguntas sobre caminos
        if exploration.paths:
            questions.append("¿Existen caminos alternativos no explorados?")

        # Preguntas sobre dominios
        if exploration.domains_touched:
            domains_list = ', '.join(list(exploration.domains_touched)[:3])
            questions.append(f"¿Cómo interactúan los dominios {domains_list}?")

        # Pregunta meta
        questions.append("¿Qué otras direcciones de exploración son posibles?")

        return questions

    def describe(self, exploration: Exploration) -> str:
        """
        Describe la exploración en lenguaje natural.
        Sin síntesis, solo mapeo.
        """
        lines = []
        lines.append(f"Exploración [{exploration.input_hash}]:")

        # Conexiones
        if exploration.connections:
            lines.append(f"  Conexiones encontradas: {len(exploration.connections)}")
            for c in exploration.connections[:4]:
                lines.append(f"    - {c.source} --[{c.type}]--> {c.target}")

        # Hipótesis
        if exploration.hypotheses:
            lines.append(f"  Hipótesis abiertas: {len(exploration.hypotheses)}")
            for h in exploration.hypotheses[:3]:
                lines.append(f"    - {h.statement}")

        # Caminos
        if exploration.paths:
            lines.append(f"  Caminos mapeados: {len(exploration.paths)}")
            for p in exploration.paths[:2]:
                path_str = ' -> '.join(p.nodes[:4])
                lines.append(f"    - {path_str}")

        # Dominios
        if exploration.domains_touched:
            lines.append(f"  Dominios tocados: {', '.join(exploration.domains_touched)}")

        # Preguntas (siempre)
        lines.append("  Preguntas abiertas:")
        for q in exploration.open_questions[:4]:
            lines.append(f"    ? {q}")

        return '\n'.join(lines)

    def to_dict(self, exploration: Exploration) -> Dict:
        """Convierte exploración a diccionario."""
        return {
            'timestamp': exploration.timestamp.isoformat(),
            'hash': exploration.input_hash,
            'connections': [
                {
                    'source': c.source,
                    'target': c.target,
                    'type': c.type,
                    'strength': c.strength,
                    'evidence': c.evidence
                }
                for c in exploration.connections
            ],
            'hypotheses': [
                {
                    'statement': h.statement,
                    'based_on': h.based_on,
                    'questions': h.questions
                }
                for h in exploration.hypotheses
            ],
            'paths': [
                {
                    'nodes': p.nodes,
                    'length': p.length
                }
                for p in exploration.paths
            ],
            'domains': list(exploration.domains_touched),
            'open_questions': exploration.open_questions
        }


# Singleton
_beta_instance = None

def get_beta() -> BetaExplorer:
    global _beta_instance
    if _beta_instance is None:
        _beta_instance = BetaExplorer()
    return _beta_instance
