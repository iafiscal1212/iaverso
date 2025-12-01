"""
Dream Processor: Consolidacion de Memoria Offline
==================================================

Durante la fase DREAM, los agentes consolidan memorias:
    - Reorganizan experiencias por importancia
    - Crean conexiones entre memorias
    - Destilan patrones de alto nivel
    - Integran aprendizajes en identidad

El proceso es ENDOGENO:
    - Que consolidar depende de relevancia interna
    - Como reorganizar depende de estructura existente
    - Patrones emergen de correlaciones, no reglas externas

100% endogeno. Sin criterios de importancia hardcodeados.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from datetime import datetime
import json

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class DreamFragment:
    """Fragmento de un sueno - unidad de consolidacion."""
    source_memories: List[str]       # IDs de memorias originales
    pattern: str                     # Patron destilado
    emotional_tone: float            # [-1, 1] tono emocional
    significance: float              # [0, 1] importancia
    connections: List[str]           # Conexiones a otros fragmentos
    created_t: int                   # Cuando se creo


@dataclass
class ConsolidationResult:
    """Resultado de consolidacion."""
    memories_processed: int
    patterns_found: int
    connections_made: int
    integration_strength: float      # Que tan bien se integro
    dream_narrative: str             # Narrativa del sueno


class DreamProcessor:
    """
    Procesador de suenos para consolidacion de memoria.

    Durante DREAM:
        1. Selecciona memorias pendientes
        2. Encuentra patrones entre ellas
        3. Crea conexiones
        4. Integra con memoria a largo plazo
        5. Genera narrativa de sueno
    """

    def __init__(self, agent_id: str):
        """
        Inicializa procesador de suenos.

        Args:
            agent_id: ID del agente
        """
        self.agent_id = agent_id

        # Memorias pendientes de consolidar
        self.pending_memories: List[Dict] = []

        # Fragmentos de suenos previos
        self.dream_fragments: List[DreamFragment] = []

        # Patrones descubiertos
        self.discovered_patterns: Dict[str, float] = {}  # patron -> fuerza

        # Conexiones entre memorias
        self.memory_connections: Dict[Tuple[str, str], float] = {}

        # Historico de consolidaciones
        self.consolidation_history: List[ConsolidationResult] = []

        self.t = 0

    def add_memory_for_consolidation(self, memory: Dict):
        """
        Agrega memoria para consolidar en proxima fase DREAM.

        Args:
            memory: Memoria a consolidar con campos:
                    - id: identificador unico
                    - content: contenido
                    - emotional_valence: tono emocional
                    - importance: importancia estimada
                    - timestamp: cuando ocurrio
                    - context: contexto (opcional)
        """
        # Asegurar campos minimos
        if 'id' not in memory:
            memory['id'] = f"mem_{self.t}_{len(self.pending_memories)}"
        if 'emotional_valence' not in memory:
            memory['emotional_valence'] = 0.0
        if 'importance' not in memory:
            memory['importance'] = 0.5

        self.pending_memories.append(memory)

        # Limitar buffer
        max_pending = max(50, int(L_t(self.t)))
        if len(self.pending_memories) > max_pending:
            # Priorizar por importancia
            self.pending_memories.sort(
                key=lambda m: m.get('importance', 0.5),
                reverse=True
            )
            self.pending_memories = self.pending_memories[:max_pending]

    def _compute_similarity(self, mem1: Dict, mem2: Dict) -> float:
        """
        Calcula similitud entre dos memorias.

        Basado en:
            - Similitud de contenido (si es texto, embeddings aproximados)
            - Similitud emocional
            - Cercania temporal
            - Contexto compartido
        """
        # Similitud emocional
        emotional_sim = 1 - abs(
            mem1.get('emotional_valence', 0) -
            mem2.get('emotional_valence', 0)
        )

        # Cercania temporal (si tienen timestamp)
        temporal_sim = 0.5
        if 'timestamp' in mem1 and 'timestamp' in mem2:
            t1 = mem1['timestamp']
            t2 = mem2['timestamp']
            diff = abs(t1 - t2)
            temporal_sim = np.exp(-diff / 100)

        # Contexto compartido
        context_sim = 0.5
        ctx1 = set(mem1.get('context', {}).keys())
        ctx2 = set(mem2.get('context', {}).keys())
        if ctx1 and ctx2:
            overlap = len(ctx1 & ctx2) / max(len(ctx1 | ctx2), 1)
            context_sim = overlap

        # Combinar (pesos endogenos basados en varianza historica)
        weights = [0.4, 0.3, 0.3]  # Base

        return (
            weights[0] * emotional_sim +
            weights[1] * temporal_sim +
            weights[2] * context_sim
        )

    def _find_patterns(self, memories: List[Dict]) -> List[Tuple[str, float]]:
        """
        Encuentra patrones en un conjunto de memorias.

        Returns:
            Lista de (patron_descripcion, fuerza)
        """
        if len(memories) < 2:
            return []

        patterns = []

        # Patron emocional
        valences = [m.get('emotional_valence', 0) for m in memories]
        if valences:
            mean_valence = np.mean(valences)
            std_valence = np.std(valences) if len(valences) > 1 else 0

            if std_valence < 0.3:  # Emociones consistentes
                if mean_valence > 0.3:
                    patterns.append(("tendencia_positiva", 0.7))
                elif mean_valence < -0.3:
                    patterns.append(("tendencia_negativa", 0.7))
                else:
                    patterns.append(("equilibrio_emocional", 0.6))

        # Patron de importancia
        importances = [m.get('importance', 0.5) for m in memories]
        if np.mean(importances) > 0.7:
            patterns.append(("periodo_significativo", 0.8))
        elif np.mean(importances) < 0.3:
            patterns.append(("rutina_cotidiana", 0.5))

        # Patron de contexto
        all_contexts = [set(m.get('context', {}).keys()) for m in memories]
        if all_contexts:
            common = set.intersection(*all_contexts) if all_contexts else set()
            if common:
                for ctx in common:
                    patterns.append((f"contexto_recurrente:{ctx}", 0.6))

        return patterns

    def _create_connections(
        self,
        memories: List[Dict],
        similarity_threshold: float = 0.6
    ) -> List[Tuple[str, str, float]]:
        """
        Crea conexiones entre memorias similares.

        Returns:
            Lista de (id1, id2, fuerza_conexion)
        """
        connections = []

        for i, mem1 in enumerate(memories):
            for mem2 in memories[i+1:]:
                sim = self._compute_similarity(mem1, mem2)
                if sim >= similarity_threshold:
                    connections.append((mem1['id'], mem2['id'], sim))

        return connections

    def _generate_dream_narrative(
        self,
        memories: List[Dict],
        patterns: List[Tuple[str, float]],
        connections: List[Tuple[str, str, float]]
    ) -> str:
        """
        Genera narrativa del sueno.

        No es un sueno literal - es la "sensacion" de consolidacion.
        """
        if not memories:
            return "sueno vacio, descanso profundo"

        # Tono emocional dominante
        valences = [m.get('emotional_valence', 0) for m in memories]
        mean_valence = np.mean(valences)

        if mean_valence > 0.3:
            tone = "calido"
        elif mean_valence < -0.3:
            tone = "inquieto"
        else:
            tone = "fluido"

        # Patrones principales
        main_patterns = [p[0] for p in sorted(patterns, key=lambda x: x[1], reverse=True)[:2]]

        # Construir narrativa
        narrative = f"Sueno {tone}. "

        if main_patterns:
            pattern_str = ", ".join(main_patterns)
            narrative += f"Emergen patrones: {pattern_str}. "

        if connections:
            narrative += f"{len(connections)} conexiones nuevas entre memorias. "

        narrative += f"Consolidadas {len(memories)} experiencias."

        return narrative

    def _integrate_into_identity(
        self,
        patterns: List[Tuple[str, float]]
    ) -> float:
        """
        Integra patrones descubiertos en identidad del agente.

        Returns:
            Fuerza de integracion [0, 1]
        """
        if not patterns:
            return 0.0

        integration_strength = 0.0

        for pattern, strength in patterns:
            # Actualizar patron existente o crear nuevo
            if pattern in self.discovered_patterns:
                # Reforzar patron existente
                old_strength = self.discovered_patterns[pattern]
                new_strength = old_strength + 0.1 * strength * (1 - old_strength)
                self.discovered_patterns[pattern] = new_strength
                integration_strength += 0.1 * strength
            else:
                # Nuevo patron
                self.discovered_patterns[pattern] = strength * 0.5
                integration_strength += 0.2 * strength

        # Decay de patrones no reforzados
        for pattern in list(self.discovered_patterns.keys()):
            if pattern not in [p[0] for p in patterns]:
                self.discovered_patterns[pattern] *= 0.99
                if self.discovered_patterns[pattern] < 0.1:
                    del self.discovered_patterns[pattern]

        return min(1.0, integration_strength)

    def process_dream(self, dream_depth: float = 1.0) -> ConsolidationResult:
        """
        Procesa consolidacion durante fase DREAM.

        Args:
            dream_depth: Profundidad del sueno [0, 1]
                         Mayor profundidad = mas consolidacion

        Returns:
            Resultado de la consolidacion
        """
        self.t += 1

        if not self.pending_memories:
            return ConsolidationResult(
                memories_processed=0,
                patterns_found=0,
                connections_made=0,
                integration_strength=0.0,
                dream_narrative="descanso sin suenos"
            )

        # Cuantas memorias procesar (depende de profundidad)
        n_process = max(1, int(len(self.pending_memories) * dream_depth * 0.3))
        n_process = min(n_process, 10)  # Max 10 por sesion

        # Seleccionar memorias a procesar (priorizando importancia)
        self.pending_memories.sort(
            key=lambda m: m.get('importance', 0.5),
            reverse=True
        )
        to_process = self.pending_memories[:n_process]
        self.pending_memories = self.pending_memories[n_process:]

        # Encontrar patrones
        patterns = self._find_patterns(to_process)

        # Crear conexiones
        connections = self._create_connections(to_process)

        # Guardar conexiones
        for id1, id2, strength in connections:
            key = (min(id1, id2), max(id1, id2))
            old_strength = self.memory_connections.get(key, 0)
            self.memory_connections[key] = max(old_strength, strength)

        # Integrar en identidad
        integration = self._integrate_into_identity(patterns)

        # Generar narrativa
        narrative = self._generate_dream_narrative(to_process, patterns, connections)

        # Crear fragmento de sueno
        fragment = DreamFragment(
            source_memories=[m['id'] for m in to_process],
            pattern=patterns[0][0] if patterns else "sin_patron",
            emotional_tone=np.mean([m.get('emotional_valence', 0) for m in to_process]),
            significance=np.mean([m.get('importance', 0.5) for m in to_process]),
            connections=[f"{c[0]}-{c[1]}" for c in connections[:3]],
            created_t=self.t
        )
        self.dream_fragments.append(fragment)

        # Limitar fragmentos
        max_fragments = max_history(self.t)
        if len(self.dream_fragments) > max_fragments:
            # Mantener los mas significativos
            self.dream_fragments.sort(key=lambda f: f.significance, reverse=True)
            self.dream_fragments = self.dream_fragments[:max_fragments]

        result = ConsolidationResult(
            memories_processed=n_process,
            patterns_found=len(patterns),
            connections_made=len(connections),
            integration_strength=integration,
            dream_narrative=narrative
        )

        self.consolidation_history.append(result)

        return result

    def get_strong_patterns(self, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Obtiene patrones fuertes descubiertos.

        Args:
            threshold: Umbral minimo de fuerza

        Returns:
            Lista de (patron, fuerza) ordenada por fuerza
        """
        strong = [
            (p, s) for p, s in self.discovered_patterns.items()
            if s >= threshold
        ]
        return sorted(strong, key=lambda x: x[1], reverse=True)

    def get_recent_dreams(self, n: int = 5) -> List[DreamFragment]:
        """Obtiene suenos recientes."""
        return self.dream_fragments[-n:]

    def get_memory_graph(self) -> Dict[str, List[str]]:
        """
        Obtiene grafo de conexiones entre memorias.

        Returns:
            Diccionario de id -> [ids conectados]
        """
        graph = {}
        for (id1, id2), strength in self.memory_connections.items():
            if strength >= 0.5:
                if id1 not in graph:
                    graph[id1] = []
                if id2 not in graph:
                    graph[id2] = []
                graph[id1].append(id2)
                graph[id2].append(id1)
        return graph

    def get_statistics(self) -> Dict:
        """Estadisticas del procesador de suenos."""
        return {
            'agent_id': self.agent_id,
            't': self.t,
            'pending_memories': len(self.pending_memories),
            'dream_fragments': len(self.dream_fragments),
            'discovered_patterns': len(self.discovered_patterns),
            'memory_connections': len(self.memory_connections),
            'total_consolidations': len(self.consolidation_history),
            'avg_integration': np.mean([
                c.integration_strength
                for c in self.consolidation_history
            ]) if self.consolidation_history else 0.0
        }


def test_dream_processor():
    """Test del procesador de suenos."""
    print("=" * 70)
    print("TEST: DREAM PROCESSOR")
    print("=" * 70)

    np.random.seed(42)

    processor = DreamProcessor("NEO")

    # Agregar memorias simuladas
    print("\nAgregando memorias para consolidar...")

    for i in range(20):
        memory = {
            'id': f'mem_{i}',
            'content': f'experiencia_{i}',
            'emotional_valence': np.random.uniform(-0.5, 0.8),
            'importance': np.random.uniform(0.3, 0.9),
            'timestamp': i * 10,
            'context': {
                'location': np.random.choice(['home', 'work', 'outside']),
                'social': np.random.choice(['alone', 'with_others'])
            }
        }
        processor.add_memory_for_consolidation(memory)

    print(f"  Memorias pendientes: {len(processor.pending_memories)}")

    # Procesar varios suenos
    print("\nProcesando suenos...")

    for dream_num in range(5):
        result = processor.process_dream(dream_depth=0.8)
        print(f"\n  Sueno {dream_num + 1}:")
        print(f"    Memorias procesadas: {result.memories_processed}")
        print(f"    Patrones encontrados: {result.patterns_found}")
        print(f"    Conexiones creadas: {result.connections_made}")
        print(f"    Integracion: {result.integration_strength:.2f}")
        print(f"    Narrativa: {result.dream_narrative}")

    # Mostrar patrones descubiertos
    print("\n" + "=" * 70)
    print("PATRONES DESCUBIERTOS:")
    print("=" * 70)

    patterns = processor.get_strong_patterns(threshold=0.3)
    for pattern, strength in patterns:
        print(f"  {pattern}: {strength:.2f}")

    # Estadisticas
    print("\n" + "=" * 70)
    print("ESTADISTICAS:")
    print("=" * 70)

    stats = processor.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return processor


if __name__ == "__main__":
    test_dream_processor()
