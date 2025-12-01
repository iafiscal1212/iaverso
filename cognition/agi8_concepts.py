"""
AGI-8: Conceptos Internos (Grafo de Co-ocurrencias)
===================================================

Grafo de co-ocurrencias episodio-símbolo-skill-régimen.

Evento estructural:
    x_t = (EPI_t, SYM_t, SK_t, REG_t)

Co-ocurrencia:
    C = Σ_t x_t x_t^T
    C̃_ij = C_ij / √(C_ii * C_jj)

Conceptos:
    Autovectores con autovalores ≥ mediana
    Cada comunidad = concepto emergente

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum


class ItemType(Enum):
    """Tipos de items en el grafo."""
    EPISODE = "episode"
    SYMBOL = "symbol"
    SKILL = "skill"
    REGIME = "regime"


@dataclass
class ConceptNode:
    """Nodo en el grafo de conceptos."""
    node_id: int
    item_type: ItemType
    item_reference: int  # ID del episodio/símbolo/skill/régimen
    activation_count: int = 0
    last_activation: int = 0


@dataclass
class EmergentConcept:
    """Concepto emergente como comunidad en el grafo."""
    concept_id: int
    nodes: List[int]  # IDs de nodos miembros
    centroid: np.ndarray  # Autovector asociado
    eigenvalue: float
    coherence: float  # Qué tan fuerte es la comunidad
    stability: float = 0.0  # Persistencia temporal


class ConceptGraph:
    """
    Grafo de conceptos internos basado en co-ocurrencias.

    Detecta comunidades que emergen de la co-activación
    de episodios, símbolos, skills y regímenes.
    """

    def __init__(self, agent_name: str, max_nodes: int = 200):
        """
        Inicializa grafo de conceptos.

        Args:
            agent_name: Nombre del agente
            max_nodes: Máximo de nodos a rastrear
        """
        self.agent_name = agent_name
        self.max_nodes = max_nodes

        # Nodos del grafo
        self.nodes: Dict[int, ConceptNode] = {}
        self.next_node_id = 0

        # Mapeo item -> nodo
        self.item_to_node: Dict[Tuple[ItemType, int], int] = {}

        # Matriz de co-ocurrencia (se construye dinámicamente)
        self.cooccurrence: Optional[np.ndarray] = None
        self.cooccurrence_normalized: Optional[np.ndarray] = None

        # Conceptos emergentes
        self.concepts: Dict[int, EmergentConcept] = {}
        self.next_concept_id = 0

        # Historial de eventos para ventana
        self.event_history: List[np.ndarray] = []

        self.t = 0

    def _get_or_create_node(self, item_type: ItemType, item_ref: int) -> int:
        """Obtiene o crea nodo para un item."""
        key = (item_type, item_ref)
        if key in self.item_to_node:
            return self.item_to_node[key]

        # Crear nuevo nodo
        node_id = self.next_node_id
        self.next_node_id += 1

        self.nodes[node_id] = ConceptNode(
            node_id=node_id,
            item_type=item_type,
            item_reference=item_ref
        )
        self.item_to_node[key] = node_id

        return node_id

    def _build_event_vector(self, active_nodes: List[int]) -> np.ndarray:
        """
        Construye vector de evento x_t.

        Dimensión = número de nodos activos hasta ahora.
        """
        n = len(self.nodes)
        x = np.zeros(n)
        for node_id in active_nodes:
            if node_id < n:
                x[node_id] = 1.0
        return x

    def record_event(self, episode_id: Optional[int] = None,
                    symbol_ids: Optional[List[int]] = None,
                    skill_ids: Optional[List[int]] = None,
                    regime_id: Optional[int] = None):
        """
        Registra un evento estructural.

        Args:
            episode_id: ID del episodio actual (o None)
            symbol_ids: IDs de símbolos activos
            skill_ids: IDs de skills usados
            regime_id: ID del régimen actual
        """
        self.t += 1

        active_nodes = []

        # Agregar nodos activos
        if episode_id is not None:
            node_id = self._get_or_create_node(ItemType.EPISODE, episode_id)
            active_nodes.append(node_id)
            self.nodes[node_id].activation_count += 1
            self.nodes[node_id].last_activation = self.t

        if symbol_ids:
            for sym_id in symbol_ids:
                node_id = self._get_or_create_node(ItemType.SYMBOL, sym_id)
                active_nodes.append(node_id)
                self.nodes[node_id].activation_count += 1
                self.nodes[node_id].last_activation = self.t

        if skill_ids:
            for sk_id in skill_ids:
                node_id = self._get_or_create_node(ItemType.SKILL, sk_id)
                active_nodes.append(node_id)
                self.nodes[node_id].activation_count += 1
                self.nodes[node_id].last_activation = self.t

        if regime_id is not None:
            node_id = self._get_or_create_node(ItemType.REGIME, regime_id)
            active_nodes.append(node_id)
            self.nodes[node_id].activation_count += 1
            self.nodes[node_id].last_activation = self.t

        # Construir vector de evento
        x = self._build_event_vector(active_nodes)
        self.event_history.append(x)

        # Limitar historial
        window = int(np.ceil(np.sqrt(self.t + 1)))
        if len(self.event_history) > window * 2:
            self.event_history = self.event_history[-window*2:]

        # Actualizar matriz de co-ocurrencia
        self._update_cooccurrence()

        # Detectar conceptos periódicamente
        if self.t % 20 == 0:
            self._detect_concepts()

        # Limitar nodos
        if len(self.nodes) > self.max_nodes:
            self._prune_nodes()

    def _update_cooccurrence(self):
        """
        Actualiza matriz de co-ocurrencia.

        C = Σ_t x_t x_t^T
        C̃_ij = C_ij / √(C_ii * C_jj)
        """
        if len(self.event_history) < 5:
            return

        n = len(self.nodes)
        if n == 0:
            return

        # Alinear dimensiones
        events = []
        for x in self.event_history:
            if len(x) < n:
                x_padded = np.zeros(n)
                x_padded[:len(x)] = x
                events.append(x_padded)
            else:
                events.append(x[:n])

        events = np.array(events)

        # Calcular C = Σ x x^T
        self.cooccurrence = events.T @ events

        # Normalizar: C̃_ij = C_ij / √(C_ii * C_jj)
        diag = np.diag(self.cooccurrence)
        diag_sqrt = np.sqrt(diag + 1e-8)
        self.cooccurrence_normalized = self.cooccurrence / np.outer(diag_sqrt, diag_sqrt)

        # Manejar NaN
        self.cooccurrence_normalized = np.nan_to_num(self.cooccurrence_normalized, 0)

    def _detect_concepts(self):
        """
        Detecta conceptos como comunidades en el grafo.

        Usa autovectores con autovalores ≥ mediana.
        """
        if self.cooccurrence_normalized is None:
            return

        C = self.cooccurrence_normalized

        if C.shape[0] < 3:
            return

        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(C)
        except:
            return

        # Conceptos = autovectores con λ ≥ mediana
        median_eig = np.median(eigenvalues)
        significant_indices = np.where(eigenvalues >= median_eig)[0]

        # Guardar conceptos anteriores para medir estabilidad
        old_concepts = {c.concept_id: c.centroid for c in self.concepts.values()}

        # Limpiar conceptos
        self.concepts.clear()

        for idx in significant_indices:
            eigenvector = eigenvectors[:, idx]
            eigenvalue = eigenvalues[idx]

            # Encontrar nodos miembros (componentes significativos)
            threshold = np.percentile(np.abs(eigenvector), 75)
            member_nodes = np.where(np.abs(eigenvector) >= threshold)[0].tolist()

            if len(member_nodes) < 2:
                continue

            # Coherencia = varianza de componentes (baja varianza = alta coherencia)
            member_values = eigenvector[member_nodes]
            coherence = 1.0 / (1.0 + np.var(member_values))

            # Estabilidad respecto a conceptos anteriores
            stability = 0.0
            for old_centroid in old_concepts.values():
                if len(old_centroid) == len(eigenvector):
                    similarity = abs(np.dot(eigenvector, old_centroid))
                    stability = max(stability, similarity)

            concept = EmergentConcept(
                concept_id=self.next_concept_id,
                nodes=member_nodes,
                centroid=eigenvector,
                eigenvalue=float(eigenvalue),
                coherence=float(coherence),
                stability=float(stability)
            )
            self.concepts[self.next_concept_id] = concept
            self.next_concept_id += 1

    def _prune_nodes(self):
        """Elimina nodos menos activos."""
        if len(self.nodes) <= self.max_nodes:
            return

        # Ordenar por activación reciente
        sorted_nodes = sorted(
            self.nodes.items(),
            key=lambda x: (x[1].last_activation, x[1].activation_count),
            reverse=True
        )

        # Mantener top nodes
        keep_ids = set(node_id for node_id, _ in sorted_nodes[:self.max_nodes])

        # Eliminar otros
        to_remove = [nid for nid in self.nodes if nid not in keep_ids]
        for nid in to_remove:
            node = self.nodes[nid]
            key = (node.item_type, node.item_reference)
            if key in self.item_to_node:
                del self.item_to_node[key]
            del self.nodes[nid]

    def get_concept_for_item(self, item_type: ItemType, item_ref: int) -> Optional[int]:
        """
        Obtiene el concepto al que pertenece un item.

        Returns:
            ID del concepto o None
        """
        key = (item_type, item_ref)
        if key not in self.item_to_node:
            return None

        node_id = self.item_to_node[key]

        for concept_id, concept in self.concepts.items():
            if node_id in concept.nodes:
                return concept_id

        return None

    def get_related_items(self, item_type: ItemType, item_ref: int,
                         target_type: Optional[ItemType] = None) -> List[int]:
        """
        Obtiene items relacionados vía conceptos.

        Args:
            item_type: Tipo del item origen
            item_ref: Referencia del item origen
            target_type: Tipo de items a buscar (opcional)

        Returns:
            Lista de referencias de items relacionados
        """
        concept_id = self.get_concept_for_item(item_type, item_ref)
        if concept_id is None:
            return []

        concept = self.concepts[concept_id]
        related = []

        for node_id in concept.nodes:
            if node_id not in self.nodes:
                continue
            node = self.nodes[node_id]
            if target_type is None or node.item_type == target_type:
                if node.item_reference != item_ref or node.item_type != item_type:
                    related.append(node.item_reference)

        return related

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del grafo de conceptos."""
        if not self.nodes:
            return {
                'agent': self.agent_name,
                't': self.t,
                'n_nodes': 0,
                'n_concepts': 0
            }

        # Estadísticas por tipo
        type_counts = {t.value: 0 for t in ItemType}
        for node in self.nodes.values():
            type_counts[node.item_type.value] += 1

        # Estadísticas de conceptos
        concept_info = []
        for concept in self.concepts.values():
            concept_info.append({
                'id': concept.concept_id,
                'size': len(concept.nodes),
                'eigenvalue': concept.eigenvalue,
                'coherence': concept.coherence,
                'stability': concept.stability
            })

        return {
            'agent': self.agent_name,
            't': self.t,
            'n_nodes': len(self.nodes),
            'n_concepts': len(self.concepts),
            'node_types': type_counts,
            'concepts': sorted(concept_info, key=lambda x: x['eigenvalue'], reverse=True),
            'mean_coherence': float(np.mean([c['coherence'] for c in concept_info])) if concept_info else 0,
            'mean_stability': float(np.mean([c['stability'] for c in concept_info])) if concept_info else 0
        }


def test_concepts():
    """Test de conceptos internos."""
    print("=" * 60)
    print("TEST AGI-8: CONCEPTOS INTERNOS (GRAFO)")
    print("=" * 60)

    graph = ConceptGraph("NEO")

    print("\nSimulando 500 eventos con co-ocurrencias...")

    # Simular patrones de co-ocurrencia
    for t in range(500):
        # Patrón 1: episodios 0-4 co-ocurren con símbolos 0-2 y skill 0
        if t % 10 < 5:
            episode_id = t % 5
            symbol_ids = [t % 3]
            skill_ids = [0]
            regime_id = 0
        # Patrón 2: episodios 5-9 co-ocurren con símbolos 3-5 y skill 1
        elif t % 10 < 8:
            episode_id = 5 + (t % 5)
            symbol_ids = [3 + (t % 3)]
            skill_ids = [1]
            regime_id = 1
        # Patrón 3: aleatorio
        else:
            episode_id = np.random.randint(10, 20)
            symbol_ids = [np.random.randint(6, 10)]
            skill_ids = [np.random.randint(2, 5)]
            regime_id = np.random.randint(0, 3)

        graph.record_event(
            episode_id=episode_id,
            symbol_ids=symbol_ids,
            skill_ids=skill_ids,
            regime_id=regime_id
        )

        if (t + 1) % 100 == 0:
            stats = graph.get_statistics()
            print(f"  t={t+1}: {stats['n_nodes']} nodos, {stats['n_concepts']} conceptos")

    # Resultados finales
    stats = graph.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS CONCEPTOS INTERNOS")
    print("=" * 60)

    print(f"\n  Nodos totales: {stats['n_nodes']}")
    print(f"  Conceptos emergentes: {stats['n_concepts']}")
    print(f"  Coherencia media: {stats['mean_coherence']:.3f}")
    print(f"  Estabilidad media: {stats['mean_stability']:.3f}")

    print("\n  Distribución por tipo:")
    for type_name, count in stats['node_types'].items():
        print(f"    {type_name}: {count}")

    print("\n  Top 5 conceptos:")
    for concept in stats['concepts'][:5]:
        print(f"    Concepto {concept['id']}: size={concept['size']}, "
              f"λ={concept['eigenvalue']:.3f}, coh={concept['coherence']:.3f}")

    # Verificar relaciones
    print("\n  Items relacionados con episodio 0:")
    related = graph.get_related_items(ItemType.EPISODE, 0)
    print(f"    {len(related)} items relacionados")

    if stats['n_concepts'] > 0:
        print("\n  ✓ Conceptos emergiendo correctamente")
    else:
        print("\n  ⚠️ No se detectaron conceptos")

    return graph


if __name__ == "__main__":
    test_concepts()
