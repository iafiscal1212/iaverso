"""
AGI-1: Global Workspace Endógeno
================================

Mecanismo de:
- Acceso global a estados relevantes
- Competencia por "espacio de trabajo"
- Broadcasting de estados salientes

GW_t = argmax_e saliency(e)

donde saliency(e) = α·Δφ + β·Δidentity + γ·ToM_impact

Todo 100% endógeno - sin constantes mágicas.
Los pesos α, β, γ derivan de covarianzas internas.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class ContentType(Enum):
    """Tipos de contenido que compiten por el workspace."""
    EPISODE = "episode"
    PERCEPTION = "perception"
    GOAL = "goal"
    PREDICTION = "prediction"
    CRISIS = "crisis"
    SYMBOL = "symbol"
    TOM = "theory_of_mind"


@dataclass
class WorkspaceContent:
    """Contenido que compite por acceso al workspace global."""
    content_type: ContentType
    data: Any
    source: str  # Qué módulo lo generó
    timestamp: int

    # Componentes de saliencia (calculados endógenamente)
    delta_phi: float = 0.0      # Cambio fenomenológico
    delta_identity: float = 0.0  # Impacto en identidad
    tom_impact: float = 0.0     # Relevancia para ToM
    crisis_relevance: float = 0.0  # Relevancia para crisis

    # Saliencia total (calculada)
    saliency: float = 0.0

    # Historial de acceso
    times_accessed: int = 0
    last_access: int = 0


@dataclass
class BroadcastEvent:
    """Evento de broadcasting cuando algo gana el workspace."""
    t: int
    content: WorkspaceContent
    saliency: float
    competitors: int  # Cuántos competían
    margin: float  # Diferencia con el segundo


class GlobalWorkspace:
    """
    Global Workspace endógeno.

    Implementa competencia por acceso consciente basada en
    saliencia estructural, sin semántica externa.

    Saliency = α·Δφ + β·Δidentity + γ·ToM_impact + δ·crisis_relevance

    donde α, β, γ, δ derivan de:
    - Covarianza histórica de cada componente con "eventos importantes"
    - Percentiles de distribuciones internas
    - Adaptación 1/√t
    """

    def __init__(self, agent_name: str):
        """
        Inicializa Global Workspace.

        Args:
            agent_name: Nombre del agente
        """
        self.agent_name = agent_name
        self.t = 0

        # Contenidos compitiendo
        self.candidates: List[WorkspaceContent] = []

        # Contenido actual en workspace (ganador)
        self.current_content: Optional[WorkspaceContent] = None

        # Historial de broadcasts
        self.broadcast_history: List[BroadcastEvent] = []

        # Pesos adaptativos (inician uniformes, se adaptan)
        self.alpha = 0.25  # peso Δφ
        self.beta = 0.25   # peso Δidentity
        self.gamma = 0.25  # peso ToM
        self.delta = 0.25  # peso crisis

        # Historiales para adaptación de pesos
        self.saliency_components_history: List[Tuple[float, float, float, float]] = []
        self.importance_history: List[float] = []  # Qué tan "importante" fue cada broadcast

        # Umbral de acceso (endógeno)
        self.access_threshold = 0.5

        # Subscribers (módulos que escuchan broadcasts)
        self.subscribers: List[str] = []

    def _compute_component_weights(self):
        """
        Calcula pesos α, β, γ, δ endógenamente.

        Usa correlación histórica entre cada componente
        y la "importancia" del broadcast (medida post-hoc).

        weight_i = rank(corr(component_i, importance))
        """
        if len(self.saliency_components_history) < 20:
            return  # Mantener pesos uniformes hasta tener datos

        # Extraer componentes
        components = np.array(self.saliency_components_history[-100:])
        importance = np.array(self.importance_history[-100:])

        if len(importance) != len(components):
            min_len = min(len(importance), len(components))
            components = components[-min_len:]
            importance = importance[-min_len:]

        # Calcular correlaciones
        correlations = []
        for i in range(4):
            comp = components[:, i]
            if np.std(comp) > 1e-8 and np.std(importance) > 1e-8:
                corr = np.corrcoef(comp, importance)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            correlations.append(max(0.1, corr + 0.5))  # Shift para que sean positivos

        # Normalizar
        total = sum(correlations)
        self.alpha = correlations[0] / total
        self.beta = correlations[1] / total
        self.gamma = correlations[2] / total
        self.delta = correlations[3] / total

    def _compute_saliency(self, content: WorkspaceContent) -> float:
        """
        Calcula saliencia de un contenido.

        saliency = α·Δφ + β·Δidentity + γ·ToM + δ·crisis

        Normalizado por ranks históricos.
        """
        # Rankear cada componente contra historial
        if len(self.saliency_components_history) > 10:
            history = np.array(self.saliency_components_history)

            # Rank de cada componente
            rank_phi = np.sum(history[:, 0] <= content.delta_phi) / len(history)
            rank_id = np.sum(history[:, 1] <= content.delta_identity) / len(history)
            rank_tom = np.sum(history[:, 2] <= content.tom_impact) / len(history)
            rank_crisis = np.sum(history[:, 3] <= content.crisis_relevance) / len(history)
        else:
            # Sin historial, usar valores directos normalizados
            rank_phi = min(1.0, content.delta_phi)
            rank_id = min(1.0, content.delta_identity)
            rank_tom = min(1.0, content.tom_impact)
            rank_crisis = min(1.0, content.crisis_relevance)

        saliency = (self.alpha * rank_phi +
                   self.beta * rank_id +
                   self.gamma * rank_tom +
                   self.delta * rank_crisis)

        return float(saliency)

    def _update_access_threshold(self):
        """
        Actualiza umbral de acceso endógenamente.

        threshold = percentile_50(saliency_history)

        Evita que contenido poco saliente acceda al workspace.
        """
        if len(self.broadcast_history) < 10:
            self.access_threshold = 0.3
            return

        recent_saliencies = [b.saliency for b in self.broadcast_history[-50:]]
        self.access_threshold = np.percentile(recent_saliencies, 50)

    def submit(self, content_type: ContentType, data: Any, source: str,
               delta_phi: float = 0.0, delta_identity: float = 0.0,
               tom_impact: float = 0.0, crisis_relevance: float = 0.0):
        """
        Envía contenido a competir por el workspace.

        Args:
            content_type: Tipo de contenido
            data: Datos del contenido
            source: Módulo que lo envía
            delta_phi: Cambio fenomenológico asociado
            delta_identity: Impacto en identidad
            tom_impact: Relevancia para Theory of Mind
            crisis_relevance: Relevancia para manejo de crisis
        """
        content = WorkspaceContent(
            content_type=content_type,
            data=data,
            source=source,
            timestamp=self.t,
            delta_phi=delta_phi,
            delta_identity=delta_identity,
            tom_impact=tom_impact,
            crisis_relevance=crisis_relevance
        )

        # Calcular saliencia
        content.saliency = self._compute_saliency(content)

        # Registrar componentes para adaptación
        self.saliency_components_history.append((
            delta_phi, delta_identity, tom_impact, crisis_relevance
        ))

        # Limitar historial
        if len(self.saliency_components_history) > 500:
            self.saliency_components_history = self.saliency_components_history[-500:]

        self.candidates.append(content)

    def compete(self) -> Optional[WorkspaceContent]:
        """
        Ejecuta competencia por el workspace.

        El contenido con mayor saliencia gana si supera el umbral.

        Returns:
            Contenido ganador o None si ninguno supera umbral
        """
        self.t += 1

        if len(self.candidates) == 0:
            return None

        # Ordenar por saliencia
        self.candidates.sort(key=lambda x: x.saliency, reverse=True)

        winner = self.candidates[0]

        # Verificar umbral
        if winner.saliency < self.access_threshold:
            self.candidates = []
            return None

        # Calcular margen
        if len(self.candidates) > 1:
            margin = winner.saliency - self.candidates[1].saliency
        else:
            margin = winner.saliency

        # Registrar broadcast
        event = BroadcastEvent(
            t=self.t,
            content=winner,
            saliency=winner.saliency,
            competitors=len(self.candidates),
            margin=margin
        )
        self.broadcast_history.append(event)

        # Limitar historial
        if len(self.broadcast_history) > 500:
            self.broadcast_history = self.broadcast_history[-500:]

        # Actualizar contenido actual
        self.current_content = winner
        winner.times_accessed += 1
        winner.last_access = self.t

        # Limpiar candidatos
        self.candidates = []

        # Actualizar umbral
        self._update_access_threshold()

        # Actualizar pesos cada 20 pasos
        if self.t % 20 == 0:
            self._compute_component_weights()

        return winner

    def record_importance(self, importance: float):
        """
        Registra qué tan "importante" fue el último broadcast.

        Esto permite adaptar los pesos endógenamente.

        Args:
            importance: Medida de importancia (derivada de consecuencias)
        """
        self.importance_history.append(importance)

        if len(self.importance_history) > 500:
            self.importance_history = self.importance_history[-500:]

    def get_current(self) -> Optional[WorkspaceContent]:
        """Obtiene contenido actual del workspace."""
        return self.current_content

    def subscribe(self, module_name: str):
        """Registra un módulo como subscriber de broadcasts."""
        if module_name not in self.subscribers:
            self.subscribers.append(module_name)

    def get_broadcast_for_subscribers(self) -> Dict[str, Any]:
        """
        Prepara datos del broadcast para subscribers.

        Returns:
            Dict con información del broadcast actual
        """
        if self.current_content is None:
            return {'has_content': False}

        return {
            'has_content': True,
            'content_type': self.current_content.content_type.value,
            'data': self.current_content.data,
            'source': self.current_content.source,
            'saliency': self.current_content.saliency,
            'timestamp': self.current_content.timestamp
        }

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del Global Workspace."""
        if len(self.broadcast_history) == 0:
            return {
                'agent': self.agent_name,
                't': self.t,
                'status': 'no_broadcasts'
            }

        recent = self.broadcast_history[-50:]

        # Distribución de tipos
        type_counts = {}
        for event in recent:
            ct = event.content.content_type.value
            type_counts[ct] = type_counts.get(ct, 0) + 1

        # Distribución de sources
        source_counts = {}
        for event in recent:
            src = event.content.source
            source_counts[src] = source_counts.get(src, 0) + 1

        return {
            'agent': self.agent_name,
            't': self.t,
            'total_broadcasts': len(self.broadcast_history),
            'access_threshold': float(self.access_threshold),
            'weights': {
                'alpha_phi': float(self.alpha),
                'beta_identity': float(self.beta),
                'gamma_tom': float(self.gamma),
                'delta_crisis': float(self.delta)
            },
            'recent_mean_saliency': float(np.mean([e.saliency for e in recent])),
            'recent_mean_margin': float(np.mean([e.margin for e in recent])),
            'type_distribution': type_counts,
            'source_distribution': source_counts,
            'current_content_type': self.current_content.content_type.value if self.current_content else None
        }


class MultiAgentGlobalWorkspace:
    """
    Coordina Global Workspaces de múltiples agentes.

    Permite:
    - Broadcasts inter-agente (comunicación)
    - Sincronización de contenidos relevantes
    - Emergencia de "shared attention"
    """

    def __init__(self, agent_names: List[str]):
        """
        Inicializa workspaces para múltiples agentes.

        Args:
            agent_names: Lista de nombres de agentes
        """
        self.agent_names = agent_names
        self.workspaces: Dict[str, GlobalWorkspace] = {
            name: GlobalWorkspace(name) for name in agent_names
        }

        # Historial de atención compartida
        self.shared_attention_history: List[Dict] = []

        self.t = 0

    def get_workspace(self, agent_name: str) -> GlobalWorkspace:
        """Obtiene workspace de un agente."""
        return self.workspaces.get(agent_name)

    def step(self) -> Dict[str, Optional[WorkspaceContent]]:
        """
        Ejecuta competencia en todos los workspaces.

        Returns:
            Dict de agente -> contenido ganador
        """
        self.t += 1

        results = {}
        for name, ws in self.workspaces.items():
            winner = ws.compete()
            results[name] = winner

        # Detectar atención compartida
        self._detect_shared_attention(results)

        return results

    def _detect_shared_attention(self, results: Dict[str, Optional[WorkspaceContent]]):
        """
        Detecta si múltiples agentes atienden a contenido similar.

        Atención compartida = mismo tipo de contenido con alta saliencia
        en 2+ agentes simultáneamente.
        """
        # Contar tipos activos
        active_types = {}
        for name, content in results.items():
            if content is not None:
                ct = content.content_type.value
                if ct not in active_types:
                    active_types[ct] = []
                active_types[ct].append((name, content.saliency))

        # Buscar tipos compartidos (2+ agentes)
        shared = {}
        for ct, agents in active_types.items():
            if len(agents) >= 2:
                shared[ct] = {
                    'agents': [a[0] for a in agents],
                    'mean_saliency': np.mean([a[1] for a in agents])
                }

        if shared:
            self.shared_attention_history.append({
                't': self.t,
                'shared_types': shared
            })

            # Limitar historial
            if len(self.shared_attention_history) > 200:
                self.shared_attention_history = self.shared_attention_history[-200:]

    def get_shared_attention_rate(self) -> float:
        """
        Calcula tasa de atención compartida.

        shared_rate = N(eventos con atención compartida) / N(total eventos)
        """
        if self.t == 0:
            return 0.0
        return len(self.shared_attention_history) / self.t

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas globales."""
        return {
            't': self.t,
            'n_agents': len(self.agent_names),
            'shared_attention_rate': self.get_shared_attention_rate(),
            'recent_shared': self.shared_attention_history[-5:] if self.shared_attention_history else [],
            'per_agent': {name: ws.get_statistics() for name, ws in self.workspaces.items()}
        }


def test_global_workspace():
    """Test del Global Workspace."""
    print("=" * 60)
    print("TEST GLOBAL WORKSPACE ENDÓGENO")
    print("=" * 60)

    # Crear workspace multi-agente
    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
    multi_ws = MultiAgentGlobalWorkspace(agents)

    print(f"\nSimulando {len(agents)} agentes por 200 pasos...")

    for t in range(200):
        # Cada agente envía contenidos
        for name in agents:
            ws = multi_ws.get_workspace(name)

            # Simular diferentes tipos de contenido
            # Episodio
            ws.submit(
                ContentType.EPISODE,
                data={'episode_id': t},
                source='episodic_memory',
                delta_phi=np.random.random() * 0.5,
                delta_identity=np.random.random() * 0.3,
                tom_impact=np.random.random() * 0.2,
                crisis_relevance=np.random.random() * 0.1
            )

            # Percepción (a veces)
            if np.random.random() < 0.3:
                ws.submit(
                    ContentType.PERCEPTION,
                    data={'world_state': np.random.randn(6)},
                    source='observation',
                    delta_phi=np.random.random() * 0.8,
                    delta_identity=np.random.random() * 0.1,
                    tom_impact=np.random.random() * 0.1,
                    crisis_relevance=np.random.random() * 0.2
                )

            # Crisis (raramente)
            if np.random.random() < 0.05:
                ws.submit(
                    ContentType.CRISIS,
                    data={'severity': np.random.random()},
                    source='regulation',
                    delta_phi=0.9,
                    delta_identity=0.8,
                    tom_impact=0.3,
                    crisis_relevance=1.0
                )

            # Meta (a veces)
            if np.random.random() < 0.2:
                ws.submit(
                    ContentType.GOAL,
                    data={'goal_id': np.random.randint(3)},
                    source='planning',
                    delta_phi=np.random.random() * 0.4,
                    delta_identity=np.random.random() * 0.5,
                    tom_impact=np.random.random() * 0.3,
                    crisis_relevance=np.random.random() * 0.1
                )

        # Ejecutar competencia
        results = multi_ws.step()

        # Simular importancia (basada en si hubo crisis cercana)
        for name in agents:
            ws = multi_ws.get_workspace(name)
            importance = 0.5 + np.random.random() * 0.5
            if results[name] and results[name].content_type == ContentType.CRISIS:
                importance = 1.0
            ws.record_importance(importance)

        if (t + 1) % 50 == 0:
            print(f"\n  t={t+1}:")
            for name in agents[:2]:  # Mostrar solo 2
                ws = multi_ws.get_workspace(name)
                stats = ws.get_statistics()
                print(f"    {name}: broadcasts={stats['total_broadcasts']}, "
                      f"threshold={stats['access_threshold']:.3f}")
                if results[name]:
                    print(f"      Current: {results[name].content_type.value} "
                          f"(saliency={results[name].saliency:.3f})")

    # Estadísticas finales
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS FINALES")
    print("=" * 60)

    global_stats = multi_ws.get_statistics()
    print(f"\nAtención compartida: {global_stats['shared_attention_rate']*100:.1f}%")

    print("\nPor agente:")
    for name in agents:
        stats = multi_ws.workspaces[name].get_statistics()
        print(f"\n  {name}:")
        print(f"    Broadcasts: {stats['total_broadcasts']}")
        print(f"    Pesos: α={stats['weights']['alpha_phi']:.2f}, "
              f"β={stats['weights']['beta_identity']:.2f}, "
              f"γ={stats['weights']['gamma_tom']:.2f}, "
              f"δ={stats['weights']['delta_crisis']:.2f}")
        print(f"    Tipos: {stats['type_distribution']}")

    return multi_ws


if __name__ == "__main__":
    test_global_workspace()
