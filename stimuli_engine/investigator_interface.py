"""
INVESTIGATOR INTERFACE - Interfaz Limpia con Investigadores
============================================================

Define el contrato entre StimulusEngine e Investigadores.

EL INVESTIGADOR SOLO RECIBE:
- Estructuras matemáticas (series, matrices, grafos)
- Metadatos de procedencia

EL INVESTIGADOR NO RECIBE:
- Nombres semánticos
- Hipótesis sugeridas
- Interpretaciones

El investigador decide autónomamente:
- Qué relaciones buscar
- Qué hipótesis formular
- Qué experimentos ejecutar
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol
import numpy as np
from abc import ABC, abstractmethod

from .stimulus_engine import StimulusBundle, TimeSeries, Matrix, Graph


# =============================================================================
# PROTOCOLO DEL INVESTIGADOR
# =============================================================================

class InvestigatorProtocol(Protocol):
    """
    Protocolo que debe cumplir cualquier investigador.

    El investigador SOLO tiene estos métodos públicos para recibir estímulos.
    """

    def observe(self, stimuli: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recibe estímulos y los procesa internamente.

        Args:
            stimuli: Diccionario con series, matrices, grafos

        Returns:
            Resultado de la observación (sin interpretación semántica)
        """
        ...

    def get_status(self) -> Dict[str, Any]:
        """Retorna estado interno del investigador."""
        ...


# =============================================================================
# ADAPTADOR DE ESTÍMULOS
# =============================================================================

@dataclass
class StimuliPacket:
    """
    Paquete de estímulos listo para entregar al investigador.

    Formato estandarizado que cualquier investigador puede consumir.
    """
    # Series temporales como arrays puros
    series: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

    # Matrices como arrays 2D
    matrices: Dict[str, np.ndarray] = field(default_factory=dict)

    # Grafos como listas de adyacencia
    graphs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Escalares
    scalars: Dict[str, float] = field(default_factory=dict)

    # Metadatos de procedencia (sin semántica)
    provenance: Dict[str, str] = field(default_factory=dict)

    # Estadísticas derivadas (calculadas automáticamente)
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para observe()."""
        return {
            'series': {
                sid: {
                    't': s['t'].tolist() if hasattr(s['t'], 'tolist') else s['t'],
                    'values': s['values'].tolist() if hasattr(s['values'], 'tolist') else s['values'],
                }
                for sid, s in self.series.items()
            },
            'matrices': {
                mid: m.tolist() if hasattr(m, 'tolist') else m
                for mid, m in self.matrices.items()
            },
            'graphs': self.graphs,
            'scalars': self.scalars,
            'provenance': self.provenance,
            'statistics': self.statistics,
        }


class StimulusAdapter:
    """
    Adapta StimulusBundle al formato que espera el investigador.

    RESPONSABILIDADES:
    - Convertir estructuras a formato estándar
    - Calcular estadísticas básicas
    - NO interpretar ni sugerir

    Este es el ÚNICO punto de contacto entre el motor y los investigadores.
    """

    def __init__(self):
        pass

    def adapt(self, bundle: StimulusBundle) -> StimuliPacket:
        """
        Adapta un bundle a paquete de estímulos.

        Args:
            bundle: StimulusBundle del motor

        Returns:
            StimuliPacket listo para investigador
        """
        packet = StimuliPacket()

        # Adaptar series
        for sid, series in bundle.series.items():
            packet.series[sid] = {
                't': series.t,
                'values': series.values,
            }

            # Calcular estadísticas básicas
            values = series.values[~np.isnan(series.values)]
            if len(values) > 0:
                packet.statistics[sid] = {
                    'n': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }

            # Registrar procedencia (sin semántica)
            if series.provenance:
                packet.provenance[sid] = series.provenance.source

        # Adaptar matrices
        for mid, matrix in bundle.matrices.items():
            packet.matrices[mid] = matrix.data

            if matrix.provenance:
                packet.provenance[mid] = matrix.provenance.source

        # Adaptar grafos
        for gid, graph in bundle.graphs.items():
            packet.graphs[gid] = {
                'nodes': graph.nodes,
                'edges': graph.edges,
                'directed': graph.directed,
            }

            if graph.provenance:
                packet.provenance[gid] = graph.provenance.source

        # Copiar escalares
        packet.scalars = bundle.scalars.copy()

        return packet


# =============================================================================
# DISPATCHER A MÚLTIPLES INVESTIGADORES
# =============================================================================

class InvestigatorDispatcher:
    """
    Distribuye estímulos a múltiples investigadores.

    Cada investigador recibe los mismos datos y decide
    autónomamente qué hacer con ellos.

    NO hay dirección ni sugerencias.
    """

    def __init__(self):
        self.investigators: Dict[str, Any] = {}
        self.adapter = StimulusAdapter()
        self.delivery_log: List[Dict[str, Any]] = []

    def register(self, investigator_id: str, investigator: Any):
        """
        Registra un investigador.

        Args:
            investigator_id: Identificador anónimo
            investigator: Objeto que cumple InvestigatorProtocol
        """
        self.investigators[investigator_id] = investigator

    def unregister(self, investigator_id: str):
        """Desregistra un investigador."""
        if investigator_id in self.investigators:
            del self.investigators[investigator_id]

    def broadcast(
        self,
        bundle: StimulusBundle
    ) -> Dict[str, Dict[str, Any]]:
        """
        Envía estímulos a todos los investigadores registrados.

        Args:
            bundle: Bundle de estímulos

        Returns:
            Diccionario con respuestas de cada investigador
        """
        packet = self.adapter.adapt(bundle)
        stimuli_dict = packet.to_dict()

        responses = {}
        for inv_id, investigator in self.investigators.items():
            if hasattr(investigator, 'observe'):
                try:
                    response = investigator.observe(stimuli_dict)
                    responses[inv_id] = {
                        'status': 'success',
                        'response': response,
                    }
                except Exception as e:
                    responses[inv_id] = {
                        'status': 'error',
                        'error': str(e),
                    }
            else:
                responses[inv_id] = {
                    'status': 'no_observe_method',
                }

        # Registrar entrega
        self.delivery_log.append({
            'bundle_id': bundle.id,
            'n_series': len(bundle.series),
            'n_matrices': len(bundle.matrices),
            'n_graphs': len(bundle.graphs),
            'n_investigators': len(self.investigators),
            'responses': {k: v['status'] for k, v in responses.items()},
        })

        return responses

    def get_investigator_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene estado de todos los investigadores."""
        statuses = {}
        for inv_id, investigator in self.investigators.items():
            if hasattr(investigator, 'get_status'):
                statuses[inv_id] = investigator.get_status()
            else:
                statuses[inv_id] = {'status': 'unknown'}
        return statuses


# =============================================================================
# BASE PARA INVESTIGADORES
# =============================================================================

class BaseInvestigator(ABC):
    """
    Clase base para investigadores.

    Define la interfaz mínima que debe cumplir cualquier investigador.
    El investigador NO recibe semántica, solo matemáticas.
    """

    def __init__(self, investigator_id: str):
        self.investigator_id = investigator_id
        self.observations: List[Dict[str, Any]] = []
        self.t = 0

    @abstractmethod
    def observe(self, stimuli: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa estímulos recibidos.

        El investigador decide internamente qué hacer.
        """
        pass

    @abstractmethod
    def decide_what_to_investigate(self) -> Optional[Dict[str, Any]]:
        """
        Decide autónomamente qué investigar.

        Returns:
            Descripción de la investigación a realizar, o None
        """
        pass

    @abstractmethod
    def run_experiment(self) -> Dict[str, Any]:
        """
        Ejecuta un experimento.

        Returns:
            Resultados del experimento
        """
        pass

    def get_status(self) -> Dict[str, Any]:
        """Retorna estado del investigador."""
        return {
            'id': self.investigator_id,
            't': self.t,
            'n_observations': len(self.observations),
        }


# =============================================================================
# EJEMPLO: INVESTIGADOR GENÉRICO
# =============================================================================

class GenericEndogenousInvestigator(BaseInvestigator):
    """
    Investigador genérico que cumple NORMA DURA.

    Solo busca patrones matemáticos.
    NO conoce semántica.
    """

    def __init__(self, investigator_id: str):
        super().__init__(investigator_id)
        self.series_cache: Dict[str, Dict] = {}
        self.hypotheses: List[Dict] = []
        self.discoveries: List[Dict] = []

    def observe(self, stimuli: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observa estímulos y los almacena internamente.
        """
        self.t += 1
        self.observations.append({
            't': self.t,
            'n_series': len(stimuli.get('series', {})),
            'n_matrices': len(stimuli.get('matrices', {})),
            'n_graphs': len(stimuli.get('graphs', {})),
        })

        # Almacenar series para análisis posterior
        for sid, series_data in stimuli.get('series', {}).items():
            self.series_cache[sid] = {
                't': np.array(series_data['t']),
                'values': np.array(series_data['values']),
            }

        return {
            'status': 'observed',
            't': self.t,
            'series_received': list(stimuli.get('series', {}).keys()),
        }

    def decide_what_to_investigate(self) -> Optional[Dict[str, Any]]:
        """
        Decide qué investigar basándose en los datos observados.

        Busca:
        - Correlaciones entre series
        - Anomalías temporales
        - Patrones de co-ocurrencia
        """
        if len(self.series_cache) < 2:
            return None

        # Elegir par de series para investigar correlación
        series_ids = list(self.series_cache.keys())

        # Calcular correlaciones entre todos los pares
        max_corr = 0
        best_pair = None

        for i, s1 in enumerate(series_ids):
            for s2 in series_ids[i+1:]:
                data1 = self.series_cache[s1]['values']
                data2 = self.series_cache[s2]['values']

                # Alinear longitudes
                min_len = min(len(data1), len(data2))
                if min_len < 5:  # Mínimo estadístico
                    continue

                corr = abs(np.corrcoef(data1[:min_len], data2[:min_len])[0, 1])

                if not np.isnan(corr) and corr > max_corr:
                    max_corr = corr
                    best_pair = (s1, s2)

        if best_pair and max_corr > 0.3:  # Umbral mínimo
            return {
                'type': 'correlation_analysis',
                'series_pair': best_pair,
                'preliminary_correlation': max_corr,
            }

        return None

    def run_experiment(self) -> Dict[str, Any]:
        """
        Ejecuta el experimento decidido.
        """
        investigation = self.decide_what_to_investigate()

        if investigation is None:
            return {'status': 'nothing_to_investigate'}

        if investigation['type'] == 'correlation_analysis':
            s1, s2 = investigation['series_pair']
            data1 = self.series_cache[s1]['values']
            data2 = self.series_cache[s2]['values']

            min_len = min(len(data1), len(data2))
            corr = np.corrcoef(data1[:min_len], data2[:min_len])[0, 1]

            # Registrar como hipótesis
            hypothesis = {
                't': self.t,
                'type': 'correlation',
                'series': (s1, s2),
                'value': float(corr),
                'n_samples': min_len,
            }
            self.hypotheses.append(hypothesis)

            # Si correlación significativa, registrar como descubrimiento
            # ORIGEN: r > 2/sqrt(n) es significativo al 5%
            significance_threshold = 2.0 / np.sqrt(min_len)
            if abs(corr) > significance_threshold:
                discovery = {
                    't': self.t,
                    'type': 'significant_correlation',
                    'series': (s1, s2),
                    'correlation': float(corr),
                    'threshold': significance_threshold,
                }
                self.discoveries.append(discovery)

            return {
                'status': 'experiment_completed',
                'hypothesis': hypothesis,
                'is_discovery': abs(corr) > significance_threshold,
            }

        return {'status': 'unknown_experiment_type'}

    def get_status(self) -> Dict[str, Any]:
        """Retorna estado completo."""
        base = super().get_status()
        base.update({
            'n_series_cached': len(self.series_cache),
            'n_hypotheses': len(self.hypotheses),
            'n_discoveries': len(self.discoveries),
        })
        return base


# =============================================================================
# TEST
# =============================================================================

def test_investigator_interface():
    """Test de la interfaz con investigadores."""
    print("=" * 70)
    print("TEST: INVESTIGATOR INTERFACE")
    print("El investigador solo recibe matemáticas, no semántica")
    print("=" * 70)

    from .stimulus_engine import StimulusEngine

    # Crear motor y generar estímulos
    engine = StimulusEngine()

    np.random.seed(42)

    # Series sintéticas (la humana sabe qué son, el sistema no)
    sources = [
        {
            'type': 'array',
            't': np.linspace(0, 100, 200),
            'values': np.cumsum(np.random.randn(200)),
            'source': 'process_alpha'
        },
        {
            'type': 'array',
            't': np.linspace(0, 100, 200),
            'values': np.sin(np.linspace(0, 10, 200)) + np.random.randn(200) * 0.1,
            'source': 'process_beta'
        },
        {
            'type': 'array',
            't': np.linspace(0, 100, 200),
            # Esta serie está correlacionada con la anterior
            'values': 0.8 * (np.sin(np.linspace(0, 10, 200)) + np.random.randn(200) * 0.1) + np.random.randn(200) * 0.2,
            'source': 'process_gamma'
        },
    ]

    bundle = engine.generate_stimuli(sources)

    print(f"\nBundle generado: {bundle.id}")
    print(f"  Series: {list(bundle.series.keys())}")

    # Crear dispatcher e investigador
    dispatcher = InvestigatorDispatcher()
    investigator = GenericEndogenousInvestigator("inv_001")
    dispatcher.register("inv_001", investigator)

    print("\n=== ENVIANDO ESTÍMULOS AL INVESTIGADOR ===")
    responses = dispatcher.broadcast(bundle)
    print(f"Respuestas: {responses}")

    print("\n=== EL INVESTIGADOR DECIDE QUÉ INVESTIGAR ===")
    decision = investigator.decide_what_to_investigate()
    print(f"Decisión autónoma: {decision}")

    print("\n=== EL INVESTIGADOR EJECUTA EXPERIMENTO ===")
    result = investigator.run_experiment()
    print(f"Resultado: {result}")

    print("\n=== ESTADO DEL INVESTIGADOR ===")
    status = investigator.get_status()
    print(f"Status: {status}")

    if investigator.discoveries:
        print("\n=== DESCUBRIMIENTOS ===")
        for d in investigator.discoveries:
            print(f"  {d}")

    return dispatcher, investigator


if __name__ == "__main__":
    test_investigator_interface()
