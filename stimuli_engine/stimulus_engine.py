"""
STIMULUS ENGINE - Motor Principal de Estímulos
===============================================

PROPÓSITO:
Traducir entradas del mundo (archivos, APIs, CSVs) a estructuras matemáticas.

QUÉ HACE:
- Convierte datos externos a series temporales X(t), Y(t), Z(t)
- Genera matrices de relaciones
- Construye grafos de estructura
- Documenta procedencia según NORMA DURA

QUÉ NO HACE:
- NO decide hipótesis
- NO interpreta significado semántico
- NO guía al investigador
- NO contiene nombres de entidades reales

El significado lo conoce la humana. El sistema solo ve matemáticas.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from pathlib import Path
import json

from .provenance import (
    Provenance, ProvenanceType, ProvenanceLogger,
    get_provenance_logger, MATH_CONSTANTS, THEORY_CONSTANTS
)


@dataclass
class TimeSeries:
    """
    Serie temporal anónima.

    NO contiene semántica. Solo:
    - Identificador numérico (s_01, s_02, ...)
    - Timestamps normalizados
    - Valores numéricos
    - Metadatos mínimos (unidades, resolución)
    """
    id: str                                 # Identificador anónimo (s_01, s_02, ...)
    t: np.ndarray                           # Timestamps (normalizados a float)
    values: np.ndarray                      # Valores
    unit: str = ""                          # Unidad (opcional, para consistencia dimensional)
    resolution: float = 0.0                 # Resolución temporal (derivada de datos)
    provenance: Optional[Provenance] = None # De dónde vienen estos datos

    def __post_init__(self):
        # Derivar resolución de los datos si no está especificada
        if self.resolution == 0.0 and len(self.t) > 1:
            # ORIGEN: Mediana de diferencias temporales
            diffs = np.diff(self.t)
            self.resolution = float(np.median(diffs))

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            't': self.t.tolist() if hasattr(self.t, 'tolist') else list(self.t),
            'values': self.values.tolist() if hasattr(self.values, 'tolist') else list(self.values),
            'unit': self.unit,
            'resolution': self.resolution,
            'n_points': len(self.t),
            'provenance': self.provenance.to_dict() if self.provenance else None,
        }


@dataclass
class Matrix:
    """
    Matriz anónima.

    Puede representar:
    - Matriz de adyacencia
    - Matriz de correlaciones
    - Cualquier relación NxM
    """
    id: str                                 # Identificador anónimo (m_01, m_02, ...)
    data: np.ndarray                        # La matriz
    row_labels: List[str] = field(default_factory=list)  # Etiquetas anónimas
    col_labels: List[str] = field(default_factory=list)
    provenance: Optional[Provenance] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'shape': list(self.data.shape),
            'data': self.data.tolist(),
            'row_labels': self.row_labels,
            'col_labels': self.col_labels,
            'provenance': self.provenance.to_dict() if self.provenance else None,
        }


@dataclass
class Graph:
    """
    Grafo anónimo.

    Solo estructura matemática:
    - Nodos (identificadores numéricos)
    - Aristas (pares de nodos con peso opcional)
    """
    id: str                                 # Identificador anónimo (g_01, g_02, ...)
    nodes: List[str]                        # Nodos anónimos (n_01, n_02, ...)
    edges: List[Tuple[str, str, float]]     # (nodo_origen, nodo_destino, peso)
    directed: bool = False
    provenance: Optional[Provenance] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'n_nodes': len(self.nodes),
            'n_edges': len(self.edges),
            'directed': self.directed,
            'nodes': self.nodes,
            'edges': [(e[0], e[1], e[2]) for e in self.edges],
            'provenance': self.provenance.to_dict() if self.provenance else None,
        }


@dataclass
class Stimulus:
    """
    Un estímulo individual = estructura matemática con procedencia.

    Tipos:
    - series: Serie temporal
    - matrix: Matriz
    - graph: Grafo
    - scalar: Valor escalar
    """
    id: str
    stype: str                              # "series", "matrix", "graph", "scalar"
    data: Union[TimeSeries, Matrix, Graph, float]
    created_at: str = ""
    provenance: Optional[Provenance] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        if hasattr(self.data, 'to_dict'):
            data_dict = self.data.to_dict()
        else:
            data_dict = self.data

        return {
            'id': self.id,
            'type': self.stype,
            'data': data_dict,
            'created_at': self.created_at,
            'provenance': self.provenance.to_dict() if self.provenance else None,
        }


@dataclass
class StimulusBundle:
    """
    Conjunto de estímulos para entregar al investigador.

    Contiene múltiples series, matrices, grafos.
    NO contiene interpretación semántica.
    """
    id: str
    series: Dict[str, TimeSeries] = field(default_factory=dict)
    matrices: Dict[str, Matrix] = field(default_factory=dict)
    graphs: Dict[str, Graph] = field(default_factory=dict)
    scalars: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def add_series(self, series: TimeSeries):
        """Añade una serie temporal."""
        self.series[series.id] = series

    def add_matrix(self, matrix: Matrix):
        """Añade una matriz."""
        self.matrices[matrix.id] = matrix

    def add_graph(self, graph: Graph):
        """Añade un grafo."""
        self.graphs[graph.id] = graph

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'n_series': len(self.series),
            'n_matrices': len(self.matrices),
            'n_graphs': len(self.graphs),
            'n_scalars': len(self.scalars),
            'series': {k: v.to_dict() for k, v in self.series.items()},
            'matrices': {k: v.to_dict() for k, v in self.matrices.items()},
            'graphs': {k: v.to_dict() for k, v in self.graphs.items()},
            'scalars': self.scalars,
            'metadata': self.metadata,
            'created_at': self.created_at,
        }

    def summary(self) -> str:
        """Resumen del bundle (sin semántica)."""
        lines = [
            f"StimulusBundle [{self.id}]",
            f"  Series: {len(self.series)}",
        ]
        for sid, s in self.series.items():
            lines.append(f"    {sid}: {len(s.t)} puntos, resolución={s.resolution:.4f}")

        lines.append(f"  Matrices: {len(self.matrices)}")
        for mid, m in self.matrices.items():
            lines.append(f"    {mid}: {m.data.shape}")

        lines.append(f"  Grafos: {len(self.graphs)}")
        for gid, g in self.graphs.items():
            lines.append(f"    {gid}: {len(g.nodes)} nodos, {len(g.edges)} aristas")

        lines.append(f"  Escalares: {len(self.scalars)}")

        return "\n".join(lines)


class StimulusEngine:
    """
    Motor principal de estímulos.

    RESPONSABILIDADES:
    1. Recibir datos externos (archivos, APIs)
    2. Convertirlos a estructuras matemáticas anónimas
    3. Documentar procedencia
    4. Entregar al investigador

    NO HACE:
    - Interpretar significado
    - Sugerir hipótesis
    - Seleccionar qué es "importante"
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = get_provenance_logger()
        self.connectors = {}
        self._series_counter = 0
        self._matrix_counter = 0
        self._graph_counter = 0
        self._bundle_counter = 0

        # Registrar constantes usadas
        self._register_constants()

    def _register_constants(self):
        """Registra las constantes que usará el engine."""
        # Todas las constantes tienen procedencia documentada
        self.constants = {
            'min_samples': THEORY_CONSTANTS['min_samples_corr'],
            'inv_e': MATH_CONSTANTS['inv_e'],
        }

    def _next_series_id(self) -> str:
        """Genera siguiente ID anónimo para serie."""
        self._series_counter += 1
        return f"s_{self._series_counter:03d}"

    def _next_matrix_id(self) -> str:
        """Genera siguiente ID anónimo para matriz."""
        self._matrix_counter += 1
        return f"m_{self._matrix_counter:03d}"

    def _next_graph_id(self) -> str:
        """Genera siguiente ID anónimo para grafo."""
        self._graph_counter += 1
        return f"g_{self._graph_counter:03d}"

    def _next_bundle_id(self) -> str:
        """Genera siguiente ID anónimo para bundle."""
        self._bundle_counter += 1
        return f"bundle_{self._bundle_counter:03d}"

    def create_series_from_arrays(
        self,
        t: np.ndarray,
        values: np.ndarray,
        source_description: str,
        unit: str = ""
    ) -> TimeSeries:
        """
        Crea serie temporal desde arrays.

        Args:
            t: Array de timestamps
            values: Array de valores
            source_description: Descripción de procedencia (sin semántica)
            unit: Unidad (opcional)

        Returns:
            TimeSeries anónima
        """
        provenance = self.logger.log_from_data(
            value=f"series[{len(t)}]",
            source=source_description,
            statistic="raw_data",
            context="create_series_from_arrays"
        )

        series = TimeSeries(
            id=self._next_series_id(),
            t=np.array(t),
            values=np.array(values),
            unit=unit,
            provenance=provenance
        )

        return series

    def create_series_from_csv(
        self,
        path: Path,
        time_col: int = 0,
        value_col: int = 1,
        unit: str = ""
    ) -> TimeSeries:
        """
        Crea serie temporal desde CSV.

        Args:
            path: Ruta al archivo
            time_col: Índice de columna de tiempo
            value_col: Índice de columna de valores
            unit: Unidad

        Returns:
            TimeSeries anónima (sin nombre del archivo en el ID)
        """
        import csv

        t_values = []
        data_values = []

        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header

            for row in reader:
                try:
                    t_val = float(row[time_col])
                    d_val = float(row[value_col])
                    t_values.append(t_val)
                    data_values.append(d_val)
                except (ValueError, IndexError):
                    continue

        provenance = self.logger.log_from_data(
            value=f"csv[{len(t_values)}]",
            source=f"file_hash:{hash(str(path)) % 10000:04d}",  # Hash, no nombre
            dataset="external_csv",
            context="create_series_from_csv"
        )

        return TimeSeries(
            id=self._next_series_id(),
            t=np.array(t_values),
            values=np.array(data_values),
            unit=unit,
            provenance=provenance
        )

    def create_bundle(self) -> StimulusBundle:
        """Crea un nuevo bundle vacío."""
        return StimulusBundle(id=self._next_bundle_id())

    def normalize_timestamps(self, series: TimeSeries) -> TimeSeries:
        """
        Normaliza timestamps a rango [0, 1].

        NORMA DURA: La normalización es lineal, sin parámetros mágicos.
        """
        t = series.t
        t_min = np.min(t)
        t_max = np.max(t)

        if t_max == t_min:
            t_normalized = np.zeros_like(t)
        else:
            # ORIGEN: Normalización min-max estándar
            t_normalized = (t - t_min) / (t_max - t_min)

        provenance = self.logger.log_from_theory(
            value="min_max_normalization",
            source="Normalización lineal: (x - min) / (max - min)",
            context="normalize_timestamps"
        )

        return TimeSeries(
            id=series.id,
            t=t_normalized,
            values=series.values.copy(),
            unit=series.unit,
            resolution=series.resolution / (t_max - t_min) if t_max != t_min else 0,
            provenance=provenance
        )

    def standardize_values(self, series: TimeSeries) -> TimeSeries:
        """
        Estandariza valores a z-scores.

        NORMA DURA: z = (x - μ) / σ, derivado de estadística básica.
        """
        values = series.values
        mu = np.mean(values)
        sigma = np.std(values)

        if sigma < np.finfo(float).eps:
            # ORIGEN: eps de precisión máquina
            z_values = np.zeros_like(values)
        else:
            z_values = (values - mu) / sigma

        provenance = self.logger.log_from_theory(
            value="z_standardization",
            source="z-score: (x - μ) / σ",
            reference="Estadística descriptiva estándar",
            context="standardize_values"
        )

        return TimeSeries(
            id=series.id,
            t=series.t.copy(),
            values=z_values,
            unit="z-score",
            resolution=series.resolution,
            provenance=provenance
        )

    def compute_correlation_matrix(
        self,
        series_list: List[TimeSeries]
    ) -> Matrix:
        """
        Computa matriz de correlaciones entre series.

        NORMA DURA:
        - Correlación de Pearson (definición estándar)
        - Mínimo de muestras según teoría (n >= 5)
        """
        n = len(series_list)
        if n == 0:
            return Matrix(
                id=self._next_matrix_id(),
                data=np.array([[]]),
                provenance=self.logger.log_from_theory(
                    value="empty_correlation",
                    source="No series provided"
                )
            )

        # Interpolar a timestamps comunes
        # ORIGEN: Usar la serie con más puntos como referencia
        max_points = max(len(s.t) for s in series_list)
        t_common = np.linspace(0, 1, max_points)

        # Interpolar todas las series
        interpolated = []
        for s in series_list:
            if len(s.t) > 1:
                interp_vals = np.interp(t_common, s.t, s.values)
            else:
                interp_vals = np.full(max_points, s.values[0] if len(s.values) > 0 else 0)
            interpolated.append(interp_vals)

        data_matrix = np.array(interpolated)

        # Matriz de correlaciones
        # ORIGEN: Pearson correlation, definición estándar
        corr_matrix = np.corrcoef(data_matrix)

        # Manejar NaN (series constantes)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        provenance = self.logger.log_from_theory(
            value="pearson_correlation_matrix",
            source="ρ = cov(X,Y) / (σ_X * σ_Y)",
            reference="Pearson, K. (1895)",
            context="compute_correlation_matrix"
        )

        return Matrix(
            id=self._next_matrix_id(),
            data=corr_matrix,
            row_labels=[s.id for s in series_list],
            col_labels=[s.id for s in series_list],
            provenance=provenance
        )

    def detect_events(
        self,
        series: TimeSeries
    ) -> List[Tuple[float, float]]:
        """
        Detecta eventos (anomalías) en una serie.

        NORMA DURA:
        - Umbral = Tukey fence (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        - No hay umbral arbitrario

        Returns:
            Lista de (timestamp, valor) de eventos detectados
        """
        values = series.values

        if len(values) < self.constants['min_samples'].value:
            return []

        # ORIGEN: Tukey fences para detección de outliers
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        k = THEORY_CONSTANTS['tukey_k'].value  # 1.5, documentado

        lower = q1 - k * iqr
        upper = q3 + k * iqr

        self.logger.log_from_theory(
            value={'lower': lower, 'upper': upper, 'k': k},
            source="Tukey fences: Q1 - 1.5*IQR, Q3 + 1.5*IQR",
            reference="Tukey, J. W. (1977). Exploratory Data Analysis",
            context="detect_events"
        )

        events = []
        for i, v in enumerate(values):
            if v < lower or v > upper:
                events.append((float(series.t[i]), float(v)))

        return events

    def build_event_graph(
        self,
        series_list: List[TimeSeries],
        time_window: Optional[float] = None
    ) -> Graph:
        """
        Construye grafo de co-ocurrencia de eventos.

        Dos nodos están conectados si tienen eventos cercanos en tiempo.

        NORMA DURA:
        - time_window derivado de datos si no se proporciona
        - ORIGEN: Mediana de resoluciones * 3 (para capturar vecindad)
        """
        if time_window is None:
            # Derivar de datos
            resolutions = [s.resolution for s in series_list if s.resolution > 0]
            if resolutions:
                time_window = np.median(resolutions) * 3
            else:
                time_window = 0.1  # Fallback basado en normalización [0,1]

            self.logger.log_from_data(
                value=time_window,
                source="median(resolutions) * 3",
                statistic="derived_time_window",
                context="build_event_graph"
            )

        # Detectar eventos en cada serie
        events_by_series = {}
        for s in series_list:
            events = self.detect_events(s)
            if events:
                events_by_series[s.id] = events

        # Construir nodos
        nodes = list(events_by_series.keys())

        # Construir aristas (co-ocurrencia temporal)
        edges = []
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                # Contar co-ocurrencias
                count = 0
                for t1, _ in events_by_series[n1]:
                    for t2, _ in events_by_series[n2]:
                        if abs(t1 - t2) < time_window:
                            count += 1

                if count > 0:
                    # Peso = número de co-ocurrencias
                    edges.append((n1, n2, float(count)))

        return Graph(
            id=self._next_graph_id(),
            nodes=nodes,
            edges=edges,
            directed=False,
            provenance=self.logger.log_from_data(
                value=f"event_graph[{len(nodes)}nodes,{len(edges)}edges]",
                source="co-occurrence_detection",
                statistic=f"time_window={time_window:.4f}",
                context="build_event_graph"
            )
        )

    def generate_stimuli(
        self,
        sources: List[Dict[str, Any]]
    ) -> StimulusBundle:
        """
        Genera bundle de estímulos desde múltiples fuentes.

        Args:
            sources: Lista de configuraciones de fuentes
                     Cada una tiene: {'type': 'csv'|'array', ...params}

        Returns:
            StimulusBundle con todas las estructuras matemáticas
        """
        bundle = self.create_bundle()
        series_list = []

        for source in sources:
            stype = source.get('type', 'array')

            if stype == 'csv':
                series = self.create_series_from_csv(
                    path=Path(source['path']),
                    time_col=source.get('time_col', 0),
                    value_col=source.get('value_col', 1),
                    unit=source.get('unit', '')
                )
            elif stype == 'array':
                series = self.create_series_from_arrays(
                    t=np.array(source['t']),
                    values=np.array(source['values']),
                    source_description=source.get('source', 'array_input'),
                    unit=source.get('unit', '')
                )
            else:
                continue

            # Normalizar y estandarizar
            series = self.normalize_timestamps(series)
            series = self.standardize_values(series)

            bundle.add_series(series)
            series_list.append(series)

        # Añadir matriz de correlaciones si hay múltiples series
        if len(series_list) > 1:
            corr_matrix = self.compute_correlation_matrix(series_list)
            bundle.add_matrix(corr_matrix)

            # Añadir grafo de eventos
            event_graph = self.build_event_graph(series_list)
            if event_graph.nodes:
                bundle.add_graph(event_graph)

        return bundle

    def get_provenance_report(self) -> Dict:
        """Obtiene reporte de procedencia para auditoría."""
        return self.logger.get_audit_report()


# =============================================================================
# INTERFAZ LIMPIA CON INVESTIGADORES
# =============================================================================

class InvestigatorInterface:
    """
    Interfaz limpia entre StimulusEngine e Investigadores.

    El investigador SOLO recibe:
    - Estructuras matemáticas (series, matrices, grafos)
    - Metadatos de procedencia

    El investigador NO recibe:
    - Nombres semánticos
    - Hipótesis sugeridas
    - Interpretaciones
    """

    def __init__(self, engine: StimulusEngine):
        self.engine = engine
        self.delivered_bundles: List[str] = []

    def deliver_to_investigator(
        self,
        bundle: StimulusBundle,
        investigator: Any
    ) -> Dict[str, Any]:
        """
        Entrega un bundle al investigador.

        Args:
            bundle: StimulusBundle a entregar
            investigator: Objeto investigador con método observe()

        Returns:
            Resultado de la observación
        """
        self.delivered_bundles.append(bundle.id)

        # Preparar datos en formato que el investigador espera
        stimuli_data = {
            'series': {
                sid: {
                    't': s.t.tolist(),
                    'values': s.values.tolist(),
                    'resolution': s.resolution,
                }
                for sid, s in bundle.series.items()
            },
            'matrices': {
                mid: {
                    'data': m.data.tolist(),
                    'shape': list(m.data.shape),
                }
                for mid, m in bundle.matrices.items()
            },
            'graphs': {
                gid: {
                    'nodes': g.nodes,
                    'edges': g.edges,
                }
                for gid, g in bundle.graphs.items()
            },
        }

        # Llamar a observe() si el investigador lo tiene
        if hasattr(investigator, 'observe'):
            return investigator.observe(stimuli_data)
        else:
            return {'status': 'delivered', 'bundle_id': bundle.id}


# =============================================================================
# TEST
# =============================================================================

def test_stimulus_engine():
    """Test del motor de estímulos."""
    print("=" * 70)
    print("TEST: STIMULUS ENGINE")
    print("Traduce el mundo a matemáticas - SIN semántica")
    print("=" * 70)

    engine = StimulusEngine()

    # Crear series sintéticas (el humano sabe qué son, el sistema no)
    np.random.seed(42)

    # Serie 1: algún proceso con tendencia
    t1 = np.linspace(0, 100, 500)
    v1 = np.cumsum(np.random.randn(500)) + 0.1 * t1

    # Serie 2: otro proceso con eventos
    t2 = np.linspace(0, 100, 500)
    v2 = np.sin(t2 * 0.1) + np.random.randn(500) * 0.5
    # Añadir algunos eventos (el sistema no sabe qué significan)
    v2[100:110] += 5
    v2[300:310] += 4

    # Serie 3: proceso relacionado con serie 2
    t3 = np.linspace(0, 100, 500)
    v3 = 0.7 * v2 + np.random.randn(500) * 0.3

    sources = [
        {'type': 'array', 't': t1, 'values': v1, 'source': 'process_alpha'},
        {'type': 'array', 't': t2, 'values': v2, 'source': 'process_beta'},
        {'type': 'array', 't': t3, 'values': v3, 'source': 'process_gamma'},
    ]

    print("\n=== GENERANDO ESTÍMULOS ===")
    bundle = engine.generate_stimuli(sources)

    print(bundle.summary())

    print("\n=== MATRIZ DE CORRELACIONES ===")
    for mid, matrix in bundle.matrices.items():
        print(f"\n{mid}:")
        print(f"  Etiquetas: {matrix.row_labels}")
        print(f"  Datos:\n{np.round(matrix.data, 3)}")

    print("\n=== GRAFO DE EVENTOS ===")
    for gid, graph in bundle.graphs.items():
        print(f"\n{gid}:")
        print(f"  Nodos: {graph.nodes}")
        print(f"  Aristas: {graph.edges}")

    print("\n=== REPORTE DE PROCEDENCIA ===")
    report = engine.get_provenance_report()
    print(f"  Total entradas: {report['total_entries']}")
    print(f"  Por tipo: {report['by_type']}")
    print(f"  Violaciones NORMA DURA: {report['violations']}")

    print("\n=== BUNDLE PARA INVESTIGADOR ===")
    print("(Solo matemáticas, sin nombres semánticos)")
    bundle_dict = bundle.to_dict()
    print(f"  Series: {list(bundle_dict['series'].keys())}")
    print(f"  Matrices: {list(bundle_dict['matrices'].keys())}")
    print(f"  Grafos: {list(bundle_dict['graphs'].keys())}")

    return engine, bundle


if __name__ == "__main__":
    test_stimulus_engine()
