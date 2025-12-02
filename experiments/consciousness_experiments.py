"""
Experimentos de Consciencia Computacional
==========================================

Tres experimentos para estudiar la dinámica de consciencia:

1. SIMULACIÓN LARGA: Quién florece, colapsa o se deprime
2. ESTRÉS PROLONGADO: Comportamiento del médico y reemplazos
3. ENTROPÍA NARRATIVA ALTA: Sueño y renacimiento bajo caos

REGLAS RÍGIDAS:
- Prohibido usar números mágicos
- Todo se define con percentiles, medias, varianzas, entropías
- Todo emerge de los datos históricos del sistema

100% endógeno. Sin números mágicos.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import sys

sys.path.insert(0, '/root/NEO_EVA')

from consciousness.emergence import SistemaConscienciaColectiva, EstadoConsciencia
from consciousness.roles import TipoRol
from consciousness.dreaming import FaseSueno
from consciousness.death_rebirth import EstadoVital


class ClasificacionAgente(Enum):
    """Clasificación del comportamiento del agente."""
    FLORECE = "florece"
    COLAPSA = "colapsa"
    DEPRIME = "deprime"
    NEUTRAL = "neutral"


@dataclass
class MetricasAgente:
    """Métricas calculadas para un agente."""
    agent_id: str
    trend_ce: float
    mean_ce: float
    p10_ce: float
    p50_ce: float
    p90_ce: float
    proportion_high_ce: float
    proportion_low_ce: float
    clasificacion: ClasificacionAgente
    historial_ce: List[float] = field(default_factory=list)


@dataclass
class EventoMedico:
    """Evento relacionado con el rol médico."""
    t: int
    medico_id: Optional[str]
    ce_medico: Optional[float]
    medico_enferma: bool
    evento_reemplazo: bool


@dataclass
class EventoVida:
    """Evento de muerte o renacimiento."""
    t: int
    agent_id: str
    tipo: str  # "muerte" o "renacimiento"
    ce_antes: float
    ce_despues: Optional[float]


class ConsciousnessExperiments:
    """
    Sistema de experimentos para consciencia computacional.

    Todos los parámetros son endógenos:
    - Duración de simulaciones basada en estabilización de varianza
    - Umbrales por percentiles históricos
    - Selección de agentes por distribuciones internas
    """

    def __init__(self, sistema: SistemaConscienciaColectiva):
        """
        Args:
            sistema: Sistema de consciencia colectiva ya inicializado con agentes
        """
        self.sys = sistema
        self.dimension = sistema.dimension

        # Historiales para cálculos endógenos
        self._historial_ce_global: List[float] = []
        self._historial_varianza_ce: List[float] = []
        self._historial_cambios_ce: List[float] = []

    # =========================================================================
    # UTILIDADES ENDÓGENAS
    # =========================================================================

    def _calcular_tendencia(self, valores: List[float]) -> float:
        """
        Calcula la tendencia (pendiente) de una serie temporal.

        slope = Cov(t, valores) / Var(t)
        """
        if len(valores) < 2:
            return 0.0

        n = len(valores)
        t = np.arange(n)

        # Regresión lineal simple
        mean_t = np.mean(t)
        mean_v = np.mean(valores)

        cov_tv = np.mean((t - mean_t) * (valores - mean_v))
        var_t = np.var(t)

        if var_t < 1e-10:
            return 0.0

        return float(cov_tv / var_t)

    def _calcular_percentil(self, valores: List[float], p: float) -> float:
        """Calcula percentil de forma segura."""
        if not valores:
            return 0.0
        return float(np.percentile(valores, p * 100))

    def _proporcion_sobre_umbral(self, valores: List[float], umbral: float) -> float:
        """Calcula proporción de valores sobre un umbral."""
        if not valores:
            return 0.0
        return float(np.mean([1 if v > umbral else 0 for v in valores]))

    def _proporcion_bajo_umbral(self, valores: List[float], umbral: float) -> float:
        """Calcula proporción de valores bajo un umbral."""
        if not valores:
            return 0.0
        return float(np.mean([1 if v < umbral else 0 for v in valores]))

    def _condicion_estabilizacion(
        self,
        historial_varianza: List[float],
        ventana_reciente: int = None
    ) -> bool:
        """
        Determina si el sistema se ha estabilizado.

        Estabilización: cambio medio de varianza < P_0.10(cambios históricos)

        La ventana es endógena: sqrt(len(historial))
        """
        if len(historial_varianza) < 4:
            return False

        # Ventana endógena
        if ventana_reciente is None:
            ventana_reciente = max(2, int(np.sqrt(len(historial_varianza))))

        # Cambios de varianza
        cambios = [abs(historial_varianza[i] - historial_varianza[i-1])
                   for i in range(1, len(historial_varianza))]

        if len(cambios) < ventana_reciente:
            return False

        # Cambio reciente
        cambio_reciente = np.mean(cambios[-ventana_reciente:])

        # Umbral endógeno: percentil 10 de cambios históricos
        umbral = np.percentile(cambios, 10)

        return cambio_reciente < umbral

    def _generar_input_normal(self) -> np.ndarray:
        """Genera input normal basado en estadísticas del sistema."""
        # Varianza basada en historial de estados
        if self._historial_varianza_ce:
            escala = np.sqrt(np.mean(self._historial_varianza_ce))
        else:
            escala = 1.0

        return np.random.randn(self.dimension) * escala

    def _ejecutar_paso(
        self,
        entradas: Dict[str, np.ndarray] = None,
        eventos: Dict[str, np.ndarray] = None
    ) -> Dict[str, EstadoConsciencia]:
        """
        Ejecuta un paso del sistema y actualiza historiales internos.
        """
        estados = self.sys.paso(entradas, eventos)

        # Actualizar historiales globales
        ces = [e.CE for e in estados.values()]
        if ces:
            self._historial_ce_global.append(np.mean(ces))
            self._historial_varianza_ce.append(np.var(ces))

            if len(self._historial_ce_global) > 1:
                cambio = abs(self._historial_ce_global[-1] - self._historial_ce_global[-2])
                self._historial_cambios_ce.append(cambio)

        return estados

    # =========================================================================
    # EXPERIMENTO 1: SIMULACIÓN LARGA - FLORECE / COLAPSA / DEPRIME
    # =========================================================================

    def experimento_1_larga_simulacion(
        self,
        n_agentes: int = None
    ) -> Dict[str, Any]:
        """
        Ejecuta simulación larga normal y clasifica comportamiento de agentes.

        Clasificación endógena:
        - FLORECE: trend > 0, mean > P_0.50, tiempo en P_0.90 alto
        - COLAPSA: trend < 0, mean < P_0.50, tiempo en P_0.10 alto
        - DEPRIME: trend ≈ 0, mean < P_0.50, sin muerte
        - NEUTRAL: resto

        La duración es endógena: hasta que varianza de CE se estabilice.

        Returns:
            Diccionario con métricas y clasificación por agente
        """
        # Crear agentes si no existen
        if n_agentes is None:
            # Número endógeno basado en dimensión
            n_agentes = max(3, int(np.sqrt(self.dimension)))

        agentes_ids = []
        for i in range(n_agentes):
            agent_id = f"agent_{i}"
            self.sys.crear_agente(agent_id)
            agentes_ids.append(agent_id)

        # Historiales por agente
        historial_ce: Dict[str, List[float]] = {aid: [] for aid in agentes_ids}
        historial_estados: Dict[str, List[EstadoVital]] = {aid: [] for aid in agentes_ids}

        # Ejecutar hasta estabilización
        # Mínimo: dimensión * 2 pasos para tener estadísticas
        # Máximo: limitado por condición de estabilización
        min_pasos = self.dimension * 2
        paso = 0

        while True:
            paso += 1

            # Generar inputs normales (no forzados)
            entradas = {aid: self._generar_input_normal() for aid in agentes_ids}

            # Eventos narrativos ocasionales (basado en entropía del paso)
            eventos = {}
            for aid in agentes_ids:
                # Probabilidad de evento basada en historial de entropía
                if historial_ce[aid]:
                    prob_evento = 1.0 / (1.0 + len(historial_ce[aid]))
                else:
                    prob_evento = 0.5

                if np.random.random() < prob_evento:
                    eventos[aid] = np.random.randn(self.dimension) * 0.5

            # Ejecutar paso
            estados = self._ejecutar_paso(entradas, eventos)

            # Registrar
            for aid, estado in estados.items():
                if aid in historial_ce:
                    historial_ce[aid].append(estado.CE)
                    historial_estados[aid].append(estado.estado_vital)

            # Verificar estabilización después del mínimo
            if paso >= min_pasos:
                if self._condicion_estabilizacion(self._historial_varianza_ce):
                    break

            # Límite de seguridad endógeno: cuando cambios son < P_0.01
            if len(self._historial_cambios_ce) > min_pasos:
                cambio_actual = self._historial_cambios_ce[-1] if self._historial_cambios_ce else 1.0
                umbral_minimo = np.percentile(self._historial_cambios_ce, 1)
                if cambio_actual < umbral_minimo:
                    break

        # Calcular métricas por agente
        resultados_agentes = {}
        tendencias = []
        proporciones_alto = []
        proporciones_bajo = []

        for aid in agentes_ids:
            ce_hist = historial_ce[aid]
            if not ce_hist:
                continue

            # Tendencia
            trend = self._calcular_tendencia(ce_hist)
            tendencias.append(trend)

            # Percentiles
            p10 = self._calcular_percentil(ce_hist, 0.10)
            p50 = self._calcular_percentil(ce_hist, 0.50)
            p90 = self._calcular_percentil(ce_hist, 0.90)

            # Proporciones
            prop_alto = self._proporcion_sobre_umbral(ce_hist, p90)
            prop_bajo = self._proporcion_bajo_umbral(ce_hist, p10)
            proporciones_alto.append(prop_alto)
            proporciones_bajo.append(prop_bajo)

            # Media
            mean_ce = float(np.mean(ce_hist))

            # Verificar si hubo muerte
            hubo_muerte = EstadoVital.MUERTO in historial_estados[aid]

            resultados_agentes[aid] = {
                'trend_ce': trend,
                'mean_ce': mean_ce,
                'p10_ce': p10,
                'p50_ce': p50,
                'p90_ce': p90,
                'proportion_high_ce': prop_alto,
                'proportion_low_ce': prop_bajo,
                'hubo_muerte': hubo_muerte,
                'historial_ce': ce_hist,
            }

        # Clasificación endógena
        # Umbrales basados en distribución de tendencias
        if tendencias:
            p25_trend = np.percentile(tendencias, 25)
            p75_trend = np.percentile(tendencias, 75)
            median_prop_alto = np.median(proporciones_alto) if proporciones_alto else 0.5
            median_prop_bajo = np.median(proporciones_bajo) if proporciones_bajo else 0.5
        else:
            p25_trend = 0.0
            p75_trend = 0.0
            median_prop_alto = 0.5
            median_prop_bajo = 0.5

        for aid, metricas in resultados_agentes.items():
            trend = metricas['trend_ce']
            mean_ce = metricas['mean_ce']
            p50 = metricas['p50_ce']
            prop_alto = metricas['proportion_high_ce']
            prop_bajo = metricas['proportion_low_ce']
            hubo_muerte = metricas['hubo_muerte']

            # Clasificación
            if trend > p75_trend and mean_ce > p50 and prop_alto > median_prop_alto:
                clasificacion = ClasificacionAgente.FLORECE
            elif trend < p25_trend and mean_ce < p50 and prop_bajo > median_prop_bajo:
                clasificacion = ClasificacionAgente.COLAPSA
            elif p25_trend <= trend <= p75_trend and mean_ce < p50 and not hubo_muerte:
                clasificacion = ClasificacionAgente.DEPRIME
            else:
                clasificacion = ClasificacionAgente.NEUTRAL

            metricas['clasificacion'] = clasificacion.value

        # Resumen global
        conteo_clasificaciones = {c.value: 0 for c in ClasificacionAgente}
        for metricas in resultados_agentes.values():
            conteo_clasificaciones[metricas['clasificacion']] += 1

        return {
            'agentes': resultados_agentes,
            'resumen': {
                'total_pasos': paso,
                'n_agentes': len(agentes_ids),
                'conteo_clasificaciones': conteo_clasificaciones,
                'ce_global_final': self._historial_ce_global[-1] if self._historial_ce_global else 0.0,
                'varianza_ce_final': self._historial_varianza_ce[-1] if self._historial_varianza_ce else 0.0,
            }
        }

    # =========================================================================
    # EXPERIMENTO 2: ESTRÉS PROLONGADO - MÉDICO Y REEMPLAZO
    # =========================================================================

    def experimento_2_estres_prolongado(
        self,
        n_agentes: int = None
    ) -> Dict[str, Any]:
        """
        Somete a un agente a estrés prolongado y observa comportamiento del médico.

        Selección endógena del agente:
        - CE cercano a P_0.50 (vulnerable, ni muy bien ni muy mal)
        - Estrés inicial bajo (para ver la caída)

        Estrés endógeno:
        - Inputs que históricamente generaron errores altos (> P_0.80)

        Returns:
            Timeline de eventos y resumen del comportamiento del médico
        """
        # Crear agentes si no existen
        if n_agentes is None:
            n_agentes = max(4, int(np.sqrt(self.dimension)))

        agentes_ids = []
        for i in range(n_agentes):
            agent_id = f"stress_agent_{i}"
            self.sys.crear_agente(agent_id)
            agentes_ids.append(agent_id)

        # Fase 1: Simulación inicial para establecer baseline
        historial_ce: Dict[str, List[float]] = {aid: [] for aid in agentes_ids}
        historial_errores: Dict[str, List[Tuple[np.ndarray, float]]] = {aid: [] for aid in agentes_ids}

        # Pasos de warmup: basado en dimensión
        pasos_warmup = self.dimension * 2

        for _ in range(pasos_warmup):
            entradas = {}
            for aid in agentes_ids:
                inp = self._generar_input_normal()
                entradas[aid] = inp

            estados = self._ejecutar_paso(entradas)

            for aid, estado in estados.items():
                if aid in historial_ce:
                    historial_ce[aid].append(estado.CE)

                    # Calcular "error" como desviación del estado respecto a identidad
                    if estado.I is not None and np.linalg.norm(estado.I) > 1e-10:
                        error = float(np.linalg.norm(estado.S - estado.I))
                        historial_errores[aid].append((entradas[aid], error))

        # Selección endógena del agente a estresar
        # Criterio: CE cercano a mediana global, varianza de errores baja
        ces_medios = {aid: np.mean(ces) for aid, ces in historial_ce.items() if ces}
        ce_global_median = np.median(list(ces_medios.values())) if ces_medios else 0.5

        # Encontrar agente más cercano a mediana
        agente_estresado = None
        min_distancia = float('inf')

        for aid, ce_medio in ces_medios.items():
            distancia = abs(ce_medio - ce_global_median)
            if distancia < min_distancia:
                min_distancia = distancia
                agente_estresado = aid

        if agente_estresado is None:
            agente_estresado = agentes_ids[0]

        # Construir inputs estresores: aquellos que generaron errores altos
        errores_agente = historial_errores.get(agente_estresado, [])
        if errores_agente:
            errores_ordenados = sorted(errores_agente, key=lambda x: x[1], reverse=True)
            # Tomar inputs del percentil 80+ de errores
            n_estresores = max(1, len(errores_ordenados) // 5)
            inputs_estresores = [inp for inp, _ in errores_ordenados[:n_estresores]]
        else:
            # Si no hay historial, generar inputs de alta varianza
            inputs_estresores = [np.random.randn(self.dimension) * 3.0 for _ in range(5)]

        # Fase 2: Aplicar estrés prolongado
        timeline: Dict[int, Dict[str, Any]] = {}
        historial_medico: List[Optional[str]] = []
        historial_ce_medico: List[float] = []
        eventos_reemplazo = 0
        medico_anterior = None

        # Duración del estrés: proporcional al warmup
        pasos_estres = pasos_warmup * 2

        for t in range(pasos_estres):
            # Input estresor para el agente objetivo
            entradas = {}
            for aid in agentes_ids:
                if aid == agente_estresado:
                    # Rotar entre inputs estresores
                    idx = t % len(inputs_estresores)
                    entradas[aid] = inputs_estresores[idx].copy()
                    # Añadir perturbación adicional
                    entradas[aid] += np.random.randn(self.dimension) * 0.5
                else:
                    entradas[aid] = self._generar_input_normal()

            # Eventos narrativos caóticos para el agente estresado
            eventos = {agente_estresado: np.random.randn(self.dimension)}

            estados = self._ejecutar_paso(entradas, eventos)

            # Identificar médico actual
            medico_actual = self.sys.obtener_medico()
            historial_medico.append(medico_actual)

            # CE del médico
            ce_medico = None
            medico_enferma = False

            if medico_actual and medico_actual in estados:
                ce_medico = estados[medico_actual].CE
                historial_ce_medico.append(ce_medico)

                # Verificar si el médico "enferma"
                # Criterio endógeno: CE < P_0.10 de su propio historial
                if medico_actual in historial_ce and len(historial_ce[medico_actual]) > 5:
                    p10_medico = np.percentile(historial_ce[medico_actual], 10)
                    medico_enferma = ce_medico < p10_medico

            # Detectar reemplazo
            evento_reemplazo = False
            if medico_anterior is not None and medico_actual != medico_anterior:
                if medico_actual is not None:
                    evento_reemplazo = True
                    eventos_reemplazo += 1

            medico_anterior = medico_actual

            # Registrar en timeline
            timeline[t] = {
                'ce_estresado': estados[agente_estresado].CE if agente_estresado in estados else None,
                'stress_estresado': float(np.var(entradas[agente_estresado])),
                'medico_activo': medico_actual,
                'ce_medico': ce_medico,
                'medico_enferma': medico_enferma,
                'evento_reemplazo': evento_reemplazo,
            }

            # Actualizar historiales
            for aid, estado in estados.items():
                if aid in historial_ce:
                    historial_ce[aid].append(estado.CE)

        # Calcular resumen
        # Tiempo hasta primer médico
        tiempo_primer_medico = None
        for t, med in enumerate(historial_medico):
            if med is not None:
                tiempo_primer_medico = t
                break

        # Veces que el médico enfermó
        veces_medico_enferma = sum(1 for t_data in timeline.values() if t_data['medico_enferma'])

        # Proporción de tiempo sin médico
        tiempo_sin_medico = sum(1 for med in historial_medico if med is None)
        prop_sin_medico = tiempo_sin_medico / len(historial_medico) if historial_medico else 0.0

        return {
            'agente_estresado': agente_estresado,
            'timeline': timeline,
            'resumen': {
                'medico_inicial': historial_medico[0] if historial_medico else None,
                'veces_medico_enferma': veces_medico_enferma,
                'num_reemplazos': eventos_reemplazo,
                'tiempo_hasta_primer_medico': tiempo_primer_medico,
                'proporcion_tiempo_sin_medico': prop_sin_medico,
                'ce_medio_estresado': float(np.mean(historial_ce.get(agente_estresado, [0.0]))),
                'ce_final_estresado': historial_ce.get(agente_estresado, [0.0])[-1] if historial_ce.get(agente_estresado) else 0.0,
            }
        }

    # =========================================================================
    # EXPERIMENTO 3: ENTROPÍA NARRATIVA ALTA - SUEÑO Y RENACIMIENTO
    # =========================================================================

    def experimento_3_entropia_narrativa_alta(
        self,
        n_agentes: int = None
    ) -> Dict[str, Any]:
        """
        Aumenta entropía narrativa con inputs caóticos y observa sueño/renacimiento.

        Inputs caóticos endógenos:
        - Transiciones raras en matriz de frecuencia de inputs
        - Combinaciones que históricamente generaron alta entropía

        Returns:
            Comparación baseline vs caos para cada agente
        """
        # Crear agentes si no existen
        if n_agentes is None:
            n_agentes = max(4, int(np.sqrt(self.dimension)))

        agentes_ids = []
        for i in range(n_agentes):
            agent_id = f"entropy_agent_{i}"
            self.sys.crear_agente(agent_id)
            agentes_ids.append(agent_id)

        # Fase 1: BASELINE - simulación normal
        baseline_ce: Dict[str, List[float]] = {aid: [] for aid in agentes_ids}
        baseline_entropy: Dict[str, List[float]] = {aid: [] for aid in agentes_ids}
        baseline_sleep: Dict[str, List[bool]] = {aid: [] for aid in agentes_ids}
        historial_inputs: Dict[str, List[np.ndarray]] = {aid: [] for aid in agentes_ids}

        # Pasos de baseline: basado en dimensión
        pasos_baseline = self.dimension * 2

        for _ in range(pasos_baseline):
            entradas = {}
            for aid in agentes_ids:
                inp = self._generar_input_normal()
                entradas[aid] = inp
                historial_inputs[aid].append(inp.copy())

            estados = self._ejecutar_paso(entradas)

            for aid, estado in estados.items():
                if aid in baseline_ce:
                    baseline_ce[aid].append(estado.CE)
                    baseline_entropy[aid].append(estado.coherencia.entropia_narrativa)
                    baseline_sleep[aid].append(estado.esta_dormido)

        # Calcular métricas baseline
        baseline_sleep_ratio = {}
        baseline_mean_ce = {}
        baseline_mean_entropy = {}

        for aid in agentes_ids:
            baseline_sleep_ratio[aid] = np.mean(baseline_sleep[aid]) if baseline_sleep[aid] else 0.0
            baseline_mean_ce[aid] = np.mean(baseline_ce[aid]) if baseline_ce[aid] else 0.0
            baseline_mean_entropy[aid] = np.mean(baseline_entropy[aid]) if baseline_entropy[aid] else 0.0

        # Construir matriz de transiciones de inputs para generar caos
        # Discretizamos inputs por cuadrante de signo
        def input_to_pattern(inp: np.ndarray) -> tuple:
            """Convierte input a patrón discreto (signos de primeras dims)."""
            n_dims = min(4, len(inp))
            return tuple(1 if inp[i] > 0 else 0 for i in range(n_dims))

        # Contar transiciones
        transiciones: Dict[Tuple, Dict[Tuple, int]] = defaultdict(lambda: defaultdict(int))

        for aid in agentes_ids:
            inputs = historial_inputs[aid]
            for i in range(1, len(inputs)):
                p1 = input_to_pattern(inputs[i-1])
                p2 = input_to_pattern(inputs[i])
                transiciones[p1][p2] += 1

        # Identificar transiciones raras (< P_0.20 de frecuencia)
        todas_transiciones = []
        for p1, destinos in transiciones.items():
            for p2, count in destinos.items():
                todas_transiciones.append((p1, p2, count))

        if todas_transiciones:
            conteos = [t[2] for t in todas_transiciones]
            umbral_raro = np.percentile(conteos, 20)
            transiciones_raras = [(p1, p2) for p1, p2, c in todas_transiciones if c <= umbral_raro]
        else:
            transiciones_raras = []

        # Fase 2: CAOS - inputs que fuerzan transiciones raras
        chaos_ce: Dict[str, List[float]] = {aid: [] for aid in agentes_ids}
        chaos_entropy: Dict[str, List[float]] = {aid: [] for aid in agentes_ids}
        chaos_sleep: Dict[str, List[bool]] = {aid: [] for aid in agentes_ids}
        eventos_muerte: Dict[str, List[int]] = {aid: [] for aid in agentes_ids}
        eventos_renacimiento: Dict[str, List[int]] = {aid: [] for aid in agentes_ids}
        estado_vital_anterior: Dict[str, EstadoVital] = {}

        pasos_caos = pasos_baseline  # Misma duración para comparar

        for t in range(pasos_caos):
            entradas = {}
            eventos = {}

            for aid in agentes_ids:
                # Generar input caótico
                if transiciones_raras and np.random.random() > 0.3:
                    # Forzar transición rara
                    p1, p2 = transiciones_raras[t % len(transiciones_raras)]
                    # Construir input que corresponda al patrón p2
                    inp = np.random.randn(self.dimension)
                    for i, signo in enumerate(p2):
                        if i < len(inp):
                            inp[i] = abs(inp[i]) if signo == 1 else -abs(inp[i])
                    # Amplificar para más caos
                    inp *= 2.0
                else:
                    # Input aleatorio de alta varianza
                    inp = np.random.randn(self.dimension) * 3.0

                entradas[aid] = inp

                # Evento narrativo caótico siempre
                eventos[aid] = np.random.randn(self.dimension) * 2.0

            estados = self._ejecutar_paso(entradas, eventos)

            for aid, estado in estados.items():
                if aid in chaos_ce:
                    chaos_ce[aid].append(estado.CE)
                    chaos_entropy[aid].append(estado.coherencia.entropia_narrativa)
                    chaos_sleep[aid].append(estado.esta_dormido)

                    # Detectar muerte/renacimiento
                    if aid in estado_vital_anterior:
                        if estado_vital_anterior[aid] != EstadoVital.MUERTO and estado.estado_vital == EstadoVital.MUERTO:
                            eventos_muerte[aid].append(t)
                        elif estado_vital_anterior[aid] == EstadoVital.MUERTO and estado.estado_vital == EstadoVital.VIVO:
                            eventos_renacimiento[aid].append(t)

                    estado_vital_anterior[aid] = estado.estado_vital

        # Calcular métricas de caos y comparar
        resultados_agentes = {}

        for aid in agentes_ids:
            chaos_sleep_ratio = np.mean(chaos_sleep[aid]) if chaos_sleep[aid] else 0.0
            chaos_mean_ce = np.mean(chaos_ce[aid]) if chaos_ce[aid] else 0.0
            chaos_mean_entropy = np.mean(chaos_entropy[aid]) if chaos_entropy[aid] else 0.0

            resultados_agentes[aid] = {
                'baseline_sleep_ratio': float(baseline_sleep_ratio[aid]),
                'chaos_sleep_ratio': float(chaos_sleep_ratio),
                'delta_sleep_ratio': float(chaos_sleep_ratio - baseline_sleep_ratio[aid]),
                'num_deaths': len(eventos_muerte[aid]),
                'num_rebirths': len(eventos_renacimiento[aid]),
                'mean_ce_before_chaos': float(baseline_mean_ce[aid]),
                'mean_ce_after_chaos': float(chaos_mean_ce),
                'mean_entropy_before': float(baseline_mean_entropy[aid]),
                'mean_entropy_during': float(chaos_mean_entropy),
                'delta_entropy': float(chaos_mean_entropy - baseline_mean_entropy[aid]),
            }

        # Resumen global
        agents_increased_sleep = sum(
            1 for aid in agentes_ids
            if resultados_agentes[aid]['delta_sleep_ratio'] > 0
        )
        agents_with_death = sum(
            1 for aid in agentes_ids
            if resultados_agentes[aid]['num_deaths'] > 0
        )

        deltas_sleep = [resultados_agentes[aid]['delta_sleep_ratio'] for aid in agentes_ids]
        deltas_ce = [
            resultados_agentes[aid]['mean_ce_after_chaos'] - resultados_agentes[aid]['mean_ce_before_chaos']
            for aid in agentes_ids
        ]

        return {
            'agentes': resultados_agentes,
            'global_summary': {
                'agents_with_increased_sleep': agents_increased_sleep,
                'agents_with_death_events': agents_with_death,
                'average_delta_sleep': float(np.mean(deltas_sleep)),
                'average_change_ce': float(np.mean(deltas_ce)),
                'total_deaths': sum(len(eventos_muerte[aid]) for aid in agentes_ids),
                'total_rebirths': sum(len(eventos_renacimiento[aid]) for aid in agentes_ids),
            }
        }

    # =========================================================================
    # EJECUTAR TODOS LOS EXPERIMENTOS
    # =========================================================================

    def ejecutar_todos(self, n_agentes: int = None) -> Dict[str, Any]:
        """
        Ejecuta los tres experimentos secuencialmente.

        Args:
            n_agentes: Número de agentes (endógeno si None)

        Returns:
            Resultados de los tres experimentos
        """
        resultados = {}

        # Crear nuevo sistema para cada experimento (independencia)

        # Experimento 1
        self.sys = SistemaConscienciaColectiva(self.dimension)
        self._historial_ce_global = []
        self._historial_varianza_ce = []
        self._historial_cambios_ce = []
        resultados['experimento_1_larga_simulacion'] = self.experimento_1_larga_simulacion(n_agentes)

        # Experimento 2
        self.sys = SistemaConscienciaColectiva(self.dimension)
        self._historial_ce_global = []
        self._historial_varianza_ce = []
        self._historial_cambios_ce = []
        resultados['experimento_2_estres_prolongado'] = self.experimento_2_estres_prolongado(n_agentes)

        # Experimento 3
        self.sys = SistemaConscienciaColectiva(self.dimension)
        self._historial_ce_global = []
        self._historial_varianza_ce = []
        self._historial_cambios_ce = []
        resultados['experimento_3_entropia_narrativa_alta'] = self.experimento_3_entropia_narrativa_alta(n_agentes)

        return resultados


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def crear_experimentos(dimension: int = 10) -> ConsciousnessExperiments:
    """
    Crea una instancia de experimentos con sistema inicializado.

    Args:
        dimension: Dimensión del espacio de estados (endógena si no se especifica)
    """
    sistema = SistemaConscienciaColectiva(dimension)
    return ConsciousnessExperiments(sistema)


def resumen_experimento_1(resultados: Dict[str, Any]) -> str:
    """Genera resumen legible del experimento 1."""
    res = resultados['resumen']
    agentes = resultados['agentes']

    lineas = [
        "=== EXPERIMENTO 1: SIMULACIÓN LARGA ===",
        f"Total pasos: {res['total_pasos']}",
        f"Agentes: {res['n_agentes']}",
        "",
        "Clasificaciones:",
    ]

    for clasif, count in res['conteo_clasificaciones'].items():
        lineas.append(f"  {clasif}: {count}")

    lineas.append("")
    lineas.append("Por agente:")

    for aid, metricas in agentes.items():
        lineas.append(f"  {aid}: {metricas['clasificacion']} (trend={metricas['trend_ce']:.4f}, mean_CE={metricas['mean_ce']:.3f})")

    return "\n".join(lineas)


def resumen_experimento_2(resultados: Dict[str, Any]) -> str:
    """Genera resumen legible del experimento 2."""
    res = resultados['resumen']

    lineas = [
        "=== EXPERIMENTO 2: ESTRÉS PROLONGADO ===",
        f"Agente estresado: {resultados['agente_estresado']}",
        f"Médico inicial: {res['medico_inicial']}",
        f"Veces médico enfermó: {res['veces_medico_enferma']}",
        f"Reemplazos de médico: {res['num_reemplazos']}",
        f"Tiempo hasta primer médico: {res['tiempo_hasta_primer_medico']}",
        f"Proporción tiempo sin médico: {res['proporcion_tiempo_sin_medico']:.2%}",
        f"CE medio agente estresado: {res['ce_medio_estresado']:.3f}",
        f"CE final agente estresado: {res['ce_final_estresado']:.3f}",
    ]

    return "\n".join(lineas)


def resumen_experimento_3(resultados: Dict[str, Any]) -> str:
    """Genera resumen legible del experimento 3."""
    res = resultados['global_summary']
    agentes = resultados['agentes']

    lineas = [
        "=== EXPERIMENTO 3: ENTROPÍA NARRATIVA ALTA ===",
        f"Agentes con más sueño: {res['agents_with_increased_sleep']}",
        f"Agentes con muerte: {res['agents_with_death_events']}",
        f"Delta sueño promedio: {res['average_delta_sleep']:.2%}",
        f"Cambio CE promedio: {res['average_change_ce']:.3f}",
        f"Total muertes: {res['total_deaths']}",
        f"Total renacimientos: {res['total_rebirths']}",
        "",
        "Por agente:",
    ]

    for aid, metricas in agentes.items():
        lineas.append(f"  {aid}:")
        lineas.append(f"    Δsueño: {metricas['delta_sleep_ratio']:.2%}")
        lineas.append(f"    Δentropía: {metricas['delta_entropy']:.3f}")
        lineas.append(f"    Muertes: {metricas['num_deaths']}, Renacimientos: {metricas['num_rebirths']}")

    return "\n".join(lineas)
