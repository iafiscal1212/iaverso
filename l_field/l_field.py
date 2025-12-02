"""
L-FIELD - Observador de Campo Latente Colectivo
===============================================

Módulo integrador que observa fenómenos colectivos emergentes
entre agentes SIN influir en ellos.

Métricas observadas:
1. LSI (Latent Synchrony Index) - Sincronía de fases
2. DIC (Deep Identity Coupling) - Acoplamiento de identidades
3. CD (Collective Drift) - Deriva grupal
4. Polarization - Formación de facciones
5. Narrative Resonance - Narrativas compartidas
6. Reinforcement Index - Sesgo de confirmación colectivo

Filosofía:
- 100% observacional
- 0% intervención
- Sin números mágicos (solo ε, 1/N, σ, percentiles)
- Los agentes son libres; L-FIELD solo mide

Este módulo es como un sociólogo estudiando una tribu:
observa patrones pero no dice "esto está mal" ni los cambia.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .synchrony import LatentSynchrony, SynchronySnapshot
from .correlations import DeepCorrelations, CorrelationSnapshot
from .collective_bias import CollectiveBias, BiasSnapshot


@dataclass
class LFieldSnapshot:
    """
    Estado completo del L-Field en un instante.

    Integra todas las métricas de los sub-observadores.
    """
    t: int

    # Sincronía
    LSI: float
    phase_coherence: float

    # Correlaciones
    DIC: float
    narrative_resonance: float

    # Sesgos
    collective_drift: float
    polarization: float
    reinforcement_index: float

    # Diagnóstico agregado
    collective_health: float        # Métrica sintética [0,1]
    dominant_phenomenon: str        # Qué fenómeno domina
    alert_level: str               # none, low, medium, high
    n_agents: int


class LField:
    """
    Campo Latente (L-Field) - Observador maestro de fenómenos colectivos.

    Integra tres sub-observadores:
    - LatentSynchrony: fases y sincronía
    - DeepCorrelations: identidades y narrativas
    - CollectiveBias: derivas y polarización

    Uso:
        l_field = LField()

        # En cada paso de simulación
        snapshot = l_field.observe(
            states={agent_id: S_vector},
            identities={agent_id: I_vector},
            narratives={agent_id: H_narr_vector},
            phases={agent_id: {'circadian_phase': θ, 'quantum_phase': φ, 'narrative_phase': ψ}}
        )

        print(f"LSI: {snapshot.LSI}, DIC: {snapshot.DIC}")
    """

    def __init__(self):
        """Inicializa el L-Field y sus sub-observadores."""
        self.t = 0
        self.eps = np.finfo(float).eps

        # Sub-observadores
        self._synchrony = LatentSynchrony()
        self._correlations = DeepCorrelations()
        self._bias = CollectiveBias()

        # Historial de snapshots integrados
        self._snapshots: List[LFieldSnapshot] = []

        # Estadísticas acumuladas
        self._n_observations = 0

    def observe(
        self,
        states: Dict[str, np.ndarray],
        identities: Dict[str, np.ndarray],
        narratives: Optional[Dict[str, np.ndarray]] = None,
        phases: Optional[Dict[str, Dict[str, float]]] = None,
        confirmation_weights: Optional[Dict[str, float]] = None
    ) -> LFieldSnapshot:
        """
        Observación completa del campo latente.

        Args:
            states: {agent_id: S(t)} - Estados internos
            identities: {agent_id: I(t)} - Vectores de identidad
            narratives: {agent_id: H_narr(t)} - Narrativas (opcional)
            phases: {agent_id: {phase_type: value}} - Fases para sincronía (opcional)
            confirmation_weights: {agent_id: w_i} - Pesos de confirmación (opcional)

        Returns:
            LFieldSnapshot con todas las métricas integradas
        """
        self.t += 1
        self._n_observations += 1
        n_agents = len(states)

        # === Observar sincronía ===
        if phases:
            sync_snapshot = self._synchrony.observe(phases)
            LSI = sync_snapshot.LSI
            phase_coherence = sync_snapshot.phase_coherence
        else:
            # Derivar fases de estados si no se proporcionan
            derived_phases = self._derive_phases(states)
            sync_snapshot = self._synchrony.observe(derived_phases)
            LSI = sync_snapshot.LSI
            phase_coherence = sync_snapshot.phase_coherence

        # === Observar correlaciones ===
        if narratives is None:
            # Usar estados como proxy de narrativas
            narratives = states

        corr_snapshot = self._correlations.observe(identities, narratives)
        DIC = corr_snapshot.DIC
        narrative_resonance = corr_snapshot.narrative_resonance

        # === Observar sesgos ===
        bias_snapshot = self._bias.observe(
            states, identities, confirmation_weights
        )
        collective_drift = bias_snapshot.collective_drift
        polarization = bias_snapshot.polarization
        reinforcement_index = bias_snapshot.reinforcement_index

        # === Calcular métricas agregadas ===
        collective_health = self._compute_collective_health(
            LSI, DIC, collective_drift, polarization, reinforcement_index
        )

        dominant_phenomenon = self._detect_dominant_phenomenon(
            LSI, DIC, collective_drift, polarization, reinforcement_index
        )

        alert_level = self._compute_alert_level(
            collective_drift, polarization, reinforcement_index
        )

        # === Crear snapshot integrado ===
        snapshot = LFieldSnapshot(
            t=self.t,
            LSI=LSI,
            phase_coherence=phase_coherence,
            DIC=DIC,
            narrative_resonance=narrative_resonance,
            collective_drift=collective_drift,
            polarization=polarization,
            reinforcement_index=reinforcement_index,
            collective_health=collective_health,
            dominant_phenomenon=dominant_phenomenon,
            alert_level=alert_level,
            n_agents=n_agents
        )

        self._snapshots.append(snapshot)

        # Limitar historial
        max_history = 500
        if len(self._snapshots) > max_history:
            self._snapshots = self._snapshots[-max_history:]

        return snapshot

    def _derive_phases(
        self,
        states: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Deriva fases de los estados cuando no se proporcionan explícitamente.

        Usa la fase del vector de estado en sus primeras 3 componentes
        principales como proxy de las fases circadiana, cuántica y narrativa.
        """
        phases = {}

        for agent_id, state in states.items():
            # Usar componentes del estado como proxies de fase
            n = len(state)

            # Fase "circadiana": ángulo de las primeras 2 componentes
            if n >= 2:
                circadian = np.arctan2(state[1], state[0])
            else:
                circadian = 0.0

            # Fase "cuántica": ángulo de componentes 2-3
            if n >= 4:
                quantum = np.arctan2(state[3], state[2])
            else:
                quantum = circadian

            # Fase "narrativa": ángulo de componentes 4-5
            if n >= 6:
                narrative = np.arctan2(state[5], state[4])
            else:
                narrative = quantum

            phases[agent_id] = {
                'circadian_phase': float(circadian),
                'quantum_phase': float(quantum),
                'narrative_phase': float(narrative)
            }

        return phases

    def _compute_collective_health(
        self,
        LSI: float,
        DIC: float,
        CD: float,
        PI: float,
        RI: float
    ) -> float:
        """
        Calcula una métrica sintética de "salud colectiva".

        NO es un juicio de valor. Es una medida de:
        - Coherencia sin homogeneización extrema
        - Diversidad sin fragmentación extrema

        Rango: [0, 1]
        - 1 = equilibrio ideal
        - 0 = extremos (ya sea homogeneización total o fragmentación total)

        Fórmula endógena:
        health = (1 - |2*LSI - 1|) * (1 - |2*DIC - 1|) *
                 (1 - CD_norm) * (1 - PI_norm) * (1 - |2*RI - 1|)

        Donde las normas son respecto a los máximos observados.
        """
        # Normalizar LSI [-1,1] -> [0,1] donde 0.5 es óptimo
        lsi_health = 1 - abs(LSI)  # |LSI|=0 es óptimo, =1 es extremo

        # DIC [-1,1] -> [0,1] donde 0 es óptimo (diversidad)
        dic_health = 1 - abs(DIC)

        # CD: normalizar por historial (endógeno)
        cd_history = self._bias.get_cd_history()
        if cd_history:
            cd_max = max(cd_history) + self.eps
            cd_norm = min(CD / cd_max, 1.0)
        else:
            cd_norm = 0.0
        cd_health = 1 - cd_norm

        # PI: normalizar por historial
        pi_history = self._bias.get_polarization_history()
        if pi_history:
            pi_max = max(pi_history) + self.eps
            pi_norm = min(PI / pi_max, 1.0)
        else:
            pi_norm = 0.0
        pi_health = 1 - pi_norm

        # RI: 0.5 es óptimo (neutral)
        ri_health = 1 - abs(2 * RI - 1)

        # Combinar con pesos iguales (1/5 cada uno)
        K = 5
        health = (lsi_health + dic_health + cd_health + pi_health + ri_health) / K

        return float(np.clip(health, 0, 1))

    def _detect_dominant_phenomenon(
        self,
        LSI: float,
        DIC: float,
        CD: float,
        PI: float,
        RI: float
    ) -> str:
        """
        Detecta qué fenómeno colectivo domina actualmente.

        Basado en cuál métrica está más alejada de su valor neutral.
        """
        # Calcular "extremidad" de cada métrica
        extremities = {
            'synchrony': abs(LSI),           # Alto |LSI| = sincronía/antisincronía fuerte
            'identity_coupling': abs(DIC),   # Alto |DIC| = acoplamiento fuerte
            'collective_drift': CD,          # CD alto = deriva fuerte
            'polarization': PI,              # PI alto = fragmentación
            'confirmation_bias': abs(2 * RI - 1)  # Lejos de 0.5 = sesgo fuerte
        }

        # Normalizar por medianas históricas si hay suficiente historial
        if len(self._snapshots) > 10:
            medians = {
                'synchrony': np.median([abs(s.LSI) for s in self._snapshots]),
                'identity_coupling': np.median([abs(s.DIC) for s in self._snapshots]),
                'collective_drift': np.median([s.collective_drift for s in self._snapshots]),
                'polarization': np.median([s.polarization for s in self._snapshots]),
                'confirmation_bias': np.median([abs(2*s.reinforcement_index-1) for s in self._snapshots])
            }

            # Calcular cuántas σ por encima de la mediana
            for key in extremities:
                if medians[key] > self.eps:
                    extremities[key] = extremities[key] / medians[key]

        # Retornar el más extremo
        dominant = max(extremities, key=extremities.get)
        return dominant

    def _compute_alert_level(
        self,
        CD: float,
        PI: float,
        RI: float
    ) -> str:
        """
        Calcula nivel de alerta basado en umbrales endógenos.

        Los umbrales se derivan del propio historial:
        - high: > percentil 90
        - medium: > percentil 75
        - low: > percentil 50
        - none: resto

        NO bloquea nada. Solo informa.
        """
        cd_history = self._bias.get_cd_history()
        pi_history = self._bias.get_polarization_history()
        ri_history = self._bias.get_ri_history()

        # Si no hay suficiente historial, no hay alerta
        if len(cd_history) < 10:
            return "none"

        # Calcular percentiles endógenos
        cd_p90, cd_p75, cd_p50 = np.percentile(cd_history, [90, 75, 50])
        pi_p90, pi_p75, pi_p50 = np.percentile(pi_history, [90, 75, 50])
        ri_extreme = [abs(2*r - 1) for r in ri_history]
        ri_p90, ri_p75, ri_p50 = np.percentile(ri_extreme, [90, 75, 50])

        ri_current = abs(2 * RI - 1)

        # Evaluar cada métrica
        high_count = 0
        medium_count = 0
        low_count = 0

        for val, p90, p75, p50 in [
            (CD, cd_p90, cd_p75, cd_p50),
            (PI, pi_p90, pi_p75, pi_p50),
            (ri_current, ri_p90, ri_p75, ri_p50)
        ]:
            if val > p90:
                high_count += 1
            elif val > p75:
                medium_count += 1
            elif val > p50:
                low_count += 1

        # Nivel de alerta agregado
        if high_count >= 2:
            return "high"
        elif high_count >= 1 or medium_count >= 2:
            return "medium"
        elif medium_count >= 1 or low_count >= 2:
            return "low"
        else:
            return "none"

    # === Acceso a historiales ===

    def get_snapshots(self) -> List[LFieldSnapshot]:
        """Retorna todos los snapshots."""
        return self._snapshots.copy()

    def get_recent_snapshots(self, n: int = 10) -> List[LFieldSnapshot]:
        """Retorna los n snapshots más recientes."""
        return self._snapshots[-n:] if self._snapshots else []

    def get_lsi_history(self) -> List[float]:
        """Retorna historial de LSI."""
        return self._synchrony.get_lsi_history()

    def get_dic_history(self) -> List[float]:
        """Retorna historial de DIC."""
        return self._correlations.get_dic_history()

    def get_cd_history(self) -> List[float]:
        """Retorna historial de Collective Drift."""
        return self._bias.get_cd_history()

    def get_polarization_history(self) -> List[float]:
        """Retorna historial de Polarization."""
        return self._bias.get_polarization_history()

    # === Estadísticas ===

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas completas del L-Field."""
        sync_stats = self._synchrony.get_statistics()
        corr_stats = self._correlations.get_statistics()
        bias_stats = self._bias.get_statistics()

        # Estadísticas de health
        if self._snapshots:
            healths = [s.collective_health for s in self._snapshots]
            health_stats = {
                'health_mean': float(np.mean(healths)),
                'health_std': float(np.std(healths)),
                'health_current': healths[-1],
                'health_min': float(np.min(healths)),
                'health_max': float(np.max(healths))
            }
        else:
            health_stats = {}

        # Distribución de fenómenos dominantes
        if self._snapshots:
            phenomena = [s.dominant_phenomenon for s in self._snapshots]
            phenomenon_counts = {}
            for p in phenomena:
                phenomenon_counts[p] = phenomenon_counts.get(p, 0) + 1
            for p in phenomenon_counts:
                phenomenon_counts[p] /= len(phenomena)
        else:
            phenomenon_counts = {}

        # Distribución de alertas
        if self._snapshots:
            alerts = [s.alert_level for s in self._snapshots]
            alert_counts = {}
            for a in alerts:
                alert_counts[a] = alert_counts.get(a, 0) + 1
            for a in alert_counts:
                alert_counts[a] /= len(alerts)
        else:
            alert_counts = {}

        return {
            't': self.t,
            'n_observations': self._n_observations,
            'synchrony': sync_stats,
            'correlations': corr_stats,
            'bias': bias_stats,
            'collective_health': health_stats,
            'dominant_phenomena': phenomenon_counts,
            'alert_distribution': alert_counts
        }

    def get_summary(self) -> str:
        """Retorna un resumen legible del estado actual."""
        if not self._snapshots:
            return "L-Field: Sin observaciones aún"

        s = self._snapshots[-1]
        stats = self.get_statistics()

        summary = f"""
L-FIELD Status (t={s.t}, agents={s.n_agents})
{'='*50}
Synchrony:
  LSI: {s.LSI:.3f} (phase coherence: {s.phase_coherence:.3f})

Correlations:
  DIC: {s.DIC:.3f} (identity coupling)
  NR:  {s.narrative_resonance:.3f} (narrative resonance)

Collective Bias:
  CD:  {s.collective_drift:.4f} (collective drift)
  PI:  {s.polarization:.4f} (polarization)
  RI:  {s.reinforcement_index:.3f} (reinforcement index)

Aggregate:
  Collective Health: {s.collective_health:.3f}
  Dominant Phenomenon: {s.dominant_phenomenon}
  Alert Level: {s.alert_level}
{'='*50}
"""
        return summary
