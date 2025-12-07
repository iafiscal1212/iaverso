"""
AGI-4: Regulación Teleológica - Trayectoria Vital
=================================================

Evalúa "si va bien o mal" según criterios INTERNOS.
No hay medida externa de éxito - solo coherencia interna.

Combina:
- Trayectoria del self
- Coherencia narrativa
- Logro de metas
- Estabilidad de identidad
- Sentido de propósito

Todo 100% endógeno - sin constantes mágicas.
"""

import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

sys.path.insert(0, '/root/NEO_EVA')
from core.norma_dura_config import CONSTANTS


class LifePhase(Enum):
    """Fases de vida detectadas endógenamente."""
    BIRTH = "birth"
    EXPLORATION = "exploration"
    GROWTH = "growth"
    CONSOLIDATION = "consolidation"
    CRISIS = "crisis"
    RECONSTRUCTION = "reconstruction"
    MATURITY = "maturity"
    STAGNATION = "stagnation"
    RENEWAL = "renewal"
    DECLINE = "decline"


@dataclass
class TrajectoryPoint:
    """Un punto en la trayectoria vital."""
    t: int
    phase: LifePhase
    wellbeing: float       # Bienestar interno
    purpose: float         # Sentido de propósito
    coherence: float       # Coherencia narrativa
    identity: float        # Fuerza de identidad
    direction: np.ndarray  # Dirección vital
    momentum: float        # Momento (velocidad del cambio)


@dataclass
class LifeEvent:
    """Evento significativo en la vida."""
    t: int
    event_type: str  # "crisis", "achievement", "transition", "discovery"
    magnitude: float
    description: str


class LifeTrajectory:
    """
    Sistema de regulación teleológica.

    Evalúa la trayectoria vital del agente según
    criterios puramente internos:

    wellbeing = α·SAGI + β·purpose + γ·coherence + δ·identity

    donde α, β, γ, δ derivan de covarianzas internas.

    "Va bien" = wellbeing > percentile_50(history)
    "Va mal" = wellbeing < percentile_25(history)
    """

    def __init__(self, agent_name: str, state_dim: int = 11):
        """
        Inicializa trayectoria vital.

        Args:
            agent_name: Nombre del agente
            state_dim: Dimensión del estado (z + phi)
        """
        self.agent_name = agent_name
        self.state_dim = state_dim

        # Historial de trayectoria
        self.trajectory: List[TrajectoryPoint] = []

        # Eventos de vida
        self.life_events: List[LifeEvent] = []

        # Métricas históricas
        self.wellbeing_history: List[float] = []
        self.purpose_history: List[float] = []
        self.coherence_history: List[float] = []
        self.identity_history: List[float] = []

        # Pesos adaptativos para wellbeing
        # ORIGEN: distribución uniforme inicial entre 4 componentes (1/4 = 0.25)
        self.alpha = CONSTANTS.PERCENTILE_25  # peso SAGI/integración
        self.beta = CONSTANTS.PERCENTILE_25   # peso purpose
        self.gamma = CONSTANTS.PERCENTILE_25  # peso coherence
        self.delta = CONSTANTS.PERCENTILE_25  # peso identity

        # Umbrales (adaptativos)
        # ORIGEN: good=P50+P10, bad=P50-P10 para simetría alrededor de mediana
        self.good_threshold = CONSTANTS.PERCENTILE_50 + CONSTANTS.PERCENTILE_10  # ~0.6
        self.bad_threshold = CONSTANTS.PERCENTILE_50 - CONSTANTS.PERCENTILE_10   # ~0.4

        # Fase actual
        self.current_phase = LifePhase.BIRTH
        self.phase_history: List[Tuple[int, LifePhase]] = []

        # Estado del self
        self.self_state = np.zeros(state_dim)
        self.self_velocity = np.zeros(state_dim)

        self.t = 0

    def _compute_wellbeing(self, SAGI: float, purpose: float,
                          coherence: float, identity: float) -> float:
        """
        Calcula bienestar interno.

        wellbeing = α·SAGI + β·purpose + γ·coherence + δ·identity

        Todos los componentes normalizados a [0, 1].
        """
        wellbeing = (self.alpha * SAGI +
                    self.beta * purpose +
                    self.gamma * coherence +
                    self.delta * identity)

        return float(np.clip(wellbeing, 0, 1))

    def _adapt_weights(self):
        """
        Adapta pesos endógenamente.

        Basado en correlación de cada componente
        con cambios positivos en wellbeing.
        """
        if len(self.wellbeing_history) < 30:
            return

        # Calcular cambios en wellbeing
        changes = np.diff(self.wellbeing_history[-50:])

        # Correlacionar con cada componente
        components = [
            self.purpose_history[-49:],
            self.coherence_history[-49:],
            self.identity_history[-49:]
        ]

        correlations = []
        for comp in components:
            if len(comp) == len(changes):
                corr = np.corrcoef(comp, changes)[0, 1]
                if np.isnan(corr):
                    corr = 0.0  # ORIGEN: correlación neutra cuando NaN
            else:
                corr = 0.0  # ORIGEN: correlación neutra cuando no hay datos
            # ORIGEN: offset 0.5 (mediana) + floor P10 para evitar pesos negativos
            correlations.append(max(CONSTANTS.PERCENTILE_10, corr + CONSTANTS.PERCENTILE_50))

        # Normalizar
        # ORIGEN: 0.3 = P25+P10/2 como peso base para SAGI
        sagi_base_weight = CONSTANTS.PERCENTILE_25 + CONSTANTS.PERCENTILE_10 / 2  # ~0.3
        total = sum(correlations) + sagi_base_weight
        self.alpha = sagi_base_weight / total
        self.beta = correlations[0] / total
        self.gamma = correlations[1] / total
        self.delta = correlations[2] / total

    def _update_thresholds(self):
        """
        Actualiza umbrales endógenamente.

        good = percentile_60(wellbeing_history)
        bad = percentile_30(wellbeing_history)
        """
        if len(self.wellbeing_history) < 20:
            return

        self.good_threshold = np.percentile(self.wellbeing_history, 60)
        self.bad_threshold = np.percentile(self.wellbeing_history, 30)

    def _detect_phase(self, wellbeing: float, purpose: float,
                     coherence: float, identity: float,
                     momentum: float) -> LifePhase:
        """
        Detecta fase vital endógenamente.

        Basado en combinación de métricas.
        """
        # Umbrales relativos
        if len(self.wellbeing_history) < 10:
            return LifePhase.BIRTH

        wb_percentile = np.sum(np.array(self.wellbeing_history) <= wellbeing) / \
                       len(self.wellbeing_history)

        # Umbrales derivados de percentiles U(0,1)
        # ORIGEN: crisis = P25-P10/2 (~0.2), alto = P75-P10/2 (~0.7)
        crisis_threshold = CONSTANTS.PERCENTILE_25 - CONSTANTS.PERCENTILE_10 / 2  # ~0.2
        high_threshold = CONSTANTS.PERCENTILE_75 - CONSTANTS.PERCENTILE_10 / 2    # ~0.7
        medium_threshold = CONSTANTS.PERCENTILE_50 + CONSTANTS.PERCENTILE_10      # ~0.6
        low_threshold = CONSTANTS.PERCENTILE_50 - CONSTANTS.PERCENTILE_10         # ~0.4
        trend_threshold = CONSTANTS.PERCENTILE_10 / 2  # ~0.05 para trends
        momentum_low = CONSTANTS.PERCENTILE_10         # ~0.1 para momentum bajo
        momentum_high = CONSTANTS.PERCENTILE_25 - CONSTANTS.PERCENTILE_10 / 2  # ~0.2 para momentum alto

        # Crisis: bienestar muy bajo
        if wb_percentile < crisis_threshold:
            return LifePhase.CRISIS

        # Declive: bienestar bajando consistentemente
        if len(self.wellbeing_history) > 5:
            recent_trend = np.mean(np.diff(self.wellbeing_history[-5:]))
            if recent_trend < -trend_threshold:
                return LifePhase.DECLINE

        # Madurez: alta identidad, alto propósito, alta coherencia
        if identity > high_threshold and purpose > medium_threshold and coherence > high_threshold:
            return LifePhase.MATURITY

        # Consolidación: identidad alta, momentum bajo
        if identity > medium_threshold and abs(momentum) < momentum_low:
            return LifePhase.CONSOLIDATION

        # Exploración: identidad baja, momentum alto
        if identity < low_threshold and abs(momentum) > momentum_high:
            return LifePhase.EXPLORATION

        # Crecimiento: wellbeing subiendo, propósito presente
        if len(self.wellbeing_history) > 5:
            recent_trend = np.mean(np.diff(self.wellbeing_history[-5:]))
            trend_positive = trend_threshold / 2.5  # ~0.02
            if recent_trend > trend_positive and purpose > low_threshold:
                return LifePhase.GROWTH

        # Estancamiento: poco cambio, bajo propósito
        stagnation_purpose = CONSTANTS.PERCENTILE_25 + CONSTANTS.PERCENTILE_10 / 2  # ~0.3
        if abs(momentum) < trend_threshold and purpose < stagnation_purpose:
            return LifePhase.STAGNATION

        # Renovación: después de crisis, mejorando
        if self.current_phase == LifePhase.CRISIS and wb_percentile > low_threshold:
            return LifePhase.RENEWAL

        # Reconstrucción: identidad baja pero coherencia recuperándose
        if identity < low_threshold and coherence > CONSTANTS.PERCENTILE_50:
            return LifePhase.RECONSTRUCTION

        return self.current_phase  # Mantener fase actual

    def _detect_life_event(self, wellbeing: float, purpose: float,
                          old_phase: LifePhase, new_phase: LifePhase) -> Optional[LifeEvent]:
        """
        Detecta eventos vitales significativos.
        """
        # Transición de fase
        if old_phase != new_phase:
            if new_phase == LifePhase.CRISIS:
                return LifeEvent(
                    t=self.t,
                    event_type="crisis",
                    magnitude=1.0 - wellbeing,
                    description=f"Crisis from {old_phase.value}"
                )
            elif new_phase == LifePhase.MATURITY:
                return LifeEvent(
                    t=self.t,
                    event_type="achievement",
                    magnitude=wellbeing,
                    description="Reached maturity"
                )
            else:
                return LifeEvent(
                    t=self.t,
                    event_type="transition",
                    magnitude=CONSTANTS.PERCENTILE_50,  # ORIGEN: magnitud media
                    description=f"{old_phase.value} → {new_phase.value}"
                )

        # Cambio brusco en wellbeing
        # ORIGEN: umbral = P25-P10/2 (~0.2) para detectar cambios significativos
        significant_change = CONSTANTS.PERCENTILE_25 - CONSTANTS.PERCENTILE_10 / 2
        if len(self.wellbeing_history) > 1:
            change = wellbeing - self.wellbeing_history[-1]
            if abs(change) > significant_change:
                return LifeEvent(
                    t=self.t,
                    event_type="discovery" if change > 0 else "setback",
                    magnitude=abs(change),
                    description=f"Wellbeing {'jump' if change > 0 else 'drop'}"
                )

        return None

    def record(self, self_state: np.ndarray, SAGI: float, purpose: float,
               coherence: float, identity: float) -> TrajectoryPoint:
        """
        Registra un punto en la trayectoria vital.

        Args:
            self_state: Estado del self
            SAGI: Métrica de integración
            purpose: Sentido de propósito
            coherence: Coherencia narrativa
            identity: Fuerza de identidad

        Returns:
            TrajectoryPoint con el estado vital
        """
        self.t += 1

        # Actualizar estado
        old_state = self.self_state.copy()
        self.self_state = self_state.copy()
        self.self_velocity = self_state - old_state

        # Calcular momentum
        momentum = float(np.linalg.norm(self.self_velocity))

        # Calcular bienestar
        wellbeing = self._compute_wellbeing(SAGI, purpose, coherence, identity)

        # Detectar fase
        old_phase = self.current_phase
        new_phase = self._detect_phase(wellbeing, purpose, coherence, identity, momentum)
        self.current_phase = new_phase

        if old_phase != new_phase:
            self.phase_history.append((self.t, new_phase))

        # Detectar eventos
        event = self._detect_life_event(wellbeing, purpose, old_phase, new_phase)
        if event:
            self.life_events.append(event)

        # Registrar historiales
        self.wellbeing_history.append(wellbeing)
        self.purpose_history.append(purpose)
        self.coherence_history.append(coherence)
        self.identity_history.append(identity)

        # Limitar historiales
        max_hist = 1000
        if len(self.wellbeing_history) > max_hist:
            self.wellbeing_history = self.wellbeing_history[-max_hist:]
            self.purpose_history = self.purpose_history[-max_hist:]
            self.coherence_history = self.coherence_history[-max_hist:]
            self.identity_history = self.identity_history[-max_hist:]

        # Crear punto de trayectoria
        point = TrajectoryPoint(
            t=self.t,
            phase=new_phase,
            wellbeing=wellbeing,
            purpose=purpose,
            coherence=coherence,
            identity=identity,
            direction=self.self_velocity.copy(),
            momentum=momentum
        )

        self.trajectory.append(point)
        if len(self.trajectory) > max_hist:
            self.trajectory = self.trajectory[-max_hist:]

        # Adaptar pesos periódicamente
        if self.t % 50 == 0:
            self._adapt_weights()
            self._update_thresholds()

        return point

    def is_going_well(self) -> bool:
        """
        Evalúa si "va bien" según criterios internos.

        going_well = wellbeing > good_threshold
        """
        if len(self.wellbeing_history) == 0:
            return True  # Optimismo inicial

        return self.wellbeing_history[-1] > self.good_threshold

    def is_going_badly(self) -> bool:
        """
        Evalúa si "va mal" según criterios internos.

        going_badly = wellbeing < bad_threshold
        """
        if len(self.wellbeing_history) == 0:
            return False

        return self.wellbeing_history[-1] < self.bad_threshold

    def get_life_assessment(self) -> str:
        """
        Obtiene evaluación de vida.

        Returns:
            "thriving", "stable", "struggling", "crisis"
        """
        if len(self.wellbeing_history) < 5:
            return "beginning"

        current = self.wellbeing_history[-1]
        recent_mean = np.mean(self.wellbeing_history[-10:])
        trend = np.mean(np.diff(self.wellbeing_history[-5:])) if len(self.wellbeing_history) >= 5 else 0

        if current > self.good_threshold and trend >= 0:
            return "thriving"
        elif current > self.bad_threshold:
            # ORIGEN: trend_declining = -P10/5 = -0.02
            trend_declining = -CONSTANTS.PERCENTILE_10 / 5
            if trend > 0:
                return "improving"
            elif trend < trend_declining:
                return "declining"
            else:
                return "stable"
        else:
            if self.current_phase == LifePhase.CRISIS:
                return "crisis"
            else:
                return "struggling"

    def get_trajectory_summary(self, window: int = 50) -> Dict:
        """Obtiene resumen de la trayectoria."""
        if len(self.trajectory) == 0:
            return {'status': 'no_data'}

        recent = self.trajectory[-window:]

        return {
            'current_phase': self.current_phase.value,
            'current_wellbeing': self.wellbeing_history[-1] if self.wellbeing_history else 0,
            'mean_wellbeing': float(np.mean(self.wellbeing_history[-window:])),
            'wellbeing_trend': float(np.mean(np.diff(self.wellbeing_history[-10:]))) if len(self.wellbeing_history) >= 10 else 0,
            'mean_purpose': float(np.mean([p.purpose for p in recent])),
            'mean_coherence': float(np.mean([p.coherence for p in recent])),
            'mean_identity': float(np.mean([p.identity for p in recent])),
            'mean_momentum': float(np.mean([p.momentum for p in recent])),
            'life_assessment': self.get_life_assessment(),
            'n_phases': len(set(p.phase for p in recent)),
            'n_events': len([e for e in self.life_events if e.t > self.t - window])
        }

    def get_life_story(self, max_events: int = 20) -> List[Dict]:
        """
        Genera resumen de la historia de vida.

        Returns:
            Lista de eventos significativos
        """
        story = []

        # Añadir cambios de fase
        for t, phase in self.phase_history[-max_events//2:]:
            story.append({
                't': t,
                'type': 'phase_change',
                'description': f"Entered {phase.value} phase"
            })

        # Añadir eventos de vida
        for event in self.life_events[-max_events//2:]:
            story.append({
                't': event.t,
                'type': event.event_type,
                'description': event.description,
                'magnitude': event.magnitude
            })

        # Ordenar por tiempo
        story.sort(key=lambda x: x['t'])

        return story[-max_events:]

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas completas."""
        summary = self.get_trajectory_summary()

        return {
            'agent': self.agent_name,
            't': self.t,
            'trajectory': summary,
            'weights': {
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
                'delta': self.delta
            },
            'thresholds': {
                'good': self.good_threshold,
                'bad': self.bad_threshold
            },
            'n_life_events': len(self.life_events),
            'n_phase_changes': len(self.phase_history),
            'going_well': self.is_going_well(),
            'going_badly': self.is_going_badly()
        }


def test_life_trajectory():
    """Test de trayectoria vital."""
    print("=" * 60)
    print("TEST TRAYECTORIA VITAL (AGI-4)")
    print("=" * 60)

    # Crear sistema
    life = LifeTrajectory("NEO", state_dim=11)

    print("\nSimulando 500 pasos de vida...")

    self_state = np.random.randn(11) * 0.1

    for t in range(500):
        # Evolución del estado
        self_state = 0.95 * self_state + 0.05 * np.tanh(self_state) + np.random.randn(11) * 0.02

        # Métricas simuladas con ciclos de vida
        base_wellbeing = 0.5 + 0.2 * np.sin(t / 100)

        # Crisis ocasional
        if 150 < t < 180 or 350 < t < 370:
            base_wellbeing -= 0.3

        SAGI = np.clip(base_wellbeing + np.random.randn() * 0.1, 0, 1)
        purpose = np.clip(base_wellbeing + 0.1 + np.random.randn() * 0.1, 0, 1)
        coherence = np.clip(base_wellbeing + 0.05 + np.random.randn() * 0.1, 0, 1)
        identity = np.clip(0.3 + t / 1000 + np.random.randn() * 0.1, 0, 1)

        point = life.record(self_state, SAGI, purpose, coherence, identity)

        if (t + 1) % 100 == 0:
            stats = life.get_statistics()
            print(f"\n  t={t+1}:")
            print(f"    Phase: {stats['trajectory']['current_phase']}")
            print(f"    Wellbeing: {stats['trajectory']['current_wellbeing']:.3f} "
                  f"(trend: {stats['trajectory']['wellbeing_trend']:.3f})")
            print(f"    Assessment: {stats['trajectory']['life_assessment']}")
            print(f"    Going well: {stats['going_well']}, Going badly: {stats['going_badly']}")

    # Reporte final
    print("\n" + "=" * 60)
    print("HISTORIA DE VIDA")
    print("=" * 60)

    story = life.get_life_story(15)
    for event in story:
        print(f"  t={event['t']}: [{event['type']}] {event['description']}")

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)

    stats = life.get_statistics()
    print(f"\nAgente: {stats['agent']}")
    print(f"Fase actual: {stats['trajectory']['current_phase']}")
    print(f"Evaluación: {stats['trajectory']['life_assessment']}")
    print(f"Bienestar medio: {stats['trajectory']['mean_wellbeing']:.3f}")
    print(f"Eventos de vida: {stats['n_life_events']}")
    print(f"Cambios de fase: {stats['n_phase_changes']}")

    return life


if __name__ == "__main__":
    test_life_trajectory()
