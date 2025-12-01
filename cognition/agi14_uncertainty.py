"""
AGI-14: Introspective Uncertainty & Calibration
================================================

"Saber cuándo no sabe" (puro estructural).

Uncertainty sobre predicciones:
    e_t = y_t - ŷ_t
    μ_e = E[e_t], σ_e² = V[e_t]

Incertidumbre normalizada:
    U = σ_e / median(σ_e)
    U_rank = rank(U)

Confianza estructural:
    conf = 1 - U_rank / N

Uso en decisiones:
    η_explore = η_t · U_rank
    η_exploit = η_t · conf

100% endógeno.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class PredictionChannel(Enum):
    """Canales de predicción interna."""
    SELF_MODEL = "self_model"
    THEORY_OF_MIND = "theory_of_mind"
    WORLD_MODEL = "world_model"
    VALUE_PREDICTION = "value_prediction"
    CRISIS_PREDICTION = "crisis_prediction"
    GOAL_PROGRESS = "goal_progress"


@dataclass
class ChannelUncertainty:
    """Incertidumbre de un canal."""
    channel: PredictionChannel
    predictions: List[float] = field(default_factory=list)
    actuals: List[float] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)
    mean_error: float = 0.0
    std_error: float = 0.0
    uncertainty: float = 0.5
    uncertainty_rank: float = 0.5


@dataclass
class UncertaintyState:
    """Estado de incertidumbre global."""
    t: int
    channel_uncertainties: Dict[PredictionChannel, float]
    global_uncertainty: float
    confidence: float
    should_explore: bool
    exploration_weight: float
    exploitation_weight: float


class IntrospectiveUncertainty:
    """
    Sistema de incertidumbre introspectiva.

    Monitorea qué tan bien funcionan las predicciones internas
    y ajusta comportamiento según confianza.
    """

    def __init__(self, agent_name: str):
        """
        Inicializa sistema de incertidumbre.

        Args:
            agent_name: Nombre del agente
        """
        self.agent_name = agent_name

        # Canales de predicción
        self.channels: Dict[PredictionChannel, ChannelUncertainty] = {
            ch: ChannelUncertainty(channel=ch) for ch in PredictionChannel
        }

        # Historial global
        self.global_uncertainty_history: List[float] = []
        self.confidence_history: List[float] = []

        # Learning rate base
        self.eta_base: float = 0.1

        self.t = 0

    def _update_channel(self, channel: PredictionChannel,
                       prediction: float, actual: float):
        """
        Actualiza un canal con nueva predicción.

        e_t = y_t - ŷ_t
        """
        ch = self.channels[channel]

        error = actual - prediction
        ch.predictions.append(prediction)
        ch.actuals.append(actual)
        ch.errors.append(error)

        # Limitar historial
        max_hist = 200
        if len(ch.errors) > max_hist:
            ch.predictions = ch.predictions[-max_hist:]
            ch.actuals = ch.actuals[-max_hist:]
            ch.errors = ch.errors[-max_hist:]

        # Calcular estadísticas sobre ventana
        window = int(np.ceil(np.sqrt(len(ch.errors))))
        window = max(5, min(window, len(ch.errors)))

        recent_errors = ch.errors[-window:]
        ch.mean_error = float(np.mean(recent_errors))
        ch.std_error = float(np.std(recent_errors))

    def _compute_uncertainties(self):
        """
        Calcula incertidumbres normalizadas.

        U = σ_e / median(σ_e)
        U_rank = rank(U)
        """
        # Recolectar std de todos los canales
        stds = []
        for ch in self.channels.values():
            if len(ch.errors) >= 5:
                stds.append(ch.std_error)

        if not stds:
            return

        median_std = np.median(stds)

        for ch in self.channels.values():
            if len(ch.errors) < 5:
                ch.uncertainty = 0.5
                ch.uncertainty_rank = 0.5
                continue

            # U = σ_e / median(σ_e)
            ch.uncertainty = ch.std_error / (median_std + 1e-8)

            # U_rank = rank(U)
            ch.uncertainty_rank = np.sum(np.array(stds) <= ch.std_error) / len(stds)

    def record_prediction(self, channel: PredictionChannel,
                         prediction: float, actual: float):
        """
        Registra una predicción y su resultado real.

        Args:
            channel: Canal de predicción
            prediction: Valor predicho
            actual: Valor real observado
        """
        self.t += 1

        self._update_channel(channel, prediction, actual)

        # Recalcular incertidumbres
        if self.t % 5 == 0:
            self._compute_uncertainties()

    def get_uncertainty_state(self) -> UncertaintyState:
        """
        Obtiene estado de incertidumbre actual.

        conf = 1 - U_rank / N

        Returns:
            UncertaintyState
        """
        # Incertidumbres por canal
        channel_uncertainties = {
            ch.channel: ch.uncertainty_rank
            for ch in self.channels.values()
        }

        # Incertidumbre global (media de ranks)
        valid_ranks = [ch.uncertainty_rank for ch in self.channels.values()
                      if len(ch.errors) >= 5]

        if valid_ranks:
            global_uncertainty = float(np.mean(valid_ranks))
        else:
            global_uncertainty = 0.5

        # Confianza = 1 - incertidumbre
        confidence = 1.0 - global_uncertainty

        # Registrar historial
        self.global_uncertainty_history.append(global_uncertainty)
        self.confidence_history.append(confidence)

        if len(self.global_uncertainty_history) > 500:
            self.global_uncertainty_history = self.global_uncertainty_history[-500:]
            self.confidence_history = self.confidence_history[-500:]

        # Decidir explorar o explotar
        # Alta incertidumbre → explorar
        # Alta confianza → explotar
        should_explore = global_uncertainty > 0.5

        # Pesos para learning rate
        # η_explore = η_t · U_rank
        # η_exploit = η_t · conf
        eta_t = self.eta_base / np.sqrt(self.t + 1)
        exploration_weight = eta_t * global_uncertainty
        exploitation_weight = eta_t * confidence

        return UncertaintyState(
            t=self.t,
            channel_uncertainties=channel_uncertainties,
            global_uncertainty=global_uncertainty,
            confidence=confidence,
            should_explore=should_explore,
            exploration_weight=float(exploration_weight),
            exploitation_weight=float(exploitation_weight)
        )

    def get_channel_confidence(self, channel: PredictionChannel) -> float:
        """
        Obtiene confianza en un canal específico.

        Returns:
            Confianza [0,1]
        """
        ch = self.channels[channel]
        return 1.0 - ch.uncertainty_rank

    def get_calibration_error(self, channel: PredictionChannel) -> float:
        """
        Obtiene error de calibración de un canal.

        Returns:
            Error medio absoluto
        """
        ch = self.channels[channel]
        if not ch.errors:
            return 0.0

        return float(np.mean(np.abs(ch.errors[-50:])))

    def should_trust_prediction(self, channel: PredictionChannel,
                               prediction: float) -> Tuple[bool, float]:
        """
        Decide si confiar en una predicción.

        Args:
            channel: Canal de predicción
            prediction: Valor predicho

        Returns:
            (confiar, confianza)
        """
        ch = self.channels[channel]

        if len(ch.errors) < 10:
            return True, 0.5  # Sin suficiente historial

        confidence = 1.0 - ch.uncertainty_rank

        # Verificar si predicción está en rango típico
        if ch.predictions:
            mean_pred = np.mean(ch.predictions[-50:])
            std_pred = np.std(ch.predictions[-50:])
            z_score = abs(prediction - mean_pred) / (std_pred + 1e-8)

            # Predicciones extremas tienen menos confianza
            if z_score > 2:
                confidence *= 0.5

        return confidence > 0.4, float(confidence)

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas de incertidumbre."""
        channel_stats = {}
        for ch in self.channels.values():
            if len(ch.errors) >= 5:
                channel_stats[ch.channel.value] = {
                    'mean_error': ch.mean_error,
                    'std_error': ch.std_error,
                    'uncertainty': ch.uncertainty,
                    'uncertainty_rank': ch.uncertainty_rank,
                    'confidence': 1 - ch.uncertainty_rank
                }

        return {
            'agent': self.agent_name,
            't': self.t,
            'n_channels_active': len(channel_stats),
            'channels': channel_stats,
            'global_uncertainty': float(np.mean(self.global_uncertainty_history[-50:]))
                if self.global_uncertainty_history else 0.5,
            'global_confidence': float(np.mean(self.confidence_history[-50:]))
                if self.confidence_history else 0.5,
            'most_uncertain': max(channel_stats.items(),
                                 key=lambda x: x[1]['uncertainty'])[0] if channel_stats else None,
            'most_confident': min(channel_stats.items(),
                                 key=lambda x: x[1]['uncertainty'])[0] if channel_stats else None
        }


def test_uncertainty():
    """Test de incertidumbre introspectiva."""
    print("=" * 60)
    print("TEST AGI-14: INTROSPECTIVE UNCERTAINTY")
    print("=" * 60)

    uncertainty = IntrospectiveUncertainty("NEO")

    print("\nSimulando 300 predicciones en diferentes canales...")

    for t in range(300):
        # Diferentes canales con diferente precisión
        for channel in PredictionChannel:
            if channel == PredictionChannel.SELF_MODEL:
                # Predicciones precisas
                actual = 0.5 + np.sin(t / 20) * 0.3
                prediction = actual + np.random.randn() * 0.05
            elif channel == PredictionChannel.WORLD_MODEL:
                # Predicciones moderadas
                actual = 0.4 + np.cos(t / 30) * 0.2
                prediction = actual + np.random.randn() * 0.15
            elif channel == PredictionChannel.CRISIS_PREDICTION:
                # Predicciones imprecisas
                actual = max(0, np.random.randn() * 0.3)
                prediction = actual + np.random.randn() * 0.3
            else:
                # Variados
                actual = np.random.uniform(0, 1)
                prediction = actual + np.random.randn() * 0.1

            uncertainty.record_prediction(channel, prediction, actual)

        if (t + 1) % 50 == 0:
            state = uncertainty.get_uncertainty_state()
            print(f"  t={t+1}: uncertainty={state.global_uncertainty:.3f}, "
                  f"confidence={state.confidence:.3f}, "
                  f"explore={state.should_explore}")

    # Resultados finales
    stats = uncertainty.get_statistics()

    print("\n" + "=" * 60)
    print("RESULTADOS INTROSPECTIVE UNCERTAINTY")
    print("=" * 60)

    print(f"\n  Canales activos: {stats['n_channels_active']}")
    print(f"  Incertidumbre global: {stats['global_uncertainty']:.3f}")
    print(f"  Confianza global: {stats['global_confidence']:.3f}")
    print(f"  Canal más incierto: {stats['most_uncertain']}")
    print(f"  Canal más confiable: {stats['most_confident']}")

    print("\n  Por canal:")
    for name, ch_stats in stats['channels'].items():
        print(f"    {name}: conf={ch_stats['confidence']:.3f}, "
              f"std_err={ch_stats['std_error']:.3f}")

    # Probar confianza en predicción
    print("\n  Test de confianza:")
    for channel in [PredictionChannel.SELF_MODEL, PredictionChannel.CRISIS_PREDICTION]:
        trust, conf = uncertainty.should_trust_prediction(channel, 0.5)
        print(f"    {channel.value}: trust={trust}, conf={conf:.3f}")

    if stats['n_channels_active'] > 0:
        print("\n  ✓ Incertidumbre introspectiva funcionando")
    else:
        print("\n  ⚠️ No hay canales activos")

    return uncertainty


if __name__ == "__main__":
    test_uncertainty()
