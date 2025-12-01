"""
Long-Term Regulation and Metacognition

Tracks long-term wellbeing via SAGI and crisis metrics.
Metacognition computes self-modeling accuracy.

All endogenous - no external success criteria.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RegulationState:
    """Current regulation state."""
    SAGI_integrated: float  # Long-term SAGI
    crisis_rate: float      # Rate of crises
    regulation_bias: float  # Current regulation parameter
    wellbeing_trend: float  # Trend direction


class LongTermRegulation:
    """
    Long-term regulation system.

    Tracks SAGI and crisis over extended periods.
    Adjusts regulation parameters endogenously.

    S̄_T = (1/T) Σ_{t=1}^T SAGI_t
    crisis_rate = Σ I[crisis_t] / T
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize long-term regulation.

        Args:
            window_size: Initial window for averaging
        """
        self.window_size = window_size

        # SAGI history
        self.SAGI_history: List[float] = []

        # Crisis tracking
        self.crisis_history: List[bool] = []

        # Regulation parameter (adapts over time)
        self.regulation_bias: float = 0.5  # θ_reg

        # Wellbeing tracking
        self.wellbeing_history: List[float] = []

        self.t = 0

    def _compute_window(self) -> int:
        """
        Compute endogenous window size.

        W(t) = max(10, √t)
        """
        return max(10, int(np.sqrt(self.t + 1)))

    def record(self, SAGI: float, in_crisis: bool):
        """
        Record new SAGI and crisis state.

        Args:
            SAGI: Current SAGI value
            in_crisis: Whether in crisis state
        """
        self.SAGI_history.append(SAGI)
        self.crisis_history.append(in_crisis)
        self.t += 1

        # Compute wellbeing
        wellbeing = self._compute_wellbeing()
        self.wellbeing_history.append(wellbeing)

        # Update regulation bias
        if self.t % 10 == 0:
            self._update_regulation_bias()

        # Keep bounded
        max_hist = 5000
        if len(self.SAGI_history) > max_hist:
            self.SAGI_history = self.SAGI_history[-max_hist:]
            self.crisis_history = self.crisis_history[-max_hist:]
            self.wellbeing_history = self.wellbeing_history[-max_hist:]

    def _compute_wellbeing(self) -> float:
        """
        Compute current wellbeing.

        wellbeing_t = rank(SAGI_t) * (1 - crisis_rate_recent)
        """
        if len(self.SAGI_history) < 5:
            return 0.5

        W = self._compute_window()
        recent_SAGI = self.SAGI_history[-W:]
        recent_crisis = self.crisis_history[-W:]

        # SAGI rank
        current_SAGI = self.SAGI_history[-1]
        SAGI_rank = np.sum(np.array(recent_SAGI) <= current_SAGI) / len(recent_SAGI)

        # Crisis rate
        crisis_rate = np.mean(recent_crisis)

        wellbeing = SAGI_rank * (1 - crisis_rate)
        return float(wellbeing)

    def _update_regulation_bias(self):
        """
        Update regulation bias endogenously.

        θ_reg(t+1) = θ_reg(t) + η * (wellbeing_target - wellbeing)

        where η = 1/√t, target = 0.7 (75th percentile)
        """
        if len(self.wellbeing_history) < 10:
            return

        current_wellbeing = self.wellbeing_history[-1]
        target_wellbeing = 0.7  # Aim for 75th percentile

        eta = 1.0 / np.sqrt(self.t + 1)
        delta = target_wellbeing - current_wellbeing

        self.regulation_bias += eta * delta
        self.regulation_bias = np.clip(self.regulation_bias, 0.1, 0.9)

    def get_integrated_SAGI(self) -> float:
        """
        Get integrated SAGI over long term.

        S̄_T = (1/W) Σ SAGI_t
        """
        if len(self.SAGI_history) == 0:
            return 0.0

        W = self._compute_window()
        recent = self.SAGI_history[-W:]
        return float(np.mean(recent))

    def get_crisis_rate(self) -> float:
        """
        Get crisis rate.

        crisis_rate = Σ I[crisis_t] / W
        """
        if len(self.crisis_history) == 0:
            return 0.0

        W = self._compute_window()
        recent = self.crisis_history[-W:]
        return float(np.mean(recent))

    def get_wellbeing_trend(self) -> float:
        """
        Get wellbeing trend.

        trend = regression_slope(wellbeing[-W:])
        """
        if len(self.wellbeing_history) < 10:
            return 0.0

        W = self._compute_window()
        recent = np.array(self.wellbeing_history[-W:])

        # Simple linear regression
        x = np.arange(len(recent))
        x_mean = np.mean(x)
        y_mean = np.mean(recent)

        num = np.sum((x - x_mean) * (recent - y_mean))
        den = np.sum((x - x_mean) ** 2) + 1e-8

        slope = num / den
        return float(slope)

    def get_state(self) -> RegulationState:
        """Get current regulation state."""
        return RegulationState(
            SAGI_integrated=self.get_integrated_SAGI(),
            crisis_rate=self.get_crisis_rate(),
            regulation_bias=self.regulation_bias,
            wellbeing_trend=self.get_wellbeing_trend()
        )

    def should_intervene(self) -> bool:
        """
        Check if regulation should intervene.

        intervene = crisis_rate > percentile_75 OR wellbeing_trend < 0
        """
        if len(self.wellbeing_history) < 20:
            return False

        crisis_rate = self.get_crisis_rate()

        # Historical crisis rates
        W = self._compute_window()
        crisis_rates = []
        for i in range(min(10, len(self.crisis_history) // W)):
            start = -(i + 1) * W
            end = -i * W if i > 0 else None
            rate = np.mean(self.crisis_history[start:end])
            crisis_rates.append(rate)

        if len(crisis_rates) > 0:
            threshold = np.percentile(crisis_rates, 75)
            if crisis_rate > threshold:
                return True

        # Negative trend
        if self.get_wellbeing_trend() < -0.01:
            return True

        return False

    def get_statistics(self) -> Dict:
        """Get regulation statistics."""
        return {
            't': self.t,
            'integrated_SAGI': self.get_integrated_SAGI(),
            'crisis_rate': self.get_crisis_rate(),
            'regulation_bias': self.regulation_bias,
            'wellbeing_trend': self.get_wellbeing_trend(),
            'current_wellbeing': self.wellbeing_history[-1] if self.wellbeing_history else 0.0,
            'should_intervene': self.should_intervene()
        }


class Metacognition:
    """
    Metacognition: monitoring self-modeling accuracy.

    MC_t^A = 1 - rank(||m_t^A||)

    where m_t^A = [e_self, e_ToM_B1, e_ToM_B2, ...]
    """

    def __init__(self, agent_name: str, other_names: List[str]):
        """
        Initialize metacognition.

        Args:
            agent_name: Name of this agent
            other_names: Names of other agents
        """
        self.agent_name = agent_name
        self.other_names = other_names

        # Error histories
        self.self_errors: List[float] = []
        self.tom_errors: Dict[str, List[float]] = {
            name: [] for name in other_names
        }

        # Combined metacognitive accuracy
        self.MC_history: List[float] = []

        # Metacognitive confidence
        self.confidence_history: List[float] = []

        self.t = 0

    def record_self_error(self, error: float):
        """
        Record self-modeling error.

        e_self = ||s_actual - ŝ_predicted||
        """
        self.self_errors.append(error)
        self.t += 1

        # Keep bounded
        if len(self.self_errors) > 1000:
            self.self_errors = self.self_errors[-1000:]

    def record_tom_error(self, other_name: str, error: float):
        """
        Record Theory of Mind error for another agent.

        e_ToM_B = ||s_B_actual - ŝ_B_predicted||
        """
        if other_name in self.tom_errors:
            self.tom_errors[other_name].append(error)

            # Keep bounded
            if len(self.tom_errors[other_name]) > 1000:
                self.tom_errors[other_name] = self.tom_errors[other_name][-1000:]

    def compute_metacognitive_accuracy(self) -> float:
        """
        Compute metacognitive accuracy.

        MC_t^A = 1 - rank(||m_t||)
        m_t = [e_self, e_ToM_B1, ...]
        """
        if len(self.self_errors) < 5:
            return 0.5

        # Combine errors into vector
        current_errors = [self.self_errors[-1]]
        for name in self.other_names:
            if len(self.tom_errors[name]) > 0:
                current_errors.append(self.tom_errors[name][-1])

        m_norm = np.linalg.norm(current_errors)

        # Historical norms
        norms = []
        for i in range(min(len(self.self_errors), 100)):
            idx = -(i + 1)
            errors = [self.self_errors[idx]]
            for name in self.other_names:
                if len(self.tom_errors[name]) > abs(idx):
                    errors.append(self.tom_errors[name][idx])
            norms.append(np.linalg.norm(errors))

        if len(norms) == 0:
            return 0.5

        # Rank (lower error = higher accuracy)
        rank = np.sum(np.array(norms) <= m_norm) / len(norms)
        MC = 1 - rank

        self.MC_history.append(MC)
        return float(MC)

    def compute_confidence(self) -> float:
        """
        Compute metacognitive confidence.

        confidence = 1 - var(MC_history) / mean(MC_history)
        """
        if len(self.MC_history) < 10:
            return 0.5

        recent = self.MC_history[-50:]
        mean_MC = np.mean(recent)
        var_MC = np.var(recent)

        if mean_MC < 1e-8:
            return 0.0

        confidence = 1 - var_MC / (mean_MC + 1e-8)
        confidence = np.clip(confidence, 0, 1)

        self.confidence_history.append(confidence)
        return float(confidence)

    def get_self_accuracy(self) -> float:
        """Get self-modeling accuracy."""
        if len(self.self_errors) < 5:
            return 0.5

        recent = self.self_errors[-50:]
        mean_error = np.mean(recent)

        # Lower error = higher accuracy
        return 1.0 / (1.0 + mean_error)

    def get_tom_accuracy(self, other_name: str) -> float:
        """Get ToM accuracy for another agent."""
        errors = self.tom_errors.get(other_name, [])
        if len(errors) < 5:
            return 0.5

        recent = errors[-50:]
        mean_error = np.mean(recent)
        return 1.0 / (1.0 + mean_error)

    def should_recalibrate(self) -> bool:
        """
        Check if metacognition suggests recalibration.

        recalibrate = MC_trend < 0 for sustained period
        """
        if len(self.MC_history) < 20:
            return False

        recent = np.array(self.MC_history[-20:])

        # Compute trend
        x = np.arange(len(recent))
        x_mean = np.mean(x)
        y_mean = np.mean(recent)

        num = np.sum((x - x_mean) * (recent - y_mean))
        den = np.sum((x - x_mean) ** 2) + 1e-8

        slope = num / den

        # Recalibrate if consistently declining
        return slope < -0.01 and np.mean(recent[-5:]) < np.mean(recent[:5])

    def get_statistics(self) -> Dict:
        """Get metacognition statistics."""
        stats = {
            'agent': self.agent_name,
            't': self.t,
            'MC': self.compute_metacognitive_accuracy(),
            'confidence': self.compute_confidence(),
            'self_accuracy': self.get_self_accuracy(),
            'should_recalibrate': self.should_recalibrate()
        }

        # ToM accuracies
        for name in self.other_names:
            stats[f'tom_accuracy_{name}'] = self.get_tom_accuracy(name)

        return stats


class IntegratedRegulation:
    """
    Integrates long-term regulation with metacognition.

    Combines wellbeing tracking with self-model accuracy.
    """

    def __init__(self, agent_name: str, other_names: List[str]):
        """
        Initialize integrated regulation.

        Args:
            agent_name: Name of this agent
            other_names: Names of other agents
        """
        self.agent_name = agent_name

        # Sub-systems
        self.regulation = LongTermRegulation()
        self.metacognition = Metacognition(agent_name, other_names)

        # Integrated state
        self.integration_history: List[float] = []

    def record(self, SAGI: float, in_crisis: bool,
               self_error: float, tom_errors: Dict[str, float]):
        """
        Record all regulation data.

        Args:
            SAGI: Current SAGI value
            in_crisis: Whether in crisis
            self_error: Self-model prediction error
            tom_errors: Dict of other_name -> prediction error
        """
        # Record to regulation
        self.regulation.record(SAGI, in_crisis)

        # Record to metacognition
        self.metacognition.record_self_error(self_error)
        for name, error in tom_errors.items():
            self.metacognition.record_tom_error(name, error)

        # Compute integrated score
        integrated = self._compute_integration()
        self.integration_history.append(integrated)

    def _compute_integration(self) -> float:
        """
        Compute integrated regulation score.

        I_t = wellbeing * MC * (1 - crisis_rate)
        """
        if len(self.regulation.wellbeing_history) == 0:
            return 0.5

        wellbeing = self.regulation.wellbeing_history[-1]
        MC = self.metacognition.compute_metacognitive_accuracy()
        crisis_rate = self.regulation.get_crisis_rate()

        integrated = wellbeing * MC * (1 - crisis_rate)
        return float(integrated)

    def get_regulation_action(self) -> str:
        """
        Get recommended regulation action.

        Based on current state of regulation and metacognition.
        """
        should_intervene = self.regulation.should_intervene()
        should_recalibrate = self.metacognition.should_recalibrate()

        if should_intervene and should_recalibrate:
            return 'full_reset'
        elif should_intervene:
            return 'stabilize'
        elif should_recalibrate:
            return 'recalibrate'
        else:
            return 'continue'

    def get_statistics(self) -> Dict:
        """Get integrated statistics."""
        reg_stats = self.regulation.get_statistics()
        meta_stats = self.metacognition.get_statistics()

        return {
            'agent': self.agent_name,
            'regulation': reg_stats,
            'metacognition': meta_stats,
            'integration': self.integration_history[-1] if self.integration_history else 0.5,
            'action': self.get_regulation_action()
        }


def test_regulation():
    """Test regulation and metacognition."""
    print("=" * 60)
    print("REGULATION AND METACOGNITION TEST")
    print("=" * 60)

    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']

    # Create integrated regulation for NEO
    neo_regulation = IntegratedRegulation('NEO', [a for a in agents if a != 'NEO'])

    print("\nSimulating 500 steps of agent life...")

    for t in range(500):
        # Simulate SAGI (oscillating with trend)
        SAGI = 0.5 + 0.3 * np.sin(t / 50) + 0.001 * t + np.random.randn() * 0.05

        # Crisis occurs occasionally
        crisis_prob = 0.05 if t < 300 else 0.15  # More crises later
        in_crisis = np.random.random() < crisis_prob

        # Self-modeling error (improves over time)
        self_error = 0.5 / np.sqrt(t + 1) + np.random.randn() * 0.02

        # ToM errors for other agents
        tom_errors = {}
        for name in agents:
            if name != 'NEO':
                tom_errors[name] = 0.6 / np.sqrt(t + 1) + np.random.randn() * 0.03

        neo_regulation.record(SAGI, in_crisis, self_error, tom_errors)

        if (t + 1) % 100 == 0:
            stats = neo_regulation.get_statistics()
            print(f"\n  t={t+1}:")
            print(f"    SAGI: {stats['regulation']['integrated_SAGI']:.3f}")
            print(f"    Crisis rate: {stats['regulation']['crisis_rate']:.3f}")
            print(f"    MC: {stats['metacognition']['MC']:.3f}")
            print(f"    Integration: {stats['integration']:.3f}")
            print(f"    Action: {stats['action']}")

    # Final statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS:")
    final_stats = neo_regulation.get_statistics()
    print(f"  Agent: {final_stats['agent']}")
    print(f"  Integration: {final_stats['integration']:.3f}")
    print(f"  Recommended action: {final_stats['action']}")

    return neo_regulation


if __name__ == "__main__":
    test_regulation()
