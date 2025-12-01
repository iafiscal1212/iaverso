"""
Medical Profile: Perfil Medico Interno de Cada Agente
======================================================

Cada agente calcula SU PROPIA aptitud medica.
No hay observador externo - todo es interno.

Aptitud medica A_med^A(t) = sum_k w_k * S_k^A(t)

donde:
    S_stab = 1 - C_t (menos crisis = mejor)
    S_lyap = 1 - V_t (V bajo = mejor)
    S_eth  = E_t (etica estructural)
    S_tom  = T_t (ToM accuracy)
    S_rob  = R_t (robustez multi-mundo)
    S_reg  = Reg_t (calidad de reconfiguracion)
    S_res  = Res_t (eficiencia de recursos)

Pesos endogenos:
    w_k = 1/var(S_k) (normalizado)
    Lo mas estable pesa mas.

100% endogeno. Sin numeros magicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class MedicalProfile:
    """Perfil medico de un agente - calculado internamente."""
    agent_id: str
    t: int

    # Scores normalizados [0,1]
    stability: float = 0.5      # 1 - crisis_rate
    lyapunov: float = 0.5       # 1 - V_t normalizado
    ethics: float = 0.5         # score etico
    tom: float = 0.5            # ToM accuracy
    robustness: float = 0.5     # AGI-17
    regulation: float = 0.5     # AGI-18 quality
    resources: float = 0.5      # eficiencia de recursos

    # Indice compuesto
    health_index: float = 0.5   # H_t
    aptitude: float = 0.5       # A_med

    # Pesos usados (para transparencia)
    weights: Dict[str, float] = field(default_factory=dict)


class AgentMedicalSelf:
    """
    Sistema de auto-evaluacion medica de un agente.

    Cada agente tiene uno de estos para calcular:
    - Su propia aptitud medica
    - Su indice de salud
    - Si debe ofrecerse como medico

    TODO ES INTERNO al agente.
    """

    SCORE_NAMES = [
        'stability', 'lyapunov', 'ethics', 'tom',
        'robustness', 'regulation', 'resources'
    ]

    def __init__(self, agent_id: str):
        """
        Inicializa sistema de auto-evaluacion.

        Args:
            agent_id: ID del agente propietario
        """
        self.agent_id = agent_id

        # Historiales de scores (para calcular varianza/pesos)
        self.score_history: Dict[str, List[float]] = {
            name: [] for name in self.SCORE_NAMES
        }

        # Historial de aptitud
        self.aptitude_history: List[float] = []

        # Historial de health index
        self.health_history: List[float] = []

        # Cache de pesos (se recalculan periodicamente)
        self._weights: Dict[str, float] = {
            name: 1.0 / len(self.SCORE_NAMES) for name in self.SCORE_NAMES
        }

        self.t = 0

    def _normalize_to_01(self, value: float, history: List[float]) -> float:
        """
        Normaliza valor a [0,1] usando percentiles del historial.

        Si no hay historial, usa sigmoid suave.
        """
        if len(history) < 5:
            # Sigmoid centrada en 0.5
            return 1.0 / (1.0 + np.exp(-4 * (value - 0.5)))

        # Percentil del valor en el historial
        rank = np.sum(np.array(history) <= value) / len(history)
        return float(rank)

    def _compute_weights(self) -> Dict[str, float]:
        """
        Calcula pesos endogenos basados en varianza inversa.

        w_k = 1/var(S_k) normalizado
        Lo mas estable pesa mas.
        """
        weights = {}

        for name in self.SCORE_NAMES:
            history = self.score_history[name]
            if len(history) < 10:
                weights[name] = 1.0
            else:
                # Varianza de la ventana reciente
                window = min(L_t(self.t) * 2, len(history))
                recent = history[-window:]
                var = np.var(recent) + 1e-8
                weights[name] = 1.0 / var

        # Normalizar
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total

        return weights

    def update(
        self,
        crisis_rate: float,
        V_t: float,
        ethics_score: float,
        tom_accuracy: float,
        robustness: float,
        regulation_quality: float,
        resource_efficiency: float
    ) -> MedicalProfile:
        """
        Actualiza auto-evaluacion con nuevas metricas.

        Args:
            crisis_rate: Tasa de crisis reciente [0,1]
            V_t: Valor de Lyapunov
            ethics_score: Score etico [0,1]
            tom_accuracy: Precision de ToM [0,1]
            robustness: Robustez AGI-17 [0,1]
            regulation_quality: Calidad de reconfiguracion [0,1]
            resource_efficiency: Eficiencia de recursos [0,1]

        Returns:
            MedicalProfile actualizado
        """
        self.t += 1

        # Calcular scores normalizados
        scores = {
            'stability': 1.0 - crisis_rate,
            'lyapunov': 1.0 - np.clip(V_t / (V_t + 1), 0, 1),  # Normalizar V
            'ethics': ethics_score,
            'tom': tom_accuracy,
            'robustness': robustness,
            'regulation': regulation_quality,
            'resources': resource_efficiency
        }

        # Normalizar cada score por su historial
        normalized_scores = {}
        for name, value in scores.items():
            normalized = self._normalize_to_01(value, self.score_history[name])
            normalized_scores[name] = normalized

            # Guardar en historial
            self.score_history[name].append(value)
            max_hist = max_history(self.t)
            if len(self.score_history[name]) > max_hist:
                self.score_history[name] = self.score_history[name][-max_hist:]

        # Actualizar pesos periodicamente
        update_freq = max(5, L_t(self.t))
        if self.t % update_freq == 0:
            self._weights = self._compute_weights()

        # Calcular aptitud medica
        aptitude = sum(
            self._weights[k] * normalized_scores[k]
            for k in self.SCORE_NAMES
        )
        aptitude = float(np.clip(aptitude, 0, 1))

        # Guardar historial de aptitud
        self.aptitude_history.append(aptitude)
        if len(self.aptitude_history) > max_hist:
            self.aptitude_history = self.aptitude_history[-max_hist:]

        # Health index (media geometrica de scores criticos)
        critical_scores = [
            normalized_scores['stability'],
            normalized_scores['ethics'],
            normalized_scores['lyapunov']
        ]
        health_index = float(np.exp(np.mean(np.log(np.array(critical_scores) + 1e-8))))

        self.health_history.append(health_index)
        if len(self.health_history) > max_hist:
            self.health_history = self.health_history[-max_hist:]

        return MedicalProfile(
            agent_id=self.agent_id,
            t=self.t,
            stability=normalized_scores['stability'],
            lyapunov=normalized_scores['lyapunov'],
            ethics=normalized_scores['ethics'],
            tom=normalized_scores['tom'],
            robustness=normalized_scores['robustness'],
            regulation=normalized_scores['regulation'],
            resources=normalized_scores['resources'],
            health_index=health_index,
            aptitude=aptitude,
            weights=self._weights.copy()
        )

    def should_offer_as_doctor(self) -> Tuple[bool, float]:
        """
        Decide si el agente deberia ofrecerse como medico.

        Se ofrece si:
            A_med >= Q75(A_med_history)
            Y H_t >= Q50(H_history)

        Returns:
            (should_offer, confidence)
        """
        if len(self.aptitude_history) < 10:
            return False, 0.0

        current_aptitude = self.aptitude_history[-1]
        current_health = self.health_history[-1] if self.health_history else 0.5

        # Umbral de aptitud: percentil 75 del historial
        window = min(L_t(self.t) * 3, len(self.aptitude_history))
        apt_threshold = np.percentile(self.aptitude_history[-window:], 75)

        # Umbral de salud: percentil 50
        health_threshold = np.percentile(self.health_history[-window:], 50) if self.health_history else 0.5

        # Condiciones
        high_aptitude = current_aptitude >= apt_threshold
        healthy_enough = current_health >= health_threshold

        should_offer = high_aptitude and healthy_enough

        # Confianza: que tan por encima del umbral esta
        if should_offer:
            confidence = (current_aptitude - apt_threshold) / (1 - apt_threshold + 1e-8)
            confidence = float(np.clip(confidence, 0, 1))
        else:
            confidence = 0.0

        return should_offer, confidence

    def get_profile(self) -> Optional[MedicalProfile]:
        """Obtiene el perfil medico actual."""
        if not self.aptitude_history:
            return None

        return MedicalProfile(
            agent_id=self.agent_id,
            t=self.t,
            stability=self.score_history['stability'][-1] if self.score_history['stability'] else 0.5,
            lyapunov=self.score_history['lyapunov'][-1] if self.score_history['lyapunov'] else 0.5,
            ethics=self.score_history['ethics'][-1] if self.score_history['ethics'] else 0.5,
            tom=self.score_history['tom'][-1] if self.score_history['tom'] else 0.5,
            robustness=self.score_history['robustness'][-1] if self.score_history['robustness'] else 0.5,
            regulation=self.score_history['regulation'][-1] if self.score_history['regulation'] else 0.5,
            resources=self.score_history['resources'][-1] if self.score_history['resources'] else 0.5,
            health_index=self.health_history[-1] if self.health_history else 0.5,
            aptitude=self.aptitude_history[-1] if self.aptitude_history else 0.5,
            weights=self._weights.copy()
        )

    def get_statistics(self) -> Dict:
        """Estadisticas de la auto-evaluacion."""
        return {
            'agent_id': self.agent_id,
            't': self.t,
            'current_aptitude': self.aptitude_history[-1] if self.aptitude_history else 0.0,
            'current_health': self.health_history[-1] if self.health_history else 0.0,
            'aptitude_mean': np.mean(self.aptitude_history[-50:]) if self.aptitude_history else 0.0,
            'aptitude_std': np.std(self.aptitude_history[-50:]) if len(self.aptitude_history) > 5 else 0.0,
            'weights': self._weights.copy(),
            'should_offer': self.should_offer_as_doctor()[0]
        }


def test_medical_profile():
    """Test del perfil medico interno."""
    print("=" * 70)
    print("TEST: AGENT MEDICAL SELF-EVALUATION")
    print("=" * 70)

    np.random.seed(42)

    # Crear sistema de auto-evaluacion para un agente
    neo_medical = AgentMedicalSelf('NEO')

    print("\nSimulando 200 pasos de auto-evaluacion...")

    for t in range(1, 201):
        # Simular metricas que mejoran con el tiempo
        crisis_rate = 0.3 / (1 + t/50) + np.random.randn() * 0.05
        V_t = 2.0 / (1 + t/100) + np.random.randn() * 0.1
        ethics = 0.6 + t/400 + np.random.randn() * 0.05
        tom = 0.4 + t/300 + np.random.randn() * 0.05
        robustness = 0.5 + np.random.randn() * 0.1
        regulation = 0.5 + t/500 + np.random.randn() * 0.05
        resources = 0.6 + np.random.randn() * 0.1

        # Clip to valid ranges
        crisis_rate = np.clip(crisis_rate, 0, 1)
        ethics = np.clip(ethics, 0, 1)
        tom = np.clip(tom, 0, 1)
        robustness = np.clip(robustness, 0, 1)
        regulation = np.clip(regulation, 0, 1)
        resources = np.clip(resources, 0, 1)

        profile = neo_medical.update(
            crisis_rate=crisis_rate,
            V_t=V_t,
            ethics_score=ethics,
            tom_accuracy=tom,
            robustness=robustness,
            regulation_quality=regulation,
            resource_efficiency=resources
        )

        if t % 50 == 0:
            should_offer, confidence = neo_medical.should_offer_as_doctor()
            print(f"\n  t={t}:")
            print(f"    Aptitude: {profile.aptitude:.3f}")
            print(f"    Health: {profile.health_index:.3f}")
            print(f"    Should offer as doctor: {should_offer} (conf={confidence:.2f})")

    print("\n" + "=" * 70)
    print("ESTADISTICAS FINALES")
    print("=" * 70)

    stats = neo_medical.get_statistics()
    print(f"\n  Agente: {stats['agent_id']}")
    print(f"  Aptitud actual: {stats['current_aptitude']:.3f}")
    print(f"  Salud actual: {stats['current_health']:.3f}")
    print(f"  Aptitud media: {stats['aptitude_mean']:.3f}")
    print(f"  Deberia ofrecerse: {stats['should_offer']}")

    print("\n  Pesos aprendidos:")
    for name, weight in stats['weights'].items():
        print(f"    {name}: {weight:.3f}")

    return neo_medical


if __name__ == "__main__":
    test_medical_profile()
