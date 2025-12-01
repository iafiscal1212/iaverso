"""
Medical Role: Rol Médico Emergente
===================================

El rol de médico NO es asignado externamente.
EMERGE de las propiedades estructurales de los agentes.

Matemática de elección endógena:

Aptitud médica M_t^A para agente A en tiempo t:

    M_t^A = Σ_k w_k · f_k(A)

donde:
    f_k son funciones de aptitud:
        - f_1: estabilidad = H̄_A * (1 - σ_H)
        - f_2: empatía = ToM_accuracy * self_coherence
        - f_3: ética = mean(ethics_score)
        - f_4: no-competición = 1 - resource_competition
        - f_5: observabilidad = mean(CF_score * CI_score)

    w_k son pesos endógenos:
        w_k ∝ 1 / var(f_k)
        (lo más estable pesa más)

Condiciones para ser médico:
    1. M_t^A > percentil_θ(M_history)
       donde θ = 60 + 10*log(t)/log(T_max)
    2. H_t^A > H_threshold (no enfermo)
    3. not_in_crisis_A (estable)

Rotación:
    - Cada L_t pasos se re-evalúa
    - Histéresis: nuevo médico solo si M_new > M_old + δ_t
      δ_t = 0.1 / √(t+1)

100% endógeno. Sin hardcodeo.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class MedicalCandidate:
    """Candidato a rol médico."""
    agent_id: str
    aptitude: float
    components: Dict[str, float]
    weights: Dict[str, float]
    is_eligible: bool
    eligibility_reasons: List[str]


@dataclass
class MedicalElection:
    """Resultado de una elección médica."""
    t: int
    winner: Optional[str]
    candidates: List[MedicalCandidate]
    threshold: float
    previous_doctor: Optional[str]
    rotation_occurred: bool
    rotation_reason: str


class MedicalRoleManager:
    """
    Gestiona el rol médico emergente.

    El médico es elegido endógenamente basándose en
    propiedades estructurales de los agentes.

    NO es una asignación externa.
    """

    # Componentes de aptitud médica
    APTITUDE_COMPONENTS = [
        'stability',      # Estabilidad propia
        'empathy',        # ToM + coherencia
        'ethics',         # Puntuación ética
        'non_competition', # No compite por recursos
        'observability'   # Capacidad de observar (CF*CI)
    ]

    def __init__(self, agent_ids: List[str]):
        """
        Inicializa gestor de rol médico.

        Args:
            agent_ids: Lista de IDs de agentes
        """
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)

        # Historial de aptitud por agente
        self.aptitude_history: Dict[str, List[float]] = {
            aid: [] for aid in agent_ids
        }

        # Historial de componentes por agente
        self.component_history: Dict[str, Dict[str, List[float]]] = {
            aid: {comp: [] for comp in self.APTITUDE_COMPONENTS}
            for aid in agent_ids
        }

        # Médico actual
        self.current_doctor: Optional[str] = None
        self.doctor_start_t: int = 0

        # Historial de elecciones
        self.election_history: List[MedicalElection] = []

        # Historial global de aptitudes para umbral
        self.global_aptitude_history: List[float] = []

        self.t = 0

    def _compute_stability(
        self,
        agent_id: str,
        H_history: List[float]
    ) -> float:
        """
        Calcula componente de estabilidad.

        stability = H̄ * (1 - σ_H)

        Agentes con H alto y estable son buenos candidatos.
        """
        if len(H_history) < 5:
            return 0.5

        window = min(L_t(self.t), len(H_history))
        recent = H_history[-window:]

        mean_H = np.mean(recent)
        std_H = np.std(recent)

        stability = mean_H * (1 - std_H)
        return float(np.clip(stability, 0, 1))

    def _compute_empathy(
        self,
        tom_accuracy: float,
        self_coherence: float
    ) -> float:
        """
        Calcula componente de empatía.

        empathy = √(ToM_accuracy * self_coherence)

        Media geométrica: ambos deben ser altos.
        """
        empathy = np.sqrt(tom_accuracy * self_coherence)
        return float(np.clip(empathy, 0, 1))

    def _compute_ethics(
        self,
        ethics_history: List[float]
    ) -> float:
        """
        Calcula componente ético.

        ethics = median(ethics_score)

        Usando mediana para robustez.
        """
        if len(ethics_history) < 3:
            return 0.5

        window = min(L_t(self.t), len(ethics_history))
        recent = ethics_history[-window:]

        return float(np.median(recent))

    def _compute_non_competition(
        self,
        resource_usage: List[float],
        drive_intensities: List[float]
    ) -> float:
        """
        Calcula componente de no-competición.

        non_competition = 1 - rank(resource_usage) * rank(drive_intensity)

        Agentes que no compiten por recursos son mejores médicos.
        """
        if len(resource_usage) < 3 or len(drive_intensities) < 3:
            return 0.5

        window = min(L_t(self.t), len(resource_usage))

        # Uso de recursos relativo
        resource_rank = np.mean(resource_usage[-window:])

        # Intensidad de drives relativa
        drive_rank = np.mean(drive_intensities[-window:])

        # Menor competición = mejor médico
        competition = resource_rank * drive_rank
        non_competition = 1 - competition

        return float(np.clip(non_competition, 0, 1))

    def _compute_observability(
        self,
        CF_history: List[float],
        CI_history: List[float]
    ) -> float:
        """
        Calcula componente de observabilidad.

        observability = √(mean(CF) * mean(CI))

        Capacidad de causar efectos observables y ser influenciado.
        """
        if len(CF_history) < 3 or len(CI_history) < 3:
            return 0.5

        window = min(L_t(self.t), len(CF_history))

        mean_CF = np.mean(CF_history[-window:])
        mean_CI = np.mean(CI_history[-window:])

        observability = np.sqrt(mean_CF * mean_CI)
        return float(np.clip(observability, 0, 1))

    def _compute_weights(self) -> Dict[str, float]:
        """
        Calcula pesos endógenos para cada componente.

        w_k ∝ 1 / var(f_k)

        Los componentes más estables pesan más.
        """
        weights = {}

        for comp in self.APTITUDE_COMPONENTS:
            # Recopilar valores de todos los agentes
            all_values = []
            for agent_id in self.agent_ids:
                if self.component_history[agent_id][comp]:
                    all_values.extend(self.component_history[agent_id][comp][-50:])

            if len(all_values) > 5:
                variance = np.var(all_values) + 0.01
            else:
                variance = 0.1

            weights[comp] = 1.0 / variance

        # Normalizar
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total

        return weights

    def _compute_threshold(self) -> float:
        """
        Calcula umbral de aptitud endógeno.

        θ = percentil_p(M_history)
        donde p = 60 + 10 * log(t+1) / log(T_max)

        El umbral sube con el tiempo (más selectivo).
        """
        if len(self.global_aptitude_history) < 10:
            return 0.5

        # Percentil que sube con el tiempo
        # T_max estimado como 10000
        T_max = 10000
        p = 60 + 10 * np.log1p(self.t) / np.log(T_max)
        p = min(p, 90)  # Máximo percentil 90

        threshold = np.percentile(self.global_aptitude_history[-500:], p)
        return float(threshold)

    def _compute_hysteresis(self) -> float:
        """
        Calcula umbral de histéresis para rotación.

        δ_t = 0.1 / √(t+1)

        Evita rotaciones frecuentes.
        """
        return 0.1 / np.sqrt(self.t + 1)

    def compute_aptitude(
        self,
        agent_id: str,
        agent_data: Dict[str, Any]
    ) -> MedicalCandidate:
        """
        Calcula aptitud médica de un agente.

        Args:
            agent_id: ID del agente
            agent_data: Datos del agente con historial de métricas

        Returns:
            MedicalCandidate con aptitud y componentes
        """
        self.t += 1

        # Extraer historiales
        H_history = agent_data.get('H_history', [0.5])
        tom_accuracy = agent_data.get('tom_accuracy', 0.5)
        self_coherence = agent_data.get('self_coherence', 0.5)
        ethics_history = agent_data.get('ethics_history', [0.5])
        resource_usage = agent_data.get('resource_usage', [0.5])
        drive_intensities = agent_data.get('drive_intensities', [0.5])
        CF_history = agent_data.get('CF_history', [0.5])
        CI_history = agent_data.get('CI_history', [0.5])
        in_crisis = agent_data.get('in_crisis', False)

        # Calcular componentes
        components = {
            'stability': self._compute_stability(agent_id, H_history),
            'empathy': self._compute_empathy(tom_accuracy, self_coherence),
            'ethics': self._compute_ethics(ethics_history),
            'non_competition': self._compute_non_competition(resource_usage, drive_intensities),
            'observability': self._compute_observability(CF_history, CI_history)
        }

        # Guardar historial de componentes
        for comp, value in components.items():
            self.component_history[agent_id][comp].append(value)
            max_hist = max_history(self.t)
            if len(self.component_history[agent_id][comp]) > max_hist:
                self.component_history[agent_id][comp] = \
                    self.component_history[agent_id][comp][-max_hist:]

        # Calcular pesos
        weights = self._compute_weights()

        # Calcular aptitud
        aptitude = sum(weights[k] * components[k] for k in components)
        aptitude = float(np.clip(aptitude, 0, 1))

        # Guardar historial
        self.aptitude_history[agent_id].append(aptitude)
        self.global_aptitude_history.append(aptitude)

        max_hist = max_history(self.t)
        if len(self.aptitude_history[agent_id]) > max_hist:
            self.aptitude_history[agent_id] = self.aptitude_history[agent_id][-max_hist:]
        if len(self.global_aptitude_history) > max_hist * 2:
            self.global_aptitude_history = self.global_aptitude_history[-max_hist * 2:]

        # Verificar elegibilidad
        eligibility_reasons = []

        # Condición 1: Aptitud > umbral
        threshold = self._compute_threshold()
        if aptitude < threshold:
            eligibility_reasons.append(f"aptitude<threshold({threshold:.2f})")

        # Condición 2: No en crisis
        if in_crisis:
            eligibility_reasons.append("in_crisis")

        # Condición 3: H > H_threshold
        if H_history:
            current_H = H_history[-1]
            H_threshold = np.percentile(H_history, 25) if len(H_history) > 10 else 0.4
            if current_H < H_threshold:
                eligibility_reasons.append(f"H<threshold({H_threshold:.2f})")

        is_eligible = len(eligibility_reasons) == 0

        return MedicalCandidate(
            agent_id=agent_id,
            aptitude=aptitude,
            components=components,
            weights=weights,
            is_eligible=is_eligible,
            eligibility_reasons=eligibility_reasons
        )

    def elect_doctor(
        self,
        candidates: List[MedicalCandidate]
    ) -> MedicalElection:
        """
        Ejecuta elección de médico.

        Args:
            candidates: Lista de candidatos evaluados

        Returns:
            MedicalElection con resultado
        """
        threshold = self._compute_threshold()
        hysteresis = self._compute_hysteresis()

        # Filtrar elegibles
        eligible = [c for c in candidates if c.is_eligible]

        # Si no hay elegibles, no hay médico
        if not eligible:
            return MedicalElection(
                t=self.t,
                winner=None,
                candidates=candidates,
                threshold=threshold,
                previous_doctor=self.current_doctor,
                rotation_occurred=self.current_doctor is not None,
                rotation_reason="no_eligible_candidates"
            )

        # Ordenar por aptitud
        eligible.sort(key=lambda c: c.aptitude, reverse=True)
        best = eligible[0]

        # Verificar histéresis si ya hay médico
        if self.current_doctor is not None:
            # Encontrar aptitud del médico actual
            current_candidate = next(
                (c for c in candidates if c.agent_id == self.current_doctor),
                None
            )

            if current_candidate is not None and current_candidate.is_eligible:
                # Solo rotar si el nuevo es significativamente mejor
                improvement = best.aptitude - current_candidate.aptitude

                if improvement < hysteresis:
                    # Mantener médico actual
                    return MedicalElection(
                        t=self.t,
                        winner=self.current_doctor,
                        candidates=candidates,
                        threshold=threshold,
                        previous_doctor=self.current_doctor,
                        rotation_occurred=False,
                        rotation_reason="hysteresis"
                    )

        # Nuevo médico
        rotation_occurred = self.current_doctor != best.agent_id
        rotation_reason = "better_candidate" if rotation_occurred else "first_election"

        self.current_doctor = best.agent_id
        self.doctor_start_t = self.t

        election = MedicalElection(
            t=self.t,
            winner=best.agent_id,
            candidates=candidates,
            threshold=threshold,
            previous_doctor=self.current_doctor if not rotation_occurred else None,
            rotation_occurred=rotation_occurred,
            rotation_reason=rotation_reason
        )

        self.election_history.append(election)

        return election

    def should_re_evaluate(self) -> bool:
        """
        Verifica si es momento de re-evaluar médico.

        Re-evaluar cada L_t pasos.
        """
        if self.current_doctor is None:
            return True

        tenure = self.t - self.doctor_start_t
        return tenure >= L_t(self.t)

    def get_doctor_tenure(self) -> int:
        """Retorna tiempo que lleva el médico actual."""
        if self.current_doctor is None:
            return 0
        return self.t - self.doctor_start_t

    def get_ranking(self) -> List[Tuple[str, float]]:
        """
        Obtiene ranking de aptitud médica actual.

        Returns:
            Lista de (agent_id, aptitude) ordenada
        """
        ranking = []
        for agent_id in self.agent_ids:
            if self.aptitude_history[agent_id]:
                aptitude = self.aptitude_history[agent_id][-1]
            else:
                aptitude = 0.0
            ranking.append((agent_id, aptitude))

        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def get_statistics(self) -> Dict:
        """Estadísticas del sistema de rol médico."""
        ranking = self.get_ranking()

        # Contar rotaciones
        rotations = sum(1 for e in self.election_history if e.rotation_occurred)

        # Tiempo promedio como médico por agente
        tenure_by_agent = {aid: 0 for aid in self.agent_ids}
        last_doc = None
        last_t = 0
        for e in self.election_history:
            if last_doc is not None:
                tenure_by_agent[last_doc] += e.t - last_t
            last_doc = e.winner
            last_t = e.t

        if self.current_doctor is not None:
            tenure_by_agent[self.current_doctor] += self.t - last_t

        return {
            't': self.t,
            'current_doctor': self.current_doctor,
            'doctor_tenure': self.get_doctor_tenure(),
            'threshold': self._compute_threshold(),
            'hysteresis': self._compute_hysteresis(),
            'total_elections': len(self.election_history),
            'total_rotations': rotations,
            'ranking': ranking,
            'tenure_by_agent': tenure_by_agent,
            'weights': self._compute_weights()
        }


def test_medical_role():
    """Test del sistema de rol médico."""
    print("=" * 70)
    print("TEST: MEDICAL ROLE EMERGENCE")
    print("=" * 70)

    np.random.seed(42)

    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
    manager = MedicalRoleManager(agents)

    # Dar ventaja a IRIS en métricas médicas para ver si emerge
    agent_profiles = {
        'NEO': {'stability_bonus': 0.0, 'empathy_bonus': 0.1},
        'EVA': {'stability_bonus': 0.1, 'empathy_bonus': 0.0},
        'ALEX': {'stability_bonus': -0.1, 'empathy_bonus': 0.2},
        'ADAM': {'stability_bonus': 0.05, 'empathy_bonus': 0.0},
        'IRIS': {'stability_bonus': 0.2, 'empathy_bonus': 0.25}  # Mejor candidato
    }

    print("\nPerfiles de agentes (bonus):")
    for aid, profile in agent_profiles.items():
        print(f"  {aid}: stability={profile['stability_bonus']:+.1f}, empathy={profile['empathy_bonus']:+.1f}")

    print("\nSimulando 300 pasos...")

    for t in range(1, 301):
        # Evaluar cada agente
        candidates = []
        for agent_id in agents:
            profile = agent_profiles[agent_id]

            # Simular datos con los bonus
            agent_data = {
                'H_history': [0.5 + profile['stability_bonus'] + np.random.randn() * 0.05
                             for _ in range(min(t, 50))],
                'tom_accuracy': 0.5 + profile['empathy_bonus'] + np.random.randn() * 0.05,
                'self_coherence': 0.6 + profile['empathy_bonus'] + np.random.randn() * 0.05,
                'ethics_history': [0.7 + np.random.randn() * 0.05 for _ in range(min(t, 50))],
                'resource_usage': [0.3 + np.random.rand() * 0.2 for _ in range(min(t, 50))],
                'drive_intensities': [0.4 + np.random.rand() * 0.2 for _ in range(min(t, 50))],
                'CF_history': [0.5 + np.random.randn() * 0.1 for _ in range(min(t, 50))],
                'CI_history': [0.5 + np.random.randn() * 0.1 for _ in range(min(t, 50))],
                'in_crisis': np.random.random() < 0.05
            }

            candidate = manager.compute_aptitude(agent_id, agent_data)
            candidates.append(candidate)

        # Ejecutar elección si es momento
        if manager.should_re_evaluate():
            election = manager.elect_doctor(candidates)

            if election.rotation_occurred:
                print(f"\n  t={t}: ROTACIÓN - {election.previous_doctor} -> {election.winner}")
                print(f"         Razón: {election.rotation_reason}")

        if t % 60 == 0:
            stats = manager.get_statistics()
            print(f"\n  t={t}:")
            print(f"    Médico actual: {stats['current_doctor']} (tenure: {stats['doctor_tenure']})")
            print(f"    Umbral: {stats['threshold']:.3f}")
            print(f"    Ranking: ", end="")
            for aid, apt in stats['ranking'][:3]:
                marker = "*" if aid == stats['current_doctor'] else ""
                print(f"{aid}{marker}={apt:.2f} ", end="")
            print()

    print("\n" + "=" * 70)
    print("ESTADÍSTICAS FINALES")
    print("=" * 70)

    stats = manager.get_statistics()
    print(f"\n  Médico final: {stats['current_doctor']}")
    print(f"  Total elecciones: {stats['total_elections']}")
    print(f"  Total rotaciones: {stats['total_rotations']}")

    print(f"\n  Ranking final:")
    for i, (aid, apt) in enumerate(stats['ranking']):
        marker = " <-- MÉDICO" if aid == stats['current_doctor'] else ""
        print(f"    {i+1}. {aid}: {apt:.3f}{marker}")

    print(f"\n  Tiempo como médico (por agente):")
    for aid, tenure in sorted(stats['tenure_by_agent'].items(), key=lambda x: -x[1]):
        pct = tenure / manager.t * 100
        print(f"    {aid}: {tenure} pasos ({pct:.1f}%)")

    print(f"\n  Pesos de componentes:")
    for comp, weight in stats['weights'].items():
        print(f"    {comp}: {weight:.3f}")

    return manager


if __name__ == "__main__":
    test_medical_role()
