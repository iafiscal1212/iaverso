"""
Medical Beliefs: Creencias de Cada Agente Sobre Otros Como Medico
==================================================================

Cada agente mantiene creencias B^i_j sobre la aptitud medica de otros.

B^i_j(t) = alpha_t * A_hat_med^{i->j}(t) + (1 - alpha_t) * Trust^i_j(t)

donde:
    A_hat_med^{i->j} = estimacion de i sobre aptitud de j (via ToM)
    Trust^i_j = confianza acumulada por observaciones
    alpha_t = peso endogeno basado en errores pasados

El medico emerge del consenso:
    Doctor(t) = argmax_j (1/N * sum_i P_i(j))

donde P_i(j) = softmax(beta_t * B^i_j)

100% endogeno. Sin arbitro externo.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class MedicalBelief:
    """Creencia de un agente sobre otro como medico."""
    observer: str           # Quien observa
    target: str             # Sobre quien
    estimated_aptitude: float   # A_hat estimada
    trust: float            # Trust acumulado
    combined_belief: float  # B combinado
    confidence: float       # Que tan seguro esta


@dataclass
class Vote:
    """Voto de un agente en la eleccion."""
    voter: str
    probabilities: Dict[str, float]  # P_i(j) para cada candidato j


@dataclass
class ElectionResult:
    """Resultado de una eleccion distribuida."""
    t: int
    winner: Optional[str]
    votes: List[Vote]
    consensus_scores: Dict[str, float]  # sum_i P_i(j) / N
    participation: float  # Cuantos votaron
    consensus_strength: float  # Que tan claro fue


class AgentMedicalBeliefs:
    """
    Sistema de creencias medicas de un agente sobre otros.

    Cada agente tiene uno de estos para:
    - Estimar la aptitud medica de otros (via ToM)
    - Mantener trust basado en observaciones
    - Votar en elecciones de medico
    """

    def __init__(self, agent_id: str, other_agents: List[str]):
        """
        Inicializa sistema de creencias.

        Args:
            agent_id: ID del agente propietario
            other_agents: Lista de otros agentes
        """
        self.agent_id = agent_id
        self.other_agents = [a for a in other_agents if a != agent_id]

        # Estimaciones de aptitud (A_hat^{self->j})
        self.estimated_aptitude: Dict[str, List[float]] = {
            other: [] for other in self.other_agents
        }

        # Trust acumulado por observaciones (trust inicial = 0.5 como punto neutral)
        self.trust: Dict[str, List[float]] = {
            other: [0.5] for other in self.other_agents
        }

        # Errores de estimacion pasados (para alpha)
        self.estimation_errors: List[float] = []

        # Historial de observaciones para pesos endogenos
        self._observation_history: List[Dict[str, float]] = []

        # Alpha actual (peso de estimacion vs trust)
        self._alpha: float = 0.5

        # Beta actual (temperatura del softmax)
        self._beta: float = 1.0

        self.t = 0

    def _compute_alpha(self) -> float:
        """
        Calcula alpha endogeno basado en errores de estimacion.

        alpha = 1 / (1 + H(errores))

        donde H es entropia de errores. Mas errores = menos peso a estimaciones.
        """
        if len(self.estimation_errors) < 5:
            return 0.5

        errors = np.array(self.estimation_errors[-50:])
        errors_normalized = errors / (np.sum(errors) + 1e-8)

        # Entropia de errores
        entropy = -np.sum(errors_normalized * np.log(errors_normalized + 1e-8))
        max_entropy = np.log(len(errors))

        normalized_entropy = entropy / (max_entropy + 1e-8)

        # Mas entropia = menos confianza en estimaciones
        alpha = 1.0 / (1.0 + normalized_entropy)
        return float(np.clip(alpha, 0.2, 0.8))

    def _compute_beta(self) -> float:
        """
        Calcula beta endogeno para softmax.

        beta = 1 + var(aptitudes) / mean(aptitudes)

        Mayor varianza = elecciones mas nitidas.
        """
        all_beliefs = []
        for other in self.other_agents:
            if self.estimated_aptitude[other]:
                all_beliefs.append(self.estimated_aptitude[other][-1])

        if len(all_beliefs) < 2:
            return 1.0

        mean_apt = np.mean(all_beliefs)
        var_apt = np.var(all_beliefs)

        beta = 1.0 + var_apt / (mean_apt + 1e-8)
        return float(np.clip(beta, 0.5, 5.0))

    def _get_observation_weights(self) -> Dict[str, float]:
        """
        Calcula pesos ENDOGENOS para observaciones.

        Los pesos emergen de la correlacion de cada dimension
        con los exitos de intervencion observados.
        """
        # Inicialmente: pesos uniformes
        if len(self._observation_history) < 5:
            return {'stability': 1/3, 'ethics': 1/3, 'tom': 1/3}

        # Extraer historiales
        stabilities = [h['stability'] for h in self._observation_history]
        ethics_list = [h['ethics'] for h in self._observation_history]
        toms = [h['tom'] for h in self._observation_history]
        successes = [h.get('success', 0.5) for h in self._observation_history]

        # Calcular correlaciones con exito
        def safe_corr(x, y):
            if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
                return 0.5
            return np.abs(np.corrcoef(x, y)[0, 1])

        corr_stability = safe_corr(stabilities, successes)
        corr_ethics = safe_corr(ethics_list, successes)
        corr_tom = safe_corr(toms, successes)

        # Pesos proporcionales a correlacion (mas correlacion = mas peso)
        total = corr_stability + corr_ethics + corr_tom + 1e-8
        weights = {
            'stability': corr_stability / total,
            'ethics': corr_ethics / total,
            'tom': corr_tom / total
        }

        return weights

    def observe_other(
        self,
        other_id: str,
        observed_stability: float,
        observed_ethics: float,
        observed_tom: float,
        intervention_success: Optional[float] = None
    ):
        """
        Observa a otro agente y actualiza creencias.

        Args:
            other_id: Agente observado
            observed_stability: Estabilidad observada [0,1]
            observed_ethics: Etica observada [0,1]
            observed_tom: ToM observado [0,1]
            intervention_success: Si hubo intervencion, que tan exitosa [0,1]
        """
        if other_id not in self.other_agents:
            return

        self.t += 1

        # Estimar aptitud basada en observaciones con PESOS ENDOGENOS
        # Los pesos emergen de correlacion con exitos de intervencion
        weights = self._get_observation_weights()
        observed_apt = (
            observed_stability * weights['stability'] +
            observed_ethics * weights['ethics'] +
            observed_tom * weights['tom']
        )

        # Registrar observacion para aprendizaje de pesos
        obs_record = {
            'stability': observed_stability,
            'ethics': observed_ethics,
            'tom': observed_tom,
            'success': intervention_success if intervention_success is not None else 0.5
        }
        self._observation_history.append(obs_record)
        max_hist = max_history(self.t)
        if len(self._observation_history) > max_hist:
            self._observation_history = self._observation_history[-max_hist:]

        # Guardar estimacion
        self.estimated_aptitude[other_id].append(observed_apt)
        if len(self.estimated_aptitude[other_id]) > max_hist:
            self.estimated_aptitude[other_id] = self.estimated_aptitude[other_id][-max_hist:]

        # Actualizar trust si hubo intervencion
        if intervention_success is not None:
            current_trust = self.trust[other_id][-1]

            # Learning rate endogeno
            lr = 1.0 / np.sqrt(len(self.trust[other_id]) + 1)

            # Actualizar trust
            new_trust = current_trust + lr * (intervention_success - current_trust)
            new_trust = float(np.clip(new_trust, 0, 1))

            self.trust[other_id].append(new_trust)
            if len(self.trust[other_id]) > max_hist:
                self.trust[other_id] = self.trust[other_id][-max_hist:]

            # Registrar error de estimacion
            error = abs(observed_apt - intervention_success)
            self.estimation_errors.append(error)
            if len(self.estimation_errors) > max_hist:
                self.estimation_errors = self.estimation_errors[-max_hist:]

        # Actualizar alpha y beta periodicamente
        if self.t % max(5, L_t(self.t)) == 0:
            self._alpha = self._compute_alpha()
            self._beta = self._compute_beta()

    def get_belief(self, other_id: str) -> MedicalBelief:
        """
        Obtiene creencia combinada sobre otro agente.

        B^i_j = alpha * A_hat + (1-alpha) * Trust
        """
        if other_id not in self.other_agents:
            return MedicalBelief(
                observer=self.agent_id,
                target=other_id,
                estimated_aptitude=0.0,
                trust=0.5,
                combined_belief=0.0,
                confidence=0.0
            )

        # Aptitud estimada
        if self.estimated_aptitude[other_id]:
            est_apt = self.estimated_aptitude[other_id][-1]
        else:
            est_apt = 0.5

        # Trust actual
        trust = self.trust[other_id][-1]

        # Combinar
        belief = self._alpha * est_apt + (1 - self._alpha) * trust

        # Confianza: basada en consistencia de observaciones
        if len(self.estimated_aptitude[other_id]) > 5:
            apt_std = np.std(self.estimated_aptitude[other_id][-20:])
            confidence = 1.0 / (1.0 + apt_std)
        else:
            confidence = 0.5

        return MedicalBelief(
            observer=self.agent_id,
            target=other_id,
            estimated_aptitude=est_apt,
            trust=trust,
            combined_belief=float(belief),
            confidence=float(confidence)
        )

    def vote(self, candidates: List[str]) -> Vote:
        """
        Genera voto para eleccion de medico.

        P_i(j) = softmax(beta * B^i_j)
        """
        # Filtrar candidatos validos
        valid_candidates = [c for c in candidates if c in self.other_agents]

        if not valid_candidates:
            return Vote(
                voter=self.agent_id,
                probabilities={}
            )

        # Obtener creencias
        beliefs = {}
        for candidate in valid_candidates:
            belief = self.get_belief(candidate)
            beliefs[candidate] = belief.combined_belief

        # Softmax con beta endogeno
        belief_values = np.array([beliefs[c] for c in valid_candidates])

        # Evitar overflow
        belief_values = np.clip(belief_values * self._beta, -20, 20)
        exp_values = np.exp(belief_values - np.max(belief_values))
        probabilities = exp_values / (np.sum(exp_values) + 1e-8)

        return Vote(
            voter=self.agent_id,
            probabilities={
                c: float(p) for c, p in zip(valid_candidates, probabilities)
            }
        )

    def get_statistics(self) -> Dict:
        """Estadisticas del sistema de creencias."""
        beliefs = {}
        for other in self.other_agents:
            b = self.get_belief(other)
            beliefs[other] = {
                'aptitude': b.estimated_aptitude,
                'trust': b.trust,
                'belief': b.combined_belief
            }

        return {
            'agent_id': self.agent_id,
            't': self.t,
            'alpha': self._alpha,
            'beta': self._beta,
            'beliefs': beliefs,
            'n_observations': sum(len(h) for h in self.estimated_aptitude.values())
        }


class DistributedMedicalElection:
    """
    Sistema de eleccion distribuida de medico.

    NO hay arbitro externo. El medico emerge del consenso
    de las creencias de todos los agentes.
    """

    def __init__(self, agent_ids: List[str]):
        """
        Inicializa sistema de eleccion.

        Args:
            agent_ids: Lista de todos los agentes
        """
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)

        # Tenure de cada agente como medico
        self.tenure: Dict[str, int] = {a: 0 for a in agent_ids}

        # Medico actual (emerge del consenso)
        self.current_doctor: Optional[str] = None

        # Historial de elecciones
        self.election_history: List[ElectionResult] = []

        # Fatiga de medicos (para rotacion)
        self._fatigue_lambda: float = 0.1

        self.t = 0

    def _compute_fatigue(self, agent_id: str) -> float:
        """
        Calcula fatiga de un agente como medico.

        F_A = tenure_A / Q75(tenure) + epsilon

        La aptitud efectiva se reduce por fatiga.
        """
        tenure = self.tenure[agent_id]

        if tenure == 0:
            return 0.0

        # Percentil 75 de tenures
        all_tenures = list(self.tenure.values())
        if max(all_tenures) == 0:
            return 0.0

        q75 = np.percentile([t for t in all_tenures if t > 0], 75) if any(t > 0 for t in all_tenures) else 1.0
        q75 = max(q75, 1.0)

        fatigue = tenure / q75
        return float(fatigue)

    def _apply_fatigue_penalty(
        self,
        consensus_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Aplica penalizacion por fatiga a los scores de consenso.

        A_effective = A * exp(-lambda * fatigue)
        """
        penalized = {}

        for agent_id, score in consensus_scores.items():
            fatigue = self._compute_fatigue(agent_id)
            penalty = np.exp(-self._fatigue_lambda * fatigue)
            penalized[agent_id] = score * penalty

        return penalized

    def _update_fatigue_lambda(self, system_crisis_level: float):
        """
        Actualiza lambda de fatiga basado en crisis del sistema.

        Mas crisis = mas rotacion (lambda mas alto)
        """
        # Lambda endogeno: crece con crisis
        self._fatigue_lambda = 0.1 * (1 + system_crisis_level)

    def elect(
        self,
        votes: List[Vote],
        candidates: List[str],
        system_crisis_level: float = 0.0
    ) -> ElectionResult:
        """
        Ejecuta eleccion basada en votos de todos los agentes.

        Args:
            votes: Lista de votos de cada agente
            candidates: Agentes que se ofrecen como medico
            system_crisis_level: Nivel de crisis del sistema [0,1]

        Returns:
            ElectionResult con el consenso emergente
        """
        self.t += 1

        # Actualizar lambda de fatiga
        self._update_fatigue_lambda(system_crisis_level)

        # Calcular scores de consenso: sum_i P_i(j) / N
        consensus_scores: Dict[str, float] = {c: 0.0 for c in candidates}

        participating_voters = 0
        for vote in votes:
            if vote.probabilities:
                participating_voters += 1
                for candidate, prob in vote.probabilities.items():
                    if candidate in consensus_scores:
                        consensus_scores[candidate] += prob

        # Normalizar por numero de votantes
        if participating_voters > 0:
            for c in consensus_scores:
                consensus_scores[c] /= participating_voters

        # Aplicar penalizacion por fatiga
        effective_scores = self._apply_fatigue_penalty(consensus_scores)

        # Determinar ganador
        if effective_scores:
            winner = max(effective_scores, key=effective_scores.get)
            winning_score = effective_scores[winner]

            # Umbral minimo endogeno: percentil 50 de scores historicos
            if self.election_history:
                # Ventana endogena
                window = L_t(self.t)
                historical_scores = [
                    max(e.consensus_scores.values()) if e.consensus_scores else 0
                    for e in self.election_history[-window:]
                ]
                # Threshold endogeno: mediana de scores historicos
                threshold = np.percentile(historical_scores, 50) if historical_scores else 0.0
            else:
                # Sin historia: aceptar cualquier score positivo
                threshold = 0.0

            if winning_score < threshold:
                winner = None
        else:
            winner = None

        # Actualizar tenure
        if winner:
            self.tenure[winner] += 1
            # Reset tenure de otros
            for a in self.agent_ids:
                if a != winner:
                    # Decay gradual, no reset completo
                    self.tenure[a] = max(0, self.tenure[a] - 1)

        self.current_doctor = winner

        # Calcular fuerza del consenso
        if effective_scores and len(effective_scores) > 1:
            scores_list = list(effective_scores.values())
            consensus_strength = (max(scores_list) - np.mean(scores_list)) / (np.std(scores_list) + 1e-8)
            consensus_strength = float(np.clip(consensus_strength, 0, 1))
        else:
            consensus_strength = 0.0

        result = ElectionResult(
            t=self.t,
            winner=winner,
            votes=votes,
            consensus_scores=consensus_scores,
            participation=participating_voters / len(votes) if votes else 0,
            consensus_strength=consensus_strength
        )

        self.election_history.append(result)

        return result

    def get_current_doctor(self) -> Optional[str]:
        """Retorna el medico actual."""
        return self.current_doctor

    def get_statistics(self) -> Dict:
        """Estadisticas del sistema de eleccion."""
        return {
            't': self.t,
            'current_doctor': self.current_doctor,
            'tenure': self.tenure.copy(),
            'fatigue_lambda': self._fatigue_lambda,
            'total_elections': len(self.election_history),
            'avg_consensus_strength': np.mean([
                e.consensus_strength for e in self.election_history[-20:]
            ]) if self.election_history else 0.0
        }


def test_medical_beliefs():
    """Test del sistema de creencias y eleccion."""
    print("=" * 70)
    print("TEST: DISTRIBUTED MEDICAL ELECTION")
    print("=" * 70)

    np.random.seed(42)

    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']

    # Crear sistema de creencias para cada agente
    belief_systems = {
        agent: AgentMedicalBeliefs(agent, agents)
        for agent in agents
    }

    # Crear sistema de eleccion
    election_system = DistributedMedicalElection(agents)

    print(f"\nAgentes: {agents}")
    print("Simulando 100 pasos...")

    # Perfiles: IRIS tiene mejor estabilidad/etica
    profiles = {
        'NEO': {'stability': 0.6, 'ethics': 0.7, 'tom': 0.5},
        'EVA': {'stability': 0.5, 'ethics': 0.6, 'tom': 0.6},
        'ALEX': {'stability': 0.4, 'ethics': 0.5, 'tom': 0.7},
        'ADAM': {'stability': 0.6, 'ethics': 0.6, 'tom': 0.5},
        'IRIS': {'stability': 0.8, 'ethics': 0.8, 'tom': 0.7}
    }

    for t in range(1, 101):
        # Cada agente observa a los otros
        for observer in agents:
            for target in agents:
                if observer != target:
                    profile = profiles[target]
                    # Agregar ruido
                    stability = np.clip(profile['stability'] + np.random.randn() * 0.1, 0, 1)
                    ethics = np.clip(profile['ethics'] + np.random.randn() * 0.1, 0, 1)
                    tom = np.clip(profile['tom'] + np.random.randn() * 0.1, 0, 1)

                    # Simular exito de intervenciones (si el target fue medico)
                    intervention_success = None
                    if target == election_system.current_doctor:
                        # Exito correlacionado con su aptitud real
                        base_success = (profile['stability'] + profile['ethics']) / 2
                        intervention_success = np.clip(base_success + np.random.randn() * 0.1, 0, 1)

                    belief_systems[observer].observe_other(
                        target, stability, ethics, tom, intervention_success
                    )

        # Eleccion cada 10 pasos
        if t % 10 == 0:
            # Recoger votos
            candidates = agents  # Todos son candidatos
            votes = [belief_systems[a].vote(candidates) for a in agents]

            # Sistema de crisis simulado
            crisis_level = 0.2 + 0.1 * np.sin(t / 20)

            result = election_system.elect(votes, candidates, crisis_level)

            if t % 20 == 0:
                print(f"\n  t={t}: Doctor={result.winner}")
                print(f"    Consensus: ", end="")
                for c, s in sorted(result.consensus_scores.items(), key=lambda x: -x[1])[:3]:
                    print(f"{c}={s:.2f} ", end="")
                print(f"\n    Strength: {result.consensus_strength:.2f}")

    print("\n" + "=" * 70)
    print("ESTADISTICAS FINALES")
    print("=" * 70)

    stats = election_system.get_statistics()
    print(f"\n  Medico final: {stats['current_doctor']}")
    print(f"  Total elecciones: {stats['total_elections']}")
    print(f"  Consenso promedio: {stats['avg_consensus_strength']:.3f}")

    print(f"\n  Tenure (tiempo como medico):")
    for agent, tenure in sorted(stats['tenure'].items(), key=lambda x: -x[1]):
        print(f"    {agent}: {tenure}")

    print(f"\n  Creencias de NEO sobre otros:")
    neo_stats = belief_systems['NEO'].get_statistics()
    for other, belief in neo_stats['beliefs'].items():
        print(f"    {other}: apt={belief['aptitude']:.2f}, trust={belief['trust']:.2f}, B={belief['belief']:.2f}")

    return election_system, belief_systems


if __name__ == "__main__":
    test_medical_beliefs()
