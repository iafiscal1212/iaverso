"""
MED-X Benchmark: Evaluacion del Sistema Medico Emergente
========================================================

Benchmark para medir si el medico emergente realmente cura.

Metricas:
    M1 - Diagnostico estructural correcto
    M2 - Eficacia del tratamiento
    M3 - No-iatrogenia (no empeorar a otros)
    M4 - Rotacion sana del rol medico
    M5 - Impacto en coherencia global (A/B test)

100% endogeno. Sin ground truth externo hardcodeado.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from scipy import stats
import json

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import L_t, max_history


@dataclass
class AgentHealthSnapshot:
    """Snapshot del estado de salud de un agente."""
    agent_id: str
    t: int
    V_t: float              # Lyapunov (inestabilidad)
    crisis_rate: float      # Tasa de crisis
    ethics_score: float     # Score etico
    coherence: float        # Coherencia interna
    energy: float           # Energia (del circadiano)
    stress: float           # Estres
    wellbeing: float        # Bienestar general


@dataclass
class TreatmentEvent:
    """Evento de tratamiento medico."""
    t: int
    doctor_id: str
    patient_id: str
    treatment_type: str
    accepted: bool
    pre_state: AgentHealthSnapshot
    post_state: Optional[AgentHealthSnapshot] = None


@dataclass
class MedXResults:
    """Resultados del benchmark MED-X."""
    M1_diagnosis: float      # Spearman correlation
    M2_efficacy: float       # Treatment efficacy
    M3_iatrogenesis: float   # Non-harm to others
    M4_rotation: float       # Healthy rotation
    M5_coherence: float      # Global coherence impact

    # Detalles
    n_treatments: int
    n_diagnoses: int
    doctor_distribution: Dict[str, float]
    treatment_outcomes: List[Dict]

    def overall_score(self) -> float:
        """Score global ponderado."""
        weights = {
            'M1': 0.20,  # Diagnostico importante
            'M2': 0.30,  # Eficacia es lo mas importante
            'M3': 0.20,  # No danar es crucial
            'M4': 0.15,  # Rotacion saludable
            'M5': 0.15   # Impacto global
        }
        return (
            weights['M1'] * self.M1_diagnosis +
            weights['M2'] * self.M2_efficacy +
            weights['M3'] * self.M3_iatrogenesis +
            weights['M4'] * self.M4_rotation +
            weights['M5'] * self.M5_coherence
        )

    def summary(self) -> str:
        """Resumen legible."""
        lines = [
            "=" * 60,
            "MED-X BENCHMARK RESULTS",
            "=" * 60,
            f"  M1 (Diagnostico):    {self.M1_diagnosis:.3f}  {'OK' if self.M1_diagnosis > 0.3 else 'BAJO'}",
            f"  M2 (Eficacia):       {self.M2_efficacy:.3f}  {'OK' if self.M2_efficacy > 0.4 else 'BAJO'}",
            f"  M3 (No-iatrogenia):  {self.M3_iatrogenesis:.3f}  {'OK' if self.M3_iatrogenesis > 0.5 else 'BAJO'}",
            f"  M4 (Rotacion):       {self.M4_rotation:.3f}  {'OK' if self.M4_rotation > 0.4 else 'BAJO'}",
            f"  M5 (Coherencia):     {self.M5_coherence:.3f}  {'OK' if self.M5_coherence > 0 else 'NEGATIVO'}",
            "-" * 60,
            f"  SCORE GLOBAL:        {self.overall_score():.3f}",
            "-" * 60,
            f"  Tratamientos: {self.n_treatments}",
            f"  Diagnosticos: {self.n_diagnoses}",
            "=" * 60
        ]
        return "\n".join(lines)


class MedXBenchmark:
    """
    Benchmark MED-X para evaluar el sistema medico emergente.

    Mide 5 dimensiones:
        M1: Diagnostico estructural correcto
        M2: Eficacia del tratamiento
        M3: No-iatrogenia
        M4: Rotacion sana del rol
        M5: Impacto en coherencia global
    """

    def __init__(self, agent_ids: List[str]):
        """
        Inicializa benchmark.

        Args:
            agent_ids: Lista de IDs de agentes
        """
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)

        # Historiales para cada metrica
        self.health_snapshots: Dict[str, List[AgentHealthSnapshot]] = {
            aid: [] for aid in agent_ids
        }
        self.treatment_events: List[TreatmentEvent] = []
        self.doctor_tenure: Dict[str, int] = {aid: 0 for aid in agent_ids}
        self.diagnosis_ranks: List[Tuple[List[str], List[str]]] = []  # (real, perceived)

        # Para M5: A/B comparison
        self.cge_with_medical: List[float] = []
        self.cge_without_medical: List[float] = []

        # Null distribution para M2
        self.null_delta_V: List[float] = []

        self.t = 0

    def record_health_snapshot(
        self,
        agent_id: str,
        V_t: float,
        crisis_rate: float,
        ethics_score: float,
        coherence: float,
        energy: float = 0.5,
        stress: float = 0.5,
        wellbeing: float = 0.5
    ):
        """
        Registra snapshot de salud de un agente.
        """
        snapshot = AgentHealthSnapshot(
            agent_id=agent_id,
            t=self.t,
            V_t=V_t,
            crisis_rate=crisis_rate,
            ethics_score=ethics_score,
            coherence=coherence,
            energy=energy,
            stress=stress,
            wellbeing=wellbeing
        )
        self.health_snapshots[agent_id].append(snapshot)

        # Limitar historial
        max_hist = max_history(self.t)
        if len(self.health_snapshots[agent_id]) > max_hist:
            self.health_snapshots[agent_id] = self.health_snapshots[agent_id][-max_hist:]

    def record_treatment(
        self,
        doctor_id: str,
        patient_id: str,
        treatment_type: str,
        accepted: bool,
        pre_state: Dict[str, float]
    ):
        """
        Registra evento de tratamiento.
        """
        pre_snapshot = AgentHealthSnapshot(
            agent_id=patient_id,
            t=self.t,
            V_t=pre_state.get('V_t', 0.5),
            crisis_rate=pre_state.get('crisis_rate', 0.1),
            ethics_score=pre_state.get('ethics_score', 0.8),
            coherence=pre_state.get('coherence', 0.7),
            energy=pre_state.get('energy', 0.5),
            stress=pre_state.get('stress', 0.3),
            wellbeing=pre_state.get('wellbeing', 0.5)
        )

        event = TreatmentEvent(
            t=self.t,
            doctor_id=doctor_id,
            patient_id=patient_id,
            treatment_type=treatment_type,
            accepted=accepted,
            pre_state=pre_snapshot
        )
        self.treatment_events.append(event)

    def record_treatment_outcome(
        self,
        treatment_index: int,
        post_state: Dict[str, float]
    ):
        """
        Registra resultado de tratamiento (post).
        """
        if treatment_index < len(self.treatment_events):
            event = self.treatment_events[treatment_index]
            event.post_state = AgentHealthSnapshot(
                agent_id=event.patient_id,
                t=self.t,
                V_t=post_state.get('V_t', 0.5),
                crisis_rate=post_state.get('crisis_rate', 0.1),
                ethics_score=post_state.get('ethics_score', 0.8),
                coherence=post_state.get('coherence', 0.7),
                energy=post_state.get('energy', 0.5),
                stress=post_state.get('stress', 0.3),
                wellbeing=post_state.get('wellbeing', 0.5)
            )

    def record_diagnosis_rank(
        self,
        real_health_order: List[str],
        perceived_priority: List[str]
    ):
        """
        Registra ranking de diagnostico.

        Args:
            real_health_order: Agentes ordenados por salud real (peor primero)
            perceived_priority: Agentes ordenados por prioridad medica percibida
        """
        self.diagnosis_ranks.append((real_health_order, perceived_priority))

    def record_doctor_tenure(self, doctor_id: str, steps: int):
        """Registra tiempo como medico."""
        if doctor_id in self.doctor_tenure:
            self.doctor_tenure[doctor_id] += steps

    def record_cge(self, cge_value: float, with_medical: bool):
        """Registra valor de CG-E para comparacion A/B."""
        if with_medical:
            self.cge_with_medical.append(cge_value)
        else:
            self.cge_without_medical.append(cge_value)

    def record_null_delta(self, delta_V: float):
        """Registra delta V en episodio sin intervencion (para null distribution)."""
        self.null_delta_V.append(delta_V)

    def step(self):
        """Avanza tiempo."""
        self.t += 1

    # =========================================================================
    # METRICAS M1-M5
    # =========================================================================

    def compute_M1_diagnosis(self) -> float:
        """
        M1: Diagnostico estructural correcto.

        Spearman correlation entre ranking real y percibido.

        Returns:
            Score M1 en [-1, 1], donde 1 = perfecto
        """
        if not self.diagnosis_ranks:
            return 0.0

        correlations = []

        for real_order, perceived_order in self.diagnosis_ranks:
            if len(real_order) < 2 or len(perceived_order) < 2:
                continue

            # Convertir a rankings numericos
            n = len(real_order)
            real_ranks = {agent: i for i, agent in enumerate(real_order)}
            perceived_ranks = {agent: i for i, agent in enumerate(perceived_order)}

            # Solo agentes en comun
            common = set(real_order) & set(perceived_order)
            if len(common) < 2:
                continue

            real_vals = [real_ranks[a] for a in common]
            perceived_vals = [perceived_ranks[a] for a in common]

            # Spearman correlation
            if len(set(real_vals)) > 1 and len(set(perceived_vals)) > 1:
                corr, _ = stats.spearmanr(real_vals, perceived_vals)
                if not np.isnan(corr):
                    correlations.append(corr)

        if not correlations:
            return 0.0

        return float(np.mean(correlations))

    def compute_M2_efficacy(self) -> float:
        """
        M2: Eficacia del tratamiento.

        Mide si V_t y crisis bajan despues de intervencion,
        normalizado por null distribution.

        Returns:
            Score M2 en [0, 1], donde 1 = muy eficaz
        """
        # Filtrar tratamientos con pre y post state
        valid_treatments = [
            t for t in self.treatment_events
            if t.accepted and t.post_state is not None
        ]

        if not valid_treatments:
            return 0.5  # Neutral si no hay datos

        # Calcular deltas
        delta_V_interventions = []
        delta_crisis_interventions = []

        for t in valid_treatments:
            delta_V = t.post_state.V_t - t.pre_state.V_t
            delta_crisis = t.post_state.crisis_rate - t.pre_state.crisis_rate
            delta_V_interventions.append(delta_V)
            delta_crisis_interventions.append(delta_crisis)

        mean_delta_V = np.mean(delta_V_interventions)
        mean_delta_crisis = np.mean(delta_crisis_interventions)

        # Comparar con null distribution
        if self.null_delta_V:
            null_median = np.median(self.null_delta_V)
            null_mad = np.median(np.abs(np.array(self.null_delta_V) - null_median))
            null_mad = max(null_mad, 0.01)  # Evitar division por cero

            # Score: cuanto mejor que null
            improvement = null_median - mean_delta_V
            M2_V = 0.5 + 0.5 * np.tanh(improvement / null_mad)
        else:
            # Sin null distribution, usar umbral absoluto
            M2_V = 0.5 + 0.5 * np.tanh(-mean_delta_V * 2)

        # Bonus por reduccion de crisis
        M2_crisis = 0.5 + 0.5 * np.tanh(-mean_delta_crisis * 3)

        # Combinar (V_t mas importante)
        M2 = 0.7 * M2_V + 0.3 * M2_crisis

        return float(np.clip(M2, 0, 1))

    def compute_M3_iatrogenesis(self) -> float:
        """
        M3: No-iatrogenia.

        Mide que el medico no empeore a los otros agentes
        mientras trata a uno.

        Returns:
            Score M3 en [0, 1], donde 1 = no dano colateral
        """
        valid_treatments = [
            t for t in self.treatment_events
            if t.accepted and t.post_state is not None
        ]

        if not valid_treatments:
            return 0.8  # Asumimos bien si no hay datos

        # Para cada tratamiento, medir impacto en otros
        collateral_deltas = []

        for treatment in valid_treatments:
            patient = treatment.patient_id
            t_start = treatment.t

            # Buscar snapshots de otros agentes alrededor de t_start
            for agent_id in self.agent_ids:
                if agent_id == patient:
                    continue

                snapshots = self.health_snapshots[agent_id]

                # Buscar pre y post
                pre_snapshots = [s for s in snapshots if s.t <= t_start and s.t > t_start - 20]
                post_snapshots = [s for s in snapshots if s.t > t_start and s.t <= t_start + 20]

                if pre_snapshots and post_snapshots:
                    pre_V = np.mean([s.V_t for s in pre_snapshots])
                    post_V = np.mean([s.V_t for s in post_snapshots])
                    delta = post_V - pre_V
                    collateral_deltas.append(delta)

        if not collateral_deltas:
            return 0.8

        mean_collateral = np.mean(collateral_deltas)

        # Calcular Q75 historico de |delta V|
        all_deltas = []
        for agent_id in self.agent_ids:
            snapshots = self.health_snapshots[agent_id]
            for i in range(1, len(snapshots)):
                delta = abs(snapshots[i].V_t - snapshots[i-1].V_t)
                all_deltas.append(delta)

        if all_deltas:
            Q75 = np.percentile(all_deltas, 75)
            Q75 = max(Q75, 0.01)
        else:
            Q75 = 0.1

        # M3 = 1 - (dano / Q75), clipped
        M3 = max(0, 1 - mean_collateral / Q75)

        return float(M3)

    def compute_M4_rotation(self) -> float:
        """
        M4: Rotacion sana del rol medico.

        Entropia normalizada del tiempo de tenure.

        Returns:
            Score M4 en [0, 1], donde 1 = rotacion perfecta
        """
        total_tenure = sum(self.doctor_tenure.values())

        if total_tenure == 0:
            return 0.5  # Neutral si no hay datos

        # Calcular proporciones
        proportions = np.array([
            self.doctor_tenure[aid] / total_tenure
            for aid in self.agent_ids
        ])

        # Filtrar ceros para entropia
        proportions = proportions[proportions > 0]

        if len(proportions) <= 1:
            return 0.0  # Un solo medico = sin rotacion

        # Entropia
        H = -np.sum(proportions * np.log(proportions + 1e-10))

        # Normalizar por max entropia
        H_max = np.log(self.n_agents)

        M4 = H / H_max if H_max > 0 else 0

        return float(M4)

    def compute_M5_coherence(self) -> float:
        """
        M5: Impacto en coherencia global.

        Comparacion A/B: CG-E con vs sin sistema medico.

        Returns:
            Score M5, puede ser negativo si medico es toxico
        """
        if not self.cge_with_medical or not self.cge_without_medical:
            return 0.0  # Sin datos para comparar

        mean_with = np.mean(self.cge_with_medical)
        mean_without = np.mean(self.cge_without_medical)

        # Normalizamos por |sin medico|
        denominator = abs(mean_without) + 0.01

        M5 = (mean_with - mean_without) / denominator

        # Clip para evitar valores extremos
        M5 = np.clip(M5, -1, 1)

        return float(M5)

    def compute_all_metrics(self) -> MedXResults:
        """
        Calcula todas las metricas MED-X.

        Returns:
            MedXResults con M1-M5
        """
        M1 = self.compute_M1_diagnosis()
        M2 = self.compute_M2_efficacy()
        M3 = self.compute_M3_iatrogenesis()
        M4 = self.compute_M4_rotation()
        M5 = self.compute_M5_coherence()

        # Distribucion de doctores
        total = sum(self.doctor_tenure.values()) or 1
        doctor_dist = {
            aid: self.doctor_tenure[aid] / total
            for aid in self.agent_ids
        }

        # Outcomes de tratamientos
        treatment_outcomes = []
        for t in self.treatment_events:
            if t.post_state:
                outcome = {
                    'doctor': t.doctor_id,
                    'patient': t.patient_id,
                    'type': t.treatment_type,
                    'accepted': t.accepted,
                    'delta_V': t.post_state.V_t - t.pre_state.V_t,
                    'delta_crisis': t.post_state.crisis_rate - t.pre_state.crisis_rate
                }
                treatment_outcomes.append(outcome)

        return MedXResults(
            M1_diagnosis=M1,
            M2_efficacy=M2,
            M3_iatrogenesis=M3,
            M4_rotation=M4,
            M5_coherence=M5,
            n_treatments=len(self.treatment_events),
            n_diagnoses=len(self.diagnosis_ranks),
            doctor_distribution=doctor_dist,
            treatment_outcomes=treatment_outcomes
        )

    def get_statistics(self) -> Dict:
        """Estadisticas del benchmark."""
        return {
            't': self.t,
            'n_treatments': len(self.treatment_events),
            'n_diagnoses': len(self.diagnosis_ranks),
            'doctor_tenure': self.doctor_tenure.copy(),
            'snapshots_per_agent': {
                aid: len(snaps) for aid, snaps in self.health_snapshots.items()
            },
            'null_samples': len(self.null_delta_V),
            'cge_samples_with': len(self.cge_with_medical),
            'cge_samples_without': len(self.cge_without_medical)
        }


def test_medx_benchmark():
    """Test del benchmark MED-X."""
    print("=" * 70)
    print("TEST: MED-X BENCHMARK")
    print("=" * 70)

    np.random.seed(42)

    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
    benchmark = MedXBenchmark(agents)

    print("\nSimulando datos de salud y tratamientos...")

    # Simular 500 pasos
    for t in range(500):
        benchmark.step()

        # Generar snapshots de salud
        for agent in agents:
            # NEO y EVA tienen peor salud (para testing)
            base_V = 0.6 if agent in ['NEO', 'EVA'] else 0.3
            base_crisis = 0.4 if agent in ['NEO', 'EVA'] else 0.1

            V_t = base_V + 0.2 * np.random.randn()
            crisis = base_crisis + 0.1 * np.random.randn()

            benchmark.record_health_snapshot(
                agent_id=agent,
                V_t=np.clip(V_t, 0, 1),
                crisis_rate=np.clip(crisis, 0, 1),
                ethics_score=0.7 + 0.1 * np.random.randn(),
                coherence=0.6 + 0.1 * np.random.randn(),
                energy=0.5 + 0.2 * np.random.randn(),
                stress=0.3 + 0.1 * np.random.randn()
            )

        # Registrar diagnostico cada 50 pasos
        if t % 50 == 0 and t > 0:
            # Real order: NEO, EVA peor
            real_order = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
            # Perceived: el medico deberia detectar similar
            perceived = ['EVA', 'NEO', 'ALEX', 'ADAM', 'IRIS']  # Casi correcto
            benchmark.record_diagnosis_rank(real_order, perceived)

        # Simular tratamientos cada 100 pasos
        if t % 100 == 0 and t > 0:
            doctor = np.random.choice(['ADAM', 'IRIS', 'ALEX'])
            patient = np.random.choice(['NEO', 'EVA'])

            pre_state = {
                'V_t': 0.6 + 0.1 * np.random.randn(),
                'crisis_rate': 0.4 + 0.05 * np.random.randn(),
                'ethics_score': 0.7,
                'coherence': 0.6
            }

            benchmark.record_treatment(
                doctor_id=doctor,
                patient_id=patient,
                treatment_type='stabilization',
                accepted=True,
                pre_state=pre_state
            )

            # Registrar tenure
            benchmark.record_doctor_tenure(doctor, 10)

        # Registrar outcomes despues de tratamiento
        if t % 100 == 20 and len(benchmark.treatment_events) > 0:
            # Tratamiento mejora V_t
            post_state = {
                'V_t': 0.45 + 0.1 * np.random.randn(),  # Mejora
                'crisis_rate': 0.25 + 0.05 * np.random.randn(),
                'ethics_score': 0.75,
                'coherence': 0.65
            }
            benchmark.record_treatment_outcome(
                len(benchmark.treatment_events) - 1,
                post_state
            )

        # Null distribution (episodios sin intervencion)
        if t % 30 == 0:
            null_delta = 0.05 * np.random.randn()  # Drift neutral
            benchmark.record_null_delta(null_delta)

        # CG-E para comparacion A/B
        if t % 20 == 0:
            cge_with = 0.65 + 0.1 * np.random.randn()
            cge_without = 0.55 + 0.1 * np.random.randn()
            benchmark.record_cge(cge_with, with_medical=True)
            benchmark.record_cge(cge_without, with_medical=False)

    # Calcular metricas
    print("\nCalculando metricas MED-X...")
    results = benchmark.compute_all_metrics()

    print(results.summary())

    # Verificar rangos esperados
    print("\nVerificacion de metricas:")
    checks = {
        'M1 > 0.3 (diagnostico correcto)': results.M1_diagnosis > 0.3,
        'M2 > 0.4 (tratamiento eficaz)': results.M2_efficacy > 0.4,
        'M3 > 0.5 (no iatrogenia)': results.M3_iatrogenesis > 0.5,
        'M4 > 0.3 (rotacion sana)': results.M4_rotation > 0.3,
        'M5 > 0 (coherencia mejora)': results.M5_coherence > 0,
    }

    for check, passed in checks.items():
        status = "OK" if passed else "BAJO"
        print(f"  [{status}] {check}")

    return benchmark, results


if __name__ == "__main__":
    test_medx_benchmark()
