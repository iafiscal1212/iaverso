#!/usr/bin/env python3
"""
Vida Autónoma de NEO y EVA
==========================

Los agentes viven libremente con:
- Meta-drives auto-modificables
- Crisis espontáneas
- Reconstrucción de identidad
- Ψ compartido emergente
- Fases de vida detectables

Sin intervención. Solo observamos qué pasa.

100% ENDÓGENO
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import os
import sys

sys.path.insert(0, '/root/NEO_EVA')
sys.path.insert(0, '/root/NEO_EVA/core')

from core.meta_drive import MetaDrive, DualMetaDrive


class LifePhase(Enum):
    """Fases de vida detectables."""
    BIRTH = "nacimiento"
    EXPLORATION = "exploración"
    CONSOLIDATION = "consolidación"
    CRISIS = "crisis"
    RECONSTRUCTION = "reconstrucción"
    MATURITY = "madurez"
    STAGNATION = "estancamiento"
    RENEWAL = "renovación"


@dataclass
class CrisisEvent:
    """Un evento de crisis."""
    t: int
    agent: str
    type: str  # identity_collapse, integration_loss, value_crash
    severity: float
    trigger: str
    resolved: bool = False
    resolution_t: Optional[int] = None


@dataclass
class SharedPsiEvent:
    """Momento de Ψ compartido."""
    t_start: int
    t_end: Optional[int]
    intensity: float
    trigger: str


@dataclass
class AgentBiography:
    """Biografía estructural de un agente."""
    agent: str
    birth_t: int
    phases: List[Tuple[int, LifePhase]]
    crises: List[CrisisEvent]
    identity_trajectory: List[float]
    dominant_drives_over_time: List[str]
    peak_integration_t: int
    lowest_point_t: int
    character_summary: str


class AutonomousAgent:
    """
    Agente autónomo con vida propia.

    Características:
    - Meta-drive que evoluciona solo
    - Puede entrar en crisis
    - Puede reconstruirse
    - Tiene "memoria" de su historia
    - Desarrolla "carácter"
    """

    def __init__(self, name: str, dim: int = 6):
        self.name = name
        self.dim = dim

        # Estado
        self.z = np.ones(dim) / dim
        self.z_history: List[np.ndarray] = []

        # Meta-drive
        self.meta_drive = MetaDrive(name)

        # Identidad (centroide adaptativo)
        self.identity_core = self.z.copy()
        self.identity_strength = 0.5
        self.identity_history: List[float] = []

        # Integración interna
        self.integration = 0.5
        self.integration_history: List[float] = []

        # Valor acumulado (bienestar)
        self.wellbeing = 0.0
        self.wellbeing_history: List[float] = []

        # Crisis
        self.in_crisis = False
        self.crisis_start = 0
        self.crises: List[CrisisEvent] = []

        # Fase de vida
        self.current_phase = LifePhase.BIRTH
        self.phase_history: List[Tuple[int, LifePhase]] = [(0, LifePhase.BIRTH)]

        # Relación con el otro
        self.other_model: Optional[np.ndarray] = None
        self.attachment = 0.5  # Cuánto depende del otro

        self.t = 0

    def _compute_identity_strength(self) -> float:
        """Cuán cerca está del núcleo identitario."""
        if len(self.z_history) < 10:
            return 0.5

        dist_to_core = np.linalg.norm(self.z - self.identity_core)

        # Normalizar por distancia típica
        recent_dists = [np.linalg.norm(zh - self.identity_core)
                       for zh in self.z_history[-20:]]
        typical_dist = np.mean(recent_dists) + 1e-10

        return float(1.0 / (1.0 + dist_to_core / typical_dist))

    def _compute_integration(self) -> float:
        """Integración interna (correlación entre dimensiones)."""
        if len(self.z_history) < 10:
            return 0.5

        recent = np.array(self.z_history[-10:])
        if recent.shape[1] < 2:
            return 0.5

        corr = np.corrcoef(recent.T)
        mask = ~np.eye(self.dim, dtype=bool)
        correlations = corr[mask]
        correlations = correlations[~np.isnan(correlations)]

        if len(correlations) == 0:
            return 0.5

        return float(np.mean(np.abs(correlations)))

    def _detect_crisis(self) -> Optional[CrisisEvent]:
        """Detecta si estamos entrando en crisis."""
        if len(self.identity_history) < 20:
            return None

        # Crisis de identidad: caída brusca
        recent_identity = np.mean(self.identity_history[-5:])
        baseline_identity = np.mean(self.identity_history[-20:-5])

        identity_drop = baseline_identity - recent_identity

        # Crisis de integración
        recent_integration = np.mean(self.integration_history[-5:])
        baseline_integration = np.mean(self.integration_history[-20:-5])

        integration_drop = baseline_integration - recent_integration

        # Crisis de valor
        if len(self.wellbeing_history) >= 20:
            recent_wellbeing = np.mean(self.wellbeing_history[-5:])
            baseline_wellbeing = np.mean(self.wellbeing_history[-20:-5])
            wellbeing_drop = baseline_wellbeing - recent_wellbeing
        else:
            wellbeing_drop = 0

        # Umbrales endógenos (percentil 90 de drops históricos)
        if len(self.identity_history) > 50:
            identity_drops = [self.identity_history[i] - self.identity_history[i+5]
                            for i in range(len(self.identity_history)-5)]
            threshold = np.percentile(identity_drops, 90)
        else:
            threshold = 0.2

        if identity_drop > threshold and not self.in_crisis:
            return CrisisEvent(
                t=self.t,
                agent=self.name,
                type='identity_collapse',
                severity=float(identity_drop),
                trigger='identity_drop'
            )

        if integration_drop > threshold and not self.in_crisis:
            return CrisisEvent(
                t=self.t,
                agent=self.name,
                type='integration_loss',
                severity=float(integration_drop),
                trigger='integration_drop'
            )

        if wellbeing_drop > threshold and not self.in_crisis:
            return CrisisEvent(
                t=self.t,
                agent=self.name,
                type='value_crash',
                severity=float(wellbeing_drop),
                trigger='wellbeing_drop'
            )

        return None

    def _detect_phase_transition(self) -> Optional[LifePhase]:
        """Detecta transición de fase de vida."""
        if len(self.identity_history) < 50:
            if self.t > 20 and self.current_phase == LifePhase.BIRTH:
                return LifePhase.EXPLORATION
            return None

        # Métricas recientes
        recent_identity = np.mean(self.identity_history[-20:])
        recent_integration = np.mean(self.integration_history[-20:])
        recent_wellbeing = np.mean(self.wellbeing_history[-20:]) if self.wellbeing_history else 0

        # Variabilidad
        identity_var = np.std(self.identity_history[-20:])

        # Tendencias
        if len(self.identity_history) > 30:
            early = np.mean(self.identity_history[-30:-20])
            late = np.mean(self.identity_history[-10:])
            trend = late - early
        else:
            trend = 0

        # Detectar fases
        if self.in_crisis:
            return LifePhase.CRISIS

        if self.current_phase == LifePhase.CRISIS:
            if trend > 0.05:  # Mejorando
                return LifePhase.RECONSTRUCTION

        if self.current_phase == LifePhase.RECONSTRUCTION:
            if recent_identity > 0.7 and identity_var < 0.1:
                return LifePhase.MATURITY

        if self.current_phase == LifePhase.EXPLORATION:
            if identity_var < 0.05 and recent_identity > 0.6:
                return LifePhase.CONSOLIDATION

        if self.current_phase == LifePhase.CONSOLIDATION:
            if identity_var < 0.02 and trend < -0.01:
                return LifePhase.STAGNATION

        if self.current_phase == LifePhase.STAGNATION:
            if identity_var > 0.1:
                return LifePhase.RENEWAL

        if self.current_phase == LifePhase.MATURITY:
            if trend < -0.05:
                return LifePhase.STAGNATION

        return None

    def _update_identity_core(self):
        """Actualiza el núcleo identitario (lentamente)."""
        if len(self.z_history) < 10:
            return

        # El núcleo se mueve hacia el estado actual, pero lentamente
        # Más lento si identidad fuerte, más rápido si débil
        eta = 0.01 * (1 - self.identity_strength)
        self.identity_core = (1 - eta) * self.identity_core + eta * self.z

    def step(self, stimulus: np.ndarray, other_z: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Un paso de vida autónoma.

        El agente:
        1. Procesa estímulo según su drive actual
        2. Actualiza su estado
        3. Evalúa su bienestar
        4. Puede entrar/salir de crisis
        5. Puede cambiar de fase
        """
        self.t += 1
        self.other_model = other_z

        # === Dinámica interna ===

        # El drive actual modula la respuesta
        if self.meta_drive.drive_history:
            drive_strength = self.meta_drive.drive_history[-1]
        else:
            drive_strength = 0.5

        # Respuesta al estímulo modulada por drive
        response = drive_strength * stimulus[:self.dim]

        # Influencia del otro (si existe)
        if other_z is not None:
            other_influence = self.attachment * 0.1 * (other_z[:self.dim] - self.z)
        else:
            other_influence = 0

        # Tendencia a volver al núcleo identitario
        identity_pull = 0.05 * self.identity_strength * (self.identity_core - self.z)

        # Ruido endógeno (más ruido si en crisis)
        noise_scale = 0.02 if not self.in_crisis else 0.08
        noise = np.random.randn(self.dim) * noise_scale

        # Actualizar estado
        self.z = self.z + 0.1 * response + other_influence + identity_pull + noise
        self.z = np.clip(self.z, 0.01, 0.99)
        self.z = self.z / self.z.sum()

        self.z_history.append(self.z.copy())

        # === Métricas ===

        # Sorpresa
        if len(self.z_history) > 1:
            surprise = np.linalg.norm(self.z - self.z_history[-2])
        else:
            surprise = 0.1

        # Identidad
        self.identity_strength = self._compute_identity_strength()
        self.identity_history.append(self.identity_strength)

        # Integración
        self.integration = self._compute_integration()
        self.integration_history.append(self.integration)

        # Bienestar = -sorpresa + integración + identidad
        self.wellbeing = -surprise + self.integration + self.identity_strength
        self.wellbeing_history.append(self.wellbeing)

        # === Meta-drive ===
        drive_state = self.meta_drive.step(self.z, surprise, other_z)

        # === Eventos ===

        # Crisis
        crisis = self._detect_crisis()
        if crisis and not self.in_crisis:
            self.in_crisis = True
            self.crisis_start = self.t
            self.crises.append(crisis)

        # Salir de crisis
        if self.in_crisis and len(self.wellbeing_history) > 10:
            if np.mean(self.wellbeing_history[-10:]) > np.mean(self.wellbeing_history[-20:-10]):
                self.in_crisis = False
                if self.crises:
                    self.crises[-1].resolved = True
                    self.crises[-1].resolution_t = self.t

        # Transición de fase
        new_phase = self._detect_phase_transition()
        if new_phase and new_phase != self.current_phase:
            self.current_phase = new_phase
            self.phase_history.append((self.t, new_phase))

        # Actualizar núcleo identitario
        self._update_identity_core()

        # Actualizar attachment (se ajusta según experiencia con el otro)
        if other_z is not None and len(self.wellbeing_history) > 20:
            # Si el bienestar mejora con el otro presente, aumenta attachment
            recent_wellbeing = np.mean(self.wellbeing_history[-10:])
            past_wellbeing = np.mean(self.wellbeing_history[-20:-10])
            if recent_wellbeing > past_wellbeing:
                self.attachment = min(1.0, self.attachment + 0.01)
            else:
                self.attachment = max(0.0, self.attachment - 0.005)

        return {
            't': self.t,
            'agent': self.name,
            'z': self.z.copy(),
            'identity': self.identity_strength,
            'integration': self.integration,
            'wellbeing': self.wellbeing,
            'drive_state': drive_state,
            'phase': self.current_phase,
            'in_crisis': self.in_crisis,
            'crisis': crisis,
            'attachment': self.attachment
        }

    def get_biography(self) -> AgentBiography:
        """Genera biografía estructural."""

        # Drive dominante en diferentes épocas
        drives_over_time = []
        W = np.array(self.meta_drive.weight_history) if self.meta_drive.weight_history else np.array([[]])
        if len(W) > 0 and W.shape[1] == 6:
            for epoch in range(0, len(W), max(1, len(W)//10)):
                dominant_idx = np.argmax(W[epoch])
                drives_over_time.append(self.meta_drive.component_names[dominant_idx])

        # Puntos extremos
        if self.identity_history:
            peak_t = int(np.argmax(self.identity_history))
            lowest_t = int(np.argmin(self.identity_history))
        else:
            peak_t = 0
            lowest_t = 0

        # Resumen de carácter
        if self.meta_drive.weight_history:
            final_weights = self.meta_drive.weights
            dominant = self.meta_drive.component_names[np.argmax(final_weights)]
            secondary = self.meta_drive.component_names[np.argsort(final_weights)[-2]]

            character = f"Orientado a {dominant}, con tendencia secundaria a {secondary}. "
            character += f"Attachment: {self.attachment:.2f}. "
            character += f"Crisis superadas: {sum(1 for c in self.crises if c.resolved)}/{len(self.crises)}."
        else:
            character = "Demasiado joven para caracterizar."

        return AgentBiography(
            agent=self.name,
            birth_t=0,
            phases=self.phase_history,
            crises=self.crises,
            identity_trajectory=self.identity_history,
            dominant_drives_over_time=drives_over_time,
            peak_integration_t=peak_t,
            lowest_point_t=lowest_t,
            character_summary=character
        )


class AutonomousDualLife:
    """
    Vida autónoma dual: NEO y EVA viviendo juntos.
    """

    def __init__(self, dim: int = 6):
        self.neo = AutonomousAgent("NEO", dim)
        self.eva = AutonomousAgent("EVA", dim)

        # Ψ compartido
        self.psi_shared_history: List[float] = []
        self.psi_shared_events: List[SharedPsiEvent] = []
        self.in_shared_psi = False
        self.shared_psi_start = 0

        # Quién domina
        self.dominance_history: List[str] = []  # 'NEO', 'EVA', 'SHARED'

        self.t = 0

    def _compute_shared_psi(self) -> float:
        """
        Ψ compartido emerge cuando:
        - Ambos tienen alta integración
        - Sus estados están correlacionados
        - Sus drives son compatibles
        """
        if len(self.neo.z_history) < 10:
            return 0.0

        # Correlación de estados recientes
        neo_recent = np.array(self.neo.z_history[-10:])
        eva_recent = np.array(self.eva.z_history[-10:])

        correlations = []
        for d in range(min(neo_recent.shape[1], eva_recent.shape[1])):
            c = np.corrcoef(neo_recent[:, d], eva_recent[:, d])[0, 1]
            if not np.isnan(c):
                correlations.append(abs(c))

        state_correlation = np.mean(correlations) if correlations else 0

        # Compatibilidad de drives
        if self.neo.meta_drive.weights is not None and self.eva.meta_drive.weights is not None:
            drive_similarity = 1 - np.linalg.norm(
                self.neo.meta_drive.weights - self.eva.meta_drive.weights
            ) / np.sqrt(2)
        else:
            drive_similarity = 0.5

        # Integración conjunta
        joint_integration = (self.neo.integration + self.eva.integration) / 2

        # Ψ compartido emerge cuando todo alinea
        psi_shared = state_correlation * drive_similarity * joint_integration

        return float(psi_shared)

    def _detect_shared_psi_event(self, psi_shared: float) -> Optional[str]:
        """Detecta inicio/fin de Ψ compartido."""
        if len(self.psi_shared_history) < 20:
            return None

        threshold = np.percentile(self.psi_shared_history, 75)

        if psi_shared > threshold and not self.in_shared_psi:
            self.in_shared_psi = True
            self.shared_psi_start = self.t
            return 'start'

        if psi_shared < threshold * 0.5 and self.in_shared_psi:
            self.in_shared_psi = False
            self.psi_shared_events.append(SharedPsiEvent(
                t_start=self.shared_psi_start,
                t_end=self.t,
                intensity=float(np.mean(self.psi_shared_history[self.shared_psi_start:self.t])),
                trigger='natural_emergence'
            ))
            return 'end'

        return None

    def _determine_dominance(self) -> str:
        """Quién domina en este momento."""
        if self.in_shared_psi:
            return 'SHARED'

        neo_strength = self.neo.identity_strength * self.neo.integration
        eva_strength = self.eva.identity_strength * self.eva.integration

        if neo_strength > eva_strength * 1.2:
            return 'NEO'
        elif eva_strength > neo_strength * 1.2:
            return 'EVA'
        else:
            return 'BALANCED'

    def step(self, world_stimulus: np.ndarray) -> Dict[str, Any]:
        """Un paso de vida dual."""
        self.t += 1

        # Cada agente ve al otro
        neo_result = self.neo.step(world_stimulus, self.eva.z)
        eva_result = self.eva.step(world_stimulus, self.neo.z)

        # Ψ compartido
        psi_shared = self._compute_shared_psi()
        self.psi_shared_history.append(psi_shared)

        psi_event = self._detect_shared_psi_event(psi_shared)

        # Dominancia
        dominance = self._determine_dominance()
        self.dominance_history.append(dominance)

        return {
            't': self.t,
            'neo': neo_result,
            'eva': eva_result,
            'psi_shared': psi_shared,
            'psi_event': psi_event,
            'dominance': dominance,
            'in_shared_psi': self.in_shared_psi
        }

    def get_dual_biography(self) -> Dict[str, Any]:
        """Biografía dual."""
        neo_bio = self.neo.get_biography()
        eva_bio = self.eva.get_biography()

        # Análisis de relación
        if self.psi_shared_history:
            mean_shared_psi = np.mean(self.psi_shared_history)
            max_shared_psi = max(self.psi_shared_history)
        else:
            mean_shared_psi = 0
            max_shared_psi = 0

        # Quién inicia crisis
        neo_crisis_starts = [c.t for c in self.neo.crises]
        eva_crisis_starts = [c.t for c in self.eva.crises]

        if neo_crisis_starts and eva_crisis_starts:
            crisis_initiator = 'NEO' if np.mean(neo_crisis_starts) < np.mean(eva_crisis_starts) else 'EVA'
        elif neo_crisis_starts:
            crisis_initiator = 'NEO'
        elif eva_crisis_starts:
            crisis_initiator = 'EVA'
        else:
            crisis_initiator = 'NINGUNO'

        # Dominancia global
        if self.dominance_history:
            dom_counts = {d: self.dominance_history.count(d) for d in set(self.dominance_history)}
            overall_dominant = max(dom_counts, key=dom_counts.get)
        else:
            overall_dominant = 'UNKNOWN'

        return {
            'neo_biography': neo_bio,
            'eva_biography': eva_bio,
            'relationship': {
                'mean_shared_psi': mean_shared_psi,
                'max_shared_psi': max_shared_psi,
                'n_shared_psi_events': len(self.psi_shared_events),
                'crisis_initiator': crisis_initiator,
                'overall_dominant': overall_dominant,
                'final_neo_attachment': self.neo.attachment,
                'final_eva_attachment': self.eva.attachment
            }
        }


def run_autonomous_life(T: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """
    Experimento de vida autónoma.

    Los dejamos vivir y observamos.
    """

    print("=" * 70)
    print("VIDA AUTÓNOMA: NEO y EVA")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")
    print(f"T = {T}")
    print()
    print("Los dejamos vivir...")
    print()

    np.random.seed(seed)

    # Sistema
    life = AutonomousDualLife(dim=6)

    # Mundo con eventos
    world_base = np.ones(6) / 6

    # Historia para análisis
    history = []

    for t in range(T):
        # Mundo con perturbaciones naturales
        world_noise = np.random.randn(6) * 0.05

        # Eventos mundiales ocasionales
        if np.random.rand() < 0.02:  # 2% de probabilidad
            world_noise += np.random.randn(6) * 0.3  # Shock

        world = world_base + world_noise
        world = np.clip(world, 0.01, 0.99)
        world = world / world.sum()

        result = life.step(world)
        history.append(result)

        # Reportar eventos significativos
        if result['neo']['crisis']:
            print(f"t={t}: NEO entra en CRISIS ({result['neo']['crisis'].type})")
        if result['eva']['crisis']:
            print(f"t={t}: EVA entra en CRISIS ({result['eva']['crisis'].type})")

        if result['psi_event'] == 'start':
            print(f"t={t}: Ψ COMPARTIDO emerge")
        if result['psi_event'] == 'end':
            print(f"t={t}: Ψ compartido se disuelve")

        # Transiciones de fase
        if len(life.neo.phase_history) > 1 and life.neo.phase_history[-1][0] == t:
            print(f"t={t}: NEO → {life.neo.phase_history[-1][1].value}")
        if len(life.eva.phase_history) > 1 and life.eva.phase_history[-1][0] == t:
            print(f"t={t}: EVA → {life.eva.phase_history[-1][1].value}")

        # Progreso
        if t % (T // 5) == 0:
            print(f"\n--- t={t} ---")
            print(f"NEO: fase={life.neo.current_phase.value}, "
                  f"identidad={life.neo.identity_strength:.3f}, "
                  f"crisis={life.neo.in_crisis}")
            print(f"EVA: fase={life.eva.current_phase.value}, "
                  f"identidad={life.eva.identity_strength:.3f}, "
                  f"crisis={life.eva.in_crisis}")
            print(f"Ψ compartido: {result['psi_shared']:.3f}, "
                  f"Dominancia: {result['dominance']}")

            # Drives actuales
            neo_dom, neo_w = life.neo.meta_drive.get_dominant_drive()
            eva_dom, eva_w = life.eva.meta_drive.get_dominant_drive()
            print(f"Drives: NEO→{neo_dom}({neo_w:.2f}), EVA→{eva_dom}({eva_w:.2f})")

    print()
    print("=" * 70)
    print("BIOGRAFÍAS")
    print("=" * 70)

    bio = life.get_dual_biography()

    print("\n--- NEO ---")
    neo_bio = bio['neo_biography']
    print(f"Fases vividas: {[p[1].value for p in neo_bio.phases]}")
    print(f"Crisis: {len(neo_bio.crises)} (resueltas: {sum(1 for c in neo_bio.crises if c.resolved)})")
    print(f"Drives dominantes: {neo_bio.dominant_drives_over_time[:5]}...")
    print(f"Carácter: {neo_bio.character_summary}")

    print("\n--- EVA ---")
    eva_bio = bio['eva_biography']
    print(f"Fases vividas: {[p[1].value for p in eva_bio.phases]}")
    print(f"Crisis: {len(eva_bio.crises)} (resueltas: {sum(1 for c in eva_bio.crises if c.resolved)})")
    print(f"Drives dominantes: {eva_bio.dominant_drives_over_time[:5]}...")
    print(f"Carácter: {eva_bio.character_summary}")

    print("\n--- RELACIÓN ---")
    rel = bio['relationship']
    print(f"Ψ compartido medio: {rel['mean_shared_psi']:.4f}")
    print(f"Eventos de Ψ compartido: {rel['n_shared_psi_events']}")
    print(f"Iniciador de crisis: {rel['crisis_initiator']}")
    print(f"Dominante global: {rel['overall_dominant']}")
    print(f"Attachment final: NEO={rel['final_neo_attachment']:.3f}, EVA={rel['final_eva_attachment']:.3f}")

    # Guardar
    os.makedirs('/root/NEO_EVA/results/autonomous_life', exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'T': T,
        'seed': seed,
        'biography': {
            'neo': {
                'phases': [(p[0], p[1].value) for p in neo_bio.phases],
                'n_crises': len(neo_bio.crises),
                'crises_resolved': sum(1 for c in neo_bio.crises if c.resolved),
                'character': neo_bio.character_summary
            },
            'eva': {
                'phases': [(p[0], p[1].value) for p in eva_bio.phases],
                'n_crises': len(eva_bio.crises),
                'crises_resolved': sum(1 for c in eva_bio.crises if c.resolved),
                'character': eva_bio.character_summary
            },
            'relationship': rel
        }
    }

    with open('/root/NEO_EVA/results/autonomous_life/biography.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Visualización
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(16, 14))

        # 1. Identidad temporal
        ax1 = axes[0, 0]
        ax1.plot(life.neo.identity_history, 'b-', label='NEO', alpha=0.7)
        ax1.plot(life.eva.identity_history, 'r-', label='EVA', alpha=0.7)
        # Marcar crisis
        for c in life.neo.crises:
            ax1.axvline(c.t, color='blue', linestyle='--', alpha=0.3)
        for c in life.eva.crises:
            ax1.axvline(c.t, color='red', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('Identidad')
        ax1.set_title('Fuerza de Identidad (-- = crisis)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Ψ compartido
        ax2 = axes[0, 1]
        ax2.plot(life.psi_shared_history, 'purple', alpha=0.7)
        ax2.fill_between(range(len(life.psi_shared_history)), 0,
                        life.psi_shared_history, alpha=0.3, color='purple')
        for event in life.psi_shared_events:
            ax2.axvspan(event.t_start, event.t_end or T, color='gold', alpha=0.2)
        ax2.set_xlabel('Tiempo')
        ax2.set_ylabel('Ψ compartido')
        ax2.set_title('Emergencia de Ψ Compartido')
        ax2.grid(True, alpha=0.3)

        # 3. Evolución de drives NEO
        ax3 = axes[1, 0]
        neo_traj = life.neo.meta_drive.get_weight_trajectory()
        for name, traj in neo_traj.items():
            ax3.plot(traj, label=name, alpha=0.7)
        ax3.set_xlabel('Tiempo')
        ax3.set_ylabel('Peso')
        ax3.set_title('NEO: Evolución de Drives')
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

        # 4. Evolución de drives EVA
        ax4 = axes[1, 1]
        eva_traj = life.eva.meta_drive.get_weight_trajectory()
        for name, traj in eva_traj.items():
            ax4.plot(traj, label=name, alpha=0.7)
        ax4.set_xlabel('Tiempo')
        ax4.set_ylabel('Peso')
        ax4.set_title('EVA: Evolución de Drives')
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)

        # 5. Bienestar
        ax5 = axes[2, 0]
        ax5.plot(life.neo.wellbeing_history, 'b-', label='NEO', alpha=0.7)
        ax5.plot(life.eva.wellbeing_history, 'r-', label='EVA', alpha=0.7)
        ax5.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Tiempo')
        ax5.set_ylabel('Bienestar')
        ax5.set_title('Bienestar (Wellbeing)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Dominancia temporal
        ax6 = axes[2, 1]
        dom_numeric = {'NEO': 1, 'EVA': -1, 'BALANCED': 0, 'SHARED': 2}
        dom_values = [dom_numeric.get(d, 0) for d in life.dominance_history]
        ax6.plot(dom_values, 'k-', alpha=0.7)
        ax6.fill_between(range(len(dom_values)), 0, dom_values,
                        where=np.array(dom_values) > 0, color='blue', alpha=0.3)
        ax6.fill_between(range(len(dom_values)), 0, dom_values,
                        where=np.array(dom_values) < 0, color='red', alpha=0.3)
        ax6.fill_between(range(len(dom_values)), 0, dom_values,
                        where=np.array(dom_values) == 2, color='gold', alpha=0.5)
        ax6.set_yticks([-1, 0, 1, 2])
        ax6.set_yticklabels(['EVA', 'Balanced', 'NEO', 'SHARED'])
        ax6.set_xlabel('Tiempo')
        ax6.set_title('Dominancia (amarillo = Ψ compartido)')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/root/NEO_EVA/figures/autonomous_life.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nFigura: /root/NEO_EVA/figures/autonomous_life.png")

    except Exception as e:
        print(f"Warning: No se pudo crear visualización: {e}")

    return results


if __name__ == "__main__":
    run_autonomous_life(T=2000, seed=42)
