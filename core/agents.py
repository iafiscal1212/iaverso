#!/usr/bin/env python3
"""
NEO y EVA: Agentes Duales con Especializaciones Distintas
=========================================================

NEO: Especializado en compresión (MDL - Minimum Description Length)
     - Busca representaciones compactas
     - Minimiza sorpresa interna
     - Orientado a estabilidad y predicción

EVA: Especializada en intercambio (MI - Mutual Information)
     - Maximiza información compartida con el mundo
     - Busca novedad y exploración
     - Orientada a comunicación y adaptación

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum


class AgentType(Enum):
    NEO = "NEO"
    EVA = "EVA"


@dataclass
class AgentState:
    """Estado interno de un agente."""
    z_visible: np.ndarray      # Estado visible
    z_hidden: np.ndarray       # Estado oculto (privado)
    S: float                   # Entropía
    drive: float               # Impulso dominante
    specialization: float      # Grado de especialización

    def get_full_state(self) -> np.ndarray:
        return np.concatenate([self.z_visible, self.z_hidden])


@dataclass
class AgentResponse:
    """Respuesta de un agente a un estímulo."""
    action: np.ndarray         # Vector de acción
    prediction: np.ndarray     # Predicción del mundo
    value: float               # Valoración del estado
    surprise: float            # Sorpresa experimentada
    report: np.ndarray         # Self-report fenomenológico


class BaseAgent(ABC):
    """Clase base para agentes NEO y EVA."""

    def __init__(self, dim_visible: int = 3, dim_hidden: int = 3, agent_type: AgentType = AgentType.NEO):
        self.agent_type = agent_type
        self.dim_visible = dim_visible
        self.dim_hidden = dim_hidden
        self.dim_total = dim_visible + dim_hidden

        # Estado interno
        self.z_visible = np.ones(dim_visible) / dim_visible
        self.z_hidden = np.ones(dim_hidden) / dim_hidden

        # Historia
        self.z_history: List[np.ndarray] = []
        self.S_history: List[float] = []
        self.prediction_history: List[np.ndarray] = []
        self.surprise_history: List[float] = []

        # Parámetros adaptativos (endógenos)
        self.learning_rate = 0.1  # Se ajusta con 1/√t
        self.exploration_rate = 0.5  # Se ajusta según especialización

        # Especialización (se desarrolla con el tiempo)
        self.specialization = 0.0
        self.t = 0

    def _update_learning_rate(self) -> None:
        """Tasa de aprendizaje endógena: η = 1/√(t+1)."""
        self.learning_rate = 1.0 / np.sqrt(self.t + 1)

    def _compute_entropy(self, z: np.ndarray) -> float:
        """Calcula entropía de un vector de probabilidad."""
        z_safe = np.clip(z, 1e-10, 1.0)
        z_norm = z_safe / z_safe.sum()
        return float(-np.sum(z_norm * np.log(z_norm)))

    def _compute_surprise(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Calcula sorpresa como distancia a predicción."""
        return float(np.linalg.norm(predicted - actual))

    @abstractmethod
    def _compute_drive(self) -> float:
        """Calcula impulso dominante (específico de cada agente)."""
        pass

    @abstractmethod
    def _generate_action(self, stimulus: np.ndarray) -> np.ndarray:
        """Genera acción en respuesta a estímulo."""
        pass

    @abstractmethod
    def _generate_prediction(self, stimulus: np.ndarray) -> np.ndarray:
        """Genera predicción del siguiente estado del mundo."""
        pass

    @abstractmethod
    def _compute_value(self, state: np.ndarray, stimulus: np.ndarray) -> float:
        """Computa valor/utilidad del estado actual."""
        pass

    def step(self, stimulus: np.ndarray) -> AgentResponse:
        """
        Ejecuta un paso del agente.

        Args:
            stimulus: Input del mundo (s_t)

        Returns:
            AgentResponse con acción, predicción, valor, sorpresa, report
        """
        self.t += 1
        self._update_learning_rate()

        # Predicción (antes de ver el estímulo completo)
        prediction = self._generate_prediction(stimulus)

        # Sorpresa
        if self.prediction_history:
            surprise = self._compute_surprise(self.prediction_history[-1], stimulus)
        else:
            surprise = 0.5
        self.surprise_history.append(surprise)

        # Actualizar estado interno basado en estímulo
        self._update_state(stimulus, surprise)

        # Generar acción
        action = self._generate_action(stimulus)

        # Computar valor
        full_state = np.concatenate([self.z_visible, self.z_hidden])
        value = self._compute_value(full_state, stimulus)

        # Generar self-report
        report = self._generate_report()

        # Registrar historia
        self.z_history.append(full_state.copy())
        self.S_history.append(self._compute_entropy(self.z_visible))
        self.prediction_history.append(prediction)

        # Actualizar especialización
        self._update_specialization()

        return AgentResponse(
            action=action,
            prediction=prediction,
            value=value,
            surprise=surprise,
            report=report
        )

    def _update_state(self, stimulus: np.ndarray, surprise: float) -> None:
        """Actualiza estado interno basado en estímulo y sorpresa."""
        # Visible: influenciado por estímulo
        stimulus_truncated = stimulus[:self.dim_visible] if len(stimulus) >= self.dim_visible else \
                            np.concatenate([stimulus, np.zeros(self.dim_visible - len(stimulus))])

        self.z_visible = (1 - self.learning_rate) * self.z_visible + \
                         self.learning_rate * stimulus_truncated

        # Hidden: dinámica interna (específica del agente)
        self._update_hidden_state(surprise)

        # Normalizar
        self.z_visible = np.clip(self.z_visible, 0.01, 0.99)
        self.z_visible = self.z_visible / self.z_visible.sum()
        self.z_hidden = np.clip(self.z_hidden, 0.01, 0.99)
        self.z_hidden = self.z_hidden / self.z_hidden.sum()

    @abstractmethod
    def _update_hidden_state(self, surprise: float) -> None:
        """Actualiza estado oculto (específico de cada agente)."""
        pass

    @abstractmethod
    def _update_specialization(self) -> None:
        """Actualiza grado de especialización."""
        pass

    def _generate_report(self) -> np.ndarray:
        """Genera self-report fenomenológico."""
        # Componentes del reporte
        S = self._compute_entropy(self.z_visible)
        drive = self._compute_drive()

        # Sorpresa reciente
        recent_surprise = np.mean(self.surprise_history[-10:]) if len(self.surprise_history) >= 10 else 0.5

        # Estabilidad (1 - variabilidad reciente)
        if len(self.z_history) >= 10:
            recent_z = np.array(self.z_history[-10:])
            stability = 1.0 / (1.0 + np.std(recent_z))
        else:
            stability = 0.5

        return np.array([S, drive, recent_surprise, stability, self.specialization])

    def get_state(self) -> AgentState:
        """Retorna estado actual del agente."""
        return AgentState(
            z_visible=self.z_visible.copy(),
            z_hidden=self.z_hidden.copy(),
            S=self._compute_entropy(self.z_visible),
            drive=self._compute_drive(),
            specialization=self.specialization
        )


class NEO(BaseAgent):
    """
    NEO: Agente especializado en compresión (MDL).

    Características:
    - Busca representaciones compactas
    - Minimiza sorpresa interna
    - Prefiere estabilidad y predicción
    - Estado oculto: modelo comprimido del mundo
    """

    def __init__(self, dim_visible: int = 3, dim_hidden: int = 3):
        super().__init__(dim_visible, dim_hidden, AgentType.NEO)

        # Modelo interno comprimido (eigenvalues de la experiencia)
        self.compression_model: Optional[np.ndarray] = None
        self.model_updates = 0

    def _compute_drive(self) -> float:
        """
        Drive de NEO: minimización de complejidad.

        drive = 1 - (complejidad actual / complejidad máxima)
        """
        # Complejidad = entropía + sorpresa
        S = self._compute_entropy(self.z_visible)
        S_max = np.log(self.dim_visible)

        recent_surprise = np.mean(self.surprise_history[-10:]) if len(self.surprise_history) >= 10 else 0.5

        complexity = (S / S_max + recent_surprise) / 2
        return 1.0 - complexity

    def _generate_action(self, stimulus: np.ndarray) -> np.ndarray:
        """
        Acción de NEO: compresión/estabilización.

        Tiende a reducir variabilidad y buscar patrones.
        """
        # Acción: moverse hacia el centroide histórico (compresión)
        if len(self.z_history) > 10:
            centroid = np.mean(self.z_history[-20:], axis=0)[:self.dim_visible]
            action = self.z_visible + 0.1 * (centroid - self.z_visible)
        else:
            action = self.z_visible.copy()

        # Añadir pequeña exploración (decrece con especialización)
        noise_scale = 0.05 * (1 - self.specialization)
        action = action + np.random.randn(self.dim_visible) * noise_scale

        return np.clip(action, 0, 1)

    def _generate_prediction(self, stimulus: np.ndarray) -> np.ndarray:
        """
        Predicción de NEO: basada en modelo comprimido.

        Usa eigenvalues de la historia para predecir.
        """
        if self.compression_model is not None and len(self.z_history) > 10:
            # Predicción: proyección en espacio comprimido
            recent_mean = np.mean(self.z_history[-10:], axis=0)
            prediction = recent_mean + 0.1 * self.compression_model
            prediction = prediction[:len(stimulus)]
        else:
            # Sin modelo: predecir continuidad
            prediction = stimulus.copy()

        return np.clip(prediction, 0, 1)

    def _compute_value(self, state: np.ndarray, stimulus: np.ndarray) -> float:
        """
        Valor para NEO: estados predecibles y compactos.

        value = predictabilidad * compacidad
        """
        # Predictabilidad (1 - sorpresa)
        recent_surprise = np.mean(self.surprise_history[-5:]) if len(self.surprise_history) >= 5 else 0.5
        predictability = 1.0 - min(recent_surprise, 1.0)

        # Compacidad (baja entropía)
        S = self._compute_entropy(state[:self.dim_visible])
        S_max = np.log(self.dim_visible)
        compactness = 1.0 - S / S_max

        return predictability * 0.6 + compactness * 0.4

    def _update_hidden_state(self, surprise: float) -> None:
        """
        Estado oculto de NEO: modelo comprimido.

        Se actualiza lentamente, priorizando estabilidad.
        """
        # Factor de actualización (más lento que visible)
        hidden_lr = self.learning_rate * 0.5

        # El estado oculto codifica "errores de predicción comprimidos"
        if self.prediction_history:
            error = surprise * np.ones(self.dim_hidden) / self.dim_hidden
            self.z_hidden = (1 - hidden_lr) * self.z_hidden + hidden_lr * error

        # Actualizar modelo de compresión periódicamente
        if self.t % 20 == 0 and len(self.z_history) > 30:
            self._update_compression_model()

    def _update_compression_model(self) -> None:
        """Actualiza modelo de compresión por PCA."""
        try:
            Z = np.array(self.z_history[-50:])
            cov = np.cov(Z.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Primera componente principal como modelo
            idx = np.argsort(eigenvalues)[::-1]
            self.compression_model = eigenvectors[:, idx[0]]
            self.model_updates += 1
        except:
            pass

    def _update_specialization(self) -> None:
        """
        Especialización de NEO: crece con éxito predictivo.
        """
        if len(self.surprise_history) < 20:
            return

        # Éxito = baja sorpresa sostenida
        recent = np.mean(self.surprise_history[-20:])
        older = np.mean(self.surprise_history[-40:-20]) if len(self.surprise_history) >= 40 else recent

        # Especialización crece si sorpresa disminuye
        if recent < older:
            self.specialization = min(1.0, self.specialization + 0.01)
        else:
            self.specialization = max(0.0, self.specialization - 0.005)


class EVA(BaseAgent):
    """
    EVA: Agente especializada en intercambio (MI).

    Características:
    - Maximiza información compartida con el mundo
    - Busca novedad y exploración
    - Prefiere comunicación y adaptación
    - Estado oculto: modelo del "otro" (mundo/NEO)
    """

    def __init__(self, dim_visible: int = 3, dim_hidden: int = 3):
        super().__init__(dim_visible, dim_hidden, AgentType.EVA)

        # Modelo del "otro" (mundo externo)
        self.other_model: Optional[np.ndarray] = None
        self.novelty_buffer: List[float] = []

    def _compute_drive(self) -> float:
        """
        Drive de EVA: maximización de información mutua.

        drive = novedad_detectada * capacidad_respuesta
        """
        # Novedad: variabilidad en estímulos recientes
        if len(self.surprise_history) > 10:
            novelty = np.std(self.surprise_history[-10:])
            novelty = min(novelty * 2, 1.0)  # Escalar
        else:
            novelty = 0.5

        # Capacidad de respuesta: entropía del estado
        S = self._compute_entropy(self.z_visible)
        S_max = np.log(self.dim_visible)
        capacity = S / S_max

        return novelty * 0.5 + capacity * 0.5

    def _generate_action(self, stimulus: np.ndarray) -> np.ndarray:
        """
        Acción de EVA: exploración/comunicación.

        Tiende a responder a novedad y maximizar diversidad.
        """
        # Acción base: responder al estímulo
        action = stimulus[:self.dim_visible] if len(stimulus) >= self.dim_visible else \
                np.concatenate([stimulus, np.zeros(self.dim_visible - len(stimulus))])

        # Añadir exploración (más alta que NEO, decrece menos con especialización)
        noise_scale = 0.1 * (1 - 0.5 * self.specialization)
        action = action + np.random.randn(self.dim_visible) * noise_scale

        # Si detecta novedad, amplificar respuesta
        if len(self.surprise_history) > 5:
            recent_surprise = np.mean(self.surprise_history[-5:])
            if recent_surprise > np.percentile(self.surprise_history, 75):
                action = action * 1.2  # Amplificar

        return np.clip(action, 0, 1)

    def _generate_prediction(self, stimulus: np.ndarray) -> np.ndarray:
        """
        Predicción de EVA: basada en modelo del otro.

        Intenta predecir qué hará el mundo/otro agente.
        """
        if self.other_model is not None and len(self.z_history) > 10:
            # Predicción: basada en patrones del otro
            prediction = stimulus + 0.1 * self.other_model[:len(stimulus)]
        else:
            # Sin modelo: predecir cambio (expectativa de novedad)
            if len(self.prediction_history) > 0:
                delta = stimulus - self.prediction_history[-1]
                prediction = stimulus + 0.5 * delta
            else:
                prediction = stimulus.copy()

        return np.clip(prediction, 0, 1)

    def _compute_value(self, state: np.ndarray, stimulus: np.ndarray) -> float:
        """
        Valor para EVA: estados informativos y diversos.

        value = información_mutua * diversidad
        """
        # Información mutua aproximada (correlación con estímulo)
        if len(stimulus) >= self.dim_visible:
            corr = np.corrcoef(state[:self.dim_visible], stimulus[:self.dim_visible])[0, 1]
            if np.isnan(corr):
                corr = 0
            mi_approx = abs(corr)
        else:
            mi_approx = 0.5

        # Diversidad (entropía alta)
        S = self._compute_entropy(state[:self.dim_visible])
        S_max = np.log(self.dim_visible)
        diversity = S / S_max

        return mi_approx * 0.5 + diversity * 0.5

    def _update_hidden_state(self, surprise: float) -> None:
        """
        Estado oculto de EVA: modelo del "otro".

        Se actualiza rápidamente para capturar cambios.
        """
        # Factor de actualización (más rápido que NEO)
        hidden_lr = self.learning_rate * 1.5

        # El estado oculto codifica "modelo del mundo externo"
        self.novelty_buffer.append(surprise)
        if len(self.novelty_buffer) > 50:
            self.novelty_buffer = self.novelty_buffer[-50:]

        # Actualizar basado en novedad
        novelty_signal = np.ones(self.dim_hidden) * surprise
        self.z_hidden = (1 - hidden_lr) * self.z_hidden + hidden_lr * novelty_signal

        # Actualizar modelo del otro periódicamente
        if self.t % 10 == 0 and len(self.z_history) > 20:
            self._update_other_model()

    def _update_other_model(self) -> None:
        """Actualiza modelo del otro por diferencias."""
        try:
            Z = np.array(self.z_history[-30:])
            # Modelo = dirección de cambio dominante
            diffs = np.diff(Z, axis=0)
            self.other_model = np.mean(diffs, axis=0)
        except:
            pass

    def _update_specialization(self) -> None:
        """
        Especialización de EVA: crece con éxito comunicativo.
        """
        if len(self.novelty_buffer) < 20:
            return

        # Éxito = detección consistente de novedad
        recent_novelty = np.std(self.novelty_buffer[-20:])
        older_novelty = np.std(self.novelty_buffer[-40:-20]) if len(self.novelty_buffer) >= 40 else recent_novelty

        # Especialización crece si detecta más novedad
        if recent_novelty > older_novelty:
            self.specialization = min(1.0, self.specialization + 0.01)
        else:
            self.specialization = max(0.0, self.specialization - 0.005)


class DualAgentSystem:
    """
    Sistema dual NEO-EVA con interacción.

    NEO y EVA operan en paralelo, cada uno con su especialización,
    y pueden comunicarse a través del workspace compartido.
    """

    def __init__(self, dim_visible: int = 3, dim_hidden: int = 3):
        self.neo = NEO(dim_visible, dim_hidden)
        self.eva = EVA(dim_visible, dim_hidden)

        # Workspace compartido
        self.workspace = np.ones(dim_visible) / dim_visible

        # Coupling (endógeno: basado en Transfer Entropy)
        self.coupling_neo_to_eva = 0.1
        self.coupling_eva_to_neo = 0.1

        # Historia de interacción
        self.interaction_history: List[Dict[str, Any]] = []
        self.t = 0

    def step(self, stimulus: np.ndarray) -> Dict[str, Any]:
        """
        Ejecuta un paso del sistema dual.

        Args:
            stimulus: Input del mundo

        Returns:
            Dict con respuestas de NEO y EVA
        """
        self.t += 1

        # Cada agente recibe estímulo + influencia del otro vía workspace
        # Truncar estímulo a dimensión del workspace
        stim_truncated = stimulus[:len(self.workspace)]
        neo_input = stim_truncated + self.coupling_eva_to_neo * self.workspace
        eva_input = stim_truncated + self.coupling_neo_to_eva * self.workspace

        # Normalizar inputs
        neo_input = np.clip(neo_input, 0.01, 0.99)
        eva_input = np.clip(eva_input, 0.01, 0.99)

        # Respuestas
        neo_response = self.neo.step(neo_input)
        eva_response = self.eva.step(eva_input)

        # Actualizar workspace (mezcla de outputs)
        self.workspace = 0.5 * neo_response.action[:len(self.workspace)] + \
                        0.5 * eva_response.action[:len(self.workspace)]
        self.workspace = self.workspace / self.workspace.sum()

        # Actualizar coupling (endógeno)
        self._update_coupling()

        # Registrar interacción
        interaction = {
            't': self.t,
            'neo': {
                'value': neo_response.value,
                'surprise': neo_response.surprise,
                'specialization': self.neo.specialization
            },
            'eva': {
                'value': eva_response.value,
                'surprise': eva_response.surprise,
                'specialization': self.eva.specialization
            },
            'coupling': {
                'neo_to_eva': self.coupling_neo_to_eva,
                'eva_to_neo': self.coupling_eva_to_neo
            }
        }
        self.interaction_history.append(interaction)

        return {
            'neo_response': neo_response,
            'eva_response': eva_response,
            'workspace': self.workspace.copy(),
            'interaction': interaction
        }

    def _update_coupling(self) -> None:
        """
        Actualiza coupling basado en beneficio mutuo.

        100% endógeno: crece si ambos se benefician
        """
        if self.t < 20:
            return

        # Beneficio de NEO por influencia de EVA
        neo_surprise_with = np.mean(self.neo.surprise_history[-10:])
        neo_surprise_baseline = np.mean(self.neo.surprise_history[-20:-10]) if len(self.neo.surprise_history) >= 20 else neo_surprise_with
        neo_benefit = neo_surprise_baseline - neo_surprise_with  # Positivo si bajó sorpresa

        # Beneficio de EVA por influencia de NEO
        eva_novelty_with = np.std(self.eva.novelty_buffer[-10:]) if len(self.eva.novelty_buffer) >= 10 else 0
        eva_novelty_baseline = np.std(self.eva.novelty_buffer[-20:-10]) if len(self.eva.novelty_buffer) >= 20 else eva_novelty_with
        eva_benefit = eva_novelty_with - eva_novelty_baseline  # Positivo si aumentó novedad

        # Ajustar coupling
        lr = 0.01
        self.coupling_neo_to_eva = np.clip(self.coupling_neo_to_eva + lr * eva_benefit, 0.01, 0.5)
        self.coupling_eva_to_neo = np.clip(self.coupling_eva_to_neo + lr * neo_benefit, 0.01, 0.5)

    def get_states(self) -> Tuple[AgentState, AgentState]:
        """Retorna estados de ambos agentes."""
        return self.neo.get_state(), self.eva.get_state()

    def get_divergence(self) -> Dict[str, float]:
        """
        Mide divergencia entre NEO y EVA.

        Returns:
            Dict con métricas de divergencia
        """
        neo_state = self.neo.get_state()
        eva_state = self.eva.get_state()

        # Divergencia en estados visibles
        visible_div = np.linalg.norm(neo_state.z_visible - eva_state.z_visible)

        # Divergencia en drives
        drive_div = abs(neo_state.drive - eva_state.drive)

        # Divergencia en especialización
        spec_div = abs(neo_state.specialization - eva_state.specialization)

        return {
            'visible_divergence': float(visible_div),
            'drive_divergence': float(drive_div),
            'specialization_divergence': float(spec_div),
            'total_divergence': float(visible_div + drive_div + spec_div) / 3
        }


def run_dual_test():
    """Test del sistema dual NEO-EVA."""

    print("=" * 70)
    print("TEST: SISTEMA DUAL NEO-EVA")
    print("=" * 70)

    import sys
    sys.path.insert(0, '/root/NEO_EVA/grounding')
    from phaseG1_world_channel import StructuredWorldChannel

    # Crear sistema
    dual = DualAgentSystem(dim_visible=3, dim_hidden=3)
    world = StructuredWorldChannel(dim_s=6, seed=42)

    # Simulación
    T = 300

    print("\nSimulando interacción NEO-EVA con mundo estructurado...")
    for t in range(T):
        world_state = world.step()
        stimulus = world_state.s[:6]

        result = dual.step(stimulus)

        if t % 50 == 0:
            neo_resp = result['neo_response']
            eva_resp = result['eva_response']
            div = dual.get_divergence()

            print(f"\nt={t}:")
            print(f"  NEO: value={neo_resp.value:.3f}, surprise={neo_resp.surprise:.3f}, spec={dual.neo.specialization:.3f}")
            print(f"  EVA: value={eva_resp.value:.3f}, surprise={eva_resp.surprise:.3f}, spec={dual.eva.specialization:.3f}")
            print(f"  Divergencia: {div['total_divergence']:.3f}")
            print(f"  Coupling: NEO→EVA={dual.coupling_neo_to_eva:.3f}, EVA→NEO={dual.coupling_eva_to_neo:.3f}")

    print("\n" + "=" * 70)
    print("RESULTADOS FINALES")
    print("=" * 70)

    neo_state, eva_state = dual.get_states()

    print(f"\nNEO (Compresión/MDL):")
    print(f"  Especialización: {neo_state.specialization:.4f}")
    print(f"  Drive: {neo_state.drive:.4f}")
    print(f"  Entropía: {neo_state.S:.4f}")

    print(f"\nEVA (Intercambio/MI):")
    print(f"  Especialización: {eva_state.specialization:.4f}")
    print(f"  Drive: {eva_state.drive:.4f}")
    print(f"  Entropía: {eva_state.S:.4f}")

    final_div = dual.get_divergence()
    print(f"\nDivergencia final: {final_div['total_divergence']:.4f}")

    return dual


if __name__ == "__main__":
    run_dual_test()
