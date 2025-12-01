"""
Soft Hook: Modulación Narrativa Suave
=====================================

El tipo de episodio modula SUAVEMENTE (sin saltos):
- Learning rate interno
- Temperatura cognitiva β_t
- Propensión a formar conceptos
- Balance planificación vs reactividad
- Sensibilidad a sorpresas

Clasificación endógena de episodios:
- crisis: φ↓, id↓
- exploración: φ↑, id↓
- consolidación: φ↓, id↑
- flow: φ↑, id↑
- transición: valores intermedios

Todo continuo, sin "modos" duros.
100% endógeno.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum


class EpisodeRegion(Enum):
    """Regiones en el espacio (φ, id) - solo para logging."""
    CRISIS = "crisis"
    EXPLORATION = "exploration"
    CONSOLIDATION = "consolidation"
    FLOW = "flow"
    TRANSITION = "transition"


@dataclass
class EpisodeCharacterization:
    """Caracterización fenomenológica de un episodio."""
    phi_norm: float      # φ normalizado [0,1]
    id_norm: float       # identidad normalizada [0,1]
    delta_S: float       # cambio en S (integración)
    delta_V: float       # cambio en valor
    crisis_prob: float   # probabilidad de crisis

    # Factores de modulación (calculados)
    learning_factor: float = 1.0
    temperature_factor: float = 1.0
    concept_factor: float = 1.0
    planning_prob: float = 0.5
    surprise_sensitivity: float = 1.0

    # Región (para debug/logging)
    region: EpisodeRegion = EpisodeRegion.TRANSITION


class SoftHook:
    """
    Gancho suave que modula comportamiento según tipo de episodio.

    Todo es CONTINUO - no hay switches ni modos discretos.
    Los factores varían suavemente según posición en espacio (φ, id).
    """

    def __init__(self):
        """Inicializa Soft Hook."""
        # Historiales para normalización endógena
        self.phi_history: list = []
        self.id_history: list = []
        self.delta_S_history: list = []
        self.delta_V_history: list = []

        self.t = 0

    def _normalize_to_history(self, value: float, history: list) -> float:
        """
        Normaliza valor usando rank en historial.

        rank(value) = percentil en historial
        """
        if len(history) < 10:
            return 0.5

        rank = np.sum(np.array(history) <= value) / len(history)
        return float(rank)

    def _compute_region_coordinates(self, phi_norm: float, id_norm: float) -> Tuple[float, float, float, float]:
        """
        Calcula coordenadas suaves en cada región.

        Cada región tiene un "peso" basado en distancia al centro.
        Los pesos suman ~1 (normalizado por softmax).

        Centros:
        - crisis: (0.2, 0.2)
        - exploration: (0.8, 0.2)
        - consolidation: (0.2, 0.8)
        - flow: (0.8, 0.8)
        """
        # Centros de cada región
        centers = {
            'crisis': (0.2, 0.2),
            'exploration': (0.8, 0.2),
            'consolidation': (0.2, 0.8),
            'flow': (0.8, 0.8)
        }

        # Distancias a cada centro
        distances = {}
        for name, (cx, cy) in centers.items():
            d = np.sqrt((phi_norm - cx)**2 + (id_norm - cy)**2)
            distances[name] = d

        # Convertir distancias a pesos (inverso de distancia, softmax)
        # Temperatura basada en varianza de distancias
        dist_values = list(distances.values())
        sigma = np.std(dist_values) + 0.1
        beta = 1.0 / sigma

        weights = {}
        for name, d in distances.items():
            weights[name] = np.exp(-beta * d)

        total = sum(weights.values())
        for name in weights:
            weights[name] /= total

        return weights['crisis'], weights['exploration'], weights['consolidation'], weights['flow']

    def characterize_episode(self, phi: float, identity: float,
                            delta_S: float, delta_V: float,
                            crisis_prob: float) -> EpisodeCharacterization:
        """
        Caracteriza un episodio y calcula factores de modulación.

        Args:
            phi: Valor fenomenológico (magnitud)
            identity: Fuerza de identidad
            delta_S: Cambio en integración
            delta_V: Cambio en valor interno
            crisis_prob: Probabilidad de estar en crisis

        Returns:
            EpisodeCharacterization con todos los factores
        """
        self.t += 1

        # Registrar en historiales
        self.phi_history.append(phi)
        self.id_history.append(identity)
        self.delta_S_history.append(delta_S)
        self.delta_V_history.append(delta_V)

        # Limitar historiales
        max_hist = 500
        if len(self.phi_history) > max_hist:
            self.phi_history = self.phi_history[-max_hist:]
            self.id_history = self.id_history[-max_hist:]
            self.delta_S_history = self.delta_S_history[-max_hist:]
            self.delta_V_history = self.delta_V_history[-max_hist:]

        # Normalizar usando ranks
        phi_norm = self._normalize_to_history(phi, self.phi_history)
        id_norm = self._normalize_to_history(identity, self.id_history)
        delta_S_norm = self._normalize_to_history(abs(delta_S),
                                                  [abs(x) for x in self.delta_S_history])

        # Obtener pesos de cada región
        w_crisis, w_expl, w_consol, w_flow = self._compute_region_coordinates(phi_norm, id_norm)

        # Calcular factores de modulación como mezcla continua
        char = EpisodeCharacterization(
            phi_norm=phi_norm,
            id_norm=id_norm,
            delta_S=delta_S,
            delta_V=delta_V,
            crisis_prob=crisis_prob
        )

        # 1. Learning rate factor
        # crisis: 0.4 + 0.6*u, exploration: 1.2 + 0.5*u, consolidation: 0.8, flow: 1.4
        u = delta_S_norm  # intensidad del episodio
        f_crisis = 0.4 + 0.6 * u
        f_expl = 1.2 + 0.5 * u
        f_consol = 0.8
        f_flow = 1.4

        char.learning_factor = (w_crisis * f_crisis +
                               w_expl * f_expl +
                               w_consol * f_consol +
                               w_flow * f_flow)

        # 2. Temperature factor (para softmax/decisiones)
        # crisis: 1.3 (más rígido), exploration: 0.7 (más aleatorio)
        # consolidation: 1.1, flow: 0.5 (creatividad máxima)
        g_crisis = 1.3
        g_expl = 0.7
        g_consol = 1.1
        g_flow = 0.5

        char.temperature_factor = (w_crisis * g_crisis +
                                  w_expl * g_expl +
                                  w_consol * g_consol +
                                  w_flow * g_flow)

        # 3. Concept formation factor (threshold multiplier)
        # exploration: 0.7 (más fácil), consolidation: 1.3 (más difícil)
        # flow: 0.8, crisis: 1.5 (casi imposible)
        h_crisis = 1.5
        h_expl = 0.7
        h_consol = 1.3
        h_flow = 0.8

        char.concept_factor = (w_crisis * h_crisis +
                              w_expl * h_expl +
                              w_consol * h_consol +
                              w_flow * h_flow)

        # 4. Planning probability
        # P(plan) = sigmoid(α*φ + γ*id - δ*crisis)
        # donde α, γ, δ son ranks
        alpha = phi_norm
        gamma = id_norm
        delta = crisis_prob

        score = alpha + gamma - delta
        char.planning_prob = 1.0 / (1.0 + np.exp(-score))

        # 5. Surprise sensitivity
        # En flow: alta sensibilidad (aprende de todo)
        # En crisis: baja sensibilidad (protección)
        s_crisis = 0.5
        s_expl = 1.0
        s_consol = 0.8
        s_flow = 1.3

        char.surprise_sensitivity = (w_crisis * s_crisis +
                                    w_expl * s_expl +
                                    w_consol * s_consol +
                                    w_flow * s_flow)

        # Determinar región dominante (para logging)
        weights = {'crisis': w_crisis, 'exploration': w_expl,
                  'consolidation': w_consol, 'flow': w_flow}
        dominant = max(weights, key=weights.get)
        char.region = EpisodeRegion(dominant)

        return char

    def modulate_learning_rate(self, base_eta: float, char: EpisodeCharacterization) -> float:
        """
        Modula learning rate base.

        η'_t = η_t * f(E_i)
        """
        return base_eta * char.learning_factor

    def modulate_temperature(self, base_beta: float, char: EpisodeCharacterization) -> float:
        """
        Modula temperatura cognitiva.

        β'_t = β_t * g(E_i)
        """
        return base_beta * char.temperature_factor

    def modulate_concept_threshold(self, base_threshold: float,
                                   char: EpisodeCharacterization) -> float:
        """
        Modula umbral de formación de conceptos.

        θ_i = P95(d_M) * h(E_i)
        """
        return base_threshold * char.concept_factor

    def should_plan(self, char: EpisodeCharacterization) -> bool:
        """
        Decide si planificar o reaccionar.

        Usa planning_prob como probabilidad.
        """
        return np.random.random() < char.planning_prob

    def modulate_surprise_response(self, base_response: float,
                                   char: EpisodeCharacterization) -> float:
        """
        Modula respuesta a sorpresas.
        """
        return base_response * char.surprise_sensitivity

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del hook."""
        if self.t == 0:
            return {'status': 'not_initialized'}

        return {
            't': self.t,
            'mean_phi': float(np.mean(self.phi_history[-50:])) if self.phi_history else 0,
            'mean_id': float(np.mean(self.id_history[-50:])) if self.id_history else 0,
            'phi_variance': float(np.var(self.phi_history[-50:])) if self.phi_history else 0,
            'id_variance': float(np.var(self.id_history[-50:])) if self.id_history else 0
        }


class DifferentiatedSoftHook(SoftHook):
    """
    Soft Hook con diferenciación por personalidad.

    Cada agente tiene sesgos únicos que modifican los factores base.
    """

    def __init__(self, agent_name: str, personality_seed: Optional[int] = None):
        """
        Inicializa con personalidad diferenciada.

        Args:
            agent_name: Nombre del agente
            personality_seed: Semilla para generar personalidad única
        """
        super().__init__()
        self.agent_name = agent_name

        # Generar personalidad única basada en nombre
        if personality_seed is None:
            personality_seed = hash(agent_name) % 10000

        # Usar RandomState local para no afectar estado global
        rng = np.random.RandomState(personality_seed)

        # Sesgos de personalidad (±25% del valor base para más variación)
        self.learning_bias = 1.0 + (rng.random() - 0.5) * 0.5
        self.temperature_bias = 1.0 + (rng.random() - 0.5) * 0.5
        self.concept_bias = 1.0 + (rng.random() - 0.5) * 0.5
        self.planning_bias = (rng.random() - 0.5) * 0.4  # Sesgo aditivo
        self.surprise_bias = 1.0 + (rng.random() - 0.5) * 0.5

        # Preferencia por regiones (qué tipo de episodio "prefiere" el agente)
        self.region_affinity = {
            'crisis': rng.random() * 0.3,  # Nadie "prefiere" crisis
            'exploration': rng.random(),
            'consolidation': rng.random(),
            'flow': rng.random()
        }

        # Normalizar afinidades
        total = sum(self.region_affinity.values())
        for k in self.region_affinity:
            self.region_affinity[k] /= total

    def characterize_episode(self, phi: float, identity: float,
                            delta_S: float, delta_V: float,
                            crisis_prob: float) -> EpisodeCharacterization:
        """
        Caracteriza con sesgos de personalidad.
        """
        # Obtener caracterización base
        char = super().characterize_episode(phi, identity, delta_S, delta_V, crisis_prob)

        # Aplicar sesgos de personalidad
        char.learning_factor *= self.learning_bias
        char.temperature_factor *= self.temperature_bias
        char.concept_factor *= self.concept_bias
        char.planning_prob = np.clip(char.planning_prob + self.planning_bias, 0.1, 0.9)
        char.surprise_sensitivity *= self.surprise_bias

        return char

    def get_personality_profile(self) -> Dict:
        """Obtiene perfil de personalidad."""
        return {
            'agent': self.agent_name,
            'learning_bias': self.learning_bias,
            'temperature_bias': self.temperature_bias,
            'concept_bias': self.concept_bias,
            'planning_bias': self.planning_bias,
            'surprise_bias': self.surprise_bias,
            'region_affinity': self.region_affinity
        }


def test_soft_hook():
    """Test del Soft Hook."""
    print("=" * 60)
    print("TEST SOFT HOOK - MODULACIÓN NARRATIVA")
    print("=" * 60)

    # Crear hooks diferenciados para cada agente
    agents = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']
    hooks = {name: DifferentiatedSoftHook(name) for name in agents}

    print("\nPerfiles de personalidad:")
    for name, hook in hooks.items():
        profile = hook.get_personality_profile()
        print(f"\n  {name}:")
        print(f"    Learning bias: {profile['learning_bias']:.3f}")
        print(f"    Temperature bias: {profile['temperature_bias']:.3f}")
        print(f"    Planning bias: {profile['planning_bias']:.3f}")
        print(f"    Exploration affinity: {profile['region_affinity']['exploration']:.3f}")
        print(f"    Consolidation affinity: {profile['region_affinity']['consolidation']:.3f}")

    print(f"\nSimulando 200 pasos con diferentes estados...")

    # Simular diferentes situaciones
    results = {name: {'regions': [], 'factors': []} for name in agents}

    for t in range(200):
        # Estado base que evoluciona
        base_phi = 0.3 + 0.4 * np.sin(t / 30) + np.random.randn() * 0.1
        base_id = 0.4 + 0.3 * np.cos(t / 40) + np.random.randn() * 0.1

        for name, hook in hooks.items():
            # Cada agente tiene ligera variación
            phi = base_phi + np.random.randn() * 0.15
            identity = base_id + np.random.randn() * 0.15
            delta_S = np.random.randn() * 0.1
            delta_V = np.random.randn() * 0.05
            crisis_prob = max(0, 0.1 + np.random.randn() * 0.1)

            char = hook.characterize_episode(phi, identity, delta_S, delta_V, crisis_prob)

            results[name]['regions'].append(char.region.value)
            results[name]['factors'].append({
                'learning': char.learning_factor,
                'temperature': char.temperature_factor,
                'planning': char.planning_prob
            })

    # Mostrar resultados
    print("\n" + "=" * 60)
    print("RESULTADOS - DIFERENCIACIÓN ENTRE AGENTES")
    print("=" * 60)

    for name in agents:
        regions = results[name]['regions']
        factors = results[name]['factors']

        # Contar regiones
        region_counts = {}
        for r in regions:
            region_counts[r] = region_counts.get(r, 0) + 1

        # Promedios de factores
        mean_learning = np.mean([f['learning'] for f in factors])
        mean_temp = np.mean([f['temperature'] for f in factors])
        mean_planning = np.mean([f['planning'] for f in factors])

        print(f"\n  {name}:")
        print(f"    Regiones visitadas: {region_counts}")
        print(f"    Mean learning factor: {mean_learning:.3f}")
        print(f"    Mean temperature factor: {mean_temp:.3f}")
        print(f"    Mean planning prob: {mean_planning:.3f}")

    # Verificar diferenciación
    print("\n" + "=" * 60)
    print("VERIFICACIÓN DE DIFERENCIACIÓN")
    print("=" * 60)

    learning_factors = [np.mean([f['learning'] for f in results[n]['factors']]) for n in agents]
    planning_probs = [np.mean([f['planning'] for f in results[n]['factors']]) for n in agents]

    learning_variance = np.var(learning_factors)
    planning_variance = np.var(planning_probs)

    print(f"\n  Varianza entre agentes (learning): {learning_variance:.4f}")
    print(f"  Varianza entre agentes (planning): {planning_variance:.4f}")

    if learning_variance > 0.01 and planning_variance > 0.001:
        print("\n  ✓ Los agentes están diferenciados correctamente")
    else:
        print("\n  ⚠️ Poca diferenciación - ajustar sesgos")

    return hooks, results


if __name__ == "__main__":
    test_soft_hook()
