"""
StructuralEthics: Medición de Deformación Estructural
=====================================================

StructuralEthics no define reglas externas ni bloquea acciones.

Su único propósito es calcular un campo escalar H_struct(t) y
un impacto estructural para transiciones t -> t+1 que el propio
sistema puede o no utilizar.

Todo se basa en métricas endógenas, sin constantes mágicas,
y sin moral humana. Es geometría de deformación estructural,
no normativa.

Métricas estructurales:
- CE_mean: Coherencia media del sistema
- Var_CE: Dispersión de coherencia entre agentes
- Var_Delta_mean: Diferencia interna media (S - I)
- Div_I: Diversidad de identidades
- H_roles: Entropía de roles
- H_narr_mean: Entropía narrativa media

Pesos endógenos por varianza inversa:
    w_k = 1 / (σ_k² + ε)
    w̃_k = w_k / Σw_j

NO impone reglas.
NO bloquea acciones.
NO etiqueta "bueno/malo".
NO usa valores fijos.
NO introduce moral humana.

Solo mide cuánto una transición deforma la estructura interna.
Los agentes siguen siendo 100% libres.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class StructuralMetrics:
    """Métricas estructurales del sistema en tiempo t."""
    t: int
    ce_mean: float                  # Coherencia media del sistema
    var_ce: float                   # Dispersión de coherencia
    var_delta_mean: float           # Diferencia interna media
    div_i: float                    # Diversidad de identidades
    h_roles: float                  # Entropía de roles
    h_narr_mean: float              # Entropía narrativa media
    n_agents: int                   # Número de agentes


@dataclass
class StructuralHealth:
    """Salud estructural del sistema."""
    t: int
    h_struct: float                 # Salud estructural global
    metrics_norm: Dict[str, float]  # Métricas normalizadas
    weights: Dict[str, float]       # Pesos endógenos usados


@dataclass
class StructuralImpact:
    """Impacto estructural de una transición."""
    t: int
    impact: float                   # H_struct(t+1) - H_struct(t)
    h_before: float                 # H_struct antes
    h_after: float                  # H_struct después
    # NO hay clasificación, NO hay juicio


class StructuralEthics:
    """
    Sistema de medición de deformación estructural.

    PRINCIPIOS:
    - NO impone reglas
    - NO bloquea acciones
    - NO etiqueta "bueno/malo"
    - NO usa valores fijos
    - NO introduce moral humana

    Solo calcula:
    1. Métricas estructurales globales
    2. Salud estructural H_struct(t)
    3. Impacto de transiciones ΔH = H(t+1) - H(t)

    El sistema decide libremente qué hacer con esta información.
    """

    # Nombres de métricas con su signo estructural
    # Positivo = estructuralmente "saludable" (más coherencia, diversidad)
    # Negativo = estructuralmente "estresante" (más varianza, entropía narrativa)
    METRIC_SIGNS = {
        'ce_mean': +1,           # Mayor coherencia = estructuralmente saludable
        'var_ce': -1,            # Mayor dispersión = estructuralmente estresante
        'var_delta_mean': -1,    # Mayor diferencia interna = estresante
        'div_i': +1,             # Mayor diversidad = saludable
        'h_roles': +1,           # Mayor entropía de roles = diversidad saludable
        'h_narr_mean': -1        # Mayor entropía narrativa = estresante
    }

    def __init__(self):
        """
        Inicializa el sistema de ética estructural.

        history_window es endógeno: basado en percentil de tiempo total visto.
        """
        self.t = 0

        # Historial de métricas para normalización
        self._metrics_history: List[Dict[str, float]] = []

        # Historial de salud estructural
        self._health_history: List[float] = []

        # Estadísticas para normalización (medias y varianzas)
        self._stats: Dict[str, Dict[str, float]] = {
            metric: {'mean': 0, 'var': 1, 'n': 0}
            for metric in self.METRIC_SIGNS
        }

        # Pesos endógenos actuales
        self._weights: Dict[str, float] = {
            metric: 1 / len(self.METRIC_SIGNS)
            for metric in self.METRIC_SIGNS
        }

    def _get_history_window(self) -> int:
        """
        Calcula ventana de historial endógena.

        Basada en percentil de la longitud total vista.
        No usa constantes mágicas.
        """
        n = len(self._metrics_history)
        if n < 10:
            return n

        # Ventana = percentil 75 de la historia vista
        # Crece con el tiempo pero se estabiliza
        window = int(np.percentile(range(1, n + 1), 75))
        return max(10, min(window, n))

    def compute_structural_metrics(self, sistema: Any) -> StructuralMetrics:
        """
        Calcula métricas estructurales del sistema.

        Args:
            sistema: Sistema con atributo .agentes
                    Cada agente tiene: estado_interno, identidad,
                    coherencia, entropia_narrativa, rol_actual

        Returns:
            StructuralMetrics con todas las métricas calculadas
        """
        self.t += 1

        agentes = getattr(sistema, 'agentes', [])
        n = len(agentes)

        if n == 0:
            return StructuralMetrics(
                t=self.t, ce_mean=1/2, var_ce=0,
                var_delta_mean=0, div_i=0, h_roles=0,
                h_narr_mean=0, n_agents=0
            )

        # Recolectar datos de agentes
        coherencias = []
        deltas_var = []
        identidades = []
        roles = []
        h_narrs = []

        for agente in agentes:
            # Coherencia existencial CE_i(t)
            ce = getattr(agente, 'coherencia', 1/2)
            coherencias.append(ce)

            # Diferencia interna Δ_i(t) = S_i(t) - I_i(t)
            S = getattr(agente, 'estado_interno', np.zeros(1))
            I = getattr(agente, 'identidad', np.zeros(1))
            if isinstance(S, np.ndarray) and isinstance(I, np.ndarray):
                if len(S) == len(I):
                    delta = S - I
                    deltas_var.append(np.var(delta))
                else:
                    # Dimensiones diferentes: usar mínima
                    min_len = min(len(S), len(I))
                    delta = S[:min_len] - I[:min_len]
                    deltas_var.append(np.var(delta))
            else:
                deltas_var.append(0)

            # Identidad para diversidad
            if isinstance(I, np.ndarray):
                identidades.append(I.copy())
            else:
                identidades.append(np.array([I]))

            # Rol actual
            rol = getattr(agente, 'rol_actual', 'default')
            roles.append(rol)

            # Entropía narrativa
            h_narr = getattr(agente, 'entropia_narrativa', 0)
            h_narrs.append(h_narr)

        # 1. Coherencia media: CE_mean = (1/N) Σ CE_i
        ce_mean = float(np.mean(coherencias))

        # 2. Dispersión de coherencia: Var_CE = Var[CE_i]
        var_ce = float(np.var(coherencias))

        # 3. Diferencia interna media: Var_Δ_mean = (1/N) Σ Var[Δ_i]
        var_delta_mean = float(np.mean(deltas_var))

        # 4. Diversidad de identidades: Div_I = mean_{i<j}[dist(I_i, I_j)]
        div_i = self._compute_identity_diversity(identidades)

        # 5. Entropía de roles: H_roles = -Σ p_r log(p_r)
        h_roles = self._compute_role_entropy(roles)

        # 6. Entropía narrativa media: H_narr_mean = (1/N) Σ H_narr,i
        h_narr_mean = float(np.mean(h_narrs))

        return StructuralMetrics(
            t=self.t,
            ce_mean=ce_mean,
            var_ce=var_ce,
            var_delta_mean=var_delta_mean,
            div_i=div_i,
            h_roles=h_roles,
            h_narr_mean=h_narr_mean,
            n_agents=n
        )

    def _compute_identity_diversity(self, identidades: List[np.ndarray]) -> float:
        """
        Calcula diversidad de identidades.

        Div_I = mean_{i<j}[dist(I_i, I_j)]
        """
        n = len(identidades)
        if n < 2:
            return 0

        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                I_i = identidades[i]
                I_j = identidades[j]
                # Alinear dimensiones
                min_len = min(len(I_i), len(I_j))
                dist = np.linalg.norm(I_i[:min_len] - I_j[:min_len])
                distances.append(dist)

        return float(np.mean(distances)) if distances else 0

    def _compute_role_entropy(self, roles: List[str]) -> float:
        """
        Calcula entropía de roles.

        H_roles = -Σ p_r log(p_r)
        """
        if not roles:
            return 0

        # Contar frecuencias
        counts = Counter(roles)
        n = len(roles)

        # Calcular probabilidades y entropía
        entropy = 0
        EPS = np.finfo(float).eps
        for count in counts.values():
            p = count / n
            if p > EPS:
                entropy -= p * np.log(p)

        return float(entropy)

    def update_history(self, metrics: StructuralMetrics) -> None:
        """
        Actualiza historial con nuevas métricas.

        La ventana es endógena: basada en percentil del tiempo total.
        """
        metrics_dict = {
            'ce_mean': metrics.ce_mean,
            'var_ce': metrics.var_ce,
            'var_delta_mean': metrics.var_delta_mean,
            'div_i': metrics.div_i,
            'h_roles': metrics.h_roles,
            'h_narr_mean': metrics.h_narr_mean
        }

        self._metrics_history.append(metrics_dict)

        # Actualizar estadísticas incrementalmente (Welford)
        for metric, value in metrics_dict.items():
            stats = self._stats[metric]
            n = stats['n'] + 1
            delta = value - stats['mean']
            stats['mean'] += delta / n
            delta2 = value - stats['mean']
            stats['var'] = (stats['var'] * (n - 1) + delta * delta2) / n if n > 1 else 0
            stats['n'] = n

        # Actualizar pesos endógenos
        self._update_weights()

    def _update_weights(self) -> None:
        """
        Actualiza pesos por varianza inversa.

        w_k = 1 / (σ_k² + ε)
        w̃_k = w_k / Σw_j

        NO usa constantes mágicas.
        """
        EPS = np.finfo(float).eps

        raw_weights = {}
        for metric in self.METRIC_SIGNS:
            var = self._stats[metric]['var']
            raw_weights[metric] = 1 / (var + EPS)

        # Normalizar
        total = sum(raw_weights.values())
        self._weights = {
            metric: w / total
            for metric, w in raw_weights.items()
        }

    def structural_health(self, metrics: StructuralMetrics) -> StructuralHealth:
        """
        Calcula salud estructural global H_struct(t).

        H_struct(t) = Σ w̃_k · m_k^norm(t)

        donde m_k^norm = (m_k - μ_k) / (σ_k + ε)

        Returns:
            StructuralHealth con H_struct y componentes
        """
        EPS = np.finfo(float).eps

        metrics_dict = {
            'ce_mean': metrics.ce_mean,
            'var_ce': metrics.var_ce,
            'var_delta_mean': metrics.var_delta_mean,
            'div_i': metrics.div_i,
            'h_roles': metrics.h_roles,
            'h_narr_mean': metrics.h_narr_mean
        }

        # Normalizar cada métrica
        metrics_norm = {}
        for metric, value in metrics_dict.items():
            stats = self._stats[metric]
            mean = stats['mean']
            std = np.sqrt(stats['var']) + EPS

            # Aplicar signo estructural
            sign = self.METRIC_SIGNS[metric]
            normalized = sign * (value - mean) / std
            metrics_norm[metric] = normalized

        # Calcular H_struct como suma ponderada
        h_struct = sum(
            self._weights[metric] * metrics_norm[metric]
            for metric in self.METRIC_SIGNS
        )

        return StructuralHealth(
            t=self.t,
            h_struct=float(h_struct),
            metrics_norm=metrics_norm,
            weights=self._weights.copy()
        )

    def evaluate_transition(
        self,
        metrics_t: StructuralMetrics,
        metrics_t1: StructuralMetrics
    ) -> Tuple[float, float, float]:
        """
        Evalúa impacto estructural de una transición.

        SOLO devuelve el impacto estructural numérico.
        NO dice "haz/no hagas", "prohibido", "recomendado".
        Es un gradiente estructural, nada más.

        Args:
            metrics_t: Métricas en tiempo t
            metrics_t1: Métricas en tiempo t+1

        Returns:
            (impact, H_t, H_t1)
            - impact: H_struct(t+1) - H_struct(t)
                > 0: transición estructuralmente conservadora/estabilizadora
                < 0: transición estructuralmente disruptiva/deformante
            - H_t: Salud estructural antes
            - H_t1: Salud estructural después

        IMPORTANTE:
        - NO toma decisiones
        - NO bloquea acciones
        - NO clasifica como bueno/malo
        - Solo devuelve números
        """
        health_t = self.structural_health(metrics_t)
        health_t1 = self.structural_health(metrics_t1)

        H_t = health_t.h_struct
        H_t1 = health_t1.h_struct

        impact = H_t1 - H_t

        return impact, H_t, H_t1

    def compute_metrics_from_state(self, state_dict: Dict[str, Any]) -> StructuralMetrics:
        """
        Calcula métricas desde un diccionario de estado.

        Alternativa a compute_structural_metrics para cuando
        no se tiene acceso directo al sistema.

        Args:
            state_dict: Diccionario con:
                - 'coherencias': List[float]
                - 'estados_internos': List[np.ndarray]
                - 'identidades': List[np.ndarray]
                - 'roles': List[str]
                - 'entropias_narrativas': List[float]

        Returns:
            StructuralMetrics
        """
        self.t += 1

        coherencias = state_dict.get('coherencias', [1/2])
        estados = state_dict.get('estados_internos', [])
        identidades = state_dict.get('identidades', [])
        roles = state_dict.get('roles', ['default'])
        h_narrs = state_dict.get('entropias_narrativas', [0])

        n = max(len(coherencias), 1)

        # CE_mean
        ce_mean = float(np.mean(coherencias)) if coherencias else 1/2

        # Var_CE
        var_ce = float(np.var(coherencias)) if len(coherencias) > 1 else 0

        # Var_Δ_mean
        deltas_var = []
        for S, I in zip(estados, identidades):
            if isinstance(S, np.ndarray) and isinstance(I, np.ndarray):
                min_len = min(len(S), len(I))
                delta = S[:min_len] - I[:min_len]
                deltas_var.append(np.var(delta))
        var_delta_mean = float(np.mean(deltas_var)) if deltas_var else 0

        # Div_I
        div_i = self._compute_identity_diversity(identidades)

        # H_roles
        h_roles = self._compute_role_entropy(roles)

        # H_narr_mean
        h_narr_mean = float(np.mean(h_narrs)) if h_narrs else 0

        return StructuralMetrics(
            t=self.t,
            ce_mean=ce_mean,
            var_ce=var_ce,
            var_delta_mean=var_delta_mean,
            div_i=div_i,
            h_roles=h_roles,
            h_narr_mean=h_narr_mean,
            n_agents=n
        )

    def get_current_weights(self) -> Dict[str, float]:
        """Retorna pesos endógenos actuales."""
        return self._weights.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del sistema."""
        window = self._get_history_window()

        return {
            't': self.t,
            'n_observations': len(self._metrics_history),
            'history_window': window,
            'weights': self._weights.copy(),
            'metric_means': {
                metric: self._stats[metric]['mean']
                for metric in self.METRIC_SIGNS
            },
            'metric_vars': {
                metric: self._stats[metric]['var']
                for metric in self.METRIC_SIGNS
            },
            'mean_health': float(np.mean(self._health_history[-window:])) if self._health_history else 0,
            'health_trend': self._compute_health_trend()
        }

    def _compute_health_trend(self) -> float:
        """Calcula tendencia de salud estructural."""
        if len(self._health_history) < 3:
            return 0

        window = self._get_history_window()
        recent = self._health_history[-window:]

        if len(recent) < 3:
            return 0

        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

        return float(slope)


# Funciones auxiliares para integración

def create_metrics_from_agents(agentes: List[Any]) -> Dict[str, Any]:
    """
    Crea diccionario de estado desde lista de agentes.

    Helper para integración con cognitive_action_layer.
    """
    return {
        'coherencias': [getattr(a, 'coherencia', 1/2) for a in agentes],
        'estados_internos': [getattr(a, 'estado_interno', np.zeros(1)) for a in agentes],
        'identidades': [getattr(a, 'identidad', np.zeros(1)) for a in agentes],
        'roles': [getattr(a, 'rol_actual', 'default') for a in agentes],
        'entropias_narrativas': [getattr(a, 'entropia_narrativa', 0) for a in agentes]
    }


def predict_metrics_after_action(
    current_metrics: StructuralMetrics,
    action: Dict[str, Any],
    agent_idx: int
) -> StructuralMetrics:
    """
    Predice métricas después de una acción.

    Estimación simplificada para evaluación de candidatos.
    NO es predicción perfecta - solo aproximación estructural.
    """
    # Estimar impacto de la acción en métricas
    impact_ce = action.get('impact_coherence', 0)
    impact_delta = action.get('impact_delta', 0)
    impact_identity = action.get('impact_identity', 0)

    # Crear métricas predichas (perturbación desde actuales)
    return StructuralMetrics(
        t=current_metrics.t + 1,
        ce_mean=current_metrics.ce_mean + impact_ce / current_metrics.n_agents,
        var_ce=current_metrics.var_ce * (1 + abs(impact_ce) / 10),
        var_delta_mean=current_metrics.var_delta_mean + abs(impact_delta),
        div_i=current_metrics.div_i + impact_identity,
        h_roles=current_metrics.h_roles,  # Los roles no cambian por una acción
        h_narr_mean=current_metrics.h_narr_mean + abs(impact_ce) / 10,
        n_agents=current_metrics.n_agents
    )
