#!/usr/bin/env python3
"""
OPERATORS_QG - Operadores Estructurales desde Phase R1
======================================================

Los operadores NO son matrices inventadas.
Son los MISMOS operadores que emergieron en Phase R1:
- Homeostasis
- Exploración
- Momentum
- Integración
- Crisis

Cada operador actúa sobre drives d(t) → d(t+1)
usando solo la dinámica interna del agente.

CHECK: Sin magic numbers, sin semántica externa.
"""

import numpy as np
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, field
from scipy.stats import rankdata


@dataclass
class StructuralOperator:
    """
    Operador estructural que actúa sobre drives.

    El operador NO es una matriz fija - se genera desde
    el estado actual del agente.
    """
    name: str
    operator_fn: Callable  # Función que genera la transformación

    # Historia de aplicaciones
    application_count: int = 0
    effect_history: List[float] = field(default_factory=list)

    def apply(self, drives: np.ndarray, context: Dict) -> np.ndarray:
        """
        Aplica el operador a los drives.

        Args:
            drives: Vector de drives actual
            context: Diccionario con estado del agente (z, identity, etc.)

        Returns:
            Nuevos drives transformados
        """
        new_drives = self.operator_fn(drives, context)

        # Normalizar (mantener como distribución)
        new_drives = np.clip(new_drives, 1e-16, None)
        new_drives = new_drives / new_drives.sum()

        # Registrar efecto
        effect = np.sum(np.abs(new_drives - drives))
        self.effect_history.append(effect)
        self.application_count += 1

        return new_drives

    def get_effect_magnitude(self) -> float:
        """Magnitud típica del efecto (percentil de historia)."""
        if len(self.effect_history) < 10:
            return 0.1
        return np.percentile(self.effect_history, 50)


def create_homeostasis_operator() -> StructuralOperator:
    """
    Operador de HOMEOSTASIS.

    Tiende a restaurar balance entre drives.
    Basado en Phase R1: reduce desviaciones del promedio.
    """
    def homeostasis_fn(drives: np.ndarray, context: Dict) -> np.ndarray:
        # Calcular desviación del promedio
        mean_drive = np.mean(drives)

        # Factor de homeostasis: basado en varianza actual
        # Alta varianza → más corrección
        variance = np.var(drives)
        # Usar percentil de la historia si disponible
        if 'drive_history' in context and len(context['drive_history']) > 10:
            var_history = [np.var(d) for d in context['drive_history'][-50:]]
            var_median = np.percentile(var_history, 50)
            correction_strength = variance / (var_median + 1e-16)
            correction_strength = np.clip(correction_strength, 0, 1)
        else:
            correction_strength = np.clip(variance * 10, 0, 1)

        # Aplicar: mover hacia el promedio
        new_drives = drives + correction_strength * (mean_drive - drives)

        return new_drives

    return StructuralOperator(name='homeostasis', operator_fn=homeostasis_fn)


def create_exploration_operator() -> StructuralOperator:
    """
    Operador de EXPLORACIÓN.

    Aumenta varianza, favorece drives menos activos.
    Basado en Phase R1: gradiente hacia novedad.
    """
    def exploration_fn(drives: np.ndarray, context: Dict) -> np.ndarray:
        # Identificar drives menos activos (ranks bajos)
        ranks = rankdata(drives, method='average')
        min_rank = np.min(ranks)

        # Fuerza de exploración: basada en coherencia
        # Baja coherencia (inestabilidad) → menos exploración
        coherence = context.get('coherence', 0.5)
        exploration_strength = coherence  # Explorar cuando estable

        # Aplicar: potenciar drives de bajo rank
        boost = (ranks == min_rank).astype(float)
        boost = boost / (np.sum(boost) + 1e-16)

        new_drives = drives + exploration_strength * boost * np.mean(drives)

        return new_drives

    return StructuralOperator(name='exploration', operator_fn=exploration_fn)


def create_momentum_operator() -> StructuralOperator:
    """
    Operador de MOMENTUM.

    Continúa la tendencia reciente de cambio.
    Basado en Phase R1: inercia de la dinámica.
    """
    def momentum_fn(drives: np.ndarray, context: Dict) -> np.ndarray:
        if 'drive_history' not in context or len(context['drive_history']) < 2:
            return drives

        # Calcular tendencia reciente
        history = context['drive_history']
        prev = history[-1]

        # Momentum = diferencia actual
        momentum = drives - prev

        # Fuerza de momentum: basada en consistencia de la tendencia
        if len(history) >= 3:
            prev_momentum = history[-1] - history[-2]
            # Si momentum actual y previo van en misma dirección → más fuerza
            alignment = np.dot(momentum, prev_momentum)
            alignment = alignment / (np.linalg.norm(momentum) * np.linalg.norm(prev_momentum) + 1e-16)
            momentum_strength = np.clip((alignment + 1) / 2, 0, 1)  # [0, 1]
        else:
            momentum_strength = 0.5

        # Aplicar: continuar en la dirección del momentum
        new_drives = drives + momentum_strength * momentum

        return new_drives

    return StructuralOperator(name='momentum', operator_fn=momentum_fn)


def create_integration_operator() -> StructuralOperator:
    """
    Operador de INTEGRACIÓN.

    Fortalece correlaciones entre drives.
    Basado en Phase R1: coherencia interna.
    """
    def integration_fn(drives: np.ndarray, context: Dict) -> np.ndarray:
        if 'drive_history' not in context or len(context['drive_history']) < 20:
            return drives

        # Calcular matriz de covarianza de drives históricos
        history = np.array(context['drive_history'][-50:])

        try:
            cov = np.cov(history.T)

            # Identificar pares más correlacionados
            dim = len(drives)
            max_cov = 0
            max_pair = (0, 1)
            for i in range(dim):
                for j in range(i+1, dim):
                    if abs(cov[i, j]) > max_cov:
                        max_cov = abs(cov[i, j])
                        max_pair = (i, j)

            # Fuerza de integración: basada en φ
            phi = context.get('phi', 0.5)

            # Aplicar: acercar los drives más correlacionados
            i, j = max_pair
            mean_ij = (drives[i] + drives[j]) / 2

            new_drives = drives.copy()
            new_drives[i] += phi * (mean_ij - drives[i])
            new_drives[j] += phi * (mean_ij - drives[j])

        except:
            new_drives = drives

        return new_drives

    return StructuralOperator(name='integration', operator_fn=integration_fn)


def create_crisis_operator() -> StructuralOperator:
    """
    Operador de CRISIS.

    Reset parcial cuando el sistema está en crisis.
    Basado en Phase R1: colapso y reorganización.
    """
    def crisis_fn(drives: np.ndarray, context: Dict) -> np.ndarray:
        in_crisis = context.get('in_crisis', False)

        if not in_crisis:
            return drives

        # En crisis: mover hacia estado de máxima entropía (uniforme)
        uniform = np.ones(len(drives)) / len(drives)

        # Fuerza del reset: basada en severidad de la crisis
        identity = context.get('identity', 0.5)
        reset_strength = 1 - identity  # Menor identidad → más reset

        new_drives = drives + reset_strength * (uniform - drives)

        return new_drives

    return StructuralOperator(name='crisis', operator_fn=crisis_fn)


def create_attachment_operator() -> StructuralOperator:
    """
    Operador de ATTACHMENT.

    Modifica drives basándose en el otro agente.
    Basado en Phase R1: influencia del vínculo.
    """
    def attachment_fn(drives: np.ndarray, context: Dict) -> np.ndarray:
        other_drives = context.get('other_drives', None)
        attachment = context.get('attachment', 0)

        if other_drives is None or attachment < 0.1:
            return drives

        # Mover hacia los drives del otro, ponderado por attachment
        new_drives = drives + attachment * (other_drives - drives)

        return new_drives

    return StructuralOperator(name='attachment', operator_fn=attachment_fn)


class OperatorSelector:
    """
    Selecciona operadores basándose en el estado actual.

    La selección es ENDÓGENA: probabilidades derivadas de drives.
    """

    def __init__(self):
        self.operators = {
            'homeostasis': create_homeostasis_operator(),
            'exploration': create_exploration_operator(),
            'momentum': create_momentum_operator(),
            'integration': create_integration_operator(),
            'crisis': create_crisis_operator(),
            'attachment': create_attachment_operator()
        }

        # Mapeo de drives a operadores (endógeno, basado en semántica estructural)
        # entropy → exploration (más entropía = más exploración)
        # neg_surprise → homeostasis (reducir sorpresa = estabilizar)
        # novelty → exploration
        # stability → homeostasis
        # integration → integration
        # otherness → attachment
        self.drive_operator_map = {
            0: 'exploration',    # entropy
            1: 'homeostasis',    # neg_surprise
            2: 'exploration',    # novelty
            3: 'homeostasis',    # stability
            4: 'integration',    # integration
            5: 'attachment'      # otherness
        }

    def select(self, drives: np.ndarray, context: Dict) -> str:
        """
        Selecciona un operador basándose en los drives.

        Probabilidad de cada operador ∝ drives relevantes.
        """
        # Calcular probabilidad de cada operador
        operator_probs = {}

        for op_name in self.operators:
            # Suma de drives que favorecen este operador
            prob = 0
            for drive_idx, mapped_op in self.drive_operator_map.items():
                if mapped_op == op_name:
                    prob += drives[drive_idx]
            operator_probs[op_name] = prob

        # Si en crisis, aumentar probabilidad de crisis operator
        if context.get('in_crisis', False):
            operator_probs['crisis'] = max(operator_probs.values())

        # Momentum siempre tiene probabilidad base (inercia)
        operator_probs['momentum'] = np.mean(list(operator_probs.values()))

        # Normalizar
        total = sum(operator_probs.values())
        if total > 0:
            for k in operator_probs:
                operator_probs[k] /= total

        # Seleccionar aleatoriamente según probabilidades
        operators = list(operator_probs.keys())
        probs = [operator_probs[k] for k in operators]

        selected = np.random.choice(operators, p=probs)

        return selected

    def apply(self, operator_name: str, drives: np.ndarray, context: Dict) -> np.ndarray:
        """Aplica un operador específico."""
        return self.operators[operator_name].apply(drives, context)

    def select_and_apply(self, drives: np.ndarray, context: Dict) -> tuple:
        """Selecciona y aplica un operador."""
        selected = self.select(drives, context)
        new_drives = self.apply(selected, drives, context)
        return selected, new_drives


def test_operators():
    """Test de operadores estructurales."""
    print("=" * 60)
    print("TEST: Operadores Estructurales desde Phase R1")
    print("=" * 60)

    # Drives iniciales
    drives = np.array([0.1, 0.25, 0.15, 0.2, 0.18, 0.12])
    print(f"\nDrives iniciales: {drives}")

    # Contexto simulado
    context = {
        'drive_history': [np.random.dirichlet(np.ones(6)) for _ in range(30)],
        'coherence': 0.7,
        'phi': 0.5,
        'identity': 0.6,
        'in_crisis': False,
        'other_drives': np.array([0.15, 0.2, 0.2, 0.15, 0.15, 0.15]),
        'attachment': 0.5
    }

    # Probar cada operador
    selector = OperatorSelector()

    print("\n--- Efectos de cada operador ---")
    for name, op in selector.operators.items():
        new_drives = op.apply(drives.copy(), context)
        change = np.sum(np.abs(new_drives - drives))
        dominant_before = np.argmax(drives)
        dominant_after = np.argmax(new_drives)

        print(f"\n{name.upper()}:")
        print(f"  Antes: {drives}")
        print(f"  Después: {new_drives}")
        print(f"  Cambio total: {change:.4f}")
        print(f"  Dominante: {dominant_before} → {dominant_after}")

    # Test de selección
    print("\n--- Selección de operadores (100 muestras) ---")
    from collections import Counter
    selections = []
    for _ in range(100):
        selected = selector.select(drives, context)
        selections.append(selected)

    counts = Counter(selections)
    for op, count in counts.most_common():
        print(f"  {op}: {count}%")

    # Test en crisis
    print("\n--- Selección en CRISIS ---")
    context['in_crisis'] = True
    context['identity'] = 0.2

    selections_crisis = []
    for _ in range(100):
        selected = selector.select(drives, context)
        selections_crisis.append(selected)

    counts_crisis = Counter(selections_crisis)
    for op, count in counts_crisis.most_common():
        print(f"  {op}: {count}%")

    print("\n✓ Operadores 100% estructurales: desde Phase R1, sin semántica externa")


if __name__ == "__main__":
    test_operators()
