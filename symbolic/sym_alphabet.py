"""
Symbolic Alphabet: Alfabeto activo de símbolos por agente
=========================================================

Mantiene el conjunto de símbolos activos y sus pesos de activación.
Todo endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import (
    L_t, max_history, adaptive_momentum, compute_adaptive_percentile
)

from symbolic.sym_atoms import Symbol, SymbolExtractor


@dataclass
class SymbolActivation:
    """Activación de un símbolo en el alfabeto."""
    symbol_id: int
    weight: float                    # Peso de activación w_k(t)
    usage_count: int                 # Veces usado
    last_used_t: int                 # Último uso
    value_contribution: float        # Contribución a ΔV histórica
    ema_value: float                 # EMA del valor episódico


class SymbolAlphabet:
    """
    Mantiene el 'alfabeto' de símbolos activos y sus pesos
    para un agente en un tiempo t.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.activations: Dict[int, SymbolActivation] = {}
        self.usage_history: Dict[int, List[int]] = {}  # symbol_id -> [tiempos de uso]
        self.value_history: Dict[int, List[float]] = {}  # symbol_id -> [valores]

        # Históricos globales
        self.all_weights: List[float] = []
        self.all_values: List[float] = []

        self.t = 0

    def update_alphabet(
        self,
        t: int,
        candidate_symbols: List[Symbol],
        value_by_symbol: Dict[int, float],
    ) -> None:
        """
        Actualiza:
        - Inclusión/exclusión en alfabeto según SymScore + uso reciente
        - Pesos de activación w_k(t) (EMA endógena sobre value_by_symbol)
        """
        self.t = t
        window = max_history(t)

        for sym in candidate_symbols:
            sym_id = sym.symbol_id

            # Valor asociado (o 0 si no hay)
            value = value_by_symbol.get(sym_id, 0.0)

            if sym_id not in self.activations:
                # Nuevo símbolo en alfabeto
                self.activations[sym_id] = SymbolActivation(
                    symbol_id=sym_id,
                    weight=sym.stats.sym_score,
                    usage_count=0,
                    last_used_t=t,
                    value_contribution=value,
                    ema_value=value
                )
                self.usage_history[sym_id] = []
                self.value_history[sym_id] = []
            else:
                # Actualizar existente
                act = self.activations[sym_id]

                # EMA del valor
                if self.value_history[sym_id]:
                    momentum = adaptive_momentum(self.value_history[sym_id][-L_t(t):])
                else:
                    momentum = 0.9

                act.ema_value = momentum * act.ema_value + (1 - momentum) * value
                act.value_contribution += value

            # Registrar valor
            self.value_history[sym_id].append(value)
            if len(self.value_history[sym_id]) > window:
                self.value_history[sym_id] = self.value_history[sym_id][-window:]

            # Actualizar peso basado en SymScore y EMA del valor
            act = self.activations[sym_id]
            act.weight = sym.stats.sym_score * (1 + act.ema_value)

            self.all_weights.append(act.weight)
            self.all_values.append(value)

        # Limitar históricos globales
        if len(self.all_weights) > window:
            self.all_weights = self.all_weights[-window:]
            self.all_values = self.all_values[-window:]

        # Eliminar símbolos obsoletos del alfabeto
        self._prune_alphabet(t)

    def record_usage(self, t: int, symbol_ids: List[int]) -> None:
        """Registra que ciertos símbolos se han usado en t."""
        for sym_id in symbol_ids:
            if sym_id in self.activations:
                self.activations[sym_id].usage_count += 1
                self.activations[sym_id].last_used_t = t

                if sym_id not in self.usage_history:
                    self.usage_history[sym_id] = []
                self.usage_history[sym_id].append(t)

                # Limitar historial
                max_h = max_history(t)
                if len(self.usage_history[sym_id]) > max_h:
                    self.usage_history[sym_id] = self.usage_history[sym_id][-max_h:]

    def _prune_alphabet(self, t: int) -> int:
        """Elimina símbolos poco usados o con bajo peso."""
        window = max_history(t)

        # Umbral endógeno de peso
        if self.all_weights:
            weight_threshold = np.percentile(self.all_weights, 10)
        else:
            weight_threshold = 0.0

        to_remove = []
        for sym_id, act in self.activations.items():
            # Eliminar si:
            # 1. Peso muy bajo
            # 2. No usado en mucho tiempo
            # 3. Uso muy bajo
            if act.weight < weight_threshold:
                to_remove.append(sym_id)
            elif t - act.last_used_t > window:
                to_remove.append(sym_id)

        for sym_id in to_remove:
            del self.activations[sym_id]
            if sym_id in self.usage_history:
                del self.usage_history[sym_id]
            if sym_id in self.value_history:
                del self.value_history[sym_id]

        return len(to_remove)

    def get_active_symbols(self, t: int) -> List[int]:
        """Devuelve IDs de símbolos activos (alfabeto A_t)."""
        window = L_t(t) * 2

        active = []
        for sym_id, act in self.activations.items():
            # Activo si usado recientemente o peso alto
            recently_used = t - act.last_used_t <= window

            if self.all_weights:
                high_weight = act.weight >= np.percentile(self.all_weights, 25)
            else:
                high_weight = act.weight > 0

            if recently_used or high_weight:
                active.append(sym_id)

        return active

    def get_symbol_weight(self, symbol_id: int, t: int) -> float:
        """Devuelve w_k(t) para un símbolo."""
        if symbol_id not in self.activations:
            return 0.0
        return self.activations[symbol_id].weight

    def get_top_symbols(self, n: int = 5) -> List[int]:
        """Devuelve los n símbolos con mayor peso."""
        sorted_syms = sorted(
            self.activations.items(),
            key=lambda x: x[1].weight,
            reverse=True
        )
        return [sym_id for sym_id, _ in sorted_syms[:n]]

    def get_symbol_activation(self, symbol_id: int) -> Optional[SymbolActivation]:
        """Obtiene información de activación de un símbolo."""
        return self.activations.get(symbol_id)

    def compute_alphabet_entropy(self) -> float:
        """Calcula entropía del alfabeto (diversidad de uso)."""
        if not self.activations:
            return 0.0

        weights = np.array([act.weight for act in self.activations.values()])
        if weights.sum() <= 0:
            return 0.0

        probs = weights / weights.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs + 1e-10)))

    def get_statistics(self) -> Dict[str, Any]:
        """Estadísticas del alfabeto."""
        active = self.get_active_symbols(self.t)

        if active:
            active_weights = [self.activations[sid].weight for sid in active]
            active_usage = [self.activations[sid].usage_count for sid in active]
        else:
            active_weights = [0]
            active_usage = [0]

        return {
            'agent_id': self.agent_id,
            't': self.t,
            'alphabet_size': len(self.activations),
            'active_size': len(active),
            'mean_weight': float(np.mean(active_weights)),
            'max_weight': float(np.max(active_weights)),
            'total_usage': sum(active_usage),
            'entropy': self.compute_alphabet_entropy()
        }


def test_symbol_alphabet():
    """Test del alfabeto simbólico."""
    print("=" * 60)
    print("TEST: SYMBOL ALPHABET")
    print("=" * 60)

    from symbolic.sym_atoms import SymbolExtractor, to_simplex

    # Crear extractor y alfabeto
    extractor = SymbolExtractor('NEO', state_dim=6)
    alphabet = SymbolAlphabet('NEO')

    np.random.seed(42)

    for t in range(200):
        # Simular estados
        mode = t % 3
        z = np.zeros(6)
        z[mode] = 0.8
        z = to_simplex(np.abs(z + np.random.randn(6) * 0.05) + 0.01)
        phi = np.random.randn(5) * 0.1
        drives = to_simplex(np.random.rand(6) + 0.1)

        extractor.record_state(t, z, phi, drives, context=mode)

        # Extraer símbolos y actualizar alfabeto periódicamente
        if (t + 1) % 20 == 0:
            symbols = extractor.extract_symbols(t)

            # Simular valores asociados a símbolos
            value_by_symbol = {}
            for sym in symbols:
                # Valor basado en SymScore + ruido
                value_by_symbol[sym.symbol_id] = sym.stats.sym_score + np.random.randn() * 0.1

            alphabet.update_alphabet(t, symbols, value_by_symbol)

            # Registrar uso aleatorio
            active = alphabet.get_active_symbols(t)
            if active:
                used = np.random.choice(active, size=min(2, len(active)), replace=False)
                alphabet.record_usage(t, list(used))

        if (t + 1) % 50 == 0:
            stats = alphabet.get_statistics()
            print(f"\n  t={t+1}:")
            print(f"    Tamaño alfabeto: {stats['alphabet_size']}")
            print(f"    Símbolos activos: {stats['active_size']}")
            print(f"    Peso medio: {stats['mean_weight']:.3f}")
            print(f"    Entropía: {stats['entropy']:.3f}")

    print("\n" + "=" * 60)
    print("ALFABETO FINAL")
    print("=" * 60)

    for sym_id in alphabet.get_top_symbols(5):
        act = alphabet.get_symbol_activation(sym_id)
        print(f"\nSímbolo {sym_id}:")
        print(f"  Peso: {act.weight:.3f}")
        print(f"  Usos: {act.usage_count}")
        print(f"  EMA valor: {act.ema_value:.3f}")

    print("\n" + "=" * 60)
    print("TEST COMPLETADO")
    print("=" * 60)

    return alphabet


if __name__ == "__main__":
    test_symbol_alphabet()
