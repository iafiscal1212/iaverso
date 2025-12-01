"""
Symbolic Binding: Combinaciones de símbolos (bigramas, trigramas)
================================================================

Define operaciones de composición entre símbolos para formar "frases".
Solo se guardan combinaciones frecuentes con consecuencia propia.

Todo endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import (
    L_t, max_history, compute_adaptive_percentile, adaptive_momentum
)


@dataclass
class SymbolBinding:
    """Binding entre símbolos (bigrama/trigrama) con consecuencia propia."""
    symbol_ids: Tuple[int, ...]
    count: int
    consequence_signature: np.ndarray     # γ_ij: consecuencia conjunta
    individual_signatures: List[np.ndarray]  # γ_i, γ_j individuales
    consistency: float                    # Cons_ij
    delta_consistency: float              # ΔCons = Cons_ij - max(Cons_i, Cons_j)
    lift: float                          # P(ij) / (P(i)*P(j))
    last_update_t: int
    value_history: List[float] = field(default_factory=list)

    def is_useful(self, lift_threshold: float, delta_threshold: float) -> bool:
        """Determina si el binding es significativo."""
        return self.lift >= lift_threshold and self.delta_consistency > delta_threshold

    def order(self) -> int:
        """Orden del binding (2=bigrama, 3=trigrama, etc.)"""
        return len(self.symbol_ids)


class SymbolBindingManager:
    """
    Detecta y mantiene bindings simbólicos útiles.
    """

    def __init__(self, agent_id: str, max_order: int = 3):
        self.agent_id = agent_id
        self.max_order = max_order

        # Bindings por tupla de IDs
        self.bindings: Dict[Tuple[int, ...], SymbolBinding] = {}

        # Contadores de co-ocurrencia
        self.cooccurrence: Dict[Tuple[int, ...], int] = defaultdict(int)
        self.symbol_counts: Dict[int, int] = defaultdict(int)
        self.total_observations: int = 0

        # Históricos para umbrales endógenos
        self.lift_history: List[float] = []
        self.delta_cons_history: List[float] = []
        self.consequence_history: List[np.ndarray] = []

        self.t = 0
        self.state_dim: int = 0  # Se inicializa con primera observación

    def observe_sequence(
        self,
        t: int,
        symbol_ids: List[int],
        states: List[np.ndarray],
        deltas: List[np.ndarray],
    ) -> None:
        """
        Observa una secuencia de símbolos usados en un episodio/narrativa.
        Actualiza co-ocurrencias, firmas de consecuencia, lift y consistencia.
        """
        self.t = t
        self.total_observations += 1

        if not symbol_ids:
            return

        if deltas and self.state_dim == 0:
            self.state_dim = len(deltas[0])

        # Contar símbolos individuales
        for sym_id in symbol_ids:
            self.symbol_counts[sym_id] += 1

        # Generar n-gramas (2 hasta max_order)
        for order in range(2, min(self.max_order + 1, len(symbol_ids) + 1)):
            for i in range(len(symbol_ids) - order + 1):
                ngram = tuple(symbol_ids[i:i + order])
                self.cooccurrence[ngram] += 1

                # Calcular consecuencia del n-grama
                if deltas and i + order <= len(deltas):
                    ngram_deltas = deltas[i:i + order]
                    ngram_consequence = np.mean(ngram_deltas, axis=0)
                else:
                    ngram_consequence = np.zeros(self.state_dim) if self.state_dim > 0 else np.zeros(1)

                # Actualizar o crear binding
                self._update_binding(t, ngram, ngram_consequence, deltas)

    def _update_binding(
        self,
        t: int,
        ngram: Tuple[int, ...],
        consequence: np.ndarray,
        all_deltas: List[np.ndarray]
    ) -> None:
        """Actualiza o crea un binding."""
        if ngram in self.bindings:
            binding = self.bindings[ngram]
            binding.count = self.cooccurrence[ngram]

            # Actualizar consecuencia con EMA
            if binding.value_history:
                momentum = adaptive_momentum(binding.value_history[-L_t(t):])
            else:
                momentum = 0.9

            binding.consequence_signature = (
                momentum * binding.consequence_signature +
                (1 - momentum) * consequence
            )
            binding.last_update_t = t

        else:
            # Crear nuevo binding
            individual_sigs = []
            for sym_id in ngram:
                # Firma individual aproximada
                if all_deltas:
                    individual_sigs.append(np.mean(all_deltas, axis=0))
                else:
                    individual_sigs.append(np.zeros_like(consequence))

            binding = SymbolBinding(
                symbol_ids=ngram,
                count=self.cooccurrence[ngram],
                consequence_signature=consequence.copy(),
                individual_signatures=individual_sigs,
                consistency=0.5,
                delta_consistency=0.0,
                lift=1.0,
                last_update_t=t
            )
            self.bindings[ngram] = binding

        # Calcular lift y consistencia
        self._compute_binding_metrics(binding)

        # Registrar históricos
        self.lift_history.append(binding.lift)
        self.delta_cons_history.append(binding.delta_consistency)
        self.consequence_history.append(consequence)

        # Limitar históricos
        max_h = max_history(t)
        if len(self.lift_history) > max_h:
            self.lift_history = self.lift_history[-max_h:]
            self.delta_cons_history = self.delta_cons_history[-max_h:]
            self.consequence_history = self.consequence_history[-max_h:]

    def _compute_binding_metrics(self, binding: SymbolBinding) -> None:
        """Calcula lift y consistencia de un binding."""
        # Lift: P(ngram) / prod(P(sym_i))
        if self.total_observations > 0:
            p_ngram = binding.count / self.total_observations

            p_individual = 1.0
            for sym_id in binding.symbol_ids:
                p_sym = self.symbol_counts.get(sym_id, 1) / self.total_observations
                p_individual *= max(p_sym, 1e-10)

            binding.lift = p_ngram / max(p_individual, 1e-10)
        else:
            binding.lift = 1.0

        # Consistencia: basada en varianza de consecuencias
        if self.consequence_history:
            cons_arr = np.array(self.consequence_history[-L_t(self.t):])
            if len(cons_arr) > 1:
                variance = np.mean(np.var(cons_arr, axis=0))
                # p95 de varianzas
                p95_var = np.percentile([np.var(c) for c in self.consequence_history], 95)
                binding.consistency = float(1.0 - variance / (p95_var + 1e-8))
                binding.consistency = np.clip(binding.consistency, 0, 1)

        # Delta consistencia: comparar con individuales
        if binding.individual_signatures:
            individual_consistencies = []
            for sig in binding.individual_signatures:
                if self.consequence_history:
                    diffs = [np.linalg.norm(c - sig) for c in self.consequence_history[-L_t(self.t):]]
                    if diffs:
                        ind_cons = 1.0 / (1 + np.mean(diffs))
                        individual_consistencies.append(ind_cons)

            if individual_consistencies:
                max_individual = max(individual_consistencies)
                binding.delta_consistency = binding.consistency - max_individual

    def get_binding(self, symbol_ids: Tuple[int, ...]) -> Optional[SymbolBinding]:
        """Obtiene un binding específico."""
        return self.bindings.get(symbol_ids)

    def prune_bindings(self, t: int) -> int:
        """Elimina bindings poco frecuentes o poco consistentes."""
        # Umbrales endógenos
        if self.lift_history:
            lift_threshold = np.percentile(self.lift_history, 25)
        else:
            lift_threshold = 0.5

        min_count = L_t(t)

        to_remove = []
        for ngram, binding in self.bindings.items():
            # Eliminar si:
            # 1. Muy poco frecuente
            # 2. Lift muy bajo
            # 3. Muy antiguo sin actualización
            if binding.count < min_count:
                to_remove.append(ngram)
            elif binding.lift < lift_threshold * 0.5:
                to_remove.append(ngram)
            elif t - binding.last_update_t > max_history(t):
                to_remove.append(ngram)

        for ngram in to_remove:
            del self.bindings[ngram]
            if ngram in self.cooccurrence:
                del self.cooccurrence[ngram]

        return len(to_remove)

    def get_useful_bindings(self, t: int) -> List[SymbolBinding]:
        """Devuelve bindings considerados 'frases útiles'."""
        # Umbrales endógenos
        if self.lift_history:
            lift_threshold = np.percentile(self.lift_history, 75)
        else:
            lift_threshold = 1.0

        if self.delta_cons_history:
            delta_threshold = np.percentile(self.delta_cons_history, 50)
        else:
            delta_threshold = 0.0

        useful = []
        for binding in self.bindings.values():
            if binding.is_useful(lift_threshold, delta_threshold):
                useful.append(binding)

        return useful

    def get_bindings_by_order(self, order: int) -> List[SymbolBinding]:
        """Obtiene bindings de un orden específico."""
        return [b for b in self.bindings.values() if b.order() == order]

    def get_bindings_containing(self, symbol_id: int) -> List[SymbolBinding]:
        """Obtiene bindings que contienen un símbolo específico."""
        return [b for b in self.bindings.values() if symbol_id in b.symbol_ids]

    def predict_consequence(self, symbol_ids: Tuple[int, ...]) -> Optional[np.ndarray]:
        """Predice consecuencia de una secuencia de símbolos."""
        binding = self.bindings.get(symbol_ids)
        if binding:
            return binding.consequence_signature.copy()
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Estadísticas del manager de bindings."""
        useful = self.get_useful_bindings(self.t)

        bigrams = self.get_bindings_by_order(2)
        trigrams = self.get_bindings_by_order(3)

        return {
            'agent_id': self.agent_id,
            't': self.t,
            'total_bindings': len(self.bindings),
            'useful_bindings': len(useful),
            'bigrams': len(bigrams),
            'trigrams': len(trigrams),
            'mean_lift': np.mean([b.lift for b in self.bindings.values()]) if self.bindings else 0,
            'mean_delta_cons': np.mean([b.delta_consistency for b in self.bindings.values()]) if self.bindings else 0,
            'total_observations': self.total_observations
        }


def test_binding_manager():
    """Test del manager de bindings."""
    print("=" * 60)
    print("TEST: SYMBOL BINDING MANAGER")
    print("=" * 60)

    manager = SymbolBindingManager('NEO', max_order=3)

    np.random.seed(42)

    # Simular secuencias de símbolos con patrones
    for t in range(200):
        # Crear secuencias con patrones recurrentes
        if t % 3 == 0:
            # Patrón A-B frecuente
            sequence = [0, 1, 2]
        elif t % 3 == 1:
            # Patrón B-C frecuente
            sequence = [1, 2, 0]
        else:
            # Patrón aleatorio
            sequence = list(np.random.randint(0, 4, size=3))

        # Simular estados y deltas
        states = [np.random.randn(6) * 0.3 for _ in sequence]
        deltas = [np.random.randn(6) * 0.1 for _ in sequence]

        manager.observe_sequence(t, sequence, states, deltas)

        if (t + 1) % 50 == 0:
            manager.prune_bindings(t)
            stats = manager.get_statistics()
            print(f"\n  t={t+1}:")
            print(f"    Total bindings: {stats['total_bindings']}")
            print(f"    Útiles: {stats['useful_bindings']}")
            print(f"    Bigramas: {stats['bigrams']}")
            print(f"    Trigramas: {stats['trigrams']}")
            print(f"    Lift medio: {stats['mean_lift']:.3f}")

    print("\n" + "=" * 60)
    print("BINDINGS ÚTILES")
    print("=" * 60)

    useful = manager.get_useful_bindings(manager.t)
    for binding in useful[:10]:
        print(f"\n  {binding.symbol_ids}:")
        print(f"    Count: {binding.count}")
        print(f"    Lift: {binding.lift:.3f}")
        print(f"    Consistencia: {binding.consistency:.3f}")
        print(f"    ΔConsistencia: {binding.delta_consistency:.3f}")

    print("\n" + "=" * 60)
    print("TEST COMPLETADO")
    print("=" * 60)

    return manager


if __name__ == "__main__":
    test_binding_manager()
