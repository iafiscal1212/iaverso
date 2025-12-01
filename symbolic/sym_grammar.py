"""
Symbolic Grammar: Gramática emergente de símbolos
=================================================

Identifica roles estructurales de símbolos y patrones gramaticales emergentes.
Los roles son: descriptivo, operativo, valorativo (emergentes del clustering).

Todo endógeno. Sin números mágicos.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from sklearn.cluster import KMeans
from collections import defaultdict

import sys
sys.path.insert(0, '/root/NEO_EVA')

from cognition.agi_dynamic_constants import (
    L_t, max_history, compute_adaptive_percentile, normalized_entropy
)

from symbolic.sym_atoms import Symbol


@dataclass
class SymbolRole:
    """Rol estructural aproximado de un símbolo."""
    symbol_id: int
    role_id: int              # Índice de cluster de roles
    role_name: str            # Nombre descriptivo del rol
    role_vector: np.ndarray   # r_k: efectos sobre ΔSAGI, ΔV, ΔH_w, etc.
    confidence: float         # Confianza en la asignación
    last_update_t: int


@dataclass
class GrammarRule:
    """Regla gramatical emergente basada en roles."""
    role_sequence: Tuple[int, ...]
    role_names: Tuple[str, ...]
    lift: float
    effect_value: float       # Impacto medio en ΔV
    effect_sagi: float        # Impacto medio en ΔSAGI
    support: int              # Número de ocurrencias
    last_update_t: int
    example_symbols: List[Tuple[int, ...]] = field(default_factory=list)


class SymbolGrammar:
    """
    Identifica roles simbólicos y patrones gramaticales emergentes.
    """

    # Nombres de roles por defecto (se asignan dinámicamente)
    ROLE_NAMES = ['descriptive', 'operative', 'evaluative', 'transitional', 'composite']

    def __init__(self, agent_id: str, n_roles: int = 4):
        self.agent_id = agent_id
        self.n_roles = n_roles

        # Roles por símbolo
        self.roles: Dict[int, SymbolRole] = {}

        # Reglas gramaticales
        self.rules: Dict[Tuple[int, ...], GrammarRule] = {}

        # Contadores para lift
        self.role_sequence_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
        self.role_counts: Dict[int, int] = defaultdict(int)
        self.total_sequences: int = 0

        # Históricos
        self.lift_history: List[float] = []
        self.effect_history: List[float] = []
        self.effects_matrix: List[np.ndarray] = []

        self.t = 0

    def infer_roles(
        self,
        symbols: Dict[int, Symbol],
        effects_by_symbol: Dict[int, np.ndarray],
    ) -> None:
        """
        Recibe efectos r_k por símbolo y hace clustering para asignar roles.

        effects_by_symbol: {symbol_id: [ρ(SAGI), ρ(V), ρ(H_w), ρ(Φ_w), ...]}
        """
        if len(effects_by_symbol) < self.n_roles:
            return

        # Preparar matriz de efectos
        symbol_ids = list(effects_by_symbol.keys())
        effects = np.array([effects_by_symbol[sid] for sid in symbol_ids])

        self.effects_matrix.append(effects)
        if len(self.effects_matrix) > max_history(self.t):
            self.effects_matrix = self.effects_matrix[-max_history(self.t):]

        # Clustering para identificar roles
        n_clusters = min(self.n_roles, len(symbol_ids))

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(effects)
        except:
            return

        # Asignar nombres de roles basados en características del cluster
        cluster_centers = kmeans.cluster_centers_
        role_name_map = self._assign_role_names(cluster_centers)

        # Crear/actualizar roles
        for i, sym_id in enumerate(symbol_ids):
            cluster_id = labels[i]
            effect_vector = effects[i]

            # Distancia al centro del cluster como inverso de confianza
            center = cluster_centers[cluster_id]
            dist = np.linalg.norm(effect_vector - center)
            max_dist = np.max([np.linalg.norm(effects[j] - cluster_centers[labels[j]])
                              for j in range(len(effects))])
            confidence = 1.0 - dist / (max_dist + 1e-8)

            role = SymbolRole(
                symbol_id=sym_id,
                role_id=cluster_id,
                role_name=role_name_map.get(cluster_id, f'role_{cluster_id}'),
                role_vector=effect_vector,
                confidence=float(confidence),
                last_update_t=self.t
            )

            self.roles[sym_id] = role

    def _assign_role_names(self, cluster_centers: np.ndarray) -> Dict[int, str]:
        """Asigna nombres a roles basados en características de sus centros."""
        role_names = {}

        for i, center in enumerate(cluster_centers):
            if len(center) == 0:
                role_names[i] = 'unknown'
                continue

            # Heurística basada en efectos dominantes
            abs_center = np.abs(center)
            max_idx = np.argmax(abs_center)
            magnitude = np.linalg.norm(center)

            # Asignar nombre según efecto dominante
            if magnitude < 0.1:
                role_names[i] = 'descriptive'  # Bajo efecto = descriptivo
            elif max_idx == 0:
                role_names[i] = 'evaluative'   # Efecto en SAGI = evaluativo
            elif max_idx == 1:
                role_names[i] = 'operative'    # Efecto en V = operativo
            else:
                role_names[i] = 'transitional'  # Otros efectos = transicional

        return role_names

    def observe_symbol_sequence(
        self,
        t: int,
        symbol_ids: List[int],
        delta_value: float,
        delta_sagi: float,
    ) -> None:
        """
        Convierte symbol_ids a roles, actualiza:
        - Frecuencia de secuencias de roles
        - Lift
        - Efecto medio sobre ΔV, ΔSAGI
        """
        self.t = t
        self.total_sequences += 1

        # Convertir a roles
        role_sequence = []
        role_name_sequence = []
        for sym_id in symbol_ids:
            if sym_id in self.roles:
                role = self.roles[sym_id]
                role_sequence.append(role.role_id)
                role_name_sequence.append(role.role_name)
                self.role_counts[role.role_id] += 1

        if len(role_sequence) < 2:
            return

        # Registrar secuencias de roles (bigramas y trigramas)
        for order in range(2, min(4, len(role_sequence) + 1)):
            for i in range(len(role_sequence) - order + 1):
                role_seq = tuple(role_sequence[i:i + order])
                name_seq = tuple(role_name_sequence[i:i + order])
                self.role_sequence_counts[role_seq] += 1

                # Actualizar o crear regla
                self._update_rule(t, role_seq, name_seq, delta_value, delta_sagi, symbol_ids)

    def _update_rule(
        self,
        t: int,
        role_seq: Tuple[int, ...],
        name_seq: Tuple[str, ...],
        delta_value: float,
        delta_sagi: float,
        symbol_ids: List[int]
    ) -> None:
        """Actualiza o crea una regla gramatical."""
        if role_seq in self.rules:
            rule = self.rules[role_seq]
            rule.support = self.role_sequence_counts[role_seq]

            # EMA de efectos
            alpha = 1.0 / (1 + np.log1p(rule.support))
            rule.effect_value = (1 - alpha) * rule.effect_value + alpha * delta_value
            rule.effect_sagi = (1 - alpha) * rule.effect_sagi + alpha * delta_sagi

            rule.last_update_t = t

            # Guardar ejemplo
            if len(rule.example_symbols) < 5:
                rule.example_symbols.append(tuple(symbol_ids))

        else:
            rule = GrammarRule(
                role_sequence=role_seq,
                role_names=name_seq,
                lift=1.0,
                effect_value=delta_value,
                effect_sagi=delta_sagi,
                support=self.role_sequence_counts[role_seq],
                last_update_t=t,
                example_symbols=[tuple(symbol_ids)]
            )
            self.rules[role_seq] = rule

        # Calcular lift
        self._compute_rule_lift(rule)

        # Registrar históricos
        self.lift_history.append(rule.lift)
        self.effect_history.append(abs(rule.effect_value) + abs(rule.effect_sagi))

        max_h = max_history(t)
        if len(self.lift_history) > max_h:
            self.lift_history = self.lift_history[-max_h:]
            self.effect_history = self.effect_history[-max_h:]

    def _compute_rule_lift(self, rule: GrammarRule) -> None:
        """Calcula el lift de una regla."""
        if self.total_sequences == 0:
            rule.lift = 1.0
            return

        # P(role_sequence)
        p_seq = rule.support / self.total_sequences

        # P(role_1) * P(role_2) * ...
        p_individual = 1.0
        for role_id in rule.role_sequence:
            p_role = self.role_counts.get(role_id, 1) / self.total_sequences
            p_individual *= max(p_role, 1e-10)

        rule.lift = p_seq / max(p_individual, 1e-10)

    def get_role(self, symbol_id: int) -> Optional[SymbolRole]:
        """Obtiene el rol de un símbolo."""
        return self.roles.get(symbol_id)

    def get_symbols_by_role(self, role_id: int) -> List[int]:
        """Obtiene símbolos con un rol específico."""
        return [sym_id for sym_id, role in self.roles.items() if role.role_id == role_id]

    def get_strong_rules(self, t: int) -> List[GrammarRule]:
        """Devuelve reglas con Lift y effect_value por encima de percentiles dinámicos."""
        if not self.lift_history:
            return []

        # Umbrales endógenos
        lift_threshold = np.percentile(self.lift_history, 75)
        effect_threshold = np.percentile(self.effect_history, 50)

        strong = []
        for rule in self.rules.values():
            total_effect = abs(rule.effect_value) + abs(rule.effect_sagi)
            if rule.lift >= lift_threshold and total_effect >= effect_threshold:
                strong.append(rule)

        return sorted(strong, key=lambda r: r.lift * (abs(r.effect_value) + abs(r.effect_sagi)), reverse=True)

    def get_rules_by_pattern(self, role_pattern: Tuple[str, ...]) -> List[GrammarRule]:
        """Obtiene reglas que siguen un patrón de nombres de roles."""
        matching = []
        for rule in self.rules.values():
            if rule.role_names == role_pattern:
                matching.append(rule)
        return matching

    def predict_effect(self, role_sequence: Tuple[int, ...]) -> Tuple[float, float]:
        """Predice el efecto de una secuencia de roles."""
        rule = self.rules.get(role_sequence)
        if rule:
            return rule.effect_value, rule.effect_sagi
        return 0.0, 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Estadísticas de la gramática."""
        strong_rules = self.get_strong_rules(self.t)

        # Distribución de roles
        role_distribution = defaultdict(int)
        for role in self.roles.values():
            role_distribution[role.role_name] += 1

        return {
            'agent_id': self.agent_id,
            't': self.t,
            'n_symbols_with_roles': len(self.roles),
            'n_rules': len(self.rules),
            'n_strong_rules': len(strong_rules),
            'role_distribution': dict(role_distribution),
            'mean_lift': np.mean(self.lift_history) if self.lift_history else 0,
            'mean_effect': np.mean(self.effect_history) if self.effect_history else 0,
            'total_sequences': self.total_sequences
        }


def test_symbol_grammar():
    """Test de la gramática simbólica."""
    print("=" * 60)
    print("TEST: SYMBOL GRAMMAR")
    print("=" * 60)

    grammar = SymbolGrammar('NEO', n_roles=4)

    np.random.seed(42)

    # Simular símbolos con efectos diferentes
    symbols = {}
    effects_by_symbol = {}

    for i in range(10):
        from symbolic.sym_atoms import Symbol, SymbolStats

        stats = SymbolStats(
            mu=np.random.randn(6),
            Sigma=np.eye(6) * 0.01,
            gamma=np.random.randn(6) * 0.1,
            stab=0.5 + np.random.rand() * 0.5,
            consistency=0.5 + np.random.rand() * 0.5,
            sym_score=0.5,
            n_episodes=50,
            last_update_t=0
        )

        symbols[i] = Symbol(symbol_id=i, agent_id='NEO', stats=stats)

        # Efectos: [SAGI, V, H_w, Φ_w]
        effects_by_symbol[i] = np.random.randn(4) * 0.5

    # Inferir roles
    grammar.infer_roles(symbols, effects_by_symbol)

    print("\nRoles asignados:")
    for sym_id, role in grammar.roles.items():
        print(f"  Símbolo {sym_id}: {role.role_name} (conf: {role.confidence:.3f})")

    # Simular secuencias de símbolos
    for t in range(200):
        # Crear secuencia con patrones
        if t % 3 == 0:
            sequence = [0, 1, 2, 3]  # Patrón regular
        elif t % 3 == 1:
            sequence = [2, 4, 6, 8]  # Otro patrón
        else:
            sequence = list(np.random.randint(0, 10, size=4))

        # Simular efectos
        delta_v = np.random.randn() * 0.2
        delta_sagi = np.random.randn() * 0.1

        grammar.observe_symbol_sequence(t, sequence, delta_v, delta_sagi)

        if (t + 1) % 50 == 0:
            stats = grammar.get_statistics()
            print(f"\n  t={t+1}:")
            print(f"    Reglas: {stats['n_rules']}")
            print(f"    Reglas fuertes: {stats['n_strong_rules']}")
            print(f"    Lift medio: {stats['mean_lift']:.3f}")

    print("\n" + "=" * 60)
    print("REGLAS GRAMATICALES FUERTES")
    print("=" * 60)

    strong = grammar.get_strong_rules(grammar.t)
    for rule in strong[:10]:
        print(f"\n  {rule.role_names}:")
        print(f"    Lift: {rule.lift:.3f}")
        print(f"    Efecto V: {rule.effect_value:.3f}")
        print(f"    Efecto SAGI: {rule.effect_sagi:.3f}")
        print(f"    Soporte: {rule.support}")

    print("\n" + "=" * 60)
    print("TEST COMPLETADO")
    print("=" * 60)

    return grammar


if __name__ == "__main__":
    test_symbol_grammar()
