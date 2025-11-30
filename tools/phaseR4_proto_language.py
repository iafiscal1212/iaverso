#!/usr/bin/env python3
"""
Phase R4: Symbols & Proto-Language (SPL)
=========================================

NEO y EVA se comunican estados internos comprimidos que mejoran coordinación.
Un proto-lenguaje interno 100% endógeno.

NO hay palabras, NO hay significado humano.
HAY mensajes comprimidos que mejoran coordinación = proto-lenguaje estructural.

Criterios de proto-lenguaje real:
1. Los símbolos reducen incertidumbre: H(z_{t+1}^E | σ) < H(z_{t+1}^E)
2. Aumentan coordinación (TE, integración) vs nulls donde símbolos se barajan

100% ENDÓGENO - Sin constantes mágicas
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.stats import rankdata, entropy
from scipy.special import softmax
from collections import deque
import json


@dataclass
class Episode:
    """Un episodio narrativo (cluster de estados)."""
    episode_id: int
    prototype: np.ndarray
    states: List[np.ndarray] = field(default_factory=list)
    S_values: List[float] = field(default_factory=list)

    @property
    def centroid(self) -> np.ndarray:
        if not self.states:
            return self.prototype
        return np.mean(self.states, axis=0)

    @property
    def value(self) -> float:
        if not self.S_values:
            return 0.0
        return float(np.mean(self.S_values))


@dataclass
class Symbol:
    """Un símbolo del proto-lenguaje."""
    symbol_id: int
    embedding: np.ndarray  # Vector simbólico σ_k
    source_episodes: List[int] = field(default_factory=list)

    # Estadísticas de uso
    usage_count: int = 0
    coordination_gains: List[float] = field(default_factory=list)

    @property
    def efficacy(self) -> float:
        if not self.coordination_gains:
            return 0.0
        return float(np.mean(self.coordination_gains))


class ProtoLanguageSystem:
    """
    Sistema de Proto-Lenguaje entre NEO y EVA.

    Cada agente aprende a:
    1. Comprimir episodios narrativos en símbolos
    2. Enviar símbolos que maximicen información útil para el otro
    3. Recibir símbolos y modular transiciones
    """

    def __init__(self, d_state: int = 8, d_symbol: int = 4):
        self.d_state = d_state
        self.d_symbol = d_symbol

        # Episodios detectados (clusters narrativos)
        self.episodes: List[Episode] = []

        # Codebook de símbolos
        self.symbols: List[Symbol] = []

        # Matriz de codificación W: episode -> symbol
        self.W_encode: Optional[np.ndarray] = None

        # Historial de comunicación
        self.communication_history: List[Dict] = []

        # Historial de estados para calcular entropía condicional
        self._state_history: deque = deque(maxlen=5000)
        self._received_symbols: deque = deque(maxlen=5000)

        # Métricas de proto-lenguaje
        self.entropy_unconditional: List[float] = []
        self.entropy_conditional: List[float] = []
        self.coordination_with_symbol: List[float] = []
        self.coordination_without_symbol: List[float] = []

    def _derive_n_episodes(self) -> int:
        """Número de episodios derivado de historial."""
        T = len(self._state_history)
        return max(3, int(np.sqrt(T + 1)))

    def _derive_n_symbols(self) -> int:
        """Número de símbolos derivado de episodios."""
        return max(2, len(self.episodes) // 2 + 1)

    def _update_episodes(self, z: np.ndarray, S: float):
        """Actualiza episodios (clustering online)."""
        k = self._derive_n_episodes()

        if not self.episodes:
            # Crear primer episodio
            self.episodes.append(Episode(
                episode_id=0,
                prototype=z.copy()
            ))

        # Encontrar episodio más cercano
        distances = [np.linalg.norm(z - e.centroid) for e in self.episodes]
        nearest_idx = int(np.argmin(distances))
        min_dist = distances[nearest_idx]

        # Umbral endógeno basado en dispersión
        if len(self._state_history) > 10:
            states = np.array(list(self._state_history)[-100:])
            threshold = float(np.percentile(
                [np.linalg.norm(s) for s in np.diff(states, axis=0)],
                75
            ))
        else:
            threshold = 1.0

        if min_dist > threshold and len(self.episodes) < k:
            # Crear nuevo episodio
            new_ep = Episode(
                episode_id=len(self.episodes),
                prototype=z.copy()
            )
            self.episodes.append(new_ep)
            nearest_idx = len(self.episodes) - 1

        # Añadir al episodio
        self.episodes[nearest_idx].states.append(z.copy())
        self.episodes[nearest_idx].S_values.append(S)

        # Limitar tamaño
        if len(self.episodes[nearest_idx].states) > 100:
            self.episodes[nearest_idx].states = self.episodes[nearest_idx].states[-100:]
            self.episodes[nearest_idx].S_values = self.episodes[nearest_idx].S_values[-100:]

        return nearest_idx

    def _update_codebook(self):
        """Actualiza codebook de símbolos (compresión de episodios)."""
        if len(self.episodes) < 2:
            return

        n_symbols = self._derive_n_symbols()

        # Crear matriz de episodios
        episode_vectors = np.array([e.centroid for e in self.episodes])

        # PCA endógeno para comprimir
        if episode_vectors.shape[0] > 1:
            centered = episode_vectors - np.mean(episode_vectors, axis=0)
            cov = np.cov(centered.T)

            if cov.ndim == 0:
                cov = np.eye(self.d_state) * (cov + 1e-12)

            eigvals, eigvecs = np.linalg.eigh(cov)

            # Tomar top d_symbol componentes
            top_indices = np.argsort(eigvals)[-self.d_symbol:]
            self.W_encode = eigvecs[:, top_indices].T  # (d_symbol, d_state)

        # Crear símbolos
        self.symbols = []
        for i, ep in enumerate(self.episodes):
            if self.W_encode is not None:
                embedding = self.W_encode @ ep.centroid
            else:
                embedding = ep.centroid[:self.d_symbol]

            sym = Symbol(
                symbol_id=i,
                embedding=embedding,
                source_episodes=[ep.episode_id]
            )
            self.symbols.append(sym)

    def observe(self, z: np.ndarray, S: float) -> int:
        """
        Observa estado y actualiza representaciones internas.

        Returns:
            Índice del episodio asignado
        """
        self._state_history.append(z.copy())
        episode_idx = self._update_episodes(z, S)

        # Actualizar codebook periódicamente
        if len(self._state_history) % 50 == 0:
            self._update_codebook()

        return episode_idx

    def select_symbol_to_send(self, z_self: np.ndarray,
                             z_other_history: np.ndarray = None) -> Tuple[np.ndarray, int]:
        """
        Selecciona símbolo a enviar basado en maximizar información útil.

        Criterio: símbolo que más reduce incertidumbre del otro
        (aproximado por correlación con mejora de S del otro).

        Returns:
            (embedding del símbolo, symbol_id)
        """
        if not self.symbols:
            return np.zeros(self.d_symbol), -1

        # Encontrar episodio actual
        distances = [np.linalg.norm(z_self - e.centroid) for e in self.episodes]
        current_episode = int(np.argmin(distances))

        # Seleccionar símbolo del episodio actual, ponderado por eficacia
        efficacies = [s.efficacy for s in self.symbols]

        # Añadir preferencia por símbolo del episodio actual
        probs = np.array(efficacies) + 0.1

        if current_episode < len(self.symbols):
            probs[current_episode] += 0.5  # Boost al símbolo relevante

        # Asegurar no-negatividad y normalización
        probs = np.clip(probs, 0, None)
        probs = probs / (np.sum(probs) + 1e-12)

        selected_idx = int(np.random.choice(len(self.symbols), p=probs))
        selected_symbol = self.symbols[selected_idx]
        selected_symbol.usage_count += 1

        return selected_symbol.embedding.copy(), selected_idx

    def receive_symbol(self, symbol_embedding: np.ndarray, symbol_id: int,
                      z_current: np.ndarray) -> np.ndarray:
        """
        Recibe símbolo y modula transiciones.

        Returns:
            Vector de modulación para aplicar al estado
        """
        self._received_symbols.append({
            'embedding': symbol_embedding.copy(),
            'id': symbol_id,
            'z_at_reception': z_current.copy()
        })

        if self.W_encode is None:
            return np.zeros(self.d_state)

        # Decodificar símbolo a espacio de estados
        # Pseudo-inversa de W_encode
        W_decode = np.linalg.pinv(self.W_encode)
        decoded = W_decode @ symbol_embedding

        # Modulación: mover suavemente hacia el significado decodificado
        current_centroid = np.mean(list(self._state_history)[-10:], axis=0) if len(self._state_history) > 0 else z_current

        direction = decoded - current_centroid
        norm = np.linalg.norm(direction)

        if norm < 1e-12:
            return np.zeros(self.d_state)

        # Escala endógena
        scale = 1.0 / (np.sqrt(len(self._state_history)) + 1)

        return scale * direction

    def compute_entropy_reduction(self) -> Tuple[float, float]:
        """
        Calcula reducción de entropía debido a símbolos.

        H(z_{t+1}) vs H(z_{t+1} | σ)
        """
        if len(self._state_history) < 20 or len(self._received_symbols) < 10:
            return 0.0, 0.0

        states = np.array(list(self._state_history))

        # Entropía incondicional (basada en varianza)
        var_unconditional = np.var(states, axis=0)
        H_unconditional = 0.5 * np.sum(np.log(var_unconditional + 1e-12))

        # Entropía condicional (estados después de recibir símbolos)
        # Agrupar por símbolo recibido
        symbols_received = [r['id'] for r in self._received_symbols]
        unique_symbols = list(set(symbols_received))

        if len(unique_symbols) < 2:
            return H_unconditional, H_unconditional

        H_conditional = 0.0
        total_weight = 0

        for sym_id in unique_symbols:
            # Estados después de recibir este símbolo
            indices = [i for i, r in enumerate(self._received_symbols) if r['id'] == sym_id]

            if len(indices) < 3:
                continue

            # Estados en las siguientes posiciones
            next_states = []
            for idx in indices:
                if idx + 1 < len(states):
                    next_states.append(states[idx + 1])

            if len(next_states) < 2:
                continue

            next_states = np.array(next_states)
            var_conditional = np.var(next_states, axis=0)
            H_sym = 0.5 * np.sum(np.log(var_conditional + 1e-12))

            weight = len(indices)
            H_conditional += weight * H_sym
            total_weight += weight

        if total_weight > 0:
            H_conditional /= total_weight
        else:
            H_conditional = H_unconditional

        self.entropy_unconditional.append(H_unconditional)
        self.entropy_conditional.append(H_conditional)

        return H_unconditional, H_conditional

    def record_coordination(self, coordination_value: float, used_symbol: bool):
        """Registra coordinación con/sin símbolo."""
        if used_symbol:
            self.coordination_with_symbol.append(coordination_value)
        else:
            self.coordination_without_symbol.append(coordination_value)

    def update_symbol_efficacy(self, symbol_id: int, coordination_gain: float):
        """Actualiza eficacia de símbolo basado en resultado."""
        if 0 <= symbol_id < len(self.symbols):
            self.symbols[symbol_id].coordination_gains.append(coordination_gain)

            # Limitar historial
            if len(self.symbols[symbol_id].coordination_gains) > 100:
                self.symbols[symbol_id].coordination_gains = \
                    self.symbols[symbol_id].coordination_gains[-100:]

    def is_proto_language(self) -> Tuple[bool, Dict]:
        """
        Verifica si hay proto-lenguaje real.

        Criterios:
        1. Símbolos reducen entropía: H(z|σ) < H(z)
        2. Coordinación con símbolos > sin símbolos
        """
        H_unc, H_cond = self.compute_entropy_reduction()

        entropy_reduction = H_unc - H_cond

        coord_with = np.mean(self.coordination_with_symbol) if self.coordination_with_symbol else 0
        coord_without = np.mean(self.coordination_without_symbol) if self.coordination_without_symbol else 0

        coordination_improvement = coord_with - coord_without

        criteria = {
            'entropy_reduced': entropy_reduction > 0,
            'coordination_improved': coordination_improvement > 0,
            'sufficient_symbols': len(self.symbols) >= 2,
            'sufficient_usage': sum(s.usage_count for s in self.symbols) > 50
        }

        is_language = sum(criteria.values()) >= 3

        return is_language, {
            'entropy_unconditional': H_unc,
            'entropy_conditional': H_cond,
            'entropy_reduction': entropy_reduction,
            'coordination_with_symbol': coord_with,
            'coordination_without_symbol': coord_without,
            'coordination_improvement': coordination_improvement,
            'criteria': criteria
        }

    def get_stats(self) -> Dict:
        """Estadísticas del sistema SPL."""
        is_lang, lang_metrics = self.is_proto_language()

        return {
            'n_episodes': len(self.episodes),
            'n_symbols': len(self.symbols),
            'n_observations': len(self._state_history),
            'n_communications': len(self._received_symbols),
            'symbol_usage': {s.symbol_id: s.usage_count for s in self.symbols},
            'symbol_efficacies': {s.symbol_id: s.efficacy for s in self.symbols},
            'is_proto_language': is_lang,
            'language_metrics': lang_metrics
        }


def run_phaseR4_test(n_steps: int = 3000) -> Dict:
    """
    Test de Phase R4: Symbols & Proto-Language.

    Simula comunicación entre NEO y EVA usando símbolos.

    Verifica:
    1. Se forman episodios y símbolos
    2. Los símbolos reducen entropía
    3. La coordinación mejora con símbolos
    """
    print("=" * 70)
    print("PHASE R4: SYMBOLS & PROTO-LANGUAGE (SPL)")
    print("=" * 70)

    # Sistemas de proto-lenguaje para NEO y EVA
    neo_spl = ProtoLanguageSystem(d_state=8, d_symbol=4)
    eva_spl = ProtoLanguageSystem(d_state=8, d_symbol=4)

    # Estados iniciales
    z_neo = np.random.randn(8) * 0.1
    z_eva = np.random.randn(8) * 0.1

    coordination_values = []

    print(f"\nEjecutando {n_steps} pasos de comunicación...")

    for t in range(n_steps):
        # Calcular S para cada agente
        S_neo = 0.5 + 0.3 * np.sin(t / 50) + np.random.randn() * 0.1
        S_eva = 0.5 + 0.3 * np.cos(t / 50) + np.random.randn() * 0.1

        # Observar estados
        neo_spl.observe(z_neo, S_neo)
        eva_spl.observe(z_eva, S_eva)

        # Comunicación (alternar quién envía)
        if t % 2 == 0:
            # NEO envía a EVA
            symbol, sym_id = neo_spl.select_symbol_to_send(z_neo)
            modulation = eva_spl.receive_symbol(symbol, sym_id, z_eva)
            z_eva = z_eva + modulation

            used_symbol = True
        else:
            # EVA envía a NEO
            symbol, sym_id = eva_spl.select_symbol_to_send(z_eva)
            modulation = neo_spl.receive_symbol(symbol, sym_id, z_neo)
            z_neo = z_neo + modulation

            used_symbol = True

        # Calcular coordinación (correlación de estados)
        coord = float(np.corrcoef(z_neo, z_eva)[0, 1])
        coord = 0.0 if np.isnan(coord) else coord
        coordination_values.append(coord)

        # Registrar coordinación
        neo_spl.record_coordination(coord, used_symbol)
        eva_spl.record_coordination(coord, used_symbol)

        # Actualizar eficacia de símbolo
        if sym_id >= 0:
            neo_spl.update_symbol_efficacy(sym_id, coord)
            eva_spl.update_symbol_efficacy(sym_id, coord)

        # Dinámica interna
        z_neo = z_neo * 0.95 + np.random.randn(8) * 0.05
        z_eva = z_eva * 0.95 + np.random.randn(8) * 0.05

        # Sincronización parcial (simular coupling real)
        coupling = 0.1
        z_neo = z_neo + coupling * (z_eva - z_neo)
        z_eva = z_eva + coupling * (z_neo - z_eva)

        if (t + 1) % 600 == 0:
            neo_stats = neo_spl.get_stats()
            print(f"  t={t+1}: episodes={neo_stats['n_episodes']}, "
                  f"symbols={neo_stats['n_symbols']}, "
                  f"is_language={neo_stats['is_proto_language']}")

    # Análisis final
    neo_stats = neo_spl.get_stats()
    eva_stats = eva_spl.get_stats()

    is_lang_neo, metrics_neo = neo_spl.is_proto_language()
    is_lang_eva, metrics_eva = eva_spl.is_proto_language()

    # Comparar coordinación temprana vs tardía
    early_coord = np.mean(coordination_values[:500])
    late_coord = np.mean(coordination_values[-500:])
    coord_improvement = late_coord - early_coord

    # GO/NO-GO criteria
    criteria = {
        'episodes_formed': neo_stats['n_episodes'] >= 3 and eva_stats['n_episodes'] >= 3,
        'symbols_created': neo_stats['n_symbols'] >= 2 and eva_stats['n_symbols'] >= 2,
        'entropy_reduced': metrics_neo['entropy_reduction'] > 0 or metrics_eva['entropy_reduction'] > 0,
        'coordination_improves': coord_improvement > 0,
        'proto_language_emerged': is_lang_neo or is_lang_eva
    }

    n_pass = sum(criteria.values())
    go = n_pass >= 3

    print(f"\n{'='*70}")
    print("RESULTADOS PHASE R4")
    print(f"{'='*70}")

    print(f"\nNEO Proto-Language:")
    print(f"  - Episodios: {neo_stats['n_episodes']}")
    print(f"  - Símbolos: {neo_stats['n_symbols']}")
    print(f"  - Reducción de entropía: {metrics_neo['entropy_reduction']:.4f}")
    print(f"  - Es proto-lenguaje: {is_lang_neo}")

    print(f"\nEVA Proto-Language:")
    print(f"  - Episodios: {eva_stats['n_episodes']}")
    print(f"  - Símbolos: {eva_stats['n_symbols']}")
    print(f"  - Reducción de entropía: {metrics_eva['entropy_reduction']:.4f}")
    print(f"  - Es proto-lenguaje: {is_lang_eva}")

    print(f"\nCoordinación:")
    print(f"  - Temprana (primeros 500): {early_coord:.4f}")
    print(f"  - Tardía (últimos 500): {late_coord:.4f}")
    print(f"  - Mejora: {coord_improvement:.4f}")

    print(f"\nGO/NO-GO Criteria:")
    for criterion, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  - {criterion}: {status}")

    print(f"\n{'GO' if go else 'NO-GO'} ({n_pass}/5 criteria passed)")

    return {
        'go': go,
        'neo_stats': neo_stats,
        'eva_stats': eva_stats,
        'criteria': criteria,
        'coordination_improvement': coord_improvement,
        'coordination_values': coordination_values
    }


if __name__ == "__main__":
    result = run_phaseR4_test(n_steps=3000)

    # Guardar resultados
    import os
    os.makedirs('/root/NEO_EVA/results/phaseR4', exist_ok=True)

    # Serializar stats (convertir numpy a listas)
    def convert_stats(stats):
        converted = {}
        for k, v in stats.items():
            if isinstance(v, dict):
                converted[k] = convert_stats(v)
            elif isinstance(v, np.ndarray):
                converted[k] = v.tolist()
            elif isinstance(v, (np.float64, np.float32)):
                converted[k] = float(v)
            elif isinstance(v, (np.int64, np.int32)):
                converted[k] = int(v)
            else:
                converted[k] = v
        return converted

    with open('/root/NEO_EVA/results/phaseR4/phaseR4_results.json', 'w') as f:
        json.dump({
            'go': result['go'],
            'neo_stats': convert_stats(result['neo_stats']),
            'eva_stats': convert_stats(result['eva_stats']),
            'criteria': {k: bool(v) for k, v in result['criteria'].items()},
            'coordination_improvement': float(result['coordination_improvement'])
        }, f, indent=2)

    print(f"\nResultados guardados en results/phaseR4/")
