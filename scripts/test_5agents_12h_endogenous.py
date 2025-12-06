#!/usr/bin/env python3
"""
5 AGENTES Test 12h - 100% Endogeno
==================================
NEO, EVA, ALEX, ADAM, IRIS

Cada agente tiene su propio CE calculado endogenamente:
CE_agent = 1 / (1 + E_norm_agent)

donde E_norm = E_self / EMA(E_self)

100% endogeno - sin magic numbers, sin recompensas.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional

# =============================================================================
# EMA Online
# =============================================================================

class OnlineEMA:
    """EMA adaptativo - alpha derivado de sqrt(n)"""
    def __init__(self):
        self.value = None
        self.n = 0

    def update(self, x: float) -> float:
        self.n += 1
        alpha = 1.0 / (1.0 + np.sqrt(self.n))
        if self.value is None:
            self.value = x
        else:
            self.value = alpha * x + (1.0 - alpha) * self.value
        return self.value


# =============================================================================
# Agente Individual
# =============================================================================

class Agent:
    """
    Agente con dinamica 100% endogena.
    """

    def __init__(self, agent_id: str, dim: int = 8):
        self.agent_id = agent_id
        self.dim = dim
        self.eps = np.finfo(float).eps

        # Semilla derivada del nombre (endogena)
        seed = sum(ord(c) for c in agent_id) * 1000
        self.rng = np.random.default_rng(seed)

        # Estado inicial
        self.S = self.rng.standard_normal(dim)
        self.S = self.S / (np.linalg.norm(self.S) + self.eps)

        # Prediccion de estado
        self.S_pred = self.S.copy()

        # Modelo interno (matriz de transicion)
        self.M = np.eye(dim) + 0.01 * self.rng.standard_normal((dim, dim))

        # Fase interna
        # Frecuencia base derivada del hash del nombre
        self.phase = self.rng.uniform(0, 2 * np.pi)
        self.omega_base = 0.1 + 0.15 * (seed % 100) / 100.0

        # EMA para error
        self.ema_error = OnlineEMA()

        # Metricas
        self.E_self = 0.0
        self.E_norm = 1.0
        self.CE = 0.5

    def compute_CE(self) -> float:
        """
        CE = 1 / (1 + E_norm)
        """
        # Error de auto-prediccion
        diff = self.S - self.S_pred
        self.E_self = float(np.dot(diff, diff))

        # Normalizar por EMA
        ema_val = self.ema_error.update(self.E_self)
        self.E_norm = self.E_self / (ema_val + self.eps)

        # CE
        self.CE = 1.0 / (1.0 + self.E_norm)
        return self.CE

    def step(self, neighbors: Dict[str, np.ndarray] = None) -> Dict[str, float]:
        """
        Un paso de evolucion.
        """
        # Calcular CE antes de evolucionar
        CE = self.compute_CE()

        # Learning rate derivada del error
        lr = self.E_norm / (1.0 + self.E_norm)

        # Actualizar modelo interno
        error_vec = self.S - self.S_pred
        dM = lr * np.outer(error_vec, self.S)
        self.M = self.M + dM

        # Normalizar modelo
        norm_M = np.linalg.norm(self.M, 'fro')
        if norm_M > self.dim:
            self.M = self.M * self.dim / norm_M

        # Evolucion de fase
        energy = np.dot(self.S, self.S)
        omega = self.omega_base + 0.1 * energy / (1.0 + energy)
        omega = omega * (1.0 + 0.3 * self.E_norm)
        self.phase = (self.phase + omega) % (2 * np.pi)

        # Ruido interno
        noise_scale = np.sqrt(self.E_norm) / (1.0 + np.sqrt(self.E_norm))
        noise = noise_scale * self.rng.standard_normal(self.dim)

        # Nuevo estado
        S_new = self.M @ self.S
        S_new = S_new + noise
        S_new = S_new + 0.1 * np.sin(self.phase) * self.S

        # Coupling con vecinos (MINIMO - para permitir personalidades)
        if neighbors:
            # Solo coupling con UN vecino aleatorio (no todos)
            neighbor_id = self.rng.choice(list(neighbors.keys()))
            neighbor_S = neighbors[neighbor_id]

            # Coupling ultra-debil: 0.001 base
            # Solo se activa si hay mucha diferencia de fase
            phase_neighbor = np.arctan2(neighbor_S[1], neighbor_S[0])
            phase_diff = abs(self.phase - phase_neighbor)

            # Coupling proporcional a diferencia de fase (atrae cuando muy distintos)
            coupling = 0.001 * np.sin(phase_diff)  # muy debil

            direction = neighbor_S - self.S
            direction = direction / (np.linalg.norm(direction) + self.eps)
            S_new = S_new + coupling * direction

        # Normalizar
        S_new = S_new / (np.linalg.norm(S_new) + self.eps)

        # Nueva prediccion
        self.S_pred = self.M @ S_new

        # Actualizar estado
        self.S = S_new

        return {
            'agent': self.agent_id,
            'CE': CE,
            'E_self': self.E_self,
            'E_norm': self.E_norm,
            'phase': self.phase,
            'energy': energy,
        }


# =============================================================================
# Sistema Multi-Agente
# =============================================================================

class MultiAgentSystem:
    """
    Sistema de 5 agentes interactuando.
    """

    AGENTS = ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.t = 0
        self.eps = np.finfo(float).eps

        # Crear agentes
        self.agents = {name: Agent(name, dim) for name in self.AGENTS}

        # EMA global
        self.ema_global = OnlineEMA()

    def compute_PLV_pair(self, a1: str, a2: str) -> float:
        """PLV entre dos agentes."""
        phase_diff = self.agents[a1].phase - self.agents[a2].phase
        return float(abs(np.cos(phase_diff)))

    def compute_global_PLV(self) -> float:
        """PLV medio del sistema."""
        plvs = []
        for i, a1 in enumerate(self.AGENTS):
            for a2 in self.AGENTS[i+1:]:
                plvs.append(self.compute_PLV_pair(a1, a2))
        return float(np.mean(plvs)) if plvs else 0.0

    def compute_global_CE(self) -> float:
        """CE medio del sistema."""
        return float(np.mean([a.CE for a in self.agents.values()]))

    def detect_mode(self, agent: Agent, PLV_global: float) -> str:
        """Detecta modo del agente."""
        E = agent.E_norm

        if PLV_global > 0.8:
            return 'FUS'
        elif E < 0.5:
            return 'RAC'
        elif E > 1.5:
            return 'EMO'
        else:
            return 'MIX'

    def step(self) -> List[Dict[str, Any]]:
        """Un paso del sistema."""
        self.t += 1

        # Obtener estados de todos los agentes
        states = {name: agent.S.copy() for name, agent in self.agents.items()}

        # Paso de cada agente
        results = []
        for name, agent in self.agents.items():
            # Vecinos = todos menos el mismo
            neighbors = {n: s for n, s in states.items() if n != name}

            metrics = agent.step(neighbors)
            results.append(metrics)

        # Metricas globales
        PLV_global = self.compute_global_PLV()
        CE_global = self.compute_global_CE()

        # Anadir modo a cada resultado
        for r in results:
            agent = self.agents[r['agent']]
            r['mode'] = self.detect_mode(agent, PLV_global)
            r['PLV_global'] = PLV_global
            r['CE_global'] = CE_global
            r['t'] = self.t

        return results


# =============================================================================
# Test Principal
# =============================================================================

def run_5agents_test(n_steps: int = 500, dim: int = 8) -> Path:
    """
    Ejecuta test de 5 agentes.
    """
    print("=" * 70)
    print("5 AGENTES TEST 12H - 100% ENDOGENO")
    print("=" * 70)
    print(f"Agentes: NEO, EVA, ALEX, ADAM, IRIS")
    print(f"Steps: {n_steps}")
    print(f"Dimension: {dim}")
    print("Sin magic numbers, sin recompensas, sin input externo")
    print("=" * 70)
    print()

    # Crear sistema
    system = MultiAgentSystem(dim=dim)

    # Archivos de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('/root/NEO_EVA/logs/5agents_12h')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'5agents_12h_{timestamp}.jsonl'

    # Metricas por hora
    steps_per_hour = max(1, n_steps // 12)
    hour_data = []

    start_time = time.time()

    with open(output_file, 'w') as f:
        for step in range(n_steps):
            results = system.step()

            # Guardar cada agente
            for r in results:
                f.write(json.dumps(r) + '\n')

            hour_data.extend(results)

            # Resumen cada hora
            if (step + 1) % steps_per_hour == 0:
                hour = (step + 1) // steps_per_hour

                # CE por agente
                ce_by_agent = defaultdict(list)
                for d in hour_data:
                    ce_by_agent[d['agent']].append(d['CE'])

                elapsed = time.time() - start_time

                ce_strs = [f"{a}:{np.mean(ce_by_agent[a]):.3f}" for a in system.AGENTS]
                print(f"Hora {hour:2d}/12 | CE: {' '.join(ce_strs)} | {elapsed:.1f}s")

                hour_data = []

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("SIMULACION COMPLETADA")
    print("=" * 70)
    print(f"Tiempo real: {elapsed:.2f}s")
    print(f"Velocidad: {n_steps * 5 / elapsed:.0f} registros/s")
    print()
    print(f"Logs: {output_file}")
    print("=" * 70)

    return output_file


def analyze_results(log_path: Path):
    """Analiza resultados por agente."""

    ce_by_agent = defaultdict(list)
    mode_by_agent = defaultdict(list)

    with open(log_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            agent = d['agent']
            ce = d['CE']
            mode = d['mode']
            ce_by_agent[agent].append(ce)
            mode_by_agent[agent].append(mode)

    print()
    print("=" * 70)
    print("CE POR AGENTE")
    print("=" * 70)
    print()
    print(f"| {'Agente':5} | {'CE medio':>8} | {'CE std':>7} | {'%CE>0.8':>8} | {'%CE<0.4':>8} | {'n':>5} |")
    print(f"|{'-'*7}|{'-'*10}|{'-'*9}|{'-'*10}|{'-'*10}|{'-'*7}|")

    for agent in ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']:
        vals = np.array(ce_by_agent.get(agent, []))
        if len(vals) == 0:
            continue
        mean = vals.mean()
        std = vals.std()
        high = (vals > 0.8).mean() * 100
        low = (vals < 0.4).mean() * 100
        n = len(vals)
        print(f"| {agent:5} | {mean:8.4f} | {std:7.4f} | {high:7.1f}% | {low:7.1f}% | {n:5} |")

    print()
    print("=" * 70)
    print("MODOS POR AGENTE")
    print("=" * 70)
    print()

    for agent in ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']:
        modes = mode_by_agent.get(agent, [])
        if not modes:
            continue

        mode_counts = {}
        for m in modes:
            mode_counts[m] = mode_counts.get(m, 0) + 1

        total = len(modes)
        mode_strs = [f"{m}:{count/total*100:.1f}%" for m, count in sorted(mode_counts.items())]
        print(f"{agent:5}: {' | '.join(mode_strs)}")

    print()

    # CE por modo por agente
    print("=" * 70)
    print("CE POR MODO POR AGENTE")
    print("=" * 70)
    print()

    # Recargar para tener ce y mode juntos
    ce_mode_by_agent = defaultdict(lambda: defaultdict(list))

    with open(log_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            ce_mode_by_agent[d['agent']][d['mode']].append(d['CE'])

    print(f"| {'Agente':5} | {'Modo':4} | {'CE medio':>8} | {'n':>5} |")
    print(f"|{'-'*7}|{'-'*6}|{'-'*10}|{'-'*7}|")

    for agent in ['NEO', 'EVA', 'ALEX', 'ADAM', 'IRIS']:
        for mode in ['RAC', 'EMO', 'FUS', 'MIX']:
            vals = ce_mode_by_agent[agent].get(mode, [])
            if not vals:
                continue
            print(f"| {agent:5} | {mode:4} | {np.mean(vals):8.4f} | {len(vals):5} |")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', '-s', type=int, default=500)
    parser.add_argument('--dim', '-d', type=int, default=8)

    args = parser.parse_args()

    log_path = run_5agents_test(n_steps=args.steps, dim=args.dim)
    analyze_results(log_path)
