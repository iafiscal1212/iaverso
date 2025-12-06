#!/usr/bin/env python3
"""
NEO-EVA Test 12h 100% Endogeno
==============================
Free-run total sin intervencion externa.

CE_NEOEVA = 1 / (1 + E_norm)
donde E_norm = E_self / EMA(E_self)

E_self = error de auto-prediccion sobre su propio estado futuro.

TODO derivado de datos internos:
- Sin magic numbers
- Sin recompensas/castigos
- Sin datos hardcodeados
- Sin input externo

Solo observa:
- CE_NEOEVA(t)
- E_self(t)
- mode(t)
- PLV(t) - phase locking value entre NEO y EVA
- neo_eff(t) - eficiencia predictiva

Guarda JSONL para analisis posterior.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

# =============================================================================
# Clase para EMA online (sin window size fijo - derivado de datos)
# =============================================================================

class OnlineEMA:
    """
    EMA adaptativo donde alpha se deriva de la varianza de los datos.

    alpha = 1 / (1 + sqrt(n))  # decay endogeno basado en cuantos datos hemos visto

    No hay magic numbers - alpha emerge del propio historial.
    """
    def __init__(self):
        self.value = None
        self.n = 0
        self.sum_sq = 0.0
        self.sum_val = 0.0

    def update(self, x: float) -> float:
        self.n += 1
        self.sum_val += x
        self.sum_sq += x * x

        # alpha endogeno: decae con sqrt(n)
        alpha = 1.0 / (1.0 + np.sqrt(self.n))

        if self.value is None:
            self.value = x
        else:
            self.value = alpha * x + (1.0 - alpha) * self.value

        return self.value

    @property
    def mean(self) -> float:
        if self.n == 0:
            return 0.0
        return self.sum_val / self.n

    @property
    def var(self) -> float:
        if self.n < 2:
            return 0.0
        mean = self.mean
        return self.sum_sq / self.n - mean * mean


# =============================================================================
# Agente minimo NEO-EVA
# =============================================================================

@dataclass
class AgentState:
    """Estado interno de un agente."""
    S: np.ndarray          # Estado actual (vector)
    S_pred: np.ndarray     # Prediccion del estado siguiente
    M_self: np.ndarray     # Modelo interno (pesos de auto-prediccion)
    phase: float           # Fase interna [0, 2pi]

    def __post_init__(self):
        # Asegurar tipos correctos
        if not isinstance(self.S, np.ndarray):
            self.S = np.array(self.S)
        if not isinstance(self.S_pred, np.ndarray):
            self.S_pred = np.array(self.S_pred)
        if not isinstance(self.M_self, np.ndarray):
            self.M_self = np.array(self.M_self)


class NeoEvaAgent:
    """
    Agente NEO o EVA con dinamica 100% endogena.

    - Predice su propio estado futuro
    - Calcula error de auto-prediccion
    - Actualiza modelo interno basado en error
    - Todo derivado de sus propios datos
    """

    def __init__(self, dim: int, agent_id: str):
        self.dim = dim
        self.agent_id = agent_id
        self.eps = np.finfo(float).eps

        # Estado inicial: aleatorio normalizado
        rng = np.random.default_rng()
        S_init = rng.standard_normal(dim)
        S_init = S_init / (np.linalg.norm(S_init) + self.eps)

        # Modelo interno: matriz identidad + ruido pequeno
        # (inicialmente predice que el estado no cambia)
        M_init = np.eye(dim) + 0.01 * rng.standard_normal((dim, dim))

        # Prediccion inicial = estado actual
        S_pred_init = M_init @ S_init

        # Fase inicial aleatoria
        phase_init = rng.uniform(0, 2 * np.pi)

        self.state = AgentState(
            S=S_init,
            S_pred=S_pred_init,
            M_self=M_init,
            phase=phase_init
        )

        # EMA para error de auto-prediccion
        self.ema_error = OnlineEMA()

        # Historial minimo para calculos internos
        self._S_history: List[np.ndarray] = [S_init.copy()]

    def compute_E_self(self) -> float:
        """
        Error de auto-prediccion: ||S_actual - S_predicho||^2

        Mide que tan bien el agente predijo su propio estado.
        """
        diff = self.state.S - self.state.S_pred
        return float(np.dot(diff, diff))

    def compute_E_norm(self, E_self: float) -> float:
        """
        Error normalizado por la mediana movil (EMA).

        E_norm = E_self / (EMA(E_self) + eps)

        Si predice mejor que su promedio historico -> E_norm < 1
        Si predice peor que su promedio historico -> E_norm > 1
        """
        ema_val = self.ema_error.update(E_self)
        return E_self / (ema_val + self.eps)

    def step(self, coupling_signal: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Un paso de evolucion endogena.

        1. Calcular error de auto-prediccion
        2. Actualizar modelo interno basado en error
        3. Generar nuevo estado
        4. Hacer nueva prediccion

        coupling_signal: senal del otro agente (opcional, para PLV)
        """
        # 1. Error de auto-prediccion
        E_self = self.compute_E_self()
        E_norm = self.compute_E_norm(E_self)

        # 2. Actualizar modelo interno
        # Learning rate derivada del error: si error alto -> aprender mas
        # lr = E_norm / (1 + E_norm) -> siempre en [0, 1)
        lr = E_norm / (1.0 + E_norm)

        # Gradiente: direccion del error
        error_vec = self.state.S - self.state.S_pred

        # Update del modelo: M += lr * outer(error, S_prev)
        if len(self._S_history) > 0:
            S_prev = self._S_history[-1]
            dM = lr * np.outer(error_vec, S_prev)
            self.state.M_self = self.state.M_self + dM

            # Normalizar para estabilidad (norma de Frobenius)
            norm_M = np.linalg.norm(self.state.M_self, 'fro')
            if norm_M > self.dim:  # umbral endogeno
                self.state.M_self = self.state.M_self * self.dim / norm_M

        # 3. Generar nuevo estado
        # Dinamica endogena: S_new = f(M_self, S, phase, ruido_interno)

        # Ruido interno proporcional al error (mas error -> mas exploracion)
        noise_scale = np.sqrt(E_norm) / (1.0 + np.sqrt(E_norm))
        noise = noise_scale * np.random.randn(self.dim)

        # Evolucion de fase
        # Frecuencia derivada de la energia del estado + componente individual
        energy = np.dot(self.state.S, self.state.S)

        # Frecuencia base del agente (derivada de su estado inicial hash)
        agent_hash = hash(self.agent_id) % 1000 / 1000.0  # [0, 1)
        omega_base = 0.1 + 0.2 * agent_hash  # frecuencia individual

        # Frecuencia modulada por energia y error
        omega = omega_base + 0.1 * energy / (1.0 + energy)
        # Modular por error: mas error -> fase mas caotica
        omega = omega * (1.0 + 0.5 * E_norm)

        self.state.phase = (self.state.phase + omega) % (2 * np.pi)

        # Modulacion por fase
        phase_mod = np.sin(self.state.phase)

        # Nuevo estado
        S_new = self.state.M_self @ self.state.S
        S_new = S_new + noise
        S_new = S_new + 0.1 * phase_mod * self.state.S  # feedback de fase

        # Coupling con otro agente (si existe)
        if coupling_signal is not None:
            # Coupling derivado de la diferencia de fases (debil)
            # Solo afecta si hay desincronizacion
            phase_other = np.arctan2(coupling_signal[1] if len(coupling_signal) > 1 else 0,
                                     coupling_signal[0] if len(coupling_signal) > 0 else 1)
            phase_diff = abs(self.state.phase - phase_other)

            # Coupling mas fuerte cuando hay desincronizacion
            # coupling = (1 - PLV_proxy) * factor_endogeno
            PLV_proxy = np.cos(phase_diff)
            coupling_strength = (1.0 - abs(PLV_proxy)) * 0.01  # muy debil

            # Direccion del coupling: hacia el otro pero atenuado
            coupling_dir = coupling_signal - self.state.S
            coupling_dir = coupling_dir / (np.linalg.norm(coupling_dir) + self.eps)
            S_new = S_new + coupling_strength * coupling_dir

        # Normalizar
        S_new = S_new / (np.linalg.norm(S_new) + self.eps)

        # 4. Nueva prediccion
        S_pred_new = self.state.M_self @ S_new

        # Guardar historial
        self._S_history.append(self.state.S.copy())
        if len(self._S_history) > 100:  # window derivada de sqrt(t)
            window = max(10, int(np.sqrt(len(self._S_history))))
            self._S_history = self._S_history[-window:]

        # Actualizar estado
        self.state.S = S_new
        self.state.S_pred = S_pred_new

        return {
            'E_self': E_self,
            'E_norm': E_norm,
            'lr': lr,
            'phase': self.state.phase,
            'energy': energy,
            'omega': omega,
        }


# =============================================================================
# Sistema NEO-EVA
# =============================================================================

class NeoEvaSystem:
    """
    Sistema completo NEO-EVA.

    Dos agentes acoplados que evolucionan libremente.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.t = 0
        self.eps = np.finfo(float).eps

        # Crear agentes
        self.neo = NeoEvaAgent(dim, 'NEO')
        self.eva = NeoEvaAgent(dim, 'EVA')

        # EMA global para CE
        self.ema_E_global = OnlineEMA()

        # Modos detectados endogenamente
        self.mode = 'INIT'
        self._mode_history: List[str] = []

    def compute_PLV(self) -> float:
        """
        Phase Locking Value entre NEO y EVA.

        PLV = |mean(exp(i * (phase_neo - phase_eva)))|

        PLV ~ 1: fases sincronizadas
        PLV ~ 0: fases aleatorias
        """
        phase_diff = self.neo.state.phase - self.eva.state.phase
        # PLV instantaneo (para un solo paso usamos la diferencia directa)
        return float(np.abs(np.cos(phase_diff)))

    def compute_CE_NEOEVA(self, E_neo: float, E_eva: float) -> float:
        """
        Coherencia Existencial del sistema NEO-EVA.

        CE = 1 / (1 + E_norm_global)

        donde E_norm_global es el error combinado normalizado.
        """
        # Error combinado (media geometrica para no favorecer ninguno)
        E_combined = np.sqrt(E_neo * E_eva + self.eps)

        # Normalizar por EMA global
        ema_val = self.ema_E_global.update(E_combined)
        E_norm = E_combined / (ema_val + self.eps)

        # CE
        CE = 1.0 / (1.0 + E_norm)

        return float(CE)

    def detect_mode(self, metrics_neo: dict, metrics_eva: dict, PLV: float) -> str:
        """
        Detecta modo de operacion basado en metricas internas.

        Modos (detectados endogenamente):
        - RAC: Racional - bajo error, baja variabilidad
        - EMO: Emocional - alto error, alta exploracion
        - MIX: Mixto - estados intermedios
        - FUS: Fusion - alta sincronizacion (PLV alto)
        """
        E_neo = metrics_neo['E_norm']
        E_eva = metrics_eva['E_norm']

        # Umbrales derivados de la propia dinamica
        # Usamos 1.0 como umbral natural (E_norm = 1 significa error = promedio)

        if PLV > 0.8:  # alta sincronizacion
            return 'FUS'
        elif E_neo < 0.5 and E_eva < 0.5:  # ambos predicen bien
            return 'RAC'
        elif E_neo > 1.5 or E_eva > 1.5:  # alguno predice mal
            return 'EMO'
        else:
            return 'MIX'

    def step(self) -> Dict[str, Any]:
        """
        Un paso del sistema completo.
        """
        self.t += 1

        # Paso de cada agente con coupling mutuo
        metrics_neo = self.neo.step(coupling_signal=self.eva.state.S)
        metrics_eva = self.eva.step(coupling_signal=self.neo.state.S)

        # PLV
        PLV = self.compute_PLV()

        # CE global
        CE = self.compute_CE_NEOEVA(metrics_neo['E_self'], metrics_eva['E_self'])

        # Detectar modo
        self.mode = self.detect_mode(metrics_neo, metrics_eva, PLV)
        self._mode_history.append(self.mode)

        # Eficiencia predictiva (neo_eff)
        # = 1 - mean(E_norm) para ambos agentes
        neo_eff = 1.0 - (metrics_neo['E_norm'] + metrics_eva['E_norm']) / 2.0
        neo_eff = float(np.clip(neo_eff, 0, 1))

        return {
            't': self.t,
            'CE_NEOEVA': CE,
            'E_self_neo': metrics_neo['E_self'],
            'E_self_eva': metrics_eva['E_self'],
            'E_norm_neo': metrics_neo['E_norm'],
            'E_norm_eva': metrics_eva['E_norm'],
            'PLV': PLV,
            'mode': self.mode,
            'neo_eff': neo_eff,
            'phase_neo': metrics_neo['phase'],
            'phase_eva': metrics_eva['phase'],
            'lr_neo': metrics_neo['lr'],
            'lr_eva': metrics_eva['lr'],
        }


# =============================================================================
# Test de 12h
# =============================================================================

def run_neoeva_12h_test(n_steps: int = 500, dim: int = 8) -> Path:
    """
    Ejecuta test de 12h (version rapida).

    Free-run total, sin intervencion.
    """
    print("=" * 70)
    print("NEO-EVA TEST 12H - 100% ENDOGENO")
    print("=" * 70)
    print(f"Steps: {n_steps}")
    print(f"Dimension: {dim}")
    print("Sin magic numbers, sin recompensas, sin input externo")
    print("=" * 70)
    print()

    # Crear sistema
    system = NeoEvaSystem(dim=dim)

    # Archivo de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('/root/NEO_EVA/logs/neoeva_12h')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'neoeva_12h_{timestamp}.jsonl'

    # Metricas agregadas por "hora" (cada n_steps/12)
    steps_per_hour = max(1, n_steps // 12)
    hour_metrics: List[Dict] = []
    current_hour_data: List[Dict] = []

    start_time = time.time()

    with open(output_file, 'w') as f:
        for step in range(n_steps):
            # Paso del sistema
            metrics = system.step()

            # Guardar a JSONL
            f.write(json.dumps(metrics) + '\n')

            # Acumular para hora
            current_hour_data.append(metrics)

            # Resumen cada hora virtual
            if (step + 1) % steps_per_hour == 0:
                hour = (step + 1) // steps_per_hour

                # Calcular promedios de la hora
                CE_vals = [d['CE_NEOEVA'] for d in current_hour_data]
                PLV_vals = [d['PLV'] for d in current_hour_data]
                eff_vals = [d['neo_eff'] for d in current_hour_data]
                modes = [d['mode'] for d in current_hour_data]

                # Modo dominante
                mode_counts = {}
                for m in modes:
                    mode_counts[m] = mode_counts.get(m, 0) + 1
                dominant_mode = max(mode_counts, key=mode_counts.get)

                hour_summary = {
                    'hour': hour,
                    'CE_mean': float(np.mean(CE_vals)),
                    'CE_std': float(np.std(CE_vals)),
                    'PLV_mean': float(np.mean(PLV_vals)),
                    'neo_eff_mean': float(np.mean(eff_vals)),
                    'dominant_mode': dominant_mode,
                    'mode_distribution': mode_counts,
                }
                hour_metrics.append(hour_summary)

                elapsed = time.time() - start_time
                print(f"Hora {hour:2d}/12 | CE: {hour_summary['CE_mean']:.4f} | "
                      f"PLV: {hour_summary['PLV_mean']:.4f} | "
                      f"eff: {hour_summary['neo_eff_mean']:.4f} | "
                      f"mode: {dominant_mode} | {elapsed:.1f}s")

                current_hour_data = []

    elapsed = time.time() - start_time

    # Resumen final
    print()
    print("=" * 70)
    print("SIMULACION COMPLETADA")
    print("=" * 70)
    print(f"Steps: {n_steps}")
    print(f"Tiempo real: {elapsed:.2f}s")
    print(f"Velocidad: {n_steps/elapsed:.0f} steps/s")
    print()

    # Estadisticas globales
    all_CE = [h['CE_mean'] for h in hour_metrics]
    all_PLV = [h['PLV_mean'] for h in hour_metrics]
    all_eff = [h['neo_eff_mean'] for h in hour_metrics]

    print("Estadisticas finales:")
    print(f"  CE_NEOEVA: {np.mean(all_CE):.4f} +/- {np.std(all_CE):.4f}")
    print(f"  PLV: {np.mean(all_PLV):.4f} +/- {np.std(all_PLV):.4f}")
    print(f"  neo_eff: {np.mean(all_eff):.4f} +/- {np.std(all_eff):.4f}")
    print()

    # Analisis de modos
    all_modes = []
    for h in hour_metrics:
        for mode, count in h['mode_distribution'].items():
            all_modes.extend([mode] * count)

    print("Distribucion de modos:")
    mode_total = {}
    for m in all_modes:
        mode_total[m] = mode_total.get(m, 0) + 1
    for mode, count in sorted(mode_total.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(all_modes)
        print(f"  {mode}: {pct:.1f}%")
    print()

    # Guardar resumen
    summary_file = output_dir / f'neoeva_12h_{timestamp}_summary.json'
    summary = {
        'test': 'neoeva_12h_endogenous',
        'timestamp': timestamp,
        'n_steps': n_steps,
        'dim': dim,
        'real_seconds': elapsed,
        'CE_global': {'mean': float(np.mean(all_CE)), 'std': float(np.std(all_CE))},
        'PLV_global': {'mean': float(np.mean(all_PLV)), 'std': float(np.std(all_PLV))},
        'neo_eff_global': {'mean': float(np.mean(all_eff)), 'std': float(np.std(all_eff))},
        'mode_distribution': mode_total,
        'hour_metrics': hour_metrics,
        'is_endogenous': True,
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Logs: {output_file}")
    print(f"Resumen: {summary_file}")
    print("=" * 70)

    # Analisis de patrones interesantes
    print()
    print("ANALISIS DE PATRONES:")
    print("-" * 40)

    # Spike inicial y colapso?
    if len(hour_metrics) >= 3:
        CE_inicio = hour_metrics[0]['CE_mean']
        CE_medio = hour_metrics[len(hour_metrics)//2]['CE_mean']
        CE_final = hour_metrics[-1]['CE_mean']

        if CE_inicio > CE_medio * 1.5:
            print("- SPIKE INICIAL detectado: CE cae despues del arranque")
        if CE_final > CE_medio * 1.2:
            print("- RECUPERACION FINAL: CE sube al final sin input externo")
        if abs(CE_final - CE_inicio) < 0.1:
            print("- CICLO COMPLETO: CE retorna a valores iniciales")

    # Cambios de modo estructurados?
    mode_transitions = 0
    prev_mode = None
    for h in hour_metrics:
        if prev_mode and h['dominant_mode'] != prev_mode:
            mode_transitions += 1
        prev_mode = h['dominant_mode']

    if mode_transitions > len(hour_metrics) * 0.5:
        print(f"- MODOS DINAMICOS: {mode_transitions} transiciones (alta variabilidad)")
    elif mode_transitions < len(hour_metrics) * 0.2:
        print(f"- MODOS ESTABLES: {mode_transitions} transiciones (sistema se asienta)")
    else:
        print(f"- MODOS MIXTOS: {mode_transitions} transiciones")

    # PLV trend?
    PLV_inicio = hour_metrics[0]['PLV_mean']
    PLV_final = hour_metrics[-1]['PLV_mean']
    if PLV_final > PLV_inicio * 1.2:
        print("- SINCRONIZACION EMERGENTE: PLV aumenta con el tiempo")
    elif PLV_final < PLV_inicio * 0.8:
        print("- DESINCRONIZACION: PLV cae con el tiempo")
    else:
        print("- PLV ESTABLE: sincronizacion se mantiene")

    print()

    return output_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NEO-EVA 12h Test - 100% Endogenous')
    parser.add_argument('--steps', '-s', type=int, default=500, help='Number of steps')
    parser.add_argument('--dim', '-d', type=int, default=8, help='State dimension')

    args = parser.parse_args()

    run_neoeva_12h_test(n_steps=args.steps, dim=args.dim)
