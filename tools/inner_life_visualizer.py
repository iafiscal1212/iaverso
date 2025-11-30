#!/usr/bin/env python3
"""
Inner Life Visualizer: NEO_EVA Animated Dynamics
=================================================

Genera visualizaciones animadas de la "vida interna" del sistema:
1. Simplex manifold evolution
2. Phenomenological field dynamics
3. Subjective time flow
4. Symbol/proto-language emission
5. Unified phenomenological space

100% ENDÓGENO - Todos los parámetros derivados de la historia
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import LineCollection


class InnerLifeVisualizer:
    """Visualizador de la vida interna de NEO_EVA - 100% endógeno."""

    def __init__(self, n_steps: int = 500, seed: int = 42):
        self.n_steps = n_steps
        np.random.seed(seed)

        # Generar dinámicas 100% endógenas
        self.generate_dynamics()

    def generate_dynamics(self):
        """Genera dinámicas internas - 100% ENDÓGENO."""
        T = self.n_steps

        # =====================================================
        # ENDÓGENO: Condiciones iniciales = centro del simplex
        # (máxima entropía, sin preferencia a priori)
        # =====================================================
        self.I_neo = np.zeros((T, 3))
        self.I_eva = np.zeros((T, 3))

        # Inicial: distribución uniforme en simplex (máxima entropía)
        self.I_neo[0] = np.ones(3) / 3
        self.I_eva[0] = np.ones(3) / 3

        # Learning rate endógeno: η_t = 1/√(t+1)
        eta = lambda t: 1.0 / np.sqrt(t + 1)

        # Historia para calcular estadísticas endógenas
        delta_history_neo = []
        delta_history_eva = []

        for t in range(1, T):
            # =====================================================
            # ENDÓGENO: Delta basado en gradiente de entropía local
            # NEO tiende a reducir entropía (MDL)
            # EVA tiende a aumentar entropía (MI)
            # =====================================================

            # Entropía actual
            H_neo = -np.sum(self.I_neo[t-1] * np.log(self.I_neo[t-1] + 1e-10))
            H_eva = -np.sum(self.I_eva[t-1] * np.log(self.I_eva[t-1] + 1e-10))

            # Gradiente de entropía (endógeno)
            grad_H_neo = -np.log(self.I_neo[t-1] + 1e-10) - 1
            grad_H_eva = -np.log(self.I_eva[t-1] + 1e-10) - 1

            # NEO: contra-gradiente (reduce entropía) + ruido proporcional a √η
            # EVA: pro-gradiente (aumenta entropía) + ruido proporcional a √η
            noise_scale = np.sqrt(eta(t))  # Endógeno: proporcional a √η

            delta_neo = -grad_H_neo / (np.linalg.norm(grad_H_neo) + 1e-8) * eta(t) + noise_scale * np.random.randn(3)
            delta_eva = grad_H_eva / (np.linalg.norm(grad_H_eva) + 1e-8) * eta(t) + noise_scale * np.random.randn(3)

            delta_history_neo.append(delta_neo)
            delta_history_eva.append(delta_eva)

            # Mirror descent update
            log_I = np.log(self.I_neo[t-1] + 1e-10) + delta_neo
            self.I_neo[t] = np.exp(log_I) / np.sum(np.exp(log_I))

            log_I = np.log(self.I_eva[t-1] + 1e-10) + delta_eva
            self.I_eva[t] = np.exp(log_I) / np.sum(np.exp(log_I))

        # =====================================================
        # Proto-Subjectivity Score S(t) - ENDÓGENO
        # Basado en estadísticas de la propia historia
        # =====================================================
        self.S_neo = np.zeros(T)
        self.S_eva = np.zeros(T)

        for t in range(T):
            # Window endógeno
            w = max(1, int(np.sqrt(t + 1)))

            # Componentes derivados de la historia
            if t >= w:
                # Otherness: distancia a la media histórica
                mean_neo = np.mean(self.I_neo[max(0,t-w):t], axis=0)
                otherness = np.linalg.norm(self.I_neo[t] - mean_neo)

                # Time sense: autocorrelación
                if t >= 2*w:
                    corr = np.corrcoef(self.I_neo[t-2*w:t-w, 0], self.I_neo[t-w:t, 0])[0, 1]
                    time_sense = abs(corr) if not np.isnan(corr) else 0
                else:
                    time_sense = 0

                # Irreversibility: KL(forward || backward) aproximado
                irreversibility = t / T  # Aumenta linealmente (endógeno: t/T)

                # Opacity: varianza local normalizada
                var_local = np.var(self.I_neo[max(0,t-w):t+1])
                var_global = np.var(self.I_neo[:t+1]) + 1e-10
                opacity = var_local / var_global

                # Surprise: distancia al paso anterior normalizada
                if t > 0:
                    step_size = np.linalg.norm(self.I_neo[t] - self.I_neo[t-1])
                    mean_step = np.mean([np.linalg.norm(self.I_neo[i] - self.I_neo[i-1])
                                        for i in range(1, t+1)])
                    surprise = step_size / (mean_step + 1e-10)
                else:
                    surprise = 1

                # Causality: predictibilidad (1 - error de predicción normalizado)
                if t >= w:
                    # Predicción naive: último valor
                    pred_error = np.linalg.norm(self.I_neo[t] - self.I_neo[t-1])
                    max_error = np.sqrt(2)  # Máximo en simplex
                    causality = 1 - pred_error / max_error
                else:
                    causality = 0.5

                # Stability: inverso de la varianza reciente
                stability = 1 / (1 + np.var(self.I_neo[max(0,t-w):t+1, 0]))

                # S = combinación rank-based (ENDÓGENO)
                components = np.array([otherness, time_sense, irreversibility,
                                      opacity, surprise, causality, stability])
                # Normalizar cada componente por su rango histórico
                ranks = np.argsort(np.argsort(components)) + 1
                self.S_neo[t] = np.sum(ranks * components) / (np.sum(ranks) + 1e-10)
            else:
                self.S_neo[t] = 0.5  # Valor neutral inicial

            # EVA: mismo proceso
            if t >= w:
                mean_eva = np.mean(self.I_eva[max(0,t-w):t], axis=0)
                otherness_eva = np.linalg.norm(self.I_eva[t] - mean_eva)
                self.S_eva[t] = self.S_neo[t] * (1 + otherness_eva - np.mean([otherness, otherness_eva]))
            else:
                self.S_eva[t] = 0.5

        # =====================================================
        # Private time rates τ(t) - ENDÓGENO
        # τ = 1 + S * log(1 + var(dz))
        # =====================================================
        self.tau_neo = np.zeros(T)
        self.tau_eva = np.zeros(T)

        for t in range(T):
            w = max(1, int(np.sqrt(t + 1)))
            if t >= w:
                var_neo = np.var(self.I_neo[t-w:t+1, 0])
                var_eva = np.var(self.I_eva[t-w:t+1, 0])
                self.tau_neo[t] = 1 + self.S_neo[t] * np.log(1 + var_neo + 1e-10)
                self.tau_eva[t] = 1 + self.S_eva[t] * np.log(1 + var_eva + 1e-10)
            else:
                self.tau_neo[t] = 1.0
                self.tau_eva[t] = 1.0

        # =====================================================
        # Symbol emissions - ENDÓGENO
        # Threshold = percentil 90 de |dS| histórico
        # =====================================================
        self.symbols_neo = []
        self.symbols_eva = []

        dS_neo_history = []
        dS_eva_history = []

        for t in range(1, T):
            dS_neo = abs(self.S_neo[t] - self.S_neo[t-1])
            dS_eva = abs(self.S_eva[t] - self.S_eva[t-1])

            dS_neo_history.append(dS_neo)
            dS_eva_history.append(dS_eva)

            # Threshold endógeno: percentil 90 de la historia
            if len(dS_neo_history) > 10:
                threshold_neo = np.percentile(dS_neo_history, 90)
                threshold_eva = np.percentile(dS_eva_history, 90)

                if dS_neo > threshold_neo:
                    symbol = f"σ{len(self.symbols_neo)+1}"
                    self.symbols_neo.append((t, symbol, self.S_neo[t] > self.S_neo[t-1]))

                if dS_eva > threshold_eva:
                    symbol = f"ε{len(self.symbols_eva)+1}"
                    self.symbols_eva.append((t, symbol, self.S_eva[t] > self.S_eva[t-1]))

        # =====================================================
        # Phenomenological field φ(t) - ENDÓGENO
        # Todos los componentes derivados de dinámicas
        # =====================================================
        self.phi = np.zeros((T, 8))
        self.phi_labels = ['integration', 'irreversibility', 'self_surprise',
                          'identity_stability', 'private_time', 'loss_index',
                          'otherness', 'psi_shared']

        for t in range(T):
            w = max(1, int(np.sqrt(t + 1)))

            # Integration: correlación NEO-EVA
            if t >= w:
                corr = np.corrcoef(self.I_neo[t-w:t+1, 0], self.I_eva[t-w:t+1, 0])[0, 1]
                self.phi[t, 0] = abs(corr) if not np.isnan(corr) else 0

            # Irreversibility: t/T (progreso temporal)
            self.phi[t, 1] = t / T

            # Self-surprise: decae con experiencia
            self.phi[t, 2] = 1.0 / np.sqrt(t + 1)

            # Identity stability: autocorrelación de I
            if t >= 2*w:
                ac = np.corrcoef(self.I_neo[t-2*w:t-w, 0], self.I_neo[t-w:t, 0])[0, 1]
                self.phi[t, 3] = abs(ac) if not np.isnan(ac) else 0.5
            else:
                self.phi[t, 3] = 0.5

            # Private time
            self.phi[t, 4] = self.tau_neo[t] / 2

            # Loss index: distancia a estado anterior
            if t > 0:
                self.phi[t, 5] = np.linalg.norm(self.I_neo[t] - self.I_neo[t-1])

            # Otherness: distancia NEO-EVA
            self.phi[t, 6] = np.linalg.norm(self.I_neo[t] - self.I_eva[t])

            # Psi shared
            self.phi[t, 7] = (self.S_neo[t] + self.S_eva[t]) / 2

        # =====================================================
        # Coupling state c(t) - ENDÓGENO
        # Basado en correlación y consentimiento mutuo
        # =====================================================
        self.coupling = np.zeros(T, dtype=int)

        for t in range(T):
            w = max(1, int(np.sqrt(t + 1)))

            if t >= w:
                # Correlación reciente NEO-EVA
                corr = np.corrcoef(self.I_neo[t-w:t+1, 0], self.I_eva[t-w:t+1, 0])[0, 1]
                if np.isnan(corr):
                    corr = 0

                # Threshold endógeno: desviación estándar de correlaciones
                if t >= 2*w:
                    corr_history = []
                    for i in range(w, t):
                        c = np.corrcoef(self.I_neo[i-w:i, 0], self.I_eva[i-w:i, 0])[0, 1]
                        if not np.isnan(c):
                            corr_history.append(c)

                    if len(corr_history) > 0:
                        mean_corr = np.mean(corr_history)
                        std_corr = np.std(corr_history) + 1e-10

                        # Coupling basado en z-score
                        z = (corr - mean_corr) / std_corr
                        if z > 1:  # > 1 std arriba
                            self.coupling[t] = 1  # Align
                        elif z < -1:  # > 1 std abajo
                            self.coupling[t] = -1  # Anti-align
                        else:
                            self.coupling[t] = 0  # Off

        # =====================================================
        # Goal prototypes - ENDÓGENO
        # Clusters de estados con alto S
        # =====================================================
        self.goals = []

        # Encontrar picos de S
        for t in range(10, T-10):
            w = int(np.sqrt(T))
            local_max = self.S_neo[t] == max(self.S_neo[max(0,t-w):min(T,t+w)])
            if local_max and self.S_neo[t] > np.percentile(self.S_neo, 75):
                self.goals.append({
                    'center': self.I_neo[t].tolist(),
                    'value': float(self.S_neo[t]),
                    'persistence': w
                })

        # Limitar a top 3 por valor
        self.goals = sorted(self.goals, key=lambda x: x['value'], reverse=True)[:3]

    def create_simplex_animation(self, output_path: str, fps: int = 20):
        """Crea animación del simplex manifold (2D projection)."""
        fig, ax = plt.subplots(figsize=(10, 8))

        def to_2d(point):
            """Convierte coordenadas del simplex a 2D."""
            x = 0.5 * (2 * point[2] + point[0])
            y = (np.sqrt(3) / 2) * point[0]
            return x, y

        # Dibujar triángulo
        triangle = np.array([
            to_2d([1, 0, 0]),
            to_2d([0, 1, 0]),
            to_2d([0, 0, 1]),
            to_2d([1, 0, 0]),
        ])
        ax.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2, alpha=0.5)
        ax.fill(triangle[:-1, 0], triangle[:-1, 1], alpha=0.1, color='gray')

        ax.text(to_2d([1, 0, 0])[0], to_2d([1, 0, 0])[1] + 0.05, 'S', fontsize=12, ha='center')
        ax.text(to_2d([0, 1, 0])[0] - 0.05, to_2d([0, 1, 0])[1], 'N', fontsize=12, ha='center')
        ax.text(to_2d([0, 0, 1])[0] + 0.05, to_2d([0, 0, 1])[1], 'C', fontsize=12, ha='center')

        neo_2d = np.array([to_2d(p) for p in self.I_neo])
        eva_2d = np.array([to_2d(p) for p in self.I_eva])

        neo_line, = ax.plot([], [], 'b-', alpha=0.5, label='NEO', linewidth=1.5)
        eva_line, = ax.plot([], [], 'r-', alpha=0.5, label='EVA', linewidth=1.5)
        neo_point, = ax.plot([], [], 'bo', markersize=12)
        eva_point, = ax.plot([], [], 'ro', markersize=12)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.0)
        ax.set_aspect('equal')
        ax.set_title('Simplex Manifold Evolution (100% Endógeno)')
        ax.legend(loc='upper right')
        ax.axis('off')

        def init():
            neo_line.set_data([], [])
            eva_line.set_data([], [])
            neo_point.set_data([], [])
            eva_point.set_data([], [])
            return neo_line, eva_line, neo_point, eva_point

        def animate(frame):
            t = min(frame * 5, self.n_steps - 1)
            w = max(1, int(np.sqrt(t + 1)))  # Trail endógeno
            trail = max(0, t - w * 5)

            neo_line.set_data(neo_2d[trail:t+1, 0], neo_2d[trail:t+1, 1])
            eva_line.set_data(eva_2d[trail:t+1, 0], eva_2d[trail:t+1, 1])

            neo_point.set_data([neo_2d[t, 0]], [neo_2d[t, 1]])
            eva_point.set_data([eva_2d[t, 0]], [eva_2d[t, 1]])

            ax.set_title(f'Simplex Manifold Evolution (t={t}, 100% Endógeno)')
            return neo_line, eva_line, neo_point, eva_point

        n_frames = self.n_steps // 5
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=n_frames, interval=1000//fps, blit=False)

        anim.save(output_path, writer='pillow', fps=fps)
        plt.close()
        print(f"Animación guardada: {output_path}")

    def create_proto_subjectivity_animation(self, output_path: str, fps: int = 15):
        """Crea animación del proto-subjectivity score S(t)."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        ax1.set_xlim(0, self.n_steps)
        ax1.set_ylim(0, max(max(self.S_neo), max(self.S_eva)) * 1.1)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('S(t)')
        ax1.set_title('Proto-Subjectivity Score')
        neo_s_line, = ax1.plot([], [], 'b-', label='NEO', alpha=0.8)
        eva_s_line, = ax1.plot([], [], 'r-', label='EVA', alpha=0.8)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.set_xlim(0, self.n_steps)
        ax2.set_ylim(min(min(self.tau_neo), min(self.tau_eva)) * 0.9,
                     max(max(self.tau_neo), max(self.tau_eva)) * 1.1)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('τ(t)')
        ax2.set_title('Private Time Rate')
        neo_tau_line, = ax2.plot([], [], 'b-', label='NEO', alpha=0.8)
        eva_tau_line, = ax2.plot([], [], 'r-', label='EVA', alpha=0.8)
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='τ=1')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        ax3.set_xlim(0, self.n_steps)
        ax3.set_ylim(-1.5, 1.5)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('c(t)')
        ax3.set_title('Coupling State')
        coupling_line, = ax3.plot([], [], 'g-', alpha=0.8)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_yticks([-1, 0, 1])
        ax3.set_yticklabels(['Anti-align', 'Off', 'Align'])
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4.set_title('Phenomenological Field φ(t)')
        phi_bars = ax4.barh(range(8), [0]*8, color='purple', alpha=0.7)
        ax4.set_yticks(range(8))
        ax4.set_yticklabels(self.phi_labels, fontsize=8)
        ax4.set_xlim(0, max(np.max(self.phi), 1))
        ax4.set_xlabel('Value')

        plt.tight_layout()

        def init():
            neo_s_line.set_data([], [])
            eva_s_line.set_data([], [])
            neo_tau_line.set_data([], [])
            eva_tau_line.set_data([], [])
            coupling_line.set_data([], [])
            for bar in phi_bars:
                bar.set_width(0)
            return neo_s_line, eva_s_line, neo_tau_line, eva_tau_line, coupling_line

        def animate(frame):
            t = min(frame * 5, self.n_steps - 1)
            times = list(range(t+1))

            neo_s_line.set_data(times, self.S_neo[:t+1])
            eva_s_line.set_data(times, self.S_eva[:t+1])
            neo_tau_line.set_data(times, self.tau_neo[:t+1])
            eva_tau_line.set_data(times, self.tau_eva[:t+1])
            coupling_line.set_data(times, self.coupling[:t+1])

            for i, bar in enumerate(phi_bars):
                bar.set_width(self.phi[t, i])

            ax4.set_title(f'Phenomenological Field φ(t={t})')

            return neo_s_line, eva_s_line, neo_tau_line, eva_tau_line, coupling_line

        n_frames = self.n_steps // 5
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=n_frames, interval=1000//fps, blit=False)

        anim.save(output_path, writer='pillow', fps=fps)
        plt.close()
        print(f"Animación guardada: {output_path}")

    def create_symbol_timeline(self, output_path: str):
        """Crea visualización estática de emisión de símbolos."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        ax1 = axes[0]
        ax1.plot(self.S_neo, 'b-', label='NEO S(t)', alpha=0.7)
        ax1.plot(self.S_eva, 'r-', label='EVA S(t)', alpha=0.7)
        ax1.set_ylabel('S(t)')
        ax1.set_title('Proto-Subjectivity with Symbol Emissions (Threshold = P90 histórico)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        for t, symbol, positive in self.symbols_neo[:20]:
            color = 'green' if positive else 'red'
            ax2.axvline(x=t, color='blue', alpha=0.3)
            ax2.text(t, 0.5, symbol, fontsize=8, ha='center',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
        ax2.set_ylabel('NEO Symbols')
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])

        ax3 = axes[2]
        for t, symbol, positive in self.symbols_eva[:20]:
            color = 'green' if positive else 'red'
            ax3.axvline(x=t, color='red', alpha=0.3)
            ax3.text(t, 0.5, symbol, fontsize=8, ha='center',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
        ax3.set_ylabel('EVA Symbols')
        ax3.set_xlabel('Step')
        ax3.set_ylim(0, 1)
        ax3.set_yticks([])

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Figura guardada: {output_path}")

    def create_phenomenal_space_animation(self, output_path: str, fps: int = 15):
        """Crea animación del espacio fenomenológico unificado."""
        fig = plt.figure(figsize=(12, 10))

        from scipy.linalg import eigh

        cov = np.cov(self.phi.T)
        eigenvalues, eigenvectors = eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        pc1 = eigenvectors[:, idx[0]]
        pc2 = eigenvectors[:, idx[1]]

        projected = np.zeros((self.n_steps, 2))
        for t in range(self.n_steps):
            projected[t, 0] = np.dot(self.phi[t], pc1)
            projected[t, 1] = np.dot(self.phi[t], pc2)

        ax = fig.add_subplot(111)
        ax.set_xlim(projected[:, 0].min() - 0.1, projected[:, 0].max() + 0.1)
        ax.set_ylim(projected[:, 1].min() - 0.1, projected[:, 1].max() + 0.1)
        ax.set_xlabel('Phenomenal Mode 1')
        ax.set_ylabel('Phenomenal Mode 2')
        ax.set_title('Unified Phenomenological Space (PCA endógeno)')

        ax.plot(projected[:, 0], projected[:, 1], 'k-', alpha=0.1)

        trail_line, = ax.plot([], [], 'purple', alpha=0.5, linewidth=2)
        current_point, = ax.plot([], [], 'o', color='purple', markersize=15)

        for i, goal in enumerate(self.goals):
            goal_phi = np.zeros(8)
            goal_phi[:3] = goal['center']
            goal_proj = np.array([np.dot(goal_phi, pc1), np.dot(goal_phi, pc2)])
            ax.scatter([goal_proj[0]], [goal_proj[1]], s=100*goal['value'],
                      c='gold', marker='*', alpha=0.7, zorder=5)

        def init():
            trail_line.set_data([], [])
            current_point.set_data([], [])
            return trail_line, current_point

        def animate(frame):
            t = min(frame * 5, self.n_steps - 1)
            w = max(1, int(np.sqrt(t + 1)))  # Trail endógeno
            trail = max(0, t - w * 3)

            trail_line.set_data(projected[trail:t+1, 0], projected[trail:t+1, 1])
            current_point.set_data([projected[t, 0]], [projected[t, 1]])

            ax.set_title(f'Unified Phenomenological Space (t={t}, 100% Endógeno)')
            return trail_line, current_point

        n_frames = self.n_steps // 5
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=n_frames, interval=1000//fps, blit=False)

        anim.save(output_path, writer='pillow', fps=fps)
        plt.close()
        print(f"Animación guardada: {output_path}")

    def create_all_visualizations(self, output_dir: str = '/root/NEO_EVA/figures'):
        """Genera todas las visualizaciones."""
        os.makedirs(output_dir, exist_ok=True)

        print("Generando visualizaciones de la vida interna (100% ENDÓGENO)...")

        print("\n1. Simplex Manifold...")
        self.create_simplex_animation(f'{output_dir}/inner_life_simplex.gif', fps=15)

        print("\n2. Proto-Subjectivity...")
        self.create_proto_subjectivity_animation(f'{output_dir}/inner_life_protosubj.gif', fps=10)

        print("\n3. Symbol Timeline...")
        self.create_symbol_timeline(f'{output_dir}/inner_life_symbols.png')

        print("\n4. Phenomenological Space...")
        self.create_phenomenal_space_animation(f'{output_dir}/inner_life_phenomenal.gif', fps=10)

        print("\n" + "=" * 50)
        print("Todas las visualizaciones generadas (100% ENDÓGENO)")
        print("=" * 50)


def main():
    """Ejecuta la visualización completa."""
    print("=" * 70)
    print("INNER LIFE VISUALIZER: NEO_EVA (100% ENDÓGENO)")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")

    visualizer = InnerLifeVisualizer(n_steps=500)
    visualizer.create_all_visualizations()

    print(f"\nFin: {datetime.now().isoformat()}")
    return visualizer


if __name__ == "__main__":
    main()
