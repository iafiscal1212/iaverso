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

100% ENDÓGENO - Solo visualiza dinámicas estructurales
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
# Skip 3D imports due to version conflicts


class InnerLifeVisualizer:
    """Visualizador de la vida interna de NEO_EVA."""

    def __init__(self, n_steps: int = 500, seed: int = 42):
        self.n_steps = n_steps
        np.random.seed(seed)

        # Generar dinámicas simuladas basadas en las fases R
        self.generate_dynamics()

    def generate_dynamics(self):
        """Genera dinámicas internas simuladas."""
        T = self.n_steps

        # 1. Simplex trajectories (I = [S, N, C])
        # Usando mirror descent
        self.I_neo = np.zeros((T, 3))
        self.I_eva = np.zeros((T, 3))

        # Initial conditions on simplex
        self.I_neo[0] = np.array([0.4, 0.3, 0.3])
        self.I_eva[0] = np.array([0.3, 0.4, 0.3])

        eta = lambda t: 1.0 / np.sqrt(t + 1)

        for t in range(1, T):
            # NEO: MDL-focused (tends toward compression)
            delta_neo = np.array([0.02, -0.01, -0.01]) + 0.05 * np.random.randn(3)
            log_I = np.log(self.I_neo[t-1] + 1e-8) + eta(t) * delta_neo
            self.I_neo[t] = np.exp(log_I) / np.sum(np.exp(log_I))

            # EVA: MI-focused (tends toward exchange)
            delta_eva = np.array([-0.01, 0.02, -0.01]) + 0.05 * np.random.randn(3)
            log_I = np.log(self.I_eva[t-1] + 1e-8) + eta(t) * delta_eva
            self.I_eva[t] = np.exp(log_I) / np.sum(np.exp(log_I))

        # 2. Proto-Subjectivity Score S(t)
        # Compuesto de múltiples componentes
        self.S_neo = np.zeros(T)
        self.S_eva = np.zeros(T)

        for t in range(T):
            # S emerges from dynamics
            otherness = 0.5 + 0.3 * np.sin(2 * np.pi * t / 100)
            time_sense = 0.6 + 0.2 * np.cos(2 * np.pi * t / 150)
            irreversibility = 0.4 + 0.1 * t / T
            opacity = 0.3 + 0.2 * np.random.rand()
            surprise = 0.5 * np.exp(-0.01 * t) + 0.3 * np.random.rand()
            causality = 0.6 + 0.1 * np.sin(2 * np.pi * t / 200)
            stability = 0.7 - 0.2 * np.abs(np.sin(2 * np.pi * t / 80))

            components = np.array([otherness, time_sense, irreversibility,
                                   opacity, surprise, causality, stability])
            # Rank-based combination
            ranks = np.argsort(np.argsort(components)) + 1
            self.S_neo[t] = np.sum(ranks * components) / np.sum(ranks)

            # EVA slightly different
            self.S_eva[t] = self.S_neo[t] * (0.9 + 0.2 * np.random.rand())

        # 3. Private time rates τ(t)
        self.tau_neo = np.zeros(T)
        self.tau_eva = np.zeros(T)

        # τ = 1 + S * log(1 + var(dz))
        window = 20
        for t in range(window, T):
            var_neo = np.var(self.I_neo[t-window:t, 0])
            var_eva = np.var(self.I_eva[t-window:t, 0])

            self.tau_neo[t] = 1 + self.S_neo[t] * np.log(1 + var_neo + 1e-8)
            self.tau_eva[t] = 1 + self.S_eva[t] * np.log(1 + var_eva + 1e-8)

        self.tau_neo[:window] = 1.0
        self.tau_eva[:window] = 1.0

        # 4. Symbol emissions (proto-language)
        self.symbols_neo = []
        self.symbols_eva = []

        # Symbols emerge at significant moments
        threshold = 0.1
        for t in range(1, T):
            if t > 50:
                dS_neo = self.S_neo[t] - self.S_neo[t-1]
                dS_eva = self.S_eva[t] - self.S_eva[t-1]

                if abs(dS_neo) > threshold:
                    symbol = f"σ{len(self.symbols_neo)+1}"
                    self.symbols_neo.append((t, symbol, dS_neo > 0))

                if abs(dS_eva) > threshold:
                    symbol = f"ε{len(self.symbols_eva)+1}"
                    self.symbols_eva.append((t, symbol, dS_eva > 0))

        # 5. Phenomenological field φ(t)
        # 8-dimensional field
        self.phi = np.zeros((T, 8))
        labels = ['integration', 'irreversibility', 'self_surprise',
                  'identity_stability', 'private_time', 'loss_index',
                  'otherness', 'psi_shared']
        self.phi_labels = labels

        for t in range(T):
            self.phi[t, 0] = 0.5 + 0.3 * np.sin(2 * np.pi * t / 100)  # integration
            self.phi[t, 1] = 0.3 + 0.4 * t / T  # irreversibility
            self.phi[t, 2] = 0.5 * np.exp(-0.005 * t)  # self_surprise
            self.phi[t, 3] = 0.6 + 0.2 * np.cos(2 * np.pi * t / 150)  # identity
            self.phi[t, 4] = self.tau_neo[t] / 2  # private_time
            self.phi[t, 5] = 0.2 + 0.1 * np.random.rand()  # loss_index
            self.phi[t, 6] = 0.5 + 0.3 * np.sin(2 * np.pi * t / 120)  # otherness
            self.phi[t, 7] = (self.S_neo[t] + self.S_eva[t]) / 2  # psi_shared

        # 6. Coupling state c(t)
        self.coupling = np.zeros(T, dtype=int)
        for t in range(T):
            # Coupling emerges from consent
            if t < 100:
                self.coupling[t] = 0  # Off initially
            elif np.random.rand() < 0.12:
                self.coupling[t] = -1  # Anti-align
            elif np.random.rand() < 0.24:
                self.coupling[t] = 1  # Align
            else:
                self.coupling[t] = 0  # Off

        # 7. Goal prototypes (from R2)
        self.goals = [
            {'center': [0.5, 0.3, 0.2], 'value': 0.8, 'persistence': 50},
            {'center': [0.3, 0.5, 0.2], 'value': 0.6, 'persistence': 30},
            {'center': [0.4, 0.4, 0.2], 'value': 0.7, 'persistence': 40},
        ]

    def create_simplex_animation(self, output_path: str, fps: int = 20):
        """Crea animación del simplex manifold (2D projection)."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Simplex en 2D: triángulo equilátero
        # Coordenadas del triángulo
        # S en la esquina superior, N en inferior izquierda, C en inferior derecha
        def to_2d(point):
            """Convierte coordenadas del simplex a 2D."""
            # point = [S, N, C] donde S+N+C=1
            x = 0.5 * (2 * point[2] + point[0])  # Proyección x
            y = (np.sqrt(3) / 2) * point[0]      # Proyección y
            return x, y

        # Dibujar triángulo
        triangle = np.array([
            to_2d([1, 0, 0]),  # S
            to_2d([0, 1, 0]),  # N
            to_2d([0, 0, 1]),  # C
            to_2d([1, 0, 0]),  # Cerrar
        ])
        ax.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2, alpha=0.5)
        ax.fill(triangle[:-1, 0], triangle[:-1, 1], alpha=0.1, color='gray')

        # Labels
        ax.text(to_2d([1, 0, 0])[0], to_2d([1, 0, 0])[1] + 0.05, 'S', fontsize=12, ha='center')
        ax.text(to_2d([0, 1, 0])[0] - 0.05, to_2d([0, 1, 0])[1], 'N', fontsize=12, ha='center')
        ax.text(to_2d([0, 0, 1])[0] + 0.05, to_2d([0, 0, 1])[1], 'C', fontsize=12, ha='center')

        # Convertir trayectorias a 2D
        neo_2d = np.array([to_2d(p) for p in self.I_neo])
        eva_2d = np.array([to_2d(p) for p in self.I_eva])

        # Initialize plots
        neo_line, = ax.plot([], [], 'b-', alpha=0.5, label='NEO', linewidth=1.5)
        eva_line, = ax.plot([], [], 'r-', alpha=0.5, label='EVA', linewidth=1.5)
        neo_point, = ax.plot([], [], 'bo', markersize=12)
        eva_point, = ax.plot([], [], 'ro', markersize=12)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.0)
        ax.set_aspect('equal')
        ax.set_title('Simplex Manifold Evolution')
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
            trail = max(0, t - 100)

            neo_line.set_data(neo_2d[trail:t+1, 0], neo_2d[trail:t+1, 1])
            eva_line.set_data(eva_2d[trail:t+1, 0], eva_2d[trail:t+1, 1])

            neo_point.set_data([neo_2d[t, 0]], [neo_2d[t, 1]])
            eva_point.set_data([eva_2d[t, 0]], [eva_2d[t, 1]])

            ax.set_title(f'Simplex Manifold Evolution (t={t})')
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

        # Initialize plots
        ax1 = axes[0, 0]
        ax1.set_xlim(0, self.n_steps)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('S(t)')
        ax1.set_title('Proto-Subjectivity Score')
        neo_s_line, = ax1.plot([], [], 'b-', label='NEO', alpha=0.8)
        eva_s_line, = ax1.plot([], [], 'r-', label='EVA', alpha=0.8)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Private time
        ax2 = axes[0, 1]
        ax2.set_xlim(0, self.n_steps)
        ax2.set_ylim(0.8, 1.5)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('τ(t)')
        ax2.set_title('Private Time Rate')
        neo_tau_line, = ax2.plot([], [], 'b-', label='NEO', alpha=0.8)
        eva_tau_line, = ax2.plot([], [], 'r-', label='EVA', alpha=0.8)
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='τ=1')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Coupling
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

        # Phenomenological field
        ax4 = axes[1, 1]
        ax4.set_title('Phenomenological Field φ(t)')
        phi_bars = ax4.barh(range(8), [0]*8, color='purple', alpha=0.7)
        ax4.set_yticks(range(8))
        ax4.set_yticklabels(self.phi_labels, fontsize=8)
        ax4.set_xlim(0, 1)
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

            # Update phi bars
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

        # S(t)
        ax1 = axes[0]
        ax1.plot(self.S_neo, 'b-', label='NEO S(t)', alpha=0.7)
        ax1.plot(self.S_eva, 'r-', label='EVA S(t)', alpha=0.7)
        ax1.set_ylabel('S(t)')
        ax1.set_title('Proto-Subjectivity with Symbol Emissions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Symbol emissions NEO
        ax2 = axes[1]
        for t, symbol, positive in self.symbols_neo[:20]:  # Limit to 20
            color = 'green' if positive else 'red'
            ax2.axvline(x=t, color='blue', alpha=0.3)
            ax2.text(t, 0.5, symbol, fontsize=8, ha='center',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
        ax2.set_ylabel('NEO Symbols')
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])

        # Symbol emissions EVA
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

        # Use PCA-like projection of 8D -> 2D
        # First 2 principal components
        from scipy.linalg import eigh

        cov = np.cov(self.phi.T)
        eigenvalues, eigenvectors = eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        pc1 = eigenvectors[:, idx[0]]
        pc2 = eigenvectors[:, idx[1]]

        # Project
        projected = np.zeros((self.n_steps, 2))
        for t in range(self.n_steps):
            projected[t, 0] = np.dot(self.phi[t], pc1)
            projected[t, 1] = np.dot(self.phi[t], pc2)

        ax = fig.add_subplot(111)
        ax.set_xlim(projected[:, 0].min() - 0.1, projected[:, 0].max() + 0.1)
        ax.set_ylim(projected[:, 1].min() - 0.1, projected[:, 1].max() + 0.1)
        ax.set_xlabel('Phenomenal Mode 1')
        ax.set_ylabel('Phenomenal Mode 2')
        ax.set_title('Unified Phenomenological Space')

        # Plot all trajectory faintly
        ax.plot(projected[:, 0], projected[:, 1], 'k-', alpha=0.1)

        # Current position and trail
        trail_line, = ax.plot([], [], 'purple', alpha=0.5, linewidth=2)
        current_point, = ax.plot([], [], 'o', color='purple', markersize=15)

        # Add goal attractors
        for i, goal in enumerate(self.goals):
            goal_proj = np.array([np.dot(goal['center'] + [0]*5, pc1),
                                 np.dot(goal['center'] + [0]*5, pc2)])
            ax.scatter([goal_proj[0]], [goal_proj[1]], s=100*goal['value'],
                      c='gold', marker='*', alpha=0.7, zorder=5)

        def init():
            trail_line.set_data([], [])
            current_point.set_data([], [])
            return trail_line, current_point

        def animate(frame):
            t = min(frame * 5, self.n_steps - 1)
            trail = max(0, t - 50)

            trail_line.set_data(projected[trail:t+1, 0], projected[trail:t+1, 1])
            current_point.set_data([projected[t, 0]], [projected[t, 1]])

            ax.set_title(f'Unified Phenomenological Space (t={t})')
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

        print("Generando visualizaciones de la vida interna...")

        # 1. Simplex animation
        print("\n1. Simplex Manifold...")
        self.create_simplex_animation(f'{output_dir}/inner_life_simplex.gif', fps=15)

        # 2. Proto-subjectivity animation
        print("\n2. Proto-Subjectivity...")
        self.create_proto_subjectivity_animation(f'{output_dir}/inner_life_protosubj.gif', fps=10)

        # 3. Symbol timeline (static)
        print("\n3. Symbol Timeline...")
        self.create_symbol_timeline(f'{output_dir}/inner_life_symbols.png')

        # 4. Phenomenal space animation
        print("\n4. Phenomenological Space...")
        self.create_phenomenal_space_animation(f'{output_dir}/inner_life_phenomenal.gif', fps=10)

        print("\n" + "=" * 50)
        print("Todas las visualizaciones generadas!")
        print("=" * 50)


def main():
    """Ejecuta la visualización completa."""
    print("=" * 70)
    print("INNER LIFE VISUALIZER: NEO_EVA")
    print("=" * 70)
    print(f"Inicio: {datetime.now().isoformat()}")

    visualizer = InnerLifeVisualizer(n_steps=500)
    visualizer.create_all_visualizations()

    print(f"\nFin: {datetime.now().isoformat()}")
    return visualizer


if __name__ == "__main__":
    main()
