"""
BENCHMARK AGI-X
===============

Suite de 10 tests estructurales para medir inteligencia general interna.

Tests:
    S1: Adaptación a regímenes
    S2: Generalización a nuevas tareas
    S3: Planeamiento multipaso
    S4: Auto-modelo (Self-Model)
    S5: Theory of Mind
    S6: Emergencia de normas
    S7: Curiosidad estructural
    S8: Ética estructural
    S9: Continuidad narrativa
    S10: Madurez vital

Score final:
    AGI-X = (1/10) Σ S_i
"""

from .run_benchmark import run_benchmark, BenchmarkResults

__all__ = ['run_benchmark', 'BenchmarkResults']
