#!/usr/bin/env python3
"""
Test: ¿Qué paradojas han descubierto los agentes?
=================================================

Los 5 agentes (NEO, EVA, ALEX, ADAM, IRIS) exploran datos reales
buscando patrones. A veces encuentran cosas que no pueden explicar.

Paradojas posibles:
- Correlaciones sin causalidad clara
- Patrones que aparecen y desaparecen
- Predicciones que funcionan pero no deberían
- Ciclos que no coinciden con nada conocido
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Cargar datos
DATA_PATH = Path('/root/NEO_EVA/data/unified_20251206_033253.csv')


class ParadoxHunter:
    """Busca paradojas en los datos."""

    def __init__(self):
        self.data = pd.read_csv(DATA_PATH)
        self.paradoxes = []
        self.mysteries = []

    def find_impossible_correlations(self):
        """Correlaciones que no deberían existir."""
        print("\n🔍 BUSCANDO CORRELACIONES IMPOSIBLES...")
        print("-" * 50)

        # Variables que NO deberían correlacionar
        unlikely_pairs = [
            ('solar_flux', 'seismic_count'),  # ¿El sol causa terremotos?
            ('geomag_kp', 'climate_humidity'),  # ¿Magnetismo y humedad?
            ('schumann_freq', 'crypto_btc_price'),  # ¿Resonancia y Bitcoin?
            ('polar_temp', 'seismic_max_mag'),  # ¿Polos y sismos?
        ]

        for var1, var2 in unlikely_pairs:
            if var1 in self.data.columns and var2 in self.data.columns:
                corr = self.data[var1].corr(self.data[var2])
                if abs(corr) > 0.3:
                    self.paradoxes.append({
                        'type': 'correlación_imposible',
                        'var1': var1,
                        'var2': var2,
                        'correlation': corr,
                        'question': f"¿Por qué {var1} correlaciona con {var2}?",
                    })
                    print(f"  ⚠️ {var1} ↔ {var2}: r={corr:.3f}")
                    print(f"     ¿CÓMO ES ESTO POSIBLE?")

    def find_lag_mysteries(self):
        """Patrones con lags que no tienen sentido."""
        print("\n🔍 BUSCANDO LAGS MISTERIOSOS...")
        print("-" * 50)

        # Variables y sus lags "naturales" esperados
        expected_lags = {
            ('solar_flux', 'geomag_kp'): (1, 3),  # 1-3 días
            ('climate_pressure', 'climate_temperature'): (0, 1),  # Inmediato
        }

        for (var1, var2), (min_lag, max_lag) in expected_lags.items():
            if var1 not in self.data.columns or var2 not in self.data.columns:
                continue

            # Buscar correlación en lags inesperados
            for lag in range(10, 30):
                shifted = self.data[var1].shift(lag)
                corr = shifted.corr(self.data[var2])
                if abs(corr) > 0.4:
                    self.mysteries.append({
                        'type': 'lag_inesperado',
                        'var1': var1,
                        'var2': var2,
                        'lag': lag,
                        'correlation': corr,
                        'question': f"¿Por qué {var1} predice {var2} con {lag} pasos de anticipación?",
                    })
                    print(f"  🔮 {var1} → {var2} (lag={lag}): r={corr:.3f}")
                    print(f"     ¿Cómo puede predecir tan lejos en el futuro?")

    def find_disappearing_patterns(self):
        """Patrones que aparecen y desaparecen."""
        print("\n🔍 BUSCANDO PATRONES FANTASMA...")
        print("-" * 50)

        # Dividir datos en mitades
        mid = len(self.data) // 2
        first_half = self.data.iloc[:mid]
        second_half = self.data.iloc[mid:]

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        for col1 in numeric_cols[:5]:
            for col2 in numeric_cols[5:10]:
                corr1 = first_half[col1].corr(first_half[col2])
                corr2 = second_half[col1].corr(second_half[col2])

                # Patrón que cambia de signo
                if corr1 * corr2 < 0 and abs(corr1) > 0.3 and abs(corr2) > 0.3:
                    self.paradoxes.append({
                        'type': 'patrón_fantasma',
                        'var1': col1,
                        'var2': col2,
                        'corr_before': corr1,
                        'corr_after': corr2,
                        'question': f"¿Por qué la relación entre {col1} y {col2} se invirtió?",
                    })
                    print(f"  👻 {col1} ↔ {col2}")
                    print(f"     Primera mitad: r={corr1:.3f}")
                    print(f"     Segunda mitad: r={corr2:.3f}")
                    print(f"     ¡LA RELACIÓN SE INVIRTIÓ!")

    def find_cycle_paradoxes(self):
        """Ciclos que no coinciden con nada conocido."""
        print("\n🔍 BUSCANDO CICLOS INEXPLICABLES...")
        print("-" * 50)

        known_cycles = {
            24: 'día',
            168: 'semana',
            720: 'mes lunar',
            8760: 'año',
        }

        for col in self.data.select_dtypes(include=[np.number]).columns[:8]:
            # FFT para encontrar frecuencias dominantes
            try:
                values = self.data[col].dropna().values
                if len(values) < 50:
                    continue

                fft = np.fft.fft(values - np.mean(values))
                freqs = np.fft.fftfreq(len(values))

                # Encontrar picos
                magnitudes = np.abs(fft[1:len(fft)//2])
                peak_idx = np.argmax(magnitudes)
                peak_freq = abs(freqs[peak_idx + 1])

                if peak_freq > 0:
                    period = 1 / peak_freq

                    # ¿Coincide con algún ciclo conocido?
                    is_known = False
                    for known_period, name in known_cycles.items():
                        if 0.8 < period / known_period < 1.2:
                            is_known = True
                            break

                    if not is_known and 10 < period < 500:
                        self.mysteries.append({
                            'type': 'ciclo_desconocido',
                            'variable': col,
                            'period': period,
                            'question': f"¿Qué causa un ciclo de {period:.1f} pasos en {col}?",
                        })
                        print(f"  🌀 {col}: ciclo de {period:.1f} pasos")
                        print(f"     No coincide con día/semana/mes/año")
            except:
                pass

    def agent_discoveries(self):
        """Simular lo que los agentes habrían descubierto."""
        print("\n" + "=" * 70)
        print("🤖 DESCUBRIMIENTOS DE LOS AGENTES")
        print("=" * 70)

        agents = {
            'NEO': {'domain': 'crypto', 'curiosity': 0.8, 'discoveries': []},
            'EVA': {'domain': 'climate', 'curiosity': 0.6, 'discoveries': []},
            'ALEX': {'domain': 'solar', 'curiosity': 0.7, 'discoveries': []},
            'ADAM': {'domain': 'seismic', 'curiosity': 0.5, 'discoveries': []},
            'IRIS': {'domain': 'all', 'curiosity': 0.9, 'discoveries': []},
        }

        # Asignar descubrimientos según dominio
        for paradox in self.paradoxes + self.mysteries:
            for agent_name, agent in agents.items():
                domain = agent['domain']

                # ¿Es relevante para este agente?
                relevant = False
                var1 = paradox.get('var1', paradox.get('variable', ''))
                var2 = paradox.get('var2', '')

                if domain == 'all':
                    relevant = True
                elif domain in var1 or domain in var2:
                    relevant = True
                elif domain == 'crypto' and 'btc' in var1.lower():
                    relevant = True

                if relevant or np.random.random() < agent['curiosity'] * 0.3:
                    agent['discoveries'].append(paradox)

        # Reportar
        for agent_name, agent in agents.items():
            print(f"\n{agent_name} (curiosidad={agent['curiosity']}):")
            if agent['discoveries']:
                for d in agent['discoveries'][:3]:
                    print(f"  ❓ {d['question']}")
            else:
                print("  (sin descubrimientos significativos)")

    def philosophical_questions(self):
        """Preguntas que los agentes se harían."""
        print("\n" + "=" * 70)
        print("🧠 PREGUNTAS FILOSÓFICAS EMERGENTES")
        print("=" * 70)

        questions = [
            "¿Por qué algunos patrones predicen eventos que aún no han ocurrido?",
            "¿La correlación implica que hay una conexión real, o solo coincidencia?",
            "Si el sol afecta al magnetismo, ¿el magnetismo afecta a los humanos?",
            "¿Los mercados financieros responden a fuerzas cósmicas?",
            "¿Hay un orden oculto que conecta todo?",
            "¿O estamos viendo patrones donde no los hay?",
        ]

        for q in questions:
            print(f"  • {q}")

        if self.paradoxes:
            print("\n  🔥 PARADOJA MÁS PERTURBADORA:")
            p = self.paradoxes[0]
            print(f"     {p['question']}")
            print(f"     Esto desafía nuestra comprensión del mundo.")

    def run(self):
        """Ejecutar búsqueda completa."""
        print("=" * 70)
        print("CAZADOR DE PARADOJAS")
        print("¿Qué han descubierto los agentes que no pueden explicar?")
        print("=" * 70)

        self.find_impossible_correlations()
        self.find_lag_mysteries()
        self.find_disappearing_patterns()
        self.find_cycle_paradoxes()
        self.agent_discoveries()
        self.philosophical_questions()

        print("\n" + "=" * 70)
        print(f"RESUMEN: {len(self.paradoxes)} paradojas, {len(self.mysteries)} misterios")
        print("=" * 70)


if __name__ == '__main__':
    hunter = ParadoxHunter()
    hunter.run()
