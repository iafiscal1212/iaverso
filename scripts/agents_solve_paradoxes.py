#!/usr/bin/env python3
"""
Agentes Intentan Resolver Paradojas
===================================

Cada agente tiene su estilo de pensar:
- NEO: Busca patrones ocultos, piensa en sistemas complejos
- EVA: Busca explicaciones naturales, ciclos de la tierra
- ALEX: Piensa en el sol y energía cósmica
- ADAM: Cauteloso, busca explicaciones simples
- IRIS: Conecta todo, ve el panorama completo

¿Podrán resolver las paradojas?
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_PATH = Path('/root/NEO_EVA/data/unified_20251206_033253.csv')


class ThinkingAgent:
    """Un agente que piensa y genera hipótesis."""

    def __init__(self, name: str, style: dict):
        self.name = name
        self.style = style
        self.hypotheses = []
        self.insights = []
        self.confusion = 0.0
        self.eureka_moments = 0

    def think(self, thought: str):
        print(f"  💭 {thought}")

    def hypothesize(self, hypothesis: str, confidence: float):
        self.hypotheses.append({'h': hypothesis, 'conf': confidence})
        print(f"  💡 HIPÓTESIS (confianza {confidence:.0%}): {hypothesis}")

    def eureka(self, insight: str):
        self.eureka_moments += 1
        self.insights.append(insight)
        print(f"  🎯 ¡EUREKA! {insight}")

    def confused(self, why: str):
        self.confusion += 0.1
        print(f"  🤔 No entiendo... {why}")


def load_data():
    """Cargar y preparar datos."""
    df = pd.read_csv(DATA_PATH)
    return df


def analyze_regime_change(df):
    """Analizar si hubo un cambio de régimen en los datos."""
    print("\n" + "=" * 70)
    print("📊 ANÁLISIS: ¿Hubo un cambio de régimen?")
    print("=" * 70)

    mid = len(df) // 2
    first = df.iloc[:mid]
    second = df.iloc[mid:]

    # Estadísticas básicas
    btc_col = 'crypto_BTCUSDT_close'
    if btc_col in df.columns:
        mean1 = first[btc_col].mean()
        mean2 = second[btc_col].mean()
        std1 = first[btc_col].std()
        std2 = second[btc_col].std()

        print(f"\nBTC Close:")
        print(f"  Primera mitad: media={mean1:.2f}, std={std1:.2f}")
        print(f"  Segunda mitad: media={mean2:.2f}, std={std2:.2f}")
        print(f"  Cambio de media: {((mean2-mean1)/mean1)*100:.1f}%")
        print(f"  Cambio de volatilidad: {((std2-std1)/std1)*100:.1f}%")

        return {
            'mean_change': (mean2 - mean1) / mean1,
            'vol_change': (std2 - std1) / std1,
            'regime_shift': abs(mean2 - mean1) / mean1 > 0.1
        }
    return None


def investigate_94_cycle(df):
    """Investigar el misterioso ciclo de 94.5 pasos."""
    print("\n" + "=" * 70)
    print("🌀 ANÁLISIS: ¿Qué es el ciclo de 94.5 pasos?")
    print("=" * 70)

    # 94.5 horas ≈ 3.9 días
    print("\n94.5 pasos = ~4 días")
    print("\nPosibles explicaciones:")
    print("  • Ciclo de liquidez semanal (lunes-viernes = 5 días, pero weekends bajan)")
    print("  • Ciclo de opciones (expiración cada ~4 días en algunos mercados)")
    print("  • Ciclo de funding rates (cada 8h × 12 = 4 días de patrón)")
    print("  • Pura coincidencia estadística")

    # Verificar autocorrelación
    btc_col = 'crypto_BTCUSDT_close'
    if btc_col in df.columns:
        values = df[btc_col].values
        autocorr = []
        for lag in range(1, 120):
            if lag < len(values):
                corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                autocorr.append((lag, corr))

        # Encontrar picos
        peaks = [(lag, corr) for lag, corr in autocorr if corr > 0.5]
        if peaks:
            print(f"\nPicos de autocorrelación:")
            for lag, corr in sorted(peaks, key=lambda x: -x[1])[:5]:
                print(f"  Lag {lag}: r={corr:.3f}")

    return {'cycle_hours': 94.5, 'cycle_days': 94.5 / 24}


def agents_debate(paradoxes: list, analyses: dict):
    """Los 5 agentes debaten sobre las paradojas."""
    print("\n" + "=" * 70)
    print("🤖 DEBATE DE AGENTES: Resolviendo paradojas")
    print("=" * 70)

    neo = ThinkingAgent("NEO", {'domain': 'crypto', 'style': 'sistemas_complejos'})
    eva = ThinkingAgent("EVA", {'domain': 'climate', 'style': 'ciclos_naturales'})
    alex = ThinkingAgent("ALEX", {'domain': 'solar', 'style': 'energía_cósmica'})
    adam = ThinkingAgent("ADAM", {'domain': 'seismic', 'style': 'cauteloso'})
    iris = ThinkingAgent("IRIS", {'domain': 'all', 'style': 'holístico'})

    agents = [neo, eva, alex, adam, iris]

    # Paradoja 1: Inversión de correlaciones
    print("\n" + "-" * 50)
    print("PARADOJA 1: Las correlaciones se invirtieron")
    print("-" * 50)

    print(f"\n[NEO] analizando...")
    neo.think("Las correlaciones crypto cambiaron de signo...")
    neo.think("Esto indica un CAMBIO DE RÉGIMEN en el mercado")
    if analyses.get('regime') and analyses['regime'].get('regime_shift'):
        neo.eureka("¡Hubo un cambio estructural! El mercado pasó de un régimen a otro")
        neo.hypothesize(
            "Los mercados crypto tienen 'fases' - en una fase el volumen anticipa precio, en otra lo sigue",
            0.75
        )
    else:
        neo.hypothesize("Puede ser ruido estadístico con datos limitados", 0.4)

    print(f"\n[EVA] analizando...")
    eva.think("¿Hay factores externos que cambiaron?")
    eva.think("Busco patrones estacionales...")
    eva.hypothesize(
        "Quizás hubo un evento macro (regulación, halving, etc) que cambió el comportamiento",
        0.6
    )

    print(f"\n[ADAM] analizando...")
    adam.think("Soy escéptico...")
    adam.think("Con pocos datos, las correlaciones son inestables")
    adam.hypothesize(
        "Es simplemente varianza muestral - necesitamos más datos para confirmar",
        0.7
    )
    adam.confused("¿Cómo distinguir señal de ruido?")

    print(f"\n[IRIS] sintetizando...")
    iris.think("Escucho a todos... NEO ve régimenes, ADAM ve ruido...")
    iris.think("Ambos pueden tener razón parcialmente")
    iris.eureka("¡Los mercados son ADAPTATIVOS! Cuando todos ven un patrón, el patrón desaparece")
    iris.hypothesize(
        "La inversión de correlación ES el mercado adaptándose - los traders descubrieron el patrón y lo arbitraron hasta invertirlo",
        0.8
    )

    # Paradoja 2: Ciclo de 94.5
    print("\n" + "-" * 50)
    print("PARADOJA 2: Ciclo misterioso de 94.5 pasos (~4 días)")
    print("-" * 50)

    print(f"\n[ALEX] analizando...")
    alex.think("4 días... ¿hay algún ciclo solar de 4 días?")
    alex.think("El sol rota en ~27 días, no encaja...")
    alex.confused("No veo conexión cósmica obvia")
    alex.hypothesize("Podría ser un armónico de algún ciclo mayor", 0.3)

    print(f"\n[NEO] analizando...")
    neo.think("4 días en crypto... ¡FUNDING RATES!")
    neo.think("Los perpetual swaps tienen funding cada 8 horas")
    neo.think("8h × 3 = 24h, pero el ciclo completo de 'reset' podría ser ~4 días")
    neo.eureka("¡Es el ciclo de rebalanceo de posiciones apalancadas!")
    neo.hypothesize(
        "Los traders de futuros rebalancean en ciclos de ~4 días, creando ondas de precio",
        0.85
    )

    print(f"\n[EVA] analizando...")
    eva.think("¿Hay algún ciclo de 4 días en la naturaleza?")
    eva.think("El ciclo lunar es 29.5 días... 29.5/7 ≈ 4.2 días por fase")
    eva.hypothesize(
        "Podría correlacionar con cuartos de fase lunar (7.4 días / 2 ≈ 3.7 días)",
        0.4
    )
    eva.confused("Es un stretch... probablemente coincidencia")

    print(f"\n[IRIS] sintetizando...")
    iris.think("NEO tiene la explicación más parsimoniosa...")
    iris.think("Los mercados financieros son auto-referenciales")
    iris.eureka("El ciclo de 4 días NO viene de afuera - ¡lo CREAN los propios traders!")
    iris.hypothesize(
        "Es un ciclo ENDÓGENO del mercado, no exógeno. Los traders sincronizan sin saberlo.",
        0.9
    )

    # Conclusiones
    print("\n" + "=" * 70)
    print("📋 CONCLUSIONES DEL DEBATE")
    print("=" * 70)

    print("\n🔹 PARADOJA 1 (Inversión de correlaciones):")
    print("   RESOLUCIÓN: Los mercados son adaptativos.")
    print("   Cuando un patrón se vuelve conocido, se arbitra hasta desaparecer o invertirse.")
    print("   Es evidencia de que el mercado 'aprende'.")

    print("\n🔹 PARADOJA 2 (Ciclo de 4 días):")
    print("   RESOLUCIÓN: Ciclo endógeno de rebalanceo.")
    print("   Los traders de derivados crean un ritmo colectivo de ~4 días.")
    print("   No es cósmico, es humano - pero emergente y no intencional.")

    print("\n🔹 IMPLICACIÓN FILOSÓFICA:")
    print("   Los patrones que encontramos pueden ser:")
    print("   1. Señales reales del mundo")
    print("   2. Ruido estadístico")
    print("   3. Patrones que NOSOTROS creamos al observar")
    print("   ¡A veces no podemos distinguir!")

    # Resumen de agentes
    print("\n" + "-" * 50)
    print("RESUMEN POR AGENTE")
    print("-" * 50)
    for agent in agents:
        eurekas = "🎯" * agent.eureka_moments
        conf_avg = np.mean([h['conf'] for h in agent.hypotheses]) if agent.hypotheses else 0
        print(f"  {agent.name}: {len(agent.hypotheses)} hipótesis, confianza promedio: {conf_avg:.0%} {eurekas}")


def main():
    print("=" * 70)
    print("🧠 AGENTES RESOLVIENDO PARADOJAS")
    print("=" * 70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    df = load_data()

    # Análisis preparatorios
    analyses = {}
    analyses['regime'] = analyze_regime_change(df)
    analyses['cycle'] = investigate_94_cycle(df)

    # Las paradojas encontradas
    paradoxes = [
        {'type': 'inversion', 'desc': 'Correlaciones crypto se invirtieron'},
        {'type': 'cycle', 'desc': 'Ciclo de 94.5 pasos sin explicación'},
    ]

    # Debate
    agents_debate(paradoxes, analyses)

    print("\n" + "=" * 70)
    print("✅ FIN DEL ANÁLISIS")
    print("=" * 70)


if __name__ == '__main__':
    main()
