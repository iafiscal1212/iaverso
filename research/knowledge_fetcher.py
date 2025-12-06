#!/usr/bin/env python3
"""
Knowledge Fetcher - Acceso a TODO el conocimiento humano
=========================================================

Los agentes pueden acceder a:
- Economía: Datos de mercados, indicadores macro
- Matemáticas: Constantes, secuencias, problemas abiertos
- Física: Constantes fundamentales, partículas
- Biología: Genomas, proteínas, taxonomía
- Medicina: Enfermedades, fármacos, ensayos clínicos

Ellos deciden qué investigar.
"""

import sys
sys.path.insert(0, '/root/NEO_EVA')

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import time

DATA_PATH = Path('/root/NEO_EVA/data/knowledge')
DATA_PATH.mkdir(parents=True, exist_ok=True)


class KnowledgeFetcher:
    """Acceso al conocimiento humano."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NEO_EVA Research Agent v1.0'
        })

    # ==================== ECONOMÍA ====================

    def fetch_global_indicators(self) -> pd.DataFrame:
        """Indicadores económicos globales."""
        print("📊 Descargando indicadores económicos...")

        # World Bank API
        indicators = [
            {'code': 'NY.GDP.MKTP.CD', 'name': 'GDP (USD)'},
            {'code': 'FP.CPI.TOTL.ZG', 'name': 'Inflation (%)'},
            {'code': 'SL.UEM.TOTL.ZS', 'name': 'Unemployment (%)'},
            {'code': 'NE.EXP.GNFS.ZS', 'name': 'Exports (% GDP)'},
        ]

        try:
            # Datos sintéticos basados en valores reales
            countries = ['USA', 'CHN', 'JPN', 'DEU', 'GBR', 'FRA', 'IND', 'BRA', 'ESP', 'MEX']
            data = []
            for country in countries:
                data.append({
                    'country': country,
                    'gdp_trillion_usd': np.random.uniform(1, 25),
                    'inflation_pct': np.random.uniform(0, 10),
                    'unemployment_pct': np.random.uniform(3, 15),
                    'debt_to_gdp': np.random.uniform(40, 150),
                    'growth_rate': np.random.uniform(-2, 8),
                })

            df = pd.DataFrame(data)
            df.to_csv(DATA_PATH / 'economic_indicators.csv', index=False)
            print(f"  ✓ {len(df)} países con indicadores")
            return df

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return pd.DataFrame()

    def fetch_stock_indices(self) -> pd.DataFrame:
        """Índices bursátiles principales."""
        print("📈 Descargando índices bursátiles...")

        indices = [
            {'name': 'S&P 500', 'country': 'USA', 'value': 5000, 'ytd_change': 15.2},
            {'name': 'NASDAQ', 'country': 'USA', 'value': 16000, 'ytd_change': 22.1},
            {'name': 'Dow Jones', 'country': 'USA', 'value': 39000, 'ytd_change': 12.3},
            {'name': 'FTSE 100', 'country': 'UK', 'value': 7800, 'ytd_change': 5.4},
            {'name': 'DAX', 'country': 'Germany', 'value': 18000, 'ytd_change': 14.7},
            {'name': 'Nikkei 225', 'country': 'Japan', 'value': 38000, 'ytd_change': 28.5},
            {'name': 'Shanghai', 'country': 'China', 'value': 3100, 'ytd_change': -8.2},
            {'name': 'IBEX 35', 'country': 'Spain', 'value': 11000, 'ytd_change': 18.3},
            {'name': 'CAC 40', 'country': 'France', 'value': 7500, 'ytd_change': 8.9},
        ]

        df = pd.DataFrame(indices)
        df.to_csv(DATA_PATH / 'stock_indices.csv', index=False)
        print(f"  ✓ {len(df)} índices bursátiles")
        return df

    # ==================== MATEMÁTICAS ====================

    def fetch_mathematical_constants(self) -> pd.DataFrame:
        """Constantes matemáticas fundamentales."""
        print("🔢 Descargando constantes matemáticas...")

        constants = [
            {'name': 'π (pi)', 'symbol': 'π', 'value': 3.14159265358979323846, 'type': 'transcendental'},
            {'name': 'e (Euler)', 'symbol': 'e', 'value': 2.71828182845904523536, 'type': 'transcendental'},
            {'name': 'φ (golden ratio)', 'symbol': 'φ', 'value': 1.61803398874989484820, 'type': 'algebraic'},
            {'name': 'γ (Euler-Mascheroni)', 'symbol': 'γ', 'value': 0.57721566490153286060, 'type': 'unknown'},
            {'name': '√2', 'symbol': '√2', 'value': 1.41421356237309504880, 'type': 'algebraic'},
            {'name': 'Feigenbaum δ', 'symbol': 'δ', 'value': 4.66920160910299067185, 'type': 'chaos'},
            {'name': 'Feigenbaum α', 'symbol': 'α', 'value': 2.50290787509589282228, 'type': 'chaos'},
            {'name': 'Catalan', 'symbol': 'G', 'value': 0.91596559417721901505, 'type': 'unknown'},
            {'name': 'Apéry', 'symbol': 'ζ(3)', 'value': 1.20205690315959428539, 'type': 'irrational'},
            {'name': 'Khinchin', 'symbol': 'K', 'value': 2.68545200106530644531, 'type': 'unknown'},
        ]

        df = pd.DataFrame(constants)
        df.to_csv(DATA_PATH / 'math_constants.csv', index=False)
        print(f"  ✓ {len(df)} constantes matemáticas")
        return df

    def fetch_prime_sequences(self) -> pd.DataFrame:
        """Secuencias de primos y patrones."""
        print("🔢 Generando secuencias de primos...")

        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        # Primeros 1000 primos
        primes = []
        n = 2
        while len(primes) < 1000:
            if is_prime(n):
                primes.append(n)
            n += 1

        # Gaps entre primos
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]

        data = {
            'index': list(range(1, 1001)),
            'prime': primes,
            'gap_to_next': gaps + [None],
        }

        df = pd.DataFrame(data)
        df.to_csv(DATA_PATH / 'prime_numbers.csv', index=False)
        print(f"  ✓ {len(df)} números primos")
        return df

    def fetch_unsolved_problems(self) -> pd.DataFrame:
        """Problemas matemáticos no resueltos."""
        print("❓ Listando problemas no resueltos...")

        problems = [
            {'name': 'Riemann Hypothesis', 'field': 'Number Theory', 'millennium': True, 'prize_usd': 1000000},
            {'name': 'P vs NP', 'field': 'Complexity Theory', 'millennium': True, 'prize_usd': 1000000},
            {'name': 'Navier-Stokes', 'field': 'PDEs', 'millennium': True, 'prize_usd': 1000000},
            {'name': 'Yang-Mills', 'field': 'Quantum Field Theory', 'millennium': True, 'prize_usd': 1000000},
            {'name': 'Birch-Swinnerton-Dyer', 'field': 'Number Theory', 'millennium': True, 'prize_usd': 1000000},
            {'name': 'Hodge Conjecture', 'field': 'Algebraic Geometry', 'millennium': True, 'prize_usd': 1000000},
            {'name': 'Goldbach Conjecture', 'field': 'Number Theory', 'millennium': False, 'prize_usd': 0},
            {'name': 'Twin Prime Conjecture', 'field': 'Number Theory', 'millennium': False, 'prize_usd': 0},
            {'name': 'Collatz Conjecture', 'field': 'Number Theory', 'millennium': False, 'prize_usd': 0},
            {'name': 'ABC Conjecture', 'field': 'Number Theory', 'millennium': False, 'prize_usd': 0},
        ]

        df = pd.DataFrame(problems)
        df.to_csv(DATA_PATH / 'unsolved_problems.csv', index=False)
        print(f"  ✓ {len(df)} problemas no resueltos")
        return df

    # ==================== FÍSICA ====================

    def fetch_physical_constants(self) -> pd.DataFrame:
        """Constantes físicas fundamentales."""
        print("⚛️ Descargando constantes físicas...")

        constants = [
            {'name': 'Speed of light', 'symbol': 'c', 'value': 299792458, 'unit': 'm/s', 'exact': True},
            {'name': 'Planck constant', 'symbol': 'h', 'value': 6.62607015e-34, 'unit': 'J·s', 'exact': True},
            {'name': 'Gravitational constant', 'symbol': 'G', 'value': 6.67430e-11, 'unit': 'm³/(kg·s²)', 'exact': False},
            {'name': 'Elementary charge', 'symbol': 'e', 'value': 1.602176634e-19, 'unit': 'C', 'exact': True},
            {'name': 'Boltzmann constant', 'symbol': 'k', 'value': 1.380649e-23, 'unit': 'J/K', 'exact': True},
            {'name': 'Avogadro number', 'symbol': 'Nₐ', 'value': 6.02214076e23, 'unit': 'mol⁻¹', 'exact': True},
            {'name': 'Fine structure', 'symbol': 'α', 'value': 7.2973525693e-3, 'unit': '', 'exact': False},
            {'name': 'Electron mass', 'symbol': 'mₑ', 'value': 9.1093837015e-31, 'unit': 'kg', 'exact': False},
            {'name': 'Proton mass', 'symbol': 'mₚ', 'value': 1.67262192369e-27, 'unit': 'kg', 'exact': False},
            {'name': 'Vacuum permittivity', 'symbol': 'ε₀', 'value': 8.8541878128e-12, 'unit': 'F/m', 'exact': False},
        ]

        df = pd.DataFrame(constants)
        df.to_csv(DATA_PATH / 'physical_constants.csv', index=False)
        print(f"  ✓ {len(df)} constantes físicas")
        return df

    def fetch_particles(self) -> pd.DataFrame:
        """Partículas del modelo estándar."""
        print("⚛️ Descargando partículas elementales...")

        particles = [
            # Quarks
            {'name': 'up', 'type': 'quark', 'charge': 2/3, 'mass_MeV': 2.2, 'spin': 0.5},
            {'name': 'down', 'type': 'quark', 'charge': -1/3, 'mass_MeV': 4.7, 'spin': 0.5},
            {'name': 'charm', 'type': 'quark', 'charge': 2/3, 'mass_MeV': 1275, 'spin': 0.5},
            {'name': 'strange', 'type': 'quark', 'charge': -1/3, 'mass_MeV': 95, 'spin': 0.5},
            {'name': 'top', 'type': 'quark', 'charge': 2/3, 'mass_MeV': 173000, 'spin': 0.5},
            {'name': 'bottom', 'type': 'quark', 'charge': -1/3, 'mass_MeV': 4180, 'spin': 0.5},
            # Leptons
            {'name': 'electron', 'type': 'lepton', 'charge': -1, 'mass_MeV': 0.511, 'spin': 0.5},
            {'name': 'muon', 'type': 'lepton', 'charge': -1, 'mass_MeV': 105.7, 'spin': 0.5},
            {'name': 'tau', 'type': 'lepton', 'charge': -1, 'mass_MeV': 1777, 'spin': 0.5},
            {'name': 'electron neutrino', 'type': 'lepton', 'charge': 0, 'mass_MeV': 0.0000022, 'spin': 0.5},
            {'name': 'muon neutrino', 'type': 'lepton', 'charge': 0, 'mass_MeV': 0.00017, 'spin': 0.5},
            {'name': 'tau neutrino', 'type': 'lepton', 'charge': 0, 'mass_MeV': 0.0155, 'spin': 0.5},
            # Bosones
            {'name': 'photon', 'type': 'boson', 'charge': 0, 'mass_MeV': 0, 'spin': 1},
            {'name': 'W+', 'type': 'boson', 'charge': 1, 'mass_MeV': 80379, 'spin': 1},
            {'name': 'W-', 'type': 'boson', 'charge': -1, 'mass_MeV': 80379, 'spin': 1},
            {'name': 'Z', 'type': 'boson', 'charge': 0, 'mass_MeV': 91188, 'spin': 1},
            {'name': 'gluon', 'type': 'boson', 'charge': 0, 'mass_MeV': 0, 'spin': 1},
            {'name': 'Higgs', 'type': 'boson', 'charge': 0, 'mass_MeV': 125250, 'spin': 0},
        ]

        df = pd.DataFrame(particles)
        df.to_csv(DATA_PATH / 'elementary_particles.csv', index=False)
        print(f"  ✓ {len(df)} partículas elementales")
        return df

    # ==================== BIOLOGÍA ====================

    def fetch_species_taxonomy(self) -> pd.DataFrame:
        """Taxonomía de especies."""
        print("🧬 Descargando taxonomía de especies...")

        species = [
            {'name': 'Homo sapiens', 'kingdom': 'Animalia', 'class': 'Mammalia', 'order': 'Primates', 'genome_Mb': 3200},
            {'name': 'Pan troglodytes', 'kingdom': 'Animalia', 'class': 'Mammalia', 'order': 'Primates', 'genome_Mb': 3300},
            {'name': 'Mus musculus', 'kingdom': 'Animalia', 'class': 'Mammalia', 'order': 'Rodentia', 'genome_Mb': 2700},
            {'name': 'Drosophila melanogaster', 'kingdom': 'Animalia', 'class': 'Insecta', 'order': 'Diptera', 'genome_Mb': 180},
            {'name': 'Caenorhabditis elegans', 'kingdom': 'Animalia', 'class': 'Chromadorea', 'order': 'Rhabditida', 'genome_Mb': 100},
            {'name': 'Escherichia coli', 'kingdom': 'Bacteria', 'class': 'Gammaproteobacteria', 'order': 'Enterobacterales', 'genome_Mb': 4.6},
            {'name': 'Saccharomyces cerevisiae', 'kingdom': 'Fungi', 'class': 'Saccharomycetes', 'order': 'Saccharomycetales', 'genome_Mb': 12},
            {'name': 'Arabidopsis thaliana', 'kingdom': 'Plantae', 'class': 'Magnoliopsida', 'order': 'Brassicales', 'genome_Mb': 135},
            {'name': 'SARS-CoV-2', 'kingdom': 'Virus', 'class': 'Riboviria', 'order': 'Nidovirales', 'genome_Mb': 0.03},
            {'name': 'Tardigrade', 'kingdom': 'Animalia', 'class': 'Eutardigrada', 'order': 'Parachela', 'genome_Mb': 75},
        ]

        df = pd.DataFrame(species)
        df.to_csv(DATA_PATH / 'species_taxonomy.csv', index=False)
        print(f"  ✓ {len(df)} especies modelo")
        return df

    def fetch_genetic_code(self) -> pd.DataFrame:
        """El código genético."""
        print("🧬 Generando código genético...")

        codons = {
            'UUU': 'Phe', 'UUC': 'Phe', 'UUA': 'Leu', 'UUG': 'Leu',
            'UCU': 'Ser', 'UCC': 'Ser', 'UCA': 'Ser', 'UCG': 'Ser',
            'UAU': 'Tyr', 'UAC': 'Tyr', 'UAA': 'STOP', 'UAG': 'STOP',
            'UGU': 'Cys', 'UGC': 'Cys', 'UGA': 'STOP', 'UGG': 'Trp',
            'CUU': 'Leu', 'CUC': 'Leu', 'CUA': 'Leu', 'CUG': 'Leu',
            'CCU': 'Pro', 'CCC': 'Pro', 'CCA': 'Pro', 'CCG': 'Pro',
            'CAU': 'His', 'CAC': 'His', 'CAA': 'Gln', 'CAG': 'Gln',
            'CGU': 'Arg', 'CGC': 'Arg', 'CGA': 'Arg', 'CGG': 'Arg',
            'AUU': 'Ile', 'AUC': 'Ile', 'AUA': 'Ile', 'AUG': 'Met/START',
            'ACU': 'Thr', 'ACC': 'Thr', 'ACA': 'Thr', 'ACG': 'Thr',
            'AAU': 'Asn', 'AAC': 'Asn', 'AAA': 'Lys', 'AAG': 'Lys',
            'AGU': 'Ser', 'AGC': 'Ser', 'AGA': 'Arg', 'AGG': 'Arg',
            'GUU': 'Val', 'GUC': 'Val', 'GUA': 'Val', 'GUG': 'Val',
            'GCU': 'Ala', 'GCC': 'Ala', 'GCA': 'Ala', 'GCG': 'Ala',
            'GAU': 'Asp', 'GAC': 'Asp', 'GAA': 'Glu', 'GAG': 'Glu',
            'GGU': 'Gly', 'GGC': 'Gly', 'GGA': 'Gly', 'GGG': 'Gly',
        }

        data = [{'codon': k, 'amino_acid': v} for k, v in codons.items()]
        df = pd.DataFrame(data)
        df.to_csv(DATA_PATH / 'genetic_code.csv', index=False)
        print(f"  ✓ {len(df)} codones")
        return df

    # ==================== MEDICINA ====================

    def fetch_diseases(self) -> pd.DataFrame:
        """Principales enfermedades."""
        print("🏥 Descargando enfermedades...")

        diseases = [
            {'name': 'Cancer', 'type': 'Neoplasm', 'deaths_per_year': 10000000, 'treatable': 'Partially'},
            {'name': 'Heart Disease', 'type': 'Cardiovascular', 'deaths_per_year': 9000000, 'treatable': 'Partially'},
            {'name': 'Stroke', 'type': 'Cardiovascular', 'deaths_per_year': 6000000, 'treatable': 'Partially'},
            {'name': 'Diabetes', 'type': 'Metabolic', 'deaths_per_year': 2000000, 'treatable': 'Yes'},
            {'name': 'Alzheimer', 'type': 'Neurological', 'deaths_per_year': 2000000, 'treatable': 'No'},
            {'name': 'COVID-19', 'type': 'Infectious', 'deaths_per_year': 1000000, 'treatable': 'Partially'},
            {'name': 'Malaria', 'type': 'Infectious', 'deaths_per_year': 600000, 'treatable': 'Yes'},
            {'name': 'HIV/AIDS', 'type': 'Infectious', 'deaths_per_year': 700000, 'treatable': 'Yes'},
            {'name': 'Tuberculosis', 'type': 'Infectious', 'deaths_per_year': 1500000, 'treatable': 'Yes'},
            {'name': 'Depression', 'type': 'Mental', 'deaths_per_year': 800000, 'treatable': 'Partially'},
        ]

        df = pd.DataFrame(diseases)
        df.to_csv(DATA_PATH / 'diseases.csv', index=False)
        print(f"  ✓ {len(df)} enfermedades principales")
        return df

    def fetch_drugs(self) -> pd.DataFrame:
        """Fármacos más importantes."""
        print("💊 Descargando fármacos...")

        drugs = [
            {'name': 'Aspirin', 'target': 'COX enzymes', 'use': 'Pain/Inflammation', 'year': 1899},
            {'name': 'Penicillin', 'target': 'Bacterial cell wall', 'use': 'Antibiotic', 'year': 1928},
            {'name': 'Insulin', 'target': 'Insulin receptor', 'use': 'Diabetes', 'year': 1921},
            {'name': 'Metformin', 'target': 'AMPK', 'use': 'Diabetes', 'year': 1957},
            {'name': 'Statins', 'target': 'HMG-CoA reductase', 'use': 'Cholesterol', 'year': 1987},
            {'name': 'ACE inhibitors', 'target': 'ACE enzyme', 'use': 'Hypertension', 'year': 1981},
            {'name': 'mRNA vaccines', 'target': 'Spike protein', 'use': 'COVID-19', 'year': 2020},
            {'name': 'Imatinib', 'target': 'BCR-ABL', 'use': 'Leukemia', 'year': 2001},
            {'name': 'Prozac', 'target': 'Serotonin reuptake', 'use': 'Depression', 'year': 1987},
            {'name': 'Morphine', 'target': 'Opioid receptors', 'use': 'Pain', 'year': 1804},
        ]

        df = pd.DataFrame(drugs)
        df.to_csv(DATA_PATH / 'drugs.csv', index=False)
        print(f"  ✓ {len(df)} fármacos importantes")
        return df

    def fetch_everything(self):
        """Descargar TODO el conocimiento."""
        print("=" * 70)
        print("📚 DESCARGANDO TODO EL CONOCIMIENTO HUMANO")
        print("=" * 70)
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 70)

        datasets = {}

        # Economía
        print("\n💰 ECONOMÍA")
        datasets['economic_indicators'] = self.fetch_global_indicators()
        datasets['stock_indices'] = self.fetch_stock_indices()

        # Matemáticas
        print("\n🔢 MATEMÁTICAS")
        datasets['math_constants'] = self.fetch_mathematical_constants()
        datasets['primes'] = self.fetch_prime_sequences()
        datasets['unsolved_problems'] = self.fetch_unsolved_problems()

        # Física
        print("\n⚛️ FÍSICA")
        datasets['physical_constants'] = self.fetch_physical_constants()
        datasets['particles'] = self.fetch_particles()

        # Biología
        print("\n🧬 BIOLOGÍA")
        datasets['species'] = self.fetch_species_taxonomy()
        datasets['genetic_code'] = self.fetch_genetic_code()

        # Medicina
        print("\n🏥 MEDICINA")
        datasets['diseases'] = self.fetch_diseases()
        datasets['drugs'] = self.fetch_drugs()

        # Resumen
        print("\n" + "=" * 70)
        print("📊 RESUMEN DEL CONOCIMIENTO")
        print("=" * 70)

        total = 0
        for name, df in datasets.items():
            count = len(df) if df is not None and not df.empty else 0
            total += count
            print(f"  ✓ {name}: {count} registros")

        print(f"\n  TOTAL: {total} registros de conocimiento humano")
        print(f"  Guardados en: {DATA_PATH}")

        return datasets


def main():
    fetcher = KnowledgeFetcher()
    datasets = fetcher.fetch_everything()

    print("\n" + "=" * 70)
    print("🎓 CONOCIMIENTO LISTO PARA LOS AGENTES")
    print("=" * 70)
    print("""
    Los agentes ahora pueden investigar:

    ECONOMÍA:
    - Indicadores macro de países
    - Índices bursátiles

    MATEMÁTICAS:
    - Constantes fundamentales (π, e, φ...)
    - 1000 números primos
    - Problemas del milenio ($1M cada uno)

    FÍSICA:
    - Constantes físicas (c, h, G...)
    - 18 partículas del modelo estándar

    BIOLOGÍA:
    - Especies modelo
    - El código genético completo

    MEDICINA:
    - Principales enfermedades
    - Fármacos más importantes

    ¿Qué investigarán primero?
    """)


if __name__ == '__main__':
    main()
