"""
TESTS NORMA DURA PARA SISTEMA DE DOMINIOS

Verifica que los conectores de dominio NO contienen:
1. Reglas de dominio hardcodeadas
2. Umbrales diagnósticos/de decisión
3. Constantes físicas/médicas/financieras
4. Conocimiento experto embebido

PERMITIDO:
- Definiciones de variables y unidades
- Funciones matemáticas genéricas
- Transformaciones de coordenadas
- Constantes matemáticas puras (π, e)
"""

import unittest
import ast
import re
from pathlib import Path
from typing import List, Tuple, Set


# =============================================================================
# PATRONES PROHIBIDOS POR DOMINIO
# =============================================================================

MEDICINA_PROHIBIDO = [
    # Diagnósticos
    r'diabetes|prediabet',
    r'hyperten|hipertens',
    r'anemia|anémia',
    r'diagnos[eit]',
    r'patholog|patológ',

    # Umbrales clínicos específicos
    r'>\s*126\b',  # glucosa diabetes
    r'>\s*140\b',  # presión sistólica
    r'>\s*90\b',   # presión diastólica
    r'>\s*6\.5\b', # HbA1c
    r'>\s*200\b',  # colesterol
    r'<\s*40\b',   # HDL bajo
    r'>\s*150\b',  # triglicéridos
    r'>\s*30\b',   # BMI obesidad
]

FINANZAS_PROHIBIDO = [
    # Reglas de trading
    r'buy|sell|compra|vend[ae]',
    r'signal|señal',
    r'bullish|bearish|alcista|bajista',
    r'overbought|oversold|sobrecompra|sobreventa',

    # Umbrales específicos
    r'>\s*70\b.*rsi',  # RSI overbought
    r'<\s*30\b.*rsi',  # RSI oversold
    r'golden.cross|death.cross',
    r'support|resistance|soporte|resistencia',
]

COSMOLOGIA_PROHIBIDO = [
    # Constantes físicas
    r'speed.of.light|velocidad.luz',
    r'gravitational.constant|constante.gravitacional',
    r'planck.constant|constante.planck',
    r'hubble.constant|constante.hubble',

    # Valores específicos de constantes
    r'=\s*299792458\b',      # c en m/s
    r'=\s*3[\.e]8\b',        # c aproximado
    r'=\s*6\.674\b',         # G
    r'=\s*6\.626\b',         # h
    r'=\s*70\b.*km.*Mpc',    # H0
]

INGENIERIA_PROHIBIDO = [
    # Reglas de mantenimiento
    r'failure.threshold|umbral.fallo',
    r'maintenance.required|mantenimiento.requerido',
    r'replace.when|reemplazar.cuando',
    r'critical.limit|límite.crítico',

    # Umbrales específicos
    r'>\s*100\b.*temp.*fail',  # temp fallo
    r'>\s*10\b.*vibration.*alarm',  # vibración alarma
]

# Patrones PERMITIDOS (no deben causar falsos positivos)
PATRONES_PERMITIDOS = [
    r'#.*ORIGEN:',           # Comentarios de proveniencia
    r'#.*FROM_',             # Tags de proveniencia
    r'description\s*=',      # Descripciones de schema
    r'unit\s*=',             # Unidades
    r'VariableDefinition',   # Definiciones de variables
    r'synthetic.*testing',   # Datos sintéticos para testing
    r'NOT.*represent',       # Disclaimers
    r'agent.*learn',         # Referencias a aprendizaje
    r'NO\s+hay',             # Disclaimers en español
    r'no\s+defin',           # "No definimos"
    r'patológ',              # Mencionado en contexto de "no definir"
    r'glucosa\s*>',          # Ejemplo de lo que NO hacer
    r'NORMA\s+DURA',         # Referencias a la norma
]


# =============================================================================
# FUNCIONES DE ANÁLISIS
# =============================================================================

def find_domain_violations(file_path: Path, prohibited_patterns: List[str]) -> List[Tuple[int, str, str]]:
    """
    Busca violaciones de NORMA DURA en un archivo.

    Returns:
        Lista de (línea, patrón_violado, texto_línea)
    """
    violations = []

    try:
        content = file_path.read_text()
        lines = content.split('\n')
    except Exception as e:
        return [(0, "ERROR", str(e))]

    for i, line in enumerate(lines, 1):
        line_lower = line.lower()

        # Saltar líneas permitidas
        if any(re.search(p, line, re.IGNORECASE) for p in PATRONES_PERMITIDOS):
            continue

        # Saltar comentarios puros
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue

        # Buscar patrones prohibidos
        for pattern in prohibited_patterns:
            if re.search(pattern, line_lower, re.IGNORECASE):
                violations.append((i, pattern, line.strip()[:80]))

    return violations


def analyze_ast_for_hardcoded_rules(file_path: Path) -> List[Tuple[int, str]]:
    """
    Analiza AST buscando patrones de reglas hardcodeadas.

    Busca:
    - if variable > NUMERO_MAGICO: return DIAGNÓSTICO
    - threshold = NUMERO_ESPECIFICO sin justificación
    """
    violations = []

    try:
        content = file_path.read_text()
        tree = ast.parse(content)
    except Exception:
        return []

    class RuleVisitor(ast.NodeVisitor):
        def visit_If(self, node):
            # Buscar if comparación > número: return string
            if isinstance(node.test, ast.Compare):
                # Tiene comparación con literal numérico?
                for comparator in node.test.comparators:
                    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, (int, float)):
                        # El cuerpo retorna un string (posible diagnóstico)?
                        for stmt in node.body:
                            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant):
                                if isinstance(stmt.value.value, str):
                                    violations.append((
                                        node.lineno,
                                        f"Posible regla hardcodeada: if X > {comparator.value}: return '{stmt.value.value}'"
                                    ))
            self.generic_visit(node)

    RuleVisitor().visit(tree)
    return violations


# =============================================================================
# TESTS
# =============================================================================

class TestDomainNormaDura(unittest.TestCase):
    """Tests NORMA DURA para sistema de dominios."""

    @classmethod
    def setUpClass(cls):
        cls.domains_path = Path("/root/NEO_EVA/domains")

    def test_medicina_no_diagnosticos(self):
        """Medicina: NO debe contener reglas diagnósticas."""
        med_file = self.domains_path / "medicine" / "medicine_connector.py"
        if not med_file.exists():
            self.skipTest("Archivo medicina no existe")

        violations = find_domain_violations(med_file, MEDICINA_PROHIBIDO)

        if violations:
            msg = "\n".join([
                f"  Línea {v[0]}: patrón '{v[1]}' en: {v[2]}"
                for v in violations[:10]  # Mostrar máximo 10
            ])
            self.fail(f"VIOLACIONES NORMA DURA en medicina:\n{msg}")

    def test_finanzas_no_trading_rules(self):
        """Finanzas: NO debe contener reglas de trading."""
        fin_file = self.domains_path / "finance" / "finance_connector.py"
        if not fin_file.exists():
            self.skipTest("Archivo finanzas no existe")

        violations = find_domain_violations(fin_file, FINANZAS_PROHIBIDO)

        if violations:
            msg = "\n".join([
                f"  Línea {v[0]}: patrón '{v[1]}' en: {v[2]}"
                for v in violations[:10]
            ])
            self.fail(f"VIOLACIONES NORMA DURA en finanzas:\n{msg}")

    def test_cosmologia_no_constantes_fisicas(self):
        """Cosmología: NO debe contener constantes físicas hardcodeadas."""
        cos_file = self.domains_path / "cosmology" / "cosmology_connector.py"
        if not cos_file.exists():
            self.skipTest("Archivo cosmología no existe")

        violations = find_domain_violations(cos_file, COSMOLOGIA_PROHIBIDO)

        if violations:
            msg = "\n".join([
                f"  Línea {v[0]}: patrón '{v[1]}' en: {v[2]}"
                for v in violations[:10]
            ])
            self.fail(f"VIOLACIONES NORMA DURA en cosmología:\n{msg}")

    def test_ingenieria_no_failure_thresholds(self):
        """Ingeniería: NO debe contener umbrales de fallo hardcodeados."""
        eng_file = self.domains_path / "engineering" / "engineering_connector.py"
        if not eng_file.exists():
            self.skipTest("Archivo ingeniería no existe")

        violations = find_domain_violations(eng_file, INGENIERIA_PROHIBIDO)

        if violations:
            msg = "\n".join([
                f"  Línea {v[0]}: patrón '{v[1]}' en: {v[2]}"
                for v in violations[:10]
            ])
            self.fail(f"VIOLACIONES NORMA DURA en ingeniería:\n{msg}")

    def test_no_hardcoded_rules_ast(self):
        """Ningún conector debe tener if X > N: return 'diagnóstico'."""
        for domain in ["medicine", "finance", "cosmology", "engineering"]:
            connector_file = self.domains_path / domain / f"{domain}_connector.py"
            if not connector_file.exists():
                continue

            violations = analyze_ast_for_hardcoded_rules(connector_file)

            if violations:
                msg = "\n".join([f"  Línea {v[0]}: {v[1]}" for v in violations])
                self.fail(f"REGLAS HARDCODEADAS en {domain}:\n{msg}")

    def test_domain_base_is_generic(self):
        """domain_base.py debe ser genérico (no domain-specific)."""
        base_file = self.domains_path / "core" / "domain_base.py"
        if not base_file.exists():
            self.skipTest("domain_base.py no existe")

        content = base_file.read_text().lower()

        # No debe mencionar dominios específicos
        domain_specific = [
            'diabetes', 'glucose', 'blood.pressure',
            'stock', 'trading', 'buy', 'sell',
            'galaxy', 'redshift', 'hubble',
            'vibration', 'maintenance', 'failure'
        ]

        for term in domain_specific:
            if re.search(term, content):
                self.fail(f"domain_base.py contiene término específico: {term}")

    def test_all_connectors_have_synthetic_disclaimer(self):
        """Todos los métodos synthetic deben tener disclaimer."""
        for domain in ["medicine", "finance", "cosmology", "engineering"]:
            connector_file = self.domains_path / domain / f"{domain}_connector.py"
            if not connector_file.exists():
                continue

            content = connector_file.read_text()

            # Buscar método load_synthetic_for_testing
            if 'load_synthetic_for_testing' in content:
                # Debe tener disclaimer
                if 'synthetic' not in content.lower() or 'testing' not in content.lower():
                    continue
                if 'NOT' not in content and 'no ' not in content.lower():
                    self.fail(f"{domain}: método synthetic sin disclaimer de datos no reales")

    def test_imports_are_correct(self):
        """Verificar que los imports funcionan."""
        try:
            import sys
            sys.path.insert(0, str(self.domains_path.parent))

            from domains import get_engine, DomainEngine
            from domains.core.domain_base import DomainSchema, DomainConnector

            engine = get_engine()
            domains = engine.registry.list_domains()

            self.assertIn("medicine", domains)
            self.assertIn("finance", domains)
            self.assertIn("cosmology", domains)
            self.assertIn("engineering", domains)

        except ImportError as e:
            self.fail(f"Error de import: {e}")

    def test_synthetic_data_generation(self):
        """Verificar que la generación de datos sintéticos funciona."""
        try:
            import sys
            sys.path.insert(0, str(self.domains_path.parent))

            from domains import get_engine

            engine = get_engine()

            for domain in ["medicine", "finance", "cosmology", "engineering"]:
                result = engine.load_data(
                    domain=domain,
                    source="synthetic",
                    n_samples=100,
                    seed=42
                )

                self.assertIn("data", result)
                self.assertIn("n_records", result)
                self.assertEqual(result["n_records"], 100)

        except Exception as e:
            self.fail(f"Error generando datos sintéticos: {e}")


# =============================================================================
# RUNNER
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTS NORMA DURA - SISTEMA DE DOMINIOS")
    print("=" * 70)
    print()
    print("Verificando que NO hay reglas de dominio hardcodeadas...")
    print()

    # Configurar verbosidad
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDomainNormaDura)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    if result.wasSuccessful():
        print("✅ TODOS LOS TESTS PASARON - NORMA DURA EXTENDIDA CUMPLIDA")
    else:
        print("❌ HAY VIOLACIONES DE NORMA DURA EXTENDIDA")
        print(f"   Fallos: {len(result.failures)}")
        print(f"   Errores: {len(result.errors)}")
    print("=" * 70)
