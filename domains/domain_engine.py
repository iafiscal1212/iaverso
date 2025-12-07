"""
MOTOR DE DOMINIOS UNIFICADO

NORMA DURA EXTENDIDA - Principio Central:
============================================
Claude es el ARQUITECTO de infraestructura, NO el experto de dominio.
Los agentes APRENDEN de los datos, no de reglas hardcodeadas.

Este motor proporciona:
1. Registro y gestión de dominios
2. Carga de datos multi-dominio
3. Análisis cross-domain
4. Sistema de hipótesis y falsificación

NO proporciona:
- Reglas específicas de dominio
- Umbrales predefinidos
- Conocimiento experto hardcodeado
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type, Callable
from pathlib import Path
import numpy as np

from .core.domain_base import (
    DomainConnector, DomainSchema, DomainAnalyzer,
    Hypothesis, HypothesisEngine
)

# Import domain-specific connectors
from .medicine.medicine_connector import MedicineConnector
from .finance.finance_connector import FinanceConnector
from .cosmology.cosmology_connector import CosmologyConnector
from .engineering.engineering_connector import EngineeringConnector


# =============================================================================
# REGISTRO DE DOMINIOS
# =============================================================================

@dataclass
class DomainRegistry:
    """
    Registro central de todos los dominios disponibles.

    Permite:
    - Registrar nuevos dominios dinámicamente
    - Obtener conectores por nombre
    - Listar dominios disponibles
    """

    _connectors: Dict[str, Type[DomainConnector]] = field(default_factory=dict)
    _instances: Dict[str, DomainConnector] = field(default_factory=dict)

    def register(self, name: str, connector_class: Type[DomainConnector]) -> None:
        """Registra un nuevo dominio."""
        self._connectors[name] = connector_class

    def get_connector(self, name: str) -> DomainConnector:
        """
        Obtiene instancia de conector para un dominio.

        Crea la instancia si no existe (singleton por dominio).
        """
        if name not in self._connectors:
            raise ValueError(f"Dominio '{name}' no registrado. Disponibles: {list(self._connectors.keys())}")

        if name not in self._instances:
            self._instances[name] = self._connectors[name]()

        return self._instances[name]

    def list_domains(self) -> List[str]:
        """Lista todos los dominios registrados."""
        return list(self._connectors.keys())

    def get_schema(self, name: str) -> DomainSchema:
        """Obtiene el schema principal de un dominio."""
        connector = self.get_connector(name)
        # Buscar atributo schema o el primero que termine en _schema
        if hasattr(connector, 'schema'):
            return connector.schema
        for attr in dir(connector):
            if attr.endswith('_schema'):
                return getattr(connector, attr)
        raise AttributeError(f"Conector {name} no tiene schema definido")


# =============================================================================
# MOTOR DE DOMINIOS
# =============================================================================

class DomainEngine:
    """
    Motor central para operaciones cross-domain.

    Proporciona:
    - Carga unificada de datos
    - Análisis cross-domain
    - Sistema de hipótesis multi-dominio
    """

    def __init__(self):
        self.registry = DomainRegistry()
        self._analyzers: Dict[str, DomainAnalyzer] = {}  # Cache de analyzers por dominio
        self._hypothesis_engines: Dict[str, HypothesisEngine] = {}  # Cache de hypothesis engines
        self._loaded_data: Dict[str, Any] = {}

        # Registrar dominios por defecto
        self._register_default_domains()

    def _get_analyzer(self, domain: str) -> DomainAnalyzer:
        """Obtiene o crea un analyzer para un dominio."""
        if domain not in self._analyzers:
            schema = self.registry.get_schema(domain)
            self._analyzers[domain] = DomainAnalyzer(schema)
        return self._analyzers[domain]

    def _get_hypothesis_engine(self, domain: str) -> HypothesisEngine:
        """Obtiene o crea un hypothesis engine para un dominio."""
        if domain not in self._hypothesis_engines:
            analyzer = self._get_analyzer(domain)
            self._hypothesis_engines[domain] = HypothesisEngine(analyzer)
        return self._hypothesis_engines[domain]

    def _register_default_domains(self) -> None:
        """Registra los dominios incluidos por defecto."""
        self.registry.register("medicine", MedicineConnector)
        self.registry.register("finance", FinanceConnector)
        self.registry.register("cosmology", CosmologyConnector)
        self.registry.register("engineering", EngineeringConnector)

    # =========================================================================
    # CARGA DE DATOS
    # =========================================================================

    def load_data(
        self,
        domain: str,
        source: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Carga datos de un dominio específico.

        Args:
            domain: Nombre del dominio ("medicine", "finance", etc.)
            source: Tipo de fuente ("csv", "synthetic", API específica)
            **kwargs: Parámetros específicos de la fuente

        Returns:
            Dict con datos cargados y metadata
        """
        connector = self.registry.get_connector(domain)

        if source == "csv":
            path = kwargs.get("path")
            if path is None:
                raise ValueError("CSV source requires 'path' parameter")
            result = connector.load_csv(Path(path), kwargs.get("column_mapping"))

        elif source == "synthetic":
            n_samples = kwargs.get("n_samples", 1000)
            seed = kwargs.get("seed")
            result = connector.load_synthetic_for_testing(n_samples, seed=seed)

        elif hasattr(connector, f"load_from_{source}"):
            # Llamar método específico del conector (load_from_yahoo, load_from_sdss, etc.)
            method = getattr(connector, f"load_from_{source}")
            result = method(**kwargs)

        else:
            raise ValueError(f"Fuente '{source}' no soportada para dominio '{domain}'")

        # Almacenar datos cargados
        data_key = f"{domain}_{source}_{len(self._loaded_data)}"
        self._loaded_data[data_key] = result

        return {
            "data_key": data_key,
            **result
        }

    def get_loaded_data(self, data_key: str) -> Any:
        """Recupera datos previamente cargados."""
        if data_key not in self._loaded_data:
            raise KeyError(f"Datos '{data_key}' no encontrados. Claves disponibles: {list(self._loaded_data.keys())}")
        return self._loaded_data[data_key]

    # =========================================================================
    # ANÁLISIS
    # =========================================================================

    def analyze(
        self,
        data_key: str,
        analysis_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ejecuta análisis genérico sobre datos cargados.

        Args:
            data_key: Clave de datos cargados
            analysis_type: Tipo de análisis
                - "correlation": Matriz de correlaciones
                - "clusters": Detección de clusters
                - "anomalies": Detección de anomalías
                - "granger": Test de causalidad Granger
            **kwargs: Parámetros específicos del análisis

        Returns:
            Dict con resultados del análisis
        """
        loaded = self.get_loaded_data(data_key)
        data = loaded.get("data")

        if data is None:
            raise ValueError("No hay datos en la clave especificada")

        # Extraer dominio del data_key (formato: domain_source_id)
        domain = data_key.split("_")[0]
        analyzer = self._get_analyzer(domain)

        if analysis_type == "correlation":
            return {
                "type": "correlation",
                "result": analyzer.correlation_matrix(data)
            }

        elif analysis_type == "clusters":
            n_clusters = kwargs.get("n_clusters", 3)
            return {
                "type": "clusters",
                "result": analyzer.detect_clusters(data, n_clusters=n_clusters)
            }

        elif analysis_type == "anomalies":
            method = kwargs.get("method", "zscore")
            return {
                "type": "anomalies",
                "result": analyzer.detect_anomalies(data, method=method)
            }

        elif analysis_type == "granger":
            x_col = kwargs.get("x")
            y_col = kwargs.get("y")
            max_lag = kwargs.get("max_lag", 5)
            if x_col is None or y_col is None:
                raise ValueError("Granger test requires 'x' and 'y' column names")
            return {
                "type": "granger",
                "result": analyzer.granger_causality_test(data, x_col, y_col, max_lag)
            }

        else:
            raise ValueError(f"Análisis '{analysis_type}' no soportado")

    # =========================================================================
    # HIPÓTESIS Y FALSIFICACIÓN
    # =========================================================================

    def formulate_hypothesis(
        self,
        domain: str,
        statement: str,
        test_function: Callable,
        falsifiable_prediction: str
    ) -> str:
        """
        Formula una hipótesis para ser testeada.

        Args:
            domain: Dominio de la hipótesis
            statement: Declaración de la hipótesis
            test_function: Función que testea la hipótesis (retorna True/False)
            falsifiable_prediction: Predicción que sería falsificada si falla

        Returns:
            ID de la hipótesis
        """
        hypothesis = Hypothesis(
            domain=domain,
            statement=statement,
            test_function=test_function,
            falsifiable_prediction=falsifiable_prediction
        )

        hypothesis_engine = self._get_hypothesis_engine(domain)
        hypothesis_id = hypothesis_engine.register_hypothesis(hypothesis)

        return hypothesis_id

    def test_hypothesis(
        self,
        hypothesis_id: str,
        data_key: str
    ) -> Dict[str, Any]:
        """
        Testea una hipótesis contra datos.

        Args:
            hypothesis_id: ID de la hipótesis
            data_key: Clave de datos para testear

        Returns:
            Dict con resultado del test
        """
        loaded = self.get_loaded_data(data_key)
        data = loaded.get("data")

        # Extraer dominio del data_key
        domain = data_key.split("_")[0]
        hypothesis_engine = self._get_hypothesis_engine(domain)

        result = hypothesis_engine.test_hypothesis(hypothesis_id, data)

        return {
            "hypothesis_id": hypothesis_id,
            "data_key": data_key,
            **result
        }

    def get_hypothesis_status(self, domain: str, hypothesis_id: str) -> Dict[str, Any]:
        """Obtiene estado actual de una hipótesis."""
        hypothesis_engine = self._get_hypothesis_engine(domain)
        return hypothesis_engine.get_hypothesis_status(hypothesis_id)

    # =========================================================================
    # CROSS-DOMAIN
    # =========================================================================

    def cross_domain_correlation(
        self,
        data_keys: List[str],
        join_on: str = "timestamp"
    ) -> Dict[str, Any]:
        """
        Calcula correlaciones entre datos de diferentes dominios.

        Args:
            data_keys: Lista de claves de datos cargados
            join_on: Columna para hacer join

        Returns:
            Dict con matriz de correlaciones cross-domain
        """
        import pandas as pd

        # Cargar y unir datasets
        merged = None
        for key in data_keys:
            loaded = self.get_loaded_data(key)
            df = loaded.get("data")
            if df is None:
                continue

            if join_on not in df.columns:
                raise ValueError(f"Columna '{join_on}' no encontrada en {key}")

            # Prefijar columnas con nombre del dataset
            prefix = key.split("_")[0]
            df_prefixed = df.copy()
            df_prefixed.columns = [
                f"{prefix}_{col}" if col != join_on else col
                for col in df.columns
            ]

            if merged is None:
                merged = df_prefixed
            else:
                merged = pd.merge(merged, df_prefixed, on=join_on, how="outer")

        if merged is None:
            return {"error": "No se pudieron unir los datos"}

        # Calcular correlaciones
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        corr_matrix = merged[numeric_cols].corr()

        return {
            "type": "cross_domain_correlation",
            "data_keys": data_keys,
            "join_column": join_on,
            "n_records": len(merged),
            "correlation_matrix": corr_matrix.to_dict()
        }


# =============================================================================
# INSTANCIA GLOBAL (singleton)
# =============================================================================

_engine_instance: Optional[DomainEngine] = None


def get_engine() -> DomainEngine:
    """
    Obtiene la instancia global del motor de dominios.

    Uso:
        from domains.domain_engine import get_engine
        engine = get_engine()
        engine.load_data("medicine", "synthetic", n_samples=1000)
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DomainEngine()
    return _engine_instance


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    """
    Ejemplo de uso del motor de dominios.

    NOTA: Este es código de demostración, no reglas de dominio.
    """
    engine = get_engine()

    print("Dominios disponibles:", engine.registry.list_domains())

    # Cargar datos sintéticos de medicina
    med_data = engine.load_data(
        domain="medicine",
        source="synthetic",
        n_samples=500,
        seed=42  # ORIGEN: reproducibilidad para testing
    )
    print(f"\nMedicina: {med_data['n_records']} registros cargados")

    # Cargar datos sintéticos de finanzas
    fin_data = engine.load_data(
        domain="finance",
        source="synthetic",
        n_samples=500,
        seed=42
    )
    print(f"Finanzas: {fin_data['n_records']} registros cargados")

    # Análisis de correlación
    corr = engine.analyze(med_data["data_key"], "correlation")
    print(f"\nCorrelación medicina: {len(corr['result'])} variables")

    # Detectar anomalías
    anomalies = engine.analyze(fin_data["data_key"], "anomalies")
    print(f"Anomalías finanzas: {anomalies['result']['n_anomalies']} detectadas")

    print("\n¡Motor de dominios funcionando correctamente!")
