"""
DOMINIO: FINANZAS / MERCADOS

NORMA DURA EXTENDIDA:
- NO hay reglas de trading hardcodeadas
- NO hay umbrales de riesgo predefinidos (VaR > 5%, etc.)
- SOLO infraestructura para que agentes aprendan de datos

El agente que use esto APRENDERÁ de series temporales,
y formulará hipótesis que serán falsificadas contra datos reales.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np

from ..core.domain_base import (
    DomainSchema, DomainConnector, VariableDefinition,
    VariableType, VariableRole
)


# =============================================================================
# SCHEMA: Define QUÉ variables existen, NO estrategias de inversión
# =============================================================================

def create_market_schema() -> DomainSchema:
    """
    Schema para datos de mercado genéricos.

    NOTA: No definimos qué es "oportunidad" o "riesgo".
    Eso lo aprende el agente de los datos.
    """
    variables = [
        # Variables OHLCV básicas
        VariableDefinition(
            name="timestamp",
            var_type=VariableType.TEMPORAL,
            role=VariableRole.INDEX,
            description="Momento del precio"
        ),
        VariableDefinition(
            name="symbol",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.INDEX,
            description="Símbolo del instrumento"
        ),
        VariableDefinition(
            name="open",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Precio de apertura"
        ),
        VariableDefinition(
            name="high",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Precio máximo"
        ),
        VariableDefinition(
            name="low",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Precio mínimo"
        ),
        VariableDefinition(
            name="close",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Precio de cierre"
        ),
        VariableDefinition(
            name="volume",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="shares",
            description="Volumen de transacciones"
        ),
        VariableDefinition(
            name="adjusted_close",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Precio ajustado por splits/dividendos"
        ),

        # Variables derivadas (calculadas, no interpretadas)
        VariableDefinition(
            name="returns",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Retorno logarítmico"
        ),
        VariableDefinition(
            name="volatility",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Volatilidad (ventana móvil)"
        ),
    ]

    return DomainSchema(
        domain_name="finance_market",
        version="1.0.0",
        variables=variables,
        metadata={
            "tipo": "mercado_genérico",
            "nota": "El agente aprende patrones de los datos"
        }
    )


def create_fundamental_schema() -> DomainSchema:
    """Schema para datos fundamentales de empresas."""
    variables = [
        VariableDefinition(
            name="symbol",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.INDEX,
            description="Símbolo de la empresa"
        ),
        VariableDefinition(
            name="report_date",
            var_type=VariableType.TEMPORAL,
            role=VariableRole.INDEX,
            description="Fecha del reporte"
        ),
        VariableDefinition(
            name="revenue",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="USD",
            description="Ingresos totales"
        ),
        VariableDefinition(
            name="net_income",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="USD",
            description="Ingreso neto"
        ),
        VariableDefinition(
            name="total_assets",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="USD",
            description="Activos totales"
        ),
        VariableDefinition(
            name="total_liabilities",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="USD",
            description="Pasivos totales"
        ),
        VariableDefinition(
            name="cash_flow_operations",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="USD",
            description="Flujo de caja operativo"
        ),
        VariableDefinition(
            name="shares_outstanding",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="shares",
            description="Acciones en circulación"
        ),
    ]

    return DomainSchema(
        domain_name="finance_fundamental",
        version="1.0.0",
        variables=variables,
        metadata={"tipo": "fundamental"}
    )


# =============================================================================
# CONNECTOR: Carga datos de diferentes fuentes
# =============================================================================

class FinanceConnector(DomainConnector):
    """
    Conector para datos financieros.

    Soporta múltiples formatos y fuentes.
    NO interpreta los datos - solo los carga y calcula derivados básicos.
    """

    def __init__(self):
        self.market_schema = create_market_schema()
        self.fundamental_schema = create_fundamental_schema()
        super().__init__(schema=self.market_schema)

    def load_data(self, source: str, **kwargs) -> np.ndarray:
        """Carga datos desde una fuente."""
        if source == "csv":
            result = self.load_csv(Path(kwargs["path"]), kwargs.get("column_mapping"))
            return result["data"].values if hasattr(result["data"], 'values') else result["data"]
        elif source == "synthetic":
            result = self.load_synthetic_for_testing(kwargs.get("n_samples", 1000), kwargs.get("seed"))
            return result["data"].values if hasattr(result["data"], 'values') else result["data"]
        elif source == "yahoo":
            result = self.load_from_yahoo(kwargs["symbol"], kwargs["start_date"], kwargs["end_date"])
            return result["data"].values if hasattr(result["data"], 'values') else result["data"]
        else:
            raise ValueError(f"Fuente '{source}' no soportada")

    def get_available_sources(self) -> List[str]:
        """Lista fuentes de datos disponibles."""
        return ["csv", "synthetic", "yahoo", "api"]

    def load_csv(self, path: Path, column_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Carga datos desde CSV."""
        import pandas as pd

        df = pd.read_csv(path)

        if column_mapping:
            df = df.rename(columns=column_mapping)

        # Si hay timestamp, convertir a datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
            df = df.drop(columns=['date'])

        return {
            "data": df,
            "source": str(path),
            "n_records": len(df),
        }

    def load_from_yahoo(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Carga datos desde Yahoo Finance.

        Requiere: yfinance instalado
        """
        try:
            import yfinance as yf
            import pandas as pd
        except ImportError:
            raise ImportError("Instalar yfinance: pip install yfinance")

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)

        # Renombrar a schema estándar
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        df['symbol'] = symbol

        # Calcular retornos (derivado básico, no interpretación)
        df['returns'] = np.log(df['close'] / df['close'].shift(1))

        return {
            "data": df,
            "source": f"yahoo_finance/{symbol}",
            "n_records": len(df),
            "date_range": (start_date, end_date),
        }

    def load_from_api(self, endpoint: str, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Carga datos desde API (Alpha Vantage, Polygon, etc.)"""
        raise NotImplementedError(
            "Implementar para API específica"
        )

    def load_synthetic_for_testing(
        self,
        n_samples: int,
        seed: Optional[int] = None,
        model: str = "gbm"
    ) -> Dict[str, Any]:
        """
        Genera datos sintéticos para testing.

        Args:
            n_samples: Número de muestras
            seed: Semilla para reproducibilidad
            model: Modelo generador ("gbm" = Geometric Brownian Motion,
                   "random_walk" = paseo aleatorio simple)

        IMPORTANTE: Estos son datos de prueba, NO representan
        patrones reales de mercado.
        """
        import pandas as pd

        if seed is not None:
            np.random.seed(seed)

        # Parámetros del modelo (genéricos, no calibrados a ningún mercado)
        initial_price = 100.0

        if model == "gbm":
            # Geometric Brownian Motion
            # dS = μS*dt + σS*dW
            dt = 1.0 / 252  # ORIGEN: días de trading por año (convención)
            mu = 0.0  # drift neutral
            sigma = np.random.uniform(0.1, 0.4)  # volatilidad aleatoria

            returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_samples)
            prices = initial_price * np.exp(np.cumsum(returns))

        else:  # random_walk
            returns = np.random.normal(0, 0.02, n_samples)
            prices = initial_price * np.cumprod(1 + returns)

        # Generar OHLC a partir de close
        noise_high = np.abs(np.random.normal(0, 0.005, n_samples))
        noise_low = np.abs(np.random.normal(0, 0.005, n_samples))
        noise_open = np.random.normal(0, 0.002, n_samples)

        df = pd.DataFrame({
            "timestamp": pd.date_range(start="2020-01-01", periods=n_samples, freq="D"),
            "symbol": "SYNTHETIC",
            "open": prices * (1 + noise_open),
            "high": prices * (1 + noise_high),
            "low": prices * (1 - noise_low),
            "close": prices,
            "volume": np.random.lognormal(mean=15, sigma=1, size=n_samples),
            "returns": np.concatenate([[0], np.diff(np.log(prices))]),
        })

        return {
            "data": df,
            "source": f"synthetic_{model}",
            "n_records": n_samples,
            "model_params": {"model": model, "sigma": sigma if model == "gbm" else None},
            "note": "Datos sintéticos - NO usar para estrategias reales"
        }

    def calculate_derived_features(self, df, window_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Calcula features derivados genéricos.

        NOTA: Estos son cálculos matemáticos puros, no reglas de trading.
        El agente decide qué features son predictivos.
        """
        import pandas as pd

        if window_sizes is None:
            window_sizes = [5, 10, 20, 50]  # ventanas típicas, no reglas

        result = df.copy()

        for w in window_sizes:
            # Volatilidad realizada (desv. std. de retornos)
            result[f'volatility_{w}d'] = result['returns'].rolling(w).std()

            # Media móvil simple (descriptivo, no señal)
            result[f'sma_{w}d'] = result['close'].rolling(w).mean()

            # Momentum (retorno sobre ventana)
            result[f'momentum_{w}d'] = result['close'] / result['close'].shift(w) - 1

        return {
            "data": result,
            "features_added": [
                f'{feat}_{w}d'
                for w in window_sizes
                for feat in ['volatility', 'sma', 'momentum']
            ]
        }


# =============================================================================
# SCHEMA ADICIONALES: Subtipos financieros
# =============================================================================

def create_options_schema() -> DomainSchema:
    """Schema para datos de opciones."""
    variables = [
        VariableDefinition(
            name="underlying_symbol",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.INDEX,
            description="Símbolo del subyacente"
        ),
        VariableDefinition(
            name="option_type",
            var_type=VariableType.CATEGORICAL,
            role=VariableRole.COVARIATE,
            categories=["call", "put"],
            description="Tipo de opción"
        ),
        VariableDefinition(
            name="strike",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Precio de ejercicio"
        ),
        VariableDefinition(
            name="expiration",
            var_type=VariableType.TEMPORAL,
            role=VariableRole.PREDICTOR,
            description="Fecha de expiración"
        ),
        VariableDefinition(
            name="bid",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Precio bid"
        ),
        VariableDefinition(
            name="ask",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Precio ask"
        ),
        VariableDefinition(
            name="implied_volatility",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Volatilidad implícita"
        ),
        VariableDefinition(
            name="open_interest",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Interés abierto"
        ),
    ]

    return DomainSchema(
        domain_name="finance_options",
        version="1.0.0",
        variables=variables,
        metadata={"tipo": "opciones"}
    )


def create_crypto_schema() -> DomainSchema:
    """Schema para datos de criptomonedas."""
    base = create_market_schema()

    additional = [
        VariableDefinition(
            name="market_cap",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            unit="USD",
            description="Capitalización de mercado"
        ),
        VariableDefinition(
            name="circulating_supply",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Suministro circulante"
        ),
        VariableDefinition(
            name="on_chain_transactions",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Transacciones on-chain"
        ),
        VariableDefinition(
            name="active_addresses",
            var_type=VariableType.CONTINUOUS,
            role=VariableRole.PREDICTOR,
            description="Direcciones activas"
        ),
    ]

    return DomainSchema(
        domain_name="finance_crypto",
        version="1.0.0",
        variables=base.variables + additional,
        metadata={"tipo": "criptomonedas"}
    )
