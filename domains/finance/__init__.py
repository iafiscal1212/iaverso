"""
Finance domain - market data infrastructure.

NO trading rules, only data infrastructure.
"""

from .finance_connector import (
    FinanceConnector,
    create_market_schema,
    create_fundamental_schema,
    create_options_schema,
    create_crypto_schema,
)

__all__ = [
    "FinanceConnector",
    "create_market_schema",
    "create_fundamental_schema",
    "create_options_schema",
    "create_crypto_schema",
]
