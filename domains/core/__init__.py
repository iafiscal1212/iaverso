"""
Core domain infrastructure - base classes and utilities.
"""

from .domain_base import (
    DomainSchema,
    DomainConnector,
    DomainAnalyzer,
    VariableDefinition,
    VariableType,
    VariableRole,
    Hypothesis,
    HypothesisEngine,
)

__all__ = [
    "DomainSchema",
    "DomainConnector",
    "DomainAnalyzer",
    "VariableDefinition",
    "VariableType",
    "VariableRole",
    "Hypothesis",
    "HypothesisEngine",
]
