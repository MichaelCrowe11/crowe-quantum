"""Crowe Synapse — scientific reasoning language with uncertainty propagation."""

__version__ = "3.0.0"

from crowe_synapse.uncertain import UncertainValue
from crowe_synapse.symbolic import Symbol, Expression, simplify
from crowe_synapse.units import Unit, Quantity
from crowe_synapse.hypothesis import HypothesisTest, chi_squared_test, t_test

__all__ = [
    "Expression",
    "HypothesisTest",
    "Quantity",
    "Symbol",
    "UncertainValue",
    "Unit",
    "chi_squared_test",
    "simplify",
    "t_test",
]
