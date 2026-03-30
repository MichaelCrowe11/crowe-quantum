"""Crowe Synapse — scientific reasoning language with uncertainty propagation."""

__version__ = "3.0.0"

from crowe_synapse.hypothesis import HypothesisTest, chi_squared_test, t_test
from crowe_synapse.symbolic import Expression, Symbol, simplify
from crowe_synapse.uncertain import UncertainValue
from crowe_synapse.units import Quantity, Unit

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
