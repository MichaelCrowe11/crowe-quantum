"""Crowe Quantum Core — shared foundation for the Crowe Quantum Platform.

Provides types, gate registry, quantum state representations, noise models,
intermediate representation, and backend protocols used by all Crowe Quantum packages.
"""

__version__ = "1.0.0"

from crowe_quantum_core.types import (
    CircuitType,
    FunctionType,
    GenericType,
    ListType,
    QubitType,
    QuantumType,
    TensorType,
    TypeKind,
    UncertainType,
)
from crowe_quantum_core.gates import Gate, GateRegistry, GateSpec, standard_gates
from crowe_quantum_core.states import DensityMatrix, PauliString, StateVector
from crowe_quantum_core.errors import (
    CloningError,
    GateArityError,
    NoCloningError,
    QuantumError,
    QubitRangeError,
    UnitarityError,
)

__all__ = [
    "CircuitType",
    "CloningError",
    "DensityMatrix",
    "FunctionType",
    "Gate",
    "GateArityError",
    "GateRegistry",
    "GateSpec",
    "GenericType",
    "ListType",
    "NoCloningError",
    "PauliString",
    "QuantumError",
    "QuantumType",
    "QubitRangeError",
    "QubitType",
    "StateVector",
    "TensorType",
    "TypeKind",
    "UncertainType",
    "UnitarityError",
    "standard_gates",
]
