"""Quantum type system.

Category-theoretic foundation: types are objects, operations are morphisms.
Enforces unitarity, no-cloning, and measurement irreversibility at the type level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class TypeKind(Enum):
    """Classification of types in the quantum type system."""

    PRIMITIVE = auto()     # int, float, complex, bool, str
    UNCERTAIN = auto()     # value ± uncertainty
    QUANTUM = auto()       # qubit, circuit, register
    TENSOR = auto()        # n-dimensional array
    FUNCTION = auto()      # callable
    COMPOSITE = auto()     # list, dict, tuple
    GENERIC = auto()       # parameterized type


@dataclass(frozen=True)
class QuantumType:
    """Base for all types in the Crowe Quantum type system."""

    kind: TypeKind
    name: str

    def is_quantum(self) -> bool:
        return self.kind == TypeKind.QUANTUM

    def is_copyable(self) -> bool:
        """Classical types can be copied; quantum types cannot (no-cloning theorem)."""
        return self.kind != TypeKind.QUANTUM

    def is_assignable_from(self, other: QuantumType) -> bool:
        if self == other:
            return True
        if isinstance(self, QubitType) and isinstance(other, QubitType):
            return True
        return False

    def size_bytes(self) -> int:
        return 8  # default: one 64-bit value


# --- Primitive Types ---

INT_TYPE = QuantumType(kind=TypeKind.PRIMITIVE, name="int")
FLOAT_TYPE = QuantumType(kind=TypeKind.PRIMITIVE, name="float")
COMPLEX_TYPE = QuantumType(kind=TypeKind.PRIMITIVE, name="complex")
BOOL_TYPE = QuantumType(kind=TypeKind.PRIMITIVE, name="bool")
STR_TYPE = QuantumType(kind=TypeKind.PRIMITIVE, name="str")


# --- Quantum Types ---

@dataclass(frozen=True)
class QubitType(QuantumType):
    """A single qubit — the fundamental unit of quantum information.

    Not copyable (no-cloning theorem). Measurement collapses it irreversibly.
    """

    kind: TypeKind = field(default=TypeKind.QUANTUM, init=False)
    name: str = field(default="qubit", init=False)
    measured: bool = False

    def size_bytes(self) -> int:
        return 16  # complex amplitude (2 floats)


@dataclass(frozen=True)
class CircuitType(QuantumType):
    """A quantum circuit — a sequence of unitary operations on qubits."""

    kind: TypeKind = field(default=TypeKind.QUANTUM, init=False)
    name: str = field(default="circuit", init=False)
    num_qubits: int = 0
    num_classical_bits: int = 0


@dataclass(frozen=True)
class RegisterType(QuantumType):
    """A quantum register — an ordered collection of qubits."""

    kind: TypeKind = field(default=TypeKind.QUANTUM, init=False)
    name: str = field(default="register", init=False)
    size: int = 1

    def size_bytes(self) -> int:
        return 16 * (2**self.size)  # full state vector


# --- Scientific Types ---

@dataclass(frozen=True)
class UncertainType(QuantumType):
    """A value with associated uncertainty — central to scientific computing."""

    kind: TypeKind = field(default=TypeKind.UNCERTAIN, init=False)
    name: str = field(default="uncertain", init=False)
    base_type: QuantumType = FLOAT_TYPE
    distribution: str = "normal"  # normal, uniform, lognormal, triangular

    def size_bytes(self) -> int:
        return self.base_type.size_bytes() * 2  # value + uncertainty


@dataclass(frozen=True)
class TensorType(QuantumType):
    """An n-dimensional tensor — the workhorse of scientific computing."""

    kind: TypeKind = field(default=TypeKind.TENSOR, init=False)
    name: str = field(default="tensor", init=False)
    shape: tuple[int, ...] = ()
    dtype: str = "float64"

    def size_bytes(self) -> int:
        from math import prod

        element_size = {"float32": 4, "float64": 8, "complex64": 8, "complex128": 16}
        total = prod(self.shape) if self.shape else 0
        return total * element_size.get(self.dtype, 8)


# --- Composite Types ---

@dataclass(frozen=True)
class ListType(QuantumType):
    """A list of elements of a given type."""

    kind: TypeKind = field(default=TypeKind.COMPOSITE, init=False)
    name: str = field(default="list", init=False)
    element_type: QuantumType = FLOAT_TYPE


@dataclass(frozen=True)
class FunctionType(QuantumType):
    """A function type with parameter types and return type."""

    kind: TypeKind = field(default=TypeKind.FUNCTION, init=False)
    name: str = field(default="function", init=False)
    param_types: tuple[QuantumType, ...] = ()
    return_type: QuantumType = FLOAT_TYPE
    is_unitary: bool = False  # True if this function preserves quantum state norms


@dataclass(frozen=True)
class GenericType(QuantumType):
    """A parameterized generic type (e.g., List[T], Circuit[N])."""

    kind: TypeKind = field(default=TypeKind.GENERIC, init=False)
    name: str = "T"
    constraints: tuple[QuantumType, ...] = ()


# --- Type predicates ---

def is_classical(t: QuantumType) -> bool:
    """Check if a type is classical (can be freely copied and measured)."""
    return t.kind in (TypeKind.PRIMITIVE, TypeKind.UNCERTAIN, TypeKind.TENSOR, TypeKind.COMPOSITE)


def is_quantum(t: QuantumType) -> bool:
    """Check if a type is quantum (subject to no-cloning and measurement collapse)."""
    return t.kind == TypeKind.QUANTUM


def types_compatible(a: QuantumType, b: QuantumType) -> bool:
    """Check if type b can be assigned to type a."""
    return a.is_assignable_from(b)
