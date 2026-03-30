"""Quantum error hierarchy.

Enforces quantum mechanical constraints at the language level:
unitarity, no-cloning theorem, qubit bounds, and gate arity.
"""

from __future__ import annotations


class QuantumError(Exception):
    """Base error for all quantum operations."""

    code: str = "Q0000"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        self.code = code or self.__class__.code
        super().__init__(f"[{self.code}] {message}")


class UnitarityError(QuantumError):
    """Operation violates unitarity — gates must preserve probability."""

    code = "Q1001"


class NoCloningError(QuantumError):
    """Attempted to copy an unknown quantum state (no-cloning theorem)."""

    code = "Q1002"


CloningError = NoCloningError  # Alias for readability


class QubitRangeError(QuantumError):
    """Qubit index is out of range for the circuit/register."""

    code = "Q2001"

    def __init__(self, index: int, num_qubits: int) -> None:
        super().__init__(
            f"Qubit index {index} out of range for {num_qubits}-qubit register",
            code=self.code,
        )
        self.index = index
        self.num_qubits = num_qubits


class GateArityError(QuantumError):
    """Gate applied to wrong number of qubits."""

    code = "Q2002"

    def __init__(self, gate_name: str, expected: int, got: int) -> None:
        super().__init__(
            f"Gate '{gate_name}' expects {expected} qubit(s), got {got}",
            code=self.code,
        )
        self.gate_name = gate_name
        self.expected = expected
        self.got = got


class GateParameterError(QuantumError):
    """Gate given wrong number of parameters."""

    code = "Q2003"


class MeasurementError(QuantumError):
    """Invalid measurement operation."""

    code = "Q3001"


class CircuitError(QuantumError):
    """Malformed circuit structure."""

    code = "Q4001"


class BackendError(QuantumError):
    """Backend execution failure."""

    code = "Q5001"


class NoiseModelError(QuantumError):
    """Invalid noise model configuration."""

    code = "Q6001"
