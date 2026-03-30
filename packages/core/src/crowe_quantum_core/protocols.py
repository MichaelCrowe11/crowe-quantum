"""Backend protocols — the abstract interface every quantum backend implements.

Inspired by Qiskit's primitive model: computation is expressed as
estimation (expectation values) or sampling (measurement outcomes),
not raw circuit execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from crowe_quantum_core.noise import NoiseModel
from crowe_quantum_core.states import DensityMatrix, PauliString, StateVector


@dataclass
class CircuitIR:
    """Intermediate representation of a quantum circuit.

    A sequence of gate operations, measurements, and classical control.
    Backend-agnostic — each backend compiles this to its native format.
    """

    num_qubits: int
    num_classical_bits: int = 0
    operations: list[Operation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_gate(self, gate_name: str, qubits: list[int], params: tuple[float, ...] = ()) -> None:
        self.operations.append(
            Operation(kind="gate", gate_name=gate_name, qubits=qubits, params=params)
        )

    def add_measurement(self, qubit: int, classical_bit: int | None = None) -> None:
        self.operations.append(
            Operation(kind="measure", qubits=[qubit], classical_bit=classical_bit)
        )

    def add_barrier(self, qubits: list[int] | None = None) -> None:
        self.operations.append(
            Operation(kind="barrier", qubits=qubits or list(range(self.num_qubits)))
        )

    def add_reset(self, qubit: int) -> None:
        self.operations.append(Operation(kind="reset", qubits=[qubit]))

    def depth(self) -> int:
        """Compute circuit depth (longest path of dependent gates)."""
        if not self.operations:
            return 0
        # Simple: count gate layers by tracking when each qubit is last used
        qubit_depth: dict[int, int] = {}
        for op in self.operations:
            if op.kind == "gate" and op.qubits:
                max_dep = max((qubit_depth.get(q, 0) for q in op.qubits), default=0)
                new_depth = max_dep + 1
                for q in op.qubits:
                    qubit_depth[q] = new_depth
        return max(qubit_depth.values(), default=0)

    def gate_count(self) -> int:
        return sum(1 for op in self.operations if op.kind == "gate")


@dataclass
class Operation:
    """A single operation in a circuit."""

    kind: str  # "gate", "measure", "barrier", "reset", "if"
    gate_name: str = ""
    qubits: list[int] = field(default_factory=list)
    params: tuple[float, ...] = ()
    classical_bit: int | None = None
    condition: tuple[int, int] | None = None  # (classical_bit, value) for conditional ops


@dataclass
class SamplerResult:
    """Result from a sampling (measurement) execution."""

    counts: dict[str, int]  # bitstring -> count
    shots: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def probabilities(self) -> dict[str, float]:
        return {k: v / self.shots for k, v in self.counts.items()}

    def most_likely(self) -> str:
        return max(self.counts, key=self.counts.get)  # type: ignore[arg-type]


@dataclass
class EstimatorResult:
    """Result from an estimation (expectation value) execution."""

    values: list[float]
    std_errors: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    """Abstract quantum backend protocol.

    Every backend (simulator, IBM, Google, AWS) implements this interface.
    The two primary operations are sample (get measurement counts) and
    estimate (get expectation values of observables).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""

    @property
    @abstractmethod
    def max_qubits(self) -> int:
        """Maximum number of qubits supported."""

    @abstractmethod
    def sample(
        self,
        circuit: CircuitIR,
        shots: int = 1024,
        noise_model: NoiseModel | None = None,
        seed: int | None = None,
    ) -> SamplerResult:
        """Execute circuit and return measurement counts."""

    @abstractmethod
    def estimate(
        self,
        circuit: CircuitIR,
        observables: list[PauliString],
        shots: int = 1024,
        noise_model: NoiseModel | None = None,
        seed: int | None = None,
    ) -> EstimatorResult:
        """Estimate expectation values of observables."""

    def statevector(self, circuit: CircuitIR) -> StateVector | None:
        """Get the final state vector (simulators only, None for hardware)."""
        return None

    def transpile(self, circuit: CircuitIR) -> CircuitIR:
        """Transpile circuit for this backend. Default: identity."""
        return circuit

    def validate_circuit(self, circuit: CircuitIR) -> list[str]:
        """Validate a circuit against backend constraints. Returns list of issues."""
        issues = []
        if circuit.num_qubits > self.max_qubits:
            issues.append(
                f"Circuit requires {circuit.num_qubits} qubits, "
                f"backend supports {self.max_qubits}"
            )
        return issues
