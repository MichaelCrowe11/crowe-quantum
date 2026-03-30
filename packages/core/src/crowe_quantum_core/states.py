"""Quantum state representations.

Three fundamental representations:
- StateVector: pure states as complex vectors in Hilbert space
- DensityMatrix: mixed states as positive semidefinite operators
- PauliString: observables as weighted sums of Pauli operators

All representations support NumPy operations and are designed for
JAX compatibility when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


class StateVector:
    """A pure quantum state as a complex vector in 2^n-dimensional Hilbert space.

    |ψ⟩ = α₀|0...0⟩ + α₁|0...1⟩ + ... + α_{2^n-1}|1...1⟩

    where Σ|αᵢ|² = 1 (normalization).
    """

    MAX_QUBITS = 30  # Safety limit: 2^30 = ~1 billion amplitudes

    def __init__(self, num_qubits: int, state: NDArray[np.complex128] | None = None) -> None:
        if num_qubits > self.MAX_QUBITS:
            raise ValueError(f"Cannot create state with {num_qubits} qubits (max {self.MAX_QUBITS})")
        self.num_qubits = num_qubits
        self._dim = 2**num_qubits
        if state is not None:
            if state.shape != (self._dim,):
                raise ValueError(f"State vector must have {self._dim} elements, got {state.shape}")
            self._data = state.astype(np.complex128)
        else:
            # Initialize to |0...0⟩
            self._data = np.zeros(self._dim, dtype=np.complex128)
            self._data[0] = 1.0

    @property
    def data(self) -> NDArray[np.complex128]:
        return self._data

    @property
    def dim(self) -> int:
        return self._dim

    @classmethod
    def from_label(cls, label: str) -> StateVector:
        """Create a computational basis state from a bitstring label.

        Example: StateVector.from_label("01") creates |01⟩
        """
        n = len(label)
        sv = cls(n)
        index = int(label, 2)
        sv._data[:] = 0
        sv._data[index] = 1.0
        return sv

    @classmethod
    def from_amplitudes(cls, amplitudes: list[complex] | NDArray[np.complex128]) -> StateVector:
        """Create a state from explicit amplitudes. Infers qubit count from length."""
        arr = np.asarray(amplitudes, dtype=np.complex128)
        n = int(np.log2(len(arr)))
        if 2**n != len(arr):
            raise ValueError(f"Amplitude count {len(arr)} is not a power of 2")
        return cls(n, arr)

    @classmethod
    def bell_state(cls, which: int = 0) -> StateVector:
        """Create one of the four Bell states.

        0: |Φ+⟩ = (|00⟩ + |11⟩)/√2
        1: |Φ-⟩ = (|00⟩ - |11⟩)/√2
        2: |Ψ+⟩ = (|01⟩ + |10⟩)/√2
        3: |Ψ-⟩ = (|01⟩ - |10⟩)/√2
        """
        sv = cls(2)
        inv_sqrt2 = 1.0 / np.sqrt(2)
        sv._data[:] = 0
        if which == 0:
            sv._data[0] = inv_sqrt2   # |00⟩
            sv._data[3] = inv_sqrt2   # |11⟩
        elif which == 1:
            sv._data[0] = inv_sqrt2
            sv._data[3] = -inv_sqrt2
        elif which == 2:
            sv._data[1] = inv_sqrt2   # |01⟩
            sv._data[2] = inv_sqrt2   # |10⟩
        elif which == 3:
            sv._data[1] = inv_sqrt2
            sv._data[2] = -inv_sqrt2
        else:
            raise ValueError(f"Bell state index must be 0-3, got {which}")
        return sv

    @classmethod
    def ghz_state(cls, n: int) -> StateVector:
        """Create an n-qubit GHZ state: (|0...0⟩ + |1...1⟩)/√2."""
        sv = cls(n)
        inv_sqrt2 = 1.0 / np.sqrt(2)
        sv._data[:] = 0
        sv._data[0] = inv_sqrt2
        sv._data[-1] = inv_sqrt2
        return sv

    def probabilities(self) -> NDArray[np.float64]:
        """Born rule: P(i) = |αᵢ|²."""
        return np.abs(self._data) ** 2

    def measure(self, rng: np.random.Generator | None = None) -> int:
        """Perform a projective measurement in the computational basis.

        Returns the measured basis state index and collapses the state.
        """
        if rng is None:
            rng = np.random.default_rng()
        probs = self.probabilities()
        outcome = rng.choice(self._dim, p=probs)
        # Collapse
        self._data[:] = 0
        self._data[outcome] = 1.0
        return int(outcome)

    def measure_qubit(self, qubit: int, rng: np.random.Generator | None = None) -> int:
        """Measure a single qubit, collapsing only that qubit's state."""
        if rng is None:
            rng = np.random.default_rng()
        if qubit < 0 or qubit >= self.num_qubits:
            from crowe_quantum_core.errors import QubitRangeError

            raise QubitRangeError(qubit, self.num_qubits)

        # Calculate probability of measuring |0⟩ on this qubit
        prob_0 = 0.0
        for i in range(self._dim):
            if not (i >> (self.num_qubits - 1 - qubit)) & 1:
                prob_0 += abs(self._data[i]) ** 2

        outcome = 0 if rng.random() < prob_0 else 1
        norm = np.sqrt(prob_0 if outcome == 0 else 1 - prob_0)

        # Collapse: zero out amplitudes inconsistent with outcome, renormalize
        for i in range(self._dim):
            bit = (i >> (self.num_qubits - 1 - qubit)) & 1
            if bit != outcome:
                self._data[i] = 0
            else:
                self._data[i] /= norm

        return outcome

    def apply_gate(self, gate_matrix: NDArray[np.complex128], qubits: list[int]) -> None:
        """Apply a gate matrix to specified qubits.

        Uses vectorized tensor contraction for performance.
        """
        n = self.num_qubits
        for q in qubits:
            if q < 0 or q >= n:
                from crowe_quantum_core.errors import QubitRangeError

                raise QubitRangeError(q, n)

        num_gate_qubits = len(qubits)

        if num_gate_qubits == 1:
            self._apply_single_qubit_gate(gate_matrix, qubits[0])
        elif num_gate_qubits == 2:
            self._apply_two_qubit_gate(gate_matrix, qubits[0], qubits[1])
        else:
            self._apply_multi_qubit_gate(gate_matrix, qubits)

    def _apply_single_qubit_gate(self, matrix: NDArray[np.complex128], qubit: int) -> None:
        """Optimized single-qubit gate application via vectorized stride access."""
        n = self.num_qubits
        target = n - 1 - qubit
        stride = 1 << target
        state = self._data

        for block_start in range(0, self._dim, stride * 2):
            idx0 = np.arange(block_start, block_start + stride)
            idx1 = idx0 + stride
            a = state[idx0].copy()
            b = state[idx1].copy()
            state[idx0] = matrix[0, 0] * a + matrix[0, 1] * b
            state[idx1] = matrix[1, 0] * a + matrix[1, 1] * b

    def _apply_two_qubit_gate(
        self, matrix: NDArray[np.complex128], qubit0: int, qubit1: int
    ) -> None:
        """Two-qubit gate via direct index computation."""
        n = self.num_qubits
        new_state = np.zeros_like(self._data)

        for i in range(self._dim):
            b0 = (i >> (n - 1 - qubit0)) & 1
            b1 = (i >> (n - 1 - qubit1)) & 1
            row = b0 * 2 + b1

            for col in range(4):
                c0 = (col >> 1) & 1
                c1 = col & 1
                # Compute source index by flipping bits
                j = i
                j = (j & ~(1 << (n - 1 - qubit0))) | (c0 << (n - 1 - qubit0))
                j = (j & ~(1 << (n - 1 - qubit1))) | (c1 << (n - 1 - qubit1))
                new_state[i] += matrix[row, col] * self._data[j]

        self._data[:] = new_state

    def _apply_multi_qubit_gate(
        self, matrix: NDArray[np.complex128], qubits: list[int]
    ) -> None:
        """General multi-qubit gate application."""
        n = self.num_qubits
        k = len(qubits)
        gate_dim = 2**k
        new_state = np.zeros_like(self._data)
        bit_positions = [n - 1 - q for q in qubits]

        for i in range(self._dim):
            row = 0
            for idx, pos in enumerate(bit_positions):
                row |= ((i >> pos) & 1) << (k - 1 - idx)

            for col in range(gate_dim):
                j = i
                for idx, pos in enumerate(bit_positions):
                    bit = (col >> (k - 1 - idx)) & 1
                    j = (j & ~(1 << pos)) | (bit << pos)
                new_state[i] += matrix[row, col] * self._data[j]

        self._data[:] = new_state

    def norm(self) -> float:
        """L2 norm (should be 1.0 for valid states)."""
        return float(np.linalg.norm(self._data))

    def normalize(self) -> None:
        """Renormalize the state vector."""
        n = self.norm()
        if n > 0:
            self._data /= n

    def fidelity(self, other: StateVector) -> float:
        """Fidelity |⟨ψ|φ⟩|² between two pure states."""
        return float(abs(np.vdot(self._data, other._data)) ** 2)

    def inner_product(self, other: StateVector) -> complex:
        """⟨ψ|φ⟩ — the inner product."""
        return complex(np.vdot(self._data, other._data))

    def to_density_matrix(self) -> DensityMatrix:
        """Convert |ψ⟩ to density matrix ρ = |ψ⟩⟨ψ|."""
        return DensityMatrix(
            self.num_qubits,
            np.outer(self._data, self._data.conj()),
        )

    def to_bitstring(self, outcome: int) -> str:
        """Convert a measurement outcome to a bitstring."""
        return format(outcome, f"0{self.num_qubits}b")

    def expectation(self, observable: NDArray[np.complex128]) -> complex:
        """⟨ψ|O|ψ⟩ — expectation value of an observable."""
        return complex(self._data.conj() @ observable @ self._data)

    def copy(self) -> StateVector:
        """Create an independent copy (classical operation on the representation)."""
        return StateVector(self.num_qubits, self._data.copy())

    def __repr__(self) -> str:
        nonzero = np.where(np.abs(self._data) > 1e-10)[0]
        if len(nonzero) <= 8:
            terms = []
            for idx in nonzero:
                amp = self._data[idx]
                label = format(idx, f"0{self.num_qubits}b")
                terms.append(f"({amp:.4f})|{label}⟩")
            return " + ".join(terms) if terms else "|vacuum⟩"
        return f"StateVector(n={self.num_qubits}, dim={self._dim})"


class DensityMatrix:
    """Mixed quantum state as a density operator ρ.

    Properties: ρ† = ρ (Hermitian), Tr(ρ) = 1, ρ ≥ 0 (positive semidefinite).
    Pure states: ρ² = ρ. Mixed states: Tr(ρ²) < 1.
    """

    def __init__(self, num_qubits: int, matrix: NDArray[np.complex128] | None = None) -> None:
        self.num_qubits = num_qubits
        dim = 2**num_qubits
        if matrix is not None:
            if matrix.shape != (dim, dim):
                raise ValueError(f"Density matrix must be {dim}x{dim}")
            self._data = matrix.astype(np.complex128)
        else:
            self._data = np.zeros((dim, dim), dtype=np.complex128)
            self._data[0, 0] = 1.0  # |0...0⟩⟨0...0|

    @property
    def data(self) -> NDArray[np.complex128]:
        return self._data

    @property
    def dim(self) -> int:
        return 2**self.num_qubits

    @classmethod
    def maximally_mixed(cls, num_qubits: int) -> DensityMatrix:
        """Create the maximally mixed state ρ = I/2^n."""
        dim = 2**num_qubits
        return cls(num_qubits, np.eye(dim, dtype=np.complex128) / dim)

    def purity(self) -> float:
        """Tr(ρ²) — 1.0 for pure states, 1/dim for maximally mixed."""
        return float(np.real(np.trace(self._data @ self._data)))

    def is_pure(self, atol: float = 1e-10) -> bool:
        return abs(self.purity() - 1.0) < atol

    def von_neumann_entropy(self) -> float:
        """S(ρ) = -Tr(ρ log ρ) — quantum entropy."""
        eigenvalues = np.real(np.linalg.eigvalsh(self._data))
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def trace(self) -> float:
        return float(np.real(np.trace(self._data)))

    def expectation(self, observable: NDArray[np.complex128]) -> complex:
        """Tr(ρO) — expectation value."""
        return complex(np.trace(self._data @ observable))

    def fidelity(self, other: DensityMatrix) -> float:
        """F(ρ, σ) = [Tr(√(√ρ σ √ρ))]²."""
        sqrt_self = _matrix_sqrt(self._data)
        product = sqrt_self @ other._data @ sqrt_self
        sqrt_product = _matrix_sqrt(product)
        return float(np.real(np.trace(sqrt_product)) ** 2)

    def partial_trace(self, keep_qubits: list[int]) -> DensityMatrix:
        """Trace out all qubits except those in keep_qubits."""
        n = self.num_qubits
        k = len(keep_qubits)
        dim_keep = 2**k
        result = np.zeros((dim_keep, dim_keep), dtype=np.complex128)
        trace_qubits = [q for q in range(n) if q not in keep_qubits]

        for i in range(dim_keep):
            for j in range(dim_keep):
                for t in range(2 ** len(trace_qubits)):
                    row = _insert_bits(i, keep_qubits, t, trace_qubits, n)
                    col = _insert_bits(j, keep_qubits, t, trace_qubits, n)
                    result[i, j] += self._data[row, col]

        return DensityMatrix(k, result)

    def __repr__(self) -> str:
        return f"DensityMatrix(n={self.num_qubits}, purity={self.purity():.4f})"


def _insert_bits(
    keep_val: int,
    keep_qubits: list[int],
    trace_val: int,
    trace_qubits: list[int],
    n: int,
) -> int:
    """Helper to reconstruct a full index from kept and traced qubit values."""
    result = 0
    for idx, q in enumerate(keep_qubits):
        bit = (keep_val >> (len(keep_qubits) - 1 - idx)) & 1
        result |= bit << (n - 1 - q)
    for idx, q in enumerate(trace_qubits):
        bit = (trace_val >> (len(trace_qubits) - 1 - idx)) & 1
        result |= bit << (n - 1 - q)
    return result


def _matrix_sqrt(m: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Matrix square root via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(m)
    eigenvalues = np.maximum(eigenvalues, 0)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T


@dataclass
class PauliString:
    """A weighted Pauli string: coefficient × P₁ ⊗ P₂ ⊗ ... ⊗ Pₙ.

    Pauli operators: I (identity), X, Y, Z.
    Example: 0.5 * Z⊗Z represents a ZZ correlation measurement.
    """

    coefficient: complex
    paulis: str  # e.g., "XZIY" — one char per qubit

    @property
    def num_qubits(self) -> int:
        return len(self.paulis)

    def to_matrix(self) -> NDArray[np.complex128]:
        """Convert to full 2^n × 2^n matrix via tensor product."""
        pauli_matrices = {
            "I": np.eye(2, dtype=np.complex128),
            "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
        }
        result = np.array([[1.0]], dtype=np.complex128)
        for p in self.paulis:
            result = np.kron(result, pauli_matrices[p.upper()])
        return self.coefficient * result

    def commutes_with(self, other: PauliString) -> bool:
        """Check if two Pauli strings commute. They anticommute if they
        differ on an odd number of non-identity positions where the Paulis are different."""
        if self.num_qubits != other.num_qubits:
            return False
        anticommute_count = 0
        for a, b in zip(self.paulis, other.paulis):
            if a != "I" and b != "I" and a != b:
                anticommute_count += 1
        return anticommute_count % 2 == 0

    def __repr__(self) -> str:
        return f"({self.coefficient}) {'⊗'.join(self.paulis)}"
