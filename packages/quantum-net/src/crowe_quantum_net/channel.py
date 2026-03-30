"""Quantum communication channels.

A quantum channel maps density matrices to density matrices via
the Kraus representation: E(rho) = sum_i K_i rho K_i†.

These are network-level channels modeling the transmission of qubits
between nodes, distinct from the gate-level noise in crowe-quantum-core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from crowe_quantum_core.states import DensityMatrix, StateVector
from numpy.typing import NDArray


@dataclass
class QuantumChannel:
    """A quantum channel defined by Kraus operators.

    The channel acts on single-qubit density matrices:
    E(rho) = sum_i K_i rho K_i†

    Completeness: sum_i K_i† K_i = I (trace-preserving).
    """

    name: str
    kraus_operators: list[NDArray[np.complex128]]
    description: str = ""

    def apply_to_density_matrix(self, dm: DensityMatrix) -> DensityMatrix:
        """Apply channel to a density matrix."""
        result = np.zeros_like(dm.data)
        for k in self.kraus_operators:
            if k.shape[0] == dm.dim:
                # Full-space operator
                result += k @ dm.data @ k.conj().T
            else:
                # Single-qubit operator — apply to each qubit
                n = dm.num_qubits
                for q in range(n):
                    full_k = _embed_operator(k, q, n)
                    result += full_k @ dm.data @ full_k.conj().T
                break  # Only apply once when embedding
        return DensityMatrix(dm.num_qubits, result)

    def apply_to_statevector(self, sv: StateVector) -> DensityMatrix:
        """Apply channel to a pure state (promotes to density matrix)."""
        dm = sv.to_density_matrix()
        return self.apply_to_density_matrix(dm)

    def is_trace_preserving(self, atol: float = 1e-10) -> bool:
        """Verify sum_i K_i† K_i = I."""
        dim = self.kraus_operators[0].shape[0]
        total = np.zeros((dim, dim), dtype=np.complex128)
        for k in self.kraus_operators:
            total += k.conj().T @ k
        return bool(np.allclose(total, np.eye(dim), atol=atol))

    def channel_fidelity(self) -> float:
        """Average channel fidelity F_avg = (d*F_e + 1)/(d+1).

        Where F_e = (1/d) * sum_i |Tr(K_i)|^2 is the entanglement fidelity.
        """
        dim = self.kraus_operators[0].shape[0]
        f_e = sum(abs(np.trace(k)) ** 2 for k in self.kraus_operators) / dim
        return float((dim * f_e + 1) / (dim + 1))


def depolarizing_channel(p: float) -> QuantumChannel:
    """Network depolarizing channel.

    With probability p, replaces the transmitted qubit with maximally mixed state.
    E(rho) = (1-p)*rho + (p/3)*(X*rho*X + Y*rho*Y + Z*rho*Z)
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Depolarizing probability must be in [0,1], got {p}")

    s0 = np.sqrt(1 - p)
    sp = np.sqrt(p / 3)

    k0 = s0 * np.eye(2, dtype=np.complex128)
    k1 = sp * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    k2 = sp * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    k3 = sp * np.array([[1, 0], [0, -1]], dtype=np.complex128)

    return QuantumChannel(
        name="depolarizing",
        kraus_operators=[k0, k1, k2, k3],
        description=f"Depolarizing channel (p={p})",
    )


def amplitude_damping_channel(gamma: float) -> QuantumChannel:
    """Network amplitude damping channel.

    Models photon loss during fiber transmission.
    |1> decays to |0> with probability gamma.
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"Damping rate must be in [0,1], got {gamma}")

    k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    k1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)

    return QuantumChannel(
        name="amplitude_damping",
        kraus_operators=[k0, k1],
        description=f"Amplitude damping channel (gamma={gamma})",
    )


def _embed_operator(op: NDArray, qubit: int, num_qubits: int) -> NDArray:
    """Embed a single-qubit operator into the full Hilbert space."""
    parts = []
    for i in range(num_qubits):
        if i == qubit:
            parts.append(op)
        else:
            parts.append(np.eye(2, dtype=np.complex128))
    result = parts[0]
    for p in parts[1:]:
        result = np.kron(result, p)
    return result
