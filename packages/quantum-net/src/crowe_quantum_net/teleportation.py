"""Quantum teleportation protocol.

Implements the standard teleportation protocol:
1. Alice and Bob share a Bell pair
2. Alice performs a Bell measurement on her qubit and the state to teleport
3. Alice communicates 2 classical bits to Bob
4. Bob applies corrections (I, X, Z, or ZX) based on Alice's measurement

Also supports noisy teleportation where the shared pair has imperfect fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from crowe_quantum_core.states import StateVector
from numpy.typing import NDArray


@dataclass(frozen=True)
class TeleportationResult:
    """Result of a quantum teleportation protocol run."""

    output_state: StateVector
    classical_bits: tuple[int, int]
    fidelity: float
    success: bool

    def __repr__(self) -> str:
        return (
            f"TeleportationResult(fidelity={self.fidelity:.6f}, "
            f"bits={self.classical_bits}, success={self.success})"
        )


class TeleportationProtocol:
    """Standard quantum teleportation protocol.

    Can simulate with perfect or imperfect Bell pairs and
    with or without channel noise.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        state: StateVector,
        bell_pair: StateVector | None = None,
    ) -> TeleportationResult:
        """Execute teleportation of a single-qubit state.

        Args:
            state: Single-qubit state to teleport (|psi> = alpha|0> + beta|1>)
            bell_pair: Shared Bell pair (default: perfect |Phi+>)

        Returns:
            TeleportationResult with the teleported state and protocol metadata
        """
        if state.num_qubits != 1:
            raise ValueError(f"Can only teleport 1-qubit states, got {state.num_qubits}")

        if bell_pair is None:
            bell_pair = StateVector.bell_state(0)

        if bell_pair.num_qubits != 2:
            raise ValueError(f"Bell pair must be 2 qubits, got {bell_pair.num_qubits}")

        original = state.copy()

        # 3-qubit system: |psi>_A ⊗ |Phi+>_{A'B}
        # Qubit 0: Alice's state to teleport
        # Qubit 1: Alice's half of Bell pair
        # Qubit 2: Bob's half of Bell pair
        full = np.kron(state.data, bell_pair.data)

        # Alice's Bell measurement on qubits 0 and 1
        # Bell basis: |Phi+>, |Phi->, |Psi+>, |Psi->
        bell_states = [StateVector.bell_state(i).data for i in range(4)]

        # Compute probabilities for each Bell measurement outcome
        probs = np.zeros(4)
        bob_states: list[NDArray] = []

        for m in range(4):
            # Projector: |Bell_m><Bell_m| ⊗ I_B
            proj_alice = np.outer(bell_states[m], bell_states[m].conj())
            proj_full = np.kron(proj_alice, np.eye(2, dtype=np.complex128))
            projected = proj_full @ full
            p = float(np.real(np.vdot(projected, projected)))
            probs[m] = p
            if p > 1e-15:
                bob_states.append(projected / np.sqrt(p))
            else:
                bob_states.append(projected)

        # Sample measurement outcome
        probs /= probs.sum()  # normalize for numerical safety
        outcome = int(self.rng.choice(4, p=probs))

        # Extract Bob's qubit by tracing out Alice's qubits
        post_state = bob_states[outcome].reshape(2, 2, 2)
        # Sum over Alice's qubits (indices 0 and 1)
        bob_dm = np.einsum("ijk,ijl->kl", post_state, post_state.conj())
        bob_dm /= np.trace(bob_dm)

        # Bob's correction
        corrections = {
            0: np.eye(2, dtype=np.complex128),           # Phi+ → I
            1: np.array([[1, 0], [0, -1]], dtype=np.complex128),  # Phi- → Z
            2: np.array([[0, 1], [1, 0]], dtype=np.complex128),   # Psi+ → X
            3: np.array([[0, -1], [1, 0]], dtype=np.complex128),  # Psi- → iY = ZX
        }

        corrected_dm = corrections[outcome] @ bob_dm @ corrections[outcome].conj().T

        # Extract state vector (should be pure after correction with perfect Bell pair)
        eigenvalues, eigenvectors = np.linalg.eigh(corrected_dm)
        max_idx = np.argmax(eigenvalues)
        bob_sv = StateVector(1, eigenvectors[:, max_idx])

        # Compute fidelity with original
        fidelity = original.fidelity(bob_sv)

        # Determine classical bits from measurement outcome
        classical = (outcome >> 1, outcome & 1)

        return TeleportationResult(
            output_state=bob_sv,
            classical_bits=classical,
            fidelity=fidelity,
            success=fidelity > 0.99,
        )


def teleport(
    state: StateVector,
    bell_pair: StateVector | None = None,
    seed: int | None = None,
) -> TeleportationResult:
    """Convenience function for single teleportation."""
    return TeleportationProtocol(seed=seed).run(state, bell_pair)
