"""Entanglement distribution and quantification.

Tools for generating, distributing, and measuring entanglement
across a quantum network. Includes entanglement swapping for
extending entanglement range beyond direct links.
"""

from __future__ import annotations

import numpy as np
from crowe_quantum_core.states import DensityMatrix, StateVector


class EntanglementSource:
    """Source of entangled pairs (Bell pairs).

    Models an entanglement source node that can generate Bell states
    with configurable fidelity (imperfect sources produce Werner states).
    """

    def __init__(self, fidelity: float = 1.0) -> None:
        """Create an entanglement source.

        Args:
            fidelity: Fidelity of produced pairs with |Phi+>. 1.0 = perfect.
                     For F < 1.0, produces Werner state:
                     rho = F*|Phi+><Phi+| + (1-F)/3 * (|Phi-><Phi-| + |Psi+><Psi+| + |Psi-><Psi-|)
        """
        if not 0 <= fidelity <= 1:
            raise ValueError(f"Fidelity must be in [0,1], got {fidelity}")
        self.fidelity = fidelity

    def generate_pair(self) -> DensityMatrix | StateVector:
        """Generate an entangled pair.

        Returns StateVector for perfect fidelity, DensityMatrix for imperfect.
        """
        if self.fidelity == 1.0:
            return StateVector.bell_state(0)

        # Werner state
        bell_states = [StateVector.bell_state(i) for i in range(4)]
        rho = np.zeros((4, 4), dtype=np.complex128)
        rho += self.fidelity * np.outer(bell_states[0].data, bell_states[0].data.conj())
        other_weight = (1 - self.fidelity) / 3
        for i in range(1, 4):
            rho += other_weight * np.outer(bell_states[i].data, bell_states[i].data.conj())

        return DensityMatrix(2, rho)


class EntanglementSwap:
    """Entanglement swapping protocol.

    Given two Bell pairs (A-B1) and (B2-C), performing a Bell measurement
    on (B1, B2) creates entanglement between (A, C) without them ever
    interacting directly.

    This extends entanglement range in quantum repeater networks.
    """

    @staticmethod
    def swap(pair_ab: StateVector, pair_bc: StateVector) -> StateVector:
        """Perform entanglement swapping.

        Takes two Bell pairs and returns the resulting entangled state
        between the outer qubits (A and C).

        Ideal swap: if both pairs are |Phi+>, output is |Phi+> between A and C.
        """
        if pair_ab.num_qubits != 2 or pair_bc.num_qubits != 2:
            raise ValueError("Both pairs must be 2-qubit states")

        # Full 4-qubit state: A, B1, B2, C
        full = np.kron(pair_ab.data, pair_bc.data)

        # Bell measurement on qubits B1 (index 1) and B2 (index 2)
        # Project onto |Phi+> on middle qubits (most likely outcome for Bell pairs)
        bell_plus = StateVector.bell_state(0).data  # |Phi+>

        # Build projector: I_A ⊗ |Phi+><Phi+|_{B1,B2} ⊗ I_C
        proj_mid = np.outer(bell_plus, bell_plus.conj())
        proj = np.kron(np.kron(np.eye(2, dtype=np.complex128), proj_mid),
                       np.eye(2, dtype=np.complex128))

        # Project and extract A-C state
        projected = proj @ full
        norm = np.linalg.norm(projected)
        if norm < 1e-15:
            raise ValueError("Swap projection failed — states may not be Bell pairs")
        projected /= norm

        # Trace out B1 and B2 (qubits 1 and 2) to get A-C state
        # Reshape to (2, 2, 2, 2) for A, B1, B2, C
        reshaped = projected.reshape(2, 2, 2, 2)
        # After projection, B1B2 are in |Phi+> state, so trace gives A-C
        ac_dm = np.einsum("ijjk->ik", reshaped)
        # Normalize
        norm = np.linalg.norm(ac_dm)
        if norm > 1e-15:
            ac_state = ac_dm.flatten() / np.linalg.norm(ac_dm.flatten())
            # This should be close to a Bell state
            return StateVector(2, ac_state)

        raise ValueError("Entanglement swap produced zero state")


def entanglement_fidelity(state: DensityMatrix | StateVector, target: int = 0) -> float:
    """Compute fidelity with respect to a target Bell state.

    Args:
        state: The state to measure
        target: Bell state index (0=Phi+, 1=Phi-, 2=Psi+, 3=Psi-)

    Returns:
        Fidelity F = <target|rho|target>
    """
    bell = StateVector.bell_state(target)

    if isinstance(state, StateVector):
        return state.fidelity(bell)
    elif isinstance(state, DensityMatrix):
        return float(np.real(bell.data.conj() @ state.data @ bell.data))
    else:
        raise TypeError(f"Expected StateVector or DensityMatrix, got {type(state)}")


def concurrence(dm: DensityMatrix) -> float:
    """Compute the concurrence of a 2-qubit density matrix.

    C(rho) = max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4)
    where lambda_i are the square roots of eigenvalues of rho * rho_tilde
    in decreasing order, and rho_tilde = (Y⊗Y) rho* (Y⊗Y).

    Concurrence ranges from 0 (separable) to 1 (maximally entangled).
    """
    if dm.num_qubits != 2:
        raise ValueError(f"Concurrence requires 2 qubits, got {dm.num_qubits}")

    rho = dm.data
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    yy = np.kron(sigma_y, sigma_y)

    rho_tilde = yy @ rho.conj() @ yy
    product = rho @ rho_tilde

    eigenvalues = np.sort(np.real(np.sqrt(np.maximum(np.linalg.eigvals(product), 0))))[::-1]
    c = eigenvalues[0] - sum(eigenvalues[1:])
    return float(max(0.0, c))
