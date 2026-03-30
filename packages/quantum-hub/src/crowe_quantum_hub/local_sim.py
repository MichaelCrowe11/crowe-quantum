"""Local state-vector quantum simulator.

A full-fidelity simulator that executes CircuitIR by applying unitary
gates to a StateVector, with optional noise model support via density
matrix promotion.
"""

from __future__ import annotations

import numpy as np
from crowe_quantum_core.gates import standard_gates
from crowe_quantum_core.noise import NoiseModel
from crowe_quantum_core.protocols import (
    Backend,
    CircuitIR,
    EstimatorResult,
    SamplerResult,
)
from crowe_quantum_core.states import DensityMatrix, PauliString, StateVector


class LocalSimulator(Backend):
    """Pure-state simulator with optional noise model.

    Supports up to 20 qubits by default (2^20 = 1M amplitudes).
    For noisy simulation, promotes to density matrix representation.
    """

    def __init__(self, max_qubits: int = 20) -> None:
        self._max_qubits = max_qubits

    @property
    def name(self) -> str:
        return "crowe-local-simulator"

    @property
    def max_qubits(self) -> int:
        return self._max_qubits

    def sample(
        self,
        circuit: CircuitIR,
        shots: int = 1024,
        noise_model: NoiseModel | None = None,
        seed: int | None = None,
    ) -> SamplerResult:
        """Execute circuit and collect measurement statistics."""
        issues = self.validate_circuit(circuit)
        if issues:
            raise ValueError(f"Circuit validation failed: {'; '.join(issues)}")

        rng = np.random.default_rng(seed)
        counts: dict[str, int] = {}

        if noise_model is None:
            # Pure-state simulation: run once, sample from probabilities
            sv = self._simulate_statevector(circuit)
            probs = sv.probabilities()
            outcomes = rng.choice(sv.dim, size=shots, p=probs)
            for outcome in outcomes:
                bs = format(outcome, f"0{circuit.num_qubits}b")
                counts[bs] = counts.get(bs, 0) + 1
        else:
            # Noisy simulation: run per-shot with density matrix
            for _ in range(shots):
                dm = self._simulate_noisy(circuit, noise_model, rng)
                probs = np.real(np.diag(dm.data))
                probs = np.maximum(probs, 0)
                probs /= probs.sum()
                outcome = rng.choice(dm.dim, p=probs)
                bs = format(outcome, f"0{circuit.num_qubits}b")
                counts[bs] = counts.get(bs, 0) + 1

        return SamplerResult(counts=counts, shots=shots)

    def estimate(
        self,
        circuit: CircuitIR,
        observables: list[PauliString],
        shots: int = 1024,
        noise_model: NoiseModel | None = None,
        seed: int | None = None,
    ) -> EstimatorResult:
        """Estimate expectation values of Pauli observables."""
        if noise_model is None:
            sv = self._simulate_statevector(circuit)
            values = []
            for obs in observables:
                mat = obs.to_matrix()
                ev = float(np.real(sv.expectation(mat)))
                values.append(ev)
            return EstimatorResult(values=values)
        else:
            rng = np.random.default_rng(seed)
            dm = self._simulate_noisy(circuit, noise_model, rng)
            values = []
            for obs in observables:
                mat = obs.to_matrix()
                ev = float(np.real(dm.expectation(mat)))
                values.append(ev)
            return EstimatorResult(values=values)

    def statevector(self, circuit: CircuitIR) -> StateVector:
        """Get the final state vector (noiseless)."""
        return self._simulate_statevector(circuit)

    def _simulate_statevector(self, circuit: CircuitIR) -> StateVector:
        """Execute circuit on a pure state vector."""
        sv = StateVector(circuit.num_qubits)

        for op in circuit.operations:
            if op.kind == "gate":
                gate = standard_gates.get_gate(op.gate_name, *op.params)
                sv.apply_gate(gate.matrix(), op.qubits)
            elif op.kind == "reset":
                q = op.qubits[0]
                outcome = sv.measure_qubit(q)
                if outcome == 1:
                    x = standard_gates.get_gate("X")
                    sv.apply_gate(x.matrix(), [q])
            # measure and barrier are no-ops in statevector simulation

        return sv

    def _simulate_noisy(
        self, circuit: CircuitIR, noise_model: NoiseModel, rng: np.random.Generator
    ) -> DensityMatrix:
        """Execute circuit with noise as a density matrix."""
        sv = StateVector(circuit.num_qubits)
        dm = sv.to_density_matrix()

        for op in circuit.operations:
            if op.kind == "gate":
                gate = standard_gates.get_gate(op.gate_name, *op.params)
                mat = gate.matrix()

                # Apply unitary: rho -> U rho U†
                if len(op.qubits) == 1:
                    full = _embed_single(mat, op.qubits[0], circuit.num_qubits)
                else:
                    full = _embed_multi(mat, op.qubits, circuit.num_qubits)
                dm._data = full @ dm.data @ full.conj().T

                # Apply noise after gate
                noise = noise_model.get_noise_for_gate(op.gate_name)
                if noise and noise.kraus_operators:
                    for q in op.qubits:
                        embedded_kraus = [
                            _embed_single(k, q, circuit.num_qubits)
                            for k in noise.kraus_operators
                        ]
                        new_dm = np.zeros_like(dm.data)
                        for ek in embedded_kraus:
                            new_dm += ek @ dm.data @ ek.conj().T
                        dm._data = new_dm

        return dm


def _embed_single(
    gate: np.ndarray, qubit: int, num_qubits: int
) -> np.ndarray:
    """Embed a single-qubit gate into the full Hilbert space."""
    ops = []
    for i in range(num_qubits):
        if i == qubit:
            ops.append(gate)
        else:
            ops.append(np.eye(2, dtype=np.complex128))
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _embed_multi(
    gate: np.ndarray, qubits: list[int], num_qubits: int
) -> np.ndarray:
    """Embed a multi-qubit gate into the full Hilbert space.

    Uses the permutation approach: permute qubits so targets are adjacent,
    apply gate, then permute back.
    """
    dim = 2**num_qubits
    result = np.zeros((dim, dim), dtype=np.complex128)
    k = len(qubits)
    2**k

    for i in range(dim):
        for j in range(dim):
            # Extract the bits at gate qubit positions
            row_bits = 0
            col_bits = 0
            for idx, q in enumerate(qubits):
                row_bits |= ((i >> (num_qubits - 1 - q)) & 1) << (k - 1 - idx)
                col_bits |= ((j >> (num_qubits - 1 - q)) & 1) << (k - 1 - idx)

            # Check non-gate qubits match
            match = True
            for q in range(num_qubits):
                if q not in qubits:
                    if ((i >> (num_qubits - 1 - q)) & 1) != ((j >> (num_qubits - 1 - q)) & 1):
                        match = False
                        break

            if match:
                result[i, j] = gate[row_bits, col_bits]

    return result
