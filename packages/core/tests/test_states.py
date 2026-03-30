"""Tests for quantum state representations."""

import numpy as np
import pytest
from crowe_quantum_core.states import DensityMatrix, PauliString, StateVector


class TestStateVector:
    def test_default_state(self):
        sv = StateVector(2)
        assert sv.num_qubits == 2
        assert sv.dim == 4
        np.testing.assert_allclose(sv.data[0], 1.0)
        np.testing.assert_allclose(np.sum(np.abs(sv.data) ** 2), 1.0)

    def test_from_label(self):
        sv = StateVector.from_label("01")
        assert sv.num_qubits == 2
        np.testing.assert_allclose(sv.data[1], 1.0)  # |01⟩ = index 1

    def test_bell_state(self):
        sv = StateVector.bell_state(0)  # |Φ+⟩
        probs = sv.probabilities()
        np.testing.assert_allclose(probs[0], 0.5, atol=1e-10)  # |00⟩
        np.testing.assert_allclose(probs[3], 0.5, atol=1e-10)  # |11⟩
        np.testing.assert_allclose(probs[1], 0.0, atol=1e-10)  # |01⟩
        np.testing.assert_allclose(probs[2], 0.0, atol=1e-10)  # |10⟩

    def test_ghz_state(self):
        sv = StateVector.ghz_state(3)
        probs = sv.probabilities()
        np.testing.assert_allclose(probs[0], 0.5, atol=1e-10)   # |000⟩
        np.testing.assert_allclose(probs[7], 0.5, atol=1e-10)   # |111⟩

    def test_measurement_collapses(self):
        sv = StateVector.bell_state(0)
        rng = np.random.default_rng(42)
        outcome = sv.measure(rng)
        assert outcome in (0, 3)  # Only |00⟩ or |11⟩
        # After measurement, state is collapsed
        np.testing.assert_allclose(sv.data[outcome], 1.0)

    def test_single_qubit_measurement(self):
        sv = StateVector.bell_state(0)
        rng = np.random.default_rng(42)
        bit = sv.measure_qubit(0, rng)
        assert bit in (0, 1)
        # After measuring qubit 0, the state should be |00⟩ or |11⟩
        probs = sv.probabilities()
        if bit == 0:
            np.testing.assert_allclose(probs[0], 1.0, atol=1e-10)
        else:
            np.testing.assert_allclose(probs[3], 1.0, atol=1e-10)

    def test_norm(self):
        sv = StateVector(3)
        np.testing.assert_allclose(sv.norm(), 1.0)

    def test_fidelity(self):
        sv1 = StateVector.from_label("00")
        sv2 = StateVector.from_label("00")
        assert sv1.fidelity(sv2) == pytest.approx(1.0)

        sv3 = StateVector.from_label("01")
        assert sv1.fidelity(sv3) == pytest.approx(0.0)

    def test_apply_hadamard(self):
        sv = StateVector(1)  # |0⟩
        h_matrix = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        sv.apply_gate(h_matrix, [0])
        # Should be |+⟩ = (|0⟩ + |1⟩)/√2
        np.testing.assert_allclose(sv.probabilities(), [0.5, 0.5], atol=1e-10)

    def test_apply_cnot(self):
        # Start with |10⟩, apply CNOT -> should get |11⟩
        sv = StateVector.from_label("10")
        cnot = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=np.complex128,
        )
        sv.apply_gate(cnot, [0, 1])
        np.testing.assert_allclose(sv.data[3], 1.0, atol=1e-10)  # |11⟩

    def test_to_density_matrix(self):
        sv = StateVector.from_label("0")
        dm = sv.to_density_matrix()
        assert dm.is_pure()
        np.testing.assert_allclose(dm.data[0, 0], 1.0, atol=1e-10)

    def test_repr(self):
        sv = StateVector.from_label("01")
        r = repr(sv)
        assert "01" in r

    def test_max_qubits_limit(self):
        with pytest.raises(ValueError, match="max"):
            StateVector(31)


class TestDensityMatrix:
    def test_default_state(self):
        dm = DensityMatrix(1)
        assert dm.num_qubits == 1
        np.testing.assert_allclose(dm.data[0, 0], 1.0)

    def test_maximally_mixed(self):
        dm = DensityMatrix.maximally_mixed(2)
        np.testing.assert_allclose(dm.purity(), 0.25, atol=1e-10)
        assert not dm.is_pure()

    def test_pure_state_purity(self):
        sv = StateVector.from_label("0")
        dm = sv.to_density_matrix()
        np.testing.assert_allclose(dm.purity(), 1.0, atol=1e-10)

    def test_von_neumann_entropy(self):
        # Pure state has zero entropy
        dm_pure = StateVector.from_label("0").to_density_matrix()
        np.testing.assert_allclose(dm_pure.von_neumann_entropy(), 0.0, atol=1e-10)

        # Maximally mixed 1-qubit has entropy = 1
        dm_mixed = DensityMatrix.maximally_mixed(1)
        np.testing.assert_allclose(dm_mixed.von_neumann_entropy(), 1.0, atol=1e-10)

    def test_partial_trace(self):
        bell = StateVector.bell_state(0)
        dm = bell.to_density_matrix()
        reduced = dm.partial_trace([0])  # Trace out qubit 1
        # Reduced state of a Bell state is maximally mixed
        np.testing.assert_allclose(reduced.purity(), 0.5, atol=1e-10)


class TestPauliString:
    def test_single_pauli(self):
        z = PauliString(coefficient=1.0, paulis="Z")
        mat = z.to_matrix()
        expected = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        np.testing.assert_allclose(mat, expected)

    def test_tensor_product(self):
        zz = PauliString(coefficient=0.5, paulis="ZZ")
        mat = zz.to_matrix()
        assert mat.shape == (4, 4)
        # ZZ|00⟩ = +|00⟩, ZZ|01⟩ = -|01⟩, ZZ|10⟩ = -|10⟩, ZZ|11⟩ = +|11⟩
        np.testing.assert_allclose(mat[0, 0], 0.5)
        np.testing.assert_allclose(mat[1, 1], -0.5)

    def test_commutation(self):
        xx = PauliString(coefficient=1.0, paulis="XX")
        zz = PauliString(coefficient=1.0, paulis="ZZ")
        assert xx.commutes_with(zz)  # XX and ZZ commute

        xz = PauliString(coefficient=1.0, paulis="XZ")
        zx = PauliString(coefficient=1.0, paulis="ZX")
        assert xz.commutes_with(zx)  # XZ and ZX commute (even anticommutation count)
