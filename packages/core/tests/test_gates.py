"""Tests for quantum gate registry."""

import numpy as np
import pytest

from crowe_quantum_core.gates import Gate, GateRegistry, standard_gates


class TestGateRegistry:
    def test_standard_gates_loaded(self):
        gates = standard_gates.list_gates()
        assert "H" in gates
        assert "CNOT" in gates
        assert "TOFFOLI" in gates
        assert len(gates) >= 20

    def test_alias_resolution(self):
        assert standard_gates.resolve("CX") == "CNOT"
        assert standard_gates.resolve("CCX") == "TOFFOLI"
        assert standard_gates.resolve("CSWAP") == "FREDKIN"

    def test_get_gate(self):
        h = standard_gates.get_gate("H")
        assert h.name == "H"
        assert h.arity == 1

    def test_parameterized_gate(self):
        rx = standard_gates.get_gate("RX", np.pi)
        mat = rx.matrix()
        # RX(π) should be -iX
        expected = np.array([[0, -1j], [-1j, 0]], dtype=np.complex128)
        np.testing.assert_allclose(mat, expected, atol=1e-10)

    def test_all_gates_unitary(self):
        for name in standard_gates.list_gates():
            spec = standard_gates.get_spec(name)
            if spec and spec.num_params == 0:
                gate = standard_gates.get_gate(name)
                assert gate.is_unitary(), f"Gate {name} is not unitary"

    def test_rotation_gates_unitary(self):
        for name in ["RX", "RY", "RZ"]:
            for theta in [0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi]:
                gate = standard_gates.get_gate(name, theta)
                assert gate.is_unitary(), f"{name}({theta}) is not unitary"

    def test_u_gate_unitary(self):
        gate = standard_gates.get_gate("U", np.pi / 3, np.pi / 4, np.pi / 6)
        assert gate.is_unitary()

    def test_adjoint(self):
        h = standard_gates.get_gate("H")
        mat = h.matrix()
        adj = h.adjoint()
        # H is self-adjoint: H† = H
        np.testing.assert_allclose(mat, adj, atol=1e-10)

    def test_pauli_involutions(self):
        """X² = Y² = Z² = I."""
        for name in ["X", "Y", "Z"]:
            gate = standard_gates.get_gate(name)
            mat = gate.matrix()
            np.testing.assert_allclose(mat @ mat, np.eye(2), atol=1e-10)

    def test_unknown_gate_error(self):
        with pytest.raises(Exception, match="Unknown gate"):
            standard_gates.get_gate("NONEXISTENT")

    def test_wrong_param_count(self):
        with pytest.raises(Exception):
            standard_gates.get_gate("RX")  # needs 1 param

    def test_cnot_matrix(self):
        cnot = standard_gates.get_gate("CNOT")
        mat = cnot.matrix()
        # CNOT flips target when control is |1⟩
        assert mat.shape == (4, 4)
        np.testing.assert_allclose(mat[2, 3], 1.0, atol=1e-10)
        np.testing.assert_allclose(mat[3, 2], 1.0, atol=1e-10)
