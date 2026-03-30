"""Tests for the QubitFlow interpreter."""

import math

import numpy as np
import pytest

from crowe_qubit_flow.interpreter import (
    Environment,
    Interpreter,
    InterpreterError,
    QubitRegister,
)


class TestEnvironment:
    def test_set_and_get(self):
        env = Environment()
        env.set("x", 42)
        assert env.get("x") == 42

    def test_parent_lookup(self):
        parent = Environment()
        parent.set("x", 10)
        child = parent.child()
        assert child.get("x") == 10

    def test_child_shadows(self):
        parent = Environment()
        parent.set("x", 10)
        child = parent.child()
        child.set("x", 20)
        assert child.get("x") == 20
        assert parent.get("x") == 10

    def test_missing_key(self):
        env = Environment()
        with pytest.raises(KeyError):
            env.get("nonexistent")


class TestQubitRegister:
    def test_creation(self):
        reg = QubitRegister("q", 2)
        assert reg.size == 2
        assert reg.state.num_qubits == 2

    def test_initial_state(self):
        reg = QubitRegister("q", 1)
        probs = reg.state.probabilities()
        np.testing.assert_allclose(probs[0], 1.0)


class TestInterpreterBasics:
    def test_variable_assignment(self):
        interp = Interpreter()
        result = interp.run("x = 42")
        assert result["env"]["x"] == 42

    def test_arithmetic(self):
        interp = Interpreter()
        result = interp.run("x = 2 + 3 * 4")
        assert result["env"]["x"] == 14

    def test_comparison(self):
        interp = Interpreter()
        result = interp.run("x = 5 > 3")
        assert result["env"]["x"] is True

    def test_boolean_logic(self):
        interp = Interpreter()
        result = interp.run("x = true and false")
        assert result["env"]["x"] is False

    def test_builtin_pi(self):
        interp = Interpreter()
        result = interp.run("x = pi")
        assert result["env"]["x"] == math.pi

    def test_unary_minus(self):
        interp = Interpreter()
        result = interp.run("x = -5")
        assert result["env"]["x"] == -5


class TestQubitOperations:
    def test_qubit_declaration(self):
        interp = Interpreter()
        result = interp.run("qubit q[2]")
        q = result["env"]["q"]
        assert isinstance(q, QubitRegister)
        assert q.size == 2

    def test_hadamard_gate(self):
        interp = Interpreter()
        result = interp.run("qubit q\nH q")
        q = result["env"]["q"]
        probs = q.state.probabilities()
        np.testing.assert_allclose(probs[0], 0.5, atol=1e-10)
        np.testing.assert_allclose(probs[1], 0.5, atol=1e-10)

    def test_pauli_x_gate(self):
        interp = Interpreter()
        result = interp.run("qubit q\nX q")
        q = result["env"]["q"]
        probs = q.state.probabilities()
        np.testing.assert_allclose(probs[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(probs[1], 1.0, atol=1e-10)

    def test_measurement(self):
        interp = Interpreter()
        result = interp.run("qubit q\nX q\nmeasure q")
        assert len(result["measurements"]) == 1
        assert result["measurements"][0]["result"] == 1

    def test_superpose(self):
        interp = Interpreter()
        result = interp.run("qubit q\nsuperpose q")
        q = result["env"]["q"]
        probs = q.state.probabilities()
        np.testing.assert_allclose(probs[0], 0.5, atol=1e-10)

    def test_entangle(self):
        interp = Interpreter()
        result = interp.run("qubit q[2]\nentangle q[0], q[1]")
        q = result["env"]["q"]
        probs = q.state.probabilities()
        # Bell state: |00⟩ + |11⟩ with equal probability
        np.testing.assert_allclose(probs[0], 0.5, atol=1e-10)
        np.testing.assert_allclose(probs[3], 0.5, atol=1e-10)

    def test_reset(self):
        interp = Interpreter()
        # X puts qubit in |1⟩, reset should bring it back to |0⟩
        result = interp.run("qubit q\nX q\nreset q")
        q = result["env"]["q"]
        probs = q.state.probabilities()
        np.testing.assert_allclose(probs[0], 1.0, atol=1e-10)


class TestGateParameters:
    def test_rx_gate(self):
        interp = Interpreter()
        result = interp.run("qubit q\nRX(pi) q")
        q = result["env"]["q"]
        probs = q.state.probabilities()
        # RX(pi)|0⟩ = -i|1⟩
        np.testing.assert_allclose(probs[1], 1.0, atol=1e-10)

    def test_rz_gate(self):
        interp = Interpreter()
        result = interp.run("qubit q\nH q\nRZ(pi) q")
        q = result["env"]["q"]
        probs = q.state.probabilities()
        # RZ only changes phase, probabilities remain 50/50
        np.testing.assert_allclose(probs[0], 0.5, atol=1e-10)

    def test_cnot_gate(self):
        interp = Interpreter()
        result = interp.run("qubit q[2]\nX q[0]\nCNOT q[0], q[1]")
        q = result["env"]["q"]
        probs = q.state.probabilities()
        # |10⟩ -> |11⟩
        np.testing.assert_allclose(probs[3], 1.0, atol=1e-10)


class TestControlFlow:
    def test_if_true(self):
        interp = Interpreter()
        result = interp.run("x = 0\nif true:\n    x = 1\n")
        assert result["env"]["x"] == 1

    def test_if_false(self):
        interp = Interpreter()
        result = interp.run("x = 0\nif false:\n    x = 1\n")
        assert result["env"]["x"] == 0

    def test_if_else(self):
        interp = Interpreter()
        result = interp.run("x = 0\nif false:\n    x = 1\nelse:\n    x = 2\n")
        assert result["env"]["x"] == 2

    def test_for_loop(self):
        interp = Interpreter()
        result = interp.run("x = 0\nfor i in range(5):\n    x = x + 1\n")
        assert result["env"]["x"] == 5

    def test_while_loop(self):
        interp = Interpreter()
        result = interp.run("x = 0\nwhile x < 3:\n    x = x + 1\n")
        assert result["env"]["x"] == 3


class TestFunctions:
    def test_function_def_and_call(self):
        source = "def double(n):\n    return n * 2\nx = double(21)\n"
        interp = Interpreter()
        result = interp.run(source)
        assert result["env"]["x"] == 42


class TestCircuit:
    def test_circuit_def_and_call(self):
        source = """circuit bell():
    qubit q[2]
    H q[0]
    CNOT q[0], q[1]
    return q
result = bell()
"""
        interp = Interpreter()
        result = interp.run(source)
        q = result["env"]["result"]
        assert isinstance(q, QubitRegister)
        probs = q.state.probabilities()
        np.testing.assert_allclose(probs[0], 0.5, atol=1e-10)
        np.testing.assert_allclose(probs[3], 0.5, atol=1e-10)


class TestComplexNumbers:
    def test_complex_literal(self):
        interp = Interpreter()
        result = interp.run("x = 3i")
        assert result["env"]["x"] == 3j
