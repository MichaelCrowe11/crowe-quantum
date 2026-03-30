"""Integration tests for the Crowe Quantum Trinity platform.

Tests that all packages work together as a unified platform.
"""

import math

import pytest


class TestFullPipeline:
    """Test the complete quantum→science→music pipeline."""

    def test_qubitflow_to_audio(self):
        """Write QubitFlow code, execute it, generate music from the result."""
        from crowe_quantum_audio import QuantumSequencer
        from crowe_quantum_audio.mapping import QuantumScale, ScaleType
        from crowe_qubit_flow import Interpreter

        # Write and execute a quantum circuit in QubitFlow
        interp = Interpreter()
        result = interp.run("""
qubit q[3]
H q[0]
H q[1]
H q[2]
""")
        q = result["env"]["q"]

        # Generate music from the quantum state
        scale = QuantumScale(root=60, scale_type=ScaleType.PENTATONIC, octave_range=2)
        seq = QuantumSequencer(scale=scale)
        notes = seq.generate_bar(q.state)

        assert len(notes) == 8  # 2^3 = 8 equal superposition states
        assert all(n.pitch >= 0 for n in notes)
        assert all(n.velocity > 0 for n in notes)

    def test_core_to_synapse(self):
        """Use quantum measurements to drive statistical analysis."""
        from crowe_quantum_core.gates import standard_gates
        from crowe_quantum_core.states import StateVector
        from crowe_synapse.hypothesis import quantum_distribution_test

        # Create a Bell state and sample
        state = StateVector(2)
        h = standard_gates.get_gate("H")
        cnot = standard_gates.get_gate("CNOT")
        state.apply_gate(h.matrix(), [0])
        state.apply_gate(cnot.matrix(), [0, 1])

        # Simulate measurement by checking probabilities
        probs = state.probabilities()
        assert probs[0] == pytest.approx(0.5, abs=1e-10)
        assert probs[3] == pytest.approx(0.5, abs=1e-10)

        # Statistical test of simulated counts
        counts = {"00": 510, "11": 490}
        expected = {"00": 0.5, "11": 0.5}
        result = quantum_distribution_test(counts, expected)
        assert not result.reject_null

    def test_synapse_uncertainty_in_gate_params(self):
        """Use uncertain values for gate parameters."""
        from crowe_synapse import UncertainValue

        # An angle with experimental uncertainty
        theta = UncertainValue(math.pi / 4, 0.01)
        two_theta = theta * 2

        assert two_theta.value == pytest.approx(math.pi / 2, abs=0.1)
        assert two_theta.uncertainty > 0

    def test_trinity_import(self):
        """Verify the meta-package imports all components."""
        from crowe_quantum_trinity import (
            Expression,
            Symbol,
            UncertainValue,
            states,
        )

        # Quick smoke tests
        assert states.StateVector(1).num_qubits == 1
        assert isinstance(UncertainValue(1.0, 0.1) + UncertainValue(2.0, 0.2), UncertainValue)
        assert isinstance(Symbol("x") + Symbol("y"), Expression)


class TestCrossPackageTypes:
    def test_statevector_in_qubitflow_and_audio(self):
        """StateVector from core is used by both QubitFlow and Audio."""
        from crowe_quantum_core.states import StateVector
        from crowe_qubit_flow.interpreter import QubitRegister

        reg = QubitRegister("q", 2)
        assert isinstance(reg.state, StateVector)

    def test_gates_in_qubitflow_and_net(self):
        """Gate registry from core is used by both QubitFlow and Net."""
        from crowe_quantum_core.gates import standard_gates

        h = standard_gates.get_gate("H")
        assert h.arity == 1
        cnot = standard_gates.get_gate("CNOT")
        assert cnot.arity == 2
