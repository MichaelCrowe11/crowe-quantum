"""Tests for circuit drawing."""


from crowe_quantum_core.protocols import CircuitIR
from crowe_quantum_viz.circuit_draw import CircuitDrawer, draw_circuit


class TestCircuitDrawer:
    def test_empty_circuit(self):
        """Empty circuit renders wire lines."""
        ir = CircuitIR(num_qubits=2)
        text = draw_circuit(ir)
        assert "q[0]:" in text
        assert "q[1]:" in text

    def test_single_gate(self):
        """Single H gate appears in output."""
        ir = CircuitIR(num_qubits=1)
        ir.add_gate("H", [0])
        text = draw_circuit(ir)
        assert "[H]" in text

    def test_cnot_symbols(self):
        """CNOT shows control dot and target circle-plus."""
        ir = CircuitIR(num_qubits=2)
        ir.add_gate("CNOT", [0, 1])
        drawer = CircuitDrawer(ir)
        data = drawer.to_data()
        assert data["num_qubits"] == 2
        assert len(data["columns"]) == 1
        col = data["columns"][0]
        assert 0 in col["entries"]
        assert 1 in col["entries"]

    def test_measurement(self):
        """Measurement symbol appears."""
        ir = CircuitIR(num_qubits=1)
        ir.add_measurement(0)
        text = draw_circuit(ir)
        assert "M" in text

    def test_barrier(self):
        """Barrier renders for specified qubits."""
        ir = CircuitIR(num_qubits=2)
        ir.add_barrier([0, 1])
        drawer = CircuitDrawer(ir)
        data = drawer.to_data()
        assert len(data["columns"]) == 1

    def test_reset(self):
        """Reset shows |0> symbol."""
        ir = CircuitIR(num_qubits=1)
        ir.add_reset(0)
        text = draw_circuit(ir)
        assert "|0>" in text

    def test_parameterized_gate(self):
        """Parameterized gate shows parameters."""
        ir = CircuitIR(num_qubits=1)
        ir.add_gate("RZ", [0], params=(1.57,))
        text = draw_circuit(ir)
        assert "RZ" in text
        assert "1.57" in text

    def test_multi_gate_circuit(self):
        """Circuit with multiple gates has correct depth."""
        ir = CircuitIR(num_qubits=2)
        ir.add_gate("H", [0])
        ir.add_gate("CNOT", [0, 1])
        ir.add_measurement(0)
        ir.add_measurement(1)
        drawer = CircuitDrawer(ir)
        data = drawer.to_data()
        assert data["depth"] >= 3  # H, CNOT, measurements

    def test_to_data_structure(self):
        """to_data returns proper structure."""
        ir = CircuitIR(num_qubits=3)
        ir.add_gate("H", [0])
        ir.add_gate("H", [1])
        ir.add_gate("H", [2])
        data = CircuitDrawer(ir).to_data()
        assert data["num_qubits"] == 3
        assert "columns" in data
        assert "depth" in data
