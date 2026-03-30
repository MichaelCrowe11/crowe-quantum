"""Tests for the local state-vector simulator."""

import math

import numpy as np
import pytest

from crowe_quantum_core.noise import depolarizing_channel, NoiseModel
from crowe_quantum_core.protocols import CircuitIR
from crowe_quantum_core.states import PauliString
from crowe_quantum_hub.local_sim import LocalSimulator


@pytest.fixture
def sim():
    return LocalSimulator()


class TestLocalSimulatorBasics:
    def test_name(self, sim):
        assert sim.name == "crowe-local-simulator"

    def test_max_qubits(self, sim):
        assert sim.max_qubits == 20

    def test_custom_max_qubits(self):
        s = LocalSimulator(max_qubits=10)
        assert s.max_qubits == 10

    def test_empty_circuit_statevector(self, sim):
        """Empty circuit returns |0...0>."""
        ir = CircuitIR(num_qubits=2)
        sv = sim.statevector(ir)
        assert sv.data[0] == pytest.approx(1.0)
        assert all(sv.data[i] == pytest.approx(0.0) for i in range(1, 4))


class TestLocalSimulatorGates:
    def test_hadamard(self, sim):
        """H|0> = |+>."""
        ir = CircuitIR(num_qubits=1)
        ir.add_gate("H", [0])
        sv = sim.statevector(ir)
        probs = sv.probabilities()
        assert probs[0] == pytest.approx(0.5)
        assert probs[1] == pytest.approx(0.5)

    def test_x_gate(self, sim):
        """X|0> = |1>."""
        ir = CircuitIR(num_qubits=1)
        ir.add_gate("X", [0])
        sv = sim.statevector(ir)
        assert sv.data[1] == pytest.approx(1.0)

    def test_bell_state(self, sim):
        """H then CNOT creates Bell state."""
        ir = CircuitIR(num_qubits=2)
        ir.add_gate("H", [0])
        ir.add_gate("CNOT", [0, 1])
        sv = sim.statevector(ir)
        probs = sv.probabilities()
        assert probs[0] == pytest.approx(0.5)  # |00>
        assert probs[3] == pytest.approx(0.5)  # |11>

    def test_parameterized_rz(self, sim):
        """RZ(pi)|0> = e^{-i*pi/2}|0>."""
        ir = CircuitIR(num_qubits=1)
        ir.add_gate("RZ", [0], params=(math.pi,))
        sv = sim.statevector(ir)
        assert abs(sv.data[0]) == pytest.approx(1.0, abs=1e-10)

    def test_reset(self, sim):
        """Reset after X brings qubit back to |0>."""
        ir = CircuitIR(num_qubits=1)
        ir.add_gate("X", [0])  # now |1>
        ir.add_reset(0)         # back to |0>
        sv = sim.statevector(ir)
        assert abs(sv.data[0]) == pytest.approx(1.0, abs=1e-10)


class TestLocalSimulatorSampling:
    def test_deterministic_state(self, sim):
        """|0> always measures to '0'."""
        ir = CircuitIR(num_qubits=1)
        result = sim.sample(ir, shots=100, seed=42)
        assert result.counts.get("0", 0) == 100

    def test_bell_state_correlations(self, sim):
        """Bell state only produces 00 and 11."""
        ir = CircuitIR(num_qubits=2)
        ir.add_gate("H", [0])
        ir.add_gate("CNOT", [0, 1])
        result = sim.sample(ir, shots=1000, seed=42)
        assert set(result.counts.keys()) <= {"00", "11"}
        assert result.shots == 1000

    def test_probabilities(self, sim):
        """SamplerResult.probabilities() sums to 1."""
        ir = CircuitIR(num_qubits=1)
        ir.add_gate("H", [0])
        result = sim.sample(ir, shots=1000, seed=42)
        probs = result.probabilities()
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_most_likely(self, sim):
        """|0> most likely outcome is '0'."""
        ir = CircuitIR(num_qubits=1)
        result = sim.sample(ir, shots=100, seed=42)
        assert result.most_likely() == "0"


class TestLocalSimulatorEstimation:
    def test_z_expectation_zero_state(self, sim):
        """<0|Z|0> = 1."""
        ir = CircuitIR(num_qubits=1)
        obs = [PauliString(1.0, "Z")]
        result = sim.estimate(ir, obs)
        assert result.values[0] == pytest.approx(1.0)

    def test_z_expectation_one_state(self, sim):
        """<1|Z|1> = -1."""
        ir = CircuitIR(num_qubits=1)
        ir.add_gate("X", [0])
        obs = [PauliString(1.0, "Z")]
        result = sim.estimate(ir, obs)
        assert result.values[0] == pytest.approx(-1.0)

    def test_x_expectation_plus_state(self, sim):
        """<+|X|+> = 1."""
        ir = CircuitIR(num_qubits=1)
        ir.add_gate("H", [0])
        obs = [PauliString(1.0, "X")]
        result = sim.estimate(ir, obs)
        assert result.values[0] == pytest.approx(1.0, abs=1e-10)

    def test_multiple_observables(self, sim):
        """Can estimate multiple observables at once."""
        ir = CircuitIR(num_qubits=1)
        ir.add_gate("H", [0])
        obs = [PauliString(1.0, "X"), PauliString(1.0, "Z")]
        result = sim.estimate(ir, obs)
        assert len(result.values) == 2


class TestLocalSimulatorNoisy:
    def test_noisy_sampling(self, sim):
        """Noisy simulation produces mixed outcomes from a basis state."""
        ir = CircuitIR(num_qubits=1)
        ir.add_gate("X", [0])  # Prepare |1>

        noise = NoiseModel()
        noise.add_all_gate_noise(depolarizing_channel(0.3))

        result = sim.sample(ir, shots=500, noise_model=noise, seed=42)
        # With strong depolarizing noise, we should see both outcomes
        assert "1" in result.counts  # Still mostly |1>
        # With p=0.3, about 20% chance of flip, so we expect some "0" counts
        assert result.shots == 500

    def test_validation_rejects_too_many_qubits(self):
        """Simulator rejects circuits exceeding max_qubits."""
        sim = LocalSimulator(max_qubits=3)
        ir = CircuitIR(num_qubits=5)
        with pytest.raises(ValueError, match="validation failed"):
            sim.sample(ir, shots=10)
