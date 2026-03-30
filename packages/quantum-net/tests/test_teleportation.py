"""Tests for quantum teleportation protocol."""

import math

import numpy as np
import pytest

from crowe_quantum_core.states import StateVector
from crowe_quantum_net.teleportation import (
    TeleportationProtocol,
    TeleportationResult,
    teleport,
)


class TestTeleportation:
    def test_teleport_zero(self):
        """Teleporting |0> produces |0> with high fidelity."""
        sv = StateVector(1)  # |0>
        result = teleport(sv, seed=42)
        assert isinstance(result, TeleportationResult)
        assert result.fidelity > 0.99
        assert result.success

    def test_teleport_one(self):
        """Teleporting |1> produces |1> with high fidelity."""
        sv = StateVector.from_label("1")
        result = teleport(sv, seed=42)
        assert result.fidelity > 0.99

    def test_teleport_superposition(self):
        """Teleporting |+> preserves superposition."""
        sv = StateVector(1, np.array([1, 1], dtype=np.complex128) / np.sqrt(2))
        result = teleport(sv, seed=42)
        assert result.fidelity > 0.99

    def test_teleport_arbitrary_state(self):
        """Teleporting an arbitrary state preserves it."""
        theta = math.pi / 3
        sv = StateVector(1, np.array([
            math.cos(theta / 2),
            math.sin(theta / 2) * np.exp(1j * math.pi / 4),
        ], dtype=np.complex128))
        result = teleport(sv, seed=42)
        assert result.fidelity > 0.99

    def test_classical_bits(self):
        """Classical bits are a tuple of two ints."""
        sv = StateVector(1)
        result = teleport(sv, seed=42)
        assert len(result.classical_bits) == 2
        assert all(b in (0, 1) for b in result.classical_bits)

    def test_rejects_multi_qubit(self):
        """Cannot teleport multi-qubit states."""
        sv = StateVector(2)
        with pytest.raises(ValueError, match="1-qubit"):
            teleport(sv)

    def test_rejects_wrong_bell_pair(self):
        """Bell pair must be 2 qubits."""
        sv = StateVector(1)
        bad_pair = StateVector(3)
        with pytest.raises(ValueError, match="2 qubits"):
            teleport(sv, bell_pair=bad_pair)

    def test_protocol_reproducible(self):
        """Same seed produces same result."""
        sv = StateVector(1)
        r1 = teleport(sv, seed=123)
        r2 = teleport(sv, seed=123)
        assert r1.classical_bits == r2.classical_bits
        assert r1.fidelity == pytest.approx(r2.fidelity)

    def test_multiple_teleportations(self):
        """Running multiple teleportations all succeed."""
        proto = TeleportationProtocol(seed=42)
        states = [
            StateVector(1),
            StateVector.from_label("1"),
            StateVector(1, np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)),
        ]
        for sv in states:
            result = proto.run(sv)
            assert result.fidelity > 0.99

    def test_repr(self):
        """TeleportationResult has readable repr."""
        result = teleport(StateVector(1), seed=42)
        r = repr(result)
        assert "fidelity" in r
        assert "bits" in r
