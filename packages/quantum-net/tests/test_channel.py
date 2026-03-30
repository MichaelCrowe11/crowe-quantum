"""Tests for quantum communication channels."""

import numpy as np
import pytest
from crowe_quantum_core.states import DensityMatrix, StateVector
from crowe_quantum_net.channel import (
    QuantumChannel,
    amplitude_damping_channel,
    depolarizing_channel,
)


class TestDepolarizingChannel:
    def test_identity_at_p_zero(self):
        """p=0 is the identity channel."""
        ch = depolarizing_channel(0.0)
        sv = StateVector(1)  # |0>
        dm = ch.apply_to_statevector(sv)
        assert dm.data[0, 0] == pytest.approx(1.0, abs=1e-10)

    def test_maximally_mixed_at_p_one(self):
        """p=1 produces maximally depolarized state (but trace-preserving Kraus)."""
        ch = depolarizing_channel(1.0)
        sv = StateVector.from_label("0")
        dm = ch.apply_to_statevector(sv)
        # At p=1, state becomes I/2
        assert dm.data[0, 0] == pytest.approx(1 / 3, abs=0.05)

    def test_trace_preserving(self):
        ch = depolarizing_channel(0.5)
        assert ch.is_trace_preserving()

    def test_channel_fidelity_decreases(self):
        """Higher p means lower channel fidelity."""
        f_low = depolarizing_channel(0.1).channel_fidelity()
        f_high = depolarizing_channel(0.9).channel_fidelity()
        assert f_low > f_high

    def test_invalid_probability(self):
        with pytest.raises(ValueError):
            depolarizing_channel(-0.1)
        with pytest.raises(ValueError):
            depolarizing_channel(1.1)


class TestAmplitudeDampingChannel:
    def test_zero_damping_preserves(self):
        """gamma=0 preserves the state."""
        ch = amplitude_damping_channel(0.0)
        sv = StateVector.from_label("1")
        dm = ch.apply_to_statevector(sv)
        assert dm.data[1, 1] == pytest.approx(1.0, abs=1e-10)

    def test_full_damping(self):
        """gamma=1 fully decays |1> to |0>."""
        ch = amplitude_damping_channel(1.0)
        sv = StateVector.from_label("1")
        dm = ch.apply_to_statevector(sv)
        assert dm.data[0, 0] == pytest.approx(1.0, abs=1e-10)

    def test_trace_preserving(self):
        ch = amplitude_damping_channel(0.5)
        assert ch.is_trace_preserving()

    def test_partial_damping(self):
        """Partial damping mixes |0> and |1>."""
        ch = amplitude_damping_channel(0.3)
        sv = StateVector.from_label("1")
        dm = ch.apply_to_statevector(sv)
        assert dm.data[0, 0] == pytest.approx(0.3, abs=1e-10)
        assert dm.data[1, 1] == pytest.approx(0.7, abs=1e-10)

    def test_invalid_gamma(self):
        with pytest.raises(ValueError):
            amplitude_damping_channel(-0.1)


class TestQuantumChannel:
    def test_custom_channel(self):
        """Can create a custom channel with arbitrary Kraus operators."""
        # Identity channel
        k = [np.eye(2, dtype=np.complex128)]
        ch = QuantumChannel(name="identity", kraus_operators=k)
        assert ch.is_trace_preserving()

    def test_apply_to_density_matrix(self):
        """Channel works directly on density matrices."""
        ch = depolarizing_channel(0.1)
        dm = DensityMatrix(1)
        result = ch.apply_to_density_matrix(dm)
        assert result.trace() == pytest.approx(1.0, abs=1e-10)
