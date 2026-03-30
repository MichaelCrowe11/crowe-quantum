"""Tests for quantum noise models."""

import numpy as np
import pytest

from crowe_quantum_core.noise import (
    NoiseModel,
    amplitude_damping_channel,
    depolarizing_channel,
    phase_damping_channel,
    thermal_relaxation_channel,
)


class TestNoiseChannels:
    def test_depolarizing_trace_preserving(self):
        for p in [0.0, 0.01, 0.1, 0.5, 1.0]:
            ch = depolarizing_channel(p)
            assert ch.is_trace_preserving()

    def test_amplitude_damping_trace_preserving(self):
        for gamma in [0.0, 0.1, 0.5, 1.0]:
            ch = amplitude_damping_channel(gamma)
            assert ch.is_trace_preserving()

    def test_phase_damping_trace_preserving(self):
        for gamma in [0.0, 0.1, 0.5, 1.0]:
            ch = phase_damping_channel(gamma)
            assert ch.is_trace_preserving()

    def test_depolarizing_identity_at_zero(self):
        ch = depolarizing_channel(0.0)
        rho = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=np.complex128)
        result = ch.apply(rho)
        np.testing.assert_allclose(result, rho, atol=1e-10)

    def test_depolarizing_fully_depolarized(self):
        ch = depolarizing_channel(0.75)  # p=3/4 gives maximally mixed
        rho = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        result = ch.apply(rho)
        # ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ) = I/2 when p=3/4
        np.testing.assert_allclose(result, np.eye(2) / 2, atol=1e-10)

    def test_amplitude_damping_decays_excited(self):
        ch = amplitude_damping_channel(1.0)
        rho_excited = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        result = ch.apply(rho_excited)
        # Full decay: |1⟩ -> |0⟩
        np.testing.assert_allclose(result[0, 0], 1.0, atol=1e-10)

    def test_invalid_probability(self):
        with pytest.raises(ValueError):
            depolarizing_channel(-0.1)
        with pytest.raises(ValueError):
            depolarizing_channel(1.5)

    def test_thermal_relaxation(self):
        ch = thermal_relaxation_channel(t1=50e-6, t2=30e-6, time=10e-6)
        assert ch.is_trace_preserving()

    def test_thermal_t2_constraint(self):
        with pytest.raises(ValueError, match="T2"):
            thermal_relaxation_channel(t1=10e-6, t2=25e-6, time=5e-6)


class TestNoiseModel:
    def test_add_gate_noise(self):
        model = NoiseModel()
        ch = depolarizing_channel(0.01)
        model.add_gate_noise("CNOT", ch)
        assert model.get_noise_for_gate("CNOT") is ch
        assert model.get_noise_for_gate("H") is None

    def test_all_gate_noise(self):
        model = NoiseModel()
        ch = depolarizing_channel(0.001)
        model.add_all_gate_noise(ch)
        assert model.get_noise_for_gate("H") is ch
        assert model.get_noise_for_gate("CNOT") is ch
