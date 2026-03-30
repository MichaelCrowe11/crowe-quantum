"""Tests for quantum-to-music mapping."""

import numpy as np
import pytest
from crowe_quantum_audio.mapping import (
    QuantumScale,
    ScaleType,
    StateToMIDI,
    amplitude_to_velocity,
    phase_to_pitch,
    probability_to_rhythm,
)
from crowe_quantum_core.states import StateVector


class TestQuantumScale:
    def test_major_scale_notes(self):
        scale = QuantumScale(root=60, scale_type=ScaleType.MAJOR, octave_range=1)
        notes = scale.notes
        assert notes == [60, 62, 64, 65, 67, 69, 71]

    def test_pentatonic_scale(self):
        scale = QuantumScale(root=60, scale_type=ScaleType.PENTATONIC, octave_range=1)
        notes = scale.notes
        assert len(notes) == 5

    def test_map_index(self):
        scale = QuantumScale(root=60, scale_type=ScaleType.MAJOR, octave_range=1)
        assert scale.map_index(0) == 60  # C4
        assert scale.map_index(2) == 64  # E4

    def test_map_index_wraps(self):
        scale = QuantumScale(root=60, scale_type=ScaleType.MAJOR, octave_range=1)
        # Should wrap around
        assert scale.map_index(7) == 60

    def test_map_state(self):
        state = StateVector(1)  # |0⟩
        scale = QuantumScale()
        result = scale.map_state(state)
        assert len(result) == 1
        assert result[0][1] == pytest.approx(1.0)  # probability 1

    def test_map_superposition(self):
        state = StateVector(1)
        from crowe_quantum_core.gates import standard_gates
        h = standard_gates.get_gate("H")
        state.apply_gate(h.matrix(), [0])
        scale = QuantumScale()
        result = scale.map_state(state)
        assert len(result) == 2
        for _, prob in result:
            assert prob == pytest.approx(0.5, abs=1e-10)


class TestAmplitudeToVelocity:
    def test_zero_amplitude(self):
        vel = amplitude_to_velocity(0.0)
        assert vel == 20  # min velocity

    def test_max_amplitude(self):
        vel = amplitude_to_velocity(1.0)
        assert vel == 127

    def test_mid_amplitude(self):
        vel = amplitude_to_velocity(0.5)
        assert 20 < vel < 127


class TestProbabilityToRhythm:
    def test_uniform(self):
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        durations = probability_to_rhythm(probs, total_beats=4.0)
        assert len(durations) == 4
        assert sum(durations) == pytest.approx(4.0)

    def test_single_note(self):
        probs = np.array([1.0, 0.0])
        durations = probability_to_rhythm(probs, total_beats=4.0)
        assert len(durations) == 1
        assert durations[0] == pytest.approx(4.0)

    def test_empty(self):
        probs = np.array([0.0, 0.0])
        durations = probability_to_rhythm(probs, total_beats=4.0)
        assert durations == [4.0]


class TestPhaseTopPitch:
    def test_returns_notes(self):
        state = StateVector(1)
        from crowe_quantum_core.gates import standard_gates
        h = standard_gates.get_gate("H")
        state.apply_gate(h.matrix(), [0])
        result = phase_to_pitch(state)
        assert len(result) == 2
        for note, amp in result:
            assert 0 <= note <= 127
            assert amp > 0


class TestStateToMIDI:
    def test_map_ground_state(self):
        state = StateVector(1)  # |0⟩
        mapper = StateToMIDI()
        events = mapper.map(state)
        assert len(events) == 1
        assert events[0]["velocity"] > 0
        assert events[0]["probability"] == pytest.approx(1.0)

    def test_map_superposition(self):
        state = StateVector(2)
        from crowe_quantum_core.gates import standard_gates
        h = standard_gates.get_gate("H")
        state.apply_gate(h.matrix(), [0])
        state.apply_gate(h.matrix(), [1])
        mapper = StateToMIDI()
        events = mapper.map(state)
        assert len(events) == 4  # 2^2 = 4 basis states
