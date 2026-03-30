"""Tests for the quantum sequencer."""

import numpy as np
import pytest

from crowe_quantum_core.states import StateVector

from crowe_quantum_audio.sequencer import NoteEvent, QuantumSequencer


class TestNoteEvent:
    def test_repr(self):
        note = NoteEvent(pitch=60, velocity=100, start=0.0, duration=1.0)
        assert "C4" in repr(note)

    def test_repr_sharp(self):
        note = NoteEvent(pitch=61, velocity=100, start=0.0, duration=1.0)
        assert "C#" in repr(note)


class TestQuantumSequencer:
    def test_generate_bar(self):
        state = StateVector(2)
        from crowe_quantum_core.gates import standard_gates
        h = standard_gates.get_gate("H")
        state.apply_gate(h.matrix(), [0])
        state.apply_gate(h.matrix(), [1])

        seq = QuantumSequencer()
        bar = seq.generate_bar(state)
        assert len(bar) > 0
        assert all(isinstance(n, NoteEvent) for n in bar)

    def test_generate_sequence(self):
        seq = QuantumSequencer()
        notes = seq.generate_sequence(num_qubits=2, num_bars=2, rng=np.random.default_rng(42))
        assert len(notes) > 0
        # Second bar notes should start after first bar
        bar1_end = max(n.start + n.duration for n in notes if n.start < 4.0)
        bar2_notes = [n for n in notes if n.start >= 4.0]
        assert len(bar2_notes) > 0

    def test_deterministic_with_seed(self):
        seq = QuantumSequencer()
        notes1 = seq.generate_sequence(num_qubits=2, num_bars=2, rng=np.random.default_rng(42))
        notes2 = seq.generate_sequence(num_qubits=2, num_bars=2, rng=np.random.default_rng(42))
        assert len(notes1) == len(notes2)
        for n1, n2 in zip(notes1, notes2):
            assert n1.pitch == n2.pitch
            assert n1.velocity == n2.velocity

    def test_sequence_to_dict(self):
        seq = QuantumSequencer()
        notes = seq.generate_sequence(num_qubits=1, num_bars=1, rng=np.random.default_rng(42))
        data = seq.sequence_to_dict(notes)
        assert all(isinstance(d, dict) for d in data)
        assert all("pitch" in d for d in data)

    def test_custom_scale(self):
        from crowe_quantum_audio.mapping import QuantumScale, ScaleType
        scale = QuantumScale(root=48, scale_type=ScaleType.BLUES, octave_range=3)
        seq = QuantumSequencer(scale=scale)
        notes = seq.generate_sequence(num_qubits=2, num_bars=1, rng=np.random.default_rng(42))
        assert all(n.pitch >= 0 for n in notes)
