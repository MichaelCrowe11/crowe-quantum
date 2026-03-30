"""Quantum sequencer — generate musical sequences from quantum circuits.

Runs a quantum circuit repeatedly, using measurement outcomes to drive
musical decisions. Each measurement collapses the superposition into
a specific musical choice, creating variation within composer-defined
harmonic and rhythmic constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from crowe_quantum_core.gates import standard_gates
from crowe_quantum_core.states import StateVector

from crowe_quantum_audio.mapping import QuantumScale, StateToMIDI, amplitude_to_velocity


@dataclass
class NoteEvent:
    """A single note event in a quantum-generated sequence."""

    pitch: int       # MIDI note number (0-127)
    velocity: int    # MIDI velocity (0-127)
    start: float     # Start time in beats
    duration: float  # Duration in beats
    channel: int = 0

    def __repr__(self) -> str:
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        name = note_names[self.pitch % 12]
        octave = self.pitch // 12 - 1
        return f"NoteEvent({name}{octave}, vel={self.velocity}, t={self.start:.2f}, dur={self.duration:.2f})"


class QuantumSequencer:
    """Generate musical sequences from quantum states.

    Takes a quantum state (or generates one from a circuit description)
    and produces a sequence of NoteEvents that can be rendered to MIDI.
    """

    def __init__(
        self,
        scale: QuantumScale | None = None,
        bpm: float = 120.0,
        beats_per_bar: int = 4,
    ) -> None:
        self.scale = scale or QuantumScale()
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self.mapper = StateToMIDI(scale=self.scale)

    def generate_bar(self, state: StateVector) -> list[NoteEvent]:
        """Generate one bar of music from a quantum state.

        Measures the state to select notes, uses amplitudes for dynamics.
        """
        events = self.mapper.map(state)
        notes = []
        current_time = 0.0

        for event in events:
            note = NoteEvent(
                pitch=event["note"],
                velocity=event["velocity"],
                start=current_time,
                duration=event["duration"],
            )
            notes.append(note)
            current_time += event["duration"]

        return notes

    def generate_sequence(
        self,
        num_qubits: int,
        num_bars: int = 4,
        gate_pattern: list[str] | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[NoteEvent]:
        """Generate a multi-bar sequence using quantum randomness.

        Creates fresh quantum states per bar, applies gates,
        and generates musical events from the resulting states.
        """
        if rng is None:
            rng = np.random.default_rng()

        if gate_pattern is None:
            gate_pattern = ["H"]  # Default: Hadamard on all qubits

        all_notes: list[NoteEvent] = []
        bar_offset = 0.0

        for bar in range(num_bars):
            state = StateVector(num_qubits)

            # Apply gate pattern
            for gate_name in gate_pattern:
                gate = standard_gates.get_gate(gate_name)
                matrix = gate.matrix()
                for q in range(num_qubits):
                    state.apply_gate(matrix, [q])

            # Add some evolution variation per bar
            if bar > 0:
                # Apply random phase rotations for variety
                for q in range(num_qubits):
                    angle = rng.uniform(0, 2 * np.pi)
                    rz = standard_gates.get_gate("RZ", angle)
                    state.apply_gate(rz.matrix(), [q])

            bar_notes = self.generate_bar(state)

            # Offset start times for this bar
            for note in bar_notes:
                note.start += bar_offset
                all_notes.append(note)

            bar_offset += self.beats_per_bar

        return all_notes

    def sequence_to_dict(self, notes: list[NoteEvent]) -> list[dict]:
        """Convert note sequence to serializable dict format."""
        return [
            {
                "pitch": n.pitch,
                "velocity": n.velocity,
                "start": n.start,
                "duration": n.duration,
                "channel": n.channel,
            }
            for n in notes
        ]
