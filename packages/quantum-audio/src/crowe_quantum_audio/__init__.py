"""Crowe Quantum Audio — quantum-music bridge for algorithmic composition."""

__version__ = "1.0.0"

from crowe_quantum_audio.mapping import (
    QuantumScale,
    StateToMIDI,
    amplitude_to_velocity,
    phase_to_pitch,
    probability_to_rhythm,
)
from crowe_quantum_audio.sequencer import QuantumSequencer, NoteEvent

__all__ = [
    "NoteEvent",
    "QuantumScale",
    "QuantumSequencer",
    "StateToMIDI",
    "amplitude_to_velocity",
    "phase_to_pitch",
    "probability_to_rhythm",
]
