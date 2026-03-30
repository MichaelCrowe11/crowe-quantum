"""Quantum-to-music mapping functions.

Maps quantum state properties (amplitudes, phases, probabilities,
entanglement measures) to musical parameters (pitch, velocity,
duration, rhythm).

The core insight: quantum measurement outcomes are inherently
probabilistic, making them natural generators of musical variation
within composer-defined constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from crowe_quantum_core.states import StateVector


class ScaleType(Enum):
    CHROMATIC = auto()
    MAJOR = auto()
    MINOR = auto()
    PENTATONIC = auto()
    BLUES = auto()
    WHOLE_TONE = auto()
    DORIAN = auto()
    MIXOLYDIAN = auto()
    HARMONIC_MINOR = auto()


# Intervals relative to root (in semitones)
SCALE_INTERVALS: dict[ScaleType, list[int]] = {
    ScaleType.CHROMATIC: list(range(12)),
    ScaleType.MAJOR: [0, 2, 4, 5, 7, 9, 11],
    ScaleType.MINOR: [0, 2, 3, 5, 7, 8, 10],
    ScaleType.PENTATONIC: [0, 2, 4, 7, 9],
    ScaleType.BLUES: [0, 3, 5, 6, 7, 10],
    ScaleType.WHOLE_TONE: [0, 2, 4, 6, 8, 10],
    ScaleType.DORIAN: [0, 2, 3, 5, 7, 9, 10],
    ScaleType.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
    ScaleType.HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],
}


@dataclass
class QuantumScale:
    """A musical scale that maps quantum states to pitches.

    Maps the 2^n basis states of an n-qubit system to notes
    within a given scale and octave range.
    """

    root: int = 60  # MIDI note (60 = middle C)
    scale_type: ScaleType = ScaleType.MAJOR
    octave_range: int = 2  # How many octaves to span

    @property
    def notes(self) -> list[int]:
        """Generate all MIDI notes in this scale across the octave range."""
        intervals = SCALE_INTERVALS[self.scale_type]
        result = []
        for octave in range(self.octave_range):
            for interval in intervals:
                note = self.root + octave * 12 + interval
                if 0 <= note <= 127:
                    result.append(note)
        return result

    def map_index(self, index: int) -> int:
        """Map a basis state index to a MIDI note."""
        notes = self.notes
        return notes[index % len(notes)]

    def map_state(self, state: StateVector) -> list[tuple[int, float]]:
        """Map a quantum state to (note, probability) pairs."""
        probs = state.probabilities()
        notes = self.notes
        result = []
        for i, p in enumerate(probs):
            if p > 1e-10:
                note = notes[i % len(notes)]
                result.append((note, float(p)))
        return result


def phase_to_pitch(
    state: StateVector,
    root: int = 60,
    pitch_range: int = 24,
) -> list[tuple[int, float]]:
    """Map amplitude phases to pitches.

    Phase ∈ [0, 2π) maps linearly to pitch range.
    Returns list of (midi_note, amplitude) pairs.
    """
    phases = np.angle(state.data)
    amplitudes = np.abs(state.data)
    result = []
    for phase, amp in zip(phases, amplitudes):
        if amp > 1e-10:
            # Map phase [0, 2π) to pitch range
            pitch_offset = int((phase + np.pi) / (2 * np.pi) * pitch_range)
            note = root + pitch_offset
            note = max(0, min(127, note))
            result.append((note, float(amp)))
    return result


def amplitude_to_velocity(amplitude: float, min_vel: int = 20, max_vel: int = 127) -> int:
    """Map quantum amplitude to MIDI velocity.

    |amplitude|² gives probability, but we use |amplitude| directly
    for a more musical (less extreme) dynamic range.
    """
    vel = int(min_vel + abs(amplitude) * (max_vel - min_vel))
    return max(min_vel, min(max_vel, vel))


def probability_to_rhythm(
    probabilities: NDArray[np.float64],
    total_beats: float = 4.0,
    min_duration: float = 0.125,
) -> list[float]:
    """Map measurement probabilities to note durations.

    Higher probability → longer duration. Durations sum to total_beats.
    """
    probs = probabilities[probabilities > 1e-10]
    if len(probs) == 0:
        return [total_beats]
    # Normalize and scale
    normalized = probs / probs.sum()
    durations = normalized * total_beats
    # Enforce minimum duration
    durations = np.maximum(durations, min_duration)
    # Renormalize to maintain total
    durations = durations / durations.sum() * total_beats
    return durations.tolist()


class StateToMIDI:
    """Complete quantum-state-to-MIDI mapping engine.

    Combines scale mapping, phase-to-pitch, amplitude-to-velocity,
    and probability-to-rhythm into a unified mapping pipeline.
    """

    def __init__(
        self,
        scale: QuantumScale | None = None,
        velocity_range: tuple[int, int] = (40, 120),
        beats: float = 4.0,
    ) -> None:
        self.scale = scale or QuantumScale()
        self.vel_min, self.vel_max = velocity_range
        self.beats = beats

    def map(self, state: StateVector) -> list[dict]:
        """Map a quantum state to a list of MIDI note events.

        Returns list of dicts with keys: note, velocity, duration, probability.
        """
        probs = state.probabilities()
        amplitudes = np.abs(state.data)
        durations = probability_to_rhythm(probs, self.beats)

        events = []
        dur_idx = 0
        for i, (amp, prob) in enumerate(zip(amplitudes, probs)):
            if prob > 1e-10:
                note = self.scale.map_index(i)
                velocity = amplitude_to_velocity(amp, self.vel_min, self.vel_max)
                duration = durations[dur_idx % len(durations)]
                dur_idx += 1
                events.append({
                    "note": note,
                    "velocity": velocity,
                    "duration": duration,
                    "probability": float(prob),
                })
        return events
