# crowe-quantum-audio

Quantum state → audio synthesis. Maps quantum probability amplitudes and phases to waveforms for sonification.

## Installation

```bash
pip install crowe-quantum-audio
```

## Features

- **Quantum Sequencer**: Build audio sequences from quantum states
- **State Sonification**: Map amplitudes to frequencies, phases to stereo panning
- **Waveform Rendering**: Generate PCM audio arrays at configurable sample rates
- **Multi-state Sequences**: Chain quantum states into evolving audio timelines

## Quick Start

```python
from crowe_quantum_audio import QuantumSequencer
from crowe_quantum_core import states

seq = QuantumSequencer(sample_rate=44100)
seq.add_state(states.PLUS, duration=1.0)
seq.add_state(states.MINUS, duration=1.0)
audio = seq.render()
print(f"Audio samples: {len(audio)}")
```

## Part of the [Crowe Quantum Platform](https://github.com/MichaelCrowe11/crowe-quantum)
