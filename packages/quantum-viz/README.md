# crowe-quantum-viz

Quantum visualization — Bloch sphere rendering, probability bar charts, phase disks, density matrix heatmaps, and ASCII circuit drawing.

## Installation

```bash
pip install crowe-quantum-viz
```

## Features

- **Bloch Sphere**: Compute Bloch coordinates via Pauli traces, render 3D sphere data
- **Probability Plots**: Bar chart data with phase-based coloring
- **Phase Disk**: Polar plot of amplitude and phase
- **Density Heatmap**: Real/imaginary component visualization
- **State Table**: Tabular state decomposition with amplitudes, probabilities, phases
- **Circuit Drawing**: ASCII circuit diagrams from CircuitIR

## Quick Start

```python
from crowe_quantum_viz import bloch_coords, BlochSphere, CircuitDrawer
from crowe_quantum_core import states

# Bloch sphere coordinates for |+⟩
coords = bloch_coords(states.PLUS)
print(f"({coords.x:.2f}, {coords.y:.2f}, {coords.z:.2f})")  # (1.0, 0.0, 0.0)

# ASCII circuit diagram
from crowe_quantum_core.circuit_ir import CircuitIR
circuit = CircuitIR(num_qubits=2)
circuit.h(0)
circuit.cnot(0, 1)
drawer = CircuitDrawer(circuit)
print(drawer.to_text())
```

## Part of the [Crowe Quantum Platform](https://github.com/MichaelCrowe11/crowe-quantum)
