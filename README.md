# Crowe Quantum Platform

A federated quantum computing framework — 8 composable Python packages for quantum state manipulation, circuit design, simulation, networking, visualization, and audio synthesis.

## Installation

Install everything:

```bash
pip install crowe-quantum-trinity
```

Or install individual packages:

```bash
pip install crowe-quantum-core      # Types, gates, states, IR
pip install crowe-synapse           # Synapse-Lang DSL compiler
pip install crowe-qubit-flow        # Qubit-Flow language interpreter
pip install crowe-quantum-net       # Quantum channels & teleportation
pip install crowe-quantum-hub       # Backend registry & local simulator
pip install crowe-quantum-viz       # Bloch sphere & circuit visualization
pip install crowe-quantum-audio     # Quantum state → audio synthesis
```

## Quick Start

```python
from crowe_quantum_trinity import states, gates, Interpreter, QuantumSequencer

# Create a Bell state with Qubit-Flow
interp = Interpreter()
result = interp.run("""
    qreg q[2];
    H q[0];
    CNOT q[0], q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
""")
print(result.counts)  # {'00': ~500, '11': ~500}

# Compile Synapse-Lang
from crowe_synapse import compile_synapse
ir = compile_synapse("qubit |+⟩ → H → CNOT(q0, q1) → measure")

# Visualize on the Bloch sphere
from crowe_quantum_viz import bloch_coords, BlochSphere
coords = bloch_coords(states.PLUS)
print(f"Bloch vector: ({coords.x:.2f}, {coords.y:.2f}, {coords.z:.2f})")

# Quantum audio synthesis
seq = QuantumSequencer(sample_rate=44100)
seq.add_state(states.PLUS, duration=1.0)
audio = seq.render()
```

## Docker

```bash
docker run -it michaelcrowe11/crowe-quantum:1.0.0
```

## Package Architecture

```
crowe-quantum-trinity  (meta-package, re-exports everything)
├── crowe-quantum-core     States, gates, types, CircuitIR
├── crowe-synapse          Synapse-Lang DSL → CircuitIR compiler
├── crowe-qubit-flow       Qubit-Flow language parser & interpreter
├── crowe-quantum-net      Quantum channels, teleportation, entanglement
├── crowe-quantum-hub      Backend registry, local state-vector simulator
├── crowe-quantum-viz      Bloch sphere, probability plots, circuit drawing
└── crowe-quantum-audio    Quantum state → waveform synthesis
```

## Requirements

- Python 3.10+
- NumPy, SymPy, SciPy, Matplotlib

## Development

```bash
git clone https://github.com/MichaelCrowe11/crowe-quantum.git
cd crowe-quantum
python -m venv .venv && source .venv/bin/activate
pip install numpy sympy scipy matplotlib pytest ruff
for pkg in core synapse qubit-flow quantum-net quantum-hub quantum-viz quantum-audio trinity; do
    pip install --no-deps -e packages/$pkg
done
pytest  # 322 tests
```

## License

MIT

## Author

Michael Crowe — [Crowe Logic Inc.](https://github.com/MichaelCrowe11)
