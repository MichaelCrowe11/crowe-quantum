# crowe-quantum-trinity

Meta-package that installs and re-exports the entire Crowe Quantum Platform. One import, everything available.

## Installation

```bash
pip install crowe-quantum-trinity
```

## Usage

```python
from crowe_quantum_trinity import (
    states, gates,           # Core types
    Interpreter,             # Qubit-Flow language
    UncertainValue,          # Uncertainty propagation
    QuantumSequencer,        # Audio synthesis
)

# Everything is available from a single import
interp = Interpreter()
result = interp.run("""
    qreg q[2];
    H q[0];
    CNOT q[0], q[1];
    measure q[0] -> c[0];
""")
print(result.counts)
```

## Included Packages

| Package | Description |
|---------|-------------|
| crowe-quantum-core | States, gates, types, CircuitIR |
| crowe-synapse | Synapse-Lang DSL compiler |
| crowe-qubit-flow | Qubit-Flow language interpreter |
| crowe-quantum-net | Quantum channels & teleportation |
| crowe-quantum-hub | Backend registry & simulator |
| crowe-quantum-viz | Bloch sphere & circuit visualization |
| crowe-quantum-audio | Quantum audio synthesis |

## Part of the [Crowe Quantum Platform](https://github.com/MichaelCrowe11/crowe-quantum)
