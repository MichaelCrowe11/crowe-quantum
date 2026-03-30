# crowe-quantum-core

Shared foundation for the Crowe Quantum Platform — quantum types, gates, states, circuit IR, and protocols.

## Installation

```bash
pip install crowe-quantum-core
```

## Features

- **Quantum States**: `ZERO`, `ONE`, `PLUS`, `MINUS`, `BELL_PHI_PLUS`, and more
- **Gate Library**: H, X, Y, Z, CNOT, CZ, Toffoli, Rx, Ry, Rz, Phase, SWAP
- **Circuit IR**: Framework-agnostic intermediate representation for quantum circuits
- **Uncertain Values**: Quantum-native numeric type with built-in uncertainty propagation
- **Protocols**: Quantum error correction, noise models, Kraus operators

## Quick Start

```python
from crowe_quantum_core import states, gates
import numpy as np

# Apply Hadamard to |0⟩ → |+⟩
plus = gates.H @ states.ZERO
print(np.allclose(plus, states.PLUS))  # True

# Build a circuit
from crowe_quantum_core.circuit_ir import CircuitIR
circuit = CircuitIR(num_qubits=2)
circuit.h(0)
circuit.cnot(0, 1)
```

## Part of the [Crowe Quantum Platform](https://github.com/MichaelCrowe11/crowe-quantum)
