# crowe-synapse

Synapse-Lang — a domain-specific language for quantum circuit design. Compiles human-readable quantum notation into CircuitIR.

## Installation

```bash
pip install crowe-synapse
```

## Features

- **Natural Syntax**: Write quantum circuits in readable notation
- **CircuitIR Output**: Compiles to the Crowe Quantum Platform's intermediate representation
- **Gate Support**: All standard single-qubit and multi-qubit gates
- **Measurement**: Built-in measurement and classical register support

## Quick Start

```python
from crowe_synapse import compile_synapse

# Compile a Bell state circuit
ir = compile_synapse("qubit |0⟩ → H → CNOT(q0, q1) → measure")
print(ir.operations)
```

## Part of the [Crowe Quantum Platform](https://github.com/MichaelCrowe11/crowe-quantum)
