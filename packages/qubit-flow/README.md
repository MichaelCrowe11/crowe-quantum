# crowe-qubit-flow

Qubit-Flow — a quantum programming language with a full parser and interpreter. Supports quantum registers, gates, measurement, classical control, loops, and teleportation.

## Installation

```bash
pip install crowe-qubit-flow
```

## Features

- **Full Language**: Variables, loops, conditionals, functions
- **Quantum Registers**: `qreg`, `creg` declarations
- **Gate Set**: H, X, Y, Z, CNOT, CZ, Toffoli, Rx, Ry, Rz, SWAP, and more
- **Measurement**: Born-rule sampling with configurable shots
- **Built-in Protocols**: Teleportation, barrier, reset

## Quick Start

```python
from crowe_qubit_flow import Interpreter

interp = Interpreter()
result = interp.run("""
    qreg q[2];
    H q[0];
    CNOT q[0], q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
""", shots=1000)

print(result.counts)  # {'00': ~500, '11': ~500}
```

## Part of the [Crowe Quantum Platform](https://github.com/MichaelCrowe11/crowe-quantum)
