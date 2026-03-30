# crowe-quantum-hub

Backend registry and local quantum simulator — run circuits on a state-vector simulator or register custom backends.

## Installation

```bash
pip install crowe-quantum-hub
```

## Features

- **Local Simulator**: Pure state-vector simulation up to 20 qubits
- **Noisy Simulation**: Density matrix promotion with Kraus operator noise
- **Backend Registry**: Register, discover, and query quantum backends by capabilities
- **Sampling**: Born-rule measurement with configurable shots
- **Expectation Values**: Compute ⟨ψ|O|ψ⟩ for arbitrary observables

## Quick Start

```python
from crowe_quantum_hub import LocalSimulator, registry

sim = LocalSimulator()

# List available backends
print(registry.list_backends())  # ['local-simulator']

# Find simulators with at least 10 qubits
backends = registry.find(min_qubits=10, simulator_only=True)
```

## Part of the [Crowe Quantum Platform](https://github.com/MichaelCrowe11/crowe-quantum)
