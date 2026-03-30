# crowe-quantum-net

Quantum networking primitives — channels, entanglement sources, entanglement swapping, teleportation, and fidelity metrics.

## Installation

```bash
pip install crowe-quantum-net
```

## Features

- **Quantum Channels**: Depolarizing, amplitude damping, custom Kraus operators
- **Entanglement Source**: Pure Bell states and Werner (noisy) states
- **Entanglement Swapping**: Extend entanglement across network nodes
- **Teleportation Protocol**: Full Bell-measurement teleportation with Pauli corrections
- **Metrics**: Fidelity, concurrence, trace preservation checks

## Quick Start

```python
from crowe_quantum_net import (
    depolarizing_channel, EntanglementSource,
    TeleportationProtocol, concurrence
)
import numpy as np

# Teleport a qubit state
proto = TeleportationProtocol()
state = np.array([0.6, 0.8], dtype=complex)  # arbitrary qubit
result = proto.run(state)
print(f"Fidelity: {result.fidelity:.4f}")  # ~1.0

# Measure entanglement
source = EntanglementSource(fidelity=0.95)
dm = source.generate_density_matrix()
print(f"Concurrence: {concurrence(dm):.4f}")
```

## Part of the [Crowe Quantum Platform](https://github.com/MichaelCrowe11/crowe-quantum)
