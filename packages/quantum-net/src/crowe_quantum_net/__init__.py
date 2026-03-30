"""Crowe Quantum Net — quantum networking protocols.

Implements quantum channels, entanglement distribution, and
teleportation protocols over noisy networks.
"""

__version__ = "1.0.0"

from crowe_quantum_net.channel import (
    QuantumChannel,
    amplitude_damping_channel,
    depolarizing_channel,
)
from crowe_quantum_net.entanglement import (
    EntanglementSource,
    EntanglementSwap,
    concurrence,
    entanglement_fidelity,
)
from crowe_quantum_net.teleportation import (
    TeleportationProtocol,
    TeleportationResult,
    teleport,
)

__all__ = [
    "EntanglementSource",
    "EntanglementSwap",
    "QuantumChannel",
    "TeleportationProtocol",
    "TeleportationResult",
    "amplitude_damping_channel",
    "concurrence",
    "depolarizing_channel",
    "entanglement_fidelity",
    "teleport",
]
