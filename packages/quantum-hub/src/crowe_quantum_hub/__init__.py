"""Crowe Quantum Hub — backend connector and local simulator.

Provides a backend registry for discovering and selecting quantum execution
targets, plus a high-fidelity local state-vector simulator.
"""

__version__ = "1.0.0"

from crowe_quantum_hub.local_sim import LocalSimulator
from crowe_quantum_hub.registry import BackendRegistry, registry

__all__ = [
    "BackendRegistry",
    "LocalSimulator",
    "registry",
]
