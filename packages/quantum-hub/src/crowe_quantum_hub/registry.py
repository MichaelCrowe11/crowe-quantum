"""Backend registry — discover, register, and select quantum backends.

The registry is the central hub through which all backend access flows.
Backends are registered by name and can be queried by capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crowe_quantum_core.protocols import Backend


@dataclass
class BackendInfo:
    """Metadata about a registered backend."""

    name: str
    backend: Backend
    max_qubits: int
    is_simulator: bool = True
    description: str = ""
    tags: list[str] = field(default_factory=list)


class BackendRegistry:
    """Registry for quantum backends with discovery and selection."""

    def __init__(self) -> None:
        self._backends: dict[str, BackendInfo] = {}

    def register(
        self,
        backend: Backend,
        is_simulator: bool = True,
        description: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """Register a backend with the hub."""
        info = BackendInfo(
            name=backend.name,
            backend=backend,
            max_qubits=backend.max_qubits,
            is_simulator=is_simulator,
            description=description,
            tags=tags or [],
        )
        self._backends[backend.name] = info

    def get(self, name: str) -> Backend:
        """Get a backend by name."""
        info = self._backends.get(name)
        if info is None:
            available = ", ".join(self._backends.keys()) or "(none)"
            raise KeyError(f"Backend '{name}' not found. Available: {available}")
        return info.backend

    def list_backends(self) -> list[BackendInfo]:
        """List all registered backends."""
        return list(self._backends.values())

    def find(
        self,
        min_qubits: int = 0,
        simulator_only: bool = False,
        tags: list[str] | None = None,
    ) -> list[BackendInfo]:
        """Find backends matching criteria."""
        results = []
        for info in self._backends.values():
            if info.max_qubits < min_qubits:
                continue
            if simulator_only and not info.is_simulator:
                continue
            if tags and not all(t in info.tags for t in tags):
                continue
            results.append(info)
        return results

    def __contains__(self, name: str) -> bool:
        return name in self._backends

    def __len__(self) -> int:
        return len(self._backends)


def _build_default_registry() -> BackendRegistry:
    """Create the default registry with the local simulator pre-registered."""
    from crowe_quantum_hub.local_sim import LocalSimulator

    reg = BackendRegistry()
    reg.register(
        LocalSimulator(),
        is_simulator=True,
        description="Local state-vector simulator (up to 20 qubits)",
        tags=["simulator", "statevector", "local"],
    )
    return reg


registry: BackendRegistry = _build_default_registry()
