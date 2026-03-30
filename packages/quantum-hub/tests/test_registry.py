"""Tests for the backend registry."""

import pytest
from crowe_quantum_hub.local_sim import LocalSimulator
from crowe_quantum_hub.registry import BackendRegistry, registry


class TestBackendRegistry:
    def test_default_registry_has_local_sim(self):
        """The default registry comes pre-loaded with a local simulator."""
        assert "crowe-local-simulator" in registry
        backend = registry.get("crowe-local-simulator")
        assert isinstance(backend, LocalSimulator)

    def test_list_backends(self):
        """Can list all registered backends."""
        backends = registry.list_backends()
        assert len(backends) >= 1
        names = [b.name for b in backends]
        assert "crowe-local-simulator" in names

    def test_register_and_get(self):
        """Can register and retrieve a custom backend."""
        reg = BackendRegistry()
        sim = LocalSimulator(max_qubits=5)
        reg.register(sim, description="test sim", tags=["test"])
        assert "crowe-local-simulator" in reg
        assert reg.get("crowe-local-simulator") is sim

    def test_get_missing_raises(self):
        """Accessing a missing backend raises KeyError."""
        reg = BackendRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent")

    def test_find_by_qubits(self):
        """Can filter backends by minimum qubit count."""
        reg = BackendRegistry()
        reg.register(LocalSimulator(max_qubits=5), tags=["small"])
        reg.register(LocalSimulator(max_qubits=20), tags=["large"])
        results = reg.find(min_qubits=10)
        # Only the 20-qubit simulator, but since both have same name, registry keeps latest
        assert len(results) == 1
        assert results[0].max_qubits == 20

    def test_find_by_tags(self):
        """Can filter backends by tags."""
        results = registry.find(tags=["simulator"])
        assert len(results) >= 1

    def test_find_simulator_only(self):
        """simulator_only filter works."""
        results = registry.find(simulator_only=True)
        assert all(b.is_simulator for b in results)

    def test_len(self):
        """Registry reports correct length."""
        reg = BackendRegistry()
        assert len(reg) == 0
        reg.register(LocalSimulator())
        assert len(reg) == 1
