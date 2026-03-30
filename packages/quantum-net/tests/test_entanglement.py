"""Tests for entanglement distribution and quantification."""

import numpy as np
import pytest

from crowe_quantum_core.states import StateVector, DensityMatrix
from crowe_quantum_net.entanglement import (
    EntanglementSource,
    EntanglementSwap,
    entanglement_fidelity,
    concurrence,
)


class TestEntanglementSource:
    def test_perfect_source(self):
        """Perfect source generates |Phi+> as StateVector."""
        source = EntanglementSource(fidelity=1.0)
        pair = source.generate_pair()
        assert isinstance(pair, StateVector)
        bell = StateVector.bell_state(0)
        assert pair.fidelity(bell) == pytest.approx(1.0, abs=1e-10)

    def test_imperfect_source(self):
        """Imperfect source generates Werner state as DensityMatrix."""
        source = EntanglementSource(fidelity=0.8)
        pair = source.generate_pair()
        assert isinstance(pair, DensityMatrix)
        assert pair.trace() == pytest.approx(1.0, abs=1e-10)

    def test_imperfect_fidelity(self):
        """Werner state has correct fidelity with target Bell state."""
        f = 0.9
        source = EntanglementSource(fidelity=f)
        pair = source.generate_pair()
        measured_f = entanglement_fidelity(pair, target=0)
        assert measured_f == pytest.approx(f, abs=1e-10)

    def test_invalid_fidelity(self):
        with pytest.raises(ValueError):
            EntanglementSource(fidelity=1.5)
        with pytest.raises(ValueError):
            EntanglementSource(fidelity=-0.1)


class TestEntanglementSwap:
    def test_perfect_swap(self):
        """Swapping two |Phi+> pairs produces entanglement between outer qubits."""
        pair_ab = StateVector.bell_state(0)
        pair_bc = StateVector.bell_state(0)
        result = EntanglementSwap.swap(pair_ab, pair_bc)
        assert result.num_qubits == 2
        # Result should be close to a Bell state
        bell_fidelity = max(
            result.fidelity(StateVector.bell_state(i)) for i in range(4)
        )
        assert bell_fidelity > 0.9

    def test_swap_rejects_wrong_size(self):
        """Swap requires 2-qubit pairs."""
        sv1 = StateVector(1)
        sv2 = StateVector.bell_state(0)
        with pytest.raises(ValueError, match="2-qubit"):
            EntanglementSwap.swap(sv1, sv2)


class TestEntanglementFidelity:
    def test_perfect_bell_state(self):
        """Bell state has fidelity 1 with itself."""
        bell = StateVector.bell_state(0)
        assert entanglement_fidelity(bell, target=0) == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal_bell_states(self):
        """Orthogonal Bell states have fidelity ~0."""
        bell_plus = StateVector.bell_state(0)
        assert entanglement_fidelity(bell_plus, target=1) == pytest.approx(0.0, abs=1e-10)

    def test_density_matrix_input(self):
        """Works with density matrix input."""
        bell = StateVector.bell_state(0)
        dm = bell.to_density_matrix()
        assert entanglement_fidelity(dm, target=0) == pytest.approx(1.0, abs=1e-10)


class TestConcurrence:
    def test_bell_state_maximally_entangled(self):
        """Bell states have concurrence = 1."""
        bell = StateVector.bell_state(0).to_density_matrix()
        assert concurrence(bell) == pytest.approx(1.0, abs=1e-6)

    def test_product_state_separable(self):
        """Product state |00> has concurrence = 0."""
        sv = StateVector.from_label("00")
        dm = sv.to_density_matrix()
        assert concurrence(dm) == pytest.approx(0.0, abs=1e-6)

    def test_rejects_non_two_qubit(self):
        """Concurrence only defined for 2-qubit states."""
        dm = DensityMatrix(1)
        with pytest.raises(ValueError, match="2 qubits"):
            concurrence(dm)

    def test_werner_state_concurrence(self):
        """Werner state with F=0.5 has concurrence = 0."""
        source = EntanglementSource(fidelity=0.5)
        pair = source.generate_pair()
        c = concurrence(pair)
        # Werner state is separable at F <= 0.5
        assert c == pytest.approx(0.0, abs=0.05)
