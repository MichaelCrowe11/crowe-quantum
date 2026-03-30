"""Tests for Bloch sphere visualization."""

import math

import numpy as np
import pytest

from crowe_quantum_core.states import StateVector, DensityMatrix
from crowe_quantum_viz.bloch import BlochCoords, BlochSphere, bloch_coords


class TestBlochCoords:
    def test_north_pole(self):
        """|0> maps to (0, 0, 1) — the north pole."""
        sv = StateVector(1)  # |0>
        coords = bloch_coords(sv)
        assert coords.x == pytest.approx(0, abs=1e-10)
        assert coords.y == pytest.approx(0, abs=1e-10)
        assert coords.z == pytest.approx(1, abs=1e-10)

    def test_south_pole(self):
        """|1> maps to (0, 0, -1) — the south pole."""
        sv = StateVector.from_label("1")
        coords = bloch_coords(sv)
        assert coords.x == pytest.approx(0, abs=1e-10)
        assert coords.y == pytest.approx(0, abs=1e-10)
        assert coords.z == pytest.approx(-1, abs=1e-10)

    def test_plus_state(self):
        """|+> = (|0>+|1>)/sqrt(2) maps to (1, 0, 0) — +X axis."""
        sv = StateVector(1, np.array([1, 1], dtype=np.complex128) / np.sqrt(2))
        coords = bloch_coords(sv)
        assert coords.x == pytest.approx(1, abs=1e-10)
        assert coords.y == pytest.approx(0, abs=1e-10)
        assert coords.z == pytest.approx(0, abs=1e-10)

    def test_minus_state(self):
        """|-> = (|0>-|1>)/sqrt(2) maps to (-1, 0, 0) — -X axis."""
        sv = StateVector(1, np.array([1, -1], dtype=np.complex128) / np.sqrt(2))
        coords = bloch_coords(sv)
        assert coords.x == pytest.approx(-1, abs=1e-10)
        assert coords.z == pytest.approx(0, abs=1e-10)

    def test_i_plus_state(self):
        """|i+> = (|0>+i|1>)/sqrt(2) maps to (0, 1, 0) — +Y axis."""
        sv = StateVector(1, np.array([1, 1j], dtype=np.complex128) / np.sqrt(2))
        coords = bloch_coords(sv)
        assert coords.x == pytest.approx(0, abs=1e-10)
        assert coords.y == pytest.approx(1, abs=1e-10)
        assert coords.z == pytest.approx(0, abs=1e-10)

    def test_maximally_mixed(self):
        """Maximally mixed state maps to origin (0, 0, 0)."""
        dm = DensityMatrix.maximally_mixed(1)
        coords = bloch_coords(dm)
        assert coords.radius == pytest.approx(0, abs=1e-10)

    def test_pure_state_radius(self):
        """Pure states have radius 1."""
        sv = StateVector(1)
        coords = bloch_coords(sv)
        assert coords.radius == pytest.approx(1.0, abs=1e-10)

    def test_theta_phi(self):
        """Verify theta and phi properties."""
        coords = BlochCoords(x=0, y=0, z=1)
        assert coords.theta == pytest.approx(0, abs=1e-10)
        coords = BlochCoords(x=0, y=0, z=-1)
        assert coords.theta == pytest.approx(math.pi, abs=1e-10)

    def test_rejects_multi_qubit(self):
        """Bloch sphere only works for single-qubit states."""
        sv = StateVector(2)
        with pytest.raises(ValueError, match="1 qubit"):
            bloch_coords(sv)

    def test_density_matrix_input(self):
        """|0><0| density matrix gives north pole."""
        dm = DensityMatrix(1)  # |0><0|
        coords = bloch_coords(dm)
        assert coords.z == pytest.approx(1, abs=1e-10)


class TestBlochSphere:
    def test_render_structure(self):
        """Render returns expected dict structure."""
        sphere = BlochSphere()
        sv = StateVector(1)
        sphere.add_state(sv, label="|0>")
        data = sphere.render()
        assert "wireframe" in data
        assert "axes" in data
        assert "points" in data
        assert len(data["points"]) == 1
        assert data["points"][0]["label"] == "|0>"

    def test_add_vector(self):
        """Can add arbitrary Bloch vectors."""
        sphere = BlochSphere()
        coords = BlochCoords(0.5, 0.5, 0.5)
        sphere.add_vector(coords, color="green", label="test")
        data = sphere.render()
        assert len(data["vectors"]) == 1
        assert data["vectors"][0]["color"] == "green"

    def test_multiple_states(self):
        """Can add multiple states."""
        sphere = BlochSphere()
        sphere.add_state(StateVector(1), label="|0>")
        sphere.add_state(StateVector.from_label("1"), label="|1>")
        data = sphere.render()
        assert len(data["points"]) == 2

    def test_fluent_api(self):
        """add_state returns self for chaining."""
        sphere = BlochSphere()
        result = sphere.add_state(StateVector(1))
        assert result is sphere
