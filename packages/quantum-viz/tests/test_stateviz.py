"""Tests for state visualization utilities."""

import numpy as np
import pytest
from crowe_quantum_core.states import DensityMatrix, StateVector
from crowe_quantum_viz.stateviz import (
    plot_density_heatmap,
    plot_phase_disk,
    plot_probabilities,
    state_table,
)


class TestPlotProbabilities:
    def test_basis_state(self):
        """|0> has probability 1 at index 0."""
        sv = StateVector(1)
        data = plot_probabilities(sv)
        assert data["labels"] == ["0", "1"]
        assert data["probabilities"][0] == pytest.approx(1.0)
        assert data["probabilities"][1] == pytest.approx(0.0)

    def test_superposition(self):
        """|+> has equal probabilities."""
        sv = StateVector(1, np.array([1, 1], dtype=np.complex128) / np.sqrt(2))
        data = plot_probabilities(sv)
        assert data["probabilities"][0] == pytest.approx(0.5)
        assert data["probabilities"][1] == pytest.approx(0.5)

    def test_two_qubit_labels(self):
        """2-qubit state has 4 labels: 00, 01, 10, 11."""
        sv = StateVector(2)
        data = plot_probabilities(sv)
        assert data["labels"] == ["00", "01", "10", "11"]

    def test_colors_are_floats(self):
        """Colors should be in [0, 1] range for colormap."""
        sv = StateVector(1)
        data = plot_probabilities(sv)
        assert all(0 <= c <= 1 for c in data["colors"])


class TestPhaseDisk:
    def test_basis_state_angles(self):
        """|0> has angle 0 and radius 1 for the first component."""
        sv = StateVector(1)
        data = plot_phase_disk(sv)
        assert data["radii"][0] == pytest.approx(1.0)
        assert data["radii"][1] == pytest.approx(0.0)

    def test_superposition_radii(self):
        """|+> has equal radii."""
        sv = StateVector(1, np.array([1, 1], dtype=np.complex128) / np.sqrt(2))
        data = plot_phase_disk(sv)
        assert data["radii"][0] == pytest.approx(1 / np.sqrt(2))
        assert data["radii"][1] == pytest.approx(1 / np.sqrt(2))

    def test_phase_difference(self):
        """i|1> component should have phase pi/2."""
        sv = StateVector(1, np.array([1, 1j], dtype=np.complex128) / np.sqrt(2))
        data = plot_phase_disk(sv)
        assert data["angles"][0] == pytest.approx(0, abs=1e-10)
        assert data["angles"][1] == pytest.approx(np.pi / 2, abs=1e-10)


class TestDensityHeatmap:
    def test_pure_state(self):
        """|0><0| has 1 at (0,0) and 0 elsewhere."""
        dm = DensityMatrix(1)
        data = plot_density_heatmap(dm)
        assert data["real"][0][0] == pytest.approx(1.0)
        assert data["real"][1][1] == pytest.approx(0.0)
        assert data["labels"] == ["0", "1"]

    def test_maximally_mixed(self):
        """Maximally mixed state has 0.5 on diagonal."""
        dm = DensityMatrix.maximally_mixed(1)
        data = plot_density_heatmap(dm)
        assert data["real"][0][0] == pytest.approx(0.5)
        assert data["real"][1][1] == pytest.approx(0.5)

    def test_two_qubit_labels(self):
        dm = DensityMatrix(2)
        data = plot_density_heatmap(dm)
        assert len(data["labels"]) == 4


class TestStateTable:
    def test_basis_state(self):
        """|0> produces one row."""
        sv = StateVector(1)
        rows = state_table(sv)
        assert len(rows) == 1
        assert rows[0]["label"] == "0"
        assert rows[0]["probability"] == pytest.approx(1.0)

    def test_superposition(self):
        """|+> produces two rows with equal probability."""
        sv = StateVector(1, np.array([1, 1], dtype=np.complex128) / np.sqrt(2))
        rows = state_table(sv)
        assert len(rows) == 2
        assert rows[0]["probability"] == pytest.approx(0.5)
        assert rows[1]["probability"] == pytest.approx(0.5)

    def test_threshold_filters(self):
        """Amplitudes below threshold are excluded."""
        sv = StateVector(1, np.array([1, 1e-15], dtype=np.complex128))
        sv.normalize()
        rows = state_table(sv, threshold=1e-10)
        assert len(rows) == 1

    def test_phase_degrees(self):
        """Phase is reported in degrees."""
        sv = StateVector(1, np.array([0, 1j], dtype=np.complex128))
        rows = state_table(sv)
        assert rows[0]["phase_deg"] == pytest.approx(90.0)
