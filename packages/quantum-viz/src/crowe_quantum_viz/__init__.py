"""Crowe Quantum Viz — quantum state visualization.

Bloch spheres, probability plots, circuit diagrams, and density matrix
heatmaps. All renderers produce matplotlib figures for maximum flexibility.
"""

__version__ = "1.0.0"

from crowe_quantum_viz.bloch import BlochSphere, bloch_coords
from crowe_quantum_viz.stateviz import (
    plot_probabilities,
    plot_phase_disk,
    plot_density_heatmap,
    state_table,
)
from crowe_quantum_viz.circuit_draw import CircuitDrawer, draw_circuit

__all__ = [
    "BlochSphere",
    "CircuitDrawer",
    "bloch_coords",
    "draw_circuit",
    "plot_density_heatmap",
    "plot_phase_disk",
    "plot_probabilities",
    "state_table",
]
