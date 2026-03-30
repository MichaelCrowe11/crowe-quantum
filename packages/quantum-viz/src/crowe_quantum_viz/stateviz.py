"""State vector and density matrix visualization utilities.

Functions for probability bar charts, phase disk plots, density matrix
heatmaps, and tabular state representations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from crowe_quantum_core.states import DensityMatrix, StateVector


def plot_probabilities(state: StateVector) -> dict:
    """Generate probability bar chart data from a state vector.

    Returns a dict with:
    - labels: list of basis state bitstrings
    - probabilities: list of measurement probabilities
    - colors: list of color values based on amplitude phase
    """
    probs = state.probabilities()
    n = state.num_qubits
    labels = [format(i, f"0{n}b") for i in range(state.dim)]
    phases = np.angle(state.data)

    # Map phases to a [0, 1] colormap range
    colors = [(p / (2 * np.pi) + 0.5) % 1.0 for p in phases]

    return {
        "labels": labels,
        "probabilities": probs.tolist(),
        "colors": colors,
        "ylabel": "Probability",
        "xlabel": "Basis State",
    }


def plot_phase_disk(state: StateVector) -> dict:
    """Generate phase disk data for a state vector.

    Each basis state is plotted as a point on a unit disk:
    - Angle = complex phase of the amplitude
    - Radius = |amplitude| (proportional to sqrt of probability)

    Returns a dict with lists of angles, radii, and labels.
    """
    amplitudes = state.data
    n = state.num_qubits
    labels = [format(i, f"0{n}b") for i in range(state.dim)]

    angles = np.angle(amplitudes).tolist()
    radii = np.abs(amplitudes).tolist()

    return {
        "labels": labels,
        "angles": angles,
        "radii": radii,
        "title": "Phase Disk",
    }


def plot_density_heatmap(dm: DensityMatrix) -> dict:
    """Generate density matrix heatmap data.

    Returns real and imaginary parts as 2D arrays, plus axis labels.
    """
    n = dm.num_qubits
    dim = dm.dim
    labels = [format(i, f"0{n}b") for i in range(dim)]

    return {
        "real": np.real(dm.data).tolist(),
        "imag": np.imag(dm.data).tolist(),
        "labels": labels,
        "title": f"Density Matrix ({n} qubit{'s' if n > 1 else ''})",
    }


def state_table(state: StateVector, threshold: float = 1e-10) -> list[dict]:
    """Generate a table of non-negligible amplitudes.

    Returns list of dicts with keys: label, amplitude, probability, phase_deg.
    """
    n = state.num_qubits
    rows = []
    for i in range(state.dim):
        amp = state.data[i]
        prob = abs(amp) ** 2
        if prob < threshold:
            continue
        rows.append({
            "label": format(i, f"0{n}b"),
            "amplitude": complex(amp),
            "probability": float(prob),
            "phase_deg": float(np.degrees(np.angle(amp))),
        })
    return rows


def to_figure_probabilities(state: StateVector, title: str = "State Probabilities"):
    """Render probability bar chart to matplotlib figure."""
    import matplotlib.pyplot as plt

    data = plot_probabilities(state)
    fig, ax = plt.subplots(figsize=(max(6, len(data["labels"]) * 0.5), 4))
    bars = ax.bar(data["labels"], data["probabilities"])

    # Color bars by phase
    cmap = plt.cm.hsv  # type: ignore[attr-defined]
    for bar, c in zip(bars, data["colors"]):
        bar.set_facecolor(cmap(c))

    ax.set_ylabel(data["ylabel"])
    ax.set_xlabel(data["xlabel"])
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45 if len(data["labels"]) > 8 else 0)
    plt.tight_layout()
    return fig, ax
