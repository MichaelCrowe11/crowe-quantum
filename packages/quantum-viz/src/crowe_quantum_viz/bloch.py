"""Bloch sphere visualization for single-qubit states.

A single-qubit pure state |psi> = cos(theta/2)|0> + e^{i*phi}*sin(theta/2)|1>
maps to the point (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
on the unit Bloch sphere.

For mixed states (density matrices), the Bloch vector lies inside the sphere:
r = (Tr(rho*X), Tr(rho*Y), Tr(rho*Z)) with |r| <= 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from crowe_quantum_core.states import DensityMatrix, StateVector


@dataclass(frozen=True)
class BlochCoords:
    """Cartesian coordinates on the Bloch sphere."""

    x: float
    y: float
    z: float

    @property
    def theta(self) -> float:
        """Polar angle from +Z axis."""
        return float(np.arccos(np.clip(self.z, -1.0, 1.0)))

    @property
    def phi(self) -> float:
        """Azimuthal angle in XY plane."""
        return float(np.arctan2(self.y, self.x))

    @property
    def radius(self) -> float:
        """Distance from origin (1.0 = pure, <1.0 = mixed)."""
        return float(np.sqrt(self.x**2 + self.y**2 + self.z**2))


def bloch_coords(state: StateVector | DensityMatrix) -> BlochCoords:
    """Compute Bloch vector from a single-qubit state.

    For a StateVector, converts to density matrix first.
    The Bloch vector is r = (Tr(rho*X), Tr(rho*Y), Tr(rho*Z)).
    """
    from crowe_quantum_core.states import StateVector, DensityMatrix

    if isinstance(state, StateVector):
        if state.num_qubits != 1:
            raise ValueError(f"Bloch sphere requires 1 qubit, got {state.num_qubits}")
        rho = np.outer(state.data, state.data.conj())
    elif isinstance(state, DensityMatrix):
        if state.num_qubits != 1:
            raise ValueError(f"Bloch sphere requires 1 qubit, got {state.num_qubits}")
        rho = state.data
    else:
        raise TypeError(f"Expected StateVector or DensityMatrix, got {type(state)}")

    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    x = float(np.real(np.trace(rho @ sigma_x)))
    y = float(np.real(np.trace(rho @ sigma_y)))
    z = float(np.real(np.trace(rho @ sigma_z)))

    return BlochCoords(x=x, y=y, z=z)


class BlochSphere:
    """Builder for Bloch sphere plots.

    Accumulates states and vectors, then renders to a matplotlib figure.
    """

    def __init__(self) -> None:
        self._points: list[tuple[BlochCoords, str, str]] = []  # coords, color, label
        self._vectors: list[tuple[BlochCoords, str, str]] = []

    def add_state(
        self,
        state: StateVector | DensityMatrix,
        color: str = "blue",
        label: str = "",
    ) -> BlochSphere:
        """Add a quantum state as a point on the sphere."""
        coords = bloch_coords(state)
        self._points.append((coords, color, label))
        return self

    def add_vector(
        self,
        coords: BlochCoords,
        color: str = "red",
        label: str = "",
    ) -> BlochSphere:
        """Add an arbitrary Bloch vector."""
        self._vectors.append((coords, color, label))
        return self

    def render(self, title: str = "Bloch Sphere") -> dict:
        """Render the Bloch sphere to a data dict (matplotlib-free for testing).

        Returns a dict with all the data needed to draw the sphere:
        - wireframe: (x, y, z) arrays for the sphere surface
        - axes: the three axis lines
        - points: list of (x, y, z, color, label)
        - vectors: list of (x, y, z, color, label)
        """
        # Sphere wireframe
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        wx = np.outer(np.cos(u), np.sin(v))
        wy = np.outer(np.sin(u), np.sin(v))
        wz = np.outer(np.ones_like(u), np.cos(v))

        result: dict = {
            "title": title,
            "wireframe": {"x": wx, "y": wy, "z": wz},
            "axes": [
                {"start": [0, 0, -1.2], "end": [0, 0, 1.2], "label": "Z"},
                {"start": [-1.2, 0, 0], "end": [1.2, 0, 0], "label": "X"},
                {"start": [0, -1.2, 0], "end": [0, 1.2, 0], "label": "Y"},
            ],
            "points": [
                {"x": p.x, "y": p.y, "z": p.z, "color": c, "label": l}
                for p, c, l in self._points
            ],
            "vectors": [
                {"x": v.x, "y": v.y, "z": v.z, "color": c, "label": l}
                for v, c, l in self._vectors
            ],
        }
        return result

    def to_figure(self, title: str = "Bloch Sphere"):
        """Render to a matplotlib 3D figure.

        Returns the (fig, ax) tuple. Requires matplotlib.
        """
        import matplotlib.pyplot as plt

        data = self.render(title=title)
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Wireframe sphere
        wf = data["wireframe"]
        ax.plot_wireframe(wf["x"], wf["y"], wf["z"], alpha=0.08, color="gray")

        # Axes
        for axis in data["axes"]:
            s, e = axis["start"], axis["end"]
            ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], "k-", alpha=0.3)
            ax.text(e[0], e[1], e[2], axis["label"], fontsize=12)

        # Points
        for pt in data["points"]:
            ax.scatter(pt["x"], pt["y"], pt["z"], color=pt["color"], s=60, depthshade=True)
            if pt["label"]:
                ax.text(pt["x"], pt["y"], pt["z"], f"  {pt['label']}", fontsize=9)

        # Vectors
        for vec in data["vectors"]:
            ax.quiver(0, 0, 0, vec["x"], vec["y"], vec["z"],
                      color=vec["color"], arrow_length_ratio=0.1, linewidth=2)
            if vec["label"]:
                ax.text(vec["x"], vec["y"], vec["z"], f"  {vec['label']}", fontsize=9)

        ax.set_title(title)
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.set_zlim([-1.3, 1.3])
        ax.set_box_aspect([1, 1, 1])
        return fig, ax
