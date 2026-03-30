"""Universal quantum gate registry.

Every gate is a unitary morphism in the category of Hilbert spaces.
The registry provides matrix representations, decomposition rules,
commutation relations, and validation for all standard gates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


class GateFamily(Enum):
    """Classification of quantum gates."""

    PAULI = auto()       # X, Y, Z
    CLIFFORD = auto()    # H, S, CNOT, CZ, SWAP
    ROTATION = auto()    # RX, RY, RZ, U
    PHASE = auto()       # S, T, P
    CONTROLLED = auto()  # CNOT, CZ, TOFFOLI
    MULTI = auto()       # SWAP, FREDKIN


@dataclass(frozen=True)
class GateSpec:
    """Specification for a quantum gate."""

    name: str
    arity: int  # number of qubits
    num_params: int  # number of continuous parameters
    family: GateFamily
    aliases: tuple[str, ...] = ()
    description: str = ""


@dataclass
class Gate:
    """A concrete quantum gate with its unitary matrix.

    Gates are morphisms: they transform quantum states while preserving
    the inner product (unitarity). U†U = I.
    """

    spec: GateSpec
    _matrix_fn: Callable[..., NDArray[np.complex128]]
    params: tuple[float, ...] = ()

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def arity(self) -> int:
        return self.spec.arity

    def matrix(self, *params: float) -> NDArray[np.complex128]:
        """Get the unitary matrix for this gate with given parameters."""
        p = params if params else self.params
        return self._matrix_fn(*p)

    def adjoint(self, *params: float) -> NDArray[np.complex128]:
        """Get the conjugate transpose (dagger) of this gate."""
        return self.matrix(*params).conj().T

    def is_unitary(self, *params: float, atol: float = 1e-10) -> bool:
        """Verify U†U = I."""
        m = self.matrix(*params)
        product = m.conj().T @ m
        return np.allclose(product, np.eye(m.shape[0]), atol=atol)

    def __repr__(self) -> str:
        if self.params:
            param_str = ", ".join(f"{p:.4f}" for p in self.params)
            return f"Gate({self.name}({param_str}))"
        return f"Gate({self.name})"


# --- Gate matrix definitions ---

_I2 = np.eye(2, dtype=np.complex128)
_SQRT2_INV = 1.0 / np.sqrt(2)


def _identity_matrix() -> NDArray[np.complex128]:
    return _I2.copy()


def _x_matrix() -> NDArray[np.complex128]:
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)


def _y_matrix() -> NDArray[np.complex128]:
    return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


def _z_matrix() -> NDArray[np.complex128]:
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _h_matrix() -> NDArray[np.complex128]:
    return np.array(
        [[_SQRT2_INV, _SQRT2_INV], [_SQRT2_INV, -_SQRT2_INV]], dtype=np.complex128
    )


def _s_matrix() -> NDArray[np.complex128]:
    return np.array([[1, 0], [0, 1j]], dtype=np.complex128)


def _sdg_matrix() -> NDArray[np.complex128]:
    return np.array([[1, 0], [0, -1j]], dtype=np.complex128)


def _t_matrix() -> NDArray[np.complex128]:
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)


def _tdg_matrix() -> NDArray[np.complex128]:
    return np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)


def _rx_matrix(theta: float) -> NDArray[np.complex128]:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry_matrix(theta: float) -> NDArray[np.complex128]:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz_matrix(theta: float) -> NDArray[np.complex128]:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=np.complex128
    )


def _phase_matrix(phi: float) -> NDArray[np.complex128]:
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=np.complex128)


def _u_matrix(theta: float, phi: float, lam: float) -> NDArray[np.complex128]:
    """General single-qubit unitary U(θ, φ, λ)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=np.complex128,
    )


def _cnot_matrix() -> NDArray[np.complex128]:
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128
    )


def _cz_matrix() -> NDArray[np.complex128]:
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.complex128
    )


def _swap_matrix() -> NDArray[np.complex128]:
    return np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
    )


def _iswap_matrix() -> NDArray[np.complex128]:
    return np.array(
        [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
    )


def _toffoli_matrix() -> NDArray[np.complex128]:
    m = np.eye(8, dtype=np.complex128)
    m[6, 6], m[6, 7] = 0, 1
    m[7, 6], m[7, 7] = 1, 0
    return m


def _fredkin_matrix() -> NDArray[np.complex128]:
    m = np.eye(8, dtype=np.complex128)
    m[5, 5], m[5, 6] = 0, 1
    m[6, 5], m[6, 6] = 1, 0
    return m


# --- Gate specifications ---

GATE_SPECS: dict[str, GateSpec] = {
    "I": GateSpec("I", 1, 0, GateFamily.PAULI, description="Identity"),
    "X": GateSpec("X", 1, 0, GateFamily.PAULI, description="Pauli-X (NOT)"),
    "Y": GateSpec("Y", 1, 0, GateFamily.PAULI, description="Pauli-Y"),
    "Z": GateSpec("Z", 1, 0, GateFamily.PAULI, description="Pauli-Z"),
    "H": GateSpec("H", 1, 0, GateFamily.CLIFFORD, description="Hadamard"),
    "S": GateSpec("S", 1, 0, GateFamily.PHASE, description="Phase (π/2)"),
    "SDG": GateSpec("SDG", 1, 0, GateFamily.PHASE, aliases=("S†", "SDAGGER"),
                     description="S-dagger"),
    "T": GateSpec("T", 1, 0, GateFamily.PHASE, description="T (π/4)"),
    "TDG": GateSpec("TDG", 1, 0, GateFamily.PHASE, aliases=("T†", "TDAGGER"),
                     description="T-dagger"),
    "RX": GateSpec("RX", 1, 1, GateFamily.ROTATION, description="X-rotation"),
    "RY": GateSpec("RY", 1, 1, GateFamily.ROTATION, description="Y-rotation"),
    "RZ": GateSpec("RZ", 1, 1, GateFamily.ROTATION, description="Z-rotation"),
    "P": GateSpec("P", 1, 1, GateFamily.PHASE, aliases=("PHASE",), description="Phase gate"),
    "U": GateSpec("U", 1, 3, GateFamily.ROTATION, description="General single-qubit unitary"),
    "CNOT": GateSpec("CNOT", 2, 0, GateFamily.CONTROLLED, aliases=("CX",),
                      description="Controlled-NOT"),
    "CZ": GateSpec("CZ", 2, 0, GateFamily.CONTROLLED, description="Controlled-Z"),
    "SWAP": GateSpec("SWAP", 2, 0, GateFamily.MULTI, description="SWAP"),
    "ISWAP": GateSpec("ISWAP", 2, 0, GateFamily.MULTI, description="iSWAP"),
    "TOFFOLI": GateSpec("TOFFOLI", 3, 0, GateFamily.CONTROLLED, aliases=("CCX",),
                         description="Toffoli (CCX)"),
    "FREDKIN": GateSpec("FREDKIN", 3, 0, GateFamily.CONTROLLED, aliases=("CSWAP",),
                         description="Fredkin (CSWAP)"),
}

# Gate name -> matrix function mapping
_GATE_MATRICES: dict[str, Callable[..., NDArray[np.complex128]]] = {
    "I": _identity_matrix,
    "X": _x_matrix,
    "Y": _y_matrix,
    "Z": _z_matrix,
    "H": _h_matrix,
    "S": _s_matrix,
    "SDG": _sdg_matrix,
    "T": _t_matrix,
    "TDG": _tdg_matrix,
    "RX": _rx_matrix,
    "RY": _ry_matrix,
    "RZ": _rz_matrix,
    "P": _phase_matrix,
    "U": _u_matrix,
    "CNOT": _cnot_matrix,
    "CZ": _cz_matrix,
    "SWAP": _swap_matrix,
    "ISWAP": _iswap_matrix,
    "TOFFOLI": _toffoli_matrix,
    "FREDKIN": _fredkin_matrix,
}


class GateRegistry:
    """Registry for quantum gates with alias resolution and validation."""

    def __init__(self) -> None:
        self._specs: dict[str, GateSpec] = {}
        self._aliases: dict[str, str] = {}
        self._matrix_fns: dict[str, Callable[..., NDArray[np.complex128]]] = {}

    def register(
        self,
        spec: GateSpec,
        matrix_fn: Callable[..., NDArray[np.complex128]],
    ) -> None:
        """Register a gate with its specification and matrix function."""
        self._specs[spec.name] = spec
        self._matrix_fns[spec.name] = matrix_fn
        for alias in spec.aliases:
            self._aliases[alias.upper()] = spec.name

    def resolve(self, name: str) -> str:
        """Resolve a gate name or alias to the canonical name."""
        upper = name.upper()
        return self._aliases.get(upper, upper)

    def get_spec(self, name: str) -> GateSpec | None:
        """Get the specification for a gate."""
        return self._specs.get(self.resolve(name))

    def get_gate(self, name: str, *params: float) -> Gate:
        """Create a Gate instance for the given name and parameters."""
        canonical = self.resolve(name)
        spec = self._specs.get(canonical)
        if spec is None:
            from crowe_quantum_core.errors import QuantumError

            raise QuantumError(f"Unknown gate: '{name}'", code="E1001")
        if len(params) != spec.num_params:
            from crowe_quantum_core.errors import GateParameterError

            raise GateParameterError(
                f"Gate '{canonical}' expects {spec.num_params} parameter(s), got {len(params)}"
            )
        return Gate(spec=spec, _matrix_fn=self._matrix_fns[canonical], params=params)

    def list_gates(self) -> list[str]:
        """List all registered gate names."""
        return sorted(self._specs.keys())

    def __contains__(self, name: str) -> bool:
        return self.resolve(name) in self._specs


def _build_standard_registry() -> GateRegistry:
    """Build the standard gate registry with all built-in gates."""
    registry = GateRegistry()
    for name, spec in GATE_SPECS.items():
        matrix_fn = _GATE_MATRICES[name]
        registry.register(spec, matrix_fn)
    return registry


standard_gates: GateRegistry = _build_standard_registry()
