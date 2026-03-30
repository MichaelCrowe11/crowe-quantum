"""Quantum noise models.

Physical noise channels represented as Kraus operators:
ρ → Σᵢ Kᵢ ρ Kᵢ† where Σᵢ Kᵢ†Kᵢ = I (trace-preserving).

Models: depolarizing, amplitude damping, phase damping,
thermal relaxation, and readout error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


class NoiseType(Enum):
    DEPOLARIZING = auto()
    AMPLITUDE_DAMPING = auto()
    PHASE_DAMPING = auto()
    THERMAL_RELAXATION = auto()
    READOUT = auto()
    CUSTOM = auto()


@dataclass
class NoiseChannel:
    """A quantum noise channel defined by Kraus operators."""

    noise_type: NoiseType
    kraus_operators: list[NDArray[np.complex128]]
    description: str = ""

    def apply(self, density_matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Apply noise channel: ρ → Σᵢ Kᵢ ρ Kᵢ†."""
        result = np.zeros_like(density_matrix)
        for k in self.kraus_operators:
            result += k @ density_matrix @ k.conj().T
        return result

    def is_trace_preserving(self, atol: float = 1e-10) -> bool:
        """Verify Σᵢ Kᵢ†Kᵢ = I."""
        dim = self.kraus_operators[0].shape[0]
        total = np.zeros((dim, dim), dtype=np.complex128)
        for k in self.kraus_operators:
            total += k.conj().T @ k
        return np.allclose(total, np.eye(dim), atol=atol)


def depolarizing_channel(p: float) -> NoiseChannel:
    """Single-qubit depolarizing channel.

    With probability p, replaces state with I/2 (maximally mixed).
    ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Depolarizing probability must be in [0, 1], got {p}")

    sqrt_1mp = np.sqrt(1 - p)
    sqrt_p3 = np.sqrt(p / 3)

    k0 = sqrt_1mp * np.eye(2, dtype=np.complex128)
    k1 = sqrt_p3 * np.array([[0, 1], [1, 0]], dtype=np.complex128)      # X
    k2 = sqrt_p3 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)   # Y
    k3 = sqrt_p3 * np.array([[1, 0], [0, -1]], dtype=np.complex128)     # Z

    return NoiseChannel(
        noise_type=NoiseType.DEPOLARIZING,
        kraus_operators=[k0, k1, k2, k3],
        description=f"Depolarizing(p={p})",
    )


def amplitude_damping_channel(gamma: float) -> NoiseChannel:
    """Amplitude damping: models energy dissipation (T1 decay).

    |1⟩ decays to |0⟩ with probability γ.
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"Damping rate must be in [0, 1], got {gamma}")

    k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    k1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)

    return NoiseChannel(
        noise_type=NoiseType.AMPLITUDE_DAMPING,
        kraus_operators=[k0, k1],
        description=f"AmplitudeDamping(γ={gamma})",
    )


def phase_damping_channel(gamma: float) -> NoiseChannel:
    """Phase damping: models dephasing (T2 decay) without energy loss.

    Off-diagonal elements decay at rate γ.
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"Phase damping rate must be in [0, 1], got {gamma}")

    k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    k1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=np.complex128)

    return NoiseChannel(
        noise_type=NoiseType.PHASE_DAMPING,
        kraus_operators=[k0, k1],
        description=f"PhaseDamping(γ={gamma})",
    )


def thermal_relaxation_channel(t1: float, t2: float, time: float) -> NoiseChannel:
    """Thermal relaxation: combined T1 (energy) and T2 (phase) decay.

    Models realistic qubit decoherence over a given time duration.
    T2 ≤ 2*T1 is required by physics.
    """
    if t2 > 2 * t1:
        raise ValueError(f"T2 ({t2}) cannot exceed 2*T1 ({2*t1})")
    if t1 <= 0 or t2 <= 0 or time < 0:
        raise ValueError("T1, T2 must be positive; time must be non-negative")

    p_reset = 1 - np.exp(-time / t1)
    p_phase = 1 - np.exp(-time / t2) if t2 < np.inf else 0

    # Combine amplitude damping and additional dephasing
    ad = amplitude_damping_channel(p_reset)
    if p_phase > p_reset:
        # Additional dephasing beyond what amplitude damping provides
        extra_dephase = (p_phase - p_reset) / (1 - p_reset) if p_reset < 1 else 0
        pd = phase_damping_channel(extra_dephase)
        # Compose channels
        return _compose_channels(ad, pd, f"ThermalRelaxation(T1={t1}, T2={t2}, t={time})")
    return NoiseChannel(
        noise_type=NoiseType.THERMAL_RELAXATION,
        kraus_operators=ad.kraus_operators,
        description=f"ThermalRelaxation(T1={t1}, T2={t2}, t={time})",
    )


def readout_error_channel(p0_given1: float, p1_given0: float) -> NoiseChannel:
    """Readout error: classical bit-flip during measurement.

    p0_given1: probability of reading 0 when state is |1⟩
    p1_given0: probability of reading 1 when state is |0⟩
    """
    if not (0 <= p0_given1 <= 1 and 0 <= p1_given0 <= 1):
        raise ValueError("Readout error probabilities must be in [0, 1]")

    return NoiseChannel(
        noise_type=NoiseType.READOUT,
        kraus_operators=[],  # Readout errors are classical, not Kraus operators
        description=f"ReadoutError(p0|1={p0_given1}, p1|0={p1_given0})",
    )


def _compose_channels(ch1: NoiseChannel, ch2: NoiseChannel, description: str) -> NoiseChannel:
    """Compose two noise channels: (ch2 ∘ ch1)(ρ) = ch2(ch1(ρ))."""
    composed_kraus = []
    for k2 in ch2.kraus_operators:
        for k1 in ch1.kraus_operators:
            composed_kraus.append(k2 @ k1)
    return NoiseChannel(
        noise_type=NoiseType.CUSTOM,
        kraus_operators=composed_kraus,
        description=description,
    )


@dataclass
class NoiseModel:
    """A noise model specifying channels for different gate operations."""

    gate_noise: dict[str, NoiseChannel] = field(default_factory=dict)
    readout_error: NoiseChannel | None = None
    idle_noise: NoiseChannel | None = None

    def add_gate_noise(self, gate_name: str, channel: NoiseChannel) -> None:
        self.gate_noise[gate_name.upper()] = channel

    def add_all_gate_noise(self, channel: NoiseChannel) -> None:
        """Apply the same noise to all gates."""
        self.gate_noise["__ALL__"] = channel

    def get_noise_for_gate(self, gate_name: str) -> NoiseChannel | None:
        return self.gate_noise.get(gate_name.upper(), self.gate_noise.get("__ALL__"))
