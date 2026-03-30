"""Uncertainty propagation for scientific measurements.

Implements Gaussian error propagation where every value carries its
uncertainty: x = μ ± σ. Arithmetic operations automatically propagate
uncertainties using the standard rules for independent variables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class UncertainValue:
    """A value with associated uncertainty: μ ± σ.

    Supports arithmetic with automatic error propagation assuming
    independent, normally-distributed uncertainties.
    """

    value: float
    uncertainty: float = 0.0

    def __post_init__(self) -> None:
        if self.uncertainty < 0:
            object.__setattr__(self, "uncertainty", abs(self.uncertainty))

    # ── Arithmetic with error propagation ────────────────────────────────

    def __add__(self, other: UncertainValue | float) -> UncertainValue:
        other = _ensure_uncertain(other)
        return UncertainValue(
            self.value + other.value,
            math.sqrt(self.uncertainty**2 + other.uncertainty**2),
        )

    def __radd__(self, other: float) -> UncertainValue:
        return self.__add__(other)

    def __sub__(self, other: UncertainValue | float) -> UncertainValue:
        other = _ensure_uncertain(other)
        return UncertainValue(
            self.value - other.value,
            math.sqrt(self.uncertainty**2 + other.uncertainty**2),
        )

    def __rsub__(self, other: float) -> UncertainValue:
        other = _ensure_uncertain(other)
        return other.__sub__(self)

    def __mul__(self, other: UncertainValue | float) -> UncertainValue:
        other = _ensure_uncertain(other)
        val = self.value * other.value
        # Relative uncertainty: σ_f/f = √((σ_a/a)² + (σ_b/b)²)
        rel_a = self.uncertainty / self.value if self.value != 0 else 0
        rel_b = other.uncertainty / other.value if other.value != 0 else 0
        unc = abs(val) * math.sqrt(rel_a**2 + rel_b**2)
        return UncertainValue(val, unc)

    def __rmul__(self, other: float) -> UncertainValue:
        return self.__mul__(other)

    def __truediv__(self, other: UncertainValue | float) -> UncertainValue:
        other = _ensure_uncertain(other)
        if other.value == 0:
            raise ZeroDivisionError("Division by zero in uncertain value")
        val = self.value / other.value
        rel_a = self.uncertainty / self.value if self.value != 0 else 0
        rel_b = other.uncertainty / other.value if other.value != 0 else 0
        unc = abs(val) * math.sqrt(rel_a**2 + rel_b**2)
        return UncertainValue(val, unc)

    def __rtruediv__(self, other: float) -> UncertainValue:
        other = _ensure_uncertain(other)
        return other.__truediv__(self)

    def __pow__(self, exponent: float) -> UncertainValue:
        val = self.value**exponent
        unc = abs(exponent * self.value ** (exponent - 1)) * self.uncertainty
        return UncertainValue(val, unc)

    def __neg__(self) -> UncertainValue:
        return UncertainValue(-self.value, self.uncertainty)

    def __abs__(self) -> UncertainValue:
        return UncertainValue(abs(self.value), self.uncertainty)

    # ── Comparison ───────────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        if isinstance(other, UncertainValue):
            return self.value == other.value and self.uncertainty == other.uncertainty
        if isinstance(other, (int, float)):
            return self.value == other and self.uncertainty == 0
        return NotImplemented

    def overlaps(self, other: UncertainValue, sigma: float = 1.0) -> bool:
        """Check if two uncertain values overlap within sigma standard deviations."""
        diff = abs(self.value - other.value)
        combined_unc = math.sqrt(self.uncertainty**2 + other.uncertainty**2)
        return diff <= sigma * combined_unc

    # ── Scientific functions with uncertainty propagation ─────────────────

    def sqrt(self) -> UncertainValue:
        return self**0.5

    def sin(self) -> UncertainValue:
        val = math.sin(self.value)
        unc = abs(math.cos(self.value)) * self.uncertainty
        return UncertainValue(val, unc)

    def cos(self) -> UncertainValue:
        val = math.cos(self.value)
        unc = abs(math.sin(self.value)) * self.uncertainty
        return UncertainValue(val, unc)

    def exp(self) -> UncertainValue:
        val = math.exp(self.value)
        unc = val * self.uncertainty
        return UncertainValue(val, unc)

    def log(self) -> UncertainValue:
        if self.value <= 0:
            raise ValueError("Cannot take log of non-positive uncertain value")
        val = math.log(self.value)
        unc = self.uncertainty / abs(self.value)
        return UncertainValue(val, unc)

    # ── Display ──────────────────────────────────────────────────────────

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty as a fraction."""
        if self.value == 0:
            return float("inf") if self.uncertainty > 0 else 0.0
        return abs(self.uncertainty / self.value)

    @property
    def significant_figures(self) -> int:
        """Estimate significant figures from uncertainty."""
        if self.uncertainty == 0:
            return 15  # float precision
        return max(0, -int(math.floor(math.log10(self.uncertainty))))

    def __repr__(self) -> str:
        if self.uncertainty == 0:
            return f"UncertainValue({self.value})"
        return f"UncertainValue({self.value} ± {self.uncertainty})"

    def __str__(self) -> str:
        if self.uncertainty == 0:
            return str(self.value)
        sf = self.significant_figures
        return f"{self.value:.{sf}f} ± {self.uncertainty:.{sf}f}"


def _ensure_uncertain(x: UncertainValue | float) -> UncertainValue:
    if isinstance(x, UncertainValue):
        return x
    return UncertainValue(float(x))
