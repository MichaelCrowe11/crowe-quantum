"""Physical unit system for scientific quantities.

Tracks dimensions through calculations to catch unit errors at runtime.
Supports SI base units with derived unit composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Unit:
    """A physical unit as a product of base dimensions.

    Dimensions stored as {dimension_name: exponent} mapping.
    e.g., velocity = {"m": 1, "s": -1}
    """

    name: str
    dimensions: dict[str, int] = field(default_factory=dict)
    scale: float = 1.0  # Scale relative to SI base

    def __mul__(self, other: Unit) -> Unit:
        dims = dict(self.dimensions)
        for dim, exp in other.dimensions.items():
            dims[dim] = dims.get(dim, 0) + exp
        # Remove zero exponents
        dims = {k: v for k, v in dims.items() if v != 0}
        name = f"{self.name}·{other.name}"
        return Unit(name=name, dimensions=dims, scale=self.scale * other.scale)

    def __truediv__(self, other: Unit) -> Unit:
        dims = dict(self.dimensions)
        for dim, exp in other.dimensions.items():
            dims[dim] = dims.get(dim, 0) - exp
        dims = {k: v for k, v in dims.items() if v != 0}
        name = f"{self.name}/{other.name}"
        return Unit(name=name, dimensions=dims, scale=self.scale / other.scale)

    def __pow__(self, exp: int) -> Unit:
        dims = {k: v * exp for k, v in self.dimensions.items()}
        dims = {k: v for k, v in dims.items() if v != 0}
        name = f"{self.name}^{exp}"
        return Unit(name=name, dimensions=dims, scale=self.scale**exp)

    def is_compatible(self, other: Unit) -> bool:
        """Check if two units have the same dimensions."""
        return self.dimensions == other.dimensions

    @property
    def is_dimensionless(self) -> bool:
        return len(self.dimensions) == 0

    def __repr__(self) -> str:
        if not self.dimensions:
            return "Unit(dimensionless)"
        parts = []
        for dim, exp in sorted(self.dimensions.items()):
            if exp == 1:
                parts.append(dim)
            else:
                parts.append(f"{dim}^{exp}")
        return f"Unit({' · '.join(parts)})"

    def __str__(self) -> str:
        if not self.dimensions:
            return ""
        parts = []
        for dim, exp in sorted(self.dimensions.items()):
            if exp == 1:
                parts.append(dim)
            else:
                parts.append(f"{dim}^{exp}")
        return " · ".join(parts)


# ── SI Base Units ────────────────────────────────────────────────────────

METER = Unit("m", {"m": 1})
KILOGRAM = Unit("kg", {"kg": 1})
SECOND = Unit("s", {"s": 1})
AMPERE = Unit("A", {"A": 1})
KELVIN = Unit("K", {"K": 1})
MOLE = Unit("mol", {"mol": 1})
CANDELA = Unit("cd", {"cd": 1})

# ── Derived Units ────────────────────────────────────────────────────────

HERTZ = Unit("Hz", {"s": -1})
NEWTON = Unit("N", {"kg": 1, "m": 1, "s": -2})
PASCAL = Unit("Pa", {"kg": 1, "m": -1, "s": -2})
JOULE = Unit("J", {"kg": 1, "m": 2, "s": -2})
WATT = Unit("W", {"kg": 1, "m": 2, "s": -3})
COULOMB = Unit("C", {"A": 1, "s": 1})
VOLT = Unit("V", {"kg": 1, "m": 2, "s": -3, "A": -1})
ELECTRONVOLT = Unit("eV", {"kg": 1, "m": 2, "s": -2}, scale=1.602176634e-19)

DIMENSIONLESS = Unit("", {})


@dataclass
class Quantity:
    """A physical quantity: value with unit.

    Arithmetic operations check unit compatibility and propagate units.
    """

    value: float
    unit: Unit

    def to(self, target_unit: Unit) -> Quantity:
        """Convert to a compatible unit."""
        if not self.unit.is_compatible(target_unit):
            raise ValueError(
                f"Cannot convert {self.unit} to {target_unit}: incompatible dimensions"
            )
        converted = self.value * (self.unit.scale / target_unit.scale)
        return Quantity(converted, target_unit)

    def __add__(self, other: Quantity) -> Quantity:
        if not isinstance(other, Quantity):
            return NotImplemented
        if not self.unit.is_compatible(other.unit):
            raise ValueError(f"Cannot add {self.unit} and {other.unit}")
        # Convert other to self's unit system
        other_converted = other.value * (other.unit.scale / self.unit.scale)
        return Quantity(self.value + other_converted, self.unit)

    def __sub__(self, other: Quantity) -> Quantity:
        if not isinstance(other, Quantity):
            return NotImplemented
        if not self.unit.is_compatible(other.unit):
            raise ValueError(f"Cannot subtract {self.unit} and {other.unit}")
        other_converted = other.value * (other.unit.scale / self.unit.scale)
        return Quantity(self.value - other_converted, self.unit)

    def __mul__(self, other: Quantity | float) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, self.unit * other.unit)
        if isinstance(other, (int, float)):
            return Quantity(self.value * other, self.unit)
        return NotImplemented

    def __rmul__(self, other: float) -> Quantity:
        return Quantity(self.value * other, self.unit)

    def __truediv__(self, other: Quantity | float) -> Quantity:
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, self.unit / other.unit)
        if isinstance(other, (int, float)):
            return Quantity(self.value / other, self.unit)
        return NotImplemented

    def __pow__(self, exp: int) -> Quantity:
        return Quantity(self.value**exp, self.unit**exp)

    def __repr__(self) -> str:
        return f"Quantity({self.value}, {self.unit})"

    def __str__(self) -> str:
        unit_str = str(self.unit)
        if unit_str:
            return f"{self.value} {unit_str}"
        return str(self.value)
