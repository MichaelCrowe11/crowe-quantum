"""Tests for physical unit system."""

import pytest

from crowe_synapse.units import (
    DIMENSIONLESS,
    JOULE,
    KILOGRAM,
    METER,
    NEWTON,
    SECOND,
    Quantity,
    Unit,
)


class TestUnit:
    def test_base_unit(self):
        assert METER.dimensions == {"m": 1}

    def test_multiply_units(self):
        area = METER * METER
        assert area.dimensions == {"m": 2}

    def test_divide_units(self):
        velocity = METER / SECOND
        assert velocity.dimensions == {"m": 1, "s": -1}

    def test_power_units(self):
        volume = METER**3
        assert volume.dimensions == {"m": 3}

    def test_compatibility(self):
        m_per_s = METER / SECOND
        velocity = Unit("v", {"m": 1, "s": -1})
        assert m_per_s.is_compatible(velocity)

    def test_not_compatible(self):
        assert not METER.is_compatible(SECOND)

    def test_dimensionless(self):
        assert DIMENSIONLESS.is_dimensionless

    def test_newton_dimensions(self):
        # N = kg·m·s⁻²
        assert NEWTON.dimensions == {"kg": 1, "m": 1, "s": -2}


class TestQuantity:
    def test_creation(self):
        q = Quantity(9.81, METER / SECOND**2)
        assert q.value == 9.81

    def test_add_compatible(self):
        a = Quantity(5.0, METER)
        b = Quantity(3.0, METER)
        c = a + b
        assert c.value == pytest.approx(8.0)

    def test_add_incompatible(self):
        a = Quantity(5.0, METER)
        b = Quantity(3.0, SECOND)
        with pytest.raises(ValueError):
            a + b

    def test_multiply(self):
        force = Quantity(10.0, NEWTON)
        distance = Quantity(5.0, METER)
        work = force * distance
        assert work.value == pytest.approx(50.0)
        assert work.unit.is_compatible(JOULE)

    def test_multiply_scalar(self):
        q = Quantity(5.0, METER)
        result = 3.0 * q
        assert result.value == pytest.approx(15.0)
        assert result.unit.dimensions == METER.dimensions

    def test_divide(self):
        d = Quantity(100.0, METER)
        t = Quantity(10.0, SECOND)
        v = d / t
        assert v.value == pytest.approx(10.0)
        assert v.unit.dimensions == {"m": 1, "s": -1}

    def test_power(self):
        side = Quantity(3.0, METER)
        area = side**2
        assert area.value == pytest.approx(9.0)
        assert area.unit.dimensions == {"m": 2}

    def test_str(self):
        q = Quantity(9.81, METER / SECOND**2)
        s = str(q)
        assert "9.81" in s
