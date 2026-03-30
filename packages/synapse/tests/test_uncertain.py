"""Tests for uncertainty propagation."""

import math

import pytest
from crowe_synapse.uncertain import UncertainValue


class TestCreation:
    def test_value_only(self):
        x = UncertainValue(3.14)
        assert x.value == 3.14
        assert x.uncertainty == 0.0

    def test_value_with_uncertainty(self):
        x = UncertainValue(9.81, 0.02)
        assert x.value == 9.81
        assert x.uncertainty == 0.02

    def test_negative_uncertainty_abs(self):
        x = UncertainValue(1.0, -0.1)
        assert x.uncertainty == 0.1


class TestArithmetic:
    def test_addition(self):
        a = UncertainValue(10.0, 0.1)
        b = UncertainValue(20.0, 0.2)
        c = a + b
        assert c.value == pytest.approx(30.0)
        assert c.uncertainty == pytest.approx(math.sqrt(0.01 + 0.04))

    def test_subtraction(self):
        a = UncertainValue(10.0, 0.1)
        b = UncertainValue(3.0, 0.2)
        c = a - b
        assert c.value == pytest.approx(7.0)
        assert c.uncertainty == pytest.approx(math.sqrt(0.01 + 0.04))

    def test_multiplication(self):
        a = UncertainValue(4.0, 0.1)
        b = UncertainValue(5.0, 0.2)
        c = a * b
        assert c.value == pytest.approx(20.0)
        rel_a = 0.1 / 4.0
        rel_b = 0.2 / 5.0
        expected_unc = 20.0 * math.sqrt(rel_a**2 + rel_b**2)
        assert c.uncertainty == pytest.approx(expected_unc)

    def test_division(self):
        a = UncertainValue(10.0, 0.5)
        b = UncertainValue(2.0, 0.1)
        c = a / b
        assert c.value == pytest.approx(5.0)

    def test_division_by_zero(self):
        a = UncertainValue(1.0, 0.1)
        b = UncertainValue(0.0, 0.0)
        with pytest.raises(ZeroDivisionError):
            a / b

    def test_power(self):
        a = UncertainValue(3.0, 0.1)
        c = a**2
        assert c.value == pytest.approx(9.0)
        assert c.uncertainty == pytest.approx(2 * 3.0 * 0.1)

    def test_negation(self):
        a = UncertainValue(5.0, 0.1)
        b = -a
        assert b.value == -5.0
        assert b.uncertainty == 0.1

    def test_add_float(self):
        a = UncertainValue(10.0, 0.1)
        c = a + 5.0
        assert c.value == pytest.approx(15.0)
        assert c.uncertainty == pytest.approx(0.1)

    def test_radd_float(self):
        a = UncertainValue(10.0, 0.1)
        c = 5.0 + a
        assert c.value == pytest.approx(15.0)

    def test_mul_by_scalar(self):
        a = UncertainValue(10.0, 0.1)
        c = 2.0 * a
        assert c.value == pytest.approx(20.0)


class TestScientificFunctions:
    def test_sqrt(self):
        a = UncertainValue(4.0, 0.1)
        c = a.sqrt()
        assert c.value == pytest.approx(2.0)

    def test_sin(self):
        a = UncertainValue(0.0, 0.01)
        c = a.sin()
        assert c.value == pytest.approx(0.0, abs=1e-10)
        assert c.uncertainty == pytest.approx(0.01)  # cos(0) = 1

    def test_cos(self):
        a = UncertainValue(0.0, 0.01)
        c = a.cos()
        assert c.value == pytest.approx(1.0)
        assert c.uncertainty == pytest.approx(0.0, abs=1e-10)  # sin(0) = 0

    def test_exp(self):
        a = UncertainValue(0.0, 0.01)
        c = a.exp()
        assert c.value == pytest.approx(1.0)
        assert c.uncertainty == pytest.approx(0.01)

    def test_log(self):
        a = UncertainValue(math.e, 0.01)
        c = a.log()
        assert c.value == pytest.approx(1.0)

    def test_log_negative(self):
        with pytest.raises(ValueError):
            UncertainValue(-1.0).log()


class TestComparison:
    def test_overlaps(self):
        a = UncertainValue(10.0, 0.5)
        b = UncertainValue(10.3, 0.5)
        assert a.overlaps(b)

    def test_no_overlap(self):
        a = UncertainValue(10.0, 0.1)
        b = UncertainValue(15.0, 0.1)
        assert not a.overlaps(b)

    def test_relative_uncertainty(self):
        a = UncertainValue(100.0, 1.0)
        assert a.relative_uncertainty == pytest.approx(0.01)


class TestDisplay:
    def test_repr(self):
        a = UncertainValue(3.14, 0.01)
        assert "±" in repr(a)

    def test_str(self):
        a = UncertainValue(3.14, 0.01)
        assert "±" in str(a)
