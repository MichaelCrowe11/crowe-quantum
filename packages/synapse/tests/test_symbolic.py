"""Tests for symbolic math engine."""

import pytest
from crowe_synapse.symbolic import Expression, Symbol, simplify


class TestSymbol:
    def test_creation(self):
        x = Symbol("x")
        assert x.name == "x"

    def test_add(self):
        x = Symbol("x")
        y = Symbol("y")
        expr = x + y
        assert isinstance(expr, Expression)
        assert "x" in str(expr)
        assert "y" in str(expr)

    def test_mul(self):
        x = Symbol("x")
        expr = x * 3
        assert isinstance(expr, Expression)

    def test_pow(self):
        x = Symbol("x")
        expr = x**2
        assert isinstance(expr, Expression)


class TestExpression:
    def test_simplify(self):
        x = Symbol("x")
        expr = x + x
        simplified = expr.simplify()
        assert "2" in str(simplified)

    def test_expand(self):
        x = Symbol("x")
        y = Symbol("y")
        expr = (x + y) * (x + y)
        expanded = expr.expand()
        expanded_str = str(expanded)
        assert "x**2" in expanded_str or "x²" in expanded_str

    def test_differentiate(self):
        x = Symbol("x")
        expr = x**3
        deriv = expr.differentiate(x)
        # d/dx(x^3) = 3x^2
        assert "3" in str(deriv)

    def test_integrate(self):
        x = Symbol("x")
        expr = x**2
        integral = expr.integrate(x)
        # ∫x² dx = x³/3
        assert "3" in str(integral)

    def test_substitute(self):
        x = Symbol("x")
        expr = x**2 + 1
        result = expr.substitute(x, 3.0)
        val = result.evaluate()
        assert val == pytest.approx(10.0)

    def test_evaluate(self):
        x = Symbol("x")
        y = Symbol("y")
        expr = x * y + 1
        val = expr.evaluate(x=2.0, y=3.0)
        assert val == pytest.approx(7.0)

    def test_free_symbols(self):
        x = Symbol("x")
        y = Symbol("y")
        expr = x + y * 2
        syms = expr.free_symbols()
        assert syms == {"x", "y"}

    def test_equality(self):
        x = Symbol("x")
        e1 = x + x
        e2 = 2 * x
        assert e1 == e2

    def test_scalar_add(self):
        x = Symbol("x")
        expr = 5.0 + x
        val = expr.evaluate(x=3.0)
        assert val == pytest.approx(8.0)


class TestSimplify:
    def test_function(self):
        x = Symbol("x")
        expr = x + x + x
        result = simplify(expr)
        assert "3" in str(result)
