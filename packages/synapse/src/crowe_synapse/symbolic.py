"""Symbolic math engine — lightweight CAS for scientific expressions.

Provides symbolic algebra for constructing and simplifying mathematical
expressions. Delegates heavy lifting to SymPy when available, but provides
a self-contained core for basic operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class Symbol:
    """A symbolic variable in an expression."""

    name: str

    def to_sympy(self) -> sp.Symbol:
        return sp.Symbol(self.name)

    def __add__(self, other: Symbol | Expression | float) -> Expression:
        return Expression(sp.Add(self.to_sympy(), _to_sympy(other)))

    def __radd__(self, other: float) -> Expression:
        return Expression(sp.Add(_to_sympy(other), self.to_sympy()))

    def __sub__(self, other: Symbol | Expression | float) -> Expression:
        return Expression(sp.Add(self.to_sympy(), sp.Mul(-1, _to_sympy(other))))

    def __rsub__(self, other: float) -> Expression:
        return Expression(sp.Add(_to_sympy(other), sp.Mul(-1, self.to_sympy())))

    def __mul__(self, other: Symbol | Expression | float) -> Expression:
        return Expression(sp.Mul(self.to_sympy(), _to_sympy(other)))

    def __rmul__(self, other: float) -> Expression:
        return Expression(sp.Mul(_to_sympy(other), self.to_sympy()))

    def __truediv__(self, other: Symbol | Expression | float) -> Expression:
        return Expression(sp.Mul(self.to_sympy(), sp.Pow(_to_sympy(other), -1)))

    def __pow__(self, exp: float | Symbol) -> Expression:
        return Expression(sp.Pow(self.to_sympy(), _to_sympy(exp)))

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"

    def __str__(self) -> str:
        return self.name


class Expression:
    """A symbolic mathematical expression backed by SymPy."""

    def __init__(self, expr: sp.Expr) -> None:
        self._expr = expr

    @property
    def sympy_expr(self) -> sp.Expr:
        return self._expr

    def simplify(self) -> Expression:
        return Expression(sp.simplify(self._expr))

    def expand(self) -> Expression:
        return Expression(sp.expand(self._expr))

    def factor(self) -> Expression:
        return Expression(sp.factor(self._expr))

    def differentiate(self, var: Symbol) -> Expression:
        return Expression(sp.diff(self._expr, var.to_sympy()))

    def integrate(self, var: Symbol) -> Expression:
        return Expression(sp.integrate(self._expr, var.to_sympy()))

    def substitute(self, var: Symbol, value: float | Symbol | Expression) -> Expression:
        return Expression(self._expr.subs(var.to_sympy(), _to_sympy(value)))

    def evaluate(self, **kwargs: float) -> float:
        """Evaluate expression with given variable values."""
        subs = {sp.Symbol(k): v for k, v in kwargs.items()}
        result = self._expr.subs(subs)
        return float(result.evalf())

    def free_symbols(self) -> set[str]:
        return {str(s) for s in self._expr.free_symbols}

    # ── Arithmetic ───────────────────────────────────────────────────────

    def __add__(self, other: Any) -> Expression:
        return Expression(sp.Add(self._expr, _to_sympy(other)))

    def __radd__(self, other: Any) -> Expression:
        return Expression(sp.Add(_to_sympy(other), self._expr))

    def __sub__(self, other: Any) -> Expression:
        return Expression(sp.Add(self._expr, sp.Mul(-1, _to_sympy(other))))

    def __mul__(self, other: Any) -> Expression:
        return Expression(sp.Mul(self._expr, _to_sympy(other)))

    def __rmul__(self, other: Any) -> Expression:
        return Expression(sp.Mul(_to_sympy(other), self._expr))

    def __truediv__(self, other: Any) -> Expression:
        return Expression(sp.Mul(self._expr, sp.Pow(_to_sympy(other), -1)))

    def __pow__(self, exp: Any) -> Expression:
        return Expression(sp.Pow(self._expr, _to_sympy(exp)))

    def __neg__(self) -> Expression:
        return Expression(sp.Mul(-1, self._expr))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Expression):
            return sp.simplify(self._expr - other._expr) == 0
        return NotImplemented

    def __repr__(self) -> str:
        return f"Expression({self._expr})"

    def __str__(self) -> str:
        return str(self._expr)


def simplify(expr: Expression) -> Expression:
    """Simplify a symbolic expression."""
    return expr.simplify()


def _to_sympy(obj: Any) -> sp.Expr:
    if isinstance(obj, Symbol):
        return obj.to_sympy()
    if isinstance(obj, Expression):
        return obj.sympy_expr
    if isinstance(obj, (int, float)):
        return sp.Float(obj) if isinstance(obj, float) else sp.Integer(obj)
    raise TypeError(f"Cannot convert {type(obj)} to sympy expression")
