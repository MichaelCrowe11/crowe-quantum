"""Tests for the QubitFlow parser."""

import pytest
from crowe_qubit_flow.ast_nodes import (
    AssignmentNode,
    BinaryOp,
    CircuitNode,
    EntanglementNode,
    ExpressionStatement,
    FloatLiteral,
    ForNode,
    FunctionDefNode,
    GateApplicationNode,
    IfNode,
    IntegerLiteral,
    KetStateNode,
    MeasurementNode,
    ProgramNode,
    QubitDeclNode,
    ReturnNode,
    SuperpositionNode,
    WhileNode,
)
from crowe_qubit_flow.parser import ParseError, Parser


def parse(source: str) -> ProgramNode:
    return Parser.from_source(source).parse()


class TestQubitDeclarations:
    def test_simple_qubit(self):
        prog = parse("qubit q")
        assert len(prog.statements) == 1
        node = prog.statements[0]
        assert isinstance(node, QubitDeclNode)
        assert node.name == "q"
        assert node.size is None

    def test_qubit_array(self):
        prog = parse("qubit q[4]")
        node = prog.statements[0]
        assert isinstance(node, QubitDeclNode)
        assert node.name == "q"
        assert isinstance(node.size, IntegerLiteral)
        assert node.size.value == 4


class TestGateApplications:
    def test_single_qubit_gate(self):
        prog = parse("H q")
        node = prog.statements[0]
        assert isinstance(node, GateApplicationNode)
        assert node.gate_name == "H"
        assert len(node.targets) == 1

    def test_parameterized_gate(self):
        prog = parse("RX(3.14) q")
        node = prog.statements[0]
        assert isinstance(node, GateApplicationNode)
        assert node.gate_name == "RX"
        assert len(node.params) == 1
        assert isinstance(node.params[0], FloatLiteral)

    def test_two_qubit_gate(self):
        prog = parse("CNOT q, r")
        node = prog.statements[0]
        assert isinstance(node, GateApplicationNode)
        assert node.gate_name == "CNOT"
        assert len(node.targets) == 2

    def test_toffoli(self):
        prog = parse("TOFFOLI a, b, c")
        node = prog.statements[0]
        assert isinstance(node, GateApplicationNode)
        assert node.gate_name == "TOFFOLI"
        assert len(node.targets) == 3


class TestQuantumOperations:
    def test_measure(self):
        prog = parse("measure q")
        node = prog.statements[0]
        assert isinstance(node, MeasurementNode)

    def test_measure_with_arrow(self):
        prog = parse("measure q -> c")
        node = prog.statements[0]
        assert isinstance(node, MeasurementNode)
        assert node.classical_target is not None

    def test_entangle(self):
        prog = parse("entangle a, b")
        node = prog.statements[0]
        assert isinstance(node, EntanglementNode)
        assert len(node.targets) == 2

    def test_superpose(self):
        prog = parse("superpose q")
        node = prog.statements[0]
        assert isinstance(node, SuperpositionNode)


class TestCircuit:
    def test_circuit_definition(self):
        source = "circuit bell():\n    qubit q[2]\n    H q\n"
        prog = parse(source)
        node = prog.statements[0]
        assert isinstance(node, CircuitNode)
        assert node.name == "bell"
        assert len(node.body) == 2

    def test_circuit_with_params(self):
        source = "circuit rotate(theta):\n    RX(theta) q\n"
        prog = parse(source)
        node = prog.statements[0]
        assert isinstance(node, CircuitNode)
        assert node.params == ["theta"]


class TestControlFlow:
    def test_if_statement(self):
        source = "if x == 1:\n    y = 2\n"
        prog = parse(source)
        node = prog.statements[0]
        assert isinstance(node, IfNode)
        assert len(node.body) == 1
        assert len(node.else_body) == 0

    def test_if_else(self):
        source = "if x == 1:\n    y = 2\nelse:\n    y = 3\n"
        prog = parse(source)
        node = prog.statements[0]
        assert isinstance(node, IfNode)
        assert len(node.body) == 1
        assert len(node.else_body) == 1

    def test_while_loop(self):
        source = "while x < 10:\n    x = x + 1\n"
        prog = parse(source)
        node = prog.statements[0]
        assert isinstance(node, WhileNode)

    def test_for_loop(self):
        source = "for i in items:\n    x = i\n"
        prog = parse(source)
        node = prog.statements[0]
        assert isinstance(node, ForNode)
        assert node.variable == "i"


class TestFunctions:
    def test_function_def(self):
        source = "def add(a, b):\n    return a + b\n"
        prog = parse(source)
        node = prog.statements[0]
        assert isinstance(node, FunctionDefNode)
        assert node.name == "add"
        assert node.params == ["a", "b"]
        assert len(node.body) == 1
        assert isinstance(node.body[0], ReturnNode)


class TestExpressions:
    def test_assignment(self):
        prog = parse("x = 42")
        node = prog.statements[0]
        assert isinstance(node, AssignmentNode)

    def test_binary_op(self):
        prog = parse("x + y")
        node = prog.statements[0]
        assert isinstance(node, ExpressionStatement)
        assert isinstance(node.expression, BinaryOp)
        assert node.expression.op == "+"

    def test_precedence(self):
        prog = parse("a + b * c")
        node = prog.statements[0]
        assert isinstance(node, ExpressionStatement)
        expr = node.expression
        assert isinstance(expr, BinaryOp)
        assert expr.op == "+"
        assert isinstance(expr.right, BinaryOp)
        assert expr.right.op == "*"


class TestDiracNotation:
    def test_ket_state(self):
        prog = parse("|0|")
        node = prog.statements[0]
        assert isinstance(node, ExpressionStatement)
        assert isinstance(node.expression, KetStateNode)
        assert node.expression.label == "0"


class TestErrors:
    def test_missing_colon(self):
        with pytest.raises(ParseError):
            parse("if true\n    x = 1\n")

    def test_unexpected_token(self):
        with pytest.raises(ParseError):
            parse(")")
