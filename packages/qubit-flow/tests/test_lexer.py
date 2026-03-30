"""Tests for the QubitFlow lexer."""

import pytest
from crowe_qubit_flow.lexer import Lexer, LexerError, TokenType


class TestBasicTokens:
    def test_empty_source(self):
        tokens = Lexer("").tokenize()
        assert tokens[-1].type == TokenType.EOF

    def test_integer(self):
        tokens = Lexer("42").tokenize()
        assert tokens[0].type == TokenType.INTEGER
        assert tokens[0].value == "42"

    def test_float(self):
        tokens = Lexer("3.14").tokenize()
        assert tokens[0].type == TokenType.FLOAT
        assert tokens[0].value == "3.14"

    def test_complex_number(self):
        tokens = Lexer("2i").tokenize()
        assert tokens[0].type == TokenType.COMPLEX
        assert tokens[0].value == "2i"

    def test_string(self):
        tokens = Lexer('"hello"').tokenize()
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == '"hello"'

    def test_boolean_true(self):
        tokens = Lexer("true").tokenize()
        assert tokens[0].type == TokenType.BOOLEAN

    def test_boolean_false(self):
        tokens = Lexer("False").tokenize()
        assert tokens[0].type == TokenType.BOOLEAN

    def test_identifier(self):
        tokens = Lexer("my_var").tokenize()
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "my_var"


class TestQuantumKeywords:
    def test_qubit(self):
        tokens = Lexer("qubit q").tokenize()
        assert tokens[0].type == TokenType.QUBIT
        assert tokens[1].type == TokenType.IDENTIFIER

    def test_measure(self):
        tokens = Lexer("measure").tokenize()
        assert tokens[0].type == TokenType.MEASURE

    def test_entangle(self):
        tokens = Lexer("entangle").tokenize()
        assert tokens[0].type == TokenType.ENTANGLE

    def test_circuit(self):
        tokens = Lexer("circuit").tokenize()
        assert tokens[0].type == TokenType.CIRCUIT

    def test_superpose(self):
        tokens = Lexer("superpose").tokenize()
        assert tokens[0].type == TokenType.SUPERPOSE

    def test_teleport(self):
        tokens = Lexer("teleport").tokenize()
        assert tokens[0].type == TokenType.TELEPORT


class TestGateKeywords:
    def test_hadamard(self):
        tokens = Lexer("H").tokenize()
        assert tokens[0].type == TokenType.GATE_H

    def test_pauli_x(self):
        tokens = Lexer("X").tokenize()
        assert tokens[0].type == TokenType.GATE_X

    def test_cnot(self):
        tokens = Lexer("CNOT").tokenize()
        assert tokens[0].type == TokenType.GATE_CNOT

    def test_cx_alias(self):
        tokens = Lexer("CX").tokenize()
        assert tokens[0].type == TokenType.GATE_CNOT

    def test_rotation_gates(self):
        for gate in ["RX", "RY", "RZ"]:
            tokens = Lexer(gate).tokenize()
            assert tokens[0].type == getattr(TokenType, f"GATE_{gate}")

    def test_toffoli_alias(self):
        tokens = Lexer("CCX").tokenize()
        assert tokens[0].type == TokenType.GATE_TOFFOLI


class TestDiracNotation:
    def test_ket_bar(self):
        tokens = Lexer("|").tokenize()
        assert tokens[0].type == TokenType.KET_BAR

    def test_bra_bar_unicode(self):
        tokens = Lexer("⟨").tokenize()
        assert tokens[0].type == TokenType.BRA_BAR

    def test_dagger(self):
        tokens = Lexer("†").tokenize()
        assert tokens[0].type == TokenType.DAGGER

    def test_tensor(self):
        tokens = Lexer("⊗").tokenize()
        assert tokens[0].type == TokenType.TENSOR


class TestOperators:
    def test_arithmetic(self):
        tokens = Lexer("+ - * /").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH]

    def test_comparison(self):
        tokens = Lexer("== != <= >=").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [
            TokenType.DOUBLE_EQUALS,
            TokenType.NOT_EQUALS,
            TokenType.LESS_EQUAL,
            TokenType.GREATER_EQUAL,
        ]

    def test_arrow(self):
        tokens = Lexer("->").tokenize()
        assert tokens[0].type == TokenType.ARROW


class TestIndentation:
    def test_indent_dedent(self):
        source = "if true:\n    x = 1\n"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        assert TokenType.INDENT in types
        assert TokenType.DEDENT in types

    def test_nested_indent(self):
        source = "if true:\n    if false:\n        x = 1\n"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        assert types.count(TokenType.INDENT) == 2


class TestComments:
    def test_comment_skipped(self):
        tokens = Lexer("# this is a comment\n42").tokenize()
        int_tokens = [t for t in tokens if t.type == TokenType.INTEGER]
        assert len(int_tokens) == 1


class TestLineTracking:
    def test_line_numbers(self):
        tokens = Lexer("x\ny\nz").tokenize()
        ids = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        assert ids[0].line == 1
        assert ids[1].line == 2
        assert ids[2].line == 3


class TestErrors:
    def test_unexpected_character(self):
        with pytest.raises(LexerError):
            Lexer("@").tokenize()


class TestComplexProgram:
    def test_full_circuit(self):
        source = """circuit bell():
    qubit q[2]
    H q[0]
    CNOT q[0], q[1]
    measure q[0]
    measure q[1]
"""
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        assert TokenType.CIRCUIT in types
        assert TokenType.GATE_H in types
        assert TokenType.GATE_CNOT in types
        assert TokenType.MEASURE in types
