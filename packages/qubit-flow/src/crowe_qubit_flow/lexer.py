"""QubitFlow lexer — tokenizer for the quantum circuit language."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    COMPLEX = auto()
    STRING = auto()
    IDENTIFIER = auto()
    BOOLEAN = auto()

    # Quantum keywords
    QUBIT = auto()
    QUDIT = auto()
    CIRCUIT = auto()
    GATE = auto()
    MEASURE = auto()
    ENTANGLE = auto()
    SUPERPOSE = auto()
    TELEPORT = auto()
    BARRIER = auto()
    RESET = auto()

    # Gate names
    GATE_H = auto()
    GATE_X = auto()
    GATE_Y = auto()
    GATE_Z = auto()
    GATE_S = auto()
    GATE_T = auto()
    GATE_SDG = auto()
    GATE_TDG = auto()
    GATE_RX = auto()
    GATE_RY = auto()
    GATE_RZ = auto()
    GATE_CNOT = auto()
    GATE_CZ = auto()
    GATE_SWAP = auto()
    GATE_TOFFOLI = auto()
    GATE_FREDKIN = auto()
    GATE_PHASE = auto()
    GATE_U = auto()

    # Algorithm keywords
    GROVERS = auto()
    SHORS = auto()
    VQE = auto()
    QAOA = auto()
    QFT = auto()
    QPE = auto()

    # Error correction
    SYNDROME = auto()
    CORRECT = auto()
    STABILIZER = auto()

    # Control flow
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    DEF = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()

    # Dirac notation
    KET_BAR = auto()      # |
    BRA_BAR = auto()      # ⟨ or <|
    DAGGER = auto()       # †
    TENSOR = auto()       # ⊗

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    CARET = auto()
    PERCENT = auto()
    EQUALS = auto()
    DOUBLE_EQUALS = auto()
    NOT_EQUALS = auto()
    LESS = auto()
    GREATER = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    ARROW = auto()         # ->

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    DOT = auto()

    # Structure
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()


KEYWORDS: dict[str, TokenType] = {
    "qubit": TokenType.QUBIT,
    "qudit": TokenType.QUDIT,
    "circuit": TokenType.CIRCUIT,
    "gate": TokenType.GATE,
    "measure": TokenType.MEASURE,
    "entangle": TokenType.ENTANGLE,
    "superpose": TokenType.SUPERPOSE,
    "teleport": TokenType.TELEPORT,
    "barrier": TokenType.BARRIER,
    "reset": TokenType.RESET,
    "grovers": TokenType.GROVERS,
    "shors": TokenType.SHORS,
    "vqe": TokenType.VQE,
    "qaoa": TokenType.QAOA,
    "qft": TokenType.QFT,
    "qpe": TokenType.QPE,
    "syndrome": TokenType.SYNDROME,
    "correct": TokenType.CORRECT,
    "stabilizer": TokenType.STABILIZER,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "in": TokenType.IN,
    "def": TokenType.DEF,
    "return": TokenType.RETURN,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "true": TokenType.BOOLEAN,
    "false": TokenType.BOOLEAN,
    "True": TokenType.BOOLEAN,
    "False": TokenType.BOOLEAN,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
}

GATE_KEYWORDS: dict[str, TokenType] = {
    "H": TokenType.GATE_H,
    "X": TokenType.GATE_X,
    "Y": TokenType.GATE_Y,
    "Z": TokenType.GATE_Z,
    "S": TokenType.GATE_S,
    "T": TokenType.GATE_T,
    "SDG": TokenType.GATE_SDG,
    "TDG": TokenType.GATE_TDG,
    "RX": TokenType.GATE_RX,
    "RY": TokenType.GATE_RY,
    "RZ": TokenType.GATE_RZ,
    "CNOT": TokenType.GATE_CNOT,
    "CX": TokenType.GATE_CNOT,
    "CZ": TokenType.GATE_CZ,
    "SWAP": TokenType.GATE_SWAP,
    "TOFFOLI": TokenType.GATE_TOFFOLI,
    "CCX": TokenType.GATE_TOFFOLI,
    "FREDKIN": TokenType.GATE_FREDKIN,
    "CSWAP": TokenType.GATE_FREDKIN,
    "PHASE": TokenType.GATE_PHASE,
    "U": TokenType.GATE_U,
}

GATE_TOKEN_TO_NAME: dict[TokenType, str] = {
    TokenType.GATE_H: "H",
    TokenType.GATE_X: "X",
    TokenType.GATE_Y: "Y",
    TokenType.GATE_Z: "Z",
    TokenType.GATE_S: "S",
    TokenType.GATE_T: "T",
    TokenType.GATE_SDG: "SDG",
    TokenType.GATE_TDG: "TDG",
    TokenType.GATE_RX: "RX",
    TokenType.GATE_RY: "RY",
    TokenType.GATE_RZ: "RZ",
    TokenType.GATE_CNOT: "CNOT",
    TokenType.GATE_CZ: "CZ",
    TokenType.GATE_SWAP: "SWAP",
    TokenType.GATE_TOFFOLI: "TOFFOLI",
    TokenType.GATE_FREDKIN: "FREDKIN",
    TokenType.GATE_PHASE: "PHASE",
    TokenType.GATE_U: "U",
}


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.column})"


class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(f"Line {line}, Col {column}: {message}")
        self.line = line
        self.column = column


class Lexer:
    """Tokenizer for the QubitFlow quantum circuit language."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []
        self.indent_stack: list[int] = [0]

    def tokenize(self) -> list[Token]:
        self.tokens = []
        self.pos = 0
        self.line = 1
        self.column = 1
        self.indent_stack = [0]

        while self.pos < len(self.source):
            if self._at_line_start():
                self._handle_indentation()
            self._skip_whitespace_inline()
            if self.pos >= len(self.source):
                break
            ch = self.source[self.pos]

            if ch == "\n":
                self._emit(TokenType.NEWLINE, "\n")
                self._advance()
                continue
            if ch == "#":
                self._skip_comment()
                continue
            if ch == '"' or ch == "'":
                self._read_string(ch)
                continue
            if ch.isdigit() or (ch == "." and self._peek_next().isdigit()):
                self._read_number()
                continue
            if ch.isalpha() or ch == "_":
                self._read_identifier()
                continue
            if ch == "|":
                self._emit(TokenType.KET_BAR, "|")
                self._advance()
                continue
            if ch == "⟨":
                self._emit(TokenType.BRA_BAR, "⟨")
                self._advance()
                continue
            if ch == "†":
                self._emit(TokenType.DAGGER, "†")
                self._advance()
                continue
            if ch == "⊗":
                self._emit(TokenType.TENSOR, "⊗")
                self._advance()
                continue

            self._read_operator_or_delimiter()

        # Close remaining indents
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self._emit(TokenType.DEDENT, "")

        self._emit(TokenType.EOF, "")
        return self.tokens

    def _at_line_start(self) -> bool:
        return self.column == 1

    def _handle_indentation(self) -> None:
        indent = 0
        while self.pos < len(self.source) and self.source[self.pos] in (" ", "\t"):
            if self.source[self.pos] == "\t":
                indent += 4
            else:
                indent += 1
            self._advance()

        if self.pos < len(self.source) and self.source[self.pos] in ("\n", "#"):
            return

        if indent > self.indent_stack[-1]:
            self.indent_stack.append(indent)
            self._emit(TokenType.INDENT, "")
        else:
            while indent < self.indent_stack[-1]:
                self.indent_stack.pop()
                self._emit(TokenType.DEDENT, "")

    def _skip_whitespace_inline(self) -> None:
        while self.pos < len(self.source) and self.source[self.pos] in (" ", "\t"):
            self._advance()

    def _skip_comment(self) -> None:
        while self.pos < len(self.source) and self.source[self.pos] != "\n":
            self._advance()

    def _read_string(self, quote: str) -> None:
        start = self.pos
        self._advance()
        while self.pos < len(self.source) and self.source[self.pos] != quote:
            if self.source[self.pos] == "\\":
                self._advance()
            self._advance()
        if self.pos < len(self.source):
            self._advance()
        self._emit(TokenType.STRING, self.source[start:self.pos])

    def _read_number(self) -> None:
        start = self.pos
        has_dot = False
        while self.pos < len(self.source) and (self.source[self.pos].isdigit() or self.source[self.pos] == "."):
            if self.source[self.pos] == ".":
                if has_dot:
                    break
                has_dot = True
            self._advance()

        # Check for complex: 2i, 3.14i
        if self.pos < len(self.source) and self.source[self.pos] == "i":
            self._advance()
            self._emit(TokenType.COMPLEX, self.source[start:self.pos])
        elif has_dot:
            self._emit(TokenType.FLOAT, self.source[start:self.pos])
        else:
            self._emit(TokenType.INTEGER, self.source[start:self.pos])

    def _read_identifier(self) -> None:
        start = self.pos
        while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == "_"):
            self._advance()
        word = self.source[start:self.pos]

        if word in KEYWORDS:
            self._emit(KEYWORDS[word], word)
        elif word in GATE_KEYWORDS:
            self._emit(GATE_KEYWORDS[word], word)
        else:
            self._emit(TokenType.IDENTIFIER, word)

    def _read_operator_or_delimiter(self) -> None:
        ch = self.source[self.pos]
        nxt = self._peek_next()

        two_char = ch + nxt
        two_char_ops: dict[str, TokenType] = {
            "==": TokenType.DOUBLE_EQUALS,
            "!=": TokenType.NOT_EQUALS,
            "<=": TokenType.LESS_EQUAL,
            ">=": TokenType.GREATER_EQUAL,
            "->": TokenType.ARROW,
            "&&": TokenType.AND,
            "||": TokenType.OR,
        }
        if two_char in two_char_ops:
            self._emit(two_char_ops[two_char], two_char)
            self._advance()
            self._advance()
            return

        one_char_ops: dict[str, TokenType] = {
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.STAR,
            "/": TokenType.SLASH,
            "^": TokenType.CARET,
            "%": TokenType.PERCENT,
            "=": TokenType.EQUALS,
            "<": TokenType.LESS,
            ">": TokenType.GREATER,
            "!": TokenType.NOT,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            "[": TokenType.LBRACKET,
            "]": TokenType.RBRACKET,
            "{": TokenType.LBRACE,
            "}": TokenType.RBRACE,
            ",": TokenType.COMMA,
            ":": TokenType.COLON,
            ";": TokenType.SEMICOLON,
            ".": TokenType.DOT,
        }
        if ch in one_char_ops:
            self._emit(one_char_ops[ch], ch)
            self._advance()
            return

        raise LexerError(f"Unexpected character: {ch!r}", self.line, self.column)

    def _peek_next(self) -> str:
        if self.pos + 1 < len(self.source):
            return self.source[self.pos + 1]
        return ""

    def _emit(self, token_type: TokenType, value: str) -> None:
        self.tokens.append(Token(token_type, value, self.line, self.column))

    def _advance(self) -> None:
        if self.pos < len(self.source):
            if self.source[self.pos] == "\n":
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.pos += 1
