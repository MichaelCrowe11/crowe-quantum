"""QubitFlow compiler — compiles AST to CircuitIR for backend execution."""

from __future__ import annotations

import math
from typing import Any

from crowe_quantum_core.protocols import CircuitIR

from crowe_qubit_flow.ast_nodes import (
    ASTNode,
    BarrierNode,
    BinaryOp,
    CircuitNode,
    EntanglementNode,
    FloatLiteral,
    ForNode,
    GateApplicationNode,
    Identifier,
    IfNode,
    IndexAccess,
    IntegerLiteral,
    MeasurementNode,
    ProgramNode,
    QFTNode,
    QubitDeclNode,
    ResetNode,
    SuperpositionNode,
    UnaryOp,
)
from crowe_qubit_flow.parser import Parser


class CompileError(Exception):
    def __init__(self, message: str, node: ASTNode) -> None:
        super().__init__(f"Line {node.line}, Col {node.column}: {message}")
        self.node = node


class Compiler:
    """Compiles QubitFlow AST to CircuitIR.

    Unlike the interpreter (which executes directly on state vectors),
    the compiler produces a CircuitIR that can be submitted to any
    backend — simulators, IBM Quantum hardware, etc.
    """

    def __init__(self) -> None:
        self.qubit_map: dict[str, int] = {}  # register_name -> start index
        self.register_sizes: dict[str, int] = {}
        self.total_qubits = 0
        self.classical_bits = 0
        self.circuit: CircuitIR | None = None
        self.constants: dict[str, float] = {
            "pi": math.pi,
            "tau": math.tau,
            "e": math.e,
        }

    def compile(self, source: str) -> CircuitIR:
        """Parse and compile a QubitFlow program to CircuitIR."""
        parser = Parser.from_source(source)
        program = parser.parse()
        return self.compile_ast(program)

    def compile_ast(self, program: ProgramNode) -> CircuitIR:
        """Compile a parsed program to CircuitIR."""
        # First pass: collect qubit declarations
        for stmt in program.statements:
            if isinstance(stmt, QubitDeclNode):
                self._register_qubits(stmt)

        self.circuit = CircuitIR(
            num_qubits=self.total_qubits,
            num_classical_bits=self.classical_bits,
        )

        # Second pass: compile operations
        for stmt in program.statements:
            if not isinstance(stmt, QubitDeclNode):
                self._compile_stmt(stmt)

        return self.circuit

    def _register_qubits(self, node: QubitDeclNode) -> None:
        size = self._eval_const(node.size) if node.size else 1
        self.qubit_map[node.name] = self.total_qubits
        self.register_sizes[node.name] = size
        self.total_qubits += size

    def _resolve_qubit_index(self, node: ASTNode) -> int:
        """Resolve a qubit reference to a global index."""
        if isinstance(node, IndexAccess) and isinstance(node.target, Identifier):
            name = node.target.name
            if name not in self.qubit_map:
                raise CompileError(f"Unknown qubit register: '{name}'", node)
            base = self.qubit_map[name]
            idx = self._eval_const(node.index)
            return base + idx
        if isinstance(node, Identifier):
            name = node.name
            if name not in self.qubit_map:
                raise CompileError(f"Unknown qubit register: '{name}'", node)
            return self.qubit_map[name]
        raise CompileError("Invalid qubit reference", node)

    def _eval_const(self, node: ASTNode | None) -> int:
        """Evaluate a compile-time constant expression."""
        if node is None:
            return 0
        if isinstance(node, IntegerLiteral):
            return node.value
        if isinstance(node, FloatLiteral):
            return int(node.value)
        if isinstance(node, Identifier):
            if node.name in self.constants:
                return int(self.constants[node.name])
            raise CompileError(f"Unknown constant: '{node.name}'", node)
        if isinstance(node, BinaryOp):
            left = self._eval_const(node.left)
            right = self._eval_const(node.right)
            if node.op == "+":
                return left + right
            if node.op == "-":
                return left - right
            if node.op == "*":
                return left * right
            if node.op == "/":
                return left // right
        if isinstance(node, UnaryOp) and node.op == "-":
            return -self._eval_const(node.operand)
        raise CompileError("Cannot evaluate as compile-time constant", node)

    def _eval_float(self, node: ASTNode) -> float:
        """Evaluate a compile-time float expression (for gate parameters)."""
        if isinstance(node, IntegerLiteral):
            return float(node.value)
        if isinstance(node, FloatLiteral):
            return node.value
        if isinstance(node, Identifier):
            if node.name in self.constants:
                return self.constants[node.name]
            raise CompileError(f"Unknown constant: '{node.name}'", node)
        if isinstance(node, BinaryOp):
            left = self._eval_float(node.left)
            right = self._eval_float(node.right)
            if node.op == "+":
                return left + right
            if node.op == "-":
                return left - right
            if node.op == "*":
                return left * right
            if node.op == "/":
                return left / right
            if node.op == "^":
                return left ** right
        if isinstance(node, UnaryOp) and node.op == "-":
            return -self._eval_float(node.operand)
        raise CompileError("Cannot evaluate as float constant", node)

    # ── Statement compilation ────────────────────────────────────────────

    def _compile_stmt(self, node: ASTNode) -> None:
        if isinstance(node, GateApplicationNode):
            self._compile_gate(node)
        elif isinstance(node, MeasurementNode):
            self._compile_measure(node)
        elif isinstance(node, EntanglementNode):
            self._compile_entangle(node)
        elif isinstance(node, SuperpositionNode):
            self._compile_superpose(node)
        elif isinstance(node, BarrierNode):
            self._compile_barrier(node)
        elif isinstance(node, ResetNode):
            self._compile_reset(node)
        elif isinstance(node, QFTNode):
            self._compile_qft(node)
        elif isinstance(node, CircuitNode):
            # Compile circuit body inline
            for stmt in node.body:
                self._compile_stmt(stmt)
        elif isinstance(node, ForNode):
            self._compile_for(node)
        elif isinstance(node, IfNode):
            self._compile_if(node)
        else:
            raise CompileError(f"Cannot compile node type: {type(node).__name__}", node)

    def _compile_gate(self, node: GateApplicationNode) -> None:
        params = tuple(self._eval_float(p) for p in node.params)
        qubits = [self._resolve_qubit_index(t) for t in node.targets]
        assert self.circuit is not None
        self.circuit.add_gate(node.gate_name, qubits, params)

    def _compile_measure(self, node: MeasurementNode) -> None:
        qubit = self._resolve_qubit_index(node.target)
        cbit = self.classical_bits
        self.classical_bits += 1
        assert self.circuit is not None
        self.circuit.num_classical_bits = self.classical_bits
        self.circuit.add_measurement(qubit, cbit)

    def _compile_entangle(self, node: EntanglementNode) -> None:
        assert self.circuit is not None
        qubits = [self._resolve_qubit_index(t) for t in node.targets]
        # H on first, CNOT pairs
        self.circuit.add_gate("H", [qubits[0]])
        for q in qubits[1:]:
            self.circuit.add_gate("CNOT", [qubits[0], q])

    def _compile_superpose(self, node: SuperpositionNode) -> None:
        assert self.circuit is not None
        qubit = self._resolve_qubit_index(node.target)
        self.circuit.add_gate("H", [qubit])

    def _compile_barrier(self, node: BarrierNode) -> None:
        assert self.circuit is not None
        if node.targets:
            qubits = [self._resolve_qubit_index(t) for t in node.targets]
        else:
            qubits = list(range(self.total_qubits))
        self.circuit.add_barrier(qubits)

    def _compile_reset(self, node: ResetNode) -> None:
        assert self.circuit is not None
        qubit = self._resolve_qubit_index(node.target)
        self.circuit.add_reset(qubit)

    def _compile_qft(self, node: QFTNode) -> None:
        assert self.circuit is not None
        qubits = [self._resolve_qubit_index(t) for t in node.targets]
        n = len(qubits)
        for i in range(n):
            self.circuit.add_gate("H", [qubits[i]])
            for j in range(i + 1, n):
                k = j - i + 1
                angle = math.pi / (2 ** (k - 1))
                self.circuit.add_gate("PHASE", [qubits[j]], (angle,))

    def _compile_for(self, node: ForNode) -> None:
        """Unroll static for-loops at compile time."""
        iterable_range = self._eval_const(node.iterable)
        if isinstance(iterable_range, int):
            iterable_range = range(iterable_range)
        for val in iterable_range:
            self.constants[node.variable] = float(val)
            for stmt in node.body:
                self._compile_stmt(stmt)
            del self.constants[node.variable]

    def _compile_if(self, node: IfNode) -> None:
        """Compile classical conditional (limited to constant conditions)."""
        # In a circuit context, if-statements with measurement results
        # become conditional operations
        for stmt in node.body:
            self._compile_stmt(stmt)
