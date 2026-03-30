"""QubitFlow interpreter — executes quantum circuit programs on state vectors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from crowe_quantum_core.gates import standard_gates
from crowe_quantum_core.states import StateVector

from crowe_qubit_flow.ast_nodes import (
    ASTNode,
    AssignmentNode,
    BarrierNode,
    BinaryOp,
    BooleanLiteral,
    BraKetNode,
    BraStateNode,
    BreakNode,
    CircuitNode,
    ComplexLiteral,
    ContinueNode,
    DaggerNode,
    EntanglementNode,
    ExpressionStatement,
    FloatLiteral,
    ForNode,
    FunctionCall,
    FunctionDefNode,
    GateApplicationNode,
    GroversNode,
    Identifier,
    IfNode,
    IndexAccess,
    IntegerLiteral,
    KetStateNode,
    ListNode,
    MeasurementNode,
    MemberAccess,
    ProgramNode,
    QFTNode,
    QubitDeclNode,
    ResetNode,
    ReturnNode,
    StringLiteral,
    SuperpositionNode,
    TensorProductNode,
    UnaryOp,
    WhileNode,
)
from crowe_qubit_flow.parser import Parser


class BreakSignal(Exception):
    pass


class ContinueSignal(Exception):
    pass


class ReturnSignal(Exception):
    def __init__(self, value: Any = None) -> None:
        self.value = value


class InterpreterError(Exception):
    def __init__(self, message: str, node: ASTNode) -> None:
        super().__init__(f"Line {node.line}, Col {node.column}: {message}")
        self.node = node


class QubitRegister:
    """Manages a collection of qubits backed by a StateVector."""

    def __init__(self, name: str, size: int) -> None:
        self.name = name
        self.size = size
        self.state = StateVector(size)
        self.classical: dict[int, int] = {}

    def __repr__(self) -> str:
        return f"QubitRegister({self.name}, size={self.size})"


class Environment:
    """Scoped variable environment with parent chain."""

    def __init__(self, parent: Environment | None = None) -> None:
        self.parent = parent
        self.bindings: dict[str, Any] = {}

    def get(self, name: str) -> Any:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise KeyError(name)

    def set(self, name: str, value: Any) -> None:
        self.bindings[name] = value

    def has(self, name: str) -> bool:
        if name in self.bindings:
            return True
        if self.parent is not None:
            return self.parent.has(name)
        return False

    def child(self) -> Environment:
        return Environment(parent=self)


class Interpreter:
    """Executes QubitFlow programs.

    Provides a quantum runtime that manages qubit registers, applies gates,
    and performs measurements using crowe_quantum_core's StateVector.
    """

    def __init__(self) -> None:
        self.env = Environment()
        self.measurements: list[dict[str, Any]] = []
        self._setup_builtins()

    def _setup_builtins(self) -> None:
        self.env.set("pi", math.pi)
        self.env.set("tau", math.tau)
        self.env.set("e", math.e)
        self.env.set("sqrt", math.sqrt)
        self.env.set("sin", math.sin)
        self.env.set("cos", math.cos)
        self.env.set("tan", math.tan)
        self.env.set("log", math.log)
        self.env.set("abs", abs)
        self.env.set("print", print)
        self.env.set("range", range)
        self.env.set("len", len)

    def run(self, source: str) -> dict[str, Any]:
        """Parse and run a QubitFlow program. Returns execution context."""
        parser = Parser.from_source(source)
        program = parser.parse()
        return self.execute(program)

    def execute(self, program: ProgramNode) -> dict[str, Any]:
        """Execute a parsed program."""
        for stmt in program.statements:
            self._exec(stmt)
        return {
            "env": self.env.bindings,
            "measurements": self.measurements,
        }

    # ── Dispatch ─────────────────────────────────────────────────────────

    def _exec(self, node: ASTNode) -> Any:
        method_name = f"_exec_{type(node).__name__}"
        method = getattr(self, method_name, None)
        if method is None:
            raise InterpreterError(f"Unhandled node type: {type(node).__name__}", node)
        return method(node)

    def _eval(self, node: ASTNode) -> Any:
        return self._exec(node)

    # ── Statements ───────────────────────────────────────────────────────

    def _exec_ExpressionStatement(self, node: ExpressionStatement) -> None:
        self._eval(node.expression)

    def _exec_AssignmentNode(self, node: AssignmentNode) -> None:
        value = self._eval(node.value)
        if isinstance(node.target, Identifier):
            self.env.set(node.target.name, value)
        elif isinstance(node.target, IndexAccess):
            target = self._eval(node.target.target)
            index = self._eval(node.target.index)
            target[index] = value
        else:
            raise InterpreterError("Invalid assignment target", node)

    # ── Quantum declarations ─────────────────────────────────────────────

    def _exec_QubitDeclNode(self, node: QubitDeclNode) -> None:
        size = self._eval(node.size) if node.size else 1
        reg = QubitRegister(node.name, size)
        self.env.set(node.name, reg)

    def _exec_QuditDeclNode(self, node: Any) -> None:
        raise InterpreterError("Qudit simulation not yet supported", node)

    # ── Circuit ──────────────────────────────────────────────────────────

    def _exec_CircuitNode(self, node: CircuitNode) -> None:
        def circuit_fn(*args: Any) -> Any:
            child_env = self.env.child()
            old_env = self.env
            self.env = child_env
            for pname, arg in zip(node.params, args):
                self.env.set(pname, arg)
            result = None
            try:
                for stmt in node.body:
                    result = self._exec(stmt)
            except ReturnSignal as r:
                result = r.value
            finally:
                self.env = old_env
            return result

        self.env.set(node.name, circuit_fn)

    # ── Gate operations ──────────────────────────────────────────────────

    def _exec_GateApplicationNode(self, node: GateApplicationNode) -> None:
        params = [self._eval(p) for p in node.params]
        gate = standard_gates.get_gate(node.gate_name, *params)
        matrix = gate.matrix()

        if gate.arity == 1:
            for target_node in node.targets:
                reg, qubit_idx = self._resolve_qubit(target_node)
                reg.state.apply_gate(matrix, [qubit_idx])
        else:
            # Multi-qubit gate — collect all target qubits
            resolved = [self._resolve_qubit(t) for t in node.targets]
            reg = resolved[0][0]
            qubits = [q for _, q in resolved]
            reg.state.apply_gate(matrix, qubits)

    def _resolve_qubit(self, node: ASTNode) -> tuple[QubitRegister, int]:
        """Resolve a qubit reference to (register, index)."""
        if isinstance(node, IndexAccess):
            reg = self._eval(node.target)
            idx = self._eval(node.index)
            if not isinstance(reg, QubitRegister):
                raise InterpreterError("Expected qubit register", node)
            return reg, idx
        elif isinstance(node, Identifier):
            reg = self.env.get(node.name)
            if isinstance(reg, QubitRegister):
                return reg, 0
            raise InterpreterError(f"'{node.name}' is not a qubit register", node)
        raise InterpreterError("Invalid qubit reference", node)

    def _exec_GateDefNode(self, node: Any) -> None:
        self.env.set(node.name, node)

    # ── Quantum operations ───────────────────────────────────────────────

    def _exec_MeasurementNode(self, node: MeasurementNode) -> int:
        reg, qubit_idx = self._resolve_qubit(node.target)
        result = reg.state.measure_qubit(qubit_idx)
        reg.classical[qubit_idx] = result
        record = {"qubit": f"{reg.name}[{qubit_idx}]", "result": result}
        self.measurements.append(record)
        if node.classical_target is not None:
            if isinstance(node.classical_target, Identifier):
                self.env.set(node.classical_target.name, result)
            elif isinstance(node.classical_target, IndexAccess):
                target = self._eval(node.classical_target.target)
                index = self._eval(node.classical_target.index)
                target[index] = result
        return result

    def _exec_EntanglementNode(self, node: EntanglementNode) -> None:
        """Create Bell pair: H on first, CNOT on first->second."""
        if len(node.targets) < 2:
            raise InterpreterError("Entangle requires at least 2 qubits", node)
        reg0, q0 = self._resolve_qubit(node.targets[0])
        h_gate = standard_gates.get_gate("H")
        reg0.state.apply_gate(h_gate.matrix(), [q0])

        for target_node in node.targets[1:]:
            reg1, q1 = self._resolve_qubit(target_node)
            if reg1 is not reg0:
                raise InterpreterError("Entangle requires qubits from the same register", node)
            cnot = standard_gates.get_gate("CNOT")
            reg0.state.apply_gate(cnot.matrix(), [q0, q1])

    def _exec_SuperpositionNode(self, node: SuperpositionNode) -> None:
        """Apply Hadamard to put qubit in superposition."""
        reg, qubit_idx = self._resolve_qubit(node.target)
        h_gate = standard_gates.get_gate("H")
        reg.state.apply_gate(h_gate.matrix(), [qubit_idx])

    def _exec_TeleportationNode(self, node: TeleportationNode) -> None:
        """Quantum teleportation protocol."""
        src_reg, src_q = self._resolve_qubit(node.source)
        dst_reg, dst_q = self._resolve_qubit(node.dest)
        if src_reg is not dst_reg:
            raise InterpreterError("Teleportation requires qubits in the same register", node)
        reg = src_reg

        if node.channel is not None:
            _, ch_q = self._resolve_qubit(node.channel)
        else:
            ch_q = max(src_q, dst_q) + 1
            if ch_q >= reg.size:
                raise InterpreterError("No channel qubit available for teleportation", node)

        h = standard_gates.get_gate("H").matrix()
        cnot = standard_gates.get_gate("CNOT").matrix()
        x = standard_gates.get_gate("X").matrix()
        z = standard_gates.get_gate("Z").matrix()

        # Create Bell pair between channel and dest
        reg.state.apply_gate(h, [ch_q])
        reg.state.apply_gate(cnot, [ch_q, dst_q])
        # Bell measurement on source and channel
        reg.state.apply_gate(cnot, [src_q, ch_q])
        reg.state.apply_gate(h, [src_q])
        m1 = reg.state.measure_qubit(src_q)
        m2 = reg.state.measure_qubit(ch_q)
        # Corrections
        if m2 == 1:
            reg.state.apply_gate(x, [dst_q])
        if m1 == 1:
            reg.state.apply_gate(z, [dst_q])

    def _exec_BarrierNode(self, node: BarrierNode) -> None:
        pass  # Barriers are synchronization hints — no-op in interpreter

    def _exec_ResetNode(self, node: ResetNode) -> None:
        reg, qubit_idx = self._resolve_qubit(node.target)
        result = reg.state.measure_qubit(qubit_idx)
        if result == 1:
            x_gate = standard_gates.get_gate("X")
            reg.state.apply_gate(x_gate.matrix(), [qubit_idx])

    # ── Algorithm stubs ──────────────────────────────────────────────────

    def _exec_GroversNode(self, node: GroversNode) -> None:
        raise InterpreterError("Grover's algorithm requires a backend", node)

    def _exec_ShorsNode(self, node: Any) -> None:
        raise InterpreterError("Shor's algorithm requires a backend", node)

    def _exec_VQENode(self, node: Any) -> None:
        raise InterpreterError("VQE requires a backend", node)

    def _exec_QAOANode(self, node: Any) -> None:
        raise InterpreterError("QAOA requires a backend", node)

    def _exec_QFTNode(self, node: QFTNode) -> None:
        """Apply quantum Fourier transform."""
        resolved = [self._resolve_qubit(t) for t in node.targets]
        if len(resolved) == 0:
            return
        reg = resolved[0][0]
        qubits = [q for _, q in resolved]
        n = len(qubits)
        h = standard_gates.get_gate("H").matrix()

        for i in range(n):
            reg.state.apply_gate(h, [qubits[i]])
            for j in range(i + 1, n):
                k = j - i + 1
                angle = math.pi / (2 ** (k - 1))
                rz = standard_gates.get_gate("RZ", angle).matrix()
                reg.state.apply_gate(rz, [qubits[j]])

    def _exec_QPENode(self, node: Any) -> None:
        raise InterpreterError("QPE requires a backend", node)

    # ── Error correction stubs ───────────────────────────────────────────

    def _exec_SyndromeNode(self, node: Any) -> None:
        raise InterpreterError("Syndrome extraction requires a backend", node)

    def _exec_CorrectNode(self, node: Any) -> None:
        raise InterpreterError("Error correction requires a backend", node)

    def _exec_StabilizerNode(self, node: Any) -> None:
        raise InterpreterError("Stabilizer simulation not yet implemented", node)

    # ── Control flow ─────────────────────────────────────────────────────

    def _exec_IfNode(self, node: IfNode) -> None:
        condition = self._eval(node.condition)
        if condition:
            for stmt in node.body:
                self._exec(stmt)
        elif node.else_body:
            for stmt in node.else_body:
                self._exec(stmt)

    def _exec_WhileNode(self, node: WhileNode) -> None:
        iterations = 0
        max_iterations = 10_000
        while self._eval(node.condition):
            iterations += 1
            if iterations > max_iterations:
                raise InterpreterError("While loop exceeded max iterations", node)
            try:
                for stmt in node.body:
                    self._exec(stmt)
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def _exec_ForNode(self, node: ForNode) -> None:
        iterable = self._eval(node.iterable)
        for item in iterable:
            self.env.set(node.variable, item)
            try:
                for stmt in node.body:
                    self._exec(stmt)
            except BreakSignal:
                break
            except ContinueSignal:
                continue

    def _exec_FunctionDefNode(self, node: FunctionDefNode) -> None:
        def user_fn(*args: Any) -> Any:
            child_env = self.env.child()
            old_env = self.env
            self.env = child_env
            for pname, arg in zip(node.params, args):
                self.env.set(pname, arg)
            result = None
            try:
                for stmt in node.body:
                    self._exec(stmt)
            except ReturnSignal as r:
                result = r.value
            finally:
                self.env = old_env
            return result

        self.env.set(node.name, user_fn)

    def _exec_ReturnNode(self, node: ReturnNode) -> None:
        value = self._eval(node.value) if node.value else None
        raise ReturnSignal(value)

    def _exec_BreakNode(self, node: BreakNode) -> None:
        raise BreakSignal()

    def _exec_ContinueNode(self, node: ContinueNode) -> None:
        raise ContinueSignal()

    # ── Expressions ──────────────────────────────────────────────────────

    def _exec_IntegerLiteral(self, node: IntegerLiteral) -> int:
        return node.value

    def _exec_FloatLiteral(self, node: FloatLiteral) -> float:
        return node.value

    def _exec_ComplexLiteral(self, node: ComplexLiteral) -> complex:
        return complex(0, node.value)

    def _exec_StringLiteral(self, node: StringLiteral) -> str:
        return node.value

    def _exec_BooleanLiteral(self, node: BooleanLiteral) -> bool:
        return node.value

    def _exec_Identifier(self, node: Identifier) -> Any:
        try:
            return self.env.get(node.name)
        except KeyError:
            raise InterpreterError(f"Undefined variable: '{node.name}'", node)

    def _exec_BinaryOp(self, node: BinaryOp) -> Any:
        left = self._eval(node.left)
        # Short-circuit for logical ops
        if node.op == "and":
            return left and self._eval(node.right)
        if node.op == "or":
            return left or self._eval(node.right)

        right = self._eval(node.right)
        ops: dict[str, Any] = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b,
            "%": lambda a, b: a % b,
            "^": lambda a, b: a ** b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            "<": lambda a, b: a < b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            ">=": lambda a, b: a >= b,
        }
        fn = ops.get(node.op)
        if fn is None:
            raise InterpreterError(f"Unknown operator: {node.op}", node)
        return fn(left, right)

    def _exec_UnaryOp(self, node: UnaryOp) -> Any:
        operand = self._eval(node.operand)
        if node.op == "-":
            return -operand
        if node.op == "not":
            return not operand
        raise InterpreterError(f"Unknown unary operator: {node.op}", node)

    def _exec_FunctionCall(self, node: FunctionCall) -> Any:
        fn = self.env.get(node.name)
        args = [self._eval(a) for a in node.args]
        return fn(*args)

    def _exec_IndexAccess(self, node: IndexAccess) -> Any:
        target = self._eval(node.target)
        index = self._eval(node.index)
        if isinstance(target, QubitRegister):
            return (target, index)  # Return tuple for qubit reference
        return target[index]

    def _exec_MemberAccess(self, node: MemberAccess) -> Any:
        target = self._eval(node.target)
        if isinstance(target, QubitRegister):
            if node.member == "size":
                return target.size
            if node.member == "state":
                return target.state
            if node.member == "probabilities":
                return target.state.probabilities()
        return getattr(target, node.member)

    def _exec_ListNode(self, node: ListNode) -> list:
        return [self._eval(el) for el in node.elements]

    # ── Dirac notation ───────────────────────────────────────────────────

    def _exec_KetStateNode(self, node: KetStateNode) -> StateVector:
        return StateVector.from_label(node.label)

    def _exec_BraStateNode(self, node: BraStateNode) -> np.ndarray:
        sv = StateVector.from_label(node.label)
        return sv.data.conj()

    def _exec_BraKetNode(self, node: BraKetNode) -> complex:
        bra = StateVector.from_label(node.bra_label)
        ket = StateVector.from_label(node.ket_label)
        return complex(np.vdot(bra.data, ket.data))

    def _exec_TensorProductNode(self, node: TensorProductNode) -> Any:
        left = self._eval(node.left)
        right = self._eval(node.right)
        if isinstance(left, StateVector) and isinstance(right, StateVector):
            combined = np.kron(left.data, right.data)
            n_qubits = left.num_qubits + right.num_qubits
            return StateVector(n_qubits, combined)
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            return np.kron(left, right)
        raise InterpreterError("Tensor product requires quantum states or arrays", node)

    def _exec_DaggerNode(self, node: DaggerNode) -> Any:
        operand = self._eval(node.operand)
        if isinstance(operand, np.ndarray):
            return operand.conj().T
        raise InterpreterError("Dagger requires a matrix or state", node)

    def _exec_RangeNode(self, node: Any) -> range:
        start = self._eval(node.start) if node.start else 0
        stop = self._eval(node.stop) if node.stop else 0
        step = self._eval(node.step) if node.step else 1
        return range(start, stop, step)
