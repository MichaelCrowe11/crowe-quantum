"""QubitFlow AST nodes — abstract syntax tree for the quantum circuit language."""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Base ─────────────────────────────────────────────────────────────────

@dataclass
class ASTNode:
    line: int = 0
    column: int = 0


# ── Program ──────────────────────────────────────────────────────────────

@dataclass
class ProgramNode(ASTNode):
    statements: list[ASTNode] = field(default_factory=list)


# ── Expressions ──────────────────────────────────────────────────────────

@dataclass
class IntegerLiteral(ASTNode):
    value: int = 0


@dataclass
class FloatLiteral(ASTNode):
    value: float = 0.0


@dataclass
class ComplexLiteral(ASTNode):
    """Complex imaginary component, e.g. 2i → ComplexLiteral(2.0)."""
    value: float = 0.0


@dataclass
class StringLiteral(ASTNode):
    value: str = ""


@dataclass
class BooleanLiteral(ASTNode):
    value: bool = False


@dataclass
class Identifier(ASTNode):
    name: str = ""


@dataclass
class BinaryOp(ASTNode):
    op: str = ""
    left: ASTNode = field(default_factory=ASTNode)
    right: ASTNode = field(default_factory=ASTNode)


@dataclass
class UnaryOp(ASTNode):
    op: str = ""
    operand: ASTNode = field(default_factory=ASTNode)


@dataclass
class FunctionCall(ASTNode):
    name: str = ""
    args: list[ASTNode] = field(default_factory=list)


@dataclass
class IndexAccess(ASTNode):
    """Array/register index: qubits[0]."""
    target: ASTNode = field(default_factory=ASTNode)
    index: ASTNode = field(default_factory=ASTNode)


@dataclass
class MemberAccess(ASTNode):
    """Dot access: circuit.depth."""
    target: ASTNode = field(default_factory=ASTNode)
    member: str = ""


# ── Dirac Notation ───────────────────────────────────────────────────────

@dataclass
class KetStateNode(ASTNode):
    """Ket state |label⟩ — e.g. |0⟩, |+⟩, |ψ⟩."""
    label: str = ""


@dataclass
class BraStateNode(ASTNode):
    """Bra state ⟨label| — e.g. ⟨0|, ⟨+|."""
    label: str = ""


@dataclass
class BraKetNode(ASTNode):
    """Inner product ⟨a|b⟩."""
    bra_label: str = ""
    ket_label: str = ""


@dataclass
class TensorProductNode(ASTNode):
    """Tensor product a ⊗ b."""
    left: ASTNode = field(default_factory=ASTNode)
    right: ASTNode = field(default_factory=ASTNode)


@dataclass
class DaggerNode(ASTNode):
    """Hermitian adjoint A†."""
    operand: ASTNode = field(default_factory=ASTNode)


# ── Quantum Declarations ────────────────────────────────────────────────

@dataclass
class QubitDeclNode(ASTNode):
    """qubit q or qubit q[4]."""
    name: str = ""
    size: ASTNode | None = None


@dataclass
class QuditDeclNode(ASTNode):
    """qudit d[3] — d-dimensional quantum system."""
    name: str = ""
    dimension: ASTNode = field(default_factory=ASTNode)
    size: ASTNode | None = None


# ── Circuit ──────────────────────────────────────────────────────────────

@dataclass
class CircuitNode(ASTNode):
    """circuit name(params): body."""
    name: str = ""
    params: list[str] = field(default_factory=list)
    body: list[ASTNode] = field(default_factory=list)


# ── Gate Operations ──────────────────────────────────────────────────────

@dataclass
class GateApplicationNode(ASTNode):
    """Apply a gate: H q[0] or CNOT q[0], q[1] or RX(pi/4) q[0]."""
    gate_name: str = ""
    targets: list[ASTNode] = field(default_factory=list)
    params: list[ASTNode] = field(default_factory=list)


@dataclass
class GateDefNode(ASTNode):
    """Custom gate definition: gate my_gate(theta) q0, q1: body."""
    name: str = ""
    param_names: list[str] = field(default_factory=list)
    qubit_names: list[str] = field(default_factory=list)
    body: list[ASTNode] = field(default_factory=list)


# ── Quantum Operations ──────────────────────────────────────────────────

@dataclass
class MeasurementNode(ASTNode):
    """measure q[0] or measure q[0] -> c[0]."""
    target: ASTNode = field(default_factory=ASTNode)
    classical_target: ASTNode | None = None


@dataclass
class EntanglementNode(ASTNode):
    """entangle q[0], q[1] — create Bell pair."""
    targets: list[ASTNode] = field(default_factory=list)


@dataclass
class SuperpositionNode(ASTNode):
    """superpose q[0] — put into equal superposition."""
    target: ASTNode = field(default_factory=ASTNode)


@dataclass
class TeleportationNode(ASTNode):
    """teleport source -> dest using channel."""
    source: ASTNode = field(default_factory=ASTNode)
    dest: ASTNode = field(default_factory=ASTNode)
    channel: ASTNode | None = None


@dataclass
class BarrierNode(ASTNode):
    """barrier q[0], q[1] — synchronization point."""
    targets: list[ASTNode] = field(default_factory=list)


@dataclass
class ResetNode(ASTNode):
    """reset q[0] — reset to |0⟩."""
    target: ASTNode = field(default_factory=ASTNode)


# ── Algorithm Nodes ──────────────────────────────────────────────────────

@dataclass
class GroversNode(ASTNode):
    """grovers(oracle, n_qubits, iterations)."""
    oracle: ASTNode = field(default_factory=ASTNode)
    n_qubits: ASTNode = field(default_factory=ASTNode)
    iterations: ASTNode | None = None


@dataclass
class ShorsNode(ASTNode):
    """shors(N) — factor integer N."""
    number: ASTNode = field(default_factory=ASTNode)


@dataclass
class VQENode(ASTNode):
    """vqe(hamiltonian, ansatz, optimizer)."""
    hamiltonian: ASTNode = field(default_factory=ASTNode)
    ansatz: ASTNode | None = None
    optimizer: ASTNode | None = None


@dataclass
class QAOANode(ASTNode):
    """qaoa(cost, mixer, depth)."""
    cost: ASTNode = field(default_factory=ASTNode)
    mixer: ASTNode | None = None
    depth: ASTNode | None = None


@dataclass
class QFTNode(ASTNode):
    """qft q[0:3] — quantum Fourier transform."""
    targets: list[ASTNode] = field(default_factory=list)


@dataclass
class QPENode(ASTNode):
    """qpe(unitary, precision_qubits)."""
    unitary: ASTNode = field(default_factory=ASTNode)
    precision: ASTNode = field(default_factory=ASTNode)


# ── Error Correction ─────────────────────────────────────────────────────

@dataclass
class SyndromeNode(ASTNode):
    """syndrome q_data, q_ancilla."""
    data_qubits: list[ASTNode] = field(default_factory=list)
    ancilla_qubits: list[ASTNode] = field(default_factory=list)


@dataclass
class CorrectNode(ASTNode):
    """correct q_data based on syndrome_result."""
    data_qubits: list[ASTNode] = field(default_factory=list)
    syndrome: ASTNode = field(default_factory=ASTNode)


@dataclass
class StabilizerNode(ASTNode):
    """stabilizer [X, Z, Z, X] on q[0:3]."""
    operators: list[ASTNode] = field(default_factory=list)
    targets: list[ASTNode] = field(default_factory=list)


# ── Control Flow ─────────────────────────────────────────────────────────

@dataclass
class IfNode(ASTNode):
    condition: ASTNode = field(default_factory=ASTNode)
    body: list[ASTNode] = field(default_factory=list)
    else_body: list[ASTNode] = field(default_factory=list)


@dataclass
class WhileNode(ASTNode):
    condition: ASTNode = field(default_factory=ASTNode)
    body: list[ASTNode] = field(default_factory=list)


@dataclass
class ForNode(ASTNode):
    variable: str = ""
    iterable: ASTNode = field(default_factory=ASTNode)
    body: list[ASTNode] = field(default_factory=list)


@dataclass
class FunctionDefNode(ASTNode):
    name: str = ""
    params: list[str] = field(default_factory=list)
    body: list[ASTNode] = field(default_factory=list)
    return_type: str | None = None


@dataclass
class ReturnNode(ASTNode):
    value: ASTNode | None = None


@dataclass
class BreakNode(ASTNode):
    pass


@dataclass
class ContinueNode(ASTNode):
    pass


# ── Statements ───────────────────────────────────────────────────────────

@dataclass
class AssignmentNode(ASTNode):
    target: ASTNode = field(default_factory=ASTNode)
    value: ASTNode = field(default_factory=ASTNode)


@dataclass
class ExpressionStatement(ASTNode):
    """Bare expression used as a statement."""
    expression: ASTNode = field(default_factory=ASTNode)


@dataclass
class RangeNode(ASTNode):
    """Range expression for slicing: start:stop or start:stop:step."""
    start: ASTNode | None = None
    stop: ASTNode | None = None
    step: ASTNode | None = None


@dataclass
class ListNode(ASTNode):
    """List literal [a, b, c]."""
    elements: list[ASTNode] = field(default_factory=list)
