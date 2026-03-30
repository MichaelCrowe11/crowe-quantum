"""QubitFlow parser — recursive-descent parser for the quantum circuit language."""

from __future__ import annotations

from crowe_qubit_flow.ast_nodes import (
    AssignmentNode,
    ASTNode,
    BarrierNode,
    BinaryOp,
    BooleanLiteral,
    BraKetNode,
    BraStateNode,
    BreakNode,
    CircuitNode,
    ComplexLiteral,
    ContinueNode,
    CorrectNode,
    DaggerNode,
    EntanglementNode,
    ExpressionStatement,
    FloatLiteral,
    ForNode,
    FunctionCall,
    FunctionDefNode,
    GateApplicationNode,
    GateDefNode,
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
    QAOANode,
    QFTNode,
    QPENode,
    QubitDeclNode,
    QuditDeclNode,
    ResetNode,
    ReturnNode,
    ShorsNode,
    StabilizerNode,
    StringLiteral,
    SuperpositionNode,
    SyndromeNode,
    TeleportationNode,
    TensorProductNode,
    UnaryOp,
    VQENode,
    WhileNode,
)
from crowe_qubit_flow.lexer import GATE_TOKEN_TO_NAME, Lexer, Token, TokenType


class ParseError(Exception):
    def __init__(self, message: str, token: Token) -> None:
        super().__init__(f"Line {token.line}, Col {token.column}: {message}")
        self.token = token


class Parser:
    """Recursive-descent parser for QubitFlow."""

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    @classmethod
    def from_source(cls, source: str) -> Parser:
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        return cls(tokens)

    def parse(self) -> ProgramNode:
        self._skip_newlines()
        stmts: list[ASTNode] = []
        while not self._check(TokenType.EOF):
            stmt = self._parse_statement()
            if stmt is not None:
                stmts.append(stmt)
            self._skip_newlines()
        return ProgramNode(statements=stmts, line=1, column=1)

    # ── Token helpers ────────────────────────────────────────────────────

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _check(self, *types: TokenType) -> bool:
        return self._current().type in types

    def _match(self, *types: TokenType) -> Token | None:
        if self._current().type in types:
            tok = self._current()
            self.pos += 1
            return tok
        return None

    def _expect(self, token_type: TokenType, message: str = "") -> Token:
        tok = self._match(token_type)
        if tok is None:
            msg = message or f"Expected {token_type.name}, got {self._current().type.name}"
            raise ParseError(msg, self._current())
        return tok

    def _skip_newlines(self) -> None:
        while self._check(TokenType.NEWLINE):
            self.pos += 1

    # ── Statement parsing ────────────────────────────────────────────────

    def _parse_statement(self) -> ASTNode | None:
        tok = self._current()
        tt = tok.type

        if tt == TokenType.QUBIT:
            return self._parse_qubit_decl()
        if tt == TokenType.QUDIT:
            return self._parse_qudit_decl()
        if tt == TokenType.CIRCUIT:
            return self._parse_circuit()
        if tt == TokenType.GATE:
            return self._parse_gate_def()
        if tt == TokenType.MEASURE:
            return self._parse_measure()
        if tt == TokenType.ENTANGLE:
            return self._parse_entangle()
        if tt == TokenType.SUPERPOSE:
            return self._parse_superpose()
        if tt == TokenType.TELEPORT:
            return self._parse_teleport()
        if tt == TokenType.BARRIER:
            return self._parse_barrier()
        if tt == TokenType.RESET:
            return self._parse_reset()
        if tt == TokenType.IF:
            return self._parse_if()
        if tt == TokenType.WHILE:
            return self._parse_while()
        if tt == TokenType.FOR:
            return self._parse_for()
        if tt == TokenType.DEF:
            return self._parse_function_def()
        if tt == TokenType.RETURN:
            return self._parse_return()
        if tt == TokenType.BREAK:
            self.pos += 1
            return BreakNode(line=tok.line, column=tok.column)
        if tt == TokenType.CONTINUE:
            self.pos += 1
            return ContinueNode(line=tok.line, column=tok.column)
        if tt in GATE_TOKEN_TO_NAME:
            return self._parse_gate_application()
        if tt == TokenType.GROVERS:
            return self._parse_grovers()
        if tt == TokenType.SHORS:
            return self._parse_shors()
        if tt == TokenType.VQE:
            return self._parse_vqe()
        if tt == TokenType.QAOA:
            return self._parse_qaoa()
        if tt == TokenType.QFT:
            return self._parse_qft()
        if tt == TokenType.QPE:
            return self._parse_qpe()
        if tt == TokenType.SYNDROME:
            return self._parse_syndrome()
        if tt == TokenType.CORRECT:
            return self._parse_correct()
        if tt == TokenType.STABILIZER:
            return self._parse_stabilizer()

        return self._parse_expression_or_assignment()

    # ── Quantum declarations ─────────────────────────────────────────────

    def _parse_qubit_decl(self) -> QubitDeclNode:
        tok = self._expect(TokenType.QUBIT)
        name_tok = self._expect(TokenType.IDENTIFIER, "Expected qubit name")
        size = None
        if self._match(TokenType.LBRACKET):
            size = self._parse_expression()
            self._expect(TokenType.RBRACKET, "Expected ']'")
        return QubitDeclNode(name=name_tok.value, size=size, line=tok.line, column=tok.column)

    def _parse_qudit_decl(self) -> QuditDeclNode:
        tok = self._expect(TokenType.QUDIT)
        name_tok = self._expect(TokenType.IDENTIFIER, "Expected qudit name")
        self._expect(TokenType.LBRACKET, "Expected '[' for qudit dimension")
        dim = self._parse_expression()
        self._expect(TokenType.RBRACKET, "Expected ']'")
        size = None
        if self._match(TokenType.LBRACKET):
            size = self._parse_expression()
            self._expect(TokenType.RBRACKET, "Expected ']'")
        return QuditDeclNode(name=name_tok.value, dimension=dim, size=size, line=tok.line, column=tok.column)

    # ── Circuit / gate defs ──────────────────────────────────────────────

    def _parse_circuit(self) -> CircuitNode:
        tok = self._expect(TokenType.CIRCUIT)
        name_tok = self._expect(TokenType.IDENTIFIER, "Expected circuit name")
        params: list[str] = []
        if self._match(TokenType.LPAREN):
            if not self._check(TokenType.RPAREN):
                params.append(self._expect(TokenType.IDENTIFIER).value)
                while self._match(TokenType.COMMA):
                    params.append(self._expect(TokenType.IDENTIFIER).value)
            self._expect(TokenType.RPAREN, "Expected ')'")
        self._expect(TokenType.COLON, "Expected ':'")
        body = self._parse_block()
        return CircuitNode(name=name_tok.value, params=params, body=body, line=tok.line, column=tok.column)

    def _parse_gate_def(self) -> GateDefNode:
        tok = self._expect(TokenType.GATE)
        name_tok = self._expect(TokenType.IDENTIFIER, "Expected gate name")
        param_names: list[str] = []
        if self._match(TokenType.LPAREN):
            if not self._check(TokenType.RPAREN):
                param_names.append(self._expect(TokenType.IDENTIFIER).value)
                while self._match(TokenType.COMMA):
                    param_names.append(self._expect(TokenType.IDENTIFIER).value)
            self._expect(TokenType.RPAREN, "Expected ')'")
        qubit_names: list[str] = []
        while self._check(TokenType.IDENTIFIER):
            qubit_names.append(self._expect(TokenType.IDENTIFIER).value)
            if not self._match(TokenType.COMMA):
                break
        self._expect(TokenType.COLON, "Expected ':'")
        body = self._parse_block()
        return GateDefNode(
            name=name_tok.value, param_names=param_names,
            qubit_names=qubit_names, body=body,
            line=tok.line, column=tok.column,
        )

    def _parse_gate_application(self) -> GateApplicationNode:
        tok = self._current()
        gate_name = GATE_TOKEN_TO_NAME[tok.type]
        self.pos += 1

        params: list[ASTNode] = []
        if self._match(TokenType.LPAREN):
            if not self._check(TokenType.RPAREN):
                params.append(self._parse_expression())
                while self._match(TokenType.COMMA):
                    params.append(self._parse_expression())
            self._expect(TokenType.RPAREN, "Expected ')'")

        targets: list[ASTNode] = []
        targets.append(self._parse_postfix())
        while self._match(TokenType.COMMA):
            targets.append(self._parse_postfix())

        return GateApplicationNode(
            gate_name=gate_name, targets=targets, params=params,
            line=tok.line, column=tok.column,
        )

    # ── Quantum operations ───────────────────────────────────────────────

    def _parse_measure(self) -> MeasurementNode:
        tok = self._expect(TokenType.MEASURE)
        target = self._parse_postfix()
        classical = None
        if self._match(TokenType.ARROW):
            classical = self._parse_postfix()
        return MeasurementNode(target=target, classical_target=classical, line=tok.line, column=tok.column)

    def _parse_entangle(self) -> EntanglementNode:
        tok = self._expect(TokenType.ENTANGLE)
        targets = [self._parse_postfix()]
        while self._match(TokenType.COMMA):
            targets.append(self._parse_postfix())
        return EntanglementNode(targets=targets, line=tok.line, column=tok.column)

    def _parse_superpose(self) -> SuperpositionNode:
        tok = self._expect(TokenType.SUPERPOSE)
        target = self._parse_postfix()
        return SuperpositionNode(target=target, line=tok.line, column=tok.column)

    def _parse_teleport(self) -> TeleportationNode:
        tok = self._expect(TokenType.TELEPORT)
        source = self._parse_postfix()
        self._expect(TokenType.ARROW, "Expected '->'")
        dest = self._parse_postfix()
        channel = None
        if self._check(TokenType.COMMA):
            self.pos += 1
            channel = self._parse_postfix()
        return TeleportationNode(source=source, dest=dest, channel=channel, line=tok.line, column=tok.column)

    def _parse_barrier(self) -> BarrierNode:
        tok = self._expect(TokenType.BARRIER)
        targets: list[ASTNode] = []
        if not self._check(TokenType.NEWLINE, TokenType.EOF):
            targets.append(self._parse_postfix())
            while self._match(TokenType.COMMA):
                targets.append(self._parse_postfix())
        return BarrierNode(targets=targets, line=tok.line, column=tok.column)

    def _parse_reset(self) -> ResetNode:
        tok = self._expect(TokenType.RESET)
        target = self._parse_postfix()
        return ResetNode(target=target, line=tok.line, column=tok.column)

    # ── Algorithm nodes ──────────────────────────────────────────────────

    def _parse_grovers(self) -> GroversNode:
        tok = self._expect(TokenType.GROVERS)
        self._expect(TokenType.LPAREN, "Expected '('")
        oracle = self._parse_expression()
        self._expect(TokenType.COMMA, "Expected ','")
        n_qubits = self._parse_expression()
        iterations = None
        if self._match(TokenType.COMMA):
            iterations = self._parse_expression()
        self._expect(TokenType.RPAREN, "Expected ')'")
        return GroversNode(oracle=oracle, n_qubits=n_qubits, iterations=iterations, line=tok.line, column=tok.column)

    def _parse_shors(self) -> ShorsNode:
        tok = self._expect(TokenType.SHORS)
        self._expect(TokenType.LPAREN, "Expected '('")
        number = self._parse_expression()
        self._expect(TokenType.RPAREN, "Expected ')'")
        return ShorsNode(number=number, line=tok.line, column=tok.column)

    def _parse_vqe(self) -> VQENode:
        tok = self._expect(TokenType.VQE)
        self._expect(TokenType.LPAREN, "Expected '('")
        hamiltonian = self._parse_expression()
        ansatz = None
        optimizer = None
        if self._match(TokenType.COMMA):
            ansatz = self._parse_expression()
            if self._match(TokenType.COMMA):
                optimizer = self._parse_expression()
        self._expect(TokenType.RPAREN, "Expected ')'")
        return VQENode(hamiltonian=hamiltonian, ansatz=ansatz, optimizer=optimizer, line=tok.line, column=tok.column)

    def _parse_qaoa(self) -> QAOANode:
        tok = self._expect(TokenType.QAOA)
        self._expect(TokenType.LPAREN, "Expected '('")
        cost = self._parse_expression()
        mixer = None
        depth = None
        if self._match(TokenType.COMMA):
            mixer = self._parse_expression()
            if self._match(TokenType.COMMA):
                depth = self._parse_expression()
        self._expect(TokenType.RPAREN, "Expected ')'")
        return QAOANode(cost=cost, mixer=mixer, depth=depth, line=tok.line, column=tok.column)

    def _parse_qft(self) -> QFTNode:
        tok = self._expect(TokenType.QFT)
        targets = [self._parse_postfix()]
        while self._match(TokenType.COMMA):
            targets.append(self._parse_postfix())
        return QFTNode(targets=targets, line=tok.line, column=tok.column)

    def _parse_qpe(self) -> QPENode:
        tok = self._expect(TokenType.QPE)
        self._expect(TokenType.LPAREN, "Expected '('")
        unitary = self._parse_expression()
        self._expect(TokenType.COMMA, "Expected ','")
        precision = self._parse_expression()
        self._expect(TokenType.RPAREN, "Expected ')'")
        return QPENode(unitary=unitary, precision=precision, line=tok.line, column=tok.column)

    # ── Error correction ─────────────────────────────────────────────────

    def _parse_syndrome(self) -> SyndromeNode:
        tok = self._expect(TokenType.SYNDROME)
        data = [self._parse_postfix()]
        while self._match(TokenType.COMMA):
            data.append(self._parse_postfix())
        ancilla: list[ASTNode] = []
        if self._match(TokenType.ARROW):
            ancilla.append(self._parse_postfix())
            while self._match(TokenType.COMMA):
                ancilla.append(self._parse_postfix())
        return SyndromeNode(data_qubits=data, ancilla_qubits=ancilla, line=tok.line, column=tok.column)

    def _parse_correct(self) -> CorrectNode:
        tok = self._expect(TokenType.CORRECT)
        data = [self._parse_postfix()]
        while self._match(TokenType.COMMA):
            data.append(self._parse_postfix())
        self._expect(TokenType.ARROW, "Expected '->' before syndrome")
        syndrome = self._parse_expression()
        return CorrectNode(data_qubits=data, syndrome=syndrome, line=tok.line, column=tok.column)

    def _parse_stabilizer(self) -> StabilizerNode:
        tok = self._expect(TokenType.STABILIZER)
        self._expect(TokenType.LBRACKET, "Expected '['")
        operators: list[ASTNode] = []
        if not self._check(TokenType.RBRACKET):
            operators.append(self._parse_expression())
            while self._match(TokenType.COMMA):
                operators.append(self._parse_expression())
        self._expect(TokenType.RBRACKET, "Expected ']'")
        targets: list[ASTNode] = []
        if not self._check(TokenType.NEWLINE, TokenType.EOF):
            targets.append(self._parse_postfix())
            while self._match(TokenType.COMMA):
                targets.append(self._parse_postfix())
        return StabilizerNode(operators=operators, targets=targets, line=tok.line, column=tok.column)

    # ── Control flow ─────────────────────────────────────────────────────

    def _parse_if(self) -> IfNode:
        tok = self._expect(TokenType.IF)
        cond = self._parse_expression()
        self._expect(TokenType.COLON, "Expected ':'")
        body = self._parse_block()
        else_body: list[ASTNode] = []
        self._skip_newlines()
        if self._match(TokenType.ELSE):
            self._expect(TokenType.COLON, "Expected ':'")
            else_body = self._parse_block()
        return IfNode(condition=cond, body=body, else_body=else_body, line=tok.line, column=tok.column)

    def _parse_while(self) -> WhileNode:
        tok = self._expect(TokenType.WHILE)
        cond = self._parse_expression()
        self._expect(TokenType.COLON, "Expected ':'")
        body = self._parse_block()
        return WhileNode(condition=cond, body=body, line=tok.line, column=tok.column)

    def _parse_for(self) -> ForNode:
        tok = self._expect(TokenType.FOR)
        var = self._expect(TokenType.IDENTIFIER, "Expected loop variable")
        self._expect(TokenType.IN, "Expected 'in'")
        iterable = self._parse_expression()
        self._expect(TokenType.COLON, "Expected ':'")
        body = self._parse_block()
        return ForNode(variable=var.value, iterable=iterable, body=body, line=tok.line, column=tok.column)

    def _parse_function_def(self) -> FunctionDefNode:
        tok = self._expect(TokenType.DEF)
        name = self._expect(TokenType.IDENTIFIER, "Expected function name")
        self._expect(TokenType.LPAREN, "Expected '('")
        params: list[str] = []
        if not self._check(TokenType.RPAREN):
            params.append(self._expect(TokenType.IDENTIFIER).value)
            while self._match(TokenType.COMMA):
                params.append(self._expect(TokenType.IDENTIFIER).value)
        self._expect(TokenType.RPAREN, "Expected ')'")
        ret_type = None
        if self._match(TokenType.ARROW):
            ret_type = self._expect(TokenType.IDENTIFIER, "Expected return type").value
        self._expect(TokenType.COLON, "Expected ':'")
        body = self._parse_block()
        return FunctionDefNode(
            name=name.value, params=params, body=body,
            return_type=ret_type, line=tok.line, column=tok.column,
        )

    def _parse_return(self) -> ReturnNode:
        tok = self._expect(TokenType.RETURN)
        value = None
        if not self._check(TokenType.NEWLINE, TokenType.EOF, TokenType.DEDENT):
            value = self._parse_expression()
        return ReturnNode(value=value, line=tok.line, column=tok.column)

    # ── Block parsing (indentation-based) ────────────────────────────────

    def _parse_block(self) -> list[ASTNode]:
        self._skip_newlines()
        self._expect(TokenType.INDENT, "Expected indented block")
        stmts: list[ASTNode] = []
        self._skip_newlines()
        while not self._check(TokenType.DEDENT, TokenType.EOF):
            stmt = self._parse_statement()
            if stmt is not None:
                stmts.append(stmt)
            self._skip_newlines()
        self._match(TokenType.DEDENT)
        return stmts

    # ── Expression / assignment ──────────────────────────────────────────

    def _parse_expression_or_assignment(self) -> ASTNode:
        expr = self._parse_expression()
        if self._match(TokenType.EQUALS):
            value = self._parse_expression()
            return AssignmentNode(target=expr, value=value, line=expr.line, column=expr.column)
        return ExpressionStatement(expression=expr, line=expr.line, column=expr.column)

    # ── Expression precedence climbing ───────────────────────────────────

    def _parse_expression(self) -> ASTNode:
        return self._parse_or()

    def _parse_or(self) -> ASTNode:
        left = self._parse_and()
        while self._match(TokenType.OR):
            right = self._parse_and()
            left = BinaryOp(op="or", left=left, right=right, line=left.line, column=left.column)
        return left

    def _parse_and(self) -> ASTNode:
        left = self._parse_not()
        while self._match(TokenType.AND):
            right = self._parse_not()
            left = BinaryOp(op="and", left=left, right=right, line=left.line, column=left.column)
        return left

    def _parse_not(self) -> ASTNode:
        if tok := self._match(TokenType.NOT):
            operand = self._parse_not()
            return UnaryOp(op="not", operand=operand, line=tok.line, column=tok.column)
        return self._parse_comparison()

    def _parse_comparison(self) -> ASTNode:
        left = self._parse_tensor()
        comp_ops = {
            TokenType.DOUBLE_EQUALS: "==",
            TokenType.NOT_EQUALS: "!=",
            TokenType.LESS: "<",
            TokenType.GREATER: ">",
            TokenType.LESS_EQUAL: "<=",
            TokenType.GREATER_EQUAL: ">=",
        }
        while self._current().type in comp_ops:
            op_str = comp_ops[self._current().type]
            self.pos += 1
            right = self._parse_tensor()
            left = BinaryOp(op=op_str, left=left, right=right, line=left.line, column=left.column)
        return left

    def _parse_tensor(self) -> ASTNode:
        left = self._parse_addition()
        while self._match(TokenType.TENSOR):
            right = self._parse_addition()
            left = TensorProductNode(left=left, right=right, line=left.line, column=left.column)
        return left

    def _parse_addition(self) -> ASTNode:
        left = self._parse_multiplication()
        while True:
            if self._match(TokenType.PLUS):
                right = self._parse_multiplication()
                left = BinaryOp(op="+", left=left, right=right, line=left.line, column=left.column)
            elif self._match(TokenType.MINUS):
                right = self._parse_multiplication()
                left = BinaryOp(op="-", left=left, right=right, line=left.line, column=left.column)
            else:
                break
        return left

    def _parse_multiplication(self) -> ASTNode:
        left = self._parse_power()
        while True:
            if self._match(TokenType.STAR):
                right = self._parse_power()
                left = BinaryOp(op="*", left=left, right=right, line=left.line, column=left.column)
            elif self._match(TokenType.SLASH):
                right = self._parse_power()
                left = BinaryOp(op="/", left=left, right=right, line=left.line, column=left.column)
            elif self._match(TokenType.PERCENT):
                right = self._parse_power()
                left = BinaryOp(op="%", left=left, right=right, line=left.line, column=left.column)
            else:
                break
        return left

    def _parse_power(self) -> ASTNode:
        base = self._parse_unary()
        if self._match(TokenType.CARET):
            exp = self._parse_power()  # right-associative
            return BinaryOp(op="^", left=base, right=exp, line=base.line, column=base.column)
        return base

    def _parse_unary(self) -> ASTNode:
        if tok := self._match(TokenType.MINUS):
            operand = self._parse_unary()
            return UnaryOp(op="-", operand=operand, line=tok.line, column=tok.column)
        return self._parse_postfix()

    def _parse_postfix(self) -> ASTNode:
        node = self._parse_primary()
        while True:
            if self._match(TokenType.LBRACKET):
                index = self._parse_expression()
                self._expect(TokenType.RBRACKET, "Expected ']'")
                node = IndexAccess(target=node, index=index, line=node.line, column=node.column)
            elif self._match(TokenType.DOT):
                member = self._expect(TokenType.IDENTIFIER, "Expected member name")
                node = MemberAccess(target=node, member=member.value, line=node.line, column=node.column)
            elif self._match(TokenType.LPAREN):
                # function call on identifier
                args: list[ASTNode] = []
                if not self._check(TokenType.RPAREN):
                    args.append(self._parse_expression())
                    while self._match(TokenType.COMMA):
                        args.append(self._parse_expression())
                self._expect(TokenType.RPAREN, "Expected ')'")
                if isinstance(node, Identifier):
                    node = FunctionCall(name=node.name, args=args, line=node.line, column=node.column)
                else:
                    node = FunctionCall(name="__call__", args=[node, *args], line=node.line, column=node.column)
            elif self._match(TokenType.DAGGER):
                node = DaggerNode(operand=node, line=node.line, column=node.column)
            else:
                break
        return node

    def _parse_primary(self) -> ASTNode:
        tok = self._current()

        # Integer
        if tok.type == TokenType.INTEGER:
            self.pos += 1
            return IntegerLiteral(value=int(tok.value), line=tok.line, column=tok.column)

        # Float
        if tok.type == TokenType.FLOAT:
            self.pos += 1
            return FloatLiteral(value=float(tok.value), line=tok.line, column=tok.column)

        # Complex (e.g. "2i", "3.14i")
        if tok.type == TokenType.COMPLEX:
            self.pos += 1
            return ComplexLiteral(value=float(tok.value.rstrip("i")), line=tok.line, column=tok.column)

        # String
        if tok.type == TokenType.STRING:
            self.pos += 1
            # Strip quotes
            val = tok.value[1:-1]
            return StringLiteral(value=val, line=tok.line, column=tok.column)

        # Boolean
        if tok.type == TokenType.BOOLEAN:
            self.pos += 1
            return BooleanLiteral(value=tok.value in ("true", "True"), line=tok.line, column=tok.column)

        # Identifier
        if tok.type == TokenType.IDENTIFIER:
            self.pos += 1
            return Identifier(name=tok.value, line=tok.line, column=tok.column)

        # Parenthesized expression
        if tok.type == TokenType.LPAREN:
            self.pos += 1
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN, "Expected ')'")
            return expr

        # List literal
        if tok.type == TokenType.LBRACKET:
            self.pos += 1
            elements: list[ASTNode] = []
            if not self._check(TokenType.RBRACKET):
                elements.append(self._parse_expression())
                while self._match(TokenType.COMMA):
                    elements.append(self._parse_expression())
            self._expect(TokenType.RBRACKET, "Expected ']'")
            return ListNode(elements=elements, line=tok.line, column=tok.column)

        # Ket: |label⟩
        if tok.type == TokenType.KET_BAR:
            self.pos += 1
            label_tok = self._current()
            if label_tok.type in (TokenType.IDENTIFIER, TokenType.INTEGER):
                label = label_tok.value
                self.pos += 1
            else:
                label = ""
            # Expect closing | or ⟩ (we accept | as ket closer)
            if self._check(TokenType.KET_BAR):
                self.pos += 1
            elif self._check(TokenType.GREATER):
                self.pos += 1
            return KetStateNode(label=label, line=tok.line, column=tok.column)

        # Bra: ⟨label|
        if tok.type == TokenType.BRA_BAR:
            self.pos += 1
            label_tok = self._current()
            if label_tok.type in (TokenType.IDENTIFIER, TokenType.INTEGER):
                label = label_tok.value
                self.pos += 1
            else:
                label = ""
            if self._check(TokenType.KET_BAR):
                self.pos += 1
                # Check if this is a braket ⟨a|b⟩
                if self._current().type in (TokenType.IDENTIFIER, TokenType.INTEGER):
                    ket_label = self._current().value
                    self.pos += 1
                    if self._check(TokenType.KET_BAR) or self._check(TokenType.GREATER):
                        self.pos += 1
                    return BraKetNode(bra_label=label, ket_label=ket_label, line=tok.line, column=tok.column)
            return BraStateNode(label=label, line=tok.line, column=tok.column)

        raise ParseError(f"Unexpected token: {tok.type.name} ({tok.value!r})", tok)
