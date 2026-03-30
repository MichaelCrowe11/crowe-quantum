"""ASCII and data-based quantum circuit drawing.

Converts a CircuitIR into human-readable circuit diagrams using
box-drawing characters, with support for multi-qubit gates,
measurements, barriers, and conditional operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crowe_quantum_core.protocols import CircuitIR


# Box-drawing elements
_WIRE = "\u2500"       # horizontal line
_CROSS = "\u253C"      # crossing
_BARRIER = "\u2502"    # vertical bar
_CTRL = "\u25CF"       # filled circle (control)
_TARGET_X = "\u2295"   # circled plus (CNOT target)
_MEASURE = "\u25A1"    # box (measurement)


@dataclass
class CircuitColumn:
    """One time-slice of the circuit diagram."""

    entries: dict[int, str] = field(default_factory=dict)
    connectors: list[tuple[int, int]] = field(default_factory=list)


class CircuitDrawer:
    """Converts CircuitIR to visual representations."""

    def __init__(self, circuit: CircuitIR) -> None:
        self.circuit = circuit
        self._columns: list[CircuitColumn] = []
        self._build()

    def _build(self) -> None:
        """Parse operations into columns."""
        # Track the next available column for each qubit
        qubit_col: dict[int, int] = {q: 0 for q in range(self.circuit.num_qubits)}

        for op in self.circuit.operations:
            if op.kind == "gate":
                # Find the earliest column all target qubits are free
                min_col = max(qubit_col.get(q, 0) for q in op.qubits)
                while len(self._columns) <= min_col:
                    self._columns.append(CircuitColumn())

                col = self._columns[min_col]
                name = op.gate_name

                if name in ("CNOT", "CX") and len(op.qubits) == 2:
                    col.entries[op.qubits[0]] = _CTRL
                    col.entries[op.qubits[1]] = _TARGET_X
                    col.connectors.append((op.qubits[0], op.qubits[1]))
                elif name in ("CZ",) and len(op.qubits) == 2:
                    col.entries[op.qubits[0]] = _CTRL
                    col.entries[op.qubits[1]] = _CTRL
                    col.connectors.append((op.qubits[0], op.qubits[1]))
                elif name in ("TOFFOLI", "CCX") and len(op.qubits) == 3:
                    col.entries[op.qubits[0]] = _CTRL
                    col.entries[op.qubits[1]] = _CTRL
                    col.entries[op.qubits[2]] = _TARGET_X
                    col.connectors.append((op.qubits[0], op.qubits[2]))
                elif name in ("SWAP",) and len(op.qubits) == 2:
                    col.entries[op.qubits[0]] = "\u2A09"  # big X
                    col.entries[op.qubits[1]] = "\u2A09"
                    col.connectors.append((op.qubits[0], op.qubits[1]))
                else:
                    # Standard gate box
                    label = name
                    if op.params:
                        param_str = ",".join(f"{p:.2f}" for p in op.params)
                        label = f"{name}({param_str})"
                    for q in op.qubits:
                        col.entries[q] = f"[{label}]"

                for q in op.qubits:
                    qubit_col[q] = min_col + 1

            elif op.kind == "measure":
                q = op.qubits[0]
                min_col = qubit_col.get(q, 0)
                while len(self._columns) <= min_col:
                    self._columns.append(CircuitColumn())
                self._columns[min_col].entries[q] = f"{_MEASURE}M"
                qubit_col[q] = min_col + 1

            elif op.kind == "barrier":
                # Barrier spans all involved qubits
                min_col = max(qubit_col.get(q, 0) for q in op.qubits)
                while len(self._columns) <= min_col:
                    self._columns.append(CircuitColumn())
                for q in op.qubits:
                    self._columns[min_col].entries[q] = _BARRIER
                    qubit_col[q] = min_col + 1

            elif op.kind == "reset":
                q = op.qubits[0]
                min_col = qubit_col.get(q, 0)
                while len(self._columns) <= min_col:
                    self._columns.append(CircuitColumn())
                self._columns[min_col].entries[q] = "|0>"
                qubit_col[q] = min_col + 1

    def to_text(self) -> str:
        """Render circuit as ASCII text."""
        n = self.circuit.num_qubits
        if not self._columns:
            return "\n".join(f"q[{i}]: {_WIRE*10}" for i in range(n))

        # Determine column widths
        col_widths = []
        for col in self._columns:
            w = 3  # minimum
            for entry in col.entries.values():
                w = max(w, len(entry) + 2)
            col_widths.append(w)

        lines = []
        for q in range(n):
            parts = [f"q[{q}]: "]
            for ci, col in enumerate(self._columns):
                w = col_widths[ci]
                if q in col.entries:
                    entry = col.entries[q]
                    padded = entry.center(w, _WIRE)
                else:
                    padded = _WIRE * w
                parts.append(padded)
            parts.append(_WIRE * 3)
            lines.append("".join(parts))

        return "\n".join(lines)

    def to_data(self) -> dict:
        """Return structured data for programmatic rendering."""
        return {
            "num_qubits": self.circuit.num_qubits,
            "columns": [
                {
                    "entries": col.entries,
                    "connectors": col.connectors,
                }
                for col in self._columns
            ],
            "depth": len(self._columns),
        }


def draw_circuit(circuit: CircuitIR) -> str:
    """Convenience function: render CircuitIR as ASCII text."""
    return CircuitDrawer(circuit).to_text()
