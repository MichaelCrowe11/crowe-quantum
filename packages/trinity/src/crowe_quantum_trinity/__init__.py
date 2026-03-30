"""Crowe Quantum Trinity — the complete quantum platform.

The Trinity unifies three pillars:
  1. QubitFlow  — quantum circuit programming language
  2. Synapse    — scientific reasoning with uncertainty
  3. Quantum Audio — quantum-music bridge

Plus the shared foundation:
  - Core       — types, gates, states, noise models
  - Hub        — backend connectors
  - Net        — quantum networking
  - Viz        — state visualization
"""

__version__ = "1.0.0"

# Re-export the platform
from crowe_quantum_audio import NoteEvent, QuantumScale, QuantumSequencer
from crowe_quantum_core import errors, gates, noise, protocols, states, tensor, types
from crowe_qubit_flow import Compiler, Interpreter, Lexer, Parser
from crowe_synapse import Expression, Quantity, Symbol, UncertainValue, Unit

__all__ = [
    # Core
    "errors",
    "gates",
    "noise",
    "protocols",
    "states",
    "tensor",
    "types",
    # QubitFlow
    "Compiler",
    "Interpreter",
    "Lexer",
    "Parser",
    # Synapse
    "Expression",
    "Quantity",
    "Symbol",
    "UncertainValue",
    "Unit",
    # Audio
    "NoteEvent",
    "QuantumScale",
    "QuantumSequencer",
]
