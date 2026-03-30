# Crowe Quantum Platform — Design Specification

**Author:** Michael Crowe / Crowe Logic Inc.
**Date:** 2026-03-29
**Status:** Approved
**PyPI Account:** Crowe-Logic
**GitHub:** MichaelCrowe11

---

## Overview

Ground-up rewrite of the Quantum Trinity (synapse-lang v2.3.4, synapse-qubit-flow v1.0.0, quantum-net-lang) as a federated family of 8 packages under the `crowe-*` namespace on PyPI. Published via Trusted Publishing (GitHub Actions OIDC) to the Crowe-Logic PyPI account.

## Design Principles

1. **"Seen and Dreamed"** — Every quantum state is immediately visible. The REPL renders Bloch spheres, Q-spheres, phase wheels, and interference patterns as you type.
2. **Dirac Notation as First-Class Syntax** — `|psi>`, `<psi|`, tensor products, daggers are language primitives, not string hacks.
3. **Category-Theoretic Foundations** — Gates are morphisms, tensor products are parallel composition, measurement is a functor from quantum to classical.
4. **Error-Aware Type System** — Unitarity enforced at compile time. No-cloning is a type error. Measurement is irreversible in the type system.
5. **Artistic Expression Native** — Quantum states map to visual art (fractals, color fields, interference patterns) and music (pitch fields, superposition chords, quantum walks).
6. **Real Hardware, Real Performance** — JAX-accelerated simulation, PennyLane differentiable computing, Qiskit Runtime for IBM hardware, Cirq for Google, Braket for AWS.

## Package Architecture

### 1. crowe-quantum-core (v1.0.0)

The shared foundation. Every other package depends on this.

**Modules:**
- `core.types` — Type system: QubitType, CircuitType, UncertainType, TensorType, FunctionType, GenericType. Unitarity constraints. No-cloning enforcement.
- `core.ir` — Intermediate representation: quantum IR nodes (gates, measurements, classical control), SSA-form for optimization passes.
- `core.gates` — Universal gate registry: 20+ gates with matrix definitions, decomposition rules, commutation relations. GateSpec with arity, parameters, aliases.
- `core.states` — StateVector, DensityMatrix, PauliString, SparseObservable. NumPy-backed with JAX-compatible interface.
- `core.protocols` — Abstract backend protocol (simulate, transpile, execute). Hardware-agnostic interface.
- `core.errors` — Error hierarchy: QuantumError, UnitarityError, NoCloningError, QubitRangeError, GateArityError.
- `core.noise` — Noise models: depolarizing, amplitude damping, phase damping, thermal relaxation, readout error. Channel representation (Kraus operators).
- `core.tensor` — Tensor network primitives: contraction, decomposition (SVD, QR), MPS/MPO representations.

**Dependencies:** numpy>=2.0, scipy>=1.14

### 2. crowe-synapse (v3.0.0)

The scientific reasoning language — complete rewrite with full interpreter coverage.

**Modules:**
- `synapse.lexer` — Tokenizer: 200+ token types including scientific, quantum, uncertainty, symbolic, parallel, pipeline keywords. Unicode math operators.
- `synapse.parser` — Single unified recursive-descent parser (no more 3 parser variants). Full language coverage: hypothesis, experiment, parallel, reason chains, pipelines, explore, uncertainty, symbolic, quantum, control flow, function definitions.
- `synapse.ast` — Consolidated AST: 50+ node types. Clean hierarchy with visitor pattern. No duplicate definitions.
- `synapse.interpreter` — Full interpreter: executes ALL constructs the parser handles. Dispatch-based with environment/scope management.
- `synapse.typechecker` — Static type checking with Hindley-Milner inference extended for scientific types. Uncertainty propagation through types.
- `synapse.jit` — JAX-based JIT compilation (replaces Numba for core path). Automatic differentiation of scientific computations. GPU/TPU dispatch.
- `synapse.uncertainty` — UncertainValue with full operator overloading. Monte Carlo, Bayesian (real implementation), interval, polynomial chaos propagation. Correlation matrix.
- `synapse.symbolic` — SymPy engine: expressions, equations, matrices, logic, proofs. Code generation to C/Fortran/JAX.
- `synapse.parallel` — Parallel execution: threading, multiprocessing, asyncio, JAX pmap for distributed.
- `synapse.stdlib` — Standard library: statistics, linear algebra, optimization, signal processing.
- `synapse.repl` — Rich-powered REPL with syntax highlighting, inline visualization, history, tab completion.
- `synapse.cli` — Click-based CLI: run files, REPL, compile, benchmark.

**Dependencies:** crowe-quantum-core, sympy>=1.14, rich>=13.0, click>=8.1, jax>=0.4.30 (optional)

### 3. crowe-qubit-flow (v2.0.0)

The quantum circuit language with Dirac notation — full rewrite from the root-level legacy modules.

**Modules:**
- `qubit_flow.lexer` — Quantum-focused tokenizer: ket/bra notation, gate names, algorithm keywords, error correction keywords, complex number literals, Unicode operators (tensor, dagger, bra).
- `qubit_flow.parser` — Recursive descent: circuits, qubit/qudit declarations, gate applications, measurements, entanglement, superposition with complex amplitudes, teleportation, algorithms, classical control, function definitions.
- `qubit_flow.ast` — 30+ node types: Circuit, Qubit, Qudit, Gate, Measurement, Entanglement (bell/ghz/w), Superposition, Teleportation, Algorithm nodes (Grover, Shor, VQE, QAOA, QFT, QPE), KetState, BraState, TensorProduct, ErrorCorrection.
- `qubit_flow.interpreter` — Full interpreter with quantum state simulation. Uses core.states for state management. Real Grover's (not stub), real Shor's (quantum period finding via QFT), real QFT (proper controlled rotations).
- `qubit_flow.compiler` — Compiles QubitFlow AST to core IR, then to Qiskit circuits or PennyLane tapes for hardware execution.
- `qubit_flow.transpiler` — Gate decomposition, routing, optimization passes. Leverages Qiskit's transpiler when available.
- `qubit_flow.error_correction` — Surface codes, toric codes, stabilizer formalism, syndrome measurement, logical qubit abstraction.
- `qubit_flow.semantics` — Pre-execution validation: gate arity, qubit range, parameter count, unitarity checks.
- `qubit_flow.repl` — Rich REPL with inline Bloch sphere rendering and circuit diagrams.
- `qubit_flow.cli` — CLI: run circuits, interactive mode, compile to OpenQASM 3.0.

**Dependencies:** crowe-quantum-core, rich>=13.0, click>=8.1

### 4. crowe-quantum-net (v1.0.0)

Quantum networking and cryptography — migrated from root-level legacy modules.

**Modules:**
- `quantum_net.lexer` — Network-focused tokenizer: node, channel, protocol, entangle, teleport, encrypt, key keywords.
- `quantum_net.parser` — Network topology declarations, protocol definitions, channel operations.
- `quantum_net.ast` — NetworkNode, QuantumChannel, EntangledPair, Protocol, BB84, E91, teleportation nodes.
- `quantum_net.interpreter` — Simulates quantum networks: QKD key exchange, entanglement distribution, teleportation, error rates.
- `quantum_net.protocols` — BB84, E91, MDI-QKD, twin-field QKD, quantum repeaters, entanglement swapping.
- `quantum_net.topology` — Network graph management: nodes, edges, routing, entanglement paths.
- `quantum_net.cli` — CLI for network simulation.

**Dependencies:** crowe-quantum-core, networkx>=3.4, rich>=13.0, click>=8.1

### 5. crowe-quantum-hub (v1.0.0)

Hardware backends and acceleration — the bridge to real quantum computers.

**Modules:**
- `quantum_hub.simulator` — Built-in statevector simulator (JAX-accelerated, 30+ qubits on CPU, 40+ on GPU).
- `quantum_hub.ibm` — Qiskit Runtime integration: EstimatorV2, SamplerV2, error mitigation (ZNE, PEC, DD, twirling), session management.
- `quantum_hub.google` — Cirq integration: device specs, noise models, qsim high-performance simulator.
- `quantum_hub.xanadu` — PennyLane integration: differentiable quantum computing, Lightning backends, Catalyst JIT.
- `quantum_hub.aws` — Amazon Braket: IonQ, QuEra, Rigetti, cloud simulators.
- `quantum_hub.jax_backend` — JAX-native quantum simulation: jit-compiled state evolution, vmap for parameter sweeps, grad for variational algorithms.
- `quantum_hub.registry` — Backend discovery and registration. Auto-detect available backends.

**Dependencies:** crowe-quantum-core, qiskit>=2.0 (optional), qiskit-ibm-runtime>=0.40 (optional), cirq>=1.4 (optional), pennylane>=0.39 (optional), pennylane-lightning>=0.44 (optional), amazon-braket-sdk>=1.70 (optional), jax>=0.4.30 (optional)

### 6. crowe-quantum-viz (v1.0.0)

Visualization — the "seen and dreamed" package.

**Modules:**
- `quantum_viz.bloch` — Bloch sphere rendering: single qubit, multi-qubit (one sphere per qubit). Terminal (Rich/Unicode art) + matplotlib + interactive 3D (PyVista).
- `quantum_viz.qsphere` — Q-sphere: multi-qubit state as single sphere. Amplitude = dot size, phase = dot color, Hamming weight = latitude.
- `quantum_viz.phase` — Phase wheels: complex amplitude constellation diagrams.
- `quantum_viz.density` — Density matrix visualization: state city plots (3D bars), Hinton diagrams (magnitude/sign squares).
- `quantum_viz.wigner` — Wigner function quasi-probability distributions. Color-mapped surface plots.
- `quantum_viz.circuits` — Circuit diagram rendering: ASCII (terminal), SVG, interactive.
- `quantum_viz.tensor_nets` — Tensor network diagrams: boxes, lines, contractions. Category-theoretic string diagrams.
- `quantum_viz.animation` — Manim-powered mathematical animations: state evolution, gate application, interference patterns, entanglement.
- `quantum_viz.generative` — Quantum generative art: measurement → color fields, circuit → fractals, walk → patterns.
- `quantum_viz.rich_repl` — Rich/Textual integration: inline terminal visualization, live-updating state displays, beautiful error messages.
- `quantum_viz.jupyter` — Jupyter widget integration: interactive Bloch spheres, circuit builders.

**Dependencies:** crowe-quantum-core, rich>=13.0, textual>=0.50 (optional), matplotlib>=3.8 (optional), manim>=0.19 (optional), pyvista>=0.44 (optional), plotly>=5.18 (optional)

### 7. crowe-quantum-audio (v1.0.0)

The quantum-music bridge — connects quantum computing to sonic art.

**Modules:**
- `quantum_audio.pitch_field` — QuantumPitchField: maps quantum state amplitudes to pitch probabilities. Born-rule note selection. Interference between intervals.
- `quantum_audio.chords` — SuperpositionChord: chord voicings in quantum superposition. Measurement collapses to specific voicing with voice-leading constraints.
- `quantum_audio.rhythm` — Quantum walk rhythm generation: walk trajectories on rhythmic grids. Interference creates structured-but-surprising patterns.
- `quantum_audio.entanglement` — EntanglementNetwork: correlated channels. Measuring one instrument's state shifts the probability landscape of entangled instruments.
- `quantum_audio.humanize` — Quantum humanization: micro-timing and velocity variations from quantum noise (genuinely random, not pseudo-random).
- `quantum_audio.sonification` — Quantum state → sound: map amplitudes to frequencies, phases to pan positions, entanglement to spatial correlation.
- `quantum_audio.midi` — MIDI output: real-time MIDI generation from quantum measurements.
- `quantum_audio.talon_bridge` — Bridge to Talon music engine: export quantum compositions to Ableton Live sessions.

**Dependencies:** crowe-quantum-core, mido>=1.3 (MIDI), numpy>=2.0

### 8. crowe-quantum-trinity (v1.0.0)

The metapackage + bridge + unified experience.

**Modules:**
- `trinity.bridge` — QuantumTrinityBridge: orchestrates hybrid programs across all three languages. Shared state, cross-language variable passing.
- `trinity.cli` — Unified CLI: `crowe-quantum run`, `crowe-quantum repl`, `crowe-quantum compile`, `crowe-quantum viz`, `crowe-quantum hardware`.
- `trinity.repl` — Master REPL: auto-detects language from syntax, routes to appropriate interpreter, unified visualization.

**Dependencies:** crowe-synapse, crowe-qubit-flow, crowe-quantum-net, crowe-quantum-hub, crowe-quantum-viz, crowe-quantum-audio

## Monorepo Structure

```
crowe-quantum/
  .github/
    workflows/
      ci.yml              — Test matrix (Python 3.10-3.13, all packages)
      publish.yml          — Trusted Publishing (OIDC, per-package)
  packages/
    core/                  — crowe-quantum-core
    synapse/               — crowe-synapse
    qubit-flow/            — crowe-qubit-flow
    quantum-net/           — crowe-quantum-net
    quantum-hub/           — crowe-quantum-hub
    quantum-viz/           — crowe-quantum-viz
    quantum-audio/         — crowe-quantum-audio
    trinity/               — crowe-quantum-trinity
  docs/
  tests/                   — Integration tests (cross-package)
  pyproject.toml           — Dev/workspace root
  LICENSE
```

Each package follows:
```
packages/<name>/
  pyproject.toml
  src/
    <package_name>/
      __init__.py
      ...
  tests/
    test_*.py
```

## Publishing Strategy

- **Trusted Publishing** via GitHub Actions OIDC to Crowe-Logic PyPI account
- Each package has its own publish job triggered by git tags: `core-v1.0.0`, `synapse-v3.0.0`, etc.
- CI tests all packages on Python 3.10, 3.11, 3.12, 3.13
- Coverage threshold: 70% (raising over time)

## Implementation Priority

1. crowe-quantum-core (foundation — everything depends on this)
2. crowe-qubit-flow (quantum circuits — the visual centerpiece)
3. crowe-synapse (scientific reasoning — the largest rewrite)
4. crowe-quantum-viz (visualization — "seen and dreamed")
5. crowe-quantum-hub (hardware backends)
6. crowe-quantum-net (networking)
7. crowe-quantum-audio (music bridge)
8. crowe-quantum-trinity (metapackage + bridge)
