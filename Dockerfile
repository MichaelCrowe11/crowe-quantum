FROM python:3.12-slim

LABEL maintainer="Michael Crowe <Crowelogicos@gmail.com>"
LABEL description="Crowe Quantum Platform — federated quantum computing framework"
LABEL version="1.0.0"

WORKDIR /quantum

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
RUN pip install --no-cache-dir \
    numpy \
    sympy \
    scipy \
    matplotlib

# Copy all packages
COPY packages/ packages/

# Install in dependency order
RUN pip install --no-cache-dir --no-deps \
    -e packages/core \
    -e packages/synapse \
    -e packages/qubit-flow \
    -e packages/quantum-net \
    -e packages/quantum-hub \
    -e packages/quantum-viz \
    -e packages/quantum-audio \
    -e packages/trinity

# Verify installation
RUN python -c "from crowe_quantum_trinity import states, gates, Interpreter, UncertainValue, QuantumSequencer; print('Crowe Quantum Platform v1.0.0 ready')"

# Default: interactive Python with the platform loaded
CMD ["python", "-c", "from crowe_quantum_trinity import *; print('Crowe Quantum Platform v1.0.0'); print('Packages: core, qubit-flow, synapse, viz, hub, net, audio, trinity'); import code; code.interact(local=globals())"]
