# MatClaw Agent Container
# Runs Claude Agent SDK in isolated Linux VM with browser automation
# and materials science computation tools (QE, LAMMPS, RASPA3, MACE, etc.)

# ==========================================
# STAGE 1: Build Quantum ESPRESSO from source
# ==========================================
FROM ubuntu:24.04 AS qe-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget ca-certificates build-essential git \
    gfortran gcc-14 g++-14 \
    libopenblas-dev \
    libfftw3-dev libfftw3-mpi-dev \
    libopenmpi-dev libscalapack-openmpi-dev \
    libhdf5-dev \
    python3-dev python3-pip python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build/qe
RUN wget -q https://gitlab.com/QEF/q-e/-/archive/qe-7.5/q-e-qe-7.5.tar.gz && \
    tar -zxf q-e-qe-7.5.tar.gz --strip-components=1 && \
    ./configure --enable-openmp --enable-parallel && \
    make -j$(nproc) all && \
    strip --strip-unneeded bin/* || true && \
    find bin/ -type l -exec sh -c 'cp -L "$1" "$1.tmp" && mv "$1.tmp" "$1"' _ {} \;

# ==========================================
# STAGE 2: Runtime (Ubuntu 24.04 for RASPA3 glibc compatibility)
# ==========================================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Node.js 22 from NodeSource
RUN apt-get update && apt-get install -y ca-certificates curl gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install system dependencies for Chromium + materials computation
RUN apt-get update && apt-get install -y \
    chromium-browser \
    fonts-liberation \
    fonts-noto-cjk \
    fonts-noto-color-emoji \
    libgbm1 \
    libnss3 \
    libatk-bridge2.0-0t64 \
    libgtk-3-0t64 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libasound2t64 \
    libpangocairo-1.0-0 \
    libcups2t64 \
    libdrm2 \
    libxshmfence1 \
    curl \
    git \
    wget \
    build-essential \
    gfortran \
    libgomp1 \
    libgfortran5 \
    libopenblas0 \
    libfftw3-double3 \
    libopenmpi3t64 \
    openmpi-bin \
    libhdf5-103-1t64 \
    lammps \
    openkim-models \
    python3 python3-pip python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Install RASPA3
RUN wget -q https://github.com/iRASPA/RASPA3/releases/download/v3.0.16/raspa_3.0.16_amd64_ubuntu-24-core-avx2.deb -O /tmp/raspa3.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends /tmp/raspa3.deb && \
    rm /tmp/raspa3.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy Quantum ESPRESSO binaries from builder stage
COPY --from=qe-builder /build/qe/bin /opt/qe/bin
RUN chmod +x /opt/qe/bin/*
ENV PATH="/opt/qe/bin:$PATH"

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"
RUN conda init bash

# Pre-install common materials science Python packages in base environment
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    numpy scipy matplotlib pandas seaborn \
    pymatgen ase mp-api spglib \
    mace-torch \
    e3nn

# MPI environment (allow running as root in container)
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Set Chromium path for agent-browser
ENV AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium-browser
ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium-browser

# Install agent-browser and claude-code globally
RUN npm install -g agent-browser @anthropic-ai/claude-code

# Create app directory
WORKDIR /app

# Copy package files first for better caching
COPY agent-runner/package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY agent-runner/ ./

# Build TypeScript
RUN npm run build

# Create workspace directories
RUN mkdir -p /workspace/group /workspace/global /workspace/extra /workspace/ipc/messages /workspace/ipc/tasks /workspace/ipc/input

# Create entrypoint script
# Secrets are passed via stdin JSON — temp file is deleted immediately after Node reads it
# Follow-up messages arrive via IPC files in /workspace/ipc/input/
RUN printf '#!/bin/bash\nset -e\ncd /app && npx tsc --outDir /tmp/dist 2>&1 >&2\nln -s /app/node_modules /tmp/dist/node_modules\nchmod -R a-w /tmp/dist\ncat > /tmp/input.json\nnode /tmp/dist/index.js < /tmp/input.json\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Create node user (provided by node:22-slim but not ubuntu base)
# Ubuntu 24.04 ships a default 'ubuntu' user at uid/gid 1000 — remove it first
RUN userdel -r ubuntu 2>/dev/null || true && \
    groupadd --gid 1000 node && \
    useradd --uid 1000 --gid node --shell /bin/bash --create-home node

# Give node user passwordless sudo for runtime package installation
RUN apt-get update && apt-get install -y sudo && rm -rf /var/lib/apt/lists/* && \
    echo "node ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set ownership to node user (non-root) for writable directories
RUN chown -R node:node /workspace && chmod 777 /home/node

# Switch to non-root user (required for --dangerously-skip-permissions)
USER node

# Set working directory to group workspace
WORKDIR /workspace/group

# Entry point reads JSON from stdin, outputs JSON to stdout
ENTRYPOINT ["/app/entrypoint.sh"]
