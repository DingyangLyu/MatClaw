# MatClaw Agent Container
# Runs Claude Agent SDK in isolated Linux VM with browser automation
# and materials science computation tools (QE, LAMMPS, RASPA3, MACE, etc.)
#
# Build variants:
#   CPU (default):  docker build -t matclaw-agent:latest .
#   CUDA (GPU):     docker build --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 \
#                                --build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cu128 \
#                                -t matclaw-agent:cuda .
#   Or use:         ./build.sh --cuda

# Runtime base image (overridden for GPU variants)
ARG BASE_IMAGE=ubuntu:24.04

# CUDA builder image for LAMMPS KOKKOS (needs nvcc); no-op for CPU builds
ARG CUDA_BUILDER_IMAGE=ubuntu:24.04
ARG ENABLE_CUDA=OFF
# Kokkos architecture for LAMMPS GPU: AMPERE86 (RTX 30xx), ADA89 (RTX 40xx)
ARG LAMMPS_KOKKOS_ARCH=AMPERE86

# ==========================================
# STAGE 1: Build Quantum ESPRESSO from source (always CPU)
# QE CUDA requires NVHPC compiler — not worth the complexity for container use.
# For GPU-accelerated DFT, use vasp-remote to offload to HPC.
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
    tar -zxf q-e-qe-7.5.tar.gz --strip-components=1

RUN mkdir -p /opt/qe/bin && \
    ./configure --enable-openmp --enable-parallel && \
    make -j$(nproc) all && \
    strip --strip-unneeded bin/* || true && \
    cp -L bin/* /opt/qe/bin/

# ==========================================
# STAGE 1b: Build LAMMPS with KOKKOS/CUDA (GPU only)
# CPU builds use the apt lammps package instead
# ==========================================
FROM ${CUDA_BUILDER_IMAGE} AS lammps-builder

ARG ENABLE_CUDA
ARG LAMMPS_KOKKOS_ARCH

ENV DEBIAN_FRONTEND=noninteractive

# Create target dirs; only do source build when CUDA is enabled
RUN mkdir -p /opt/lammps/bin /opt/lammps/lib && \
    if [ "$ENABLE_CUDA" != "ON" ]; then \
      echo "Skipping LAMMPS source build (CPU: will use apt package)" && exit 0; \
    fi && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      wget ca-certificates build-essential cmake \
      gfortran libopenmpi-dev libfftw3-dev libjpeg-dev libpng-dev \
      python3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    wget -q https://github.com/lammps/lammps/archive/refs/tags/stable_29Aug2024_update1.tar.gz \
      -O /tmp/lammps.tar.gz && \
    mkdir -p /build/lammps && cd /build/lammps && \
    tar -zxf /tmp/lammps.tar.gz --strip-components=1 && \
    echo "Building LAMMPS with KOKKOS/CUDA (arch: $LAMMPS_KOKKOS_ARCH)..." && \
    cmake -B build -S cmake \
      -DCMAKE_INSTALL_PREFIX=/opt/lammps \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/build/lammps/lib/kokkos/bin/nvcc_wrapper \
      -DBUILD_MPI=ON \
      -DPKG_KOKKOS=ON \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_${LAMMPS_KOKKOS_ARCH}=ON \
      -DPKG_MANYBODY=ON \
      -DPKG_KSPACE=ON \
      -DPKG_MOLECULE=ON \
      -DPKG_RIGID=ON \
      -DPKG_REPLICA=ON \
      -DPKG_MC=ON \
      -DPKG_EXTRA-PAIR=ON \
      -DPKG_EXTRA-COMPUTE=ON \
      -DPKG_EXTRA-FIX=ON \
      -DPKG_EXTRA-DUMP=ON \
      -DPKG_ML-SNAP=ON \
      -DPKG_ML-PACE=ON && \
    cmake --build build -j$(nproc) && \
    cmake --install build && \
    rm -rf /build/lammps /tmp/lammps.tar.gz

# ==========================================
# STAGE 2: Runtime (Ubuntu 24.04 for RASPA3 glibc compatibility)
# For GPU: nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 (also Ubuntu 24.04 based)
# ==========================================
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Re-declare ARGs after FROM (Docker scoping rule)
ARG PYTORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG ENABLE_CUDA=OFF

ENV DEBIAN_FRONTEND=noninteractive

# NVIDIA container runtime configuration (no-op without GPU)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

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
    openssh-client \
    python3 python3-pip python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Install RASPA3 (amd64 only — no official arm64 release)
RUN dpkgArch="$(dpkg --print-architecture)" && \
    if [ "$dpkgArch" = "amd64" ]; then \
      wget -q https://github.com/iRASPA/RASPA3/releases/download/v3.0.16/raspa_3.0.16_amd64_ubuntu-24-core-avx2.deb -O /tmp/raspa3.deb && \
      apt-get update && \
      apt-get install -y --no-install-recommends /tmp/raspa3.deb && \
      rm /tmp/raspa3.deb && \
      apt-get clean && \
      rm -rf /var/lib/apt/lists/*; \
    else \
      echo "RASPA3: skipping (no arm64 release available, MC simulations will not work)"; \
    fi

# Copy Quantum ESPRESSO binaries from builder stage
COPY --from=qe-builder /opt/qe/bin /opt/qe/bin
RUN chmod +x /opt/qe/bin/*
ENV PATH="/opt/qe/bin:$PATH"

# LAMMPS: source-built with KOKKOS/CUDA for GPU, apt package for CPU
COPY --from=lammps-builder /opt/lammps /opt/lammps
RUN if [ "$ENABLE_CUDA" = "ON" ]; then \
      chmod +x /opt/lammps/bin/* 2>/dev/null || true; \
    else \
      apt-get update && apt-get install -y --no-install-recommends lammps openkim-models && \
      rm -rf /var/lib/apt/lists/*; \
    fi
ENV PATH="/opt/lammps/bin:$PATH"

# Install Miniconda (auto-detect architecture)
RUN dpkgArch="$(dpkg --print-architecture)" && \
    if [ "$dpkgArch" = "amd64" ]; then \
      CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    else \
      CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    fi && \
    wget -q "$CONDA_URL" -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"
RUN conda init bash

# Pre-install common materials science Python packages in base environment
# PYTORCH_INDEX selects CPU or CUDA wheels at build time
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url ${PYTORCH_INDEX}
RUN pip install --no-cache-dir \
    numpy scipy matplotlib pandas seaborn \
    pymatgen ase mp-api spglib \
    mace-torch \
    e3nn \
    chgnet sevenn matgl

# MPI environment (allow running as root in container)
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Set Chromium path for agent-browser
ENV AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium-browser
ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium-browser

# Install vasp-remote script (VASP execution via SSH or local)
COPY scripts/vasp-remote /usr/local/bin/vasp-remote
RUN chmod +x /usr/local/bin/vasp-remote

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

# Bundle computation skills into the image so standalone `docker run` works.
# When running via the MatClaw orchestrator, the host mounts its own skills
# (from container/skills/) over /home/node/.claude, overriding these defaults.
COPY skills/ /home/node/.claude/skills/

# Set ownership to node user (non-root) for writable directories
RUN chown -R node:node /workspace /home/node/.claude && chmod 777 /home/node

# Switch to non-root user (required for --dangerously-skip-permissions)
USER node

# Set working directory to group workspace
WORKDIR /workspace/group

# Entry point reads JSON from stdin, outputs JSON to stdout
ENTRYPOINT ["/app/entrypoint.sh"]
