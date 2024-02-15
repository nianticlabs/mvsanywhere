# Stage: system-dependencies
# Base Ubuntu 22.10 that is extended with all system dependencies needed by this repo for training, benchmarking and other tools
FROM ubuntu:22.04@sha256:b492494d8e0113c4ad3fe4528a4b5ff89faa5331f7d52c5c138196f69ce176a6 AS system-dependencies

WORKDIR /opt/geometryhints

RUN groupadd -g 65537 researchers && \
    useradd -m -r -u 1100 -g researchers masala

RUN apt-get update && \
    apt-get install -yq dumb-init curl git make libgl1-mesa-glx libgomp1 libomp-dev wget && \
    rm -rf /var/lib/apt/lists/*

# --------------------------------------
# Stage: python-dependencies
# Extends system-dependencies with additional tools (like Conda) to pull and build project
# environment with all the Python dependencies
FROM system-dependencies AS python-dependencies

RUN apt-get update && \
    apt-get install -yq gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh" && \
    mkdir -p $HOME/.conda && \
    bash Mambaforge-Linux-x86_64.sh -b -p /opt/conda && \
    rm -f Mambaforge-Linux-x86_64.sh
ENV PATH="/opt/conda/envs/geometryhints/bin:/opt/conda/mambaforge/bin:/opt/conda/bin:${PATH}"

COPY environment.yml .

RUN --mount=type=secret,id=pip_conf mamba env update -f environment.yml -n geometryhints && \
    export PIP_CONFIG_FILE=/run/secrets/pip_conf && \
    mamba clean -y -a -f && \
    chmod -R 777 /opt/conda

# --------------------------------------
# Stage: geometryhints
# Extends system-dependencies and includes all Python dependencies copied from
# python-dependencies (but without all build-time dependencies like Conda)
FROM system-dependencies AS geometryhints

ENV PATH="/opt/conda/envs/geometryhints/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"

COPY --from=python-dependencies /opt/conda /opt/conda

LABEL org.opencontainers.image.authors="filippoaleotti@nianticlabs.com"

# IMPORTANT: Copy files in the order from least to most often changed
# Proper order of copies improves Docker layers cache reuse
COPY src src
COPY setup.py setup.py

ARG CI_COMMIT_SHA
ENV CI_COMMIT_SHA=$CI_COMMIT_SHA

RUN pip install -e .

