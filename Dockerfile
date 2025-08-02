# Multi-stage Dockerfile for Liquid AI Vision Kit
# Optimized for embedded AI development and deployment

ARG BUILDPLATFORM=linux/amd64
ARG TARGETPLATFORM=linux/amd64

# Base builder stage with common tools
FROM --platform=$BUILDPLATFORM ubuntu:22.04 AS base-builder

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install essential build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    pkg-config \
    libtool \
    autoconf \
    automake \
    python3 \
    python3-pip \
    python3-dev \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Setup ccache for faster builds
ENV CCACHE_DIR=/tmp/ccache
ENV CMAKE_C_COMPILER_LAUNCHER=ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
RUN ccache --max-size=2G

# ARM cross-compilation builder stage
FROM base-builder AS arm-builder

# Install ARM toolchain and dependencies
RUN apt-get update && apt-get install -y \
    gcc-arm-none-eabi \
    binutils-arm-none-eabi \
    libnewlib-arm-none-eabi \
    gdb-multiarch \
    openocd \
    && rm -rf /var/lib/apt/lists/*

# Set up build environment
WORKDIR /workspace
COPY . .

# Configure ARM build
ENV CMAKE_TOOLCHAIN_FILE=/workspace/cmake/arm-none-eabi.cmake
ENV CMAKE_BUILD_TYPE=Release

# Build for ARM Cortex-M7
RUN mkdir -p build-arm && cd build-arm && \
    cmake .. \
        -G Ninja \
        -DTARGET_PLATFORM=ARM_CORTEX_M7 \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_TESTS=OFF \
        -DENABLE_PYTHON_BINDINGS=OFF && \
    ninja -j$(nproc)

# Development stage with full toolchain
FROM ubuntu:22.04 AS development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-venv \
    gcc-arm-none-eabi \
    gdb-multiarch \
    openocd \
    clang-format \
    cppcheck \
    valgrind \
    && rm -rf /var/lib/apt/lists/*

# Create development user
RUN useradd -m -s /bin/bash dev && \
    usermod -aG dialout dev

# Set up Python environment
COPY requirements-dev.txt /tmp/
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Configure development environment
WORKDIR /workspace
USER dev

# Set environment variables
ENV CMAKE_BUILD_TYPE=Debug
ENV TARGET_PLATFORM=X86_SIMULATION

# Expose common ports
EXPOSE 8080 3333 5760

# Default command for development
CMD ["/bin/bash"]

# Production simulation stage
FROM ubuntu:22.04 AS simulation

# Minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install minimal Python dependencies
COPY requirements.txt /tmp/
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Copy simulation binaries from builder
COPY --from=arm-builder /workspace/build-arm/liquid_vision_sim /usr/local/bin/

# Create runtime user
RUN useradd -m -s /bin/bash runner
USER runner

WORKDIR /app

# Runtime command
CMD ["liquid_vision_sim"]

# Final embedded deployment stage
FROM scratch AS embedded

# Copy only the embedded firmware
COPY --from=arm-builder /workspace/build-arm/liquid_vision_core.elf /firmware.elf
COPY --from=arm-builder /workspace/build-arm/liquid_vision_core.bin /firmware.bin

# Metadata
LABEL org.opencontainers.image.title="Liquid AI Vision Kit"
LABEL org.opencontainers.image.description="Embedded neural network firmware"
LABEL org.opencontainers.image.version="1.0.0"