# Development Environment Setup

## Prerequisites

### Required Tools
- **GCC ARM Toolchain 10.3+**: For embedded compilation
- **CMake 3.16+**: Build system
- **Python 3.8+**: Development tools and model conversion
- **Git 2.30+**: Version control

### Platform-Specific Requirements

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    gcc-arm-none-eabi \
    cmake \
    python3-dev \
    python3-pip \
    build-essential \
    git
```

#### macOS
```bash
brew install --cask gcc-arm-embedded
brew install cmake python git
```

#### Windows
- Install [ARM GNU Toolchain](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm)
- Install [CMake](https://cmake.org/download/)
- Install [Python](https://python.org/downloads/)

## Development Workflow

### 1. Repository Setup
```bash
git clone https://github.com/yourusername/liquid-ai-vision-kit.git
cd liquid-ai-vision-kit
git checkout -b feature/your-feature
```

### 2. Environment Configuration
```bash
# Python virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install Python dependencies
pip install -r requirements-dev.txt
```

### 3. Build Configuration
```bash
mkdir build && cd build

# For simulation/development
cmake .. -DTARGET_PLATFORM=X86_SIMULATION -DENABLE_TESTS=ON

# For embedded target
cmake .. -DTARGET_PLATFORM=ARM_CORTEX_M7 -DCMAKE_BUILD_TYPE=Release

# For PX4 integration
cmake .. -DTARGET_PLATFORM=PX4 -DCMAKE_TOOLCHAIN_FILE=../cmake/px4.cmake
```

### 4. Development Commands
```bash
# Build everything
make -j$(nproc)

# Run tests
make test

# Format code
make format

# Static analysis
make analyze

# Generate documentation
make docs
```

## IDE Configuration

### VS Code (Recommended)
Install extensions:
- C/C++ (Microsoft)
- CMake Tools
- Python
- GitLens

Workspace settings (`.vscode/settings.json`):
```json
{
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
    "cmake.buildDirectory": "${workspaceFolder}/build",
    "python.defaultInterpreterPath": "./venv/bin/python",
    "[cpp]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "ms-vscode.cpptools"
    }
}
```

### CLion
- Import project as CMake project
- Set toolchain to ARM GCC for embedded builds
- Configure remote debugging for hardware targets

## Debugging

### Software Debugging
```bash
# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# GDB debugging
gdb ./liquid_vision_sim
```

### Hardware Debugging
```bash
# Using OpenOCD (for STM32)
openocd -f interface/stlink.cfg -f target/stm32h7x.cfg

# GDB remote debugging
arm-none-eabi-gdb -ex "target remote localhost:3333" build/firmware.elf
```

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Use fixed-point arithmetic for embedded compatibility
- Mock hardware interfaces

### Integration Tests  
- Test component interactions
- Validate end-to-end data flow
- Hardware-in-the-loop when possible

### Performance Tests
- Measure inference latency
- Monitor power consumption
- Validate real-time constraints

## Code Quality

### Style Guidelines
- Follow Google C++ Style Guide (modified for embedded)
- Use clang-format for consistent formatting
- 100-character line limit
- Descriptive variable names

### Static Analysis
```bash
# Comprehensive analysis
cppcheck --enable=all src/ include/
clang-static-analyzer src/

# Security analysis
flawfinder src/
```

### Memory Safety
- Use RAII patterns
- Avoid dynamic allocation in embedded code
- Validate array bounds
- Initialize all variables

## Continuous Integration

### Local Pre-commit Checks
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Manual run
pre-commit run --all-files
```

### CI Pipeline
- Build for all target platforms
- Run comprehensive test suite
- Static analysis and security scanning
- Documentation generation
- Performance benchmarking

## Hardware Integration

### Simulation Environment
```bash
# Gazebo simulation
export GAZEBO_MODEL_PATH=$PWD/simulation/models
gazebo simulation/worlds/test_world.world

# SITL with PX4
make px4_sitl gazebo
```

### Real Hardware Testing
```bash
# Flash firmware (STM32)
make flash-stm32

# Flash PX4 module
make px4_upload

# Monitor system
make monitor
```

## Model Development

### Training Pipeline
```python
# Model training
python scripts/train_model.py \
    --dataset datasets/drone_navigation \
    --model_type liquid_cnn \
    --target_platform cortex_m7

# Model conversion
python scripts/convert_model.py \
    --input models/trained_model.onnx \
    --output models/embedded_model.lnn \
    --quantization int8
```

### Validation
```bash
# Model accuracy validation
python scripts/validate_model.py \
    --model models/embedded_model.lnn \
    --test_data datasets/validation

# Performance profiling
python scripts/profile_model.py \
    --model models/embedded_model.lnn \
    --platform cortex_m7
```

## Troubleshooting

### Common Build Issues
- **ARM toolchain not found**: Check PATH and installation
- **CMake configuration failed**: Verify all dependencies installed
- **Linker errors**: Check memory layout and symbol conflicts

### Runtime Issues  
- **Stack overflow**: Increase stack size or optimize recursion
- **Hard fault**: Enable fault handlers and check memory access
- **Performance issues**: Profile and optimize critical paths

## Performance Optimization

### Profiling Tools
```bash
# CPU profiling
perf record -g ./liquid_vision_sim
perf report

# Memory profiling  
valgrind --tool=massif ./liquid_vision_sim
```

### Optimization Techniques
- Use fixed-point arithmetic
- Optimize memory layout
- Minimize dynamic allocations
- Vectorize computations where possible

## Documentation

### API Documentation
```bash
# Generate Doxygen docs
doxygen Doxyfile
```

### User Documentation
- Keep README.md up to date
- Update tutorials for new features
- Document breaking changes in CHANGELOG.md

---

Happy coding! 🚀