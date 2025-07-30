# Contributing to Liquid AI Vision Kit

Thank you for your interest in contributing to the Liquid AI Vision Kit! This guide will help you get started with development and contributing to this embedded AI project.

## 🚀 Quick Start

### Prerequisites

- **GCC ARM Toolchain**: For embedded development
- **CMake 3.16+**: Build system
- **Python 3.8+**: For model conversion and tooling
- **Git**: Version control
- **Docker**: For containerized builds (optional)

### Development Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/liquid-ai-vision-kit.git
   cd liquid-ai-vision-kit
   ```

2. **Install dependencies**:
   ```bash
   # Install ARM toolchain
   sudo apt-get install gcc-arm-none-eabi
   
   # Install Python dependencies
   pip install -r requirements-dev.txt
   ```

3. **Build the project**:
   ```bash
   mkdir build && cd build
   cmake .. -DTARGET_PLATFORM=X86_SIMULATION
   make -j$(nproc)
   ```

## 📋 Development Workflow

### Before Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Run existing tests**:
   ```bash
   make test
   ```

### Making Changes

1. **Follow coding standards** (see [Coding Standards](#coding-standards))
2. **Write tests** for new functionality
3. **Update documentation** as needed
4. **Test on target platforms** when possible

### Submitting Changes

1. **Run the full test suite**:
   ```bash
   make test-all
   make benchmark
   ```

2. **Format your code**:
   ```bash
   make format
   ```

3. **Create a pull request** with:
   - Clear description of changes
   - Reference to related issues
   - Test results on different platforms

## 🎯 Priority Areas for Contributions

- **Additional Flight Controller Support** (ArduPilot, Betaflight)
- **More Pre-trained Models** (navigation, tracking, detection)
- **Hardware Accelerator Integration** (GPU, NPU, FPGA)
- **ROS2 Packages** and integration
- **Power Optimization** techniques
- **Documentation** and tutorials

## 📏 Coding Standards

### C++ Guidelines

- **C++17** standard
- **Google C++ Style Guide** with modifications for embedded
- **Fixed-point arithmetic** for MCU compatibility
- **Memory-efficient** data structures
- **Real-time constraints** awareness

### Example Code Style

```cpp
// Use snake_case for variables and functions
int neuron_count = 128;
void process_frame(const Frame& input);

// Use PascalCase for classes
class LiquidNeuron {
private:
    // Member variables with trailing underscore
    FixedPoint<16, 16> state_;
    const LiquidParams* params_;
    
public:
    // Clear, documented interfaces
    void update(const InputVector& input, float timestep);
    FixedPoint<16, 16> get_output() const;
};
```

### Python Guidelines

- **PEP 8** style guide
- **Type hints** for all functions
- **Docstrings** in Google format
- **Unit tests** with pytest

## 🧪 Testing

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Component interaction testing
3. **Hardware-in-Loop Tests**: Real embedded platform testing
4. **Simulation Tests**: Gazebo/simulation environment testing
5. **Performance Tests**: Latency, power, accuracy benchmarks

### Running Tests

```bash
# All tests
make test-all

# Specific test categories
make test-unit
make test-integration
make test-hil

# Performance benchmarks
make benchmark
```

## 📦 Pull Request Guidelines

### PR Requirements

- [ ] **Tests pass** on all supported platforms
- [ ] **Code follows style guidelines**
- [ ] **Documentation updated** if needed
- [ ] **Changelog entry** added (if applicable)
- [ ] **Performance impact** assessed

### PR Description Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Tested on hardware: [platform]
- [ ] Performance impact measured

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
```

## 🛠️ Development Tools

### Recommended IDEs

- **VS Code** with C++ and CMake extensions
- **CLion** for advanced debugging
- **Vim/Neovim** with language server support

### Useful Commands

```bash
# Format code
make format

# Static analysis
make analyze

# Generate documentation
make docs

# Cross-compile for ARM
make arm-cross-compile

# Flash to hardware
make flash-px4
```

## 🐛 Bug Reports

### Before Reporting

1. **Search existing issues**
2. **Test on latest version**
3. **Reproduce consistently**

### Bug Report Template

- **Platform**: Target hardware/OS
- **Compiler**: Version and flags
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **Debug logs** (if available)

## 📚 Documentation

- **Code comments** for complex algorithms
- **API documentation** in header files
- **Tutorial updates** for new features
- **Performance characteristics** for optimizations

## 🚁 Safety and Testing

⚠️ **Important**: This software controls drones and robots. Always:

- **Test in simulation first**
- **Use fail-safe mechanisms**
- **Follow local regulations**
- **Never skip hardware validation**

## 📞 Getting Help

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Check [docs/](docs/) directory
- **Community**: Join our developer chat

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Academic papers citing this work (with permission)

---

**Thank you for helping make embedded AI more accessible for robotics!** 🤖✨