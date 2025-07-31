# CI/CD Workflow Setup

This document outlines the recommended GitHub Actions workflows for the Liquid AI Vision Kit project.

## Overview

The CI/CD pipeline is designed for embedded AI systems with multi-platform builds, security scanning, and safety-critical testing requirements.

## Required Workflows

### 1. Main CI Pipeline

**File**: `.github/workflows/ci.yml`

**Purpose**: Build, test, and validate changes across all supported platforms

**Key Features**:
- Multi-platform builds (x86_64, ARM Cortex-M7, PX4)
- Comprehensive test suite with coverage reporting
- Static analysis and security scanning
- Performance benchmarking
- Documentation generation

**Required Secrets**:
- `CODECOV_TOKEN`: For coverage reporting
- `SONAR_TOKEN`: For code quality analysis (optional)

### 2. Security Scanning

**File**: `.github/workflows/security.yml`

**Purpose**: Comprehensive security analysis for safety-critical systems

**Key Features**:
- CodeQL analysis for C++ vulnerabilities
- Dependency vulnerability scanning
- Container security scanning
- SLSA compliance verification

**Required Permissions**:
- `security-events: write` for CodeQL uploads

### 3. Release Automation

**File**: `.github/workflows/release.yml`

**Purpose**: Automated release management with embedded artifacts

**Key Features**:
- Semantic versioning
- Cross-platform binary builds
- Model artifact packaging
- GitHub release creation
- Documentation deployment

**Required Secrets**:
- `GITHUB_TOKEN`: For release creation (automatically provided)

### 4. Hardware-in-Loop Testing

**File**: `.github/workflows/hil-testing.yml`

**Purpose**: Real hardware validation on self-hosted runners

**Key Features**:
- PX4 SITL simulation testing
- Hardware-specific performance validation
- Flight controller integration tests

**Requirements**:
- Self-hosted runners with PX4 simulation environment
- Hardware test fixtures (when available)

## Workflow Templates

### Basic CI Template

```yaml
name: CI Pipeline
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    strategy:
      matrix:
        platform: [ubuntu-latest, macos-latest]
        target: [X86_SIMULATION, ARM_CORTEX_M7]
    
    runs-on: ${{ matrix.platform }}
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Build Environment
      run: |
        sudo apt-get update
        sudo apt-get install -y gcc-arm-none-eabi cmake
    
    - name: Configure CMake
      run: |
        cmake -B build -DTARGET_PLATFORM=${{ matrix.target }}
    
    - name: Build
      run: cmake --build build --parallel
    
    - name: Test
      run: cd build && ctest --output-on-failure
```

### Security Scanning Template

```yaml
name: Security Analysis
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly scans

jobs:
  codeql-analysis:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: cpp
    
    - name: Build for Analysis
      run: |
        cmake -B build -DTARGET_PLATFORM=X86_SIMULATION
        cmake --build build
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3
```

## Setup Instructions

### 1. Create Workflow Files

Since I cannot create GitHub Actions files directly, you'll need to manually create:

```bash
mkdir -p .github/workflows
```

Then create the workflow files using the templates above.

### 2. Configure Branch Protection

Recommended branch protection rules for `main`:

- Require pull request reviews (2 reviewers minimum)
- Require status checks to pass before merging
- Required status checks:
  - `build-and-test (ubuntu-latest, X86_SIMULATION)`
  - `build-and-test (ubuntu-latest, ARM_CORTEX_M7)`
  - `codeql-analysis`
- Require branches to be up to date before merging
- Include administrators in restrictions

### 3. Required Repository Settings

**Secrets** (Repository Settings → Secrets and variables → Actions):
- `CODECOV_TOKEN`: Get from codecov.io
- `SONAR_TOKEN`: Get from SonarCloud (optional)

**Environment Variables**:
- `CMAKE_BUILD_TYPE`: Release
- `TARGET_PLATFORMS`: X86_SIMULATION,ARM_CORTEX_M7,PX4

### 4. Self-Hosted Runners (Optional)

For hardware-in-loop testing, set up self-hosted runners with:

- PX4 development environment
- Physical hardware test setups
- Secure network isolation for safety

## Integration with External Services

### Code Coverage (Codecov)

Add to your CI workflow:

```yaml
- name: Upload Coverage
  uses: codecov/codecov-action@v4
  with:
    token: ${{ secrets.CODECOV_TOKEN }}
    file: ./build/coverage.xml
```

### Code Quality (SonarCloud)

```yaml
- name: SonarCloud Scan
  uses: SonarSource/sonarcloud-github-action@master
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

## Performance Monitoring

Track key metrics in CI:

- Build times by platform
- Test execution times
- Binary size changes
- Memory usage on embedded targets
- Inference latency benchmarks

## Safety Considerations

For safety-critical embedded systems:

1. **Mandatory Reviews**: Never merge without human review
2. **Hardware Validation**: Require HIL testing for control changes
3. **Rollback Capability**: Maintain ability to quickly revert
4. **Documentation**: Update safety documentation with changes
5. **Compliance**: Ensure regulatory compliance (when applicable)

## Troubleshooting

### Common Issues

**ARM Toolchain Not Found**:
```yaml
- name: Install ARM Toolchain
  run: |
    sudo apt-get update
    sudo apt-get install -y gcc-arm-none-eabi
    arm-none-eabi-gcc --version
```

**CMake Configuration Fails**:
```yaml
- name: Debug CMake
  run: |
    cmake --version
    cmake -B build -DTARGET_PLATFORM=X86_SIMULATION --debug-output
```

**Tests Timeout on Embedded Builds**:
```yaml
- name: Run Tests with Timeout
  run: |
    cd build
    timeout 300 ctest --output-on-failure
```

## Migration Guide

### From Manual Builds

1. Start with basic CI for simulation builds
2. Add ARM cross-compilation gradually
3. Integrate hardware testing last
4. Monitor build times and optimize

### Adding New Platforms

1. Update matrix strategy in workflows
2. Add platform-specific setup steps
3. Update test configurations
4. Verify all combinations work

## Maintenance

- Review and update workflows quarterly
- Monitor for security advisories on actions
- Update toolchain versions regularly
- Performance tune build times as needed

---

**Note**: This setup prioritizes safety and reliability for embedded AI systems. Always test workflows thoroughly before deploying to production environments.