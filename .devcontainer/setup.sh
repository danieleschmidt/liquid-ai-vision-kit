#!/bin/bash

# Development container setup script for Liquid AI Vision Kit

set -e

echo "🚀 Setting up Liquid AI Vision Kit development environment..."

# Update package lists
sudo apt-get update

# Install essential development tools
echo "📦 Installing development tools..."
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    clang \
    clang-format \
    clang-tidy \
    cppcheck \
    valgrind \
    gdb \
    git-lfs \
    doxygen \
    graphviz \
    lcov \
    gcovr

# Install embedded development tools
echo "🔧 Installing embedded development tools..."
sudo apt-get install -y \
    gcc-arm-none-eabi \
    binutils-arm-none-eabi \
    openocd \
    stlink-tools \
    minicom \
    picocom

# Install Python dependencies for development tools
echo "🐍 Installing Python development tools..."
pip3 install --user \
    pytest \
    pytest-cov \
    black \
    pylint \
    mypy \
    sphinx \
    breathe \
    sphinx-rtd-theme \
    pre-commit \
    cmakelang

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# Create build directory with proper permissions
echo "🏗️ Setting up build environment..."
mkdir -p build
mkdir -p build-embedded
mkdir -p docs/_build

# Set up ccache for faster builds
echo "⚡ Configuring ccache..."
ccache --max-size=2G
ccache --set-config=compression=true

# Configure git if not already configured
if ! git config --get user.name > /dev/null; then
    echo "⚙️ Configuring git..."
    git config --global user.name "Dev Container User"
    git config --global user.email "dev@liquid-ai-vision-kit.local"
fi

# Set up git LFS
git lfs install

# Create useful aliases
echo "🔗 Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# Liquid AI Vision Kit Development Aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias build-debug='npm run build:debug'
alias build-release='npm run build:release'
alias build-embedded='npm run build:embedded'
alias test-all='npm run test'
alias lint-all='npm run lint'
alias format-all='npm run format'
alias clean-build='npm run clean'
alias docs-serve='npm run docs:serve'

# Quick development commands
alias dev='npm run dev'
alias check='npm run pre-commit'

# Git shortcuts
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline'

# Build and test shortcuts
alias bt='npm run build && npm run test'
alias btr='npm run build:release && npm run test'

# Documentation shortcuts
alias docs='npm run docs:build && npm run docs:serve'

# Embedded development
alias flash='openocd -f interface/stlink.cfg -f target/stm32h7x.cfg -c "program build-embedded/liquid_vision.elf verify reset exit"'
alias debug='arm-none-eabi-gdb build-embedded/liquid_vision.elf'

EOF

# Set up VSCode workspace settings
echo "🎨 Configuring VSCode workspace..."
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
    "cmake.buildDirectory": "${workspaceFolder}/build",
    "cmake.configureOnOpen": true,
    "cmake.buildBeforeRun": true,
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
    "C_Cpp.default.cStandard": "c17",
    "C_Cpp.default.cppStandard": "c++17",
    "C_Cpp.default.intelliSenseMode": "linux-gcc-x64",
    "C_Cpp.default.compilerPath": "/usr/bin/gcc",
    "C_Cpp.errorSquiggles": "enabled",
    "C_Cpp.autoAddFileAssociations": true,
    "editor.formatOnSave": true,
    "editor.formatOnPaste": true,
    "editor.insertSpaces": true,
    "editor.tabSize": 4,
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,
    "files.trimFinalNewlines": true,
    "python.defaultInterpreterPath": "/usr/local/bin/python3",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "markdownlint.config": {
        "MD013": false,
        "MD033": false
    },
    "cSpell.words": [
        "cortex",
        "embedded",
        "stlink",
        "openocd",
        "cmake",
        "ccache",
        "devcontainer",
        "liquid",
        "neural",
        "vision"
    ],
    "terminal.integrated.defaultProfile.linux": "bash",
    "git.autofetch": true,
    "git.enableSmartCommit": true,
    "search.exclude": {
        "**/build": true,
        "**/build-*": true,
        "**/.git": true,
        "**/node_modules": true,
        "**/*.tmp": true,
        "**/*.log": true
    }
}
EOF

# Create launch configurations for debugging
cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Tests",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/tests/test_runner",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "build-debug"
        },
        {
            "name": "Debug Main Application",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/liquid_vision_demo",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "build-debug"
        },
        {
            "name": "Cortex-Debug STM32H7",
            "type": "cortex-debug",
            "request": "launch",
            "servertype": "openocd",
            "cwd": "${workspaceFolder}",
            "executable": "${workspaceFolder}/build-embedded/liquid_vision.elf",
            "configFiles": [
                "interface/stlink.cfg",
                "target/stm32h7x.cfg"
            ],
            "svdFile": "${workspaceFolder}/docs/reference/STM32H7x3.svd",
            "runToMain": true,
            "showDevDebugOutput": true
        }
    ]
}
EOF

# Create tasks for VSCode
cat > .vscode/tasks.json << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build-debug",
            "type": "shell",
            "command": "npm",
            "args": ["run", "build:debug"],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "problemMatcher": "$gcc"
        },
        {
            "label": "build-release",
            "type": "shell",
            "command": "npm",
            "args": ["run", "build:release"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "problemMatcher": "$gcc"
        },
        {
            "label": "build-embedded",
            "type": "shell",
            "command": "npm",
            "args": ["run", "build:embedded"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "problemMatcher": "$gcc"
        },
        {
            "label": "test",
            "type": "shell",
            "command": "npm",
            "args": ["run", "test"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "dependsOn": "build-debug"
        },
        {
            "label": "clean",
            "type": "shell",
            "command": "npm",
            "args": ["run", "clean"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "lint",
            "type": "shell",
            "command": "npm",
            "args": ["run", "lint"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "format",
            "type": "shell",
            "command": "npm",
            "args": ["run", "format"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        }
    ]
}
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "  1. Run 'npm run dev' to build and test"
echo "  2. Use 'npm run docs:serve' to view documentation"
echo "  3. Start coding! The environment is ready."
echo ""
echo "🔧 Available commands:"
echo "  npm run build:debug    - Build debug version"
echo "  npm run build:release  - Build optimized release"
echo "  npm run build:embedded - Build for embedded target"
echo "  npm run test          - Run all tests"
echo "  npm run lint          - Check code style"
echo "  npm run format        - Format all code"
echo "  npm run dev           - Build debug + test"
echo "  npm run clean         - Clean build artifacts"
echo ""