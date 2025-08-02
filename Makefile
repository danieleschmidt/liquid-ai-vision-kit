# Makefile for Liquid AI Vision Kit
# Provides convenient targets for common development tasks

# Configuration
PROJECT_NAME := liquid_ai_vision_kit
VERSION := $(shell git describe --tags --always --dirty)
BUILD_DIR := build
BUILD_ARM_DIR := build-arm
BUILD_EMBEDDED_DIR := build-embedded

# Platform configuration
TARGET_PLATFORM ?= X86_SIMULATION
CMAKE_BUILD_TYPE ?= Debug

# Docker configuration
DOCKER_REGISTRY ?= ghcr.io
DOCKER_IMAGE := $(DOCKER_REGISTRY)/liquid-ai-vision-kit
DOCKER_TAG ?= $(VERSION)

# Compiler configuration
CC ?= gcc
CXX ?= g++
ARM_CC := arm-none-eabi-gcc
ARM_CXX := arm-none-eabi-g++

# Parallel build configuration
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
MAKEFLAGS += -j$(NPROC)

# Color output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

.PHONY: all build build-debug build-release build-embedded clean test test-unit test-integration \
        test-performance format lint install uninstall docker docker-build docker-push \
        docs help benchmark flash debug size analyze coverage

# Default target
all: build test

# Help target
help:
	@echo "$(BLUE)Liquid AI Vision Kit Build System$(RESET)"
	@echo ""
	@echo "$(GREEN)Main targets:$(RESET)"
	@echo "  all             - Build and test (default)"
	@echo "  build           - Build for current platform"
	@echo "  build-debug     - Build debug version"
	@echo "  build-release   - Build optimized release"
	@echo "  build-embedded  - Build for embedded target"
	@echo "  test            - Run all tests"
	@echo "  clean           - Clean build artifacts"
	@echo ""
	@echo "$(GREEN)Testing targets:$(RESET)"
	@echo "  test-unit       - Run unit tests only"
	@echo "  test-integration- Run integration tests"
	@echo "  test-performance- Run performance benchmarks"
	@echo "  coverage        - Generate code coverage report"
	@echo ""
	@echo "$(GREEN)Code quality targets:$(RESET)"
	@echo "  format          - Format source code"
	@echo "  lint            - Run static analysis"
	@echo "  analyze         - Run detailed static analysis"
	@echo ""
	@echo "$(GREEN)Embedded targets:$(RESET)"
	@echo "  flash           - Flash firmware to target"
	@echo "  debug           - Start debugging session"
	@echo "  size            - Show firmware size analysis"
	@echo ""
	@echo "$(GREEN)Docker targets:$(RESET)"
	@echo "  docker          - Build all Docker images"
	@echo "  docker-build    - Build development Docker image"
	@echo "  docker-push     - Push images to registry"
	@echo ""
	@echo "$(GREEN)Documentation targets:$(RESET)"
	@echo "  docs            - Build documentation"
	@echo ""
	@echo "$(GREEN)Configuration:$(RESET)"
	@echo "  TARGET_PLATFORM=$(TARGET_PLATFORM)"
	@echo "  CMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)"
	@echo "  VERSION=$(VERSION)"

# Build targets
build: $(BUILD_DIR)/Makefile
	@echo "$(BLUE)Building $(PROJECT_NAME) for $(TARGET_PLATFORM)...$(RESET)"
	$(MAKE) -C $(BUILD_DIR)
	@echo "$(GREEN)Build completed successfully$(RESET)"

build-debug: CMAKE_BUILD_TYPE=Debug
build-debug: build

build-release: CMAKE_BUILD_TYPE=Release
build-release: build

build-embedded: TARGET_PLATFORM=ARM_CORTEX_M7
build-embedded: CMAKE_BUILD_TYPE=MinSizeRel
build-embedded: $(BUILD_EMBEDDED_DIR)/Makefile
	@echo "$(BLUE)Building embedded firmware...$(RESET)"
	$(MAKE) -C $(BUILD_EMBEDDED_DIR)
	@echo "$(GREEN)Embedded build completed$(RESET)"

# CMake configuration
$(BUILD_DIR)/Makefile:
	@echo "$(BLUE)Configuring build for $(TARGET_PLATFORM)...$(RESET)"
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DTARGET_PLATFORM=$(TARGET_PLATFORM) \
		-DENABLE_TESTS=ON \
		-DENABLE_BENCHMARKS=ON

$(BUILD_EMBEDDED_DIR)/Makefile:
	@echo "$(BLUE)Configuring embedded build...$(RESET)"
	mkdir -p $(BUILD_EMBEDDED_DIR)
	cd $(BUILD_EMBEDDED_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DTARGET_PLATFORM=$(TARGET_PLATFORM) \
		-DCMAKE_TOOLCHAIN_FILE=../cmake/arm-none-eabi.cmake \
		-DENABLE_TESTS=OFF \
		-DENABLE_BENCHMARKS=OFF

# Test targets
test: build
	@echo "$(BLUE)Running all tests...$(RESET)"
	cd $(BUILD_DIR) && ctest --output-on-failure --parallel $(NPROC)
	@echo "$(GREEN)All tests completed$(RESET)"

test-unit: build
	@echo "$(BLUE)Running unit tests...$(RESET)"
	cd $(BUILD_DIR) && ctest --output-on-failure -R "test_.*" -E "(integration|e2e|performance)"

test-integration: build
	@echo "$(BLUE)Running integration tests...$(RESET)"
	cd $(BUILD_DIR) && ctest --output-on-failure -R ".*integration.*"

test-performance: build
	@echo "$(BLUE)Running performance tests...$(RESET)"
	cd $(BUILD_DIR) && ctest --output-on-failure -R ".*performance.*"

benchmark: build-release
	@echo "$(BLUE)Running benchmarks...$(RESET)"
	$(BUILD_DIR)/liquid_vision_benchmark

# Coverage analysis
coverage: CMAKE_BUILD_TYPE=Debug
coverage: CXXFLAGS += --coverage
coverage: build test
	@echo "$(BLUE)Generating coverage report...$(RESET)"
	cd $(BUILD_DIR) && \
	lcov --directory . --capture --output-file coverage.info && \
	lcov --remove coverage.info '/usr/*' '*/tests/*' '*/build/*' --output-file coverage_filtered.info && \
	genhtml coverage_filtered.info --output-directory coverage_html
	@echo "$(GREEN)Coverage report generated in $(BUILD_DIR)/coverage_html/$(RESET)"

# Code quality targets
format:
	@echo "$(BLUE)Formatting source code...$(RESET)"
	find src include tests -name "*.cpp" -o -name "*.hpp" -o -name "*.h" | \
		xargs clang-format -i
	@echo "$(GREEN)Code formatting completed$(RESET)"

lint:
	@echo "$(BLUE)Running static analysis...$(RESET)"
	npm run lint
	@echo "$(GREEN)Linting completed$(RESET)"

analyze: build
	@echo "$(BLUE)Running detailed static analysis...$(RESET)"
	cd $(BUILD_DIR) && \
	cppcheck --enable=all --inconclusive --force --xml \
		--output-file=cppcheck-report.xml \
		../src ../include
	clang-tidy src/**/*.cpp include/**/*.hpp \
		-p $(BUILD_DIR) \
		-- -std=c++17
	@echo "$(GREEN)Static analysis completed$(RESET)"

# Embedded development targets
flash: build-embedded
	@echo "$(BLUE)Flashing firmware to target...$(RESET)"
	openocd -f interface/stlink.cfg -f target/stm32h7x.cfg \
		-c "program $(BUILD_EMBEDDED_DIR)/liquid_vision_core.elf verify reset exit"
	@echo "$(GREEN)Firmware flashed successfully$(RESET)"

debug: build-embedded
	@echo "$(BLUE)Starting debugging session...$(RESET)"
	arm-none-eabi-gdb $(BUILD_EMBEDDED_DIR)/liquid_vision_core.elf

size: build-embedded
	@echo "$(BLUE)Firmware size analysis:$(RESET)"
	arm-none-eabi-size $(BUILD_EMBEDDED_DIR)/liquid_vision_core.elf
	@echo ""
	@echo "$(BLUE)Section breakdown:$(RESET)"
	arm-none-eabi-objdump -h $(BUILD_EMBEDDED_DIR)/liquid_vision_core.elf

# Docker targets
docker: docker-build

docker-build:
	@echo "$(BLUE)Building Docker images...$(RESET)"
	docker build --target development -t $(DOCKER_IMAGE):dev-$(DOCKER_TAG) .
	docker build --target simulation -t $(DOCKER_IMAGE):sim-$(DOCKER_TAG) .
	docker build --target embedded -t $(DOCKER_IMAGE):embedded-$(DOCKER_TAG) .
	@echo "$(GREEN)Docker images built successfully$(RESET)"

docker-push: docker-build
	@echo "$(BLUE)Pushing Docker images...$(RESET)"
	docker push $(DOCKER_IMAGE):dev-$(DOCKER_TAG)
	docker push $(DOCKER_IMAGE):sim-$(DOCKER_TAG)
	docker push $(DOCKER_IMAGE):embedded-$(DOCKER_TAG)
	@echo "$(GREEN)Docker images pushed successfully$(RESET)"

docker-run-dev:
	@echo "$(BLUE)Starting development container...$(RESET)"
	docker run -it --rm \
		-v $(PWD):/workspace \
		-p 8080:8080 \
		$(DOCKER_IMAGE):dev-$(DOCKER_TAG)

# Documentation targets
docs:
	@echo "$(BLUE)Building documentation...$(RESET)"
	doxygen Doxyfile
	npm run docs:build
	@echo "$(GREEN)Documentation built successfully$(RESET)"
	@echo "Open build/docs/html/index.html to view"

docs-serve: docs
	@echo "$(BLUE)Serving documentation on http://localhost:8080$(RESET)"
	npm run docs:serve

# Installation targets
install: build-release
	@echo "$(BLUE)Installing $(PROJECT_NAME)...$(RESET)"
	cd $(BUILD_DIR) && make install
	@echo "$(GREEN)Installation completed$(RESET)"

uninstall:
	@echo "$(BLUE)Uninstalling $(PROJECT_NAME)...$(RESET)"
	cd $(BUILD_DIR) && make uninstall
	@echo "$(GREEN)Uninstallation completed$(RESET)"

# Utility targets
clean:
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf $(BUILD_DIR) $(BUILD_ARM_DIR) $(BUILD_EMBEDDED_DIR)
	rm -rf docs/_build coverage_html *.log
	find . -name "*.o" -o -name "*.so" -o -name "*.a" -o -name "*.elf" -o -name "*.bin" | xargs rm -f
	@echo "$(GREEN)Clean completed$(RESET)"

distclean: clean
	@echo "$(BLUE)Deep cleaning...$(RESET)"
	rm -rf .cache node_modules __pycache__ .pytest_cache
	git clean -fdx
	@echo "$(GREEN)Deep clean completed$(RESET)"

# Development convenience targets
dev: build-debug test-unit

quick: CMAKE_BUILD_TYPE=Debug
quick: build

release: CMAKE_BUILD_TYPE=Release
release: build test

package: build-release
	@echo "$(BLUE)Creating distribution package...$(RESET)"
	cd $(BUILD_DIR) && cpack
	@echo "$(GREEN)Package created successfully$(RESET)"

# Pre-commit checks
pre-commit: format lint test
	@echo "$(GREEN)Pre-commit checks passed$(RESET)"

# CI/CD targets
ci-build: CMAKE_BUILD_TYPE=Release
ci-build: build test coverage analyze

ci-embedded: build-embedded size

# Print configuration
config:
	@echo "$(BLUE)Build Configuration:$(RESET)"
	@echo "  Project: $(PROJECT_NAME)"
	@echo "  Version: $(VERSION)"
	@echo "  Platform: $(TARGET_PLATFORM)"
	@echo "  Build Type: $(CMAKE_BUILD_TYPE)"
	@echo "  Build Dir: $(BUILD_DIR)"
	@echo "  Parallel Jobs: $(NPROC)"
	@echo "  Docker Image: $(DOCKER_IMAGE):$(DOCKER_TAG)"

# Dependency checks
deps:
	@echo "$(BLUE)Checking dependencies...$(RESET)"
	@command -v cmake >/dev/null 2>&1 || { echo "$(RED)cmake not found$(RESET)"; exit 1; }
	@command -v $(CXX) >/dev/null 2>&1 || { echo "$(RED)$(CXX) not found$(RESET)"; exit 1; }
	@command -v git >/dev/null 2>&1 || { echo "$(RED)git not found$(RESET)"; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo "$(RED)python3 not found$(RESET)"; exit 1; }
	@echo "$(GREEN)All dependencies found$(RESET)"

# Show build environment
env:
	@echo "$(BLUE)Build Environment:$(RESET)"
	@echo "  OS: $$(uname -s)"
	@echo "  Arch: $$(uname -m)"
	@echo "  Kernel: $$(uname -r)"
	@echo "  Shell: $$SHELL"
	@echo "  Make: $$(make --version | head -1)"
	@echo "  CMake: $$(cmake --version | head -1)"
	@echo "  GCC: $$($(CXX) --version | head -1)"
	@echo "  Python: $$(python3 --version)"
	@echo "  Git: $$(git --version)"

# Watch for changes and rebuild
watch:
	@echo "$(BLUE)Watching for changes...$(RESET)"
	@while inotifywait -r -e modify,create,delete src include tests; do \
		echo "$(YELLOW)Changes detected, rebuilding...$(RESET)"; \
		make build; \
	done