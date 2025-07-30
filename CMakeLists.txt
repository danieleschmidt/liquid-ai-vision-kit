cmake_minimum_required(VERSION 3.16)

# Project configuration
project(liquid_ai_vision_kit
    VERSION 1.0.0
    DESCRIPTION "Liquid Neural Networks for embedded vision systems"
    LANGUAGES CXX C
)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform configuration
option(TARGET_PLATFORM "Target platform" "X86_SIMULATION")
option(ENABLE_TESTS "Build tests" ON)
option(ENABLE_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_PYTHON_BINDINGS "Build Python bindings" OFF)

# Platform-specific settings
if(TARGET_PLATFORM STREQUAL "PX4")
    include(cmake/px4.cmake)
    set(EMBEDDED_BUILD ON)
elseif(TARGET_PLATFORM STREQUAL "ARM_CORTEX_M7")
    include(cmake/cortex_m7.cmake)
    set(EMBEDDED_BUILD ON)
elseif(TARGET_PLATFORM STREQUAL "X86_SIMULATION")
    set(EMBEDDED_BUILD OFF)
else()
    message(FATAL_ERROR "Unsupported target platform: ${TARGET_PLATFORM}")
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

if(EMBEDDED_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Find dependencies
if(NOT EMBEDDED_BUILD)
    find_package(Threads REQUIRED)
    
    # Optional dependencies for simulation
    find_package(OpenCV QUIET)
    if(OpenCV_FOUND)
        add_definitions(-DHAVE_OPENCV)
    endif()
    
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(MAVLINK mavlink)
        if(MAVLINK_FOUND)
            add_definitions(-DHAVE_MAVLINK)
        endif()
    endif()
endif()

# Source files
set(LIQUID_VISION_SOURCES
    src/core/liquid_neuron.cpp
    src/core/liquid_network.cpp
    src/core/ode_solver.cpp
    src/vision/image_processor.cpp
    src/vision/feature_extractor.cpp
    src/control/flight_controller.cpp
    src/utils/fixed_point.cpp
    src/utils/math_utils.cpp
)

set(LIQUID_VISION_HEADERS
    include/liquid_vision/core/liquid_neuron.h
    include/liquid_vision/core/liquid_network.h
    include/liquid_vision/core/ode_solver.h
    include/liquid_vision/vision/image_processor.h
    include/liquid_vision/vision/feature_extractor.h
    include/liquid_vision/control/flight_controller.h
    include/liquid_vision/utils/fixed_point.h
    include/liquid_vision/utils/math_utils.h
    include/liquid_vision/lnn_controller.hpp
)

# Main library
add_library(liquid_vision_core STATIC
    ${LIQUID_VISION_SOURCES}
    ${LIQUID_VISION_HEADERS}
)

target_include_directories(liquid_vision_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Platform-specific linking
if(NOT EMBEDDED_BUILD)
    target_link_libraries(liquid_vision_core PRIVATE Threads::Threads)
    
    if(OpenCV_FOUND)
        target_link_libraries(liquid_vision_core PRIVATE ${OpenCV_LIBS})
    endif()
endif()

# Executable targets
if(NOT EMBEDDED_BUILD)
    # Simulation executable
    add_executable(liquid_vision_sim
        src/simulation/main.cpp
        src/simulation/simulator.cpp
    )
    
    target_link_libraries(liquid_vision_sim PRIVATE liquid_vision_core)
    
    # Benchmark executable
    if(ENABLE_BENCHMARKS)
        add_executable(liquid_vision_benchmark
            src/benchmark/main.cpp
            src/benchmark/performance_tests.cpp
        )
        
        target_link_libraries(liquid_vision_benchmark PRIVATE liquid_vision_core)
    endif()
endif()

# Tests
if(ENABLE_TESTS AND NOT EMBEDDED_BUILD)
    enable_testing()
    
    # Find or build a simple test framework
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        # Use a simple test framework for embedded compatibility
        add_library(simple_test STATIC tests/framework/simple_test.cpp)
        target_include_directories(simple_test PUBLIC tests/framework)
    endif()
    
    # Test executables
    set(TEST_SOURCES
        tests/test_liquid_neuron.cpp
        tests/test_ode_solver.cpp
        tests/test_fixed_point.cpp
        tests/test_image_processor.cpp
    )
    
    foreach(test_source ${TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        
        if(GTest_FOUND)
            target_link_libraries(${test_name} PRIVATE 
                liquid_vision_core 
                GTest::gtest_main
            )
        else()
            target_link_libraries(${test_name} PRIVATE 
                liquid_vision_core 
                simple_test
            )
        endif()
        
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# Python bindings
if(ENABLE_PYTHON_BINDINGS AND NOT EMBEDDED_BUILD)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        pybind11_add_module(liquid_vision_py 
            src/python/bindings.cpp
        )
        
        target_link_libraries(liquid_vision_py PRIVATE liquid_vision_core)
    endif()
endif()

# Custom targets
add_custom_target(format
    COMMAND find ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include 
            -name "*.cpp" -o -name "*.h" -o -name "*.hpp" | 
            xargs clang-format -i
    COMMENT "Formatting source code"
)

add_custom_target(analyze
    COMMAND cppcheck --enable=all --suppress=missingIncludeSystem 
            ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include
    COMMENT "Running static analysis"
)

# Installation
install(TARGETS liquid_vision_core
    EXPORT liquid_vision_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Export package
install(EXPORT liquid_vision_targets
    FILE liquid_vision_targets.cmake
    NAMESPACE liquid_vision::
    DESTINATION lib/cmake/liquid_vision
)

# Create package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    liquid_vision_config_version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/liquid_vision_config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/liquid_vision_config.cmake
    INSTALL_DESTINATION lib/cmake/liquid_vision
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/liquid_vision_config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/liquid_vision_config_version.cmake
    DESTINATION lib/cmake/liquid_vision
)

# Print configuration summary
message(STATUS "")
message(STATUS "Liquid AI Vision Kit Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Target platform: ${TARGET_PLATFORM}")
message(STATUS "  Embedded build: ${EMBEDDED_BUILD}")
message(STATUS "  Tests enabled: ${ENABLE_TESTS}")
message(STATUS "  Benchmarks enabled: ${ENABLE_BENCHMARKS}")
message(STATUS "  Python bindings: ${ENABLE_PYTHON_BINDINGS}")
message(STATUS "")