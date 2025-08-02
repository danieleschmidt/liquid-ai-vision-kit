# Test configuration for Liquid AI Vision Kit
cmake_minimum_required(VERSION 3.16)

# Test configuration options
option(ENABLE_UNIT_TESTS "Build unit tests" ON)
option(ENABLE_INTEGRATION_TESTS "Build integration tests" ON)
option(ENABLE_E2E_TESTS "Build end-to-end tests" OFF)
option(ENABLE_PERFORMANCE_TESTS "Build performance tests" ON)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(ENABLE_VALGRIND "Enable valgrind memory checking" OFF)
option(ENABLE_SANITIZERS "Enable sanitizers" OFF)

# Code coverage setup
if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
        
        # Find coverage tools
        find_program(GCOV_PATH gcov)
        find_program(LCOV_PATH lcov)
        find_program(GENHTML_PATH genhtml)
        
        if(GCOV_PATH AND LCOV_PATH AND GENHTML_PATH)
            # Coverage target
            add_custom_target(coverage
                COMMAND ${CMAKE_COMMAND} -E make_directory coverage
                COMMAND ${LCOV_PATH} --directory . --zerocounters
                COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                COMMAND ${LCOV_PATH} --directory . --capture --output-file coverage/app.info
                COMMAND ${LCOV_PATH} --remove coverage/app.info 
                    '*/tests/*' '*/build/*' '/usr/*' '*/third_party/*'
                    --output-file coverage/app_filtered.info
                COMMAND ${GENHTML_PATH} coverage/app_filtered.info --output-directory coverage/html
                COMMAND ${CMAKE_COMMAND} -E echo "Coverage report generated in coverage/html/index.html"
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMENT "Generating code coverage report"
            )
        endif()
    endif()
endif()

# Sanitizer setup
if(ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(SANITIZER_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_FLAGS}")
    endif()
endif()

# Test framework selection
set(TEST_FRAMEWORK "auto" CACHE STRING "Test framework to use (auto, gtest, catch2, simple)")

# Find or configure test framework
if(TEST_FRAMEWORK STREQUAL "auto" OR TEST_FRAMEWORK STREQUAL "gtest")
    find_package(GTest QUIET)
    if(GTest_FOUND)
        set(USING_GTEST TRUE)
        message(STATUS "Using Google Test framework")
    endif()
endif()

if(NOT USING_GTEST AND (TEST_FRAMEWORK STREQUAL "auto" OR TEST_FRAMEWORK STREQUAL "catch2"))
    find_package(Catch2 QUIET)
    if(Catch2_FOUND)
        set(USING_CATCH2 TRUE)
        message(STATUS "Using Catch2 framework")
    endif()
endif()

if(NOT USING_GTEST AND NOT USING_CATCH2)
    # Use simple test framework
    set(USING_SIMPLE_TEST TRUE)
    message(STATUS "Using simple test framework")
    
    add_library(simple_test STATIC 
        framework/simple_test.cpp
        framework/simple_test.h
    )
    
    target_include_directories(simple_test PUBLIC framework)
    
    # Mock framework for embedded testing
    add_library(test_mocks STATIC
        mocks/mock_hardware.cpp
        mocks/mock_camera.cpp
        mocks/mock_flight_controller.cpp
    )
    
    target_include_directories(test_mocks PUBLIC mocks)
    target_link_libraries(test_mocks PUBLIC liquid_vision_core)
endif()

# Test utilities library
add_library(test_utils STATIC
    utils/test_helpers.cpp
    utils/performance_timer.cpp
    utils/memory_tracker.cpp
    utils/data_generator.cpp
)

target_include_directories(test_utils PUBLIC utils)
target_link_libraries(test_utils PUBLIC liquid_vision_core)

if(USING_GTEST)
    target_link_libraries(test_utils PUBLIC GTest::gtest)
elseif(USING_CATCH2)
    target_link_libraries(test_utils PUBLIC Catch2::Catch2)
elseif(USING_SIMPLE_TEST)
    target_link_libraries(test_utils PUBLIC simple_test)
endif()

# Test data management
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/test_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/fixtures/test_config.h
)

# Function to add a test with common configuration
function(add_liquid_test test_name test_source)
    add_executable(${test_name} ${test_source})
    
    target_include_directories(${test_name} PRIVATE 
        ${CMAKE_CURRENT_BINARY_DIR}/fixtures
    )
    
    target_link_libraries(${test_name} PRIVATE 
        liquid_vision_core
        test_utils
    )
    
    if(USING_GTEST)
        target_link_libraries(${test_name} PRIVATE GTest::gtest_main)
    elseif(USING_CATCH2)
        target_link_libraries(${test_name} PRIVATE Catch2::Catch2WithMain)
    elseif(USING_SIMPLE_TEST)
        target_link_libraries(${test_name} PRIVATE simple_test test_mocks)
    endif()
    
    # Add test to CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
    
    # Set test properties
    set_tests_properties(${test_name} PROPERTIES
        TIMEOUT 30
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    # Add valgrind target if enabled
    if(ENABLE_VALGRIND AND VALGRIND_EXECUTABLE)
        add_test(NAME ${test_name}_valgrind 
            COMMAND ${VALGRIND_EXECUTABLE} 
                --tool=memcheck 
                --leak-check=full 
                --show-reachable=yes 
                --error-exitcode=1
                $<TARGET_FILE:${test_name}>
        )
        
        set_tests_properties(${test_name}_valgrind PROPERTIES
            TIMEOUT 120
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endfunction()

# Unit tests
if(ENABLE_UNIT_TESTS)
    message(STATUS "Building unit tests")
    
    # Core component tests
    add_liquid_test(test_liquid_neuron unit/test_liquid_neuron.cpp)
    add_liquid_test(test_liquid_network unit/test_liquid_network.cpp)
    add_liquid_test(test_ode_solver unit/test_ode_solver.cpp)
    add_liquid_test(test_fixed_point unit/test_fixed_point.cpp)
    add_liquid_test(test_math_utils unit/test_math_utils.cpp)
    
    # Vision component tests
    add_liquid_test(test_image_processor unit/test_image_processor.cpp)
    add_liquid_test(test_feature_extractor unit/test_feature_extractor.cpp)
    
    # Control component tests
    add_liquid_test(test_flight_controller unit/test_flight_controller.cpp)
    
    # Safety and validation tests
    add_liquid_test(test_safety_manager unit/test_safety_manager.cpp)
    add_liquid_test(test_parameter_validation unit/test_parameter_validation.cpp)
endif()

# Integration tests
if(ENABLE_INTEGRATION_TESTS)
    message(STATUS "Building integration tests")
    
    add_liquid_test(test_vision_pipeline integration/test_vision_pipeline.cpp)
    add_liquid_test(test_control_loop integration/test_control_loop.cpp)
    add_liquid_test(test_model_loading integration/test_model_loading.cpp)
    add_liquid_test(test_hardware_interfaces integration/test_hardware_interfaces.cpp)
    add_liquid_test(test_memory_management integration/test_memory_management.cpp)
endif()

# End-to-end tests
if(ENABLE_E2E_TESTS)
    message(STATUS "Building end-to-end tests")
    
    add_liquid_test(test_full_system e2e/test_full_system.cpp)
    add_liquid_test(test_simulation_integration e2e/test_simulation_integration.cpp)
    add_liquid_test(test_real_hardware e2e/test_real_hardware.cpp)
endif()

# Performance tests
if(ENABLE_PERFORMANCE_TESTS)
    message(STATUS "Building performance tests")
    
    add_liquid_test(test_inference_performance performance/test_inference_performance.cpp)
    add_liquid_test(test_memory_usage performance/test_memory_usage.cpp)
    add_liquid_test(test_power_consumption performance/test_power_consumption.cpp)
    add_liquid_test(test_real_time_constraints performance/test_real_time_constraints.cpp)
    
    # Benchmark executable
    add_executable(liquid_vision_benchmark
        performance/benchmark_main.cpp
        performance/benchmark_liquid_network.cpp
        performance/benchmark_vision_pipeline.cpp
        performance/benchmark_ode_solver.cpp
    )
    
    target_link_libraries(liquid_vision_benchmark PRIVATE
        liquid_vision_core
        test_utils
    )
    
    # Benchmark data generation
    add_custom_target(generate_benchmark_data
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_test_data.py
            --output ${CMAKE_CURRENT_BINARY_DIR}/fixtures/benchmark_data
        COMMENT "Generating benchmark test data"
    )
endif()

# Custom test targets
add_custom_target(test_unit
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R "test_.*" -E "(integration|e2e|performance)"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running unit tests"
)

add_custom_target(test_integration
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R ".*integration.*"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running integration tests"
)

add_custom_target(test_performance
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R ".*performance.*"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running performance tests"
)

add_custom_target(test_all
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests"
)

# Test reporting
add_custom_target(test_report
    COMMAND ${CMAKE_COMMAND} -E echo "Generating test report..."
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --output-junit test_results.xml
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating test report"
)

# Memory testing with valgrind
if(ENABLE_VALGRIND)
    find_program(VALGRIND_EXECUTABLE valgrind)
    if(VALGRIND_EXECUTABLE)
        add_custom_target(test_memory
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R ".*_valgrind"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running memory tests with Valgrind"
        )
    endif()
endif()

# Installation of test utilities
if(INSTALL_TESTS)
    install(TARGETS test_utils
        EXPORT liquid_vision_test_targets
        ARCHIVE DESTINATION lib
    )
    
    install(DIRECTORY fixtures/
        DESTINATION share/liquid_vision/test_fixtures
        FILES_MATCHING PATTERN "*.dat" PATTERN "*.json" PATTERN "*.bin"
    )
endif()

# Print test configuration summary
message(STATUS "")
message(STATUS "Test Configuration:")
message(STATUS "  Unit tests: ${ENABLE_UNIT_TESTS}")
message(STATUS "  Integration tests: ${ENABLE_INTEGRATION_TESTS}")
message(STATUS "  E2E tests: ${ENABLE_E2E_TESTS}")
message(STATUS "  Performance tests: ${ENABLE_PERFORMANCE_TESTS}")
message(STATUS "  Code coverage: ${ENABLE_COVERAGE}")
message(STATUS "  Memory checking: ${ENABLE_VALGRIND}")
message(STATUS "  Sanitizers: ${ENABLE_SANITIZERS}")
if(USING_GTEST)
    message(STATUS "  Test framework: Google Test")
elseif(USING_CATCH2)
    message(STATUS "  Test framework: Catch2")
else()
    message(STATUS "  Test framework: Simple Test")
endif()
message(STATUS "")