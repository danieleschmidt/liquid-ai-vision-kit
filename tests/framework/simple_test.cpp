#include "simple_test.h"
#include <iostream>
#include <vector>
#include <chrono>

namespace SimpleTest {

static std::vector<TestCase*> test_cases;
static TestStats global_stats;

void register_test(TestCase* test_case) {
    test_cases.push_back(test_case);
}

int run_all_tests() {
    std::cout << "Starting test execution..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (auto* test : test_cases) {
        std::cout << "Running " << test->name << "... ";
        
        auto test_start = std::chrono::high_resolution_clock::now();
        
        try {
            test->run();
            auto test_end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                test_end - test_start).count();
            
            std::cout << "PASSED (" << duration << "μs)" << std::endl;
            global_stats.passed++;
            
        } catch (const AssertionFailure& e) {
            std::cout << "FAILED" << std::endl;
            std::cout << "  " << e.what() << std::endl;
            global_stats.failed++;
            
        } catch (const std::exception& e) {
            std::cout << "ERROR" << std::endl;
            std::cout << "  Unexpected exception: " << e.what() << std::endl;
            global_stats.failed++;
        }
        
        global_stats.total++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    std::cout << std::endl;
    std::cout << "Test Results:" << std::endl;
    std::cout << "  Total: " << global_stats.total << std::endl;
    std::cout << "  Passed: " << global_stats.passed << std::endl;
    std::cout << "  Failed: " << global_stats.failed << std::endl;
    std::cout << "  Duration: " << total_duration << "ms" << std::endl;
    
    return global_stats.failed == 0 ? 0 : 1;
}

void assert_true(bool condition, const std::string& message, 
                const char* file, int line) {
    if (!condition) {
        throw AssertionFailure("Assertion failed at " + std::string(file) + 
                              ":" + std::to_string(line) + " - " + message);
    }
}

void assert_false(bool condition, const std::string& message,
                 const char* file, int line) {
    assert_true(!condition, message, file, line);
}

void assert_equal(int expected, int actual, const std::string& message,
                 const char* file, int line) {
    if (expected != actual) {
        throw AssertionFailure("Assertion failed at " + std::string(file) + 
                              ":" + std::to_string(line) + " - " + message +
                              " (expected: " + std::to_string(expected) + 
                              ", actual: " + std::to_string(actual) + ")");
    }
}

void assert_equal(float expected, float actual, float tolerance,
                 const std::string& message, const char* file, int line) {
    float diff = std::abs(expected - actual);
    if (diff > tolerance) {
        throw AssertionFailure("Assertion failed at " + std::string(file) + 
                              ":" + std::to_string(line) + " - " + message +
                              " (expected: " + std::to_string(expected) + 
                              ", actual: " + std::to_string(actual) + 
                              ", tolerance: " + std::to_string(tolerance) + ")");
    }
}

} // namespace SimpleTest