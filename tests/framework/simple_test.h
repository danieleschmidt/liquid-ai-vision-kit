#pragma once

#include <string>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace SimpleTest {

/**
 * @brief Exception thrown when an assertion fails
 */
class AssertionFailure : public std::runtime_error {
public:
    explicit AssertionFailure(const std::string& message) 
        : std::runtime_error(message) {}
};

/**
 * @brief Test statistics
 */
struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
};

/**
 * @brief Register a test case
 */
void register_test(class TestCase* test_case);

/**
 * @brief Base class for test cases
 */
class TestCase {
public:
    TestCase(const std::string& test_name) : name(test_name) {
        register_test(this);
    }
    
    virtual ~TestCase() = default;
    virtual void run() = 0;
    
    std::string name;
};

/**
 * @brief Run all registered tests
 * @return 0 if all tests passed, 1 if any failed
 */
int run_all_tests();

/**
 * @brief Assert that a condition is true
 */
void assert_true(bool condition, const std::string& message,
                const char* file, int line);

/**
 * @brief Assert that a condition is false
 */
void assert_false(bool condition, const std::string& message,
                 const char* file, int line);

/**
 * @brief Assert that two integers are equal
 */
void assert_equal(int expected, int actual, const std::string& message,
                 const char* file, int line);

/**
 * @brief Assert that two floats are approximately equal
 */
void assert_equal(float expected, float actual, float tolerance,
                 const std::string& message, const char* file, int line);

} // namespace SimpleTest

// Convenience macros
#define ASSERT_TRUE(condition, message) \
    SimpleTest::assert_true(condition, message, __FILE__, __LINE__)

#define ASSERT_FALSE(condition, message) \
    SimpleTest::assert_false(condition, message, __FILE__, __LINE__)

#define ASSERT_EQ(expected, actual, message) \
    SimpleTest::assert_equal(expected, actual, message, __FILE__, __LINE__)

#define ASSERT_FLOAT_EQ(expected, actual, tolerance, message) \
    SimpleTest::assert_equal(expected, actual, tolerance, message, __FILE__, __LINE__)

#define ASSERT_FLOAT_NEAR(expected, actual, tolerance) \
    SimpleTest::assert_equal(expected, actual, tolerance, "Values should be near", __FILE__, __LINE__)

#define ASSERT_GE(actual, expected, message) \
    SimpleTest::assert_true((actual) >= (expected), message, __FILE__, __LINE__)

#define RUN_TEST(test_func) \
    do { \
        try { \
            test_func(); \
            std::cout << "PASSED: " << #test_func << std::endl; \
        } catch (const std::exception& e) { \
            std::cout << "FAILED: " << #test_func << " - " << e.what() << std::endl; \
        } \
    } while(0)

#define TEST_CLASS(class_name, test_name) \
    class class_name : public SimpleTest::TestCase { \
    public: \
        class_name() : TestCase(test_name) {} \
        void run() override; \
    }; \
    static class_name class_name##_instance; \
    void class_name::run()

#define TEST_MAIN() \
    int main() { \
        return SimpleTest::run_all_tests(); \
    }