#include "test_helpers.h"
#include "liquid_vision/utils/fixed_point.h"

#ifdef USING_GTEST
#include <gtest/gtest.h>
#elif defined(USING_CATCH2)
#include <catch2/catch_test_macros.hpp>
#else
#include "simple_test.h"
#endif

using namespace LiquidVision;
using namespace LiquidVision::Testing;

// Test fixture for fixed-point arithmetic
class FixedPointTest {
public:
    static void testBasicArithmetic() {
        // Test basic construction and conversion
        FixedPoint<16, 16> a(1.5f);
        FixedPoint<16, 16> b(2.25f);
        
        // Test addition
        auto sum = a + b;
        float sum_float = sum.toFloat();
        ASSERT_FLOAT_EQ(3.75f, sum_float, TestHelpers::EMBEDDED_TOLERANCE);
        
        // Test subtraction
        auto diff = b - a;
        float diff_float = diff.toFloat();
        ASSERT_FLOAT_EQ(0.75f, diff_float, TestHelpers::EMBEDDED_TOLERANCE);
        
        // Test multiplication
        auto prod = a * b;
        float prod_float = prod.toFloat();
        ASSERT_FLOAT_EQ(3.375f, prod_float, TestHelpers::EMBEDDED_TOLERANCE);
        
        // Test division
        auto quot = b / a;
        float quot_float = quot.toFloat();
        ASSERT_FLOAT_EQ(1.5f, quot_float, TestHelpers::EMBEDDED_TOLERANCE);
    }
    
    static void testRangeAndPrecision() {
        // Test range limits
        FixedPoint<16, 16> max_val(32767.0f);
        FixedPoint<16, 16> min_val(-32768.0f);
        
        ASSERT_FLOAT_EQ(32767.0f, max_val.toFloat(), 1.0f);
        ASSERT_FLOAT_EQ(-32768.0f, min_val.toFloat(), 1.0f);
        
        // Test precision
        FixedPoint<16, 16> precise(1.0f / 65536.0f); // Smallest representable value
        ASSERT_FLOAT_EQ(1.0f / 65536.0f, precise.toFloat(), 2.0f / 65536.0f);
    }
    
    static void testOverflowHandling() {
        // Test overflow protection
        FixedPoint<16, 16> large(30000.0f);
        FixedPoint<16, 16> small(2.0f);
        
        // This should trigger overflow protection
        auto result = large + large; // Would overflow without protection
        // Result should be clamped to maximum value
        ASSERT_FLOAT_EQ(32767.0f, result.toFloat(), 1.0f);
    }
    
    static void testNegativeNumbers() {
        FixedPoint<16, 16> neg_a(-1.5f);
        FixedPoint<16, 16> pos_b(2.25f);
        
        // Test negative arithmetic
        auto sum = neg_a + pos_b;
        ASSERT_FLOAT_EQ(0.75f, sum.toFloat(), TestHelpers::EMBEDDED_TOLERANCE);
        
        auto prod = neg_a * pos_b;
        ASSERT_FLOAT_EQ(-3.375f, prod.toFloat(), TestHelpers::EMBEDDED_TOLERANCE);
    }
    
    static void testComparisonOperators() {
        FixedPoint<16, 16> a(1.5f);
        FixedPoint<16, 16> b(2.25f);
        FixedPoint<16, 16> c(1.5f);
        
        // Test equality
        if (!(a == c)) {
            throw std::runtime_error("Equality test failed");
        }
        if (a == b) {
            throw std::runtime_error("Inequality test failed");
        }
        
        // Test ordering
        if (!(a < b)) {
            throw std::runtime_error("Less than test failed");
        }
        if (!(b > a)) {
            throw std::runtime_error("Greater than test failed");
        }
    }
    
    static void testMathematicalFunctions() {
        FixedPoint<16, 16> angle(0.5f); // ~28.6 degrees
        
        // Test trigonometric functions (if implemented)
        // auto sin_result = sin(angle);
        // auto cos_result = cos(angle);
        
        // Test square root (if implemented)
        FixedPoint<16, 16> value(4.0f);
        // auto sqrt_result = sqrt(value);
        // ASSERT_FLOAT_EQ(2.0f, sqrt_result.toFloat(), TestHelpers::EMBEDDED_TOLERANCE);
    }
    
    static void testConversionAccuracy() {
        // Test round-trip conversion accuracy
        std::vector<float> test_values = {
            0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 3.14159f, -2.71828f,
            0.001f, -0.001f, 100.0f, -100.0f, 1000.0f, -1000.0f
        };
        
        for (float value : test_values) {
            if (std::abs(value) < 32767.0f) { // Within range
                FixedPoint<16, 16> fixed(value);
                float converted = fixed.toFloat();
                
                float error = std::abs(value - converted);
                float tolerance = std::max(TestHelpers::EMBEDDED_TOLERANCE, 
                                         std::abs(value) * 1e-4f);
                
                if (error > tolerance) {
                    throw std::runtime_error("Conversion accuracy test failed");
                }
            }
        }
    }
    
    static void testPerformance() {
        const int iterations = 10000;
        FixedPoint<16, 16> a(1.5f);
        FixedPoint<16, 16> b(2.25f);
        
        TestHelpers::Timer timer;
        timer.start();
        
        // Perform many operations
        for (int i = 0; i < iterations; ++i) {
            auto result = a * b + a - b;
            (void)result; // Suppress unused variable warning
        }
        
        double elapsed = timer.elapsed();
        
        // Performance should be much faster than floating point on embedded systems
        // This is a placeholder - actual thresholds would depend on target platform
        if (elapsed > 100.0) { // 100ms for 10k operations seems reasonable
            throw std::runtime_error("Performance test failed");
        }
    }
};

// Framework-specific test implementations
#ifdef USING_GTEST

TEST(FixedPointTest, BasicArithmetic) {
    FixedPointTest::testBasicArithmetic();
}

TEST(FixedPointTest, RangeAndPrecision) {
    FixedPointTest::testRangeAndPrecision();
}

TEST(FixedPointTest, OverflowHandling) {
    FixedPointTest::testOverflowHandling();
}

TEST(FixedPointTest, NegativeNumbers) {
    FixedPointTest::testNegativeNumbers();
}

TEST(FixedPointTest, ComparisonOperators) {
    FixedPointTest::testComparisonOperators();
}

TEST(FixedPointTest, MathematicalFunctions) {
    FixedPointTest::testMathematicalFunctions();
}

TEST(FixedPointTest, ConversionAccuracy) {
    FixedPointTest::testConversionAccuracy();
}

TEST(FixedPointTest, Performance) {
    FixedPointTest::testPerformance();
}

#elif defined(USING_CATCH2)

TEST_CASE("FixedPoint Basic Arithmetic", "[fixed_point]") {
    FixedPointTest::testBasicArithmetic();
}

TEST_CASE("FixedPoint Range and Precision", "[fixed_point]") {
    FixedPointTest::testRangeAndPrecision();
}

TEST_CASE("FixedPoint Overflow Handling", "[fixed_point]") {
    FixedPointTest::testOverflowHandling();
}

TEST_CASE("FixedPoint Negative Numbers", "[fixed_point]") {
    FixedPointTest::testNegativeNumbers();
}

TEST_CASE("FixedPoint Comparison Operators", "[fixed_point]") {
    FixedPointTest::testComparisonOperators();
}

TEST_CASE("FixedPoint Mathematical Functions", "[fixed_point]") {
    FixedPointTest::testMathematicalFunctions();
}

TEST_CASE("FixedPoint Conversion Accuracy", "[fixed_point]") {
    FixedPointTest::testConversionAccuracy();
}

TEST_CASE("FixedPoint Performance", "[fixed_point]") {
    FixedPointTest::testPerformance();
}

#else

// Simple test framework implementation
int main() {
    TestHelpers::setupTestEnvironment();
    
    int tests_passed = 0;
    int tests_failed = 0;
    
    std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"Basic Arithmetic", FixedPointTest::testBasicArithmetic},
        {"Range and Precision", FixedPointTest::testRangeAndPrecision},
        {"Overflow Handling", FixedPointTest::testOverflowHandling},
        {"Negative Numbers", FixedPointTest::testNegativeNumbers},
        {"Comparison Operators", FixedPointTest::testComparisonOperators},
        {"Mathematical Functions", FixedPointTest::testMathematicalFunctions},
        {"Conversion Accuracy", FixedPointTest::testConversionAccuracy},
        {"Performance", FixedPointTest::testPerformance}
    };
    
    for (const auto& test : tests) {
        try {
            test.second();
            printf("PASS: %s\n", test.first.c_str());
            tests_passed++;
        } catch (const std::exception& e) {
            printf("FAIL: %s - %s\n", test.first.c_str(), e.what());
            tests_failed++;
        }
    }
    
    printf("\nTest Results: %d passed, %d failed\n", tests_passed, tests_failed);
    
    TestHelpers::teardownTestEnvironment();
    
    return tests_failed == 0 ? 0 : 1;
}

#endif