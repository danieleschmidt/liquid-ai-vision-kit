#include "../tests/framework/simple_test.h"
#include "../include/liquid_vision/utils/fixed_point.h"
#include <cmath>

using namespace LiquidVision;

void test_fixed_point_construction() {
    FixedPoint<16, 16> fp1(3.14159f);
    FixedPoint<16, 16> fp2(42);
    FixedPoint<16, 16> fp3; // Default constructor
    
    ASSERT_FLOAT_NEAR(fp1.to_float(), 3.14159f, 0.001f);
    ASSERT_FLOAT_NEAR(fp2.to_float(), 42.0f, 0.001f);
    ASSERT_FLOAT_EQ(fp3.to_float(), 0.0f, 0.01f, "Default construction should be zero");
}

void test_fixed_point_arithmetic() {
    FixedPoint<16, 16> a(5.5f);
    FixedPoint<16, 16> b(2.25f);
    
    // Addition
    auto sum = a + b;
    ASSERT_FLOAT_NEAR(sum.to_float(), 7.75f, 0.001f);
    
    // Subtraction
    auto diff = a - b;
    ASSERT_FLOAT_NEAR(diff.to_float(), 3.25f, 0.001f);
    
    // Multiplication
    auto prod = a * b;
    ASSERT_FLOAT_NEAR(prod.to_float(), 12.375f, 0.001f);
    
    // Division
    auto quot = a / b;
    ASSERT_FLOAT_NEAR(quot.to_float(), 2.444f, 0.01f);
}

void test_fixed_point_precision() {
    // Test different precision settings
    FixedPoint<8, 24> high_frac(0.123456789f);
    FixedPoint<24, 8> high_int(12345.67f);
    
    // High fractional precision should preserve more decimal places
    ASSERT_FLOAT_NEAR(high_frac.to_float(), 0.123456789f, 0.00001f);
    
    // High integer precision should handle large numbers
    ASSERT_FLOAT_NEAR(high_int.to_float(), 12345.67f, 0.1f);
}

void test_fixed_point_overflow() {
    FixedPoint<8, 8> small_range(100.0f);
    
    // Should handle values within range
    ASSERT_FLOAT_NEAR(small_range.to_float(), 100.0f, 1.0f);
    
    // Test edge cases
    FixedPoint<8, 8> max_val(255.0f);
    FixedPoint<8, 8> min_val(-128.0f);
    
    ASSERT_GE(max_val.to_float(), 254.0f, "Max value should be >= 254");
    ASSERT_LE(min_val.to_float(), -127.0f, "Min value should be <= -127");
}

void test_fixed_point_comparison() {
    FixedPoint<16, 16> a(3.5f);
    FixedPoint<16, 16> b(3.5f);
    FixedPoint<16, 16> c(4.0f);
    
    ASSERT_TRUE(a == b, "Equal values should be equal");
    ASSERT_FALSE(a == c, "Unequal values should not be equal");
    ASSERT_TRUE(a < c, "Smaller value should be less than larger");
    ASSERT_TRUE(c > a, "Larger value should be greater than smaller");
    ASSERT_TRUE(a <= b, "Equal values should be less than or equal");
    ASSERT_TRUE(a >= b, "Equal values should be greater than or equal");
}

void test_fixed_point_saturation() {
    // Test saturation behavior on overflow
    FixedPoint<4, 4> tiny(-20.0f); // Should saturate
    FixedPoint<4, 4> large(20.0f);  // Should saturate
    
    // Values should be clamped to representable range
    ASSERT_GE(tiny.to_float(), -8.0f, "Saturated tiny value should be >= -8");   // Approximately minimum value
    ASSERT_LE(large.to_float(), 7.9f, "Saturated large value should be <= 7.9");   // Approximately maximum value
}

void test_fixed_point_performance() {
    const int iterations = 10000;
    
    FixedPoint<16, 16> a(1.5f);
    FixedPoint<16, 16> b(2.3f);
    FixedPoint<16, 16> result;
    
    // This should complete without timing out
    for (int i = 0; i < iterations; ++i) {
        result = a * b + a - b / a;
    }
    
    // Just verify the result is reasonable
    ASSERT_TRUE(std::isfinite(result.to_float()), "Performance test result should be finite");
}

int main() {
    RUN_TEST(test_fixed_point_construction);
    RUN_TEST(test_fixed_point_arithmetic);
    RUN_TEST(test_fixed_point_precision);
    RUN_TEST(test_fixed_point_overflow);
    RUN_TEST(test_fixed_point_comparison);
    RUN_TEST(test_fixed_point_saturation);
    RUN_TEST(test_fixed_point_performance);
    
    return 0;
}