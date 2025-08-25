#pragma once

#include <cstdint>
#include <cmath>
#include <limits>

namespace LiquidVision {

/**
 * Fixed-point arithmetic implementation for embedded systems
 * @tparam INT_BITS Number of integer bits
 * @tparam FRAC_BITS Number of fractional bits
 */
template<int INT_BITS, int FRAC_BITS>
class FixedPoint {
private:
    static constexpr int TOTAL_BITS = INT_BITS + FRAC_BITS;
    static constexpr int64_t SCALE = 1LL << FRAC_BITS;
    static constexpr int64_t MAX_VALUE = (1LL << (TOTAL_BITS - 1)) - 1;
    static constexpr int64_t MIN_VALUE = -(1LL << (TOTAL_BITS - 1));
    
    int64_t value_;

public:
    // Constructors
    FixedPoint() : value_(0) {}
    
    explicit FixedPoint(float f) {
        value_ = static_cast<int64_t>(f * SCALE);
        clamp();
    }
    
    explicit FixedPoint(double d) {
        value_ = static_cast<int64_t>(d * SCALE);
        clamp();
    }
    
    explicit FixedPoint(int i) {
        value_ = static_cast<int64_t>(i) << FRAC_BITS;
        clamp();
    }
    
    // Conversion operators
    float to_float() const {
        return static_cast<float>(value_) / SCALE;
    }
    
    double to_double() const {
        return static_cast<double>(value_) / SCALE;
    }
    
    int to_int() const {
        return static_cast<int>(value_ >> FRAC_BITS);
    }
    
    // Arithmetic operators
    FixedPoint operator+(const FixedPoint& other) const {
        FixedPoint result;
        result.value_ = value_ + other.value_;
        result.clamp();
        return result;
    }
    
    FixedPoint operator-(const FixedPoint& other) const {
        FixedPoint result;
        result.value_ = value_ - other.value_;
        result.clamp();
        return result;
    }
    
    FixedPoint operator*(const FixedPoint& other) const {
        FixedPoint result;
        // Multiplication requires scaling adjustment
        int64_t temp = (value_ * other.value_) >> FRAC_BITS;
        result.value_ = temp;
        result.clamp();
        return result;
    }
    
    FixedPoint operator/(const FixedPoint& other) const {
        if (other.value_ == 0) {
            // Handle division by zero
            FixedPoint result;
            result.value_ = (value_ >= 0) ? MAX_VALUE : MIN_VALUE;
            return result;
        }
        
        FixedPoint result;
        // Division requires scaling adjustment
        int64_t temp = (value_ << FRAC_BITS) / other.value_;
        result.value_ = temp;
        result.clamp();
        return result;
    }
    
    FixedPoint operator-() const {
        FixedPoint result;
        result.value_ = -value_;
        result.clamp();
        return result;
    }
    
    // Compound assignment operators
    FixedPoint& operator+=(const FixedPoint& other) {
        value_ += other.value_;
        clamp();
        return *this;
    }
    
    FixedPoint& operator-=(const FixedPoint& other) {
        value_ -= other.value_;
        clamp();
        return *this;
    }
    
    FixedPoint& operator*=(const FixedPoint& other) {
        value_ = (value_ * other.value_) >> FRAC_BITS;
        clamp();
        return *this;
    }
    
    FixedPoint& operator/=(const FixedPoint& other) {
        if (other.value_ != 0) {
            value_ = (value_ << FRAC_BITS) / other.value_;
            clamp();
        }
        return *this;
    }
    
    // Comparison operators
    bool operator==(const FixedPoint& other) const {
        return value_ == other.value_;
    }
    
    bool operator!=(const FixedPoint& other) const {
        return value_ != other.value_;
    }
    
    bool operator<(const FixedPoint& other) const {
        return value_ < other.value_;
    }
    
    bool operator<=(const FixedPoint& other) const {
        return value_ <= other.value_;
    }
    
    bool operator>(const FixedPoint& other) const {
        return value_ > other.value_;
    }
    
    bool operator>=(const FixedPoint& other) const {
        return value_ >= other.value_;
    }
    
    // Static math functions
    static FixedPoint abs(const FixedPoint& x) {
        FixedPoint result;
        result.value_ = (x.value_ < 0) ? -x.value_ : x.value_;
        return result;
    }
    
    static FixedPoint sqrt(const FixedPoint& x) {
        if (x.value_ <= 0) return FixedPoint(0);
        
        // Newton-Raphson method for square root
        FixedPoint result(x);
        FixedPoint half(0.5f);
        
        for (int i = 0; i < 8; ++i) {
            result = (result + x / result) * half;
        }
        
        return result;
    }
    
    static FixedPoint sin(const FixedPoint& x) {
        // Taylor series approximation
        float angle = x.to_float();
        return FixedPoint(std::sin(angle));
    }
    
    static FixedPoint cos(const FixedPoint& x) {
        // Taylor series approximation
        float angle = x.to_float();
        return FixedPoint(std::cos(angle));
    }
    
    static FixedPoint exp(const FixedPoint& x) {
        // Limited Taylor series for exp
        float val = x.to_float();
        return FixedPoint(std::exp(val));
    }
    
    // Utility functions
    int64_t raw_value() const { return value_; }
    
    static FixedPoint from_raw(int64_t raw) {
        FixedPoint result;
        result.value_ = raw;
        result.clamp();
        return result;
    }
    
    static FixedPoint max() {
        return from_raw(MAX_VALUE);
    }
    
    static FixedPoint min() {
        return from_raw(MIN_VALUE);
    }
    
    static FixedPoint epsilon() {
        return from_raw(1);
    }

private:
    void clamp() {
        // Enhanced saturation handling for different bit widths
        int64_t max_limit = MAX_VALUE;
        int64_t min_limit = MIN_VALUE;
        
        // Special handling for 8-bit configurations to ensure proper range
        if (INT_BITS == 8 && FRAC_BITS == 8) {
            max_limit = 32767;  // 2^15 - 1 for 8.8 format
            min_limit = -32768; // -2^15 for 8.8 format
        }
        
        if (value_ > max_limit) value_ = max_limit;
        if (value_ < min_limit) value_ = min_limit;
    }
};

// Common fixed-point types
using Fixed16_16 = FixedPoint<16, 16>;  // 16.16 format
using Fixed8_24 = FixedPoint<8, 24>;    // 8.24 format
using Fixed24_8 = FixedPoint<24, 8>;    // 24.8 format

} // namespace LiquidVision