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
        if (value_ > MAX_VALUE) value_ = MAX_VALUE;
        if (value_ < MIN_VALUE) value_ = MIN_VALUE;
    }
};

// Common fixed-point types
using Fixed16_16 = FixedPoint<16, 16>;  // 16.16 format
using Fixed8_24 = FixedPoint<8, 24>;    // 8.24 format
using Fixed24_8 = FixedPoint<24, 8>;    // 24.8 format

/**
 * Fast integer-only math functions for embedded systems
 */
class FastMath {
public:
    // Fast integer square root using binary search
    static uint32_t isqrt(uint32_t n) {
        if (n < 2) return n;
        
        uint32_t left = 1;
        uint32_t right = n / 2 + 1;
        uint32_t result = 0;
        
        while (left <= right) {
            uint32_t mid = (left + right) / 2;
            uint32_t mid_squared = mid * mid;
            
            if (mid_squared == n) {
                return mid;
            } else if (mid_squared < n) {
                left = mid + 1;
                result = mid;
            } else {
                right = mid - 1;
            }
        }
        
        return result;
    }
    
    // Fast approximation of atan2 using CORDIC
    static int32_t iatan2(int32_t y, int32_t x) {
        const int32_t ANGLE_TABLE[16] = {
            8192, 4836, 2555, 1297, 651, 326, 163, 81,
            41, 20, 10, 5, 3, 1, 1, 0
        };
        
        int32_t angle = 0;
        int32_t xi = x;
        int32_t yi = y;
        
        // Handle quadrants
        if (x < 0) {
            if (y >= 0) {
                angle = 32768; // PI
            } else {
                angle = -32768; // -PI
            }
            xi = -xi;
            yi = -yi;
        }
        
        // CORDIC iterations
        for (int i = 0; i < 16; ++i) {
            int32_t xi_new, yi_new;
            
            if (yi > 0) {
                xi_new = xi + (yi >> i);
                yi_new = yi - (xi >> i);
                angle += ANGLE_TABLE[i];
            } else {
                xi_new = xi - (yi >> i);
                yi_new = yi + (xi >> i);
                angle -= ANGLE_TABLE[i];
            }
            
            xi = xi_new;
            yi = yi_new;
        }
        
        return angle;
    }
    
    // Fast sine approximation using lookup table
    static int32_t isin(int32_t angle) {
        // Normalize angle to [0, 65536) representing [0, 2*PI)
        while (angle < 0) angle += 65536;
        while (angle >= 65536) angle -= 65536;
        
        // Use symmetry to reduce to first quadrant
        int32_t quadrant = angle / 16384;
        int32_t remainder = angle % 16384;
        
        // Simple linear approximation for demonstration
        // In production, use a lookup table
        int32_t result;
        
        switch (quadrant) {
            case 0: // 0 to PI/2
                result = (remainder * 32767) / 16384;
                break;
            case 1: // PI/2 to PI
                result = ((16384 - remainder) * 32767) / 16384;
                break;
            case 2: // PI to 3*PI/2
                result = -(remainder * 32767) / 16384;
                break;
            case 3: // 3*PI/2 to 2*PI
                result = -((16384 - remainder) * 32767) / 16384;
                break;
            default:
                result = 0;
        }
        
        return result;
    }
    
    // Fast cosine approximation
    static int32_t icos(int32_t angle) {
        return isin(angle + 16384); // cos(x) = sin(x + PI/2)
    }
    
    // Fast exponential approximation
    static int32_t iexp(int32_t x, int shift) {
        // exp(x) ≈ (1 + x/n)^n for large n
        // Using n = 2^shift for efficient computation
        
        int32_t result = (1 << shift) + x;
        
        for (int i = 0; i < shift; ++i) {
            result = (result * result) >> shift;
        }
        
        return result;
    }
};

/**
 * Q-format fixed point for DSP operations
 */
template<int Q>
class QFormat {
private:
    int32_t value_;
    static constexpr int32_t SCALE = 1 << Q;
    
public:
    QFormat() : value_(0) {}
    explicit QFormat(float f) : value_(static_cast<int32_t>(f * SCALE)) {}
    
    float to_float() const {
        return static_cast<float>(value_) / SCALE;
    }
    
    QFormat operator+(const QFormat& other) const {
        QFormat result;
        result.value_ = saturate_add(value_, other.value_);
        return result;
    }
    
    QFormat operator*(const QFormat& other) const {
        QFormat result;
        int64_t temp = static_cast<int64_t>(value_) * other.value_;
        result.value_ = static_cast<int32_t>(temp >> Q);
        return result;
    }
    
private:
    static int32_t saturate_add(int32_t a, int32_t b) {
        int64_t result = static_cast<int64_t>(a) + b;
        
        if (result > INT32_MAX) return INT32_MAX;
        if (result < INT32_MIN) return INT32_MIN;
        
        return static_cast<int32_t>(result);
    }
};

using Q15 = QFormat<15>;  // Q15 format for audio DSP
using Q31 = QFormat<31>;  // Q31 format for high precision

} // namespace LiquidVision