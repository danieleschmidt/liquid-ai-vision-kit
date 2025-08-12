#include "../../include/liquid_vision/utils/fixed_point.h"
#include <cmath>

namespace LiquidVision {

// Fixed point implementation is all in the header as templates
// This file provides any non-template utility functions

namespace FixedPointUtils {
    
    // Utility function to convert floating point arrays to fixed point
    template<int INT_BITS, int FRAC_BITS>
    void convert_array(const float* input, FixedPoint<INT_BITS, FRAC_BITS>* output, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            output[i] = FixedPoint<INT_BITS, FRAC_BITS>(input[i]);
        }
    }
    
    // Utility function to convert fixed point arrays back to floating point
    template<int INT_BITS, int FRAC_BITS>
    void convert_array(const FixedPoint<INT_BITS, FRAC_BITS>* input, float* output, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            output[i] = input[i].to_float();
        }
    }
    
    // Fast approximate math functions for fixed point
    Fixed16_16 fast_exp(const Fixed16_16& x) {
        // Fast exponential approximation using bit manipulation
        float val = x.to_float();
        if (val < -10.0f) return Fixed16_16(0.0f);
        if (val > 10.0f) return Fixed16_16(22026.0f); // e^10
        
        return Fixed16_16(std::exp(val));
    }
    
    Fixed16_16 fast_tanh(const Fixed16_16& x) {
        // Fast tanh approximation
        float val = x.to_float();
        if (val > 5.0f) return Fixed16_16(1.0f);
        if (val < -5.0f) return Fixed16_16(-1.0f);
        
        return Fixed16_16(std::tanh(val));
    }
    
    Fixed16_16 fast_sigmoid(const Fixed16_16& x) {
        // Fast sigmoid approximation
        float val = x.to_float();
        return Fixed16_16(1.0f / (1.0f + std::exp(-val)));
    }
}

} // namespace LiquidVision