#include "../../include/liquid_vision/utils/fixed_point.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace LiquidVision {

/**
 * Math utilities for embedded neural network processing
 */
namespace MathUtils {
    
    // Vector operations
    float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) return 0.0f;
        
        float result = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    void normalize_vector(std::vector<float>& vec) {
        float magnitude = 0.0f;
        for (float val : vec) {
            magnitude += val * val;
        }
        magnitude = std::sqrt(magnitude);
        
        if (magnitude > 1e-6f) {
            for (float& val : vec) {
                val /= magnitude;
            }
        }
    }
    
    // Activation functions optimized for embedded systems
    float fast_tanh(float x) {
        // Piecewise linear approximation of tanh
        if (x > 2.5f) return 1.0f;
        if (x < -2.5f) return -1.0f;
        
        // Linear approximation in middle range
        if (x > -1.0f && x < 1.0f) {
            return x;
        }
        
        // Use actual tanh for intermediate values
        return std::tanh(x);
    }
    
    float fast_sigmoid(float x) {
        // Fast sigmoid approximation
        if (x > 5.0f) return 1.0f;
        if (x < -5.0f) return 0.0f;
        
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    float relu(float x) {
        return std::max(0.0f, x);
    }
    
    float leaky_relu(float x, float alpha = 0.01f) {
        return x > 0.0f ? x : alpha * x;
    }
    
    // Statistical functions
    float mean(const std::vector<float>& data) {
        if (data.empty()) return 0.0f;
        
        float sum = 0.0f;
        for (float val : data) {
            sum += val;
        }
        return sum / data.size();
    }
    
    float variance(const std::vector<float>& data) {
        if (data.empty()) return 0.0f;
        
        float m = mean(data);
        float var = 0.0f;
        
        for (float val : data) {
            float diff = val - m;
            var += diff * diff;
        }
        
        return var / data.size();
    }
    
    float standard_deviation(const std::vector<float>& data) {
        return std::sqrt(variance(data));
    }
    
    // Matrix operations for small matrices
    std::vector<std::vector<float>> matrix_multiply(
        const std::vector<std::vector<float>>& A,
        const std::vector<std::vector<float>>& B) {
        
        if (A.empty() || B.empty() || A[0].size() != B.size()) {
            return std::vector<std::vector<float>>();
        }
        
        int rows_A = A.size();
        int cols_A = A[0].size();
        int cols_B = B[0].size();
        
        std::vector<std::vector<float>> C(rows_A, std::vector<float>(cols_B, 0.0f));
        
        for (int i = 0; i < rows_A; ++i) {
            for (int j = 0; j < cols_B; ++j) {
                for (int k = 0; k < cols_A; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return C;
    }
    
    // Clamp function for safety
    float clamp(float value, float min_val, float max_val) {
        return std::max(min_val, std::min(max_val, value));
    }
    
    // Linear interpolation
    float lerp(float a, float b, float t) {
        t = clamp(t, 0.0f, 1.0f);
        return a + t * (b - a);
    }
    
    // Exponential smoothing
    float exp_smooth(float current, float new_value, float alpha) {
        alpha = clamp(alpha, 0.0f, 1.0f);
        return alpha * new_value + (1.0f - alpha) * current;
    }
    
    // Fast square root approximation
    float fast_sqrt(float x) {
        if (x <= 0.0f) return 0.0f;
        
        // Newton-Raphson approximation
        float guess = x * 0.5f;
        for (int i = 0; i < 3; ++i) {
            guess = 0.5f * (guess + x / guess);
        }
        return guess;
    }
    
    // Fast inverse square root (Quake algorithm variant)
    float fast_inv_sqrt(float x) {
        if (x <= 0.0f) return 0.0f;
        return 1.0f / fast_sqrt(x);
    }
    
} // namespace MathUtils

} // namespace LiquidVision