#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

namespace LiquidVision {
namespace MathUtils {

/**
 * Mathematical utility functions for embedded systems
 */

// Fast approximations for transcendental functions
float fast_exp(float x);
float fast_tanh(float x);
float fast_sigmoid(float x);
float fast_sqrt(float x);

// Vector operations
std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b);
std::vector<float> vector_multiply(const std::vector<float>& a, float scalar);
float vector_dot(const std::vector<float>& a, const std::vector<float>& b);
float vector_norm(const std::vector<float>& v);
std::vector<float> vector_normalize(const std::vector<float>& v);

// Activation functions
float relu(float x);
float leaky_relu(float x, float alpha = 0.01f);
float swish(float x);
float gelu(float x);

// Numerical stability
float safe_log(float x, float epsilon = 1e-8f);
float clip(float x, float min_val, float max_val);
bool is_finite(float x);

// Statistics
float mean(const std::vector<float>& data);
float variance(const std::vector<float>& data);
float standard_deviation(const std::vector<float>& data);

// Interpolation
float linear_interpolate(float a, float b, float t);
float cubic_interpolate(float p0, float p1, float p2, float p3, float t);

// Random number generation (deterministic for embedded)
class FastRNG {
private:
    uint32_t state_;

public:
    explicit FastRNG(uint32_t seed = 12345) : state_(seed) {}
    
    uint32_t next();
    float uniform_float(float min_val = 0.0f, float max_val = 1.0f);
    float gaussian_float(float mean = 0.0f, float stddev = 1.0f);
};

// Memory-efficient matrix operations
class Matrix {
private:
    std::vector<float> data_;
    size_t rows_, cols_;

public:
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, const std::vector<float>& data);
    
    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;
    
    Matrix multiply(const Matrix& other) const;
    std::vector<float> multiply_vector(const std::vector<float>& vec) const;
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    void zero();
    void identity();
    void fill(float value);
};

} // namespace MathUtils
} // namespace LiquidVision