#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <functional>
#include <cmath>

namespace LiquidVision {
namespace Testing {

// Test data structures
struct TestNeuronParameters {
    float tau;      // Time constant
    float A, B, C;  // ODE parameters
    float sigma;    // Noise level
    float mu;       // Bias
    float W_in;     // Input weight
    float W_rec;    // Recurrent weight
    float bias;     // Neuron bias
};

struct TestImageData {
    int width;
    int height;
    int channels;
    std::vector<uint8_t> data;
};

struct TestControlOutput {
    float forward_velocity;
    float lateral_velocity;
    float yaw_rate;
    float altitude_rate;
    float confidence;
    uint64_t timestamp_us;
};

struct PerformanceMetrics {
    double min_time;
    double max_time;
    double avg_time;
    double total_time;
    double std_dev;
    size_t memory_usage;
    int iterations;
};

class TestHelpers {
public:
    // Random data generation
    static std::vector<float> generateRandomFloats(size_t count, float min = -1.0f, float max = 1.0f);
    static std::vector<uint8_t> generateRandomImage(int width, int height, int channels = 3);
    
    // Comparison utilities
    static bool areFloatsEqual(float a, float b, float tolerance = 1e-6f);
    static bool areArraysEqual(const float* a, const float* b, size_t count, float tolerance = 1e-6f);
    static bool areVectorsEqual(const std::vector<float>& a, const std::vector<float>& b, float tolerance = 1e-6f);
    
    // File I/O utilities
    static bool loadTestData(const std::string& filename, std::vector<float>& data);
    static bool saveTestData(const std::string& filename, const std::vector<float>& data);
    static std::string readTextFile(const std::string& filename);
    
    // Timing utilities
    class Timer {
    public:
        void start();
        double elapsed() const; // Returns elapsed time in milliseconds
        
    private:
        std::chrono::high_resolution_clock::time_point start_time;
    };
    
    // Memory utilities
    static size_t getCurrentMemoryUsage();
    static size_t getPeakMemoryUsage();
    
    // Test data generators
    static TestNeuronParameters generateTestNeuronParams();
    static TestImageData generateTestImage(int width = 160, int height = 120);
    static TestControlOutput generateTestControlOutput();
    
    // Performance testing utilities
    static PerformanceMetrics measureInferencePerformance(
        std::function<void()> inference_func, 
        int iterations = 100
    );
    
    // Validation utilities
    static bool validateNeuronOutput(float output, float expected, float tolerance = 1e-3f);
    static bool validateControlOutput(const TestControlOutput& output);
    static bool validateImageData(const TestImageData& image);
    
    // Setup/teardown utilities
    static void setupTestEnvironment();
    static void teardownTestEnvironment();
    
    // Mathematical utilities for testing
    static std::vector<float> generateSineWave(size_t samples, float frequency = 1.0f, float amplitude = 1.0f);
    static std::vector<float> generateGaussianNoise(size_t samples, float mean = 0.0f, float stddev = 1.0f);
    
    // Constants for testing
    static constexpr float DEFAULT_TOLERANCE = 1e-6f;
    static constexpr float EMBEDDED_TOLERANCE = 1e-3f;  // Looser tolerance for fixed-point
    static constexpr int DEFAULT_TEST_ITERATIONS = 100;
    static constexpr int PERFORMANCE_TEST_ITERATIONS = 1000;
    
    // Test image dimensions
    static constexpr int TEST_IMAGE_WIDTH = 160;
    static constexpr int TEST_IMAGE_HEIGHT = 120;
    static constexpr int TEST_IMAGE_CHANNELS = 3;
};

// Macros for common test assertions (framework-agnostic)
#define ASSERT_FLOAT_EQ(expected, actual, tolerance) \
    do { \
        if (!TestHelpers::areFloatsEqual(expected, actual, tolerance)) { \
            throw std::runtime_error("Float assertion failed"); \
        } \
    } while(0)

#define ASSERT_VECTOR_EQ(expected, actual, tolerance) \
    do { \
        if (!TestHelpers::areVectorsEqual(expected, actual, tolerance)) { \
            throw std::runtime_error("Vector assertion failed"); \
        } \
    } while(0)

#define ASSERT_PERFORMANCE(func, max_time_ms) \
    do { \
        TestHelpers::Timer timer; \
        timer.start(); \
        func(); \
        double elapsed = timer.elapsed(); \
        if (elapsed > max_time_ms) { \
            throw std::runtime_error("Performance assertion failed"); \
        } \
    } while(0)

} // namespace Testing
} // namespace LiquidVision