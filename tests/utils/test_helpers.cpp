#include "test_helpers.h"
#include <random>
#include <cstring>
#include <fstream>
#include <sstream>

namespace LiquidVision {
namespace Testing {

// Random data generation
std::vector<float> TestHelpers::generateRandomFloats(size_t count, float min, float max) {
    std::vector<float> data(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    for (size_t i = 0; i < count; ++i) {
        data[i] = dis(gen);
    }
    
    return data;
}

std::vector<uint8_t> TestHelpers::generateRandomImage(int width, int height, int channels) {
    std::vector<uint8_t> image(width * height * channels);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis(0, 255);
    
    for (size_t i = 0; i < image.size(); ++i) {
        image[i] = dis(gen);
    }
    
    return image;
}

// Comparison utilities
bool TestHelpers::areFloatsEqual(float a, float b, float tolerance) {
    return std::abs(a - b) <= tolerance;
}

bool TestHelpers::areArraysEqual(const float* a, const float* b, size_t count, float tolerance) {
    for (size_t i = 0; i < count; ++i) {
        if (!areFloatsEqual(a[i], b[i], tolerance)) {
            return false;
        }
    }
    return true;
}

bool TestHelpers::areVectorsEqual(const std::vector<float>& a, const std::vector<float>& b, float tolerance) {
    if (a.size() != b.size()) {
        return false;
    }
    
    for (size_t i = 0; i < a.size(); ++i) {
        if (!areFloatsEqual(a[i], b[i], tolerance)) {
            return false;
        }
    }
    return true;
}

// File I/O utilities
bool TestHelpers::loadTestData(const std::string& filename, std::vector<float>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Read file size
    file.seekg(0, std::ios::end);
    size_t size = file.tellg() / sizeof(float);
    file.seekg(0, std::ios::beg);
    
    // Read data
    data.resize(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
    
    return file.good();
}

bool TestHelpers::saveTestData(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    return file.good();
}

std::string TestHelpers::readTextFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Timing utilities
void TestHelpers::Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

double TestHelpers::Timer::elapsed() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0; // Return milliseconds
}

// Memory utilities
size_t TestHelpers::getCurrentMemoryUsage() {
    // Platform-specific implementation would go here
    // For now, return 0 as placeholder
    return 0;
}

size_t TestHelpers::getPeakMemoryUsage() {
    // Platform-specific implementation would go here
    // For now, return 0 as placeholder
    return 0;
}

// Test data generators
TestNeuronParameters TestHelpers::generateTestNeuronParams() {
    TestNeuronParameters params;
    params.tau = 0.1f;
    params.A = -1.0f;
    params.B = 1.0f;
    params.C = 0.5f;
    params.sigma = 0.02f;
    params.mu = 0.0f;
    params.W_in = 1.0f;
    params.W_rec = 0.5f;
    params.bias = 0.0f;
    return params;
}

TestImageData TestHelpers::generateTestImage(int width, int height) {
    TestImageData image;
    image.width = width;
    image.height = height;
    image.channels = 3;
    image.data = generateRandomImage(width, height, 3);
    return image;
}

TestControlOutput TestHelpers::generateTestControlOutput() {
    TestControlOutput output;
    output.forward_velocity = 1.0f;
    output.lateral_velocity = 0.0f;
    output.yaw_rate = 0.1f;
    output.altitude_rate = 0.0f;
    output.confidence = 0.95f;
    output.timestamp_us = 1000000; // 1 second
    return output;
}

// Performance testing utilities
PerformanceMetrics TestHelpers::measureInferencePerformance(
    std::function<void()> inference_func, 
    int iterations
) {
    PerformanceMetrics metrics = {};
    
    std::vector<double> times;
    times.reserve(iterations);
    
    size_t initial_memory = getCurrentMemoryUsage();
    
    for (int i = 0; i < iterations; ++i) {
        Timer timer;
        timer.start();
        
        inference_func();
        
        double elapsed = timer.elapsed();
        times.push_back(elapsed);
        
        if (elapsed < metrics.min_time || i == 0) {
            metrics.min_time = elapsed;
        }
        if (elapsed > metrics.max_time) {
            metrics.max_time = elapsed;
        }
        metrics.total_time += elapsed;
    }
    
    metrics.avg_time = metrics.total_time / iterations;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (double time : times) {
        variance += (time - metrics.avg_time) * (time - metrics.avg_time);
    }
    metrics.std_dev = std::sqrt(variance / iterations);
    
    metrics.memory_usage = getPeakMemoryUsage() - initial_memory;
    metrics.iterations = iterations;
    
    return metrics;
}

// Validation utilities
bool TestHelpers::validateNeuronOutput(float output, float expected, float tolerance) {
    return areFloatsEqual(output, expected, tolerance) && 
           output >= -1.0f && output <= 1.0f; // Bounded output check
}

bool TestHelpers::validateControlOutput(const TestControlOutput& output) {
    return output.confidence >= 0.0f && output.confidence <= 1.0f &&
           std::abs(output.forward_velocity) <= 10.0f && // Reasonable velocity limits
           std::abs(output.lateral_velocity) <= 10.0f &&
           std::abs(output.yaw_rate) <= 5.0f &&
           std::abs(output.altitude_rate) <= 5.0f;
}

bool TestHelpers::validateImageData(const TestImageData& image) {
    return image.width > 0 && image.height > 0 &&
           image.channels > 0 && image.channels <= 4 &&
           image.data.size() == static_cast<size_t>(image.width * image.height * image.channels);
}

// Setup/teardown utilities
void TestHelpers::setupTestEnvironment() {
    // Initialize any global test state
    // Setup mock hardware interfaces
    // Configure logging for tests
}

void TestHelpers::teardownTestEnvironment() {
    // Cleanup any global test state
    // Reset mock hardware interfaces
    // Flush test logs
}

// Mathematical utilities for testing
std::vector<float> TestHelpers::generateSineWave(size_t samples, float frequency, float amplitude) {
    std::vector<float> wave(samples);
    const float pi = 3.14159265359f;
    
    for (size_t i = 0; i < samples; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(samples);
        wave[i] = amplitude * std::sin(2.0f * pi * frequency * t);
    }
    
    return wave;
}

std::vector<float> TestHelpers::generateGaussianNoise(size_t samples, float mean, float stddev) {
    std::vector<float> noise(samples);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(mean, stddev);
    
    for (size_t i = 0; i < samples; ++i) {
        noise[i] = dis(gen);
    }
    
    return noise;
}

} // namespace Testing
} // namespace LiquidVision