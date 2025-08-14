#include "../include/liquid_vision/core/liquid_neuron.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace LiquidVision;

void assert_float_eq(float a, float b, const std::string& msg = "") {
    if (std::abs(a - b) > 1e-6f) {
        throw std::runtime_error("Float assertion failed: " + msg);
    }
}

void assert_ne(float a, float b, const std::string& msg = "") {
    if (a == b) {
        throw std::runtime_error("Not-equal assertion failed: " + msg);
    }
}

void assert_ge(float a, float b, const std::string& msg = "") {
    if (a < b) {
        throw std::runtime_error("Greater-equal assertion failed: " + msg);
    }
}

void assert_le(float a, float b, const std::string& msg = "") {
    if (a > b) {
        throw std::runtime_error("Less-equal assertion failed: " + msg);
    }
}

void test_neuron_initialization() {
    LiquidNeuron::Parameters params;
    params.tau = 1.0f;
    params.leak = 0.1f;
    
    LiquidNeuron neuron(params);
    
    assert_float_eq(neuron.get_state(), 0.0f);
    assert_float_eq(neuron.get_parameters().tau, 1.0f);
    assert_float_eq(neuron.get_parameters().leak, 0.1f);
}

void test_neuron_forward_pass() {
    LiquidNeuron neuron;
    
    std::vector<float> weights = {0.5f, -0.3f, 0.8f};
    neuron.set_weights(weights);
    neuron.set_bias(0.1f);
    
    std::vector<float> inputs = {1.0f, 0.5f, -0.2f};
    
    // Test multiple timesteps
    float output1 = neuron.update(inputs, 0.01f);
    float output2 = neuron.update(inputs, 0.01f);
    
    // Output should be different after updates (temporal dynamics)
    assert_ne(output1, output2);
    
    // Output should be bounded by activation function
    assert_ge(output1, -1.0f);
    assert_le(output1, 1.0f);
    assert_ge(output2, -1.0f);
    assert_le(output2, 1.0f);
}

void test_neuron_stability() {
    LiquidNeuron neuron;
    
    std::vector<float> weights = {0.1f, 0.1f};
    neuron.set_weights(weights);
    
    std::vector<float> inputs = {1.0f, 1.0f};
    
    // Run for many timesteps to check stability
    for (int i = 0; i < 1000; ++i) {
        float output = neuron.update(inputs, 0.001f);
        
        // Should remain bounded
        assert_ge(output, -10.0f);
        assert_le(output, 10.0f);
        if (!std::isfinite(output)) {
            throw std::runtime_error("Output not finite");
        }
    }
}

void test_neuron_reset() {
    LiquidNeuron neuron;
    
    std::vector<float> weights = {0.5f};
    neuron.set_weights(weights);
    
    std::vector<float> inputs = {1.0f};
    
    // Update state
    neuron.update(inputs, 0.1f);
    assert_ne(neuron.get_state(), 0.0f);
    
    // Reset should restore initial state
    neuron.reset();
    assert_float_eq(neuron.get_state(), 0.0f);
}

void run_test(void (*test_func)(), const std::string& name) {
    try {
        test_func();
        std::cout << "PASS: " << name << std::endl;
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << name << " - " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Running Liquid Neuron Tests..." << std::endl;
    
    run_test(test_neuron_initialization, "neuron_initialization");
    run_test(test_neuron_forward_pass, "neuron_forward_pass");
    run_test(test_neuron_stability, "neuron_stability");
    run_test(test_neuron_reset, "neuron_reset");
    
    std::cout << "Tests completed." << std::endl;
    return 0;
}