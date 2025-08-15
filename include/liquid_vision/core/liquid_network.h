#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <string>
#include "../utils/fixed_point.h"

namespace LiquidVision {

/**
 * Core Liquid Neural Network implementation
 * Uses continuous-time neural dynamics with adaptive computation
 */
class LiquidNetwork {
public:
    struct NeuronParams {
        float tau = 1.0f;           // Time constant
        float leak = 0.1f;           // Leak rate
        float threshold = 0.5f;      // Activation threshold
        float adaptation = 0.01f;    // Adaptation rate
    };

    struct LayerConfig {
        int num_neurons = 32;
        NeuronParams params;
        bool use_fixed_point = true;
    };

    struct NetworkConfig {
        std::vector<LayerConfig> layers;
        float timestep = 0.01f;
        int max_iterations = 10;
        bool adaptive_timestep = true;
    };

    struct InferenceResult {
        std::vector<float> outputs;
        float confidence = 0.0f;
        uint32_t computation_time_us = 0;
        float energy_consumed_mj = 0.0f;
    };

private:
    NetworkConfig config_;
    std::vector<std::vector<float>> states_;       // Neuron states
    std::vector<std::vector<float>> weights_;      // Connection weights
    std::vector<std::vector<float>> biases_;       // Neuron biases
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    bool initialized_ = false;

    // Adaptive computation tracking
    float complexity_estimate_ = 0.5f;
    float average_iterations_ = 5.0f;

public:
    explicit LiquidNetwork(const NetworkConfig& config);
    ~LiquidNetwork() = default;

    bool initialize();
    bool load_weights(const std::string& model_path);
    InferenceResult forward(const std::vector<float>& input);
    
    // Core liquid dynamics computation
    void update_neurons(float timestep);
    std::vector<float> compute_derivatives(const std::vector<float>& state,
                                          const std::vector<float>& input);
    
    // Adaptive computation
    float estimate_complexity(const std::vector<float>& input);
    int compute_required_iterations(float complexity);
    
    // Energy and performance monitoring
    float get_power_consumption() const;
    float get_memory_usage() const;
    
    const NetworkConfig& get_config() const { return config_; }
    bool is_initialized() const { return initialized_; }

private:
    // ODE solver implementations
    void runge_kutta_step(float timestep);
    void euler_step(float timestep);
    
    // Activation functions
    float activation(float x) const;
    float activation_derivative(float x) const;
    
    // Utility functions
    void reset_states();
    float compute_confidence(const std::vector<float>& outputs) const;
};


} // namespace LiquidVision