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

/**
 * Specialized liquid neuron with continuous dynamics
 */
template<typename T, int STATE_DIM>
class LiquidNeuron {
private:
    FixedPoint<16, 16> state_[STATE_DIM];
    FixedPoint<16, 16> tau_;
    FixedPoint<16, 16> leak_;
    FixedPoint<16, 16> threshold_;
    std::vector<FixedPoint<16, 16>> weights_;

public:
    LiquidNeuron() : tau_(1.0f), leak_(0.1f), threshold_(0.5f) {
        for (int i = 0; i < STATE_DIM; ++i) {
            state_[i] = FixedPoint<16, 16>(0.0f);
        }
    }

    void update(const std::vector<T>& inputs, T timestep) {
        // Compute weighted input sum
        FixedPoint<16, 16> weighted_sum(0.0f);
        for (size_t i = 0; i < inputs.size() && i < weights_.size(); ++i) {
            weighted_sum = weighted_sum + weights_[i] * FixedPoint<16, 16>(inputs[i]);
        }

        // Update state using liquid dynamics
        for (int i = 0; i < STATE_DIM; ++i) {
            // dx/dt = (-leak * x + weighted_sum) / tau
            FixedPoint<16, 16> derivative = (weighted_sum - leak_ * state_[i]) / tau_;
            state_[i] = state_[i] + FixedPoint<16, 16>(timestep) * derivative;
            
            // Apply bounds
            if (state_[i].to_float() > 1.0f) {
                state_[i] = FixedPoint<16, 16>(1.0f);
            } else if (state_[i].to_float() < -1.0f) {
                state_[i] = FixedPoint<16, 16>(-1.0f);
            }
        }
    }

    T get_output() const {
        FixedPoint<16, 16> sum(0.0f);
        for (int i = 0; i < STATE_DIM; ++i) {
            sum = sum + state_[i];
        }
        float output = sum.to_float() / STATE_DIM;
        
        // Apply activation
        return static_cast<T>(std::tanh(output));
    }

    void set_weights(const std::vector<T>& new_weights) {
        weights_.clear();
        for (const auto& w : new_weights) {
            weights_.push_back(FixedPoint<16, 16>(w));
        }
    }

    void reset() {
        for (int i = 0; i < STATE_DIM; ++i) {
            state_[i] = FixedPoint<16, 16>(0.0f);
        }
    }
};

} // namespace LiquidVision