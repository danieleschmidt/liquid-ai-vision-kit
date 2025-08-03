#include "liquid_network.h"
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cstring>

namespace LiquidVision {

LiquidNetwork::LiquidNetwork(const NetworkConfig& config) 
    : config_(config) {
    // Initialize state vectors for each layer
    states_.resize(config.layers.size());
    for (size_t i = 0; i < config.layers.size(); ++i) {
        states_[i].resize(config.layers[i].num_neurons, 0.0f);
    }
    
    // Initialize weight matrices
    weights_.resize(config.layers.size());
    biases_.resize(config.layers.size());
    
    if (!config.layers.empty()) {
        input_buffer_.resize(config.layers[0].num_neurons);
        output_buffer_.resize(config.layers.back().num_neurons);
    }
}

bool LiquidNetwork::initialize() {
    if (config_.layers.empty()) {
        return false;
    }
    
    // Initialize weights with Xavier initialization
    for (size_t layer = 0; layer < config_.layers.size(); ++layer) {
        int current_size = config_.layers[layer].num_neurons;
        int prev_size = (layer > 0) ? config_.layers[layer - 1].num_neurons : current_size;
        
        weights_[layer].resize(current_size * prev_size);
        biases_[layer].resize(current_size);
        
        // Xavier initialization
        float scale = std::sqrt(2.0f / (prev_size + current_size));
        for (auto& w : weights_[layer]) {
            w = ((rand() / float(RAND_MAX)) * 2.0f - 1.0f) * scale;
        }
        
        for (auto& b : biases_[layer]) {
            b = 0.01f * ((rand() / float(RAND_MAX)) * 2.0f - 1.0f);
        }
    }
    
    initialized_ = true;
    return true;
}

bool LiquidNetwork::load_weights(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Read model header
    uint32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    if (magic_number != 0x4C4E4E00) { // "LNN\0"
        return false;
    }
    
    // Read layer count
    uint32_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    if (num_layers != config_.layers.size()) {
        return false;
    }
    
    // Read weights and biases for each layer
    for (size_t layer = 0; layer < num_layers; ++layer) {
        uint32_t weight_count, bias_count;
        
        file.read(reinterpret_cast<char*>(&weight_count), sizeof(weight_count));
        if (weight_count != weights_[layer].size()) {
            return false;
        }
        
        file.read(reinterpret_cast<char*>(weights_[layer].data()), 
                 weight_count * sizeof(float));
        
        file.read(reinterpret_cast<char*>(&bias_count), sizeof(bias_count));
        if (bias_count != biases_[layer].size()) {
            return false;
        }
        
        file.read(reinterpret_cast<char*>(biases_[layer].data()),
                 bias_count * sizeof(float));
    }
    
    file.close();
    initialized_ = true;
    return true;
}

LiquidNetwork::InferenceResult LiquidNetwork::forward(const std::vector<float>& input) {
    auto start_time = std::chrono::high_resolution_clock::now();
    InferenceResult result;
    
    if (!initialized_ || input.empty()) {
        return result;
    }
    
    // Reset neuron states
    reset_states();
    
    // Copy input to first layer
    size_t input_neurons = std::min(input.size(), states_[0].size());
    for (size_t i = 0; i < input_neurons; ++i) {
        states_[0][i] = input[i];
    }
    
    // Estimate complexity for adaptive computation
    float complexity = estimate_complexity(input);
    int iterations = compute_required_iterations(complexity);
    
    // Run liquid dynamics
    for (int iter = 0; iter < iterations; ++iter) {
        float adaptive_timestep = config_.timestep;
        
        if (config_.adaptive_timestep) {
            // Adjust timestep based on state changes
            adaptive_timestep *= (1.0f + 0.5f * complexity);
        }
        
        // Update all neurons
        update_neurons(adaptive_timestep);
    }
    
    // Extract output from last layer
    result.outputs = states_.back();
    
    // Compute confidence
    result.confidence = compute_confidence(result.outputs);
    
    // Compute timing
    auto end_time = std::chrono::high_resolution_clock::now();
    result.computation_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    // Estimate energy consumption (simplified model)
    result.energy_consumed_mj = (result.computation_time_us / 1000.0f) * get_power_consumption();
    
    return result;
}

void LiquidNetwork::update_neurons(float timestep) {
    // Process each layer sequentially
    for (size_t layer = 0; layer < config_.layers.size(); ++layer) {
        std::vector<float> new_states(states_[layer].size());
        
        const auto& layer_config = config_.layers[layer];
        int current_size = layer_config.num_neurons;
        
        for (int n = 0; n < current_size; ++n) {
            float weighted_input = biases_[layer][n];
            
            // Compute input from previous layer (or external input for first layer)
            if (layer > 0) {
                int prev_size = config_.layers[layer - 1].num_neurons;
                for (int p = 0; p < prev_size; ++p) {
                    int weight_idx = n * prev_size + p;
                    weighted_input += weights_[layer][weight_idx] * states_[layer - 1][p];
                }
            }
            
            // Apply liquid dynamics
            float current_state = states_[layer][n];
            float tau = layer_config.params.tau;
            float leak = layer_config.params.leak;
            
            // dx/dt = (-leak * x + weighted_input) / tau
            float derivative = (-leak * current_state + weighted_input) / tau;
            
            // Update state using Euler integration
            new_states[n] = current_state + timestep * derivative;
            
            // Apply activation and bounds
            new_states[n] = activation(new_states[n]);
            
            // Apply adaptation
            if (layer_config.params.adaptation > 0) {
                new_states[n] *= (1.0f - layer_config.params.adaptation * std::abs(new_states[n]));
            }
        }
        
        states_[layer] = new_states;
    }
}

std::vector<float> LiquidNetwork::compute_derivatives(const std::vector<float>& state,
                                                      const std::vector<float>& input) {
    std::vector<float> derivatives(state.size());
    
    for (size_t i = 0; i < state.size(); ++i) {
        float weighted_sum = 0.0f;
        
        // Add input contribution
        if (i < input.size()) {
            weighted_sum += input[i];
        }
        
        // Simple liquid dynamics model
        float tau = 1.0f;
        float leak = 0.1f;
        derivatives[i] = (-leak * state[i] + weighted_sum) / tau;
    }
    
    return derivatives;
}

float LiquidNetwork::estimate_complexity(const std::vector<float>& input) {
    // Estimate complexity based on input variance and magnitude
    float mean = 0.0f;
    float variance = 0.0f;
    
    for (float val : input) {
        mean += val;
    }
    mean /= input.size();
    
    for (float val : input) {
        float diff = val - mean;
        variance += diff * diff;
    }
    variance /= input.size();
    
    // Normalize complexity estimate to [0, 1]
    float complexity = std::min(1.0f, std::sqrt(variance) + 0.1f * std::abs(mean));
    
    // Update running average
    complexity_estimate_ = 0.9f * complexity_estimate_ + 0.1f * complexity;
    
    return complexity_estimate_;
}

int LiquidNetwork::compute_required_iterations(float complexity) {
    if (!config_.adaptive_timestep) {
        return config_.max_iterations;
    }
    
    // Scale iterations based on complexity
    int base_iterations = config_.max_iterations / 2;
    int additional_iterations = static_cast<int>(complexity * config_.max_iterations / 2);
    
    return std::min(config_.max_iterations, base_iterations + additional_iterations);
}

float LiquidNetwork::get_power_consumption() const {
    // Estimate power based on network size and activity
    float total_neurons = 0;
    for (const auto& layer : config_.layers) {
        total_neurons += layer.num_neurons;
    }
    
    // Base power consumption model (mW)
    float base_power = 50.0f;  // MCU baseline
    float per_neuron_power = 0.5f;  // Per neuron overhead
    float dynamic_power = 0.2f;  // Activity-based power
    
    return base_power + total_neurons * per_neuron_power + 
           dynamic_power * complexity_estimate_ * total_neurons;
}

float LiquidNetwork::get_memory_usage() const {
    // Calculate memory footprint in KB
    float total_memory = 0.0f;
    
    // States
    for (const auto& layer_states : states_) {
        total_memory += layer_states.size() * sizeof(float);
    }
    
    // Weights and biases
    for (const auto& layer_weights : weights_) {
        total_memory += layer_weights.size() * sizeof(float);
    }
    
    for (const auto& layer_biases : biases_) {
        total_memory += layer_biases.size() * sizeof(float);
    }
    
    return total_memory / 1024.0f;  // Convert to KB
}

void LiquidNetwork::runge_kutta_step(float timestep) {
    // RK4 implementation for more accurate integration
    for (size_t layer = 0; layer < states_.size(); ++layer) {
        std::vector<float> k1, k2, k3, k4;
        std::vector<float> temp_state = states_[layer];
        
        // k1 = f(state, t)
        k1 = compute_derivatives(states_[layer], input_buffer_);
        
        // k2 = f(state + timestep/2 * k1, t + timestep/2)
        for (size_t i = 0; i < temp_state.size(); ++i) {
            temp_state[i] = states_[layer][i] + timestep * 0.5f * k1[i];
        }
        k2 = compute_derivatives(temp_state, input_buffer_);
        
        // k3 = f(state + timestep/2 * k2, t + timestep/2)
        for (size_t i = 0; i < temp_state.size(); ++i) {
            temp_state[i] = states_[layer][i] + timestep * 0.5f * k2[i];
        }
        k3 = compute_derivatives(temp_state, input_buffer_);
        
        // k4 = f(state + timestep * k3, t + timestep)
        for (size_t i = 0; i < temp_state.size(); ++i) {
            temp_state[i] = states_[layer][i] + timestep * k3[i];
        }
        k4 = compute_derivatives(temp_state, input_buffer_);
        
        // Update state: state += (timestep/6) * (k1 + 2*k2 + 2*k3 + k4)
        for (size_t i = 0; i < states_[layer].size(); ++i) {
            states_[layer][i] += (timestep / 6.0f) * 
                (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
        }
    }
}

void LiquidNetwork::euler_step(float timestep) {
    // Simple Euler integration
    for (size_t layer = 0; layer < states_.size(); ++layer) {
        auto derivatives = compute_derivatives(states_[layer], input_buffer_);
        
        for (size_t i = 0; i < states_[layer].size(); ++i) {
            states_[layer][i] += timestep * derivatives[i];
        }
    }
}

float LiquidNetwork::activation(float x) const {
    // Tanh activation for bounded outputs
    return std::tanh(x);
}

float LiquidNetwork::activation_derivative(float x) const {
    float tanh_x = std::tanh(x);
    return 1.0f - tanh_x * tanh_x;
}

void LiquidNetwork::reset_states() {
    for (auto& layer_states : states_) {
        std::fill(layer_states.begin(), layer_states.end(), 0.0f);
    }
    
    // Reset adaptation tracking
    average_iterations_ = config_.max_iterations / 2.0f;
}

float LiquidNetwork::compute_confidence(const std::vector<float>& outputs) const {
    if (outputs.empty()) {
        return 0.0f;
    }
    
    // Confidence based on output strength and consistency
    float max_output = 0.0f;
    float mean_output = 0.0f;
    
    for (float out : outputs) {
        float abs_out = std::abs(out);
        max_output = std::max(max_output, abs_out);
        mean_output += abs_out;
    }
    mean_output /= outputs.size();
    
    // Higher confidence when outputs are strong and consistent
    float strength = max_output;
    float consistency = 1.0f - std::abs(max_output - mean_output);
    
    return 0.5f * (strength + consistency);
}

} // namespace LiquidVision