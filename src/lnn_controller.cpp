#include "../include/liquid_vision/lnn_controller.hpp"
#include "../include/liquid_vision/core/liquid_network.h"
#include "../include/liquid_vision/vision/image_processor.h"
#include "../include/liquid_vision/control/flight_controller.h"
#include <memory>
#include <chrono>
#include <algorithm>

namespace LiquidVision {

class LNNController::Impl {
public:
    std::unique_ptr<LiquidNetwork> network_;
    std::unique_ptr<ImageProcessor> image_processor_;
    Config config_;
    bool initialized_ = false;
    
    // Performance metrics
    uint32_t inference_count_ = 0;
    float total_inference_time_ms_ = 0.0f;
    float total_power_consumption_mw_ = 0.0f;
    
    Impl(const Config& config) : config_(config) {}
    
    bool initialize() {
        // Create liquid network configuration
        LiquidNetwork::NetworkConfig net_config;
        
        // Input layer for vision features
        LiquidNetwork::LayerConfig input_layer;
        input_layer.num_neurons = 64;  // Vision feature neurons
        input_layer.params.tau = 0.8f;
        input_layer.params.leak = 0.15f;
        input_layer.use_fixed_point = config_.use_fixed_point;
        
        // Hidden layer for processing
        LiquidNetwork::LayerConfig hidden_layer;
        hidden_layer.num_neurons = 32;
        hidden_layer.params.tau = 1.2f;
        hidden_layer.params.leak = 0.1f;
        hidden_layer.use_fixed_point = config_.use_fixed_point;
        
        // Output layer for control signals
        LiquidNetwork::LayerConfig output_layer;
        output_layer.num_neurons = 8;  // Control outputs
        output_layer.params.tau = 0.5f;
        output_layer.params.leak = 0.2f;
        output_layer.use_fixed_point = config_.use_fixed_point;
        
        net_config.layers = {input_layer, hidden_layer, output_layer};
        net_config.timestep = config_.ode_timestep;
        net_config.max_iterations = config_.max_iterations;
        net_config.adaptive_timestep = config_.adaptive_timestep;
        
        // Create network
        network_ = std::make_unique<LiquidNetwork>(net_config);
        
        // Initialize weights
        if (!config_.model_path.empty()) {
            if (!network_->load_weights(config_.model_path)) {
                // Fall back to random initialization
                if (!network_->initialize()) {
                    return false;
                }
            }
        } else {
            if (!network_->initialize()) {
                return false;
            }
        }
        
        // Create image processor
        ImageProcessor::Config img_config;
        img_config.target_width = config_.input_width;
        img_config.target_height = config_.input_height;
        img_config.use_temporal_filter = config_.use_temporal_filtering;
        
        image_processor_ = std::make_unique<ImageProcessor>(img_config);
        
        initialized_ = true;
        return true;
    }
    
    ControlOutput process_frame(const std::vector<uint8_t>& image_data) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        ControlOutput output;
        if (!initialized_) {
            return output;
        }
        
        // Process image to extract features - simple conversion for now
        std::vector<float> features;
        int expected_size = config_.input_width * config_.input_height;
        
        // Convert raw image data to features (simplified)
        if (image_data.size() >= expected_size * 3) {
            // RGB to grayscale conversion
            features.reserve(expected_size);
            for (int i = 0; i < expected_size; ++i) {
                float r = image_data[i * 3 + 0] / 255.0f;
                float g = image_data[i * 3 + 1] / 255.0f;
                float b = image_data[i * 3 + 2] / 255.0f;
                features.push_back(0.299f * r + 0.587f * g + 0.114f * b);
            }
        } else {
            // Fill with zeros if data insufficient
            features.resize(expected_size, 0.0f);
        }
        
        // Downsample to network input size
        std::vector<float> network_input;
        int network_input_size = network_->get_config().layers[0].num_neurons;
        int skip = std::max(1, static_cast<int>(features.size() / network_input_size));
        
        for (size_t i = 0; i < features.size() && network_input.size() < network_input_size; i += skip) {
            network_input.push_back(features[i]);
        }
        network_input.resize(network_input_size, 0.0f);
        
        // Run liquid neural network inference
        auto result = network_->forward(network_input);
        
        // Convert network outputs to control signals
        if (result.outputs.size() >= 8) {
            output.velocity_x = std::tanh(result.outputs[0]) * config_.max_velocity;
            output.velocity_y = std::tanh(result.outputs[1]) * config_.max_velocity;
            output.velocity_z = std::tanh(result.outputs[2]) * config_.max_velocity;
            output.yaw_rate = std::tanh(result.outputs[3]) * config_.max_yaw_rate;
            output.thrust = (std::tanh(result.outputs[4]) + 1.0f) * 0.5f; // [0,1]
            
            // Legacy fields for compatibility
            output.forward_velocity = output.velocity_x;
            output.target_altitude = 5.0f + output.velocity_z; // Relative to base altitude
            
            // Obstacle avoidance signals
            output.obstacle_left = std::max(0.0f, result.outputs[5]);
            output.obstacle_right = std::max(0.0f, result.outputs[6]);
            output.obstacle_front = std::max(0.0f, result.outputs[7]);
        }
        
        output.confidence = result.confidence;
        output.processing_time_ms = result.computation_time_us / 1000.0f;
        output.power_consumption_mw = network_->get_power_consumption();
        output.inference_time_us = result.computation_time_us;
        
        // Update metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        float total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count() / 1000.0f;
        
        inference_count_++;
        total_inference_time_ms_ += total_time_ms;
        total_power_consumption_mw_ += output.power_consumption_mw;
        
        return output;
    }
    
    InferenceStats get_stats() const {
        InferenceStats stats;
        if (inference_count_ > 0) {
            stats.average_inference_time_ms = total_inference_time_ms_ / inference_count_;
            stats.average_power_consumption_mw = total_power_consumption_mw_ / inference_count_;
        }
        stats.total_inferences = inference_count_;
        stats.memory_usage_kb = network_ ? network_->get_memory_usage() : 0.0f;
        return stats;
    }
};

// LNNController implementation
LNNController::LNNController(const Config& config) 
    : config_(config), pImpl(std::make_unique<Impl>(config)) {
}

LNNController::~LNNController() = default;

bool LNNController::initialize() {
    if (!validate_config()) {
        return false;
    }
    
    if (!pImpl->initialize()) {
        return false;
    }
    
    initialized_ = true;
    model_loaded_ = true;
    return true;
}

LNNController::ControlOutput LNNController::process_frame(const std::vector<uint8_t>& image_data) {
    return pImpl->process_frame(image_data);
}

ProcessedFrame LNNController::preprocess(const uint8_t* image_data,
                                        int width, int height, int channels) {
    ProcessedFrame frame;
    if (!pImpl->image_processor_) {
        return frame;
    }
    
    return pImpl->image_processor_->process(image_data, width, height, channels);
}

LNNController::ControlOutput LNNController::infer(const ProcessedFrame& frame) {
    ControlOutput output;
    
    if (!initialized_ || !model_loaded_ || !pImpl->network_) {
        return output;
    }
    
    // Convert ProcessedFrame to network input
    std::vector<float> nn_input;
    int input_size = pImpl->network_->get_config().layers[0].num_neurons;
    int skip = std::max(1, static_cast<int>(frame.data.size() / input_size));
    
    for (size_t i = 0; i < frame.data.size() && nn_input.size() < input_size; i += skip) {
        nn_input.push_back(frame.data[i]);
    }
    
    nn_input.resize(input_size, 0.0f);
    
    // Run inference
    auto result = pImpl->network_->forward(nn_input);
    
    // Map to control output
    if (result.outputs.size() >= 3) {
        output.forward_velocity = result.outputs[0] * 5.0f;
        output.yaw_rate = result.outputs[1] * 1.0f;
        output.target_altitude = (result.outputs[2] + 1.0f) * 5.0f;
        output.confidence = result.confidence;
        output.inference_time_us = result.computation_time_us;
    }
    
    return output;
}

InferenceStats LNNController::get_performance_stats() const {
    return pImpl->get_stats();
}

bool LNNController::is_initialized() const {
    return initialized_;
}

bool LNNController::load_model() {
    // Model loading is handled in initialize()
    return true;
}

bool LNNController::validate_config() const {
    if (config_.input_width <= 0 || config_.input_height <= 0) {
        return false;
    }
    
    if (config_.max_inference_time_ms <= 0) {
        return false;
    }
    
    if (config_.memory_limit_kb <= 0) {
        return false;
    }
    
    return true;
}

} // namespace LiquidVision