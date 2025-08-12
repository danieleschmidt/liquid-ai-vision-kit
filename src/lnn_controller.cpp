#include "../include/liquid_vision/lnn_controller.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace LiquidVision {

class LNNController::Impl {
public:
    Config config;
    bool initialized = false;
    bool model_loaded = false;
    
    std::unique_ptr<LiquidNetwork> network;
    std::unique_ptr<ImageProcessor> processor;
    std::unique_ptr<FlightController> controller;
    
    PerformanceStats stats;
    
    Impl(const Config& cfg) : config(cfg) {}
    
    bool initialize() {
        // Initialize image processor
        ImageProcessor::Config proc_config;
        proc_config.target_width = config.input_resolution.first;
        proc_config.target_height = config.input_resolution.second;
        proc_config.use_temporal_filter = config.timestep_adaptive;
        processor = std::make_unique<ImageProcessor>(proc_config);
        
        // Initialize neural network
        LiquidNetwork::NetworkConfig net_config;
        
        // Create a 3-layer network for vision processing
        net_config.layers.push_back({64, {1.0f, 0.1f, 0.5f, 0.01f}, true});  // Input layer
        net_config.layers.push_back({32, {0.8f, 0.15f, 0.6f, 0.02f}, true}); // Hidden layer
        net_config.layers.push_back({4, {0.5f, 0.2f, 0.7f, 0.03f}, true});   // Output layer
        
        net_config.timestep = 0.01f;
        net_config.max_iterations = 10;
        net_config.adaptive_timestep = config.timestep_adaptive;
        
        network = std::make_unique<LiquidNetwork>(net_config);
        
        if (!network->initialize()) {
            return false;
        }
        
        // Initialize flight controller
        controller = std::make_unique<FlightController>(FlightController::ControllerType::SIMULATION);
        
        if (!controller->initialize()) {
            return false;
        }
        
        initialized = true;
        return true;
    }
    
    bool load_model() {
        if (!network) return false;
        
        // Try to load model from file
        if (!config.model_path.empty()) {
            if (network->load_weights(config.model_path)) {
                model_loaded = true;
                return true;
            }
        }
        
        // If no model file, we're already initialized with random weights
        model_loaded = true;
        return true;
    }
    
    ControlOutput process_frame(const uint8_t* image_data, int width, int height, int channels) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        ControlOutput output;
        
        if (!initialized || !model_loaded) {
            return output;
        }
        
        // Process image
        ProcessedFrame frame = processor->process(image_data, width, height, channels);
        
        // Prepare input for neural network
        std::vector<float> nn_input;
        
        // Downsample frame data to match network input size
        int input_size = network->get_config().layers[0].num_neurons;
        int skip = frame.data.size() / input_size;
        
        for (size_t i = 0; i < frame.data.size() && nn_input.size() < input_size; i += skip) {
            nn_input.push_back(frame.data[i]);
        }
        
        // Ensure correct input size
        nn_input.resize(input_size, 0.0f);
        
        // Add temporal difference as additional input
        if (input_size > 1) {
            nn_input[input_size - 1] = frame.temporal_diff;
        }
        
        // Run neural network inference
        auto inference_result = network->forward(nn_input);
        
        // Map network outputs to control commands
        if (inference_result.outputs.size() >= 3) {
            output.forward_velocity = inference_result.outputs[0] * 5.0f;  // Scale to m/s
            output.yaw_rate = inference_result.outputs[1] * 1.0f;         // Scale to rad/s
            output.target_altitude = (inference_result.outputs[2] + 1.0f) * 5.0f; // 0-10m range
            
            if (inference_result.outputs.size() >= 4) {
                output.confidence = std::abs(inference_result.outputs[3]);
            } else {
                output.confidence = inference_result.confidence;
            }
        }
        
        // Calculate timing
        auto end_time = std::chrono::high_resolution_clock::now();
        output.inference_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();
        
        // Update statistics
        update_stats(output, inference_result);
        
        return output;
    }
    
    void update_stats(const ControlOutput& output, const LiquidNetwork::InferenceResult& inference) {
        stats.total_frames++;
        
        // Update running averages
        float alpha = 0.95f;  // Exponential moving average factor
        
        float inference_time_ms = output.inference_time_us / 1000.0f;
        stats.average_inference_time_ms = alpha * stats.average_inference_time_ms + 
                                         (1 - alpha) * inference_time_ms;
        
        stats.average_confidence = alpha * stats.average_confidence + 
                                  (1 - alpha) * output.confidence;
        
        stats.total_energy_consumed_j += inference.energy_consumed_mj / 1000.0f;
        
        float current_power = network->get_power_consumption();
        stats.average_power_mw = alpha * stats.average_power_mw + (1 - alpha) * current_power;
    }
    
    bool validate_config() const {
        if (config.input_resolution.first <= 0 || config.input_resolution.second <= 0) {
            return false;
        }
        
        if (config.max_inference_time_ms <= 0) {
            return false;
        }
        
        if (config.memory_limit_kb <= 0) {
            return false;
        }
        
        return true;
    }
};

// Main LNNController implementation
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
    
    if (!load_model()) {
        return false;
    }
    
    initialized_ = true;
    return true;
}

LNNController::ControlOutput LNNController::process_frame(const uint8_t* image_data,
                                                         int width, int height, int channels) {
    return pImpl->process_frame(image_data, width, height, channels);
}

ProcessedFrame LNNController::preprocess(const uint8_t* image_data,
                                        int width, int height, int channels) {
    if (!pImpl->processor) {
        return ProcessedFrame();
    }
    
    return pImpl->processor->process(image_data, width, height, channels);
}

LNNController::ControlOutput LNNController::infer(const ProcessedFrame& frame) {
    ControlOutput output;
    
    if (!initialized_ || !model_loaded_ || !pImpl->network) {
        return output;
    }
    
    // Convert ProcessedFrame to network input
    std::vector<float> nn_input;
    int input_size = pImpl->network->get_config().layers[0].num_neurons;
    int skip = std::max(1, static_cast<int>(frame.data.size() / input_size));
    
    for (size_t i = 0; i < frame.data.size() && nn_input.size() < input_size; i += skip) {
        nn_input.push_back(frame.data[i]);
    }
    
    nn_input.resize(input_size, 0.0f);
    
    // Run inference
    auto result = pImpl->network->forward(nn_input);
    
    // Map to control output
    if (result.outputs.size() >= 3) {
        output.forward_velocity = result.outputs[0] * 5.0f;
        output.yaw_rate = result.outputs[1] * 1.0f;
        output.target_altitude = (result.outputs[2] + 1.0f) * 5.0f;
        output.confidence = result.confidence;
        output.inference_time_us = result.computation_time_us;
    }
    
    pImpl->update_stats(output, result);
    
    return output;
}

PerformanceStats LNNController::get_performance_stats() const {
    return pImpl->stats;
}

bool LNNController::load_model() {
    model_loaded_ = pImpl->load_model();
    return model_loaded_;
}

bool LNNController::validate_config() const {
    return pImpl->validate_config();
}

} // namespace LiquidVision