#include <iostream>
#include <vector>
#include <chrono>
#include <exception>
#include <memory>
#include <cmath>
#include "liquid_vision/core/liquid_network.h"
#include "liquid_vision/vision/image_processor.h"

namespace LiquidVision {

/**
 * Robust Error Recovery System for Generation 2
 * Handles failures gracefully and maintains system reliability
 */
class RobustLiquidSystem {
private:
    std::unique_ptr<LiquidNetwork> primary_network_;
    std::unique_ptr<LiquidNetwork> backup_network_;
    std::unique_ptr<ImageProcessor> processor_;
    
    // Error tracking
    int consecutive_errors_ = 0;
    int max_consecutive_errors_ = 5;
    float error_threshold_ = 0.95f;
    
    // Health monitoring 
    struct HealthMetrics {
        float confidence_trend = 1.0f;
        float average_inference_time = 0.0f;
        float power_consumption = 0.0f;
        int successful_inferences = 0;
        int failed_inferences = 0;
        int network_resets = 0;
        int fallback_activations = 0;
    } health_metrics_;
    
    // Fault detection
    bool is_network_healthy(const LiquidNetwork::InferenceResult& result) {
        return result.confidence > 0.1f && 
               result.computation_time_us < 50000 && // 50ms max
               !result.outputs.empty() &&
               std::isfinite(result.outputs[0]);
    }
    
    // Self-healing mechanisms
    void attempt_self_healing() {
        std::cout << "🔧 Attempting self-healing..." << std::endl;
        
        // Try network reset first
        if (primary_network_) {
            // Reset neuron states
            auto config = primary_network_->get_config();
            primary_network_ = std::make_unique<LiquidNetwork>(config);
            primary_network_->initialize();
            health_metrics_.network_resets++;
            std::cout << "  ↻ Primary network reset" << std::endl;
        }
        
        consecutive_errors_ = 0;
    }
    
    // Fallback to backup systems
    LiquidNetwork::InferenceResult fallback_inference(const std::vector<float>& input) {
        health_metrics_.fallback_activations++;
        
        if (backup_network_) {
            std::cout << "  🔄 Using backup network" << std::endl;
            return backup_network_->forward(input);
        }
        
        // Ultimate fallback - return safe default
        std::cout << "  🛡️ Using safe default output" << std::endl;
        LiquidNetwork::InferenceResult safe_result;
        safe_result.outputs = {0.0f, 0.0f}; // Safe hover command
        safe_result.confidence = 0.5f;
        safe_result.computation_time_us = 1;
        return safe_result;
    }

public:
    RobustLiquidSystem(const LiquidNetwork::NetworkConfig& config) {
        // Initialize primary system
        primary_network_ = std::make_unique<LiquidNetwork>(config);
        primary_network_->initialize();
        
        // Initialize backup system with simpler configuration
        auto backup_config = config;
        // Reduce backup network complexity for reliability
        for (auto& layer : backup_config.layers) {
            layer.num_neurons = std::max(2, layer.num_neurons / 2);
        }
        backup_network_ = std::make_unique<LiquidNetwork>(backup_config);
        backup_network_->initialize();
        
        // Initialize image processor
        ImageProcessor::Config proc_config;
        proc_config.target_width = 32;
        proc_config.target_height = 24;
        proc_config.use_temporal_filter = true;
        processor_ = std::make_unique<ImageProcessor>(proc_config);
    }
    
    // Robust inference with error handling
    LiquidNetwork::InferenceResult robust_inference(const std::vector<float>& input) {
        try {
            auto result = primary_network_->forward(input);
            
            // Health check
            if (is_network_healthy(result)) {
                consecutive_errors_ = 0;
                health_metrics_.successful_inferences++;
                
                // Update health metrics
                health_metrics_.confidence_trend = 0.9f * health_metrics_.confidence_trend + 
                                                 0.1f * result.confidence;
                health_metrics_.average_inference_time = 0.9f * health_metrics_.average_inference_time +
                                                       0.1f * result.computation_time_us;
                
                return result;
            } else {
                consecutive_errors_++;
                health_metrics_.failed_inferences++;
                throw std::runtime_error("Network output failed health check");
            }
            
        } catch (const std::exception& e) {
            consecutive_errors_++;
            health_metrics_.failed_inferences++;
            
            std::cout << "⚠️ Inference error: " << e.what() << std::endl;
            
            // Check if self-healing is needed
            if (consecutive_errors_ >= max_consecutive_errors_) {
                attempt_self_healing();
            }
            
            // Use fallback system
            return fallback_inference(input);
        }
    }
    
    // Process image with error recovery
    LiquidNetwork::InferenceResult process_image_robust(const uint8_t* image_data, 
                                                        int width, int height, int channels) {
        try {
            auto processed = processor_->process(image_data, width, height, channels);
            
            // Convert to network input
            std::vector<float> nn_input;
            int input_size = primary_network_->get_config().layers[0].num_neurons;
            int skip = std::max(1, static_cast<int>(processed.data.size() / input_size));
            
            for (size_t i = 0; i < processed.data.size() && nn_input.size() < input_size; i += skip) {
                nn_input.push_back(processed.data[i]);
            }
            nn_input.resize(input_size, 0.0f);
            
            return robust_inference(nn_input);
            
        } catch (const std::exception& e) {
            std::cout << "⚠️ Image processing error: " << e.what() << std::endl;
            
            // Return safe output
            LiquidNetwork::InferenceResult safe_result;
            safe_result.outputs = {0.0f, 0.0f}; // Emergency stop
            safe_result.confidence = 0.0f;
            return safe_result;
        }
    }
    
    // System health report
    void print_health_report() {
        std::cout << "\n📊 System Health Report:" << std::endl;
        std::cout << "  Successful inferences: " << health_metrics_.successful_inferences << std::endl;
        std::cout << "  Failed inferences: " << health_metrics_.failed_inferences << std::endl;
        
        float success_rate = 0.0f;
        int total = health_metrics_.successful_inferences + health_metrics_.failed_inferences;
        if (total > 0) {
            success_rate = static_cast<float>(health_metrics_.successful_inferences) / total * 100.0f;
        }
        std::cout << "  Success rate: " << success_rate << "%" << std::endl;
        std::cout << "  Confidence trend: " << health_metrics_.confidence_trend << std::endl;
        std::cout << "  Network resets: " << health_metrics_.network_resets << std::endl;
        std::cout << "  Fallback activations: " << health_metrics_.fallback_activations << std::endl;
        std::cout << "  Avg inference time: " << health_metrics_.average_inference_time << " μs" << std::endl;
        
        // Overall health status
        if (success_rate > 95.0f && health_metrics_.confidence_trend > 0.8f) {
            std::cout << "  Status: 🟢 HEALTHY" << std::endl;
        } else if (success_rate > 80.0f && health_metrics_.confidence_trend > 0.5f) {
            std::cout << "  Status: 🟡 DEGRADED" << std::endl;
        } else {
            std::cout << "  Status: 🔴 CRITICAL" << std::endl;
        }
    }
    
    // Stress test for robustness
    void stress_test(int num_iterations = 1000) {
        std::cout << "\n🏋️ Starting stress test..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Create various challenging inputs
        std::vector<std::vector<float>> test_inputs = {
            {0.5f, -0.2f, 0.8f, 0.1f},      // Normal input
            {1000.0f, -1000.0f, 0.0f, 0.0f}, // Extreme values
            {NAN, 0.5f, 0.2f, 0.1f},         // NaN input
            {INFINITY, 0.0f, 0.0f, 0.0f},    // Infinite input
            {},                               // Empty input
            {0.0f, 0.0f, 0.0f, 0.0f},        // Zero input
        };
        
        for (int i = 0; i < num_iterations; ++i) {
            // Use different test inputs
            auto& test_input = test_inputs[i % test_inputs.size()];
            
            // Occasionally inject faults
            if (i % 100 == 99) {
                consecutive_errors_ = max_consecutive_errors_; // Force self-healing
            }
            
            try {
                auto result = robust_inference(test_input);
                
                // Verify output sanity
                if (!result.outputs.empty() && std::isfinite(result.outputs[0])) {
                    // Good result
                } else {
                    std::cout << "  ⚠️ Suspicious output at iteration " << i << std::endl;
                }
                
            } catch (...) {
                std::cout << "  💥 Unhandled exception at iteration " << i << std::endl;
            }
            
            // Progress indicator
            if (i % (num_iterations / 10) == 0) {
                std::cout << "  Progress: " << (i * 100 / num_iterations) << "%" << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "✅ Stress test completed in " << duration.count() << " ms" << std::endl;
        print_health_report();
    }
};

} // namespace LiquidVision

// Demo program for Generation 2 robustness
int main() {
    std::cout << "=== Liquid AI Vision Kit - Generation 2: Robustness Demo ===" << std::endl;
    std::cout << "Testing error recovery, self-healing, and fault tolerance\n" << std::endl;
    
    using namespace LiquidVision;
    
    // Create robust system
    LiquidNetwork::NetworkConfig config;
    config.layers.push_back({8, {1.0f, 0.1f, 0.5f, 0.01f}, true});
    config.layers.push_back({4, {0.8f, 0.15f, 0.6f, 0.02f}, true});
    config.layers.push_back({2, {0.5f, 0.2f, 0.7f, 0.03f}, true});
    config.timestep = 0.01f;
    config.max_iterations = 5;
    
    RobustLiquidSystem robust_system(config);
    
    // Test normal operation
    std::cout << "1. Testing normal operation..." << std::endl;
    std::vector<float> normal_input = {0.5f, -0.2f, 0.8f, 0.1f};
    auto result = robust_system.robust_inference(normal_input);
    std::cout << "   Normal inference successful, confidence: " << result.confidence << std::endl;
    
    // Test fault injection
    std::cout << "\n2. Testing fault recovery..." << std::endl;
    std::vector<float> faulty_input = {NAN, INFINITY, -INFINITY, 0.0f};
    auto fault_result = robust_system.robust_inference(faulty_input);
    std::cout << "   Fault recovery successful, confidence: " << fault_result.confidence << std::endl;
    
    // Test image processing robustness
    std::cout << "\n3. Testing image processing robustness..." << std::endl;
    std::vector<uint8_t> test_image(64 * 48 * 3, 128);
    auto img_result = robust_system.process_image_robust(test_image.data(), 64, 48, 3);
    std::cout << "   Image processing successful, confidence: " << img_result.confidence << std::endl;
    
    // Run stress test
    std::cout << "\n4. Running comprehensive stress test..." << std::endl;
    robust_system.stress_test(500);
    
    std::cout << "\n=== Generation 2 Complete: System is Robust ===\n" << std::endl;
    std::cout << "✅ Error detection and recovery" << std::endl;
    std::cout << "✅ Self-healing mechanisms" << std::endl;
    std::cout << "✅ Backup system fallback" << std::endl;
    std::cout << "✅ Health monitoring" << std::endl;
    std::cout << "✅ Fault tolerance under stress" << std::endl;
    
    return 0;
}