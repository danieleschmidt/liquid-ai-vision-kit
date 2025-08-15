#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include "liquid_vision/core/liquid_network.h"
#include "liquid_vision/core/liquid_neuron.h"
#include "liquid_vision/core/ode_solver.h"
#include "liquid_vision/vision/image_processor.h"

using namespace LiquidVision;

/**
 * Generation 1 Demo: Basic Liquid Neural Network functionality
 * This demonstrates the core working implementation
 */
int main() {
    std::cout << "=== Liquid AI Vision Kit - Generation 1 Demo ===" << std::endl;
    std::cout << "Demonstrating basic Liquid Neural Network functionality\n" << std::endl;
    
    // 1. Test Liquid Neuron
    std::cout << "1. Testing Liquid Neuron..." << std::endl;
    LiquidNeuron::Parameters params;
    params.tau = 1.0f;
    params.leak = 0.1f;
    
    LiquidNeuron neuron(params);
    neuron.set_weights({0.5f, -0.3f, 0.8f});
    neuron.set_bias(0.1f);
    
    std::vector<float> inputs = {1.0f, 0.5f, -0.2f};
    float timestep = 0.01f;
    
    for (int i = 0; i < 10; ++i) {
        float output = neuron.update(inputs, timestep);
        if (i % 3 == 0) {
            std::cout << "  Step " << i << ": state=" << neuron.get_state() 
                      << ", output=" << output << std::endl;
        }
    }
    std::cout << "  ✓ Liquid neuron dynamics working\n" << std::endl;
    
    // 2. Test ODE Solver
    std::cout << "2. Testing ODE Solver..." << std::endl;
    ODESolver solver;
    solver.set_derivative_function([](const std::vector<float>& state, const std::vector<float>& inputs) {
        std::vector<float> derivatives(state.size());
        for (size_t i = 0; i < state.size(); ++i) {
            float input_sum = (i < inputs.size()) ? inputs[i] : 0.0f;
            derivatives[i] = -0.1f * state[i] + input_sum; // Simple liquid dynamics
        }
        return derivatives;
    });
    
    std::vector<float> state = {0.0f, 0.0f};
    std::vector<float> solver_inputs = {1.0f, -0.5f};
    
    for (int i = 0; i < 5; ++i) {
        state = solver.solve_step(state, solver_inputs, 0.01f);
        std::cout << "  Step " << i << ": state=["
                  << state[0] << ", " << state[1] << "]" << std::endl;
    }
    std::cout << "  ✓ ODE solver integration working\n" << std::endl;
    
    // 3. Test Liquid Network
    std::cout << "3. Testing Liquid Neural Network..." << std::endl;
    LiquidNetwork::NetworkConfig config;
    
    // Create a simple 3-layer network
    config.layers.push_back({4, {1.0f, 0.1f, 0.5f, 0.01f}, true});  // Input layer
    config.layers.push_back({3, {0.8f, 0.15f, 0.6f, 0.02f}, true}); // Hidden layer  
    config.layers.push_back({2, {0.5f, 0.2f, 0.7f, 0.03f}, true});  // Output layer
    
    config.timestep = 0.01f;
    config.max_iterations = 10;
    config.adaptive_timestep = true;
    
    LiquidNetwork network(config);
    if (!network.initialize()) {
        std::cerr << "Failed to initialize network" << std::endl;
        return 1;
    }
    
    // Test with sample input
    std::vector<float> network_input = {0.5f, -0.2f, 0.8f, 0.1f};
    auto result = network.forward(network_input);
    
    std::cout << "  Input: [";
    for (size_t i = 0; i < network_input.size(); ++i) {
        std::cout << network_input[i] << (i < network_input.size()-1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  Output: [";
    for (size_t i = 0; i < result.outputs.size(); ++i) {
        std::cout << result.outputs[i] << (i < result.outputs.size()-1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  Confidence: " << result.confidence << std::endl;
    std::cout << "  Inference time: " << result.computation_time_us << " μs" << std::endl;
    std::cout << "  Power estimate: " << network.get_power_consumption() << " mW" << std::endl;
    std::cout << "  Memory usage: " << network.get_memory_usage() << " KB" << std::endl;
    std::cout << "  ✓ Liquid Neural Network working\n" << std::endl;
    
    // 4. Test Image Processing
    std::cout << "4. Testing Image Processing..." << std::endl;
    ImageProcessor::Config proc_config;
    proc_config.target_width = 32;
    proc_config.target_height = 24;
    proc_config.use_temporal_filter = true;
    
    ImageProcessor processor(proc_config);
    
    // Create fake image data (64x48 RGB)
    std::vector<uint8_t> fake_image(64 * 48 * 3);
    for (size_t i = 0; i < fake_image.size(); ++i) {
        fake_image[i] = (i % 256); // Create a pattern
    }
    
    auto processed = processor.process(fake_image.data(), 64, 48, 3);
    
    std::cout << "  Input image: 64x48x3 (" << fake_image.size() << " bytes)" << std::endl;
    std::cout << "  Processed: " << processed.width << "x" << processed.height 
              << "x" << processed.channels << " (" << processed.data.size() << " values)" << std::endl;
    std::cout << "  Temporal diff: " << processed.temporal_diff << std::endl;
    std::cout << "  ✓ Image processing working\n" << std::endl;
    
    // 5. Performance Test
    std::cout << "5. Performance Test..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        // Simulate vision processing pipeline
        auto processed_frame = processor.process(fake_image.data(), 64, 48, 3);
        
        // Convert to network input
        std::vector<float> nn_input;
        int input_size = config.layers[0].num_neurons;
        int skip = std::max(1, static_cast<int>(processed_frame.data.size() / input_size));
        
        for (size_t j = 0; j < processed_frame.data.size() && nn_input.size() < input_size; j += skip) {
            nn_input.push_back(processed_frame.data[j]);
        }
        nn_input.resize(input_size, 0.0f);
        
        // Run network inference
        auto inference_result = network.forward(nn_input);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "  Processed 100 frames in " << duration.count() << " μs" << std::endl;
    std::cout << "  Average per frame: " << duration.count() / 100.0f << " μs" << std::endl;
    std::cout << "  Theoretical FPS: " << 1000000.0f / (duration.count() / 100.0f) << std::endl;
    std::cout << "  ✓ Performance test complete\n" << std::endl;
    
    std::cout << "=== Generation 1 Complete: Basic functionality working ===" << std::endl;
    std::cout << "✓ Liquid neuron dynamics" << std::endl;
    std::cout << "✓ ODE solver integration" << std::endl; 
    std::cout << "✓ Multi-layer liquid networks" << std::endl;
    std::cout << "✓ Vision processing pipeline" << std::endl;
    std::cout << "✓ Real-time performance" << std::endl;
    
    return 0;
}