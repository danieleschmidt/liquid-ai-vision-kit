#include "../include/liquid_vision/lnn_controller.hpp"
#include "../include/liquid_vision/core/liquid_network.h"
#include "../include/liquid_vision/vision/image_processor.h"
#include "../include/liquid_vision/control/flight_controller.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace LiquidVision;

/**
 * Generation 1 Demo: MAKE IT WORK (Simple)
 * Demonstrates basic functionality of the Liquid AI Vision Kit
 */

void demo_liquid_neuron() {
    std::cout << "=== Liquid Neuron Demo ===" << std::endl;
    
    // Create a simple liquid neuron
    LiquidNeuron::Parameters params;
    params.tau = 1.0f;
    params.leak = 0.1f;
    LiquidNeuron neuron(params);
    
    // Set some weights
    std::vector<float> weights = {0.5f, -0.3f, 0.8f};
    neuron.set_weights(weights);
    neuron.set_bias(0.1f);
    
    // Test with different inputs
    std::vector<float> inputs = {1.0f, 0.5f, -0.2f};
    
    std::cout << "Input: ";
    for (float val : inputs) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Update neuron multiple times to see dynamics
    std::cout << "Neuron dynamics over time:" << std::endl;
    float timestep = 0.01f;
    for (int t = 0; t < 10; ++t) {
        float output = neuron.update(inputs, timestep);
        std::cout << "t=" << t << ": state=" << neuron.get_state() 
                  << ", output=" << output << std::endl;
    }
    std::cout << std::endl;
}

void demo_ode_solver() {
    std::cout << "=== ODE Solver Demo ===" << std::endl;
    
    ODESolver::Config config;
    config.method = ODESolver::Method::RUNGE_KUTTA_4;
    config.timestep = 0.01f;
    
    ODESolver solver(config);
    
    // Define a simple oscillator: dx/dt = -k*x
    solver.set_derivative_function([](const std::vector<float>& state, 
                                     const std::vector<float>& inputs) {
        std::vector<float> derivatives(state.size());
        float k = 2.0f; // Spring constant
        for (size_t i = 0; i < state.size(); ++i) {
            derivatives[i] = -k * state[i];
        }
        return derivatives;
    });
    
    // Initial conditions
    std::vector<float> state = {1.0f};
    
    std::cout << "Simple harmonic oscillator (dx/dt = -2x):" << std::endl;
    for (int step = 0; step < 20; ++step) {
        std::cout << "t=" << step * config.timestep << ": x=" << state[0] << std::endl;
        state = solver.solve_step(state, {});
    }
    std::cout << std::endl;
}

void demo_liquid_network() {
    std::cout << "=== Liquid Network Demo ===" << std::endl;
    
    // Create a simple 3-layer network
    LiquidNetwork::NetworkConfig config;
    
    LiquidNetwork::LayerConfig input_layer;
    input_layer.num_neurons = 4;
    input_layer.params.tau = 1.0f;
    input_layer.params.leak = 0.1f;
    
    LiquidNetwork::LayerConfig hidden_layer;
    hidden_layer.num_neurons = 8;
    hidden_layer.params.tau = 0.8f;
    hidden_layer.params.leak = 0.15f;
    
    LiquidNetwork::LayerConfig output_layer;
    output_layer.num_neurons = 2;
    output_layer.params.tau = 0.5f;
    output_layer.params.leak = 0.2f;
    
    config.layers = {input_layer, hidden_layer, output_layer};
    config.timestep = 0.01f;
    config.max_iterations = 10;
    
    LiquidNetwork network(config);
    
    if (!network.initialize()) {
        std::cerr << "Failed to initialize network!" << std::endl;
        return;
    }
    
    // Test with random input
    std::vector<float> input = {0.5f, -0.3f, 0.8f, 0.1f};
    
    std::cout << "Network inference test:" << std::endl;
    std::cout << "Input: ";
    for (float val : input) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    auto result = network.forward(input);
    
    std::cout << "Output: ";
    for (float val : result.outputs) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Confidence: " << result.confidence << std::endl;
    std::cout << "Computation time: " << result.computation_time_us << " µs" << std::endl;
    std::cout << "Power consumption: " << network.get_power_consumption() << " mW" << std::endl;
    std::cout << "Memory usage: " << network.get_memory_usage() << " KB" << std::endl;
    std::cout << std::endl;
}

void demo_image_processing() {
    std::cout << "=== Image Processing Demo ===" << std::endl;
    
    ImageProcessor::Config config;
    config.target_width = 32;
    config.target_height = 32;
    config.use_temporal_filter = true;
    
    ImageProcessor processor(config);
    
    // Create synthetic image data (simulated camera frame)
    int width = 64, height = 64, channels = 3;
    std::vector<uint8_t> image_data(width * height * channels);
    
    // Fill with a simple pattern
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t intensity = static_cast<uint8_t>((x + y) * 255 / (width + height));
            int idx = (y * width + x) * channels;
            image_data[idx + 0] = intensity;      // R
            image_data[idx + 1] = intensity / 2;  // G
            image_data[idx + 2] = intensity / 4;  // B
        }
    }
    
    std::cout << "Processing " << width << "x" << height 
              << " image to " << config.target_width << "x" << config.target_height << std::endl;
    
    auto processed = processor.process(image_data.data(), width, height, channels);
    
    std::cout << "Processed image size: " << processed.width << "x" << processed.height 
              << " with " << processed.data.size() << " pixels" << std::endl;
    std::cout << "Temporal difference: " << processed.temporal_diff << std::endl;
    
    // Show some pixel values
    std::cout << "Sample pixel values: ";
    for (int i = 0; i < std::min(10, static_cast<int>(processed.data.size())); ++i) {
        std::cout << std::fixed << std::setprecision(3) << processed.data[i] << " ";
    }
    std::cout << std::endl << std::endl;
}

void demo_flight_controller() {
    std::cout << "=== Flight Controller Demo ===" << std::endl;
    
    FlightController controller(FlightController::ControllerType::SIMULATION);
    
    if (!controller.initialize()) {
        std::cerr << "Failed to initialize flight controller!" << std::endl;
        return;
    }
    
    if (!controller.connect()) {
        std::cerr << "Failed to connect to flight controller!" << std::endl;
        return;
    }
    
    std::cout << "Flight controller connected successfully" << std::endl;
    
    // Test basic commands
    std::cout << "Testing basic flight commands..." << std::endl;
    
    if (controller.arm()) {
        std::cout << "Armed successfully" << std::endl;
    }
    
    // Get current state
    auto state = controller.get_current_state();
    std::cout << "Current position: (" << state.x << ", " << state.y << ", " << state.z << ")" << std::endl;
    std::cout << "Battery: " << state.battery_percent << "%" << std::endl;
    
    // Send some test commands
    controller.send_control(1.0f, 0.1f, 2.0f); // Forward 1 m/s, yaw 0.1 rad/s, altitude 2m
    std::cout << "Sent test control command" << std::endl;
    
    if (controller.disarm()) {
        std::cout << "Disarmed successfully" << std::endl;
    }
    
    std::cout << "Safety status: " << (controller.is_safe() ? "Safe" : "Unsafe") << std::endl;
    std::cout << std::endl;
}

void demo_lnn_controller() {
    std::cout << "=== LNN Controller Demo ===" << std::endl;
    
    LNNController::Config config;
    config.input_width = 32;
    config.input_height = 32;
    config.ode_timestep = 0.01f;
    config.max_iterations = 10;
    config.use_fixed_point = false; // Use floating point for demo
    config.max_velocity = 2.0f;
    config.max_yaw_rate = 0.5f;
    
    LNNController controller(config);
    
    if (!controller.initialize()) {
        std::cerr << "Failed to initialize LNN controller!" << std::endl;
        return;
    }
    
    std::cout << "LNN Controller initialized successfully" << std::endl;
    
    // Create synthetic camera frame
    std::vector<uint8_t> camera_frame(config.input_width * config.input_height * 3);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    
    for (auto& pixel : camera_frame) {
        pixel = static_cast<uint8_t>(dis(gen));
    }
    
    std::cout << "Processing camera frame..." << std::endl;
    auto output = controller.process_frame(camera_frame);
    
    std::cout << "Control outputs:" << std::endl;
    std::cout << "  Velocity X: " << output.velocity_x << " m/s" << std::endl;
    std::cout << "  Velocity Y: " << output.velocity_y << " m/s" << std::endl;
    std::cout << "  Velocity Z: " << output.velocity_z << " m/s" << std::endl;
    std::cout << "  Yaw rate: " << output.yaw_rate << " rad/s" << std::endl;
    std::cout << "  Thrust: " << output.thrust << std::endl;
    std::cout << "  Confidence: " << output.confidence << std::endl;
    std::cout << "  Processing time: " << output.processing_time_ms << " ms" << std::endl;
    std::cout << "  Power consumption: " << output.power_consumption_mw << " mW" << std::endl;
    
    // Get performance statistics
    auto stats = controller.get_performance_stats();
    std::cout << "Performance stats:" << std::endl;
    std::cout << "  Total inferences: " << stats.total_inferences << std::endl;
    std::cout << "  Average inference time: " << stats.average_inference_time_ms << " ms" << std::endl;
    std::cout << "  Average power: " << stats.average_power_consumption_mw << " mW" << std::endl;
    std::cout << "  Memory usage: " << stats.memory_usage_kb << " KB" << std::endl;
    std::cout << std::endl;
}

void demo_integration_test() {
    std::cout << "=== Integration Test Demo ===" << std::endl;
    
    // Test complete vision-to-control pipeline
    LNNController::Config config;
    config.input_width = 64;
    config.input_height = 48;
    config.max_velocity = 3.0f;
    config.max_yaw_rate = 1.0f;
    
    LNNController vision_controller(config);
    FlightController flight_controller(FlightController::ControllerType::SIMULATION);
    
    if (!vision_controller.initialize()) {
        std::cerr << "Failed to initialize vision controller!" << std::endl;
        return;
    }
    
    if (!flight_controller.initialize() || !flight_controller.connect()) {
        std::cerr << "Failed to initialize flight controller!" << std::endl;
        return;
    }
    
    std::cout << "Integrated system initialized successfully" << std::endl;
    
    // Simulate a vision-guided flight sequence
    std::cout << "Simulating vision-guided flight..." << std::endl;
    
    flight_controller.arm();
    
    for (int frame = 0; frame < 5; ++frame) {
        // Generate synthetic camera data
        std::vector<uint8_t> camera_data(config.input_width * config.input_height * 3);
        
        // Simulate changing scene (moving pattern)
        for (int i = 0; i < camera_data.size(); i += 3) {
            uint8_t pattern = static_cast<uint8_t>((i / 3 + frame * 10) % 256);
            camera_data[i + 0] = pattern;
            camera_data[i + 1] = pattern / 2;
            camera_data[i + 2] = pattern / 4;
        }
        
        // Process frame through vision system
        auto vision_output = vision_controller.process_frame(camera_data);
        
        // Convert to flight commands
        FlightCommand cmd;
        cmd.velocity_x = vision_output.velocity_x;
        cmd.velocity_y = vision_output.velocity_y;
        cmd.velocity_z = vision_output.velocity_z;
        cmd.yaw_rate = vision_output.yaw_rate;
        
        // Send to flight controller
        flight_controller.send_command(cmd);
        
        // Get current state
        auto state = flight_controller.get_current_state();
        
        std::cout << "Frame " << frame << ":" << std::endl;
        std::cout << "  Vision confidence: " << vision_output.confidence << std::endl;
        std::cout << "  Command velocities: (" << cmd.velocity_x << ", " 
                  << cmd.velocity_y << ", " << cmd.velocity_z << ")" << std::endl;
        std::cout << "  Drone position: (" << state.x << ", " << state.y << ", " << state.z << ")" << std::endl;
        std::cout << "  Processing time: " << vision_output.processing_time_ms << " ms" << std::endl;
        
        // Small delay to simulate real-time operation
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    flight_controller.disarm();
    std::cout << "Integration test completed successfully!" << std::endl;
}

int main() {
    std::cout << "🚀 Liquid AI Vision Kit - Generation 1 Demo" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Testing core functionality and basic integration" << std::endl << std::endl;
    
    try {
        demo_liquid_neuron();
        demo_ode_solver();
        demo_liquid_network();
        demo_image_processing();
        demo_flight_controller();
        demo_lnn_controller();
        demo_integration_test();
        
        std::cout << "🎉 All demos completed successfully!" << std::endl;
        std::cout << "\nGeneration 1 (MAKE IT WORK) ✅ COMPLETE" << std::endl;
        std::cout << "Ready for Generation 2: MAKE IT ROBUST" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Demo failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}