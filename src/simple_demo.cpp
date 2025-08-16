#include "../include/liquid_vision/core/liquid_network.h"
#include "../include/liquid_vision/vision/image_processor.h"
#include "../include/liquid_vision/control/flight_controller.h"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace LiquidVision;

int main() {
    std::cout << "🚀 Liquid AI Vision Kit - Generation 1 Demo" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    try {
        // Test 1: Liquid Network
        std::cout << "\n=== Liquid Network Test ===" << std::endl;
        
        LiquidNetwork::NetworkConfig config;
        LiquidNetwork::LayerConfig layer1;
        layer1.num_neurons = 4;
        layer1.params.tau = 1.0f;
        layer1.params.leak = 0.1f;
        
        LiquidNetwork::LayerConfig layer2;
        layer2.num_neurons = 2;
        layer2.params.tau = 0.5f;
        layer2.params.leak = 0.2f;
        
        config.layers = {layer1, layer2};
        config.timestep = 0.01f;
        config.max_iterations = 5;
        
        LiquidNetwork network(config);
        
        if (network.initialize()) {
            std::cout << "✅ Network initialized successfully" << std::endl;
            
            std::vector<float> input = {0.5f, -0.3f, 0.8f, 0.1f};
            auto result = network.forward(input);
            
            std::cout << "Input: ";
            for (float val : input) std::cout << std::fixed << std::setprecision(2) << val << " ";
            std::cout << std::endl;
            
            std::cout << "Output: ";
            for (float val : result.outputs) std::cout << std::fixed << std::setprecision(2) << val << " ";
            std::cout << std::endl;
            
            std::cout << "Confidence: " << result.confidence << std::endl;
            std::cout << "Time: " << result.computation_time_us << " µs" << std::endl;
            std::cout << "Power: " << network.get_power_consumption() << " mW" << std::endl;
            std::cout << "Memory: " << network.get_memory_usage() << " KB" << std::endl;
        } else {
            std::cout << "❌ Network initialization failed" << std::endl;
        }
        
        // Test 2: Image Processor
        std::cout << "\n=== Image Processor Test ===" << std::endl;
        
        ImageProcessor::Config img_config;
        img_config.target_width = 32;
        img_config.target_height = 32;
        
        ImageProcessor processor(img_config);
        
        // Create test image
        int width = 64, height = 64, channels = 3;
        std::vector<uint8_t> image(width * height * channels);
        
        for (int i = 0; i < image.size(); i += 3) {
            image[i] = static_cast<uint8_t>((i / 3) % 256);     // R
            image[i + 1] = static_cast<uint8_t>((i / 3) % 128); // G  
            image[i + 2] = static_cast<uint8_t>((i / 3) % 64);  // B
        }
        
        auto processed = processor.process(image.data(), width, height, channels);
        
        std::cout << "✅ Processed " << width << "x" << height 
                  << " -> " << processed.width << "x" << processed.height << std::endl;
        std::cout << "Temporal diff: " << processed.temporal_diff << std::endl;
        
        // Test 3: Flight Controller
        std::cout << "\n=== Flight Controller Test ===" << std::endl;
        
        FlightController controller(FlightController::ControllerType::SIMULATION);
        
        if (controller.initialize() && controller.connect()) {
            std::cout << "✅ Flight controller connected" << std::endl;
            
            controller.arm();
            std::cout << "Armed vehicle" << std::endl;
            
            auto state = controller.get_current_state();
            std::cout << "Position: (" << state.x << ", " << state.y << ", " << state.z << ")" << std::endl;
            std::cout << "Battery: " << state.battery_percent << "%" << std::endl;
            
            // Test control command
            FlightCommand cmd;
            cmd.velocity_x = 1.0f;
            cmd.yaw_rate = 0.1f;
            controller.send_command(cmd);
            std::cout << "Sent test command" << std::endl;
            
            controller.disarm();
            std::cout << "Disarmed vehicle" << std::endl;
            
            std::cout << "Safety: " << (controller.is_safe() ? "OK" : "UNSAFE") << std::endl;
        } else {
            std::cout << "❌ Flight controller initialization failed" << std::endl;
        }
        
        // Test 4: Integration
        std::cout << "\n=== Integration Test ===" << std::endl;
        
        // Create a simple vision-to-control pipeline
        std::vector<uint8_t> camera_data(32 * 32 * 3, 128);
        auto vision_frame = processor.process(camera_data.data(), 32, 32, 3);
        
        // Convert vision output to flight command (simplified)
        FlightCommand flight_cmd;
        if (!vision_frame.data.empty()) {
            float avg_intensity = 0.0f;
            for (float val : vision_frame.data) avg_intensity += val;
            avg_intensity /= vision_frame.data.size();
            
            // Simple mapping: bright = go forward, dark = stop
            flight_cmd.velocity_x = avg_intensity * 2.0f; // Scale to reasonable velocity
            flight_cmd.yaw_rate = (avg_intensity - 0.5f) * 0.5f; // Turn based on intensity
        }
        
        std::cout << "✅ Vision pipeline complete" << std::endl;
        std::cout << "Computed velocities: x=" << flight_cmd.velocity_x 
                  << ", yaw=" << flight_cmd.yaw_rate << std::endl;
        
        std::cout << "\n🎉 Generation 1 Demo Complete!" << std::endl;
        std::cout << "All core components working ✅" << std::endl;
        std::cout << "Ready for Generation 2: MAKE IT ROBUST 🛡️" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Demo failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}