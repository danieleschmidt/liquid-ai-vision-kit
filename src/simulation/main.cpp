#include <iostream>
#include <chrono>
#include <vector>
#include "../core/liquid_network.h"
#include "../vision/image_processor.h"

using namespace LiquidVision;

int main() {
    std::cout << "Liquid Vision Simulation" << std::endl;
    
    // Create a simple test network
    LiquidNetwork::NetworkConfig config;
    LiquidNetwork::LayerConfig layer1;
    layer1.num_neurons = 16;
    layer1.params.tau = 1.0f;
    layer1.params.leak = 0.1f;
    
    LiquidNetwork::LayerConfig layer2;
    layer2.num_neurons = 8;
    layer2.params.tau = 0.5f;
    layer2.params.leak = 0.2f;
    
    config.layers = {layer1, layer2};
    config.timestep = 0.01f;
    config.adaptive_timestep = true;
    
    LiquidNetwork network(config);
    
    if (!network.initialize()) {
        std::cerr << "Failed to initialize network" << std::endl;
        return 1;
    }
    
    // Run simulation with random inputs
    std::vector<float> input(16);
    for (int i = 0; i < 1000; ++i) {
        // Generate random input
        for (auto& val : input) {
            val = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = network.forward(input);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (i % 100 == 0) {
            std::cout << "Step " << i << ": ";
            std::cout << "Time: " << duration.count() << "us, ";
            std::cout << "Confidence: " << result.confidence << ", ";
            std::cout << "Output: [";
            for (size_t j = 0; j < result.outputs.size() && j < 3; ++j) {
                std::cout << result.outputs[j];
                if (j < result.outputs.size() - 1 && j < 2) std::cout << ", ";
            }
            if (result.outputs.size() > 3) std::cout << "...";
            std::cout << "]" << std::endl;
        }
    }
    
    std::cout << "Simulation completed successfully" << std::endl;
    return 0;
}