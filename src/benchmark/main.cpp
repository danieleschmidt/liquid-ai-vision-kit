#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include "../core/liquid_network.h"

using namespace LiquidVision;

struct BenchmarkResult {
    float avg_inference_time_us = 0.0f;
    float min_inference_time_us = 1e6f;
    float max_inference_time_us = 0.0f;
    float avg_power_mw = 0.0f;
    float avg_confidence = 0.0f;
    int total_samples = 0;
};

void run_performance_benchmark() {
    std::cout << "Running Performance Benchmarks..." << std::endl;
    
    // Test different network configurations
    std::vector<int> neuron_counts = {8, 16, 32, 64};
    std::vector<float> timesteps = {0.001f, 0.01f, 0.05f};
    
    for (int neurons : neuron_counts) {
        for (float timestep : timesteps) {
            LiquidNetwork::NetworkConfig config;
            LiquidNetwork::LayerConfig layer;
            layer.num_neurons = neurons;
            layer.params.tau = 1.0f;
            layer.params.leak = 0.1f;
            
            config.layers = {layer};
            config.timestep = timestep;
            config.adaptive_timestep = false;
            
            LiquidNetwork network(config);
            network.initialize();
            
            BenchmarkResult result;
            std::vector<float> input(neurons);
            
            // Warm-up
            for (int i = 0; i < 10; ++i) {
                for (auto& val : input) {
                    val = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
                }
                network.forward(input);
            }
            
            // Actual benchmark
            const int num_samples = 1000;
            for (int i = 0; i < num_samples; ++i) {
                for (auto& val : input) {
                    val = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
                }
                
                auto start = std::chrono::high_resolution_clock::now();
                auto inference_result = network.forward(input);
                auto end = std::chrono::high_resolution_clock::now();
                
                float time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                
                result.avg_inference_time_us += time_us;
                result.min_inference_time_us = std::min(result.min_inference_time_us, time_us);
                result.max_inference_time_us = std::max(result.max_inference_time_us, time_us);
                result.avg_confidence += inference_result.confidence;
                result.total_samples++;
            }
            
            result.avg_inference_time_us /= num_samples;
            result.avg_confidence /= num_samples;
            result.avg_power_mw = network.get_power_consumption();
            
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Neurons: " << std::setw(3) << neurons
                      << ", Timestep: " << std::setw(6) << timestep
                      << ", Avg Time: " << std::setw(8) << result.avg_inference_time_us << "us"
                      << ", Min: " << std::setw(6) << result.min_inference_time_us << "us"
                      << ", Max: " << std::setw(6) << result.max_inference_time_us << "us"
                      << ", Power: " << std::setw(6) << result.avg_power_mw << "mW"
                      << ", Conf: " << std::setw(5) << result.avg_confidence
                      << std::endl;
        }
    }
}

void run_memory_benchmark() {
    std::cout << "\nRunning Memory Benchmarks..." << std::endl;
    
    std::vector<int> layer_sizes = {8, 16, 32, 64, 128};
    
    for (int size : layer_sizes) {
        LiquidNetwork::NetworkConfig config;
        LiquidNetwork::LayerConfig layer;
        layer.num_neurons = size;
        
        config.layers = {layer};
        
        LiquidNetwork network(config);
        network.initialize();
        
        float memory_kb = network.get_memory_usage();
        
        std::cout << "Layer Size: " << std::setw(3) << size
                  << ", Memory Usage: " << std::setw(8) << memory_kb << " KB"
                  << std::endl;
    }
}

int main() {
    std::cout << "=====================================" << std::endl;
    std::cout << "   Liquid Vision Benchmark Suite    " << std::endl;
    std::cout << "=====================================" << std::endl;
    
    run_performance_benchmark();
    run_memory_benchmark();
    
    std::cout << "\nBenchmark completed successfully!" << std::endl;
    return 0;
}