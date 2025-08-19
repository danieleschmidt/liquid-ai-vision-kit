#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <iomanip>
#include "../core/liquid_network.h"
#include "../vision/image_processor.h"

namespace LiquidVision {

class PerformanceProfiler {
private:
    struct Measurement {
        std::string test_name;
        float duration_us;
        float memory_kb;
        float power_mw;
        float accuracy;
    };
    
    std::vector<Measurement> measurements_;

public:
    void add_measurement(const std::string& name, float duration_us, 
                        float memory_kb, float power_mw, float accuracy = 0.0f) {
        measurements_.push_back({name, duration_us, memory_kb, power_mw, accuracy});
    }
    
    void export_results(const std::string& filename) {
        std::ofstream file(filename);
        file << "Test Name,Duration (us),Memory (KB),Power (mW),Accuracy\n";
        
        for (const auto& m : measurements_) {
            file << m.test_name << "," << m.duration_us << "," 
                 << m.memory_kb << "," << m.power_mw << "," << m.accuracy << "\n";
        }
        
        std::cout << "Results exported to " << filename << std::endl;
    }
    
    void print_summary() {
        std::cout << "\nPerformance Summary:" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        std::cout << std::setw(20) << "Test" 
                  << std::setw(12) << "Time (us)"
                  << std::setw(12) << "Memory (KB)"
                  << std::setw(12) << "Power (mW)" 
                  << std::setw(12) << "Accuracy"
                  << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (const auto& m : measurements_) {
            std::cout << std::setw(20) << m.test_name
                      << std::setw(12) << std::fixed << std::setprecision(1) << m.duration_us
                      << std::setw(12) << std::fixed << std::setprecision(1) << m.memory_kb
                      << std::setw(12) << std::fixed << std::setprecision(1) << m.power_mw
                      << std::setw(12) << std::fixed << std::setprecision(3) << m.accuracy
                      << std::endl;
        }
    }
};

void benchmark_ode_solvers(PerformanceProfiler& profiler) {
    std::cout << "Benchmarking ODE Solvers..." << std::endl;
    
    // Test different ODE solver methods
    std::vector<std::string> solver_names = {"Euler", "RK4", "Adaptive_RK4"};
    
    for (const auto& solver_name : solver_names) {
        LiquidNetwork::NetworkConfig config;
        LiquidNetwork::LayerConfig layer;
        layer.num_neurons = 32;
        config.layers = {layer};
        
        if (solver_name == "Euler") {
            config.timestep = 0.001f;  // Smaller timestep for stability
        } else {
            config.timestep = 0.01f;
        }
        
        LiquidNetwork network(config);
        network.initialize();
        
        std::vector<float> input(32);
        for (auto& val : input) {
            val = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Run multiple iterations
        for (int i = 0; i < 100; ++i) {
            network.forward(input);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        profiler.add_measurement(
            "ODE_" + solver_name,
            duration / 100.0f,  // Average per iteration
            network.get_memory_usage(),
            network.get_power_consumption()
        );
    }
}

void benchmark_vision_pipeline(PerformanceProfiler& profiler) {
    std::cout << "Benchmarking Vision Pipeline..." << std::endl;
    
    std::vector<std::pair<int, int>> resolutions = {
        {80, 60}, {160, 120}, {320, 240}, {640, 480}
    };
    
    for (const auto& res : resolutions) {
        ImageProcessor::Config config;
        config.target_width = res.first;
        config.target_height = res.second;
        ImageProcessor processor(config);
        
        // Config already applied in constructor
        
        // Generate test image
        std::vector<uint8_t> test_image(res.first * res.second * 3);
        for (auto& pixel : test_image) {
            pixel = rand() % 256;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Process image multiple times
        for (int i = 0; i < 50; ++i) {
            auto processed = processor.process(test_image.data(), res.first, res.second, 3);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        std::string test_name = "Vision_" + std::to_string(res.first) + "x" + std::to_string(res.second);
        
        profiler.add_measurement(
            test_name,
            duration / 50.0f,  // Average per frame
            (res.first * res.second * 3) / 1024.0f,  // Memory approximation
            50.0f  // Estimated power consumption
        );
    }
}

void benchmark_integrated_system(PerformanceProfiler& profiler) {
    std::cout << "Benchmarking Integrated System..." << std::endl;
    
    // Create full pipeline
    LiquidNetwork::NetworkConfig net_config;
    LiquidNetwork::LayerConfig layer1, layer2;
    
    layer1.num_neurons = 64;
    layer2.num_neurons = 16;
    
    net_config.layers = {layer1, layer2};
    net_config.adaptive_timestep = true;
    
    LiquidNetwork network(net_config);
    network.initialize();
    
    ImageProcessor::Config img_config;
    img_config.target_width = 160;
    img_config.target_height = 120;
    ImageProcessor processor(img_config);
    
    // Generate test data
    std::vector<uint8_t> camera_frame(160 * 120 * 3);
    for (auto& pixel : camera_frame) {
        pixel = rand() % 256;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Full pipeline
    for (int i = 0; i < 20; ++i) {
        // Image processing
        auto processed_image = processor.process(camera_frame.data(), 160, 120, 3);
        
        // Neural network inference
        auto network_result = network.forward(processed_image.data);
        
        // Simulate control output processing
        float control_output = network_result.outputs.empty() ? 0.0f : network_result.outputs[0];
        (void)control_output; // Suppress unused variable warning
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    profiler.add_measurement(
        "Integrated_Pipeline",
        duration / 20.0f,
        network.get_memory_usage() + (160 * 120 * 3) / 1024.0f,
        network.get_power_consumption() + 100.0f  // Vision processing power
    );
}

} // namespace LiquidVision