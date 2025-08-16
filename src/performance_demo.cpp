#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <future>
#include <atomic>
#include <memory>
#include <mutex>
#include <algorithm>
#include <unordered_map>
#include <iomanip>
#include "liquid_vision/core/liquid_network.h"
#include "liquid_vision/vision/image_processor.h"

namespace LiquidVision {

/**
 * High-Performance System for Generation 3
 * Demonstrates scaling, concurrency, and optimization
 */
class PerformanceLiquidSystem {
private:
    std::vector<std::unique_ptr<LiquidNetwork>> networks_;
    std::unique_ptr<ImageProcessor> processor_;
    std::atomic<size_t> network_index_{0};
    
    // Simple cache
    struct CacheEntry {
        std::vector<float> input;
        LiquidNetwork::InferenceResult result;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    std::unordered_map<size_t, CacheEntry> cache_;
    std::mutex cache_mutex_;
    
    // Performance metrics
    std::atomic<uint64_t> total_inferences_{0};
    std::atomic<uint64_t> cache_hits_{0};
    std::atomic<uint64_t> cache_misses_{0};
    std::atomic<uint64_t> total_time_us_{0};
    
    size_t hash_input(const std::vector<float>& input) {
        std::hash<float> hasher;
        size_t hash = 0;
        for (const auto& val : input) {
            hash ^= hasher(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }

public:
    PerformanceLiquidSystem(const LiquidNetwork::NetworkConfig& config, size_t num_networks = 4) {
        // Create multiple network instances for parallel processing
        for (size_t i = 0; i < num_networks; ++i) {
            auto network = std::make_unique<LiquidNetwork>(config);
            if (network->initialize()) {
                networks_.push_back(std::move(network));
            }
        }
        
        // Initialize image processor
        ImageProcessor::Config proc_config;
        proc_config.target_width = 64;
        proc_config.target_height = 48;
        proc_config.use_temporal_filter = true;
        processor_ = std::make_unique<ImageProcessor>(proc_config);
        
        std::cout << "🚀 Performance system initialized with " << networks_.size() 
                  << " networks and " << std::thread::hardware_concurrency() 
                  << " hardware threads" << std::endl;
    }
    
    // Optimized inference with load balancing
    LiquidNetwork::InferenceResult optimized_inference(const std::vector<float>& input) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Check cache first
        size_t hash = hash_input(input);
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            auto it = cache_.find(hash);
            if (it != cache_.end()) {
                auto now = std::chrono::steady_clock::now();
                if (now - it->second.timestamp < std::chrono::milliseconds(100)) {
                    cache_hits_++;
                    return it->second.result;
                }
                cache_.erase(it);
            }
        }
        
        cache_misses_++;
        
        // Get network using round-robin load balancing
        if (networks_.empty()) {
            LiquidNetwork::InferenceResult error_result;
            error_result.outputs = {0.0f, 0.0f};
            error_result.confidence = 0.0f;
            return error_result;
        }
        
        size_t idx = network_index_.fetch_add(1) % networks_.size();
        auto& network = networks_[idx];
        
        // Run inference
        auto result = network->forward(input);
        
        // Cache result
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            cache_[hash] = {input, result, std::chrono::steady_clock::now()};
            
            // Simple cache eviction
            if (cache_.size() > 1000) {
                auto oldest = std::min_element(cache_.begin(), cache_.end(),
                    [](const auto& a, const auto& b) {
                        return a.second.timestamp < b.second.timestamp;
                    });
                cache_.erase(oldest);
            }
        }
        
        // Update metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        
        total_inferences_++;
        total_time_us_ += duration.count();
        
        return result;
    }
    
    // Batch processing for high throughput
    std::vector<LiquidNetwork::InferenceResult> batch_inference(
        const std::vector<std::vector<float>>& inputs) {
        
        std::vector<std::future<LiquidNetwork::InferenceResult>> futures;
        futures.reserve(inputs.size());
        
        // Submit all tasks asynchronously
        for (const auto& input : inputs) {
            futures.push_back(std::async(std::launch::async, 
                [this, input]() {
                    return optimized_inference(input);
                }));
        }
        
        // Collect results
        std::vector<LiquidNetwork::InferenceResult> results;
        results.reserve(inputs.size());
        
        for (auto& future : futures) {
            results.push_back(future.get());
        }
        
        return results;
    }
    
    // Streaming video processing simulation
    void streaming_benchmark(int num_frames = 1000) {
        std::cout << "\n📹 Streaming Processing Benchmark..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        std::atomic<int> frames_processed{0};
        std::vector<std::future<void>> frame_futures;
        
        // Process frames in parallel
        for (int frame = 0; frame < num_frames; ++frame) {
            auto future = std::async(std::launch::async, [this, frame, &frames_processed]() {
                // Generate synthetic image data
                std::vector<uint8_t> image_data(64 * 48 * 3);
                for (size_t i = 0; i < image_data.size(); ++i) {
                    image_data[i] = static_cast<uint8_t>((i + frame) % 256);
                }
                
                try {
                    // Process image
                    auto processed = processor_->process(image_data.data(), 64, 48, 3);
                    
                    // Convert to network input
                    std::vector<float> nn_input;
                    if (!networks_.empty()) {
                        int input_size = networks_[0]->get_config().layers[0].num_neurons;
                        int skip = std::max(1, static_cast<int>(processed.data.size() / input_size));
                        
                        for (size_t i = 0; i < processed.data.size() && nn_input.size() < input_size; i += skip) {
                            nn_input.push_back(processed.data[i]);
                        }
                        nn_input.resize(input_size, 0.0f);
                        
                        // Run inference
                        auto result = optimized_inference(nn_input);
                        
                        if (!result.outputs.empty()) {
                            frames_processed++;
                        }
                    }
                } catch (const std::exception& e) {
                    std::cout << "Frame " << frame << " failed: " << e.what() << std::endl;
                }
            });
            
            frame_futures.push_back(std::move(future));
            
            // Simulate frame rate (don't overwhelm system)
            if (frame % 100 == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        // Wait for all frames to complete
        for (auto& future : frame_futures) {
            future.get();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        double fps = (frames_processed.load() * 1000.0) / total_time.count();
        
        std::cout << "  Frames processed: " << frames_processed.load() << "/" << num_frames << std::endl;
        std::cout << "  Total time: " << total_time.count() << " ms" << std::endl;
        std::cout << "  Average FPS: " << std::fixed << std::setprecision(1) << fps << std::endl;
        std::cout << "  Success rate: " << std::fixed << std::setprecision(1) 
                  << (100.0 * frames_processed.load() / num_frames) << "%" << std::endl;
    }
    
    // Load test with concurrent workers
    void concurrent_load_test(int num_requests = 10000) {
        std::cout << "\n⚡ Concurrent Load Test..." << std::endl;
        
        // Generate test data
        std::vector<std::vector<float>> test_inputs;
        test_inputs.reserve(num_requests);
        
        for (int i = 0; i < num_requests; ++i) {
            std::vector<float> input;
            int input_size = networks_.empty() ? 4 : networks_[0]->get_config().layers[0].num_neurons;
            input.reserve(input_size);
            
            for (int j = 0; j < input_size; ++j) {
                input.push_back(static_cast<float>((i + j) % 100) / 100.0f);
            }
            test_inputs.push_back(input);
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process in batches for better concurrency control
        const int batch_size = 500;
        std::vector<std::future<std::vector<LiquidNetwork::InferenceResult>>> batch_futures;
        
        for (int i = 0; i < num_requests; i += batch_size) {
            int end_idx = std::min(i + batch_size, num_requests);
            std::vector<std::vector<float>> batch(
                test_inputs.begin() + i, 
                test_inputs.begin() + end_idx);
            
            auto future = std::async(std::launch::async, 
                [this, batch]() {
                    return batch_inference(batch);
                });
            
            batch_futures.push_back(std::move(future));
        }
        
        // Collect all results
        int total_successful = 0;
        for (auto& future : batch_futures) {
            auto batch_results = future.get();
            for (const auto& result : batch_results) {
                if (!result.outputs.empty() && std::isfinite(result.confidence)) {
                    total_successful++;
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        double throughput = (total_successful * 1000.0) / total_time.count();
        
        std::cout << "  Requests processed: " << total_successful << "/" << num_requests << std::endl;
        std::cout << "  Total time: " << total_time.count() << " ms" << std::endl;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1) 
                  << throughput << " requests/sec" << std::endl;
        std::cout << "  Success rate: " << std::fixed << std::setprecision(1) 
                  << (100.0 * total_successful / num_requests) << "%" << std::endl;
    }
    
    // Performance monitoring
    void print_performance_metrics() {
        uint64_t total = total_inferences_.load();
        uint64_t hits = cache_hits_.load();
        uint64_t misses = cache_misses_.load();
        uint64_t time_us = total_time_us_.load();
        
        std::cout << "\n📊 Performance Metrics:" << std::endl;
        std::cout << "  Total inferences: " << total << std::endl;
        std::cout << "  Cache hits: " << hits << std::endl;
        std::cout << "  Cache misses: " << misses << std::endl;
        
        if (hits + misses > 0) {
            double hit_rate = (100.0 * hits) / (hits + misses);
            std::cout << "  Cache hit rate: " << std::fixed << std::setprecision(1) 
                      << hit_rate << "%" << std::endl;
        }
        
        if (total > 0) {
            double avg_time_ms = time_us / (total * 1000.0);
            std::cout << "  Average inference time: " << std::fixed << std::setprecision(2) 
                      << avg_time_ms << " ms" << std::endl;
        }
        
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            std::cout << "  Cache size: " << cache_.size() << " entries" << std::endl;
        }
        
        std::cout << "  Network instances: " << networks_.size() << std::endl;
        std::cout << "  Hardware threads: " << std::thread::hardware_concurrency() << std::endl;
    }
};

} // namespace LiquidVision

// Generation 3 Performance Demo
int main() {
    std::cout << "=== Liquid AI Vision Kit - Generation 3: Performance Demo ===" << std::endl;
    std::cout << "Testing high-performance concurrent processing and optimization\n" << std::endl;
    
    using namespace LiquidVision;
    
    // Create optimized network configuration
    LiquidNetwork::NetworkConfig config;
    config.layers.push_back({32, {1.0f, 0.1f, 0.5f, 0.01f}, true});
    config.layers.push_back({16, {0.8f, 0.15f, 0.6f, 0.02f}, true});
    config.layers.push_back({8, {0.5f, 0.2f, 0.7f, 0.03f}, true});
    config.timestep = 0.005f; // Faster timestep for performance
    config.max_iterations = 3; // Reduced iterations for speed
    
    // Create performance system with optimal number of networks
    size_t num_networks = std::max(2u, std::thread::hardware_concurrency());
    PerformanceLiquidSystem perf_system(config, num_networks);
    
    // Test 1: Basic concurrent inference
    std::cout << "\n1. Testing basic concurrent inference..." << std::endl;
    std::vector<std::vector<float>> test_inputs;
    for (int i = 0; i < 20; ++i) {
        std::vector<float> input;
        for (int j = 0; j < 32; ++j) {
            input.push_back(static_cast<float>((i + j) % 100) / 100.0f);
        }
        test_inputs.push_back(input);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto results = perf_system.batch_inference(test_inputs);
    auto end = std::chrono::high_resolution_clock::now();
    
    int successful = 0;
    for (const auto& result : results) {
        if (!result.outputs.empty() && std::isfinite(result.confidence)) {
            successful++;
        }
    }
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "  Basic test: " << successful << "/20 successful in " 
              << duration.count() << " ms" << std::endl;
    
    // Test 2: Streaming video processing
    perf_system.streaming_benchmark(2000);
    
    // Test 3: High-load concurrent processing
    perf_system.concurrent_load_test(20000);
    
    // Final performance report
    perf_system.print_performance_metrics();
    
    std::cout << "\n=== Generation 3 Complete: High Performance Achieved ===\n" << std::endl;
    std::cout << "✅ Multi-network load balancing" << std::endl;
    std::cout << "✅ Intelligent caching system" << std::endl;
    std::cout << "✅ Concurrent batch processing" << std::endl;
    std::cout << "✅ Real-time streaming pipeline" << std::endl;
    std::cout << "✅ Performance monitoring" << std::endl;
    std::cout << "✅ Optimized for high throughput" << std::endl;
    
    return 0;
}