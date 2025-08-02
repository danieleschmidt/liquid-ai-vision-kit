#include "test_helpers.h"
#include "liquid_vision/core/liquid_network.h"
#include "liquid_vision/vision/image_processor.h"
#include "test_config.h"

#ifdef USING_GTEST
#include <gtest/gtest.h>
#elif defined(USING_CATCH2)
#include <catch2/catch_test_macros.hpp>
#else
#include "simple_test.h"
#endif

using namespace LiquidVision;
using namespace LiquidVision::Testing;

// Performance test fixture
class InferencePerformanceTest {
public:
    static void testSingleInferenceLatency() {
        // Create a test network configuration
        LiquidNetwork::Config config;
        config.num_neurons = TEST_NETWORK_NEURONS;
        config.num_inputs = TEST_NETWORK_INPUTS;
        config.num_outputs = TEST_NETWORK_OUTPUTS;
        config.timestep = TEST_ODE_TIMESTEP;
        config.num_iterations = TEST_ODE_ITERATIONS;
        
        // Initialize network with test parameters
        LiquidNetwork network(config);
        
        // Generate test input
        std::vector<float> input = TestHelpers::generateRandomFloats(TEST_NETWORK_INPUTS);
        
        // Measure single inference time
        TestHelpers::Timer timer;
        timer.start();
        
        auto output = network.infer(input);
        
        double elapsed = timer.elapsed();
        
        // Verify inference completed within target time
        if (elapsed > TARGET_INFERENCE_TIME_MS) {
            throw std::runtime_error("Single inference latency exceeded target");
        }
        
        // Verify output validity
        if (output.size() != TEST_NETWORK_OUTPUTS) {
            throw std::runtime_error("Invalid output size");
        }
        
        for (float val : output) {
            if (std::isnan(val) || std::isinf(val)) {
                throw std::runtime_error("Invalid output values");
            }
        }
    }
    
    static void testBatchInferencePerformance() {
        LiquidNetwork::Config config;
        config.num_neurons = TEST_NETWORK_NEURONS;
        config.num_inputs = TEST_NETWORK_INPUTS;
        config.num_outputs = TEST_NETWORK_OUTPUTS;
        config.timestep = TEST_ODE_TIMESTEP;
        config.num_iterations = TEST_ODE_ITERATIONS;
        
        LiquidNetwork network(config);
        
        // Test batch inference performance
        const int batch_size = 100;
        std::vector<std::vector<float>> inputs;
        
        for (int i = 0; i < batch_size; ++i) {
            inputs.push_back(TestHelpers::generateRandomFloats(TEST_NETWORK_INPUTS));
        }
        
        auto inference_func = [&network, &inputs]() {
            for (const auto& input : inputs) {
                auto output = network.infer(input);
                (void)output; // Suppress unused variable warning
            }
        };
        
        auto metrics = TestHelpers::measureInferencePerformance(inference_func, 10);
        
        // Check performance requirements
        double avg_per_inference = metrics.avg_time / batch_size;
        if (avg_per_inference > TARGET_INFERENCE_TIME_MS) {
            throw std::runtime_error("Batch inference performance below target");
        }
        
        // Check consistency (low standard deviation)
        if (metrics.std_dev > metrics.avg_time * 0.1) { // 10% variation threshold
            throw std::runtime_error("Inference time too variable");
        }
    }
    
    static void testVisionPipelinePerformance() {
        // Test complete vision pipeline performance
        ImageProcessor::Config vision_config;
        vision_config.input_width = MOCK_CAMERA_WIDTH;
        vision_config.input_height = MOCK_CAMERA_HEIGHT;
        vision_config.output_width = 80;  // Downsampled
        vision_config.output_height = 60;
        vision_config.enable_temporal_filter = true;
        
        ImageProcessor processor(vision_config);
        
        // Generate test image
        auto test_image = TestHelpers::generateTestImage(MOCK_CAMERA_WIDTH, MOCK_CAMERA_HEIGHT);
        
        auto pipeline_func = [&processor, &test_image]() {
            // Process image through vision pipeline
            auto processed = processor.process(test_image.data.data(), 
                                             test_image.width, 
                                             test_image.height);
            (void)processed; // Suppress unused variable warning
        };
        
        auto metrics = TestHelpers::measureInferencePerformance(pipeline_func, 50);
        
        // Vision processing should be even faster than inference
        if (metrics.avg_time > TARGET_INFERENCE_TIME_MS * 0.5) {
            throw std::runtime_error("Vision pipeline performance below target");
        }
    }
    
    static void testMemoryUsagePerformance() {
        size_t initial_memory = TestHelpers::getCurrentMemoryUsage();
        
        // Create multiple networks to test memory usage
        std::vector<std::unique_ptr<LiquidNetwork>> networks;
        
        LiquidNetwork::Config config;
        config.num_neurons = TEST_NETWORK_NEURONS;
        config.num_inputs = TEST_NETWORK_INPUTS;
        config.num_outputs = TEST_NETWORK_OUTPUTS;
        
        const int num_networks = 10;
        for (int i = 0; i < num_networks; ++i) {
            networks.push_back(std::make_unique<LiquidNetwork>(config));
        }
        
        // Perform inference on all networks
        std::vector<float> input = TestHelpers::generateRandomFloats(TEST_NETWORK_INPUTS);
        
        for (auto& network : networks) {
            auto output = network->infer(input);
            (void)output;
        }
        
        size_t peak_memory = TestHelpers::getPeakMemoryUsage();
        size_t memory_used = peak_memory - initial_memory;
        
        // Convert to KB
        size_t memory_used_kb = memory_used / 1024;
        
        // Check memory usage is within target
        size_t expected_max_kb = TARGET_MEMORY_USAGE_KB * num_networks;
        if (memory_used_kb > expected_max_kb) {
            throw std::runtime_error("Memory usage exceeded target");
        }
    }
    
    static void testRealTimeConstraints() {
        // Test that the system can maintain real-time performance
        LiquidNetwork::Config config;
        config.num_neurons = TEST_NETWORK_NEURONS;
        config.num_inputs = TEST_NETWORK_INPUTS;
        config.num_outputs = TEST_NETWORK_OUTPUTS;
        
        LiquidNetwork network(config);
        
        const double target_frequency = 50.0; // 50 Hz
        const double target_period_ms = 1000.0 / target_frequency;
        const int test_duration_cycles = 100;
        
        std::vector<float> input = TestHelpers::generateRandomFloats(TEST_NETWORK_INPUTS);
        
        TestHelpers::Timer overall_timer;
        overall_timer.start();
        
        std::vector<double> cycle_times;
        cycle_times.reserve(test_duration_cycles);
        
        for (int cycle = 0; cycle < test_duration_cycles; ++cycle) {
            TestHelpers::Timer cycle_timer;
            cycle_timer.start();
            
            // Simulate full processing cycle
            auto output = network.infer(input);
            
            // Simulate some additional processing overhead
            for (int i = 0; i < 1000; ++i) {
                volatile float dummy = output[0] * 1.001f;
                (void)dummy;
            }
            
            double cycle_time = cycle_timer.elapsed();
            cycle_times.push_back(cycle_time);
            
            // Check if this cycle exceeded real-time constraint
            if (cycle_time > target_period_ms) {
                throw std::runtime_error("Real-time constraint violated");
            }
        }
        
        double total_time = overall_timer.elapsed();
        double avg_cycle_time = total_time / test_duration_cycles;
        
        // Check average performance
        if (avg_cycle_time > target_period_ms * 0.8) { // Allow 80% utilization
            throw std::runtime_error("Average real-time performance insufficient");
        }
        
        // Check for timing consistency
        double max_cycle_time = *std::max_element(cycle_times.begin(), cycle_times.end());
        double min_cycle_time = *std::min_element(cycle_times.begin(), cycle_times.end());
        
        if (max_cycle_time > min_cycle_time * 2.0) { // Max should not be more than 2x min
            throw std::runtime_error("Timing consistency insufficient");
        }
    }
    
    static void testConcurrentInference() {
        // Test performance under concurrent load (if threading enabled)
        #ifdef ENABLE_THREADING
        
        LiquidNetwork::Config config;
        config.num_neurons = TEST_NETWORK_NEURONS;
        config.num_inputs = TEST_NETWORK_INPUTS;
        config.num_outputs = TEST_NETWORK_OUTPUTS;
        
        const int num_threads = 4;
        const int inferences_per_thread = 25;
        
        std::vector<std::unique_ptr<LiquidNetwork>> networks;
        for (int i = 0; i < num_threads; ++i) {
            networks.push_back(std::make_unique<LiquidNetwork>(config));
        }
        
        TestHelpers::Timer timer;
        timer.start();
        
        std::vector<std::thread> threads;
        std::atomic<int> total_inferences(0);
        
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&networks, t, inferences_per_thread, &total_inferences]() {
                std::vector<float> input = TestHelpers::generateRandomFloats(TEST_NETWORK_INPUTS);
                
                for (int i = 0; i < inferences_per_thread; ++i) {
                    auto output = networks[t]->infer(input);
                    total_inferences.fetch_add(1);
                    (void)output;
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        double elapsed = timer.elapsed();
        double inferences_per_second = (total_inferences.load() * 1000.0) / elapsed;
        
        // Should achieve significant throughput with multiple threads
        double expected_min_throughput = 50.0 * num_threads * 0.7; // 70% efficiency
        if (inferences_per_second < expected_min_throughput) {
            throw std::runtime_error("Concurrent inference throughput insufficient");
        }
        
        #endif
    }
};

// Framework-specific test implementations
#ifdef USING_GTEST

TEST(InferencePerformanceTest, SingleInferenceLatency) {
    InferencePerformanceTest::testSingleInferenceLatency();
}

TEST(InferencePerformanceTest, BatchInferencePerformance) {
    InferencePerformanceTest::testBatchInferencePerformance();
}

TEST(InferencePerformanceTest, VisionPipelinePerformance) {
    InferencePerformanceTest::testVisionPipelinePerformance();
}

TEST(InferencePerformanceTest, MemoryUsagePerformance) {
    InferencePerformanceTest::testMemoryUsagePerformance();
}

TEST(InferencePerformanceTest, RealTimeConstraints) {
    InferencePerformanceTest::testRealTimeConstraints();
}

TEST(InferencePerformanceTest, ConcurrentInference) {
    InferencePerformanceTest::testConcurrentInference();
}

#elif defined(USING_CATCH2)

TEST_CASE("Single Inference Latency", "[performance]") {
    InferencePerformanceTest::testSingleInferenceLatency();
}

TEST_CASE("Batch Inference Performance", "[performance]") {
    InferencePerformanceTest::testBatchInferencePerformance();
}

TEST_CASE("Vision Pipeline Performance", "[performance]") {
    InferencePerformanceTest::testVisionPipelinePerformance();
}

TEST_CASE("Memory Usage Performance", "[performance]") {
    InferencePerformanceTest::testMemoryUsagePerformance();
}

TEST_CASE("Real-Time Constraints", "[performance]") {
    InferencePerformanceTest::testRealTimeConstraints();
}

TEST_CASE("Concurrent Inference", "[performance]") {
    InferencePerformanceTest::testConcurrentInference();
}

#else

// Simple test framework implementation
int main() {
    TestHelpers::setupTestEnvironment();
    
    int tests_passed = 0;
    int tests_failed = 0;
    
    std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"Single Inference Latency", InferencePerformanceTest::testSingleInferenceLatency},
        {"Batch Inference Performance", InferencePerformanceTest::testBatchInferencePerformance},
        {"Vision Pipeline Performance", InferencePerformanceTest::testVisionPipelinePerformance},
        {"Memory Usage Performance", InferencePerformanceTest::testMemoryUsagePerformance},
        {"Real-Time Constraints", InferencePerformanceTest::testRealTimeConstraints},
        {"Concurrent Inference", InferencePerformanceTest::testConcurrentInference}
    };
    
    printf("=== Inference Performance Tests ===\n");
    
    for (const auto& test : tests) {
        printf("Running: %s...", test.first.c_str());
        fflush(stdout);
        
        try {
            TestHelpers::Timer timer;
            timer.start();
            
            test.second();
            
            double elapsed = timer.elapsed();
            printf(" PASS (%.2fms)\n", elapsed);
            tests_passed++;
        } catch (const std::exception& e) {
            printf(" FAIL - %s\n", e.what());
            tests_failed++;
        }
    }
    
    printf("\n=== Performance Test Results ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("Total:  %d\n", tests_passed + tests_failed);
    
    if (tests_failed == 0) {
        printf("All performance tests PASSED!\n");
    } else {
        printf("Some performance tests FAILED!\n");
    }
    
    TestHelpers::teardownTestEnvironment();
    
    return tests_failed == 0 ? 0 : 1;
}

#endif