#include "../tests/framework/simple_test.h"
#include "../include/liquid_vision/vision/image_processor.h"
#include <vector>

using namespace LiquidVision;

void test_image_processor_initialization() {
    ImageProcessor::Config config;
    config.target_width = 160;
    config.target_height = 120;
    config.use_temporal_filter = false;
    
    ImageProcessor processor(config);
    ASSERT_TRUE(true, "ImageProcessor initialized successfully");
}

void test_image_resize() {
    ImageProcessor::Config config;
    config.target_width = 80;
    config.target_height = 60;
    config.use_temporal_filter = false;
    
    ImageProcessor processor(config);
    
    // Create test image (160x120 -> 80x60)
    std::vector<uint8_t> input_image(160 * 120 * 3);
    
    // Fill with a simple pattern
    for (int y = 0; y < 120; ++y) {
        for (int x = 0; x < 160; ++x) {
            int idx = (y * 160 + x) * 3;
            input_image[idx] = x % 256;     // R
            input_image[idx + 1] = y % 256; // G
            input_image[idx + 2] = 128;     // B
        }
    }
    
    auto result = processor.process(input_image.data(), 160, 120, 3);
    
    // Should resize to target dimensions
    int expected_size = 80 * 60 * 3;
    ASSERT_EQ(expected_size, result.data.size(), "Data size should match target dimensions");
    ASSERT_EQ(80, result.width, "Width should match target");
    ASSERT_EQ(60, result.height, "Height should match target");
    ASSERT_EQ(3, result.channels, "Channels should be preserved");
}

void test_image_normalization() {
    ImageProcessor::Config config;
    config.target_width = 4;
    config.target_height = 4;
    config.use_temporal_filter = false;
    
    ImageProcessor processor(config);
    
    // Create test image with known values
    std::vector<uint8_t> input_image = {
        255, 0, 128,  // Pixel 1
        0, 255, 64,   // Pixel 2
        128, 128, 192, // etc...
        64, 192, 255,
    };
    
    // Fill remaining pixels
    while (input_image.size() < 4 * 4 * 3) {
        input_image.push_back(100);
    }
    
    auto result = processor.process(input_image.data(), 4, 4, 3);
    
    // Values should be normalized to [0, 1] range for typical image processing
    for (float val : result.data) {
        ASSERT_GE(val, 0.0f, "Normalized values should be >= 0.0");
        ASSERT_LE(val, 1.0f, "Normalized values should be <= 1.0");
    }
}

void test_performance() {
    ImageProcessor::Config config;
    config.target_width = 160;
    config.target_height = 120;
    config.use_temporal_filter = false;
    
    ImageProcessor processor(config);
    
    // Create test image
    std::vector<uint8_t> input_image(640 * 480 * 3, 128);
    
    // Process image and measure basic functionality
    auto result = processor.process(input_image.data(), 640, 480, 3);
    
    ASSERT_EQ(160, result.width, "Performance test width");
    ASSERT_EQ(120, result.height, "Performance test height");
    ASSERT_TRUE(result.data.size() > 0, "Performance test should produce output");
}

int main() {
    std::cout << "Running Image Processor Tests..." << std::endl;
    
    RUN_TEST(test_image_processor_initialization);
    RUN_TEST(test_image_resize);
    RUN_TEST(test_image_normalization);
    RUN_TEST(test_performance);
    
    std::cout << "All tests completed." << std::endl;
    return 0;
}