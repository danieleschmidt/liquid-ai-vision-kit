#include "../tests/framework/simple_test.h"
#include "../include/liquid_vision/vision/image_processor.h"
#include <vector>

using namespace LiquidVision;

void test_image_processor_initialization() {
    ImageProcessor processor;
    ImageProcessor::Config config;
    
    config.target_width = 160;
    config.target_height = 120;
    config.normalize = true;
    
    bool success = processor.configure(config);
    ASSERT_TRUE(success);
}

void test_image_resize() {
    ImageProcessor processor;
    ImageProcessor::Config config;
    
    config.target_width = 80;
    config.target_height = 60;
    config.normalize = false;
    
    processor.configure(config);
    
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
    ASSERT_EQ(result.data.size(), expected_size);
    ASSERT_EQ(result.width, 80);
    ASSERT_EQ(result.height, 60);
    ASSERT_EQ(result.channels, 3);
}

void test_image_normalization() {
    ImageProcessor processor;
    ImageProcessor::Config config;
    
    config.target_width = 4;
    config.target_height = 4;
    config.normalize = true;
    
    processor.configure(config);
    
    // Create test image with known values
    std::vector<uint8_t> input_image = {
        255, 0, 128,  // Pixel 1
        0, 255, 64,   // Pixel 2
        128, 128, 192, // etc...
        64, 192, 255,
        // ... continue for 4x4 image
    };
    
    // Fill remaining pixels
    while (input_image.size() < 4 * 4 * 3) {
        input_image.push_back(100);
    }
    
    auto result = processor.process(input_image.data(), 4, 4, 3);
    
    // Values should be normalized to [-1, 1] range
    for (float val : result.data) {
        ASSERT_GE(val, -1.0f);
        ASSERT_LE(val, 1.0f);
    }
    
    // Check specific known conversions
    // 255 -> 1.0, 0 -> -1.0, 128 -> ~0.0
    ASSERT_FLOAT_NEAR(result.data[0], 1.0f, 0.01f);   // 255 normalized
    ASSERT_FLOAT_NEAR(result.data[1], -1.0f, 0.01f);  // 0 normalized
    ASSERT_FLOAT_NEAR(result.data[2], 0.0f, 0.1f);    // 128 normalized
}

void test_grayscale_conversion() {
    ImageProcessor processor;
    ImageProcessor::Config config;
    
    config.target_width = 2;
    config.target_height = 2;
    config.convert_to_grayscale = true;
    
    processor.configure(config);
    
    // RGB input
    std::vector<uint8_t> rgb_image = {
        255, 0, 0,    // Red pixel
        0, 255, 0,    // Green pixel
        0, 0, 255,    // Blue pixel
        255, 255, 255 // White pixel
    };
    
    auto result = processor.process(rgb_image.data(), 2, 2, 3);
    
    // Should output single channel
    ASSERT_EQ(result.channels, 1);
    ASSERT_EQ(result.data.size(), 2 * 2 * 1);
    
    // Grayscale values should be reasonable
    for (float val : result.data) {
        ASSERT_GE(val, -1.0f);
        ASSERT_LE(val, 1.0f);
    }
}

void test_edge_cases() {
    ImageProcessor processor;
    ImageProcessor::Config config;
    
    config.target_width = 1;
    config.target_height = 1;
    
    processor.configure(config);
    
    // Single pixel image
    std::vector<uint8_t> single_pixel = {128, 128, 128};
    auto result = processor.process(single_pixel.data(), 1, 1, 3);
    
    ASSERT_EQ(result.width, 1);
    ASSERT_EQ(result.height, 1);
    ASSERT_EQ(result.data.size(), 3);
    
    // Empty image should be handled gracefully
    std::vector<uint8_t> empty_image;
    auto empty_result = processor.process(nullptr, 0, 0, 0);
    
    ASSERT_EQ(empty_result.width, 0);
    ASSERT_EQ(empty_result.height, 0);
    ASSERT_TRUE(empty_result.data.empty());
}

void test_performance() {
    ImageProcessor processor;
    ImageProcessor::Config config;
    
    config.target_width = 160;
    config.target_height = 120;
    config.normalize = true;
    
    processor.configure(config);
    
    // Large test image
    std::vector<uint8_t> test_image(640 * 480 * 3);
    for (auto& pixel : test_image) {
        pixel = rand() % 256;
    }
    
    // Should process without timeout
    for (int i = 0; i < 10; ++i) {
        auto result = processor.process(test_image.data(), 640, 480, 3);
        ASSERT_EQ(result.width, 160);
        ASSERT_EQ(result.height, 120);
    }
}

int main() {
    RUN_TEST(test_image_processor_initialization);
    RUN_TEST(test_image_resize);
    RUN_TEST(test_image_normalization);
    RUN_TEST(test_grayscale_conversion);
    RUN_TEST(test_edge_cases);
    RUN_TEST(test_performance);
    
    return 0;
}