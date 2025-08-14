#pragma once

#include <vector>
#include <memory>
#include <cstdint>

namespace LiquidVision {

/**
 * Feature extraction for vision preprocessing
 */
class FeatureExtractor {
public:
    struct Config {
        int patch_size = 8;
        int stride = 4;
        bool normalize = true;
        float contrast_threshold = 0.1f;
    };

    struct FeatureMap {
        std::vector<float> features;
        int width = 0;
        int height = 0;
        int channels = 0;
    };

private:
    Config config_;
    std::vector<float> filter_bank_;

public:
    explicit FeatureExtractor(const Config& config = Config());
    ~FeatureExtractor() = default;

    bool initialize();
    
    FeatureMap extract_features(
        const uint8_t* image_data,
        int width,
        int height,
        int channels
    );

    FeatureMap extract_optical_flow_features(
        const uint8_t* current_frame,
        const uint8_t* previous_frame,
        int width,
        int height,
        int channels
    );

private:
    void create_gabor_filters();
    void create_edge_filters();
    
    std::vector<float> apply_filter(
        const float* image,
        int width,
        int height,
        const float* filter,
        int filter_size
    );

    std::vector<float> normalize_features(const std::vector<float>& features);
    
    float compute_local_contrast(
        const uint8_t* image,
        int x, int y,
        int width, int height,
        int patch_size
    );
};

} // namespace LiquidVision