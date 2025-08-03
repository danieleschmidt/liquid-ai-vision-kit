#pragma once

#include <vector>
#include <memory>
#include <cstdint>

namespace LiquidVision {

/**
 * Processed frame structure for neural network input
 */
struct ProcessedFrame {
    std::vector<float> data;
    int width = 0;
    int height = 0;
    int channels = 0;
    float temporal_diff = 0.0f;  // Motion estimate
    uint32_t timestamp_us = 0;
};

/**
 * Image preprocessing pipeline optimized for embedded systems
 */
class ImageProcessor {
public:
    struct Config {
        int target_width = 160;
        int target_height = 120;
        bool use_temporal_filter = true;
        bool use_edge_detection = false;
        float downsample_ratio = 1.0f;
        float noise_threshold = 0.01f;
    };

private:
    Config config_;
    ProcessedFrame previous_frame_;
    std::vector<float> temporal_buffer_;
    bool has_previous_frame_ = false;
    
    // Edge detection kernels
    static constexpr float SOBEL_X[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    static constexpr float SOBEL_Y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

public:
    explicit ImageProcessor(const Config& config);
    ~ImageProcessor() = default;

    ProcessedFrame process(const uint8_t* raw_data, int width, int height, int channels);
    
    // Core processing functions
    ProcessedFrame resize(const uint8_t* data, int src_width, int src_height, 
                         int channels, int dst_width, int dst_height);
    
    void normalize(ProcessedFrame& frame);
    void apply_temporal_filter(ProcessedFrame& frame);
    void detect_edges(ProcessedFrame& frame);
    float compute_motion_estimate(const ProcessedFrame& current, const ProcessedFrame& previous);
    
    // Utility functions
    static void rgb_to_grayscale(const uint8_t* rgb, float* gray, int pixels);
    static void apply_gaussian_blur(float* data, int width, int height);
    
    const Config& get_config() const { return config_; }
    void reset() { has_previous_frame_ = false; }

private:
    // Fast resize using nearest neighbor or bilinear interpolation
    void fast_resize_nearest(const uint8_t* src, float* dst,
                            int src_width, int src_height,
                            int dst_width, int dst_height, int channels);
    
    void fast_resize_bilinear(const uint8_t* src, float* dst,
                             int src_width, int src_height,
                             int dst_width, int dst_height, int channels);
    
    // Convolution for edge detection
    void convolve_3x3(const float* input, float* output, 
                     int width, int height, const float* kernel);
};

/**
 * Region of Interest detector for adaptive processing
 */
class ROIDetector {
public:
    struct Region {
        int x, y, width, height;
        float importance_score;
    };

private:
    std::vector<Region> regions_;
    int frame_width_ = 0;
    int frame_height_ = 0;

public:
    std::vector<Region> detect(const ProcessedFrame& frame);
    ProcessedFrame extract_roi(const ProcessedFrame& frame, const Region& roi);
    
    static float compute_saliency(const float* data, int width, int height);
    static std::vector<Region> non_max_suppression(const std::vector<Region>& regions, 
                                                   float threshold);
};

/**
 * Temporal stabilization for smooth control outputs
 */
class TemporalStabilizer {
private:
    static constexpr int HISTORY_SIZE = 5;
    std::vector<ProcessedFrame> frame_history_;
    std::vector<float> motion_history_;
    int history_index_ = 0;
    bool history_full_ = false;

public:
    ProcessedFrame stabilize(const ProcessedFrame& frame);
    float get_motion_trend() const;
    void reset();
    
private:
    ProcessedFrame weighted_average(const std::vector<ProcessedFrame>& frames);
    float estimate_global_motion(const ProcessedFrame& frame1, const ProcessedFrame& frame2);
};

} // namespace LiquidVision