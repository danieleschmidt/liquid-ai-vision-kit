#include "image_processor.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>

namespace LiquidVision {

ImageProcessor::ImageProcessor(const Config& config) 
    : config_(config) {
    int buffer_size = config.target_width * config.target_height * 3;
    temporal_buffer_.resize(buffer_size, 0.0f);
}

ProcessedFrame ImageProcessor::process(const uint8_t* raw_data, int width, int height, int channels) {
    auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
    ProcessedFrame frame;
    frame.timestamp_us = timestamp;
    
    // Resize to target dimensions
    frame = resize(raw_data, width, height, channels, 
                  config_.target_width, config_.target_height);
    
    // Normalize pixel values to [-1, 1]
    normalize(frame);
    
    // Apply edge detection if enabled
    if (config_.use_edge_detection) {
        detect_edges(frame);
    }
    
    // Apply temporal filtering
    if (config_.use_temporal_filter && has_previous_frame_) {
        frame.temporal_diff = compute_motion_estimate(frame, previous_frame_);
        apply_temporal_filter(frame);
    }
    
    // Store for next iteration
    previous_frame_ = frame;
    has_previous_frame_ = true;
    
    return frame;
}

ProcessedFrame ImageProcessor::resize(const uint8_t* data, int src_width, int src_height,
                                     int channels, int dst_width, int dst_height) {
    ProcessedFrame result;
    result.width = dst_width;
    result.height = dst_height;
    result.channels = channels;
    result.data.resize(dst_width * dst_height * channels);
    
    // Use bilinear interpolation for better quality
    if (src_width > dst_width || src_height > dst_height) {
        // Downsampling - use area averaging for anti-aliasing
        fast_resize_bilinear(data, result.data.data(), 
                           src_width, src_height, 
                           dst_width, dst_height, channels);
    } else {
        // Upsampling - use nearest neighbor for speed
        fast_resize_nearest(data, result.data.data(),
                          src_width, src_height,
                          dst_width, dst_height, channels);
    }
    
    return result;
}

void ImageProcessor::normalize(ProcessedFrame& frame) {
    if (frame.data.empty()) return;
    
    // Find min and max values
    float min_val = 255.0f;
    float max_val = 0.0f;
    
    for (float& val : frame.data) {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    
    // Normalize to [0, 1] for stability
    float range = max_val - min_val;
    if (range > 0.001f) {
        for (float& val : frame.data) {
            val = (val - min_val) / range;
        }
    }
}

void ImageProcessor::apply_temporal_filter(ProcessedFrame& frame) {
    if (!has_previous_frame_ || frame.data.size() != temporal_buffer_.size()) {
        temporal_buffer_ = frame.data;
        return;
    }
    
    // Exponential moving average filter
    const float alpha = 0.7f;  // Current frame weight
    const float beta = 0.3f;   // Previous frame weight
    
    for (size_t i = 0; i < frame.data.size(); ++i) {
        temporal_buffer_[i] = alpha * frame.data[i] + beta * temporal_buffer_[i];
        frame.data[i] = temporal_buffer_[i];
    }
}

void ImageProcessor::detect_edges(ProcessedFrame& frame) {
    if (frame.channels != 1) {
        // Convert to grayscale first
        std::vector<float> gray(frame.width * frame.height);
        for (int i = 0; i < frame.width * frame.height; ++i) {
            if (frame.channels == 3) {
                // RGB to grayscale using luminance formula
                gray[i] = 0.299f * frame.data[i * 3] + 
                         0.587f * frame.data[i * 3 + 1] + 
                         0.114f * frame.data[i * 3 + 2];
            } else {
                gray[i] = frame.data[i];
            }
        }
        frame.data = gray;
        frame.channels = 1;
    }
    
    // Apply Sobel edge detection
    std::vector<float> edges_x(frame.width * frame.height);
    std::vector<float> edges_y(frame.width * frame.height);
    
    convolve_3x3(frame.data.data(), edges_x.data(), frame.width, frame.height, SOBEL_X);
    convolve_3x3(frame.data.data(), edges_y.data(), frame.width, frame.height, SOBEL_Y);
    
    // Compute gradient magnitude
    for (size_t i = 0; i < frame.data.size(); ++i) {
        frame.data[i] = std::sqrt(edges_x[i] * edges_x[i] + edges_y[i] * edges_y[i]);
    }
}

float ImageProcessor::compute_motion_estimate(const ProcessedFrame& current, 
                                             const ProcessedFrame& previous) {
    if (current.data.size() != previous.data.size()) {
        return 0.0f;
    }
    
    // Compute mean absolute difference
    float total_diff = 0.0f;
    for (size_t i = 0; i < current.data.size(); ++i) {
        total_diff += std::abs(current.data[i] - previous.data[i]);
    }
    
    return total_diff / current.data.size();
}

void ImageProcessor::rgb_to_grayscale(const uint8_t* rgb, float* gray, int pixels) {
    for (int i = 0; i < pixels; ++i) {
        gray[i] = 0.299f * rgb[i * 3] + 
                 0.587f * rgb[i * 3 + 1] + 
                 0.114f * rgb[i * 3 + 2];
    }
}

void ImageProcessor::apply_gaussian_blur(float* data, int width, int height) {
    // 3x3 Gaussian kernel
    const float kernel[9] = {
        1/16.0f, 2/16.0f, 1/16.0f,
        2/16.0f, 4/16.0f, 2/16.0f,
        1/16.0f, 2/16.0f, 1/16.0f
    };
    
    std::vector<float> temp(width * height);
    
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float sum = 0.0f;
            int kernel_idx = 0;
            
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int idx = (y + dy) * width + (x + dx);
                    sum += data[idx] * kernel[kernel_idx++];
                }
            }
            
            temp[y * width + x] = sum;
        }
    }
    
    std::memcpy(data, temp.data(), width * height * sizeof(float));
}

void ImageProcessor::fast_resize_nearest(const uint8_t* src, float* dst,
                                        int src_width, int src_height,
                                        int dst_width, int dst_height, int channels) {
    float x_ratio = static_cast<float>(src_width) / dst_width;
    float y_ratio = static_cast<float>(src_height) / dst_height;
    
    for (int y = 0; y < dst_height; ++y) {
        int src_y = static_cast<int>(y * y_ratio);
        for (int x = 0; x < dst_width; ++x) {
            int src_x = static_cast<int>(x * x_ratio);
            
            for (int c = 0; c < channels; ++c) {
                int src_idx = (src_y * src_width + src_x) * channels + c;
                int dst_idx = (y * dst_width + x) * channels + c;
                dst[dst_idx] = static_cast<float>(src[src_idx]);
            }
        }
    }
}

void ImageProcessor::fast_resize_bilinear(const uint8_t* src, float* dst,
                                         int src_width, int src_height,
                                         int dst_width, int dst_height, int channels) {
    float x_ratio = static_cast<float>(src_width - 1) / dst_width;
    float y_ratio = static_cast<float>(src_height - 1) / dst_height;
    
    for (int y = 0; y < dst_height; ++y) {
        float src_y = y * y_ratio;
        int y_low = static_cast<int>(src_y);
        int y_high = std::min(y_low + 1, src_height - 1);
        float y_weight = src_y - y_low;
        
        for (int x = 0; x < dst_width; ++x) {
            float src_x = x * x_ratio;
            int x_low = static_cast<int>(src_x);
            int x_high = std::min(x_low + 1, src_width - 1);
            float x_weight = src_x - x_low;
            
            for (int c = 0; c < channels; ++c) {
                // Bilinear interpolation
                float a = src[(y_low * src_width + x_low) * channels + c];
                float b = src[(y_low * src_width + x_high) * channels + c];
                float c_val = src[(y_high * src_width + x_low) * channels + c];
                float d = src[(y_high * src_width + x_high) * channels + c];
                
                float value = a * (1 - x_weight) * (1 - y_weight) +
                             b * x_weight * (1 - y_weight) +
                             c_val * (1 - x_weight) * y_weight +
                             d * x_weight * y_weight;
                
                dst[(y * dst_width + x) * channels + c] = value;
            }
        }
    }
}

void ImageProcessor::convolve_3x3(const float* input, float* output,
                                 int width, int height, const float* kernel) {
    // Apply 3x3 convolution
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float sum = 0.0f;
            int kernel_idx = 0;
            
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int idx = (y + dy) * width + (x + dx);
                    sum += input[idx] * kernel[kernel_idx++];
                }
            }
            
            output[y * width + x] = sum;
        }
    }
    
    // Handle borders (copy from input)
    for (int x = 0; x < width; ++x) {
        output[x] = input[x];
        output[(height - 1) * width + x] = input[(height - 1) * width + x];
    }
    for (int y = 0; y < height; ++y) {
        output[y * width] = input[y * width];
        output[y * width + width - 1] = input[y * width + width - 1];
    }
}

// ROIDetector implementation
std::vector<ROIDetector::Region> ROIDetector::detect(const ProcessedFrame& frame) {
    regions_.clear();
    frame_width_ = frame.width;
    frame_height_ = frame.height;
    
    // Divide frame into grid and compute saliency
    const int grid_size = 32;
    int grid_x = frame.width / grid_size;
    int grid_y = frame.height / grid_size;
    
    for (int y = 0; y < grid_y; ++y) {
        for (int x = 0; x < grid_x; ++x) {
            Region region;
            region.x = x * grid_size;
            region.y = y * grid_size;
            region.width = grid_size;
            region.height = grid_size;
            
            // Extract region data
            std::vector<float> region_data(grid_size * grid_size);
            for (int dy = 0; dy < grid_size; ++dy) {
                for (int dx = 0; dx < grid_size; ++dx) {
                    int src_idx = ((region.y + dy) * frame.width + (region.x + dx));
                    region_data[dy * grid_size + dx] = frame.data[src_idx];
                }
            }
            
            region.importance_score = compute_saliency(region_data.data(), grid_size, grid_size);
            
            if (region.importance_score > 0.3f) {
                regions_.push_back(region);
            }
        }
    }
    
    // Apply non-maximum suppression
    regions_ = non_max_suppression(regions_, 0.5f);
    
    return regions_;
}

ProcessedFrame ROIDetector::extract_roi(const ProcessedFrame& frame, const Region& roi) {
    ProcessedFrame result;
    result.width = roi.width;
    result.height = roi.height;
    result.channels = frame.channels;
    result.data.resize(roi.width * roi.height * frame.channels);
    
    for (int y = 0; y < roi.height; ++y) {
        for (int x = 0; x < roi.width; ++x) {
            for (int c = 0; c < frame.channels; ++c) {
                int src_idx = ((roi.y + y) * frame.width + (roi.x + x)) * frame.channels + c;
                int dst_idx = (y * roi.width + x) * frame.channels + c;
                result.data[dst_idx] = frame.data[src_idx];
            }
        }
    }
    
    return result;
}

float ROIDetector::compute_saliency(const float* data, int width, int height) {
    // Simple saliency based on local contrast
    float mean = 0.0f;
    float variance = 0.0f;
    int pixels = width * height;
    
    for (int i = 0; i < pixels; ++i) {
        mean += data[i];
    }
    mean /= pixels;
    
    for (int i = 0; i < pixels; ++i) {
        float diff = data[i] - mean;
        variance += diff * diff;
    }
    variance /= pixels;
    
    return std::sqrt(variance);
}

std::vector<ROIDetector::Region> ROIDetector::non_max_suppression(
    const std::vector<Region>& regions, float threshold) {
    
    std::vector<Region> result;
    std::vector<bool> suppressed(regions.size(), false);
    
    // Sort by importance score
    std::vector<size_t> indices(regions.size());
    for (size_t i = 0; i < regions.size(); ++i) {
        indices[i] = i;
    }
    
    std::sort(indices.begin(), indices.end(), 
        [&regions](size_t a, size_t b) {
            return regions[a].importance_score > regions[b].importance_score;
        });
    
    for (size_t i : indices) {
        if (!suppressed[i]) {
            result.push_back(regions[i]);
            
            // Suppress overlapping regions
            for (size_t j = 0; j < regions.size(); ++j) {
                if (i != j && !suppressed[j]) {
                    // Compute overlap
                    int x1 = std::max(regions[i].x, regions[j].x);
                    int y1 = std::max(regions[i].y, regions[j].y);
                    int x2 = std::min(regions[i].x + regions[i].width,
                                    regions[j].x + regions[j].width);
                    int y2 = std::min(regions[i].y + regions[i].height,
                                    regions[j].y + regions[j].height);
                    
                    if (x2 > x1 && y2 > y1) {
                        int overlap_area = (x2 - x1) * (y2 - y1);
                        int area_j = regions[j].width * regions[j].height;
                        
                        if (static_cast<float>(overlap_area) / area_j > threshold) {
                            suppressed[j] = true;
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

// TemporalStabilizer implementation
ProcessedFrame TemporalStabilizer::stabilize(const ProcessedFrame& frame) {
    // Add to history
    if (frame_history_.size() < HISTORY_SIZE) {
        frame_history_.push_back(frame);
    } else {
        frame_history_[history_index_] = frame;
    }
    
    history_index_ = (history_index_ + 1) % HISTORY_SIZE;
    
    if (frame_history_.size() < 2) {
        return frame;
    }
    
    // Compute motion between consecutive frames
    motion_history_.clear();
    for (size_t i = 1; i < frame_history_.size(); ++i) {
        float motion = estimate_global_motion(frame_history_[i-1], frame_history_[i]);
        motion_history_.push_back(motion);
    }
    
    // Apply weighted averaging for stabilization
    return weighted_average(frame_history_);
}

float TemporalStabilizer::get_motion_trend() const {
    if (motion_history_.empty()) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (float motion : motion_history_) {
        sum += motion;
    }
    
    return sum / motion_history_.size();
}

void TemporalStabilizer::reset() {
    frame_history_.clear();
    motion_history_.clear();
    history_index_ = 0;
    history_full_ = false;
}

ProcessedFrame TemporalStabilizer::weighted_average(const std::vector<ProcessedFrame>& frames) {
    if (frames.empty()) {
        return ProcessedFrame();
    }
    
    ProcessedFrame result = frames.back();
    
    // Gaussian weights for temporal averaging
    std::vector<float> weights = {0.05f, 0.1f, 0.2f, 0.3f, 0.35f};
    
    if (frames.size() < weights.size()) {
        weights.resize(frames.size());
        float sum = 0.0f;
        for (float& w : weights) {
            sum += w;
        }
        for (float& w : weights) {
            w /= sum;
        }
    }
    
    // Apply weighted average
    std::fill(result.data.begin(), result.data.end(), 0.0f);
    
    for (size_t f = 0; f < frames.size(); ++f) {
        float weight = weights[f];
        for (size_t i = 0; i < result.data.size(); ++i) {
            result.data[i] += weight * frames[f].data[i];
        }
    }
    
    return result;
}

float TemporalStabilizer::estimate_global_motion(const ProcessedFrame& frame1, 
                                                const ProcessedFrame& frame2) {
    if (frame1.data.size() != frame2.data.size()) {
        return 0.0f;
    }
    
    // Simple motion estimate using mean absolute difference
    float total_diff = 0.0f;
    for (size_t i = 0; i < frame1.data.size(); ++i) {
        total_diff += std::abs(frame1.data[i] - frame2.data[i]);
    }
    
    return total_diff / frame1.data.size();
}

} // namespace LiquidVision