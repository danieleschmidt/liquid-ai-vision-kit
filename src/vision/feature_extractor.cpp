#include "../../include/liquid_vision/vision/image_processor.h"
#include <cmath>
#include <algorithm>

namespace LiquidVision {

// Feature extraction for vision processing
class FeatureExtractor {
public:
    struct Features {
        std::vector<float> edge_features;
        std::vector<float> texture_features;
        std::vector<float> motion_features;
        float complexity_score = 0.0f;
    };
    
    static Features extract_features(const ProcessedFrame& frame) {
        Features features;
        
        if (frame.data.empty()) {
            return features;
        }
        
        // Extract edge features using gradient magnitude
        features.edge_features = extract_edge_features(frame);
        
        // Extract texture features using local variance
        features.texture_features = extract_texture_features(frame);
        
        // Motion features from temporal difference
        features.motion_features = {frame.temporal_diff};
        
        // Compute overall complexity
        features.complexity_score = compute_complexity(features);
        
        return features;
    }
    
private:
    static std::vector<float> extract_edge_features(const ProcessedFrame& frame) {
        std::vector<float> edges;
        
        // Simple Sobel edge detection
        int width = frame.width;
        int height = frame.height;
        
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int idx = y * width + x;
                
                // Sobel X kernel
                float gx = -frame.data[(y-1)*width + (x-1)] + frame.data[(y-1)*width + (x+1)]
                          -2*frame.data[y*width + (x-1)] + 2*frame.data[y*width + (x+1)]
                          -frame.data[(y+1)*width + (x-1)] + frame.data[(y+1)*width + (x+1)];
                
                // Sobel Y kernel
                float gy = -frame.data[(y-1)*width + (x-1)] - 2*frame.data[(y-1)*width + x] - frame.data[(y-1)*width + (x+1)]
                          +frame.data[(y+1)*width + (x-1)] + 2*frame.data[(y+1)*width + x] + frame.data[(y+1)*width + (x+1)];
                
                float magnitude = std::sqrt(gx*gx + gy*gy);
                edges.push_back(magnitude);
            }
        }
        
        return edges;
    }
    
    static std::vector<float> extract_texture_features(const ProcessedFrame& frame) {
        std::vector<float> texture;
        
        // Local variance in 8x8 blocks
        int block_size = 8;
        int width = frame.width;
        int height = frame.height;
        
        for (int y = 0; y < height - block_size; y += block_size) {
            for (int x = 0; x < width - block_size; x += block_size) {
                float mean = 0.0f;
                float variance = 0.0f;
                int count = 0;
                
                // Compute mean
                for (int dy = 0; dy < block_size; ++dy) {
                    for (int dx = 0; dx < block_size; ++dx) {
                        int idx = (y + dy) * width + (x + dx);
                        if (idx < frame.data.size()) {
                            mean += frame.data[idx];
                            count++;
                        }
                    }
                }
                mean /= count;
                
                // Compute variance
                for (int dy = 0; dy < block_size; ++dy) {
                    for (int dx = 0; dx < block_size; ++dx) {
                        int idx = (y + dy) * width + (x + dx);
                        if (idx < frame.data.size()) {
                            float diff = frame.data[idx] - mean;
                            variance += diff * diff;
                        }
                    }
                }
                variance /= count;
                
                texture.push_back(std::sqrt(variance));
            }
        }
        
        return texture;
    }
    
    static float compute_complexity(const Features& features) {
        float edge_complexity = 0.0f;
        for (float edge : features.edge_features) {
            edge_complexity += edge;
        }
        edge_complexity /= std::max(1.0f, static_cast<float>(features.edge_features.size()));
        
        float texture_complexity = 0.0f;
        for (float tex : features.texture_features) {
            texture_complexity += tex;
        }
        texture_complexity /= std::max(1.0f, static_cast<float>(features.texture_features.size()));
        
        float motion_complexity = std::abs(features.motion_features.empty() ? 0.0f : features.motion_features[0]);
        
        return 0.4f * edge_complexity + 0.4f * texture_complexity + 0.2f * motion_complexity;
    }
};

} // namespace LiquidVision