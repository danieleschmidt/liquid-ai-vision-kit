#include <iostream>
#include <vector>
#include <random>
#include <cmath>

namespace LiquidVision {

class DroneSimulator {
private:
    float x_, y_, z_;
    float vx_, vy_, vz_;
    float yaw_, pitch_, roll_;
    std::mt19937 rng_;
    std::normal_distribution<float> noise_dist_;

public:
    DroneSimulator() 
        : x_(0), y_(0), z_(0)
        , vx_(0), vy_(0), vz_(0)
        , yaw_(0), pitch_(0), roll_(0)
        , rng_(12345)
        , noise_dist_(0.0f, 0.1f) {
    }
    
    void update(float dt, float thrust, float yaw_rate) {
        // Simple physics simulation
        float gravity = -9.81f;
        
        // Add control inputs
        vz_ += (thrust - gravity) * dt;
        yaw_ += yaw_rate * dt;
        
        // Update position
        x_ += vx_ * dt;
        y_ += vy_ * dt;
        z_ += vz_ * dt;
        
        // Add noise
        x_ += noise_dist_(rng_) * dt;
        y_ += noise_dist_(rng_) * dt;
        
        // Ground constraint
        if (z_ < 0) {
            z_ = 0;
            vz_ = 0;
        }
        
        // Apply drag
        vx_ *= 0.95f;
        vy_ *= 0.95f;
        vz_ *= 0.98f;
    }
    
    std::vector<uint8_t> generate_camera_frame(int width, int height) {
        std::vector<uint8_t> frame(width * height * 3);
        
        // Generate synthetic terrain with obstacles
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = (y * width + x) * 3;
                
                // Terrain color based on drone position
                float world_x = x_ + (x - width/2) * 0.01f;
                float world_y = y_ + (y - height/2) * 0.01f;
                
                // Simple pattern generation
                float pattern = std::sin(world_x * 0.1f) * std::cos(world_y * 0.1f);
                
                if (pattern > 0.5f) {
                    // Obstacle (red)
                    frame[idx] = 200;
                    frame[idx + 1] = 50;
                    frame[idx + 2] = 50;
                } else if (pattern > 0.0f) {
                    // Ground (green)
                    frame[idx] = 50;
                    frame[idx + 1] = 150;
                    frame[idx + 2] = 50;
                } else {
                    // Sky (blue)
                    frame[idx] = 100;
                    frame[idx + 1] = 150;
                    frame[idx + 2] = 200;
                }
                
                // Add altitude effect
                float altitude_factor = std::max(0.1f, 1.0f - z_ * 0.1f);
                frame[idx] *= altitude_factor;
                frame[idx + 1] *= altitude_factor;
                frame[idx + 2] *= altitude_factor;
            }
        }
        
        return frame;
    }
    
    float get_x() const { return x_; }
    float get_y() const { return y_; }
    float get_z() const { return z_; }
    float get_yaw() const { return yaw_; }
};

} // namespace LiquidVision