#include "mock_hardware.h"
#include <random>
#include <cstring>

namespace LiquidVision {
namespace Testing {
namespace Mocks {

// Mock Camera Implementation
MockCamera::MockCamera(int width, int height, int channels) 
    : width_(width), height_(height), channels_(channels),
      frame_count_(0), noise_level_(0.1f) {
    
    // Initialize frame buffer
    frame_buffer_.resize(width_ * height_ * channels_);
    generateSyntheticFrame();
}

bool MockCamera::isOpen() const {
    return true; // Mock camera is always "open"
}

bool MockCamera::captureFrame(uint8_t* buffer, size_t buffer_size) {
    if (!buffer || buffer_size < frame_buffer_.size()) {
        return false;
    }
    
    // Generate new synthetic frame
    generateSyntheticFrame();
    
    // Copy to output buffer
    std::memcpy(buffer, frame_buffer_.data(), frame_buffer_.size());
    frame_count_++;
    
    return true;
}

void MockCamera::setFramePattern(FramePattern pattern) {
    pattern_ = pattern;
}

void MockCamera::setNoiseLevel(float noise_level) {
    noise_level_ = std::max(0.0f, std::min(1.0f, noise_level));
}

int MockCamera::getWidth() const {
    return width_;
}

int MockCamera::getHeight() const {
    return height_;
}

int MockCamera::getChannels() const {
    return channels_;
}

uint64_t MockCamera::getFrameCount() const {
    return frame_count_;
}

void MockCamera::generateSyntheticFrame() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise_dist(-noise_level_, noise_level_);
    
    switch (pattern_) {
        case FramePattern::SOLID_COLOR:
            generateSolidColor();
            break;
            
        case FramePattern::CHECKERBOARD:
            generateCheckerboard();
            break;
            
        case FramePattern::GRADIENT:
            generateGradient();
            break;
            
        case FramePattern::MOVING_OBJECT:
            generateMovingObject();
            break;
            
        case FramePattern::RANDOM_NOISE:
            generateRandomNoise();
            break;
            
        default:
            generateSolidColor();
            break;
    }
    
    // Add noise if specified
    if (noise_level_ > 0.0f) {
        for (size_t i = 0; i < frame_buffer_.size(); ++i) {
            float noisy_value = frame_buffer_[i] + noise_dist(gen) * 255.0f;
            frame_buffer_[i] = static_cast<uint8_t>(
                std::max(0.0f, std::min(255.0f, noisy_value))
            );
        }
    }
}

void MockCamera::generateSolidColor() {
    uint8_t color = static_cast<uint8_t>((frame_count_ * 5) % 256);
    std::fill(frame_buffer_.begin(), frame_buffer_.end(), color);
}

void MockCamera::generateCheckerboard() {
    const int square_size = 16;
    
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            bool is_white = ((x / square_size) + (y / square_size)) % 2 == 0;
            uint8_t value = is_white ? 255 : 0;
            
            for (int c = 0; c < channels_; ++c) {
                frame_buffer_[(y * width_ + x) * channels_ + c] = value;
            }
        }
    }
}

void MockCamera::generateGradient() {
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            uint8_t value = static_cast<uint8_t>((x * 255) / width_);
            
            for (int c = 0; c < channels_; ++c) {
                frame_buffer_[(y * width_ + x) * channels_ + c] = value;
            }
        }
    }
}

void MockCamera::generateMovingObject() {
    // Fill with dark background
    std::fill(frame_buffer_.begin(), frame_buffer_.end(), 32);
    
    // Draw moving circle
    int center_x = (frame_count_ * 2) % (width_ + 40) - 20;
    int center_y = height_ / 2;
    int radius = 20;
    
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            int dx = x - center_x;
            int dy = y - center_y;
            
            if (dx * dx + dy * dy <= radius * radius) {
                for (int c = 0; c < channels_; ++c) {
                    frame_buffer_[(y * width_ + x) * channels_ + c] = 255;
                }
            }
        }
    }
}

void MockCamera::generateRandomNoise() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    
    for (size_t i = 0; i < frame_buffer_.size(); ++i) {
        frame_buffer_[i] = dist(gen);
    }
}

// Mock Flight Controller Implementation
MockFlightController::MockFlightController() 
    : armed_(false), mode_(FlightMode::MANUAL),
      position_{0.0f, 0.0f, 0.0f}, velocity_{0.0f, 0.0f, 0.0f},
      attitude_{0.0f, 0.0f, 0.0f}, angular_velocity_{0.0f, 0.0f, 0.0f},
      battery_voltage_(12.6f), command_count_(0) {
}

bool MockFlightController::isConnected() const {
    return true; // Mock FC is always "connected"
}

bool MockFlightController::arm() {
    armed_ = true;
    return true;
}

bool MockFlightController::disarm() {
    armed_ = false;
    return true;
}

bool MockFlightController::setFlightMode(FlightMode mode) {
    mode_ = mode;
    return true;
}

bool MockFlightController::sendVelocityCommand(float vx, float vy, float vz, float yaw_rate) {
    // Simulate vehicle response to commands
    if (armed_ && mode_ == FlightMode::OFFBOARD) {
        velocity_.x += vx * 0.1f; // Simple integration
        velocity_.y += vy * 0.1f;
        velocity_.z += vz * 0.1f;
        angular_velocity_.z = yaw_rate;
        
        // Update position based on velocity
        position_.x += velocity_.x * 0.02f; // 50Hz update rate
        position_.y += velocity_.y * 0.02f;
        position_.z += velocity_.z * 0.02f;
        
        // Update attitude based on angular velocity
        attitude_.z += angular_velocity_.z * 0.02f;
        
        command_count_++;
        return true;
    }
    return false;
}

bool MockFlightController::sendPositionCommand(float x, float y, float z, float yaw) {
    if (armed_ && mode_ == FlightMode::OFFBOARD) {
        // Simple position control simulation
        position_.x = x;
        position_.y = y;
        position_.z = z;
        attitude_.z = yaw;
        
        command_count_++;
        return true;
    }
    return false;
}

Position MockFlightController::getPosition() const {
    return position_;
}

Velocity MockFlightController::getVelocity() const {
    return velocity_;
}

Attitude MockFlightController::getAttitude() const {
    return attitude_;
}

AngularVelocity MockFlightController::getAngularVelocity() const {
    return angular_velocity_;
}

float MockFlightController::getBatteryVoltage() const {
    // Simulate slowly draining battery
    return battery_voltage_ - (command_count_ * 0.0001f);
}

bool MockFlightController::isArmed() const {
    return armed_;
}

MockFlightController::FlightMode MockFlightController::getFlightMode() const {
    return mode_;
}

uint64_t MockFlightController::getCommandCount() const {
    return command_count_;
}

void MockFlightController::simulateGPSLoss() {
    // Simulate GPS loss by adding noise to position
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 1.0f);
    
    position_.x += noise(gen);
    position_.y += noise(gen);
}

void MockFlightController::simulateWindDisturbance(float wind_x, float wind_y) {
    // Add wind disturbance to velocity
    velocity_.x += wind_x * 0.1f;
    velocity_.y += wind_y * 0.1f;
}

void MockFlightController::reset() {
    armed_ = false;
    mode_ = FlightMode::MANUAL;
    position_ = {0.0f, 0.0f, 0.0f};
    velocity_ = {0.0f, 0.0f, 0.0f};
    attitude_ = {0.0f, 0.0f, 0.0f};
    angular_velocity_ = {0.0f, 0.0f, 0.0f};
    battery_voltage_ = 12.6f;
    command_count_ = 0;
}

// Mock Sensor Implementation
MockSensor::MockSensor(SensorType type) 
    : type_(type), enabled_(true), sample_count_(0) {
}

MockSensor::SensorData MockSensor::readData() {
    SensorData data = {};
    
    if (!enabled_) {
        return data;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.1f);
    
    switch (type_) {
        case SensorType::ACCELEROMETER:
            data.x = 0.0f + noise(gen); // Gravity on Z would be -9.81
            data.y = 0.0f + noise(gen);
            data.z = -9.81f + noise(gen);
            break;
            
        case SensorType::GYROSCOPE:
            data.x = 0.0f + noise(gen);
            data.y = 0.0f + noise(gen);
            data.z = 0.0f + noise(gen);
            break;
            
        case SensorType::MAGNETOMETER:
            data.x = 0.3f + noise(gen); // Typical Earth magnetic field
            data.y = 0.1f + noise(gen);
            data.z = 0.5f + noise(gen);
            break;
            
        case SensorType::BAROMETER:
            data.pressure = 101325.0f + noise(gen) * 100.0f; // Sea level pressure
            data.temperature = 20.0f + noise(gen);
            break;
            
        case SensorType::GPS:
            data.latitude = 37.7749f + noise(gen) * 0.0001f; // San Francisco
            data.longitude = -122.4194f + noise(gen) * 0.0001f;
            data.altitude = 100.0f + noise(gen);
            break;
    }
    
    data.timestamp = sample_count_++;
    return data;
}

void MockSensor::setEnabled(bool enabled) {
    enabled_ = enabled;
}

bool MockSensor::isEnabled() const {
    return enabled_;
}

MockSensor::SensorType MockSensor::getType() const {
    return type_;
}

uint64_t MockSensor::getSampleCount() const {
    return sample_count_;
}

void MockSensor::reset() {
    sample_count_ = 0;
}

} // namespace Mocks
} // namespace Testing
} // namespace LiquidVision