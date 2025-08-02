#pragma once

#include <vector>
#include <cstdint>

namespace LiquidVision {
namespace Testing {
namespace Mocks {

// Mock Camera for testing vision pipeline
class MockCamera {
public:
    enum class FramePattern {
        SOLID_COLOR,
        CHECKERBOARD,
        GRADIENT,
        MOVING_OBJECT,
        RANDOM_NOISE
    };
    
    MockCamera(int width = 160, int height = 120, int channels = 3);
    
    // Camera interface
    bool isOpen() const;
    bool captureFrame(uint8_t* buffer, size_t buffer_size);
    
    // Configuration
    void setFramePattern(FramePattern pattern);
    void setNoiseLevel(float noise_level);
    
    // Properties
    int getWidth() const;
    int getHeight() const;
    int getChannels() const;
    uint64_t getFrameCount() const;
    
private:
    int width_, height_, channels_;
    std::vector<uint8_t> frame_buffer_;
    FramePattern pattern_ = FramePattern::CHECKERBOARD;
    uint64_t frame_count_;
    float noise_level_;
    
    void generateSyntheticFrame();
    void generateSolidColor();
    void generateCheckerboard();
    void generateGradient();
    void generateMovingObject();
    void generateRandomNoise();
};

// Mock Flight Controller for testing control loops
class MockFlightController {
public:
    enum class FlightMode {
        MANUAL,
        STABILIZE,
        ALTITUDE_HOLD,
        POSITION_HOLD,
        OFFBOARD,
        RETURN_TO_LAND,
        LAND
    };
    
    struct Position {
        float x, y, z;
    };
    
    struct Velocity {
        float x, y, z;
    };
    
    struct Attitude {
        float roll, pitch, yaw; // radians
    };
    
    struct AngularVelocity {
        float x, y, z; // rad/s
    };
    
    MockFlightController();
    
    // Connection interface
    bool isConnected() const;
    
    // Arming interface
    bool arm();
    bool disarm();
    bool isArmed() const;
    
    // Flight mode interface
    bool setFlightMode(FlightMode mode);
    FlightMode getFlightMode() const;
    
    // Command interface
    bool sendVelocityCommand(float vx, float vy, float vz, float yaw_rate);
    bool sendPositionCommand(float x, float y, float z, float yaw);
    
    // State interface
    Position getPosition() const;
    Velocity getVelocity() const;
    Attitude getAttitude() const;
    AngularVelocity getAngularVelocity() const;
    float getBatteryVoltage() const;
    
    // Testing utilities
    uint64_t getCommandCount() const;
    void simulateGPSLoss();
    void simulateWindDisturbance(float wind_x, float wind_y);
    void reset();
    
private:
    bool armed_;
    FlightMode mode_;
    Position position_;
    Velocity velocity_;
    Attitude attitude_;
    AngularVelocity angular_velocity_;
    float battery_voltage_;
    uint64_t command_count_;
};

// Mock Sensor for testing sensor fusion
class MockSensor {
public:
    enum class SensorType {
        ACCELEROMETER,
        GYROSCOPE,
        MAGNETOMETER,
        BAROMETER,
        GPS
    };
    
    struct SensorData {
        float x, y, z;           // For IMU sensors
        float pressure;          // For barometer
        float temperature;       // For barometer
        double latitude;         // For GPS
        double longitude;        // For GPS
        float altitude;          // For GPS
        uint64_t timestamp;
    };
    
    MockSensor(SensorType type);
    
    // Sensor interface
    SensorData readData();
    void setEnabled(bool enabled);
    bool isEnabled() const;
    
    // Properties
    SensorType getType() const;
    uint64_t getSampleCount() const;
    
    // Testing utilities
    void reset();
    
private:
    SensorType type_;
    bool enabled_;
    uint64_t sample_count_;
};

// Mock System Monitor for testing resource usage
class MockSystemMonitor {
public:
    struct SystemStats {
        float cpu_usage_percent;
        size_t memory_usage_kb;
        float temperature_celsius;
        float power_consumption_mw;
        uint64_t uptime_ms;
    };
    
    MockSystemMonitor();
    
    SystemStats getStats() const;
    void setCPULoad(float load_percent);
    void setMemoryUsage(size_t usage_kb);
    void setTemperature(float temp_celsius);
    void setPowerConsumption(float power_mw);
    
    void reset();
    
private:
    SystemStats stats_;
    uint64_t start_time_ms_;
};

// Mock Timer for testing real-time constraints
class MockTimer {
public:
    MockTimer();
    
    void start();
    uint64_t elapsed_us() const;
    uint64_t elapsed_ms() const;
    
    // Testing utilities
    void setTimeScale(float scale); // Speed up/slow down time for testing
    void addDelay(uint64_t delay_us);
    
private:
    uint64_t start_time_us_;
    float time_scale_;
    uint64_t artificial_delay_us_;
    
    uint64_t getCurrentTimeUs() const;
};

} // namespace Mocks
} // namespace Testing
} // namespace LiquidVision