#pragma once

#include <string>
#include <vector>
#include <memory>
#include <queue>
#include <mutex>
#include <thread>

namespace LiquidVision {

/**
 * Flight control command structure
 */
struct FlightCommand {
    float velocity_x = 0.0f;      // Forward/backward velocity (m/s)
    float velocity_y = 0.0f;      // Left/right velocity (m/s)
    float velocity_z = 0.0f;      // Up/down velocity (m/s)
    float yaw_rate = 0.0f;        // Yaw rate (rad/s)
    float roll_angle = 0.0f;      // Roll angle (rad)
    float pitch_angle = 0.0f;     // Pitch angle (rad)
    uint32_t timestamp_us = 0;
    uint8_t mode = 0;             // Flight mode
};

/**
 * Drone state information
 */
struct DroneState {
    // Position
    float x = 0.0f, y = 0.0f, z = 0.0f;
    
    // Orientation (Euler angles)
    float roll = 0.0f, pitch = 0.0f, yaw = 0.0f;
    
    // Velocities
    float vx = 0.0f, vy = 0.0f, vz = 0.0f;
    
    // Angular velocities
    float roll_rate = 0.0f, pitch_rate = 0.0f, yaw_rate = 0.0f;
    
    // Battery and status
    float battery_voltage = 0.0f;
    float battery_percent = 0.0f;
    bool armed = false;
    uint8_t flight_mode = 0;
    
    uint32_t timestamp_us = 0;
};

/**
 * Safety monitor for flight operations
 */
class SafetyMonitor {
public:
    struct Limits {
        float max_velocity = 5.0f;        // m/s
        float max_altitude = 100.0f;      // m
        float min_altitude = 0.5f;        // m
        float max_angle = 0.5f;           // rad (~30 degrees)
        float max_angular_rate = 1.0f;    // rad/s
        float min_battery_voltage = 10.0f; // V
        float geofence_radius = 50.0f;    // m
    };

private:
    Limits limits_;
    bool safety_enabled_ = true;
    bool emergency_stop_ = false;
    std::vector<std::string> violations_;

public:
    SafetyMonitor() : limits_() {}
    explicit SafetyMonitor(const Limits& limits) : limits_(limits) {}
    
    bool check_command(FlightCommand& cmd, const DroneState& state);
    bool check_state(const DroneState& state);
    void trigger_emergency_stop() { emergency_stop_ = true; }
    void reset_emergency_stop() { emergency_stop_ = false; }
    
    const std::vector<std::string>& get_violations() const { return violations_; }
    bool is_safe() const { return violations_.empty() && !emergency_stop_; }
    
private:
    void limit_command(FlightCommand& cmd);
    bool check_geofence(float x, float y, float z);
    bool check_battery(float voltage, float percent);
};

/**
 * Abstract base class for flight controller interfaces
 */
class FlightControllerInterface {
public:
    virtual ~FlightControllerInterface() = default;
    
    virtual bool connect() = 0;
    virtual bool disconnect() = 0;
    virtual bool is_connected() const = 0;
    
    virtual bool send_command(const FlightCommand& cmd) = 0;
    virtual DroneState get_state() = 0;
    
    virtual bool arm() = 0;
    virtual bool disarm() = 0;
    virtual bool takeoff(float altitude) = 0;
    virtual bool land() = 0;
    virtual bool set_mode(uint8_t mode) = 0;
};

/**
 * PX4 flight controller implementation
 */
class PX4Controller : public FlightControllerInterface {
private:
    bool connected_ = false;
    DroneState current_state_;
    std::mutex state_mutex_;
    
    // Communication parameters
    std::string port_name_ = "/dev/ttyACM0";
    int baudrate_ = 57600;
    
    // MAVLink parameters
    uint8_t system_id_ = 1;
    uint8_t component_id_ = 1;
    uint8_t target_system_ = 1;
    uint8_t target_component_ = 0;

public:
    PX4Controller(const std::string& port = "/dev/ttyACM0", int baudrate = 57600);
    ~PX4Controller() override;
    
    bool connect() override;
    bool disconnect() override;
    bool is_connected() const override { return connected_; }
    
    bool send_command(const FlightCommand& cmd) override;
    DroneState get_state() override;
    
    bool arm() override;
    bool disarm() override;
    bool takeoff(float altitude) override;
    bool land() override;
    bool set_mode(uint8_t mode) override;
    
private:
    bool send_heartbeat();
    bool send_velocity_setpoint(float vx, float vy, float vz, float yaw_rate);
    bool send_position_setpoint(float x, float y, float z, float yaw);
    void update_telemetry();
};

/**
 * Simulation controller for testing
 */
class SimulationController : public FlightControllerInterface {
private:
    DroneState simulated_state_;
    bool connected_ = false;
    std::thread simulation_thread_;
    bool running_ = false;
    std::mutex state_mutex_;
    std::queue<FlightCommand> command_queue_;

public:
    SimulationController();
    ~SimulationController() override;
    
    bool connect() override;
    bool disconnect() override;
    bool is_connected() const override { return connected_; }
    
    bool send_command(const FlightCommand& cmd) override;
    DroneState get_state() override;
    
    bool arm() override;
    bool disarm() override;
    bool takeoff(float altitude) override;
    bool land() override;
    bool set_mode(uint8_t mode) override;
    
private:
    void simulation_loop();
    void update_physics(float dt);
    void apply_command(const FlightCommand& cmd, float dt);
};

/**
 * Main flight controller with safety features
 */
class FlightController {
private:
    std::unique_ptr<FlightControllerInterface> controller_;
    SafetyMonitor safety_monitor_;
    bool safety_override_ = false;
    
    // Command smoothing
    FlightCommand previous_command_;
    float command_smoothing_factor_ = 0.7f;
    
    // State estimation
    DroneState estimated_state_;
    std::chrono::steady_clock::time_point last_update_;

public:
    enum class ControllerType {
        PX4,
        SIMULATION,
        ARDUPILOT
    };

    FlightController(ControllerType type = ControllerType::SIMULATION);
    ~FlightController();
    
    bool initialize();
    bool connect();
    bool disconnect();
    
    bool send_control(float forward_velocity, float yaw_rate, float altitude);
    bool send_command(const FlightCommand& cmd);
    
    DroneState get_current_state();
    bool is_safe() const { return safety_monitor_.is_safe(); }
    
    bool arm();
    bool disarm();
    bool takeoff(float altitude);
    bool land();
    bool emergency_stop();
    
    void enable_safety(bool enabled) { safety_override_ = !enabled; }
    const SafetyMonitor& get_safety_monitor() const { return safety_monitor_; }
    
private:
    FlightCommand smooth_command(const FlightCommand& cmd);
    void update_state_estimation();
};

/**
 * Waypoint navigation controller
 */
class WaypointNavigator {
public:
    struct Waypoint {
        float x, y, z;
        float yaw;
        float speed;
        float acceptance_radius;
    };

private:
    std::vector<Waypoint> waypoints_;
    size_t current_waypoint_ = 0;
    bool mission_active_ = false;
    
    // PID controllers for position control
    struct PIDController {
        float kp = 1.0f, ki = 0.1f, kd = 0.5f;
        float integral = 0.0f;
        float previous_error = 0.0f;
        float max_output = 5.0f;
        
        float update(float error, float dt);
        void reset();
    };
    
    PIDController pid_x_, pid_y_, pid_z_, pid_yaw_;

public:
    void add_waypoint(const Waypoint& wp);
    void clear_waypoints();
    void start_mission();
    void stop_mission();
    
    FlightCommand compute_command(const DroneState& current_state);
    bool is_mission_complete() const;
    float get_distance_to_waypoint(const DroneState& state) const;
    
private:
    bool is_waypoint_reached(const DroneState& state, const Waypoint& wp) const;
    float compute_bearing(float x1, float y1, float x2, float y2) const;
};

} // namespace LiquidVision