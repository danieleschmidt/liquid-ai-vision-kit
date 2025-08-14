#include "flight_controller.h"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>

namespace LiquidVision {

// SafetyMonitor implementation
// Constructor implementations are now inline in header

bool SafetyMonitor::check_command(FlightCommand& cmd, const DroneState& state) {
    violations_.clear();
    
    if (emergency_stop_) {
        violations_.push_back("Emergency stop activated");
        cmd.velocity_x = 0;
        cmd.velocity_y = 0;
        cmd.velocity_z = 0;
        cmd.yaw_rate = 0;
        return false;
    }
    
    // Check velocity limits
    float velocity_magnitude = std::sqrt(cmd.velocity_x * cmd.velocity_x + 
                                        cmd.velocity_y * cmd.velocity_y + 
                                        cmd.velocity_z * cmd.velocity_z);
    
    if (velocity_magnitude > limits_.max_velocity) {
        violations_.push_back("Velocity limit exceeded");
        // Scale down velocities
        float scale = limits_.max_velocity / velocity_magnitude;
        cmd.velocity_x *= scale;
        cmd.velocity_y *= scale;
        cmd.velocity_z *= scale;
    }
    
    // Check angular rate limits
    if (std::abs(cmd.yaw_rate) > limits_.max_angular_rate) {
        violations_.push_back("Angular rate limit exceeded");
        cmd.yaw_rate = std::copysign(limits_.max_angular_rate, cmd.yaw_rate);
    }
    
    // Check angle limits
    if (std::abs(cmd.roll_angle) > limits_.max_angle) {
        violations_.push_back("Roll angle limit exceeded");
        cmd.roll_angle = std::copysign(limits_.max_angle, cmd.roll_angle);
    }
    
    if (std::abs(cmd.pitch_angle) > limits_.max_angle) {
        violations_.push_back("Pitch angle limit exceeded");
        cmd.pitch_angle = std::copysign(limits_.max_angle, cmd.pitch_angle);
    }
    
    // Check altitude limits
    float predicted_altitude = state.z + cmd.velocity_z * 0.1f; // 100ms prediction
    if (predicted_altitude > limits_.max_altitude) {
        violations_.push_back("Maximum altitude limit approached");
        cmd.velocity_z = std::min(cmd.velocity_z, 0.0f);
    }
    
    if (predicted_altitude < limits_.min_altitude) {
        violations_.push_back("Minimum altitude limit approached");
        cmd.velocity_z = std::max(cmd.velocity_z, 0.0f);
    }
    
    return violations_.empty();
}

bool SafetyMonitor::check_state(const DroneState& state) {
    violations_.clear();
    
    // Check battery
    if (!check_battery(state.battery_voltage, state.battery_percent)) {
        violations_.push_back("Low battery");
    }
    
    // Check geofence
    if (!check_geofence(state.x, state.y, state.z)) {
        violations_.push_back("Geofence violation");
    }
    
    // Check attitude
    if (std::abs(state.roll) > limits_.max_angle || 
        std::abs(state.pitch) > limits_.max_angle) {
        violations_.push_back("Attitude limit exceeded");
    }
    
    return violations_.empty();
}

void SafetyMonitor::limit_command(FlightCommand& cmd) {
    // Apply hard limits to all command values
    cmd.velocity_x = std::clamp(cmd.velocity_x, -limits_.max_velocity, limits_.max_velocity);
    cmd.velocity_y = std::clamp(cmd.velocity_y, -limits_.max_velocity, limits_.max_velocity);
    cmd.velocity_z = std::clamp(cmd.velocity_z, -limits_.max_velocity, limits_.max_velocity);
    cmd.yaw_rate = std::clamp(cmd.yaw_rate, -limits_.max_angular_rate, limits_.max_angular_rate);
    cmd.roll_angle = std::clamp(cmd.roll_angle, -limits_.max_angle, limits_.max_angle);
    cmd.pitch_angle = std::clamp(cmd.pitch_angle, -limits_.max_angle, limits_.max_angle);
}

bool SafetyMonitor::check_geofence(float x, float y, float z) {
    float distance = std::sqrt(x * x + y * y);
    return distance < limits_.geofence_radius && 
           z < limits_.max_altitude && 
           z > limits_.min_altitude;
}

bool SafetyMonitor::check_battery(float voltage, float percent) {
    return voltage > limits_.min_battery_voltage || percent > 20.0f;
}

// PX4Controller implementation
PX4Controller::PX4Controller(const std::string& port, int baudrate) 
    : port_name_(port), baudrate_(baudrate) {}

PX4Controller::~PX4Controller() {
    disconnect();
}

bool PX4Controller::connect() {
    // In real implementation, this would open serial port and establish MAVLink connection
    // For now, simulate connection
    connected_ = true;
    
    // Initialize state
    current_state_ = DroneState();
    current_state_.battery_voltage = 12.6f;
    current_state_.battery_percent = 100.0f;
    
    return true;
}

bool PX4Controller::disconnect() {
    connected_ = false;
    return true;
}

bool PX4Controller::send_command(const FlightCommand& cmd) {
    if (!connected_) return false;
    
    // Convert to MAVLink velocity setpoint
    return send_velocity_setpoint(cmd.velocity_x, cmd.velocity_y, 
                                 cmd.velocity_z, cmd.yaw_rate);
}

DroneState PX4Controller::get_state() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    update_telemetry();
    return current_state_;
}

bool PX4Controller::arm() {
    if (!connected_) return false;
    
    // Send MAVLink ARM command
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_state_.armed = true;
    return true;
}

bool PX4Controller::disarm() {
    if (!connected_) return false;
    
    // Send MAVLink DISARM command
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_state_.armed = false;
    return true;
}

bool PX4Controller::takeoff(float altitude) {
    if (!connected_ || !current_state_.armed) return false;
    
    // Send MAVLink TAKEOFF command
    return send_position_setpoint(current_state_.x, current_state_.y, 
                                 altitude, current_state_.yaw);
}

bool PX4Controller::land() {
    if (!connected_) return false;
    
    // Send MAVLink LAND command
    return send_velocity_setpoint(0, 0, -0.5f, 0);
}

bool PX4Controller::set_mode(uint8_t mode) {
    if (!connected_) return false;
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_state_.flight_mode = mode;
    return true;
}

bool PX4Controller::send_heartbeat() {
    // Send MAVLink heartbeat message
    return connected_;
}

bool PX4Controller::send_velocity_setpoint(float vx, float vy, float vz, float yaw_rate) {
    // Send MAVLink SET_POSITION_TARGET_LOCAL_NED message
    // with velocity control bits set
    return connected_;
}

bool PX4Controller::send_position_setpoint(float x, float y, float z, float yaw) {
    // Send MAVLink SET_POSITION_TARGET_LOCAL_NED message
    // with position control bits set
    return connected_;
}

void PX4Controller::update_telemetry() {
    // In real implementation, this would parse incoming MAVLink messages
    // For now, simulate some telemetry updates
    auto now = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
    current_state_.timestamp_us = now;
    
    // Simulate some movement
    current_state_.x += current_state_.vx * 0.01f;
    current_state_.y += current_state_.vy * 0.01f;
    current_state_.z += current_state_.vz * 0.01f;
}

// SimulationController implementation
SimulationController::SimulationController() {
    simulated_state_ = DroneState();
    simulated_state_.battery_voltage = 12.6f;
    simulated_state_.battery_percent = 100.0f;
    simulated_state_.z = 0.0f; // Start on ground
}

SimulationController::~SimulationController() {
    disconnect();
}

bool SimulationController::connect() {
    if (connected_) return true;
    
    connected_ = true;
    running_ = true;
    simulation_thread_ = std::thread(&SimulationController::simulation_loop, this);
    
    return true;
}

bool SimulationController::disconnect() {
    if (!connected_) return true;
    
    running_ = false;
    if (simulation_thread_.joinable()) {
        simulation_thread_.join();
    }
    
    connected_ = false;
    return true;
}

bool SimulationController::send_command(const FlightCommand& cmd) {
    if (!connected_) return false;
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    command_queue_.push(cmd);
    
    return true;
}

DroneState SimulationController::get_state() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return simulated_state_;
}

bool SimulationController::arm() {
    if (!connected_) return false;
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    simulated_state_.armed = true;
    return true;
}

bool SimulationController::disarm() {
    if (!connected_) return false;
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    simulated_state_.armed = false;
    simulated_state_.vx = 0;
    simulated_state_.vy = 0;
    simulated_state_.vz = 0;
    return true;
}

bool SimulationController::takeoff(float altitude) {
    if (!connected_ || !simulated_state_.armed) return false;
    
    // Simulate takeoff
    FlightCommand cmd;
    cmd.velocity_z = 1.0f; // Climb at 1 m/s
    send_command(cmd);
    
    return true;
}

bool SimulationController::land() {
    if (!connected_) return false;
    
    // Simulate landing
    FlightCommand cmd;
    cmd.velocity_z = -0.5f; // Descend at 0.5 m/s
    send_command(cmd);
    
    return true;
}

bool SimulationController::set_mode(uint8_t mode) {
    if (!connected_) return false;
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    simulated_state_.flight_mode = mode;
    return true;
}

void SimulationController::simulation_loop() {
    const float dt = 0.01f; // 100Hz simulation rate
    
    while (running_) {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            
            // Process commands
            if (!command_queue_.empty()) {
                FlightCommand cmd = command_queue_.front();
                command_queue_.pop();
                apply_command(cmd, dt);
            }
            
            // Update physics
            update_physics(dt);
            
            // Update timestamp
            simulated_state_.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            
            // Simulate battery drain
            if (simulated_state_.armed) {
                simulated_state_.battery_percent -= 0.001f; // Slow battery drain
                simulated_state_.battery_voltage = 10.0f + 2.6f * (simulated_state_.battery_percent / 100.0f);
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void SimulationController::update_physics(float dt) {
    // Simple physics simulation
    
    // Update position from velocity
    simulated_state_.x += simulated_state_.vx * dt;
    simulated_state_.y += simulated_state_.vy * dt;
    simulated_state_.z += simulated_state_.vz * dt;
    
    // Update orientation from angular velocities
    simulated_state_.roll += simulated_state_.roll_rate * dt;
    simulated_state_.pitch += simulated_state_.pitch_rate * dt;
    simulated_state_.yaw += simulated_state_.yaw_rate * dt;
    
    // Apply damping
    const float damping = 0.98f;
    simulated_state_.vx *= damping;
    simulated_state_.vy *= damping;
    simulated_state_.vz *= damping;
    simulated_state_.roll_rate *= damping;
    simulated_state_.pitch_rate *= damping;
    simulated_state_.yaw_rate *= damping;
    
    // Ground constraint
    if (simulated_state_.z < 0) {
        simulated_state_.z = 0;
        simulated_state_.vz = 0;
    }
    
    // Normalize yaw to [-pi, pi]
    while (simulated_state_.yaw > M_PI) simulated_state_.yaw -= 2 * M_PI;
    while (simulated_state_.yaw < -M_PI) simulated_state_.yaw += 2 * M_PI;
}

void SimulationController::apply_command(const FlightCommand& cmd, float dt) {
    if (!simulated_state_.armed) return;
    
    // Apply velocities with acceleration limits
    const float max_acceleration = 2.0f; // m/s^2
    
    float dvx = cmd.velocity_x - simulated_state_.vx;
    float dvy = cmd.velocity_y - simulated_state_.vy;
    float dvz = cmd.velocity_z - simulated_state_.vz;
    
    // Limit acceleration
    float max_dv = max_acceleration * dt;
    dvx = std::clamp(dvx, -max_dv, max_dv);
    dvy = std::clamp(dvy, -max_dv, max_dv);
    dvz = std::clamp(dvz, -max_dv, max_dv);
    
    simulated_state_.vx += dvx;
    simulated_state_.vy += dvy;
    simulated_state_.vz += dvz;
    
    // Apply angular rates
    simulated_state_.yaw_rate = cmd.yaw_rate;
    simulated_state_.roll_rate = 0; // Simplified - no direct roll/pitch control
    simulated_state_.pitch_rate = 0;
}

// FlightController implementation
FlightController::FlightController(ControllerType type) {
    switch (type) {
        case ControllerType::PX4:
            controller_ = std::make_unique<PX4Controller>();
            break;
        case ControllerType::SIMULATION:
            controller_ = std::make_unique<SimulationController>();
            break;
        default:
            controller_ = std::make_unique<SimulationController>();
    }
    
    last_update_ = std::chrono::steady_clock::now();
}

FlightController::~FlightController() {
    disconnect();
}

bool FlightController::initialize() {
    return controller_ != nullptr;
}

bool FlightController::connect() {
    if (!controller_) return false;
    return controller_->connect();
}

bool FlightController::disconnect() {
    if (!controller_) return false;
    return controller_->disconnect();
}

bool FlightController::send_control(float forward_velocity, float yaw_rate, float altitude) {
    FlightCommand cmd;
    cmd.velocity_x = forward_velocity;
    cmd.yaw_rate = yaw_rate;
    
    // Simple altitude hold
    DroneState current = get_current_state();
    float altitude_error = altitude - current.z;
    cmd.velocity_z = std::clamp(altitude_error * 0.5f, -1.0f, 1.0f);
    
    return send_command(cmd);
}

bool FlightController::send_command(const FlightCommand& cmd) {
    if (!controller_ || !controller_->is_connected()) return false;
    
    FlightCommand safe_cmd = cmd;
    
    // Apply safety checks unless overridden
    if (!safety_override_) {
        DroneState current_state = controller_->get_state();
        safety_monitor_.check_command(safe_cmd, current_state);
        
        if (!safety_monitor_.is_safe()) {
            std::cerr << "Safety violation detected!" << std::endl;
            for (const auto& violation : safety_monitor_.get_violations()) {
                std::cerr << "  - " << violation << std::endl;
            }
        }
    }
    
    // Apply command smoothing
    safe_cmd = smooth_command(safe_cmd);
    
    // Send to controller
    bool success = controller_->send_command(safe_cmd);
    
    if (success) {
        previous_command_ = safe_cmd;
    }
    
    return success;
}

DroneState FlightController::get_current_state() {
    if (!controller_) return DroneState();
    
    DroneState state = controller_->get_state();
    
    // Update state estimation
    update_state_estimation();
    
    // Check state safety
    if (!safety_override_) {
        safety_monitor_.check_state(state);
    }
    
    return state;
}

bool FlightController::arm() {
    if (!controller_) return false;
    return controller_->arm();
}

bool FlightController::disarm() {
    if (!controller_) return false;
    return controller_->disarm();
}

bool FlightController::takeoff(float altitude) {
    if (!controller_) return false;
    
    // Safety check
    DroneState current = get_current_state();
    if (current.z > 0.5f) {
        std::cerr << "Already airborne, cannot takeoff" << std::endl;
        return false;
    }
    
    return controller_->takeoff(altitude);
}

bool FlightController::land() {
    if (!controller_) return false;
    return controller_->land();
}

bool FlightController::emergency_stop() {
    safety_monitor_.trigger_emergency_stop();
    
    // Send zero velocity command
    FlightCommand stop_cmd;
    stop_cmd.velocity_x = 0;
    stop_cmd.velocity_y = 0;
    stop_cmd.velocity_z = 0;
    stop_cmd.yaw_rate = 0;
    
    return controller_->send_command(stop_cmd);
}

FlightCommand FlightController::smooth_command(const FlightCommand& cmd) {
    FlightCommand smoothed = cmd;
    
    // Apply exponential smoothing
    smoothed.velocity_x = command_smoothing_factor_ * cmd.velocity_x + 
                         (1 - command_smoothing_factor_) * previous_command_.velocity_x;
    smoothed.velocity_y = command_smoothing_factor_ * cmd.velocity_y + 
                         (1 - command_smoothing_factor_) * previous_command_.velocity_y;
    smoothed.velocity_z = command_smoothing_factor_ * cmd.velocity_z + 
                         (1 - command_smoothing_factor_) * previous_command_.velocity_z;
    smoothed.yaw_rate = command_smoothing_factor_ * cmd.yaw_rate + 
                       (1 - command_smoothing_factor_) * previous_command_.yaw_rate;
    
    return smoothed;
}

void FlightController::update_state_estimation() {
    // Simple state estimation with prediction
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - last_update_).count();
    last_update_ = now;
    
    if (controller_) {
        DroneState current = controller_->get_state();
        
        // Simple velocity estimation from position changes
        float dx = current.x - estimated_state_.x;
        float dy = current.y - estimated_state_.y;
        float dz = current.z - estimated_state_.z;
        
        if (dt > 0.001f && dt < 1.0f) {
            estimated_state_.vx = 0.8f * estimated_state_.vx + 0.2f * (dx / dt);
            estimated_state_.vy = 0.8f * estimated_state_.vy + 0.2f * (dy / dt);
            estimated_state_.vz = 0.8f * estimated_state_.vz + 0.2f * (dz / dt);
        }
        
        estimated_state_ = current;
    }
}

// WaypointNavigator implementation
float WaypointNavigator::PIDController::update(float error, float dt) {
    // Proportional term
    float p = kp * error;
    
    // Integral term
    integral += error * dt;
    integral = std::clamp(integral, -max_output, max_output);
    float i = ki * integral;
    
    // Derivative term
    float derivative = (error - previous_error) / dt;
    float d = kd * derivative;
    
    previous_error = error;
    
    // Total output
    float output = p + i + d;
    return std::clamp(output, -max_output, max_output);
}

void WaypointNavigator::PIDController::reset() {
    integral = 0;
    previous_error = 0;
}

void WaypointNavigator::add_waypoint(const Waypoint& wp) {
    waypoints_.push_back(wp);
}

void WaypointNavigator::clear_waypoints() {
    waypoints_.clear();
    current_waypoint_ = 0;
    mission_active_ = false;
}

void WaypointNavigator::start_mission() {
    if (!waypoints_.empty()) {
        current_waypoint_ = 0;
        mission_active_ = true;
        pid_x_.reset();
        pid_y_.reset();
        pid_z_.reset();
        pid_yaw_.reset();
    }
}

void WaypointNavigator::stop_mission() {
    mission_active_ = false;
}

FlightCommand WaypointNavigator::compute_command(const DroneState& current_state) {
    FlightCommand cmd;
    
    if (!mission_active_ || waypoints_.empty() || current_waypoint_ >= waypoints_.size()) {
        return cmd;
    }
    
    const Waypoint& target = waypoints_[current_waypoint_];
    
    // Check if waypoint reached
    if (is_waypoint_reached(current_state, target)) {
        current_waypoint_++;
        if (current_waypoint_ >= waypoints_.size()) {
            mission_active_ = false;
            return cmd;
        }
    }
    
    // Compute position errors
    float error_x = target.x - current_state.x;
    float error_y = target.y - current_state.y;
    float error_z = target.z - current_state.z;
    
    // Compute desired heading
    float desired_yaw = compute_bearing(current_state.x, current_state.y, target.x, target.y);
    float error_yaw = desired_yaw - current_state.yaw;
    
    // Normalize yaw error to [-pi, pi]
    while (error_yaw > M_PI) error_yaw -= 2 * M_PI;
    while (error_yaw < -M_PI) error_yaw += 2 * M_PI;
    
    // Update PID controllers
    const float dt = 0.01f; // Assume 100Hz update rate
    cmd.velocity_x = pid_x_.update(error_x, dt);
    cmd.velocity_y = pid_y_.update(error_y, dt);
    cmd.velocity_z = pid_z_.update(error_z, dt);
    cmd.yaw_rate = pid_yaw_.update(error_yaw, dt);
    
    // Limit to target speed
    float velocity_magnitude = std::sqrt(cmd.velocity_x * cmd.velocity_x + 
                                        cmd.velocity_y * cmd.velocity_y);
    if (velocity_magnitude > target.speed) {
        float scale = target.speed / velocity_magnitude;
        cmd.velocity_x *= scale;
        cmd.velocity_y *= scale;
    }
    
    return cmd;
}

bool WaypointNavigator::is_mission_complete() const {
    return !mission_active_ || current_waypoint_ >= waypoints_.size();
}

float WaypointNavigator::get_distance_to_waypoint(const DroneState& state) const {
    if (waypoints_.empty() || current_waypoint_ >= waypoints_.size()) {
        return 0;
    }
    
    const Waypoint& target = waypoints_[current_waypoint_];
    float dx = target.x - state.x;
    float dy = target.y - state.y;
    float dz = target.z - state.z;
    
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool WaypointNavigator::is_waypoint_reached(const DroneState& state, const Waypoint& wp) const {
    float dx = wp.x - state.x;
    float dy = wp.y - state.y;
    float dz = wp.z - state.z;
    float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    
    return distance < wp.acceptance_radius;
}

float WaypointNavigator::compute_bearing(float x1, float y1, float x2, float y2) const {
    return std::atan2(y2 - y1, x2 - x1);
}

} // namespace LiquidVision