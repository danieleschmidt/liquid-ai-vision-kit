#include "../../include/liquid_vision/utils/logger.h"
#include <iostream>
#include <iomanip>
#include <thread>
#include <filesystem>
#include <cstdio>

namespace LiquidVision {

// Static member definitions
std::unique_ptr<Logger> Logger::instance_;
std::mutex Logger::instance_mutex_;

Logger::Logger(const Config& config) : config_(config) {
    if (config_.enable_file) {
        file_stream_ = std::make_unique<std::ofstream>(config_.log_file_path, std::ios::app);
        if (!file_stream_->is_open()) {
            std::cerr << "Warning: Could not open log file: " << config_.log_file_path << std::endl;
            config_.enable_file = false;
        }
    }
}

Logger::~Logger() {
    flush();
}

Logger& Logger::get_instance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (!instance_) {
        instance_ = std::make_unique<Logger>();
    }
    return *instance_;
}

void Logger::initialize(const Config& config) {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    instance_ = std::make_unique<Logger>(config);
}

void Logger::log(LogLevel level, const std::string& message, 
                const char* file, int line, const char* function) {
    if (!is_enabled(level)) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    std::string formatted_message = format_message(level, message, file, line, function);
    write_to_outputs(formatted_message);
    
    // Check if log file needs rotation
    if (config_.enable_file && current_file_size_ > config_.max_file_size_mb * 1024 * 1024) {
        rotate_log_file();
    }
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (file_stream_ && file_stream_->is_open()) {
        file_stream_->flush();
    }
}

std::string Logger::format_message(LogLevel level, const std::string& message,
                                 const char* file, int line, const char* function) {
    std::ostringstream oss;
    
    // Timestamp
    if (config_.include_timestamp) {
        oss << "[" << get_timestamp() << "] ";
    }
    
    // Log level
    oss << "[" << level_to_string(level) << "] ";
    
    // Thread ID
    if (config_.include_thread_id) {
        oss << "[" << std::this_thread::get_id() << "] ";
    }
    
    // File, line, function (if provided)
    if (file && line > 0) {
        std::string filename = std::filesystem::path(file).filename().string();
        oss << "[" << filename << ":" << line;
        if (function) {
            oss << " " << function << "()";
        }
        oss << "] ";
    }
    
    // Message
    oss << message;
    
    return oss.str();
}

std::string Logger::level_to_string(LogLevel level) const {
    switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

std::string Logger::get_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    
    return oss.str();
}

void Logger::rotate_log_file() {
    if (!file_stream_ || !file_stream_->is_open()) {
        return;
    }
    
    file_stream_->close();
    
    // Rotate backup files
    for (int i = config_.max_backup_files - 1; i > 0; --i) {
        std::string old_backup = config_.log_file_path + "." + std::to_string(i);
        std::string new_backup = config_.log_file_path + "." + std::to_string(i + 1);
        
        if (std::filesystem::exists(old_backup)) {
            if (i == config_.max_backup_files - 1) {
                std::filesystem::remove(new_backup);
            }
            std::filesystem::rename(old_backup, new_backup);
        }
    }
    
    // Move current log to .1
    std::string first_backup = config_.log_file_path + ".1";
    if (std::filesystem::exists(config_.log_file_path)) {
        std::filesystem::rename(config_.log_file_path, first_backup);
    }
    
    // Reopen new log file
    file_stream_ = std::make_unique<std::ofstream>(config_.log_file_path, std::ios::app);
    current_file_size_ = 0;
}

void Logger::write_to_outputs(const std::string& formatted_message) {
    // Console output
    if (config_.enable_console) {
        std::cout << formatted_message << std::endl;
    }
    
    // File output
    if (config_.enable_file && file_stream_ && file_stream_->is_open()) {
        *file_stream_ << formatted_message << std::endl;
        current_file_size_ += formatted_message.length() + 1;
    }
}

// PerformanceLogger implementation
void PerformanceLogger::log_frame_metrics(const Metrics& metrics) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    metrics_history_.push_back(metrics);
    
    // Keep only the last max_history_size_ entries
    if (metrics_history_.size() > max_history_size_) {
        metrics_history_.erase(metrics_history_.begin());
    }
}

PerformanceLogger::Metrics PerformanceLogger::get_average_metrics(size_t last_n_frames) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    if (metrics_history_.empty()) {
        return Metrics{};
    }
    
    size_t start_idx = metrics_history_.size() > last_n_frames ? 
                      metrics_history_.size() - last_n_frames : 0;
    
    Metrics avg;
    size_t count = 0;
    
    for (size_t i = start_idx; i < metrics_history_.size(); ++i) {
        const auto& m = metrics_history_[i];
        avg.inference_time_ms += m.inference_time_ms;
        avg.preprocessing_time_ms += m.preprocessing_time_ms;
        avg.total_frame_time_ms += m.total_frame_time_ms;
        avg.power_consumption_mw += m.power_consumption_mw;
        avg.confidence += m.confidence;
        avg.memory_usage_kb += m.memory_usage_kb;
        count++;
    }
    
    if (count > 0) {
        avg.inference_time_ms /= count;
        avg.preprocessing_time_ms /= count;
        avg.total_frame_time_ms /= count;
        avg.power_consumption_mw /= count;
        avg.confidence /= count;
        avg.memory_usage_kb /= count;
    }
    
    return avg;
}

void PerformanceLogger::export_csv(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        return;
    }
    
    // CSV header
    file << "timestamp,inference_time_ms,preprocessing_time_ms,total_frame_time_ms,"
         << "power_consumption_mw,confidence,memory_usage_kb\n";
    
    // CSV data
    for (const auto& metrics : metrics_history_) {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        file << time_t << ","
             << metrics.inference_time_ms << ","
             << metrics.preprocessing_time_ms << ","
             << metrics.total_frame_time_ms << ","
             << metrics.power_consumption_mw << ","
             << metrics.confidence << ","
             << metrics.memory_usage_kb << "\n";
    }
}

double PerformanceLogger::get_fps() const {
    auto avg_metrics = get_average_metrics(10); // Last 10 frames
    if (avg_metrics.total_frame_time_ms > 0) {
        return 1000.0 / avg_metrics.total_frame_time_ms;
    }
    return 0.0;
}

float PerformanceLogger::get_average_confidence() const {
    return get_average_metrics(100).confidence;
}

float PerformanceLogger::get_average_power() const {
    return get_average_metrics(100).power_consumption_mw;
}

void PerformanceLogger::clear_history() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_history_.clear();
}

// HealthLogger implementation
void HealthLogger::update_component_health(const std::string& component, 
                                         ComponentStatus status,
                                         const std::string& message) {
    std::lock_guard<std::mutex> lock(health_mutex_);
    
    ComponentHealth health;
    health.component_name = component;
    health.status = status;
    health.message = message;
    health.last_update = std::chrono::steady_clock::now();
    
    component_status_[component] = health;
    
    // Log health status changes
    std::string status_str;
    switch (status) {
        case ComponentStatus::HEALTHY: status_str = "HEALTHY"; break;
        case ComponentStatus::WARNING: status_str = "WARNING"; break;
        case ComponentStatus::CRITICAL: status_str = "CRITICAL"; break;
        case ComponentStatus::OFFLINE: status_str = "OFFLINE"; break;
    }
    
    Logger::get_instance().log_formatted(LogLevel::INFO, 
        "Component %s status: %s - %s", 
        component.c_str(), status_str.c_str(), message.c_str());
}

HealthLogger::ComponentHealth HealthLogger::get_component_health(const std::string& component) const {
    std::lock_guard<std::mutex> lock(health_mutex_);
    
    auto it = component_status_.find(component);
    if (it != component_status_.end()) {
        return it->second;
    }
    
    // Return offline status for unknown components
    ComponentHealth unknown;
    unknown.component_name = component;
    unknown.status = ComponentStatus::OFFLINE;
    unknown.message = "Component not found";
    return unknown;
}

std::map<std::string, HealthLogger::ComponentHealth> HealthLogger::get_all_health() const {
    std::lock_guard<std::mutex> lock(health_mutex_);
    return component_status_;
}

bool HealthLogger::is_system_healthy() const {
    std::lock_guard<std::mutex> lock(health_mutex_);
    
    for (const auto& [component, health] : component_status_) {
        if (health.status == ComponentStatus::CRITICAL || 
            health.status == ComponentStatus::OFFLINE) {
            return false;
        }
    }
    
    return true;
}

void HealthLogger::start_health_monitoring(std::chrono::seconds interval) {
    if (health_monitoring_active_) {
        return;
    }
    
    health_monitoring_active_ = true;
    health_monitor_thread_ = std::thread(&HealthLogger::health_check_loop, this, interval);
}

void HealthLogger::stop_health_monitoring() {
    health_monitoring_active_ = false;
    if (health_monitor_thread_.joinable()) {
        health_monitor_thread_.join();
    }
}

void HealthLogger::health_check_loop(std::chrono::seconds interval) {
    while (health_monitoring_active_) {
        // Check for stale components (not updated recently)
        auto now = std::chrono::steady_clock::now();
        auto stale_threshold = std::chrono::seconds(30);
        
        {
            std::lock_guard<std::mutex> lock(health_mutex_);
            for (auto& [component, health] : component_status_) {
                if (now - health.last_update > stale_threshold && 
                    health.status != ComponentStatus::OFFLINE) {
                    health.status = ComponentStatus::WARNING;
                    health.message = "Component not reporting";
                    
                    Logger::get_instance().log_formatted(LogLevel::WARN,
                        "Component %s appears stale (last update: %ld seconds ago)",
                        component.c_str(), 
                        std::chrono::duration_cast<std::chrono::seconds>(now - health.last_update).count());
                }
            }
        }
        
        // Sleep for the specified interval
        std::this_thread::sleep_for(interval);
    }
}

} // namespace LiquidVision