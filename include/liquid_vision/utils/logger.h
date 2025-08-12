#pragma once

#include <string>
#include <memory>
#include <mutex>
#include <fstream>
#include <chrono>
#include <sstream>

namespace LiquidVision {

/**
 * Logging levels for different message types
 */
enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5
};

/**
 * Thread-safe logger with configurable outputs
 */
class Logger {
public:
    struct Config {
        LogLevel min_level = LogLevel::INFO;
        bool enable_console = true;
        bool enable_file = false;
        std::string log_file_path = "liquid_vision.log";
        bool include_timestamp = true;
        bool include_thread_id = true;
        size_t max_file_size_mb = 50;
        int max_backup_files = 5;
    };

private:
    Config config_;
    std::mutex log_mutex_;
    std::unique_ptr<std::ofstream> file_stream_;
    size_t current_file_size_ = 0;
    
    static std::unique_ptr<Logger> instance_;
    static std::mutex instance_mutex_;

public:
    explicit Logger(const Config& config = Config{});
    ~Logger();
    
    // Singleton access
    static Logger& get_instance();
    static void initialize(const Config& config = Config{});
    
    // Core logging methods
    void log(LogLevel level, const std::string& message, 
             const char* file = nullptr, int line = 0, const char* function = nullptr);
    
    template<typename... Args>
    void log_formatted(LogLevel level, const char* format, Args... args) {
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), format, args...);
        log(level, std::string(buffer));
    }
    
    // Convenience methods
    void trace(const std::string& message) { log(LogLevel::TRACE, message); }
    void debug(const std::string& message) { log(LogLevel::DEBUG, message); }
    void info(const std::string& message) { log(LogLevel::INFO, message); }
    void warn(const std::string& message) { log(LogLevel::WARN, message); }
    void error(const std::string& message) { log(LogLevel::ERROR, message); }
    void fatal(const std::string& message) { log(LogLevel::FATAL, message); }
    
    // Configuration
    void set_level(LogLevel level) { config_.min_level = level; }
    LogLevel get_level() const { return config_.min_level; }
    
    // Utility methods
    bool is_enabled(LogLevel level) const { return level >= config_.min_level; }
    void flush();
    
private:
    std::string format_message(LogLevel level, const std::string& message,
                              const char* file, int line, const char* function);
    std::string level_to_string(LogLevel level) const;
    std::string get_timestamp() const;
    void rotate_log_file();
    void write_to_outputs(const std::string& formatted_message);
};

// Performance monitoring logger
class PerformanceLogger {
public:
    struct Metrics {
        double inference_time_ms = 0.0;
        double preprocessing_time_ms = 0.0;
        double total_frame_time_ms = 0.0;
        float power_consumption_mw = 0.0f;
        float confidence = 0.0f;
        size_t memory_usage_kb = 0;
    };

private:
    std::mutex metrics_mutex_;
    std::vector<Metrics> metrics_history_;
    size_t max_history_size_ = 1000;

public:
    void log_frame_metrics(const Metrics& metrics);
    Metrics get_average_metrics(size_t last_n_frames = 100) const;
    void export_csv(const std::string& filename) const;
    void clear_history();
    
    // Real-time metrics
    double get_fps() const;
    float get_average_confidence() const;
    float get_average_power() const;
};

// System health logger
class HealthLogger {
public:
    enum class ComponentStatus {
        HEALTHY,
        WARNING,
        CRITICAL,
        OFFLINE
    };
    
    struct ComponentHealth {
        std::string component_name;
        ComponentStatus status;
        std::string message;
        std::chrono::steady_clock::time_point last_update;
    };

private:
    std::mutex health_mutex_;
    std::map<std::string, ComponentHealth> component_status_;
    std::chrono::steady_clock::time_point last_health_check_;

public:
    void update_component_health(const std::string& component, 
                               ComponentStatus status,
                               const std::string& message = "");
    
    ComponentHealth get_component_health(const std::string& component) const;
    std::map<std::string, ComponentHealth> get_all_health() const;
    bool is_system_healthy() const;
    
    // Periodic health checks
    void start_health_monitoring(std::chrono::seconds interval = std::chrono::seconds(5));
    void stop_health_monitoring();
    
private:
    void health_check_loop(std::chrono::seconds interval);
    std::thread health_monitor_thread_;
    std::atomic<bool> health_monitoring_active_{false};
};

} // namespace LiquidVision

// Logging macros for convenience
#define LOG_TRACE(msg) LiquidVision::Logger::get_instance().trace(msg)
#define LOG_DEBUG(msg) LiquidVision::Logger::get_instance().debug(msg)
#define LOG_INFO(msg) LiquidVision::Logger::get_instance().info(msg)
#define LOG_WARN(msg) LiquidVision::Logger::get_instance().warn(msg)
#define LOG_ERROR(msg) LiquidVision::Logger::get_instance().error(msg)
#define LOG_FATAL(msg) LiquidVision::Logger::get_instance().fatal(msg)

// Detailed logging with file/line/function info
#define LOG_TRACE_F(msg) LiquidVision::Logger::get_instance().log(LiquidVision::LogLevel::TRACE, msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_DEBUG_F(msg) LiquidVision::Logger::get_instance().log(LiquidVision::LogLevel::DEBUG, msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_INFO_F(msg) LiquidVision::Logger::get_instance().log(LiquidVision::LogLevel::INFO, msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_WARN_F(msg) LiquidVision::Logger::get_instance().log(LiquidVision::LogLevel::WARN, msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_ERROR_F(msg) LiquidVision::Logger::get_instance().log(LiquidVision::LogLevel::ERROR, msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_FATAL_F(msg) LiquidVision::Logger::get_instance().log(LiquidVision::LogLevel::FATAL, msg, __FILE__, __LINE__, __FUNCTION__)

// Formatted logging
#define LOG_INFO_FMT(...) LiquidVision::Logger::get_instance().log_formatted(LiquidVision::LogLevel::INFO, __VA_ARGS__)
#define LOG_WARN_FMT(...) LiquidVision::Logger::get_instance().log_formatted(LiquidVision::LogLevel::WARN, __VA_ARGS__)
#define LOG_ERROR_FMT(...) LiquidVision::Logger::get_instance().log_formatted(LiquidVision::LogLevel::ERROR, __VA_ARGS__)