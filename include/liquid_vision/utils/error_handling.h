#pragma once

#include <stdexcept>
#include <string>
#include <memory>
#include <functional>
#include <vector>
#include <chrono>

namespace LiquidVision {

/**
 * Custom exception hierarchy for Liquid Vision system
 */
class LiquidVisionException : public std::runtime_error {
public:
    explicit LiquidVisionException(const std::string& message) 
        : std::runtime_error(message) {}
};

class InitializationException : public LiquidVisionException {
public:
    explicit InitializationException(const std::string& component)
        : LiquidVisionException("Failed to initialize component: " + component) {}
};

class ProcessingException : public LiquidVisionException {
public:
    explicit ProcessingException(const std::string& message)
        : LiquidVisionException("Processing error: " + message) {}
};

class NetworkException : public LiquidVisionException {
public:
    explicit NetworkException(const std::string& message)
        : LiquidVisionException("Neural network error: " + message) {}
};

class SafetyException : public LiquidVisionException {
public:
    explicit SafetyException(const std::string& message)
        : LiquidVisionException("Safety violation: " + message) {}
};

class HardwareException : public LiquidVisionException {
public:
    explicit HardwareException(const std::string& device)
        : LiquidVisionException("Hardware error with device: " + device) {}
};

class TimeoutException : public LiquidVisionException {
public:
    explicit TimeoutException(const std::string& operation, int timeout_ms)
        : LiquidVisionException("Timeout after " + std::to_string(timeout_ms) + 
                               "ms in operation: " + operation) {}
};

/**
 * Result type for operations that can fail
 */
template<typename T>
class Result {
private:
    bool success_;
    T value_;
    std::string error_message_;

public:
    // Success constructor
    explicit Result(T value) : success_(true), value_(std::move(value)) {}
    
    // Error constructor
    explicit Result(const std::string& error) : success_(false), error_message_(error) {}
    
    bool is_success() const { return success_; }
    bool is_error() const { return !success_; }
    
    const T& get_value() const {
        if (!success_) {
            throw std::runtime_error("Attempted to access value of failed Result: " + error_message_);
        }
        return value_;
    }
    
    T get_value_or(const T& default_value) const {
        return success_ ? value_ : default_value;
    }
    
    const std::string& get_error() const { return error_message_; }
    
    // Conversion operators
    explicit operator bool() const { return success_; }
    const T& operator*() const { return get_value(); }
    const T* operator->() const { return &get_value(); }
};

/**
 * Circuit breaker pattern for handling repeated failures
 */
class CircuitBreaker {
public:
    enum class State {
        CLOSED,     // Normal operation
        OPEN,       // Failing, rejecting calls
        HALF_OPEN   // Testing if service recovered
    };
    
    struct Config {
        int failure_threshold = 5;
        std::chrono::milliseconds timeout = std::chrono::milliseconds(5000);
        int success_threshold = 2; // For half-open to closed transition
    };

private:
    Config config_;
    State state_;
    int failure_count_;
    int success_count_;
    std::chrono::steady_clock::time_point last_failure_time_;

public:
    explicit CircuitBreaker(const Config& config = Config{});
    
    template<typename Func>
    auto execute(Func&& func) -> Result<decltype(func())> {
        if (state_ == State::OPEN) {
            if (std::chrono::steady_clock::now() - last_failure_time_ < config_.timeout) {
                return Result<decltype(func())>("Circuit breaker is OPEN");
            } else {
                state_ = State::HALF_OPEN;
                success_count_ = 0;
            }
        }
        
        try {
            auto result = func();
            on_success();
            return Result<decltype(func())>(result);
        } catch (const std::exception& e) {
            on_failure();
            return Result<decltype(func())>(e.what());
        }
    }
    
    State get_state() const { return state_; }
    int get_failure_count() const { return failure_count_; }
    
private:
    void on_success();
    void on_failure();
};

/**
 * Retry mechanism with exponential backoff
 */
class RetryPolicy {
public:
    struct Config {
        int max_attempts = 3;
        std::chrono::milliseconds initial_delay = std::chrono::milliseconds(100);
        double backoff_multiplier = 2.0;
        std::chrono::milliseconds max_delay = std::chrono::milliseconds(5000);
    };

private:
    Config config_;

public:
    explicit RetryPolicy(const Config& config = Config{}) : config_(config) {}
    
    template<typename Func>
    auto execute(Func&& func) -> Result<decltype(func())> {
        std::string last_error;
        
        for (int attempt = 0; attempt < config_.max_attempts; ++attempt) {
            try {
                auto result = func();
                return Result<decltype(func())>(result);
            } catch (const std::exception& e) {
                last_error = e.what();
                
                if (attempt < config_.max_attempts - 1) {
                    // Calculate delay with exponential backoff
                    auto delay = std::chrono::milliseconds(static_cast<int>(
                        config_.initial_delay.count() * std::pow(config_.backoff_multiplier, attempt)
                    ));
                    
                    if (delay > config_.max_delay) {
                        delay = config_.max_delay;
                    }
                    
                    std::this_thread::sleep_for(delay);
                }
            }
        }
        
        return Result<decltype(func())>("Retry exhausted after " + 
                                       std::to_string(config_.max_attempts) + 
                                       " attempts. Last error: " + last_error);
    }
};

/**
 * Error recovery strategies
 */
class ErrorRecoveryManager {
public:
    using RecoveryFunction = std::function<bool()>;
    
    struct RecoveryStrategy {
        std::string name;
        RecoveryFunction recovery_func;
        int max_attempts = 3;
        std::chrono::milliseconds delay = std::chrono::milliseconds(1000);
    };

private:
    std::vector<RecoveryStrategy> strategies_;

public:
    void add_recovery_strategy(const RecoveryStrategy& strategy);
    bool attempt_recovery(const std::string& error_context);
    
    // Common recovery strategies
    static RecoveryStrategy restart_component(const std::string& component_name,
                                            std::function<bool()> restart_func);
    static RecoveryStrategy reset_to_defaults(std::function<bool()> reset_func);
    static RecoveryStrategy fallback_mode(std::function<bool()> fallback_func);
};

/**
 * Graceful degradation manager
 */
class DegradationManager {
public:
    enum class DegradationLevel {
        NORMAL = 0,
        REDUCED_QUALITY = 1,
        MINIMAL_FEATURES = 2,
        EMERGENCY_MODE = 3,
        SHUTDOWN = 4
    };

private:
    DegradationLevel current_level_;
    std::map<DegradationLevel, std::function<void()>> level_handlers_;

public:
    DegradationManager() : current_level_(DegradationLevel::NORMAL) {}
    
    void set_degradation_level(DegradationLevel level);
    DegradationLevel get_degradation_level() const { return current_level_; }
    
    void register_level_handler(DegradationLevel level, std::function<void()> handler);
    
    // Convenience methods
    bool is_normal_operation() const { return current_level_ == DegradationLevel::NORMAL; }
    bool is_degraded() const { return current_level_ > DegradationLevel::NORMAL; }
    bool is_emergency_mode() const { return current_level_ >= DegradationLevel::EMERGENCY_MODE; }
};

/**
 * Timeout manager for operations
 */
class TimeoutManager {
public:
    template<typename Func>
    static auto execute_with_timeout(Func&& func, std::chrono::milliseconds timeout) 
        -> Result<decltype(func())> {
        
        std::promise<Result<decltype(func())>> promise;
        auto future = promise.get_future();
        
        std::thread worker([&promise, func = std::forward<Func>(func)]() mutable {
            try {
                auto result = func();
                promise.set_value(Result<decltype(func())>(result));
            } catch (const std::exception& e) {
                promise.set_value(Result<decltype(func())>(e.what()));
            }
        });
        
        if (future.wait_for(timeout) == std::future_status::timeout) {
            worker.detach(); // Let it finish in background
            return Result<decltype(func())>("Operation timed out after " + 
                                           std::to_string(timeout.count()) + "ms");
        }
        
        worker.join();
        return future.get();
    }
};

/**
 * Resource guard for RAII cleanup
 */
template<typename Resource, typename Deleter>
class ResourceGuard {
private:
    Resource resource_;
    Deleter deleter_;
    bool released_;

public:
    ResourceGuard(Resource resource, Deleter deleter) 
        : resource_(resource), deleter_(deleter), released_(false) {}
    
    ~ResourceGuard() {
        if (!released_) {
            deleter_(resource_);
        }
    }
    
    // Move constructor
    ResourceGuard(ResourceGuard&& other) noexcept 
        : resource_(std::move(other.resource_)), 
          deleter_(std::move(other.deleter_)), 
          released_(other.released_) {
        other.released_ = true;
    }
    
    // No copy
    ResourceGuard(const ResourceGuard&) = delete;
    ResourceGuard& operator=(const ResourceGuard&) = delete;
    ResourceGuard& operator=(ResourceGuard&&) = delete;
    
    Resource& get() { return resource_; }
    const Resource& get() const { return resource_; }
    
    void release() { released_ = true; }
};

// Helper function to create resource guards
template<typename Resource, typename Deleter>
ResourceGuard<Resource, Deleter> make_resource_guard(Resource resource, Deleter deleter) {
    return ResourceGuard<Resource, Deleter>(resource, deleter);
}

/**
 * Error context for debugging
 */
struct ErrorContext {
    std::string component;
    std::string operation;
    std::string file;
    int line;
    std::string additional_info;
    std::chrono::steady_clock::time_point timestamp;
    
    ErrorContext(const std::string& comp, const std::string& op, 
                const std::string& f, int l, const std::string& info = "")
        : component(comp), operation(op), file(f), line(l), 
          additional_info(info), timestamp(std::chrono::steady_clock::now()) {}
    
    std::string to_string() const;
};

// Macro for creating error context
#define ERROR_CONTEXT(component, operation, info) \
    LiquidVision::ErrorContext(component, operation, __FILE__, __LINE__, info)

} // namespace LiquidVision