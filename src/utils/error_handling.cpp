#include "../../include/liquid_vision/utils/error_handling.h"
#include "../../include/liquid_vision/utils/logger.h"
#include <thread>
#include <sstream>

namespace LiquidVision {

// CircuitBreaker implementation
CircuitBreaker::CircuitBreaker(const Config& config) 
    : config_(config), state_(State::CLOSED), failure_count_(0), success_count_(0) {}

void CircuitBreaker::on_success() {
    if (state_ == State::HALF_OPEN) {
        success_count_++;
        if (success_count_ >= config_.success_threshold) {
            state_ = State::CLOSED;
            failure_count_ = 0;
            LOG_INFO_FMT("Circuit breaker transitioning to CLOSED state after %d successes", 
                        success_count_);
        }
    } else if (state_ == State::CLOSED) {
        // Reset failure count on success in closed state
        failure_count_ = 0;
    }
}

void CircuitBreaker::on_failure() {
    failure_count_++;
    last_failure_time_ = std::chrono::steady_clock::now();
    
    if (state_ == State::CLOSED && failure_count_ >= config_.failure_threshold) {
        state_ = State::OPEN;
        LOG_ERROR_FMT("Circuit breaker transitioning to OPEN state after %d failures", 
                     failure_count_);
    } else if (state_ == State::HALF_OPEN) {
        state_ = State::OPEN;
        LOG_WARN("Circuit breaker returning to OPEN state due to failure in HALF_OPEN");
    }
}

// ErrorRecoveryManager implementation
void ErrorRecoveryManager::add_recovery_strategy(const RecoveryStrategy& strategy) {
    strategies_.push_back(strategy);
    LOG_INFO_FMT("Added recovery strategy: %s", strategy.name.c_str());
}

bool ErrorRecoveryManager::attempt_recovery(const std::string& error_context) {
    LOG_WARN_FMT("Attempting recovery for error context: %s", error_context.c_str());
    
    for (const auto& strategy : strategies_) {
        LOG_INFO_FMT("Trying recovery strategy: %s", strategy.name.c_str());
        
        for (int attempt = 0; attempt < strategy.max_attempts; ++attempt) {
            try {
                if (strategy.recovery_func()) {
                    LOG_INFO_FMT("Recovery strategy '%s' succeeded on attempt %d", 
                                strategy.name.c_str(), attempt + 1);
                    return true;
                }
            } catch (const std::exception& e) {
                LOG_WARN_FMT("Recovery strategy '%s' failed on attempt %d: %s", 
                            strategy.name.c_str(), attempt + 1, e.what());
            }
            
            if (attempt < strategy.max_attempts - 1) {
                std::this_thread::sleep_for(strategy.delay);
            }
        }
        
        LOG_WARN_FMT("Recovery strategy '%s' exhausted all %d attempts", 
                    strategy.name.c_str(), strategy.max_attempts);
    }
    
    LOG_ERROR_FMT("All recovery strategies failed for error context: %s", 
                 error_context.c_str());
    return false;
}

ErrorRecoveryManager::RecoveryStrategy ErrorRecoveryManager::restart_component(
    const std::string& component_name, 
    std::function<bool()> restart_func) {
    
    RecoveryStrategy strategy;
    strategy.name = "Restart " + component_name;
    strategy.recovery_func = restart_func;
    strategy.max_attempts = 2;
    strategy.delay = std::chrono::milliseconds(2000);
    
    return strategy;
}

ErrorRecoveryManager::RecoveryStrategy ErrorRecoveryManager::reset_to_defaults(
    std::function<bool()> reset_func) {
    
    RecoveryStrategy strategy;
    strategy.name = "Reset to Defaults";
    strategy.recovery_func = reset_func;
    strategy.max_attempts = 1;
    strategy.delay = std::chrono::milliseconds(500);
    
    return strategy;
}

ErrorRecoveryManager::RecoveryStrategy ErrorRecoveryManager::fallback_mode(
    std::function<bool()> fallback_func) {
    
    RecoveryStrategy strategy;
    strategy.name = "Fallback Mode";
    strategy.recovery_func = fallback_func;
    strategy.max_attempts = 1;
    strategy.delay = std::chrono::milliseconds(100);
    
    return strategy;
}

// DegradationManager implementation
void DegradationManager::set_degradation_level(DegradationLevel level) {
    if (level == current_level_) {
        return;
    }
    
    std::string level_name;
    switch (level) {
        case DegradationLevel::NORMAL: level_name = "NORMAL"; break;
        case DegradationLevel::REDUCED_QUALITY: level_name = "REDUCED_QUALITY"; break;
        case DegradationLevel::MINIMAL_FEATURES: level_name = "MINIMAL_FEATURES"; break;
        case DegradationLevel::EMERGENCY_MODE: level_name = "EMERGENCY_MODE"; break;
        case DegradationLevel::SHUTDOWN: level_name = "SHUTDOWN"; break;
    }
    
    LOG_WARN_FMT("System degradation level changing from %d to %s", 
                static_cast<int>(current_level_), level_name.c_str());
    
    current_level_ = level;
    
    // Execute level handler if registered
    auto it = level_handlers_.find(level);
    if (it != level_handlers_.end()) {
        try {
            it->second();
        } catch (const std::exception& e) {
            LOG_ERROR_FMT("Error executing degradation level handler: %s", e.what());
        }
    }
}

void DegradationManager::register_level_handler(DegradationLevel level, 
                                               std::function<void()> handler) {
    level_handlers_[level] = handler;
    LOG_DEBUG_FMT("Registered handler for degradation level %d", static_cast<int>(level));
}

// ErrorContext implementation
std::string ErrorContext::to_string() const {
    std::ostringstream oss;
    oss << "[" << component << "::" << operation << "] ";
    oss << "at " << file << ":" << line;
    if (!additional_info.empty()) {
        oss << " (" << additional_info << ")";
    }
    
    auto time_since_epoch = timestamp.time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_epoch);
    oss << " [" << ms.count() << "ms]";
    
    return oss.str();
}

} // namespace LiquidVision