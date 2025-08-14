#include "../../include/liquid_vision/core/ode_solver.h"
#include <algorithm>
#include <cmath>

namespace LiquidVision {

// Constructor implementations are now inline in header

void ODESolver::set_derivative_function(const DerivativeFunction& func) {
    derivative_func_ = func;
}

std::vector<float> ODESolver::solve_step(
    const std::vector<float>& current_state,
    const std::vector<float>& inputs,
    float timestep
) {
    if (!derivative_func_) {
        return current_state;
    }
    
    if (timestep <= 0.0f) {
        timestep = config_.timestep;
    }
    
    switch (config_.method) {
        case Method::EULER:
            return euler_step(current_state, inputs, timestep);
        
        case Method::RUNGE_KUTTA_4:
            return runge_kutta_4_step(current_state, inputs, timestep);
        
        case Method::ADAPTIVE_RK4:
            return adaptive_rk4_step(current_state, inputs, timestep);
        
        default:
            return current_state;
    }
}

std::vector<float> ODESolver::euler_step(
    const std::vector<float>& state,
    const std::vector<float>& inputs,
    float timestep
) {
    auto derivatives = derivative_func_(state, inputs);
    std::vector<float> next_state(state.size());
    
    for (size_t i = 0; i < state.size(); ++i) {
        next_state[i] = state[i] + timestep * derivatives[i];
    }
    
    return next_state;
}

std::vector<float> ODESolver::runge_kutta_4_step(
    const std::vector<float>& state,
    const std::vector<float>& inputs,
    float timestep
) {
    auto k1 = derivative_func_(state, inputs);
    
    std::vector<float> temp_state(state.size());
    for (size_t i = 0; i < state.size(); ++i) {
        temp_state[i] = state[i] + 0.5f * timestep * k1[i];
    }
    auto k2 = derivative_func_(temp_state, inputs);
    
    for (size_t i = 0; i < state.size(); ++i) {
        temp_state[i] = state[i] + 0.5f * timestep * k2[i];
    }
    auto k3 = derivative_func_(temp_state, inputs);
    
    for (size_t i = 0; i < state.size(); ++i) {
        temp_state[i] = state[i] + timestep * k3[i];
    }
    auto k4 = derivative_func_(temp_state, inputs);
    
    std::vector<float> next_state(state.size());
    for (size_t i = 0; i < state.size(); ++i) {
        next_state[i] = state[i] + timestep * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6.0f;
    }
    
    return next_state;
}

std::vector<float> ODESolver::adaptive_rk4_step(
    const std::vector<float>& state,
    const std::vector<float>& inputs,
    float& timestep
) {
    // Try full step
    auto full_step = runge_kutta_4_step(state, inputs, timestep);
    
    // Try two half steps
    auto half_step1 = runge_kutta_4_step(state, inputs, timestep * 0.5f);
    auto half_step2 = runge_kutta_4_step(half_step1, inputs, timestep * 0.5f);
    
    // Estimate error
    float error = estimate_error(full_step, half_step2);
    
    // Adjust timestep
    if (error > config_.tolerance && timestep > config_.min_timestep) {
        timestep *= 0.8f;
        timestep = std::max(timestep, config_.min_timestep);
        return adaptive_rk4_step(state, inputs, timestep);
    } else if (error < config_.tolerance * 0.1f && timestep < config_.max_timestep) {
        timestep *= 1.2f;
        timestep = std::min(timestep, config_.max_timestep);
    }
    
    return half_step2;
}

float ODESolver::estimate_error(
    const std::vector<float>& rk4_result,
    const std::vector<float>& rk2_result
) {
    float max_error = 0.0f;
    for (size_t i = 0; i < rk4_result.size(); ++i) {
        float local_error = std::abs(rk4_result[i] - rk2_result[i]);
        max_error = std::max(max_error, local_error);
    }
    return max_error;
}

float ODESolver::get_adaptive_timestep(
    const std::vector<float>& state,
    const std::vector<float>& inputs
) {
    if (!config_.adaptive) {
        return config_.timestep;
    }
    
    auto derivatives = derivative_func_(state, inputs);
    
    float max_derivative = 0.0f;
    for (float d : derivatives) {
        max_derivative = std::max(max_derivative, std::abs(d));
    }
    
    if (max_derivative < 1e-8f) {
        return config_.max_timestep;
    }
    
    float adaptive_timestep = config_.tolerance / max_derivative;
    return std::clamp(adaptive_timestep, config_.min_timestep, config_.max_timestep);
}

} // namespace LiquidVision