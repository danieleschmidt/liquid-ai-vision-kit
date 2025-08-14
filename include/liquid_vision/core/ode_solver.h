#pragma once

#include <vector>
#include <functional>

namespace LiquidVision {

/**
 * ODE solver for liquid neural network dynamics
 */
class ODESolver {
public:
    enum class Method {
        EULER,
        RUNGE_KUTTA_4,
        ADAPTIVE_RK4
    };

    using DerivativeFunction = std::function<std::vector<float>(
        const std::vector<float>&,  // current state
        const std::vector<float>&   // inputs
    )>;

    struct Config {
        Method method = Method::RUNGE_KUTTA_4;
        float timestep = 0.01f;
        float min_timestep = 0.001f;
        float max_timestep = 0.1f;
        float tolerance = 1e-6f;
        bool adaptive = false;
    };

private:
    Config config_;
    DerivativeFunction derivative_func_;

public:
    ODESolver() : config_() {}
    explicit ODESolver(const Config& config) : config_(config) {}
    ~ODESolver() = default;

    void set_derivative_function(const DerivativeFunction& func);
    
    std::vector<float> solve_step(
        const std::vector<float>& current_state,
        const std::vector<float>& inputs,
        float timestep = 0.0f
    );

    float get_adaptive_timestep(
        const std::vector<float>& state,
        const std::vector<float>& inputs
    );

private:
    std::vector<float> euler_step(
        const std::vector<float>& state,
        const std::vector<float>& inputs,
        float timestep
    );

    std::vector<float> runge_kutta_4_step(
        const std::vector<float>& state,
        const std::vector<float>& inputs,
        float timestep
    );

    std::vector<float> adaptive_rk4_step(
        const std::vector<float>& state,
        const std::vector<float>& inputs,
        float& timestep
    );

    float estimate_error(
        const std::vector<float>& rk4_result,
        const std::vector<float>& rk2_result
    );
};

} // namespace LiquidVision