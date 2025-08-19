#include "../tests/framework/simple_test.h"
#include "../include/liquid_vision/core/ode_solver.h"
#include <vector>
#include <cmath>

using namespace LiquidVision;

void test_euler_method() {
    ODESolver::Config config;
    config.method = ODESolver::Method::EULER;
    config.timestep = 0.01f;
    
    ODESolver solver(config);
    
    // Simple test: dx/dt = -x (exponential decay)
    solver.set_derivative_function([](const std::vector<float>& state, const std::vector<float>&) {
        std::vector<float> derivatives(state.size());
        for (size_t i = 0; i < state.size(); ++i) {
            derivatives[i] = -state[i];
        }
        return derivatives;
    });
    
    std::vector<float> initial_state = {1.0f};
    std::vector<float> inputs;
    
    auto result = solver.solve_step(initial_state, inputs);
    
    // After one timestep with Euler: x_new = x_old + dt * (-x_old) = x_old * (1 - dt)
    float expected = 1.0f * (1.0f - 0.01f);
    ASSERT_FLOAT_NEAR(result[0], expected, 1e-6f);
}

void test_runge_kutta_4() {
    ODESolver::Config config;
    config.method = ODESolver::Method::RUNGE_KUTTA_4;
    config.timestep = 0.1f;
    
    ODESolver solver(config);
    
    // Test: dx/dt = -x
    solver.set_derivative_function([](const std::vector<float>& state, const std::vector<float>&) {
        std::vector<float> derivatives(state.size());
        for (size_t i = 0; i < state.size(); ++i) {
            derivatives[i] = -state[i];
        }
        return derivatives;
    });
    
    std::vector<float> state = {1.0f};
    std::vector<float> inputs;
    
    // Run multiple steps
    for (int i = 0; i < 10; ++i) {
        state = solver.solve_step(state, inputs);
    }
    
    // Analytical solution: x(t) = x0 * exp(-t)
    float analytical = std::exp(-1.0f); // t = 10 * 0.1 = 1.0
    
    // RK4 should be much more accurate than Euler
    ASSERT_FLOAT_NEAR(state[0], analytical, 1e-3f);
}

void test_adaptive_timestep() {
    ODESolver::Config config;
    config.method = ODESolver::Method::ADAPTIVE_RK4;
    config.timestep = 0.1f;
    config.min_timestep = 0.001f;
    config.max_timestep = 0.5f;
    config.tolerance = 1e-4f;
    config.adaptive = true;
    
    ODESolver solver(config);
    
    // Stiff equation: dx/dt = -100*x (fast decay)
    solver.set_derivative_function([](const std::vector<float>& state, const std::vector<float>&) {
        std::vector<float> derivatives(state.size());
        for (size_t i = 0; i < state.size(); ++i) {
            derivatives[i] = -100.0f * state[i];
        }
        return derivatives;
    });
    
    std::vector<float> state = {1.0f};
    std::vector<float> inputs;
    
    // Adaptive solver should handle this without instability
    for (int i = 0; i < 50; ++i) {
        state = solver.solve_step(state, inputs);
        
        // Should remain stable and finite
        ASSERT_TRUE(std::isfinite(state[0]), "State should be finite");
        ASSERT_GE(state[0], 0.0f, "State should decay towards zero");
    }
}

void test_multi_dimensional() {
    ODESolver::Config config;
    config.method = ODESolver::Method::RUNGE_KUTTA_4;
    config.timestep = 0.01f;
    
    ODESolver solver(config);
    
    // Coupled oscillator: dx/dt = y, dy/dt = -x
    solver.set_derivative_function([](const std::vector<float>& state, const std::vector<float>&) {
        std::vector<float> derivatives(2);
        derivatives[0] = state[1];   // dx/dt = y
        derivatives[1] = -state[0];  // dy/dt = -x
        return derivatives;
    });
    
    std::vector<float> state = {1.0f, 0.0f}; // Initial condition: x=1, y=0
    std::vector<float> inputs;
    
    // This should be a circular motion, energy should be conserved
    float initial_energy = 0.5f * (state[0]*state[0] + state[1]*state[1]);
    
    for (int i = 0; i < 100; ++i) {
        state = solver.solve_step(state, inputs);
    }
    
    float final_energy = 0.5f * (state[0]*state[0] + state[1]*state[1]);
    
    // Energy should be approximately conserved (within numerical error)
    ASSERT_FLOAT_NEAR(final_energy, initial_energy, 0.1f);
}

int main() {
    RUN_TEST(test_euler_method);
    RUN_TEST(test_runge_kutta_4);
    RUN_TEST(test_adaptive_timestep);
    RUN_TEST(test_multi_dimensional);
    
    return 0;
}