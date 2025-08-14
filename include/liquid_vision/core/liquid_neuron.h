#pragma once

#include <vector>
#include <cmath>
#include "../utils/fixed_point.h"

namespace LiquidVision {

/**
 * Single liquid neuron with continuous-time dynamics
 */
class LiquidNeuron {
public:
    struct Parameters {
        float tau = 1.0f;          // Time constant
        float leak = 0.1f;         // Leak rate  
        float threshold = 0.5f;    // Activation threshold
        float adaptation = 0.01f;  // Adaptation rate
    };

private:
    Parameters params_;
    float state_ = 0.0f;
    std::vector<float> weights_;
    float bias_ = 0.0f;
    bool use_fixed_point_ = true;

public:
    LiquidNeuron() : params_(), state_(0.0f), bias_(0.0f), use_fixed_point_(true) {}
    explicit LiquidNeuron(const Parameters& params) : params_(params), state_(0.0f), bias_(0.0f), use_fixed_point_(true) {}
    ~LiquidNeuron() = default;

    void set_weights(const std::vector<float>& weights);
    void set_bias(float bias) { bias_ = bias; }
    
    float update(const std::vector<float>& inputs, float timestep);
    float get_state() const { return state_; }
    float get_output() const;
    
    void reset() { state_ = 0.0f; }
    
    const Parameters& get_parameters() const { return params_; }

private:
    float activation(float x) const;
    float compute_derivative(const std::vector<float>& inputs) const;
};

} // namespace LiquidVision