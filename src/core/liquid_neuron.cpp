#include "../../include/liquid_vision/core/liquid_neuron.h"
#include <cmath>
#include <algorithm>

namespace LiquidVision {

// Constructor implementations are now inline in header

void LiquidNeuron::set_weights(const std::vector<float>& weights) {
    weights_ = weights;
}

float LiquidNeuron::update(const std::vector<float>& inputs, float timestep) {
    float derivative = compute_derivative(inputs);
    
    // Euler integration: dx/dt = f(x, u)
    state_ += timestep * derivative;
    
    // Apply bounds to prevent instability
    state_ = std::max(-10.0f, std::min(10.0f, state_));
    
    return get_output();
}

float LiquidNeuron::get_output() const {
    return activation(state_);
}

float LiquidNeuron::activation(float x) const {
    // Tanh activation with saturation protection
    return std::tanh(x);
}

float LiquidNeuron::compute_derivative(const std::vector<float>& inputs) const {
    // Liquid neuron dynamics: dx/dt = (-leak * x + weighted_input) / tau
    
    float weighted_sum = bias_;
    size_t min_size = std::min(inputs.size(), weights_.size());
    
    for (size_t i = 0; i < min_size; ++i) {
        weighted_sum += weights_[i] * inputs[i];
    }
    
    float derivative = (-params_.leak * state_ + weighted_sum) / params_.tau;
    
    return derivative;
}

} // namespace LiquidVision