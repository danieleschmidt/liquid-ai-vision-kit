# ADR-0002: Fixed-Point Arithmetic for Liquid Neural Networks

## Status

Accepted

## Context

Liquid Neural Networks require numerical computations for ODE solving and neural state updates. Embedded platforms (STM32H7, Cortex-M7) have limited floating-point performance and power constraints. We need to choose an arithmetic representation that balances:

- Computational efficiency on embedded hardware
- Numerical precision for network convergence
- Power consumption optimization
- Memory usage constraints
- Real-time performance requirements

## Decision

We will implement fixed-point arithmetic using Q16.16 format (16-bit integer, 16-bit fractional) for all Liquid Neural Network computations.

### Implementation Details

- **Primary Format**: Q16.16 (32-bit total)
- **Range**: -32768.0 to +32767.99998 (approximately)
- **Precision**: ~1.5e-5 (sufficient for neural network weights)
- **Operations**: Custom saturated arithmetic with overflow protection
- **Fallback**: Floating-point available for simulation/debugging

### Code Example

```cpp
template<int INTEGER_BITS, int FRACTIONAL_BITS>
class FixedPoint {
    static constexpr int SCALE = 1 << FRACTIONAL_BITS;
    int32_t value;
    
public:
    FixedPoint(float f) : value(static_cast<int32_t>(f * SCALE)) {}
    
    FixedPoint operator+(const FixedPoint& other) const {
        return FixedPoint::from_raw(saturated_add(value, other.value));
    }
    
    FixedPoint operator*(const FixedPoint& other) const {
        int64_t temp = static_cast<int64_t>(value) * other.value;
        return FixedPoint::from_raw(static_cast<int32_t>(temp >> FRACTIONAL_BITS));
    }
};

using Q16_16 = FixedPoint<16, 16>;
```

## Consequences

### Positive

- **Performance**: 5-10x faster than software floating-point on Cortex-M7
- **Power Efficiency**: ~30% reduction in computation power
- **Deterministic**: Reproducible results across platforms
- **Memory Efficient**: Smaller memory footprint than double precision
- **Hardware Friendly**: Leverages integer ALU performance

### Negative

- **Precision Limitations**: May affect convergence for some models
- **Range Constraints**: Requires careful scaling of network parameters
- **Development Complexity**: Additional debugging and validation needed
- **Overflow Risk**: Requires saturated arithmetic implementation
- **Conversion Overhead**: Interface complexity with floating-point APIs

### Mitigation Strategies

1. **Model Validation**: Extensive testing against floating-point reference
2. **Overflow Protection**: Saturated arithmetic with runtime checks
3. **Precision Monitoring**: Runtime precision loss detection
4. **Fallback Mode**: Ability to switch to floating-point for debugging
5. **Scaling Tools**: Automated parameter scaling during model conversion

## Alternatives Considered

### 1. IEEE 754 Single Precision (float)
- **Pros**: Standard, well-supported, good precision
- **Cons**: Slow on Cortex-M without FPU, high power consumption
- **Verdict**: Rejected due to performance constraints

### 2. IEEE 754 Half Precision (fp16)
- **Pros**: Smaller memory, hardware support on some platforms
- **Cons**: Limited range, not widely supported on target MCUs
- **Verdict**: Considered for future when hardware support improves

### 3. Custom 8-bit Fixed Point (Q4.4)
- **Pros**: Very fast, minimal memory
- **Cons**: Insufficient precision for neural networks
- **Verdict**: Rejected due to accuracy requirements

### 4. Dynamic Fixed Point with Multiple Scales
- **Pros**: Adaptive precision, better range utilization
- **Cons**: Complex implementation, runtime overhead
- **Verdict**: Rejected due to complexity

### 5. Integer-only with Manual Scaling
- **Pros**: Maximum performance
- **Cons**: Very complex model conversion, error-prone
- **Verdict**: Rejected due to development complexity

## Implementation Plan

### Phase 1: Core Infrastructure
- Fixed-point arithmetic library
- Basic operations (add, subtract, multiply, divide)
- Saturated arithmetic with overflow detection
- Unit tests for numerical accuracy

### Phase 2: Network Integration
- Liquid neuron state representation
- ODE solver integration
- Weight and bias conversion tools
- Numerical validation framework

### Phase 3: Model Conversion
- ONNX to fixed-point conversion pipeline
- Automatic scaling factor determination
- Precision loss analysis tools
- Reference implementation validation

### Phase 4: Optimization
- Platform-specific optimizations
- SIMD instruction utilization
- Memory layout optimization
- Performance profiling and tuning

## Validation Criteria

### Numerical Accuracy
- Maximum 1% accuracy degradation vs floating-point reference
- Convergence stability over 1000+ inference cycles
- Overflow rate <0.1% under normal operating conditions

### Performance Targets
- >5x speedup vs software floating-point on Cortex-M7
- <15ms inference time for 64-neuron network
- <400mW total power consumption

### Robustness Requirements
- Stable operation across input range [-1, +1]
- Graceful degradation under overflow conditions
- Deterministic behavior across temperature range

## References

- [Fixed-Point Arithmetic for Embedded Systems](https://en.wikipedia.org/wiki/Fixed-point_arithmetic)
- [ARM Cortex-M7 Performance Analysis](https://developer.arm.com/documentation)
- [Neural Network Quantization Survey](https://arxiv.org/abs/2103.13630)
- [Q-format Number Representation](https://en.wikipedia.org/wiki/Q_(number_format))

## Date

January 2025