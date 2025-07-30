#pragma once

#include "core/liquid_network.h"
#include "vision/image_processor.h"
#include "control/flight_controller.h"
#include "utils/fixed_point.h"
#include <memory>
#include <string>

namespace LiquidVision {

/**
 * @brief Main controller integrating Liquid Neural Networks with vision processing
 * 
 * This class provides the high-level interface for drone vision and control,
 * combining image processing, neural network inference, and flight control.
 */
class LNNController {
public:
    /**
     * @brief Configuration parameters for the LNN Controller
     */
    struct Config {
        std::string model_path;           ///< Path to the LNN model file
        std::pair<int, int> input_resolution{160, 120}; ///< Input image resolution
        ODESolver::Type ode_solver = ODESolver::Type::FIXED_POINT_RK4; ///< ODE solver type
        bool timestep_adaptive = true;    ///< Enable adaptive timestep
        float max_inference_time_ms = 20.0f; ///< Maximum allowed inference time
        int memory_limit_kb = 256;        ///< Memory limit for embedded deployment
    };

    /**
     * @brief Control output from the neural network
     */
    struct ControlOutput {
        float forward_velocity = 0.0f;    ///< Forward velocity command (m/s)
        float yaw_rate = 0.0f;           ///< Yaw rate command (rad/s)
        float target_altitude = 0.0f;    ///< Target altitude (m)
        float confidence = 0.0f;         ///< Confidence in the output (0-1)
        uint32_t inference_time_us = 0;  ///< Inference time in microseconds
    };

public:
    /**
     * @brief Construct a new LNN Controller
     * @param config Configuration parameters
     */
    explicit LNNController(const Config& config);

    /**
     * @brief Destructor
     */
    ~LNNController();

    /**
     * @brief Initialize the controller and load the model
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * @brief Process a camera frame and generate control commands
     * @param image_data Raw image data
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param channels Number of color channels
     * @return Control output with velocity commands and metadata
     */
    ControlOutput process_frame(const uint8_t* image_data, 
                               int width, int height, int channels);

    /**
     * @brief Preprocess image for neural network input
     * @param image_data Raw image data  
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @return Preprocessed image ready for inference
     */
    ProcessedFrame preprocess(const uint8_t* image_data, 
                             int width, int height, int channels);

    /**
     * @brief Run neural network inference on preprocessed image
     * @param frame Preprocessed image frame
     * @return Control output from the network
     */
    ControlOutput infer(const ProcessedFrame& frame);

    /**
     * @brief Get current performance statistics
     * @return Performance metrics structure
     */
    PerformanceStats get_performance_stats() const;

    /**
     * @brief Check if controller is ready for inference
     * @return true if ready, false otherwise
     */
    bool is_ready() const { return initialized_ && model_loaded_; }

    /**
     * @brief Get configuration parameters
     * @return Current configuration
     */
    const Config& get_config() const { return config_; }

private:
    Config config_;                              ///< Configuration parameters
    bool initialized_ = false;                   ///< Initialization status
    bool model_loaded_ = false;                  ///< Model loading status
    
    std::unique_ptr<LiquidNetwork> network_;     ///< Neural network instance
    std::unique_ptr<ImageProcessor> processor_;   ///< Image preprocessing
    std::unique_ptr<FlightController> controller_; ///< Flight control interface
    
    PerformanceStats stats_;                     ///< Performance statistics
    
    /**
     * @brief Load the neural network model from file
     * @return true if successful, false otherwise
     */
    bool load_model();
    
    /**
     * @brief Validate configuration parameters
     * @return true if valid, false otherwise
     */
    bool validate_config() const;
};

} // namespace LiquidVision