#include <iostream>
#include <chrono>
#include <thread>
#include <signal.h>
#include <vector>
#include <cstring>
#include "../include/liquid_vision/lnn_controller.hpp"
#include "core/liquid_network.h"
#include "vision/image_processor.h"
#include "control/flight_controller.h"

using namespace LiquidVision;

// Global flag for graceful shutdown
volatile bool g_running = true;

void signal_handler(int signum) {
    std::cout << "\nShutdown signal received. Stopping..." << std::endl;
    g_running = false;
}

/**
 * Simulate camera input for testing
 */
class CameraSimulator {
private:
    int width_;
    int height_;
    int channels_;
    std::vector<uint8_t> frame_buffer_;
    int frame_count_ = 0;

public:
    CameraSimulator(int width = 640, int height = 480, int channels = 3) 
        : width_(width), height_(height), channels_(channels) {
        frame_buffer_.resize(width * height * channels);
    }
    
    void capture_frame(uint8_t* buffer) {
        // Generate synthetic test pattern
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                int idx = (y * width_ + x) * channels_;
                
                // Create moving gradient pattern
                uint8_t r = (x + frame_count_) % 256;
                uint8_t g = (y + frame_count_ / 2) % 256;
                uint8_t b = ((x + y) + frame_count_ / 3) % 256;
                
                if (channels_ == 3) {
                    buffer[idx] = r;
                    buffer[idx + 1] = g;
                    buffer[idx + 2] = b;
                } else {
                    // Grayscale
                    buffer[idx] = (r + g + b) / 3;
                }
            }
        }
        
        frame_count_++;
    }
    
    int get_width() const { return width_; }
    int get_height() const { return height_; }
    int get_channels() const { return channels_; }
};

/**
 * Main application class
 */
class LiquidVisionApp {
private:
    LNNController controller_;
    FlightController flight_controller_;
    CameraSimulator camera_;
    
    // Statistics
    uint32_t total_frames_ = 0;
    float total_inference_time_ms_ = 0;
    float min_inference_time_ms_ = 1000.0f;
    float max_inference_time_ms_ = 0.0f;

public:
    LiquidVisionApp(const LNNController::Config& config) 
        : controller_(config),
          flight_controller_(FlightController::ControllerType::SIMULATION),
          camera_(640, 480, 3) {
    }
    
    bool initialize() {
        std::cout << "Initializing Liquid Vision System..." << std::endl;
        
        // Initialize controller
        if (!controller_.initialize()) {
            std::cerr << "Failed to initialize LNN controller" << std::endl;
            return false;
        }
        
        // Initialize flight controller
        if (!flight_controller_.initialize()) {
            std::cerr << "Failed to initialize flight controller" << std::endl;
            return false;
        }
        
        if (!flight_controller_.connect()) {
            std::cerr << "Failed to connect to flight controller" << std::endl;
            return false;
        }
        
        std::cout << "System initialized successfully" << std::endl;
        return true;
    }
    
    void run() {
        std::cout << "Starting main processing loop..." << std::endl;
        std::cout << "Press Ctrl+C to stop" << std::endl;
        
        // Allocate frame buffer
        std::vector<uint8_t> frame_buffer(
            camera_.get_width() * camera_.get_height() * camera_.get_channels()
        );
        
        // Main processing loop
        const int target_fps = 30;
        const auto frame_duration = std::chrono::milliseconds(1000 / target_fps);
        
        while (g_running) {
            auto frame_start = std::chrono::high_resolution_clock::now();
            
            // Capture frame from camera
            camera_.capture_frame(frame_buffer.data());
            
            // Process frame through neural network
            auto control_output = controller_.process_frame(
                frame_buffer.data(),
                camera_.get_width(),
                camera_.get_height(),
                camera_.get_channels()
            );
            
            // Send control commands to flight controller
            FlightCommand cmd;
            cmd.velocity_x = control_output.forward_velocity;
            cmd.yaw_rate = control_output.yaw_rate;
            cmd.velocity_z = 0; // Maintain altitude
            
            flight_controller_.send_command(cmd);
            
            // Update statistics
            update_statistics(control_output);
            
            // Display status every second
            if (total_frames_ % target_fps == 0) {
                display_status(control_output);
            }
            
            // Maintain target FPS
            auto frame_end = std::chrono::high_resolution_clock::now();
            auto elapsed = frame_end - frame_start;
            
            if (elapsed < frame_duration) {
                std::this_thread::sleep_for(frame_duration - elapsed);
            }
            
            total_frames_++;
        }
    }
    
    void shutdown() {
        std::cout << "\nShutting down system..." << std::endl;
        
        // Land drone if airborne
        DroneState state = flight_controller_.get_current_state();
        if (state.z > 0.5f) {
            std::cout << "Landing drone..." << std::endl;
            flight_controller_.land();
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
        
        // Disarm
        flight_controller_.disarm();
        
        // Disconnect
        flight_controller_.disconnect();
        
        // Display final statistics
        display_final_statistics();
    }

private:
    void update_statistics(const LNNController::ControlOutput& output) {
        float inference_ms = output.inference_time_us / 1000.0f;
        total_inference_time_ms_ += inference_ms;
        min_inference_time_ms_ = std::min(min_inference_time_ms_, inference_ms);
        max_inference_time_ms_ = std::max(max_inference_time_ms_, inference_ms);
    }
    
    void display_status(const LNNController::ControlOutput& output) {
        DroneState state = flight_controller_.get_current_state();
        
        std::cout << "\r[Frame " << total_frames_ << "] "
                  << "Vel: " << output.forward_velocity << " m/s, "
                  << "Yaw: " << output.yaw_rate << " rad/s, "
                  << "Alt: " << state.z << " m, "
                  << "Conf: " << (output.confidence * 100) << "%, "
                  << "Inf: " << (output.inference_time_us / 1000.0f) << " ms"
                  << std::flush;
    }
    
    void display_final_statistics() {
        std::cout << "\n\n=== Final Statistics ===" << std::endl;
        std::cout << "Total frames processed: " << total_frames_ << std::endl;
        
        if (total_frames_ > 0) {
            float avg_inference = total_inference_time_ms_ / total_frames_;
            std::cout << "Average inference time: " << avg_inference << " ms" << std::endl;
            std::cout << "Min inference time: " << min_inference_time_ms_ << " ms" << std::endl;
            std::cout << "Max inference time: " << max_inference_time_ms_ << " ms" << std::endl;
            
            auto stats = controller_.get_performance_stats();
            std::cout << "Average confidence: " << (stats.average_confidence * 100) << "%" << std::endl;
            std::cout << "Average power: " << stats.average_power_mw << " mW" << std::endl;
            std::cout << "Total energy: " << stats.total_energy_consumed_j << " J" << std::endl;
        }
    }
};

/**
 * Command-line argument parser
 */
struct CommandLineArgs {
    std::string model_path = "";
    int width = 160;
    int height = 120;
    bool adaptive = true;
    float max_inference_ms = 20.0f;
    int memory_kb = 256;
    bool help = false;
    
    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg == "--help" || arg == "-h") {
                help = true;
            } else if (arg == "--model" && i + 1 < argc) {
                model_path = argv[++i];
            } else if (arg == "--width" && i + 1 < argc) {
                width = std::atoi(argv[++i]);
            } else if (arg == "--height" && i + 1 < argc) {
                height = std::atoi(argv[++i]);
            } else if (arg == "--no-adaptive") {
                adaptive = false;
            } else if (arg == "--max-inference" && i + 1 < argc) {
                max_inference_ms = std::atof(argv[++i]);
            } else if (arg == "--memory" && i + 1 < argc) {
                memory_kb = std::atoi(argv[++i]);
            }
        }
    }
    
    void print_usage() {
        std::cout << "Liquid Vision - Adaptive Neural Networks for Drone Vision\n"
                  << "\nUsage: liquid_vision [options]\n"
                  << "\nOptions:\n"
                  << "  --model <path>        Path to LNN model file\n"
                  << "  --width <pixels>      Input image width (default: 160)\n"
                  << "  --height <pixels>     Input image height (default: 120)\n"
                  << "  --no-adaptive         Disable adaptive timestep\n"
                  << "  --max-inference <ms>  Maximum inference time (default: 20.0)\n"
                  << "  --memory <kb>         Memory limit in KB (default: 256)\n"
                  << "  --help, -h            Show this help message\n"
                  << std::endl;
    }
};

int main(int argc, char* argv[]) {
    // Parse command line arguments
    CommandLineArgs args;
    args.parse(argc, argv);
    
    if (args.help) {
        args.print_usage();
        return 0;
    }
    
    // Display banner
    std::cout << "╔════════════════════════════════════════════════╗\n"
              << "║     Liquid Vision - Drone Neural Networks     ║\n"
              << "║         Adaptive AI for Micro-UAVs            ║\n"
              << "╚════════════════════════════════════════════════╝\n"
              << std::endl;
    
    // Register signal handler for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Configure LNN controller
    LNNController::Config config;
    config.model_path = args.model_path;
    config.input_resolution = {args.width, args.height};
    config.timestep_adaptive = args.adaptive;
    config.max_inference_time_ms = args.max_inference_ms;
    config.memory_limit_kb = args.memory_kb;
    
    // Display configuration
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Model: " << (config.model_path.empty() ? "Random weights" : config.model_path) << std::endl;
    std::cout << "  Input resolution: " << config.input_resolution.first 
              << "x" << config.input_resolution.second << std::endl;
    std::cout << "  Adaptive timestep: " << (config.timestep_adaptive ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Max inference time: " << config.max_inference_time_ms << " ms" << std::endl;
    std::cout << "  Memory limit: " << config.memory_limit_kb << " KB" << std::endl;
    std::cout << std::endl;
    
    // Create and run application
    try {
        LiquidVisionApp app(config);
        
        if (!app.initialize()) {
            std::cerr << "Failed to initialize application" << std::endl;
            return 1;
        }
        
        app.run();
        app.shutdown();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Application terminated successfully" << std::endl;
    return 0;
}