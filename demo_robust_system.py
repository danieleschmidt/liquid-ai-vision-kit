#!/usr/bin/env python3
"""
Robust System Demo for Liquid AI Vision Kit
Generation 2: MAKE IT ROBUST

This script demonstrates enhanced error handling, logging, monitoring, and
graceful degradation capabilities of the Liquid Neural Network system.
"""

import sys
import os
import time
import numpy as np
import threading
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

# Add src path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/python'))

class LogLevel(Enum):
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4
    FATAL = 5

class ComponentStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"

class SystemMode(Enum):
    NORMAL = "normal"
    REDUCED_QUALITY = "reduced_quality"
    MINIMAL_FEATURES = "minimal_features"
    EMERGENCY_MODE = "emergency_mode"
    SHUTDOWN = "shutdown"

@dataclass
class HealthMetrics:
    component_name: str
    status: ComponentStatus
    message: str
    last_update: float
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    error_count: int = 0
    uptime_seconds: float = 0.0

@dataclass
class PerformanceMetrics:
    timestamp: float
    frame_id: int
    inference_time_ms: float
    preprocessing_time_ms: float
    total_frame_time_ms: float
    power_consumption_mw: float
    confidence: float
    memory_usage_kb: int
    fps: float = 0.0
    success: bool = True
    error_message: str = ""

class Logger:
    """
    Robust logging system with multiple outputs and error handling
    """
    
    def __init__(self, min_level=LogLevel.INFO, log_file="liquid_vision_robust.log"):
        self.min_level = min_level
        self.log_file = log_file
        self.lock = threading.Lock()
        
        # Create log directory if it doesn't exist
        log_dir = Path(log_file).parent
        log_dir.mkdir(exist_ok=True)
        
    def log(self, level: LogLevel, message: str, component: str = "SYSTEM"):
        if level.value < self.min_level.value:
            return
            
        with self.lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            thread_id = threading.current_thread().ident
            level_str = level.name.ljust(5)
            
            log_entry = f"[{timestamp}] [{level_str}] [{thread_id}] [{component}] {message}"
            
            # Console output with color coding
            color_codes = {
                LogLevel.TRACE: "\033[36m",    # Cyan
                LogLevel.DEBUG: "\033[37m",    # White
                LogLevel.INFO: "\033[32m",     # Green
                LogLevel.WARN: "\033[33m",     # Yellow
                LogLevel.ERROR: "\033[31m",    # Red
                LogLevel.FATAL: "\033[35m"     # Magenta
            }
            reset_code = "\033[0m"
            
            colored_entry = f"{color_codes.get(level, '')}{log_entry}{reset_code}"
            print(colored_entry)
            
            # File output (without colors)
            try:
                with open(self.log_file, 'a') as f:
                    f.write(log_entry + '\n')
            except Exception as e:
                print(f"Failed to write to log file: {e}")
    
    def trace(self, message, component="SYSTEM"):
        self.log(LogLevel.TRACE, message, component)
    
    def debug(self, message, component="SYSTEM"):
        self.log(LogLevel.DEBUG, message, component)
    
    def info(self, message, component="SYSTEM"):
        self.log(LogLevel.INFO, message, component)
    
    def warn(self, message, component="SYSTEM"):
        self.log(LogLevel.WARN, message, component)
    
    def error(self, message, component="SYSTEM"):
        self.log(LogLevel.ERROR, message, component)
    
    def fatal(self, message, component="SYSTEM"):
        self.log(LogLevel.FATAL, message, component)

class CircuitBreaker:
    """
    Circuit breaker pattern for handling repeated failures
    """
    
    def __init__(self, failure_threshold=3, timeout_seconds=10, success_threshold=2):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        with self.lock:
            current_time = time.time()
            
            if self.state == "OPEN":
                if current_time - self.last_failure_time < self.timeout_seconds:
                    raise Exception(f"Circuit breaker is OPEN (timeout: {self.timeout_seconds}s)")
                else:
                    self.state = "HALF_OPEN"
                    self.success_count = 0
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _on_success(self):
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = 0
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state in ["CLOSED", "HALF_OPEN"] and self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class HealthMonitor:
    """
    Comprehensive health monitoring system
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.components = {}
        self.lock = threading.Lock()
        self.monitoring_active = False
        self.monitor_thread = None
    
    def register_component(self, name: str):
        """Register a component for health monitoring"""
        with self.lock:
            self.components[name] = HealthMetrics(
                component_name=name,
                status=ComponentStatus.OFFLINE,
                message="Component registered",
                last_update=time.time(),
                uptime_seconds=0.0
            )
        self.logger.info(f"Registered component for health monitoring: {name}", "HEALTH_MONITOR")
    
    def update_component_health(self, name: str, status: ComponentStatus, 
                              message: str = "", **kwargs):
        """Update component health status"""
        with self.lock:
            if name not in self.components:
                self.register_component(name)
            
            metrics = self.components[name]
            old_status = metrics.status
            
            metrics.status = status
            metrics.message = message
            metrics.last_update = time.time()
            
            # Update additional metrics
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
            
            # Log status changes
            if old_status != status:
                self.logger.warn(f"Component {name} status changed: {old_status.value} -> {status.value} ({message})", 
                               "HEALTH_MONITOR")
    
    def get_component_health(self, name: str) -> Optional[HealthMetrics]:
        """Get health metrics for a specific component"""
        with self.lock:
            return self.components.get(name)
    
    def get_all_health(self) -> Dict[str, HealthMetrics]:
        """Get health metrics for all components"""
        with self.lock:
            return self.components.copy()
    
    def is_system_healthy(self) -> bool:
        """Check if the entire system is healthy"""
        with self.lock:
            for component in self.components.values():
                if component.status in [ComponentStatus.CRITICAL, ComponentStatus.OFFLINE]:
                    return False
            return True
    
    def start_monitoring(self, interval_seconds=5):
        """Start periodic health monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, 
                                             args=(interval_seconds,), daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"Started health monitoring (interval: {interval_seconds}s)", "HEALTH_MONITOR")
    
    def stop_monitoring(self):
        """Stop periodic health monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)
        self.logger.info("Stopped health monitoring", "HEALTH_MONITOR")
    
    def _monitor_loop(self, interval_seconds):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                current_time = time.time()
                stale_threshold = 30.0  # seconds
                
                with self.lock:
                    for name, metrics in self.components.items():
                        # Check for stale components
                        if (current_time - metrics.last_update > stale_threshold and 
                            metrics.status != ComponentStatus.OFFLINE):
                            
                            metrics.status = ComponentStatus.WARNING
                            metrics.message = f"Component appears stale (last update: {current_time - metrics.last_update:.1f}s ago)"
                            self.logger.warn(f"Component {name} is stale", "HEALTH_MONITOR")
                        
                        # Update uptime
                        metrics.uptime_seconds = current_time - (metrics.last_update - metrics.uptime_seconds)
                
                time.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}", "HEALTH_MONITOR")
                time.sleep(1)

class PerformanceTracker:
    """
    Performance metrics tracking and analysis
    """
    
    def __init__(self, logger, max_history=1000):
        self.logger = logger
        self.max_history = max_history
        self.metrics_history = []
        self.lock = threading.Lock()
    
    def log_frame_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics for a frame"""
        with self.lock:
            self.metrics_history.append(metrics)
            
            # Keep only the latest metrics
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)
    
    def get_average_metrics(self, last_n_frames=100) -> Dict[str, float]:
        """Get average performance metrics"""
        with self.lock:
            if not self.metrics_history:
                return {}
            
            recent_metrics = self.metrics_history[-last_n_frames:]
            
            total_frames = len(recent_metrics)
            successful_frames = sum(1 for m in recent_metrics if m.success)
            
            avg_metrics = {
                'total_frames': total_frames,
                'successful_frames': successful_frames,
                'success_rate': successful_frames / total_frames if total_frames > 0 else 0,
                'avg_inference_time_ms': sum(m.inference_time_ms for m in recent_metrics) / total_frames if total_frames > 0 else 0,
                'avg_preprocessing_time_ms': sum(m.preprocessing_time_ms for m in recent_metrics) / total_frames if total_frames > 0 else 0,
                'avg_total_frame_time_ms': sum(m.total_frame_time_ms for m in recent_metrics) / total_frames if total_frames > 0 else 0,
                'avg_power_consumption_mw': sum(m.power_consumption_mw for m in recent_metrics) / total_frames if total_frames > 0 else 0,
                'avg_confidence': sum(m.confidence for m in recent_metrics) / total_frames if total_frames > 0 else 0,
                'avg_fps': sum(m.fps for m in recent_metrics) / total_frames if total_frames > 0 else 0
            }
            
            return avg_metrics
    
    def export_metrics_csv(self, filename="performance_metrics.csv"):
        """Export metrics to CSV file"""
        with self.lock:
            try:
                with open(filename, 'w') as f:
                    # CSV header
                    f.write("timestamp,frame_id,inference_time_ms,preprocessing_time_ms,total_frame_time_ms,"
                           "power_consumption_mw,confidence,memory_usage_kb,fps,success,error_message\n")
                    
                    # CSV data
                    for m in self.metrics_history:
                        f.write(f"{m.timestamp},{m.frame_id},{m.inference_time_ms},{m.preprocessing_time_ms},"
                               f"{m.total_frame_time_ms},{m.power_consumption_mw},{m.confidence},{m.memory_usage_kb},"
                               f"{m.fps},{m.success},{m.error_message}\n")
                
                self.logger.info(f"Exported {len(self.metrics_history)} metrics to {filename}", "PERF_TRACKER")
            except Exception as e:
                self.logger.error(f"Failed to export metrics: {e}", "PERF_TRACKER")

class RobustLNNController:
    """
    Enhanced LNN Controller with comprehensive error handling and monitoring
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.health_monitor = HealthMonitor(logger)
        self.perf_tracker = PerformanceTracker(logger)
        
        # Circuit breakers for different components
        self.vision_breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=5)
        self.inference_breaker = CircuitBreaker(failure_threshold=5, timeout_seconds=10)
        self.control_breaker = CircuitBreaker(failure_threshold=2, timeout_seconds=3)
        
        self.system_mode = SystemMode.NORMAL
        self.frame_count = 0
        self.error_counts = {
            'vision': 0,
            'inference': 0,
            'control': 0,
            'safety': 0
        }
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components with error handling"""
        try:
            # Register components for health monitoring
            self.health_monitor.register_component("vision_processor")
            self.health_monitor.register_component("neural_network")
            self.health_monitor.register_component("flight_controller")
            self.health_monitor.register_component("safety_monitor")
            
            # Start health monitoring
            self.health_monitor.start_monitoring(interval_seconds=3)
            
            # Update component status to healthy
            self.health_monitor.update_component_health("vision_processor", ComponentStatus.HEALTHY, "Initialized")
            self.health_monitor.update_component_health("neural_network", ComponentStatus.HEALTHY, "Model loaded")
            self.health_monitor.update_component_health("flight_controller", ComponentStatus.HEALTHY, "Connected")
            self.health_monitor.update_component_health("safety_monitor", ComponentStatus.HEALTHY, "Active")
            
            self.logger.info("All components initialized successfully", "LNN_CONTROLLER")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}", "LNN_CONTROLLER")
            raise
    
    def _preprocess_frame_robust(self, image, frame_id):
        """Robust image preprocessing with error handling"""
        try:
            result = self.vision_breaker.call(self._preprocess_frame_internal, image)
            
            self.health_monitor.update_component_health("vision_processor", 
                                                      ComponentStatus.HEALTHY, 
                                                      f"Processed frame {frame_id}")
            return result
            
        except Exception as e:
            self.error_counts['vision'] += 1
            self.logger.error(f"Vision preprocessing failed for frame {frame_id}: {e}", "VISION")
            
            self.health_monitor.update_component_health("vision_processor", 
                                                      ComponentStatus.CRITICAL, 
                                                      f"Processing error: {str(e)[:100]}",
                                                      error_count=self.error_counts['vision'])
            
            # Return fallback preprocessed frame
            return self._get_fallback_frame()
    
    def _preprocess_frame_internal(self, image):
        """Internal preprocessing logic"""
        # Simulate occasional failures for demo
        if random.random() < 0.1:  # 10% failure rate
            raise Exception("Simulated vision processing failure")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        else:
            gray = image
        
        # Simple resize and normalize
        h, w = gray.shape
        target_h, target_w = 120, 160
        
        resized = np.zeros((target_h, target_w))
        for i in range(target_h):
            for j in range(target_w):
                orig_i = int(i * h / target_h)
                orig_j = int(j * w / target_w)
                resized[i, j] = gray[orig_i, orig_j]
        
        normalized = (resized / 127.5) - 1.0
        
        return {
            'data': normalized.flatten(),
            'width': target_w,
            'height': target_h,
            'success': True
        }
    
    def _get_fallback_frame(self):
        """Generate a safe fallback frame when processing fails"""
        self.logger.info("Using fallback frame due to processing failure", "VISION")
        return {
            'data': np.zeros(160 * 120),
            'width': 160,
            'height': 120,
            'success': False
        }
    
    def _run_inference_robust(self, processed_frame, frame_id):
        """Robust neural network inference with error handling"""
        try:
            result = self.inference_breaker.call(self._run_inference_internal, processed_frame)
            
            self.health_monitor.update_component_health("neural_network", 
                                                      ComponentStatus.HEALTHY, 
                                                      f"Inference completed for frame {frame_id}",
                                                      memory_usage_mb=50.5)
            return result
            
        except Exception as e:
            self.error_counts['inference'] += 1
            self.logger.error(f"Neural network inference failed for frame {frame_id}: {e}", "INFERENCE")
            
            self.health_monitor.update_component_health("neural_network", 
                                                      ComponentStatus.CRITICAL, 
                                                      f"Inference error: {str(e)[:100]}",
                                                      error_count=self.error_counts['inference'])
            
            # Return safe fallback control output
            return self._get_fallback_control_output()
    
    def _run_inference_internal(self, processed_frame):
        """Internal inference logic"""
        # Simulate occasional failures for demo
        if random.random() < 0.05:  # 5% failure rate
            raise Exception("Simulated neural network failure")
        
        # Simulate neural network processing
        input_data = processed_frame['data']
        
        # Simple simulation with some realistic complexity
        weights = np.random.randn(3, len(input_data)) * 0.1
        raw_outputs = np.dot(weights, input_data)
        
        # Apply activation and scaling
        forward_velocity = np.tanh(raw_outputs[0]) * 2.0
        yaw_rate = np.tanh(raw_outputs[1]) * 1.0
        target_altitude = (np.tanh(raw_outputs[2]) + 1) * 5.0
        
        confidence = 1.0 - np.var(raw_outputs) / (1.0 + np.var(raw_outputs))
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            'forward_velocity': forward_velocity,
            'yaw_rate': yaw_rate,
            'target_altitude': target_altitude,
            'confidence': confidence,
            'success': True
        }
    
    def _get_fallback_control_output(self):
        """Generate safe fallback control output"""
        self.logger.info("Using fallback control output due to inference failure", "INFERENCE")
        return {
            'forward_velocity': 0.0,  # Stop moving
            'yaw_rate': 0.0,          # Stop turning
            'target_altitude': 2.0,   # Maintain safe altitude
            'confidence': 0.0,
            'success': False
        }
    
    def _apply_safety_checks_robust(self, control_output, frame_id):
        """Robust safety validation with error handling"""
        try:
            result = self.control_breaker.call(self._apply_safety_checks_internal, control_output)
            
            self.health_monitor.update_component_health("safety_monitor", 
                                                      ComponentStatus.HEALTHY, 
                                                      f"Safety check passed for frame {frame_id}")
            return result
            
        except Exception as e:
            self.error_counts['safety'] += 1
            self.logger.error(f"Safety check failed for frame {frame_id}: {e}", "SAFETY")
            
            self.health_monitor.update_component_health("safety_monitor", 
                                                      ComponentStatus.CRITICAL, 
                                                      f"Safety error: {str(e)[:100]}",
                                                      error_count=self.error_counts['safety'])
            
            # Apply emergency stop
            return self._get_emergency_stop_output()
    
    def _apply_safety_checks_internal(self, control_output):
        """Internal safety check logic"""
        # Simulate occasional safety violations for demo
        if random.random() < 0.02:  # 2% failure rate
            raise Exception("Simulated safety violation detected")
        
        # Apply safety constraints
        safe_output = control_output.copy()
        
        # Velocity limits
        safe_output['forward_velocity'] = np.clip(safe_output['forward_velocity'], -5.0, 5.0)
        safe_output['yaw_rate'] = np.clip(safe_output['yaw_rate'], -2.0, 2.0)
        safe_output['target_altitude'] = np.clip(safe_output['target_altitude'], 0.5, 50.0)
        
        # Confidence-based scaling
        if safe_output['confidence'] < 0.3:
            safe_output['forward_velocity'] *= 0.5
            safe_output['yaw_rate'] *= 0.5
        
        return safe_output
    
    def _get_emergency_stop_output(self):
        """Generate emergency stop control output"""
        self.logger.warn("Activating emergency stop due to safety failure", "SAFETY")
        return {
            'forward_velocity': 0.0,
            'yaw_rate': 0.0,
            'target_altitude': 2.0,
            'confidence': 1.0,  # High confidence in stop command
            'emergency_stop': True
        }
    
    def _adapt_system_mode(self):
        """Adapt system mode based on health status"""
        health_status = self.health_monitor.get_all_health()
        critical_components = 0
        warning_components = 0
        
        for component, metrics in health_status.items():
            if metrics.status == ComponentStatus.CRITICAL:
                critical_components += 1
            elif metrics.status == ComponentStatus.WARNING:
                warning_components += 1
        
        new_mode = self.system_mode
        
        # Determine appropriate system mode
        if critical_components > 0:
            if critical_components >= 2:
                new_mode = SystemMode.EMERGENCY_MODE
            else:
                new_mode = SystemMode.MINIMAL_FEATURES
        elif warning_components > 1:
            new_mode = SystemMode.REDUCED_QUALITY
        else:
            new_mode = SystemMode.NORMAL
        
        if new_mode != self.system_mode:
            self.logger.warn(f"System mode changing: {self.system_mode.value} -> {new_mode.value}", "SYSTEM")
            self.system_mode = new_mode
    
    def process_frame(self, image):
        """Complete robust frame processing pipeline"""
        self.frame_count += 1
        frame_start_time = time.time()
        
        self.logger.debug(f"Processing frame {self.frame_count} in {self.system_mode.value} mode", "LNN_CONTROLLER")
        
        # Initialize metrics
        metrics = PerformanceMetrics(
            timestamp=frame_start_time,
            frame_id=self.frame_count,
            inference_time_ms=0.0,
            preprocessing_time_ms=0.0,
            total_frame_time_ms=0.0,
            power_consumption_mw=0.0,
            confidence=0.0,
            memory_usage_kb=256
        )
        
        try:
            # Step 1: Vision preprocessing
            preprocess_start = time.time()
            processed_frame = self._preprocess_frame_robust(image, self.frame_count)
            metrics.preprocessing_time_ms = (time.time() - preprocess_start) * 1000
            
            # Step 2: Neural network inference
            inference_start = time.time()
            control_output = self._run_inference_robust(processed_frame, self.frame_count)
            metrics.inference_time_ms = (time.time() - inference_start) * 1000
            metrics.confidence = control_output.get('confidence', 0.0)
            
            # Step 3: Safety validation
            safe_control = self._apply_safety_checks_robust(control_output, self.frame_count)
            
            # Step 4: Adapt system mode based on health
            self._adapt_system_mode()
            
            # Calculate final metrics
            metrics.total_frame_time_ms = (time.time() - frame_start_time) * 1000
            metrics.power_consumption_mw = 450 + random.uniform(-50, 50)  # Simulate power usage
            metrics.fps = 1000.0 / metrics.total_frame_time_ms if metrics.total_frame_time_ms > 0 else 0
            metrics.success = processed_frame.get('success', True) and control_output.get('success', True)
            
            # Log metrics
            self.perf_tracker.log_frame_metrics(metrics)
            
            # Update flight controller health
            self.health_monitor.update_component_health("flight_controller", 
                                                      ComponentStatus.HEALTHY, 
                                                      f"Commands sent for frame {self.frame_count}",
                                                      cpu_usage=25.5)
            
            return {
                'frame_id': self.frame_count,
                'system_mode': self.system_mode.value,
                'control_output': safe_control,
                'metrics': metrics,
                'health_status': self.health_monitor.get_all_health()
            }
            
        except Exception as e:
            self.logger.fatal(f"Critical error in frame processing: {e}", "LNN_CONTROLLER")
            
            metrics.success = False
            metrics.error_message = str(e)
            metrics.total_frame_time_ms = (time.time() - frame_start_time) * 1000
            self.perf_tracker.log_frame_metrics(metrics)
            
            # Force emergency mode
            self.system_mode = SystemMode.EMERGENCY_MODE
            
            return {
                'frame_id': self.frame_count,
                'system_mode': self.system_mode.value,
                'control_output': self._get_emergency_stop_output(),
                'metrics': metrics,
                'error': str(e)
            }
    
    def get_system_status(self):
        """Get comprehensive system status"""
        health_status = self.health_monitor.get_all_health()
        performance_metrics = self.perf_tracker.get_average_metrics(last_n_frames=50)
        
        return {
            'system_mode': self.system_mode.value,
            'frame_count': self.frame_count,
            'system_healthy': self.health_monitor.is_system_healthy(),
            'component_health': {name: asdict(metrics) for name, metrics in health_status.items()},
            'performance': performance_metrics,
            'error_counts': self.error_counts,
            'circuit_breaker_states': {
                'vision': self.vision_breaker.state,
                'inference': self.inference_breaker.state,
                'control': self.control_breaker.state
            }
        }
    
    def shutdown(self):
        """Graceful system shutdown"""
        self.logger.info("Initiating graceful system shutdown", "LNN_CONTROLLER")
        
        try:
            # Stop health monitoring
            self.health_monitor.stop_monitoring()
            
            # Export performance metrics
            self.perf_tracker.export_metrics_csv("robust_demo_metrics.csv")
            
            # Update all components to offline
            for component_name in ["vision_processor", "neural_network", "flight_controller", "safety_monitor"]:
                self.health_monitor.update_component_health(component_name, 
                                                          ComponentStatus.OFFLINE, 
                                                          "System shutdown")
            
            self.logger.info("System shutdown completed successfully", "LNN_CONTROLLER")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", "LNN_CONTROLLER")

def simulate_camera_frame(width=160, height=120, frame_number=0):
    """Simulate camera frame with varying complexity"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add moving patterns
    stripe_pos = (frame_number * 3) % height
    for i in range(3):
        y = (stripe_pos + i * 30) % height
        image[max(0, y-3):min(height, y+4), :, :] = [80 + i*40, 60, 180-i*30]
    
    # Add moving objects
    center_x = int(width * (0.5 + 0.4 * np.sin(frame_number * 0.15)))
    center_y = int(height * (0.5 + 0.4 * np.cos(frame_number * 0.1)))
    
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 <= 20**2
    image[mask] = [255, 200, 50]
    
    # Add noise (more noise = more processing challenge)
    noise_level = 30 + 20 * np.sin(frame_number * 0.05)
    noise = np.random.randint(0, int(noise_level), (height, width, 3), dtype=np.uint8)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

def main():
    """
    Main robust system demo
    """
    print("🧠 Liquid AI Vision Kit - Generation 2 Demo")
    print("Making it ROBUST with comprehensive error handling...")
    print()
    
    # Initialize logger
    logger = Logger(min_level=LogLevel.INFO, log_file="robust_demo.log")
    
    # Create robust controller
    controller = RobustLNNController(logger)
    
    logger.info("Starting robust system demonstration", "MAIN")
    
    try:
        # Run demonstration
        num_frames = 25
        status_report_interval = 10
        
        for frame_num in range(num_frames):
            # Generate test frame
            camera_frame = simulate_camera_frame(frame_number=frame_num)
            
            # Process frame through robust pipeline
            result = controller.process_frame(camera_frame)
            
            # Display results
            metrics = result['metrics']
            control = result['control_output']
            
            logger.info(f"Frame {result['frame_id']:3d} | Mode: {result['system_mode']:15s} | "
                       f"Time: {metrics.total_frame_time_ms:6.2f}ms | "
                       f"Conf: {metrics.confidence:4.1%} | "
                       f"Vel: {control.get('forward_velocity', 0):5.2f}m/s | "
                       f"Yaw: {control.get('yaw_rate', 0):5.2f}rad/s", "DEMO")
            
            # Periodic status reports
            if (frame_num + 1) % status_report_interval == 0:
                status = controller.get_system_status()
                logger.info("=== SYSTEM STATUS REPORT ===", "STATUS")
                logger.info(f"System Health: {'✅ HEALTHY' if status['system_healthy'] else '⚠️ DEGRADED'}", "STATUS")
                logger.info(f"System Mode: {status['system_mode']}", "STATUS")
                logger.info(f"Performance: {status['performance']['success_rate']*100:.1f}% success rate, "
                           f"{status['performance']['avg_fps']:.1f} FPS", "STATUS")
                logger.info(f"Errors: Vision={status['error_counts']['vision']}, "
                           f"Inference={status['error_counts']['inference']}, "
                           f"Safety={status['error_counts']['safety']}", "STATUS")
                logger.info("============================", "STATUS")
            
            # Simulate real-time processing
            time.sleep(0.02)
        
        # Final system status
        final_status = controller.get_system_status()
        
        print("\n🎯 ROBUST SYSTEM DEMONSTRATION COMPLETED")
        print("=" * 60)
        print(f"📊 Total frames processed: {final_status['frame_count']}")
        print(f"🏥 System health: {'✅ HEALTHY' if final_status['system_healthy'] else '⚠️ DEGRADED'}")
        print(f"📈 Success rate: {final_status['performance']['success_rate']*100:.1f}%")
        print(f"⚡ Average FPS: {final_status['performance']['avg_fps']:.1f}")
        print(f"🔧 Average processing time: {final_status['performance']['avg_total_frame_time_ms']:.2f}ms")
        print(f"🔋 Average power consumption: {final_status['performance']['avg_power_consumption_mw']:.0f}mW")
        print(f"❌ Total errors: {sum(final_status['error_counts'].values())}")
        
        print(f"\n🔄 Circuit Breaker States:")
        for component, state in final_status['circuit_breaker_states'].items():
            print(f"  {component}: {state}")
        
        print(f"\n📁 Performance data exported to: robust_demo_metrics.csv")
        
    except KeyboardInterrupt:
        logger.warn("Demo interrupted by user", "MAIN")
    except Exception as e:
        logger.fatal(f"Demo failed with critical error: {e}", "MAIN")
        import traceback
        traceback.print_exc()
    finally:
        # Graceful shutdown
        controller.shutdown()
        logger.info("Demo completed", "MAIN")
    
    print("\n✅ Generation 2 COMPLETED: System is now ROBUST")
    print("🚀 Ready for Generation 3: Optimize for SCALE and PERFORMANCE")

if __name__ == "__main__":
    main()