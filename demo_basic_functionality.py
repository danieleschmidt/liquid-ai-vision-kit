#!/usr/bin/env python3
"""
Basic Functionality Demo for Liquid AI Vision Kit
Generation 1: MAKE IT WORK

This script demonstrates the core functionality of the Liquid Neural Network
system including vision processing, neural network inference, and flight control.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add src path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/python'))

def simulate_camera_frame(width=160, height=120, frame_number=0):
    """
    Simulate a camera frame with some basic patterns for testing
    """
    # Create a synthetic image with some patterns
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some horizontal stripes that move over time
    stripe_pos = (frame_number * 2) % height
    for i in range(5):
        y = (stripe_pos + i * 20) % height
        image[max(0, y-2):min(height, y+3), :, :] = [100 + i*30, 50, 200-i*20]
    
    # Add a moving circle
    center_x = int(width * (0.5 + 0.3 * np.sin(frame_number * 0.1)))
    center_y = int(height * (0.5 + 0.3 * np.cos(frame_number * 0.1)))
    
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 <= 15**2
    image[mask] = [255, 255, 0]  # Yellow circle
    
    # Add some noise
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

class BasicLNNDemo:
    """
    Basic demonstration of LNN functionality without C++ dependencies
    """
    
    def __init__(self):
        self.frame_count = 0
        self.stats = {
            'total_frames': 0,
            'successful_inferences': 0,
            'total_processing_time': 0.0,
            'average_confidence': 0.0
        }
        
        print("🚁 Liquid AI Vision Kit - Basic Demo")
        print("=" * 50)
        print("Initializing core components...")
        
    def preprocess_frame(self, image):
        """
        Basic image preprocessing pipeline
        """
        start_time = time.time()
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        else:
            gray = image
            
        # Resize to target resolution (160x120)
        # In real implementation this would use optimized C++ code
        target_height, target_width = 120, 160
        
        # Simple nearest neighbor resize for demo
        h, w = gray.shape
        resized = np.zeros((target_height, target_width))
        for i in range(target_height):
            for j in range(target_width):
                orig_i = int(i * h / target_height)
                orig_j = int(j * w / target_width)
                resized[i, j] = gray[orig_i, orig_j]
        
        # Normalize to [-1, 1]
        normalized = (resized / 127.5) - 1.0
        
        # Apply temporal filtering (simple exponential moving average)
        if hasattr(self, 'previous_frame'):
            alpha = 0.7
            filtered = alpha * normalized + (1 - alpha) * self.previous_frame
        else:
            filtered = normalized
            
        self.previous_frame = filtered.copy()
        
        processing_time = time.time() - start_time
        
        return {
            'data': filtered.flatten(),
            'width': target_width,
            'height': target_height,
            'processing_time_ms': processing_time * 1000
        }
    
    def simulate_lnn_inference(self, processed_frame):
        """
        Simulate liquid neural network inference
        """
        start_time = time.time()
        
        # Simulate network computation with some realistic complexity
        input_data = processed_frame['data']
        
        # Simple feedforward simulation with liquid-like dynamics
        # Layer 1: Input processing (160*120 -> 64 neurons)
        layer1_size = 64
        weights1 = np.random.randn(layer1_size, len(input_data)) * 0.1
        hidden1 = np.tanh(np.dot(weights1, input_data))
        
        # Layer 2: Liquid dynamics simulation (64 -> 32 neurons)
        layer2_size = 32
        weights2 = np.random.randn(layer2_size, layer1_size) * 0.2
        
        # Simulate continuous-time dynamics with simple Euler integration
        dt = 0.01
        state = np.zeros(layer2_size)
        tau = 1.0
        leak = 0.1
        
        for _ in range(10):  # 10 integration steps
            input_current = np.dot(weights2, hidden1)
            derivative = (-leak * state + input_current) / tau
            state = state + dt * derivative
            state = np.tanh(state)  # Apply activation
        
        # Layer 3: Output (32 -> 3 control outputs)
        output_weights = np.random.randn(3, layer2_size) * 0.3
        raw_outputs = np.dot(output_weights, state)
        
        # Map to control commands
        forward_velocity = np.tanh(raw_outputs[0]) * 2.0    # ±2 m/s
        yaw_rate = np.tanh(raw_outputs[1]) * 1.0            # ±1 rad/s  
        target_altitude = (np.tanh(raw_outputs[2]) + 1) * 5 # 0-10m
        
        # Compute confidence based on output consistency
        confidence = 1.0 - np.var(raw_outputs) / (1.0 + np.var(raw_outputs))
        confidence = max(0.0, min(1.0, confidence))
        
        inference_time = time.time() - start_time
        
        return {
            'forward_velocity': forward_velocity,
            'yaw_rate': yaw_rate,
            'target_altitude': target_altitude,
            'confidence': confidence,
            'inference_time_ms': inference_time * 1000
        }
    
    def apply_safety_constraints(self, control_output):
        """
        Apply safety constraints to control commands
        """
        # Velocity limits
        control_output['forward_velocity'] = np.clip(
            control_output['forward_velocity'], -5.0, 5.0)
        control_output['yaw_rate'] = np.clip(
            control_output['yaw_rate'], -2.0, 2.0)
        
        # Altitude limits
        control_output['target_altitude'] = np.clip(
            control_output['target_altitude'], 0.5, 50.0)
        
        # Confidence-based scaling
        if control_output['confidence'] < 0.5:
            control_output['forward_velocity'] *= 0.5
            control_output['yaw_rate'] *= 0.5
            
        return control_output
    
    def simulate_flight_response(self, control_output):
        """
        Simulate flight controller response
        """
        # This would interface with PX4/ArduPilot in real implementation
        commands = {
            'velocity_x': control_output['forward_velocity'],
            'velocity_y': 0.0,
            'velocity_z': 0.0,  # Simplified - no altitude control in demo
            'yaw_rate': control_output['yaw_rate']
        }
        
        # Simulate command execution delay
        time.sleep(0.001)  # 1ms simulated latency
        
        return {
            'commands_sent': commands,
            'flight_mode': 'GUIDED',
            'armed': True,
            'altitude': control_output['target_altitude']
        }
    
    def process_frame(self, image):
        """
        Complete frame processing pipeline
        """
        self.frame_count += 1
        frame_start_time = time.time()
        
        print(f"\n📸 Processing Frame {self.frame_count}")
        print("-" * 30)
        
        # Step 1: Preprocess image
        print("1️⃣ Image preprocessing...")
        processed_frame = self.preprocess_frame(image)
        print(f"   ✓ Resolution: {processed_frame['width']}x{processed_frame['height']}")
        print(f"   ✓ Processing time: {processed_frame['processing_time_ms']:.2f}ms")
        
        # Step 2: Neural network inference
        print("2️⃣ LNN inference...")
        control_output = self.simulate_lnn_inference(processed_frame)
        print(f"   ✓ Forward velocity: {control_output['forward_velocity']:.2f} m/s")
        print(f"   ✓ Yaw rate: {control_output['yaw_rate']:.2f} rad/s")
        print(f"   ✓ Target altitude: {control_output['target_altitude']:.1f} m")
        print(f"   ✓ Confidence: {control_output['confidence']:.1%}")
        print(f"   ✓ Inference time: {control_output['inference_time_ms']:.2f}ms")
        
        # Step 3: Apply safety constraints
        print("3️⃣ Safety validation...")
        safe_control = self.apply_safety_constraints(control_output)
        print("   ✓ Safety constraints applied")
        
        # Step 4: Send to flight controller
        print("4️⃣ Flight control...")
        flight_response = self.simulate_flight_response(safe_control)
        print(f"   ✓ Commands sent to flight controller")
        print(f"   ✓ Mode: {flight_response['flight_mode']}")
        
        # Update statistics
        total_time = time.time() - frame_start_time
        self.update_stats(control_output, total_time)
        
        print(f"⏱️ Total processing time: {total_time*1000:.2f}ms")
        
        return {
            'processed_frame': processed_frame,
            'control_output': safe_control,
            'flight_response': flight_response,
            'total_time_ms': total_time * 1000
        }
    
    def update_stats(self, control_output, total_time):
        """
        Update performance statistics
        """
        self.stats['total_frames'] += 1
        self.stats['total_processing_time'] += total_time
        
        if control_output['confidence'] > 0.5:
            self.stats['successful_inferences'] += 1
            
        # Update running average of confidence
        alpha = 0.9
        self.stats['average_confidence'] = (
            alpha * self.stats['average_confidence'] + 
            (1 - alpha) * control_output['confidence']
        )
    
    def print_stats(self):
        """
        Print performance statistics
        """
        print("\n📊 Performance Statistics")
        print("=" * 50)
        print(f"Total frames processed: {self.stats['total_frames']}")
        print(f"Successful inferences: {self.stats['successful_inferences']}")
        print(f"Success rate: {self.stats['successful_inferences']/max(1, self.stats['total_frames'])*100:.1f}%")
        print(f"Average processing time: {self.stats['total_processing_time']/max(1, self.stats['total_frames'])*1000:.2f}ms")
        print(f"Average confidence: {self.stats['average_confidence']:.1%}")
        
        # Estimate power consumption
        frames_per_second = self.stats['total_frames'] / max(0.1, self.stats['total_processing_time'])
        estimated_power = 400 + frames_per_second * 50  # Base + processing power in mW
        print(f"Estimated power consumption: {estimated_power:.0f}mW")
    
    def run_demo(self, num_frames=20):
        """
        Run the complete demo
        """
        print(f"🎬 Starting demo with {num_frames} frames")
        print("Simulating camera input and processing pipeline...\n")
        
        try:
            for i in range(num_frames):
                # Generate synthetic camera frame
                camera_frame = simulate_camera_frame(frame_number=i)
                
                # Process the frame
                result = self.process_frame(camera_frame)
                
                # Small delay to simulate real-time processing
                time.sleep(0.05)
                
            # Print final statistics
            self.print_stats()
            
            print("\n✅ Demo completed successfully!")
            print("🚀 Ready for Generation 2: Enhanced robustness and error handling")
            
        except KeyboardInterrupt:
            print("\n⏹️ Demo interrupted by user")
            self.print_stats()
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """
    Main demo function
    """
    print("🧠 Liquid AI Vision Kit - Generation 1 Demo")
    print("Making it work with basic functionality...")
    print()
    
    # Create and run demo
    demo = BasicLNNDemo()
    demo.run_demo(num_frames=10)
    
    print("\n🎯 Next Steps:")
    print("- ✅ Generation 1: Basic functionality working")
    print("- 🔄 Generation 2: Add robustness and error handling")
    print("- ⚡ Generation 3: Optimize for scale and performance")
    print("- 🧪 Add comprehensive testing and validation")

if __name__ == "__main__":
    main()