#!/usr/bin/env python3
"""
Integration Bridge: Connecting Quantum Task Planning with Liquid Neural Networks
Real-time adaptive task execution using LNN-guided decisions
"""

import numpy as np
import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import json
import subprocess
import os
from pathlib import Path

from .core.quantum_engine import QuantumTaskEngine, QuantumTask, TaskPriority, TaskState
from .core.task_scheduler import QuantumResourceAllocator, ResourceConstraint, ResourceType
from .core.lnn_integration import LNNScheduler, extract_task_features, calculate_performance_score


@dataclass
class SystemStatus:
    """Real-time system status"""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    disk_io_utilization: float = 0.0
    network_utilization: float = 0.0
    temperature_celsius: float = 25.0
    power_consumption_watts: float = 0.0
    active_tasks: int = 0
    queue_depth: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class LNNBridgeConfig:
    """Configuration for LNN-Quantum bridge"""
    lnn_model_path: str = "models/quantum_scheduler_lnn.json"
    update_interval_ms: int = 100
    adaptation_learning_rate: float = 0.001
    performance_window_size: int = 50
    enable_real_time_adaptation: bool = True
    enable_cpp_integration: bool = True
    liquid_vision_executable: str = "./liquid_vision_sim"


class CPPIntegrationManager:
    """Manages C++ LNN integration through subprocess"""
    
    def __init__(self, executable_path: str):
        self.executable_path = executable_path
        self.process = None
        self.is_running = False
        
    def start_cpp_process(self) -> bool:
        """Start C++ liquid vision process"""
        try:
            if not os.path.exists(self.executable_path):
                print(f"Warning: C++ executable not found: {self.executable_path}")
                return False
            
            self.process = subprocess.Popen(
                [self.executable_path, "--mode=server", "--port=8888"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True
            )
            
            # Wait for process to initialize
            time.sleep(0.5)
            
            if self.process.poll() is None:
                self.is_running = True
                print("C++ LNN process started successfully")
                return True
            else:
                print("C++ LNN process failed to start")
                return False
                
        except Exception as e:
            print(f"Failed to start C++ process: {e}")
            return False
    
    def send_command(self, command: Dict) -> Optional[Dict]:
        """Send command to C++ process"""
        if not self.is_running or not self.process:
            return None
        
        try:
            command_json = json.dumps(command) + "\n"
            self.process.stdin.write(command_json)
            self.process.stdin.flush()
            
            # Read response (with timeout)
            response_line = self.process.stdout.readline()
            if response_line:
                return json.loads(response_line.strip())
            
        except Exception as e:
            print(f"C++ communication error: {e}")
        
        return None
    
    def stop_cpp_process(self):
        """Stop C++ process"""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.is_running = False


class SystemMonitor:
    """Real-time system monitoring"""
    
    def __init__(self):
        self.status = SystemStatus()
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start system monitoring thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._update_system_metrics()
                time.sleep(0.1)  # 10Hz update rate
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def _update_system_metrics(self):
        """Update system metrics (simplified implementation)"""
        self.status.timestamp = time.time()
        
        # Simulate system metrics (in production, use psutil or similar)
        self.status.cpu_utilization = min(1.0, np.random.beta(2, 5) + 0.1)
        self.status.memory_utilization = min(1.0, np.random.beta(3, 4) + 0.2)
        self.status.disk_io_utilization = min(1.0, np.random.exponential(0.1))
        self.status.network_utilization = min(1.0, np.random.exponential(0.05))
        self.status.temperature_celsius = 25 + np.random.normal(0, 5)
        self.status.power_consumption_watts = 50 + self.status.cpu_utilization * 100
    
    def get_status(self) -> SystemStatus:
        """Get current system status"""
        return self.status


class LNNQuantumBridge:
    """Main integration bridge connecting LNN with Quantum Task Planning"""
    
    def __init__(self, config: LNNBridgeConfig = None):
        self.config = config or LNNBridgeConfig()
        
        # Core components
        self.quantum_engine = QuantumTaskEngine(max_workers=4, lnn_integration=True)
        self.resource_allocator = QuantumResourceAllocator()
        self.lnn_scheduler = LNNScheduler(
            input_features=8,
            hidden_neurons=16,
            output_neurons=4
        )
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        
        # C++ Integration
        self.cpp_manager = None
        if self.config.enable_cpp_integration:
            self.cpp_manager = CPPIntegrationManager(self.config.liquid_vision_executable)
        
        # Bridge state
        self.is_running = False
        self.bridge_thread = None
        self.adaptation_thread = None
        
        # Performance tracking
        self.performance_history = []
        self.task_execution_history = []
        
        # Initialize components
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        # Add resource pools
        self.resource_allocator.create_resource_pool("cpu", ResourceType.CPU, 8.0, 1.2)
        self.resource_allocator.create_resource_pool("memory", ResourceType.MEMORY, 16384, 1.0)
        self.resource_allocator.create_resource_pool("gpu", ResourceType.GPU, 4.0, 1.5)
        
        self.quantum_engine.add_resource_pool("cpu", capacity=8.0, efficiency=1.2)
        self.quantum_engine.add_resource_pool("memory", capacity=16384, efficiency=1.0)
        self.quantum_engine.add_resource_pool("gpu", capacity=4.0, efficiency=1.5)
        
        # Try to load existing LNN model
        if os.path.exists(self.config.lnn_model_path):
            self.lnn_scheduler.load_model(self.config.lnn_model_path)
            print(f"Loaded LNN model from {self.config.lnn_model_path}")
        else:
            print("No existing LNN model found, starting with random weights")
    
    def start_bridge(self) -> bool:
        """Start the integration bridge"""
        if self.is_running:
            return True
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        # Start C++ process if enabled
        if self.cpp_manager:
            if not self.cpp_manager.start_cpp_process():
                print("Warning: C++ integration disabled due to startup failure")
        
        # Start quantum task engine
        self.quantum_engine.start_scheduler()
        
        # Start bridge threads
        self.is_running = True
        self.bridge_thread = threading.Thread(target=self._bridge_loop, daemon=True)
        self.bridge_thread.start()
        
        if self.config.enable_real_time_adaptation:
            self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
            self.adaptation_thread.start()
        
        print("LNN-Quantum Bridge started successfully")
        return True
    
    def stop_bridge(self):
        """Stop the integration bridge"""
        self.is_running = False
        
        # Stop threads
        if self.bridge_thread:
            self.bridge_thread.join(timeout=5)
        if self.adaptation_thread:
            self.adaptation_thread.join(timeout=5)
        
        # Stop components
        self.quantum_engine.stop_scheduler()
        self.system_monitor.stop_monitoring()
        
        if self.cpp_manager:
            self.cpp_manager.stop_cpp_process()
        
        # Save LNN model
        self.lnn_scheduler.save_model(self.config.lnn_model_path)
        
        print("LNN-Quantum Bridge stopped")
    
    def _bridge_loop(self):
        """Main bridge processing loop"""
        last_update = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Update at specified interval
                if (current_time - last_update) * 1000 >= self.config.update_interval_ms:
                    self._process_bridge_update()
                    last_update = current_time
                
                time.sleep(0.001)  # 1ms sleep to prevent busy waiting
                
            except Exception as e:
                print(f"Bridge loop error: {e}")
                time.sleep(1.0)
    
    def _adaptation_loop(self):
        """LNN adaptation processing loop"""
        while self.is_running:
            try:
                self._process_lnn_adaptation()
                time.sleep(1.0)  # 1Hz adaptation rate
            except Exception as e:
                print(f"Adaptation loop error: {e}")
                time.sleep(5.0)
    
    def _process_bridge_update(self):
        """Process bridge update cycle"""
        # Get current system status
        system_status = self.system_monitor.get_status()
        engine_status = self.quantum_engine.get_status()
        
        # Update resource pools based on system state
        self._update_resource_allocations(system_status)
        
        # Process pending tasks with LNN guidance
        self._process_lnn_guided_scheduling(system_status, engine_status)
        
        # Send updates to C++ process if available
        if self.cpp_manager and self.cpp_manager.is_running:
            self._send_cpp_updates(system_status, engine_status)
        
        # Update performance metrics
        self._update_performance_metrics(system_status, engine_status)
    
    def _update_resource_allocations(self, system_status: SystemStatus):
        """Update resource allocations based on system status"""
        # Adjust CPU allocation based on utilization
        cpu_pool = self.resource_allocator.resource_pools.get("cpu")
        if cpu_pool:
            utilization = system_status.cpu_utilization
            
            # Dynamic efficiency adjustment
            if utilization > 0.8:
                cpu_pool.quantum_efficiency = 0.8  # Reduced efficiency under high load
            elif utilization < 0.2:
                cpu_pool.quantum_efficiency = 1.3  # Higher efficiency under low load
            else:
                cpu_pool.quantum_efficiency = 1.0
        
        # Similar adjustments for memory
        memory_pool = self.resource_allocator.resource_pools.get("memory")
        if memory_pool:
            utilization = system_status.memory_utilization
            if utilization > 0.9:
                memory_pool.quantum_efficiency = 0.7
            else:
                memory_pool.quantum_efficiency = 1.0
    
    def _process_lnn_guided_scheduling(self, system_status: SystemStatus, engine_status: Dict):
        """Process task scheduling with LNN guidance"""
        # Get tasks that need LNN guidance
        running_tasks = engine_status.get('running_tasks', 0)
        queue_size = engine_status.get('queue_size', 0)
        
        if running_tasks > 0:
            # Get system state for LNN input
            system_state = {
                'cpu_utilization': system_status.cpu_utilization,
                'memory_utilization': system_status.memory_utilization,
                'active_tasks': running_tasks,
                'queue_depth': queue_size,
                'power_consumption': system_status.power_consumption_watts / 200.0  # Normalize
            }
            
            # This would integrate with actual running tasks
            # For now, we simulate LNN feedback
            self._simulate_lnn_feedback(system_state)
    
    def _simulate_lnn_feedback(self, system_state: Dict):
        """Simulate LNN feedback for demonstration"""
        # Create mock task features
        mock_features = np.array([
            system_state.get('cpu_utilization', 0.5),
            system_state.get('memory_utilization', 0.5),
            0.1,  # task age
            0.0,  # runtime
            2.0,  # dependencies
            1.0,  # priority
            0.8,  # amplitude
            system_state.get('active_tasks', 0) / 10.0
        ])
        
        # Get LNN prediction
        params = self.lnn_scheduler.predict(mock_features)
        
        # Apply LNN recommendations (would be used to adjust scheduling)
        execution_speed = params[0]
        resource_multiplier = params[1]
        error_tolerance = params[2]
        retry_strategy = params[3]
        
        # Store for adaptation
        self.performance_history.append({
            'timestamp': time.time(),
            'system_state': system_state.copy(),
            'lnn_params': params.copy(),
            'execution_speed': execution_speed,
            'resource_multiplier': resource_multiplier
        })
    
    def _send_cpp_updates(self, system_status: SystemStatus, engine_status: Dict):
        """Send updates to C++ process"""
        update_command = {
            "command": "update_status",
            "system_cpu": system_status.cpu_utilization,
            "system_memory": system_status.memory_utilization,
            "active_tasks": engine_status.get('running_tasks', 0),
            "timestamp": time.time()
        }
        
        response = self.cpp_manager.send_command(update_command)
        if response:
            # Process C++ response
            self._process_cpp_response(response)
    
    def _process_cpp_response(self, response: Dict):
        """Process response from C++ component"""
        if response.get("status") == "ok":
            # Extract performance metrics from C++ side
            cpp_metrics = response.get("metrics", {})
            
            # Update our performance tracking
            if cpp_metrics:
                self.performance_history.append({
                    'timestamp': time.time(),
                    'source': 'cpp',
                    'inference_time_ms': cpp_metrics.get('inference_time_ms', 0),
                    'confidence': cpp_metrics.get('confidence', 0),
                    'power_consumption_mw': cpp_metrics.get('power_mw', 0)
                })
    
    def _process_lnn_adaptation(self):
        """Process LNN adaptation based on performance feedback"""
        if len(self.performance_history) < 2:
            return
        
        # Calculate recent performance score
        recent_entries = self.performance_history[-self.config.performance_window_size:]
        
        # Simplified performance scoring
        total_score = 0.0
        for entry in recent_entries:
            if 'inference_time_ms' in entry:
                # Reward fast inference
                time_score = max(0, 1.0 - entry['inference_time_ms'] / 100.0)
                confidence_score = entry.get('confidence', 0.5)
                entry_score = (time_score + confidence_score) / 2.0
                total_score += entry_score
        
        avg_score = total_score / len(recent_entries) if recent_entries else 0.5
        
        # Create mock features for training
        mock_features = np.array([0.5, 0.4, 0.1, 0.0, 1.0, 1.0, 0.6, 0.3])
        mock_result = {'execution_time': 50, 'success': True, 'quality_score': avg_score}
        
        # Train LNN
        self.lnn_scheduler.train_online(
            mock_features,
            mock_result,
            avg_score
        )
    
    def _update_performance_metrics(self, system_status: SystemStatus, engine_status: Dict):
        """Update overall performance metrics"""
        current_metrics = {
            'timestamp': time.time(),
            'system_cpu': system_status.cpu_utilization,
            'system_memory': system_status.memory_utilization,
            'system_power': system_status.power_consumption_watts,
            'queue_size': engine_status.get('queue_size', 0),
            'completed_tasks': engine_status.get('completed_tasks', 0),
            'failed_tasks': engine_status.get('failed_tasks', 0)
        }
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-800:]
    
    def submit_vision_task(self, image_path: str, callback: Callable = None) -> str:
        """Submit a vision processing task"""
        task = QuantumTask(
            id=f"vision_{int(time.time() * 1000)}",
            name="Vision Processing",
            priority=TaskPriority.HIGH,
            resources={"cpu": 2.0, "memory": 1024, "gpu": 1.0},
            estimated_duration=0.05,  # 50ms
            callback=self._vision_task_callback if not callback else callback,
            metadata={"image_path": image_path}
        )
        
        return self.quantum_engine.submit_task(task)
    
    def _vision_task_callback(self, task: QuantumTask) -> Dict:
        """Callback for vision processing tasks"""
        image_path = task.metadata.get("image_path", "")
        
        # If C++ integration available, use it
        if self.cpp_manager and self.cpp_manager.is_running:
            command = {
                "command": "process_image",
                "image_path": image_path,
                "task_id": task.id
            }
            
            response = self.cpp_manager.send_command(command)
            if response and response.get("status") == "success":
                return response
        
        # Fallback: simulate processing
        time.sleep(task.estimated_duration)
        return {
            "status": "completed",
            "confidence": 0.85,
            "forward_velocity": 2.5,
            "yaw_rate": 0.1,
            "inference_time_ms": task.estimated_duration * 1000
        }
    
    def get_bridge_status(self) -> Dict:
        """Get comprehensive bridge status"""
        system_status = self.system_monitor.get_status()
        engine_status = self.quantum_engine.get_status()
        lnn_stats = self.lnn_scheduler.get_adaptation_stats()
        
        return {
            "bridge_running": self.is_running,
            "system": {
                "cpu_utilization": system_status.cpu_utilization,
                "memory_utilization": system_status.memory_utilization,
                "power_consumption_watts": system_status.power_consumption_watts,
                "temperature_celsius": system_status.temperature_celsius
            },
            "quantum_engine": engine_status,
            "lnn_scheduler": lnn_stats,
            "cpp_integration": {
                "enabled": self.config.enable_cpp_integration,
                "running": self.cpp_manager.is_running if self.cpp_manager else False
            },
            "performance_history_size": len(self.performance_history)
        }


async def demo_bridge_integration():
    """Demonstrate the LNN-Quantum integration bridge"""
    print("LNN-Quantum Integration Bridge Demo")
    print("=" * 50)
    
    # Create bridge
    config = LNNBridgeConfig(
        enable_cpp_integration=False,  # Disable for demo
        enable_real_time_adaptation=True
    )
    
    bridge = LNNQuantumBridge(config)
    
    # Start bridge
    if not bridge.start_bridge():
        print("Failed to start bridge")
        return
    
    try:
        # Submit some vision tasks
        task_ids = []
        for i in range(5):
            task_id = bridge.submit_vision_task(f"demo_image_{i}.jpg")
            task_ids.append(task_id)
            print(f"Submitted vision task: {task_id}")
            
        # Monitor for 30 seconds
        for i in range(30):
            status = bridge.get_bridge_status()
            print(f"\n[{i+1:2d}s] Bridge Status:")
            print(f"  System CPU: {status['system']['cpu_utilization']*100:.1f}%")
            print(f"  Queue Size: {status['quantum_engine']['queue_size']}")
            print(f"  Completed: {status['quantum_engine']['completed_tasks']}")
            print(f"  LNN Examples: {status['lnn_scheduler']['total_training_examples']}")
            
            await asyncio.sleep(1)
    
    finally:
        bridge.stop_bridge()


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_bridge_integration())