#!/usr/bin/env python3
"""
Advanced Auto-Scaling System with Quantum-Inspired Load Balancing
Dynamic resource scaling based on workload patterns and predictive analytics
"""

import time
import threading
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import asyncio
import concurrent.futures


class ScalingDirection(Enum):
    """Scaling direction options"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ResourceType(Enum):
    """Types of resources that can be scaled"""
    WORKERS = "workers"
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    QUANTUM_COHERENCE = "quantum_coherence"


@dataclass
class ScalingMetric:
    """Metric used for scaling decisions"""
    name: str
    current_value: float
    target_value: float
    threshold_up: float
    threshold_down: float
    weight: float = 1.0
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_value(self, value: float):
        """Add metric value to history"""
        self.current_value = value
        self.history.append((time.time(), value))
    
    def get_trend(self, window_seconds: float = 300) -> str:
        """Calculate trend over time window"""
        if len(self.history) < 2:
            return "stable"
        
        cutoff_time = time.time() - window_seconds
        recent_values = [val for ts, val in self.history if ts >= cutoff_time]
        
        if len(recent_values) < 2:
            return "stable"
        
        # Linear regression for trend
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"


@dataclass
class ScalingEvent:
    """Record of scaling action"""
    timestamp: float = field(default_factory=time.time)
    resource_type: ResourceType = ResourceType.WORKERS
    direction: ScalingDirection = ScalingDirection.STABLE
    before_value: int = 0
    after_value: int = 0
    trigger_metrics: Dict[str, float] = field(default_factory=dict)
    success: bool = False
    duration: float = 0.0
    cost_impact: float = 0.0


class PredictiveScaler:
    """Predictive scaling using machine learning techniques"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.workload_history = deque(maxlen=history_size)
        self.scaling_history = deque(maxlen=history_size)
        
        # Simple neural network weights (can be replaced with proper ML model)
        self.prediction_weights = np.random.normal(0, 0.1, (8, 4, 1))  # 8->4->1 network
        self.learning_rate = 0.001
    
    def record_workload(self, metrics: Dict[str, float]):
        """Record current workload metrics"""
        workload_vector = self._metrics_to_vector(metrics)
        self.workload_history.append((time.time(), workload_vector))
    
    def predict_future_load(self, horizon_minutes: int = 15) -> Dict[str, float]:
        """Predict future workload"""
        if len(self.workload_history) < 10:
            # Not enough data, return current metrics
            if self.workload_history:
                return self._vector_to_metrics(self.workload_history[-1][1])
            else:
                return {"cpu_usage": 0.5, "memory_usage": 0.5, "task_rate": 10.0}
        
        # Extract recent patterns
        recent_vectors = [vec for _, vec in list(self.workload_history)[-50:]]
        
        # Simple pattern matching and extrapolation
        avg_vector = np.mean(recent_vectors, axis=0)
        trend_vector = np.polyfit(range(len(recent_vectors)), recent_vectors, 1)[0]
        
        # Predict future state
        future_vector = avg_vector + trend_vector * (horizon_minutes / 5.0)  # 5-minute intervals
        future_vector = np.clip(future_vector, 0, 1)  # Keep in valid range
        
        return self._vector_to_metrics(future_vector)
    
    def should_preemptive_scale(self, current_metrics: Dict[str, float]) -> Optional[ScalingDirection]:
        """Determine if preemptive scaling is needed"""
        predicted = self.predict_future_load(10)  # 10 minutes ahead
        
        # Check if predicted load requires scaling
        if predicted.get("cpu_usage", 0) > 0.8 or predicted.get("memory_usage", 0) > 0.8:
            return ScalingDirection.UP
        elif predicted.get("cpu_usage", 0) < 0.3 and predicted.get("memory_usage", 0) < 0.3:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.STABLE
    
    def _metrics_to_vector(self, metrics: Dict[str, float]) -> np.ndarray:
        """Convert metrics dict to numpy vector"""
        return np.array([
            metrics.get("cpu_usage", 0.0),
            metrics.get("memory_usage", 0.0),
            metrics.get("task_rate", 0.0) / 100.0,  # Normalize
            metrics.get("queue_depth", 0.0) / 50.0,  # Normalize
            metrics.get("error_rate", 0.0) / 10.0,   # Normalize
            metrics.get("response_time", 0.0) / 1000.0,  # Normalize
            time.time() % 86400 / 86400,  # Time of day (0-1)
            (time.time() % 604800) / 604800  # Day of week (0-1)
        ])
    
    def _vector_to_metrics(self, vector: np.ndarray) -> Dict[str, float]:
        """Convert numpy vector back to metrics dict"""
        return {
            "cpu_usage": float(vector[0]),
            "memory_usage": float(vector[1]),
            "task_rate": float(vector[2] * 100.0),
            "queue_depth": float(vector[3] * 50.0),
            "error_rate": float(vector[4] * 10.0),
            "response_time": float(vector[5] * 1000.0)
        }


class QuantumLoadBalancer:
    """Quantum-inspired load balancing with superposition principles"""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.worker_loads = np.zeros(num_workers)
        self.worker_capabilities = np.ones(num_workers)  # Can vary by worker
        self.quantum_state = np.ones(num_workers) / np.sqrt(num_workers)  # Equal superposition
        
        # Quantum parameters
        self.coherence_decay = 0.95
        self.measurement_strength = 0.1
    
    def assign_task(self, task_weight: float = 1.0) -> int:
        """Assign task to worker using quantum load balancing"""
        # Update quantum state based on current loads
        load_factors = 1.0 / (1.0 + self.worker_loads)  # Inverse of load
        capability_factors = self.worker_capabilities
        
        # Quantum interference patterns
        interference = load_factors * capability_factors
        
        # Apply quantum measurement
        probabilities = np.abs(self.quantum_state * interference) ** 2
        probabilities /= np.sum(probabilities)  # Normalize
        
        # Select worker based on quantum probabilities
        selected_worker = np.random.choice(self.num_workers, p=probabilities)
        
        # Update worker load
        self.worker_loads[selected_worker] += task_weight
        
        # Apply quantum decoherence
        self.quantum_state *= self.coherence_decay
        self.quantum_state[selected_worker] *= (1.0 + self.measurement_strength)
        
        # Renormalize quantum state
        norm = np.linalg.norm(self.quantum_state)
        if norm > 0:
            self.quantum_state /= norm
        
        return selected_worker
    
    def complete_task(self, worker_id: int, task_weight: float = 1.0):
        """Mark task as completed and update worker load"""
        if 0 <= worker_id < self.num_workers:
            self.worker_loads[worker_id] = max(0, self.worker_loads[worker_id] - task_weight)
    
    def get_load_distribution(self) -> Dict[int, float]:
        """Get current load distribution across workers"""
        return {i: load for i, load in enumerate(self.worker_loads)}
    
    def rebalance(self):
        """Rebalance loads across workers"""
        total_load = np.sum(self.worker_loads)
        target_load = total_load / self.num_workers
        
        # Gradually move towards balanced state
        rebalance_rate = 0.1
        self.worker_loads = (1 - rebalance_rate) * self.worker_loads + rebalance_rate * target_load
    
    def add_worker(self, capability: float = 1.0):
        """Add new worker to the pool"""
        self.num_workers += 1
        self.worker_loads = np.append(self.worker_loads, 0.0)
        self.worker_capabilities = np.append(self.worker_capabilities, capability)
        
        # Extend quantum state
        new_state = np.zeros(self.num_workers)
        new_state[:-1] = self.quantum_state * np.sqrt((self.num_workers - 1) / self.num_workers)
        new_state[-1] = 1.0 / np.sqrt(self.num_workers)
        self.quantum_state = new_state
    
    def remove_worker(self, worker_id: int) -> bool:
        """Remove worker from the pool"""
        if worker_id < 0 or worker_id >= self.num_workers or self.num_workers <= 1:
            return False
        
        # Redistribute load of removed worker
        removed_load = self.worker_loads[worker_id]
        remaining_workers = self.num_workers - 1
        
        if remaining_workers > 0:
            load_per_worker = removed_load / remaining_workers
            
            # Remove worker from arrays
            self.worker_loads = np.delete(self.worker_loads, worker_id)
            self.worker_capabilities = np.delete(self.worker_capabilities, worker_id)
            self.worker_loads += load_per_worker
            
            # Update quantum state
            self.quantum_state = np.delete(self.quantum_state, worker_id)
            norm = np.linalg.norm(self.quantum_state)
            if norm > 0:
                self.quantum_state /= norm
            
            self.num_workers = remaining_workers
            return True
        
        return False


class AutoScaler:
    """Advanced auto-scaling system with quantum load balancing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Scaling configuration
        self.min_workers = self.config.get("min_workers", 1)
        self.max_workers = self.config.get("max_workers", 16)
        self.current_workers = self.config.get("initial_workers", 4)
        self.scale_up_cooldown = self.config.get("scale_up_cooldown", 60.0)  # seconds
        self.scale_down_cooldown = self.config.get("scale_down_cooldown", 300.0)  # seconds
        
        # Metrics and thresholds
        self.metrics = {
            "cpu_usage": ScalingMetric("cpu_usage", 0.0, 0.6, 0.8, 0.3, weight=1.0),
            "memory_usage": ScalingMetric("memory_usage", 0.0, 0.6, 0.8, 0.3, weight=1.0),
            "task_rate": ScalingMetric("task_rate", 0.0, 20.0, 40.0, 5.0, weight=0.8),
            "queue_depth": ScalingMetric("queue_depth", 0.0, 10.0, 20.0, 2.0, weight=0.9),
            "response_time": ScalingMetric("response_time", 0.0, 100.0, 200.0, 50.0, weight=0.7),
            "error_rate": ScalingMetric("error_rate", 0.0, 1.0, 5.0, 0.1, weight=1.2)
        }
        
        # Components
        self.predictor = PredictiveScaler()
        self.load_balancer = QuantumLoadBalancer(self.current_workers)
        
        # State tracking
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        self.scaling_history = deque(maxlen=1000)
        self.is_scaling = False
        
        # Callbacks
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None
        
        # Control
        self.is_monitoring = False
        self.monitor_thread = None
        
    def set_scale_callbacks(self, scale_up: Callable[[int], bool], scale_down: Callable[[int], bool]):
        """Set callbacks for scaling actions"""
        self.scale_up_callback = scale_up
        self.scale_down_callback = scale_down
    
    def start_monitoring(self):
        """Start auto-scaling monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"Auto-scaler started monitoring (workers: {self.current_workers})")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring"""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Auto-scaler stopped monitoring")
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update scaling metrics"""
        for name, value in metrics.items():
            if name in self.metrics:
                self.metrics[name].add_value(value)
        
        # Record for predictive scaling
        self.predictor.record_workload(metrics)
    
    def _monitoring_loop(self):
        """Main monitoring and scaling loop"""
        while self.is_monitoring:
            try:
                # Check if scaling is needed
                scaling_decision = self._make_scaling_decision()
                
                if scaling_decision != ScalingDirection.STABLE and not self.is_scaling:
                    self._execute_scaling_decision(scaling_decision)
                
                # Rebalance load across workers
                self.load_balancer.rebalance()
                
                # Log status periodically
                if int(time.time()) % 60 == 0:  # Every minute
                    self._log_status()
                
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Auto-scaler monitoring error: {e}")
                time.sleep(30.0)
    
    def _make_scaling_decision(self) -> ScalingDirection:
        """Make scaling decision based on metrics and predictions"""
        current_time = time.time()
        
        # Check cooldown periods
        if (current_time - self.last_scale_up) < self.scale_up_cooldown:
            return ScalingDirection.STABLE
        
        if (current_time - self.last_scale_down) < self.scale_down_cooldown:
            return ScalingDirection.STABLE
        
        # Calculate weighted scaling score
        scale_up_score = 0.0
        scale_down_score = 0.0
        
        for metric in self.metrics.values():
            if metric.current_value > metric.threshold_up:
                scale_up_score += metric.weight * (metric.current_value - metric.threshold_up) / metric.threshold_up
            elif metric.current_value < metric.threshold_down:
                scale_down_score += metric.weight * (metric.threshold_down - metric.current_value) / metric.threshold_down
        
        # Consider predictive scaling
        predictive_decision = self.predictor.should_preemptive_scale({
            name: metric.current_value for name, metric in self.metrics.items()
        })
        
        if predictive_decision == ScalingDirection.UP:
            scale_up_score += 0.5  # Boost for predictive scaling
        elif predictive_decision == ScalingDirection.DOWN:
            scale_down_score += 0.3
        
        # Consider trends
        for metric in self.metrics.values():
            trend = metric.get_trend()
            if trend == "increasing" and metric.current_value > metric.target_value:
                scale_up_score += 0.2 * metric.weight
            elif trend == "decreasing" and metric.current_value < metric.target_value:
                scale_down_score += 0.1 * metric.weight
        
        # Make decision
        decision_threshold = 1.0
        
        if scale_up_score > decision_threshold and self.current_workers < self.max_workers:
            return ScalingDirection.UP
        elif scale_down_score > decision_threshold and self.current_workers > self.min_workers:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.STABLE
    
    def _execute_scaling_decision(self, direction: ScalingDirection):
        """Execute scaling decision"""
        if self.is_scaling:
            return
        
        self.is_scaling = True
        start_time = time.time()
        
        try:
            before_workers = self.current_workers
            success = False
            
            if direction == ScalingDirection.UP:
                target_workers = min(self.current_workers + self._calculate_scale_amount(True), self.max_workers)
                success = self._scale_up(target_workers)
                if success:
                    self.last_scale_up = time.time()
                    
            elif direction == ScalingDirection.DOWN:
                target_workers = max(self.current_workers - self._calculate_scale_amount(False), self.min_workers)
                success = self._scale_down(target_workers)
                if success:
                    self.last_scale_down = time.time()
            
            # Record scaling event
            scaling_event = ScalingEvent(
                resource_type=ResourceType.WORKERS,
                direction=direction,
                before_value=before_workers,
                after_value=self.current_workers,
                trigger_metrics={name: metric.current_value for name, metric in self.metrics.items()},
                success=success,
                duration=time.time() - start_time
            )
            
            self.scaling_history.append(scaling_event)
            
            if success:
                self.logger.info(f"Scaling {direction.value}: {before_workers} -> {self.current_workers} workers")
            else:
                self.logger.warning(f"Scaling {direction.value} failed: {before_workers} workers (unchanged)")
        
        finally:
            self.is_scaling = False
    
    def _calculate_scale_amount(self, scale_up: bool) -> int:
        """Calculate how many workers to add or remove"""
        if scale_up:
            # Aggressive scaling up for high urgency metrics
            urgency_metrics = ["error_rate", "response_time"]
            high_urgency = any(
                self.metrics[name].current_value > self.metrics[name].threshold_up * 1.5
                for name in urgency_metrics if name in self.metrics
            )
            
            if high_urgency:
                return max(2, self.current_workers // 2)  # Scale up by 50% or at least 2
            else:
                return max(1, self.current_workers // 4)  # Scale up by 25% or at least 1
        else:
            # Conservative scaling down
            return 1  # Remove one worker at a time
    
    def _scale_up(self, target_workers: int) -> bool:
        """Scale up to target number of workers"""
        if self.scale_up_callback:
            success = self.scale_up_callback(target_workers)
            if success:
                # Add workers to load balancer
                while self.load_balancer.num_workers < target_workers:
                    self.load_balancer.add_worker()
                self.current_workers = target_workers
            return success
        else:
            # Simulate scaling without callback
            self.current_workers = target_workers
            while self.load_balancer.num_workers < target_workers:
                self.load_balancer.add_worker()
            return True
    
    def _scale_down(self, target_workers: int) -> bool:
        """Scale down to target number of workers"""
        if self.scale_down_callback:
            success = self.scale_down_callback(target_workers)
            if success:
                # Remove workers from load balancer
                while self.load_balancer.num_workers > target_workers:
                    self.load_balancer.remove_worker(self.load_balancer.num_workers - 1)
                self.current_workers = target_workers
            return success
        else:
            # Simulate scaling without callback
            self.current_workers = target_workers
            while self.load_balancer.num_workers > target_workers:
                self.load_balancer.remove_worker(self.load_balancer.num_workers - 1)
            return True
    
    def _log_status(self):
        """Log current auto-scaler status"""
        current_metrics = {name: metric.current_value for name, metric in self.metrics.items()}
        load_distribution = self.load_balancer.get_load_distribution()
        
        self.logger.info(
            f"Auto-scaler status: {self.current_workers} workers, "
            f"CPU: {current_metrics.get('cpu_usage', 0):.1f}%, "
            f"Memory: {current_metrics.get('memory_usage', 0):.1f}%, "
            f"Queue: {current_metrics.get('queue_depth', 0):.0f}, "
            f"Load balance: {len(load_distribution)} workers"
        )
    
    def assign_task_to_worker(self, task_weight: float = 1.0) -> int:
        """Assign task to optimal worker using quantum load balancing"""
        return self.load_balancer.assign_task(task_weight)
    
    def complete_task_on_worker(self, worker_id: int, task_weight: float = 1.0):
        """Mark task as completed on specific worker"""
        self.load_balancer.complete_task(worker_id, task_weight)
    
    def get_scaling_report(self) -> Dict[str, Any]:
        """Get comprehensive scaling report"""
        recent_events = list(self.scaling_history)[-10:]  # Last 10 events
        
        return {
            "timestamp": time.time(),
            "current_workers": self.current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "is_scaling": self.is_scaling,
            "last_scale_up": self.last_scale_up,
            "last_scale_down": self.last_scale_down,
            "current_metrics": {
                name: {
                    "value": metric.current_value,
                    "target": metric.target_value,
                    "trend": metric.get_trend()
                }
                for name, metric in self.metrics.items()
            },
            "load_distribution": self.load_balancer.get_load_distribution(),
            "recent_scaling_events": [
                {
                    "timestamp": event.timestamp,
                    "direction": event.direction.value,
                    "before": event.before_value,
                    "after": event.after_value,
                    "success": event.success,
                    "duration": event.duration
                }
                for event in recent_events
            ],
            "scaling_efficiency": self._calculate_scaling_efficiency(),
            "predictive_insights": self.predictor.predict_future_load(30)  # 30 minutes ahead
        }
    
    def _calculate_scaling_efficiency(self) -> float:
        """Calculate scaling efficiency based on historical data"""
        if not self.scaling_history:
            return 1.0
        
        successful_events = [event for event in self.scaling_history if event.success]
        if not successful_events:
            return 0.0
        
        efficiency_scores = []
        
        for event in successful_events:
            # Calculate efficiency based on metrics improvement after scaling
            base_score = 0.8 if event.success else 0.0
            
            # Bonus for quick scaling
            if event.duration < 30.0:
                base_score += 0.1
            
            # Penalty for excessive scaling
            time_since = time.time() - event.timestamp
            if time_since < 300:  # Recent event
                recent_events = [e for e in self.scaling_history 
                               if time.time() - e.timestamp < 300 and e.timestamp != event.timestamp]
                if len(recent_events) > 2:  # Too many recent scaling events
                    base_score -= 0.2
            
            efficiency_scores.append(max(0.0, min(1.0, base_score)))
        
        return np.mean(efficiency_scores) if efficiency_scores else 0.0


def demo_auto_scaler():
    """Demonstrate auto-scaling capabilities"""
    print("⚡ Auto-Scaling System Demo")
    print("=" * 50)
    
    # Create auto-scaler
    config = {
        "min_workers": 2,
        "max_workers": 12,
        "initial_workers": 4,
        "scale_up_cooldown": 10.0,    # Shorter for demo
        "scale_down_cooldown": 20.0   # Shorter for demo
    }
    
    scaler = AutoScaler(config)
    
    # Set up scaling callbacks
    def scale_up_callback(target_workers: int) -> bool:
        print(f"🔼 Scaling UP to {target_workers} workers")
        return True  # Simulate successful scaling
    
    def scale_down_callback(target_workers: int) -> bool:
        print(f"🔽 Scaling DOWN to {target_workers} workers")
        return True  # Simulate successful scaling
    
    scaler.set_scale_callbacks(scale_up_callback, scale_down_callback)
    
    # Start monitoring
    scaler.start_monitoring()
    
    try:
        print("✓ Auto-scaler started, simulating load patterns...")
        
        # Simulate different load patterns
        scenarios = [
            # Normal load
            {"cpu_usage": 0.4, "memory_usage": 0.3, "task_rate": 15.0, "queue_depth": 5.0, "response_time": 80.0, "error_rate": 0.1},
            # Increasing load
            {"cpu_usage": 0.6, "memory_usage": 0.5, "task_rate": 25.0, "queue_depth": 10.0, "response_time": 120.0, "error_rate": 0.3},
            # High load - should trigger scale up
            {"cpu_usage": 0.85, "memory_usage": 0.8, "task_rate": 45.0, "queue_depth": 25.0, "response_time": 250.0, "error_rate": 2.0},
            # Spike in errors - should trigger immediate scale up
            {"cpu_usage": 0.9, "memory_usage": 0.85, "task_rate": 50.0, "queue_depth": 35.0, "response_time": 400.0, "error_rate": 8.0},
            # Decreasing load
            {"cpu_usage": 0.6, "memory_usage": 0.5, "task_rate": 20.0, "queue_depth": 8.0, "response_time": 150.0, "error_rate": 1.0},
            # Low load - should trigger scale down
            {"cpu_usage": 0.2, "memory_usage": 0.2, "task_rate": 5.0, "queue_depth": 1.0, "response_time": 40.0, "error_rate": 0.05}
        ]
        
        for i, scenario in enumerate(scenarios):
            print(f"\n📊 Scenario {i+1}: {scenario}")
            
            # Update metrics multiple times to establish pattern
            for _ in range(3):
                scaler.update_metrics(scenario)
                time.sleep(2)
            
            # Get scaling report
            report = scaler.get_scaling_report()
            
            print(f"   Workers: {report['current_workers']}")
            print(f"   Scaling active: {report['is_scaling']}")
            print(f"   Load distribution: {list(report['load_distribution'].values())}")
            
            # Test task assignment
            worker_assignments = []
            for j in range(5):
                worker_id = scaler.assign_task_to_worker(task_weight=1.0)
                worker_assignments.append(worker_id)
            
            print(f"   Task assignments: {worker_assignments}")
            
            # Complete tasks
            for worker_id in worker_assignments:
                scaler.complete_task_on_worker(worker_id, task_weight=1.0)
            
            time.sleep(5)  # Wait for scaling decisions
        
        # Final report
        final_report = scaler.get_scaling_report()
        
        print(f"\n🎯 Final Auto-Scaling Report:")
        print(f"   Final workers: {final_report['current_workers']}")
        print(f"   Scaling efficiency: {final_report['scaling_efficiency']:.3f}")
        print(f"   Recent scaling events: {len(final_report['recent_scaling_events'])}")
        
        print(f"   Predictive insights (30min ahead):")
        predictions = final_report['predictive_insights']
        for metric, value in predictions.items():
            print(f"     {metric}: {value:.2f}")
        
    finally:
        scaler.stop_monitoring()
        print("\n✓ Auto-scaler demo completed")


if __name__ == "__main__":
    demo_auto_scaler()