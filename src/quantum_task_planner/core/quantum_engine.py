#!/usr/bin/env python3
"""
Quantum-Inspired Task Planning Engine
Leverages quantum computing principles for optimal task scheduling and resource allocation
"""

import numpy as np
import json
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import heapq
from collections import defaultdict
import asyncio


class TaskPriority(Enum):
    """Task priority levels using quantum-inspired states"""
    CRITICAL = 0    # |1⟩ - Highest priority
    HIGH = 1       # |0⟩ + |1⟩ - Superposition
    MEDIUM = 2     # |0⟩ - Standard priority  
    LOW = 3        # |ψ⟩ - Background tasks


class TaskState(Enum):
    """Quantum-inspired task states"""
    PENDING = "pending"        # |0⟩
    RUNNING = "running"        # |1⟩
    SUPERPOSITION = "superposition"  # |0⟩ + |1⟩
    COMPLETED = "completed"    # measured |1⟩
    FAILED = "failed"         # measured |0⟩
    CANCELLED = "cancelled"   # collapsed


@dataclass
class QuantumTask:
    """Quantum-inspired task representation"""
    id: str
    name: str
    priority: TaskPriority
    state: TaskState = TaskState.PENDING
    amplitude: float = 1.0  # Quantum amplitude
    phase: float = 0.0      # Quantum phase
    dependencies: List[str] = field(default_factory=list)
    resources: Dict[str, float] = field(default_factory=dict)
    estimated_duration: float = 0.0
    deadline: Optional[float] = None
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    @property
    def age(self) -> float:
        """Age of task in seconds"""
        return time.time() - self.created_at
    
    @property
    def runtime(self) -> float:
        """Current runtime in seconds"""
        if self.started_at:
            return time.time() - self.started_at
        return 0.0
    
    @property
    def quantum_weight(self) -> float:
        """Compute quantum-inspired priority weight"""
        base_weight = {
            TaskPriority.CRITICAL: 1000,
            TaskPriority.HIGH: 100,
            TaskPriority.MEDIUM: 10,
            TaskPriority.LOW: 1
        }[self.priority]
        
        # Apply quantum amplitude and urgency factors
        urgency_factor = 1.0
        if self.deadline:
            time_left = self.deadline - time.time()
            if time_left > 0:
                urgency_factor = 1.0 / time_left
            else:
                urgency_factor = 1000  # Overdue
        
        age_factor = 1.0 + (self.age / 3600)  # Increase priority with age
        return base_weight * self.amplitude * urgency_factor * age_factor


@dataclass
class ResourcePool:
    """Quantum-inspired resource management"""
    name: str
    total_capacity: float
    available_capacity: float
    allocated: Dict[str, float] = field(default_factory=dict)
    quantum_efficiency: float = 1.0
    
    def can_allocate(self, amount: float) -> bool:
        """Check if resources can be allocated"""
        return self.available_capacity >= amount
    
    def allocate(self, task_id: str, amount: float) -> bool:
        """Allocate resources using quantum-inspired efficiency"""
        if self.can_allocate(amount):
            self.allocated[task_id] = amount
            self.available_capacity -= amount
            return True
        return False
    
    def deallocate(self, task_id: str) -> bool:
        """Release allocated resources"""
        if task_id in self.allocated:
            amount = self.allocated.pop(task_id)
            self.available_capacity += amount
            return True
        return False
    
    @property
    def utilization(self) -> float:
        """Current resource utilization percentage"""
        return (self.total_capacity - self.available_capacity) / self.total_capacity


class QuantumPriorityQueue:
    """Quantum-inspired priority queue with superposition effects"""
    
    def __init__(self):
        self._heap = []
        self._task_finder = {}
        self._removed = set()
        self._lock = threading.Lock()
    
    def add_task(self, task: QuantumTask):
        """Add task to quantum priority queue"""
        with self._lock:
            # Remove existing entry if updating
            if task.id in self._task_finder:
                self._removed.add(task.id)
            
            # Calculate quantum-inspired priority
            priority = -task.quantum_weight  # Negative for min-heap
            entry = [priority, time.time(), task.id, task]
            
            heapq.heappush(self._heap, entry)
            self._task_finder[task.id] = entry
    
    def pop_task(self) -> Optional[QuantumTask]:
        """Pop highest priority task with quantum collapse"""
        with self._lock:
            while self._heap:
                priority, timestamp, task_id, task = heapq.heappop(self._heap)
                
                if task_id not in self._removed:
                    del self._task_finder[task_id]
                    # Quantum collapse - task transitions from superposition to running
                    if task.state == TaskState.SUPERPOSITION:
                        task.state = TaskState.RUNNING
                    return task
                
                self._removed.discard(task_id)
        return None
    
    def remove_task(self, task_id: str):
        """Remove task from queue"""
        with self._lock:
            if task_id in self._task_finder:
                self._removed.add(task_id)
                del self._task_finder[task_id]
    
    def update_priorities(self):
        """Update quantum priorities based on current state"""
        with self._lock:
            # Rebuild heap with updated priorities
            tasks = []
            for priority, timestamp, task_id, task in self._heap:
                if task_id not in self._removed:
                    tasks.append(task)
            
            self._heap.clear()
            self._task_finder.clear()
            self._removed.clear()
            
            for task in tasks:
                self.add_task(task)
    
    def size(self) -> int:
        """Get queue size"""
        with self._lock:
            return len(self._heap) - len(self._removed)
    
    def peek(self) -> Optional[QuantumTask]:
        """Peek at highest priority task without removing"""
        with self._lock:
            while self._heap:
                priority, timestamp, task_id, task = self._heap[0]
                if task_id not in self._removed:
                    return task
                heapq.heappop(self._heap)
                self._removed.discard(task_id)
        return None


class QuantumTaskEngine:
    """Main quantum-inspired task planning and execution engine"""
    
    def __init__(self, max_workers: int = 4, lnn_integration: bool = True):
        self.max_workers = max_workers
        self.lnn_integration = lnn_integration
        
        # Core components
        self.task_queue = QuantumPriorityQueue()
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.resource_pools = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = False
        self._scheduler_thread = None
        self._lock = threading.Lock()
        
        # Metrics
        self.metrics = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'average_wait_time': 0.0,
            'resource_utilization': {}
        }
        
        # Quantum-inspired parameters
        self.coherence_time = 1.0  # Quantum coherence duration
        self.entanglement_threshold = 0.7
        self.superposition_decay = 0.1
        
        # LNN integration
        if lnn_integration:
            self._init_lnn_components()
    
    def _init_lnn_components(self):
        """Initialize Liquid Neural Network components for adaptive scheduling"""
        try:
            from ..core.lnn_integration import LNNScheduler
            self.lnn_scheduler = LNNScheduler(
                input_features=8,  # Task features for scheduling decisions
                hidden_neurons=16,
                output_neurons=4   # Scheduling actions
            )
        except ImportError:
            print("Warning: LNN integration not available, using classical scheduling")
            self.lnn_integration = False
    
    def add_resource_pool(self, name: str, capacity: float, efficiency: float = 1.0):
        """Add quantum resource pool"""
        self.resource_pools[name] = ResourcePool(
            name=name,
            total_capacity=capacity,
            available_capacity=capacity,
            quantum_efficiency=efficiency
        )
    
    def submit_task(self, task: QuantumTask) -> str:
        """Submit task to quantum scheduler"""
        with self._lock:
            # Apply quantum superposition for certain priority levels
            if task.priority in [TaskPriority.HIGH, TaskPriority.MEDIUM]:
                task.state = TaskState.SUPERPOSITION
                task.amplitude = 1.0 / np.sqrt(2)  # Balanced superposition
            
            self.task_queue.add_task(task)
            self.metrics['tasks_submitted'] += 1
            
        return task.id
    
    def start_scheduler(self):
        """Start the quantum task scheduler"""
        if self.is_running:
            return
        
        self.is_running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        print(f"Quantum Task Engine started with {self.max_workers} workers")
    
    def stop_scheduler(self):
        """Stop the quantum task scheduler"""
        self.is_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)
        print("Quantum Task Engine stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop with quantum-inspired decisions"""
        last_priority_update = time.time()
        
        while self.is_running:
            try:
                # Update quantum priorities periodically
                if time.time() - last_priority_update > 1.0:
                    self.task_queue.update_priorities()
                    self._apply_quantum_decoherence()
                    last_priority_update = time.time()
                
                # Get next task from quantum queue
                task = self.task_queue.pop_task()
                if not task:
                    time.sleep(0.1)
                    continue
                
                # Check dependencies
                if not self._check_dependencies(task):
                    # Put back in queue with lower priority
                    task.quantum_weight *= 0.9
                    self.task_queue.add_task(task)
                    continue
                
                # Check resource allocation
                if not self._allocate_resources(task):
                    # Put back in queue
                    self.task_queue.add_task(task)
                    time.sleep(0.5)
                    continue
                
                # Submit task for execution
                if len(self.running_tasks) < self.max_workers:
                    self._execute_task(task)
                else:
                    # Return to queue if no workers available
                    self._deallocate_resources(task)
                    self.task_queue.add_task(task)
                    time.sleep(0.1)
                
            except Exception as e:
                print(f"Scheduler error: {e}")
                time.sleep(1.0)
    
    def _check_dependencies(self, task: QuantumTask) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def _allocate_resources(self, task: QuantumTask) -> bool:
        """Allocate resources using quantum efficiency"""
        for resource_name, amount in task.resources.items():
            if resource_name not in self.resource_pools:
                continue
            
            pool = self.resource_pools[resource_name]
            if not pool.can_allocate(amount):
                # Rollback previous allocations
                self._deallocate_resources(task)
                return False
            
            if not pool.allocate(task.id, amount):
                self._deallocate_resources(task)
                return False
        
        return True
    
    def _deallocate_resources(self, task: QuantumTask):
        """Release allocated resources"""
        for resource_name in task.resources.keys():
            if resource_name in self.resource_pools:
                self.resource_pools[resource_name].deallocate(task.id)
    
    def _execute_task(self, task: QuantumTask):
        """Execute task using quantum-inspired optimization"""
        task.state = TaskState.RUNNING
        task.started_at = time.time()
        
        with self._lock:
            self.running_tasks[task.id] = task
        
        # Submit to thread pool
        future = self.executor.submit(self._run_task, task)
        future.add_done_callback(lambda f: self._task_completed(task, f))
    
    def _run_task(self, task: QuantumTask) -> Any:
        """Run individual task with quantum error correction"""
        try:
            if task.callback:
                # Apply quantum-inspired adaptive execution
                if self.lnn_integration and hasattr(self, 'lnn_scheduler'):
                    execution_params = self._get_lnn_execution_params(task)
                    return task.callback(task, **execution_params)
                else:
                    return task.callback(task)
            else:
                # Default task execution
                time.sleep(task.estimated_duration)
                return {"status": "completed", "result": f"Task {task.name} completed"}
                
        except Exception as e:
            raise Exception(f"Task {task.id} failed: {e}")
    
    def _get_lnn_execution_params(self, task: QuantumTask) -> Dict:
        """Use LNN to determine optimal execution parameters"""
        if not hasattr(self, 'lnn_scheduler'):
            return {}
        
        # Extract task features for LNN
        features = np.array([
            task.quantum_weight,
            task.age,
            task.runtime,
            len(task.dependencies),
            task.priority.value,
            task.amplitude,
            task.phase,
            sum(self.resource_pools[r].utilization for r in task.resources.keys() if r in self.resource_pools) / max(1, len(task.resources))
        ])
        
        # Get LNN recommendations
        recommendations = self.lnn_scheduler.predict(features)
        
        return {
            'execution_speed': float(recommendations[0]),
            'resource_multiplier': float(recommendations[1]),
            'error_tolerance': float(recommendations[2]),
            'retry_strategy': float(recommendations[3])
        }
    
    def _task_completed(self, task: QuantumTask, future):
        """Handle task completion with quantum measurement"""
        try:
            result = future.result()
            task.state = TaskState.COMPLETED
            task.completed_at = time.time()
            
            with self._lock:
                self.running_tasks.pop(task.id, None)
                self.completed_tasks[task.id] = task
                self.metrics['tasks_completed'] += 1
                self.metrics['total_execution_time'] += task.runtime
            
            self._deallocate_resources(task)
            self._update_metrics()
            
        except Exception as e:
            task.state = TaskState.FAILED
            task.retry_count += 1
            
            with self._lock:
                self.running_tasks.pop(task.id, None)
                
                # Quantum retry with exponential backoff
                if task.retry_count <= task.max_retries:
                    # Apply quantum error correction
                    task.amplitude *= 0.8  # Reduce amplitude on failure
                    task.state = TaskState.PENDING
                    
                    # Re-submit with delay
                    retry_delay = 2 ** task.retry_count
                    self.executor.submit(self._delayed_resubmit, task, retry_delay)
                else:
                    self.failed_tasks[task.id] = task
                    self.metrics['tasks_failed'] += 1
            
            self._deallocate_resources(task)
    
    def _delayed_resubmit(self, task: QuantumTask, delay: float):
        """Re-submit failed task after delay"""
        time.sleep(delay)
        self.task_queue.add_task(task)
    
    def _apply_quantum_decoherence(self):
        """Apply quantum decoherence effects to maintain system stability"""
        current_time = time.time()
        
        # Apply decoherence to superposition tasks
        for task_id, task in list(self.running_tasks.items()):
            if task.state == TaskState.SUPERPOSITION:
                coherence_loss = (current_time - task.started_at) / self.coherence_time
                task.amplitude *= np.exp(-coherence_loss * self.superposition_decay)
                
                # Collapse if amplitude too low
                if task.amplitude < 0.1:
                    task.state = TaskState.RUNNING
                    task.amplitude = 1.0
    
    def _update_metrics(self):
        """Update system metrics"""
        # Update resource utilization
        for name, pool in self.resource_pools.items():
            self.metrics['resource_utilization'][name] = pool.utilization
        
        # Update average wait time
        if self.metrics['tasks_completed'] > 0:
            total_wait = sum(
                (task.started_at - task.created_at) 
                for task in self.completed_tasks.values() 
                if task.started_at
            )
            self.metrics['average_wait_time'] = total_wait / self.metrics['tasks_completed']
    
    def get_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'is_running': self.is_running,
            'queue_size': self.task_queue.size(),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'metrics': self.metrics,
            'resource_pools': {
                name: {
                    'total_capacity': pool.total_capacity,
                    'available_capacity': pool.available_capacity,
                    'utilization': pool.utilization,
                    'quantum_efficiency': pool.quantum_efficiency
                } for name, pool in self.resource_pools.items()
            }
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get specific task status"""
        # Check running tasks
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
        elif task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
        elif task_id in self.failed_tasks:
            task = self.failed_tasks[task_id]
        else:
            return None
        
        return {
            'id': task.id,
            'name': task.name,
            'state': task.state.value,
            'priority': task.priority.name,
            'amplitude': task.amplitude,
            'phase': task.phase,
            'age': task.age,
            'runtime': task.runtime,
            'retry_count': task.retry_count,
            'quantum_weight': task.quantum_weight
        }


def create_sample_tasks() -> List[QuantumTask]:
    """Create sample tasks for testing"""
    tasks = []
    
    # Critical system task
    tasks.append(QuantumTask(
        id="task_001",
        name="System Health Check",
        priority=TaskPriority.CRITICAL,
        resources={"cpu": 0.5, "memory": 100},
        estimated_duration=2.0,
        callback=lambda t: {"status": "healthy", "cpu_usage": 0.3}
    ))
    
    # High priority data processing
    tasks.append(QuantumTask(
        id="task_002", 
        name="Data Processing Pipeline",
        priority=TaskPriority.HIGH,
        resources={"cpu": 2.0, "memory": 500},
        estimated_duration=5.0,
        dependencies=["task_001"],
        callback=lambda t: {"processed_records": 1000}
    ))
    
    # Medium priority analytics
    tasks.append(QuantumTask(
        id="task_003",
        name="Analytics Generation",
        priority=TaskPriority.MEDIUM,
        resources={"cpu": 1.0, "memory": 200},
        estimated_duration=3.0,
        deadline=time.time() + 300,  # 5 minutes
        callback=lambda t: {"analytics": "generated"}
    ))
    
    # Low priority cleanup
    tasks.append(QuantumTask(
        id="task_004",
        name="Log Cleanup",
        priority=TaskPriority.LOW,
        resources={"disk": 1.0},
        estimated_duration=1.0,
        callback=lambda t: {"cleaned_files": 50}
    ))
    
    return tasks


if __name__ == "__main__":
    # Example usage
    print("Quantum Task Planning Engine")
    print("=" * 50)
    
    # Create engine
    engine = QuantumTaskEngine(max_workers=2)
    
    # Add resource pools
    engine.add_resource_pool("cpu", capacity=4.0, efficiency=1.2)
    engine.add_resource_pool("memory", capacity=2048, efficiency=1.0)
    engine.add_resource_pool("disk", capacity=10.0, efficiency=0.8)
    
    # Start scheduler
    engine.start_scheduler()
    
    # Submit sample tasks
    sample_tasks = create_sample_tasks()
    for task in sample_tasks:
        task_id = engine.submit_task(task)
        print(f"Submitted task: {task_id}")
    
    # Monitor execution
    try:
        for i in range(30):
            status = engine.get_status()
            print(f"\n[{i+1:2d}] Queue: {status['queue_size']}, "
                  f"Running: {status['running_tasks']}, "
                  f"Completed: {status['completed_tasks']}")
            
            # Print resource utilization
            for name, pool in status['resource_pools'].items():
                print(f"    {name}: {pool['utilization']*100:.1f}% utilized")
            
            time.sleep(1)
            
            # Exit when all tasks complete
            if (status['completed_tasks'] + status['failed_tasks']) == len(sample_tasks):
                break
    
    finally:
        engine.stop_scheduler()
        
        # Final status
        final_status = engine.get_status()
        print(f"\nFinal Status:")
        print(f"Tasks completed: {final_status['completed_tasks']}")
        print(f"Tasks failed: {final_status['failed_tasks']}")
        print(f"Average wait time: {final_status['metrics']['average_wait_time']:.2f}s")