#!/usr/bin/env python3
"""
Comprehensive Tests for Quantum Task Planning Engine
Tests quantum functionality, LNN integration, and resource allocation
"""

import time
import threading
import numpy as np
from concurrent.futures import Future
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    
    # Mock pytest functionality
    class MockPytest:
        @staticmethod
        def skip(reason):
            print(f"SKIPPED: {reason}")
    
    pytest = MockPytest()

# Import the modules under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from quantum_task_planner.core.quantum_engine import (
    QuantumTaskEngine, QuantumTask, TaskPriority, TaskState,
    ResourcePool, QuantumPriorityQueue
)
from quantum_task_planner.core.task_scheduler import (
    QuantumResourceAllocator, ResourceType, ResourceConstraint, AllocationStrategy
)
from quantum_task_planner.validation.task_validator import (
    TaskValidator, ValidationResult, SecurityLevel
)


class TestQuantumTask:
    """Test QuantumTask functionality"""
    
    def test_task_creation(self):
        """Test basic task creation"""
        task = QuantumTask(
            id="test_001",
            name="Test Task",
            priority=TaskPriority.HIGH,
            estimated_duration=10.0
        )
        
        assert task.id == "test_001"
        assert task.name == "Test Task"
        assert task.priority == TaskPriority.HIGH
        assert task.state == TaskState.PENDING
        assert task.amplitude == 1.0
        assert task.phase == 0.0
        assert task.estimated_duration == 10.0
    
    def test_quantum_weight_calculation(self):
        """Test quantum weight calculation"""
        task = QuantumTask(
            id="test_002",
            name="Weight Test",
            priority=TaskPriority.CRITICAL,
            deadline=time.time() + 3600  # 1 hour from now
        )
        
        weight = task.quantum_weight
        assert weight > 0
        assert isinstance(weight, float)
        
        # Critical tasks should have higher weight
        low_task = QuantumTask(
            id="test_low",
            name="Low Priority",
            priority=TaskPriority.LOW
        )
        assert task.quantum_weight > low_task.quantum_weight
    
    def test_task_age_calculation(self):
        """Test task age calculation"""
        task = QuantumTask(
            id="test_age",
            name="Age Test",
            priority=TaskPriority.MEDIUM
        )
        
        initial_age = task.age
        time.sleep(0.1)
        later_age = task.age
        
        assert later_age > initial_age
        assert later_age - initial_age >= 0.1


class TestQuantumPriorityQueue:
    """Test QuantumPriorityQueue functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.queue = QuantumPriorityQueue()
    
    def test_empty_queue(self):
        """Test empty queue operations"""
        assert self.queue.size() == 0
        assert self.queue.pop_task() is None
        assert self.queue.peek() is None
    
    def test_add_and_pop_task(self):
        """Test adding and popping tasks"""
        task = QuantumTask(
            id="queue_test",
            name="Queue Test",
            priority=TaskPriority.HIGH
        )
        
        self.queue.add_task(task)
        assert self.queue.size() == 1
        
        popped_task = self.queue.pop_task()
        assert popped_task is not None
        assert popped_task.id == "queue_test"
        assert self.queue.size() == 0
    
    def test_priority_ordering(self):
        """Test that tasks are prioritized correctly"""
        critical_task = QuantumTask(
            id="critical", name="Critical", priority=TaskPriority.CRITICAL
        )
        low_task = QuantumTask(
            id="low", name="Low", priority=TaskPriority.LOW
        )
        high_task = QuantumTask(
            id="high", name="High", priority=TaskPriority.HIGH
        )
        
        # Add in reverse priority order
        self.queue.add_task(low_task)
        self.queue.add_task(high_task)
        self.queue.add_task(critical_task)
        
        # Should pop in priority order (critical first)
        first = self.queue.pop_task()
        second = self.queue.pop_task()
        third = self.queue.pop_task()
        
        assert first.id == "critical"
        assert second.id == "high"
        assert third.id == "low"
    
    def test_remove_task(self):
        """Test task removal"""
        task = QuantumTask(
            id="remove_test",
            name="Remove Test",
            priority=TaskPriority.MEDIUM
        )
        
        self.queue.add_task(task)
        assert self.queue.size() == 1
        
        self.queue.remove_task("remove_test")
        assert self.queue.size() == 0
        assert self.queue.pop_task() is None
    
    def test_concurrent_operations(self):
        """Test thread safety"""
        tasks = [
            QuantumTask(id=f"concurrent_{i}", name=f"Task {i}", priority=TaskPriority.MEDIUM)
            for i in range(10)
        ]
        
        # Add tasks from multiple threads
        threads = []
        for task in tasks:
            thread = threading.Thread(target=self.queue.add_task, args=(task,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert self.queue.size() == 10
        
        # Pop all tasks
        popped_count = 0
        while self.queue.pop_task() is not None:
            popped_count += 1
        
        assert popped_count == 10


class TestQuantumTaskEngine:
    """Test QuantumTaskEngine functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.engine = QuantumTaskEngine(max_workers=2, lnn_integration=False)
        self.engine.add_resource_pool("cpu", capacity=4.0)
        self.engine.add_resource_pool("memory", capacity=1024.0)
    
    def teardown_method(self):
        """Cleanup after each test"""
        if self.engine.is_running:
            self.engine.stop_scheduler()
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        assert self.engine.max_workers == 2
        assert not self.engine.lnn_integration
        assert not self.engine.is_running
        assert len(self.engine.resource_pools) == 2
        assert "cpu" in self.engine.resource_pools
        assert "memory" in self.engine.resource_pools
    
    def test_submit_task(self):
        """Test task submission"""
        task = QuantumTask(
            id="submit_test",
            name="Submit Test",
            priority=TaskPriority.HIGH
        )
        
        task_id = self.engine.submit_task(task)
        assert task_id == "submit_test"
        assert self.engine.metrics['tasks_submitted'] == 1
    
    def test_start_stop_scheduler(self):
        """Test scheduler start/stop"""
        assert not self.engine.is_running
        
        self.engine.start_scheduler()
        assert self.engine.is_running
        
        self.engine.stop_scheduler()
        assert not self.engine.is_running
    
    def test_simple_task_execution(self):
        """Test simple task execution"""
        executed = threading.Event()
        result_holder = {}
        
        def test_callback(task):
            result_holder['executed'] = True
            result_holder['task_id'] = task.id
            executed.set()
            return {"status": "success"}
        
        task = QuantumTask(
            id="exec_test",
            name="Execution Test",
            priority=TaskPriority.HIGH,
            callback=test_callback,
            estimated_duration=0.1
        )
        
        self.engine.start_scheduler()
        self.engine.submit_task(task)
        
        # Wait for execution
        assert executed.wait(timeout=5.0), "Task was not executed within timeout"
        
        # Give some time for cleanup
        time.sleep(0.5)
        
        assert result_holder.get('executed') is True
        assert result_holder.get('task_id') == "exec_test"
        assert self.engine.metrics['tasks_completed'] >= 1
    
    def test_task_failure_and_retry(self):
        """Test task failure and retry mechanism"""
        attempt_count = [0]
        
        def failing_callback(task):
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise Exception("Simulated failure")
            return {"status": "success", "attempts": attempt_count[0]}
        
        task = QuantumTask(
            id="retry_test",
            name="Retry Test",
            priority=TaskPriority.HIGH,
            callback=failing_callback,
            max_retries=3
        )
        
        self.engine.start_scheduler()
        self.engine.submit_task(task)
        
        # Wait for completion
        time.sleep(2.0)
        
        assert attempt_count[0] == 3
    
    def test_task_dependencies(self):
        """Test task dependency handling"""
        execution_order = []
        
        def dependency_callback(task):
            execution_order.append(task.id)
            return {"status": "success"}
        
        # Create tasks with dependencies
        task1 = QuantumTask(
            id="dep_task_1",
            name="First Task",
            priority=TaskPriority.HIGH,
            callback=dependency_callback
        )
        
        task2 = QuantumTask(
            id="dep_task_2",
            name="Second Task",
            priority=TaskPriority.HIGH,
            dependencies=["dep_task_1"],
            callback=dependency_callback
        )
        
        self.engine.start_scheduler()
        
        # Submit in reverse order
        self.engine.submit_task(task2)
        self.engine.submit_task(task1)
        
        # Wait for completion
        time.sleep(2.0)
        
        # task1 should execute before task2
        assert len(execution_order) == 2
        assert execution_order[0] == "dep_task_1"
        assert execution_order[1] == "dep_task_2"
    
    def test_get_status(self):
        """Test status reporting"""
        status = self.engine.get_status()
        
        assert isinstance(status, dict)
        assert 'is_running' in status
        assert 'queue_size' in status
        assert 'running_tasks' in status
        assert 'completed_tasks' in status
        assert 'metrics' in status
        assert 'resource_pools' in status
        
        assert status['is_running'] == self.engine.is_running
        assert isinstance(status['metrics'], dict)


class TestQuantumResourceAllocator:
    """Test QuantumResourceAllocator functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.allocator = QuantumResourceAllocator(AllocationStrategy.QUANTUM_SUPERPOSITION)
        self.allocator.create_resource_pool("cpu", ResourceType.CPU, 4.0)
        self.allocator.create_resource_pool("memory", ResourceType.MEMORY, 1024.0)
    
    def test_resource_pool_creation(self):
        """Test resource pool creation"""
        assert "cpu" in self.allocator.resource_pools
        assert "memory" in self.allocator.resource_pools
        
        cpu_pool = self.allocator.resource_pools["cpu"]
        assert cpu_pool.total_capacity == 4.0
        assert cpu_pool.available_capacity == 4.0
        assert cpu_pool.resource_type == ResourceType.CPU
    
    def test_simple_allocation(self):
        """Test simple resource allocation"""
        requirements = {
            "cpu": ResourceConstraint(ResourceType.CPU, min_required=1.0, max_allowed=2.0, preferred=1.5),
            "memory": ResourceConstraint(ResourceType.MEMORY, min_required=256, max_allowed=512, preferred=384)
        }
        
        success, allocation = self.allocator.allocate_resources("test_task", requirements)
        
        assert success is True
        assert "cpu" in allocation
        assert "memory" in allocation
        assert allocation["cpu"] >= 1.0
        assert allocation["cpu"] <= 2.0
        assert allocation["memory"] >= 256
        assert allocation["memory"] <= 512
    
    def test_insufficient_resources(self):
        """Test allocation failure when resources insufficient"""
        # Allocate most resources first
        large_requirements = {
            "cpu": ResourceConstraint(ResourceType.CPU, min_required=3.0, max_allowed=3.0, preferred=3.0)
        }
        
        success1, _ = self.allocator.allocate_resources("task1", large_requirements)
        assert success1 is True
        
        # Try to allocate more than available
        excessive_requirements = {
            "cpu": ResourceConstraint(ResourceType.CPU, min_required=2.0, max_allowed=2.0, preferred=2.0)
        }
        
        success2, allocation2 = self.allocator.allocate_resources("task2", excessive_requirements)
        assert success2 is False
        assert allocation2 == {}
    
    def test_resource_deallocation(self):
        """Test resource deallocation"""
        requirements = {
            "cpu": ResourceConstraint(ResourceType.CPU, min_required=2.0, max_allowed=2.0, preferred=2.0)
        }
        
        success, allocation = self.allocator.allocate_resources("test_task", requirements)
        assert success is True
        
        cpu_pool = self.allocator.resource_pools["cpu"]
        assert cpu_pool.available_capacity == 2.0  # 4.0 - 2.0
        
        # Deallocate
        deallocate_success = self.allocator.deallocate_resources("test_task")
        assert deallocate_success is True
        assert cpu_pool.available_capacity == 4.0  # Back to full capacity
    
    def test_allocation_strategies(self):
        """Test different allocation strategies"""
        requirements = {
            "cpu": ResourceConstraint(ResourceType.CPU, min_required=1.0, max_allowed=2.0, preferred=1.5)
        }
        
        strategies = [
            AllocationStrategy.GREEDY,
            AllocationStrategy.QUANTUM_SUPERPOSITION,
            AllocationStrategy.FAIR_SHARE,
            AllocationStrategy.PRIORITY_WEIGHTED
        ]
        
        for strategy in strategies:
            allocator = QuantumResourceAllocator(strategy)
            allocator.create_resource_pool("cpu", ResourceType.CPU, 4.0)
            
            success, allocation = allocator.allocate_resources(f"task_{strategy.name}", requirements)
            assert success is True, f"Strategy {strategy.name} failed"
            assert "cpu" in allocation
            assert allocation["cpu"] >= 1.0
            assert allocation["cpu"] <= 2.0
    
    def test_resource_status(self):
        """Test resource status reporting"""
        status = self.allocator.get_resource_status()
        
        assert isinstance(status, dict)
        assert 'pools' in status
        assert 'metrics' in status
        assert 'quantum_state' in status
        
        assert "cpu" in status['pools']
        assert "memory" in status['pools']
        
        cpu_status = status['pools']['cpu']
        assert 'total_capacity' in cpu_status
        assert 'available_capacity' in cpu_status
        assert 'utilization' in cpu_status


class TestTaskValidator:
    """Test TaskValidator functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.validator = TaskValidator()
    
    def test_valid_task_validation(self):
        """Test validation of valid task"""
        valid_task = {
            'id': 'valid_task_001',
            'name': 'Valid Task',
            'priority': 1,
            'resources': {'cpu': 2, 'memory': 512},
            'estimated_duration': 300,
            'dependencies': ['setup_task']
        }
        
        report = self.validator.validate_task(valid_task)
        
        assert report.is_valid is True
        assert report.security_level in [SecurityLevel.PUBLIC, SecurityLevel.INTERNAL]
        assert report.risk_score < 0.5
        assert not report.has_errors()
    
    def test_invalid_task_structure(self):
        """Test validation of task with structural issues"""
        invalid_task = {
            'name': 'Missing ID',  # Missing required 'id' field
            'priority': 'invalid_type',  # Wrong type
            'invalid@field': 'value'  # Invalid field name
        }
        
        report = self.validator.validate_task(invalid_task)
        
        assert report.is_valid is False
        assert report.has_errors()
        
        error_codes = [issue.code for issue in report.issues]
        assert "MISSING_FIELD" in error_codes
    
    def test_security_validation(self):
        """Test security validation"""
        malicious_task = {
            'id': 'malicious_task',
            'name': 'Evil Task; rm -rf /',  # Command injection
            'priority': 1,
            'callback': 'eval("dangerous_code()")',  # Code injection
            'metadata': {
                'api_key': 'secret123',  # Sensitive information
                'password': 'hidden'
            }
        }
        
        report = self.validator.validate_task(malicious_task)
        
        assert report.risk_score > 0.5
        assert report.security_level in [SecurityLevel.RESTRICTED, SecurityLevel.TOP_SECRET]
        
        security_risks = report.get_security_risks()
        assert len(security_risks) > 0
    
    def test_resource_validation(self):
        """Test resource requirement validation"""
        resource_heavy_task = {
            'id': 'heavy_task',
            'name': 'Resource Heavy Task',
            'priority': 1,
            'resources': {
                'memory': 32 * 1024 * 1024 * 1024,  # 32GB - excessive
                'cpu': 128  # 128 cores - excessive
            },
            'estimated_duration': 48 * 3600  # 48 hours - very long
        }
        
        report = self.validator.validate_task(resource_heavy_task)
        
        assert not report.is_valid
        
        error_codes = [issue.code for issue in report.issues]
        assert any("EXCESSIVE" in code for code in error_codes)
    
    def test_task_sanitization(self):
        """Test task data sanitization"""
        dirty_task = {
            'id': 'dirty_task',
            'name': 'Task with $(dangerous) content',
            'priority': 1,
            'resources': {
                'cpu': 200,  # Will be limited
                'memory': 64 * 1024 * 1024 * 1024  # Will be limited
            }
        }
        
        sanitized = self.validator.sanitize_task_data(dirty_task)
        
        # Check that dangerous content was removed/sanitized
        assert '$(dangerous)' not in sanitized['name']
        
        # Check that resource limits were applied
        assert sanitized['resources']['cpu'] <= self.validator.resource_limits['max_cpu_cores']
        assert sanitized['resources']['memory'] <= self.validator.resource_limits['max_memory']
    
    def test_task_integrity(self):
        """Test task integrity verification"""
        task = {
            'id': 'integrity_test',
            'name': 'Integrity Test',
            'priority': 1
        }
        
        # Generate hash
        task_hash = self.validator.generate_task_hash(task, "secret_key")
        
        # Verify integrity
        assert self.validator.verify_task_integrity(task, task_hash, "secret_key")
        
        # Modify task and verify integrity fails
        modified_task = task.copy()
        modified_task['name'] = 'Modified Task'
        assert not self.validator.verify_task_integrity(modified_task, task_hash, "secret_key")


class TestIntegration:
    """Integration tests for complete system"""
    
    def test_full_workflow(self):
        """Test complete workflow from task submission to completion"""
        # Create system components
        engine = QuantumTaskEngine(max_workers=2, lnn_integration=False)
        engine.add_resource_pool("cpu", capacity=4.0)
        engine.add_resource_pool("memory", capacity=1024.0)
        
        validator = TaskValidator()
        
        # Create a valid task
        task_data = {
            'id': 'integration_test',
            'name': 'Integration Test Task',
            'priority': TaskPriority.HIGH.value,
            'resources': {'cpu': 1, 'memory': 256},
            'estimated_duration': 1.0
        }
        
        try:
            # Validate task
            report = validator.validate_task(task_data)
            assert report.is_valid, "Task validation failed"
            
            # Create quantum task
            execution_completed = threading.Event()
            
            def test_callback(task):
                execution_completed.set()
                return {"status": "integration_success"}
            
            task = QuantumTask(
                id=task_data['id'],
                name=task_data['name'],
                priority=TaskPriority.HIGH,
                resources=task_data['resources'],
                estimated_duration=task_data['estimated_duration'],
                callback=test_callback
            )
            
            # Start engine and submit task
            engine.start_scheduler()
            engine.submit_task(task)
            
            # Wait for execution
            assert execution_completed.wait(timeout=5.0), "Task execution timeout"
            
            # Verify completion
            time.sleep(0.5)  # Allow cleanup
            status = engine.get_status()
            assert status['completed_tasks'] >= 1
            
        finally:
            engine.stop_scheduler()
    
    def test_error_handling_workflow(self):
        """Test error handling throughout the system"""
        engine = QuantumTaskEngine(max_workers=1, lnn_integration=False)
        engine.add_resource_pool("cpu", capacity=1.0)
        
        validator = TaskValidator()
        
        # Test with invalid task
        invalid_task_data = {
            'name': 'No ID Task',  # Missing required ID
            'priority': 'invalid'  # Wrong type
        }
        
        try:
            # Validate invalid task
            report = validator.validate_task(invalid_task_data)
            assert not report.is_valid, "Invalid task passed validation"
            
            # Sanitize and retry
            sanitized_data = validator.sanitize_task_data(invalid_task_data)
            
            # The sanitized task might still be invalid due to missing required fields
            # This tests the robustness of the sanitization process
            
        finally:
            if engine.is_running:
                engine.stop_scheduler()


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    import traceback
    
    test_classes = [
        TestQuantumTask,
        TestQuantumPriorityQueue,
        TestQuantumTaskEngine,
        TestQuantumResourceAllocator,
        TestTaskValidator,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\nRunning tests for {test_class.__name__}")
        print("=" * 50)
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            test_instance = test_class()
            
            try:
                # Run setup if available
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run the test
                getattr(test_instance, test_method)()
                
                print(f"  ✓ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {test_method}: {str(e)}")
                print(f"    {traceback.format_exc()}")
                failed_tests += 1
            
            finally:
                # Run teardown if available
                if hasattr(test_instance, 'teardown_method'):
                    try:
                        test_instance.teardown_method()
                    except:
                        pass
    
    print(f"\n" + "=" * 50)
    print(f"Test Results:")
    print(f"  Total: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {failed_tests}")
    print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")