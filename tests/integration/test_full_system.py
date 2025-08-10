#!/usr/bin/env python3
"""
Comprehensive Full System Integration Tests
Testing the complete Liquid AI Vision Kit with Quantum Task Planning
"""

import os
import sys
import pytest
import asyncio
import time
import numpy as np
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from quantum_task_planner.core.quantum_engine import QuantumTaskEngine, QuantumTask, TaskPriority
from quantum_task_planner.core.lnn_integration import LNNScheduler
from quantum_task_planner.integration_bridge import LNNQuantumBridge, LNNBridgeConfig
from quantum_task_planner.reliability.error_recovery import ErrorRecoverySystem, SystemComponent, ErrorSeverity
from quantum_task_planner.monitoring.comprehensive_monitor import ComprehensiveMonitor


class TestSystemIntegration:
    """Comprehensive system integration test suite"""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test"""
        self.test_components = []
        yield
        # Cleanup components
        for component in self.test_components:
            if hasattr(component, 'stop_scheduler'):
                component.stop_scheduler()
            elif hasattr(component, 'stop_monitoring'):
                component.stop_monitoring()
            elif hasattr(component, 'stop_bridge'):
                component.stop_bridge()
    
    def test_quantum_engine_basic_functionality(self):
        """Test basic quantum task engine functionality"""
        engine = QuantumTaskEngine(max_workers=2, lnn_integration=False)
        self.test_components.append(engine)
        
        # Add resource pools
        engine.add_resource_pool("cpu", capacity=4.0, efficiency=1.0)
        engine.add_resource_pool("memory", capacity=2048, efficiency=1.0)
        
        # Start engine
        engine.start_scheduler()
        
        # Create test task
        task_executed = False
        
        def test_callback(task):
            nonlocal task_executed
            task_executed = True
            return {"status": "completed", "result": "test_passed"}
        
        task = QuantumTask(
            id="test_001",
            name="Basic Test Task",
            priority=TaskPriority.HIGH,
            resources={"cpu": 1.0, "memory": 512},
            estimated_duration=0.1,
            callback=test_callback
        )
        
        # Submit task
        task_id = engine.submit_task(task)
        assert task_id == "test_001"
        
        # Wait for execution
        timeout = 10.0
        start_time = time.time()
        while not task_executed and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        assert task_executed, "Task was not executed within timeout"
        
        # Check engine status
        status = engine.get_status()
        assert status['completed_tasks'] >= 1
        assert status['is_running'] is True
    
    def test_lnn_scheduler_functionality(self):
        """Test LNN scheduler functionality"""
        scheduler = LNNScheduler(
            input_features=8,
            hidden_neurons=16,
            output_neurons=4
        )
        
        # Test prediction
        features = np.array([0.5, 0.3, 0.1, 0.0, 1.0, 0.8, 0.6, 0.4])
        params = scheduler.predict(features)
        
        assert len(params) == 4
        assert all(isinstance(p, (int, float, np.number)) for p in params)
        
        # Test online training
        mock_result = {
            'execution_time': 1.0,
            'resource_usage': 0.8,
            'success': True,
            'quality_score': 0.9
        }
        
        from quantum_task_planner.core.lnn_integration import calculate_performance_score
        score = calculate_performance_score(None, mock_result, {'target_time': 1.5, 'resource_limit': 1.0})
        
        scheduler.train_online(features, mock_result, score)
        
        # Check adaptation stats
        stats = scheduler.get_adaptation_stats()
        assert stats['total_training_examples'] >= 1
        assert 'recent_performance' in stats
    
    def test_integration_bridge_functionality(self):
        """Test LNN-Quantum integration bridge"""
        config = LNNBridgeConfig(
            enable_cpp_integration=False,  # Disable for testing
            enable_real_time_adaptation=False,  # Simplified for testing
            update_interval_ms=500
        )
        
        bridge = LNNQuantumBridge(config)
        self.test_components.append(bridge)
        
        # Start bridge
        assert bridge.start_bridge(), "Bridge failed to start"
        
        # Submit a vision task
        task_id = bridge.submit_vision_task("test_image.jpg")
        assert task_id is not None
        
        # Wait for processing
        timeout = 10.0
        start_time = time.time()
        initial_completed = 0
        
        while (time.time() - start_time) < timeout:
            status = bridge.get_bridge_status()
            completed = status['quantum_engine']['completed_tasks']
            if completed > initial_completed:
                break
            time.sleep(0.1)
        
        # Check final status
        final_status = bridge.get_bridge_status()
        assert final_status['bridge_running'] is True
        assert final_status['quantum_engine']['completed_tasks'] > 0
    
    def test_error_recovery_system(self):
        """Test error recovery system functionality"""
        recovery_system = ErrorRecoverySystem()
        self.test_components.append(recovery_system)
        
        recovery_system.start_monitoring()
        
        # Report test errors
        test_exception = ValueError("Test error for recovery system")
        error_id = recovery_system.report_error(
            SystemComponent.QUANTUM_ENGINE,
            test_exception,
            {"test_context": "integration_test"}
        )
        
        assert error_id is not None
        
        # Wait for recovery attempt
        time.sleep(2)
        
        # Get health report
        health_report = recovery_system.get_system_health_report()
        
        assert 'overall_health_score' in health_report
        assert 'error_statistics' in health_report
        assert health_report['error_statistics']['total_errors'] >= 1
        
        # Test manual recovery
        recovery_success = recovery_system.attempt_recovery(error_id)
        assert isinstance(recovery_success, bool)
    
    def test_comprehensive_monitoring(self):
        """Test comprehensive monitoring system"""
        monitor = ComprehensiveMonitor()
        self.test_components.append(monitor)
        
        monitor.start_monitoring()
        
        # Add test metrics
        monitor.record_metric("test_metric", 42.0, {"source": "integration_test"})
        monitor.record_metric("cpu_usage_percent", 25.0)
        monitor.record_metric("memory_usage_percent", 30.0)
        
        # Wait for monitoring cycle
        time.sleep(2)
        
        # Get dashboard
        dashboard = monitor.get_monitoring_dashboard()
        
        assert 'timestamp' in dashboard
        assert 'system_overview' in dashboard
        assert 'metrics_summary' in dashboard
        assert 'health_status' in dashboard
        
        # Check system overview
        system_overview = dashboard['system_overview']
        assert 'cpu_percent' in system_overview
        assert 'memory_percent' in system_overview
        
        # Test metric data retrieval
        metric_data = monitor.get_metric_data("cpu_usage_percent")
        assert 'statistics' in metric_data
        assert metric_data['statistics']['count'] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self):
        """Test concurrent task processing under load"""
        engine = QuantumTaskEngine(max_workers=4, lnn_integration=False)
        self.test_components.append(engine)
        
        # Add resource pools
        engine.add_resource_pool("cpu", capacity=8.0, efficiency=1.0)
        engine.add_resource_pool("memory", capacity=4096, efficiency=1.0)
        
        engine.start_scheduler()
        
        # Submit multiple concurrent tasks
        completed_tasks = []
        
        def task_callback(task):
            # Simulate work
            time.sleep(np.random.uniform(0.1, 0.3))
            completed_tasks.append(task.id)
            return {"status": "completed", "task_id": task.id}
        
        task_count = 20
        submitted_tasks = []
        
        for i in range(task_count):
            task = QuantumTask(
                id=f"concurrent_task_{i:03d}",
                name=f"Concurrent Task {i}",
                priority=TaskPriority.HIGH if i % 3 == 0 else TaskPriority.MEDIUM,
                resources={"cpu": np.random.uniform(0.5, 2.0), "memory": np.random.uniform(256, 1024)},
                estimated_duration=np.random.uniform(0.1, 0.5),
                callback=task_callback
            )
            
            task_id = engine.submit_task(task)
            submitted_tasks.append(task_id)
        
        # Wait for all tasks to complete
        timeout = 30.0
        start_time = time.time()
        
        while len(completed_tasks) < task_count and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        # Verify results
        assert len(completed_tasks) == task_count, f"Only {len(completed_tasks)}/{task_count} tasks completed"
        
        # Check engine final status
        final_status = engine.get_status()
        assert final_status['completed_tasks'] >= task_count
        assert final_status['failed_tasks'] == 0
    
    def test_resource_allocation_stress(self):
        """Test resource allocation under stress"""
        engine = QuantumTaskEngine(max_workers=3, lnn_integration=False)
        self.test_components.append(engine)
        
        # Add limited resource pools
        engine.add_resource_pool("cpu", capacity=2.0, efficiency=1.0)  # Limited CPU
        engine.add_resource_pool("memory", capacity=1024, efficiency=1.0)  # Limited memory
        
        engine.start_scheduler()
        
        # Submit resource-intensive tasks
        completed_count = 0
        failed_count = 0
        
        def resource_task_callback(task):
            nonlocal completed_count
            # Simulate resource-intensive work
            time.sleep(0.2)
            completed_count += 1
            return {"status": "completed"}
        
        # Submit tasks that collectively require more resources than available
        for i in range(15):
            task = QuantumTask(
                id=f"resource_task_{i:03d}",
                name=f"Resource Task {i}",
                priority=TaskPriority.MEDIUM,
                resources={"cpu": 0.8, "memory": 512},  # Each task requires significant resources
                estimated_duration=0.2,
                callback=resource_task_callback
            )
            
            engine.submit_task(task)
        
        # Wait for processing
        timeout = 20.0
        start_time = time.time()
        
        while (completed_count + failed_count) < 15 and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        # Verify that tasks were processed (some may have been queued and completed serially)
        final_status = engine.get_status()
        total_processed = final_status['completed_tasks'] + final_status['failed_tasks']
        
        assert total_processed > 0, "No tasks were processed"
        assert completed_count > 0, "No tasks completed successfully"
    
    def test_system_resilience_under_errors(self):
        """Test system resilience when components fail"""
        # Create system with error recovery
        recovery_system = ErrorRecoverySystem()
        self.test_components.append(recovery_system)
        
        engine = QuantumTaskEngine(max_workers=2, lnn_integration=False)
        self.test_components.append(engine)
        
        recovery_system.start_monitoring()
        engine.start_scheduler()
        
        engine.add_resource_pool("cpu", capacity=4.0, efficiency=1.0)
        
        # Submit tasks, some of which will fail
        successful_tasks = 0
        failed_tasks = 0
        
        def sometimes_failing_callback(task):
            nonlocal successful_tasks, failed_tasks
            
            # Simulate random failures
            if np.random.random() < 0.3:  # 30% failure rate
                failed_tasks += 1
                # Report error to recovery system
                error = RuntimeError(f"Simulated failure in task {task.id}")
                recovery_system.report_error(SystemComponent.QUANTUM_ENGINE, error)
                raise error
            else:
                successful_tasks += 1
                return {"status": "completed"}
        
        # Submit tasks
        task_count = 20
        for i in range(task_count):
            task = QuantumTask(
                id=f"resilience_task_{i:03d}",
                name=f"Resilience Task {i}",
                priority=TaskPriority.MEDIUM,
                resources={"cpu": 1.0},
                estimated_duration=0.1,
                callback=sometimes_failing_callback,
                max_retries=2  # Allow retries
            )
            
            engine.submit_task(task)
        
        # Wait for processing
        timeout = 30.0
        start_time = time.time()
        
        while (successful_tasks + failed_tasks) < task_count and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        # Get health report
        health_report = recovery_system.get_system_health_report()
        
        # Verify system handled errors gracefully
        assert successful_tasks > 0, "No tasks completed successfully"
        assert health_report['error_statistics']['total_errors'] > 0, "No errors were reported"
        assert health_report['overall_health_score'] >= 0.0, "Health score is invalid"
        
        # System should still be running despite errors
        final_status = engine.get_status()
        assert final_status['is_running'] is True
    
    def test_performance_benchmarks(self):
        """Test system performance benchmarks"""
        engine = QuantumTaskEngine(max_workers=4, lnn_integration=False)
        self.test_components.append(engine)
        
        # Configure for performance testing
        engine.add_resource_pool("cpu", capacity=8.0, efficiency=1.2)
        engine.add_resource_pool("memory", capacity=8192, efficiency=1.0)
        
        engine.start_scheduler()
        
        # Benchmark task throughput
        completed_tasks = []
        
        def benchmark_callback(task):
            completed_tasks.append((task.id, time.time()))
            return {"status": "completed"}
        
        # Submit batch of lightweight tasks
        task_count = 100
        start_time = time.time()
        
        for i in range(task_count):
            task = QuantumTask(
                id=f"benchmark_task_{i:04d}",
                name=f"Benchmark Task {i}",
                priority=TaskPriority.MEDIUM,
                resources={"cpu": 0.1, "memory": 64},
                estimated_duration=0.01,  # Very fast tasks
                callback=benchmark_callback
            )
            
            engine.submit_task(task)
        
        # Wait for all tasks to complete
        timeout = 30.0
        while len(completed_tasks) < task_count and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate performance metrics
        throughput = len(completed_tasks) / total_time
        
        # Performance assertions
        assert len(completed_tasks) == task_count, f"Only {len(completed_tasks)}/{task_count} tasks completed"
        assert throughput > 10.0, f"Throughput too low: {throughput:.1f} tasks/sec"
        assert total_time < 20.0, f"Execution took too long: {total_time:.2f} seconds"
        
        print(f"Performance Benchmark Results:")
        print(f"  Tasks completed: {len(completed_tasks)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} tasks/sec")
    
    def test_memory_usage_stability(self):
        """Test memory usage stability over time"""
        import psutil
        process = psutil.Process(os.getpid())
        
        # Record initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        engine = QuantumTaskEngine(max_workers=2, lnn_integration=False)
        self.test_components.append(engine)
        
        engine.add_resource_pool("cpu", capacity=4.0, efficiency=1.0)
        engine.start_scheduler()
        
        # Run many tasks to test for memory leaks
        def memory_test_callback(task):
            # Create some temporary data
            temp_data = np.random.random((100, 100))
            time.sleep(0.01)
            return {"status": "completed", "data_size": temp_data.size}
        
        # Submit tasks in batches
        total_tasks = 200
        batch_size = 20
        
        for batch in range(total_tasks // batch_size):
            # Submit batch
            for i in range(batch_size):
                task = QuantumTask(
                    id=f"memory_test_{batch:03d}_{i:03d}",
                    name=f"Memory Test Task {batch}-{i}",
                    priority=TaskPriority.LOW,
                    resources={"cpu": 0.5},
                    estimated_duration=0.02,
                    callback=memory_test_callback
                )
                engine.submit_task(task)
            
            # Wait for batch to complete
            time.sleep(1.0)
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = current_memory - initial_memory
            
            # Memory growth should be reasonable (less than 100MB growth)
            assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f}MB after {(batch+1)*batch_size} tasks"
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_growth = final_memory - initial_memory
        
        print(f"Memory Usage Test Results:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Total growth: {total_growth:.1f}MB")
        print(f"  Tasks processed: {total_tasks}")
        
        # Final assertion
        assert total_growth < 150, f"Memory leak detected: {total_growth:.1f}MB growth"


def run_integration_tests():
    """Run integration tests manually"""
    print("🧪 Running Full System Integration Tests")
    print("=" * 60)
    
    test_suite = TestSystemIntegration()
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        print(f"\n🔄 Running {method_name}...")
        
        try:
            # Setup
            test_suite.setup_teardown().__next__()
            
            # Run test
            method = getattr(test_suite, method_name)
            if asyncio.iscoroutinefunction(method):
                asyncio.run(method())
            else:
                method()
            
            print(f"✅ {method_name} PASSED")
            passed += 1
            
        except Exception as e:
            print(f"❌ {method_name} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
        
        finally:
            # Teardown
            try:
                test_suite.setup_teardown().__next__()
            except StopIteration:
                pass
        
        time.sleep(0.5)  # Brief pause between tests
    
    print(f"\n📊 Integration Test Results:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        return True
    else:
        print("💥 SOME INTEGRATION TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)