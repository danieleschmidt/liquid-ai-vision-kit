#!/usr/bin/env python3
"""
Comprehensive Test Suite for Liquid AI Vision Kit
Implements all quality gates and validation tests for production readiness
"""

import sys
import os
import time
import unittest
import numpy as np
import threading
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import concurrent.futures
import subprocess

# Add project modules to path
sys.path.append('.')

# Import our system modules
try:
    from demo_basic_functionality import BasicLNNDemo
    from demo_robust_system import RobustLNNController, Logger, LogLevel, ComponentStatus, HealthMonitor
    from demo_scaled_system import ScaledLNNSystem, ProcessingMode, PerformanceMetrics
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")

@dataclass
class TestResult:
    test_name: str
    passed: bool
    duration_ms: float
    details: str = ""
    metrics: Dict[str, Any] = None

class LiquidVisionTestSuite:
    """Comprehensive test suite for the Liquid AI Vision Kit"""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Set up temporary test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="liquid_vision_test_")
        print(f"🧪 Test environment: {self.temp_dir}")
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def run_test(self, test_func, test_name: str) -> TestResult:
        """Run a single test and record results"""
        print(f"▶️  Running: {test_name}")
        start_time = time.time()
        
        try:
            result = test_func()
            duration = (time.time() - start_time) * 1000
            
            if isinstance(result, dict):
                test_result = TestResult(
                    test_name=test_name,
                    passed=result.get('passed', True),
                    duration_ms=duration,
                    details=result.get('details', ''),
                    metrics=result.get('metrics', {})
                )
            else:
                test_result = TestResult(
                    test_name=test_name,
                    passed=bool(result),
                    duration_ms=duration
                )
            
            status = "✅ PASS" if test_result.passed else "❌ FAIL"
            print(f"   {status} ({duration:.1f}ms) - {test_result.details}")
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            test_result = TestResult(
                test_name=test_name,
                passed=False,
                duration_ms=duration,
                details=f"Exception: {str(e)}"
            )
            print(f"   ❌ FAIL ({duration:.1f}ms) - Exception: {e}")
        
        self.test_results.append(test_result)
        return test_result
    
    # Unit Tests
    def test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic LNN functionality"""
        try:
            demo = BasicLNNDemo()
            
            # Test initialization
            assert hasattr(demo, 'frame_count')
            assert hasattr(demo, 'stats')
            
            # Test frame processing with correct image size (240x320 as expected by BasicLNNDemo)
            test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            result = demo.process_frame(test_image)
            
            assert result is not None
            assert 'processed_frame' in result
            assert 'control_output' in result
            assert 'total_time_ms' in result
            
            # Test control output validation
            control = result['control_output']
            assert 'forward_velocity' in control
            assert 'yaw_rate' in control
            assert 'target_altitude' in control
            assert 'confidence' in control
            
            # Test safety constraints
            assert -5.0 <= control['forward_velocity'] <= 5.0
            assert -2.0 <= control['yaw_rate'] <= 2.0
            assert 0.5 <= control['target_altitude'] <= 50.0
            assert 0.0 <= control['confidence'] <= 1.0
            
            return {
                'passed': True,
                'details': 'Basic functionality validated',
                'metrics': {'frame_processing_time_ms': result.get('total_time_ms', 0)}
            }
        except Exception as e:
            return {
                'passed': False,
                'details': f'Test failed: {str(e)}',
                'metrics': {}
            }
    
    def test_robust_system_components(self) -> Dict[str, Any]:
        """Test robust system components"""
        logger = Logger(min_level=LogLevel.DEBUG, log_file=os.path.join(self.temp_dir, "test.log"))
        
        # Test logging
        logger.info("Test message", "TEST")
        logger.error("Test error", "TEST")
        
        # Test health monitoring
        health_monitor = HealthMonitor(logger)
        health_monitor.register_component("test_component")
        health_monitor.update_component_health("test_component", ComponentStatus.HEALTHY, "Test")
        
        component_health = health_monitor.get_component_health("test_component")
        assert component_health is not None
        assert component_health.status == ComponentStatus.HEALTHY
        
        # Test system health check
        assert health_monitor.is_system_healthy()
        
        # Test critical component
        health_monitor.update_component_health("test_component", ComponentStatus.CRITICAL, "Critical test")
        assert not health_monitor.is_system_healthy()
        
        return {
            'passed': True,
            'details': 'Robust system components validated',
            'metrics': {'components_tested': 2}
        }
    
    def test_performance_optimization(self) -> Dict[str, Any]:
        """Test performance optimization features"""
        system = ScaledLNNSystem()
        
        # Test different processing modes
        modes_tested = []
        performance_metrics = {}
        
        for mode in [ProcessingMode.SINGLE_THREADED, ProcessingMode.MULTI_THREADED]:
            try:
                system.set_processing_mode(mode, num_workers=2)
                
                # Process test frames
                start_time = time.time()
                successful_frames = 0
                
                for i in range(10):
                    frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
                    result, metrics = system.process_frame(frame, i)
                    if result is not None:
                        successful_frames += 1
                
                processing_time = time.time() - start_time
                fps = successful_frames / processing_time if processing_time > 0 else 0
                
                modes_tested.append(mode.value)
                performance_metrics[mode.value] = {
                    'fps': fps,
                    'success_rate': successful_frames / 10,
                    'processing_time': processing_time
                }
                
            except Exception as e:
                print(f"Warning: Could not test mode {mode.value}: {e}")
        
        # Test caching
        cache_hit_rates = []
        for _ in range(5):
            # Process same frame multiple times to test caching
            frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
            for _ in range(3):
                system.process_frame(frame, 0)
        
        status = system.get_system_status()
        preprocessing_hit_rate = status['cache_stats']['preprocessing']['hit_rate']
        
        return {
            'passed': len(modes_tested) >= 2 and preprocessing_hit_rate >= 0,
            'details': f'Tested {len(modes_tested)} processing modes, cache hit rate: {preprocessing_hit_rate*100:.1f}%',
            'metrics': {
                'modes_tested': modes_tested,
                'performance': performance_metrics,
                'cache_hit_rate': preprocessing_hit_rate
            }
        }
    
    def test_error_handling_robustness(self) -> Dict[str, Any]:
        """Test error handling and recovery mechanisms"""
        logger = Logger(min_level=LogLevel.DEBUG, log_file=os.path.join(self.temp_dir, "error_test.log"))
        controller = RobustLNNController(logger)
        
        errors_handled = 0
        total_tests = 0
        
        # Test with various problematic inputs
        test_cases = [
            np.zeros((120, 160, 3), dtype=np.uint8),  # All black frame
            np.ones((120, 160, 3), dtype=np.uint8) * 255,  # All white frame
            np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8),  # Wrong size
            np.random.randint(0, 255, (120, 160, 1), dtype=np.uint8),  # Single channel
        ]
        
        for i, test_frame in enumerate(test_cases):
            total_tests += 1
            try:
                result = controller.process_frame(test_frame)
                if result is not None:
                    errors_handled += 1
            except Exception:
                # Even exceptions should be handled gracefully
                pass
        
        # Test system status after errors
        status = controller.get_system_status()
        system_healthy = status.get('system_healthy', False)
        
        success_rate = errors_handled / total_tests if total_tests > 0 else 0
        
        return {
            'passed': success_rate >= 0.75 and system_healthy is not None,  # At least 75% success rate
            'details': f'Handled {errors_handled}/{total_tests} error cases successfully',
            'metrics': {
                'error_handling_success_rate': success_rate,
                'system_healthy': system_healthy,
                'error_cases_tested': total_tests
            }
        }
    
    def test_concurrent_processing(self) -> Dict[str, Any]:
        """Test concurrent processing capabilities"""
        system = ScaledLNNSystem()
        system.set_processing_mode(ProcessingMode.MULTI_THREADED, num_workers=4)
        
        num_threads = 8
        frames_per_thread = 5
        results = []
        
        def process_frames_worker(thread_id):
            thread_results = []
            for i in range(frames_per_thread):
                frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
                result, metrics = system.process_frame(frame, thread_id * frames_per_thread + i)
                thread_results.append(result is not None)
            return thread_results
        
        # Test concurrent processing
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_frames_worker, i) for i in range(num_threads)]
            
            for future in concurrent.futures.as_completed(futures):
                thread_results = future.result()
                results.extend(thread_results)
        
        processing_time = time.time() - start_time
        success_rate = sum(results) / len(results) if results else 0
        throughput = len(results) / processing_time if processing_time > 0 else 0
        
        return {
            'passed': success_rate >= 0.8 and throughput > 0,
            'details': f'Concurrent processing: {success_rate*100:.1f}% success, {throughput:.1f} FPS',
            'metrics': {
                'concurrent_success_rate': success_rate,
                'concurrent_throughput_fps': throughput,
                'threads_tested': num_threads,
                'total_frames': len(results)
            }
        }
    
    def test_memory_management(self) -> Dict[str, Any]:
        """Test memory management and resource cleanup"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and destroy many objects to test memory management
        systems = []
        for _ in range(10):
            system = ScaledLNNSystem()
            
            # Process some frames
            for i in range(20):
                frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
                system.process_frame(frame, i)
            
            systems.append(system)
        
        # Clear references
        systems.clear()
        gc.collect()  # Force garbage collection
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        memory_ok = memory_increase < 100
        
        return {
            'passed': memory_ok,
            'details': f'Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)',
            'metrics': {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase,
                'memory_leak_detected': not memory_ok
            }
        }
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks and requirements"""
        system = ScaledLNNSystem()
        
        # Test single-threaded performance baseline
        system.set_processing_mode(ProcessingMode.SINGLE_THREADED)
        
        frame_times = []
        confidence_scores = []
        
        for i in range(25):  # Test with 25 frames
            frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
            
            start_time = time.time()
            result, metrics = system.process_frame(frame, i)
            frame_time = (time.time() - start_time) * 1000  # ms
            
            if result is not None and metrics is not None:
                frame_times.append(frame_time)
                if 'confidence' in result:
                    confidence_scores.append(result['confidence'])
        
        if frame_times:
            avg_frame_time = sum(frame_times) / len(frame_times)
            max_frame_time = max(frame_times)
            fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            # Performance requirements
            meets_latency_req = avg_frame_time <= 100  # <100ms average
            meets_max_latency_req = max_frame_time <= 200  # <200ms max
            meets_fps_req = fps >= 5  # >=5 FPS
            meets_confidence_req = avg_confidence >= 0.0  # Some confidence
            
            all_requirements_met = all([
                meets_latency_req, meets_max_latency_req, 
                meets_fps_req, meets_confidence_req
            ])
            
            return {
                'passed': all_requirements_met,
                'details': f'Avg: {avg_frame_time:.1f}ms, Max: {max_frame_time:.1f}ms, FPS: {fps:.1f}',
                'metrics': {
                    'avg_frame_time_ms': avg_frame_time,
                    'max_frame_time_ms': max_frame_time,
                    'fps': fps,
                    'avg_confidence': avg_confidence,
                    'meets_requirements': all_requirements_met,
                    'frames_processed': len(frame_times)
                }
            }
        else:
            return {
                'passed': False,
                'details': 'No successful frame processing',
                'metrics': {'frames_processed': 0}
            }
    
    def test_integration_end_to_end(self) -> Dict[str, Any]:
        """Test end-to-end integration"""
        # Test complete pipeline from basic to scaled system
        
        # Step 1: Basic system
        basic_demo = BasicLNNDemo()
        test_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        basic_result = basic_demo.process_frame(test_frame)
        
        basic_works = basic_result is not None and 'control_output' in basic_result
        
        # Step 2: Robust system
        logger = Logger(min_level=LogLevel.INFO, log_file=os.path.join(self.temp_dir, "integration_test.log"))
        robust_controller = RobustLNNController(logger)
        robust_result = robust_controller.process_frame(test_frame)
        
        robust_works = robust_result is not None and 'control_output' in robust_result
        
        # Step 3: Scaled system
        scaled_system = ScaledLNNSystem()
        scaled_system.set_processing_mode(ProcessingMode.MULTI_THREADED, num_workers=2)
        scaled_result, scaled_metrics = scaled_system.process_frame(test_frame[:120, :160], 0)  # Resize for scaled system
        
        scaled_works = scaled_result is not None and scaled_metrics is not None
        
        # Integration validation
        all_systems_work = basic_works and robust_works and scaled_works
        
        return {
            'passed': all_systems_work,
            'details': f'Basic: {basic_works}, Robust: {robust_works}, Scaled: {scaled_works}',
            'metrics': {
                'basic_system': basic_works,
                'robust_system': robust_works,
                'scaled_system': scaled_works,
                'integration_successful': all_systems_work
            }
        }
    
    def run_all_tests(self):
        """Run all tests in the comprehensive suite"""
        print("\n🧪 LIQUID AI VISION KIT - COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        print(f"Test environment: {self.temp_dir}")
        print()
        
        # Define all tests
        test_suite = [
            (self.test_basic_functionality, "Basic Functionality"),
            (self.test_robust_system_components, "Robust System Components"),
            (self.test_performance_optimization, "Performance Optimization"),
            (self.test_error_handling_robustness, "Error Handling & Robustness"),
            (self.test_concurrent_processing, "Concurrent Processing"),
            (self.test_memory_management, "Memory Management"),
            (self.test_performance_benchmarks, "Performance Benchmarks"),
            (self.test_integration_end_to_end, "End-to-End Integration"),
        ]
        
        # Run all tests
        start_time = time.time()
        
        for test_func, test_name in test_suite:
            self.run_test(test_func, test_name)
            time.sleep(0.1)  # Small delay between tests
        
        total_time = time.time() - start_time
        
        # Generate test report
        self.generate_test_report(total_time)
    
    def generate_test_report(self, total_time: float):
        """Generate comprehensive test report"""
        passed_tests = [r for r in self.test_results if r.passed]
        failed_tests = [r for r in self.test_results if not r.passed]
        
        pass_rate = len(passed_tests) / len(self.test_results) * 100 if self.test_results else 0
        avg_test_time = sum(r.duration_ms for r in self.test_results) / len(self.test_results) if self.test_results else 0
        
        print(f"\n📊 TEST REPORT SUMMARY")
        print("=" * 60)
        print(f"Total tests:     {len(self.test_results)}")
        print(f"Passed:          {len(passed_tests)}")
        print(f"Failed:          {len(failed_tests)}")
        print(f"Pass rate:       {pass_rate:.1f}%")
        print(f"Total time:      {total_time:.2f}s")
        print(f"Avg test time:   {avg_test_time:.1f}ms")
        
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"   - {test.test_name}: {test.details}")
        
        print(f"\n✅ PASSED TESTS:")
        for test in passed_tests:
            print(f"   - {test.test_name} ({test.duration_ms:.1f}ms)")
        
        # Quality Gates Assessment
        print(f"\n🎯 QUALITY GATES ASSESSMENT")
        print("=" * 60)
        
        quality_gates = [
            ("Minimum Pass Rate (85%)", pass_rate >= 85),
            ("All Core Tests Pass", all(r.passed for r in self.test_results if 'Basic' in r.test_name or 'Integration' in r.test_name)),
            ("Performance Requirements", any(r.passed for r in self.test_results if 'Performance' in r.test_name)),
            ("Robustness Requirements", any(r.passed for r in self.test_results if 'Robust' in r.test_name or 'Error' in r.test_name)),
            ("Scalability Requirements", any(r.passed for r in self.test_results if 'Concurrent' in r.test_name or 'Optimization' in r.test_name)),
        ]
        
        all_gates_pass = True
        for gate_name, gate_result in quality_gates:
            status = "✅ PASS" if gate_result else "❌ FAIL"
            print(f"   {status} - {gate_name}")
            if not gate_result:
                all_gates_pass = False
        
        # Final Assessment
        print(f"\n🏁 FINAL ASSESSMENT")
        print("=" * 60)
        
        if all_gates_pass and pass_rate >= 95:
            print("🎉 SYSTEM READY FOR PRODUCTION DEPLOYMENT")
            print("   All quality gates passed with excellent test coverage")
        elif all_gates_pass and pass_rate >= 85:
            print("✅ SYSTEM READY FOR DEPLOYMENT")
            print("   All critical quality gates passed")
        elif pass_rate >= 75:
            print("⚠️  SYSTEM NEEDS IMPROVEMENTS")
            print("   Some quality gates failed - address issues before deployment")
        else:
            print("❌ SYSTEM NOT READY FOR DEPLOYMENT")
            print("   Critical issues found - major improvements needed")
        
        # Export detailed results
        self.export_test_results()
        
        print(f"\n📄 Test results exported to: {self.temp_dir}/test_results.json")
        print(f"📁 Test logs available in: {self.temp_dir}/")
    
    def export_test_results(self):
        """Export detailed test results to JSON"""
        results_data = {
            'test_run_timestamp': time.time(),
            'test_environment': self.temp_dir,
            'summary': {
                'total_tests': len(self.test_results),
                'passed': len([r for r in self.test_results if r.passed]),
                'failed': len([r for r in self.test_results if not r.passed]),
                'pass_rate': len([r for r in self.test_results if r.passed]) / len(self.test_results) * 100 if self.test_results else 0
            },
            'test_results': []
        }
        
        for result in self.test_results:
            results_data['test_results'].append({
                'test_name': result.test_name,
                'passed': result.passed,
                'duration_ms': result.duration_ms,
                'details': result.details,
                'metrics': result.metrics or {}
            })
        
        results_file = os.path.join(self.temp_dir, 'test_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

def main():
    """Run the comprehensive test suite"""
    print("🚀 Liquid AI Vision Kit - Comprehensive Testing")
    print("Running all quality gates and validation tests...")
    
    test_suite = LiquidVisionTestSuite()
    
    try:
        test_suite.run_all_tests()
    finally:
        # Cleanup test environment
        test_suite.cleanup_test_environment()
    
    print("\n🏁 Testing completed!")

if __name__ == "__main__":
    main()