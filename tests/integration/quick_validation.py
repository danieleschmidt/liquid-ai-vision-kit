#!/usr/bin/env python3
"""
QUANTUM TASK PLANNING ENGINE - QUICK VALIDATION
Fast integration test for core functionality
"""

import time
import sys
import os

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_core_imports():
    """Test that all core modules can be imported"""
    try:
        from quantum_task_planner.core.quantum_engine import QuantumTaskEngine, QuantumTask, TaskPriority
        from quantum_task_planner.core.lnn_integration import LNNScheduler
        from quantum_task_planner.performance.cache_manager import QuantumCacheManager
        from quantum_task_planner.validation.task_validator import TaskValidator
        from quantum_task_planner.monitoring.health_monitor import HealthMonitor
        from quantum_task_planner.security.audit_logger import AuditLogger
        from quantum_task_planner.resilience.fault_tolerance import FaultToleranceManager
        from quantum_task_planner.scaling.performance_optimizer import PerformanceOptimizer
        from quantum_task_planner.deployment.global_orchestrator import GlobalOrchestrator
        print("✅ All core modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_engine_operation():
    """Test basic engine functionality"""
    try:
        from quantum_task_planner.core.quantum_engine import QuantumTaskEngine, QuantumTask, TaskPriority
        
        engine = QuantumTaskEngine(max_workers=1, lnn_integration=False)
        engine.add_resource_pool("cpu", capacity=2.0)
        
        # Test task creation
        task = QuantumTask(
            id="quick_test",
            name="Quick Test Task", 
            priority=TaskPriority.HIGH,
            estimated_duration=0.1
        )
        
        # Test submission
        task_id = engine.submit_task(task)
        
        # Test status
        status = engine.get_status()
        
        print("✅ Basic engine operations working")
        return task_id == "quick_test" and status is not None
        
    except Exception as e:
        print(f"❌ Engine test failed: {e}")
        return False

def test_cache_basic_operations():
    """Test basic cache operations"""
    try:
        from quantum_task_planner.performance.cache_manager import QuantumCacheManager
        
        cache = QuantumCacheManager()
        
        # Test put/get
        cache.put("test_key", {"data": "test_value"}, priority=2.0)
        result = cache.get("test_key")
        
        # Test stats
        stats = cache.get_comprehensive_stats()
        
        print("✅ Cache operations working")
        return result is not None and stats is not None
        
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False

def test_validation_framework():
    """Test validation framework"""
    try:
        from quantum_task_planner.validation.task_validator import TaskValidator
        
        validator = TaskValidator()
        
        # Test valid task
        valid_task = {
            'id': 'test_task',
            'name': 'Test Task',
            'priority': 1,
            'resources': {'cpu': 1, 'memory': 256}
        }
        
        report = validator.validate_task(valid_task)
        
        print("✅ Validation framework working")
        return report is not None and hasattr(report, 'is_valid')
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def test_global_orchestrator():
    """Test global orchestrator"""
    try:
        from quantum_task_planner.deployment.global_orchestrator import GlobalOrchestrator, RegionConfig, Region, ComplianceRegime, LanguageCode
        
        orchestrator = GlobalOrchestrator()
        
        # Test region addition with proper config
        region_config = RegionConfig(
            region=Region.US_EAST_1,
            endpoint="https://api.us-east-1.example.com",
            capacity=1000,
            compliance_regimes=[ComplianceRegime.CCPA],
            supported_languages=[LanguageCode.EN],
            data_residency_required=True
        )
        orchestrator.add_region(region_config)
        
        # Test health check
        health = orchestrator.is_healthy()
        
        print("✅ Global orchestrator working")
        return health is not None
        
    except Exception as e:
        print(f"❌ Global orchestrator test failed: {e}")
        return False

def run_quick_validation():
    """Run quick validation suite"""
    print("🧪 QUANTUM TASK PLANNING ENGINE - QUICK VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Basic Engine", test_basic_engine_operation),
        ("Cache Operations", test_cache_basic_operations),
        ("Validation Framework", test_validation_framework),
        ("Global Orchestrator", test_global_orchestrator)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Testing {test_name}...")
        start_time = time.time()
        
        try:
            result = test_func()
            duration = time.time() - start_time
            
            if result:
                print(f"✅ {test_name}: PASS ({duration:.3f}s)")
                passed += 1
            else:
                print(f"❌ {test_name}: FAIL ({duration:.3f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ {test_name}: ERROR - {str(e)} ({duration:.3f}s)")
    
    print(f"\n" + "=" * 60)
    print(f"🏁 QUICK VALIDATION RESULTS")
    print(f"=" * 60)
    print(f"📊 Passed: {passed}/{total} ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print(f"🟢 ALL TESTS PASSED - SYSTEM OPERATIONAL!")
        print(f"🚀 QUANTUM TASK PLANNING ENGINE v3.0 - READY!")
    elif passed >= total * 0.8:
        print(f"🟡 MOSTLY FUNCTIONAL - Minor issues detected")
    else:
        print(f"🔴 CRITICAL ISSUES - System needs attention")
    
    print(f"=" * 60)
    return passed, total

if __name__ == "__main__":
    run_quick_validation()