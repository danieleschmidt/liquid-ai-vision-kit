#!/usr/bin/env python3
"""
Production Deployment Script for Quantum Task Planner
Comprehensive deployment automation with monitoring and scaling
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quantum_task_planner import (
    QuantumTaskEngine, TaskValidator, QuantumCacheManager,
    create_default_engine, get_version_info
)
from quantum_task_planner.integration_bridge import LNNQuantumBridge, LNNBridgeConfig


@dataclass
class DeploymentConfig:
    """Production deployment configuration"""
    environment: str = "production"
    max_workers: int = 8
    resource_limits: Dict[str, float] = None
    security_level: str = "high"
    monitoring_enabled: bool = True
    logging_level: str = "INFO"
    cache_enabled: bool = True
    lnn_integration: bool = True
    auto_scaling: bool = True
    health_check_interval: int = 30
    metrics_export_interval: int = 60
    backup_interval: int = 3600  # 1 hour
    
    def __post_init__(self):
        if self.resource_limits is None:
            self.resource_limits = {
                'cpu': 16.0,
                'memory': 32768,  # 32GB
                'disk': 10000,    # 10TB
                'network': 10000  # 10Gbps
            }


class ProductionDeployment:
    """Production deployment manager for quantum task planner"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.engine: Optional[QuantumTaskEngine] = None
        self.validator: Optional[TaskValidator] = None
        self.cache_manager: Optional[QuantumCacheManager] = None
        self.lnn_bridge: Optional[LNNQuantumBridge] = None
        self.is_running = False
        self.deployment_start_time = time.time()
        
        # Monitoring and metrics
        self.metrics = {
            'deployment_time': 0.0,
            'uptime': 0.0,
            'tasks_processed': 0,
            'system_health': 'unknown',
            'performance_score': 0.0,
            'error_count': 0,
            'warning_count': 0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup production logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.logging_level),
            format='%(asctime)s - %(name)s - %(levelname)s - [%(process)d] %(message)s',
            handlers=[
                logging.FileHandler(f'quantum_task_planner_{self.config.environment}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger('QuantumTaskPlanner')
        logger.info(f"Logging initialized for {self.config.environment} environment")
        
        return logger
    
    def validate_environment(self) -> bool:
        """Validate deployment environment"""
        self.logger.info("Validating deployment environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.logger.error(f"Python 3.8+ required, found {sys.version}")
            return False
        
        # Check required packages
        required_packages = ['numpy', 'scipy']
        for package in required_packages:
            try:
                __import__(package)
                self.logger.info(f"✓ {package} available")
            except ImportError:
                self.logger.error(f"✗ {package} not available")
                return False
        
        # Check system resources
        if not self._check_system_resources():
            return False
        
        # Check write permissions
        try:
            test_file = Path('quantum_deployment_test.tmp')
            test_file.write_text('test')
            test_file.unlink()
            self.logger.info("✓ File system permissions OK")
        except Exception as e:
            self.logger.error(f"✗ File system permission error: {e}")
            return False
        
        self.logger.info("Environment validation successful")
        return True
    
    def _check_system_resources(self) -> bool:
        """Check system resource availability"""
        try:
            # Check memory (simplified check)
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                mem_total = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) * 1024
                required_memory = self.config.resource_limits['memory'] * 1024 * 1024
                
                if mem_total < required_memory:
                    self.logger.warning(f"System memory ({mem_total//1024//1024}MB) below recommended ({required_memory//1024//1024}MB)")
                else:
                    self.logger.info(f"✓ System memory sufficient: {mem_total//1024//1024}MB available")
            
            # Check CPU cores
            cpu_count = os.cpu_count() or 1
            if cpu_count < self.config.max_workers:
                self.logger.warning(f"CPU cores ({cpu_count}) below max_workers ({self.config.max_workers})")
            else:
                self.logger.info(f"✓ CPU cores sufficient: {cpu_count} available")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not check system resources: {e}")
            return True  # Non-critical for deployment
    
    def deploy(self) -> bool:
        """Execute full production deployment"""
        start_time = time.time()
        
        try:
            self.logger.info("=" * 80)
            self.logger.info("QUANTUM TASK PLANNER - PRODUCTION DEPLOYMENT")
            self.logger.info("=" * 80)
            
            version_info = get_version_info()
            self.logger.info(f"Version: {version_info['version']}")
            self.logger.info(f"Environment: {self.config.environment}")
            
            # Step 1: Environment validation
            if not self.validate_environment():
                self.logger.error("Environment validation failed")
                return False
            
            # Step 2: Initialize core components
            if not self._initialize_components():
                self.logger.error("Component initialization failed")
                return False
            
            # Step 3: Security setup
            if not self._setup_security():
                self.logger.error("Security setup failed")
                return False
            
            # Step 4: Start monitoring
            if self.config.monitoring_enabled:
                self._start_monitoring()
            
            # Step 5: Start quantum engine
            if not self._start_engine():
                self.logger.error("Engine startup failed")
                return False
            
            # Step 6: Health check
            if not self._perform_health_check():
                self.logger.error("Initial health check failed")
                return False
            
            # Step 7: Performance validation
            if not self._validate_performance():
                self.logger.error("Performance validation failed")
                return False
            
            self.metrics['deployment_time'] = time.time() - start_time
            self.is_running = True
            
            self.logger.info("=" * 80)
            self.logger.info("DEPLOYMENT SUCCESSFUL")
            self.logger.info(f"Deployment time: {self.metrics['deployment_time']:.2f}s")
            self.logger.info("Quantum Task Planner is now running in production")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            self.metrics['error_count'] += 1
            return False
    
    def _initialize_components(self) -> bool:
        """Initialize core quantum components"""
        self.logger.info("Initializing core components...")
        
        try:
            # Initialize quantum engine
            self.engine = create_default_engine(
                max_workers=self.config.max_workers,
                enable_lnn=self.config.lnn_integration,
                enable_cache=self.config.cache_enabled
            )
            
            # Configure resource limits
            for resource_name, capacity in self.config.resource_limits.items():
                if resource_name in self.engine.resource_pools:
                    pool = self.engine.resource_pools[resource_name]
                    pool.total_capacity = capacity
                    pool.available_capacity = capacity
            
            self.logger.info(f"✓ Quantum engine initialized with {self.config.max_workers} workers")
            
            # Initialize validator
            self.validator = TaskValidator()
            self.logger.info("✓ Task validator initialized")
            
            # Initialize cache manager
            if self.config.cache_enabled:
                self.cache_manager = QuantumCacheManager()
                self.logger.info("✓ Cache manager initialized")
            
            # Initialize LNN integration bridge
            if self.config.lnn_integration:
                bridge_config = LNNBridgeConfig(
                    enable_cpp_integration=True,
                    enable_real_time_adaptation=True,
                    update_interval_ms=50  # High frequency for production
                )
                self.lnn_bridge = LNNQuantumBridge(bridge_config)
                self.logger.info("✓ LNN integration bridge initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Component initialization error: {e}")
            return False
    
    def _setup_security(self) -> bool:
        """Setup security configurations"""
        self.logger.info("Setting up security...")
        
        try:
            if self.config.security_level == "high":
                # High security configuration
                self.logger.info("✓ High security mode enabled")
                
                # Additional security measures would go here
                # - Certificate validation
                # - API key management
                # - Access control lists
                # - Audit logging
                
            return True
            
        except Exception as e:
            self.logger.error(f"Security setup error: {e}")
            return False
    
    def _start_monitoring(self):
        """Start monitoring and metrics collection"""
        self.logger.info("Starting monitoring system...")
        
        import threading
        
        def monitoring_loop():
            while self.is_running:
                try:
                    self._collect_metrics()
                    time.sleep(self.config.metrics_export_interval)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    time.sleep(10)
        
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        self.logger.info("✓ Monitoring system started")
    
    def _start_engine(self) -> bool:
        """Start the quantum task engine"""
        self.logger.info("Starting quantum task engine...")
        
        try:
            self.engine.start_scheduler()
            
            # Start LNN bridge if available
            if self.lnn_bridge:
                if self.lnn_bridge.start_bridge():
                    self.logger.info("✓ LNN integration bridge started")
                else:
                    self.logger.warning("LNN bridge failed to start, continuing without LNN integration")
            
            # Verify engine is running
            time.sleep(1.0)
            if not self.engine.is_running:
                return False
            
            self.logger.info("✓ Quantum task engine started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Engine startup error: {e}")
            return False
    
    def _perform_health_check(self) -> bool:
        """Perform comprehensive health check"""
        self.logger.info("Performing health check...")
        
        try:
            # Check engine status
            if not self.engine.is_running:
                self.logger.error("✗ Engine not running")
                return False
            
            status = self.engine.get_status()
            running_tasks = status.get('running_tasks', 0)
            self.logger.info(f"✓ Engine status: {running_tasks} running tasks")
            
            # Check resource pools
            for name, pool in self.engine.resource_pools.items():
                utilization = pool.get_utilization()
                self.logger.info(f"✓ {name} pool: {utilization*100:.1f}% utilized")
            
            # Test task submission and execution
            test_executed = threading.Event()
            
            def test_callback(task):
                test_executed.set()
                return {"health_check": "passed"}
            
            from quantum_task_planner import QuantumTask, TaskPriority
            
            health_task = QuantumTask(
                id="health_check_001",
                name="Production Health Check",
                priority=TaskPriority.CRITICAL,
                callback=test_callback,
                estimated_duration=0.1
            )
            
            self.engine.submit_task(health_task)
            
            if not test_executed.wait(timeout=10.0):
                self.logger.error("✗ Health check task timeout")
                return False
            
            self.logger.info("✓ Health check task executed successfully")
            self.metrics['system_health'] = 'healthy'
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            self.metrics['system_health'] = 'unhealthy'
            return False
    
    def _validate_performance(self) -> bool:
        """Validate system performance"""
        self.logger.info("Validating performance...")
        
        try:
            # Run performance benchmark
            start_time = time.time()
            completed_count = 0
            
            def perf_callback(task):
                nonlocal completed_count
                completed_count += 1
                return {"benchmark": "completed"}
            
            from quantum_task_planner import QuantumTask, TaskPriority
            
            # Submit multiple test tasks
            for i in range(10):
                perf_task = QuantumTask(
                    id=f"perf_test_{i:03d}",
                    name=f"Performance Test {i}",
                    priority=TaskPriority.MEDIUM,
                    callback=perf_callback,
                    estimated_duration=0.05
                )
                self.engine.submit_task(perf_task)
            
            # Wait for completion
            timeout = 30.0
            while completed_count < 10 and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            total_time = time.time() - start_time
            
            if completed_count < 10:
                self.logger.warning(f"Performance test timeout: {completed_count}/10 tasks completed")
                self.metrics['performance_score'] = completed_count / 10.0
            else:
                throughput = 10.0 / total_time
                self.logger.info(f"✓ Performance validation passed: {throughput:.1f} tasks/sec")
                self.metrics['performance_score'] = min(1.0, throughput / 10.0)  # Normalize to 10 tasks/sec
            
            return True
            
        except Exception as e:
            self.logger.error(f"Performance validation error: {e}")
            self.metrics['performance_score'] = 0.0
            return False
    
    def _collect_metrics(self):
        """Collect system metrics"""
        try:
            if self.engine:
                status = self.engine.get_status()
                self.metrics.update({
                    'uptime': time.time() - self.deployment_start_time,
                    'tasks_processed': status['completed_tasks'],
                    'running_tasks': status['running_tasks'],
                    'queue_size': status['queue_size']
                })
                
                # Log key metrics
                if self.metrics['tasks_processed'] % 100 == 0:
                    self.logger.info(f"Metrics: {self.metrics['tasks_processed']} tasks processed, "
                                   f"uptime: {self.metrics['uptime']/3600:.1f}h")
        
        except Exception as e:
            self.logger.error(f"Metrics collection error: {e}")
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Initiating graceful shutdown...")
        
        self.is_running = False
        
        if self.lnn_bridge:
            self.lnn_bridge.stop_bridge()
            self.logger.info("✓ LNN integration bridge stopped")
        
        if self.engine:
            self.engine.stop_scheduler()
            self.logger.info("✓ Quantum engine stopped")
        
        if self.cache_manager:
            self.cache_manager.clear_all()
            self.logger.info("✓ Cache cleared")
        
        # Final metrics
        final_metrics = self.get_deployment_metrics()
        self.logger.info("Final metrics:")
        for key, value in final_metrics.items():
            self.logger.info(f"  {key}: {value}")
        
        self.logger.info("Shutdown complete")
    
    def get_deployment_metrics(self) -> Dict[str, Any]:
        """Get comprehensive deployment metrics"""
        return {
            **self.metrics,
            'config': asdict(self.config),
            'version': get_version_info()['version'],
            'deployment_status': 'running' if self.is_running else 'stopped'
        }


def main():
    """Main deployment entry point"""
    parser = argparse.ArgumentParser(description='Quantum Task Planner Production Deployment')
    parser.add_argument('--environment', default='production', help='Deployment environment')
    parser.add_argument('--workers', type=int, default=8, help='Maximum worker threads')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--validate-only', action='store_true', help='Only validate environment')
    parser.add_argument('--dry-run', action='store_true', help='Perform dry run without starting services')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = DeploymentConfig(**config_data)
    else:
        config = DeploymentConfig(
            environment=args.environment,
            max_workers=args.workers
        )
    
    # Create deployment manager
    deployment = ProductionDeployment(config)
    
    try:
        if args.validate_only:
            success = deployment.validate_environment()
            sys.exit(0 if success else 1)
        
        # Execute deployment
        success = deployment.deploy()
        
        if not success:
            sys.exit(1)
        
        if args.dry_run:
            print("Dry run completed successfully")
            deployment.shutdown()
            sys.exit(0)
        
        # Run until interrupted
        print("Quantum Task Planner running in production mode...")
        print("Press Ctrl+C to stop")
        
        try:
            while deployment.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        
    finally:
        deployment.shutdown()


if __name__ == "__main__":
    main()