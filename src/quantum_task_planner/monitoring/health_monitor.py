#!/usr/bin/env python3
"""
Quantum System Health Monitoring and Alerting
Advanced monitoring with self-healing capabilities and predictive alerting
"""

import time
import threading
import json
import logging
import traceback
import psutil
import queue
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILURE = "failure"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics being monitored"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    @property
    def age(self) -> float:
        """Age of metric in seconds"""
        return time.time() - self.timestamp
    
    def get_status(self) -> HealthStatus:
        """Get health status based on thresholds"""
        if self.threshold_critical and self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.threshold_warning and self.value >= self.threshold_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


@dataclass
class Alert:
    """System alert"""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def resolve(self):
        """Mark alert as resolved"""
        self.resolved = True
        self.resolved_at = time.time()
    
    @property
    def duration(self) -> float:
        """Duration of alert in seconds"""
        end_time = self.resolved_at if self.resolved else time.time()
        return end_time - self.timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'timestamp': self.timestamp,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at,
            'duration': self.duration,
            'metadata': self.metadata
        }


class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, 
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def __call__(self, func):
        """Decorator for circuit breaker"""
        def wrapper(*args, **kwargs):
            with self.lock:
                if self.state == 'OPEN':
                    if time.time() - self.last_failure_time < self.recovery_timeout:
                        raise Exception(f"Circuit breaker OPEN for {func.__name__}")
                    else:
                        self.state = 'HALF_OPEN'
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Success - reset if in HALF_OPEN or CLOSED
                    if self.state == 'HALF_OPEN':
                        self.state = 'CLOSED'
                        self.failure_count = 0
                    
                    return result
                    
                except self.expected_exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = 'OPEN'
                    
                    raise e
        
        return wrapper
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time,
            'recovery_timeout': self.recovery_timeout
        }


class HealthMonitor:
    """Comprehensive system health monitoring"""
    
    def __init__(self, check_interval: float = 30.0, metric_retention: int = 3600):
        self.check_interval = check_interval
        self.metric_retention = metric_retention  # seconds
        
        # Monitoring data
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: Dict[str, Alert] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # System state
        self.overall_status = HealthStatus.HEALTHY
        self.last_check_time = 0
        self.running = False
        
        # Callbacks and handlers
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.metric_handlers: List[Callable[[HealthMetric], None]] = []
        self.health_checks: List[Callable[[], Dict[str, Any]]] = []
        
        # Threading
        self.monitor_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.alert_queue = queue.Queue()
        self.lock = threading.RLock()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize built-in checks
        self._initialize_builtin_checks()
    
    def _initialize_builtin_checks(self):
        """Initialize built-in health checks"""
        self.add_health_check(self._check_system_resources)
        self.add_health_check(self._check_memory_usage)
        self.add_health_check(self._check_disk_usage)
        self.add_health_check(self._check_network_connectivity)
    
    def start(self):
        """Start health monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Start alert processing thread
        alert_thread = threading.Thread(target=self._process_alerts, daemon=True)
        alert_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop(self):
        """Stop health monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        self.logger.info("Health monitoring stopped")
    
    def add_metric(self, metric: HealthMetric):
        """Add a metric to monitoring"""
        with self.lock:
            self.metrics[metric.name].append(metric)
            
            # Check thresholds and generate alerts
            status = metric.get_status()
            if status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                self._generate_threshold_alert(metric, status)
            
            # Notify handlers
            for handler in self.metric_handlers:
                try:
                    handler(metric)
                except Exception as e:
                    self.logger.error(f"Error in metric handler: {e}")
    
    def add_alert(self, alert: Alert):
        """Add an alert"""
        with self.lock:
            self.alerts[alert.id] = alert
            self.alert_queue.put(alert)
            
            # Update overall status based on alert severity
            self._update_overall_status()
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        with self.lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolve()
                self._update_overall_status()
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler"""
        self.alert_handlers.append(handler)
    
    def add_metric_handler(self, handler: Callable[[HealthMetric], None]):
        """Add metric handler"""
        self.metric_handlers.append(handler)
    
    def add_health_check(self, check_func: Callable[[], Dict[str, Any]]):
        """Add custom health check"""
        self.health_checks.append(check_func)
    
    def create_circuit_breaker(self, name: str, failure_threshold: int = 5, 
                              recovery_timeout: float = 60.0) -> CircuitBreaker:
        """Create and register a circuit breaker"""
        breaker = CircuitBreaker(failure_threshold, recovery_timeout)
        self.circuit_breakers[name] = breaker
        return breaker
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                start_time = time.time()
                
                # Run health checks
                self._run_health_checks()
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                # Update overall status
                self._update_overall_status()
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.check_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                self.last_check_time = time.time()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self.logger.error(traceback.format_exc())
                time.sleep(self.check_interval)
    
    def _run_health_checks(self):
        """Run all registered health checks"""
        for check_func in self.health_checks:
            try:
                result = check_func()
                if isinstance(result, dict):
                    for metric_name, metric_value in result.items():
                        if isinstance(metric_value, (int, float)):
                            metric = HealthMetric(
                                name=metric_name,
                                value=float(metric_value),
                                metric_type=MetricType.GAUGE
                            )
                            self.add_metric(metric)
                
            except Exception as e:
                self.logger.error(f"Error in health check {check_func.__name__}: {e}")
                
                # Generate alert for failed health check
                alert = Alert(
                    id=f"health_check_failed_{check_func.__name__}_{int(time.time())}",
                    severity=AlertSeverity.ERROR,
                    title="Health Check Failed",
                    message=f"Health check {check_func.__name__} failed: {str(e)}",
                    source="health_monitor",
                    metadata={'check_function': check_func.__name__, 'error': str(e)}
                )
                self.add_alert(alert)
    
    def _check_system_resources(self) -> Dict[str, float]:
        """Check system CPU and memory usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_used_gb': memory.used / (1024**3)
            }
        except Exception as e:
            self.logger.error(f"Error checking system resources: {e}")
            return {}
    
    def _check_memory_usage(self) -> Dict[str, float]:
        """Check detailed memory usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'process_memory_rss_mb': memory_info.rss / (1024**2),
                'process_memory_vms_mb': memory_info.vms / (1024**2),
                'process_memory_percent': process.memory_percent()
            }
        except Exception as e:
            self.logger.error(f"Error checking memory usage: {e}")
            return {}
    
    def _check_disk_usage(self) -> Dict[str, float]:
        """Check disk space usage"""
        try:
            disk_usage = psutil.disk_usage('/')
            
            return {
                'disk_usage_percent': (disk_usage.used / disk_usage.total) * 100,
                'disk_free_gb': disk_usage.free / (1024**3),
                'disk_used_gb': disk_usage.used / (1024**3)
            }
        except Exception as e:
            self.logger.error(f"Error checking disk usage: {e}")
            return {}
    
    def _check_network_connectivity(self) -> Dict[str, float]:
        """Check network connectivity"""
        try:
            # Simple connectivity check
            import socket
            
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            
            try:
                result = sock.connect_ex(('8.8.8.8', 53))  # Google DNS
                latency = (time.time() - start_time) * 1000  # ms
                
                return {
                    'network_connectivity': 1.0 if result == 0 else 0.0,
                    'network_latency_ms': latency if result == 0 else -1
                }
            finally:
                sock.close()
                
        except Exception as e:
            self.logger.error(f"Error checking network connectivity: {e}")
            return {'network_connectivity': 0.0}
    
    def _generate_threshold_alert(self, metric: HealthMetric, status: HealthStatus):
        """Generate alert when metric crosses threshold"""
        alert_id = f"threshold_{metric.name}_{int(time.time())}"
        
        severity = AlertSeverity.WARNING if status == HealthStatus.WARNING else AlertSeverity.CRITICAL
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=f"Metric Threshold Exceeded: {metric.name}",
            message=f"Metric {metric.name} value {metric.value} exceeded threshold",
            source="health_monitor",
            metadata={
                'metric_name': metric.name,
                'metric_value': metric.value,
                'threshold_warning': metric.threshold_warning,
                'threshold_critical': metric.threshold_critical
            }
        )
        
        self.add_alert(alert)
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics beyond retention period"""
        cutoff_time = time.time() - self.metric_retention
        
        with self.lock:
            for metric_name, metric_deque in self.metrics.items():
                # Remove old metrics from the front of deque
                while metric_deque and metric_deque[0].timestamp < cutoff_time:
                    metric_deque.popleft()
    
    def _update_overall_status(self):
        """Update overall system status based on active alerts"""
        with self.lock:
            active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
            
            if not active_alerts:
                self.overall_status = HealthStatus.HEALTHY
            else:
                # Determine worst status
                severities = [alert.severity for alert in active_alerts]
                
                if AlertSeverity.EMERGENCY in severities:
                    self.overall_status = HealthStatus.FAILURE
                elif AlertSeverity.CRITICAL in severities:
                    self.overall_status = HealthStatus.CRITICAL
                elif AlertSeverity.ERROR in severities:
                    self.overall_status = HealthStatus.DEGRADED
                else:
                    self.overall_status = HealthStatus.WARNING
    
    def _process_alerts(self):
        """Process alerts from queue"""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1.0)
                
                # Notify all alert handlers
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        self.logger.error(f"Error in alert handler: {e}")
                
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing alerts: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self.lock:
            active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
            resolved_alerts = [alert for alert in self.alerts.values() if alert.resolved]
            
            # Calculate metric statistics
            metric_stats = {}
            for metric_name, metric_deque in self.metrics.items():
                if metric_deque:
                    values = [m.value for m in metric_deque]
                    metric_stats[metric_name] = {
                        'current': values[-1] if values else 0,
                        'average': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            
            # Circuit breaker states
            breaker_states = {
                name: breaker.get_state() 
                for name, breaker in self.circuit_breakers.items()
            }
            
            return {
                'overall_status': self.overall_status.value,
                'last_check_time': self.last_check_time,
                'monitoring_running': self.running,
                'active_alerts': [alert.to_dict() for alert in active_alerts],
                'resolved_alerts_count': len(resolved_alerts),
                'metric_stats': metric_stats,
                'circuit_breakers': breaker_states,
                'alert_handlers_count': len(self.alert_handlers),
                'health_checks_count': len(self.health_checks)
            }
    
    def get_metrics_history(self, metric_name: str, duration: float = 3600) -> List[Dict[str, Any]]:
        """Get metric history for specified duration"""
        cutoff_time = time.time() - duration
        
        with self.lock:
            if metric_name not in self.metrics:
                return []
            
            return [
                {
                    'timestamp': metric.timestamp,
                    'value': metric.value,
                    'age': metric.age,
                    'tags': metric.tags
                }
                for metric in self.metrics[metric_name]
                if metric.timestamp >= cutoff_time
            ]
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        status = self.get_status()
        
        # Add additional analysis
        report = {
            'timestamp': time.time(),
            'system_status': status,
            'recommendations': self._generate_recommendations(),
            'performance_analysis': self._analyze_performance_trends(),
            'risk_assessment': self._assess_risks()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on current state"""
        recommendations = []
        
        # Analyze metrics for recommendations
        with self.lock:
            for metric_name, metric_deque in self.metrics.items():
                if not metric_deque:
                    continue
                
                recent_values = [m.value for m in list(metric_deque)[-10:]]
                
                if 'cpu_usage_percent' in metric_name:
                    avg_cpu = np.mean(recent_values)
                    if avg_cpu > 80:
                        recommendations.append("High CPU usage detected - consider scaling resources")
                    elif avg_cpu < 10:
                        recommendations.append("Low CPU usage - resources may be over-provisioned")
                
                elif 'memory_usage_percent' in metric_name:
                    avg_memory = np.mean(recent_values)
                    if avg_memory > 85:
                        recommendations.append("High memory usage - monitor for memory leaks")
                
                elif 'disk_usage_percent' in metric_name:
                    current_disk = recent_values[-1] if recent_values else 0
                    if current_disk > 90:
                        recommendations.append("Critical disk space - immediate cleanup required")
                    elif current_disk > 80:
                        recommendations.append("High disk usage - plan for storage expansion")
        
        # Check circuit breaker states
        for name, breaker in self.circuit_breakers.items():
            state = breaker.get_state()
            if state['state'] == 'OPEN':
                recommendations.append(f"Circuit breaker '{name}' is OPEN - investigate underlying issues")
        
        return recommendations
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends"""
        trends = {}
        
        with self.lock:
            for metric_name, metric_deque in self.metrics.items():
                if len(metric_deque) < 10:
                    continue
                
                values = [m.value for m in metric_deque]
                timestamps = [m.timestamp for m in metric_deque]
                
                # Calculate trend (simple linear regression slope)
                if len(values) >= 2:
                    x = np.array(timestamps)
                    y = np.array(values)
                    
                    # Normalize timestamps
                    x = x - x[0]
                    
                    # Calculate slope
                    if np.std(x) > 0:
                        slope = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
                        
                        trends[metric_name] = {
                            'trend_direction': 'increasing' if slope > 0.1 else 'decreasing' if slope < -0.1 else 'stable',
                            'slope': slope,
                            'current_value': values[-1],
                            'average_value': np.mean(values),
                            'volatility': np.std(values)
                        }
        
        return trends
    
    def _assess_risks(self) -> Dict[str, Any]:
        """Assess system risks"""
        risks = {
            'high_risk_factors': [],
            'medium_risk_factors': [],
            'low_risk_factors': [],
            'overall_risk_score': 0.0
        }
        
        risk_score = 0.0
        
        # Assess based on active alerts
        with self.lock:
            active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
            
            for alert in active_alerts:
                if alert.severity == AlertSeverity.CRITICAL:
                    risks['high_risk_factors'].append(f"Critical alert: {alert.title}")
                    risk_score += 30
                elif alert.severity == AlertSeverity.ERROR:
                    risks['medium_risk_factors'].append(f"Error alert: {alert.title}")
                    risk_score += 15
                elif alert.severity == AlertSeverity.WARNING:
                    risks['low_risk_factors'].append(f"Warning alert: {alert.title}")
                    risk_score += 5
        
        # Assess based on circuit breaker states
        open_breakers = [
            name for name, breaker in self.circuit_breakers.items()
            if breaker.get_state()['state'] == 'OPEN'
        ]
        
        if open_breakers:
            risks['high_risk_factors'].append(f"Open circuit breakers: {', '.join(open_breakers)}")
            risk_score += 20 * len(open_breakers)
        
        # Assess based on metric trends
        trends = self._analyze_performance_trends()
        for metric_name, trend_data in trends.items():
            if 'usage_percent' in metric_name and trend_data['current_value'] > 90:
                risks['high_risk_factors'].append(f"High resource usage: {metric_name}")
                risk_score += 25
            elif 'usage_percent' in metric_name and trend_data['current_value'] > 80:
                risks['medium_risk_factors'].append(f"Elevated resource usage: {metric_name}")
                risk_score += 10
        
        risks['overall_risk_score'] = min(100.0, risk_score)
        
        return risks


# Example alert handlers
def console_alert_handler(alert: Alert):
    """Simple console alert handler"""
    print(f"[{alert.severity.value.upper()}] {alert.title}: {alert.message}")


def file_alert_handler(alert: Alert, log_file: str = "/tmp/alerts.log"):
    """File-based alert handler"""
    try:
        with open(log_file, 'a') as f:
            f.write(f"{time.ctime(alert.timestamp)} [{alert.severity.value.upper()}] {alert.title}: {alert.message}\n")
    except Exception as e:
        print(f"Error writing alert to file: {e}")


if __name__ == "__main__":
    # Example usage and testing
    print("Quantum Health Monitoring System")
    print("=" * 50)
    
    # Create health monitor
    monitor = HealthMonitor(check_interval=5.0)
    
    # Add alert handlers
    monitor.add_alert_handler(console_alert_handler)
    monitor.add_alert_handler(lambda alert: file_alert_handler(alert, "/tmp/quantum_alerts.log"))
    
    # Create circuit breakers
    db_breaker = monitor.create_circuit_breaker("database", failure_threshold=3, recovery_timeout=30.0)
    api_breaker = monitor.create_circuit_breaker("external_api", failure_threshold=5, recovery_timeout=60.0)
    
    # Add custom health check
    def custom_check():
        return {
            'custom_metric_1': np.random.uniform(0, 100),
            'custom_metric_2': np.random.uniform(0, 50)
        }
    
    monitor.add_health_check(custom_check)
    
    # Start monitoring
    monitor.start()
    
    try:
        print("Health monitoring started. Running for 30 seconds...")
        
        # Simulate some metrics and alerts
        for i in range(6):
            time.sleep(5)
            
            # Add some test metrics
            monitor.add_metric(HealthMetric(
                name="test_latency_ms",
                value=np.random.uniform(10, 200),
                metric_type=MetricType.TIMER,
                threshold_warning=100.0,
                threshold_critical=150.0
            ))
            
            monitor.add_metric(HealthMetric(
                name="test_throughput",
                value=np.random.uniform(50, 1000),
                metric_type=MetricType.RATE
            ))
            
            # Occasionally trigger a test alert
            if i == 3:
                test_alert = Alert(
                    id=f"test_alert_{int(time.time())}",
                    severity=AlertSeverity.WARNING,
                    title="Test Alert",
                    message="This is a test alert for demonstration",
                    source="test_system"
                )
                monitor.add_alert(test_alert)
            
            # Show current status
            status = monitor.get_status()
            print(f"\nStatus check {i+1}:")
            print(f"  Overall Status: {status['overall_status']}")
            print(f"  Active Alerts: {len(status['active_alerts'])}")
            print(f"  Metrics Tracked: {len(status['metric_stats'])}")
        
        # Generate and show health report
        print("\nGenerating comprehensive health report...")
        report = monitor.generate_health_report()
        
        print(f"\nHealth Report Summary:")
        print(f"  Overall Status: {report['system_status']['overall_status']}")
        print(f"  Risk Score: {report['risk_assessment']['overall_risk_score']:.1f}/100")
        print(f"  Recommendations: {len(report['recommendations'])}")
        
        if report['recommendations']:
            print(f"  Top Recommendations:")
            for rec in report['recommendations'][:3]:
                print(f"    - {rec}")
        
        # Test circuit breaker
        print(f"\nTesting circuit breaker functionality...")
        
        @db_breaker
        def simulated_db_call():
            if np.random.random() < 0.7:  # 70% chance of failure
                raise Exception("Database connection failed")
            return "Success"
        
        # Try multiple calls to trigger circuit breaker
        for i in range(8):
            try:
                result = simulated_db_call()
                print(f"  DB Call {i+1}: {result}")
            except Exception as e:
                print(f"  DB Call {i+1}: FAILED - {e}")
        
        breaker_state = db_breaker.get_state()
        print(f"  Circuit Breaker State: {breaker_state['state']}")
        print(f"  Failure Count: {breaker_state['failure_count']}")
        
    finally:
        print("\nStopping health monitoring...")
        monitor.stop()
        
        # Final status
        final_status = monitor.get_status()
        print(f"Final monitoring session:")
        print(f"  Total metrics collected: {sum(len(deque) for deque in monitor.metrics.values())}")
        print(f"  Total alerts generated: {len(monitor.alerts)}")
        print(f"  Circuit breakers created: {len(monitor.circuit_breakers)}")
    
    print("\nHealth monitoring demonstration completed!")
