#!/usr/bin/env python3
"""
Comprehensive System Monitoring and Health Checks
Multi-dimensional monitoring with predictive analytics and alerting
"""

import time
import threading
import logging
import json
import psutil
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import asyncio
import subprocess
import os


class MetricType(Enum):
    """Types of metrics to monitor"""
    SYSTEM = "system"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SECURITY = "security"
    CUSTOM = "custom"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class MetricPoint:
    """Single metric measurement"""
    timestamp: float = field(default_factory=time.time)
    value: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric points"""
    name: str
    metric_type: MetricType
    unit: str = ""
    description: str = ""
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    aggregation_window: float = 60.0  # seconds
    
    def add_point(self, value: float, tags: Dict[str, str] = None):
        """Add metric point"""
        point = MetricPoint(
            value=value,
            tags=tags or {},
            timestamp=time.time()
        )
        self.points.append(point)
    
    def get_recent_values(self, window_seconds: float = None) -> List[float]:
        """Get recent values within time window"""
        if window_seconds is None:
            window_seconds = self.aggregation_window
        
        cutoff_time = time.time() - window_seconds
        return [
            point.value for point in self.points
            if point.timestamp >= cutoff_time
        ]
    
    def get_statistics(self, window_seconds: float = None) -> Dict[str, float]:
        """Get statistical summary of recent values"""
        values = self.get_recent_values(window_seconds)
        
        if not values:
            return {"count": 0}
        
        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "p50": np.percentile(values, 50),
            "p90": np.percentile(values, 90),
            "p99": np.percentile(values, 99)
        }


@dataclass
class Alert:
    """System alert"""
    id: str = field(default_factory=lambda: f"alert_{int(time.time()*1000):013x}")
    timestamp: float = field(default_factory=time.time)
    level: AlertLevel = AlertLevel.INFO
    title: str = ""
    message: str = ""
    component: str = ""
    metric_name: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    def resolve(self):
        """Mark alert as resolved"""
        self.resolved = True
        self.resolution_time = time.time()


class ThresholdRule:
    """Threshold-based alerting rule"""
    
    def __init__(self, metric_name: str, threshold: float, comparison: str = "greater",
                 level: AlertLevel = AlertLevel.WARNING, duration: float = 60.0):
        self.metric_name = metric_name
        self.threshold = threshold
        self.comparison = comparison  # greater, less, equal
        self.level = level
        self.duration = duration  # How long condition must persist
        self.violation_start = None
        
    def check_violation(self, current_value: float) -> bool:
        """Check if threshold is violated"""
        violated = False
        
        if self.comparison == "greater":
            violated = current_value > self.threshold
        elif self.comparison == "less":
            violated = current_value < self.threshold
        elif self.comparison == "equal":
            violated = abs(current_value - self.threshold) < 1e-6
        
        current_time = time.time()
        
        if violated:
            if self.violation_start is None:
                self.violation_start = current_time
            return (current_time - self.violation_start) >= self.duration
        else:
            self.violation_start = None
            return False
    
    def create_alert(self, current_value: float, component: str = "") -> Alert:
        """Create alert for threshold violation"""
        return Alert(
            level=self.level,
            title=f"Threshold violation: {self.metric_name}",
            message=f"{self.metric_name} = {current_value:.3f} {self.comparison} {self.threshold:.3f}",
            component=component,
            metric_name=self.metric_name,
            current_value=current_value,
            threshold_value=self.threshold
        )


class HealthChecker:
    """Component health checker"""
    
    def __init__(self, name: str, check_function: Callable[[], Dict[str, Any]], 
                 interval: float = 30.0, timeout: float = 10.0):
        self.name = name
        self.check_function = check_function
        self.interval = interval
        self.timeout = timeout
        self.last_check_time = 0.0
        self.last_result = {"healthy": True, "message": "Not checked yet"}
        self.consecutive_failures = 0
        
    def perform_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            start_time = time.time()
            result = self.check_function()
            check_duration = time.time() - start_time
            
            if check_duration > self.timeout:
                result = {
                    "healthy": False,
                    "message": f"Health check timed out ({check_duration:.2f}s > {self.timeout:.2f}s)"
                }
            
            self.last_check_time = time.time()
            
            if result.get("healthy", False):
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
            
            self.last_result = result
            return result
            
        except Exception as e:
            self.consecutive_failures += 1
            self.last_result = {
                "healthy": False,
                "message": f"Health check failed: {str(e)}",
                "exception": str(e)
            }
            return self.last_result
    
    def is_due_for_check(self) -> bool:
        """Check if health check is due"""
        return (time.time() - self.last_check_time) >= self.interval


class PredictiveAnalyzer:
    """Predictive analytics for system metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.trend_coefficients = {}
        
    def analyze_trend(self, metric_series: MetricSeries, prediction_horizon: float = 300.0) -> Dict[str, Any]:
        """Analyze metric trend and predict future values"""
        values = metric_series.get_recent_values(window_seconds=3600)  # Last hour
        
        if len(values) < 10:
            return {"trend": "insufficient_data", "prediction": None}
        
        # Simple linear regression for trend analysis
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope, intercept = coeffs
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(x, values)[0, 1]
        
        # Predict future value
        future_x = len(values) + (prediction_horizon / 60.0)  # Assuming 1 minute intervals
        predicted_value = slope * future_x + intercept
        
        # Determine trend direction
        if abs(slope) < 0.001:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return {
            "trend": trend,
            "slope": slope,
            "correlation": correlation,
            "prediction": {
                "value": predicted_value,
                "horizon_seconds": prediction_horizon,
                "confidence": abs(correlation)
            },
            "analysis_timestamp": time.time()
        }
    
    def detect_anomalies(self, metric_series: MetricSeries, sensitivity: float = 2.0) -> List[Dict[str, Any]]:
        """Detect anomalies using statistical analysis"""
        values = metric_series.get_recent_values(window_seconds=3600)
        
        if len(values) < 30:
            return []
        
        # Calculate rolling statistics
        window_size = min(30, len(values) // 3)
        anomalies = []
        
        for i in range(window_size, len(values)):
            window_values = values[i-window_size:i]
            mean = np.mean(window_values)
            std = np.std(window_values)
            
            current_value = values[i]
            z_score = abs(current_value - mean) / (std + 1e-8)
            
            if z_score > sensitivity:
                anomalies.append({
                    "timestamp": time.time() - (len(values) - i) * 60,  # Approximate timestamp
                    "value": current_value,
                    "expected_value": mean,
                    "z_score": z_score,
                    "severity": "high" if z_score > sensitivity * 1.5 else "medium"
                })
        
        return anomalies[-10:]  # Return last 10 anomalies


class ComprehensiveMonitor:
    """Comprehensive system monitoring with predictive analytics"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core monitoring components
        self.metrics = {}  # metric_name -> MetricSeries
        self.alerts = deque(maxlen=10000)
        self.threshold_rules = {}
        self.health_checkers = {}
        
        # Analytics
        self.predictor = PredictiveAnalyzer()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.health_check_thread = None
        self.analytics_thread = None
        
        # Callbacks
        self.alert_callbacks = []
        
        # Initialize default metrics and rules
        self._initialize_default_metrics()
        self._initialize_default_rules()
        self._initialize_health_checkers()
        
    def _initialize_default_metrics(self):
        """Initialize default system metrics"""
        system_metrics = [
            ("cpu_usage_percent", MetricType.SYSTEM, "%", "CPU utilization percentage"),
            ("memory_usage_percent", MetricType.SYSTEM, "%", "Memory utilization percentage"),
            ("disk_usage_percent", MetricType.SYSTEM, "%", "Disk utilization percentage"),
            ("network_io_mbps", MetricType.SYSTEM, "MB/s", "Network I/O throughput"),
            ("task_throughput", MetricType.PERFORMANCE, "tasks/sec", "Task processing throughput"),
            ("task_latency_ms", MetricType.PERFORMANCE, "ms", "Average task processing latency"),
            ("error_rate", MetricType.PERFORMANCE, "errors/min", "Error occurrence rate"),
            ("queue_depth", MetricType.PERFORMANCE, "tasks", "Task queue depth"),
            ("active_connections", MetricType.BUSINESS, "connections", "Active system connections"),
            ("memory_leaks", MetricType.SECURITY, "MB/hr", "Memory leak detection rate")
        ]
        
        for name, metric_type, unit, description in system_metrics:
            self.metrics[name] = MetricSeries(
                name=name,
                metric_type=metric_type,
                unit=unit,
                description=description
            )
    
    def _initialize_default_rules(self):
        """Initialize default alerting rules"""
        default_rules = [
            ("cpu_usage_percent", ThresholdRule("cpu_usage_percent", 80.0, "greater", AlertLevel.WARNING, 120.0)),
            ("cpu_usage_percent_critical", ThresholdRule("cpu_usage_percent", 95.0, "greater", AlertLevel.CRITICAL, 60.0)),
            ("memory_usage_percent", ThresholdRule("memory_usage_percent", 85.0, "greater", AlertLevel.WARNING, 180.0)),
            ("memory_usage_percent_critical", ThresholdRule("memory_usage_percent", 95.0, "greater", AlertLevel.CRITICAL, 60.0)),
            ("disk_usage_percent", ThresholdRule("disk_usage_percent", 90.0, "greater", AlertLevel.ERROR, 300.0)),
            ("error_rate_high", ThresholdRule("error_rate", 10.0, "greater", AlertLevel.ERROR, 120.0)),
            ("task_latency_high", ThresholdRule("task_latency_ms", 1000.0, "greater", AlertLevel.WARNING, 180.0)),
            ("queue_depth_high", ThresholdRule("queue_depth", 100.0, "greater", AlertLevel.WARNING, 240.0))
        ]
        
        for rule_name, rule in default_rules:
            self.threshold_rules[rule_name] = rule
    
    def _initialize_health_checkers(self):
        """Initialize health checkers"""
        self.health_checkers = {
            "system_resources": HealthChecker(
                "system_resources",
                self._check_system_resources,
                interval=30.0
            ),
            "disk_space": HealthChecker(
                "disk_space",
                self._check_disk_space,
                interval=60.0
            ),
            "network_connectivity": HealthChecker(
                "network_connectivity",
                self._check_network_connectivity,
                interval=45.0
            ),
            "process_health": HealthChecker(
                "process_health",
                self._check_process_health,
                interval=30.0
            )
        }
    
    def start_monitoring(self):
        """Start comprehensive monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        # Start monitoring threads
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.analytics_thread = threading.Thread(target=self._analytics_loop, daemon=True)
        
        self.monitor_thread.start()
        self.health_check_thread.start()
        self.analytics_thread.start()
        
        self.logger.info("Comprehensive monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        
        # Wait for threads to finish
        for thread in [self.monitor_thread, self.health_check_thread, self.analytics_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        self.logger.info("Comprehensive monitoring stopped")
    
    def add_metric(self, name: str, metric_type: MetricType, unit: str = "", description: str = ""):
        """Add custom metric"""
        self.metrics[name] = MetricSeries(
            name=name,
            metric_type=metric_type,
            unit=unit,
            description=description
        )
        self.logger.info(f"Added custom metric: {name}")
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record metric value"""
        if name in self.metrics:
            self.metrics[name].add_point(value, tags)
        else:
            self.logger.warning(f"Unknown metric: {name}")
    
    def add_threshold_rule(self, rule_name: str, rule: ThresholdRule):
        """Add threshold-based alerting rule"""
        self.threshold_rules[rule_name] = rule
        self.logger.info(f"Added threshold rule: {rule_name}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check threshold rules
                self._check_threshold_rules()
                
                # Update metric statistics
                self._update_metric_statistics()
                
                time.sleep(5.0)  # 5-second intervals
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10.0)
    
    def _health_check_loop(self):
        """Health check loop"""
        while self.is_monitoring:
            try:
                for checker in self.health_checkers.values():
                    if checker.is_due_for_check():
                        result = checker.perform_check()
                        
                        if not result.get("healthy", True):
                            alert = Alert(
                                level=AlertLevel.ERROR,
                                title=f"Health check failed: {checker.name}",
                                message=result.get("message", "Unknown health check failure"),
                                component=checker.name
                            )
                            self._trigger_alert(alert)
                
                time.sleep(10.0)  # 10-second intervals
                
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                time.sleep(20.0)
    
    def _analytics_loop(self):
        """Predictive analytics loop"""
        while self.is_monitoring:
            try:
                # Run predictive analysis on key metrics
                key_metrics = ["cpu_usage_percent", "memory_usage_percent", "task_throughput", "error_rate"]
                
                for metric_name in key_metrics:
                    if metric_name in self.metrics:
                        metric_series = self.metrics[metric_name]
                        
                        # Trend analysis
                        trend_analysis = self.predictor.analyze_trend(metric_series)
                        
                        # Anomaly detection
                        anomalies = self.predictor.detect_anomalies(metric_series)
                        
                        # Create predictive alerts
                        if trend_analysis.get("trend") == "increasing" and metric_name in ["cpu_usage_percent", "memory_usage_percent", "error_rate"]:
                            prediction = trend_analysis.get("prediction", {})
                            if prediction.get("value", 0) > 90:
                                alert = Alert(
                                    level=AlertLevel.WARNING,
                                    title=f"Predictive alert: {metric_name}",
                                    message=f"Metric trending upward, predicted to reach {prediction.get('value', 0):.1f} in {prediction.get('horizon_seconds', 0)/60:.0f} minutes",
                                    component="predictive_analytics",
                                    metric_name=metric_name
                                )
                                self._trigger_alert(alert)
                        
                        # Create anomaly alerts
                        for anomaly in anomalies:
                            if anomaly["severity"] == "high":
                                alert = Alert(
                                    level=AlertLevel.WARNING,
                                    title=f"Anomaly detected: {metric_name}",
                                    message=f"Anomalous value {anomaly['value']:.2f} (expected ~{anomaly['expected_value']:.2f}, z-score: {anomaly['z_score']:.1f})",
                                    component="anomaly_detection",
                                    metric_name=metric_name
                                )
                                self._trigger_alert(alert)
                
                time.sleep(300.0)  # 5-minute intervals
                
            except Exception as e:
                self.logger.error(f"Analytics loop error: {e}")
                time.sleep(600.0)  # 10-minute retry on error
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1.0)
            self.record_metric("cpu_usage_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric("memory_usage_percent", memory.percent)
            
            # Disk usage (root filesystem)
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric("disk_usage_percent", disk_percent)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if hasattr(self, '_last_net_io'):
                bytes_sent_delta = net_io.bytes_sent - self._last_net_io.bytes_sent
                bytes_recv_delta = net_io.bytes_recv - self._last_net_io.bytes_recv
                total_mbps = (bytes_sent_delta + bytes_recv_delta) / (1024 * 1024)  # MB/s
                self.record_metric("network_io_mbps", total_mbps)
            self._last_net_io = net_io
            
        except Exception as e:
            self.logger.error(f"System metrics collection error: {e}")
    
    def _check_threshold_rules(self):
        """Check all threshold rules"""
        for rule_name, rule in self.threshold_rules.items():
            if rule.metric_name in self.metrics:
                metric_series = self.metrics[rule.metric_name]
                recent_values = metric_series.get_recent_values(window_seconds=60)
                
                if recent_values:
                    current_value = recent_values[-1]  # Most recent value
                    
                    if rule.check_violation(current_value):
                        alert = rule.create_alert(current_value, component="threshold_monitor")
                        self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Alert):
        """Trigger alert and notify callbacks"""
        self.alerts.append(alert)
        
        # Log alert
        log_level = logging.INFO
        if alert.level >= AlertLevel.ERROR:
            log_level = logging.ERROR
        elif alert.level >= AlertLevel.WARNING:
            log_level = logging.WARNING
        
        self.logger.log(log_level, f"ALERT [{alert.level.name}] {alert.title}: {alert.message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def _update_metric_statistics(self):
        """Update metric statistics for all metrics"""
        # This would update derived metrics, aggregations, etc.
        # For now, we just ensure data is being collected properly
        pass
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Health check for system resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1.0)
            memory = psutil.virtual_memory()
            
            if cpu_percent > 95:
                return {"healthy": False, "message": f"CPU usage critically high: {cpu_percent:.1f}%"}
            
            if memory.percent > 95:
                return {"healthy": False, "message": f"Memory usage critically high: {memory.percent:.1f}%"}
            
            return {"healthy": True, "message": f"System resources OK (CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%)"}
            
        except Exception as e:
            return {"healthy": False, "message": f"System resource check failed: {e}"}
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Health check for disk space"""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent > 95:
                return {"healthy": False, "message": f"Disk space critically low: {usage_percent:.1f}% used"}
            
            return {"healthy": True, "message": f"Disk space OK: {usage_percent:.1f}% used"}
            
        except Exception as e:
            return {"healthy": False, "message": f"Disk space check failed: {e}"}
    
    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Health check for network connectivity"""
        try:
            # Simple ping test
            result = subprocess.run(
                ['ping', '-c', '1', '-W', '3', '8.8.8.8'],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return {"healthy": True, "message": "Network connectivity OK"}
            else:
                return {"healthy": False, "message": "Network connectivity test failed"}
                
        except Exception as e:
            return {"healthy": False, "message": f"Network connectivity check failed: {e}"}
    
    def _check_process_health(self) -> Dict[str, Any]:
        """Health check for process health"""
        try:
            process = psutil.Process(os.getpid())
            
            # Check for excessive resource usage
            if process.memory_percent() > 50:  # More than 50% of system memory
                return {"healthy": False, "message": f"Process using excessive memory: {process.memory_percent():.1f}%"}
            
            # Check for zombie threads
            thread_count = process.num_threads()
            if thread_count > 100:  # Arbitrary high threshold
                return {"healthy": False, "message": f"Excessive thread count: {thread_count}"}
            
            return {"healthy": True, "message": f"Process health OK (Memory: {process.memory_percent():.1f}%, Threads: {thread_count})"}
            
        except Exception as e:
            return {"healthy": False, "message": f"Process health check failed: {e}"}
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        dashboard = {
            "timestamp": time.time(),
            "system_overview": {},
            "metrics_summary": {},
            "active_alerts": [],
            "health_status": {},
            "predictive_insights": {}
        }
        
        # System overview
        try:
            dashboard["system_overview"] = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100,
                "uptime_seconds": time.time() - psutil.boot_time(),
                "process_count": len(psutil.pids())
            }
        except Exception as e:
            self.logger.error(f"System overview error: {e}")
        
        # Metrics summary
        for name, metric_series in self.metrics.items():
            stats = metric_series.get_statistics()
            if stats.get("count", 0) > 0:
                dashboard["metrics_summary"][name] = {
                    "current": stats.get("mean", 0),
                    "unit": metric_series.unit,
                    "trend": "stable"  # Would be calculated from trend analysis
                }
        
        # Active alerts
        recent_alerts = [alert for alert in self.alerts if not alert.resolved and (time.time() - alert.timestamp) < 3600]
        dashboard["active_alerts"] = [
            {
                "level": alert.level.name,
                "title": alert.title,
                "message": alert.message,
                "age_seconds": time.time() - alert.timestamp
            }
            for alert in recent_alerts[-10:]  # Last 10 alerts
        ]
        
        # Health status
        dashboard["health_status"] = {}
        for name, checker in self.health_checkers.items():
            dashboard["health_status"][name] = {
                "healthy": checker.last_result.get("healthy", True),
                "message": checker.last_result.get("message", ""),
                "consecutive_failures": checker.consecutive_failures,
                "last_check": checker.last_check_time
            }
        
        return dashboard
    
    def get_metric_data(self, metric_name: str, window_seconds: float = 3600) -> Dict[str, Any]:
        """Get detailed metric data"""
        if metric_name not in self.metrics:
            return {"error": f"Metric {metric_name} not found"}
        
        metric_series = self.metrics[metric_name]
        values = metric_series.get_recent_values(window_seconds)
        stats = metric_series.get_statistics(window_seconds)
        
        # Get trend analysis
        trend_analysis = self.predictor.analyze_trend(metric_series)
        
        # Get anomalies
        anomalies = self.predictor.detect_anomalies(metric_series)
        
        return {
            "name": metric_name,
            "type": metric_series.metric_type.value,
            "unit": metric_series.unit,
            "description": metric_series.description,
            "data_points": len(values),
            "statistics": stats,
            "trend_analysis": trend_analysis,
            "recent_anomalies": anomalies,
            "timestamp": time.time()
        }
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve alert"""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolve()
                self.logger.info(f"Alert {alert_id} manually resolved")
                return True
        return False


def demo_comprehensive_monitoring():
    """Demonstrate comprehensive monitoring capabilities"""
    print("📊 Comprehensive System Monitor Demo")
    print("=" * 60)
    
    # Create monitor
    monitor = ComprehensiveMonitor()
    
    # Add custom alert callback
    def alert_callback(alert: Alert):
        print(f"🚨 ALERT: [{alert.level.name}] {alert.title}")
    
    monitor.add_alert_callback(alert_callback)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        print("✓ Monitoring started, collecting metrics...")
        
        # Let it run for 30 seconds
        for i in range(6):  # 6 * 5 = 30 seconds
            time.sleep(5)
            
            # Add some custom metrics
            monitor.record_metric("task_throughput", np.random.uniform(10, 50))
            monitor.record_metric("task_latency_ms", np.random.uniform(50, 200))
            monitor.record_metric("error_rate", np.random.exponential(2))
            monitor.record_metric("queue_depth", np.random.poisson(20))
            
            # Get dashboard update
            dashboard = monitor.get_monitoring_dashboard()
            
            print(f"\n📈 Dashboard Update {i+1}:")
            print(f"   CPU: {dashboard['system_overview']['cpu_percent']:.1f}%")
            print(f"   Memory: {dashboard['system_overview']['memory_percent']:.1f}%")
            print(f"   Active Alerts: {len(dashboard['active_alerts'])}")
            
            # Show health status
            healthy_components = sum(
                1 for status in dashboard['health_status'].values()
                if status['healthy']
            )
            total_components = len(dashboard['health_status'])
            print(f"   Health: {healthy_components}/{total_components} components healthy")
        
        # Get detailed analysis
        print(f"\n🔍 Detailed Analysis:")
        for metric_name in ["cpu_usage_percent", "task_throughput", "error_rate"]:
            metric_data = monitor.get_metric_data(metric_name, window_seconds=300)  # 5 minutes
            if "statistics" in metric_data:
                stats = metric_data["statistics"]
                trend = metric_data["trend_analysis"]
                print(f"   {metric_name}:")
                print(f"     Mean: {stats.get('mean', 0):.2f} {metric_data['unit']}")
                print(f"     Trend: {trend.get('trend', 'unknown')}")
                print(f"     Data points: {stats.get('count', 0)}")
                
                if metric_data["recent_anomalies"]:
                    print(f"     Anomalies: {len(metric_data['recent_anomalies'])}")
        
        print(f"\n✅ Monitoring demo completed successfully")
        
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    demo_comprehensive_monitoring()