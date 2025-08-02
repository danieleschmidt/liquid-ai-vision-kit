# System Monitoring and Observability

## Overview

This document outlines monitoring and observability setup for production deployments of the Liquid AI Vision Kit.

## Health Check Endpoints

### HTTP Health Check
```cpp
// Health check endpoint configuration
class HealthMonitor {
public:
    struct HealthStatus {
        bool system_ok;
        float cpu_usage_percent;
        size_t memory_usage_kb;
        float temperature_celsius;
        uint64_t uptime_seconds;
        std::string version;
    };
    
    HealthStatus getSystemHealth() const;
    void enableHTTPEndpoint(int port = 8090);
};
```

### Implementation
```bash
# Health check endpoint returns JSON
curl http://localhost:8090/health
{
  "status": "healthy",
  "cpu_usage": 45.2,
  "memory_usage_kb": 128000,
  "temperature": 65.5,
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "inference_rate": 48.5,
  "last_inference_ms": 15.2
}
```

## Prometheus Metrics

### Core Metrics
```yaml
# Prometheus configuration
metrics:
  - name: liquid_vision_inference_duration_seconds
    type: histogram
    help: Time spent on neural network inference
    buckets: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    
  - name: liquid_vision_memory_usage_bytes
    type: gauge
    help: Current memory usage in bytes
    
  - name: liquid_vision_cpu_temperature_celsius
    type: gauge
    help: CPU temperature in Celsius
    
  - name: liquid_vision_inference_errors_total
    type: counter
    help: Total number of inference errors
```

### Custom Metrics
```cpp
// Metrics collection example
void recordInferenceMetrics(double duration, bool success) {
    static auto inference_duration = prometheus::BuildHistogram()
        .Name("liquid_vision_inference_duration_seconds")
        .Help("Neural network inference time")
        .Register(*registry);
        
    static auto inference_errors = prometheus::BuildCounter()
        .Name("liquid_vision_inference_errors_total")
        .Help("Total inference errors")
        .Register(*registry);
        
    inference_duration.Observe(duration);
    if (!success) {
        inference_errors.Increment();
    }
}
```

## Logging Configuration

### Structured Logging
```json
{
  "timestamp": "2025-01-01T12:00:00.000Z",
  "level": "INFO",
  "component": "liquid_network",
  "message": "Inference completed",
  "duration_ms": 12.5,
  "confidence": 0.92,
  "input_size": [160, 120, 3],
  "output": [0.1, 0.2, 0.3, 0.4]
}
```

### Log Levels
- **FATAL**: System cannot continue
- **ERROR**: Operation failed but system continues
- **WARN**: Potential issues or degraded performance
- **INFO**: Normal operation events
- **DEBUG**: Detailed execution information
- **TRACE**: Very detailed execution traces

## Performance Monitoring

### Real-time Metrics Dashboard
```yaml
grafana_dashboard:
  panels:
    - title: "Inference Latency"
      query: "rate(liquid_vision_inference_duration_seconds[5m])"
      
    - title: "Memory Usage"
      query: "liquid_vision_memory_usage_bytes"
      
    - title: "CPU Temperature"
      query: "liquid_vision_cpu_temperature_celsius"
      
    - title: "Error Rate"
      query: "rate(liquid_vision_inference_errors_total[5m])"
```

### Alerting Rules
```yaml
groups:
  - name: liquid_vision_alerts
    rules:
      - alert: HighInferenceLatency
        expr: liquid_vision_inference_duration_seconds > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Inference latency too high"
          
      - alert: HighTemperature
        expr: liquid_vision_cpu_temperature_celsius > 80
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "CPU temperature critical"
```

## System Resource Monitoring

### Memory Tracking
```cpp
class MemoryMonitor {
public:
    struct MemoryStats {
        size_t total_kb;
        size_t used_kb;
        size_t available_kb;
        size_t peak_usage_kb;
        float fragmentation_percent;
    };
    
    MemoryStats getCurrentStats() const;
    void setMemoryThreshold(size_t threshold_kb);
    bool isMemoryPressureHigh() const;
};
```

### CPU and Thermal Monitoring
```cpp
class SystemMonitor {
public:
    float getCPUUsagePercent() const;
    float getTemperatureCelsius() const;
    uint64_t getUptimeSeconds() const;
    bool isThermalThrottling() const;
    
    void enableThermalProtection(float max_temp = 85.0f);
};
```

## Network Monitoring

### Telemetry Streaming
```cpp
// MAVLink telemetry integration
class TelemetryMonitor {
public:
    void sendSystemStatus(const SystemStatus& status);
    void sendPerformanceMetrics(const PerformanceMetrics& metrics);
    void sendDiagnosticData(const DiagnosticInfo& info);
    
    void setUpdateRate(float hz = 10.0f);
    void enableCompression(bool enable = true);
};
```

### Data Export
```yaml
# Data export configuration
telemetry:
  protocols:
    - mavlink
    - mqtt
    - websocket
    
  formats:
    - json
    - msgpack
    - protobuf
    
  endpoints:
    - url: "mqtt://telemetry.example.com:1883"
      topic: "devices/{device_id}/metrics"
      
    - url: "ws://dashboard.example.com:8080/ws"
      format: "json"
```

## Safety Monitoring

### Critical System Checks
```cpp
class SafetyMonitor {
public:
    enum class SafetyStatus {
        SAFE,
        WARNING,
        CRITICAL,
        EMERGENCY
    };
    
    SafetyStatus checkSystemSafety() const;
    void enableWatchdog(uint32_t timeout_ms = 1000);
    void registerSafetyCallback(std::function<void(SafetyStatus)> callback);
    
private:
    bool checkMemoryLimits() const;
    bool checkTemperatureLimits() const;
    bool checkInferencePerformance() const;
    bool checkFlightControllerConnection() const;
};
```

### Emergency Procedures
```cpp
class EmergencyHandler {
public:
    void triggerEmergencyLanding();
    void enableFailsafeMode();
    void logEmergencyEvent(const std::string& reason);
    void notifyGroundStation(const EmergencyInfo& info);
    
    void setEmergencyCallback(std::function<void()> callback);
};
```

## Deployment Monitoring

### Kubernetes Integration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: liquid-vision-metrics
spec:
  selector:
    app: liquid-vision
  ports:
    - name: metrics
      port: 8090
      targetPort: 8090
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: liquid-vision
spec:
  selector:
    matchLabels:
      app: liquid-vision
  endpoints:
    - port: metrics
      interval: 30s
      path: /metrics
```

### Docker Health Checks
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8090/health || exit 1
```

## Troubleshooting Integration

### Diagnostic Data Collection
```cpp
class DiagnosticCollector {
public:
    void collectSystemInfo();
    void collectPerformanceLogs();
    void collectErrorLogs();
    void collectConfigurationDump();
    
    void generateDiagnosticReport(const std::string& output_path);
    void uploadDiagnostics(const std::string& support_url);
};
```

### Remote Debugging
```yaml
debug_server:
  enabled: false  # Only enable when needed
  port: 3333
  authentication:
    username: "debug"
    password_hash: "..."
  
  capabilities:
    - memory_inspection
    - log_streaming
    - configuration_viewing
    # NOTE: No code modification for security
```

## Maintenance Windows

### Automated Maintenance
```cpp
class MaintenanceScheduler {
public:
    void scheduleMaintenance(const std::chrono::system_clock::time_point& when);
    void enableAutomaticMaintenance(bool enable);
    void setMaintenanceWindow(int hour_start, int hour_end);
    
    void performRoutineMaintenance();
    void performLogRotation();
    void performConfigBackup();
};
```

### Update Monitoring
```cpp
class UpdateMonitor {
public:
    void checkForUpdates();
    void downloadUpdate(const std::string& version);
    void validateUpdate(const std::string& package_path);
    void installUpdate(bool immediate = false);
    
    void setUpdateCallback(std::function<void(UpdateStatus)> callback);
};
```